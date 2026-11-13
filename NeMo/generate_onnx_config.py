#!/usr/bin/env python3
"""
This script takes a basic extracted config and adds ONNX-specific information.

Usage:
    python generate_onnx_config.py <extracted_config.json> <encoder.onnx> <decoder.onnx> [output_config.json]
"""

import json
import argparse
import onnx
from pathlib import Path
from typing import Dict, Any, List, Tuple


def get_onnx_io_info(model_path: str) -> Dict[str, Any]:
    """Extract input/output information from ONNX model."""
    model = onnx.load(model_path)
    
    def parse_tensor_info(tensor):
        # Map ONNX type codes to string names
        type_map = {
            1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 
            9: "bool", 10: "float16", 11: "double", 12: "uint32", 13: "uint64"
        }
        
        tensor_info = {
            "name": tensor.name,
            "type": type_map.get(tensor.type.tensor_type.elem_type, f"unknown_{tensor.type.tensor_type.elem_type}"),
            "shape": []
        }
        
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_param:
                tensor_info["shape"].append(dim.dim_param)
            elif dim.dim_value:
                tensor_info["shape"].append(dim.dim_value)
            else:
                tensor_info["shape"].append("dynamic")
        
        return tensor_info
    
    inputs = [parse_tensor_info(inp) for inp in model.graph.input]
    outputs = [parse_tensor_info(out) for out in model.graph.output]
    
    return {"inputs": inputs, "outputs": outputs}


def calculate_quantizer_info(vector_quantizer: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate additional quantizer information."""
    if "num_levels_per_group" in vector_quantizer:
        levels = vector_quantizer["num_levels_per_group"]
        total_size = 1
        for level in levels:
            total_size *= level
        
        return {
            **vector_quantizer,
            "total_codebook_size": total_size,
            "notes": "Quantizer is integrated into encoder wrapper. Encoder outputs discrete int32 tokens."
        }
    
    return vector_quantizer


def calculate_bitrate(sample_rate: int, samples_per_frame: int, num_groups: int, num_levels_per_group: List[int]) -> float:
    """Calculate approximate bitrate based on quantizer configuration."""
    frame_rate = sample_rate / samples_per_frame
    
    # Bits per token (sum of log2 of each group's levels)
    import math
    bits_per_frame = sum(math.log2(levels) for levels in num_levels_per_group)
    
    bits_per_second = frame_rate * bits_per_frame * num_groups
    
    return bits_per_second / 1000  # kbps


def enhance_config(extracted_config: Dict[str, Any], encoder_info: Dict[str, Any], decoder_info: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance extracted config with ONNX-specific information."""
    
    config = extracted_config.copy()
    
    config["_name_or_path"] = f"nvidia/nemo-nano-codec-{config['sample_rate']//1000}khz"
    config["onnx_opset_version"] = 20  # torch>=2.9 dynamo export default
    config["export_method"] = "bypass_padding"
    config["dynamic_shapes_support"] = True
    
    if "bitrate_kbps" not in config and "vector_quantizer" in config:
        vq = config["vector_quantizer"]
        if "num_groups" in vq and "num_levels_per_group" in vq:
            bitrate = calculate_bitrate(
                config["sample_rate"],
                config["samples_per_frame"], 
                vq["num_groups"],
                vq["num_levels_per_group"]
            )
            config["bitrate_kbps"] = round(bitrate, 2)
    
    if "encoder" in config:
        enc_inputs = {inp["name"]: inp["type"] for inp in encoder_info["inputs"]}
        enc_outputs = {out["name"]: out["type"] for out in encoder_info["outputs"]}
        enc_input_shapes = {inp["name"]: inp["shape"] for inp in encoder_info["inputs"]}
        enc_output_shapes = {out["name"]: out["shape"] for out in encoder_info["outputs"]}
        
        config["encoder"].update({
            "total_downsampling_ratio": config["samples_per_frame"],
            "bypass_internal_padding": True,
            "requires_pre_padded_input": True,
            "input_names": [inp["name"] for inp in encoder_info["inputs"]],
            "output_names": [out["name"] for out in encoder_info["outputs"]],
            "input_types": enc_inputs,
            "output_types": enc_outputs,
            "input_shapes": enc_input_shapes,
            "output_shapes": enc_output_shapes,
            "notes": "Input audio MUST be pre-padded to complete frames (num_samples must be multiple of {}). Use pad_audio_to_frames() helper before encoding.".format(config["samples_per_frame"])
        })
    
    if "decoder" in config:
        dec_inputs = {inp["name"]: inp["type"] for inp in decoder_info["inputs"]}
        dec_outputs = {out["name"]: out["type"] for out in decoder_info["outputs"]}
        dec_input_shapes = {inp["name"]: inp["shape"] for inp in decoder_info["inputs"]}
        dec_output_shapes = {out["name"]: out["shape"] for out in decoder_info["outputs"]}
        
        config["decoder"].update({
            "total_upsampling_ratio": config["samples_per_frame"],
            "input_names": [inp["name"] for inp in decoder_info["inputs"]],
            "output_names": [out["name"] for out in decoder_info["outputs"]],
            "input_types": dec_inputs,
            "output_types": dec_outputs,
            "input_shapes": dec_input_shapes,
            "output_shapes": dec_output_shapes,
            "notes": "Expects discrete tokens from encoder. Fully supports dynamic input/output lengths."
        })
    
    if "vector_quantizer" in config:
        config["vector_quantizer"] = calculate_quantizer_info(config["vector_quantizer"])
    
    if "audio_channels" not in config:
        config["audio_channels"] = 1
    
    config["export_notes"] = "Models exported with bypassed internal padding to enable dynamic shapes. Encoder requires external padding to complete frames (multiples of samples_per_frame)."
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced ONNX config from extracted NeMo config")
    parser.add_argument("extracted_config", help="Path to extracted config.json from NeMo model")
    parser.add_argument("encoder_onnx", help="Path to encoder ONNX model")
    parser.add_argument("decoder_onnx", help="Path to decoder ONNX model")
    parser.add_argument("output_config", nargs="?", help="Output path for enhanced config (default: enhanced_config.json)")
    
    args = parser.parse_args()
    
    with open(args.extracted_config, 'r') as f:
        extracted_config = json.load(f)
    
    print("Analyzing encoder ONNX model...")
    encoder_info = get_onnx_io_info(args.encoder_onnx)
    
    print("Analyzing decoder ONNX model...")
    decoder_info = get_onnx_io_info(args.decoder_onnx)
    
    print("Enhancing config with ONNX-specific information...")
    enhanced_config = enhance_config(extracted_config, encoder_info, decoder_info)
    
    output_path = args.output_config or "enhanced_config.json"
    with open(output_path, 'w') as f:
        json.dump(enhanced_config, f, indent=2)
    
    print(f"✓ Enhanced config saved to {output_path}")
    
    print(f"\nSummary:")
    print(f"  Model: {enhanced_config.get('model_type', 'unknown')}")
    print(f"  Sample Rate: {enhanced_config.get('sample_rate', 'unknown')} Hz")
    print(f"  Bitrate: {enhanced_config.get('bitrate_kbps', 'unknown')} kbps")
    print(f"  Codebooks: {enhanced_config.get('vector_quantizer', {}).get('num_groups', 'unknown')}")
    print(f"  Frame Rate: {enhanced_config.get('frame_rate', 'unknown')} fps")


if __name__ == "__main__":
    main()