"""
Extract configuration from NeMo models and generate config.json and preprocessor_config.json files.

Usage:
    python extract_nemo_config.py <path_to_nemo_file> [output_dir]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def load_nemo_model(nemo_path: str):
    """
    Load a NeMo model from a .nemo file.
    
    Args:
        nemo_path: Path to the .nemo file
        
    Returns:
        The loaded NeMo model
    """
    try:
        import nemo.collections.tts.models as tts_models
        import nemo.collections.asr.models as asr_models
        
        # Try TTS AudioCodecModel first
        try:
            model = tts_models.AudioCodecModel.restore_from(nemo_path)
            return model, "audio_codec"
        except Exception:
            pass
        
        tts_model_classes = [
            ("FastPitchModel", tts_models.FastPitchModel),
            ("HifiGanModel", tts_models.HifiGanModel),
            ("VitsModel", tts_models.VitsModel),
            ("Tacotron2Model", tts_models.Tacotron2Model),
        ]
        
        for model_name, model_class in tts_model_classes:
            try:
                model = model_class.restore_from(nemo_path)
                return model, model_name.lower().replace("model", "")
            except Exception:
                continue
        
        asr_model_classes = [
            ("EncDecCTCModel", asr_models.EncDecCTCModel),
            ("EncDecRNNTModel", asr_models.EncDecRNNTModel),
        ]
        
        for model_name, model_class in asr_model_classes:
            try:
                model = model_class.restore_from(nemo_path)
                return model, model_name.lower().replace("model", "")
            except Exception:
                continue
        
        raise RuntimeError(f"Could not load model from {nemo_path} - unsupported model type")
        
    except ImportError as e:
        print(f"Error: NeMo is not installed or cannot be imported: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading NeMo model: {e}")
        sys.exit(1)

def extract_config_dict(model) -> Dict[str, Any]:
    """
    Extract configuration from a NeMo model as a dictionary.
    
    Args:
        model: NeMo model instance
        
    Returns:
        Configuration dictionary
    """
    try:
        # Convert OmegaConf to dict
        from omegaconf import OmegaConf
        
        if hasattr(model, 'cfg'):
            cfg_dict = OmegaConf.to_container(model.cfg, resolve=True)
            return cfg_dict
        else:
            print("Warning: Model does not have a 'cfg' attribute")
            return {}
    except Exception as e:
        print(f"Error extracting config: {e}")
        return {}

def detect_model_architecture(model, model_type: str) -> Dict[str, Any]:
    """
    Detect and extract model architecture information.
    
    Args:
        model: NeMo model instance
        model_type: Type of model (audio_codec, fastpitch, etc.)
        
    Returns:
        Dictionary with architecture information
    """
    arch_info = {
        "model_type": model_type,
        "architectures": [],
    }
    
    model_class = model.__class__.__name__
    arch_info["architectures"].append(model_class)
    
    # Try to get sample rate
    if hasattr(model, 'cfg'):
        cfg = model.cfg
        if hasattr(cfg, 'sample_rate'):
            arch_info["sample_rate"] = cfg.sample_rate
        elif hasattr(cfg, 'preprocessor') and hasattr(cfg.preprocessor, 'sample_rate'):
            arch_info["sample_rate"] = cfg.preprocessor.sample_rate
    
    return arch_info

def create_config_json(model, model_type: str, cfg_dict: Dict[str, Any], output_path: str):
    """
    Create a config.json file from the model.
    
    Args:
        model: NeMo model instance
        model_type: Type of model
        cfg_dict: Full configuration dictionary
        output_path: Path to save config.json
    """
    config = detect_model_architecture(model, model_type)
    
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'name'):
        config["_name_or_path"] = model.cfg.name
    
    if model_type == "audio_codec":
        if hasattr(model.cfg, 'sample_rate'):
            config["sample_rate"] = model.cfg.sample_rate
        if hasattr(model.cfg, 'samples_per_frame'):
            config["samples_per_frame"] = model.cfg.samples_per_frame
            config["frame_rate"] = config["sample_rate"] / config["samples_per_frame"]
        
        if 'audio_encoder' in cfg_dict:
            encoder_cfg = cfg_dict['audio_encoder']
            if isinstance(encoder_cfg, dict):
                config["encoder"] = {
                    "type": encoder_cfg.get('_target_', 'Unknown').split('.')[-1],
                    **{k: v for k, v in encoder_cfg.items() if k not in ['_target_']}
                }
        
        if 'audio_decoder' in cfg_dict:
            decoder_cfg = cfg_dict['audio_decoder']
            if isinstance(decoder_cfg, dict):
                config["decoder"] = {
                    "type": decoder_cfg.get('_target_', 'Unknown').split('.')[-1],
                    **{k: v for k, v in decoder_cfg.items() if k not in ['_target_']}
                }
        
        if 'vector_quantizer' in cfg_dict:
            quantizer_cfg = cfg_dict['vector_quantizer']
            if isinstance(quantizer_cfg, dict):
                config["vector_quantizer"] = {
                    "type": quantizer_cfg.get('_target_', 'Unknown').split('.')[-1],
                    **{k: v for k, v in quantizer_cfg.items() if k not in ['_target_']}
                }
    
    elif model_type in ["fastpitch", "tacotron2"]:
        # Acoustic model parameters
        if hasattr(model.cfg, 'n_speakers'):
            config["n_speakers"] = model.cfg.n_speakers
        if hasattr(model.cfg, 'symbols_embedding_dim'):
            config["symbols_embedding_dim"] = model.cfg.symbols_embedding_dim
    
    elif model_type == "hifigan":
        # Vocoder parameters
        if hasattr(model.cfg, 'generator'):
            config["generator"] = cfg_dict.get('generator', {})
    
    config["torch_dtype"] = "float32"
    
    if 'nemo_version' in cfg_dict:
        config["nemo_version"] = cfg_dict['nemo_version']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  Created config.json at {output_path}")

def create_preprocessor_config(model, model_type: str, cfg_dict: Dict[str, Any], output_path: str):
    """
    Create a preprocessor_config.json file.
    
    Args:
        model: NeMo model instance
        model_type: Type of model
        cfg_dict: Full configuration dictionary
        output_path: Path to save preprocessor_config.json
    """
    preprocessor_config = {
        "feature_extractor_type": f"NeMo{model_type.title()}FeatureExtractor",
        "feature_size": 1,
    }
    
    sample_rate = None
    if hasattr(model.cfg, 'sample_rate'):
        sample_rate = model.cfg.sample_rate
    elif hasattr(model.cfg, 'preprocessor') and hasattr(model.cfg.preprocessor, 'sample_rate'):
        sample_rate = model.cfg.preprocessor.sample_rate
    
    if sample_rate:
        preprocessor_config["sampling_rate"] = sample_rate
    
    if model_type == "audio_codec":
        if hasattr(model.cfg, 'samples_per_frame'):
            preprocessor_config["hop_length"] = model.cfg.samples_per_frame
            if sample_rate:
                preprocessor_config["frame_rate"] = sample_rate / model.cfg.samples_per_frame
        
        preprocessor_config["padding_side"] = "right"
        preprocessor_config["padding_value"] = 0.0
        preprocessor_config["return_attention_mask"] = False
        preprocessor_config["do_normalize"] = False
    
    elif model_type in ["fastpitch", "tacotron2", "hifigan"]:
        # TTS preprocessing
        if hasattr(model.cfg, 'preprocessor'):
            prep_cfg = cfg_dict.get('preprocessor', {})
            if 'n_fft' in prep_cfg:
                preprocessor_config["n_fft"] = prep_cfg['n_fft']
            if 'hop_length' in prep_cfg:
                preprocessor_config["hop_length"] = prep_cfg['hop_length']
            if 'win_length' in prep_cfg:
                preprocessor_config["win_length"] = prep_cfg['win_length']
    
    preprocessor_config["processor_class"] = f"NeMo{model_type.title()}Processor"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessor_config, f, indent=2, ensure_ascii=False)
    
    print(f"  Created preprocessor_config.json at {output_path}")

def create_full_config_dump(cfg_dict: Dict[str, Any], output_path: str):
    """
    Create a full configuration dump for reference.
    
    Args:
        cfg_dict: Full configuration dictionary
        output_path: Path to save full_config.json
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  Created full_config.json (complete dump) at {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract configuration from NeMo models and generate config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract config from a codec model
  python extract_nemo_config.py nemo-nano-codec.nemo
  
  # Extract config and save to specific directory
  python extract_nemo_config.py nemo-nano-codec.nemo ./output_configs
        """
    )
    
    parser.add_argument(
        'nemo_path',
        type=str,
        help='Path to the .nemo model file'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',
        default=None,
        help='Output directory for config files (default: same directory as .nemo file)'
    )
    parser.add_argument(
        '--full-dump',
        action='store_true',
        help='Also create a full_config.json with complete configuration dump'
    )
    
    args = parser.parse_args()
    
    nemo_path = Path(args.nemo_path)
    if not nemo_path.exists():
        print(f"Error: File not found: {nemo_path}")
        sys.exit(1)
    
    if not nemo_path.suffix == '.nemo':
        print(f"Warning: File does not have .nemo extension: {nemo_path}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = nemo_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading NeMo model from: {nemo_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    model, model_type = load_nemo_model(str(nemo_path))
    print(f"  Loaded model type: {model_type}")
    print(f"  Model class: {model.__class__.__name__}")
    
    print("\nExtracting configuration")
    cfg_dict = extract_config_dict(model)
    
    if not cfg_dict:
        print("Error: Could not extract configuration from model")
        sys.exit(1)
    
    print(f"  Extracted {len(cfg_dict)} configuration keys")
    
    print("\nCreating config files")
    
    config_json_path = output_dir / "config.json"
    create_config_json(model, model_type, cfg_dict, str(config_json_path))
    
    preprocessor_config_path = output_dir / "preprocessor_config.json"
    create_preprocessor_config(model, model_type, cfg_dict, str(preprocessor_config_path))
    
    if args.full_dump:
        full_config_path = output_dir / "full_config.json"
        create_full_config_dump(cfg_dict, str(full_config_path))
    
    print("\n  All config files created successfully!")
    print(f"\nOutput files:")
    print(f"  - {config_json_path}")
    print(f"  - {preprocessor_config_path}")
    if args.full_dump:
        print(f"  - {full_config_path}")

if __name__ == "__main__":
    main()
