"""
Export NeMo Nano Codec Models to ONNX format.

Supported Models:
- nemo-nano-codec-22khz-1.78kbps-12.5fps (13 codebooks, 12.5 fps)
- nemo-nano-codec-22khz-1.89kbps-21.5fps (8 codebooks, 21.5 fps)
- nemo-nano-codec-22khz-0.6kbps-12.5fps (4 codebooks, 12.5 fps)

Uses the public API (encode/decode) which includes quantization, 
unlike the internal audio_encoder/audio_decoder methods.

Key differences from internal API:
- Uses model.encode() -> returns discrete tokens (int32) with quantization
- Uses model.decode() -> expects discrete tokens, not continuous features
- Exports models that work with the token-based format used in the NeMo README

Requirements:
    pip install nemo_toolkit[all] torch onnx onnxruntime
"""

import os
import argparse
import torch
import onnx
from pathlib import Path

# Model configuration lookup based on NVIDIA README
# https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps
MODEL_CONFIGS = {
    "1.78kbps-12.5fps": {
        "sample_rate": 22050,
        "frame_rate": 12.5,
        "bit_rate": "1.78kbps",
        "num_codebooks": 13,
        "codebook_size": 2016,
        "embed_dim": 52,
        "fsq_levels": [8, 7, 6, 6]
    },
    "0.6kbps-12.5fps": {
        "sample_rate": 22050,
        "frame_rate": 12.5,
        "bit_rate": "0.6kbps",
        "num_codebooks": 4,
        "codebook_size": 4032,
        "embed_dim": 16,
        "fsq_levels": [9, 8, 8, 7]
    },
    "1.89kbps-21.5fps": {
        "sample_rate": 22050,
        "frame_rate": 21.5,
        "bit_rate": "1.89kbps",
        "num_codebooks": 8,
        "codebook_size": 2016,
        "embed_dim": 32,
        "fsq_levels": [8, 7, 6, 6]
    }
}

def detect_model_variant(checkpoint_path: str, model) -> dict:
    """
    Detect model variant from checkpoint path or model configuration.
    
    Args:
        checkpoint_path: Path to the .nemo checkpoint file
        model: Loaded NeMo model
        
    Returns:
        Dictionary with model configuration including variant name
    """
    checkpoint_name = Path(checkpoint_path).stem
    
    # Try to extract variant from checkpoint name
    for variant_key in MODEL_CONFIGS.keys():
        if variant_key in checkpoint_name:
            config = MODEL_CONFIGS[variant_key].copy()
            config['variant'] = variant_key
            return config
    
    # Fallback: detect from model properties
    sample_rate = model.sample_rate
    samples_per_frame = model.samples_per_frame
    frame_rate = sample_rate / samples_per_frame
    
    # Match to closest known configuration
    for variant_key, config in MODEL_CONFIGS.items():
        if abs(config['frame_rate'] - frame_rate) < 0.1:
            result_config = config.copy()
            result_config['variant'] = variant_key
            return result_config
    
    # Unknown variant, create config from model
    print(f"Warning: Unknown model variant, using detected values")
    return {
        'variant': f"{sample_rate//1000}khz-{frame_rate:.1f}fps",
        'sample_rate': sample_rate,
        'frame_rate': frame_rate,
        'num_codebooks': 4,     # Default fallback
        'codebook_size': 5000,  # Safe upper bound
        'bit_rate': 'unknown',
        'embed_dim': None,
        'fsq_levels': None
    }

def load_nemo_model(checkpoint_path: str):
    """
    Load NeMo audio codec model from checkpoint and print configuration details.
    
    Args:
        checkpoint_path: Path to the .nemo checkpoint file
        
    Returns:
        Tuple of (loaded NeMo model, configuration dict)
    """
    try:
        from nemo.collections.tts.models.audio_codec import AudioCodecModel
        
        print(f"Loading NeMo model from: {checkpoint_path}")
        model = AudioCodecModel.restore_from(checkpoint_path)
        model.eval()
        
        config = detect_model_variant(checkpoint_path, model)
        
        print("\n=== Model Configuration ===")
        print(f"Detected variant: {config['variant']}")
        print(f"Sample rate: {model.sample_rate} Hz")
        print(f"Samples per frame: {model.samples_per_frame}")
        print(f"Frame rate: {model.sample_rate / model.samples_per_frame:.2f} fps")
        print(f"Bit rate: {config['bit_rate']}")
        print(f"Number of codebooks: {config['num_codebooks']}")
        print(f"Codebook size: {config['codebook_size']}")
        if config['embed_dim']:
            print(f"Embed dimension: {config['embed_dim']}")
        if config['fsq_levels']:
            print(f"FSQ levels: {config['fsq_levels']}")
        
        print("=" * 50)
        
        return model, config
    except ImportError as e:
        print(f"Error: NeMo toolkit not installed properly. {e}")
        print("Install with: pip install nemo_toolkit[all]")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class EncoderWrapper(torch.nn.Module):
    """Wrapper for NeMo encoder that bypasses internal padding to support dynamic shapes.
    
    IMPORTANT: Input audio MUST be pre-padded to complete frames (multiples of samples_per_frame).
    The model expects audio that is already padded externally to avoid dynamic padding issues
    during ONNX export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.samples_per_frame = model.samples_per_frame  # Store for reference (typically 1764)
    
    def forward(self, audio, audio_len):
        """Forward pass that bypasses internal padding (expects pre-padded audio).
        
        Args:
            audio: Input audio tensor of shape [batch, samples] 
                   MUST be pre-padded to complete frames externally
            audio_len: Length tensor of shape [batch] containing actual lengths BEFORE padding
        
        Returns:
            encoded_tokens: Discrete tokens [batch, codebooks, frames]
            encoded_len: Frame counts [batch]
        """
        # Bypass encode_audio() which does internal padding
        # Call audio_encoder directly with pre-padded audio
        encoded, encoded_len = self.model.audio_encoder(audio=audio, audio_len=audio_len)
        
        # Apply quantization to get discrete tokens
        tokens = self.model.quantize(encoded=encoded, encoded_len=encoded_len)
        
        return tokens, encoded_len

class DecoderWrapper(torch.nn.Module):
    """Wrapper for NeMo decoder that bypasses internal masking for ONNX dynamo export.
    
    IMPORTANT: Bypasses decode_audio() which uses mask_sequence_tensor that creates
    data-dependent masks incompatible with dynamo export. Calls audio_decoder directly.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, tokens, tokens_len):
        """Forward pass bypassing internal masking for ONNX export compatibility.
        
        Args:
            tokens: Input token tensor of shape [batch, codebooks, frames]
            tokens_len: Length tensor of shape [batch] containing actual frame counts
        
        Note: For dynamo export, we bypass decode_audio() which uses mask_sequence_tensor
              that creates data-dependent masks incompatible with ONNX export.
        """
        print(f"DecoderWrapper.forward: tokens shape: {tokens.shape}, tokens_len shape: {tokens_len.shape}")
        print(f"DecoderWrapper.forward: tokens dtype: {tokens.dtype}, tokens_len dtype: {tokens_len.dtype}")
        
        return self.model.decode(tokens=tokens, tokens_len=tokens_len)

def pad_audio_to_frames(audio: torch.Tensor, samples_per_frame: int = 1764) -> torch.Tensor:
    """
    Pad audio tensor to complete frames externally (required for encoder export).
    
    This is the external padding that users must apply before passing audio to the 
    exported ONNX encoder, since we bypass the model's internal padding to enable
    dynamic shapes.
    
    Args:
        audio: Audio tensor of shape [batch, samples]
        samples_per_frame: Number of samples per frame (typically 1764 for Nano codec)
        
    Returns:
        Padded audio tensor with length as a multiple of samples_per_frame
        
    Example:
        >>> audio = torch.randn(1, 100000)  # ~4.5 seconds at 22050 Hz
        >>> padded_audio = pad_audio_to_frames(audio, samples_per_frame=1764)
        >>> print(padded_audio.shape)  # torch.Size([1, 100548]) = 57 frames * 1764
    """
    current_len = audio.shape[-1]
    num_frames = (current_len + samples_per_frame - 1) // samples_per_frame  # Round up
    target_len = num_frames * samples_per_frame
    padding_needed = target_len - current_len
    
    if padding_needed > 0:
        # Pad with zeros on the right
        audio = torch.nn.functional.pad(audio, (0, padding_needed), mode='constant', value=0.0)
    
    return audio

@torch.inference_mode()
def export_encoder_to_onnx(model, output_path: str, audio_length_sec: float = 10.0, optimize: bool = True, dynamo: bool = True):
    """
    Export the encoder portion of the codec model to ONNX bypassing internal padding.
    
    NOTE: The exported model expects PRE-PADDED audio input that is a multiple of 
    samples_per_frame. You must pad audio externally before passing it to this model.
    
    Args:
        model: NeMo AudioCodecModel instance
        output_path: Path to save the ONNX encoder model
        audio_length_sec: Length of example audio in seconds
    """
    print(f"\n=== Exporting Encoder to ONNX (bypassing internal padding) ===")
    
    # Get configuration from model
    sample_rate = model.sample_rate
    samples_per_frame = model.samples_per_frame
    frame_rate = sample_rate / samples_per_frame
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples per frame: {samples_per_frame}")
    print(f"Frame rate: {frame_rate:.2f} fps")
    
    # NeMo encoder expects shape (batch_size, num_samples)
    num_samples = int(sample_rate * audio_length_sec)
    
    # Pad to complete frames externally
    num_frames = (num_samples + samples_per_frame - 1) // samples_per_frame
    padded_samples = num_frames * samples_per_frame
    
    dummy_audio = torch.randn(1, padded_samples)
    
    print(f"Original audio length: {audio_length_sec} seconds ({num_samples} samples)")
    print(f"Padded to {num_frames} frames ({padded_samples} samples)")
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Sample rate: {sample_rate} Hz")
    
    try:
        # Use the wrapper that bypasses internal padding
        encoder = EncoderWrapper(model)
        encoder.eval()
        
        # audio_len should be the ORIGINAL length before padding (for calculating frame counts)
        # But since we're pre-padding, we use the padded length
        dummy_audio_len = torch.tensor([padded_samples], dtype=torch.int64)
        
        # Test the encoder to get output shape for dynamic axes
        with torch.no_grad():
            test_tokens, test_len = encoder(dummy_audio, dummy_audio_len)
            print(f"Encoder output tokens shape (bypassing padding): {test_tokens.shape}")
            print(f"Encoder output tokens dtype: {test_tokens.dtype}")
            print(f"Encoder output len shape: {test_len.shape}")
            print(f"Expected frames: {num_frames}, Got: {test_len.item()}")
        
        print(f"Exporting encoder to: {output_path}")
        print(f"Exported model requires pre-padded input (multiple of {samples_per_frame} samples)")
        
        dynamic_shapes = {
            'audio': {0: 'batch_size', 1: 'num_samples'},  # Dynamic batch and time (must be multiple of 1764)
            'audio_len': {0: 'batch_size'},  # Dynamic batch for lengths
        }     
        dynamic_axes = {
            'audio': {0: 'batch_size', 1: 'num_samples'},  # Dynamic batch and time (must be multiple of 1764)
            'audio_len': {0: 'batch_size'},  # Dynamic batch for lengths
            'tokens': {0: 'batch_size', 2: 'num_frames'},  # Tokens have shape [batch, codebooks, frames] (matches NeMo model)
            'tokens_len': {0: 'batch_size'}  # Dynamic batch for output lengths (matches NeMo model)
        }     

        if not dynamo:
            torch.onnx.export(
                encoder,
                (dummy_audio, dummy_audio_len),
                output_path,
                input_names=['audio', 'audio_len'],
                output_names=['tokens', 'tokens_len'],
                dynamic_axes=dynamic_axes,
                export_params=True,
                optimize=optimize,
                dynamo=False
            )
        elif optimize:
            torch.onnx.export(
                encoder,
                (dummy_audio, dummy_audio_len),
                output_path,
                input_names=['audio', 'audio_len'],
                output_names=['tokens', 'tokens_len'],
                dynamic_shapes=dynamic_shapes,
                export_params=True,
                optimize=True,
                external_data=False,
                dynamo=True
            )
        else:
            # Export to ONNX with dynamic axes for discrete tokens
            torch.onnx.export(
                encoder,
                (dummy_audio, dummy_audio_len),
                output_path,
                input_names=['audio', 'audio_len'],
                output_names=['tokens', 'tokens_len'],
                dynamic_axes=dynamic_axes,
                export_params=True,
            )

        print(f"Encoder exported successfully")
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model validation passed")
        
    except Exception as e:
        print(f"Error exporting encoder: {str(e)[:100]}")
        print("\nTrying fallback export without dynamo")
        
        # Fallback to TorchScript-based export
        torch.onnx.export(
            encoder,
            (dummy_audio, dummy_audio_len),
            output_path,
            input_names=['audio', 'audio_len'],
            output_names=['tokens', 'tokens_len'],
            dynamic_axes=dynamic_axes,
            export_params=True,
            optimize=optimize,
            #verbose=True,
            dynamo=False
        )
        
        print(f"Encoder exported with fallback method")

@torch.inference_mode()
def export_decoder_to_onnx(model, config, output_path: str, num_frames: int = 100, optimize: bool = True, dynamo: bool = True):
    """
    Export the decoder portion of the codec model to ONNX bypassing internal masking.
    
    NOTE: The exported model bypasses decode_audio() which uses mask_sequence_tensor
    that creates data-dependent masks incompatible with dynamo export. Instead calls
    audio_decoder directly without masking.
    
    Args:
        model: NeMo AudioCodecModel instance
        config: Model configuration dictionary from detect_model_variant
        output_path: Path to save the ONNX decoder model
        num_frames: Number of frames in the encoded representation
        optimize: Whether to optimize during export
        dynamo: Whether to use dynamo export (recommended for dynamic shapes)
    """
    print(f"\n=== Exporting Decoder to ONNX (bypassing internal masking) ===")
    
    num_codebooks = config['num_codebooks']
    codebook_size = config['codebook_size']
    
    print(f"Number of codebooks: {num_codebooks}")
    print(f"Codebook size (max token value): {codebook_size}")
    
    # Dummy input: discrete tokens from encoder, shape: (batch_size, num_codebooks, num_frames)
    # Tokens should be integers in the range [0, codebook_size)
    # Note: Using int64 (long) for ONNX export compatibility, even though NeMo outputs int32
    dummy_tokens = torch.randint(0, codebook_size, (1, num_codebooks, num_frames), dtype=torch.int32)
    
    print(f"Input tokens shape: {dummy_tokens.shape}")
    print(f"Input tokens dtype: {dummy_tokens.dtype}")
    print(f"Num codebooks: {num_codebooks}")
    print(f"Num frames: {num_frames}")
    print(f"Token range: [{dummy_tokens.min().item()}, {dummy_tokens.max().item()}]")
    
    try:
        decoder = DecoderWrapper(model)
        decoder.eval()
        
        # Create tokens_len tensor as a separate input
        dummy_tokens_len = torch.tensor([num_frames], dtype=torch.int64)

        print(f"Dummy tokens_len shape: {dummy_tokens_len.shape}")
        
        # Test the decoder to get output shape
        with torch.no_grad():
            test_audio, test_audio_len = decoder(dummy_tokens, dummy_tokens_len)
            print(f"Decoder output audio shape: {test_audio.shape}")
            print(f"Decoder output audio dtype: {test_audio.dtype}")
            print(f"Decoder output audio_len shape: {test_audio_len.shape}")
        
        dynamic_shapes = {
            'tokens': {0: torch.export.Dim("batch_size"), 2: torch.export.Dim("num_frames")},
            'tokens_len': {0: torch.export.Dim("batch_size")},
        }
        
        dynamic_axes = {
            'tokens': {0: 'batch_size', 2: 'num_frames'},  # Tokens input (matches NeMo model)
            'tokens_len': {0: 'batch_size'},  # Dynamic batch for token lengths
            'audio': {0: 'batch_size', 1: 'num_samples'},  # Audio output (matches NeMo model)
            'audio_len': {0: 'batch_size'}  # Dynamic batch for audio lengths (matches NeMo model)
        } 
        path = str(output_path + "op_nody_decoder.onnx")
        print(f"Exporting decoder to: {path}")

        if not dynamo:
            torch.onnx.export(
                decoder,
                (dummy_tokens, dummy_tokens_len),
                output_path,
                input_names=['tokens', 'tokens_len'],
                output_names=['audio', 'audio_len'],
                dynamic_axes=dynamic_axes,
                export_params=True,
                optimize=optimize,
                dynamo=False
            )
        elif optimize:
            # Keep int32 dummy tokens to matches encoder output
            torch.onnx.export(
                decoder,
                (dummy_tokens, dummy_tokens_len),
                output_path,
                input_names=['tokens', 'tokens_len'],
                output_names=['audio', 'audio_len'],
                export_params=True,
                verbose=True,
                dynamo=True,
                optimize=True,
                external_data=False,
                dynamic_shapes=dynamic_shapes
            )
        else:
            # Export to ONNX with dynamic axes for discrete tokens
            torch.onnx.export(
                decoder,
                (dummy_tokens, dummy_tokens_len),
                output_path,
                input_names=['tokens', 'tokens_len'],
                output_names=['audio', 'audio_len'],
                dynamic_axes=dynamic_axes,
                export_params=True,
                verbose=False,
                external_data=False,
                dynamo=True
            )

        print(f"Decoder exported successfully")
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model validation passed")
        
    except Exception as e:
        print(f"Error exporting decoder: {e}")
        print("\nTrying fallback export without dynamo")
        
        # Fallback to TorchScript-based export
        torch.onnx.export(
            decoder,
            (dummy_tokens, dummy_tokens_len),
            output_path,
            input_names=['tokens', 'tokens_len'],
            output_names=['audio', 'audio_len'],
            dynamic_axes=dynamic_axes,
            export_params=True,
            verbose=False,
            optimize=optimize,
            dynamo=False
        )
        
        print(f"Decoder exported with fallback method")

def test_onnx_inference(encoder_path: str, decoder_path: str, 
                       model):
    """
    Test the exported ONNX models with ONNX Runtime.
    
    Args:
        encoder_path: Path to encoder ONNX model
        decoder_path: Path to decoder ONNX model
        model: Original NeMo model for configuration
    """
    print(f"\n=== Testing ONNX Inference ===")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Get configuration from model
        sample_rate = model.sample_rate
        samples_per_frame = model.samples_per_frame
        
        # Create test audio padded to complete frames - 1 second of audio
        num_frames = int(sample_rate / samples_per_frame)
        test_audio_samples = num_frames * samples_per_frame
        test_audio = np.random.randn(1, test_audio_samples).astype(np.float32)
        test_audio_len = np.array([test_audio_samples], dtype=np.int64)
        
        print(f"Testing encoder...")
        encoder_session = ort.InferenceSession(encoder_path)
        encoder_inputs = {
            encoder_session.get_inputs()[0].name: test_audio,
            encoder_session.get_inputs()[1].name: test_audio_len
        }
        encoder_outputs = encoder_session.run(None, encoder_inputs)
        encoded_tokens = encoder_outputs[0]
        encoded_len = encoder_outputs[1]
        print(f"  Input audio shape: {test_audio.shape}")
        print(f"  Input audio_len: {test_audio_len}")
        print(f"  Encoded tokens shape: {encoded_tokens.shape}")
        print(f"  Encoded tokens dtype: {encoded_tokens.dtype}")
        print(f"  Encoded len: {encoded_len}")
        print(f"  Encoded len dtype: {encoded_len.dtype}")
        print(f"  Token range: [{encoded_tokens.min()}, {encoded_tokens.max()}]")
        
        print(f"\nTesting decoder")
        decoder_session = ort.InferenceSession(decoder_path)
        decoder_inputs = {
            decoder_session.get_inputs()[0].name: encoded_tokens,
            decoder_session.get_inputs()[1].name: encoded_len
        }
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        reconstructed_audio = decoder_outputs[0]
        reconstructed_audio_len = decoder_outputs[1]
        print(f"  Reconstructed audio shape: {reconstructed_audio.shape}")
        print(f"  Reconstructed audio dtype: {reconstructed_audio.dtype}")
        print(f"  Reconstructed audio_len: {reconstructed_audio_len}")
        print(f"  Audio range: [{reconstructed_audio.min():.4f}, {reconstructed_audio.max():.4f}]")
        
        # Simple quality check
        if np.isnan(reconstructed_audio).any() or np.isinf(reconstructed_audio).any():
            print(f"  WARNING: Output contains NaN or Inf values")
        else:
            print(f"  Output is numerically clean")
        
        print(f"\nFull pipeline test completed successfully")
        
    except ImportError:
        print("ONNX Runtime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"Error during ONNX inference test: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Export NeMo Nano Codec to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NeMo .nemo checkpoint file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_models",
        help="Directory to save ONNX models (default: ./onnx_models)"
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        default=False,
        help="Skip exporting encoder model"
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        default=False,
        help="Skip exporting decoder model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test exported ONNX models"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimization during export"
    )
    parser.add_argument(
        "--no-dynamo",
        action="store_true",
        help="Disable dynamo during export"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("NeMo Nano Codec to ONNX Exporter")
    print("=" * 70)
    
    model, config = load_nemo_model(args.checkpoint)

    model_variant = config['variant']
    print(f"\nModel variant: {model_variant}")
    

    if not args.skip_encoder:
        encoder_path = output_dir / f"encoder.onnx"
        export_encoder_to_onnx(model, str(encoder_path), optimize=not args.no_optimize, dynamo=not args.no_dynamo)
    
    if not args.skip_decoder:
        decoder_path = output_dir / f"decoder.onnx"
        export_decoder_to_onnx(model, config, str(decoder_path), optimize=not args.no_optimize, dynamo=not args.no_dynamo)
    
    if args.test:
        encoder_path = output_dir / f"encoder.onnx"
        decoder_path = output_dir / f"decoder.onnx"
        test_onnx_inference(str(encoder_path), str(decoder_path), model)
    
    print("\n" + "=" * 70)
    print("Export completed")
    print(f"Models saved to: {output_dir.absolute()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
