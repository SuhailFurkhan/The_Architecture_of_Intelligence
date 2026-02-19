"""
check_vram.py — Pre-flight check: will your model fit on your GPU?

Estimates VRAM needed for full fine-tuning (weights + grads + optimizer).
Run this BEFORE training to avoid OOM surprises.
"""

import torch
import yaml
from pathlib import Path


def estimate_vram(num_params: int, batch_size: int, seq_len: int,
                  bf16: bool = True, grad_checkpoint: bool = True) -> dict:
    """Estimate VRAM usage for full fine-tuning."""

    bytes_per_param = 2 if bf16 else 4  # fp16/bf16 = 2 bytes, fp32 = 4

    # Model weights
    model_mem = num_params * bytes_per_param

    # Gradients (same dtype as weights)
    grad_mem = num_params * bytes_per_param

    # AdamW optimizer states: 2 copies in fp32 (momentum + variance)
    optimizer_mem = num_params * 4 * 2  # always fp32

    # Activations (rough estimate — varies wildly by architecture)
    # Gradient checkpointing reduces this by ~60-70%
    activation_mem = batch_size * seq_len * num_params * 0.000002  # rough heuristic
    if grad_checkpoint:
        activation_mem *= 0.3  # ~70% reduction

    total = model_mem + grad_mem + optimizer_mem + activation_mem

    return {
        "model_weights_gb": model_mem / 1e9,
        "gradients_gb": grad_mem / 1e9,
        "optimizer_states_gb": optimizer_mem / 1e9,
        "activations_gb": activation_mem / 1e9,
        "total_estimated_gb": total / 1e9,
    }


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Known param counts (avoids downloading the model just to check)
    param_counts = {
        "unsloth/Llama-3.2-1B-Instruct": 1_240_000_000,
        "meta-llama/Llama-3.2-1B-Instruct": 1_240_000_000,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1_100_000_000,
        "HuggingFaceTB/SmolLM2-360M-Instruct": 360_000_000,
        "openai-community/gpt2": 124_000_000,
    }

    model_name = config["model_name"]
    num_params = param_counts.get(model_name, None)

    if num_params is None:
        print(f"⚠️  Unknown model '{model_name}'. Add its param count to check_vram.py.")
        return

    print(f"{'='*60}")
    print(f"  VRAM Estimation: Full Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Model:       {model_name}")
    print(f"  Parameters:  {num_params / 1e9:.2f}B")
    print(f"  Precision:   {'bf16' if config.get('bf16') else 'fp32'}")
    print(f"  Grad Ckpt:   {config.get('gradient_checkpointing', False)}")
    print(f"  Batch Size:  {config['per_device_train_batch_size']}")
    print(f"  Seq Length:  {config['max_seq_length']}")
    print()

    est = estimate_vram(
        num_params=num_params,
        batch_size=config["per_device_train_batch_size"],
        seq_len=config["max_seq_length"],
        bf16=config.get("bf16", True),
        grad_checkpoint=config.get("gradient_checkpointing", True),
    )

    print(f"  Model Weights:     {est['model_weights_gb']:.1f} GB")
    print(f"  Gradients:         {est['gradients_gb']:.1f} GB")
    print(f"  Optimizer (AdamW): {est['optimizer_states_gb']:.1f} GB")
    print(f"  Activations:       {est['activations_gb']:.1f} GB (estimated)")
    print(f"  {'─'*40}")
    print(f"  TOTAL ESTIMATED:   {est['total_estimated_gb']:.1f} GB")
    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        fits = "✅ Should fit!" if est['total_estimated_gb'] < gpu_mem * 0.90 else "⚠️  Tight! May OOM."
        if est['total_estimated_gb'] > gpu_mem:
            fits = "❌ Won't fit. Use a smaller model or reduce seq_length."
        print(f"  Your GPU:  {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  Verdict:   {fits}")
    else:
        print(f"  ⚠️  No GPU detected. Training will be extremely slow on CPU.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()