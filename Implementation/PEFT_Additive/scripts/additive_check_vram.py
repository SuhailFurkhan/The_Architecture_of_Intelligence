"""
additive_check_vram.py — VRAM estimation for Bottleneck Adapters and (IA)³.

╔══════════════════════════════════════════════════════════════════════════╗
║  HOW ADDITIVE METHODS CHANGE VRAM vs LoRA                                ║
║                                                                          ║
║  LoRA:                                                                   ║
║    Frozen base (2.5 GB) + LoRA params (~40 MB) + LoRA grads + optimizer  ║
║    Activations: similar to full FT (gradients flow through frozen layers)║
║                                                                          ║
║  Bottleneck Adapters:                                                    ║
║    Frozen base (2.5 GB) + Adapter params (~64 MB) + Adapter grads        ║
║    + optimizer (just for adapters)                                       ║
║    Activations: SLIGHTLY MORE than LoRA — adapters add extra compute     ║
║    graph nodes that hold intermediate values (down-proj output before    ║
║    GELU, GELU output before up-proj, and the residual input h_in)        ║
║                                                                          ║
║  (IA)³:                                                                  ║
║    Frozen base (2.5 GB) + l vectors (~0.5 MB) + l grads (~0.5 MB)        ║
║    + optimizer (~2 MB) + activations (very similar to full FT)           ║
║    LOWEST training footprint of any PEFT method                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python additive_check_vram.py
"""

import torch
import yaml
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Model dimension registry
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIMS = \
    {
    "unsloth/Llama-3.2-1B-Instruct":      {"d": 2048, "kv_d": 256,  "ffn": 8192,  "layers": 16, "total": 1_240_000_000},
    "meta-llama/Llama-3.2-1B-Instruct":   {"d": 2048, "kv_d": 256,  "ffn": 8192,  "layers": 16, "total": 1_240_000_000},
    "unsloth/Llama-3.2-3B-Instruct":      {"d": 3072, "kv_d": 1024, "ffn": 8192,  "layers": 28, "total": 3_210_000_000},
    "unsloth/Meta-Llama-3.1-8B-Instruct": {"d": 4096, "kv_d": 1024, "ffn": 14336, "layers": 32, "total": 8_030_000_000},
    "HuggingFaceTB/SmolLM2-360M-Instruct":{"d": 960,  "kv_d": 320,  "ffn": 2560,  "layers": 32, "total": 360_000_000},
    }


def calculate_bottleneck_params(dims: dict, bottleneck_dim: int,
                                 placement: str) -> int:
    """
    Calculate trainable parameter count for Bottleneck Adapters.

    Per adapter module (Pfeiffer — 1 per layer):
      W_down: [hidden_dim × bottleneck_dim]
      b_down: [bottleneck_dim]
      W_up:   [bottleneck_dim × hidden_dim]
      b_up:   [hidden_dim]

    Houlsby (2 per layer) doubles this.
    """
    d = dims["d"]
    L = dims["layers"]
    adapters_per_layer = 2 if placement == "after_attn_and_ffn" else 1

    per_adapter = (d * bottleneck_dim) + bottleneck_dim + (bottleneck_dim * d) + d
    total = per_adapter * adapters_per_layer * L

    print(f"\n  Bottleneck Adapter Parameter Breakdown:")
    print(f"  {'Component':<25} {'Per adapter':>12}  {'× layers':>8}  {'× per_layer':>10}  {'Total':>12}")
    print(f"  {'─'*72}")
    print(f"  {'W_down':<25} {d * bottleneck_dim:>12,}  {L:>8}  {adapters_per_layer:>10}  {d * bottleneck_dim * L * adapters_per_layer:>12,}")
    print(f"  {'b_down':<25} {bottleneck_dim:>12,}  {L:>8}  {adapters_per_layer:>10}  {bottleneck_dim * L * adapters_per_layer:>12,}")
    print(f"  {'W_up':<25} {bottleneck_dim * d:>12,}  {L:>8}  {adapters_per_layer:>10}  {bottleneck_dim * d * L * adapters_per_layer:>12,}")
    print(f"  {'b_up':<25} {d:>12,}  {L:>8}  {adapters_per_layer:>10}  {d * L * adapters_per_layer:>12,}")
    print(f"  {'─'*72}")
    print(f"  {'TOTAL':<25} {per_adapter:>12,}  {'':>8}  {'':>10}  {total:>12,}")

    return total


def calculate_ia3_params(dims: dict) -> int:
    """
    Calculate trainable parameter count for (IA)³.

    Per layer:
      l_k:  [kv_d]   — rescales key projections
      l_v:  [kv_d]   — rescales value projections
      l_ff: [ffn]    — rescales FFN gate activations
    """
    kv_d = dims["kv_d"]
    ffn = dims["ffn"]
    L = dims["layers"]

    per_layer = kv_d + kv_d + ffn
    total = per_layer * L

    print(f"\n  (IA)³ Parameter Breakdown:")
    print(f"  {'Vector':<15} {'Shape':>12}  {'Per layer':>10}  {'× layers':>8}  {'Total':>12}")
    print(f"  {'─'*62}")
    print(f"  {'l_k (keys)':<15} {f'[{kv_d}]':>12}  {kv_d:>10,}  {L:>8}  {kv_d * L:>12,}")
    print(f"  {'l_v (values)':<15} {f'[{kv_d}]':>12}  {kv_d:>10,}  {L:>8}  {kv_d * L:>12,}")
    print(f"  {'l_ff (ffn)':<15} {f'[{ffn}]':>12}  {ffn:>10,}  {L:>8}  {ffn * L:>12,}")
    print(f"  {'─'*62}")
    print(f"  {'TOTAL':<15} {'':>12}  {per_layer:>10,}  {'':>8}  {total:>12,}")

    return total


def estimate_vram(config: dict, trainable_params: int, dims: dict) -> dict:
    """Estimate VRAM usage for additive PEFT training."""
    total_params = dims["total"]
    bf16 = config.get("bf16", True)
    grad_checkpoint = config.get("gradient_checkpointing", True)
    batch_size = config.get("per_device_train_batch_size", 4)
    seq_len = config.get("max_seq_length", 512)
    bytes_model = 2 if bf16 else 4

    # 1. Frozen base model weights
    base_mem = total_params * bytes_model

    # 2. Trainable param weights
    trainable_weights_mem = trainable_params * bytes_model

    # 3. Gradients (only for trainable params)
    grad_mem = trainable_params * bytes_model

    # 4. Optimizer states (AdamW: 2 × fp32 per trainable param)
    optimizer_mem = trainable_params * 4 * 2

    # 5. Activations
    # Bottleneck adapters store slightly more activations than LoRA because
    # the adapter module adds extra nodes to the compute graph:
    #   - h_in (the residual)
    #   - W_down(h) before GELU
    #   - GELU(W_down(h)) before W_up
    method = config.get("adapter_method", "bottleneck")
    activation_base = batch_size * seq_len * total_params * 0.000002
    if grad_checkpoint:
        activation_base *= 0.3
    if method == "bottleneck":
        activation_mem = activation_base * 1.15  # ~15% extra for adapter activations
    else:
        activation_mem = activation_base  # (IA)³ adds negligible activation overhead

    total = base_mem + trainable_weights_mem + grad_mem + optimizer_mem + activation_mem

    return {
        "base_model_gb": base_mem / 1e9,
        "trainable_weights_gb": trainable_weights_mem / 1e9,
        "gradients_gb": grad_mem / 1e9,
        "optimizer_gb": optimizer_mem / 1e9,
        "activations_gb": activation_mem / 1e9,
        "total_gb": total / 1e9,
    }


def main():
    config_path = Path(__file__).parent.parent / "configs" / "additive_training_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
    method = config.get("adapter_method", "bottleneck")
    dims = MODEL_DIMS.get(model_name)

    if dims is None:
        print(f"  ⚠️  Unknown model '{model_name}'.")
        return

    print(f"{'='*65}")
    print(f"  VRAM Estimation: Additive PEFT ({method.upper()})")
    print(f"{'='*65}")
    print(f"  Model:     {model_name}")
    print(f"  Method:    {method}")
    print(f"  Precision: {'bf16' if config.get('bf16') else 'fp32'}")
    print(f"  Grad Ckpt: {config.get('gradient_checkpointing', True)}")
    print(f"  Batch:     {config['per_device_train_batch_size']}")
    print(f"  Seq len:   {config['max_seq_length']}")

    if method == "bottleneck":
        bottleneck_dim = config.get("bottleneck_dim", 64)
        placement = config.get("adapter_placement", "after_ffn")
        print(f"  Bottleneck dim: {bottleneck_dim}")
        print(f"  Placement:      {placement}")
        trainable_params = calculate_bottleneck_params(dims, bottleneck_dim, placement)
    else:  # ia3
        trainable_params = calculate_ia3_params(dims)

    pct = trainable_params / dims["total"] * 100
    print(f"\n  Trainable parameters: {trainable_params:,}  ({pct:.3f}% of {dims['total']/1e9:.1f}B)")

    est = estimate_vram(config, trainable_params, dims)

    print(f"\n  {'─'*50}")
    print(f"  VRAM Breakdown:")
    print(f"  {'─'*50}")
    print(f"  Frozen base model:           {est['base_model_gb']:.2f} GB")
    print(f"  Trainable weights:           {est['trainable_weights_gb']:.4f} GB")
    print(f"  Gradients:                   {est['gradients_gb']:.4f} GB")
    print(f"  Optimizer states (AdamW):    {est['optimizer_gb']:.4f} GB")
    print(f"  Activations (estimated):     {est['activations_gb']:.2f} GB")
    print(f"  {'─'*50}")
    print(f"  TOTAL ESTIMATED:             {est['total_gb']:.1f} GB")

    full_ft = (dims["total"] * 2 + dims["total"] * 2 + dims["total"] * 8) / 1e9
    savings = full_ft - est["total_gb"]
    print(f"\n  vs Full Fine-Tuning estimate: ~{full_ft:.0f} GB")
    print(f"  Estimated savings:            ~{savings:.0f} GB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        comfortable = gpu_mem * 0.85
        if est["total_gb"] < comfortable:
            verdict = f"✅ Fits comfortably! ({gpu_mem - est['total_gb']:.1f} GB to spare)"
        elif est["total_gb"] < gpu_mem:
            verdict = f"⚠️  Tight — may fit. Reduce batch_size or bottleneck_dim."
        else:
            verdict = f"❌ Won't fit. Reduce bottleneck_dim or batch_size."
        print(f"\n  Your GPU: {gpu_name} ({gpu_mem:.0f} GB)")
        print(f"  Verdict:  {verdict}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
