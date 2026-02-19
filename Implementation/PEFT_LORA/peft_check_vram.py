"""
peft_check_vram.py — Pre-flight VRAM check for LoRA / PEFT fine-tuning.

╔══════════════════════════════════════════════════════════════════════════╗
║  HOW LoRA CHANGES VRAM USAGE                                             ║
║                                                                          ║
║  In full fine-tuning, you need VRAM for:                                 ║
║    1. Model weights            (e.g., 2.5 GB for 1B model in bf16)       ║
║    2. Gradients                (same size as weights: 2.5 GB)            ║
║    3. AdamW optimizer states   (2× fp32 copies: 10 GB for 1B model)      ║
║    4. Activations              (varies, ~1-4 GB with grad checkpointing)  ║
║    TOTAL: ~16-18 GB for 1B model — tight on a 3090                       ║
║                                                                          ║
║  With LoRA:                                                              ║
║    1. Model weights            SAME (2.5 GB) — still loaded in memory   ║
║    2. Gradients                ONLY for adapters (~0.02 GB)              ║
║    3. AdamW optimizer states   ONLY for adapters (~0.08 GB)              ║
║    4. Activations              SIMILAR (gradients still flow forward)    ║
║    TOTAL: ~8-10 GB for same model — 2× headroom!                         ║
║                                                                          ║
║  KEY INSIGHT: The base model is FROZEN (no gradients stored for it).     ║
║  Only the tiny adapter parameters (A and B matrices) need gradients      ║
║  and optimizer states. The base model weights are read-only.             ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python peft_check_vram.py
"""

import torch
import yaml
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# LoRA Parameter Calculator
# ──────────────────────────────────────────────────────────────────────────────

def calculate_lora_params(config: dict) -> int:
    """
    Calculate the number of trainable LoRA parameters given the config.

    For each target module at rank r:
      - Matrix A: [r × d_in]
      - Matrix B: [d_out × r]
      - Total per module: r × (d_in + d_out)

    Architecture dimensions for Llama-3.2-1B:
      - num_layers:   16
      - hidden_dim:   2048  (d_model)
      - num_heads:    32    (Q heads)
      - num_kv_heads: 8     (GQA K/V heads — Grouped Query Attention)
      - head_dim:     64    (hidden_dim / num_heads)
      - intermediate: 8192  (FFN hidden size)
    """
    r = config.get("lora_r", 16)
    target_modules = config.get("lora_target_modules", ["q_proj", "v_proj"])

    # Dimensions for Llama-3.2-1B
    # These come from the model's config.json
    model_dims = {
        "unsloth/Llama-3.2-1B-Instruct":        {"d": 2048, "kv_d": 256,  "ffn": 8192,  "layers": 16},
        "meta-llama/Llama-3.2-1B-Instruct":     {"d": 2048, "kv_d": 256,  "ffn": 8192,  "layers": 16},
        "unsloth/Llama-3.2-3B-Instruct":        {"d": 3072, "kv_d": 1024, "ffn": 8192,  "layers": 28},
        "unsloth/Meta-Llama-3.1-8B-Instruct":   {"d": 4096, "kv_d": 1024, "ffn": 14336, "layers": 32},
        "Qwen/Qwen2.5-7B-Instruct":             {"d": 3584, "kv_d": 512,  "ffn": 18944, "layers": 28},
        "HuggingFaceTB/SmolLM2-360M-Instruct":  {"d": 960,  "kv_d": 320,  "ffn": 2560,  "layers": 32},
        "openai-community/gpt2":                {"d": 768,  "kv_d": 768,  "ffn": 3072,  "layers": 12},
    }

    model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
    dims = model_dims.get(model_name)

    if dims is None:
        print(f"  ⚠️  Unknown model '{model_name}' — cannot calculate LoRA params precisely.")
        return 0

    d = dims["d"]         # hidden dimension (e.g. 2048)
    kv_d = dims["kv_d"]  # K/V head dim (smaller with GQA)
    ffn = dims["ffn"]     # FFN intermediate size
    L = dims["layers"]    # number of transformer layers

    # Module shapes: (d_in, d_out) for each projection
    module_shapes = {
        "q_proj":    (d, d),
        "k_proj":    (d, kv_d),
        "v_proj":    (d, kv_d),
        "o_proj":    (d, d),
        "gate_proj": (d, ffn),
        "up_proj":   (d, ffn),
        "down_proj": (ffn, d),
    }

    total_params = 0
    print(f"\n  LoRA Adapter Parameter Breakdown (r={r}):")
    print(f"  {'Module':<15} {'d_in':>6} {'d_out':>6} {'per layer':>12} {'× layers':>10} {'total':>12}")
    print(f"  {'─'*65}")

    for module in target_modules:
        if module not in module_shapes:
            print(f"  ⚠️  Unknown module '{module}' — skipping.")
            continue

        d_in, d_out = module_shapes[module]
        params_per_layer = r * (d_in + d_out)   # A matrix + B matrix
        params_total = params_per_layer * L

        print(f"  {module:<15} {d_in:>6,} {d_out:>6,} {params_per_layer:>12,} {L:>10} {params_total:>12,}")
        total_params += params_total

    print(f"  {'─'*65}")
    print(f"  {'TOTAL LoRA params':<15} {'':>6} {'':>6} {'':>12} {'':>10} {total_params:>12,}")
    return total_params


def estimate_vram_lora(config: dict, lora_params: int) -> dict:
    """
    Estimate VRAM usage for LoRA fine-tuning.

    The key difference vs full fine-tuning:
      - Base model weights:   loaded in bf16 (frozen, no grad)
      - LoRA adapter params:  trained in bf16/fp32
      - Gradients:            only for LoRA params (not base model)
      - Optimizer states:     only for LoRA params (AdamW: 2× fp32 states)
      - Activations:          roughly same as full FT (gradients still
                              flow through frozen layers forward pass)
    """
    known_model_params = {
        "unsloth/Llama-3.2-1B-Instruct":        1_240_000_000,
        "meta-llama/Llama-3.2-1B-Instruct":     1_240_000_000,
        "unsloth/Llama-3.2-3B-Instruct":        3_210_000_000,
        "unsloth/Meta-Llama-3.1-8B-Instruct":   8_030_000_000,
        "Qwen/Qwen2.5-7B-Instruct":             7_610_000_000,
        "HuggingFaceTB/SmolLM2-360M-Instruct":  360_000_000,
        "openai-community/gpt2":                124_000_000,
    }

    model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
    total_params = known_model_params.get(model_name, 1_240_000_000)
    bf16 = config.get("bf16", True)
    grad_checkpoint = config.get("gradient_checkpointing", True)
    batch_size = config.get("per_device_train_batch_size", 4)
    seq_len = config.get("max_seq_length", 512)

    bytes_model = 2 if bf16 else 4  # bf16 = 2 bytes per param

    # 1. Base model weights (frozen, loaded in bf16)
    base_model_mem = total_params * bytes_model

    # 2. LoRA adapter weights (small — trainable)
    adapter_weights_mem = lora_params * bytes_model

    # 3. Gradients — ONLY for LoRA params (base model is requires_grad=False)
    adapter_grad_mem = lora_params * bytes_model

    # 4. Optimizer states — ONLY for LoRA params (AdamW: 2× fp32 copies)
    optimizer_mem = lora_params * 4 * 2  # fp32 momentum + variance

    # 5. Activations — gradients still pass through frozen layers, so activations
    # are still stored (or recomputed with gradient checkpointing)
    activation_mem = batch_size * seq_len * total_params * 0.000002
    if grad_checkpoint:
        activation_mem *= 0.3

    total = base_model_mem + adapter_weights_mem + adapter_grad_mem + optimizer_mem + activation_mem

    return {
        "base_model_weights_gb": base_model_mem / 1e9,
        "adapter_weights_gb": adapter_weights_mem / 1e9,
        "adapter_gradients_gb": adapter_grad_mem / 1e9,
        "optimizer_states_gb": optimizer_mem / 1e9,
        "activations_gb": activation_mem / 1e9,
        "total_estimated_gb": total / 1e9,
    }


def main():
    # ── Load config ──────────────────────────────────────────────────────────
    config_path = Path(__file__).parent / "peft_training_config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "peft_training_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
    r = config.get("lora_r", 16)
    alpha = config.get("lora_alpha", 32)
    target_modules = config.get("lora_target_modules", ["q_proj", "v_proj"])

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  VRAM Estimation: PEFT / LoRA Fine-Tuning")
    print(f"{'='*65}")
    print(f"  Model:         {model_name}")
    print(f"  LoRA Rank:     r={r}  |  alpha={alpha}  |  scale={alpha/r:.1f}")
    print(f"  Target mods:   {target_modules}")
    print(f"  Precision:     {'bf16' if config.get('bf16') else 'fp32'}")
    print(f"  Grad Ckpt:     {config.get('gradient_checkpointing', True)}")
    print(f"  Batch Size:    {config['per_device_train_batch_size']}")
    print(f"  Seq Length:    {config['max_seq_length']}")

    # ── Calculate LoRA params ────────────────────────────────────────────────
    lora_params = calculate_lora_params(config)

    # ── Known total model params ─────────────────────────────────────────────
    known_total = {
        "unsloth/Llama-3.2-1B-Instruct": 1_240_000_000,
        "unsloth/Llama-3.2-3B-Instruct": 3_210_000_000,
        "unsloth/Meta-Llama-3.1-8B-Instruct": 8_030_000_000,
    }
    total_params = known_total.get(model_name, 1_240_000_000)

    pct = (lora_params / total_params) * 100 if total_params > 0 else 0
    print(f"\n  Trainable parameters: {lora_params:,}  ({pct:.2f}% of {total_params/1e9:.1f}B total)")

    # ── VRAM estimate ────────────────────────────────────────────────────────
    est = estimate_vram_lora(config, lora_params)

    print(f"\n  {'─'*50}")
    print(f"  VRAM Breakdown:")
    print(f"  {'─'*50}")
    print(f"  Base model weights (frozen):   {est['base_model_weights_gb']:.2f} GB")
    print(f"  LoRA adapter weights:          {est['adapter_weights_gb']:.4f} GB")
    print(f"  LoRA adapter gradients:        {est['adapter_gradients_gb']:.4f} GB")
    print(f"  Optimizer states (AdamW):      {est['optimizer_states_gb']:.4f} GB")
    print(f"  Activations (est.):            {est['activations_gb']:.2f} GB")
    print(f"  {'─'*50}")
    print(f"  TOTAL ESTIMATED:               {est['total_estimated_gb']:.1f} GB")

    # ── Comparison vs full FT ────────────────────────────────────────────────
    full_ft_estimate = (total_params * 2 + total_params * 2 + total_params * 8) / 1e9
    savings = full_ft_estimate - est["total_estimated_gb"]
    print(f"\n  vs Full Fine-Tuning estimate:  ~{full_ft_estimate:.0f} GB")
    print(f"  Estimated VRAM savings:        ~{savings:.0f} GB ({savings/full_ft_estimate*100:.0f}% reduction)")

    # ── GPU check ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        comfortable = gpu_mem * 0.85

        if est["total_estimated_gb"] < comfortable:
            verdict = f"✅ Fits comfortably! ({gpu_mem - est['total_estimated_gb']:.1f} GB to spare)"
        elif est["total_estimated_gb"] < gpu_mem:
            verdict = f"⚠️  Tight — may fit but watch for OOM. Consider reducing batch size."
        else:
            verdict = f"❌ Won't fit. Reduce batch_size or lora_r."

        print(f"\n  Your GPU:  {gpu_name} ({gpu_mem:.0f} GB)")
        print(f"  Verdict:   {verdict}")
    else:
        print(f"\n  ⚠️  No GPU detected.")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
