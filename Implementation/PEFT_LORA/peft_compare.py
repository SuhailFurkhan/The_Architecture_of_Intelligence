"""
peft_compare.py — Side-by-side: Original vs LoRA Fine-Tuned model.

╔══════════════════════════════════════════════════════════════════════════╗
║  WHAT'S DIFFERENT FROM THE FULL FT compare.py?                          ║
║                                                                          ║
║  Full FT comparison:                                                     ║
║    - Load original model                                                 ║
║    - Load fine-tuned model (full, 2.5 GB)                                ║
║    - Compare weight drifts across ALL layers                             ║
║    - Drift should be uniform (all layers got gradients)                  ║
║                                                                          ║
║  LoRA comparison (this file):                                            ║
║    - Load original model                                                 ║
║    - Load LoRA adapter + base model, then MERGE to compare fairly        ║
║    - Compare weight drifts across layers                                 ║
║    - EXPECTED: drift only in LoRA target layers (q,k,v,o,gate,up,down) ║
║    - NON-target layers (embed, layernorm, lm_head) should drift ≈ 0     ║
║                                                                          ║
║  BONUS: This file also shows the raw adapter matrices (A and B)          ║
║  so you can see the structure of what LoRA actually learned.             ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python peft_compare.py
    python peft_compare.py --adapter_path ./outputs/llama-3.2-1B-lora/final
"""

import argparse
import json
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


TEST_PROMPTS = [
    "What is machine learning? Explain in 2 sentences.",
    "Write a short poem about the ocean.",
    "Translate 'Good morning, how are you?' to French.",
    "List 3 benefits of regular exercise.",
    "Explain the difference between a list and a tuple in Python.",
]


def load_original(model_name: str):
    """Load the original (unmodified) base model."""
    print(f"  📦 Loading ORIGINAL model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_lora_and_merge(adapter_path: str):
    """
    Load the LoRA adapter and MERGE it into the base model.

    Why merge before comparison?
      The weight drift analysis compares W_original vs W_finetuned.
      For a LoRA model, W_finetuned = W + B·A·(alpha/r).
      To see this delta in the same space as the base weights,
      we need to merge first so we can do: W_finetuned - W_original.
    """
    from peft import PeftModel

    # Read base model name from adapter config
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path")
        r = adapter_cfg.get("r", "?")
        alpha = adapter_cfg.get("lora_alpha", "?")
        targets = adapter_cfg.get("target_modules", [])
        print(f"  📋 Adapter config: r={r}, alpha={alpha}, targets={targets}")
    else:
        # Might be a merged model already
        base_model_name = None

    if base_model_name:
        print(f"  📦 Loading base model: {base_model_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print(f"  🔧 Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base, adapter_path)
        print(f"  🔀 Merging LoRA weights (B·A → W) for comparison...")
        model = model.merge_and_unload()
    else:
        # Already merged — load directly
        print(f"  📦 Loading merged model from: {adapter_path}")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def show_lora_matrices(adapter_path: str):
    """
    Print statistics about the raw A and B adapter matrices.

    ┌──────────────────────────────────────────────────────────────────┐
    │  What are we looking at?                                         │
    │                                                                  │
    │  For each adapted layer, PEFT saves:                             │
    │    lora_A.weight: shape [r, d_in]   — initialized N(0, 1/r)     │
    │    lora_B.weight: shape [d_out, r]  — initialized to 0          │
    │                                                                  │
    │  After training, B is no longer zero — it's learned the delta.   │
    │  The effective weight change is: B·A × (alpha / r)               │
    │                                                                  │
    │  Looking at |B·A| tells you:                                     │
    │    - Large norm → LoRA made big changes to this layer            │
    │    - Small norm → this layer barely changed (maybe less important)│
    └──────────────────────────────────────────────────────────────────┘
    """
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"  ⚠️  No adapter_config.json found — skipping adapter matrix analysis.")
        return

    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)

    r = adapter_cfg.get("r", 16)
    alpha = adapter_cfg.get("lora_alpha", 32)
    scale = alpha / r

    print(f"\n  {'─'*65}")
    print(f"  LORA ADAPTER MATRIX ANALYSIS")
    print(f"  {'─'*65}")
    print(f"  Rank r={r}  |  Alpha={alpha}  |  Scale={scale:.2f}")
    print(f"  {'─'*65}")

    # Load raw adapter weights (safetensors or pytorch)
    from safetensors.torch import load_file as load_safetensors
    adapter_model_path = Path(adapter_path) / "adapter_model.safetensors"

    if not adapter_model_path.exists():
        adapter_model_path = Path(adapter_path) / "adapter_model.bin"
        if not adapter_model_path.exists():
            print(f"  ⚠️  Adapter weights file not found.")
            return
        weights = torch.load(adapter_model_path, map_location="cpu")
    else:
        weights = load_safetensors(str(adapter_model_path))

    # Group by layer and compute B·A product norm
    # Key format in PEFT: "base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"
    layers = {}
    for key, tensor in weights.items():
        if "lora_A" in key or "lora_B" in key:
            # Extract layer identifier
            parts = key.split(".")
            # Find the layer number and module name
            module_key = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            if module_key not in layers:
                layers[module_key] = {}
            if "lora_A" in key:
                layers[module_key]["A"] = tensor
            elif "lora_B" in key:
                layers[module_key]["B"] = tensor

    print(f"\n  {'Layer':<60} {'‖B·A‖':<12} {'rank'}")
    print(f"  {'─'*80}")

    layer_norms = []
    for module_key in sorted(layers.keys()):
        mats = layers[module_key]
        if "A" in mats and "B" in mats:
            A = mats["A"].float()   # [r, d_in]
            B = mats["B"].float()   # [d_out, r]
            delta = (B @ A) * scale  # [d_out, d_in] — the effective weight change
            norm = delta.norm().item()
            rank = A.shape[0]
            # Trim the key for display
            display_key = module_key.replace("base_model.model.model.", "")[-60:]
            print(f"  {display_key:<60} {norm:<12.4f} {rank}")
            layer_norms.append((display_key, norm))

    if layer_norms:
        layer_norms.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 most-adapted layers (largest ‖B·A‖):")
        for name, norm in layer_norms[:5]:
            bar = "█" * int(norm * 100)
            print(f"    {norm:.4f}  {bar[:40]:40s}  {name}")

        print(f"\n  Bottom 5 least-adapted layers:")
        for name, norm in layer_norms[-5:]:
            print(f"    {norm:.6f}  {name}")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a response (same as compare.py)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )

    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, temperature=0.7,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compare(original_path: str, adapter_path: str):
    """Run comparison between original and LoRA fine-tuned model."""
    print("=" * 70)
    print("  BEFORE vs AFTER — LoRA Fine-Tuning Comparison")
    print("=" * 70)

    # ── Load models ────────────────────────────────────────────────────────
    original_model, original_tok = load_original(original_path)
    lora_model, lora_tok = load_lora_and_merge(adapter_path)
    print()

    # ── Qualitative response comparison ───────────────────────────────────
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"{'─'*70}")
        print(f"  PROMPT {i}: {prompt}")
        print(f"{'─'*70}")

        original_resp = generate(original_model, original_tok, prompt)
        lora_resp     = generate(lora_model, lora_tok, prompt)

        print(f"\n  🔵 BEFORE (original):     {original_resp[:400]}")
        print(f"\n  🟢 AFTER  (LoRA adapted): {lora_resp[:400]}")
        print()

    # ── Weight drift analysis (merged LoRA vs original) ────────────────────
    print(f"{'─'*70}")
    print(f"  WEIGHT DRIFT ANALYSIS (post-merge)")
    print(f"{'─'*70}")
    print(f"  Compares: W_original vs W_finetuned (= W + B·A·scale)")
    print(f"  Non-target layers (embed, norms) should drift ≈ 0\n")

    orig_state = original_model.state_dict()
    lora_state = lora_model.state_dict()

    drifts = []
    for key in orig_state:
        if key not in lora_state:
            continue
        if orig_state[key].dtype in (torch.bfloat16, torch.float16, torch.float32):
            diff = (lora_state[key].float() - orig_state[key].float()).abs().mean().item()
            drifts.append((key, diff))

    drifts.sort(key=lambda x: x[1], reverse=True)

    print(f"  Top 10 most-changed layers (LoRA target modules expected here):")
    for name, drift in drifts[:10]:
        bar = "█" * int(drift * 5000)
        print(f"    {drift:.6f}  {bar:20s}  {name}")

    print(f"\n  Bottom 5 least-changed layers (non-target layers should be here):")
    for name, drift in drifts[-5:]:
        print(f"    {drift:.6f}  {name}")

    avg_drift = sum(d for _, d in drifts) / len(drifts)
    print(f"\n  Average weight drift: {avg_drift:.6f}")
    print(f"  Total layers compared: {len(drifts)}")

    # ── Adapter matrix analysis ────────────────────────────────────────────
    show_lora_matrices(adapter_path)

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Compare original vs LoRA fine-tuned model")

    config_path = Path(__file__).parent / "peft_training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    parser.add_argument("--original_path", default=config["model_name"])
    parser.add_argument("--adapter_path",
                        default=config["output_dir"] + "/final")
    args = parser.parse_args()

    compare(args.original_path, args.adapter_path)


if __name__ == "__main__":
    main()
