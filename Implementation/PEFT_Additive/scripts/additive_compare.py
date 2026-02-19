"""
additive_compare.py — Analyze what Bottleneck Adapters and (IA)³ learned.

╔══════════════════════════════════════════════════════════════════════════╗
║  WHAT TO ANALYZE — HOW THIS DIFFERS FROM LoRA COMPARE                   ║
║                                                                          ║
║  LoRA analysis:                                                          ║
║    • Weight drift in existing matrices (W_q, W_k, etc.)                 ║
║    • Adapter matrix norms: ‖B·A‖ per layer                              ║
║    • Which layers adapted most vs least                                 ║
║                                                                          ║
║  Bottleneck Adapter analysis:                                            ║
║    • Adapter weight magnitudes: ‖W_down‖, ‖W_up‖ per layer             ║
║    • Base model weight drift: SHOULD BE ZERO (they're frozen)           ║
║    • Effective output magnitude: ‖W_up · GELU(W_down)‖                  ║
║    • Input/output similarity: how much does adapter change h?           ║
║                                                                          ║
║  (IA)³ analysis:                                                         ║
║    • l vector values: which dimensions are amplified vs suppressed?     ║
║    • Distance from ones: ‖l - 1‖ measures total deviation from identity ║
║    • Top amplified/suppressed feature dimensions per layer              ║
║    • Base model weight drift: SHOULD BE ZERO (they're frozen)           ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python additive_compare.py
    python additive_compare.py --adapter_path ./outputs/my-adapter
    python additive_compare.py --method ia3
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_models(adapter_path: str):
    """Load base model and adapted model for comparison."""
    config_file = Path(adapter_path) / "adapter_config.json"
    with open(config_file) as f:
        cfg = json.load(f)
    base_model_name = cfg["base_model_name_or_path"]
    peft_type = cfg.get("peft_type", "").lower()

    if "bottleneck" in peft_type or "adapter" in peft_type:
        method = "bottleneck"
    elif "ia3" in peft_type:
        method = "ia3"
    else:
        method = "unknown"

    print(f"  Base model:   {base_model_name}")
    print(f"  Adapter path: {adapter_path}")
    print(f"  Method:       {method}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # CPU for analysis — we don't need GPU speed
    )
    adapted_model = PeftModel.from_pretrained(base_model, adapter_path)

    return base_model, adapted_model, tokenizer, method


# ──────────────────────────────────────────────────────────────────────────────
# Bottleneck Adapter Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_bottleneck_adapters(adapted_model):
    """
    Analyze what the Bottleneck Adapters learned.

    Key metrics:
    1. W_down norm:  ‖W_down‖_F  — how much the down-projection is using
    2. W_up norm:    ‖W_up‖_F    — how much the up-projection is using
    3. Effective delta magnitude: ‖W_up · random_input‖ (rough proxy)
    4. Rank analysis: approximate rank of W_down and W_up (should be full
       since they're small matrices)
    5. Deviation of W_up from zero (initialized to zeros):
       large deviation = adapter learned significant transformation
       small deviation = adapter barely changed = undertrained

    Note: We CANNOT analyze the combined nonlinear transform W_up·GELU(W_down·h)
    analytically because GELU depends on the actual input values.
    This is the fundamental difference from LoRA where B·A is computable statically.
    """
    print("\n  BOTTLENECK ADAPTER ANALYSIS")
    print("  " + "─" * 56)

    layer_stats = []

    for name, module in adapted_model.named_modules():
        # Look for adapter down-projection weights
        if "adapter_down" in name and hasattr(module, "weight"):
            layer_id = name.split(".")[2] if len(name.split(".")) > 2 else name
            w_down = module.weight.float()

            # Find corresponding up-projection
            up_name = name.replace("adapter_down", "adapter_up")
            up_module = None
            for n2, m2 in adapted_model.named_modules():
                if n2 == up_name and hasattr(m2, "weight"):
                    up_module = m2
                    break

            if up_module is None:
                continue

            w_up = up_module.weight.float()

            # Metrics
            down_norm = torch.norm(w_down, "fro").item()
            up_norm = torch.norm(w_up, "fro").item()

            # Deviation of W_up from zero (its initialization)
            up_zero_deviation = torch.norm(w_up, "fro").item()  # W_up init = 0, so this IS the deviation

            # Effective signal: W_up · W_down acts like a rank-r matrix
            # Compute ‖W_up · W_down‖_F as a proxy for adapter output strength
            combined = (w_up @ w_down)
            combined_norm = torch.norm(combined, "fro").item()

            # Approximate rank via singular value analysis
            try:
                _, s, _ = torch.svd(w_down[:min(16, w_down.shape[0]), :min(16, w_down.shape[1])])
                effective_rank = (s > s[0] * 0.01).sum().item()
            except Exception:
                effective_rank = -1

            layer_stats.append({
                "layer": layer_id,
                "down_norm": down_norm,
                "up_norm": up_norm,
                "up_deviation": up_zero_deviation,
                "combined_norm": combined_norm,
                "effective_rank": effective_rank,
            })

    if not layer_stats:
        print("  No bottleneck adapter weights found in model state.")
        print("  (Model may need to be loaded differently — check peft version)")
        return

    # Sort by combined norm to find most/least adapted
    layer_stats.sort(key=lambda x: x["combined_norm"], reverse=True)

    print(f"\n  {'Layer':<12} {'‖W_down‖':>10} {'‖W_up‖':>10} {'‖W_up·W_down‖':>14} {'Rank(W_down)':>14}")
    print(f"  {'─'*66}")
    for s in layer_stats:
        print(f"  {s['layer']:<12} {s['down_norm']:>10.4f} {s['up_norm']:>10.4f} "
              f"{s['combined_norm']:>14.4f} {s['effective_rank']:>14}")

    print(f"\n  Most adapted layers (highest ‖W_up·W_down‖):")
    for s in layer_stats[:5]:
        print(f"    {s['layer']:<30} combined_norm = {s['combined_norm']:.4f}")

    print(f"\n  Least adapted layers:")
    for s in layer_stats[-5:]:
        print(f"    {s['layer']:<30} combined_norm = {s['combined_norm']:.4f}")

    avg_deviation = np.mean([s["up_deviation"] for s in layer_stats])
    print(f"\n  Average W_up deviation from init (zero): {avg_deviation:.4f}")
    print(f"  (Higher = adapter learned stronger transformations)")
    print(f"\n  ⚠️  NOTE: These metrics are LINEAR approximations.")
    print(f"  The actual adapter output W_up·GELU(W_down·h) depends on input h.")
    print(f"  Use compare_base_vs_adapted() in additive_inference.py for true behavioral comparison.")


# ──────────────────────────────────────────────────────────────────────────────
# (IA)³ Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_ia3_vectors(adapted_model):
    """
    Analyze what the (IA)³ l vectors learned.

    This is richer than the Bottleneck analysis because:
    - l vectors are just numbers (no non-linearity to hide the insight)
    - Each element directly tells you: amplify (>1) or suppress (<1) this feature
    - ‖l - 1‖ tells you exactly how much the adapter deviated from identity
    - We can see WHICH feature dimensions were amplified/suppressed

    This gives a direct window into what the model thinks matters for the task:
    high l values = "this feature dimension is important for my new task"
    low l values = "this feature dimension is noise/irrelevant for my new task"
    """
    print("\n  (IA)³ SCALING VECTOR ANALYSIS")
    print("  " + "─" * 56)

    l_vectors = {}
    for name, param in adapted_model.named_parameters():
        if "ia3_l" in name.lower() or ("lora" not in name.lower() and "weight" not in name.lower()
                                        and param.requires_grad and param.numel() < 10000):
            l_vectors[name] = param.data.float()

    if not l_vectors:
        # Try alternative naming
        for name, param in adapted_model.named_parameters():
            if param.requires_grad:
                l_vectors[name] = param.data.float()

    if not l_vectors:
        print("  No (IA)³ l vectors found.")
        return

    print(f"\n  Found {len(l_vectors)} learned scaling vectors\n")
    print(f"  {'Vector':<55} {'Size':>6} {'Mean':>8} {'Std':>8} {'‖l-1‖':>10} {'Max':>8} {'Min':>8}")
    print(f"  {'─'*110}")

    all_deviations = []

    for name, l in l_vectors.items():
        deviation = torch.norm(l - 1.0).item()
        all_deviations.append((name, deviation, l))
        short_name = name[-55:] if len(name) > 55 else name
        print(f"  {short_name:<55} {l.numel():>6} {l.mean():>8.4f} {l.std():>8.4f} "
              f"{deviation:>10.4f} {l.max():>8.4f} {l.min():>8.4f}")

    all_deviations.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Most changed l vectors (highest ‖l-1‖ = most deviation from identity):")
    for name, dev, l in all_deviations[:5]:
        print(f"    {name[-60:]}")
        print(f"      ‖l-1‖ = {dev:.4f} | mean={l.mean():.4f} | max={l.max():.4f} | min={l.min():.4f}")

        # Show top 5 amplified and suppressed dimensions
        top_amp = torch.topk(l, min(5, len(l))).indices.tolist()
        top_sup = torch.topk(-l, min(5, len(l))).indices.tolist()
        print(f"      Top amplified dims:  {[(i, f'{l[i]:.3f}') for i in top_amp]}")
        print(f"      Top suppressed dims: {[(i, f'{l[i]:.3f}') for i in top_sup]}")

    print(f"\n  Least changed (closest to identity, i.e., adapter barely affected these):")
    for name, dev, l in all_deviations[-3:]:
        print(f"    {name[-60:]} → ‖l-1‖ = {dev:.4f}")

    total_dev = sum(x[1] for x in all_deviations)
    print(f"\n  Total deviation across all l vectors: {total_dev:.4f}")
    print(f"  (IA)³ trained with {len(l_vectors)} vectors, avg ‖l-1‖ = {total_dev/len(l_vectors):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Base weight drift check (should be zero for both methods)
# ──────────────────────────────────────────────────────────────────────────────

def check_base_weight_drift(base_model, adapted_model):
    """
    Verify that the base model's frozen weights were not modified.

    Expected outcome:
    - ALL base model weight matrices (q_proj, k_proj, etc.) should have
      ZERO drift — they were frozen during training.
    - If drift > 0 for any base weight, that indicates a bug.
    - Only the adapter weights (W_down/W_up or l vectors) should differ.
    """
    print("\n  BASE WEIGHT DRIFT CHECK (should all be zero)")
    print("  " + "─" * 56)

    base_state = dict(base_model.named_parameters())
    max_drift = 0.0
    nonzero_drifts = []

    for name, adapted_param in adapted_model.named_parameters():
        # Only check params that exist in both (i.e., base model params)
        # Skip adapter-specific params (W_down, W_up, l vectors)
        clean_name = name.replace("base_model.model.", "")
        if clean_name not in base_state:
            continue

        base_param = base_state[clean_name].float()
        adapted_p = adapted_param.data.float()

        if base_param.shape != adapted_p.shape:
            continue

        drift = torch.norm(adapted_p - base_param).item()
        if drift > 1e-6:
            nonzero_drifts.append((clean_name, drift))
        max_drift = max(max_drift, drift)

    if len(nonzero_drifts) == 0:
        print(f"  ✅ All base weights perfectly frozen — zero drift.")
        print(f"     This confirms only the adapter parameters were updated.")
    else:
        print(f"  ⚠️  Found {len(nonzero_drifts)} base weights with non-zero drift!")
        print(f"  This may indicate a training configuration issue.")
        for n, d in sorted(nonzero_drifts, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {n:<60}  drift = {d:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str,
                        default="./outputs/llama-additive-bottleneck")
    args = parser.parse_args()

    adapter_path = args.adapter_path

    print("=" * 60)
    print("  Additive PEFT — Model Analysis")
    print("=" * 60)

    base_model, adapted_model, tokenizer, method = load_models(adapter_path)

    check_base_weight_drift(base_model, adapted_model)

    if method == "bottleneck":
        analyze_bottleneck_adapters(adapted_model)
    elif method == "ia3":
        analyze_ia3_vectors(adapted_model)
    else:
        print(f"\n  Unknown method '{method}' — running both analyses.")
        analyze_bottleneck_adapters(adapted_model)
        analyze_ia3_vectors(adapted_model)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
