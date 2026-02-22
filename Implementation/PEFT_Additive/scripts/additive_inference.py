"""
additive_inference.py — Inference for Bottleneck Adapters and (IA)³.

╔══════════════════════════════════════════════════════════════════════════╗
║  INFERENCE: HOW ADDITIVE METHODS DIFFER FROM LoRA                        ║
║                                                                          ║
║  LoRA (merged):                                                          ║
║    Load model → looks like a normal model → zero overhead                ║
║                                                                          ║
║  LoRA (unmerged):                                                        ║
║    Load base + PeftModel.from_pretrained → parallel bypass active        ║
║    → can swap adapters, slight extra compute                             ║
║                                                                          ║
║  Bottleneck Adapters:                                                    ║
║    ALWAYS need PeftModel.from_pretrained (cannot merge non-linear)       ║
║    Every inference call routes tokens through:                           ║
║      frozen_layer → adapter (down → GELU → up → residual) → ...          ║
║    ~5–15% slower than base model (depends on bottleneck_dim)             ║
║                                                                          ║
║  (IA)³ (unmerged):                                                       ║
║    Load base + PeftModel.from_pretrained → l vectors applied to K/V/FF   ║
║    Overhead: element-wise multiply → nearly zero                         ║
║                                                                          ║
║  (IA)³ (merged):                                                         ║
║    If trained with merge_before_save=true (or merged manually),          ║
║    loads as a standard model → exactly zero overhead                     ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python additive_inference.py
    python additive_inference.py --prompt "Explain what a neural network is"
    python additive_inference.py --adapter_path ./outputs/llama-additive-bottleneck
    python additive_inference.py --compare  # compare base vs adapted
"""

import argparse
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def detect_adapter_type(adapter_path: str) -> str:
    """
    Read adapter_config.json to determine which PEFT method was used.

    Returns "bottleneck", "ia3", "lora", or "merged" (if merge_before_save was used).
    """
    import json
    config_file = Path(adapter_path) / "adapter_config.json"

    if not config_file.exists():
        # Might be a merged model (no adapter_config.json)
        return "merged"

    with open(config_file) as f:
        cfg = json.load(f)

    peft_type = cfg.get("peft_type", "").lower()
    if "bottleneck" in peft_type or "adapter" in peft_type:
        return "bottleneck"
    elif "ia3" in peft_type:
        return "ia3"
    elif "lora" in peft_type:
        return "lora"
    return "unknown"


def load_model_for_inference(adapter_path: str, device: str = "auto"):
    """
    Load a trained adapter and its base model for inference.

    Routing logic:
      1. Detect adapter type from adapter_config.json
      2. If merged (no adapter config): load as standard model
      3. If bottleneck/ia3/lora: load base model, then wrap with PeftModel
    """
    adapter_type = detect_adapter_type(adapter_path)
    print(f"  Detected adapter type: {adapter_type}")

    import json
    config_file = Path(adapter_path) / "adapter_config.json"

    if adapter_type == "merged":
        # Merged (IA)³ or full model — load directly, zero overhead
        print(f"  Loading merged model from: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print("  ✅ Merged model loaded — zero PEFT overhead at inference.")
    else:
        # Load base model + adapter
        with open(config_file) as f:
            cfg = json.load(f)
        base_model_name = cfg.get("base_model_name_or_path")
        print(f"  Base model: {base_model_name}")
        print(f"  Adapter:    {adapter_path}")

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Wrap the base model with the adapter
        # For bottleneck: re-inserts adapter modules into the forward path
        # For (IA)³: re-attaches l vectors to the relevant projections
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        if adapter_type == "bottleneck":
            print("  ⚠️  Bottleneck adapter active — permanent inference overhead.")
            print("  Each token passes through: frozen_layer → adapter → ...")
        elif adapter_type == "ia3":
            print("  ℹ️  (IA)³ adapter active — negligible overhead (element-wise multiply).")
            print("  Tip: use model.merge_and_unload() to fold l vectors into weights.")

    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Run a single inference call and return the generated response."""
    formatted = ALPACA_PROMPT_NO_INPUT.format(instruction=prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    tokens_per_sec = len(generated) / elapsed
    return response, tokens_per_sec


def compare_base_vs_adapted(adapter_path: str, prompts: list[str]):
    """
    Load both the base model and the adapted model and compare their outputs.

    Useful for verifying the adapter is actually changing behavior and
    for estimating the inference overhead introduced by the adapter.
    """
    import json
    with open(Path(adapter_path) / "adapter_config.json") as f:
        cfg = json.load(f)
    base_model_name = cfg["base_model_name_or_path"]

    print("\n  Loading BASE model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()

    print("  Loading ADAPTED model...")
    adapted_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapted_model.eval()

    print("\n" + "=" * 60)
    for prompt in prompts:
        print(f"\n  PROMPT: {prompt}")
        print("  " + "─" * 56)

        base_response, base_tps = run_inference(base_model, tokenizer, prompt)
        print(f"  BASE ({base_tps:.1f} tok/s):\n  {base_response[:300]}")

        adapted_response, adapted_tps = run_inference(adapted_model, tokenizer, prompt)
        overhead_pct = (base_tps - adapted_tps) / base_tps * 100
        print(f"\n  ADAPTED ({adapted_tps:.1f} tok/s, {overhead_pct:+.1f}% vs base):\n  {adapted_response[:300]}")
        print("  " + "─" * 56)


def main():
    parser = argparse.ArgumentParser(description="Additive PEFT inference")
    parser.add_argument("--adapter_path", type=str,
                        default="./outputs/llama-additive-bottleneck",
                        help="Path to saved adapter or merged model")
    parser.add_argument("--prompt", type=str,
                        default="Explain the difference between supervised and unsupervised learning.",
                        help="Instruction to run")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--compare", action="store_true",
                        help="Compare base model vs adapted model side by side")
    args = parser.parse_args()

    if args.compare:
        compare_base_vs_adapted(
            args.adapter_path,
            prompts=[
                args.prompt,
                "What are the steps to bake sourdough bread?",
            ]
        )
        return

    print("=" * 60)
    print("  Additive PEFT — Inference")
    print("=" * 60)

    model, tokenizer = load_model_for_inference(args.adapter_path)
    model.eval()

    print(f"\n  Prompt: {args.prompt}\n")
    response, tps = run_inference(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"  Response:\n  {response}")
    print(f"\n  Speed: {tps:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
