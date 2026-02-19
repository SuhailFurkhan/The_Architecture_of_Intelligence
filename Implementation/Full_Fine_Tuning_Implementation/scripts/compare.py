"""
compare.py — Side-by-side comparison: Original vs Fine-Tuned model.

Run AFTER training to see what changed.

Usage:
    python scripts/compare.py
    python scripts/compare.py --finetuned_path ./outputs/llama-3.2-1B-full-ft/final
"""

import argparse
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Test prompts — same questions go to both models
# ──────────────────────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    "What is machine learning? Explain in 2 sentences.",
    "Write a short poem about the ocean.",
    "Translate 'Good morning, how are you?' to French.",
    "List 3 benefits of regular exercise.",
    "Explain the difference between a list and a tuple in Python.",
]


def load_model(model_path: str, label: str):
    """Load a model + tokenizer."""
    print(f"  📦 Loading {label}: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a response using chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    # BatchEncoding (transformers v5) inherits from UserDict, not dict.
    # Check for Tensor first to handle both old and new versions.
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compare(original_path: str, finetuned_path: str):
    """Run same prompts through both models and display side by side."""

    print("=" * 70)
    print("  BEFORE vs AFTER — Full Fine-Tuning Comparison")
    print("=" * 70)
    print()

    # Load both models
    original_model, original_tok = load_model(original_path, "ORIGINAL")
    finetuned_model, finetuned_tok = load_model(finetuned_path, "FINE-TUNED")
    print()

    # Run each prompt through both
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"{'─' * 70}")
        print(f"  PROMPT {i}: {prompt}")
        print(f"{'─' * 70}")

        original_response = generate(original_model, original_tok, prompt)
        finetuned_response = generate(finetuned_model, finetuned_tok, prompt)

        print(f"\n  🔵 BEFORE (original):")
        print(f"     {original_response[:500]}")
        print(f"\n  🟢 AFTER (fine-tuned):")
        print(f"     {finetuned_response[:500]}")
        print()

    # ──────────────────────────────────────────────────────────────────────
    # Bonus: weight drift — how much did the parameters actually change?
    # ──────────────────────────────────────────────────────────────────────
    print(f"{'─' * 70}")
    print(f"  WEIGHT DRIFT ANALYSIS")
    print(f"{'─' * 70}")
    print(f"  Comparing how much each layer's weights moved during training.\n")

    orig_state = original_model.state_dict()
    ft_state = finetuned_model.state_dict()

    drifts = []
    for key in orig_state:
        if orig_state[key].dtype in (torch.bfloat16, torch.float16, torch.float32):
            diff = (ft_state[key].float() - orig_state[key].float()).abs().mean().item()
            drifts.append((key, diff))

    # Sort by biggest change
    drifts.sort(key=lambda x: x[1], reverse=True)

    print(f"  Top 10 most-changed layers:")
    for name, drift in drifts[:10]:
        bar = "█" * int(drift * 5000)  # visual bar
        print(f"    {drift:.6f}  {bar:20s}  {name}")

    print(f"\n  Bottom 5 least-changed layers:")
    for name, drift in drifts[-5:]:
        print(f"    {drift:.6f}  {'░':20s}  {name}")

    avg_drift = sum(d for _, d in drifts) / len(drifts)
    print(f"\n  Average weight drift: {avg_drift:.6f}")
    print(f"  Total layers compared: {len(drifts)}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Compare original vs fine-tuned model")

    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    parser.add_argument("--original_path", default=config["model_name"],
                        help="Original model (HuggingFace name or local path)")
    parser.add_argument("--finetuned_path",
                        default=config["output_dir"] + "/final",
                        help="Fine-tuned model path")
    args = parser.parse_args()

    compare(args.original_path, args.finetuned_path)


if __name__ == "__main__":
    main()