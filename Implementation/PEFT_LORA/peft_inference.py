"""
peft_inference.py — Test your LoRA fine-tuned model.

╔══════════════════════════════════════════════════════════════════════════╗
║  LOADING A LoRA MODEL IS DIFFERENT FROM FULL FINE-TUNING               ║
║                                                                          ║
║  Full fine-tuning inference:                                             ║
║    model = AutoModelForCausalLM.from_pretrained("./outputs/final")      ║
║    # ← loads the entire modified model, self-contained                  ║
║                                                                          ║
║  LoRA inference (adapter-only save):                                     ║
║    base  = AutoModelForCausalLM.from_pretrained("original_model_name")  ║
║    model = PeftModel.from_pretrained(base, "./outputs/final")            ║
║    # ← loads base + tiny adapter deltas, applies them on top             ║
║                                                                          ║
║  LoRA inference (merged model save):                                     ║
║    model = AutoModelForCausalLM.from_pretrained("./outputs/final")      ║
║    # ← same as full FT — works without PEFT library                     ║
║                                                                          ║
║  This script auto-detects which type was saved and loads accordingly.   ║
║                                                                          ║
║  ADAPTER SWAPPING (advanced use case):                                   ║
║    Because adapters are small and separate, you can:                     ║
║      1. Load ONE base model into GPU memory                              ║
║      2. Hot-swap different adapters for different tasks!                  ║
║    Example:                                                              ║
║      base = AutoModelForCausalLM.from_pretrained("Llama-3.2-1B")        ║
║      model = PeftModel.from_pretrained(base, "adapter_for_coding")       ║
║      # ... generate code ...                                             ║
║      model.load_adapter("adapter_for_medical", adapter_name="medical")   ║
║      model.set_adapter("medical")                                        ║
║      # ... now generates medical text with same base model ...           ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python peft_inference.py
    python peft_inference.py --prompt "Explain quantum computing"
    python peft_inference.py --model_path ./outputs/llama-3.2-1B-lora/final
    python peft_inference.py --merge        # merge on-the-fly before generating
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_model_type(model_path: str) -> str:
    """
    Detect whether the saved model is:
      - "adapter"  → only adapter weights saved (needs base model)
      - "merged"   → full model with weights already merged in
      - "unknown"  → can't determine

    We check for the presence of 'adapter_config.json', which is the file
    PEFT writes when saving an adapter (it contains the LoRA hyperparameters
    and the name of the base model). If that file exists, it's an adapter save.
    """
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        return "adapter"

    # Check for standard model config (merged or full fine-tuned)
    model_config = Path(model_path) / "config.json"
    if model_config.exists():
        return "merged"

    return "unknown"


def get_base_model_name(model_path: str) -> str:
    """
    Read the base model name from adapter_config.json.

    adapter_config.json (written by PEFT) looks like:
    {
      "base_model_name_or_path": "unsloth/Llama-3.2-1B-Instruct",
      "r": 16,
      "lora_alpha": 32,
      "target_modules": ["q_proj", "v_proj", ...],
      ...
    }
    """
    adapter_config_path = Path(model_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    return adapter_config.get("base_model_name_or_path", "unsloth/Llama-3.2-1B-Instruct")


def load_model(model_path: str, merge: bool = False):
    """
    Load the fine-tuned model — auto-detects adapter vs merged save.

    Args:
        model_path:  path to the saved model or adapter
        merge:       if True and model is adapter-type, merge LoRA into base
                     before returning (useful for faster inference)

    Returns:
        (model, tokenizer, model_type)
    """
    from peft import PeftModel

    model_type = detect_model_type(model_path)
    print(f"  🔍 Detected model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "adapter":
        # Need to load base model first, then apply adapter on top
        base_model_name = get_base_model_name(model_path)
        print(f"  📦 Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"  🔧 Applying LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)

        if merge:
            # Merge the adapter into the base model weights
            # After merging, the model behaves exactly like a full fine-tuned model
            # at the cost of losing the ability to swap/disable the adapter.
            print(f"  🔀 Merging LoRA weights into base model (merge=True)...")
            model = model.merge_and_unload()
            print(f"  ✅ Adapter merged — model is now a standalone model")

    elif model_type == "merged":
        # Already merged — load directly (same as full fine-tuning inference)
        print(f"  📦 Loading merged model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise FileNotFoundError(
            f"Cannot load model from '{model_path}'.\n"
            f"  Expected either 'adapter_config.json' (adapter save) "
            f"or 'config.json' (merged save)."
        )

    model.eval()
    return model, tokenizer, model_type


def generate(model, tokenizer, prompt: str,
             system_prompt: str = "You are a helpful assistant.",
             max_new_tokens: int = 256,
             temperature: float = 0.7) -> str:
    """
    Generate a response. Identical to full fine-tuning inference.py.
    The model API is the same regardless of whether it was full FT or LoRA.
    """
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": prompt},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    # Handle transformers v4 (Tensor) and v5 (BatchEncoding)
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
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Test your LoRA fine-tuned model")
    parser.add_argument("--model_path", default="./outputs/llama-3.2-1B-lora/final",
                        help="Path to adapter or merged model")
    parser.add_argument("--prompt", default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--merge", action="store_true",
                        help="If adapter-type, merge weights before inference (slightly faster)")
    args = parser.parse_args()

    model, tokenizer, model_type = load_model(args.model_path, merge=args.merge)

    if args.prompt:
        response = generate(model, tokenizer, args.prompt)
        print(f"\n  💬 Prompt:   {args.prompt}")
        print(f"  🤖 Response: {response}")
    else:
        print("\n" + "=" * 55)
        print("  LoRA Model Interactive Chat (type 'quit' to exit)")
        print(f"  Model type: {model_type}")
        print("=" * 55 + "\n")

        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            response = generate(model, tokenizer, prompt)
            print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
