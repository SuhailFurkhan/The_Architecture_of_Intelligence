"""
inference.py — Test your fully fine-tuned model.

Usage:
    python scripts/inference.py
    python scripts/inference.py --prompt "Explain quantum computing"
    python scripts/inference.py --model_path ./outputs/llama-3.2-1B-full-ft/final
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    """Load the full fine-tuned model (not an adapter — the whole thing)."""

    print(f"📦 Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,              # renamed from torch_dtype in transformers v5
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, prompt: str, system_prompt: str = "You are a helpful assistant.",
             max_new_tokens: int = 256, temperature: float = 0.7):
    """Generate a response using the chat template."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # apply_chat_template formats the messages with special tokens:
    #   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    #   You are a helpful assistant.<|eot_id|>
    #   <|start_header_id|>user<|end_header_id|>
    #   {prompt}<|eot_id|>
    #   <|start_header_id|>assistant<|end_header_id|>
    #
    # In transformers v5, this returns a BatchEncoding dict with keys:
    #   { "input_ids": tensor, "attention_mask": tensor }
    # (In older versions it returned a plain tensor)
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    # Handle both transformers v5 (BatchEncoding, a UserDict subclass)
    # and older versions (plain tensor).
    # NOTE: BatchEncoding inherits from UserDict, NOT dict,
    # so isinstance(encoded, dict) returns False — check for Tensor instead.
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

    # Decode only the NEW tokens (skip the prompt)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Test your fine-tuned model")
    parser.add_argument("--model_path", default="./outputs/llama-3.2-1B-full-ft/final",
                        help="Path to saved model")
    parser.add_argument("--prompt", default=None, help="Single prompt to test")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    if args.prompt:
        # Single prompt mode
        response = generate(model, tokenizer, args.prompt)
        print(f"\n💬 Prompt:   {args.prompt}")
        print(f"🤖 Response: {response}")
    else:
        # Interactive mode
        print("\n" + "=" * 50)
        print("  Interactive Chat (type 'quit' to exit)")
        print("=" * 50 + "\n")

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

