"""
additive_prepare_data.py — Dataset preparation for Bottleneck Adapters & (IA)³.

╔══════════════════════════════════════════════════════════════════════════╗
║  DATA PIPELINE — IDENTICAL TO LoRA                                       ║
║                                                                          ║
║  The data pipeline is completely unchanged between LoRA, Bottleneck      ║
║  Adapters, and (IA)³. The model sees the same tokens, the same labels,   ║
║  and the same loss mask regardless of which PEFT method is used.         ║
║                                                                          ║
║  The methods only differ in what happens INSIDE the model during the     ║
║  forward pass — not in how the input is prepared.                        ║
║                                                                          ║
║  RESPONSE-ONLY MASKING:                                                  ║
║  We still use response masking (labels=-100 for instruction tokens)      ║
║  for the same reason as LoRA: with few trainable params, every           ║
║  gradient step must be maximally informative. Grading the model only     ║
║  on its own responses, not on the prompt it was given, ensures           ║
║  gradients carry real task signal.                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python additive_prepare_data.py
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template (identical to LoRA system — same model, same chat format)
# ──────────────────────────────────────────────────────────────────────────────
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(example: dict) -> tuple[str, str]:
    """
    Format a raw JSONL example into (prompt, response) strings.

    Returns the prompt (instruction + input, no response) and the response
    separately, so we can tokenize the boundary precisely for loss masking.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if inp:
        prompt = ALPACA_PROMPT.format(instruction=instruction, input=inp)
    else:
        prompt = ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)

    return prompt, output


def tokenize_with_response_masking(
    example: dict,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> dict:
    """
    Tokenize a single example with response-only masking.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  HOW RESPONSE MASKING WORKS                                          │
    │                                                                      │
    │  Full conversation tokens:                                           │
    │  [bos, sys, user_tokens..., asst_tokens..., eos]                     │
    │                                                                      │
    │  Labels:                                                             │
    │  [-100, -100, -100...,      asst_tokens..., eos]                     │
    │   ↑ ignored by loss          ↑ these tokens count toward loss        │
    │                                                                      │
    │  How boundary is found:                                              │
    │  1. Tokenize just the prompt (no response)                           │
    │  2. len(prompt_tokens) = N                                           │
    │  3. labels[:N] = -100   (mask everything up to and including prompt) │
    │  4. labels[N:] = input_ids[N:]  (response contributes to loss)      │
    └──────────────────────────────────────────────────────────────────────┘
    """
    prompt, response = format_prompt(example)
    full_text = prompt + response

    # Tokenize full conversation
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )

    # Tokenize prompt-only to find exact boundary
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    prompt_len = len(prompt_tokens["input_ids"])

    # Build labels: -100 for prompt, actual IDs for response
    labels = [-100] * len(full_tokens["input_ids"])
    response_start = min(prompt_len, len(full_tokens["input_ids"]))
    for i in range(response_start, len(full_tokens["input_ids"])):
        labels[i] = full_tokens["input_ids"][i]

    # Filter out examples where response was fully truncated
    has_response = any(l != -100 for l in labels)
    if not has_response:
        return None

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }


def prepare_datasets(config_path: str = None) -> tuple[Dataset, Dataset, AutoTokenizer]:
    """
    Load, format, tokenize, and return train/eval datasets + tokenizer.

    Pipeline:
        Raw JSONL
         → format_prompt()         : add chat template, separate prompt/response
         → tokenize_with_masking() : convert to token IDs with -100 label masking
         → filter()                : remove examples with no response tokens
         → return                  : HuggingFace Datasets ready for Trainer
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "additive_training_config.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    dataset_name = config.get("dataset_name", "yahma/alpaca-cleaned")
    max_seq_length = config.get("max_seq_length", 512)
    train_split = config.get("train_split", "train[:90%]")
    eval_split = config.get("eval_split", "train[90%:]")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"  Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load raw dataset ──────────────────────────────────────────────────────
    print(f"  Loading dataset: {dataset_name}")
    train_raw = load_dataset(dataset_name, split=train_split)
    eval_raw  = load_dataset(dataset_name, split=eval_split)
    print(f"  Train: {len(train_raw):,}  |  Eval: {len(eval_raw):,}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    def tokenize_fn(example):
        return tokenize_with_response_masking(example, tokenizer, max_seq_length)

    print("  Tokenizing training set...")
    train_tokenized = train_raw.map(tokenize_fn, batched=False, remove_columns=train_raw.column_names)
    train_tokenized = train_tokenized.filter(lambda x: x is not None and len(x["input_ids"]) > 0)

    print("  Tokenizing eval set...")
    eval_tokenized = eval_raw.map(tokenize_fn, batched=False, remove_columns=eval_raw.column_names)
    eval_tokenized = eval_tokenized.filter(lambda x: x is not None and len(x["input_ids"]) > 0)

    print(f"  After filtering — Train: {len(train_tokenized):,}  |  Eval: {len(eval_tokenized):,}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    lengths = [len(x["input_ids"]) for x in train_tokenized.select(range(min(500, len(train_tokenized))))]
    print(f"\n  Sequence length stats (first 500 examples):")
    print(f"    Mean:   {np.mean(lengths):.0f}")
    print(f"    Median: {np.median(lengths):.0f}")
    print(f"    Max:    {max(lengths)}")
    print(f"    % at max_len ({max_seq_length}): {sum(1 for l in lengths if l == max_seq_length) / len(lengths) * 100:.1f}%")

    return train_tokenized, eval_tokenized, tokenizer


def get_data_collator(tokenizer: AutoTokenizer, model=None) -> DataCollatorForSeq2Seq:
    """
    Return a DataCollatorForSeq2Seq.

    This collator:
    - Pads input_ids and attention_mask to the longest sequence in the batch
    - Pads labels with -100 (not with pad_token_id!) so padding positions
      don't contribute to the loss
    - Handles the label shifting needed for causal LM training

    This is identical for both LoRA and Additive PEFT — the collator doesn't
    know or care about the PEFT method used.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  Additive PEFT — Data Preparation")
    print("=" * 60)
    train_ds, eval_ds, tokenizer = prepare_datasets()
    print(f"\n  Sample — input_ids[:10]: {train_ds[0]['input_ids'][:10]}")
    print(f"  Sample — labels[:10]:    {train_ds[0]['labels'][:10]}")
    non_masked = sum(1 for l in train_ds[0]["labels"] if l != -100)
    total = len(train_ds[0]["labels"])
    print(f"  Response tokens: {non_masked} / {total}  ({non_masked/total*100:.1f}% of sequence graded)")
    print("\n  ✅ Data ready.")
