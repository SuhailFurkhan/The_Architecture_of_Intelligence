"""
peft_prepare_data.py — Load & format dataset for LoRA / PEFT fine-tuning.

╔══════════════════════════════════════════════════════════════════════════╗
║  HOW DATA PREP DIFFERS FOR PEFT vs FULL FINE-TUNING                     ║
║                                                                          ║
║  The tokenization pipeline is IDENTICAL — text → token IDs → labels.    ║
║                                                                          ║
║  The key difference is RESPONSE-ONLY MASKING (also called "instruction   ║
║  masking" or "loss masking"):                                            ║
║                                                                          ║
║  Full fine-tuning (what your original prepare_data.py does):            ║
║    labels = input_ids  (compute loss on EVERY token — instruction + answer)║
║                                                                          ║
║  PEFT best practice (what this file does):                               ║
║    labels[instruction_tokens] = -100   ← IGNORED by cross-entropy loss  ║
║    labels[response_tokens]    = actual token IDs                         ║
║                                                                          ║
║  WHY MASK THE INSTRUCTION?                                               ║
║    The model should learn HOW TO ANSWER, not how to reproduce the        ║
║    question. If you train on the full sequence:                          ║
║                                                                          ║
║      System: "You are a helpful assistant."                              ║
║      User:   "What is Python?"                                           ║
║      Bot:    "Python is a programming language..."                        ║
║                                                                          ║
║    Without masking: loss computed on ALL tokens including "You are...",  ║
║    "What is Python?" — the model wastes capacity memorizing prompts.     ║
║                                                                          ║
║    With masking: loss computed only on "Python is a programming..."      ║
║    — the model focuses entirely on generating good RESPONSES.            ║
║                                                                          ║
║  This is especially important for LoRA because:                          ║
║    • Fewer trainable params → each gradient update must be maximally     ║
║      informative. Wasting gradients on prompt tokens hurts convergence.  ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python peft_prepare_data.py          # Preview first few examples
    from peft_prepare_data import load_and_prepare_dataset   # in train.py
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login


# ── Default model ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Authentication  (identical to full fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

def load_env_file(env_file: str = "Keys.env") -> tuple[str | None, str | None]:
    """
    Load HuggingFace token from an env file — same as full FT version.
    Searches script dir and up to 3 parent directories.
    """
    possible_keys = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    search_paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    for _ in range(4):
        search_paths.append(os.path.join(current, env_file))
        current = os.path.dirname(current)
    search_paths.append(os.path.join(os.getcwd(), env_file))
    search_paths.append(os.path.expanduser(f"~/{env_file}"))

    found_path = None
    for path in search_paths:
        if os.path.exists(path):
            found_path = path
            break
    if not found_path:
        return None, None

    print(f"  📂 Found env file: {found_path}")
    try:
        with open(found_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    if key in possible_keys:
                        print(f"  ✅ Loaded {key} from {env_file}")
                        return value, key
    except Exception as e:
        print(f"  ⚠️  Error reading {found_path}: {e}")
    return None, None


def load_hf_token(env_file: str = "Keys.env") -> str | None:
    """Load token from env vars or Keys.env, then authenticate with HF."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    source = "environment variable" if token else None

    if not token:
        token, key_name = load_env_file(env_file)
        source = f"{env_file}" if token else None

    if not token:
        print(f"  ⚠️  No HF token found. Create Keys.env with HF_TOKEN=hf_xxx")
        return None

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    login(token=token)
    print(f"  ✅ Authenticated with HuggingFace (source: {source})")
    return token


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Dataset Formatting with Response Masking
# ══════════════════════════════════════════════════════════════════════════════

def format_alpaca_to_chat(example: dict) -> dict:
    """
    Convert Alpaca-format example to chat messages list.

    Input (Alpaca format):
        {
            "instruction": "Explain the difference between lists and tuples.",
            "input": "",          # optional context / extra input
            "output": "Lists are mutable..."
        }

    Output (chat messages):
        [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "Explain the difference..."},
            {"role": "assistant", "content": "Lists are mutable..."}
        ]

    The messages format matches what tokenizer.apply_chat_template() expects.
    Having separate messages lets us identify WHERE the response starts,
    which we need for masking (see tokenize_with_masking below).
    """
    instruction = example["instruction"]
    context = example.get("input", "")
    response = example["output"]

    # Combine instruction and context if context is non-empty
    if context and context.strip():
        user_content = f"{instruction}\n\nContext: {context}"
    else:
        user_content = instruction

    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": response},
    ]

    return {"messages": messages}


def tokenize_with_masking(example: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Apply chat template, tokenize, and MASK the instruction tokens.

    ┌──────────────────────────────────────────────────────────────────┐
    │  HOW RESPONSE MASKING WORKS                                      │
    │                                                                  │
    │  The full tokenized sequence looks like:                         │
    │                                                                  │
    │  <bos> [system] You are... [/system] [user] Explain... [/user]  │
    │  [assistant] Lists are mutable... [/assistant] <eos>             │
    │  ├──────────────────────────────────────┤ ├──────────────────┤  │
    │           "prompt" portion                  "response" portion   │
    │           labels = -100 (ignored)           labels = token IDs  │
    │                                                                  │
    │  Cross-entropy loss only computes on positions where label ≠ -100│
    │  So gradients only flow from the RESPONSE tokens.               │
    │                                                                  │
    │  Implementation strategy:                                        │
    │  1. Tokenize the FULL conversation (system+user+assistant)       │
    │  2. Tokenize just the PROMPT (system+user) separately            │
    │  3. prompt_length = len(prompt_ids)                              │
    │  4. labels[:prompt_length] = -100                                │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        example:          dict with "messages" key (list of role/content dicts)
        tokenizer:        the model's tokenizer
        max_seq_length:   truncate to this length (controls VRAM)

    Returns:
        dict with "input_ids", "attention_mask", "labels"
    """
    messages = example["messages"]

    # ── Step 1: Tokenize the FULL conversation ────────────────────────────
    # add_generation_prompt=False — we don't want the trailing [assistant] marker
    # because the assistant response is already in the messages list.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,    # return plain Python lists
    )

    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]

    # ── Step 2: Tokenize just the PROMPT (system + user turn) ────────────
    # We use apply_chat_template with add_generation_prompt=True to get the
    # exact token sequence that ends right before the assistant's response.
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # adds "[assistant]" marker at the end
    )

    prompt_tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )

    prompt_length = len(prompt_tokenized["input_ids"])

    # ── Step 3: Build labels with prompt tokens masked to -100 ───────────
    # PyTorch cross-entropy ignores positions where label == -100.
    # This is the standard way to exclude positions from the loss.
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    # Ensure labels is same length as input_ids (truncation might mismatch)
    labels = labels[:len(input_ids)]

    # Sanity check: at least some labels should be valid (not all -100)
    valid_labels = [l for l in labels if l != -100]
    if len(valid_labels) == 0:
        # The response was truncated away — return None to filter this example
        # (DatasetDict.filter() can remove these later)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "_skip": True}

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "_skip": False,
    }


def tokenize_without_masking(example: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Alternative: tokenize WITHOUT masking (same as full fine-tuning approach).

    Use this if you want to compare the effect of response masking,
    or if your dataset has very short instructions relative to responses.

    When to use this instead:
      - Very short prompts (masking removes most of the signal)
      - Datasets where instruction quality matters too (e.g., generating prompts)
      - You want identical behaviour to your full fine-tuning pipeline
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    tokenized["_skip"] = False
    return tokenized


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Main Dataset Loading Function
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_dataset(config: dict, tokenizer, mask_instruction: bool = True):
    """
    Load the Alpaca dataset, format, tokenize, and return train/eval splits.

    Args:
        config:            training config dict (from peft_training_config.yaml)
        tokenizer:         the model's tokenizer (already loaded)
        mask_instruction:  True = response-only loss (recommended for LoRA)
                           False = full sequence loss (same as full fine-tuning)

    Returns:
        (train_dataset, eval_dataset) — HuggingFace Dataset objects
    """
    dataset_name = config.get("dataset_name", "yahma/alpaca-cleaned")
    max_seq_length = config.get("max_seq_length", 512)
    seed = config.get("seed", 42)
    save_locally = config.get("save_dataset_locally", False)
    mask_suffix = "_masked" if mask_instruction else "_unmasked"

    # ── Local cache check ─────────────────────────────────────────────────
    local_dir = Path(config.get("local_data_dir", "./data"))
    dataset_folder = dataset_name.split("/")[-1] + "-lora" + mask_suffix + "-processed"
    local_path = local_dir / dataset_folder

    if save_locally and local_path.exists():
        print(f"  📂 Loading processed dataset from disk: {local_path}")
        processed = load_from_disk(str(local_path))
        train_data = processed["train"]
        eval_data = processed["eval"]
        print(f"  ✅ Train: {len(train_data)} | Eval: {len(eval_data)} (from disk cache)")
        return train_data, eval_data

    # ── Download from HuggingFace ─────────────────────────────────────────
    print(f"\n  📥 Downloading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name)

    # ── Step A: Format to chat messages ───────────────────────────────────
    print(f"  💬 Formatting to chat template...")
    dataset = raw_dataset.map(
        format_alpaca_to_chat,
        remove_columns=raw_dataset["train"].column_names,
        desc="Formatting to chat",
    )

    # ── Step B: Tokenize (with or without response masking) ───────────────
    tokenize_fn = tokenize_with_masking if mask_instruction else tokenize_without_masking
    mask_label = "response-only masking" if mask_instruction else "full sequence"
    print(f"  🔤 Tokenizing (max_length={max_seq_length}, mode={mask_label})...")

    dataset = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_seq_length),
        remove_columns=["messages"],
        desc=f"Tokenizing [{mask_label}]",
    )

    # ── Step C: Filter out examples where response was truncated ──────────
    before = len(dataset["train"])
    dataset = dataset.filter(lambda ex: not ex["_skip"])
    after = len(dataset["train"])
    skipped = before - after
    if skipped > 0:
        print(f"  ⚠️  Filtered {skipped} examples (response was fully truncated at max_seq_length={max_seq_length})")

    # Remove the helper column
    dataset = dataset.remove_columns(["_skip"])

    # ── Step D: Train / eval split ────────────────────────────────────────
    train_data = dataset["train"]
    if "test" not in dataset and "validation" not in dataset:
        split = train_data.train_test_split(test_size=0.1, seed=seed)
        train_data = split["train"]
        eval_data = split["test"]
    elif "validation" in dataset:
        eval_data = dataset["validation"]
    else:
        eval_data = dataset["test"]

    print(f"\n  ✅ Dataset ready:")
    print(f"     Train: {len(train_data):,} examples")
    print(f"     Eval:  {len(eval_data):,} examples")
    print(f"     Masking: {'instruction tokens masked (-100)' if mask_instruction else 'full sequence'}")

    # ── Step E: Optional local save ───────────────────────────────────────
    if save_locally:
        print(f"  💾 Saving processed dataset to: {local_path}")
        local_path.mkdir(parents=True, exist_ok=True)
        DatasetDict({"train": train_data, "eval": eval_data}).save_to_disk(str(local_path))
        print(f"  ✅ Saved! Next run will load from disk.")

    return train_data, eval_data


# ══════════════════════════════════════════════════════════════════════════════
# Standalone preview
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml

    print("=" * 65)
    print("  PEFT Data Preparation — Preview Mode")
    print("=" * 65)

    # Auth
    token = load_hf_token()
    if not token:
        print("  ❌ No HF token. Create Keys.env with HF_TOKEN=hf_xxx")
        sys.exit(1)

    # Config
    config_path = Path(__file__).parent / "peft_training_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model_name": DEFAULT_MODEL,
            "dataset_name": "yahma/alpaca-cleaned",
            "max_seq_length": 512,
            "seed": 42,
            "save_dataset_locally": False,
        }

    # Load tokenizer
    print(f"\n  📦 Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset WITH masking
    train_data, eval_data = load_and_prepare_dataset(config, tokenizer, mask_instruction=True)

    # Preview first example
    print("\n" + "=" * 65)
    print("  PREVIEW: First example decoded (with masking)")
    print("=" * 65)
    sample = train_data[0]
    ids = sample["input_ids"]
    labels = sample["labels"]

    # Show the full sequence, marking which parts are masked
    print("\n  [MASKED - not in loss] Prompt portion:")
    masked_ids = [i for i, l in zip(ids, labels) if l == -100]
    print(f"  {tokenizer.decode(masked_ids, skip_special_tokens=False)[:300]}")

    print("\n  [ACTIVE - in loss] Response portion:")
    active_ids = [i for i, l in zip(ids, labels) if l != -100]
    print(f"  {tokenizer.decode(active_ids, skip_special_tokens=False)[:300]}")

    print(f"\n  Total tokens: {len(ids)}")
    print(f"  Masked (instruction): {len(masked_ids)} tokens ({len(masked_ids)/len(ids)*100:.1f}%)")
    print(f"  Active (response):    {len(active_ids)} tokens ({len(active_ids)/len(ids)*100:.1f}%)")
