"""

unsloth/Llama-3.2-1B-Instruct
prepare_data.py — Load & format dataset for full fine-tuning.


What This Module Does:
  1. Authenticates with HuggingFace (loads token from Keys.env)
  2. Validates that you have download access to the target model
  3. Downloads the dataset (yahma/alpaca-cleaned — 52K instruction examples)
  4. Downloads the TOKENIZER (not the full model — just the vocabulary files, ~few MB)
  5. Formats each example into Llama 3's chat template
  6. Tokenizes all examples (converts text → numerical token IDs)
  7. Returns train/eval splits ready for training

What is Tokenization?
======================
  LLMs don't understand text — they work with numbers. Tokenization is the
  process of converting raw text into a sequence of integer IDs that the model
  can process.

  Example:
    Text:    "Hello, how are you?"
    Tokens:  ["Hello", ",", " how", " are", " you", "?"]
    IDs:     [9906, 11, 1268, 527, 499, 30]

  Each model has its own tokenizer trained alongside it. The tokenizer defines:
    • Vocabulary — the set of all tokens the model knows (~128K for Llama 3)
    • Merge rules — how characters get combined into tokens (BPE algorithm)
    • Special tokens — control tokens like <|begin_of_text|>, <|eot_id|>

  Llama 3 uses Byte-Pair Encoding (BPE) via tiktoken:
    - Starts with individual bytes (256 base tokens)
    - Iteratively merges the most frequent adjacent pairs
    - Common words become single tokens ("the" → 1 token)
    - Rare words get split into subwords ("tokenization" → "token" + "ization")
    - This balances vocabulary size vs sequence length

Why the Tokenizer Must Match the Model:
  During pre-training, the model learned that token ID 9906 means "Hello".
  If you used a different tokenizer where ID 9906 means "purple", every
  input would be scrambled. Always use the tokenizer that ships with the model.

Chat Template:
  Beyond basic tokenization, instruction-tuned models expect a specific
  conversation format. Llama 3 uses special tokens to mark role boundaries:

    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful assistant.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is Python?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    Python is a programming language...<|eot_id|>

  The tokenizer.apply_chat_template() method handles this formatting
  automatically, ensuring training data matches the model's expected structure.

Best Practices for Tokenization in Fine-Tuning:
  1. Always use AutoTokenizer.from_pretrained(model_name) — never a generic one
  2. Set a max_seq_length to control VRAM usage (512 here; longer = more memory)
  3. Truncate, don't pad — the data collator handles dynamic padding at batch time
  4. Copy input_ids to labels — for causal LM, the model predicts the next token
  5. Use the chat template — instruction-tuned models need the role markers

Supports two modes (set in training_config.yaml):
  save_dataset_locally: true   → saves to ./data/, loads from disk next time
  save_dataset_locally: false  → streams from HuggingFace cache (~/.cache/)

Loads HuggingFace API token from Keys.env (searches current dir + 3 levels up).
Validates token and model access before proceeding.
Converts instruction datasets into the Llama 3 chat template format.

Token Loading (no external dependencies like python-dotenv):
  Searches for Keys.env in script dir → 3 levels up → CWD → home dir.
  Supports: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACE_HUB_TOKEN
  Supports: export prefix, quoted values, comments
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import login


# ── Default model ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


# ──────────────────────────────────────────────────────────────────────────────
# Environment / Auth  (self-contained — no python-dotenv required)
# ──────────────────────────────────────────────────────────────────────────────

def load_env_file(env_file: str = "Keys.env") -> tuple[str | None, str | None]:
    """
    Load HuggingFace token from an env file.

    Supports formats:
        HF_TOKEN=hf_xxx
        HUGGINGFACE_TOKEN=hf_xxx
        HF_TOKEN="hf_xxx"
        export HF_TOKEN=hf_xxx

    Search order:
        1. Script's own directory
        2. 1 level up from script
        3. 2 levels up from script
        4. 3 levels up from script
        5. Current working directory
        6. Home directory (~/)

    Returns: (token, key_name) or (None, None)
    """
    possible_keys = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]

    # Build search paths: script dir + 3 levels up
    search_paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    for _ in range(4):  # script dir + 3 levels up
        search_paths.append(os.path.join(current, env_file))
        current = os.path.dirname(current)

    # Also check CWD and home dir as fallback
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
        with open(found_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Remove 'export ' prefix if present
                if line.startswith('export '):
                    line = line[7:]

                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                            (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    # Check if this is a token key
                    if key in possible_keys:
                        print(f"  ✅ Loaded {key} from {env_file}")
                        return value, key

        return None, None

    except Exception as e:
        print(f"  ⚠️  Error reading {found_path}: {e}")
        return None, None


def load_hf_token(env_file: str = "Keys.env") -> str | None:
    """
    Load HuggingFace token and authenticate.

    Priority:
      1. Environment variable (HF_TOKEN / HUGGINGFACE_TOKEN)
      2. Keys.env file (self-contained parser, no python-dotenv)

    Sets os.environ["HF_TOKEN"] so transformers/hub pick it up.
    Returns the token so callers can pass it directly to API calls.
    """
    token = None
    source = None

    # 1. Check environment variables first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        source = "environment variable"

    # 2. Fall back to Keys.env file
    if not token:
        token, key_name = load_env_file(env_file)
        if token:
            source = f"{env_file} ({key_name})"

    if not token:
        print(f"  ⚠️  No HF token found in environment or {env_file}")
        print(f"     Create a {env_file} file with: HF_TOKEN=hf_your_token_here")
        print(f"     Or set the HF_TOKEN environment variable")
        return None

    # ── Force the correct token everywhere ───────────────────────────────
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    login(token=token)

    masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
    print(f"  🔑 Authenticated with HuggingFace (token: {masked})")
    if source:
        print(f"     Source: {source}")

    return token


def validate_token_and_model(token: str, model_name: str) -> bool:
    """
    Verify the token is valid and has DOWNLOAD access to the target model.

    Checks:
      1. Token is valid (whoami)
      2. Account details
      3. Model exists and is accessible (metadata)
      4. If gated, actually tests file download access (not just metadata)
         — model_info() can succeed even without download permission

    Returns True if everything passes, False otherwise.
    """
    from huggingface_hub import HfApi, model_info, hf_hub_download

    print(f"\n── Validating Token & Model Access ──")

    # ── 1. Token validity ────────────────────────────────────────────────
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info.get("name", "unknown")
        print(f"  ✅ Token valid — logged in as: {username}")
    except Exception as e:
        print(f"  ❌ Token is INVALID or expired")
        print(f"     Error: {e}")
        print(f"     → Get a new token: https://huggingface.co/settings/tokens")
        return False

    # ── 2. Model metadata ────────────────────────────────────────────────
    print(f"  🔍 Checking access to: {model_name}")

    try:
        info = model_info(model_name, token=token)
        gated = info.gated  # False, "manual", or "auto"

        if gated:
            print(f"  ℹ️  Model is gated (type: {gated}) — testing download access...")
        else:
            print(f"  ✅ Model is public (no gate): {model_name}")

        # Show model details
        if info.safetensors and info.safetensors.total:
            params = info.safetensors.total
            print(f"     Parameters: {params:,} ({params / 1e9:.2f}B)")

        # ── 3. Actual download test (the real gate check) ────────────────
        # model_info() only reads metadata — it can succeed even without
        # download permission. We must test downloading an actual file.
        if gated:
            try:
                import tempfile
                hf_hub_download(
                    repo_id=model_name,
                    filename="config.json",
                    token=token,
                    cache_dir=tempfile.mkdtemp(),
                )
                print(f"  ✅ Download access CONFIRMED for: {model_name}")
            except Exception as dl_err:
                dl_str = str(dl_err)
                if "403" in dl_str or "gated" in dl_str.lower():
                    print(f"  ❌ Download access DENIED to: {model_name}")
                    print(f"     ⚠️  model_info() succeeded but file downloads are blocked.")
                    print(f"     Your account ({username}) has not been granted download access.")
                    print(f"     → Visit: https://huggingface.co/{model_name}")
                    print(f"       Click 'Agree and access repository' and wait for approval.")
                    print(f"\n  💡 Ungated alternatives you can use right now:")
                    _suggest_alternatives(model_name)
                    return False
                else:
                    print(f"  ❌ Download test failed: {dl_err}")
                    return False

        return True

    except Exception as e:
        error_str = str(e)

        if "403" in error_str or "gated" in error_str.lower():
            print(f"  ❌ Access DENIED to: {model_name}")
            print(f"     Your account ({username}) has not been granted access.")
            print(f"     → Visit: https://huggingface.co/{model_name}")
            print(f"       Click 'Agree and access repository' and wait for approval.")
            print(f"\n  💡 Ungated alternatives you can use right now:")
            _suggest_alternatives(model_name)

        elif "401" in error_str:
            print(f"  ❌ Authentication rejected for: {model_name}")
            print(f"     → Regenerate your token: https://huggingface.co/settings/tokens")

        elif "404" in error_str:
            print(f"  ❌ Model not found: {model_name}")
            print(f"     → Check spelling in training_config.yaml")
            print(f"     → Search: https://huggingface.co/models?search={model_name.split('/')[-1]}")

        else:
            print(f"  ❌ Unexpected error: {e}")

        return False


def _suggest_alternatives(model_name: str):
    """Suggest ungated alternatives based on the model they tried to access."""

    alternatives = {
        "llama-3-8b-instruct": [
            ("unsloth/llama-3-8b-Instruct", "Same model, no gate"),
            ("NousResearch/Meta-Llama-3-8B-Instruct", "Community mirror"),
        ],
        "llama-3-8b": [
            ("unsloth/llama-3-8b-Instruct", "Same model, no gate"),
            ("NousResearch/Meta-Llama-3-8B-Instruct", "Community mirror"),
        ],
        "llama-3.2-1b": [
            ("unsloth/Llama-3.2-1B-Instruct", "Same model, no gate"),
        ],
        "llama-3.2-3b": [
            ("unsloth/Llama-3.2-3B-Instruct", "Same model, no gate"),
        ],
        "llama-3.1-8b": [
            ("unsloth/Meta-Llama-3.1-8B-Instruct", "Same model, no gate"),
        ],
    }

    model_lower = model_name.lower()
    matched = False

    for key, alts in alternatives.items():
        if key in model_lower:
            for alt_name, desc in alts:
                print(f"       • {alt_name}  ({desc})")
            matched = True
            break

    if not matched:
        print(f"       • Try searching: https://huggingface.co/unsloth")
        print(f"       • Or use: openai-community/gpt2 (for testing pipeline)")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_alpaca_to_chat(example: dict) -> dict:
    """
    Convert Alpaca-format examples to Llama 3 chat messages.

    Alpaca format:  { "instruction": ..., "input": ..., "output": ... }
    Chat format:    [ {"role": "user", "content": ...}, {"role": "assistant", "content": ...} ]
    """
    instruction = example["instruction"]
    context = example.get("input", "")
    response = example["output"]

    if context and context.strip():
        user_content = f"{instruction}\n\nContext: {context}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]

    return {"messages": messages}


def tokenize_chat(example: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Apply the chat template and tokenize.

    Pipeline:
      1. apply_chat_template() — wraps messages in Llama 3's special tokens
         (e.g., <|start_header_id|>user<|end_header_id|>) producing a single string
      2. tokenizer() — converts that string into integer token IDs using BPE
      3. labels = input_ids — for causal LM training, the model learns to predict
         each token from the previous ones (next-token prediction)

    Args:
        example: Dict with "messages" key (list of role/content dicts)
        tokenizer: The model's tokenizer (must match the model being fine-tuned)
        max_seq_length: Truncate sequences longer than this (saves VRAM)
    """
    # Step 1: Format messages into the model's expected chat template string
    # tokenize=False means we get a string back, not token IDs yet
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    # Step 2: Convert the formatted string into token IDs
    # padding=False — the DataCollator handles padding dynamically per batch
    # truncation=True — cut sequences that exceed max_seq_length
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    # Step 3: For causal language modeling, labels = input_ids
    # The model learns: given tokens [0..n-1], predict token [n]
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────────────────────

def _get_local_path(config: dict) -> Path:
    """Get the local save path for the processed dataset."""
    local_dir = Path(config.get("local_data_dir", "./data"))
    dataset_folder = config["dataset_name"].split("/")[-1] + "-processed"
    return local_dir / dataset_folder


def load_and_prepare_dataset(config: dict, tokenizer):
    """
    Main entry point: load dataset, format, tokenize, split.

    If save_dataset_locally=true in config:
      - First run:  downloads → processes → saves to ./data/
      - Next runs:  loads from ./data/ (no internet needed)

    If save_dataset_locally=false:
      - Uses HuggingFace cache at ~/.cache/huggingface/datasets/

    Returns: (train_dataset, eval_dataset)
    """
    save_locally = config.get("save_dataset_locally", False)
    local_path = _get_local_path(config)

    # ── Try loading from local disk first ────────────────────────────────
    if save_locally and local_path.exists():
        print(f"📂 Loading processed dataset from disk: {local_path}")
        processed = load_from_disk(str(local_path))
        train_data = processed["train"]
        eval_data = processed["eval"]
        print(f"   ✅ Train: {len(train_data)} | Eval: {len(eval_data)} (loaded from disk)")
        return train_data, eval_data

    # ── Download from HuggingFace ────────────────────────────────────────
    print(f"📥 Downloading dataset: {config['dataset_name']}")
    dataset = load_dataset(config["dataset_name"])

    print("💬 Formatting to chat template...")
    dataset = dataset.map(
        format_alpaca_to_chat,
        remove_columns=dataset["train"].column_names,
        desc="Formatting",
    )

    print(f"🔤 Tokenizing (max_length={config['max_seq_length']})...")
    dataset = dataset.map(
        lambda ex: tokenize_chat(ex, tokenizer, config["max_seq_length"]),
        remove_columns=["messages"],
        desc="Tokenizing",
    )

    # Split
    train_data = dataset["train"]
    if "test" not in dataset:
        split = train_data.train_test_split(test_size=0.1, seed=config.get("seed", 42))
        train_data = split["train"]
        eval_data = split["test"]
    else:
        eval_data = dataset["test"]

    print(f"   ✅ Train: {len(train_data)} | Eval: {len(eval_data)}")

    # ── Save locally if configured ───────────────────────────────────────
    if save_locally:
        print(f"💾 Saving processed dataset to: {local_path}")
        local_path.mkdir(parents=True, exist_ok=True)

        from datasets import DatasetDict
        to_save = DatasetDict({"train": train_data, "eval": eval_data})
        to_save.save_to_disk(str(local_path))
        print(f"   ✅ Saved! Next run loads from disk (no internet needed)")

    return train_data, eval_data


# ---------------------------------------------------------------------------
# Standalone: preview formatted examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("  Dataset Preparation — unsloth/Llama-3.2-1B-Instruct")
    print("=" * 60)

    # 1. Authenticate (self-contained — no python-dotenv needed)
    token = load_hf_token()
    if not token:
        print("\n  ❌ Cannot proceed without a valid HF token.")
        sys.exit(1)

    # 2. Load config
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"  ⚠️  Config not found at {config_path}, using defaults")
        config = {
            "model_name": DEFAULT_MODEL,
            "dataset_name": "tatsu-lab/alpaca",
            "max_seq_length": 2048,
            "seed": 42,
            "save_dataset_locally": True,
        }

    # Ensure model_name defaults to unsloth/Llama-3.2-1B-Instruct
    config.setdefault("model_name", DEFAULT_MODEL)

    # 3. Validate token + model access BEFORE downloading anything
    if not validate_token_and_model(token, config["model_name"]):
        print("\n  ❌ Cannot proceed — fix the issues above and try again.")
        sys.exit(1)

    # 4. Load tokenizer (only runs if validation passed)
    print(f"\n📦 Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Prepare dataset
    train_data, eval_data = load_and_prepare_dataset(config, tokenizer)

    # 6. Preview
    print("\n" + "=" * 60)
    print("  PREVIEW: First training example (decoded)")
    print("=" * 60)
    sample = train_data[0]
    print(tokenizer.decode(sample["input_ids"][:200], skip_special_tokens=False))
    print("...")

