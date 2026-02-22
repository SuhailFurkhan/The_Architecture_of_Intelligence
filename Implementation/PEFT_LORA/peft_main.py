"""
peft_main.py — Master Controller for PEFT / LoRA Fine-Tuning Pipeline

Mirrors the structure of your Full Fine-Tuning main.py.

Usage:
    python peft_main.py                  # Interactive menu
    python peft_main.py --run all        # Full pipeline
    python peft_main.py --run prepare    # Data prep only
    python peft_main.py --run train      # Training only
    python peft_main.py --run inference  # Inference only
    python peft_main.py --run compare    # Comparison only
    python peft_main.py --run vram       # VRAM check only
    python peft_main.py --run token      # Token verification only
    python peft_main.py --run train --yes  # Auto-confirm all prompts
"""

import os
import sys
import time
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

AUTO_CONFIRM = False

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       LLM PEFT / LoRA Fine-Tuning Pipeline — Master Controller       ║
║                                                                      ║
║  PEFT = Parameter-Efficient Fine-Tuning                              ║
║  LoRA = Low-Rank Adaptation (Hu et al., 2021)                        ║
║                                                                      ║
║  Why LoRA?                                                           ║
║    Full fine-tuning trains ALL 1.24B parameters.                     ║
║    LoRA trains only ~10M adapter params (0.82%) — 120x fewer.        ║
║    Same task performance, fraction of the VRAM and time.             ║
║                                                                      ║
║  Modules:                                                            ║
║    1. Token Verification    — Validate HuggingFace credentials       ║
║    2. VRAM Check            — LoRA-aware GPU memory estimation       ║
║    3. Data Preparation      — Format with response-only masking      ║
║    4. LoRA Training         — Adapter-only training loop             ║
║    5. Inference             — Test adapter (auto-loads base+adapter) ║
║    6. Compare               — Original vs LoRA + adapter analysis    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

TRAINING_WARNING = """
\033[93m\033[1m
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                WARNING: LORA TRAINING TIME ESTIMATE                        ║
║                                                                            ║
║   LoRA is FASTER than full fine-tuning but still takes significant time.   ║
║                                                                            ║
║   Estimated time (1B model, 52K examples, 3 epochs):                      ║
║     • RTX 3090 (24 GB):    ~1-2 hours    (vs ~3-6h full FT)              ║
║     • RTX 4090 (24 GB):    ~40min-1h     (vs ~2-4h full FT)              ║
║     • A100 (40/80 GB):     ~20-40 min    (vs ~1-2h full FT)              ║
║     • CPU only:            Hours/Days    (not recommended)                ║
║                                                                            ║
║   Key differences from full fine-tuning:                                  ║
║     ✓ Only ~10M adapter params updated (vs 1.24B)                         ║
║     ✓ Larger batch size (4 instead of 1)                                  ║
║     ✓ Higher learning rate safe (2e-4 vs 2e-5)                            ║
║     ✓ Saves only ~15 MB adapter (vs 2.5 GB full model)                    ║
║                                                                            ║
║   Do NOT close the terminal during training.                               ║
║   Checkpoints are saved every 400 steps so you can resume.                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
\033[0m"""


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    width = 65
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def confirm_proceed(message: str = "Do you want to proceed?") -> bool:
    if AUTO_CONFIRM:
        print(f"\n  {message} (auto-confirmed: --yes)")
        return True
    while True:
        response = input(f"\n  {message} (yes/no): ").strip().lower()
        if response in ("yes", "y"):
            return True
        if response in ("no", "n"):
            return False
        print("  Please enter 'yes' or 'no'.")


def load_config() -> dict:
    """Load peft_training_config.yaml."""
    import yaml

    config_path = Path(__file__).parent / "peft_training_config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "peft_training_config.yaml"

    if not config_path.exists():
        print(f"  ⚠️  peft_training_config.yaml not found. Using defaults.")
        return {
            "model_name": "unsloth/Llama-3.2-1B-Instruct",
            "dataset_name": "yahma/alpaca-cleaned",
            "max_seq_length": 512,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "bf16": True,
            "gradient_checkpointing": True,
            "output_dir": "./outputs/llama-3.2-1B-lora",
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 200,
            "save_strategy": "steps",
            "save_steps": 400,
            "save_total_limit": 2,
            "seed": 42,
            "merge_before_save": False,
        }

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"  Config loaded: {config_path}")
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Steps
# ──────────────────────────────────────────────────────────────────────────────

def run_token_check():
    """Step 1: HuggingFace token verification (reuses your existing check)."""
    print_header("Step 1: HuggingFace Token Verification")

    # Reuse the same token verification script
    try:
        from HF_Token_Verificaiton import (
            check_system_env, check_keys_env,
            check_conflict, check_token_valid, check_model_access,
        )
    except ImportError:
        print("  ⚠️  HF_Token_Verificaiton.py not found.")
        print("  Falling back to simple token check...")
        from peft_prepare_data import load_hf_token
        token = load_hf_token()
        return token is not None

    config = load_config()
    model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")

    sys_token = check_system_env()
    env_path, env_token = check_keys_env()
    check_conflict(sys_token, env_token)
    active_token = sys_token or env_token

    if active_token:
        user_info = check_token_valid(active_token)
        if user_info:
            check_model_access(active_token, model_name)
            print("\n  ✅ Token verification passed!")
            return True
        else:
            print("\n  ❌ Token is invalid.")
            return False
    else:
        print("\n  ❌ No token found.")
        return False


def run_vram_check():
    """Step 2: LoRA-aware VRAM estimation."""
    print_header("Step 2: VRAM Check (LoRA Mode)")

    from peft_check_vram import main as vram_main
    vram_main()
    return True


def run_data_preparation():
    """Step 3: Prepare dataset with response-only masking."""
    print_header("Step 3: Data Preparation (LoRA — Response Masking)")

    from peft_prepare_data import load_hf_token, load_and_prepare_dataset
    from transformers import AutoTokenizer

    config = load_config()

    token = load_hf_token()
    if not token:
        print("  ❌ No HF token. Create Keys.env with HF_TOKEN=hf_xxx")
        return None, None, None

    print(f"\n  📦 Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n  📊 Loading dataset with response-only masking...")
    train_data, eval_data = load_and_prepare_dataset(config, tokenizer, mask_instruction=True)

    print(f"\n  ✅ Data preparation complete!")
    print(f"     Train: {len(train_data):,} examples | Eval: {len(eval_data):,} examples")
    print(f"     Masking: instruction tokens set to -100 (excluded from loss)")

    return train_data, eval_data, tokenizer


def run_training(train_data=None, eval_data=None, tokenizer=None):
    """Step 4: LoRA fine-tuning."""
    print_header("Step 4: LoRA Fine-Tuning")

    config = load_config()

    # Show warning and get confirmation
    print(TRAINING_WARNING)

    summary = (
        f"  About to start LoRA training with:\n"
        f"    Model:          {config['model_name']}\n"
        f"    LoRA rank:      r={config.get('lora_r', 16)}  |  alpha={config.get('lora_alpha', 32)}\n"
        f"    Target modules: {config.get('lora_target_modules', ['q_proj', 'v_proj'])}\n"
        f"    Epochs:         {config.get('num_train_epochs', 3)}\n"
        f"    Learning rate:  {config.get('learning_rate', 2e-4)}\n"
        f"    Output dir:     {config.get('output_dir', './outputs/llama-3.2-1B-lora')}\n"
        f"    Save strategy:  {'adapter only (~15 MB)' if not config.get('merge_before_save') else 'merged model (~2.5 GB)'}\n"
    )
    print(summary)

    if not confirm_proceed("Ready to start LoRA training?"):
        print("  Training cancelled.")
        return False

    from peft_train import train
    start_time = time.time()

    if train_data is not None and eval_data is not None and tokenizer is not None:
        train(train_data, eval_data, tokenizer, config)
    else:
        train(config=config)

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n  ⏱️  Training completed in {int(hours)}h {int(mins)}m {int(secs)}s")
    return True


def run_inference(prompt: str = None):
    """Step 5: Test the fine-tuned model."""
    print_header("Step 5: Inference Test")

    config = load_config()
    adapter_path = os.path.join(config.get("output_dir", "./outputs/llama-3.2-1B-lora"), "final")

    if not Path(adapter_path).exists():
        print(f"  ❌ No trained adapter found at: {adapter_path}")
        print(f"     Run training first (option 4).")
        return False

    from peft_inference import load_model, generate
    model, tokenizer, model_type = load_model(adapter_path)

    print(f"  Model type: {model_type}")

    if prompt:
        response = generate(model, tokenizer, prompt)
        print(f"\n  💬 Prompt:   {prompt}")
        print(f"  🤖 Response: {response}")
    else:
        print("\n  Interactive Chat (type 'quit' to exit)\n")
        while True:
            user_input = input("  You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            response = generate(model, tokenizer, user_input)
            print(f"  Bot: {response}\n")

    return True


def run_compare():
    """Step 6: Compare original vs LoRA fine-tuned."""
    print_header("Step 6: Model Comparison (Original vs LoRA)")

    from peft_compare import compare
    config = load_config()

    original_path = config["model_name"]
    adapter_path = os.path.join(config.get("output_dir", "./outputs/llama-3.2-1B-lora"), "final")

    if not Path(adapter_path).exists():
        print(f"  ❌ No adapter found at: {adapter_path}")
        print(f"     Run training first (option 4).")
        return False

    compare(original_path, adapter_path)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline():
    """Run the complete PEFT pipeline: token → vram → data → train → compare."""
    print_header("Running Full PEFT Pipeline")

    print("\n  Step 1/6: Verifying HuggingFace token...")
    if not run_token_check():
        print("\n  Pipeline aborted: token verification failed.")
        return

    if not confirm_proceed("Token verified. Continue to VRAM check?"):
        return

    print("\n  Step 2/6: Checking VRAM requirements...")
    run_vram_check()

    if not confirm_proceed("VRAM check done. Continue to data preparation?"):
        return

    print("\n  Step 3/6: Preparing dataset...")
    train_data, eval_data, tokenizer = run_data_preparation()
    if train_data is None:
        print("\n  Pipeline aborted: data preparation failed.")
        return

    print("\n  Step 4/6: LoRA Training...")
    training_success = run_training(train_data, eval_data, tokenizer)
    if not training_success:
        print("\n  Pipeline stopped: training was skipped or failed.")
        return

    print("\n  Step 5/6: Quick inference test...")
    run_inference(prompt="What is machine learning? Explain in 2 sentences.")

    if confirm_proceed("Run full comparison (loads both models — needs extra VRAM)?"):
        print("\n  Step 6/6: Comparing models...")
        run_compare()

    print_header("PEFT Pipeline Complete!")
    print("  ✅ All steps finished successfully.")
    print(f"  Your LoRA adapter is ready for use.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Menu
# ──────────────────────────────────────────────────────────────────────────────

def interactive_menu():
    print(BANNER)

    menu_options = {
        "1": ("Verify HuggingFace Token",             run_token_check),
        "2": ("Check VRAM Requirements (LoRA-aware)",  run_vram_check),
        "3": ("Prepare Dataset (with response masking)", run_data_preparation),
        "4": ("Start LoRA Training",                   run_training),
        "5": ("Test Model (Inference)",                run_inference),
        "6": ("Compare Original vs LoRA Fine-Tuned",   run_compare),
        "7": ("Run Full Pipeline (1 → 6)",             run_full_pipeline),
        "0": ("Exit",                                  None),
    }

    while True:
        print("\n  ┌───────────────────────────────────────────────┐")
        print("  │         PEFT / LoRA Pipeline Menu              │")
        print("  ├───────────────────────────────────────────────┤")
        for key, (label, _) in menu_options.items():
            prefix = " │"
            if key == "7":
                print("  ├────────────────────────────────────────────────┤")
            print(f"{prefix}   [{key}]  {label:<42s}│")
        print("  └───────────────────────────────────────────────┘")

        choice = input("\n  Enter your choice (0-7): ").strip()

        if choice == "0":
            print("\n  Goodbye!\n")
            break

        if choice in menu_options:
            label, func = menu_options[choice]
            if func is not None:
                try:
                    func()
                except KeyboardInterrupt:
                    print("\n\n  Operation interrupted by user.")
                except Exception as e:
                    print(f"\n  Error during '{label}': {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("  Invalid choice. Please enter 0-7.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Master controller for the PEFT / LoRA fine-tuning pipeline."
    )
    parser.add_argument(
        "--run",
        choices=["all", "token", "vram", "prepare", "train", "inference", "compare"],
        default=None,
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--yes", action="store_true", default=False,
                        help="Auto-confirm all prompts (for non-interactive use).")

    args = parser.parse_args()

    if args.yes:
        global AUTO_CONFIRM
        AUTO_CONFIRM = True

    if args.run is None:
        interactive_menu()
    else:
        dispatch = {
            "all":       run_full_pipeline,
            "token":     run_token_check,
            "vram":      run_vram_check,
            "prepare":   run_data_preparation,
            "train":     run_training,
            "inference": lambda: run_inference(prompt=args.prompt),
            "compare":   run_compare,
        }
        try:
            dispatch[args.run]()
        except KeyboardInterrupt:
            print("\n\n  Operation interrupted by user.")
            sys.exit(1)


if __name__ == "__main__":
    main()


# """
# peft_main.py — Master Controller for PEFT / LoRA Fine-Tuning Pipeline
#
# Mirrors the structure of your Full Fine-Tuning main.py.
#
# Usage:
#     python peft_main.py                  # Interactive menu
#     python peft_main.py --run all        # Full pipeline
#     python peft_main.py --run prepare    # Data prep only
#     python peft_main.py --run train      # Training only
#     python peft_main.py --run inference  # Inference only
#     python peft_main.py --run compare    # Comparison only
#     python peft_main.py --run vram       # VRAM check only
#     python peft_main.py --run token      # Token verification only
#     python peft_main.py --run train --yes  # Auto-confirm all prompts
# """
#
# import os
# import sys
# import time
# import argparse
# from pathlib import Path
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Constants
# # ──────────────────────────────────────────────────────────────────────────────
#
# AUTO_CONFIRM = False
#
# BANNER = r"""
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║       LLM PEFT / LoRA Fine-Tuning Pipeline — Master Controller       ║
# ║                                                                      ║
# ║  PEFT = Parameter-Efficient Fine-Tuning                              ║
# ║  LoRA = Low-Rank Adaptation (Hu et al., 2021)                        ║
# ║                                                                      ║
# ║  Why LoRA?                                                           ║
# ║    Full fine-tuning trains ALL 1.24B parameters.                     ║
# ║    LoRA trains only ~10M adapter params (0.82%) — 120x fewer.        ║
# ║    Same task performance, fraction of the VRAM and time.             ║
# ║                                                                      ║
# ║  Modules:                                                            ║
# ║    1. Token Verification    — Validate HuggingFace credentials       ║
# ║    2. VRAM Check            — LoRA-aware GPU memory estimation       ║
# ║    3. Data Preparation      — Format with response-only masking      ║
# ║    4. LoRA Training         — Adapter-only training loop             ║
# ║    5. Inference             — Test adapter (auto-loads base+adapter) ║
# ║    6. Compare               — Original vs LoRA + adapter analysis    ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
# """
#
# TRAINING_WARNING = """
# \033[93m\033[1m
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║                WARNING: LORA TRAINING TIME ESTIMATE                        ║
# ║                                                                            ║
# ║   LoRA is FASTER than full fine-tuning but still takes significant time.   ║
# ║                                                                            ║
# ║   Estimated time (1B model, 52K examples, 3 epochs):                      ║
# ║     • RTX 3090 (24 GB):    ~1-2 hours    (vs ~3-6h full FT)              ║
# ║     • RTX 4090 (24 GB):    ~40min-1h     (vs ~2-4h full FT)              ║
# ║     • A100 (40/80 GB):     ~20-40 min    (vs ~1-2h full FT)              ║
# ║     • CPU only:            Hours/Days    (not recommended)                ║
# ║                                                                            ║
# ║   Key differences from full fine-tuning:                                  ║
# ║     ✓ Only ~10M adapter params updated (vs 1.24B)                         ║
# ║     ✓ Larger batch size (4 instead of 1)                                  ║
# ║     ✓ Higher learning rate safe (2e-4 vs 2e-5)                            ║
# ║     ✓ Saves only ~15 MB adapter (vs 2.5 GB full model)                    ║
# ║                                                                            ║
# ║   Do NOT close the terminal during training.                               ║
# ║   Checkpoints are saved every 400 steps so you can resume.                 ║
# ║                                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# \033[0m"""
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Utilities
# # ──────────────────────────────────────────────────────────────────────────────
#
# def print_header(title: str):
#     width = 65
#     print()
#     print("=" * width)
#     print(f"  {title}")
#     print("=" * width)
#     print()
#
#
# def confirm_proceed(message: str = "Do you want to proceed?") -> bool:
#     if AUTO_CONFIRM:
#         print(f"\n  {message} (auto-confirmed: --yes)")
#         return True
#     while True:
#         response = input(f"\n  {message} (yes/no): ").strip().lower()
#         if response in ("yes", "y"):
#             return True
#         if response in ("no", "n"):
#             return False
#         print("  Please enter 'yes' or 'no'.")
#
#
# def load_config() -> dict:
#     """Load peft_training_config.yaml."""
#     import yaml
#
#     config_path = Path(__file__).parent / "peft_training_config.yaml"
#     if not config_path.exists():
#         config_path = Path(__file__).parent.parent / "configs" / "peft_training_config.yaml"
#
#     if not config_path.exists():
#         print(f"  ⚠️  peft_training_config.yaml not found. Using defaults.")
#         return {
#             "model_name": "unsloth/Llama-3.2-1B-Instruct",
#             "dataset_name": "yahma/alpaca-cleaned",
#             "max_seq_length": 512,
#             "lora_r": 16,
#             "lora_alpha": 32,
#             "lora_dropout": 0.05,
#             "lora_bias": "none",
#             "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
#                                     "gate_proj", "up_proj", "down_proj"],
#             "per_device_train_batch_size": 4,
#             "per_device_eval_batch_size": 4,
#             "gradient_accumulation_steps": 4,
#             "num_train_epochs": 3,
#             "learning_rate": 2e-4,
#             "weight_decay": 0.01,
#             "warmup_ratio": 0.03,
#             "lr_scheduler_type": "cosine",
#             "bf16": True,
#             "gradient_checkpointing": True,
#             "output_dir": "./outputs/llama-3.2-1B-lora",
#             "logging_steps": 10,
#             "eval_strategy": "steps",
#             "eval_steps": 200,
#             "save_strategy": "steps",
#             "save_steps": 400,
#             "save_total_limit": 2,
#             "seed": 42,
#             "merge_before_save": False,
#         }
#
#     with open(config_path, encoding="utf-8") as f:
#         config = yaml.safe_load(f)
#
#     print(f"  Config loaded: {config_path}")
#     return config
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Pipeline Steps
# # ──────────────────────────────────────────────────────────────────────────────
#
# def run_token_check():
#     """Step 1: HuggingFace token verification (reuses your existing check)."""
#     print_header("Step 1: HuggingFace Token Verification")
#
#     # Reuse the same token verification script
#     try:
#         from HF_Token_Verificaiton import (
#             check_system_env, check_keys_env,
#             check_conflict, check_token_valid, check_model_access,
#         )
#     except ImportError:
#         print("  ⚠️  HF_Token_Verificaiton.py not found.")
#         print("  Falling back to simple token check...")
#         from peft_prepare_data import load_hf_token
#         token = load_hf_token()
#         return token is not None
#
#     config = load_config()
#     model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
#
#     sys_token = check_system_env()
#     env_path, env_token = check_keys_env()
#     check_conflict(sys_token, env_token)
#     active_token = sys_token or env_token
#
#     if active_token:
#         user_info = check_token_valid(active_token)
#         if user_info:
#             check_model_access(active_token, model_name)
#             print("\n  ✅ Token verification passed!")
#             return True
#         else:
#             print("\n  ❌ Token is invalid.")
#             return False
#     else:
#         print("\n  ❌ No token found.")
#         return False
#
#
# def run_vram_check():
#     """Step 2: LoRA-aware VRAM estimation."""
#     print_header("Step 2: VRAM Check (LoRA Mode)")
#
#     from peft_check_vram import main as vram_main
#     vram_main()
#     return True
#
#
# def run_data_preparation():
#     """Step 3: Prepare dataset with response-only masking."""
#     print_header("Step 3: Data Preparation (LoRA — Response Masking)")
#
#     from peft_prepare_data import load_hf_token, load_and_prepare_dataset
#     from transformers import AutoTokenizer
#
#     config = load_config()
#
#     token = load_hf_token()
#     if not token:
#         print("  ❌ No HF token. Create Keys.env with HF_TOKEN=hf_xxx")
#         return None, None, None
#
#     print(f"\n  📦 Loading tokenizer: {config['model_name']}")
#     tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token, use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     print(f"\n  📊 Loading dataset with response-only masking...")
#     train_data, eval_data = load_and_prepare_dataset(config, tokenizer, mask_instruction=True)
#
#     print(f"\n  ✅ Data preparation complete!")
#     print(f"     Train: {len(train_data):,} examples | Eval: {len(eval_data):,} examples")
#     print(f"     Masking: instruction tokens set to -100 (excluded from loss)")
#
#     return train_data, eval_data, tokenizer
#
#
# def run_training(train_data=None, eval_data=None, tokenizer=None):
#     """Step 4: LoRA fine-tuning."""
#     print_header("Step 4: LoRA Fine-Tuning")
#
#     config = load_config()
#
#     # Show warning and get confirmation
#     print(TRAINING_WARNING)
#
#     summary = (
#         f"  About to start LoRA training with:\n"
#         f"    Model:          {config['model_name']}\n"
#         f"    LoRA rank:      r={config.get('lora_r', 16)}  |  alpha={config.get('lora_alpha', 32)}\n"
#         f"    Target modules: {config.get('lora_target_modules', ['q_proj', 'v_proj'])}\n"
#         f"    Epochs:         {config.get('num_train_epochs', 3)}\n"
#         f"    Learning rate:  {config.get('learning_rate', 2e-4)}\n"
#         f"    Output dir:     {config.get('output_dir', './outputs/llama-3.2-1B-lora')}\n"
#         f"    Save strategy:  {'adapter only (~15 MB)' if not config.get('merge_before_save') else 'merged model (~2.5 GB)'}\n"
#     )
#     print(summary)
#
#     if not confirm_proceed("Ready to start LoRA training?"):
#         print("  Training cancelled.")
#         return False
#
#     from peft_train import train
#     start_time = time.time()
#
#     if train_data is not None and eval_data is not None and tokenizer is not None:
#         train(train_data, eval_data, tokenizer, config)
#     else:
#         train(config=config)
#
#     elapsed = time.time() - start_time
#     hours, rem = divmod(elapsed, 3600)
#     mins, secs = divmod(rem, 60)
#     print(f"\n  ⏱️  Training completed in {int(hours)}h {int(mins)}m {int(secs)}s")
#     return True
#
#
# def run_inference(prompt: str = None):
#     """Step 5: Test the fine-tuned model."""
#     print_header("Step 5: Inference Test")
#
#     config = load_config()
#     adapter_path = os.path.join(config.get("output_dir", "./outputs/llama-3.2-1B-lora"), "final")
#
#     if not Path(adapter_path).exists():
#         print(f"  ❌ No trained adapter found at: {adapter_path}")
#         print(f"     Run training first (option 4).")
#         return False
#
#     from peft_inference import load_model, generate
#     model, tokenizer, model_type = load_model(adapter_path)
#
#     print(f"  Model type: {model_type}")
#
#     if prompt:
#         response = generate(model, tokenizer, prompt)
#         print(f"\n  💬 Prompt:   {prompt}")
#         print(f"  🤖 Response: {response}")
#     else:
#         print("\n  Interactive Chat (type 'quit' to exit)\n")
#         while True:
#             user_input = input("  You: ").strip()
#             if user_input.lower() in ("quit", "exit", "q"):
#                 break
#             if not user_input:
#                 continue
#             response = generate(model, tokenizer, user_input)
#             print(f"  Bot: {response}\n")
#
#     return True
#
#
# def run_compare():
#     """Step 6: Compare original vs LoRA fine-tuned."""
#     print_header("Step 6: Model Comparison (Original vs LoRA)")
#
#     from peft_compare import compare
#     config = load_config()
#
#     original_path = config["model_name"]
#     adapter_path = os.path.join(config.get("output_dir", "./outputs/llama-3.2-1B-lora"), "final")
#
#     if not Path(adapter_path).exists():
#         print(f"  ❌ No adapter found at: {adapter_path}")
#         print(f"     Run training first (option 4).")
#         return False
#
#     compare(original_path, adapter_path)
#     return True
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Full Pipeline
# # ──────────────────────────────────────────────────────────────────────────────
#
# def run_full_pipeline():
#     """Run the complete PEFT pipeline: token → vram → data → train → compare."""
#     print_header("Running Full PEFT Pipeline")
#
#     print("\n  Step 1/6: Verifying HuggingFace token...")
#     if not run_token_check():
#         print("\n  Pipeline aborted: token verification failed.")
#         return
#
#     if not confirm_proceed("Token verified. Continue to VRAM check?"):
#         return
#
#     print("\n  Step 2/6: Checking VRAM requirements...")
#     run_vram_check()
#
#     if not confirm_proceed("VRAM check done. Continue to data preparation?"):
#         return
#
#     print("\n  Step 3/6: Preparing dataset...")
#     train_data, eval_data, tokenizer = run_data_preparation()
#     if train_data is None:
#         print("\n  Pipeline aborted: data preparation failed.")
#         return
#
#     print("\n  Step 4/6: LoRA Training...")
#     training_success = run_training(train_data, eval_data, tokenizer)
#     if not training_success:
#         print("\n  Pipeline stopped: training was skipped or failed.")
#         return
#
#     print("\n  Step 5/6: Quick inference test...")
#     run_inference(prompt="What is machine learning? Explain in 2 sentences.")
#
#     if confirm_proceed("Run full comparison (loads both models — needs extra VRAM)?"):
#         print("\n  Step 6/6: Comparing models...")
#         run_compare()
#
#     print_header("PEFT Pipeline Complete!")
#     print("  ✅ All steps finished successfully.")
#     print(f"  Your LoRA adapter is ready for use.\n")
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Interactive Menu
# # ──────────────────────────────────────────────────────────────────────────────
#
# def interactive_menu():
#     print(BANNER)
#
#     menu_options = {
#         "1": ("Verify HuggingFace Token",             run_token_check),
#         "2": ("Check VRAM Requirements (LoRA-aware)",  run_vram_check),
#         "3": ("Prepare Dataset (with response masking)", run_data_preparation),
#         "4": ("Start LoRA Training",                   run_training),
#         "5": ("Test Model (Inference)",                run_inference),
#         "6": ("Compare Original vs LoRA Fine-Tuned",   run_compare),
#         "7": ("Run Full Pipeline (1 → 6)",             run_full_pipeline),
#         "0": ("Exit",                                  None),
#     }
#
#     while True:
#         print("\n  ┌───────────────────────────────────────────────┐")
#         print("  │         PEFT / LoRA Pipeline Menu              │")
#         print("  ├───────────────────────────────────────────────┤")
#         for key, (label, _) in menu_options.items():
#             prefix = " │"
#             if key == "7":
#                 print("  ├────────────────────────────────────────────────┤")
#             print(f"{prefix}   [{key}]  {label:<42s}│")
#         print("  └───────────────────────────────────────────────┘")
#
#         choice = input("\n  Enter your choice (0-7): ").strip()
#
#         if choice == "0":
#             print("\n  Goodbye!\n")
#             break
#
#         if choice in menu_options:
#             label, func = menu_options[choice]
#             if func is not None:
#                 try:
#                     func()
#                 except KeyboardInterrupt:
#                     print("\n\n  Operation interrupted by user.")
#                 except Exception as e:
#                     print(f"\n  Error during '{label}': {e}")
#                     import traceback
#                     traceback.print_exc()
#         else:
#             print("  Invalid choice. Please enter 0-7.")
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # CLI Entry Point
# # ──────────────────────────────────────────────────────────────────────────────
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Master controller for the PEFT / LoRA fine-tuning pipeline."
#     )
#     parser.add_argument(
#         "--run",
#         choices=["all", "token", "vram", "prepare", "train", "inference", "compare"],
#         default=None,
#     )
#     parser.add_argument("--prompt", type=str, default=None)
#     parser.add_argument("--yes", action="store_true", default=False,
#                         help="Auto-confirm all prompts (for non-interactive use).")
#
#     args = parser.parse_args()
#
#     if args.yes:
#         global AUTO_CONFIRM
#         AUTO_CONFIRM = True
#
#     if args.run is None:
#         interactive_menu()
#     else:
#         dispatch = {
#             "all":       run_full_pipeline,
#             "token":     run_token_check,
#             "vram":      run_vram_check,
#             "prepare":   run_data_preparation,
#             "train":     run_training,
#             "inference": lambda: run_inference(prompt=args.prompt),
#             "compare":   run_compare,
#         }
#         try:
#             dispatch[args.run]()
#         except KeyboardInterrupt:
#             print("\n\n  Operation interrupted by user.")
#             sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()
