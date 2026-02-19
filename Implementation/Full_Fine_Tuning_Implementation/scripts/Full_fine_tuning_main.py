"""
main.py — Master Controller for Full Fine-Tuning Pipeline

Orchestrates the entire fine-tuning workflow from a single entry point.

Usage:
    python main.py                  # Interactive menu
    python main.py --run all        # Run full pipeline (with confirmation)
    python main.py --run prepare    # Run only data preparation
    python main.py --run train      # Run only training
    python main.py --run inference  # Run only inference
    python main.py --run compare    # Run only comparison
    python main.py --run vram       # Run only VRAM check
    python main.py --run token      # Run only token verification
    python main.py --run train --yes  # Run training, auto-confirm all prompts
"""

import os
import sys
import time
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Global flag: when True, all confirm_proceed() calls auto-approve
AUTO_CONFIRM = False

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          LLM Full Fine-Tuning Pipeline — Master Controller           ║
║                                                                      ║
║   Modules:                                                           ║
║     1. Token Verification    — Validate HuggingFace credentials      ║
║     2. VRAM Check            — Estimate GPU memory requirements      ║
║     3. Data Preparation      — Download, format & tokenize dataset   ║
║     4. Training              — Full fine-tuning (all parameters)     ║
║     5. Inference             — Test your fine-tuned model            ║
║     6. Compare               — Side-by-side: original vs fine-tuned  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║             WARNING: TRAINING IS A TIME-CONSUMING PROCESS!               ║
# ║                                                                          ║
# ║  Full fine-tuning of even a 1B parameter model on 52K examples           ║
# ║  can take SEVERAL HOURS (3-8+ hours on an RTX 3090, longer on            ║
# ║  slower GPUs). The pipeline involves:                                    ║
# ║                                                                          ║
# ║    • Downloading model weights (~2.5 GB)                                 ║
# ║    • Downloading & processing the dataset (~52K examples)                ║
# ║    • ~17,000+ optimizer steps across 3 epochs                            ║
# ║    • Saving the full model (~2.5 GB) after training                      ║
# ║                                                                          ║
# ║  Make sure you have:                                                     ║
# ║    ✓  Stable power supply / UPS                                          ║
# ║    ✓  Sufficient disk space (~10 GB free)                                ║
# ║    ✓  A valid HuggingFace token with model access                        ║
# ║    ✓  Adequate GPU VRAM (run VRAM check first!)                          ║
# ║                                                                          ║
# ║  You will be asked to confirm before training begins.                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAINING_WARNING = """
\033[93m\033[1m
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║            WARNING: TRAINING IS EXTREMELY TIME-CONSUMING!                  ║
║                                                                            ║
║   Full fine-tuning will take SEVERAL HOURS to complete.                    ║
║                                                                            ║
║   Estimated time:                                                          ║
║     • RTX 3090 (24 GB):    ~3-6 hours                                      ║
║     • RTX 4090 (24 GB):    ~2-4 hours                                      ║
║     • A100 (40/80 GB):     ~1-2 hours                                      ║
║     • CPU only:            Days (not recommended)                          ║
║                                                                            ║
║   The process involves ~17,000+ optimizer steps across 3 epochs            ║
║   over 52K training examples. It CANNOT be paused and resumed              ║
║   (checkpoints are saved, but restarting picks up from the last one).      ║
║                                                                            ║
║   Do NOT close the terminal or shut down your machine during training.     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
\033[0m"""


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def print_header(title: str):
    """Print a formatted section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def confirm_proceed(message: str = "Do you want to proceed?") -> bool:
    """Ask the user for yes/no confirmation. Returns True if yes."""
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
    """Load the training config YAML file."""
    import yaml

    config_path = Path(__file__).parent / "configs" / "training_config.yaml"

    # Also try parent directory (if scripts/ is one level deep)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"

    if not config_path.exists():
        print(f"  Config not found. Searched:")
        print(f"     {Path(__file__).parent / 'configs' / 'training_config.yaml'}")
        print(f"     {Path(__file__).parent.parent / 'configs' / 'training_config.yaml'}")
        print(f"  Using default configuration values.")
        return {
            "model_name": "unsloth/Llama-3.2-1B-Instruct",
            "dataset_name": "yahma/alpaca-cleaned",
            "max_seq_length": 512,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "bf16": True,
            "gradient_checkpointing": True,
            "output_dir": "./outputs/llama-3.2-1B-full-ft",
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 200,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 2,
            "seed": 42,
            "save_dataset_locally": False,
        }

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"  Config loaded from: {config_path}")
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Steps
# ──────────────────────────────────────────────────────────────────────────────

def run_token_check():
    """Step 1: Verify HuggingFace token and model access."""
    print_header("Step 1: HuggingFace Token Verification")

    from HF_Token_Verificaiton import (
        check_system_env,
        check_keys_env,
        check_conflict,
        check_token_valid,
        check_model_access,
    )

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
            print("\n  Token verification passed!")
            return True
        else:
            print("\n  Token is invalid.")
            return False
    else:
        print("\n  No token found. Create a Keys.env file with: HF_TOKEN=hf_your_token_here")
        return False


def run_vram_check():
    """Step 2: Estimate VRAM requirements."""
    print_header("Step 2: VRAM Estimation")

    from check_vram import estimate_vram
    import torch

    config = load_config()

    # Known param counts
    param_counts = {
        "unsloth/Llama-3.2-1B-Instruct": 1_240_000_000,
        "meta-llama/Llama-3.2-1B-Instruct": 1_240_000_000,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1_100_000_000,
        "HuggingFaceTB/SmolLM2-360M-Instruct": 360_000_000,
        "openai-community/gpt2": 124_000_000,
    }

    model_name = config["model_name"]
    num_params = param_counts.get(model_name)

    if num_params is None:
        print(f"  ⚠️  Unknown model '{model_name}'. Cannot estimate VRAM.")
        print(f"     Add its parameter count to check_vram.py or main.py.")
        return False

    est = estimate_vram(
        num_params=num_params,
        batch_size=config["per_device_train_batch_size"],
        seq_len=config["max_seq_length"],
        bf16=config.get("bf16", True),
        grad_checkpoint=config.get("gradient_checkpointing", True),
    )

    print(f"  Model:             {model_name}")
    print(f"  Parameters:        {num_params / 1e9:.2f}B")
    print(f"  Model Weights:     {est['model_weights_gb']:.1f} GB")
    print(f"  Gradients:         {est['gradients_gb']:.1f} GB")
    print(f"  Optimizer (AdamW): {est['optimizer_states_gb']:.1f} GB")
    print(f"  Activations:       {est['activations_gb']:.1f} GB (estimated)")
    print(f"  {'─' * 40}")
    print(f"  TOTAL ESTIMATED:   {est['total_estimated_gb']:.1f} GB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  Your GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        if est['total_estimated_gb'] > gpu_mem:
            print(f"  Won't fit. Use a smaller model or reduce seq_length.")
            return False
        elif est['total_estimated_gb'] > gpu_mem * 0.90:
            print(f"  Tight! May OOM under certain conditions.")
            return True
        else:
            print(f"  Should fit with headroom.")
            return True
    else:
        print(f"\n  No GPU detected. Training on CPU will be extremely slow.")
        return True


def run_data_preparation():
    """Step 3: Download, format, and tokenize the dataset."""
    print_header("Step 3: Data Preparation")

    from prepare_data import load_hf_token, validate_token_and_model, load_and_prepare_dataset
    from transformers import AutoTokenizer

    config = load_config()

    # Authenticate
    print("  Authenticating with HuggingFace...")
    token = load_hf_token()
    if not token:
        print("\n  Cannot proceed without a valid HF token.")
        return None, None, None

    # Validate access
    if not validate_token_and_model(token, config["model_name"]):
        print("\n  Cannot access model. Fix the issues above and retry.")
        return None, None, None

    # Load tokenizer
    print(f"\n  Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    train_data, eval_data = load_and_prepare_dataset(config, tokenizer)

    print(f"\n  Data preparation complete!")
    print(f"    Train examples: {len(train_data):,}")
    print(f"    Eval examples:  {len(eval_data):,}")

    return train_data, eval_data, tokenizer


def run_training(train_data=None, eval_data=None, tokenizer=None):
    """
    Step 4: Full fine-tuning.

    ╔════════════════════════════════════════════════════════════════════╗
    ║    ️THIS STEP IS EXTREMELY TIME-CONSUMING!                           ║
    ║                                                                    ║
    ║  Expect 3-8+ hours depending on your GPU.                          ║
    ║  ~17,000 optimizer steps across 3 epochs on 52K examples.          ║
    ║                                                                    ║
    ║  The user MUST confirm before this step proceeds.                  ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    print_header("Step 4: Full Fine-Tuning")

    # ══════════════════════════════════════════════════════════════════════
    #   DISPLAY PROMINENT WARNING TO THE USER
    # ══════════════════════════════════════════════════════════════════════
    print(TRAINING_WARNING)

    if not confirm_proceed(
        "Training may take SEVERAL HOURS. Do you want to proceed?"
    ):
        print("\n  Training skipped by user.")
        return False

    print("\n User confirmed. Starting training pipeline...\n")

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from prepare_data import load_hf_token, load_and_prepare_dataset

    config = load_config()

    # If data wasn't passed in, prepare it now
    if train_data is None or eval_data is None or tokenizer is None:
        print("  Loading tokenizer...")
        token = load_hf_token()
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        train_data, eval_data = load_and_prepare_dataset(config, tokenizer)

    # Load model
    print("  Loading model in bf16 (full weights, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (100%)")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=int(config.get("warmup_ratio", 0.03) * (
            len(train_data) / config["per_device_train_batch_size"]
            / config["gradient_accumulation_steps"]
        ) * config["num_train_epochs"]),
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config.get("optim", "adamw_torch_fused"),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.get("logging_steps", 10),
        report_to=config.get("report_to", "tensorboard"),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 200),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    effective_bs = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    print(f"\n  Starting full fine-tuning...")
    print(f"     Effective batch size: {config['per_device_train_batch_size']} × {config['gradient_accumulation_steps']} = {effective_bs}")
    print(f"     Epochs: {config['num_train_epochs']}")
    print()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Save the full model
    final_dir = os.path.join(config["output_dir"], "final")
    print(f"\n  Saving full model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n  Training complete!")
    print(f"     Duration:       {hours}h {minutes}m {seconds}s")
    print(f"     Model saved to: {final_dir}")
    print(f"     TensorBoard:    tensorboard --logdir {config['output_dir']}")

    return True


def run_inference(prompt: str = None):
    """Step 5: Test the fine-tuned model."""
    print_header("Step 5: Inference")

    from inference import load_model, generate

    config = load_config()
    model_path = os.path.join(config["output_dir"], "final")

    if not Path(model_path).exists():
        print(f"  Fine-tuned model not found at: {model_path}")
        print(f"  Run training first (option 4).")
        return False

    model, tokenizer = load_model(model_path)

    if prompt:
        response = generate(model, tokenizer, prompt)
        print(f"\n Prompt:   {prompt}")
        print(f"   Response: {response}")
    else:
        # Interactive mode
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
    """Step 6: Compare original vs fine-tuned model."""
    print_header("Step 6: Model Comparison")

    from compare import compare

    config = load_config()
    original_path = config["model_name"]
    finetuned_path = os.path.join(config["output_dir"], "final")

    if not Path(finetuned_path).exists():
        print(f"  Fine-tuned model not found at: {finetuned_path}")
        print(f"  Run training first (option 4).")
        return False

    compare(original_path, finetuned_path)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline():
    """Run the complete pipeline: token check → VRAM → data → train → compare."""
    print_header("Running Full Pipeline")

    # Step 1: Token check
    print("\n  Step 1/6: Verifying HuggingFace token...")
    if not run_token_check():
        print("\n  Pipeline aborted: token verification failed.")
        return

    if not confirm_proceed("Token verified. Continue to VRAM check?"):
        return

    # Step 2: VRAM check
    print("\n  Step 2/6: Checking VRAM requirements...")
    if not run_vram_check():
        if not confirm_proceed("VRAM check flagged issues. Continue anyway?"):
            return

    if not confirm_proceed("Pre-flight checks passed. Continue to data preparation?"):
        return

    # Step 3: Data preparation
    print("\n  Step 3/6: Preparing dataset...")
    train_data, eval_data, tokenizer = run_data_preparation()
    if train_data is None:
        print("\n  Pipeline aborted: data preparation failed.")
        return

    # Step 4: Training (includes its own confirmation prompt)
    print("\n  Step 4/6: Training...")
    training_success = run_training(train_data, eval_data, tokenizer)
    if not training_success:
        print("\n  Pipeline stopped: training was skipped or failed.")
        return

    # Step 5: Quick inference test
    print("\n  Step 5/6: Quick inference test...")
    run_inference(prompt="What is machine learning? Explain in 2 sentences.")

    # Step 6: Comparison
    if confirm_proceed("Run full comparison (loads both models — needs extra VRAM)?"):
        print("\n  Step 6/6: Comparing models...")
        run_compare()

    print_header("Pipeline Complete!")
    print("   All steps finished successfully.")
    print(f"  Your fine-tuned model is ready for use.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Menu
# ──────────────────────────────────────────────────────────────────────────────

def interactive_menu():
    """Display an interactive menu for selecting pipeline steps."""
    print(BANNER)

    menu_options = {
        "1": ("Verify HuggingFace Token",     run_token_check),
        "2": ("Check VRAM Requirements",       run_vram_check),
        "3": ("Prepare Dataset",               run_data_preparation),
        "4": ("Start Training",                run_training),
        "5": ("Test Model (Inference)",        run_inference),
        "6": ("Compare Original vs Fine-Tuned", run_compare),
        "7": ("Run Full Pipeline (1 → 6)",     run_full_pipeline),
        "0": ("Exit",                          None),
    }

    while True:
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         Select an Operation             │")
        print("  ├─────────────────────────────────────────┤")
        for key, (label, _) in menu_options.items():
            prefix = " │"
            if key == "7":
                print("  ├──────────────────────────────────────────┤")
            print(f"{prefix}   [{key}]  {label:<35s}│")
        print("  └─────────────────────────────────────────┘")

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
            print("  Invalid choice. Please enter a number 0-7.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Master controller for the LLM full fine-tuning pipeline."
    )
    parser.add_argument(
        "--run",
        choices=["all", "token", "vram", "prepare", "train", "inference", "compare"],
        default=None,
        help="Run a specific step directly (skips interactive menu).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for inference mode (used with --run inference).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Auto-confirm all interactive prompts (for non-interactive/Streamlit usage).",
    )

    args = parser.parse_args()

    # Enable auto-confirm for non-interactive usage (e.g., Streamlit)
    if args.yes:
        global AUTO_CONFIRM
        AUTO_CONFIRM = True

    if args.run is None:
        # No argument — show interactive menu
        interactive_menu()
    else:
        # Direct execution of a specific step
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
# main.py — Master Controller for Full Fine-Tuning Pipeline
#
# Orchestrates the entire fine-tuning workflow from a single entry point.
#
# Usage:
#     python main.py                  # Interactive menu
#     python main.py --run all        # Run full pipeline (with confirmation)
#     python main.py --run prepare    # Run only data preparation
#     python main.py --run train      # Run only training
#     python main.py --run inference  # Run only inference
#     python main.py --run compare    # Run only comparison
#     python main.py --run vram       # Run only VRAM check
#     python main.py --run token      # Run only token verification
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
# BANNER = r"""
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║          LLM Full Fine-Tuning Pipeline — Master Controller           ║
# ║                                                                      ║
# ║   Modules:                                                           ║
# ║     1. Token Verification    — Validate HuggingFace credentials      ║
# ║     2. VRAM Check            — Estimate GPU memory requirements      ║
# ║     3. Data Preparation      — Download, format & tokenize dataset   ║
# ║     4. Training              — Full fine-tuning (all parameters)     ║
# ║     5. Inference             — Test your fine-tuned model            ║
# ║     6. Compare               — Side-by-side: original vs fine-tuned  ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
# """
#
# # ╔══════════════════════════════════════════════════════════════════════════╗
# # ║             WARNING: TRAINING IS A TIME-CONSUMING PROCESS!               ║
# # ║                                                                          ║
# # ║  Full fine-tuning of even a 1B parameter model on 52K examples           ║
# # ║  can take SEVERAL HOURS (3-8+ hours on an RTX 3090, longer on            ║
# # ║  slower GPUs). The pipeline involves:                                    ║
# # ║                                                                          ║
# # ║    • Downloading model weights (~2.5 GB)                                 ║
# # ║    • Downloading & processing the dataset (~52K examples)                ║
# # ║    • ~17,000+ optimizer steps across 3 epochs                            ║
# # ║    • Saving the full model (~2.5 GB) after training                      ║
# # ║                                                                          ║
# # ║  Make sure you have:                                                     ║
# # ║    ✓  Stable power supply / UPS                                          ║
# # ║    ✓  Sufficient disk space (~10 GB free)                                ║
# # ║    ✓  A valid HuggingFace token with model access                        ║
# # ║    ✓  Adequate GPU VRAM (run VRAM check first!)                          ║
# # ║                                                                          ║
# # ║  You will be asked to confirm before training begins.                    ║
# # ╚══════════════════════════════════════════════════════════════════════════╝
#
# TRAINING_WARNING = """
# \033[93m\033[1m
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║            WARNING: TRAINING IS EXTREMELY TIME-CONSUMING!                  ║
# ║                                                                            ║
# ║   Full fine-tuning will take SEVERAL HOURS to complete.                    ║
# ║                                                                            ║
# ║   Estimated time:                                                          ║
# ║     • RTX 3090 (24 GB):    ~3-6 hours                                      ║
# ║     • RTX 4090 (24 GB):    ~2-4 hours                                      ║
# ║     • A100 (40/80 GB):     ~1-2 hours                                      ║
# ║     • CPU only:            Days (not recommended)                          ║
# ║                                                                            ║
# ║   The process involves ~17,000+ optimizer steps across 3 epochs            ║
# ║   over 52K training examples. It CANNOT be paused and resumed              ║
# ║   (checkpoints are saved, but restarting picks up from the last one).      ║
# ║                                                                            ║
# ║   Do NOT close the terminal or shut down your machine during training.     ║
# ║                                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# \033[0m"""
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Utility
# # ──────────────────────────────────────────────────────────────────────────────
#
# def print_header(title: str):
#     """Print a formatted section header."""
#     width = 60
#     print()
#     print("=" * width)
#     print(f"  {title}")
#     print("=" * width)
#     print()
#
#
# def confirm_proceed(message: str = "Do you want to proceed?") -> bool:
#     """Ask the user for yes/no confirmation. Returns True if yes."""
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
#     """Load the training config YAML file."""
#     import yaml
#
#     config_path = Path(__file__).parent / "configs" / "training_config.yaml"
#
#     # Also try parent directory (if scripts/ is one level deep)
#     if not config_path.exists():
#         config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
#
#     if not config_path.exists():
#         print(f"  Config not found. Searched:")
#         print(f"     {Path(__file__).parent / 'configs' / 'training_config.yaml'}")
#         print(f"     {Path(__file__).parent.parent / 'configs' / 'training_config.yaml'}")
#         print(f"  Using default configuration values.")
#         return {
#             "model_name": "unsloth/Llama-3.2-1B-Instruct",
#             "dataset_name": "yahma/alpaca-cleaned",
#             "max_seq_length": 512,
#             "per_device_train_batch_size": 1,
#             "per_device_eval_batch_size": 2,
#             "gradient_accumulation_steps": 8,
#             "num_train_epochs": 3,
#             "learning_rate": 2e-5,
#             "weight_decay": 0.01,
#             "warmup_ratio": 0.03,
#             "lr_scheduler_type": "cosine",
#             "bf16": True,
#             "gradient_checkpointing": True,
#             "output_dir": "./outputs/llama-3.2-1B-full-ft",
#             "logging_steps": 10,
#             "eval_strategy": "steps",
#             "eval_steps": 200,
#             "save_strategy": "steps",
#             "save_steps": 500,
#             "save_total_limit": 2,
#             "seed": 42,
#             "save_dataset_locally": False,
#         }
#
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
#
#     print(f"  Config loaded from: {config_path}")
#     return config
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Pipeline Steps
# # ──────────────────────────────────────────────────────────────────────────────
#
# def run_token_check():
#     """Step 1: Verify HuggingFace token and model access."""
#     print_header("Step 1: HuggingFace Token Verification")
#
#     from HF_Token_Verificaiton import (
#         check_system_env,
#         check_keys_env,
#         check_conflict,
#         check_token_valid,
#         check_model_access,
#     )
#
#     config = load_config()
#     model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
#
#     sys_token = check_system_env()
#     env_path, env_token = check_keys_env()
#     check_conflict(sys_token, env_token)
#
#     active_token = sys_token or env_token
#
#     if active_token:
#         user_info = check_token_valid(active_token)
#         if user_info:
#             check_model_access(active_token, model_name)
#             print("\n  Token verification passed!")
#             return True
#         else:
#             print("\n  Token is invalid.")
#             return False
#     else:
#         print("\n  No token found. Create a Keys.env file with: HF_TOKEN=hf_your_token_here")
#         return False
#
#
# def run_vram_check():
#     """Step 2: Estimate VRAM requirements."""
#     print_header("Step 2: VRAM Estimation")
#
#     from check_vram import estimate_vram
#     import torch
#
#     config = load_config()
#
#     # Known param counts
#     param_counts = {
#         "unsloth/Llama-3.2-1B-Instruct": 1_240_000_000,
#         "meta-llama/Llama-3.2-1B-Instruct": 1_240_000_000,
#         "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1_100_000_000,
#         "HuggingFaceTB/SmolLM2-360M-Instruct": 360_000_000,
#         "openai-community/gpt2": 124_000_000,
#     }
#
#     model_name = config["model_name"]
#     num_params = param_counts.get(model_name)
#
#     if num_params is None:
#         print(f"  ⚠️  Unknown model '{model_name}'. Cannot estimate VRAM.")
#         print(f"     Add its parameter count to check_vram.py or main.py.")
#         return False
#
#     est = estimate_vram(
#         num_params=num_params,
#         batch_size=config["per_device_train_batch_size"],
#         seq_len=config["max_seq_length"],
#         bf16=config.get("bf16", True),
#         grad_checkpoint=config.get("gradient_checkpointing", True),
#     )
#
#     print(f"  Model:             {model_name}")
#     print(f"  Parameters:        {num_params / 1e9:.2f}B")
#     print(f"  Model Weights:     {est['model_weights_gb']:.1f} GB")
#     print(f"  Gradients:         {est['gradients_gb']:.1f} GB")
#     print(f"  Optimizer (AdamW): {est['optimizer_states_gb']:.1f} GB")
#     print(f"  Activations:       {est['activations_gb']:.1f} GB (estimated)")
#     print(f"  {'─' * 40}")
#     print(f"  TOTAL ESTIMATED:   {est['total_estimated_gb']:.1f} GB")
#
#     if torch.cuda.is_available():
#         gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
#         gpu_name = torch.cuda.get_device_name(0)
#         print(f"\n  Your GPU: {gpu_name} ({gpu_mem:.1f} GB)")
#
#         if est['total_estimated_gb'] > gpu_mem:
#             print(f"  Won't fit. Use a smaller model or reduce seq_length.")
#             return False
#         elif est['total_estimated_gb'] > gpu_mem * 0.90:
#             print(f"  Tight! May OOM under certain conditions.")
#             return True
#         else:
#             print(f"  Should fit with headroom.")
#             return True
#     else:
#         print(f"\n  No GPU detected. Training on CPU will be extremely slow.")
#         return True
#
#
# def run_data_preparation():
#     """Step 3: Download, format, and tokenize the dataset."""
#     print_header("Step 3: Data Preparation")
#
#     from prepare_data import load_hf_token, validate_token_and_model, load_and_prepare_dataset
#     from transformers import AutoTokenizer
#
#     config = load_config()
#
#     # Authenticate
#     print("  Authenticating with HuggingFace...")
#     token = load_hf_token()
#     if not token:
#         print("\n  Cannot proceed without a valid HF token.")
#         return None, None, None
#
#     # Validate access
#     if not validate_token_and_model(token, config["model_name"]):
#         print("\n  Cannot access model. Fix the issues above and retry.")
#         return None, None, None
#
#     # Load tokenizer
#     print(f"\n  Loading tokenizer: {config['model_name']}")
#     tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     # Prepare dataset
#     train_data, eval_data = load_and_prepare_dataset(config, tokenizer)
#
#     print(f"\n  Data preparation complete!")
#     print(f"    Train examples: {len(train_data):,}")
#     print(f"    Eval examples:  {len(eval_data):,}")
#
#     return train_data, eval_data, tokenizer
#
#
# def run_training(train_data=None, eval_data=None, tokenizer=None):
#     """
#     Step 4: Full fine-tuning.
#
#     ╔════════════════════════════════════════════════════════════════════╗
#     ║    ️THIS STEP IS EXTREMELY TIME-CONSUMING!                           ║
#     ║                                                                    ║
#     ║  Expect 3-8+ hours depending on your GPU.                          ║
#     ║  ~17,000 optimizer steps across 3 epochs on 52K examples.          ║
#     ║                                                                    ║
#     ║  The user MUST confirm before this step proceeds.                  ║
#     ╚════════════════════════════════════════════════════════════════════╝
#     """
#     print_header("Step 4: Full Fine-Tuning")
#
#     # ══════════════════════════════════════════════════════════════════════
#     #   DISPLAY PROMINENT WARNING TO THE USER
#     # ══════════════════════════════════════════════════════════════════════
#     print(TRAINING_WARNING)
#
#     if not confirm_proceed(
#         "Training may take SEVERAL HOURS. Do you want to proceed?"
#     ):
#         print("\n  Training skipped by user.")
#         return False
#
#     print("\n User confirmed. Starting training pipeline...\n")
#
#     import torch
#     from transformers import (
#         AutoModelForCausalLM,
#         AutoTokenizer,
#         TrainingArguments,
#         Trainer,
#         DataCollatorForLanguageModeling,
#     )
#     from prepare_data import load_hf_token, load_and_prepare_dataset
#
#     config = load_config()
#
#     # If data wasn't passed in, prepare it now
#     if train_data is None or eval_data is None or tokenizer is None:
#         print("  Loading tokenizer...")
#         token = load_hf_token()
#         tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#         train_data, eval_data = load_and_prepare_dataset(config, tokenizer)
#
#     # Load model
#     print("  Loading model in bf16 (full weights, no quantization)...")
#     model = AutoModelForCausalLM.from_pretrained(
#         config["model_name"],
#         dtype=torch.bfloat16,
#         device_map="auto",
#         attn_implementation="sdpa",
#     )
#
#     if config.get("gradient_checkpointing", True):
#         model.gradient_checkpointing_enable()
#         print("  ✓ Gradient checkpointing enabled")
#
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  Total parameters:     {total_params:,}")
#     print(f"  Trainable parameters: {trainable_params:,} (100%)")
#
#     # Data collator
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=config["output_dir"],
#         num_train_epochs=config["num_train_epochs"],
#         per_device_train_batch_size=config["per_device_train_batch_size"],
#         per_device_eval_batch_size=config["per_device_eval_batch_size"],
#         gradient_accumulation_steps=config["gradient_accumulation_steps"],
#         learning_rate=config["learning_rate"],
#         weight_decay=config["weight_decay"],
#         warmup_steps=int(config.get("warmup_ratio", 0.03) * (
#             len(train_data) / config["per_device_train_batch_size"]
#             / config["gradient_accumulation_steps"]
#         ) * config["num_train_epochs"]),
#         lr_scheduler_type=config["lr_scheduler_type"],
#         optim=config.get("optim", "adamw_torch_fused"),
#         bf16=config.get("bf16", True),
#         gradient_checkpointing=config.get("gradient_checkpointing", True),
#         gradient_checkpointing_kwargs={"use_reentrant": False},
#         logging_steps=config.get("logging_steps", 10),
#         report_to=config.get("report_to", "tensorboard"),
#         eval_strategy=config.get("eval_strategy", "steps"),
#         eval_steps=config.get("eval_steps", 200),
#         save_strategy=config.get("save_strategy", "steps"),
#         save_steps=config.get("save_steps", 500),
#         save_total_limit=config.get("save_total_limit", 2),
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         seed=config.get("seed", 42),
#         dataloader_num_workers=config.get("dataloader_num_workers", 2),
#         remove_unused_columns=False,
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=eval_data,
#         data_collator=data_collator,
#         processing_class=tokenizer,
#     )
#
#     effective_bs = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
#     print(f"\n  Starting full fine-tuning...")
#     print(f"     Effective batch size: {config['per_device_train_batch_size']} × {config['gradient_accumulation_steps']} = {effective_bs}")
#     print(f"     Epochs: {config['num_train_epochs']}")
#     print()
#
#     start_time = time.time()
#     trainer.train()
#     elapsed = time.time() - start_time
#
#     hours, remainder = divmod(int(elapsed), 3600)
#     minutes, seconds = divmod(remainder, 60)
#
#     # Save the full model
#     final_dir = os.path.join(config["output_dir"], "final")
#     print(f"\n  Saving full model to {final_dir}")
#     trainer.save_model(final_dir)
#     tokenizer.save_pretrained(final_dir)
#
#     print(f"\n  Training complete!")
#     print(f"     Duration:       {hours}h {minutes}m {seconds}s")
#     print(f"     Model saved to: {final_dir}")
#     print(f"     TensorBoard:    tensorboard --logdir {config['output_dir']}")
#
#     return True
#
#
# def run_inference(prompt: str = None):
#     """Step 5: Test the fine-tuned model."""
#     print_header("Step 5: Inference")
#
#     from inference import load_model, generate
#
#     config = load_config()
#     model_path = os.path.join(config["output_dir"], "final")
#
#     if not Path(model_path).exists():
#         print(f"  Fine-tuned model not found at: {model_path}")
#         print(f"  Run training first (option 4).")
#         return False
#
#     model, tokenizer = load_model(model_path)
#
#     if prompt:
#         response = generate(model, tokenizer, prompt)
#         print(f"\n Prompt:   {prompt}")
#         print(f"   Response: {response}")
#     else:
#         # Interactive mode
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
#     """Step 6: Compare original vs fine-tuned model."""
#     print_header("Step 6: Model Comparison")
#
#     from compare import compare
#
#     config = load_config()
#     original_path = config["model_name"]
#     finetuned_path = os.path.join(config["output_dir"], "final")
#
#     if not Path(finetuned_path).exists():
#         print(f"  Fine-tuned model not found at: {finetuned_path}")
#         print(f"  Run training first (option 4).")
#         return False
#
#     compare(original_path, finetuned_path)
#     return True
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Full Pipeline
# # ──────────────────────────────────────────────────────────────────────────────
#
# def run_full_pipeline():
#     """Run the complete pipeline: token check → VRAM → data → train → compare."""
#     print_header("Running Full Pipeline")
#
#     # Step 1: Token check
#     print("\n  Step 1/6: Verifying HuggingFace token...")
#     if not run_token_check():
#         print("\n  Pipeline aborted: token verification failed.")
#         return
#
#     if not confirm_proceed("Token verified. Continue to VRAM check?"):
#         return
#
#     # Step 2: VRAM check
#     print("\n  Step 2/6: Checking VRAM requirements...")
#     if not run_vram_check():
#         if not confirm_proceed("VRAM check flagged issues. Continue anyway?"):
#             return
#
#     if not confirm_proceed("Pre-flight checks passed. Continue to data preparation?"):
#         return
#
#     # Step 3: Data preparation
#     print("\n  Step 3/6: Preparing dataset...")
#     train_data, eval_data, tokenizer = run_data_preparation()
#     if train_data is None:
#         print("\n  Pipeline aborted: data preparation failed.")
#         return
#
#     # Step 4: Training (includes its own confirmation prompt)
#     print("\n  Step 4/6: Training...")
#     training_success = run_training(train_data, eval_data, tokenizer)
#     if not training_success:
#         print("\n  Pipeline stopped: training was skipped or failed.")
#         return
#
#     # Step 5: Quick inference test
#     print("\n  Step 5/6: Quick inference test...")
#     run_inference(prompt="What is machine learning? Explain in 2 sentences.")
#
#     # Step 6: Comparison
#     if confirm_proceed("Run full comparison (loads both models — needs extra VRAM)?"):
#         print("\n  Step 6/6: Comparing models...")
#         run_compare()
#
#     print_header("Pipeline Complete!")
#     print("   All steps finished successfully.")
#     print(f"  Your fine-tuned model is ready for use.\n")
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # Interactive Menu
# # ──────────────────────────────────────────────────────────────────────────────
#
# def interactive_menu():
#     """Display an interactive menu for selecting pipeline steps."""
#     print(BANNER)
#
#     menu_options = {
#         "1": ("Verify HuggingFace Token",     run_token_check),
#         "2": ("Check VRAM Requirements",       run_vram_check),
#         "3": ("Prepare Dataset",               run_data_preparation),
#         "4": ("Start Training",                run_training),
#         "5": ("Test Model (Inference)",        run_inference),
#         "6": ("Compare Original vs Fine-Tuned", run_compare),
#         "7": ("Run Full Pipeline (1 → 6)",     run_full_pipeline),
#         "0": ("Exit",                          None),
#     }
#
#     while True:
#         print("\n  ┌─────────────────────────────────────────┐")
#         print("  │         Select an Operation             │")
#         print("  ├─────────────────────────────────────────┤")
#         for key, (label, _) in menu_options.items():
#             prefix = " │"
#             if key == "7":
#                 print("  ├──────────────────────────────────────────┤")
#             print(f"{prefix}   [{key}]  {label:<35s}│")
#         print("  └─────────────────────────────────────────┘")
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
#             print("  Invalid choice. Please enter a number 0-7.")
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# # CLI Entry Point
# # ──────────────────────────────────────────────────────────────────────────────
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Master controller for the LLM full fine-tuning pipeline."
#     )
#     parser.add_argument(
#         "--run",
#         choices=["all", "token", "vram", "prepare", "train", "inference", "compare"],
#         default=None,
#         help="Run a specific step directly (skips interactive menu).",
#     )
#     parser.add_argument(
#         "--prompt",
#         type=str,
#         default=None,
#         help="Prompt for inference mode (used with --run inference).",
#     )
#
#     args = parser.parse_args()
#
#     if args.run is None:
#         # No argument — show interactive menu
#         interactive_menu()
#     else:
#         # Direct execution of a specific step
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