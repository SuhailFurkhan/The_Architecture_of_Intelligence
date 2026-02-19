"""
peft_train.py — LoRA / PEFT Fine-Tuning of LLaMA 3.2 1B

╔══════════════════════════════════════════════════════════════════════════╗
║  WHAT HAPPENS DIFFERENTLY HERE vs full fine-tuning (train.py)           ║
║                                                                          ║
║  Full fine-tuning (train.py):                                            ║
║    model = AutoModelForCausalLM.from_pretrained(...)                     ║
║    # ALL 1.24B parameters are trainable                                  ║
║    # AdamW tracks momentum/variance for ALL 1.24B params                 ║
║    # Saves entire 2.5 GB model                                           ║
║                                                                          ║
║  LoRA (this file):                                                        ║
║    model = AutoModelForCausalLM.from_pretrained(...)                     ║
║    # ① Freeze ALL base model params (requires_grad=False)                ║
║    model = get_peft_model(model, lora_config)                            ║
║    # ② Inject A and B matrices into target layers                        ║
║    # ③ Only A and B matrices are trainable (0.8% of params)              ║
║    # AdamW only tracks momentum/variance for those tiny adapters         ║
║    # Saves only ~15 MB of adapter deltas                                 ║
║                                                                          ║
║  THE FORWARD PASS WITH LoRA:                                             ║
║                                                                          ║
║  Original linear layer:   output = W · x                                 ║
║    W: [d_out × d_in]  ← frozen, not updated                              ║
║                                                                          ║
║  LoRA-augmented layer:    output = W · x  +  (B · A) · x · (α/r)        ║
║    W: [d_out × d_in]  ← FROZEN                                           ║
║    A: [r × d_in]      ← TRAINABLE (initialized randomly)                 ║
║    B: [d_out × r]     ← TRAINABLE (initialized to ZERO)                  ║
║                                                                          ║
║    At init:  B = 0  →  B·A = 0  →  model starts IDENTICAL to base.      ║
║    Training: gradients update A and B to capture the task-specific delta.║
║                                                                          ║
║  Why B=0 at initialization?                                              ║
║    If A were also 0, no gradients would flow (dead start).               ║
║    If B were random, the model would immediately differ from base.       ║
║    B=0 + A=random means: delta = B·A = 0·(random) = 0 at step 0.        ║
║    This preserves the pretrained behaviour from the very first token.    ║
╚══════════════════════════════════════════════════════════════════════════╝

DATA FLOW COMPARISON:
═══════════════════════════════════════════════════════════════════════════

Full fine-tuning gradient flow:
  Token IDs → Embed → Layer 0 → ... → Layer 15 → LM Head → Loss
                 ↑         ↑                 ↑          ↑       ↓
              (grad)    (grad)             (grad)    (grad)  Backprop
  Every weight gets a gradient. AdamW updates 1.24B params.

LoRA gradient flow:
  Token IDs → Embed → Layer 0 → ... → Layer 15 → LM Head → Loss
               [frz]    [frz]            [frz]     [frz]       ↓
               BUT: LoRA(A,B) injected at q,k,v,o,gate,up,down Backprop
                    Gradients flow THROUGH frozen layers but only
                    ACCUMULATE at A and B matrices.
  Only A and B get updated. AdamW handles ~10M params.

Usage:
    python peft_train.py                       # uses peft_training_config.yaml
    from peft_train import train               # call from peft_main.py
"""

import os
import torch
import yaml
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

from peft_prepare_data import load_hf_token, load_and_prepare_dataset


def load_config(config_path: str = None) -> dict:
    """Load PEFT training config from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "peft_training_config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "configs" / "peft_training_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"  ✅ Config loaded: {config_path}")
    return config


def build_lora_config(config: dict) -> LoraConfig:
    """
    Build the PEFT LoraConfig from the training config dict.

    LoraConfig tells PEFT:
      - Which layers to inject adapters into  (target_modules)
      - The rank of the decomposition         (r)
      - The scaling factor                    (lora_alpha)
      - Regularization                        (lora_dropout)
      - Whether to train biases               (bias)
      - The task type (needed for correct     (task_type)
        handling of the model output format)

    What PEFT does with this:
      For each module name in target_modules, PEFT:
        1. Finds that nn.Linear layer inside the model
        2. Replaces it with a LoraLinear wrapper that contains:
              - The original weight W (frozen)
              - New matrix A [r × d_in] (trainable, random init)
              - New matrix B [d_out × r] (trainable, zero init)
        3. Modifies forward() to compute W·x + (B·A)·x·(alpha/r)
    """
    return LoraConfig(
        r=config.get("lora_r", 16),                        # Rank
        lora_alpha=config.get("lora_alpha", 32),           # Scale factor
        lora_dropout=config.get("lora_dropout", 0.05),     # Regularization
        bias=config.get("lora_bias", "none"),              # Bias training
        task_type=TaskType.CAUSAL_LM,                      # Causal language model
        target_modules=config.get("lora_target_modules",  # Which layers to adapt
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]
        ),
        # inference_mode=False — set automatically during training
    )


def print_trainable_parameters(model):
    """
    Print the number of trainable vs total parameters.

    This is a critical check! Before training, you should always verify that:
      - Trainable params ≈ 1-5% of total (typical for LoRA at r=16)
      - Non-trainable ≈ base model params
      - Trainable are ONLY the A and B matrices, NOT the base weights

    Example output for Llama-3.2-1B with r=16, all 7 target modules:
      Trainable params:   10,248,192  (0.82%)
      Non-trainable:   1,240,000,000
      Total:           1,250,248,192
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    non_trainable = all_params - trainable_params
    pct = 100 * trainable_params / all_params

    print(f"\n  {'─'*55}")
    print(f"  Model Parameter Summary:")
    print(f"  {'─'*55}")
    print(f"  Trainable parameters:   {trainable_params:>15,}  ({pct:.2f}%)")
    print(f"  Non-trainable (frozen): {non_trainable:>15,}  ({100-pct:.2f}%)")
    print(f"  Total parameters:       {all_params:>15,}")
    print(f"  {'─'*55}")
    print(f"  LoRA rank:    r={config_global.get('lora_r',16)}")
    print(f"  LoRA alpha:   α={config_global.get('lora_alpha',32)}  →  scale={config_global.get('lora_alpha',32)/config_global.get('lora_r',16):.2f}")
    print(f"  {'─'*55}\n")


# Global config reference used by print_trainable_parameters
config_global = {}


def train(train_dataset=None, eval_dataset=None, tokenizer=None, config: dict = None):
    """
    Main LoRA training function.

    Can be called:
      1. Standalone:  python peft_train.py
      2. From main:   train(train_data, eval_data, tokenizer, config)
    """
    global config_global

    # ── 0. Load config if not provided ────────────────────────────────────
    if config is None:
        config = load_config()
    config_global = config

    # ── 1. Authenticate ────────────────────────────────────────────────────
    token = load_hf_token()
    if not token:
        raise RuntimeError("No HuggingFace token found. Create Keys.env with HF_TOKEN=hf_xxx")

    # ── 2. Load tokenizer and dataset ─────────────────────────────────────
    if tokenizer is None:
        print(f"\n  📦 Loading tokenizer: {config['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            token=token,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if train_dataset is None or eval_dataset is None:
        print(f"\n  📊 Preparing dataset...")
        train_dataset, eval_dataset = load_and_prepare_dataset(
            config, tokenizer, mask_instruction=True
        )

    # ── 3. Load base model ─────────────────────────────────────────────────
    #
    # IMPORTANT: We load the SAME base model as full fine-tuning.
    # The difference is what happens NEXT (get_peft_model below).
    #
    # With LoRA, we can even load in 4-bit (QLoRA) to save more VRAM,
    # but here we use bf16 (same as full FT) for simplicity and speed.
    # 4-bit QLoRA would let you train an 8B model on a 3090!
    print(f"\n  📦 Loading base model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        token=token,
        torch_dtype=torch.bfloat16,    # bf16 weights = 2 bytes per param
        device_map="auto",             # puts model on GPU automatically
    )

    # Disable the use_cache optimisation — incompatible with gradient checkpointing.
    # use_cache=True caches K/V pairs during forward pass for fast inference.
    # But during training with grad checkpointing, we recompute activations,
    # so caching is not only useless but causes a warning. Disable it here.
    model.config.use_cache = False

    # ── 4. Apply LoRA — THE KEY STEP ──────────────────────────────────────
    #
    # get_peft_model() does the following:
    #   a) Freezes ALL existing parameters (requires_grad=False)
    #   b) Wraps each target module with a LoraLinear layer
    #   c) Adds A and B matrices to each wrapped module
    #   d) Sets requires_grad=True ONLY on A and B matrices
    #
    # After this call:
    #   model.print_trainable_parameters()  →  ~10M trainable (0.82%)
    #
    # The base model weights are still there — they're just frozen.
    # During forward pass, PEFT computes: W·x + (B·A·x) × (alpha/r)
    print(f"\n  🔧 Applying LoRA adapters...")
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # ── 5. Training arguments ──────────────────────────────────────────────
    output_dir = config.get("output_dir", "./outputs/llama-3.2-1B-lora")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),

        # BATCH SIZE NOTE:
        # With LoRA we can use larger batches (less VRAM).
        # batch_size=4 instead of 1 for full FT.
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),

        # LEARNING RATE NOTE:
        # LoRA can tolerate higher LR than full FT.
        # Base model is frozen so there's no risk of destroying pretrained weights.
        # 2e-4 is typical; some practitioners use up to 1e-3.
        learning_rate=config.get("learning_rate", 2e-4),

        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),

        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        optim=config.get("optim", "adamw_torch_fused"),

        logging_steps=config.get("logging_steps", 10),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 200),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 400),
        save_total_limit=config.get("save_total_limit", 2),

        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        report_to=config.get("report_to", "tensorboard"),

        # PEFT-specific: when saving checkpoints, only save the adapter,
        # not the full model (saves huge disk space).
        # Trainer auto-detects PEFT and saves adapters by default.
    )

    # ── 6. Data Collator ────────────────────────────────────────────────────
    #
    # DataCollatorForSeq2Seq is better for our masked labels:
    #   - Pads input_ids to the same length within a batch
    #   - Pads labels with -100 (so padded positions are excluded from loss)
    #   - This is important because we already have -100 for instruction tokens
    #
    # vs DataCollatorForLanguageModeling (used in full FT):
    #   - That collator assumes label = input_id everywhere (no masking support)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,               # Aligns tensor dims for efficiency
        label_pad_token_id=-100,            # Pad labels with -100 (excluded from loss)
    )

    # ── 7. Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ── 8. Train ────────────────────────────────────────────────────────────
    effective_batch = (config.get("per_device_train_batch_size", 4) *
                       config.get("gradient_accumulation_steps", 4))
    print(f"  🚀 Starting LoRA fine-tuning...")
    print(f"     Effective batch size: {config.get('per_device_train_batch_size',4)} × {config.get('gradient_accumulation_steps',4)} = {effective_batch}")
    print(f"     Trainable parameters: only A and B matrices (LoRA adapters)")
    print()

    trainer.train()

    # ── 9. Save adapter weights ─────────────────────────────────────────────
    #
    # TWO SAVING STRATEGIES — understand the trade-off:
    #
    # Strategy A: Save ADAPTER ONLY (default, recommended)
    # ────────────────────────────────────────────────────
    # model.save_pretrained(final_dir)  →  saves ~15 MB adapter_model.safetensors
    # tokenizer.save_pretrained(final_dir)
    #
    # Pros: tiny file, fast to save/share
    # Cons: loading requires PEFT + base model at inference time
    #
    #   Loading later:
    #     base = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    #     model = PeftModel.from_pretrained(base, "./final")
    #
    # Strategy B: MERGE and save full model
    # ──────────────────────────────────────
    # merged = model.merge_and_unload()  →  merges B·A into W, removes adapter overhead
    # merged.save_pretrained(final_dir)  →  saves full ~2.5 GB model
    #
    # Pros: standalone model, no PEFT dependency at inference
    # Cons: large file, loses ability to swap adapters
    #
    #   Loading later:
    #     model = AutoModelForCausalLM.from_pretrained("./final")  # just like regular model
    #
    # MERGE MATH:
    #   During inference, LoRA computes: output = (W + B·A·(α/r)) · x
    #   Merging pre-computes:  W_merged = W + B·A·(α/r)
    #   Then inference is just: output = W_merged · x  (no adapter overhead!)
    #   The merged model is IDENTICAL in output to the adapter model.

    final_dir = os.path.join(output_dir, "final")
    merge_before_save = config.get("merge_before_save", False)

    if merge_before_save:
        # Strategy B: merge adapters into base model
        print(f"\n  🔀 Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        print(f"  💾 Saving merged model to {final_dir} (~2.5 GB)")
        merged_model.save_pretrained(final_dir, safe_serialization=True)
    else:
        # Strategy A: save only adapter weights
        print(f"\n  💾 Saving LoRA adapter to {final_dir} (~15 MB)")
        model.save_pretrained(final_dir)

    tokenizer.save_pretrained(final_dir)

    print(f"\n  ✅ LoRA fine-tuning complete!")
    print(f"     Adapter saved to:  {final_dir}")
    print(f"     TensorBoard:       tensorboard --logdir {output_dir}")
    if not merge_before_save:
        print(f"\n  To load for inference:")
        print(f"     from peft import PeftModel")
        print(f"     from transformers import AutoModelForCausalLM")
        print(f"     base = AutoModelForCausalLM.from_pretrained('{config['model_name']}')")
        print(f"     model = PeftModel.from_pretrained(base, '{final_dir}')")
        print(f"     # Or use peft_inference.py which handles this automatically")


if __name__ == "__main__":
    config = load_config()
    train(config=config)
