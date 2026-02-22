"""
additive_train.py — Training loop for Bottleneck Adapters and (IA)³.

╔══════════════════════════════════════════════════════════════════════════╗
║  THE KEY DIFFERENCE FROM LoRA TRAINING                                   ║
║                                                                          ║
║  LoRA uses get_peft_model(model, LoraConfig(...))                        ║
║    → Freezes all base params                                             ║
║    → Injects A and B matrices in PARALLEL to existing weight matrices    ║
║    → New params live inside modified Linear layers (side by side)        ║
║                                                                          ║
║  Bottleneck Adapters use get_peft_model(model, AdapterConfig(...))       ║
║    → Freezes all base params                                             ║
║    → Inserts NEW BotleneckAdapterLayer modules SEQUENTIALLY              ║
║      between existing transformer sub-layers                             ║
║    → Model architecture grows: each layer now has an extra module        ║
║    → These modules stay at inference time — cannot be removed            ║
║                                                                          ║
║  (IA)³ uses get_peft_model(model, IA3Config(...))                        ║
║    → Freezes all base params                                             ║
║    → Injects tiny l_k, l_v, l_ff scaling vectors                         ║
║    → Vectors are applied element-wise to existing activations            ║
║    → Can be merged (linear) or kept separate                             ║
║                                                                          ║
║  HuggingFace PEFT supports:                                              ║
║    - LoRA, DoRA, (IA)³, Prompt Tuning, Prefix Tuning, etc.               ║
║    - Does NOT have Bottleneck Adapters                                   ║
║                                                                          ║
║  AdapterHub `adapters` library supports:                                 ║
║    - Bottleneck Adapters (Houlsby, Pfeiffer, Parallel variants)          ║
║    - Has its own API separate from PEFT                                  ║
║    - pip install adapters                                                ║
║                                                                          ║
║  So in this file:                                                        ║
║    adapter_method = "ia3"        -> uses peft library                    ║
║    adapter_method = "bottleneck" -> uses adapters library                ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python additive_train.py
"""

import yaml
import torch
import sys
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# -- (IA)3 -- from HuggingFace PEFT -------------------------------------------
from peft import (
    get_peft_model,
    TaskType,
    IA3Config,
    PeftModel,
)

# -- Bottleneck Adapters -- from AdapterHub `adapters` library ----------------
# If not installed: pip install adapters
try:
    import adapters
    from adapters import SeqBnConfig, DoubleSeqBnConfig
    # SeqBnConfig       = Pfeiffer style (1 adapter per layer, after FFN only)
    # DoubleSeqBnConfig = Houlsby style  (2 adapters per layer, after attn AND FFN)
    #
    # We use adapters.init(model) on a standard AutoModelForCausalLM rather than
    # AutoAdapterModel. This preserves the original LM head and loss computation
    # exactly, and just adds adapter support on top of the existing model.
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

from additive_prepare_data import prepare_datasets, get_data_collator


# =============================================================================
# (IA)3 -- HuggingFace PEFT
# =============================================================================

def build_ia3_config(config: dict) -> IA3Config:
    """
    Build IA3Config for HuggingFace PEFT.

    target_modules: which projections get a learned scaling vector
      k_proj   -> l_k scales key projections    K = (l_k o W_k) . h
      v_proj   -> l_v scales value projections  V = (l_v o W_v) . h
      down_proj -> l_ff scales FFN gate output

    feedforward_modules: which of the above targets are in the FFN
      PEFT uses this to distinguish attention vs FFN rescaling

    Initialization: all l vectors = ones -> identity at step 0
    """
    target_modules = config.get("ia3_target_modules", ["k_proj", "v_proj", "down_proj"])
    ff_modules = config.get("ia3_feedforward_modules", ["down_proj"])

    return IA3Config(
        target_modules=target_modules,
        feedforward_modules=ff_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_ia3(config: dict):
    """
    Load base model and apply (IA)3 via HuggingFace PEFT.

    get_peft_model():
      1. Freezes ALL base model parameters (requires_grad = False)
      2. Adds tiny l_k, l_v, l_ff scaling vectors alongside projections
      3. Returns PeftModel -- forward pass applies l o activations at each target
    """
    model_name = config["model_name"]

    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    print("  Applying (IA)3 configuration via HuggingFace PEFT...")
    peft_config = build_ia3_config(config)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\n  Parameter breakdown:")
    print(f"    Total:     {total:>12,}")
    print(f"    Frozen:    {frozen:>12,}  ({frozen/total*100:.2f}%)")
    print(f"    Trainable: {trainable:>12,}  ({trainable/total*100:.4f}%)")
    print(f"\n  Architecture note:")
    print(f"  (IA)3 scaling vectors added alongside existing projections.")
    print(f"  After training, these CAN be merged -- linear operation.")
    print(f"  Use merge_and_unload() to fold l vectors into weight matrices.")

    return model


# =============================================================================
# Bottleneck Adapters -- AdapterHub `adapters` library
# =============================================================================

def load_model_bottleneck(config: dict):
    """
    Load base model and apply Bottleneck Adapters via the `adapters` library.

    We use adapters.init(model) on a standard AutoModelForCausalLM rather than
    AutoAdapterModel. This is the recommended pattern when you want to keep the
    original model head and loss computation intact:

      AutoModelForCausalLM.from_pretrained()   <- standard model, full LM head
      adapters.init(model)                     <- bolt adapter support on top
      model.add_adapter("task_adapter", ...)   <- insert bottleneck modules
      model.set_active_adapters("task_adapter")<- activate for forward pass
      model.train_adapter("task_adapter")      <- freeze base, unfreeze adapters

    Config classes (current adapters library API):
      SeqBnConfig       = Pfeiffer (2020): 1 adapter/layer, after FFN only
      DoubleSeqBnConfig = Houlsby (2019):  2 adapters/layer, after attn AND FFN

    Both take:
      reduction_factor: hidden_dim / bottleneck_dim
                        e.g. hidden=2048, bottleneck=64 -> reduction_factor=32
      non_linearity:    "gelu", "relu", "swish"

    Forward pass (SeqBnConfig / Pfeiffer):
      h -> [frozen attn] -> [frozen FFN] -> [W_down -> GELU -> W_up] + h

    Initialization:
      W_up = zeros -> output = h + 0 = h at step 0 (transparent / identity)
      W_down = random Gaussian
    """
    if not ADAPTERS_AVAILABLE:
        print("\n  ERROR: Bottleneck Adapters require the 'adapters' library.")
        print("  Install with:  pip install adapters")
        print("  Then re-run this script.")
        sys.exit(1)

    model_name     = config["model_name"]
    bottleneck_dim = config.get("bottleneck_dim", 64)
    placement      = config.get("adapter_placement", "after_ffn")
    non_linearity  = config.get("adapter_non_linearity", "gelu")

    # ── Step 1: Load as a standard AutoModelForCausalLM ──────────────────────
    # This gives us the full LM head and cross-entropy loss for free.
    # AutoAdapterModel was avoided because it strips the head and requires
    # add_causal_lm_head(), which causes loss computation issues with the Trainer.
    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    # ── Step 2: Add adapter support to the existing model ────────────────────
    # adapters.init() patches the model in-place to support:
    #   .add_adapter(), .train_adapter(), .set_active_adapters(), .save_adapter()
    # The model class, LM head, and loss computation are unchanged.
    adapters.init(model)
    print("  Adapter support initialised on AutoModelForCausalLM")

    # ── Step 3: Compute reduction_factor ─────────────────────────────────────
    # The adapters library uses reduction_factor, not absolute bottleneck_dim.
    hidden_size       = model.config.hidden_size
    reduction_factor  = max(1, hidden_size // bottleneck_dim)
    actual_bottleneck = hidden_size // reduction_factor
    print(f"  hidden_size={hidden_size}, requested bottleneck_dim={bottleneck_dim}")
    print(f"  reduction_factor={reduction_factor} -> actual bottleneck={actual_bottleneck}")

    # ── Step 4: Choose config class and insert adapter modules ────────────────
    use_houlsby = (placement == "after_attn_and_ffn")

    if use_houlsby:
        print("  Placement: Houlsby (DoubleSeqBnConfig) -- 2 adapters per layer")
        adapter_config = DoubleSeqBnConfig(
            reduction_factor=reduction_factor,
            non_linearity=non_linearity,
        )
    else:
        print("  Placement: Pfeiffer (SeqBnConfig) -- 1 adapter per layer (after FFN)")
        adapter_config = SeqBnConfig(
            reduction_factor=reduction_factor,
            non_linearity=non_linearity,
        )

    model.add_adapter("task_adapter", config=adapter_config)

    # ── Step 5: Activate adapter for the forward pass ─────────────────────────
    # Without this the adapter modules exist in the model but are bypassed.
    # The warning "adapters available but none are activated" means this was missing.
    model.set_active_adapters("task_adapter")

    # ── Step 6: Freeze base, unfreeze adapter only ────────────────────────────
    # train_adapter() sets requires_grad=False on all base params and
    # requires_grad=True on the named adapter's W_down, W_up, biases.
    model.train_adapter("task_adapter")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    style     = "Houlsby (2/layer)" if use_houlsby else "Pfeiffer (1/layer)"
    print(f"\n  Parameter breakdown ({style}):")
    print(f"    Total:     {total:>12,}")
    print(f"    Frozen:    {frozen:>12,}  ({frozen/total*100:.2f}%)")
    print(f"    Trainable: {trainable:>12,}  ({trainable/total*100:.4f}%)")
    print(f"\n  Architecture note:")
    print(f"  Bottleneck adapter modules inserted INTO the transformer layers.")
    print(f"  Forward path: frozen_layer -> [W_down -> GELU -> W_up + residual]")
    print(f"  These modules stay at inference -- cannot merge (non-linear GELU).")

    return model



# =============================================================================
# Unified entry point
# =============================================================================

def load_model_and_apply_peft(config: dict):
    method = config.get("adapter_method", "ia3")
    if method == "ia3":
        return load_model_ia3(config)
    elif method == "bottleneck":
        return load_model_bottleneck(config)
    else:
        raise ValueError(f"Unknown adapter_method: '{method}'. Choose 'ia3' or 'bottleneck'.")


# =============================================================================
# Save helpers
# =============================================================================

def save_ia3(model, tokenizer, output_dir: str, merge: bool):
    """Save an (IA)3 adapter -- optionally merged into base weights."""
    if merge:
        print("\n  Merging (IA)3 l vectors into base weights before saving...")
        # merge_and_unload() computes:
        #   W_k_merged[i, :] = l_k[i] * W_k[i, :]  for all i
        # Result is a standard AutoModelForCausalLM -- no PEFT overhead at inference
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(f"{output_dir}/merged")
        tokenizer.save_pretrained(f"{output_dir}/merged")
        print(f"  Merged model saved to: {output_dir}/merged")
        print(f"  Zero inference overhead -- l vectors baked into weights.")
    else:
        # Save adapter weights only (~0.5 MB for 1B model)
        # Saves: adapter_model.safetensors + adapter_config.json
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  (IA)3 adapter saved to: {output_dir}")


def save_bottleneck(model, tokenizer, output_dir: str):
    """
    Save a Bottleneck Adapter using the adapters library format.

    save_adapter() saves only the adapter weights (W_down, W_up per layer).
    Saved to: {output_dir}/task_adapter/
      adapter_config.json  -- reduction_factor, non_linearity, placement
      pytorch_model.bin    -- W_down, b_down, W_up, b_up for all layers
    """
    adapter_save_path = str(Path(output_dir) / "task_adapter")
    model.save_adapter(adapter_save_path, "task_adapter")
    tokenizer.save_pretrained(output_dir)
    print(f"  Bottleneck adapter saved to: {adapter_save_path}")
    print(f"  Load with: model.load_adapter('{adapter_save_path}')")


# =============================================================================
# Main training entry point
# =============================================================================

def train(config_path: str = None):
    """
    Full training pipeline for additive PEFT (Bottleneck or (IA)3).

    Training loop is the same for both methods -- the Trainer only updates
    parameters where requires_grad=True, which is exclusively the adapter weights.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "additive_training_config.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    method = config.get("adapter_method", "ia3")
    output_dir = config.get("output_dir", f"./outputs/llama-additive-{method}")

    print("=" * 60)
    print(f"  Additive PEFT Training  --  method: {method.upper()}")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1/3] Preparing data...")
    train_dataset, eval_dataset, tokenizer = prepare_datasets(config_path)

    # 2. Load model + apply adapter
    print(f"\n[2/3] Loading model and applying {method.upper()}...")
    model = load_model_and_apply_peft(config)

    # 3. Train
    print("\n[3/3] Training...")

    collator = get_data_collator(tokenizer, model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1e-4),
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    print("\n  Training complete.")

    # 4. Save
    merge_before_save = config.get("merge_before_save", False)

    if method == "ia3":
        save_ia3(model, tokenizer, output_dir, merge=merge_before_save)
    elif method == "bottleneck":
        if merge_before_save:
            print("\n  WARNING: Bottleneck Adapters cannot be merged (GELU is non-linear).")
            print("  Saving as adapter-only format instead.")
        save_bottleneck(model, tokenizer, output_dir)

    print(f"\n  Done. Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()
