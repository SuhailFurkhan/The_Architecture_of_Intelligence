"""
train.py — Full Fine-Tuning of LLaMA 3.2 1B on RTX 3090

╔═══════════════════════════════════════════════════════════════════╗
║  THIS IS FULL FINE-TUNING — NOT LoRA/QLoRA                        ║
║                                                                   ║
║  Key differences from your QLoRA notebooks:                       ║
║  • No BitsAndBytesConfig (no quantization)                        ║
║  • No LoraConfig / get_peft_model (no adapters)                   ║
║  • No prepare_model_for_kbit_training                             ║
║  • ALL model parameters are trainable                             ║
║  • Output is the ENTIRE model, not just adapter weights           ║
║                                                                   ║
║  Memory fits on 3090 via:                                         ║
║  • bf16 mixed precision                                           ║
║  • gradient checkpointing                                         ║
║  • batch_size=1 + gradient accumulation                           ║
╚═══════════════════════════════════════════════════════════════════╝


Step 1 — prepare_data.py (already ran, data saved to disk)
    This already happened before training. It converted raw text into token IDs:

        "Hello, how are you?" → [9906, 11, 1268, 527, 499, 30]

    These are just integer lookups — not yet embeddings. The output saved to disk is a dataset of integer sequences.

Step 2 — train.py (what happens each training step)
    This is where the actual model forward pass happens, and it goes through several stages:

    Token IDs → Embedding Layer — The model's first layer is an embedding table (a big matrix of shape [128256 vocab × 2048 hidden_dim]).
    Each token ID is used as an index to look up its corresponding vector. So token 9906 pulls out row 9906, which is a 2048-dimensional vector.
    This is where numbers become meaningful vectors.

    Embeddings → Transformer Layers (×16 layers) —
    Those vectors flow through the transformer stack: self-attention (tokens attend to each other), feed-forward networks (transform each vector), and layer norms.
    This is where the model "reasons" about the relationships between tokens.

    Final Layer → Prediction — The output of the last transformer layer goes through a linear head that projects back to vocabulary size [2048 → 128256].
    This produces a probability distribution over all possible next tokens at each position.

    Loss Computation — The labels (which are just input_ids shifted by one) tell the model what the correct next token should have been.
    Cross-entropy loss measures how wrong the prediction was.

    Backpropagation — The loss gradient flows backward through every layer, computing how much each of the 1.24 billion parameters contributed to the error.
    Then AdamW updates every parameter to reduce that error.


        So the full flow per training step is:
        Token IDs  →  Embedding Lookup  →  16 Transformer Layers  →  Next-Token Prediction
           [ints]        [vectors]             [attention + FFN]          [probabilities]
                                                                               ↓
                                                                            Loss
                                                                               ↓
                                                                      Backprop + Update
                                                                     (all 1.24B params)

The key thing is: tokenization (text → integers) is not the same as embedding (integers → vectors).

COMPLETE DATA FLOW — What happens during training:
═══════════════════════════════════════════════════

OFFLINE (prepare_data.py — already ran before this script):
  Raw text        →  Tokenizer (BPE)    →  Token IDs saved to disk
  "Hello, how"    →  [9906, 11, 1268]   →  stored as integers on disk

PER TRAINING STEP (this script — happens inside trainer.train()):

  ┌─────────────────────────────────────────────────────────────────┐
  │ STEP 1: DataLoader reads token IDs from dataset                 │
  │                                                                 │
  │   input_ids:  [128000, 9906, 11, 1268, 527, 499, 30, 128001]    │
  │   labels:     [128000, 9906, 11, 1268, 527, 499, 30, 128001]    │
  │                                                                 │
  │   These are just integers — no meaning yet. The DataCollator    │
  │   pads shorter sequences in the batch to the same length.       │
  ├─────────────────────────────────────────────────────────────────┤
  │ STEP 2: Embedding Layer (model.model.embed_tokens)              │
  │                                                                 │
  │   Shape: [128256 vocab_size × 2048 hidden_dim] — a lookup table │
  │                                                                 │
  │   Token ID 9906 → looks up row 9906 → gets a 2048-dim vector    │
  │   Token ID 11   → looks up row 11   → gets a 2048-dim vector    │
  │                                                                 │
  │   Result: [seq_len × 2048] matrix of embedding vectors          │
  │                                                                 │
  │   THIS is where integers become meaningful vectors.             │
  │   These embedding weights are TRAINABLE — they get updated      │
  │   during backprop just like all other parameters.               │
  ├─────────────────────────────────────────────────────────────────┤
  │ STEP 3: 16 Transformer Layers (model.model.layers[0..15])       │
  │                                                                 │
  │   Each layer applies IN ORDER:                                  │
  │                                                                 │
  │   a) RMSNorm — normalize the vectors (stabilizes training)      │
  │                                                                 │
  │   b) Self-Attention (with Grouped Query Attention / GQA):       │
  │      • Q, K, V projections: [2048] → [2048] each                │
  │      • Each token attends to all PREVIOUS tokens (causal mask)  │
  │      • "What should I pay attention to?"                        │
  │      • Output projection: [2048] → [2048]                       │
  │                                                                 │
  │   c) RMSNorm — normalize again                                  │
  │                                                                 │
  │   d) Feed-Forward Network (MLP):                                │
  │      • gate_proj:  [2048] → [8192]  (expand)                    │
  │      • up_proj:    [2048] → [8192]  (expand)                    │
  │      • SiLU activation + element-wise multiply                  │
  │      • down_proj:  [8192] → [2048]  (compress back)             │
  │      • "Process and transform the information"                  │
  │                                                                 │
  │   e) Residual connections add the input back to the output      │
  │      (prevents vanishing gradients in deep networks)            │
  │                                                                 │
  │   After 16 layers: [seq_len × 2048] contextual representations  │
  │   Every token's vector now encodes info from all prior tokens.  │
  ├─────────────────────────────────────────────────────────────────┤
  │ STEP 4: Final Norm + LM Head (model.lm_head)                    │
  │                                                                 │
  │   RMSNorm: normalize the final hidden states                    │
  │   Linear:  [2048] → [128256]  (project to vocabulary size)      │
  │                                                                 │
  │   Result: logits — a score for every token in the vocabulary    │
  │   at every position in the sequence.                            │
  │                                                                 │
  │   Example at position 3 (after "Hello , how"):                  │
  │     "are"  → score 8.2  (high — model is confident)             │
  │     "is"   → score 3.1  (possible but less likely)              │
  │     "zebra"→ score -5.0 (very unlikely)                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ STEP 5: Loss Computation (Cross-Entropy)                        │
  │                                                                 │
  │   Compare predictions vs labels (shifted by 1):                 │
  │                                                                 │
  │   Position:    0        1       2       3        4              │
  │   Input:     <bos>   "Hello"  ","    "how"    "are"             │
  │   Predict:   "Hello"   ","    "how"   "are"   "you"             │
  │   Label:     "Hello"   ","    "how"   "are"   "you"             │
  │                                                                 │
  │   Loss = how wrong were the predictions?                        │
  │   Lower loss = model predicted the training data better.        │
  ├─────────────────────────────────────────────────────────────────┤
  │ STEP 6: Backpropagation                                         │
  │                                                                 │
  │   Loss gradient flows BACKWARD through every layer:             │
  │     lm_head → layer 15 → layer 14 → ... → layer 0 → embed       │
  │                                                                 │
  │   Computes: "how much did each parameter contribute to the      │
  │   error?" — this is the gradient for each of 1.24B params.      │
  │                                                                 │
  │   Gradient checkpointing (enabled): instead of storing all      │
  │   intermediate activations in VRAM, re-computes them during     │
  │   backprop. Trades ~30% more compute for ~40% less VRAM.        │
  ├─────────────────────────────────────────────────────────────────┤
  │ STEP 7: Parameter Update (AdamW optimizer)                      │
  │                                                                 │
  │   For each of the 1,235,814,400 parameters:                     │
  │     • Update momentum (running average of gradients)            │
  │     • Update variance (running average of squared gradients)    │
  │     • Compute adaptive learning rate per parameter              │
  │     • Apply weight decay (regularization)                       │
  │     • new_weight = old_weight - lr * adjusted_gradient          │
  │                                                                 │
  │   AdamW states are kept in FP32 for numerical stability,        │
  │   even though the model weights are in BF16.                    │
  │                                                                 │
  │   With gradient_accumulation_steps=8, this update only happens  │
  │   every 8 mini-batches. Gradients accumulate across batches     │
  │   giving an effective batch size of 8 (1 × 8).                  │
  └─────────────────────────────────────────────────────────────────┘

  This cycle repeats for every batch across all 3 epochs:
    46,584 examples ÷ 8 effective batch = 5,823 steps/epoch × 3 = ~17,469 total steps


"""

import os
import yaml
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from prepare_data import load_and_prepare_dataset


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    print("=" * 60)
    print("  FULL FINE-TUNING (all parameters trainable)")
    print("=" * 60)
    print(f"  Model:  {config['model_name']}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # ──────────────────────────────────────────────────────────
    # 1. Load Tokenizer
    #
    #    Downloads the vocabulary files (~few MB), NOT the model.
    #    The tokenizer converts text → token IDs (integers).
    #    Llama 3 uses BPE via tiktoken with 128,256 vocab tokens.
    #
    #    We need this BEFORE training to:
    #    a) Tokenize the dataset (done in prepare_data.py)
    #    b) Handle padding during batch creation
    #    c) Save alongside the model so inference uses the same vocab
    # ──────────────────────────────────────────────────────────
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ──────────────────────────────────────────────────────────
    # 2. Load Model — NO quantization, full precision
    #
    #    Downloads the full model weights (~2.5 GB in bf16).
    #    This is the actual neural network with 1.24B parameters:
    #
    #    MODEL ARCHITECTURE (what gets loaded):
    #    ┌────────────────────────────────────────────────────┐
    #    │  embed_tokens: [128256 × 2048]  — Embedding table  │
    #    │    Token IDs → 2048-dim vectors (lookup, not math) │
    #    │                                                    │
    #    │  layers[0..15]: 16 Transformer blocks, each with:  │
    #    │    ├─ self_attn: Q,K,V,O projections (attention)   │
    #    │    ├─ mlp: gate_proj, up_proj, down_proj (FFN)     │
    #    │    └─ input/post_attention layernorms (RMSNorm)    │
    #    │                                                    │
    #    │  norm: final RMSNorm                               │
    #    │  lm_head: [2048 → 128256]  — Predicts next token   │
    #    └────────────────────────────────────────────────────┘
    #
    #    bf16: each parameter = 2 bytes (vs 4 bytes for fp32)
    #    So 1.24B params × 2 bytes ≈ 2.5 GB for weights alone
    #
    #    THIS IS THE KEY DIFFERENCE FROM QLORA:
    #    - No BitsAndBytesConfig (no 4-bit quantization)
    #    - Model loaded in bf16 (full precision, not compressed)
    #    - All parameters will be trained (no frozen layers)
    # ──────────────────────────────────────────────────────────
    print("📦 Loading model in bf16 (full weights, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16,        # bf16 for memory savings
        device_map="auto",                  # Place on GPU automatically
        attn_implementation="sdpa",         # Efficient attention (PyTorch 2.0+)
    )



    # Enable gradient checkpointing — critical for fitting in 24GB
    # Gradient checkpointing: during backprop, instead of storing ALL
    # intermediate activations from the forward pass (very VRAM-hungry),
    # it discards them and re-computes them on the fly.
    # Cost: ~30% more compute. Savings: ~40% less VRAM.
    # Without this, a 1.24B model won't fit on 24GB with optimizer states.
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled (saves ~40% VRAM)")


    # Verify: ALL parameters are trainable
    # In full fine-tuning, every single weight gets updated.
    # (In LoRA, only ~0.1-1% of params would be trainable)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %:          {100 * trainable_params / total_params:.1f}%")
    assert trainable_params == total_params, "Not all parameters are trainable!"
    print(f"  ✓ Confirmed: 100% of parameters are trainable (FULL fine-tuning)\n")

    # VRAM BUDGET (approximate for 1.24B params in bf16):
    #   Model weights:      ~2.5 GB  (1.24B × 2 bytes)
    #   Gradients:          ~2.5 GB  (same size as weights)
    #   AdamW optimizer:   ~10.0 GB  (2 × FP32 states = 1.24B × 4 bytes × 2)
    #   Activations:        ~3-5 GB  (reduced by gradient checkpointing)
    #   ─────────────────────────────
    #   Total:             ~18-20 GB  (fits in 24GB 3090 with headroom)
    print()

    # ──────────────────────────────────────────────────────────
    # 3. Load & Prepare Dataset
    #
    #    The dataset was already tokenized by prepare_data.py.
    #    Each example is a dict with:
    #      input_ids: [128000, 9906, 11, ...]  — token integers
    #      labels:    [128000, 9906, 11, ...]  — same (for next-token prediction)
    #      attention_mask: [1, 1, 1, ...]      — which tokens to attend to
    #
    #    NO embeddings yet — just integers. The embedding lookup
    #    happens inside the model's forward pass (Step 2 in the
    #    data flow diagram above).
    # ──────────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)

    # DataCollator: handles dynamic padding at batch time.
    #   - Each example has a different length (e.g., 87, 234, 156 tokens)
    #   - The collator pads all examples in a batch to the longest one
    #   - mlm=False means Causal LM (predict next token), not Masked LM (BERT-style)
    #
    #   - It also shifts labels left by 1 so the model predicts the NEXT token:
    #     input:  [<bos>, "Hello", ",", "how", "are"]
    #     label:  ["Hello", ",", "how", "are", "you"]
    # Data collator handles dynamic padding + label shifting
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # ──────────────────────────────────────────────────────────
    # 4. Training Arguments
    #
    #    Standard HuggingFace Trainer — no SFTTrainer needed
    #    since we're doing vanilla supervised fine-tuning
    # ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config["output_dir"],

        # Training
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],

        # Optimizer
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=int(config.get("warmup_ratio", 0.03) * (
            len(train_dataset) / config["per_device_train_batch_size"] / config["gradient_accumulation_steps"]
        ) * config["num_train_epochs"]),  # Convert warmup_ratio to steps
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config.get("optim", "adamw_torch_fused"),

        # Precision
        bf16=config.get("bf16", True),

        # Gradient checkpointing (already enabled on model, but Trainer needs to know)
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=config.get("logging_steps", 10),
        report_to=config.get("report_to", "tensorboard"),

        # Evaluation & Saving
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 200),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Other
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
    )

    # ──────────────────────────────────────────────────────────
    # 5. Create Trainer & Train
    #
    #    The Trainer orchestrates the entire training loop:
    #
    #    for each epoch (3 total):
    #      for each batch of token IDs:
    #        ① DataLoader loads batch of input_ids (integers)
    #        ② Forward pass:
    #           input_ids → Embedding lookup → 2048-dim vectors
    #           → 16 Transformer layers (attention + FFN)
    #           → LM head → logits (scores for all 128K vocab tokens)
    #        ③ Loss: cross-entropy between predicted vs actual next tokens
    #        ④ Backward pass: compute gradients for ALL 1.24B parameters
    #        ⑤ Every 8 batches (gradient_accumulation_steps):
    #           AdamW updates all parameters using accumulated gradients
    #        ⑥ Every 200 steps: evaluate on eval set (compute eval_loss)
    #        ⑦ Every 400 steps: save a checkpoint to disk
    # ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # renamed from 'tokenizer' in transformers v5+
    )

    print("🚀 Starting full fine-tuning...")
    print(f"   Effective batch size: {config['per_device_train_batch_size']} × {config['gradient_accumulation_steps']} = {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print()

    trainer.train()

    # ──────────────────────────────────────────────────────────
    # 6. Save the FULL model (not just adapters!)
    #
    #    After training, every one of the 1.24B parameters has
    #    been modified by backprop. We save the ENTIRE model:
    #
    #    What gets saved (~2.5 GB total):
    #      model.safetensors  — all updated weights (embed, layers, lm_head)
    #      config.json        — model architecture config
    #      tokenizer files    — vocab, merges, special tokens
    #
    #    With QLoRA you'd only save ~100MB of adapter deltas.
    #    Here we save the complete, standalone model that can
    #    be loaded directly for inference without any base model.
    # ──────────────────────────────────────────────────────────
    final_dir = os.path.join(config["output_dir"], "final")
    print(f"\n💾 Saving full model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n✅ Full fine-tuning complete!")
    print(f"   Model saved to: {final_dir}")
    print(f"   TensorBoard:    tensorboard --logdir {config['output_dir']}")


if __name__ == "__main__":
    main()


"""

How training is done 
The training loop:  yahma/alpaca-cleaned (52K instruction examples) which has 52K instructions 

    * Dataset: 46,584 training examples (52K minus 10% eval split)
        per_device_train_batch_size: 1 (one example at a time — VRAM limit)
        gradient_accumulation_steps: 8
        num_train_epochs: 3


EPOCH 1 (full pass through all 46,584 examples):
│
├─ Batch 1:  example #1     → forward pass → compute gradients (accumulate)
├─ Batch 2:  example #2     → forward pass → compute gradients (accumulate)
├─ Batch 3:  example #3     → forward pass → compute gradients (accumulate)
├─ Batch 4:  example #4     → forward pass → compute gradients (accumulate)
├─ Batch 5:  example #5     → forward pass → compute gradients (accumulate)
├─ Batch 6:  example #6     → forward pass → compute gradients (accumulate)
├─ Batch 7:  example #7     → forward pass → compute gradients (accumulate)
├─ Batch 8:  example #8     → forward pass → ★ UPDATE all 1.24B weights (step 1)
│
├─ Batch 9:  example #9     → forward pass → compute gradients (accumulate)
├─ ...
├─ Batch 16: example #16    → forward pass → ★ UPDATE weights (step 2)
│
├─ ... continues ...
│
├─ Batch 46,584: last example → ★ UPDATE weights (step 5,823)
│
EPOCH 2 (same 46,584 examples again, shuffled differently):
├─ Batch 1:  example #37201  → forward pass → accumulate...
├─ ...
├─ Step 5,824 through step 11,646
│
EPOCH 3 (same examples, shuffled again):
├─ ...
├─ Step 11,647 through step 17,469
│
DONE — total of ~17,469 optimizer steps


Each "instruction" in the Alpaca dataset looks like this:

{
  "instruction": "Explain the difference between a list and a tuple in Python.",
  "input": "",
  "output": "A list is mutable, meaning you can change its contents..."
}
```

After `prepare_data.py` processes it, that becomes a single sequence of token IDs — let's say 187 tokens long. That's **one example, one instruction, one training sample**.

With your config:

**`per_device_train_batch_size: 1`** means the GPU processes **1 instruction per forward pass**. 

That single instruction goes through embedding → 16 transformer layers → loss → backward pass. 

Then the gradients sit in memory.

**`gradient_accumulation_steps: 8`** means it repeats this for **8 separate instructions** before updating weights.

So one "optimizer step" looks like:
```
Instruction 1: "Explain lists vs tuples..."         → forward → backward → hold gradients
Instruction 2: "Write a poem about rain..."         → forward → backward → add to gradients
Instruction 3: "Translate hello to French..."       → forward → backward → add to gradients
Instruction 4: "List 3 benefits of exercise..."     → forward → backward → add to gradients
Instruction 5: "What is machine learning?..."       → forward → backward → add to gradients
Instruction 6: "Sort this list: [3,1,2]..."         → forward → backward → add to gradients
Instruction 7: "Summarize this paragraph..."        → forward → backward → add to gradients
Instruction 8: "Fix this Python code..."            → forward → backward → ★ UPDATE weights

1 instruction per batch, 8 instructions per weight update.

The reason it's 1 and not, say, 32 is purely a VRAM constraint. 

With a batch size of 32, the GPU would need to hold 32 sets of activations simultaneously during the forward pass — that would blow past 24 GB(3090). 

Gradient accumulation is the workaround: you get the mathematical benefit of a larger batch without the memory cost, 
at the expense of taking 8× more forward passes per update.
"""