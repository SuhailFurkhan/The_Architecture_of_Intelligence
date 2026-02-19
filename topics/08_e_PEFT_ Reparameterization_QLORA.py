"""
Fine Tuning - PEFT (Parameter-Efficient Fine-Tuning) Detailed Breakdown
========================================================================

A comprehensive deep dive into PEFT methods — LoRA, QLoRA, Adapters,
Prompt Tuning, and more. Covers the math, the memory, the data flow,
and the practical trade-offs that make PEFT the dominant fine-tuning
paradigm for modern LLMs.
"""

TOPIC_NAME = "Fine_Tuning_PEFT_QLORA"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

PEFT has five categories, and each one answers the same question differently: 

            How do we adapt a frozen model with minimal trainable parameters?

### PEFT (Parameter-Efficient Fine-Tuning) — Detailed Breakdown


                    ════════════════════════════════════════════════════════════════════════════════════
                                               PEFT — WHERE IT SITS IN THE LANDSCAPE
                    ════════════════════════════════════════════════════════════════════════════════════


                                                     ┌──────────────────────┐
                                                     │   FOUNDATION MODEL   │
                                                     │  (LLaMA, Mistral,    │
                                                     │   Qwen, GPT, etc.)   │
                                                     └──────────┬───────────┘
                                                                │
                              ┌─────────────────────────────────┼─────────────────────────────────┐
                              │                                 │                                 │
                              ▼                                 ▼                                 ▼
                ┌──────────────────────────┐   ┌──────────────────────────────────┐   ┌──────────────────────────┐
                │    FULL FINE-TUNING      │   │  ██████████████████████████████  │   │    ALIGNMENT TUNING      │
                │                          │   │  ██  PEFT (Parameter-        ██  │   │                          │
                │  • ALL params updated    │   │  ██  Efficient Fine-Tuning)  ██  │   │  • RLHF, DPO, ORPO       │
                │  • 120+ GB for 7B model  │   │  ██                          ██  │   │  • Human preference      │
                │  • Catastrophic          │   │  ██  • 0.01% – 3% of params  ██  │   │    based optimization    │
                │    forgetting risk       │   │  ██  • Base model FROZEN     ██  │   │  • Often combined with   │
                │  • Full copy per task    │   │  ██  • ~16 GB for 7B model   ██  │   │    PEFT (LoRA + DPO)     │
                └──────────────────────────┘   │  ██  • No forgetting         ██  │   └──────────────────────────┘
                                               │  ██  • Modular adapters      ██  │
                                               │  ██████████████████████████████  │
                                               └───────────────┬──────────────────┘
                                                               │
                                                         THIS MODULE
                                                      covers everything
                                                         below here

---

### The Big Picture — Why PEFT Exists

Full fine-tuning updates ALL parameters. For a 7B model that means:

    - ~28 GB just for weights (FP32)
    - ~28 GB for gradients
    - ~56 GB for optimizer states (Adam)
    - ~10-30 GB for activations
    ────────────────────────────────
    Total: ~120+ GB of GPU VRAM

And every fine-tuned variant is a full copy of the model. Ten tasks = ten 14-28 GB checkpoints.

PEFT's core insight: **you don't need to update all the parameters.**

Research showed that the "intrinsic dimensionality" of fine-tuning is low —
meaning the weight changes needed to adapt a model to a new task live in a much
smaller subspace than the full parameter space. You can capture most of the
adaptation with a tiny fraction of trainable parameters.

Think of it this way: a pre-trained model is a massive building.
Full fine-tuning demolishes and rebuilds every room.
PEFT just redecorates specific rooms — same structural integrity, fraction of the cost.

---


                    ══════════════════════════════════════════════════════════════════════════════
                                              PEFT METHODS — TAXONOMY
                    ══════════════════════════════════════════════════════════════════════════════


                                                ┌───────────────────┐
                                                │    PEFT METHODS   │
                                                └─────────┬─────────┘
                                                          │
              ┌──────────────────┬────────────────────────┼────────────────────────┬──────────────────┐
              │                  │                        │                        │                  │
              ▼                  ▼                        ▼                        ▼                  ▼
    ┌──────────────────┐ ┌───────────────────┐ ┌──────────────────────┐ ┌──────────────────┐ ┌───────────────────┐
    │   ADDITIVE       │ │ REPARAMETERIZATION│ │     SELECTIVE        │ │     HYBRID       │ │   PROMPT-BASED    │
    │                  │ │                   │ │                      │ │                  │ │                   │
    │ Insert NEW       │ │ Decompose weight  │ │ Pick WHICH existing  │ │ Combine multiple │ │ Learn soft tokens │
    │ modules/params   │ │ updates into      │ │ params to train,     │ │ strategies       │ │ prepended to      │
    │ while freezing   │ │ low-rank matrices │ │ freeze the rest      │ │ (e.g. quantize   │ │ the input, no     │
    │ originals        │ │                   │ │                      │ │  + adapters)     │ │ weight changes    │
    │                  │ │                   │ │                      │ │                  │ │                   │
    │ • Bottleneck     │ │ • LoRA  ★         │ │ • BitFit             │ │ • QLoRA  ★       │ │ • Prefix Tuning   │
    │   Adapters       │ │ • DoRA            │ │ • Fish Mask          │ │ • LongLoRA       │ │ • Prompt Tuning   │
    │ • (IA)³          │ │ • LoRA+           │ │ • Diff Pruning       │ │ • LoRA-FA        │ │ • P-Tuning v2     │
    │ • Soft Prompts   │ │ • rsLoRA          │ │                      │ │                  │ │                   │
    │                  │ │ • AdaLoRA         │ │                      │ │                  │ │                   │
    └──────────────────┘ └───────────────────┘ └──────────────────────┘ └──────────────────┘ └───────────────────┘
                                  │                                              │
                                  │                                              │
                           ★ Most popular                                 ★ Most popular
                           general-purpose                                for large models
                           PEFT method                                    on consumer GPUs

---

Additive —  Build new small modules and insert them into the model. 
            The original architecture gets new components bolted on. 
            Bottleneck Adapters are the classic example. 
            Downside: those new modules stay in the model forever, adding inference latency.

Reparameterization — don't change the architecture at all. 
            Instead, decompose the weight updates into smaller matrices (LoRA's A × B). 
            Trains alongside the frozen weights as a parallel bypass, then merges back in and vanishes. 
            Zero inference overhead. This is why LoRA dominates.

Selective — add nothing new, change nothing structurally. 
            Just pick a tiny subset of the model's existing parameters 
            (like bias terms in BitFit, or Fisher-information-selected weights in Fish Mask) and unfreeze only those. 
            Everything else stays frozen. Extremely lightweight but limited expressiveness.

Hybrid — combine strategies from the categories above. 
            QLoRA is the poster child: it takes reparameterization (LoRA adapters in 16-bit) and combines it with 
            quantization (base model compressed to 4-bit). 
            Each technique solves a different bottleneck — LoRA reduces trainable parameters, 
            quantization reduces the memory footprint of the frozen base.

Prompt-based — the most radical approach. Don't touch any weights at all. 
            Instead, learn continuous "soft prompt" vectors that get prepended to the input and steer the model's 
            behavior from the outside. 
            The model itself is completely untouched — you're just learning a better way to talk to it. 
            Fewest parameters (~0.001%), but also the least expressive for smaller models.

---

### The Fundamental PEFT Principle — Freeze, Then Add or Select

Every PEFT method follows the same two-step pattern:

    Step 1: FREEZE the pre-trained model weights (they become read-only)
    Step 2: Either ADD new small trainable components, or SELECT a tiny subset of existing params to train

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                         │
    │   PRE-TRAINED MODEL (FROZEN)                    TRAINABLE PARTS (TINY)                  │
    │                                                                                         │
    │   ┌────────────────────────────────┐            ┌───────────────────────┐               │
    │   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │            │ █████████████████████ │               │
    │   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │            │ ██ 0.1% - 3%       ██ │               │
    │   │ ░░░ 7 Billion Parameters ░░░░  │     +      │ ██ of total params ██ │               │
    │   │ ░░░ (ALL FROZEN)         ░░░░  │            │ ██ (trainable)     ██ │               │
    │   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │            │ █████████████████████ │               │
    │   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │            └───────────────────────┘               │
    │   └────────────────────────────────┘                                                    │
    │                                                                                         │
    │   Gradients: NOT computed for these             Gradients: ONLY computed here           │
    │   Optimizer states: NOT stored                  Optimizer states: ONLY stored here      │
    │   Memory: just inference cost                   Memory: tiny training overhead          │
    │                                                                                         │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

Because the frozen base model only does forward passes (no gradients, no optimizer states),
the memory footprint drops dramatically:

    Full Fine-Tuning (7B model, BF16 + Adam):
        Weights:           14 GB
        Gradients:         14 GB     ← eliminated in PEFT
        Optimizer states:  56 GB     ← eliminated in PEFT
        Activations:    10-30 GB     ← reduced (fewer backward-pass paths)
        ─────────────────────────
        Total:          ~94-114 GB

    PEFT / LoRA (7B model, BF16 base + LoRA adapters):
        Frozen weights:        14 GB   (forward pass only, no gradients/optimizer)
        LoRA adapter weights:  ~20 MB  (trainable)
        LoRA gradients:        ~20 MB
        LoRA optimizer states: ~80 MB
        Activations:         4-10 GB   (reduced)
        ─────────────────────────
        Total:               ~16-24 GB


---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                 QLoRA — QUANTIZED LoRA
                           (Making LoRA Work on Consumer GPUs)
═══════════════════════════════════════════════════════════════════════════════════════════════


### QLoRA — The Core Idea

QLoRA (Quantized LoRA, Dettmers et al. 2023) combines two techniques:

    1. Quantize the frozen base model to 4-bit precision (dramatically reduces memory)
    2. Attach standard LoRA adapters in 16-bit precision (for training)

This lets you fine-tune a 65B parameter model on a single 48GB GPU — something that would
normally require a cluster of 8+ GPUs with full fine-tuning.

---

### What Is Quantization?

Quantization reduces the precision of numbers to save memory:

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                         NUMBER PRECISION FORMATS                                 │
    │                                                                                  │
    │   Format        Bits    Bytes per param    7B model size    Range/Precision      │
    │   ──────        ────    ───────────────    ─────────────    ────────────────     │
    │   FP32          32      4 bytes            28.0 GB          Full precision       │
    │   BF16/FP16     16      2 bytes            14.0 GB          Half precision       │
    │   INT8           8      1 byte              7.0 GB          256 levels           │
    │   NF4            4      0.5 bytes            3.5 GB         16 levels    ★       │
    │   INT4           4      0.5 bytes            3.5 GB         16 levels            │
    │                                                                                  │
    │   ★ NF4 = Normal Float 4-bit, designed specifically for neural network weights   │
    │     which follow a roughly normal (bell curve) distribution                      │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘


**Naive 4-bit quantization destroys model quality.** The key insight of QLoRA is that
the quantized model is ONLY used for the frozen forward pass. The LoRA adapters that
actually get trained remain in full 16-bit precision. Quality loss from quantization
is compensated by the adapters learning corrective adjustments.

---

### QLoRA's Three Innovations

**1. NF4 (4-bit NormalFloat):**

    Standard 4-bit integers divide the number range uniformly:
        INT4 levels: {-8, -7, -6, ..., 0, ..., 5, 6, 7}  (evenly spaced)

    But neural network weights aren't uniformly distributed — they follow a bell curve
    (normal distribution), clustered around zero.

    NF4 places quantization levels according to the normal distribution:
        More levels near zero (where most weights live) → finer precision where it matters
        Fewer levels at extremes (where few weights live) → less precision where it doesn't

    This gives NF4 ~0.5-1% better accuracy than naive INT4 for the same 4-bit budget.


**2. Double Quantization:**

    Quantization requires storing "quantization constants" — scaling factors for each
    block of weights (typically one constant per 64 weights).

    For a 7B model, these constants can consume ~500 MB.

    Double quantization: quantize the quantization constants themselves (from FP32 to FP8).
    Saves ~375 MB. Small on its own, but adds up for larger models.

    ┌────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                │
    │   WITHOUT double quantization:                                                 │
    │       Weights: 4-bit      Constants: FP32 (~500 MB for 7B)                     │
    │                                                                                │
    │   WITH double quantization:                                                    │
    │       Weights: 4-bit      Constants: FP8  (~125 MB for 7B)  ← saved 375 MB     │
    │                                                                                │
    └────────────────────────────────────────────────────────────────────────────────┘


**3. Paged Optimizers:**

    During training, GPU memory usage can spike temporarily (e.g., long sequences).
    If these spikes exceed available VRAM, training crashes with OOM (out of memory).

    Paged optimizers (from bitsandbytes) use CPU RAM as overflow:
    when GPU VRAM is full, optimizer states are automatically paged to CPU RAM,
    then paged back when needed — like virtual memory for GPU training.

    This prevents OOM crashes at the cost of slightly slower training during spikes.

---

### QLoRA Memory Breakdown

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   QLoRA MEMORY BUDGET — 7B MODEL                                                 │
    │                                                                                  │
    │   Component                       Memory         Notes                           │
    │   ─────────                       ──────         ─────                           │
    │   Frozen base model (NF4)         ~3.5 GB        4-bit quantized                 │
    │   Quantization constants          ~0.125 GB      Double-quantized (FP8)          │
    │   LoRA adapters (BF16)            ~0.02 GB       Trainable A and B matrices      │
    │   LoRA gradients                  ~0.02 GB       For adapter params only         │
    │   LoRA optimizer states           ~0.08 GB       Adam states for adapters only   │
    │   Activations + overhead          ~4-8 GB        Forward/backward pass cache     │
    │   ──────────────────────────────────────────────────────────────────────         │
    │   Total:                          ~8-12 GB       ★ Fits on a single 24GB GPU!    │
    │                                                                                  │
    │                                                                                  │
    │   Compare:                                                                       │
    │       Full FT (BF16 + Adam):      ~94-114 GB                                     │
    │       Standard LoRA (BF16):       ~16-24  GB                                     │
    │       QLoRA (NF4 + LoRA):         ~8-12   GB     ★                               │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

    For a 70B model with QLoRA: ~36-48 GB → fits on a single A100 80GB
    (vs. ~1 TB+ for full fine-tuning)

---

### QLoRA Data Flow — What Happens During Training

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   QLoRA FORWARD + BACKWARD PASS                                                      │
    │                                                                                      │
    │   ┌───────────────┐       ┌────────────────────────────────────────────────┐         │
    │   │               │       │              TRANSFORMER LAYER                 │         │
    │   │  Input tokens │──────▶│                                                │         │
    │   │  (BF16)       │       │   ┌────────────────────┐                       │         │
    │   │               │       │   │  W₀ (frozen, NF4)  │                       │         │
    │   └───────────────┘       │   │                    │                       │         │
    │                           │   │  Dequantize on     │──── W₀x ────┐         │         │
    │                           │   │  the fly: NF4→BF16 │             │         │         │
    │                           │   │  (not stored,      │             ▼         │         │
    │                           │   │   computed fresh   │         ┌────────┐    │         │
    │                           │   │   each time)       │         │   +    │──▶ │ output  │
    │                           │   └────────────────────┘         └────────┘    │         │
    │                           │                                      ▲         │         │
    │                           │   ┌──────────┐  ┌──────────┐        │          │         │
    │                           │   │ A (BF16) │─▶│ B (BF16) │── BAx ─┘          │         │
    │                           │   │ trainable│  │ trainable│   × (α/r)         │         │
    │                           │   └──────────┘  └──────────┘                   │         │
    │                           │                                                │         │
    │                           └────────────────────────────────────────────────┘         │
    │                                                                                      │
    │   BACKWARD PASS:                                                                     │
    │   • Gradients flow back through the LoRA path (A and B only)                         │
    │   • Frozen NF4 weights: no gradients computed, no updates                            │
    │   • Only A and B get updated by the optimizer                                        │
    │   • Base weights are dequantized again during backprop (NF4 → BF16 on the fly)       │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘


    Key detail: the NF4 weights are NEVER stored in BF16. Every time the model needs them
    (forward or backward), it dequantizes on the fly. This costs compute but saves memory.
    The trade-off: QLoRA training is ~30-50% slower than standard LoRA, but uses ~50% less memory.

---

### QLoRA vs LoRA — When to Use Which

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   Scenario                                 Recommendation                        │
    │   ────────                                 ──────────────                        │
    │   Single 24GB GPU (RTX 3090/4090)          QLoRA  (only option that fits)        │
    │   Single 48GB GPU (A6000)                  LoRA   (faster, no quant overhead)    │
    │   Single 80GB GPU (A100/H100)              LoRA   (plenty of room)               │
    │   70B model on one GPU                     QLoRA  (essential)                    │
    │   Maximum training speed                   LoRA   (~30-50% faster than QLoRA)    │
    │   Maximum quality at 7B scale              LoRA   (no quantization noise)        │
    │   Tight budget, large model                QLoRA  (the whole point)              │
    │                                                                                  │
    │   Quality difference: typically <1% between LoRA and QLoRA on benchmarks.        │
    │   For most practical purposes, QLoRA quality ≈ LoRA quality.                     │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---


What Problem Does QLoRA Solve ?

Standard LoRA already reduced trainable parameters to ~0.1% of the model. 
But there's still a bottleneck: the frozen base model itself still takes up massive memory just sitting in GPU VRAM doing forward passes.

For a 7B model with LoRA:

    Frozen weights (BF16):       14 GB      ← THIS IS THE PROBLEM
    LoRA weights:               ~17 MB
    LoRA gradients:             ~17 MB
    LoRA optimizer:             ~67 MB
    Activations:               4-10 GB
   ────────────────────────────────────
    Total:                   ~18-24 GB


14 GB of that is just the frozen model sitting there. 
It's never updated, never trained — it only does forward passes. 
So QLoRA asks: can we compress those frozen weights to use less memory, while keeping the LoRA training quality intact?

The answer is yes — compress the frozen model to 4-bit precision (from 16-bit), cutting its memory footprint by ~4×. 
The LoRA adapters that actually get trained stay in full 16-bit precision so training quality isn't compromised.

---

Step 0: Understand the Precision Formats

Before diving into QLoRA's steps, you need to understand what 4-bit means versus 16-bit:

A single model weight (one number out of 7 billion):

FP32:    0.0234375                          stored as 32 bits (4 bytes)
         ┌─────────────────────────────────┐
         │ 0 01111001 10000000000000000000 │   full precision
         └─────────────────────────────────┘

BF16:    0.0234                             stored as 16 bits (2 bytes)
         ┌─────────────────┐
         │ 0 01111001 1000 │                   half precision — what LoRA uses
         └─────────────────┘

NF4:     0.02                               stored as 4 bits (0.5 bytes)
         ┌──────┐
         │ 0110 │                               only 16 possible values!
         └──────┘

7B parameters × bytes per param:
    FP32:  7B × 4 bytes  = 28 GB
    BF16:  7B × 2 bytes  = 14 GB
    NF4:   7B × 0.5 bytes = 3.5 GB     ← QLoRA uses this for frozen weights

The catch: 4-bit only has 16 possible values. 
You're mapping 7 billion numbers that could each be any decimal into just 16 discrete levels. 
That's a massive information loss. This is where NF4 (NormalFloat 4-bit) comes in — QLoRA's first innovation.

---

Step 1: Quantize the Pre-Trained Model to 4-Bit NF4

This is the first step that differs from standard LoRA. 
Instead of loading the model in BF16 (14 GB), you load it in 4-bit NF4 (3.5 GB).

What NF4 does differently from naive 4-bit:
Neural network weights follow a bell curve (normal distribution) — most values cluster near zero, with few extreme values. 
Naive INT4 spaces its 16 levels evenly across the range, wasting precision:


    Weight distribution (bell curve):
    
         ▲ frequency
         │
         │           ████
         │         ████████
         │       ████████████
         │     ████████████████
         │   ████████████████████
         │ ████████████████████████
         └──────────────────────────▶  weight value
        -1.0       0.0        +1.0
    
        Most weights live HERE          Few weights out HERE
        (near zero)                     (at extremes)
    
    
    INT4 quantization levels (evenly spaced):
        │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
       -1.0                     0.0                      +1.0
    
        Same precision everywhere.
        WASTES levels at extremes where few weights live.
        INSUFFICIENT precision near zero where most weights live.
    
    
    NF4 quantization levels (spaced by normal distribution):
        │        │    │  │ │││ │  │    │        │
       -1.0                0.0                +1.0
    
        MORE levels packed near zero (where weights are dense).
        FEWER levels at extremes (where weights are sparse).
        Better approximation for the SAME 4-bit budget.

---

How quantization actually works — the block structure:
You can't just map each weight to its nearest NF4 level globally because weights in different layers have different scales. 
QLoRA quantizes in blocks:


Take a block of 64 consecutive weights:

    Original (BF16): [0.023, -0.891, 0.445, 0.012, -0.338, ..., 0.112]   (64 values)
    
    Step 1: Find the absmax (largest absolute value) in this block
            absmax = 0.891
    
    Step 2: Normalize all values to [-1, 1] range
    
            normalized = original / absmax
    
            [0.026, -1.000, 0.499, 0.013, -0.379, ..., 0.126]
    
    Step 3: Map each normalized value to its nearest NF4 level
    
            NF4 levels: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844,
                         -0.1848, -0.0911, 0.0,
                          0.0796, 0.1609, 0.2461, 0.3379, 0.4407,
                          0.5626, 0.7230, 1.0]
            
            0.026  → nearest is 0.0     → stored as index 7  (4 bits)
            -1.000 → nearest is -1.0    → stored as index 0  (4 bits)
            0.499  → nearest is 0.4407  → stored as index 12 (4 bits)
            ...
    
    Step 4: Store the absmax (0.891) as the quantization constant
            This is needed to dequantize: real_value = NF4_level × absmax
    
    Result:
        64 weights stored as 64 × 4 bits = 32 bytes   (was 64 × 2 bytes = 128 bytes in BF16)
        + 1 quantization constant (FP32): 4 bytes
    
        Total: 36 bytes for 64 weights = 0.5625 bytes/weight

For the entire 7B model:

    7B weights ÷ 64 per block = ~109 million blocks
    
    NF4 weights:              7B × 0.5 bytes  = 3.5 GB
    Quantization constants:   109M × 4 bytes  = ~437 MB
    ────────────────────────────────────────────────────
    Total:                                      ~3.94 GB

---

Step 2: Double Quantization — Compressing the Constants

Those 437 MB of quantization constants (one FP32 float per block of 64 weights) are themselves wasteful. 
QLoRA's second innovation: quantize the quantization constants.

WITHOUT double quantization:
    Weights: NF4 (4-bit)                    =   3.5 GB
    Quantization constants: FP32 (32-bit)   =  ~437 MB
    ──────────────────────────────────────────────────
    Total:                                    ~3.94 GB


WITH double quantization:
    Weights: NF4 (4-bit)                    =  3.5 GB
    Quantization constants: FP8 (8-bit)     = ~109 MB       ← 4× smaller
    Second-level constants: FP32            =   ~7 MB       ← tiny (1 per 256 blocks)
    ─────────────────────────────────────────────────
    Total:                                   ~3.62 GB

    Savings: ~320 MB

Not huge on its own, but for a 70B model this saves ~3.2 GB — meaningful when you're pushing against VRAM limits.

---

Step 3: Prepare the Model for Training — prepare_model_for_kbit_training

After loading in 4-bit, the model needs some adjustments before LoRA can be applied. 
This is a QLoRA-specific step that doesn't exist in standard LoRA:

    model = prepare_model_for_kbit_training(model)

What this does:
    
    1. Enable gradient checkpointing
       Activations from the forward pass are normally stored for backprop.
       With 4-bit weights, memory is tight. Gradient checkpointing trades
       compute for memory: discard intermediate activations, recompute them
       during the backward pass.
       
    2. Cast LayerNorm layers to FP32
       LayerNorm is sensitive to precision. With 4-bit weights feeding into
       it, keeping LayerNorm in FP32 stabilizes training. These are tiny
       layers (just a few thousand parameters each) so the memory cost
       is negligible.
       
    3. Cast the LM head output layer to FP32
       The final vocabulary projection needs full precision for stable
       loss computation.
       
    4. Enable input gradient computation
       Even though the base weights are frozen, gradients must flow
       THROUGH the quantized layers to reach the LoRA adapters deeper
       in the network. This flag ensures the computation graph stays
       connected.

---

Step 4: Attach LoRA Adapters (Same as Standard LoRA)

This step is identical to standard LoRA. You create A and B matrices for each target module:

For each target weight matrix in each layer:
    A: [r × d_in]      initialized with random Gaussian    requires_grad = True
    B: [d_out × r]     initialized to ALL ZEROS            requires_grad = True

These LoRA adapters are in BF16 (16-bit) — NOT quantized.
Only the frozen base weights are in NF4.

The crucial asymmetry:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   FROZEN BASE MODEL          LoRA ADAPTERS                     │
│                                                                │
│   Precision: NF4 (4-bit)     Precision: BF16 (16-bit)          │
│   Purpose:   Forward pass    Purpose:   Training               │
│   Gradients: None            Gradients: Yes                    │
│   Optimizer: None            Optimizer: Adam (FP32)            │
│   Memory:    3.5 GB          Memory:    ~100 MB total          │
│                                                                │
│   LOSSY compression is OK    FULL precision is needed          │
│   because these weights      because these are the params      │
│   are never updated.         that actually learn.              │
│                                                                │
└────────────────────────────────────────────────────────────────┘

---

Step 5: Data Pipeline (Identical to LoRA and Full Fine-Tuning)

Nothing changes here whatsoever:

    Raw JSONL → Template formatting → Tokenize → Labels + loss mask → Pad → Attention mask → Collate into tensors → Move to GPU

batch = 
    {
        "input_ids":      [batch_size, seq_len],
        "attention_mask":  [batch_size, seq_len],
        "labels":         [batch_size, seq_len]
    }

The data has no idea whether the model is full precision, LoRA, or QLoRA. Same inputs, same format, same pipeline.

---

Step 6: Forward Pass — Where QLoRA Gets Interesting

This is the key difference from standard LoRA. 
Every time a frozen weight matrix is needed during the forward pass, it must be dequantized on the fly from NF4 back to BF16:

Standard LoRA forward pass (one target weight):

    h = W₀x + (α/r)·BAx
        ───
         ↑
         W₀ is already in BF16 — just do the multiplication


QLoRA forward pass (same computation, but W₀ is in NF4):

    Step 6a: Dequantize W₀ from NF4 → BF16
    
        For each block of 64 weights:
            real_value = NF4_level × absmax_constant
        
        NF4 stored:    [index_7, index_0, index_12, ...]    (4-bit integers)
        absmax:        0.891                                 (from quantization constant)
        
        Dequantized:   [0.0 × 0.891,  -1.0 × 0.891,  0.4407 × 0.891, ...]
                     = [0.0,           -0.891,          0.3926,         ...]
        
        This produces a BF16 copy of W₀ for THIS computation.
        This BF16 copy is TEMPORARY — it's NOT stored.
        It's computed, used for the matrix multiply, then discarded.
    
    
    Step 6b: Compute frozen path
    
        h_frozen = dequantized_W₀ × x        (standard matmul, BF16)
        
        Then DISCARD dequantized_W₀ — don't store it.
    
    
    Step 6c: Compute LoRA path (unchanged from standard LoRA)
    
        h_lora = B × (A × x)                  (BF16, trainable)
    
    
    Step 6d: Combine
    
        h = h_frozen + (α/r) × h_lora         (element-wise add)

Traced through one full layer:

    ┌──────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                          │
    │   LAYER 0 — QLoRA FORWARD PASS                                                           │
    │                                                                                          │
    │   Input: h = [batch, seq, 4096] (BF16)                                                   │
    │                                                                                          │
    │   ┌─────────────────────────────────────────────────────────────────────────────────┐    │
    │   │  QUERY COMPUTATION (W_q has LoRA)                                               │    │
    │   │                                                                                 │    │
    │   │  x ──────┬──────────────────────────────────────┐                               │    │
    │   │          │                                      │                               │    │
    │   │          ▼                                      ▼                               │    │
    │   │  ┌─────────────────────────────┐    ┌──────────────────────────┐                │    │
    │   │  │  W_q (NF4, frozen)          │    │  LoRA A_q, B_q (BF16)    │                │    │
    │   │  │                             │    │                          │                │    │
    │   │  │  Step 1: Dequantize         │    │  Standard LoRA:          │                │    │
    │   │  │    NF4 → BF16 on the fly    │    │  x_down = A_q × x  [8]   │                │    │
    │   │  │    (NOT stored, temporary)  │    │  x_up   = B_q × x_down   │                │    │
    │   │  │                             │    │          [4096]          │                │    │
    │   │  │  Step 2: Multiply           │    │                          │                │    │
    │   │  │    q_frozen = deq(W_q) × x  │    │  q_lora = (α/r) × x_up   │                │    │
    │   │  │                             │    │                          │                │    │
    │   │  │  Step 3: Discard deq(W_q)   │    │  Gradients: YES          │                │    │
    │   │  │    (don't keep it)          │    │                          │                │    │
    │   │  │                             │    │                          │                │    │
    │   │  │  Gradients: flow through    │    │                          │                │    │
    │   │  │  but NOT stored for W_q     │    │                          │                │    │
    │   │  └──────────────┬──────────────┘    └────────────┬─────────────┘                │    │
    │   │                 │                                │                              │    │
    │   │                 └──────────── + ────────────────┘                               │    │
    │   │                               │                                                 │    │
    │   │                               ▼                                                 │    │
    │   │                          q = q_frozen + q_lora   [4096]                         │    │
    │   │                                                                                 │    │
    │   └─────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                          │
    │   (same process for K, V, O projections)                                                 │
    │   (same process for MLP weights if "all-linear" targeting)                               │
    │                                                                                          │
    │   IMPORTANT: the dequantization happens EVERY TIME the weight is needed.                 │
    │   Forward pass: dequantize W_q → use → discard                                           │
    │   Backward pass: dequantize W_q AGAIN → use for chain rule → discard again               │
    │                                                                                          │
    │   This is why QLoRA is ~30-50% slower than standard LoRA —                               │
    │   you're paying compute to dequantize instead of paying memory to store.                 │
    │                                                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────┘

---

Step 7: Loss Computation (Identical)

    logits = [batch, seq_len, vocab_size]
    labels = [-100, -100, ..., 6374, 2]
    
    loss = CrossEntropyLoss(logits, labels)
    
---

Step 8: Backward Pass — Dequantize Again

During backpropagation, the chain rule needs the weight values to compute gradients for the LoRA matrices. 
The NF4 weights must be dequantized a second time:


    Forward pass:  dequantize W₀ (NF4 → BF16) → compute → discard
    Backward pass: dequantize W₀ (NF4 → BF16) → compute gradients for A, B → discard
    
    Standard LoRA: W₀ is already in BF16, always available — no dequantization cost.
    QLoRA:         W₀ must be reconstructed TWICE per training step.
    
    What gets gradient storage:
        Frozen NF4 weights:    NO gradients stored (requires_grad = False)
        LoRA A and B:          YES gradients stored (requires_grad = True)
        
    Same as standard LoRA — only the adapter gradients consume memory.
    The dequantization is pure compute overhead, not memory overhead.   

---

Step 9: Paged Optimizers — QLoRA's Third Innovation

    During training, GPU memory usage isn't constant. Long sequences, large batches, 
    or gradient accumulation can create memory spikes that temporarily exceed VRAM. 
    
    In standard LoRA with plenty of headroom (18-24 GB on a 48 GB GPU), this isn't a problem. 
    In QLoRA where you're operating at 8-12 GB on a 24 GB GPU, any spike can crash training with an OOM error.
    
    QLoRA introduces paged optimizers from the bitsandbytes library:
    
    NORMAL OPTIMIZER:
        All optimizer states live in GPU VRAM at all times.
        Memory spike → exceeds VRAM → OOM crash → training dies.
    
        GPU VRAM [24 GB]:
        ┌──────────────────────────────────────────────────────────────┐
        │ Model (3.5GB) │ LoRA │ Activations │ Optimizer │  SPIKE!██   │ ← OOM!
        └──────────────────────────────────────────────────────────────┘
    
    
    PAGED OPTIMIZER:
        Optimizer states can overflow to CPU RAM when GPU is full.
        Memory spike → optimizer states automatically paged to CPU →
        spike passes → states paged back to GPU. Training continues.
    
        GPU VRAM [24 GB]:
        ┌──────────────────────────────────────────────────────────────┐
        │ Model (3.5GB) │ LoRA │ Activations │ Optim(partial) │ SPIKE  │ ← survives
        └──────────────────────────────────────────────────────────────┘
                                                   │
                                          paged to │ CPU RAM
                                                   ▼
        CPU RAM:
        ┌──────────────────────────────────────────────────────────────┐
        │ Overflow optimizer states (temporary)                        │
        └──────────────────────────────────────────────────────────────┘
    
        Cost: slightly slower during spikes (CPU↔GPU transfer)
        Benefit: training doesn't crash

This is implemented by using paged_adamw_8bit or paged_adamw_32bit optimizers from bitsandbytes instead of the standard PyTorch Adam.

---

Step 10: Optimizer Update (Same Logic as LoRA)

    optimizer.step():
    Only LoRA A and B matrices get updated.
    Frozen NF4 weights: untouched (still in 4-bit, never modified).
    
    Adam computes momentum and variance for each LoRA parameter.
    Updates the A and B matrices.
    
    Memory: ~67 MB optimizer states (same as standard LoRA —
    the optimizer only sees the LoRA params, it doesn't know
    or care that the base model is quantized).
    
---

Step 11: Save Adapter (Identical to Standard LoRA)

    Saved:
    adapter_model.safetensors     ~33 MB    (LoRA A and B matrices, in BF16)
    adapter_config.json           ~1 KB     (r, α, targets, base model name)

    NOT saved:
        The 4-bit quantized model — not needed.
        The adapter config references the original base model by name.

The saved adapter is identical to a standard LoRA adapter. 
There's no difference in the checkpoint files. 
The quantization was only a training-time optimization — it doesn't affect what gets saved.

---

Step 12: Deploy — Merge or Swap (With a Wrinkle)

    Here's where QLoRA has a subtle difference from standard LoRA:
    
Option A: Merge into a full-precision model

    Step 1: Load the ORIGINAL base model in BF16 (not the 4-bit version)
    Step 2: Load the LoRA adapter
    Step 3: Merge: W_merged = W₀(BF16) + (α/r) · B × A
    Step 4: Save the merged model
    
    The merged model is full BF16 — no quantization artifacts.
    Same quality as if you'd trained with standard LoRA.

Option B: Merge into a quantized model (for serving)

    Step 1: Load the base model in 4-bit (for memory-efficient serving)
    Step 2: Load the LoRA adapter
    Step 3: Use as-is (two-path forward, same as during training)
    
    OR:
    
    Step 1: Load the base model in BF16
    Step 2: Merge the adapter
    Step 3: Re-quantize to GGUF/GPTQ/AWQ for serving
    
    This is the common production path:
        Train with QLoRA → merge at full precision → re-quantize for deployment
        
        
Option C: Swap adapters (same as standard LoRA)
        
    Load 4-bit base model once → swap LoRA adapters per task.
    Same adapter-swapping workflow as standard LoRA.

---

**The Full QLoRA Memory Stack — Concrete Numbers**

QLoRA (7B model, NF4 base, LoRA rank=8, 4 attention targets):

    Component                         Memory         Why
    ─────────                         ──────         ───
    Frozen base (NF4)                 3.50 GB        4-bit quantized weights
    Quantization constants (FP8)      0.11 GB        Double-quantized (Step 2)
    Second-level constants (FP32)     0.007 GB       Constants of constants
    LayerNorm layers (FP32)           0.003 GB       Cast to FP32 for stability
    LM Head (FP32)                    0.13 GB        Full precision for loss
    LoRA A and B matrices (BF16)      0.017 GB       Trainable adapters
    LoRA gradients (BF16)             0.017 GB       Gradient storage
    LoRA optimizer states (FP32)      0.067 GB       Adam momentum + variance
    Activations (mixed)               4-8 GB         Forward/backward cache
    ─────────────────────────────────────────────────────────
    Total:                            ~8-12 GB

Compare:
    Full fine-tuning:   94-114 GB    (requires multi-GPU cluster)
    Standard LoRA:      18-24 GB     (requires A6000 or better)
    QLoRA:              8-12 GB      (fits on RTX 3090/4090!)


What QLoRA Loses — The Trade-Offs

QLoRA isn't free. You're trading:

Speed for memory. Every weight matrix must be dequantized from NF4 → BF16 twice per training step (forward and backward).
This makes QLoRA ~30-50% slower than standard LoRA per step. 
A training run that takes 4 hours with LoRA takes ~5-6 hours with QLoRA.

Precision for memory. The 4-bit quantization introduces approximation error. 
When the frozen path computes dequantized(W₀) × x, the dequantized weight isn't exactly the original W₀ — it's been rounded to one of 16 levels. 
This quantization noise flows through the entire network. 
The LoRA adapters partially compensate (they learn corrections that account for the quantization error), but there's a small quality gap.

In practice:

    Task                    LoRA Quality     QLoRA Quality     Gap
    ───────────────────     ────────────     ─────────────     ─────
    Instruction tuning       92.1%            91.4%            -0.7%
    Code generation          84.3%            83.8%            -0.5%
    Summarization            89.7%            89.0%            -0.7%
    Classification           94.2%            93.6%            -0.6%
    
    The gap is typically <1%. For most practical purposes,
    QLoRA ≈ LoRA ≈ Full fine-tuning in quality.


The Complete QLoRA Pipeline — All Steps in Sequence

    Step 1:   Load base model in NF4 (4-bit quantization with NormalFloat4)
          → 14 GB model compressed to ~3.5 GB
          
    Step 2:   Double quantization (compress quantization constants FP32 → FP8)
              → Saves ~320 MB more
    
    Step 3:   prepare_model_for_kbit_training()
              → Enable gradient checkpointing
              → Cast LayerNorm and LM Head to FP32
              → Enable input gradients for chain rule
    
    Step 4:   Attach LoRA adapters in BF16 (same as standard LoRA)
              → A initialized random, B initialized zero
              → ΔW = 0 at start → model starts as pre-trained
    
    Step 5:   Data pipeline (100% identical to full fine-tuning / LoRA)
              → JSONL → template → tokenize → labels → pad → batch
    
    Step 6:   Forward pass
              → Dequantize each NF4 weight block → BF16 on the fly
              → Compute frozen path: dequantized(W₀) × x
              → Compute LoRA path: B(Ax)
              → Combine: h = frozen + (α/r) · LoRA
              → Discard dequantized weights (not stored)
    
    Step 7:   Loss = CrossEntropy(logits, labels) — identical
    
    Step 8:   Backward pass
              → Dequantize weights AGAIN for chain rule
              → Compute and store gradients for LoRA A and B only
              → Discard dequantized weights again
    
    Step 9:   Paged optimizer handles memory spikes
              → Overflow optimizer states to CPU if GPU VRAM is full
    
    Step 10:  Optimizer updates LoRA A and B only — identical to LoRA
    
    Step 11:  Save adapter files (~33 MB) — identical to LoRA
    
    Step 12:  Deploy: merge adapter into full-precision base model
              → Re-quantize for serving if needed


The Core Insight — Why QLoRA Works

The entire trick rests on one asymmetry: the frozen weights and the trainable weights serve different purposes and need different precision.

The frozen weights just need to produce approximately correct forward pass outputs. 
The LoRA adapters then learn to compensate for any approximation errors alongside learning the task itself. 
It's like having a rough sketch (4-bit base) that a precise artist (16-bit LoRA) refines.

If you quantized the LoRA adapters to 4-bit too, training would collapse — you need full precision for gradient computation and weight updates. 
The key insight is that quantization is safe for inference-only weights but destructive for training weights, and QLoRA exploits this asymmetry perfectly.





═══════════════════════════════════════════════════════════════════════════════════════════════
                                    LoRA VARIANTS
═══════════════════════════════════════════════════════════════════════════════════════════════


### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA (Liu et al. 2024) decomposes weight updates into magnitude and direction components:

    Standard LoRA:    W = W₀ + BA         (single combined update)
    DoRA:             W = m · (W₀ + BA) / ||W₀ + BA||

    Where m is a learnable magnitude vector and the rest captures direction.

    Intuition: LoRA conflates "how much" and "which way" to adjust weights.
    DoRA separates them — like adjusting both the brightness (magnitude) and
    the hue (direction) of a color independently instead of changing both at once.

    Result: DoRA consistently outperforms LoRA by 1-3% on benchmarks,
    at the cost of slightly more parameters (~10% more) and training time.


### LoRA+ (Different Learning Rates for A and B)

    Standard LoRA: same learning rate for both A and B matrices.
    LoRA+: uses different learning rates — typically B gets a higher LR than A.

    Why? A (down-projection) maps to a compressed space.
    B (up-projection) maps back to the original space.
    They play different roles and benefit from different update magnitudes.

    Typically: lr_B = 2× to 8× lr_A
    Result: faster convergence, slightly better final performance.


### rsLoRA (Rank-Stabilized LoRA)

    Standard LoRA scaling:  (α/r) · BAx
    rsLoRA scaling:         (α/√r) · BAx

    Problem: as rank r increases, the standard α/r scaling shrinks the adapter
    contribution, requiring retuning α. rsLoRA replaces r with √r so that
    the scaling remains stable across different ranks.

    Result: you can freely change rank without retuning α. Practical convenience.


### AdaLoRA (Adaptive Low-Rank Adaptation)

    Standard LoRA uses the same rank r for every target layer.
    AdaLoRA dynamically allocates rank across layers based on importance.

    Important layers (measured by sensitivity analysis) get higher rank.
    Unimportant layers get lower rank or are pruned entirely.

    Think of it as a budget allocation problem — given a fixed parameter budget,
    spend more on the layers that matter most.

    Uses SVD-based parameterization (W = P × Λ × Q) instead of the standard
    A × B decomposition, which enables pruning by zeroing out singular values.


### LoRA-FA (Frozen-A LoRA)

    Freezes matrix A after initialization and only trains B.
    Cuts trainable parameters roughly in half.

    A is initialized with random Gaussian and stays fixed.
    B learns to adapt the random projections from A to the task.

    Surprisingly effective — the random projection in A provides sufficient
    diversity, and B alone can learn task-specific adaptations.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   LoRA VARIANTS COMPARISON                                                       │
    │                                                                                  │
    │   Method      Key Change                  Params vs LoRA    Quality vs LoRA      │
    │   ──────      ──────────                  ──────────────    ───────────────      │
    │   LoRA        Baseline                    1×                Baseline             │
    │   DoRA        Magnitude + direction       ~1.1×             +1-3%                │
    │   LoRA+       Different LR for A, B       1×                +0.5-2%              │
    │   rsLoRA      √r scaling                  1×                Same (more stable)   │
    │   AdaLoRA     Dynamic rank allocation     ≤1× (pruned)      +1-2%                │
    │   LoRA-FA     Freeze A, train B only      ~0.5×             -0.5% to same        │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                 BOTTLENECK ADAPTERS
                           (The Original PEFT Method)
═══════════════════════════════════════════════════════════════════════════════════════════════


### Bottleneck Adapters — How They Work

Adapters (Houlsby et al. 2019) were the first major PEFT method. They insert small
feed-forward networks (bottleneck modules) between existing transformer layers.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   TRANSFORMER LAYER — WITHOUT vs. WITH ADAPTERS                                  │
    │                                                                                  │
    │   WITHOUT ADAPTERS:              WITH ADAPTERS:                                  │
    │                                                                                  │
    │   ┌───────────────┐               ┌───────────────┐                              │
    │   │ Self-Attention│               │ Self-Attention│ (frozen)                     │
    │   └──────┬────────┘               └───────┬───────┘                              │
    │          │                                │                                      │
    │          │                         ┌──────┴────────┐                             │
    │          │                         │ ██ ADAPTER ██ │ ← NEW, trainable            │
    │          │                         │ ██ down+up ██ │                             │
    │          │                         └──────┬────────┘                             │
    │          │                                │                                      │
    │   ┌──────┴────────┐                ┌──────┴───────┐                              │
    │   │   Add & Norm  │                │   Add & Norm │                              │
    │   └──────┬────────┘                └──────┬───────┘                              │
    │          │                                │                                      │
    │   ┌──────┴───────┐                 ┌──────┴───────┐                              │
    │   │ Feed-Forward │                 │ Feed-Forward │ (frozen)                     │
    │   └──────┬───────┘                 └──────┬───────┘                              │
    │          │                                │                                      │
    │          │                         ┌──────┴────────┐                             │
    │          │                         │ ██ ADAPTER ██ │ ← NEW, trainable            │
    │          │                         │ ██ down+up ██ │                             │
    │          │                         └──────┬────────┘                             │
    │          │                                │                                      │
    │   ┌──────┴───────┐                 ┌──────┴───────┐                              │
    │   │   Add & Norm │                 │   Add & Norm │                              │
    │   └──────────────┘                 └──────────────┘                              │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘


**Inside an Adapter Module:**

    Input (hidden_dim = 4096)
        ↓
    Down-projection: [4096 → bottleneck_dim]    (e.g., 4096 → 64)
        ↓
    Non-linearity (ReLU or GELU)
        ↓
    Up-projection: [bottleneck_dim → 4096]      (e.g., 64 → 4096)
        ↓
    Residual connection: output = adapter_output + input
        ↓
    Output (hidden_dim = 4096)

    The bottleneck_dim controls capacity — like LoRA's rank.
    A bottleneck of 64 with hidden_dim 4096 gives a 64× compression.


**Adapters vs LoRA:**

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   Property                    Adapters                   LoRA                    │
    │   ────────                    ────────                   ────                    │
    │   Architecture                Sequential (in-series)     Parallel (bypass)       │
    │   Inference overhead          Yes (extra layers)         None (after merge)      │
    │   Merge into base model       No                         Yes                     │
    │   Non-linearity               Yes (ReLU/GELU)            No (purely linear)      │
    │   Parameters (typical)        ~0.5-3%                    ~0.1-1%                 │
    │   Popularity (2024+)          Declining                  Dominant                │
    │                                                                                  │
    │   LoRA won because: (1) no inference overhead, (2) mergeable, (3) simpler.       │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                 (IA)³ — INFUSED ADAPTER BY INHIBITING
                                       AND AMPLIFYING INNER ACTIVATIONS
═══════════════════════════════════════════════════════════════════════════════════════════════

### (IA)³ — Rescaling Vectors

(IA)³ (Liu et al. 2022) is the most parameter-efficient method — it learns three vectors
that rescale the keys, values, and feed-forward activations:

    Standard attention: Q × K^T × V
    (IA)³ attention:    Q × (l_k ⊙ K)^T × (l_v ⊙ V)

    Standard FFN:       FFN(x)
    (IA)³ FFN:          l_ff ⊙ FFN(x)

    Where ⊙ is element-wise multiplication and l_k, l_v, l_ff are learned vectors.

    Total trainable parameters: just 3 vectors per layer.
    For a 7B model: ~0.01% of parameters (~700K total).

    Extremely lightweight, but less expressive than LoRA.
    Best for few-shot learning where you need many task-specific adapters
    and memory is at an absolute premium.

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                SELECTIVE METHODS
═══════════════════════════════════════════════════════════════════════════════════════════════


### BitFit — Training Only Bias Terms

BitFit (Zaken et al. 2021) freezes all weight matrices and only trains the bias terms.

    Standard linear layer:   y = Wx + b
    BitFit:                  y = W(frozen)x + b(trainable)

    In a typical transformer, bias terms are ~0.05-0.1% of total parameters.
    For a 7B model, that's roughly 3.5-7 million trainable parameters.

    Surprisingly effective for many NLU tasks (classification, NER, etc.).
    Less effective for complex generation tasks where the model needs to 
    learn new patterns of output.

    Note: many modern LLMs (LLaMA, Mistral) don't use bias terms at all,
    making BitFit inapplicable. Check your model architecture first.


### Fish Mask — Learned Binary Masks

Fish Mask (Sung et al. 2021) uses Fisher information to identify the most important
parameters, then creates a binary mask: 1 = train this parameter, 0 = freeze it.

    Step 1: Compute Fisher information for each parameter (measures sensitivity)
    Step 2: Select top-k% most important parameters
    Step 3: Train only those parameters, freeze the rest

    Like surgical fine-tuning — identifying exactly which weights matter most
    for your specific task and only updating those.

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                PROMPT-BASED METHODS
═══════════════════════════════════════════════════════════════════════════════════════════════

### Prompt-Based Methods — No Weight Changes at All

These methods don't modify any model weights. Instead, they learn "soft prompts" —
continuous vectors that are prepended to the input and guide the model's behavior.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   HARD PROMPT (what humans write):                                               │
    │                                                                                  │
    │       "Classify the sentiment of this review: This movie was great"              │
    │        ↓ tokenize                                                                │
    │       [518, 25580, ..., 2107]  ← fixed, discrete token IDs                       │
    │                                                                                  │
    │                                                                                  │
    │   SOFT PROMPT (what the model learns):                                           │
    │                                                                                  │
    │       [v₁, v₂, ..., v₂₀]  +  "This movie was great"                              │
    │        ↑                        ↑                                                │
    │        Learned continuous        Actual input                                    │
    │        vectors (trainable)       (normal tokens)                                 │
    │                                                                                  │
    │   These v₁...v₂₀ don't correspond to any real words.                             │
    │   They're free-floating vectors in embedding space that the model                │
    │   learns to interpret as task instructions.                                      │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘


### Prompt Tuning

Prompt Tuning (Lester et al. 2021) prepends k learned vectors to the input embedding:

    Input:  [soft_1, soft_2, ..., soft_k, token_1, token_2, ..., token_n]

    Only the soft tokens are trainable. The rest of the model is completely frozen.

    Trainable parameters: k × hidden_dim
    For k=20, hidden_dim=4096: just 81,920 parameters (~0.001% of a 7B model)

    Extremely efficient, but performance lags behind LoRA for smaller models.
    Scales better with model size — at 10B+ parameters, prompt tuning 
    approaches full fine-tuning quality.


### Prefix Tuning

Prefix Tuning (Li & Liang 2021) inserts learned vectors at every attention layer,
not just the input:

    Standard attention at each layer:
        Q, K, V from the input sequence

    Prefix Tuning:
        Q from input, K = [prefix_K; input_K], V = [prefix_V; input_V]
        Learned prefix vectors prepended to keys and values at EVERY layer

    More expressive than Prompt Tuning because it can influence attention
    patterns at every depth of the model, not just at the input.

    Trainable parameters: 2 × num_layers × prefix_length × hidden_dim
    For 32 layers, prefix_length=20, hidden_dim=4096: ~5.2M parameters


### P-Tuning v2

P-Tuning v2 (Liu et al. 2022) extends Prefix Tuning with deeper integration:
learned prompts are added across all layers with layer-specific parameters.

    Essentially Prefix Tuning + learned prompts at the input layer + 
    optional task-specific classification heads.

    Designed to close the gap between prompt-based methods and full fine-tuning
    for smaller models and NLU tasks.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   PROMPT METHODS COMPARISON                                                      │
    │                                                                                  │
    │   Method            Where Prompts Live         Params         Quality            │
    │   ──────            ──────────────────         ──────         ───────            │
    │   Prompt Tuning     Input layer only           ~0.001%        Good (at scale)    │
    │   Prefix Tuning     K, V at every layer        ~0.1%          Better             │
    │   P-Tuning v2       All layers + input         ~0.1-1%        Best (prompt-based)│
    │                                                                                  │
    │   All prompt methods: zero inference overhead if prompts are cached.             │
    │   But none can be "merged" like LoRA — the prompts must always be present.       │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                HYBRID METHODS
═══════════════════════════════════════════════════════════════════════════════════════════════


### LongLoRA — Efficient Attention for Long Contexts

LongLoRA (Chen et al. 2023) extends LoRA with shifted sparse attention (S²-Attn)
to efficiently fine-tune models for longer context windows:

    Standard attention:  O(n²) memory, where n = sequence length
    Shifted sparse attention: splits sequence into groups, applies attention
    within groups, then shifts groups to capture cross-group dependencies.

    This allows fine-tuning with 8K-100K+ context lengths on limited hardware.
    The LoRA adapters handle task adaptation while S²-Attn handles the
    extended context — each solving a different bottleneck.


### VeRA (Vector-based Random Matrix Adaptation)

VeRA (Kopiczko et al. 2024) takes parameter efficiency even further:

    LoRA:  ΔW = B × A           (B and A are trainable)
    VeRA:  ΔW = Λ_b × B × Λ_d × A    (B and A are FROZEN random matrices,
                                         Λ_b and Λ_d are trainable diagonal matrices)

    Instead of learning the projection matrices, VeRA shares frozen random
    projections across all layers and only learns per-layer scaling vectors.

    Result: 10× fewer parameters than LoRA with competitive performance.

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                         PEFT TRAINING — UNDER THE HOOD
                    (The Data Pipeline with LoRA / QLoRA)
═══════════════════════════════════════════════════════════════════════════════════════════════


### How PEFT Training Differs from Full Fine-Tuning

The data pipeline (JSONL → tokenize → pad → batch → tensors) is IDENTICAL to full fine-tuning.
The difference is entirely in what happens during the forward pass, backward pass, and weight update.


    ┌──────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                          │
    │   FULL FINE-TUNING                               PEFT (LoRA)                             │
    │                                                                                          │
    │   ┌────────────────────────────────┐             ┌────────────────────────────────┐      │
    │   │ FORWARD PASS                   │             │ FORWARD PASS                   │      │
    │   │                                │             │                                │      │
    │   │ All layers compute normally    │             │ Frozen layers: normal compute  │      │
    │   │                                │             │ + LoRA bypass: BAx added       │      │
    │   │ ✓ Same                         │             │ ✓ ~Same compute cost           │      │
    │   └────────────────┬───────────────┘             └────────────────┬───────────────┘      │
    │                    │                                              │                      │
    │   ┌────────────────▼───────────────┐             ┌────────────────▼───────────────┐      │
    │   │ LOSS COMPUTATION               │             │ LOSS COMPUTATION               │      │
    │   │                                │             │                                │      │
    │   │ Cross-entropy on output tokens │             │ Cross-entropy on output tokens │      │
    │   │                                │             │                                │      │
    │   │ ✓ Identical                    │             │ ✓ Identical                    │      │
    │   └────────────────┬───────────────┘             └────────────────┬───────────────┘      │
    │                    │                                              │                      │
    │   ┌────────────────▼───────────────┐             ┌────────────────▼───────────────┐      │
    │   │ BACKWARD PASS                  │             │ BACKWARD PASS                  │      │
    │   │                                │             │                                │      │
    │   │ Compute gradients for          │             │ Compute gradients ONLY for     │      │
    │   │ ALL 7B parameters              │             │ LoRA A and B matrices          │      │
    │   │                                │             │ (~8M parameters)               │      │
    │   │ ✗ 14 GB of gradients stored    │             │ ✓ ~20 MB of gradients stored   │      │
    │   └────────────────┬───────────────┘             └────────────────┬───────────────┘      │
    │                    │                                              │                      │
    │   ┌────────────────▼───────────────┐             ┌────────────────▼───────────────┐      │
    │   │ OPTIMIZER UPDATE               │             │ OPTIMIZER UPDATE               │      │
    │   │                                │             │                                │      │
    │   │ Adam updates ALL 7B params     │             │ Adam updates ONLY adapters     │      │
    │   │ Stores momentum + variance     │             │ Stores momentum + variance     │      │
    │   │ for every parameter            │             │ for ~8M params only            │      │
    │   │                                │             │                                │      │
    │   │ ✗ 56 GB optimizer states       │             │ ✓ ~80 MB optimizer states      │      │
    │   └────────────────────────────────┘             └────────────────────────────────┘      │
    │                                                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────┘


### What Requires Gradients — The requires_grad Flag

In PyTorch, every parameter has a `requires_grad` flag:

    # Full fine-tuning: ALL parameters have requires_grad=True
    for param in model.parameters():
        param.requires_grad = True     # ALL 7B parameters → compute gradients for all

    # PEFT / LoRA: ONLY adapter parameters have requires_grad=True
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True   # ~8M LoRA params → compute gradients
        else:
            param.requires_grad = False  # ~7B frozen params → skip gradient computation

    This single flag is what makes PEFT memory-efficient.
    PyTorch's autograd engine skips gradient computation and storage
    for any parameter with requires_grad=False.

---

### Learning Rate and Hyperparameters for PEFT

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   HYPERPARAMETER COMPARISON                                                      │
    │                                                                                  │
    │   Hyperparameter          Full Fine-Tuning         LoRA / QLoRA                  │
    │   ──────────────          ────────────────         ────────────                  │
    │   Learning rate           1e-6 to 5e-5             1e-4 to 3e-4    ★ higher      │
    │   Epochs                  1-5                      1-5             (similar)     │
    │   Batch size              32-128                   16-64           (similar)     │
    │   Warmup                  5-10% of steps           3-10% of steps                │
    │   Weight decay            0.01-0.1                 0.0-0.05       ★ lower        │
    │   Scheduler               Cosine decay             Cosine decay    (similar)     │
    │   Gradient accumulation   Often needed             Often needed    (similar)     │
    │                                                                                  │
    │   ★ LoRA can afford HIGHER learning rates because:                               │
    │     1. Only updating ~0.1% of params → less risk of catastrophic change          │
    │     2. Low-rank adapters are less prone to overshooting                          │
    │     3. Base model is frozen → built-in regularization                            │
    │                                                                                  │
    │   ★ LoRA needs LOWER weight decay because:                                       │
    │     1. Already heavily constrained by low-rank structure                         │
    │     2. Few parameters → less need for regularization                             │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                               PRACTICAL PEFT WORKFLOW
                          (End-to-End with HuggingFace PEFT)
═══════════════════════════════════════════════════════════════════════════════════════════════


### The Typical PEFT Pipeline

    ┌────────────────┐     ┌─────────────────┐     ┌────────────────────────────┐
    │  Load Base     │────▶│  Apply PEFT     │────▶│  Prepare Data              │
    │  Model         │     │  Config         │     │  (same as full fine-tuning)│
    │  (from HF Hub) │     │  (freeze base,  │     │  JSONL → tokenize → pad    │
    │                │     │   add adapters) │     │                            │
    └────────────────┘     └─────────────────┘     └─────────────┬──────────────┘
                                                                 │
                                                                 ▼
    ┌────────────────┐     ┌────────────────┐     ┌────────────────────────────┐
    │  Deploy        │◀────│  Save / Merge  │◀────│  Train                     │
    │                │     │  Adapters      │     │  (only adapters get        │
    │  • Merged model│     │                │     │   gradient updates)        │
    │  • Or base +   │     │  Option A:     │     │                            │
    │    swappable   │     │   Save adapter │     │  SFTTrainer / Trainer      │
    │    adapters    │     │   files (~MB)  │     │  handles the loop          │
    │                │     │  Option B:     │     │                            │
    │                │     │   Merge into   │     │                            │
    │                │     │   base model   │     │                            │
    └────────────────┘     └────────────────┘     └────────────────────────────┘


### What Gets Saved — Adapter Files

When you save a LoRA-trained model, you save ONLY the adapter:

    Full fine-tuning checkpoint:          LoRA adapter checkpoint:
    ─────────────────────────────         ─────────────────────────────
    model.safetensors      14 GB         adapter_model.safetensors   33 MB
    config.json                          adapter_config.json
    tokenizer files                      (base model referenced by name)

    To use the adapter later:
    1. Load the original base model (from HuggingFace Hub or local)
    2. Load the adapter on top
    3. Optionally merge for zero-overhead inference

    adapter_config.json contains:
    {
        "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "task_type": "CAUSAL_LM"
    }

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                         PEFT WITH ALIGNMENT — LoRA + DPO / RLHF
═══════════════════════════════════════════════════════════════════════════════════════════════


### Combining PEFT with Alignment Tuning

The most common production pipeline today uses PEFT at multiple stages:

    ┌───────────────┐     ┌──────────────────────────────┐     ┌───────────────────────┐
    │  Base Model   │────▶│  Stage 1: SFT with LoRA      │────▶│  Stage 2: DPO with    │
    │  (LLaMA,      │     │  Train on instruction pairs  │     │  LoRA                 │
    │   Mistral)    │     │  LoRA rank=16, α=32          │     │  Align to preferences │
    │               │     │  → save adapter_sft/         │     │  LoRA rank=8, α=16    │
    └───────────────┘     └──────────────────────────────┘     │  → save adapter_dpo/  │
                                                               └───────────┬───────────┘
                                                                           │
                                                                           ▼
                                                               ┌───────────────────────┐
                                                               │  Merge both adapters  │
                                                               │  into base model      │
                                                               │  → deploy             │
                                                               └───────────────────────┘

    This entire pipeline can run on a single 24GB GPU with QLoRA.
    That's remarkable — alignment-tuning a 7B model used to require a GPU cluster.

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                         ADAPTER MERGING — COMBINING MULTIPLE ADAPTERS
═══════════════════════════════════════════════════════════════════════════════════════════════


### Merging Multiple LoRA Adapters

When you have trained separate adapters for different capabilities, you can combine them:


**Linear Merge (simplest):**

    W_merged = W₀ + λ₁·B₁A₁ + λ₂·B₂A₂ + ...

    Where λ₁, λ₂ are weighting coefficients (how much of each adapter to blend).
    Simple weighted average. Works when tasks are related and adapters don't conflict.


**TIES Merging (Trim, Elect Sign & Merge):**

    Step 1: Trim — remove small-magnitude changes (noise)
    Step 2: Elect Sign — for each parameter, pick the majority sign direction
    Step 3: Merge — average only the values that agree on direction

    More robust than linear merge. Handles conflicting adapters better.


**DARE (Drop And REscale):**

    Randomly drop a fraction of adapter parameters (set to zero),
    then rescale the remaining ones to compensate.

    Combined with TIES or linear merge for better generalization.
    Think of it like dropout but applied to the adapter delta weights.


    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   ADAPTER MERGING EXAMPLE                                                        │
    │                                                                                  │
    │   Base LLaMA-7B                                                                  │
    │       + Medical QA adapter    (λ=0.5)                                            │
    │       + Code generation adapter (λ=0.3)                                          │
    │       + Summarization adapter (λ=0.2)                                            │
    │       ────────────────────────────────                                           │
    │       = Single merged model that can do all three                                │
    │         (quality depends on task compatibility)                                  │
    │                                                                                  │
    │   No guarantee of quality — conflicting tasks may degrade each other.            │
    │   Always evaluate merged models carefully.                                       │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                              WHEN TO USE WHICH PEFT METHOD
═══════════════════════════════════════════════════════════════════════════════════════════════


    ┌───────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                           │
    │   DECISION TREE — CHOOSING A PEFT METHOD                                                  │
    │                                                                                           │
    │   Start here: Do you have a GPU?                                                          │
    │       │                                                                                   │
    │       ├── No  →  Use API-based fine-tuning (OpenAI, Anthropic, etc.)                      │
    │       │                                                                                   │
    │       └── Yes →  How much VRAM?                                                           │
    │               │                                                                           │
    │               ├── 8-16 GB   → QLoRA (7B model, 4-bit)                                     │
    │               │               or Prompt Tuning if task is simple                          │
    │               │                                                                           │
    │               ├── 24 GB     → QLoRA (7B-13B) or LoRA (7B)                                 │
    │               │                                                                           │
    │               ├── 48 GB     → LoRA (7B-13B) or QLoRA (70B)                                │
    │               │                                                                           │
    │               └── 80+ GB   → LoRA (up to 70B) or Full FT (7B)                             │
    │                                                                                           │
    │                                                                                           │
    │   Model Size    Budget GPU        Best Method       Approx VRAM                           │
    │   ──────────    ──────────        ───────────       ───────────                           │
    │   7B            RTX 3090 (24GB)   QLoRA             ~10 GB                                │
    │   7B            A6000 (48GB)      LoRA              ~18 GB                                │
    │   13B           RTX 3090 (24GB)   QLoRA             ~16 GB                                │
    │   13B           A100 (80GB)       LoRA              ~30 GB                                │
    │   70B           A100 (80GB)       QLoRA             ~40 GB                                │
    │   70B           8× A100s          LoRA (FSDP)       ~20 GB/GPU                            │
    │                                                                                           │
    └───────────────────────────────────────────────────────────────────────────────────────────┘

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                    SUMMARY MENTAL MODEL
═══════════════════════════════════════════════════════════════════════════════════════════════


    FULL FINE-TUNING                              PEFT (LoRA / QLoRA)

    ┌───────────────────────┐                       ┌─────────────────────┐
    │ █████████████████████ │                       │ ░░░░░░░░░░░░░░░░░░░ │
    │ █████████████████████ │                       │ ░░░░░░░░░░░░░░░░░░░ │
    │ ██ ALL 7B weights  ██ │                       │ ░░ 7B weights ░░░░░ │
    │ ██ updated         ██ │                       │ ░░ FROZEN    ░░░░░░ │
    │ ██                 ██ │                       │ ░░░░░░░░░░░░░░░░░░░ │
    │ █████████████████████ │                       │ ░░░░░░░░░░░░░░░░░░░ │
    │ █████████████████████ │                       │ ░░ + ███ tiny ███░░ │
    │ █████████████████████ │                       │ ░░ + ███ adapters██ │
    └───────────────────────┘                       │ ░░ + ███ (~0.1%) ██ │
                                                    └─────────────────────┘

    Memory:    94-114 GB                          Memory:    8-24 GB
    Storage:   14 GB per task                     Storage:   33 MB per task
    Forgetting: High risk                         Forgetting: None (base frozen)
    Speed:     Slow                               Speed:     Fast
    Quality:   Maximum                            Quality:   95-99% of full FT
    Use when:  Unlimited compute                  Use when:  Everything else

---

### The Full PEFT Landscape — One Final View

    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                           │
    │   Method              Category            Params Trained    Memory    Quality       Best For              │
    │   ──────              ────────            ──────────────    ──────    ───────       ────────              │
    │   LoRA                Reparameterization   0.1-1%           Low       Very Good     General purpose   ★   │
    │   QLoRA               Hybrid               0.1-1%           V. Low    Very Good     Large models      ★   │
    │   DoRA                Reparameterization   0.1-1.1%         Low       Excellent     When +1-3% matters    │
    │   AdaLoRA             Reparameterization   ≤0.1-1%          Low       Very Good     Optimal rank alloc    │
    │   LoRA+               Reparameterization   0.1-1%           Low       Very Good     Faster convergence    │
    │   Bottleneck Adapt.   Additive             0.5-3%           Medium    Good          Legacy / research     │
    │   (IA)³               Selective            ~0.01%           V. Low    Moderate      Many adapters         │
    │   BitFit              Selective            ~0.05-0.1%       V. Low    Moderate      NLU tasks (w/ bias)   │
    │   Prefix Tuning       Prompt-based         ~0.1%            Lowest    Moderate      Task switching        │
    │   Prompt Tuning       Prompt-based         ~0.001%          Lowest    Moderate      Large models, few-shot│
    │   VeRA                Reparameterization   ~0.01%           V. Low    Good          Extreme efficiency    │
    │                                                                                                           │
    │   ★ = Recommended starting point for most practitioners                                                   │
    │                                                                                                           │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘

---

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect                     | Full Fine-Tuning        | LoRA                     | QLoRA                    |
|----------------------------|-------------------------|--------------------------|--------------------------|
| Trainable Parameters       | 100% (~7B)              | ~0.1-1% (~8M)            | ~0.1-1% (~8M)            |
| GPU Memory (7B model)      | 94-114 GB               | 16-24 GB                 | 8-12 GB                  |
| GPU Memory (70B model)     | 1+ TB (multi-GPU)       | 160-200 GB (multi-GPU)   | 36-48 GB (single GPU)    |
| Training Speed             | Slowest                 | Fast                     | ~30-50% slower than LoRA |
| Inference Overhead         | None                    | None (after merge)       | None (after merge)       |
| Checkpoint Size            | 14-140 GB               | 20-200 MB                | 20-200 MB                |
| Catastrophic Forgetting    | High risk               | Minimal                  | Minimal                  |
| Quality (vs Full FT)       | Baseline (100%)         | 95-99%                   | 94-99%                   |
| Multi-task Serving         | Separate models         | Swap adapters            | Swap adapters            |
| Key Hyperparameters        | lr, epochs, batch size  | + rank, alpha, targets   | + quant type, paged opt  |
| Typical Learning Rate      | 1e-6 to 5e-5            | 1e-4 to 3e-4             | 1e-4 to 3e-4             |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "LoRA Config Setup (HuggingFace PEFT)": {
        "description": "Standard LoRA configuration using the PEFT library",
        "runnable": False,
        "code": '''from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                                          # Rank — start here, increase if needed
    lora_alpha=16,                                # Scaling factor — typically 2× rank
    target_modules=[                              # Which weight matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj"    # Attention projections (standard)
    ],
    lora_dropout=0.05,                            # Dropout on adapter outputs
    bias="none",                                  # Don't train bias terms
    task_type=TaskType.CAUSAL_LM,                 # Causal language modeling
)

# Apply LoRA to a pre-loaded base model
model = get_peft_model(base_model, lora_config)

# Check trainable vs total parameters
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243
'''
    },

    "QLoRA Full Setup": {
        "description": "Complete QLoRA setup with 4-bit quantization and LoRA adapters",
        "runnable": False,
        "code": '''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",            # NormalFloat4 (better than INT4)
    bnb_4bit_compute_dtype="bfloat16",    # Compute in BF16 during forward pass
    bnb_4bit_use_double_quant=True,       # Double quantization (saves ~375 MB)
)

# Step 2: Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Step 3: Prepare for k-bit training (handles gradient checkpointing, layer norms)
model = prepare_model_for_kbit_training(model)

# Step 4: Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",          # Apply to all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
'''
    },

    "Training with SFTTrainer": {
        "description": "Training loop using TRL's SFTTrainer for instruction tuning with LoRA",
        "runnable": False,
        "code": '''from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl", split="train")

training_args = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,       # Effective batch size = 4 × 8 = 32
    learning_rate=2e-4,                  # Higher than full FT (LoRA can handle it)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,                           # Mixed precision
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    gradient_checkpointing=True,         # Save memory
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./lora-output/final")  # Saves ONLY the adapter files (~33 MB)
'''
    },

    "Merge and Deploy LoRA Adapter": {
        "description": "Merge LoRA adapter back into base model for zero-overhead inference",
        "runnable": False,
        "code": '''from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model (full precision for merging)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)

# Load adapter on top
model = PeftModel.from_pretrained(base_model, "./lora-output/final")

# Merge adapter into base weights: W_merged = W₀ + (α/r) · BA
model = model.merge_and_unload()

# Save the merged model (full size, no adapter overhead at inference)
model.save_pretrained("./merged-model")
# This is now a standard model — no PEFT dependency needed to load it
'''
    },

    "Swap LoRA Adapters at Runtime": {
        "description": "Load different LoRA adapters for different tasks on the same base model",
        "runnable": False,
        "code": '''from peft import PeftModel

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load first adapter
model = PeftModel.from_pretrained(base_model, "./adapters/medical-qa")

# Use for medical QA...
output = model.generate(medical_input)

# Swap to code generation adapter (base model stays in memory)
model.load_adapter("./adapters/code-gen", adapter_name="code")
model.set_adapter("code")

# Use for code generation...
output = model.generate(code_input)

# Swap back to medical
model.set_adapter("default")
'''
    },

    "Inspect LoRA Adapter Structure": {
        "description": "Examine what LoRA actually adds to the model",
        "runnable": False,
        "code": '''# After applying LoRA, inspect the model structure
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"TRAINABLE: {name:60s} shape={str(param.shape):20s} params={param.numel():>10,}")
    # else: frozen (not printed — there are 7 billion of these)

# Example output:
# TRAINABLE: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight  shape=torch.Size([8, 4096])    params=    32,768
# TRAINABLE: base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight  shape=torch.Size([4096, 8])    params=    32,768
# TRAINABLE: base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight  shape=torch.Size([8, 4096])    params=    32,768
# TRAINABLE: base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight  shape=torch.Size([4096, 8])    params=    32,768
# ... (repeated for all 32 layers × all target modules)
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    return {
        "theory": THEORY,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }