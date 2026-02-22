"""
Fine Tuning - PEFT (Parameter-Efficient Fine-Tuning) Detailed Breakdown
========================================================================

A comprehensive deep dive into PEFT methods — LoRA, QLoRA, Adapters,
Prompt Tuning, and more. Covers the math, the memory, the data flow,
and the practical trade-offs that make PEFT the dominant fine-tuning
paradigm for modern LLMs.
"""


import os
import re
import sys
import subprocess
from pathlib import Path

TOPIC_NAME = "Fine_Tuning_PEFT_LORA"

# ─────────────────────────────────────────────────────────────────────────────
# PATH TO THE PIPELINE SCRIPT
# topics/08_c_PEFT_Reparameterization_LORA.py
# Implementation/PEFT_LORA/peft_main.py
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / "Implementation" / "PEFT_LORA"
_MAIN_SCRIPT  = _SCRIPTS_DIR / "peft_main.py"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

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
    │ • Bottleneck     │ │ • LoRA  *         │ │ • BitFit             │ │ • QLoRA  *       │ │ • Prefix Tuning   │
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
        Weights             :     14 GB
        Gradients           :     14 GB     ← eliminated in PEFT
        Optimizer states    :     56 GB     ← eliminated in PEFT
        Activations         :  10-30 GB     ← reduced (fewer backward-pass paths)
        ─────────────────────────────────────────────────────────────────────────
        Total:               ~94-114 GB

    PEFT / LoRA (7B model, BF16 base + LoRA adapters):
        Frozen weights          :    14 GB   (forward pass only, no gradients/optimizer)
        LoRA adapter weights    :   ~20 MB  (trainable)
        LoRA gradients          :   ~20 MB
        LoRA optimizer states   :   ~80 MB
        Activations             :  4-10 GB   (reduced)
        ───────────────────────────────────────────────
        Total:                   ~16-24 GB

---

═══════════════════════════════════════════════════════════════════════════════════════════════
                                    LoRA — LOW-RANK ADAPTATION
                             (The Most Important PEFT Method to Understand)
═══════════════════════════════════════════════════════════════════════════════════════════════

### LoRA — The Core Idea

LoRA (Low-Rank Adaptation of Large Language Models, Hu et al. 2021) is built on one key insight:

**The weight changes during fine-tuning have low intrinsic rank.**

What does that mean? When you fine-tune a model, the difference between the original weights
and the fine-tuned weights (ΔW) can be well-approximated by a low-rank matrix — meaning it
can be decomposed into two much smaller matrices multiplied together.

Instead of learning a full ΔW (which is huge), LoRA learns two small matrices A and B
whose product approximates ΔW.

---

### The Problem LoRA Is Solving:

When you fine-tune a large model the traditional way (full fine-tuning), every weight in the model gets updated. 
For a single weight matrix W₀ of shape [4096 × 4096], the change you're learning — called ΔW — is itself a full [4096 × 4096] matrix. 

That's 16.7 million parameters just for one layer. 
Across a 7B parameter model, the gradients and optimizer states (Adam stores two states per parameter) 
push memory requirements to 94–114 GB. That's inaccessible for most people.

LoRA's starting insight is this: you don't need all that. 
Research showed that when you actually analyze what ΔW looks like after fine-tuning, it's almost always low-rank. 
Most of those 16.7 million changes are redundant — the real adaptation lives in a much smaller subspace. 
The matrix is fat, but the information inside it is slim.


---

### The Math — Step by Step

**In full fine-tuning:**

    For a weight matrix W₀ of shape [d_out × d_in] (e.g., [4096 × 4096]):

    W_new = W₀ + ΔW

    ΔW is also [4096 × 4096] = 16,777,216 parameters to learn. That's the problem.


**<u>In LoRA: The Core Idea:</u>**

**Decompose ΔW Into Two Tiny Matrices**
Instead of learning the full ΔW, LoRA approximates it as the product of two small matrices:
    
    When you fine-tune a neural network layer, you normally update its weight matrix:
    
        W_new = W_original + ΔW   
    
    Where: 
        W_original  → pretrained weights (frozen in LoRA)
        ΔW          → the learned update during fine-tuning

    Instead of learning the full ΔW, decompose it:

        ΔW ≈ B × A

    Where:
        ΔW   =  Change in the original weight matrix
        A    =  [r × d_in]    →  e.g., [8 × 4096] = 32,768 parameters
        B    =  [d_out × r]   →  e.g., [4096 × 8] = 32,768 parameters
        
    In LoRA, we do not learn a full ΔW matrix directly.
    
    Instead we parameterize it as:
        
        ΔW = BA
    
    So fine-tuning becomes:
        
        W_new = W_original + B × A
 
 
    For the same [4096 × 4096] example with r=8:

        A   :   [8 × 4096]  =   32,768 params
        B   :   [4096 × 8]  =   32,768 params
        
        Total: 65,536 vs. 16,777,216 — a 256× reduction for one weight matrix.
        
    
    Total LoRA parameters :  32,768 + 32,768 = 65,536
    vs. full ΔW           :  16,777,216

    That's a 256× reduction for this single weight matrix.

    r is the "rank" — the key hyperparameter. Typical values: 4, 8, 16, 32, 64.
    
    
    The big question is: **why does this work ?** 
    
    Because the hypothesis (backed empirically) is that the "task adaptation" signal can be captured in just a few dimensions of change. 
    You don't need a full-rank matrix to express "now be a medical assistant instead of a general assistant."
    
---

### The Forward Pass

During a forward pass through any targeted layer, the computation becomes:

    h = W₀x + (α/r) · BAx
    
    - W₀x — the frozen original weight doing its job, unchanged
    
    - BAx — the small adapter's contribution, running in parallel
    
    - α/r — a scaling factor (more on this below)
    
Both paths process the same input x simultaneously. Their outputs are just added together. 
This is the key structural difference from Adapters (which insert modules in series, adding latency). 
LoRA runs in parallel — and can be merged away completely at inference.

---

**The forward pass becomes:**

    h = W₀x + (B × A)x      (where x is the input)
    
      = W₀x + BAx           (just matrix multiplication)

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                 │
    │                          LoRA FORWARD PASS                                      │                          
    │                    ┌──────────────────────┐                                     │
    │         x ────────▶│   W₀ (FROZEN)        │────────────┐                        │
    │         │          │   [4096 × 4096]      │            │                        │
    │         │          │   (original weights) │            │                        │
    │         │          └──────────────────────┘            │                        │
    │         │                                              ▼                        │
    │         │                                          ┌────────┐                   │
    │         │                                          │   +    │──────▶ h (output) │
    │         │                                          └────────┘                   │
    │         │          ┌───────────┐  ┌────────────┐       ▲                        │
    │         └─────────▶│  A        │─▶│  B         │───────┘                        │
    │                    │ [r × 4096]│  │ [4096 × r] │                                │
    │                    │ (DOWN     │  │ (UP        │   × (α/r) scaling              │
    │                    │  project) │  │  project)  │                                │
    │                    │ TRAINABLE │  │ TRAINABLE  │                                │
    │                    └───────────┘  └────────────┘                                │
    │                                                                                 │
    │    x: input tensor                                                              │
    │    h = W₀x + (α/r) · BAx                                                        │
    └─────────────────────────────────────────────────────────────────────────────────┘

---

### Initialization — Why It Matters

LoRA uses a very specific initialization strategy:

    Matrix A: initialized with random Gaussian (small random values)
    Matrix B: initialized to ALL ZEROS

    Why zeros for B?

    At the start of training: ΔW = B × A = 0 × A = 0

    This means the model starts EXACTLY where the pre-trained model left off.
    The LoRA adapters add zero contribution initially, so training begins from
    a known-good state — the pre-trained model's behavior is perfectly preserved
    at step 0.

    As training progresses, B learns non-zero values and the adapters
    gradually steer the model toward the new task.

    If both A and B were randomly initialized, the model would start with
    random perturbations to every adapted layer — potentially destroying
    pre-trained knowledge immediately.

---

### The Scaling Factor α (Alpha)

Alpha (α) is a scaling constant that acts like a volume knob on the adapter's contribution. The actual scaling applied is α/r:

LoRA includes a scaling factor α (alpha) that controls how much the adapters
contribute to the output:

    h = W₀x + (α/r) · BAx

    α is a constant (set before training, typical values: 8, 16, 32)
    r is the rank

    The ratio α/r controls the "magnitude" of the adaptation.

    Think of it like a volume knob:
        α/r too low  →  adapters contribute too little, model barely adapts
        α/r too high →  adapters dominate, can destabilize training

    Common practice:
        Set α = r      →  effective scaling of 1.0 (neutral)
        Set α = 2r     →  effective scaling of 2.0 (stronger adaptation)
        Set α = r/2    →  effective scaling of 0.5 (gentler adaptation)

    Rule of thumb: α = r or α = 2r works well for most tasks.
    When you increase r, increase α proportionally to maintain similar scaling.

The reason α exists separately from r is so you can change r without needing to recalibrate the magnitude of updates.
You fix α and tune r.

---

### Which Layers Get LoRA? — Target Modules

**<u>Which Layers to Target</u>**

You choose. The original LoRA paper only targeted W_q and W_v in the attention blocks. Modern practice has evolved:

    Minimal     :   W_q, W_v only
    Standard    :   W_q, W_k, W_v, W_o (all attention projections)
    Aggressive  :   all linear layers including the MLP's gate, up, and down projections

For a 32-layer model with standard 4 targets and r=8: that's 256 small matrices, 
~8.4 million trainable parameters — roughly 0.12% of the 7B model. 

Yet performance on fine-tuning benchmarks is often competitive with full fine-tuning.

---

You don't have to apply LoRA to every weight matrix. In a transformer, the key weight matrices are:

    ┌──────────────────────────────────────────────────────────────────────────────────────────┐
    │                          TRANSFORMER LAYER — Weight Matrices                             │
    │                                                                                          │
    │   ┌─────────────────────────────────────────────────────────┐                            │
    │   │              SELF-ATTENTION BLOCK                       │                            │
    │   │                                                         │                            │
    │   │   W_q  (Query projection)      [hidden × hidden]        │                            │
    │   │   W_k  (Key projection)        [hidden × hidden]        │                            │
    │   │   W_v  (Value projection)      [hidden × hidden]        │                            │
    │   │   W_o  (Output projection)     [hidden × hidden]        │                            │
    │   │                                                         │                            │
    │   └─────────────────────────────────────────────────────────┘                            │
    │                                                                                          │
    │   ┌─────────────────────────────────────────────────────────┐                            │
    │   │              FEED-FORWARD NETWORK (MLP)                 │                            │
    │   │                                                         │                            │
    │   │   W_gate (Gate projection)     [hidden × intermediate]  │                            │
    │   │   W_up   (Up projection)       [hidden × intermediate]  │                            │
    │   │   W_down (Down projection)     [intermediate × hidden]  │                            │
    │   │                                                         │                            │
    │   └─────────────────────────────────────────────────────────┘                            │
    │                                                                                          │
    │   ┌─────────────────────────────────────────────────────────┐                            │
    │   │              OTHER                                      │                            │
    │   │   Embedding layer              [vocab × hidden]         │  ← rarely adapted          │
    │   │   LM head                      [hidden × vocab]         │  ← rarely adapted          │
    │   └─────────────────────────────────────────────────────────┘                            │
    │                                                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────┘

**Common LoRA target configurations:**

    Minimal (original paper)        :       W_q, W_v  only
    Standard (most popular)         :       W_q, W_k, W_v, W_o
    Aggressive (best performance)   :       W_q, W_k, W_v, W_o, W_gate, W_up, W_down  (all linear layers)

    More targets = more trainable parameters = better performance but more memory.

    In HuggingFace PEFT config, this looks like:
    
        target_modules  =   ["q_proj", "v_proj"]                       # minimal
        target_modules  =   ["q_proj", "k_proj", "v_proj", "o_proj"]   # standard
        target_modules  =   "all-linear"                               # aggressive

    Each target module in EACH transformer layer gets its own A and B matrices.
    For a 32-layer model with 4 attention targets and rank 8:
        32 layers × 4 targets × 2 matrices (A, B) × (8 × 4096) = ~8.4M parameters
        vs. 7B total = ~0.12% of the model

---

### The Rank (r) — How to Choose It

    r = 1:    Extremely compressed. Each adapter captures only 1 direction of variation.
              Works for very simple tasks or when data is tiny.

    r = 4:    Very lean. Good for straightforward classification or single-skill tasks - simple tasks

    r = 8:    The default sweet spot. Works well for most instruction-tuning and chat tasks.
              This is where most practitioners start.
              Works for most instruction-tuning.

    r = 16:   Better for complex tasks requiring nuanced adaptation. 
              Moderate memory increase.
              More expressive, complex behavioral shifts

    r = 32:   Approaching diminishing returns for many tasks.
              Good for very different target domains (e.g., English model → code model).
              More expressive, complex behavioral shifts

    r = 64+:  Rarely needed. If you need this much capacity, consider whether
              full fine-tuning or a larger rank is actually buying you anything.
              Approaching full fine-tuning territory, diminishing returns

    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                    │
    │    RANK vs. PERFORMANCE vs. MEMORY  (for a 7B model, 4 attention targets)          │
    │                                                                                    │
    │    Rank    Trainable Params     % of Model     Adapter Size     Quality            │
    │    ────    ────────────────     ──────────     ────────────     ───────            │
    │      4       ~4.2M                0.06%          ~16 MB         Good               │
    │      8       ~8.4M                0.12%          ~33 MB         Very Good   *      │
    │     16      ~16.8M                0.24%          ~67 MB         Excellent          │
    │     32      ~33.5M                0.48%         ~134 MB         Excellent          │
    │     64      ~67.1M                0.96%         ~268 MB         Marginal gain      │
    │                                                                                    │
    │    * = recommended starting point                                                  │
    │                                                                                    │
    │    Note: these numbers assume LoRA on q_proj, k_proj, v_proj, o_proj only.         │
    │    Adding MLP targets roughly doubles the parameter count.                         │
    │                                                                                    │
    └────────────────────────────────────────────────────────────────────────────────────┘

---

### LoRA Merge — Deploying Without Overhead

A beautiful property of LoRA: after training, you can **merge** the adapters back into the base model.

    W_merged = W₀ + (α/r) · B × A

    This is just matrix addition. Once merged:
    - No inference overhead (no extra computation at runtime)
    - No adapter files needed
    - Model behaves exactly as if it had been fully fine-tuned
    - The merged model is the same size and architecture as the original

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   DURING TRAINING:                              AFTER MERGE:                         │
    │                                                                                      │
    │   ┌──────────────┐   ┌───────┐                 ┌──────────────┐                      │
    │   │   W₀         │ + │ B × A │   ──merge──▶    │  W_merged    │                      │
    │   │   (frozen)   │   │(adapt)│                 │  (single     │                      │
    │   │   14 GB      │   │ 33 MB │                 │   matrix)    │                      │
    │   └──────────────┘   └───────┘                 │  14 GB       │                      │
    │                                                │              │                      │
    │   Two-path forward pass                        │  Standard    │                      │
    │   (slightly slower)                            │  forward     │                      │
    │                                                │  pass        │                      │
    │                                                │  (no extra   │                      │
    │                                                │   cost)      │                      │
    │                                                └──────────────┘                      │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

    BUT — once merged, you can't "un-merge." If you want to swap tasks,
    keep the adapters separate and swap them at inference time (see below).

---

### LoRA Adapter Swapping — Multi-Task Serving

This is one of LoRA's killer features: one base model, many tasks.

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │                        ONE BASE MODEL, MANY LoRA ADAPTERS                            │
    │                                                                                      │
    │                          ┌───────────────────────┐                                   │
    │                          │   BASE MODEL          │                                   │
    │                          │   (LLaMA-7B)          │                                   │
    │                          │   14 GB (loaded once) │                                   │
    │                          └──────────┬────────────┘                                   │
    │                                     │                                                │
    │                ┌────────────────────┼────────────────────────┐                       │
    │                │                    │                        │                       │
    │                ▼                    ▼                        ▼                       │
    │         ┌──────────────┐      ┌──────────────┐        ┌──────────────┐               │
    │         │ LoRA Adapter │      │ LoRA Adapter │        │ LoRA Adapter │   ...         │
    │         │ Medical QA   │      │ Code Gen     │        │ Legal Summ.  │               │
    │         │ ~33 MB       │      │ ~33 MB       │        │ ~33 MB       │               │
    │         └──────────────┘      └──────────────┘        └──────────────┘               │
    │                                                                                      │
    │   Full fine-tuning: 3 tasks = 3 × 14 GB = 42 GB of model checkpoints                 │
    │   LoRA:             3 tasks = 14 GB base + 3 × 33 MB = ~14.1 GB total                │
    │                                                                                      │
    │   Swap adapters at runtime in milliseconds — no model reloading!                     │
    │   Frameworks like vLLM and LoRAX can serve multiple adapters simultaneously.         │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

---

**Step 0: Understand What Problem LoRA Is Solving**

In full fine-tuning, when you update a weight matrix W₀ of shape [4096 × 4096], you're learning a completely new matrix:

    W_new = W₀ + ΔW

    ΔW is [4096 × 4096] = 16,777,216 parameters to learn for this ONE matrix.

Researchers discovered that ΔW (the change needed to adapt a model to a new task) has low intrinsic rank. 
That means most of those 16.7 million changes are redundant — the real adaptation lives in a much smaller subspace.

LoRA exploits this by decomposing ΔW into two small matrices whose product approximates the full update:

    ΔW ≈ B × A

    A: [r × 4096]     (down-projection, compresses)
    B: [4096 × r]     (up-projection, expands)

    With r=8:
    A: [8 × 4096]  =  32,768 params
    B: [4096 × 8]  =  32,768 params
    ═══════════════════════════════
    Total:            65,536 params

    vs. full ΔW:      16,777,216 params

    That's a 256× reduction for ONE weight matrix.

Now let's walk through every step of how this actually gets implemented and trained.

---

Step 1: Load the Pre-Trained Model and Freeze Everything

You start with a pre-trained model — say LLaMA-2-7B. Every parameter gets frozen:

    model = load_pretrained("meta-llama/Llama-2-7b-hf")

    for param in model.parameters():
        param.requires_grad = False     # All 7 billion parameters → read-only

At this point the model is in inference mode. No gradients will be computed for any original weight. 
This is identical to what Additive PEFT does — the difference comes in Step 2.

Load the pre-trained model. Set requires_grad = False on every single parameter. 
The base model becomes read-only. No gradients will flow through it. 
This is what eliminates 56 GB of optimizer states and 14 GB of gradients.

---

**Step 2: Attach LoRA Matrices (A and B) to Target Weight Matrices**

Here's where LoRA diverges from Adapters. Instead of inserting new modules between layers (in-series), 
LoRA attaches a small parallel bypass alongside existing weight matrices.

For each target weight matrix W₀ in the model, LoRA creates two new small matrices A and B:

    Target: W_q (query projection) in Layer 0
            Shape: [4096 × 4096]
            Status: FROZEN (requires_grad = False)

    LoRA creates:
            A: [r × 4096]  = [8 × 4096]    → requires_grad = True (TRAINABLE)
            B: [4096 × r]  = [4096 × 8]    → requires_grad = True (TRAINABLE)

Which weight matrices get LoRA ? You choose. 

This is the target_modules configuration:

    Minimal:      W_q, W_v                                  (original paper)
    Standard:     W_q, W_k, W_v, W_o                        (most popular)
    Aggressive:   W_q, W_k, W_v, W_o, W_gate, W_up, W_down  (all linear layers)

For a 32-layer model with the standard 4 attention targets and rank 8:

    32 layers × 4 targets × 2 matrices (A + B) = 256 small matrices

    Total trainable params:
        256 matrices × (8 × 4096) = ~8.4 million

    vs. 7 billion total → 0.12% of the model


For each weight matrix you choose to adapt (e.g., W_q, W_k, W_v, W_o in every attention block), 
create two new small matrices A and B. These are the only trainable parameters. 
They sit alongside W₀ as a parallel bypass.

**Step 3: Initialize A and B — This Is Critical**

Initialize Carefully (This Is Critical)

    A → initialized with small random Gaussian values
    
    B → initialized to all zeros


LoRA uses a very specific initialization:

    Matrix A:  Random Gaussian (small random values, like normal init)

    Matrix B:  ALL ZEROS

Why does B start at zero? Because at training step 0:

    ΔW = B × A = 0 × A = 0     (zero matrix)

    So: W_effective = W₀ + ΔW = W₀ + 0 = W₀

A is initialized with small random Gaussian values. B is initialized to all zeros.
This ensures ΔW = B × A = 0 at step 0, so the model starts exactly as the pre-trained checkpoint
with no disruption to pre-trained knowledge. See Diagram 3 below for the full visual breakdown.

Explanation 2:
The model starts exactly as the pre-trained model. The LoRA adapters contribute nothing initially. 
Training begins from a known-good state — the pre-trained model's behavior is perfectly preserved at step 0.
If both A and B were randomly initialized, B × A would produce random noise added to every weight matrix, 
potentially destroying the pre-trained knowledge immediately.
As training progresses, B learns non-zero values and the product B × A gradually steers the model toward the new task.

Explanation 3:
At training step zero: ΔW = B × A = 0 × A = 0. So the effective weight is exactly W₀ + 0 = W₀. 
The model starts identically to the pre-trained checkpoint. No pre-trained knowledge is disrupted. 
As training progresses, B learns non-zero values and the adapters gradually steer behavior. 
If both A and B were randomly initialized, you'd be injecting random noise into every layer from the start — potentially catastrophic.



---

**Step 4: Prepare the Data (Identical to Full Fine-Tuning)**

The data pipeline is completely unchanged from full fine-tuning. 
LoRA doesn't change what goes into the model, only what happens inside it.

4a. Raw data on disk:

    train.jsonl:

    {"instruction": "Classify the sentiment", "input": "This movie was breathtaking", "output": "positive"}
    {"instruction": "Classify the sentiment", "input": "Terrible waste of time", "output": "negative"}
    {"instruction": "Translate to French", "input": "The cat sat on the mat", "output": "Le chat était assis sur le tapis"}

    Just text. Nothing has happened yet.

4b. Template formatting (text → structured text):

    The DataLoader applies a chat template to combine the fields. This is model-specific:

    LLaMA format:
    "<s>[INST] Classify the sentiment: This movie was breathtaking [/INST] positive</s>"

    ChatML format:
    "<|im_start|>user\nClassify the sentiment: This movie was breathtaking<|im_end|>\n<|im_start|>assistant\npositive<|im_end|>"

    Still just text strings. 
    Using the wrong template degrades performance because the model was pre-trained expecting a specific format.

4c. Tokenization (text → integer IDs):

    "<s>[INST] Classify the sentiment: This movie was breathtaking [/INST] positive</s>"
                ↓ tokenizer
    [1, 518, 25580, 29962, 4134, 1598, 278, 19688, 29901, 910, 14064, 471, 4800, 28107, 518, 29914, 25580, 29962, 6374, 2]

    That's 20 integer IDs. No vectors yet.

4d. Create labels and loss mask:

    Input IDs: [1,     518,  25580, 29962, 4134, 1598, 278,  19688, 29901, 910,  14064, 471,  4800, 28107, 518,  29914, 25580, 29962, 6374, 2]

    Labels:    [-100, -100, -100,   -100,  -100, -100, -100, -100,  -100,  -100, -100,  -100, -100, -100,  -100, -100,  -100,  -100,  6374, 2]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                instruction/input tokens: IGNORED by loss                                                                     output: GRADED


    -100 tells PyTorch's CrossEntropyLoss to skip that position. 
    The model sees the full sequence during forward pass but only gets graded on the response tokens ("positive").

4e. Padding and attention mask:

    Different examples have different lengths. GPUs need rectangular tensors, so shorter examples get padded:

    Before padding:
    Example 1: [1, 518, 25580, ..., 6374, 2]               → 20 tokens
    Example 2: [1, 518, 25580, ..., 11690, 2]              → 29 tokens (longest)

    After padding to length 29:
    Example 1: [1, 518, 25580, ..., 6374, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]      → 29 tokens
    Example 2: [1, 518, 25580, ..., 11690, 2]                                → 29 tokens

    Attention mask (1 = real, 0 = padding):
    Example 1: [1, 1, 1, ..., 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Example 2: [1, 1, 1, ..., 1, 1]

4f. Collate into tensors and move to GPU:

    batch = 
    {
        "input_ids":      tensor  [batch_size, seq_len]      e.g. [4, 29]
        "attention_mask":  tensor [batch_size, seq_len]      e.g. [4, 29]
        "labels":         tensor  [batch_size, seq_len]      e.g. [4, 29]
    }

    This entire data pipeline (4a through 4f) is identical for full fine-tuning, LoRA, QLoRA, Adapters — all of them. 
    The data doesn't know or care what training method you're using. The difference is entirely in what happens next.

---

**Step 5: Forward Pass — Where LoRA Changes Things**

    This is where LoRA diverges. In full fine-tuning, the forward pass through a weight matrix is:

        h = W₀ × x          (standard matrix multiplication)

    With LoRA, every target weight matrix gets an additional parallel path:

    h = W₀x + (α/r) · B(Ax)
    ────   ──────────────
     │           │
     │           └── LoRA bypass: x goes through A (compress), then B (expand)
     │                scaled by α/r
     │
     └── Original frozen path: unchanged

     Let's trace one token through Layer 0's self-attention with LoRA on W_q, W_k, W_v, W_o:

     Input: x = [4096] (one token's hidden state)

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  QUERY COMPUTATION (W_q has LoRA)                                       │
    │                                                                         │
    │  Frozen path:   q_frozen = W_q × x                    [4096 × 4096]     │
    │                                                       → [4096]          │
    │                                                                         │
    │  LoRA path:     x_down   = A_q × x                    [8 × 4096]        │
    │                                                       → [8]             │
    │                                                                         │
    │                 x_up     = B_q × x_down               [4096 × 8]        │
    │                                                       → [4096]          │
    │                                                                         │   
    │                 q_lora   = (α/r) × x_up               (scaling)         │
    │                                                                         │
    │  Combined:      q = q_frozen + q_lora                 [4096]            │
    │                   = W_q·x + (α/r)·B_q·A_q·x                             │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  KEY COMPUTATION (W_k has LoRA) — same pattern                          │
    │                                                                         │
    │  k = W_k·x + (α/r)·B_k·A_k·x                                            │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  VALUE COMPUTATION (W_v has LoRA) — same pattern                        │
    │                                                                         │
    │  v = W_v·x + (α/r)·B_v·A_v·x                                            │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  ATTENTION: softmax(Q·K^T / √d) · V  →  attention output                │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  OUTPUT PROJECTION (W_o has LoRA)                                       │
    │                                                                         │
    │  o = W_o·attn + (α/r)·B_o·A_o·attn                                      │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
                  Add & LayerNorm
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  FEED-FORWARD MLP                     │
        │                                       │
        │  If target_modules="all-linear":      │
        │    Each MLP weight ALSO gets LoRA     │
        │                                       │
        │  If targeting attention only:         │
        │    MLP weights are frozen, no LoRA    │
        │    Standard: h = W_down(act(W_gate·x  │
        │                    * W_up·x))         │
        └───────────────────────────────────────┘
                        │
                        ▼
                  Add & LayerNorm
                        │
                        ▼
                  → Layer 1 (same process)
                  → Layer 2 ...
                  → Layer 31
                        │
                        ▼
                  LM Head (FROZEN, no LoRA typically)
                        │
                        ▼
                  logits: [batch_size, seq_len, vocab_size]

---

**Step 6: Loss Computation (Identical to Full Fine-Tuning)**

    logits = model output: [batch_size, seq_len, 32000]    (32000 = vocab size)
    labels = [-100, -100, ..., -100, 6374, 2]              (-100 = ignore)

    loss = CrossEntropyLoss(logits, labels)

    Only positions where labels ≠ -100 contribute.
    The model is graded ONLY on predicting the response tokens.

Nothing changes here between LoRA and full fine-tuning. Same loss function, same masking.

---

**Step 7: Backward Pass — Where the Memory Savings Happen**

This is the key step. loss.backward() triggers backpropagation.

In full fine-tuning:
    Gradients computed for ALL 7B parameters:
    ∂Loss/∂W_q      [4096 × 4096]   → stored (67 MB per matrix)
    ∂Loss/∂W_k      [4096 × 4096]   → stored
    ∂Loss/∂W_v      [4096 × 4096]   → stored
    ∂Loss/∂W_o      [4096 × 4096]   → stored
    ∂Loss/∂W_gate   [4096 × 11008]  → stored
    ∂Loss/∂W_up     [4096 × 11008]  → stored
    ∂Loss/∂W_down   [11008 × 4096]  → stored

    ... × 32 layers = ~14 GB of gradient storage


In LoRA:

    Frozen weights (W_q, W_k, etc.):
        Gradients flow THROUGH them (needed for chain rule to reach LoRA matrices)
        But gradients are NOT STORED (requires_grad = False)
        NO memory allocated for their gradients

    LoRA matrices (A and B):
        ∂Loss/∂A_q     [8 × 4096]    → stored (131 KB per matrix)
        ∂Loss/∂B_q     [4096 × 8]    → stored (131 KB per matrix)
        ∂Loss/∂A_k     [8 × 4096]    → stored
        ∂Loss/∂B_k     [4096 × 8]    → stored
        ... × 4 targets × 32 layers  = ~33 MB of gradient storage

        vs. 14 GB in full fine-tuning → 400× less gradient memory

    The chain rule detail matters here. 
    When PyTorch computes gradients for the LoRA matrices, it needs to know how the loss changed with respect to the 
    input of each frozen layer (so it can propagate back further). This means gradients do flow through the 
    frozen weights — the computation happens — but the gradient tensors for the frozen weights themselves are never allocated or stored. 
    The frozen weights are treated as constants in the computation graph.


The data pipeline, tokenization, loss masking, and training loop are identical to full fine-tuning. 
The only difference is which parameters receive gradient updates.
Gradients flow back through the output, through B, through A — and stop there. W₀ never accumulates a gradient.

---

**Step 8: Optimizer Update — Only Adapters Move**

    optimizer = AdamW(model.parameters(), lr=2e-4)   
    # BUT only params with requires_grad=True are in the optimizer

    optimizer.step():
        For each LoRA parameter (A and B matrices):
            momentum[param]  = β₁ × momentum  + (1-β₁) × gradient
            variance[param]  = β₂ × variance  + (1-β₂) × gradient²
            param           -= lr × momentum / (√variance + ε)

        Adam stores 2 extra values per trainable parameter:
            8.4M LoRA params × 2 × 4 bytes (FP32) = ~67 MB optimizer states

        vs. full fine-tuning:
            7B params × 2 × 4 bytes = ~56 GB optimizer states

        That's an 800× reduction in optimizer memory.

    Frozen parameters:
        No gradient → no momentum → no variance → no update
        They are literally unchanged from the pre-trained values

---

**Step 9: Repeat the Loop**

    for epoch in range(num_epochs):           # typically 1-5 epochs
    for batch in dataloader:
        outputs = model(batch)            # forward (frozen + LoRA)
        loss = compute_loss(outputs)      # cross-entropy
        loss.backward()                   # gradients for LoRA only
        optimizer.step()                  # update LoRA only
        optimizer.zero_grad()             # clear gradients

    After enough batches and epochs, the A and B matrices have learned the task-specific adjustments. 
    The frozen base model hasn't changed at all.

---

**Step 10: Save — Only the Adapter**

    After training, you save ONLY the LoRA matrices:

    Saved files:
    adapter_model.safetensors     ~33 MB    (all A and B matrices)
    adapter_config.json           ~1 KB     (rank, alpha, targets, base model name)

    Compare to full fine-tuning:
        model.safetensors             ~14 GB    (entire model)


    The adapter_config.json looks like:

    {
        "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }

    To use this adapter later, you load the base model (from HuggingFace Hub) and load the adapter on top. 
    The base model is never duplicated — you just store the tiny delta.

---

**Step 11: Deploy — Merge or Swap**

You have two deployment options:

    Option A: Merge the adapter into the base model

        W_merged = W₀ + (α/r) · B × A

        This is just matrix addition. One-time computation.

        After merging:
            - No adapter files needed at runtime
            - No extra computation during inference
            - Model behaves as if it was fully fine-tuned
            - Same size, same architecture, same speed as original
            - But you can't "un-merge" — the adapter is baked in

    Option B: Keep adapters separate and swap them

        Load base model once (14 GB in GPU memory)
        Load medical adapter  (~33 MB) → do medical QA
        Swap to code adapter  (~33 MB) → do code generation
        Swap to legal adapter (~33 MB) → do legal summarization

        One base model, many tasks. Swap in milliseconds.

        Full fine-tuning equivalent: 3 × 14 GB = 42 GB of checkpoints
        LoRA equivalent:            14 GB base + 3 × 33 MB = ~14.1 GB total


    The Scaling Factor α — What It Actually Does

    Every forward pass through a LoRA target computes:

    h = W₀x + (α/r) · BAx

    The α/r ratio is a volume knob controlling how much the adapters contribute:

    α = 16, r = 8  →  α/r = 2.0   (adapters contribute 2× their raw output)
    α = 8,  r = 8  →  α/r = 1.0   (neutral — adapters contribute as-is)
    α = 8,  r = 16 →  α/r = 0.5   (adapters contribute half their raw output)

    Why not just use the learning rate for this? Because α is a structural scaling applied at every forward pass, 
    while learning rate controls how fast the optimizer updates. 
    They serve different purposes. α controls the magnitude of the adaptation signal, 
    learning rate controls the magnitude of the gradient step.

    Common practice: set α = r (neutral scaling of 1.0) or α = 2r (slightly amplified). 
    When you increase rank, increase α proportionally to keep the effective scaling stable.

---

The Actual Data at Each Stage — Concrete Numbers

    Let me trace a single training step with concrete shapes for a 7B model, batch_size=4, seq_len=512, rank=8, 4 attention targets:

    STAGE                               SHAPE                               SIZE IN MEMORY
    ─────                               ─────                               ──────────────

    Raw JSONL                           text strings                        ~few KB per example

    After tokenize                      [4, 512] int32                      ~8 KB

    After embed (frozen)                [4, 512, 4096] BF16                 ~16 MB

    At each attention layer:
      W_q frozen forward                [4096, 4096] × [4, 512, 4096]       → [4, 512, 4096]
      A_q LoRA down                     [8, 4096] × [4, 512, 4096]          → [4, 512, 8]
      B_q LoRA up                       [4096, 8] × [4, 512, 8]             → [4, 512, 4096]
      Sum both paths                    [4, 512, 4096]                      (element-wise add)
      (same for K, V, O)

    After all 32 layers                 [4, 512, 4096] BF16                 ~16 MB

    After LM Head (frozen)              [4, 512, 32000] BF16                ~128 MB (logits)

    Loss                                scalar                              4 bytes

    Gradients (LoRA only):
      Per A matrix                      [8, 4096] BF16                      ~64 KB
      Per B matrix                      [4096, 8] BF16                      ~64 KB
      Total: 256 matrices                                                   ~33 MB

    Optimizer states (LoRA only):
      Momentum + variance               per LoRA param, FP32                ~67 MB

Total VRAM for this training step: ~16-24 GB (frozen weights + activations + LoRA overhead).

---

What Makes LoRA Different From Every Other PEFT Method — The One Key Property

After training, you can compute:

    W_merged = W₀ + (α/r) · B × A

This is a permanent merge. The A and B matrices disappear. 
The merged weight matrix is the same shape as the original. 
The model architecture is completely unchanged from the pre-trained version.

At inference time, there is zero additional computation, zero additional memory, zero additional latency. 
The model doesn't know it was LoRA-trained. It's just a standard model with different weight values.

No other PEFT method has this property. Adapters stay in the forward path permanently. 
Prompt-based methods need their soft tokens prepended to every input forever. 
(IA)³ needs its rescaling vectors applied at every forward pass.

LoRA's mergeability is why it became the dominant PEFT method and why it's the default recommendation 
for almost every fine-tuning scenario today.


Merge and Disappear (At Inference)

After training, you can compute:
        
    W_merged = W₀ + (α/r) · B × A
    
This produces a single weight matrix, identical in shape to the original W₀. 
You discard A and B.
The merged model has zero inference overhead — it's structurally identical to the base model, 
just with slightly different weights. No PEFT library needed at inference.

---

LORA Diagram 

                        ================================================================
                         LoRA (Low-Rank Adaptation) — Complete Visual Diagram Breakdown
                        ================================================================

Textual diagrams covering every aspect of LoRA:
the math, initialization, architecture, forward/backward pass,
data flow through every stage, memory layout, merging,
adapter swapping, scaling factor, rank selection, and
comparison with full fine-tuning and other PEFT methods.



    ══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    LoRA — COMPLETE VISUAL BREAKDOWN
    ══════════════════════════════════════════════════════════════════════════════════════════════════════



    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 1:  THE CORE PROBLEM LoRA SOLVES
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    The core problem: full ΔW (16.7M params) vs LoRA decomposition B×A (65K params), with visual size comparison of the matrices
    
    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                               │
    │   FULL FINE-TUNING: Learning the full weight update ΔW                                        │
    │                                                                                               │
    │   W₀ (pre-trained)          ΔW (what we need to learn)           W_new (fine-tuned)           │
    │   [4096 × 4096]             [4096 × 4096]                        [4096 × 4096]                │
    │                                                                                               │
    │   ┌─────────────────┐       ┌───────────────────┐                  ┌──────────────────┐       │
    │   │░░░░░░░░░░░░░░░░░│       │███████████████████│                  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│       │
    │   │░░░░░░░░░░░░░░░░░│       │███████████████████│                  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│       │
    │   │░░░ 16,777,216  ░│   +   │██ 16,777,216    ██│      =           │▓▓▓ 16,777,216  ▓▓│       │
    │   │░░░ parameters  ░│       │██ parameters    ██│                  │▓▓▓ parameters  ▓▓│       │
    │   │░░░ (frozen)    ░│       │██ (ALL learned) ██│                  │▓▓▓ (updated)   ▓▓│       │
    │   │░░░░░░░░░░░░░░░░░│       │███████████████████│                  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│       │
    │   │░░░░░░░░░░░░░░░░░│       │███████████████████│                  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│       │
    │   └─────────────────┘       └───────────────────┘                  └──────────────────┘       │
    │                                                                                               │
    │                     PROBLEM: ΔW is HUGE. For every target matrix,                             │
    │                     we're learning 16.7 million parameters.                                   │
    │                     Multiply by all layers and targets = billions.                            │
    │                                                                                               │
    │                                                                                               │
    │       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │                                                                                               │
    │                                                                                               │
    │   LoRA INSIGHT: ΔW has LOW RANK — it can be decomposed into two tiny matrices                 │
    │                                                                                               │
    │   W₀ (pre-trained)          ΔW ≈ B × A                          W_new (fine-tuned)            │
    │   [4096 × 4096]             [4096×r] × [r×4096]                  [4096 × 4096]                │
    │                                                                                               │
    │   ┌─────────────────┐       ┌────┐   ┌─────────────────┐        ┌──────────────────┐          │
    │   │░░░░░░░░░░░░░░░░░│       │████│   │█████████████████│        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│          │
    │   │░░░░░░░░░░░░░░░░░│       │████│   │                 │        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│          │
    │   │░░░ 16,777,216  ░│   +   │████│ × │  A [8 × 4096]   │  =     │▓▓▓ 16,777,216  ▓▓│          │
    │   │░░░ parameters  ░│       │████│   │  = 32,768       │        │▓▓▓ parameters  ▓▓│          │
    │   │░░░ (frozen)    ░│       │████│   │  params         │        │▓▓▓ (updated)   ▓▓│          │
    │   │░░░░░░░░░░░░░░░░░│       │████│   └─────────────────┘        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│          │
    │   │░░░░░░░░░░░░░░░░░│       │████│   B [4096 × 8]               │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│          │
    │   └─────────────────┘       │████│   = 32,768 params            └──────────────────┘          │
    │                             └────┘                                                            │
    │                                                                                               │
    │                     Total LoRA params: 32,768 + 32,768 = 65,536                               │
    │                     vs. full ΔW:       16,777,216                                             │
    │                                                                                               │
    │                     That's a 256× REDUCTION for this one weight matrix.                       │
    │                                                                                               │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 2:  THE MATH — WHAT HAPPENS TO A SINGLE INPUT VECTOR
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
    The math traced step by step: input x splits into frozen path (W₀x) and LoRA path (A compresses → B expands → scale by α/r → sum), 
    with shapes and compute costs at each step

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   FULL FINE-TUNING:       h = W_new × x       =  (W₀ + ΔW) × x                                   │
    │                                                                                                  │
    │   LoRA:                   h = W₀x + (α/r)·BAx                                                    │
    │                                                                                                  │
    │                                                                                                  │
    │   Input x                                                                                        │
    │   [4096]                                                                                         │
    │     │                                                                                            │
    │     ├─────────────────────────────────┐                                                          │
    │     │                                 │                                                          │
    │     │  FROZEN PATH                    │  LoRA PATH                                               │
    │     │                                 │                                                          │
    │     ▼                                 ▼                                                          │
    │   ┌────────────────────────┐        ┌────────────────────────┐                                   │
    │   │                        │        │                        │                                   │
    │   │   W₀ × x               │        │   A × x                │                                   │
    │   │                        │        │                        │                                   │
    │   │   [4096 × 4096]        │        │   [8 × 4096]           │                                   │
    │   │   × [4096]             │        │   × [4096]             │                                   │
    │   │   = [4096]             │        │   = [8]                │    ← COMPRESSED to 8 dimensions   │
    │   │                        │        │                        │                                   │
    │   │   16.7M multiplications│        │   32K multiplications  │    ← 500× less compute            │
    │   │                        │        │                        │                                   │
    │   │   FROZEN               │        │   TRAINABLE            │                                   │
    │   │   (no gradients stored)│        │   (gradients stored)   │                                   │
    │   │                        │        │                        │                                   │
    │   └───────────┬────────────┘        └───────────┬────────────┘                                   │
    │               │                                 │                                                │
    │               │                                 ▼                                                │
    │               │                     ┌────────────────────────┐                                   │
    │               │                     │                        │                                   │
    │               │                     │   B × (Ax)             │                                   │
    │               │                     │                        │                                   │
    │               │                     │   [4096 × 8]           │                                   │
    │               │                     │   × [8]                │                                   │
    │               │                     │   = [4096]             │    ← EXPANDED back to 4096        │
    │               │                     │                        │                                   │
    │               │                     │   TRAINABLE            │                                   │
    │               │                     │                        │                                   │
    │               │                     └───────────┬────────────┘                                   │
    │               │                                 │                                                │
    │               │                                 ▼                                                │
    │               │                     ┌────────────────────────┐                                   │
    │               │                     │                        │                                   │
    │               │                     │   × (α / r)            │    ← SCALE by alpha/rank          │
    │               │                     │                        │                                   │
    │               │                     │   e.g., 16/8 = 2.0     │                                   │
    │               │                     │                        │                                   │
    │               │                     └───────────┬────────────┘                                   │
    │               │                                 │                                                │
    │               ▼                                 ▼                                                │
    │           ┌─────────────────────────────────────────┐                                            │
    │           │                                         │                                            │
    │           │            h = W₀x + (α/r)·BAx          │                                            │
    │           │                                         │                                            │
    │           │            Element-wise addition        │                                            │
    │           │            Both are [4096] vectors      │                                            │
    │           │                                         │                                            │
    │           └────────────────────┬────────────────────┘                                            │
    │                                │                                                                 │
    │                                ▼                                                                 │
    │                          Output h [4096]                                                         │
    │                          (continues through                                                      │
    │                           rest of transformer)                                                   │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 3:  INITIALIZATION — WHY B STARTS AT ZERO
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
    Initialization: why A gets random values and B gets zeros, what ΔW looks like at step 0 vs after training, 
    and why this preserves pre-trained knowledge

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   AT TRAINING STEP 0:                                                                            │
    │                                                                                                  │
    │                                                                                                  │
    │   Matrix A:   initialized with small random Gaussian values                                      │
    │                                                                                                  │
    │   ┌───────────────────────────────────────────────────────────────┐                              │
    │   │  0.012  -0.034   0.008   0.041  -0.019   0.027  ...  4096 cols│                              │
    │   │ -0.007   0.051  -0.023   0.016   0.038  -0.011  ...           │                              │
    │   │  0.029  -0.018   0.044  -0.031   0.009   0.033  ...           │                              │
    │   │   ...    ...     ...     ...     ...     ...                  │                              │
    │   │  (8 rows × 4096 cols)                                         │                              │
    │   └───────────────────────────────────────────────────────────────┘                              │
    │                                                                                                  │
    │                                                                                                  │
    │   Matrix B:   initialized to ALL ZEROS                                                           │
    │                                                                                                  │
    │   ┌────────────────────┐                                                                         │
    │   │  0  0  0  0  0  0  │                                                                         │
    │   │  0  0  0  0  0  0  │                                                                         │
    │   │  0  0  0  0  0  0  │                                                                         │
    │   │  0  0  0  0  0  0  │                                                                         │
    │   │  ...               │                                                                         │
    │   │(4096 rows × 8 cols)│                                                                         │
    │   └────────────────────┘                                                                         │
    │                                                                                                  │
    │                                                                                                  │
    │   Therefore at step 0:                                                                           │
    │                                                                                                  │
    │       ΔW = B × A = ZEROS × A = ZERO MATRIX                                                       │
    │                                                                                                  │
    │       W_effective = W₀ + ΔW = W₀ + 0 = W₀                                                        │
    │                                                                                                  │
    │       ┌─────────────────────────────────────────────────────────────────────────┐                │
    │       │                                                                         │                │
    │       │  The model starts EXACTLY as the pre-trained model.                     │                │
    │       │  Zero perturbation. Zero risk of destroying knowledge at init.          │                │
    │       │  LoRA adapters are invisible until B learns non-zero values.            │                │
    │       │                                                                         │                │
    │       └─────────────────────────────────────────────────────────────────────────┘                │
    │                                                                                                  │
    │                                                                                                  │
    │   AS TRAINING PROGRESSES (step 100, 500, 1000...):                                               │
    │                                                                                                  │
    │   B gradually learns non-zero values:                                                            │
    │                                                                                                  │
    │   ┌────────────────────┐                                                                         │
    │   │ 0.15 -0.08  0.23   │                                                                         │
    │   │-0.41  0.19 -0.07   │     ΔW = B × A ≠ 0 anymore                                              │
    │   │ 0.03  0.37 -0.29   │                                                                         │
    │   │ ...                │     Model gradually adapts to new task                                  │
    │   └────────────────────┘     while preserving pre-trained knowledge                              │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 4:  WHICH LAYERS GET LoRA — TARGET MODULES INSIDE A TRANSFORMER
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Target modules: every weight matrix inside a transformer layer labeled with whether it gets LoRA, 
    plus the three common configurations (minimal/standard/aggressive) with parameter counts
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   ONE TRANSFORMER LAYER (e.g., Layer 0 of 32)                                                    │
    │                                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                           SELF-ATTENTION BLOCK                                           │   │
    │   │                                                                                          │   │
    │   │    ┌──────────────────────────┐     ┌──────────────────────────┐                         │   │
    │   │    │   W_q [4096 × 4096]      │     │   W_k [4096 × 4096]      │                         │   │
    │   │    │   Query projection       │     │   Key projection         │                         │   │
    │   │    │                          │     │                          │                         │   │
    │   │    │   ██ + LoRA (A_q, B_q) ██│     │   ██ + LoRA (A_k, B_k) ██│  ← target_modules       │   │
    │   │    └──────────────────────────┘     └──────────────────────────┘    = ["q_proj",         │   │
    │   │                                                                       "k_proj",          │   │
    │   │    ┌──────────────────────────┐     ┌──────────────────────────┐      "v_proj",          │   │
    │   │    │   W_v [4096 × 4096]      │     │   W_o [4096 × 4096]      │      "o_proj"]          │   │
    │   │    │   Value projection       │     │   Output projection      │                         │   │
    │   │    │                          │     │                          │                         │   │
    │   │    │   ██ + LoRA (A_v, B_v) ██│     │   ██ + LoRA (A_o, B_o) ██│                         │   │
    │   │    └──────────────────────────┘     └──────────────────────────┘                         │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                           FEED-FORWARD (MLP) BLOCK                                       │   │
    │   │                                                                                          │   │
    │   │    ┌───────────────────────────┐  ┌───────────────────────────┐                          │   │
    │   │    │   W_gate [4096 × 11008]   │  │   W_up [4096 × 11008]     │                          │   │
    │   │    │                           │  │                           │                          │   │
    │   │    │   No LoRA (standard)      │  │   No LoRA (standard)      │  ← NOT targeted          │   │
    │   │    │       — OR —              │  │       — OR —              │    (unless you use       │   │
    │   │    │   ██ + LoRA ██            │  │   ██ + LoRA ██            │    "all-linear")         │   │
    │   │    │   (if "all-linear")       │  │   (if "all-linear")       │                          │   │
    │   │    └───────────────────────────┘  └───────────────────────────┘                          │   │
    │   │                                                                                          │   │
    │   │    ┌───────────────────────────┐                                                         │   │
    │   │    │   W_down [11008 × 4096]   │                                                         │   │
    │   │    │                           │                                                         │   │
    │   │    │   No LoRA (standard)      │                                                         │   │
    │   │    │       — OR —              │                                                         │   │
    │   │    │   ██ + LoRA ██            │                                                         │   │
    │   │    └───────────────────────────┘                                                         │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    │                                                                                                  │
    │   × 32 layers = 32 copies of this structure, each with its OWN A and B matrices                  │
    │                                                                                                  │
    │                                                                                                  │
    │   COMMON CONFIGURATIONS:                                                                         │
    │                                                                                                  │
    │   ┌───────────────────┬──────────────────────────────────┬──────────────┬────────────────────┐   │
    │   │  Config           │  target_modules                  │  LoRA pairs  │  Trainable params  │   │
    │   │                   │                                  │  (32 layers) │  (rank=8)          │   │
    │   ├───────────────────┼──────────────────────────────────┼──────────────┼────────────────────┤   │
    │   │  Minimal          │  ["q_proj", "v_proj"]            │  64          │  ~4.2M  (0.06%)    │   │
    │   │  Standard    *    │  ["q_proj","k_proj","v_proj",    │  128         │  ~8.4M  (0.12%)    │   │
    │   │                   │   "o_proj"]                      │              │                    │   │
    │   │  Aggressive       │  "all-linear"  (attn + MLP)      │  224         │  ~18.9M (0.27%)    │   │
    │   └───────────────────┴──────────────────────────────────┴──────────────┴────────────────────┘   │
    │                                                                                                  │
    │   * = most common starting point                                                                 │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 5:  THE SCALING FACTOR α/r — THE VOLUME KNOB
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
    The α/r scaling factor visualized as a volume knob with bar charts showing different ratios
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   h = W₀x  +  (α/r) · BAx                                                                        │
    │       ─────    ────── ─────                                                                      │
    │         │         │      │                                                                       │
    │         │         │      └── raw LoRA output                                                     │
    │         │         └── scaling factor (volume knob)                                               │
    │         └── frozen output                                                                        │
    │                                                                                                  │
    │                                                                                                  │
    │   α (alpha) = constant set before training (typical: 8, 16, 32)                                  │
    │   r (rank)  = LoRA rank (typical: 4, 8, 16, 32)                                                  │
    │                                                                                                  │
    │                                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                                          │   │
    │   │   SCALING EXAMPLES:                                                                      │   │
    │   │                                                                                          │   │
    │   │   α=8,  r=8   →  α/r = 1.0    Neutral — LoRA output used as-is                           │   │
    │   │                                                                                          │   │
    │   │       W₀x:  ████████████████████████████████████████  (large, frozen)                    │   │
    │   │       BAx:  ████████                                   (1.0× scaling)                    │   │
    │   │                                                                                          │   │
    │   │                                                                                          │   │
    │   │   α=16, r=8   →  α/r = 2.0    Amplified — LoRA contributes more strongly                 │   │
    │   │                                                                                          │   │
    │   │       W₀x:  ████████████████████████████████████████  (large, frozen)                    │   │
    │   │       BAx:  ████████████████                           (2.0× scaling)                    │   │
    │   │                                                                                          │   │
    │   │                                                                                          │   │
    │   │   α=4,  r=8   →  α/r = 0.5    Dampened — LoRA contributes gently                         │   │
    │   │                                                                                          │   │
    │   │       W₀x:  ████████████████████████████████████████  (large, frozen)                    │   │
    │   │       BAx:  ████                                       (0.5× scaling)                    │   │
    │   │                                                                                          │   │
    │   │                                                                                          │   │
    │   │   RULE OF THUMB:                                                                         │   │
    │   │                                                                                          │   │
    │   │     α = r      →  safe default (scaling = 1.0)                                           │   │
    │   │     α = 2r     →  common choice (scaling = 2.0, slightly stronger adaptation)            │   │
    │   │     If you increase r, increase α proportionally to keep scaling stable                  │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 6:  FULL DATA PIPELINE — FROM RAW TEXT TO LoRA WEIGHT UPDATE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
    Complete data pipeline: 
    
        raw JSONL → template → tokenize → labels/loss mask → padding/attention mask → GPU tensors → embedding, 
    
    with concrete examples at every stage


    STAGE 1: RAW DATA (on disk — identical to full fine-tuning)
    ═══════════════════════════════════════════════════════════

    train.jsonl:
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ {"instruction": "Classify the sentiment", "input": "This movie was breathtaking", "output": "positive"} │
    │ {"instruction": "Classify the sentiment", "input": "Terrible waste of time", "output": "negative"}      │
    │ {"instruction": "Translate to French", "input": "The cat sat on the mat", "output": "Le chat..."}       │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │  DataLoader reads batch of examples
                                              ▼

    STAGE 2: TEMPLATE FORMATTING (text → structured text)
    ═══════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  "<s>[INST] Classify the sentiment: This movie was breathtaking [/INST] positive</s>"            │
    │  "<s>[INST] Classify the sentiment: Terrible waste of time [/INST] negative</s>"                 │
    │  "<s>[INST] Translate to French: The cat sat on the mat [/INST] Le chat était assis...</s>"      │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │  Still just text strings
                                              ▼

    STAGE 3: TOKENIZATION (text → integer IDs)
    ═══════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  Example 1: [1, 518, 25580, 29962, 4134, 1598, ..., 6374, 2]               → 20 tokens           │
    │  Example 2: [1, 518, 25580, 29962, 4134, 1598, ..., 8178, 2]               → 18 tokens           │
    │  Example 3: [1, 518, 25580, 29962, 4103, 9632, ..., 11690, 2]              → 29 tokens           │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │  Just integer arrays — different lengths
                                              ▼

    STAGE 4: LABELS + LOSS MASK (which tokens to grade)
    ═══════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │  Example 1:                                                                                      │
    │  input_ids: [1, 518, 25580, 29962, 4134, 1598, ...,    518, 29914, 25580, 29962, 6374,  2]       │
    │  labels:    [-100, -100, -100, -100, -100, -100, ..., -100,  -100,  -100,  -100, 6374,  2]       │
    │              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       │
    │              instruction tokens: IGNORED by loss (-100)                          output: GRADED  │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │  Model learns to predict ONLY the response
                                              ▼

    STAGE 5: PADDING + ATTENTION MASK (make all same length)
    ═══════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │  Pad to longest example in batch (29 tokens):                                                    │
    │                                                                                                  │
    │  input_ids [3 × 29]:                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐             │
    │  │  1  518  25580 ...  6374  2   0   0   0   0   0   0   0   0   0  │  Example 1   │             │
    │  │  1  518  25580 ...  8178  2   0   0   0   0   0   0   0   0   0  │  Example 2   │             │
    │  │  1  518  25580 ... 11690  2                                      │  Example 3   │             │
    │  └─────────────────────────────────────────────────────────────────────────────────┘             │
    │                                            ↑                                                     │
    │                                          PAD = 0 (meaningless filler)                            │
    │                                                                                                  │
    │  attention_mask [3 × 29]:                                                                        │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐             │
    │  │  1   1   1  ...   1   1   0   0   0   0   0   0   0   0   0     │  1=real 0=pad │             │
    │  │  1   1   1  ...   1   1   0   0   0   0   0   0   0   0   0     │               │             │
    │  │  1   1   1  ...   1   1                                         │               │             │
    │  └─────────────────────────────────────────────────────────────────────────────────┘             │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │  Rectangular tensors, ready for GPU
                                              ▼

    STAGE 6: MOVE TO GPU + EMBEDDING (integer IDs → vectors)
    ═══════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │  input_ids [3, 29]    →    Embedding Layer (FROZEN)    →    hidden_states [3, 29, 4096]          │
    │  (integers)                 (lookup table)                   (dense vectors)                     │
    │                                                                                                  │
    │  Token ID 518  → [0.023, -0.891, 0.445, ..., 0.112]   (4096 floats)                              │
    │  Token ID 6374 → [0.771,  0.034, -0.562, ..., -0.338] (4096 floats)                              │
    │  Token ID 0    → [0.0, 0.0, 0.0, ..., 0.0]            (PAD — masked out by attention_mask)       │
    │                                                                                                  │
    │  NOTE: In full fine-tuning, the embedding layer gets updated.                                    │
    │        In LoRA, it's typically FROZEN (unless you add LoRA to it, which is rare).                │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │  3D tensor [batch, seq_len, hidden_dim]
                                              │  enters the transformer layers
                                              ▼

    STAGE 7: FORWARD PASS THROUGH TRANSFORMER LAYERS (where LoRA happens)
    ═════════════════════════════════════════════════════════════════════════
                        (see DIAGRAM 7 below for detailed view)

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 7:  FORWARD PASS — INSIDE ONE TRANSFORMER LAYER WITH LoRA
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Full forward pass through one transformer layer: 
    
        input splits to Q/K/V/O each with frozen + LoRA paths, through attention, through MLP, out to logits and loss

    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                        │
    │   LAYER 0 (of 32)          Input: h = [batch=3, seq=29, dim=4096]                      │
    │                                                                                        │
    │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                                │   │
    │   │   SELF-ATTENTION WITH LoRA                                                     │   │
    │   │                                                                                │   │
    │   │         h                                                                      │   │
    │   │         │                                                                      │   │
    │   │    ┌────┴────┬────────────┬────────────┐                                       │   │
    │   │    │         │            │            │                                       │   │
    │   │    ▼         ▼            ▼            ▼                                       │   │
    │   │  ┌──────┐  ┌──────┐   ┌──────┐      ┌──────┐                                   │   │
    │   │  │ W_q  │  │ W_k  │   │ W_v  │      │      │                                   │   │
    │   │  │FROZEN│  │FROZEN│   │FROZEN│      │      │                                   │   │
    │   │  │      │  │      │   │      │      │      │                                   │   │
    │   │  │+ LoRA│  │+ LoRA│   │+ LoRA│      │      │                                   │   │
    │   │  │ A_q  │  │ A_k  │   │ A_v  │      │      │                                   │   │
    │   │  │ B_q  │  │ B_k  │   │ B_v  │      │      │                                   │   │
    │   │  └──┬───┘  └──┬───┘   └───┬──┘      │      │                                   │   │
    │   │     │         │           │         │      │                                   │   │
    │   │     ▼         ▼           ▼         │      │                                   │   │
    │   │     Q         K           V         │      │                                   │   │
    │   │     │         │           │         │      │                                   │   │
    │   │     └────┬────┘           │         │      │                                   │   │
    │   │          │                │         │      │                                   │   │
    │   │          ▼                │         │      │                                   │   │
    │   │    ┌───────────┐          │         │      │                                   │   │
    │   │    │ Q × K^T   │          │         │      │                                   │   │
    │   │    │ / √d_k    │          │         │      │                                   │   │
    │   │    │ + attn    │          │         │      │                                   │   │
    │   │    │   mask    │          │         │      │                                   │   │
    │   │    │ softmax   │          │         │      │                                   │   │
    │   │    └─────┬─────┘          │         │      │                                   │   │
    │   │          │                │         │      │                                   │   │
    │   │          └───── × ────────┘         │      │                                   │   │
    │   │                 │                   │      │                                   │   │
    │   │                 ▼                   │      │                                   │   │
    │   │          attention_output           │      │                                   │   │
    │   │                 │                   │      │                                   │   │
    │   │                 ▼                   │      │                                   │   │
    │   │           ┌──────────┐              │      │                                   │   │
    │   │           │   W_o    │              │      │                                   │   │
    │   │           │  FROZEN  │              │      │                                   │   │
    │   │           │ + LoRA   │              │      │                                   │   │
    │   │           │  A_o, B_o│              │      │                                   │   │
    │   │           └────┬─────┘              │      │                                   │   │
    │   │                │                    │      │                                   │   │
    │   │                ▼                    │      │                                   │   │
    │   │          attn_output                │      │                                   │   │
    │   │                                     │      │                                   │   │
    │   └─────────────────────────────────────┼──────┘                                   │   │
    │                    │                    │                                          │   │
    │                    ▼                    │                                          │   │
    │   ┌────────────────────────────────┐    │                                          │   │
    │   │  Residual Add: h + attn_output │    │                                          │   │
    │   │  LayerNorm                     │    │                                          │   │
    │   └────────────────┬───────────────┘    │                                          │   │
    │                    │                    │                                          │   │
    │                    ▼                    │                                          │   │
    │   ┌────────────────────────────────┐    │                                          │   │
    │   │  FEED-FORWARD MLP (FROZEN)     │    │  (no LoRA here in standard config)       │   │
    │   │                                │    │  (LoRA here if "all-linear" is used)     │   │
    │   │  gate = W_gate × h             │    │                                          │   │
    │   │  up   = W_up × h               │    │                                          │   │
    │   │  out  = W_down × (act(gate)    │    │                                          │   │
    │   │                    × up)       │    │                                          │   │
    │   └────────────────┬───────────────┘    │                                          │   │
    │                    │                    │                                          │   │
    │                    ▼                    │                                          │   │
    │   ┌────────────────────────────────┐    │                                          │   │
    │   │  Residual Add + LayerNorm      │    │                                          │   │
    │   └────────────────┬───────────────┘    │                                          │   │
    │                    │                                                               │   │
    │                    ▼                                                               │   │
    │              Output: h' = [3, 29, 4096]  →  passes to Layer 1                      │   │
    │                                                                                    │   │
    └────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                        │
    │    ... repeat for Layers 1-31, each with its OWN A and B matrices ...                  │
    │                                                                                        │
    │                    ▼                                                                   │
    │              ┌──────────────────────────────────┐                                      │
    │              │  LM HEAD (FROZEN, no LoRA)       │                                      │
    │              │  h_final × W_lm_head             │                                      │
    │              │  [3, 29, 4096] × [4096, 32000]   │                                      │
    │              │  = logits [3, 29, 32000]         │                                      │
    │              └──────────────────┬───────────────┘                                      │
    │                                 │                                                      │
    │                                 ▼                                                      │
    │              ┌──────────────────────────────────┐                                      │
    │              │  LOSS = CrossEntropy(logits,     │                                      │
    │              │         labels)                  │                                      │
    │              │  Only where labels ≠ -100        │                                      │
    │              └──────────────────────────────────┘                                      │
    │                                                                                        │
    └────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 8:  BACKWARD PASS — GRADIENT FLOW IN LoRA
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Backward pass gradient flow: where gradients flow through (frozen layers) vs where they're stored (LoRA A and B only),
    with memory comparison bars
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   Loss (scalar)                                                                                  │
    │     │                                                                                            │
    │     │  loss.backward()                                                                           │
    │     │                                                                                            │
    │     ▼                                                                                            │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │  LM HEAD (FROZEN)                                                                        │   │
    │   │                                                                                          │   │
    │   │  ∂Loss/∂W_lm_head    →  computed for chain rule, NOT STORED, NOT UPDATED                 │   │
    │   │  ∂Loss/∂h_final      →  passed backward to Layer 31                                      │   │
    │   │                                                                                          │   │
    │   └───────────────────────────────────────────────┬──────────────────────────────────────────┘   │
    │                                                   │                                              │
    │                                                   ▼                                              │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │  LAYER 31 → LAYER 30 → ... → LAYER 1 → LAYER 0                                           │   │
    │   │                                                                                          │   │
    │   │  At each layer, for each weight matrix with LoRA:                                        │   │
    │   │                                                                                          │   │
    │   │  ┌──────────────────────────────────────────────────────────────────────────────────┐    │   │
    │   │  │                                                                                  │    │   │
    │   │  │   FROZEN PATH (W₀):                                                              │    │   │
    │   │  │                                                                                  │    │   │
    │   │  │     ∂Loss/∂W₀  →  Computed (chain rule needs it)                                 │    │   │
    │   │  │                    but NOT STORED in memory                                      │    │   │
    │   │  │                    and NOT used for any update                                   │    │   │
    │   │  │                    W₀ remains exactly at pre-trained values                      │    │   │
    │   │  │                                                                                  │    │   │
    │   │  │   LoRA PATH (A, B):                                                              │    │   │
    │   │  │                                                                                  │    │   │
    │   │  │     ∂Loss/∂B   →  COMPUTED and STORED   ← gradient tensor allocated (~64 KB)     │    │   │
    │   │  │     ∂Loss/∂A   →  COMPUTED and STORED   ← gradient tensor allocated (~64 KB)     │    │   │
    │   │  │                                                                                  │    │   │
    │   │  │   These are the ONLY gradients that consume memory at this layer.                │    │   │
    │   │  │                                                                                  │    │   │
    │   │  └──────────────────────────────────────────────────────────────────────────────────┘    │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    │                                                                                                  │
    │   GRADIENT MEMORY COMPARISON:                                                                    │
    │                                                                                                  │
    │   Full fine-tuning:                                                                              │
    │     7B params × 2 bytes (BF16)  =  ~14 GB of gradient storage                                    │
    │     ████████████████████████████████████████████████████████████████████                         │
    │                                                                                                  │
    │   LoRA (rank=8, 4 targets):                                                                      │
    │     8.4M params × 2 bytes       =  ~17 MB of gradient storage                                    │
    │     █                                                                                            │
    │                                                                                                  │
    │   ~800× less gradient memory                                                                     │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 9:  OPTIMIZER UPDATE — WHAT ACTUALLY GETS CHANGED
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
     Optimizer update: Adam formula applied only to LoRA params, with optimizer state memory comparison (56 GB vs 67 MB)

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   optimizer.step()                                                                               │
    │                                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                                          │   │
    │   │   FROZEN PARAMETERS (7B):                                                                │   │
    │   │                                                                                          │   │
    │   │     W_q, W_k, W_v, W_o (all 32 layers)     ──▶  NO UPDATE.  No gradient.  No momentum.   │   │
    │   │     W_gate, W_up, W_down (all 32 layers)         No variance. No memory allocated.       │   │
    │   │     Embedding, LM Head                           Values identical to pre-trained model.  │   │
    │   │                                                                                          │   │
    │   │                                                                                          │   │
    │   │   LoRA PARAMETERS (~8.4M):                                                               │   │
    │   │                                                                                          │   │
    │   │     For each A_q, B_q, A_k, B_k, A_v, B_v, A_o, B_o  (× 32 layers):                      │   │
    │   │                                                                                          │   │
    │   │       ┌──────────────────────────────────────────────────────────────────────┐           │   │
    │   │       │                                                                      │           │   │
    │   │       │  Adam optimizer computes:                                            │           │   │
    │   │       │                                                                      │           │   │
    │   │       │  momentum[A] = 0.9 × momentum[A] + 0.1 × gradient[A]                 │           │   │
    │   │       │  variance[A] = 0.999 × variance[A] + 0.001 × gradient[A]²            │           │   │
    │   │       │                                                                      │           │   │
    │   │       │  A -= learning_rate × momentum[A] / (√variance[A] + 1e-8)            │           │   │
    │   │       │                                                                      │           │   │
    │   │       │  (same for B)                                                        │           │   │
    │   │       │                                                                      │           │   │
    │   │       └──────────────────────────────────────────────────────────────────────┘           │   │
    │   │                                                                                          │   │
    │   │                                                                                          │   │
    │   │   OPTIMIZER STATE MEMORY:                                                                │   │
    │   │                                                                                          │   │
    │   │   Full fine-tuning:    7B × 2 states × 4 bytes (FP32)  =  ~56 GB                         │   │
    │   │   ████████████████████████████████████████████████████████████████████████████████████   │   │
    │   │                                                                                          │   │
    │   │   LoRA:               8.4M × 2 states × 4 bytes       =  ~67 MB                          │   │
    │   │   █                                                                                      │   │
    │   │                                                                                          │   │
    │   │   ~850× less optimizer memory                                                            │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 10:  TOTAL MEMORY LAYOUT — LoRA vs FULL FINE-TUNING
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Total memory layout comparison: Full FT vs LoRA vs QLoRA with proportional bar visualizations
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   FULL FINE-TUNING (7B model, BF16 + Adam):                                                      │
    │                                                                                                  │
    │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │ Weights (BF16)          ██████████████                         14 GB                    │    │
    │   │ Gradients (BF16)        ██████████████                         14 GB                    │    │
    │   │ Optimizer (FP32)        ████████████████████████████████████████████████████████ 56 GB  │    │
    │   │ Activations             ████████████                           10-30 GB                 │    │
    │   │                                                                ──────────               │    │
    │   │                                                          TOTAL: 94-114 GB               │    │
    │   └─────────────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                                  │
    │                                                                                                  │
    │   LoRA (7B model, BF16 frozen base + LoRA rank=8):                                               │
    │                                                                                                  │
    │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │ Frozen weights (BF16)   ██████████████                         14 GB  (forward only)    │    │
    │   │ LoRA weights (BF16)     ▌                                      ~17 MB                   │    │
    │   │ LoRA gradients (BF16)   ▌                                      ~17 MB                   │    │
    │   │ LoRA optimizer (FP32)   ▌                                      ~67 MB                   │    │
    │   │ Activations             ██████                                 4-10 GB (reduced)        │    │
    │   │                                                                ──────────               │    │
    │   │                                                          TOTAL: 18-24 GB                │    │
    │   └─────────────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                                  │
    │                                                                                                  │
    │   QLoRA (7B model, NF4 frozen base + LoRA rank=8):                                               │
    │                                                                                                  │
    │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │ Frozen weights (NF4)    ███████                                3.5 GB (4-bit!)          │    │
    │   │ Quant constants (FP8)   ▌                                      ~125 MB                  │    │
    │   │ LoRA weights (BF16)     ▌                                      ~17 MB                   │    │
    │   │ LoRA gradients (BF16)   ▌                                      ~17 MB                   │    │
    │   │ LoRA optimizer (FP32)   ▌                                      ~67 MB                   │    │
    │   │ Activations             ████                                   4-8 GB                   │    │
    │   │                                                                ──────────               │    │
    │   │                                                          TOTAL: 8-12 GB                 │    │
    │   └─────────────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                                  │
    │                                                                                                  │
    │   VISUAL COMPARISON (scale: █ = ~2 GB):                                                          │
    │                                                                                                  │
    │   Full FT:  ██████████████████████████████████████████████████████   ~100 GB                     │
    │   LoRA:     █████████████                                            ~20 GB                      │
    │   QLoRA:    ██████                                                   ~10 GB                      │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 11:  END-TO-END TRAINING LOOP
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    End-to-end training loop: 
            
            freeze → attach LoRA → data pipeline → forward/backward/update cycle → save adapter
    
    
        ┌────────────────────┐             ┌──────────────────────┐
        │  Pre-trained Model │             │  Task-Specific Data  │
        │  (LLaMA-7B)        │             │  (train.jsonl)       │
        └──────┬─────────────┘             └──────────┬───────────┘
               │                                      │
               ▼                                      │
        ┌──────────────────────────────┐              │
        │  STEP 1: FREEZE ALL PARAMS   │              │
        │                              │              │
        │  for p in model.parameters():│              │
        │      p.requires_grad = False │              │
        │                              │              │
        │  7B params → all read-only   │              │
        └──────────────┬───────────────┘              │
                       │                              │
                       ▼                              │
        ┌──────────────────────────────┐              │
        │  STEP 2: ATTACH LoRA         │              │
        │                              │              │
        │  For each target module      │              │
        │  in each of 32 layers:       │              │
        │                              │              │
        │    Create A [r × d_in]       │              │
        │      init: random Gaussian   │              │
        │      requires_grad = True    │              │
        │                              │              │
        │    Create B [d_out × r]      │              │
        │      init: ALL ZEROS         │              │
        │      requires_grad = True    │              │
        │                              │              │
        │  Total: ~8.4M new params     │              │
        │  (0.12% of model)            │              │
        └──────────────┬───────────────┘              │
                       │                              │
                       ▼                              ▼
        ┌─────────────────────────────────────────────────────┐
        │  STEP 3: DATA PIPELINE                              │
        │                                                     │
        │  JSONL → Template → Tokenize → Labels/Mask →        │
        │  Pad → Attention Mask → Collate → GPU Tensors       │
        │                                                     │
        │  (100% identical to full fine-tuning)               │
        └─────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                                                                     │
        │  STEP 4: TRAINING LOOP                                              │
        │                                                                     │
        │  for epoch in range(num_epochs):        # typically 1-5             │
        │    for batch in dataloader:                                         │
        │                                                                     │
        │      ┌──────────────────────────────────────────────────────┐       │
        │      │  A) FORWARD PASS                                     │       │
        │      │                                                      │       │
        │      │  For each layer, for each target weight W₀:          │       │
        │      │                                                      │       │
        │      │    frozen_out = W₀ × x             (original path)   │       │
        │      │    lora_out   = B × (A × x)         (LoRA bypass)    │       │
        │      │    h          = frozen_out + (α/r) × lora_out        │       │
        │      │                                                      │       │
        │      │  → logits [batch, seq_len, vocab_size]               │       │
        │      └──────────────────────────┬───────────────────────────┘       │
        │                                 │                                   │
        │      ┌──────────────────────────▼───────────────────────────┐       │
        │      │  B) LOSS = CrossEntropy(logits, labels)              │       │
        │      │     Only where labels ≠ -100                         │       │
        │      └──────────────────────────┬───────────────────────────┘       │
        │                                 │                                   │
        │      ┌──────────────────────────▼───────────────────────────┐       │
        │      │  C) BACKWARD PASS                                    │       │
        │      │                                                      │       │
        │      │  loss.backward()                                     │       │
        │      │                                                      │       │
        │      │  Frozen W₀: gradients flow through, NOT stored       │       │
        │      │  LoRA A, B:  gradients COMPUTED and STORED           │       │
        │      │                                                      │       │
        │      │  Memory: ~17 MB gradients (vs. 14 GB full FT)        │       │
        │      └──────────────────────────┬───────────────────────────┘       │
        │                                 │                                   │
        │      ┌──────────────────────────▼───────────────────────────┐       │
        │      │  D) OPTIMIZER UPDATE                                 │       │
        │      │                                                      │       │
        │      │  optimizer.step()                                    │       │
        │      │    → updates ONLY A and B matrices                   │       │
        │      │    → ~67 MB optimizer states (vs. 56 GB full FT)     │       │
        │      │                                                      │       │
        │      │  optimizer.zero_grad()                               │       │
        │      │    → clears LoRA gradients for next batch            │       │
        │      │                                                      │       │
        │      │  Frozen params: COMPLETELY UNTOUCHED                 │       │
        │      └──────────────────────────────────────────────────────┘       │
        │                                                                     │
        │  Repeat for all batches × all epochs                                │
        │                                                                     │
        └─────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │                                                                      │
        │  STEP 5: SAVE ADAPTER                                                │
        │                                                                      │
        │  Saved:                                                              │
        │    adapter_model.safetensors     ~33 MB   (all A and B matrices)     │
        │    adapter_config.json           ~1 KB    (r, α, targets, base)      │
        │                                                                      │
        │  NOT saved: base model (14 GB) — referenced by name                  │
        │                                                                      │
        └──────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 12:  MERGING — LoRA'S KILLER FEATURE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
    Merging: 
        
            during-training two-path diagram → merge operation (matrix addition) → after-merge single-path diagram, 
    
    with the key comparison to adapters that can't merge

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                 │
    │   DURING TRAINING (two separate paths):                                                         │
    │                                                                                                 │
    │        x ──────┬────────────────────────────────┐                                               │
    │                │                                │                                               │
    │                ▼                                ▼                                               │
    │   ┌────────────────────────┐     ┌──────────────────────────┐                                   │
    │   │  W₀ (FROZEN)           │     │  LoRA: B × A (TRAINABLE) │                                   │
    │   │  [4096 × 4096]         │     │  [4096×8] × [8×4096]     │                                   │
    │   │  14 GB total           │     │  33 MB total             │                                   │
    │   └───────────┬────────────┘     └─────────────┬────────────┘                                   │
    │               │                                │                                                │
    │               │              h = W₀x + (α/r)·BAx                                                │
    │               └────────────── + ───────────────┘                                                │
    │                               │                                                                 │
    │                               ▼                                                                 │
    │                          output h                                                               │
    │                                                                                                 │
    │                                                                                                 │
    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
    │                                                                                                 │
    │   MERGE OPERATION (one-time, after training):                                                   │
    │                                                                                                 │
    │   ┌────────────────────────┐     ┌──────────────────────────┐                                   │
    │   │  W₀                    │     │  (α/r) · B × A           │                                   │
    │   │  [4096 × 4096]         │  +  │  [4096 × 4096]           │                                   │
    │   │                        │     │  (computed once)         │                                   │
    │   └───────────┬────────────┘     └─────────────┬────────────┘                                   │
    │               │                                │                                                │
    │               └──────── matrix addition ───────┘                                                │
    │                               │                                                                 │
    │                               ▼                                                                 │
    │               ┌────────────────────────────────────┐                                            │
    │               │                                    │                                            │
    │               │  W_merged = W₀ + (α/r)·BA          │                                            │
    │               │  [4096 × 4096]                     │                                            │
    │               │                                    │                                            │
    │               │  This is now a STANDARD weight     │                                            │
    │               │  matrix. No LoRA. No bypass.       │                                            │
    │               │  No adapter. Just a normal model.  │                                            │
    │               │                                    │                                            │
    │               └────────────────────────────────────┘                                            │
    │                                                                                                 │
    │                                                                                                 │
    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
    │                                                                                                 │
    │   AFTER MERGE (single path — zero overhead):                                                    │
    │                                                                                                 │
    │        x ──────────────────────────────┐                                                        │
    │                                        │                                                        │
    │                                        ▼                                                        │
    │                          ┌─────────────────────────┐                                            │
    │                          │  W_merged               │                                            │
    │                          │  [4096 × 4096]          │   ← looks like any normal model            │
    │                          │                         │      no adapters, no bypass                │
    │                          │  h = W_merged × x       │      zero extra compute                    │
    │                          └───────────┬─────────────┘      zero extra memory                     │
    │                                      │                    zero extra latency                    │
    │                                      ▼                                                          │
    │                                 output h                                                        │
    │                                                                                                 │
    │                                                                                                 │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
    │   │                                                                                          │  │
    │   │   WHY THIS MATTERS:                                                                      │  │
    │   │                                                                                          │  │
    │   │   Adapters (Additive PEFT):    adapter modules stay in forward path FOREVER              │  │
    │   │                                 → permanent inference overhead                           │  │
    │   │                                 → cannot be removed                                      │  │
    │   │                                                                                          │  │
    │   │   LoRA:                         merges into W₀ and DISAPPEARS                            │  │
    │   │                                 → zero inference overhead                                │  │
    │   │                                 → model is indistinguishable from full fine-tuning       │  │
    │   │                                                                                          │  │
    │   │   This is the single most important reason LoRA dominates.                               │  │
    │   │                                                                                          │  │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 13:  ADAPTER SWAPPING — ONE MODEL, MANY TASKS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Adapter swapping: one base model serving multiple tasks, with storage comparison scaling to 100 tasks (1.4 TB full FT vs 17.3 GB LoRA)
        
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │                                                                                                  │
    │                           ┌──────────────────────────────┐                                       │
    │                           │      BASE MODEL              │                                       │
    │                           │      (LLaMA-7B)              │                                       │
    │                           │      14 GB — loaded ONCE     │                                       │
    │                           │      in GPU memory           │                                       │
    │                           └──────────────┬───────────────┘                                       │
    │                                          │                                                       │
    │                     ┌────────────────────┼────────────────────┐                                  │
    │                     │                    │                    │                                  │
    │                     ▼                    ▼                    ▼                                  │
    │             ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                           │
    │             │ LoRA Adapter │     │ LoRA Adapter │     │ LoRA Adapter │     ...more               │
    │             │              │     │              │     │              │                           │
    │             │  Medical QA  │     │  Code Gen    │     │  Legal Summ  │                           │
    │             │              │     │              │     │              │                           │
    │             │  ~33 MB      │     │  ~33 MB      │     │  ~33 MB      │                           │
    │             │  rank=8      │     │  rank=16     │     │  rank=8      │                           │
    │             │  α=16        │     │  α=32        │     │  α=16        │                           │
    │             └──────────────┘     └──────────────┘     └──────────────┘                           │
    │                                                                                                  │
    │                                                                                                  │
    │   HOW SWAPPING WORKS:                                                                            │
    │                                                                                                  │
    │     1. Load base model once                      (14 GB, stays in GPU)                           │
    │     2. Load medical adapter                      (+33 MB, attach A & B matrices)                 │
    │     3. Process medical queries                   (model uses frozen W₀ + medical LoRA)           │
    │     4. Detach medical adapter                    (-33 MB)                                        │
    │     5. Load code adapter                         (+33 MB, attach different A & B)                │
    │     6. Process code queries                      (model uses frozen W₀ + code LoRA)              │
    │                                                                                                  │
    │   Swap time: milliseconds (just loading ~33 MB of small matrices)                                │
    │   No model reloading. No GPU re-initialization.                                                  │
    │                                                                                                  │
    │                                                                                                  │
    │   STORAGE COMPARISON:                                                                            │
    │                                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                                          │   │
    │   │   Full fine-tuning:  3 tasks × 14 GB  =  42 GB on disk                                   │   │
    │   │   ████████████████████████████████████████████                                           │   │
    │   │                                                                                          │   │
    │   │   LoRA adapters:     14 GB base + 3 × 33 MB = ~14.1 GB on disk                           │   │
    │   │   ██████████████▌                                                                        │   │
    │   │                                                                                          │   │
    │   │   10 tasks:                                                                              │   │
    │   │   Full FT:   10 × 14 GB  =  140 GB                                                       │   │
    │   │   LoRA:      14 GB + 10 × 33 MB = ~14.3 GB                                               │   │
    │   │                                                                                          │   │
    │   │   100 tasks:                                                                             │   │
    │   │   Full FT:   100 × 14 GB  =  1.4 TB                                                      │   │
    │   │   LoRA:      14 GB + 100 × 33 MB = ~17.3 GB                                              │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 14:  RANK SELECTION — WHAT DIFFERENT RANKS LOOK LIKE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    
    Rank selection: visual representation of different ranks as "bandwidth," plus a quality-vs-rank curve showing diminishing returns

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   RANK = how many dimensions the compressed representation has                                   │
    │                                                                                                  │
    │   Think of rank as the "bandwidth" of the adaptation:                                            │
    │                                                                                                  │
    │                                                                                                  │
    │   ORIGINAL WEIGHT MATRIX:    [4096 × 4096]   (full bandwidth — 4096 "channels")                  │
    │   ════════════════════════════════════════════════════════════════                               │
    │                                                                                                  │
    │                                                                                                  │
    │   Rank 1:    A=[1 × 4096]    B=[4096 × 1]                                                        │
    │              ─                ─                                                                  │
    │              1 channel of adaptation                                                             │
    │              Can only learn ONE direction of change                                              │
    │              Params: 8,192                                                                       │
    │                                                                                                  │
    │   Rank 4:    A=[4 × 4096]    B=[4096 × 4]                                                        │
    │              ────             ────                                                               │
    │              4 channels of adaptation                                                            │
    │              Params: 32,768                                                                      │
    │                                                                                                  │
    │   Rank 8:    A=[8 × 4096]    B=[4096 × 8]         ★ DEFAULT                                      │
    │              ────────         ────────                                                           │
    │              8 channels — captures most task adaptations well                                    │
    │              Params: 65,536                                                                      │
    │                                                                                                  │
    │   Rank 16:   A=[16 × 4096]   B=[4096 × 16]                                                       │
    │              ────────────     ────────────                                                       │
    │              16 channels — more expressive                                                       │
    │              Params: 131,072                                                                     │
    │                                                                                                  │
    │   Rank 32:   A=[32 × 4096]   B=[4096 × 32]                                                       │
    │              ════════════     ════════════                                                       │
    │              32 channels — approaching diminishing returns                                       │
    │              Params: 262,144                                                                     │
    │                                                                                                  │
    │   Rank 64:   A=[64 × 4096]   B=[4096 × 64]                                                       │
    │              ════════════════ ════════════════                                                   │
    │              64 channels — rarely needed                                                         │
    │              Params: 524,288                                                                     │
    │                                                                                                  │
    │                                                                                                  │
    │   QUALITY vs RANK (typical instruction-tuning task):                                             │
    │                                                                                                  │
    │   Quality                                                                                        │
    │     ▲                                                                                            │
    │     │                           ┌─────────────────────────────── Full FT baseline                │
    │     │                    ╱──────                                                                 │
    │     │                 ╱──                                                                        │
    │     │              ╱─                                                                            │
    │     │           ╱─     ← diminishing returns beyond rank 16-32                                   │
    │     │        ╱─                                                                                  │
    │     │      ╱                                                                                     │
    │     │    ╱      ★ sweet spot (rank 8-16)                                                         │
    │     │  ╱                                                                                         │
    │     │╱                                                                                           │
    │     └──────────────────────────────────────────────────────────▶  Rank                           │
    │          1    4    8    16    32    64    128                                                    │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 15:  LoRA vs ADDITIVE ADAPTERS vs FULL FINE-TUNING — SIDE BY SIDE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Three-way architecture comparison: Full FT vs Adapters (serial) vs LoRA (parallel), with full property comparison table
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   HOW EACH METHOD MODIFIES THE FORWARD PASS:                                                     │
    │                                                                                                  │
    │                                                                                                  │
    │   FULL FINE-TUNING                ADDITIVE (Adapters)               LoRA                         │
    │   All weights updated             New modules inserted              Parallel bypass added        │
    │                                   IN SERIES                         IN PARALLEL                  │
    │                                                                                                  │
    │        ┌──────────┐                  ┌──────────┐                  ┌──────────┐                  │
    │        │  Input x │                  │  Input x │                  │  Input x │                  │
    │        └────┬─────┘                  └────┬─────┘                  └───┬──┬───┘                  │
    │             │                             │                            │  │                      │
    │             ▼                             ▼                            │  │                      │
    │   ┌─────────────────┐           ┌─────────────────┐                    │  │                      │
    │   │ W (TRAINABLE)   │           │ W₀ (FROZEN)     │             ┌──────┘  └───────┐              │
    │   │ all params      │           │                 │             │                 │              │
    │   │ updated by      │           └────────┬────────┘             ▼                 ▼              │
    │   │ gradients       │                    │               ┌──────────┐     ┌───────────┐          │
    │   └────────┬────────┘                    ▼               │ W₀       │     │ A (down)  │          │
    │            │                   ╔══════════════════╗      │ FROZEN   │     │ TRAINABLE │          │
    │            │                   ║  ADAPTER         ║      └────┬─────┘     └─────┬─────┘          │
    │            │                   ║  4096→64→4096    ║           │                 │                │
    │            │                   ║  + ReLU          ║           │                 ▼                │
    │            │                   ║  + residual      ║           │           ┌───────────┐          │
    │            │                   ║  TRAINABLE       ║           │           │ B (up)    │          │
    │            │                   ╚════════╤═════════╝           │           │ TRAINABLE │          │
    │            │                            │                     │           └─────┬─────┘          │
    │            │                            │                     │                 │                │
    │            │                            │                     │    × (α/r)      │                │
    │            │                            │                     │                 │                │
    │            ▼                            ▼                     └──────┬──────────┘                │
    │        ┌──────┐                    ┌──────┐                          │                           │
    │        │  h   │                    │  h   │                     ┌────▼────┐                      │
    │        └──────┘                    └──────┘                     │  h = +  │                      │
    │                                                                 └────┬────┘                      │
    │                                                                      │                           │
    │                                                                 ┌────▼────┐                      │
    │                                                                 │    h    │                      │
    │                                                                 └─────────┘                      │
    │                                                                                                  │
    │   After training:              After training:                  After training:                  │
    │   New W stored (14 GB)         Adapters stay (extra latency)    MERGE: W = W₀ + BA               │
    │   1 full copy per task         1 adapter file per task          Adapters VANISH                  │
    │                                Cannot merge                     Zero overhead                    │
    │                                                                 1 tiny file per task             │
    │                                                                                                  │
    │                                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                                          │   │
    │   │   Property             Full FT           Adapters           LoRA                         │   │
    │   │   ────────             ───────           ────────           ────                         │   │
    │   │   Architecture         Unchanged         Modified (serial)  Unchanged (parallel)         │   │
    │   │   Trainable params     100%              ~0.5-3%            ~0.1-1%                      │   │
    │   │   Inference overhead   None              Yes (permanent)    None (after merge)           │   │
    │   │   Mergeable            N/A               No                 Yes  ★                       │   │
    │   │   Checkpoint size      14 GB             ~130 MB            ~33 MB                       │   │
    │   │   Memory (training)    94-114 GB         19-25 GB           18-24 GB                     │   │
    │   │   Non-linearity        Yes (in model)    Yes (ReLU)         No (purely linear)           │   │
    │   │   Forgetting risk      High              Minimal            Minimal                      │   │
    │   │                                                                                          │   │
    │   │   ★ = the defining advantage of LoRA                                                     │   │
    │   │                                                                                          │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

Think of it this way. You have a painting (W₀). LoRA doesn't cut the painting into pieces and try to reassemble it. 
LoRA keeps the painting perfectly intact and learns a small transparent overlay (ΔW = B×A) that sits on top of it. 
The overlay adjusts the colors slightly for your specific task. 
The overlay itself is tiny because it's built from two small matrices multiplied together.


The reason B×A works as an approximation of ΔW (and not of W₀) is the low intrinsic rank insight. 
The changes needed to adapt a model to a new task are simple and low-dimensional, even though the original weights
 themselves are complex and high-dimensional. 
 The original W₀ has full rank — you could NOT reconstruct it from an 8-dimensional decomposition. 
 But ΔW (the adjustment) lives in a much simpler subspace that can be captured by rank 8.
 
 
A concrete analogy: imagine you have a 4K photograph (W₀). You couldn't compress that entire photo down to 8 pixels and reconstruct it. 
But the difference between that photo and a slightly color-corrected version (ΔW) — "make it a bit warmer, 
boost the blues slightly" — is simple enough to describe with just a few numbers.

---

{{LORA_IMAGE}}

---

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

##### Recap

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
    │          │                         │ ██ ADAPTER ██ │  ← NEW, trainable           │
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
    │   Learning rate           1e-6 to 5e-5             1e-4 to 3e-4    * higher      │
    │   Epochs                  1-5                      1-5             (similar)     │
    │   Batch size              32-128                   16-64           (similar)     │
    │   Warmup                  5-10% of steps           3-10% of steps                │
    │   Weight decay            0.01-0.1                 0.0-0.05        * lower       │
    │   Scheduler               Cosine decay             Cosine decay    (similar)     │
    │   Gradient accumulation   Often needed             Often needed    (similar)     │
    │                                                                                  │
    │   * LoRA can afford HIGHER learning rates because:                               │
    │     1. Only updating ~0.1% of params → less risk of catastrophic change          │
    │     2. Low-rank adapters are less prone to overshooting                          │
    │     3. Base model is frozen → built-in regularization                            │
    │                                                                                  │
    │   * LoRA needs LOWER weight decay because:                                       │
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
                                  WHEN LoRA DOESN'T WORK WELL
                               (Failure Modes and Limitations)
    ═══════════════════════════════════════════════════════════════════════════════════════════════

LoRA is powerful, but it's not a universal solution. Understanding where it breaks down
is just as important as knowing where it shines.

**1. Tasks requiring vocabulary expansion**

    LoRA cannot add new tokens to the vocabulary. If your target domain uses
    specialized terminology that was not in the base model's training data
    (rare chemical names, proprietary product codes, newly coined terms),
    LoRA may tokenize them awkwardly and struggle to represent them well.

    The LM head (the final projection from hidden states to vocab logits)
    is typically frozen in LoRA. You can't give the model new output tokens.

    Workaround: add new tokens and fine-tune the embedding + LM head layers
    separately (selectively unfreeze those weights), or use full fine-tuning
    for vocabulary-heavy domain adaptation.

**2. Very large domain gaps**

    LoRA's low-rank constraint works because task adaptation is assumed to be
    low-dimensional. But if you are taking a general English model and trying
    to make it fluent in a technical language with fundamentally different
    syntax and semantics (e.g., translating between natural language and a
    domain-specific programming language the base model has rarely seen),
    the adaptation may not fit within the low-rank subspace.

    Rule of thumb: the further your target domain is from pre-training,
    the higher the rank you need — and eventually, full fine-tuning wins.

**3. Small datasets with high rank**

    Counterintuitively, using too high a rank with too little data causes
    overfitting. The low-rank constraint is a form of regularization.
    If you have only a few hundred examples and use rank=64, the adapter
    has enough capacity to memorize the training set rather than generalize.

    With small datasets, start with r=4 or r=8. Let the rank constraint
    do the regularization work for you.

**4. Tasks where non-linearity matters**

    LoRA is purely linear: ΔW = B × A. No activation function between A and B.
    Bottleneck Adapters include a ReLU or GELU between the down- and up-projections,
    which can capture non-linear task-specific transformations.

    For most instruction-tuning tasks this doesn't matter — the non-linearity
    in the frozen transformer itself is sufficient. But for tasks requiring
    a very different "shape" of computation, adapters may outperform LoRA.

**5. Continual learning across many tasks**

    If you need to sequentially fine-tune a single adapter across many tasks
    without merging, earlier tasks will be overwritten (catastrophic forgetting
    within the adapter). LoRA prevents forgetting of the base model, but the
    adapter itself has no such protection.

    Mitigation: train separate adapters per task and swap at inference time,
    rather than training one adapter incrementally.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   WHEN TO USE FULL FINE-TUNING INSTEAD                                           │
    │                                                                                  │
    │   Signal                                        Recommendation                   │
    │   ──────                                        ──────────────                   │
    │   Need new vocabulary / tokens                  Full FT or selective unfreeze    │
    │   Domain gap is extremely large                 Full FT with high LR warmup      │
    │   Dataset < 500 examples                        LoRA r=4, or prompt tuning       │
    │   Dataset > 1M examples, unlimited compute      Full FT (max quality)            │
    │   Multiple sequential tasks, one adapter        Separate adapters per task       │
    │   LoRA at r=64 still underperforms              Full FT                          │
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
    │   LoRA                Reparameterization   0.1-1%           Low       Very Good     General purpose   *   │
    │   QLoRA               Hybrid               0.1-1%           V. Low    Very Good     Large models      *   │
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
    │   * = Recommended starting point for most practitioners                                                   │
    │                                                                                                           │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘

---

Why It Works — The Intuition

Think of a pre-trained model as having already organized the world's knowledge into a high-dimensional structure. 
When you fine-tune for a specific task, you're not restructuring the whole thing — you're just rotating a small subspace
of that structure to be more relevant to your domain. That rotation can be described with a low-rank matrix. 
LoRA is essentially finding that rotation directly, without wasting parameters on the parts of the weight space that 
don't need to change.

The Inference Advantage Over Adapters:

Bottleneck Adapters (the main Additive PEFT alternative) insert new modules between layers. 
Those modules are there at inference time, adding extra computation on every forward pass. 
LoRA's adapters merge back into the original weights and vanish. Net inference overhead: zero. 
This is the main practical reason LoRA dominates despite the similar parameter counts.

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

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE OPERATIONS — Runnable steps that execute the real LoRA training
# pipeline via peft_main.py  (valid --run choices: vram, prepare, train,
# inference, compare, all)
# ─────────────────────────────────────────────────────────────────────────────

    "1. Check VRAM Requirements": {
        "description": "Estimates GPU memory needed for LoRA training based on your config — much lighter than full fine-tuning",
        "runnable": True,
        "pipeline_cmd": "vram",
        "code": '''# VRAM Estimation for LoRA
# =========================
# Frozen base model (BF16, no gradients):  P × 2 bytes
# LoRA adapter weights (BF16):             A × 2 bytes  (tiny — r=8 ≈ 32 MB)
# LoRA gradients (BF16):                   A × 2 bytes
# LoRA optimizer states (AdamW FP32):      A × 8 bytes
# Activations (with grad checkpointing):   ~1-4 GB
#
# Example: LLaMA-3.2-1B + LoRA r=8, alpha=16
#   Frozen weights:  1.24B × 2 = 2.5   GB
#   LoRA weights:    4.2M  × 2 = 8.4   MB
#   Grads + optim:   4.2M  × 10 = 42   MB
#   Activations:               ~1.5 GB
#   TOTAL:                     ~5 GB   (vs ~16 GB for full FT)
#
# CLI equivalent:
#   python peft_main.py --run vram --yes
''',
    },

    "2. Prepare Dataset": {
        "description": "Downloads dataset from HuggingFace, applies the model chat template, and tokenizes for LoRA training",
        "runnable": True,
        "pipeline_cmd": "prepare",
        "code": '''# Data Preparation Pipeline (peft_prepare_data.py):
# ===================================================
# 1. Authenticate with HuggingFace using your token
# 2. Download dataset (default: yahma/alpaca-cleaned, ~52K examples)
# 3. Apply Llama chat template to each example:
#       {"instruction": "...", "input": "...", "output": "..."}
# 4. Tokenize all examples using the model's tokenizer
# 5. Create train/eval split
# 6. Return tokenized datasets ready for the Trainer
#
# Identical pipeline to full fine-tuning — LoRA sits transparently on
# top of the frozen base model, so tokenization is unchanged.
#
# CLI equivalent:
#   python peft_main.py --run prepare --yes
''',
    },

    "3. Train LoRA Adapters": {
        "description": "TAKES TIME — Freezes base model, attaches LoRA adapters, trains only A and B matrices",
        "runnable": True,
        "pipeline_cmd": "train",
        "needs_confirmation": True,
        "code": '''# LoRA Training Loop (peft_train.py):
# =====================================
# 1. Load base model in BF16 with device_map="auto"
# 2. Apply LoRA via get_peft_model(model, lora_config):
#       - Injects A [r × d_in] and B [d_out × r] matrices into target modules
#       - B initialized to zero → adapter outputs 0 at step 0 (identity start)
#       - A initialized with Gaussian noise
#       - ALL base model params set requires_grad = False
# 3. Configure TrainingArguments:
#       - AdamW (fused), Cosine LR scheduler, BF16 mixed precision
#       - Gradient checkpointing to save VRAM
# 4. Run trainer.train()
#       - Only LoRA A and B matrices get gradients + optimizer states
#       - Effective weight update: W_effective = W_frozen + (alpha/r) * B * A
# 5. Save adapter checkpoint (NOT the full model):
#       adapter_model.safetensors  (~32 MB for r=8 on 1B model)
#       adapter_config.json        (~1 KB)
#
# ESTIMATED TIME:
#   RTX 3090:  ~30-60 min
#   RTX 4090:  ~20-40 min
#   A100:      ~10-20 min
#
# CLI equivalent:
#   python peft_main.py --run train --yes
''',
    },

    "4. Test Inference": {
        "description": "Load the trained LoRA adapter on top of the base model and generate text to verify it works",
        "runnable": True,
        "pipeline_cmd": "inference",
        "code": '''# Inference Testing (peft_inference.py):
# ========================================
# 1. Load the FROZEN base model
# 2. Load saved LoRA adapter weights from outputs/final/
# 3. Attach adapter via PeftModel.from_pretrained(base, adapter_path)
# 4. Send a test prompt through the model
# 5. Display the generated response
#
# At inference the effective weight is:
#   W_effective = W_frozen + (alpha/r) × B × A
# This is computed on-the-fly. To eliminate this overhead entirely,
# use merge_and_unload() to bake A×B into W_frozen permanently.
#
# CLI equivalent:
#   python peft_main.py --run inference --yes --prompt "..."
''',
    },

    "5. Compare Original vs Fine-Tuned": {
        "description": "Side-by-side comparison of the frozen base model vs your LoRA-adapted version",
        "runnable": True,
        "pipeline_cmd": "compare",
        "code": '''# Model Comparison (peft_compare.py):
# =====================================
# 1. Load the ORIGINAL base model (no adapters)
# 2. Load base model + LoRA adapter weights
# 3. Send identical prompts to both
# 4. Display side-by-side responses
#
# The base model is shared between both runs — LoRA adds only ~32 MB
# on top, so both typically fit in VRAM simultaneously.
#
# CLI equivalent:
#   python peft_main.py --run compare --yes
''',
    },

    "6. Run Full Pipeline (1 to 5)": {
        "description": "Runs ALL steps sequentially — VRAM, Data, Train, Inference, Compare",
        "runnable": True,
        "pipeline_cmd": "all",
        "needs_confirmation": True,
        "code": '''# Full LoRA Pipeline (all steps in sequence):
# =============================================
# Step 1: Check VRAM requirements
# Step 2: Prepare dataset (download + tokenize)
# Step 3: Train LoRA adapters (frozen base + A/B matrices)
# Step 4: Quick inference test
# Step 5: Compare base model vs LoRA model
#
# Each step must succeed before the next one starts.
#
# CLI equivalent:
#   python peft_main.py --run all --yes
''',
    },

# ─────────────────────────────────────────────────────────────────────────────
# EDUCATIONAL CODE SNIPPETS — Reference implementations below
# ─────────────────────────────────────────────────────────────────────────────

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
# CUSTOM STREAMLIT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_operations():
    """
    Custom Streamlit UI for the Operations tab.

    Called by app_Testing.py instead of the default code-in-expander rendering.
    Provides:
    - Configuration panel (rank, alpha, target modules, model, etc.)
    - Run buttons for each pipeline step
    - Real-time streaming output from subprocess
    - Step status tracking (pending / running / success / failed)
    """
    import streamlit as st

    # ── Session State Init ──
    if "lora_step_outputs" not in st.session_state:
        st.session_state.lora_step_outputs = {}
    if "lora_step_status" not in st.session_state:
        st.session_state.lora_step_status = {}

    # ── Verify Script Exists ──
    script_path = _MAIN_SCRIPT
    scripts_dir = _SCRIPTS_DIR

    if not script_path.exists():
        st.error(
            f"**Pipeline script not found!**\n\n"
            f"Expected at:\n`{script_path}`\n\n"
            f"Edit the path variables `_SCRIPTS_DIR` / `_MAIN_SCRIPT` at the "
            f"top of `08_c_PEFT_Reparameterization_LORA.py` to match your layout."
        )
        st.markdown("---")
        st.caption("Falling back to code-only view:")
        _render_code_only(st)
        return

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION PANEL
    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ LoRA Configuration — Edit before running", expanded=False):
        st.caption("These values will be used when running the pipeline steps.")

        col1, col2 = st.columns(2)

        with col1:
            model_name = st.selectbox(
                "Base Model",
                options=[
                    "unsloth/Llama-3.2-1B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    "openai-community/gpt2",
                ],
                index=0,
                key="lora_model_name",
            )
            lora_rank = st.select_slider(
                "LoRA Rank (r)",
                options=[1, 2, 4, 8, 16, 32, 64],
                value=8,
                key="lora_rank",
            )
            lora_alpha = st.select_slider(
                "LoRA Alpha (α)",
                options=[4, 8, 16, 32, 64, 128],
                value=16,
                key="lora_alpha",
            )
            target_modules = st.multiselect(
                "Target Modules",
                options=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                default=["q_proj", "k_proj", "v_proj", "o_proj"],
                key="lora_target_modules",
            )

        with col2:
            max_seq_length = st.select_slider(
                "Max Sequence Length",
                options=[128, 256, 512, 1024, 2048],
                value=512,
                key="lora_seq_len",
            )
            num_epochs = st.number_input(
                "Epochs", min_value=1, max_value=10, value=3,
                key="lora_epochs",
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4],
                value=2e-4,
                format_func=lambda x: f"{x:.0e}",
                key="lora_lr",
            )
            batch_size = st.number_input(
                "Per-Device Batch Size",
                min_value=1, max_value=16, value=2,
                key="lora_batch_size",
            )
            grad_accum = st.number_input(
                "Gradient Accumulation Steps",
                min_value=1, max_value=64, value=8,
                key="lora_grad_accum",
            )

        effective_bs = batch_size * grad_accum
        scaling = lora_alpha / lora_rank
        st.info(
            f"**LoRA scaling (α/r):** {lora_alpha}/{lora_rank} = **{scaling:.2f}**  \n"
            f"**Effective batch size:** {batch_size} × {grad_accum} = **{effective_bs}**"
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE STEPS
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### Pipeline Steps")
    st.caption("Click **Run** to execute. Output streams in real-time.")
    st.markdown("")

    pipeline_ops = {k: v for k, v in OPERATIONS.items() if v.get("pipeline_cmd")}

    for op_name, op_data in pipeline_ops.items():
        pipeline_cmd  = op_data.get("pipeline_cmd", "")
        needs_confirm = op_data.get("needs_confirmation", False)
        step_key      = pipeline_cmd

        status = st.session_state.lora_step_status.get(step_key, "pending")
        icon   = {"pending": "⬜", "running": "🔄", "success": "✅", "failed": "❌"}.get(status, "⬜")

        has_output = step_key in st.session_state.lora_step_outputs
        with st.expander(f"{icon} {op_name}", expanded=has_output):

            st.markdown(f"**{op_data['description']}**")

            if st.checkbox("Show code details", key=f"lora_showcode_{step_key}", value=False):
                st.code(op_data["code"], language="python")

            st.markdown("---")

            run_disabled = False
            if needs_confirm:
                confirmed = st.checkbox(
                    "⚠️ I understand this will take time and I'm ready to proceed",
                    key=f"lora_confirm_{step_key}",
                    value=False,
                )
                run_disabled = not confirmed

            custom_prompt = None
            if step_key == "inference":
                custom_prompt = st.text_input(
                    "Test prompt:",
                    value="Explain the difference between supervised and unsupervised learning.",
                    key="lora_inference_prompt",
                )

            col_run, col_clear = st.columns([3, 1])

            with col_run:
                if st.button(
                    "▶️ Run",
                    key=f"lora_run_{step_key}",
                    disabled=run_disabled,
                    use_container_width=True,
                    type="primary" if step_key in ("train", "all") else "secondary",
                ):
                    _run_pipeline_step(
                        st, step_key, op_name,
                        script_path, scripts_dir,
                        prompt=custom_prompt,
                    )

            with col_clear:
                if has_output:
                    if st.button(
                        "Clear", key=f"lora_clear_{step_key}", use_container_width=True
                    ):
                        del st.session_state.lora_step_outputs[step_key]
                        st.session_state.lora_step_status[step_key] = "pending"
                        st.rerun()

            if has_output:
                output = st.session_state.lora_step_outputs[step_key]
                if status == "success":
                    st.success("Completed successfully")
                elif status == "failed":
                    st.error("Step failed — see output below")
                st.code(output, language="text")

    st.markdown("---")
    st.markdown("### Educational Code Snippets")
    st.caption("Reference implementations — read the theory in action.")

    educational_ops = {k: v for k, v in OPERATIONS.items() if not v.get("pipeline_cmd")}
    for op_name, op_data in educational_ops.items():
        is_runnable = op_data.get("runnable", False)
        with st.expander(f"{'▶️' if is_runnable else '📖'} {op_name}", expanded=False):
            st.markdown(f"**Description:** {op_data['description']}")
            st.markdown("---")
            st.code(op_data["code"], language="python")


# ─────────────────────────────────────────────────────────────────────────────
# SUBPROCESS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_step(st, step_key, step_label, script_path, scripts_dir, prompt=None):
    """Run a pipeline step via subprocess, streaming stdout to Streamlit."""
    st.session_state.lora_step_status[step_key] = "running"

    cmd = [
        sys.executable,
        str(script_path),
        "--run", step_key,
        "--yes",
    ]

    if step_key == "inference" and prompt:
        cmd.extend(["--prompt", prompt])

    output_lines     = []
    output_placeholder = st.empty()

    try:
        output_placeholder.info(f"🔄 Starting: {step_label} ...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(scripts_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            clean = _strip_ansi(line)
            output_lines.append(clean)
            output_placeholder.code("".join(output_lines), language="text")

        process.wait()

        if process.returncode == 0:
            st.session_state.lora_step_status[step_key] = "success"
            output_lines.append(
                f"\n{'='*50}\n"
                f"  {step_label} — completed successfully.\n"
            )
        else:
            st.session_state.lora_step_status[step_key] = "failed"
            output_lines.append(
                f"\n{'='*50}\n"
                f"  {step_label} — failed (exit code {process.returncode}).\n"
            )

    except FileNotFoundError:
        st.session_state.lora_step_status[step_key] = "failed"
        output_lines.append(
            f"Could not find Python or script.\n"
            f"  Python: {sys.executable}\n"
            f"  Script: {script_path}\n"
        )
    except Exception as e:
        st.session_state.lora_step_status[step_key] = "failed"
        output_lines.append(f"Unexpected error: {e}\n")

    final_output = "".join(output_lines)
    st.session_state.lora_step_outputs[step_key] = final_output
    output_placeholder.code(final_output, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK — Code-only view if script path is wrong
# ─────────────────────────────────────────────────────────────────────────────

def _render_code_only(st):
    """Render operations as plain code expanders (no run buttons)."""
    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**Description:** {op_data['description']}")
            st.markdown("---")
            st.code(op_data["code"], language="python")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    """Remove ANSI color/formatting escape codes from terminal output."""
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    from Required_Images.lora_visual import LORA_VISUAL_HEIGHT, LORA_VISUAL_HTML


    # Interactive component data — app.py splits theory_raw at each
    # placeholder and injects an st.components.v1.html() iframe instead.
    interactive_components = [

        {
            "placeholder": "{{LORA_IMAGE}}",
            "html": LORA_VISUAL_HTML,
            "height": LORA_VISUAL_HEIGHT,
        },

    ]
    return {
        "theory": THEORY,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
        "render_operations": render_operations,
    }