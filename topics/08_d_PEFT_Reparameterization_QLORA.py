"""
Fine Tuning - PEFT (Parameter-Efficient Fine-Tuning) Deep Dive: QLoRA
======================================================================

A comprehensive deep dive into QLoRA — Quantized Low-Rank Adaptation.
Covers the quantization math, the memory arithmetic, every innovation
in the original Dettmers et al. 2023 paper, and the complete
forward/backward pass in extreme detail.
"""

TOPIC_NAME = "Fine_Tuning_PEFT_QLORA"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

QLoRA's core insight: **LoRA is already parameter-efficient. Make the base model memory-efficient too.**

LoRA solves the *parameter* problem — instead of training 7 billion weights, you train 8 million.
But the base model still needs to live in GPU memory during training. For a 7B model in BF16,
that's 14 GB just for weights. Add optimizer states, gradients, and activations, and you need 16–24 GB.
A 70B model? 160+ GB — requiring multiple A100s.

QLoRA (Dettmers et al., 2023) asks: what if we could compress the frozen base model down to
4 bits per weight, leaving room for the LoRA adapters to run in full precision alongside it?
The result: a 7B model in ~4.5 GB, trainable with 16-bit LoRA adapters on a single consumer GPU.

Think of it this way:
    LoRA is a small team renovating a single floor of a skyscraper.
    QLoRA compresses the entire building into a scale model — the renovation team still works
    in full scale, but the building they're working around takes up 75% less space.

---


                ══════════════════════════════════════════════════════════════════════════════
                                         QLoRA — THE FOUR INNOVATIONS
                ══════════════════════════════════════════════════════════════════════════════


    QLoRA is not a single idea. It's four engineering innovations stacked on top of LoRA,
    each solving a specific bottleneck in running large model training on limited hardware.


    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                         │
    │                              THE FOUR QLoRA INNOVATIONS                                 │
    │                                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
    │  │  Innovation 1: NF4 — NormalFloat 4-bit                                          │    │
    │  │  Store base model weights in 4-bit NF4 format (not INT4, not FP4)               │    │
    │  │  An information-theoretically optimal quantization for normally-distributed     │    │
    │  │  neural network weights. Loses less information than INT4 at the same bitwidth. │    │
    │  └─────────────────────────────────────────────────────────────────────────────────┘    │
    │                   │                                                                     │
    │                   ▼                                                                     │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
    │  │  Innovation 2: Double Quantization (DQ)                                         │    │
    │  │  Quantize the quantization constants themselves.                                │    │
    │  │  Block-wise quantization requires storing one scale factor per block.           │    │
    │  │  DQ quantizes those scale factors too, saving an additional ~0.5 bits/param.    │    │
    │  └─────────────────────────────────────────────────────────────────────────────────┘    │
    │                   │                                                                     │
    │                   ▼                                                                     │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
    │  │  Innovation 3: Paged Optimizers                                                 │    │
    │  │  Use NVIDIA's unified memory to page optimizer states between GPU and CPU RAM.  │    │
    │  │  Handles memory spikes during gradient checkpointing without OOM crashes.       │    │
    │  └─────────────────────────────────────────────────────────────────────────────────┘    │
    │                   │                                                                     │
    │                   ▼                                                                     │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
    │  │  Innovation 4: LoRA Adapters in BF16 on top of the 4-bit base                   │    │
    │  │  The frozen base model is stored in NF4 (4-bit).                                │    │
    │  │  During computation, weights are dequantized to BF16 on-the-fly.                │    │
    │  │  LoRA adapters (A and B matrices) are stored and computed in BF16.              │    │
    │  │  Result: full-precision adapter training on a quantized base.                   │    │
    │  └─────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                         │
    └─────────────────────────────────────────────────────────────────────────────────────────┘


Together, these four innovations allow a 65B parameter model (Guanaco) to be trained on a single
48 GB GPU, achieving 99.3% of ChatGPT's performance on the Vicuna benchmark — results that
previously required at least 8× A100s running full fine-tuning.

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                          INNOVATION 1 — QUANTIZATION & NF4 IN DEPTH
                  (The Core Memory Compression — Understanding Every Bit)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### What Is Quantization?

Quantization is the process of representing numerical values in fewer bits.
Every weight in a neural network is a floating-point number. In BF16 (the standard training format),
each weight takes 16 bits = 2 bytes.

The goal of quantization is to represent the same weight using only 4 bits = 0.5 bytes.
That's a 4× compression of the weight memory footprint.

The challenge: you lose information. A 4-bit number can only represent 16 distinct values (2⁴ = 16).
A BF16 number can represent ~65,000 distinct values. The quantization question is:
which 16 values should we choose, and how should we map the original weights to them?

---

### Why Not Just Use INT4?

INT4 maps to 16 evenly spaced integers: {-8, -7, -6, ..., 6, 7}.
This looks like a natural choice — it covers the full range uniformly.

But here's the problem: **neural network weights are not uniformly distributed.**

After pre-training, weight values follow an approximately normal (Gaussian) distribution.
Most weights cluster near zero, with fewer at the extremes:

    Weight value distribution (typical LLM layer):

    Frequency
        │        ████
        │       ██████
        │      █████████
        │    █████████████
        │  ████████████████
        │████████████████████████
        └─────────────────────────────── Weight value
         -3  -2  -1   0   1   2   3

    Most weights are near 0. Very few weights are near ±3.

If you use INT4's evenly spaced quantization levels, you waste most of your 16 slots
representing extreme values that almost no weight has. Your quantization grid looks like:

    INT4 grid (uniform):
     │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    -8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7   8

    For a normally distributed weight, most values fall in [-2, 2].
    Only 6 of the 16 slots serve the majority of the data. The rest go to waste.

---

### NF4 — Information-Theoretically Optimal for Normal Distributions

NF4 (NormalFloat 4-bit) solves this by **distributing the 16 quantization levels
according to the quantiles of the normal distribution** — not uniformly.

Instead of equal spacing between -8 and 7, NF4 places more levels near zero
(where most weights cluster) and fewer at the extremes (where weights rarely appear).

The NF4 quantization levels are computed so that each level covers an equal probability
mass under the standard normal distribution N(0, 1):

    NF4 grid (quantile-based, normalized to [-1, 1]):

    {-1.0000, -0.6962, -0.5251, -0.3946, -0.2844, -0.1848, -0.0911, 0.0000,
      0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000}

    ← sparse                 dense near 0                  sparse →

    The levels are NOT evenly spaced. There are 8 levels between -0.4 and +0.4
    and only 3 levels below -0.5 and 3 above 0.5.

    This matches how neural network weights are actually distributed.
    Each level represents roughly the same number of weights (equal probability mass),
    so no information is "wasted" on empty regions of weight space.

Why "NormalFloat"? Because it's designed to be optimal when weights are normally distributed.
The original paper proves that NF4 has lower quantization error than INT4 for Gaussian data
at any given bitwidth — it's the information-theoretically correct format for this task.

---

### Block-Wise Quantization (Absmax Quantization)

Before applying NF4, each weight needs to be mapped to the [-1, 1] range where
the NF4 codebook lives. This is done via absmax quantization, applied block-by-block.

**Why block-wise?**

If you quantize a full weight matrix with a single scale factor (the global max),
one outlier value could distort the entire matrix. A single weight of magnitude 100
would compress all other weights into a tiny range near 0, destroying precision.

Block-wise quantization divides the weight matrix into small blocks
(default: 64 weights per block) and computes a separate scale factor for each block:

    Weight matrix W  [4096 × 4096] = 16,777,216 weights
                     │
                     ├── Block 1:  weights[0:64]       → scale₁   = max(|weights[0:64]|)
                     ├── Block 2:  weights[64:128]      → scale₂   = max(|weights[64:128]|)
                     ├── Block 3:  weights[128:192]     → scale₃   = max(|weights[128:192]|)
                     │   ...
                     └── Block N:  weights[N×64:(N+1)×64] → scaleₙ

    Total blocks: 16,777,216 / 64 = 262,144 blocks
    Scale factors: 262,144 values (one per block), stored in FP32

    Memory for scale factors: 262,144 × 4 bytes = 1,048,576 bytes ≈ 1.0 MB
    (small relative to the weight matrix itself)

**The quantization process (per block):**

    Step 1: Find the absolute maximum in the block
            absmax = max(|w| for w in block)                   e.g., absmax = 2.47

    Step 2: Normalize all weights in the block to [-1, 1]
            w_normalized = w / absmax                          maps [-2.47, 2.47] → [-1, 1]

    Step 3: Map to nearest NF4 quantization level
            w_nf4 = nearest_nf4_level(w_normalized)           maps float → 4-bit index (0-15)

    Step 4: Store the 4-bit index and the scale factor (absmax)
            Stored: w_nf4 (4 bits), absmax (float32 per block)

**The dequantization process (during forward pass):**

    Step 1: Look up the NF4 codebook value for the 4-bit index
            w_normalized = NF4_CODEBOOK[w_nf4]                4-bit index → float in [-1, 1]

    Step 2: Rescale back to original magnitude
            w_dequantized = w_normalized × absmax              e.g., → ~original value

    Step 3: Use w_dequantized (in BF16) for computation

This dequantization happens on-the-fly during each forward pass.
The weights are NEVER stored in BF16 — they live as 4-bit integers in GPU memory
and are promoted to BF16 only for the matrix multiplication, then discarded.

---

### The Concrete Memory Math

For a 7B parameter model (LLaMA-2-7B):

    WITHOUT quantization (BF16):
        7,000,000,000 weights × 2 bytes/weight = 14,000,000,000 bytes = ~13.0 GB

    WITH NF4 (4-bit):
        7,000,000,000 weights × 0.5 bytes/weight = 3,500,000,000 bytes = ~3.26 GB

        Plus scale factors (one FP32 per 64-weight block):
        
        7,000,000,000 / 64 = 109,375,000 blocks
        
        109,375,000 × 4 bytes = 437,500,000 bytes ≈ 0.41 GB

        Total: 3.26 + 0.41 = ~3.67 GB

    COMPRESSION RATIO: 13.0 GB → 3.67 GB = 3.54 × compression

    After double quantization (Innovation 2), the scale factor overhead drops further.
    Final QLoRA base model: ~3.5 GB for a 7B model.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   MODEL SIZE COMPARISON (weights only)                                           │
    │                                                                                  │
    │   Model Size    FP32         BF16         INT8         NF4 (4-bit)               │
    │   ──────────    ────         ────         ────         ──────────                │
    │   7B            28 GB        14 GB        7 GB         ~3.5 GB                   │
    │   13B           52 GB        26 GB        13 GB        ~6.5 GB                   │
    │   30B           120 GB       60 GB        30 GB        ~15 GB                    │
    │   65B           260 GB       130 GB       65 GB        ~33 GB                    │
    │   70B           280 GB       140 GB       70 GB        ~35 GB                    │
    │                                                                                  │
    │   * NF4 includes ~12% overhead for block-wise scale factors                      │
    │   * After double quantization, overhead drops to ~8%                             │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                             INNOVATION 2 — DOUBLE QUANTIZATION (DQ)
                      (Quantizing the Quantization Constants — Saving 0.5 bits/param)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### The Problem with Block-Wise Scale Factors

Block-wise quantization requires storing one FP32 scale factor per block of 64 weights.
FP32 = 32 bits = 4 bytes per scale factor.

    Scale factor overhead per parameter = 32 bits / 64 parameters = 0.5 bits/parameter

That's a non-trivial overhead on top of the 4 bits we're using per weight.
Effective cost: 4 + 0.5 = 4.5 bits per parameter. We can do better.

---

### Double Quantization — The Solution

Double Quantization quantizes the FP32 scale factors themselves using INT8.

    FIRST quantization:  Weights (BF16) → NF4 codes (4-bit)
                         Scale: one FP32 absmax per block of 64 weights

    SECOND quantization: Scale factors (FP32) → INT8 (8-bit)
                         Scale: one FP32 absmax per 256 scale factors (a "super-block")

Let's trace the math:

    First level (weights → NF4):
        Block size:         64 weights
        Scale precision:    FP32 (32 bits per block)
        Scale overhead:     32 bits / 64 weights = 0.5 bits per weight

    Second level (scale factors → INT8):
        Super-block size:   256 scale factors (covering 256 × 64 = 16,384 weights)
        Super-scale:        FP32 (one per super-block)
        INT8 scale factors: 256 × 8 bits = 2,048 bits instead of 256 × 32 = 8,192 bits
        Savings per weight: (8,192 - 2,048) bits / 16,384 weights ≈ 0.375 bits/weight

    Net overhead after double quantization:
        NF4 weights:              4.000 bits/param
        INT8 scale factors:       8/64  = 0.125 bits/param
        FP32 super-scale factors: 32/16384 ≈ 0.002 bits/param
        ─────────────────────────────────────────────────────
        Total:                    4.127 bits/param

    vs. standard NF4 without DQ:
        NF4 weights:              4.000 bits/param
        FP32 scale factors:       32/64 = 0.500 bits/param
        ─────────────────────────────────────────────────────
        Total:                    4.500 bits/param

    Savings from Double Quantization: 0.5 - 0.127 = 0.373 bits per parameter

    For a 7B parameter model:
        Savings = 0.373 bits × 7,000,000,000 / 8 bits/byte ≈ 327 MB

That 327 MB (~0.37 bits/param) is not enormous in absolute terms, but on a 10 GB GPU
it represents a meaningful fraction of available memory — potentially the difference
between a 7B model fitting at batch_size=4 vs. crashing with OOM.

---

### The Two-Level Hierarchy Visualized

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │                     DOUBLE QUANTIZATION HIERARCHY                                   │
    │                                                                                     │
    │   ┌─────────────────────────────────────────────────────────────────────────────┐   │
    │   │ SUPER-BLOCK (covers 16,384 weights)                                         │   │
    │   │                                                                             │   │
    │   │  FP32 super-scale: 1 value (4 bytes)                                        │   │
    │   │                                                                             │   │
    │   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
    │   │  │ BLOCK 1 (64 weights)                                                  │  │   │
    │   │  │  INT8 scale:    1 value (1 byte)  ← quantized with super-scale        │  │   │
    │   │  │  NF4 weights:   64 values (32 bytes = 64 × 4 bits)                    │  │   │
    │   │  └───────────────────────────────────────────────────────────────────────┘  │   │
    │   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
    │   │  │ BLOCK 2 (64 weights)                                                  │  │   │
    │   │  │  INT8 scale:    1 value (1 byte)                                      │  │   │
    │   │  │  NF4 weights:   64 values (32 bytes)                                  │  │   │
    │   │  └───────────────────────────────────────────────────────────────────────┘  │   │
    │   │  ...  (256 blocks per super-block)                                          │   │
    │   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
    │   │  │ BLOCK 256 (64 weights)                                                │  │   │
    │   │  │  INT8 scale:    1 value (1 byte)                                      │  │   │
    │   │  │  NF4 weights:   64 values (32 bytes)                                  │  │   │
    │   │  └───────────────────────────────────────────────────────────────────────┘  │   │
    │   └─────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                     │
    │   Dequantization order:                                                             │
    │   1. Use FP32 super-scale to dequantize INT8 block-scales → FP32 block-scales       │
    │   2. Use FP32 block-scales to dequantize NF4 weights → BF16 weights                 │
    │   3. Use BF16 weights for matrix multiplication                                     │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                              INNOVATION 3 — PAGED OPTIMIZERS
                     (Handling Memory Spikes Without OOM — GPU ↔ CPU Paging)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### The Memory Spike Problem

During training, GPU memory usage is not constant. Two events cause sharp spikes:

    1. Gradient checkpointing recomputation:
       When recomputing activations during backprop, all activations for a layer
       are momentarily in memory simultaneously. For long sequences, this can
       spike memory usage by 2-4 GB for just a few milliseconds.

    2. Microbatch boundary processing:
       When processing the final tokens in a packed batch, the model may briefly
       need more memory than the steady-state consumption.

These spikes cause OOM (Out of Memory) crashes at the worst time — deep into a training run.
The rest of the time, there's plenty of memory. The spike is the problem.

---

### Paged Optimizers — The Solution

QLoRA uses NVIDIA's unified memory feature, which allows GPU memory to automatically
page to and from CPU RAM when the GPU runs out — exactly like virtual memory in an OS.

Paged optimizers take the AdamW optimizer states (momentum and variance buffers)
and allocate them in CUDA unified memory rather than standard GPU memory:

    Standard AdamW optimizer states:
        momentum buffer:   1 FP32 value per trainable parameter → allocated in GPU VRAM
        variance buffer:   1 FP32 value per trainable parameter → allocated in GPU VRAM

        For 8.4M LoRA parameters:
        momentum: 8.4M × 4 bytes = 33.6 MB (in GPU VRAM, always)
        variance: 8.4M × 4 bytes = 33.6 MB (in GPU VRAM, always)

    Paged AdamW optimizer states:
        Same 33.6 MB each, but allocated in CUDA unified memory.

        During normal operation: lives in GPU VRAM (same speed as before)
        During memory spikes: automatically pages to CPU RAM (slower, but no OOM)
        After spike subsides: pages back to GPU VRAM

    The paging is transparent to the training code. You don't have to manage it.
    The speed cost is only paid during the rare spike — the rest of training runs at GPU speed.

---

### Why This Matters More Than It Sounds

Without paged optimizers, a training run on a 10 GB GPU might proceed perfectly for hours,
then crash with OOM on batch 847 because of a single long sequence that caused a temporary
memory spike. The entire run is lost.

With paged optimizers:
    - The spike gets handled by temporarily using CPU RAM
    - Training continues without interruption
    - The cost is a brief slowdown on that batch (milliseconds)
    - No data loss, no restart

This is particularly important for consumer GPUs (RTX 3090, 4090) where GPU memory is
precious and there's typically 32-64 GB of CPU RAM available as a buffer.

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   PAGED OPTIMIZER MEMORY FLOW                                                        │
    │                                                                                      │
    │   Normal operation:                                                                  │
    │                                                                                      │
    │   GPU VRAM (10 GB)                              CPU RAM (32 GB)                      │
    │   ┌─────────────────────────────────┐           ┌───────────────────────────────┐    │
    │   │ Base model (NF4):      3.5 GB   │           │                               │    │
    │   │ LoRA adapters (BF16):  0.1 GB   │           │   Optimizer states            │    │
    │   │ Opt. states (paged):   0.1 GB   │◀─────────▶│   (backup, mostly unused)     │    │
    │   │ Activations:           3.0 GB   │           │                               │    │
    │   │ Available:             3.3 GB   │           └───────────────────────────────┘    │
    │   └─────────────────────────────────┘                                                │
    │                                                                                      │
    │   During memory spike (gradient checkpointing):                                      │
    │                                                                                      │
    │   GPU VRAM (10 GB)                              CPU RAM (32 GB)                      │
    │   ┌─────────────────────────────────┐           ┌───────────────────────────────┐    │
    │   │ Base model (NF4):      3.5 GB   │           │                               │    │
    │   │ LoRA adapters (BF16):  0.1 GB   │           │   Optimizer states            │    │
    │   │ Opt. states:           0.0 GB   │◀── PAGE ──│   (paged out to CPU RAM)      │    │
    │   │ Activations (spike):   6.0 GB   │           │   0.1 GB                      │    │
    │   │ Available:             0.4 GB   │           └───────────────────────────────┘    │
    │   └─────────────────────────────────┘                                                │
    │                                                                                      │
    │   Spike handled: no OOM. After spike, optimizer states page back to GPU.            │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                        INNOVATION 4 — LORA IN BF16 ON TOP OF NF4 BASE
                  (The Compute Dtype Trick — Why This Works Numerically)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### The Storage vs. Compute Dtype Distinction

This is one of the most important and often-misunderstood aspects of QLoRA.

There are TWO distinct dtypes at play:

    Storage dtype:  How weights are stored in GPU memory (NF4 = 4-bit)
    Compute dtype:  What dtype is used during actual matrix multiplication (BF16 = 16-bit)

These are different. The 4-bit weights are NOT used directly in matrix multiplications.
Modern GPU hardware (CUDA cores, Tensor Cores) doesn't natively support 4-bit arithmetic.
Instead, the workflow is:

    1. Retrieve 4-bit NF4 weights from GPU memory (very fast — tiny data transfer)
    2. Dequantize to BF16 on-the-fly (cheap — a lookup + multiply per weight)
    3. Run matrix multiplication in BF16 (full Tensor Core performance)
    4. Discard the BF16 weights (they're NOT stored back — the NF4 remains the source of truth)

This is why QLoRA doesn't sacrifice compute speed as much as you'd expect.
The actual math happens in BF16. The savings come from the storage size.

---

### How LoRA Adapters Sit On Top

The LoRA A and B matrices are stored and computed in BF16 throughout.
They are never quantized. This is deliberate:

    The base model (NF4): frozen, read-only, static information
                          → aggressively compress it (4-bit is fine, it won't change)

    The LoRA adapters (BF16): actively trained, gradients flow through them
                               → must preserve precision for gradient descent to work

    The frozen model is like old books in a warehouse — compress them.
    The adapters are like the active work-in-progress — keep those in full quality.

During a forward pass through a QLoRA-adapted layer:

    h = (dequant(W₀_nf4) × x) + (α/r) · B(Ax)
    ─────────────────────────    ────────────────
             │                          │
             │                          └── LoRA path: A and B in BF16
             │                              gradients flow here
             │
             └── Frozen path: W₀ stored in NF4, dequantized to BF16 for compute
                              no gradients stored (requires_grad = False)

---

### Why NF4 + BF16 Compute Works Numerically

You might worry: if we're dequantizing to BF16 and computing in BF16, aren't we
introducing quantization error that corrupts the gradients?

Yes — but it's bounded, and the LoRA structure limits the impact:

    1. The quantization error in W₀_nf4 is constant per weight (fixed after quantization).
       It appears as a fixed bias in the frozen path's output, not as noise.

    2. The LoRA adapters can partially compensate for systematic quantization errors
       in the frozen base — they "see" the quantized base and adapt accordingly.

    3. The NF4 format is chosen precisely because it minimizes this quantization error
       for normally-distributed weights. The errors are small to begin with.

    4. In practice, QLoRA-trained models are within 1-4% quality of LoRA-trained models
       on the same base, which are within 1-5% of full fine-tuning. The compounding
       error is acceptable for most use cases.

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                             THE COMPLETE QLoRA FORWARD PASS — STEP BY STEP
                   (Tracing a Single Token Through Every Layer of a QLoRA Model)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


Let's trace one token through a QLoRA-adapted LLaMA-7B model.
Model specs: 32 transformer layers, d_model=4096, LoRA on q/k/v/o_proj, rank=8.

**Input to Layer 0:**

    x = [4096]   (one token's hidden state, BF16)

**At every QLoRA attention layer (example: W_q with LoRA):**

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   STEP 1: LOAD 4-BIT WEIGHTS FROM GPU MEMORY                                        │
    │                                                                                     │
    │   W_q is stored as NF4 on GPU:                                                      │
    │       NF4 weight tensor:    [4096 × 4096]  packed into 4 bits each                  │
    │       Size in memory:       4096 × 4096 × 0.5 bytes = 8 MB                          │
    │                             (vs. 32 MB in BF16)                                     │
    │       Block-scale factors:  FP32 or INT8 (after double quantization)                │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   STEP 2: DEQUANTIZE TO BF16 (ON-THE-FLY)                                           │
    │                                                                                     │
    │   For each block of 64 weights:                                                     │
    │       1. Dequantize scale: INT8_scale → FP32_scale (using super-scale)              │
    │       2. Look up NF4 codebook: 4-bit index → float in [-1, 1]                       │
    │       3. Rescale:            w_bf16 = nf4_value × FP32_scale                        │
    │                                                                                     │
    │   Output: W_q_bf16 [4096 × 4096] in BF16 (temporary, in L2/L3 cache or VRAM)        │
    │                                                                                     │
    │   Note: W_q_bf16 is NOT stored persistently. It's created, used for one             │
    │         matrix multiply, then discarded. Only the NF4 version persists.             │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   STEP 3: FROZEN PATH — COMPUTE W_q × x IN BF16                                     │
    │                                                                                     │
    │   q_frozen = W_q_bf16 × x                                                           │
    │                                                                                     │
    │   Shapes:   [4096 × 4096] × [4096]  →  [4096]                                       │
    │   Dtype:    BF16 × BF16             →  BF16                                         │
    │   Hardware: Tensor Cores (full speed)                                               │
    │   Gradient: NOT stored (requires_grad = False for W_q)                              │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                            │
                            │                     ┌──────────────────────────────────────────┐
                            │                     │                                          │
                            │                     │  STEP 3B: LoRA PATH (PARALLEL)           │
                            │                     │                                          │
                            │                     │  A_q is stored in BF16 [8 × 4096]        │
                            │                     │  B_q is stored in BF16 [4096 × 8]        │
                            │                     │  Both have requires_grad = True          │
                            │                     │                                          │
                            │                     │  x_down = A_q × x                        │
                            │                     │  Shapes: [8×4096] × [4096] → [8]         │
                            │                     │  (compress to rank-8 subspace)           │
                            │                     │                                          │
                            │                     │  x_up = B_q × x_down                     │
                            │                     │  Shapes: [4096×8] × [8] → [4096]         │
                            │                     │  (expand back to full dimension)         │
                            │                     │                                          │
                            │                     │  q_lora = (α/r) × x_up   (scale)         │
                            │                     │                                          │
                            │                     └──────────────────┬───────────────────────┘
                            │                                        │
                            ▼                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   STEP 4: COMBINE PATHS                                                             │
    │                                                                                     │
    │   q = q_frozen + q_lora                                                             │
    │     = (dequant(W_q_nf4) × x) + (α/r) · B_q(A_q × x)                                 │
    │     = [4096]                                                                        │
    │                                                                                     │
    │   Same operation as LoRA, except W₀ was loaded from 4-bit storage first.            │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    The same process repeats for W_k, W_v, W_o, and optionally the MLP layers.
    After 32 such layers, the output hits the LM Head (frozen, no LoRA typically):

    logits = LM_head(hidden_state)
           = [batch_size, seq_len, vocab_size]

---

**Complete Forward Pass Data Flow (7B model, batch=4, seq=512, rank=8):**

    COMPONENT                           SHAPE                   DTYPE     SIZE IN MEMORY
    ─────────                           ─────                   ─────     ──────────────

    Input token IDs                     [4, 512]                INT32     ~4 KB
    Embeddings (frozen, NF4-like)       [4, 512, 4096]          BF16      ~16 MB

    Per layer (×32):
      W_q (stored NF4)                  [4096, 4096]            NF4       8 MB (persistent)
      W_q (dequantized, temporary)      [4096, 4096]            BF16      32 MB (created/discarded)
      q_frozen output                   [4, 512, 4096]          BF16      ~16 MB
      A_q (LoRA, stored)                [8, 4096]               BF16      0.064 MB (persistent)
      B_q (LoRA, stored)                [4096, 8]               BF16      0.064 MB (persistent)
      q_lora output                     [4, 512, 4096]          BF16      ~16 MB
      q combined                        [4, 512, 4096]          BF16      ~16 MB
      (same for k, v, o)

    Logits                              [4, 512, 32000]         BF16      ~125 MB

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                              THE COMPLETE QLoRA TRAINING LOOP — STEP BY STEP
                      (Every Stage from Raw Data to Saved Adapter, in Extreme Detail)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


**Step 0: Understand What Problem QLoRA Is Solving**

LoRA on a 7B model requires ~16-24 GB GPU memory (14 GB for BF16 weights + adapters + activations).
This works on an A100 (80 GB) but not on consumer hardware.

QLoRA's target: a 7B model in ~8-12 GB. That's an RTX 3090 or RTX 4090.
A 70B model: ~40 GB. That's a single A100 80 GB instead of 8×A100s.

The recipe:
    1. Quantize base model weights to NF4 (4-bit) → 75% memory reduction for weights
    2. Double-quantize scale factors → additional ~375 MB savings
    3. Use paged optimizers → handle memory spikes without OOM
    4. Train LoRA adapters in BF16 on top

---

**Step 1: Install and Configure the 4-Bit Base Model**

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Enable 4-bit loading
        bnb_4bit_quant_type="nf4",              # Use NF4 (not INT4, not FP4)
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 (not FP32)
        bnb_4bit_use_double_quant=True,         # Enable double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",                      # Automatically assign layers to GPU(s)
    )

    What happens during from_pretrained() with 4-bit config:

    1. Download model weights (14 GB BF16 checkpoint)
    2. For each weight matrix, quantize to NF4:
           a. Divide weights into blocks of 64
           b. Compute absmax per block
           c. Normalize and map to NF4 codebook
           d. Store 4-bit codes + scale factors
    3. Load quantized model into GPU VRAM:
           NF4 weights: ~3.5 GB
           Scale factors: ~0.4 GB (or ~0.1 GB after double quant)
    4. Register all base model parameters as requires_grad = False

    After loading: ~3.6 GB of GPU memory used for the base model
    (vs. 14 GB in standard BF16 LoRA)

---

**Step 2: prepare_model_for_kbit_training()**

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    This function does three specific things:

    2a. Enables gradient checkpointing on the model
        (saves activation memory at the cost of recomputation during backward)

    2b. Casts all non-quantized modules to FP32
        (layer norms, embedding layers — these don't get quantized but need FP32 for stability)

        Why? Layer norms work on statistics (mean, variance) of activations.
        In BF16, these computations lose precision. QLoRA keeps them in FP32
        to avoid numerical instability in normalization.

    2c. Configures gradient checkpointing to handle the frozen/quantized base
        (ensures the checkpointing recomputation correctly dequantizes weights again)

---

**Step 3: Apply LoRA Adapters**

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",           # Or ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    After this step, the model has:
        - Frozen NF4 base weights (no gradients, ~3.6 GB)
        - Trainable BF16 LoRA A and B matrices (full gradients, ~0.1 GB)

    model.print_trainable_parameters()
    # trainable params: 83,886,080 || all params: 6,830,690,304 || trainable%: 1.2280
    # (varies by target_modules and rank)

---

**Step 4: Set Up Paged Optimizer**

    import bitsandbytes as bnb

    # Paged AdamW — same as AdamW but optimizer states live in unified memory
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=2e-4,
    )

    # This creates momentum and variance buffers in CUDA unified memory:
    # They live in GPU VRAM during normal operation
    # They page to CPU RAM during memory spikes
    # PagedAdamW8bit also quantizes the optimizer states to INT8,
    # reducing their memory footprint by 4× compared to FP32 optimizer states

    Optimizer state memory comparison:
        Standard AdamW (FP32 states, 8.4M params):
            momentum:  8.4M × 4 bytes = 33.6 MB
            variance:  8.4M × 4 bytes = 33.6 MB
            Total: 67.2 MB

        PagedAdamW8bit (INT8 states):
            momentum:  8.4M × 1 byte  = 8.4 MB
            variance:  8.4M × 1 byte  = 8.4 MB
            Total: 16.8 MB   (+ negligible scale factors per block)

---

**Step 5: Data Preparation (Identical to Full Fine-Tuning and LoRA)**

    The data pipeline is completely unchanged from standard training.
    QLoRA doesn't affect how you tokenize, format, or batch your data.

    5a. Raw JSONL:
        {"instruction": "Summarize:", "input": "Long article...", "output": "Summary..."}

    5b. Chat template formatting:
        "<s>[INST] Summarize: Long article... [/INST] Summary... </s>"

    5c. Tokenization:
        [1, 518, 25580, 29962, 6264, 279, 675, 29901, ...]  (integer IDs)

    5d. Label masking:
        Input tokens: [-100, -100, ..., -100]  (ignored in loss)
        Output tokens: [actual token IDs]       (graded by loss)

    5e. Padding + attention mask, collated into batch tensors

    5f. batch = {
            "input_ids":      [batch_size, seq_len]
            "attention_mask": [batch_size, seq_len]
            "labels":         [batch_size, seq_len]
        }

---

**Step 6: Forward Pass — Where NF4 Dequantization Happens**

    outputs = model(**batch)

    During this forward pass, for each QLoRA-adapted linear layer:

    6a. The NF4 kernel is invoked (bitsandbytes custom CUDA kernel):
            - Reads 4-bit packed weights from GPU memory (fast: small data)
            - Reads INT8 block-scales, reads FP32 super-scales
            - Dequantizes in a fused GPU kernel: INT8_scale → FP32_scale → BF16_weight
            - Performs the matrix multiplication in BF16
            - Returns output in BF16

    6b. Simultaneously, the LoRA path computes A × x and B × (Ax) in BF16

    6c. Both paths' outputs are summed and passed to the next layer

    This dequantize-compute-discard pattern repeats for every linear layer in every forward pass.
    The 4-bit weights are NEVER stored as BF16 — they're reconstructed transiently.

    6d. Output logits: [batch_size, seq_len, vocab_size] in BF16

---

**Step 7: Loss Computation (Unchanged)**

    loss = CrossEntropyLoss(logits, labels)

    Only output positions (labels ≠ -100) contribute to the loss.
    The dequantization error in the frozen path shows up as a small constant bias
    in the loss — it doesn't prevent convergence.

---

**Step 8: Backward Pass — Where Quantization Meets Gradients**

    loss.backward()

    This is where QLoRA's design pays off most clearly:

    8a. Gradients flow backward through the output logits to the last layer

    8b. At each QLoRA layer:
        - Gradients DO flow through the dequantized W₀ (as if it were BF16)
          The backward pass through the frozen path uses the same dequantized weights
          that were used in the forward pass (held in memory for backward computation)
        - But W₀ itself accumulates NO gradient (requires_grad = False)
        - Gradients flow to LoRA A and B matrices (requires_grad = True)

    8c. The gradient for LoRA A:
        ∂Loss/∂A = (α/r) × Bᵀ × ∂Loss/∂output × xᵀ
        Shape: [r × d_in]  (e.g., [8 × 4096])

        The ∂Loss/∂output term comes from the frozen path flowing backward through
        the dequantized W₀. This is where quantization error can theoretically
        corrupt gradients — but NF4's low quantization error keeps this manageable.

    8d. Memory used during backward:
        LoRA gradients:     ~33 MB  (for rank=16, all-linear targets)
        Activation buffer:  varies (depends on gradient checkpointing settings)
        NO storage for W₀ gradients (they're never computed/stored)

---

**Step 9: Optimizer Step — Only LoRA Parameters Update**

    optimizer.step()

    The paged AdamW8bit optimizer updates only the LoRA A and B matrices:

    For each LoRA parameter θ:
        g  = ∂Loss/∂θ                            (BF16 gradient)
        m  = β₁ × m + (1-β₁) × g                (momentum, stored INT8, dequant for update)
        v  = β₂ × v + (1-β₂) × g²               (variance, stored INT8, dequant for update)
        θ ← θ - lr × m̂ / (√v̂ + ε)              (update in BF16)

    If a memory spike occurs during this step:
        The paged memory for m and v gets offloaded to CPU RAM
        The update computation pauses briefly
        After the spike, states page back to GPU
        Training continues

    optimizer.zero_grad()                        # Clear the ~33 MB of LoRA gradients

---

**Step 10: Gradient Checkpointing Integration**

    QLoRA strongly recommends enabling gradient checkpointing:

    model.gradient_checkpointing_enable()

    Without gradient checkpointing:
        All layer activations must be kept in memory during the forward pass
        so they're available for the backward pass.
        For a 7B model: activations can easily reach 10-30 GB for longer sequences.

    With gradient checkpointing:
        Only checkpoint activations are kept (every N layers, configurable).
        During backward pass, the sections between checkpoints are recomputed.
        Memory reduction: 4-8× for activations.
        Compute cost: ~30-35% more compute per step (each non-checkpointed activation
        is computed twice: once in forward, once in backward recomputation).

    For QLoRA, the interaction with quantization:
        During recomputation, the NF4 dequantization happens AGAIN for those layers.
        This is handled correctly by bitsandbytes — the same NF4 → BF16 kernel runs.
        The recomputed activations are numerically identical because the NF4 weights
        are deterministic (no randomness in dequantization).

---

**Step 11: Save — Only the Adapter (Identical to LoRA)**

    After training, save ONLY the LoRA adapter:

    model.save_pretrained("./qlora-adapter/")

    Saved files:
        adapter_model.safetensors    ~33-300 MB (all A and B matrices, in BF16)
        adapter_config.json          ~1 KB

    The NF4 base model is NOT saved by default — it's available on HuggingFace Hub.
    Only the tiny adapter delta is saved.

    adapter_config.json:
    {
        "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }

---

**Step 12: Inference — Merge or Keep Separate**

    Option A: Keep adapter separate (for adapter swapping)

        base = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,   # Still use 4-bit at inference
        )
        model = PeftModel.from_pretrained(base, "./qlora-adapter/")
        # Inference in 4-bit base + BF16 LoRA: same memory as training, minimal overhead

    Option B: Merge adapter into base model (for zero-overhead inference)

        # Step 1: Load model in BF16 (not 4-bit) for merging
        base = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,       # Full precision for merge
        )
        model = PeftModel.from_pretrained(base, "./qlora-adapter/")

        # Step 2: Merge: W_merged = W₀_bf16 + (α/r) · B × A
        model = model.merge_and_unload()

        # Step 3: Optionally re-quantize for deployment
        # Or save as BF16 for inference on larger GPUs
        model.save_pretrained("./merged-model-bf16/")

    Note on merge-then-requantize:
        It's common to merge the BF16 LoRA into the BF16 base, then re-quantize to GGUF
        or GPTQ for efficient deployment. The merged model has the adapter's learning
        "baked in" and can be quantized like any standard model afterward.

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                          MEMORY BREAKDOWN — QLORA VS. LORA VS. FULL FINE-TUNING
                               (Concrete Numbers for a 7B and 70B Model)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### 7B Model (e.g., LLaMA-2-7B), batch=4, seq=512, rank=16

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   MEMORY COMPONENT            Full FT         LoRA (BF16)     QLoRA (NF4)           │
    │   ────────────────────        ───────         ───────────     ───────────           │
    │   Base weights                14 GB           14 GB           3.5 GB ✓              │
    │   Gradients (base)            14 GB           0 GB ✓          0 GB ✓                │
    │   Optimizer states (base)     56 GB           0 GB ✓          0 GB ✓                │
    │   LoRA adapter weights        N/A             ~0.1 GB         ~0.1 GB               │
    │   LoRA gradients              N/A             ~0.1 GB         ~0.1 GB               │
    │   LoRA optimizer states       N/A             ~0.3 GB         ~0.08 GB ✓            │
    │   Activations (grad. ckpt)    ~5 GB           ~5 GB           ~3.5 GB ✓             │
    │   ───────────────────────────────────────────────────────────────────────────────   │
    │   Total                       ~89-109 GB      ~19-23 GB       ~7-11 GB ✓            │
    │                                                                                     │
    │   Required hardware            8× A100         1-2× A100      1× RTX 3090/4090      │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘


### 70B Model (e.g., LLaMA-2-70B), batch=2, seq=512, rank=16

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   MEMORY COMPONENT            Full FT         LoRA (BF16)     QLoRA (NF4)           │
    │   ────────────────────        ───────         ───────────     ───────────           │
    │   Base weights                140 GB          140 GB          35 GB ✓               │
    │   Gradients (base)            140 GB          0 GB ✓          0 GB ✓                │
    │   Optimizer states (base)     560 GB          0 GB ✓          0 GB ✓                │
    │   LoRA adapter weights        N/A             ~0.5 GB         ~0.5 GB               │
    │   LoRA gradients              N/A             ~0.5 GB         ~0.5 GB               │
    │   LoRA optimizer states       N/A             ~2.0 GB         ~0.5 GB ✓             │
    │   Activations (grad. ckpt)    ~20 GB          ~15 GB          ~10 GB ✓              │
    │   ───────────────────────────────────────────────────────────────────────────────   │
    │   Total                       ~860 GB+        ~160 GB         ~46 GB ✓              │
    │                                                                                     │
    │   Required hardware            16-20× A100     2× A100 (80G)  1× A100 (80G) ✓       │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘

    QLoRA is what makes 70B model training accessible to teams without massive GPU clusters.

---


═══════════════════════════════════════════════════════════════════════════════════════════════
                             QLORA VISUAL DIAGRAMS — COMPLETE BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════════════════════


    ──────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 1: HOW NF4 COMPRESSES A WEIGHT MATRIX
    ──────────────────────────────────────────────────────────────────────────────────────────

    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                        │
    │   ORIGINAL WEIGHT MATRIX W_q: [4096 × 4096]                                            │
    │   Stored in BF16: 4096 × 4096 × 2 bytes = 33,554,432 bytes = 32 MB                     │
    │                                                                                        │
    │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
    │   │ -0.032  0.018  -0.007  0.041  -0.089  0.003  0.027  ...  (BF16 floats)         │   │
    │   │  0.011 -0.045   0.033 -0.019   0.056 -0.011  0.008  ...                        │   │
    │   │  ...                                                                           │   │
    │   └────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                        │
    │                              ↓ QUANTIZE (block-wise, block_size=64)                    │
    │                                                                                        │
    │   QUANTIZED WEIGHT MATRIX W_q in NF4:                                                  │
    │   NF4 codes: 4096 × 4096 × 0.5 bytes  =  8 MB   (4 bits per weight)                    │
    │   Scales:    16,777,216 / 64 blocks × 4 bytes = 1 MB  (FP32 per block)                 │
    │   After DQ:  scales compressed to ~0.25 MB in INT8                                     │
    │   Total:     8 + 0.25 = ~8.25 MB  (vs 32 MB original) = 3.88× compression              │
    │                                                                                        │
    │   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
    │   │ Block 0:  scale=0.089  codes=[7, 11, 8, 13, 4, 9, 11, 7, ...]  (64 values)      │  │
    │   │ Block 1:  scale=0.056  codes=[9, 6, 11, 8, 12, 8, 9, 8, ...]   (64 values)      │  │
    │   │ ...       (262,144 total blocks for this matrix)                                │  │
    │   └─────────────────────────────────────────────────────────────────────────────────┘  │
    │                                                                                        │
    │                              ↓ DEQUANTIZE (during forward pass)                        │
    │                                                                                        │
    │   RECOVERED W_q in BF16:                                                               │
    │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
    │   │ -0.031  0.017  -0.006  0.041  -0.089  0.003  0.027  ...  (BF16, approximate)   │   │
    │   │  0.011 -0.044   0.033 -0.019   0.057 -0.011  0.008  ...                        │   │
    │   │  ...           (small quantization errors — values ~0.001 off)                 │   │
    │   └────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                        │
    │   Error: each weight ≈ ±0.001 off from original. Small enough to not destroy learning. │
    │                                                                                        │
    └────────────────────────────────────────────────────────────────────────────────────────┘


    ──────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 2: INT4 vs. NF4 — WHY NF4 IS BETTER FOR NEURAL NETWORKS
    ──────────────────────────────────────────────────────────────────────────────────────────

    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                        │
    │   Weight distribution (from a real LLM layer, normalized):                             │
    │                                                                                        │
    │                          ████████                                                      │
    │                       █████████████                                                    │
    │                     █████████████████                                                  │
    │                   █████████████████████                                                │
    │                 ██████████████████████████                                             │
    │             ████████████████████████████████                                           │
    │        ─────────────────────────────────────────────                                   │
    │       -3.0   -2.0   -1.0    0.0    1.0    2.0    3.0                                   │
    │                                                                                        │
    │   INT4 quantization levels (16 evenly spaced from -8 to 7, normalized to [-1,1]):      │
    │                                                                                        │
    │     │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │                      │   
    │  -1.0         -0.5         0.0         0.5         1.0         2.0                     │
    │                                                                                        │
    │   Problem: equally-spaced levels waste resolution on extreme values where              │
    │   almost no weights live. Most weight values are represented by only 4-6 levels.       │
    │                                                                                        │
    │   NF4 quantization levels (quantile-based, more levels near 0):                        │
    │                                                                                        │
    │   │ │ │  │  │   │    │         │    │   │  │  │ │ │                                    │
    │  -1.0   -0.5           0.0           0.5       1.0                                     │
    │                                                                                        │
    │   NF4 concentrates quantization levels where most weights actually live.               │
    │   Result: same 4 bits, dramatically lower average quantization error.                  │
    │                                                                                        │
    │   Quantization SNR comparison (higher = better quality):                               │
    │       INT4:  ~21.0 dB                                                                  │
    │       NF4:   ~24.5 dB    (+3.5 dB, roughly 1.5× better signal/noise ratio)             │
    │                                                                                        │
    └────────────────────────────────────────────────────────────────────────────────────────┘


    ──────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 3: QLORA LAYER ARCHITECTURE — FROZEN NF4 BASE + BF16 LORA ADAPTERS
    ──────────────────────────────────────────────────────────────────────────────────────────

    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                       │
    │                          QLoRA ATTENTION LAYER — FULL DETAIL                          │
    │                                                                                       │
    │                                                                                       │
    │                          x (input hidden state, BF16 [4096])                          │
    │                          │                                                            │
    │                          ├─────────────────────────────────────────┐                  │
    │                          │                                         │                  │
    │                          │   FROZEN NF4 PATH                       │  LoRA BF16 PATH  │
    │                          │                                         │                  │
    │                          ▼                                         ▼                  │
    │            ┌─────────────────────────────┐           ┌────────────────────┐           │
    │            │ Step 1: Load NF4 weights    │           │ A_q in BF16        │           │
    │            │ from GPU memory (8 MB)      │           │ [8 × 4096]         │           │
    │            │                             │           │ requires_grad=True │           │
    │            │ Step 2: Dequantize to BF16  │           │ No quantization    │           │
    │            │ INT8_scale → FP32_scale     │           └─────────┬──────────┘           │
    │            │ NF4_code  → float in [-1,1] │                     │                      │
    │            │ × FP32_scale → BF16 weight  │                     ▼                      │
    │            │ Output: W_q_bf16 (32 MB,    │           ┌────────────────────┐           │
    │            │         TEMPORARY)          │           │ x_down = A_q × x   │           │
    │            │                             │           │ [8]  (bottleneck)  │           │
    │            │ Step 3: Matrix multiply     │           └─────────┬──────────┘           │
    │            │ q_frozen = W_q_bf16 × x     │                     │                      │
    │            │ [4096 × 4096] × [4096]      │                     ▼                      │
    │            │ → q_frozen [4096]           │           ┌────────────────────┐           │
    │            │                             │           │ B_q in BF16        │           │
    │            │ Step 4: Discard W_q_bf16    │           │ [4096 × 8]         │           │
    │            │ (temporary BF16 not stored) │           │ requires_grad=True │           │
    │            │ NF4 remains as source       │           └─────────┬──────────┘           │
    │            │ requires_grad = False       │                     │                      │
    │            └──────────────┬──────────────┘                     ▼                      │
    │                           │                         ┌────────────────────┐            │
    │                           │                         │ x_up = B_q × x_down│            │
    │                           │                         │ [4096] (expanded)  │            │
    │                           │                         │                    │            │
    │                           │                         │ q_lora = (α/r)×x_up│            │
    │                           │                         └─────────┬──────────┘            │
    │                           │                                   │                       │
    │                           ▼                                   ▼                       │
    │                       ┌─────────────────────────────────────────┐                     │
    │                       │             q = q_frozen + q_lora       │                     │
    │                       │             [4096], BF16                │                     │
    │                       └─────────────────────────────────────────┘                     │
    │                                                                                       │
    │   ═══════════════════════════════════════════════════════════════                     │
    │   MEMORY AT STEADY STATE:                                                             │
    │   W_q (NF4):     8 MB   (always in GPU VRAM)                                          │
    │   W_q (BF16):    32 MB  (TEMPORARY — created and discarded each forward pass)         │
    │   A_q, B_q:      0.13 MB (always in GPU VRAM, BF16, trainable)                        │
    │   ═══════════════════════════════════════════════════════════════                     │
    │                                                                                       │
    └───────────────────────────────────────────────────────────────────────────────────────┘


    ──────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 4: BACKWARD PASS — GRADIENT FLOW IN QLORA
    ──────────────────────────────────────────────────────────────────────────────────────────

    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                        │
    │                          QLORA BACKWARD PASS GRADIENT FLOW                             │
    │                                                                                        │
    │   ∂Loss/∂output (from next layer)                                                      │
    │          │                                                                             │
    │          ▼                                                                             │
    │    ┌─────────────────────────────────────────────────────────────────┐                 │
    │    │            ∂Loss/∂q = ∂Loss/∂output (combined output gradient)  │                 │
    │    └─────────────────────────────────────────────────────────────────┘                 │
    │          │                                                                             │
    │          ├─────────────────────────────────────────────────┐                           │
    │          │                                                 │                           │
    │          │   FROZEN NF4 PATH (backward)                    │  LoRA BF16 PATH (back)    │
    │          │                                                 │                           │
    │          ▼                                                 ▼                           │
    │   Gradient flows through dequantized W_q                ∂Loss/∂B_q:                    │
    │   (W_q was held in BF16 from forward pass               = ∂Loss/∂q × (A_q×x)ᵀ          │
    │   for this exact purpose)                                stored ✓ (~0.13 MB)           │
    │                                                                                        │
    │   ∂Loss/∂W_q:  NOT computed (requires_grad=False)       ∂Loss/∂A_q:                    │
    │   NOT stored   ← this is where the 14 GB gradient       = (α/r) × B_qᵀ × ∂Loss/∂q      │ 
    │   savings come from                                       × xᵀ                         │
    │                                                          stored ✓ (~0.13 MB)           │
    │          │                                                                             │
    │          ▼                                                                             │
    │   ∂Loss/∂x (passed to previous layer for chain rule)                                   │
    │   = W_q_bf16ᵀ × ∂Loss/∂q_frozen                                                        │
    │     + (α/r) × A_qᵀ × B_qᵀ × ∂Loss/∂q_lora                                              │
    │                                                                                        │
    │   After computing ∂Loss/∂x, the dequantized W_q_bf16 is DISCARDED.                     │
    │   (gradient checkpointing may require re-dequantizing for earlier layers)              │
    │                                                                                        │
    │   Total gradient storage: ~33-300 MB (LoRA only, no base model gradients)              │
    │                                                                                        │
    └────────────────────────────────────────────────────────────────────────────────────────┘


    ──────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 5: PAGED OPTIMIZER — MEMORY UNDER PRESSURE
    ──────────────────────────────────────────────────────────────────────────────────────────

    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                        │
    │   SCENARIO: Training on RTX 3090 (24 GB), 7B model, QLoRA                              │
    │                                                                                        │
    │                                                                                        │
    │   Normal batch (seq_len=512):                                                          │
    │                                                                                        │
    │   GPU VRAM [24 GB]                               CPU RAM [64 GB]                       │
    │   ┌──────────────────────────────────────┐       ┌──────────────────────────────────┐  │
    │   │ ■ NF4 base weights:      3.5 GB      │       │                                  │  │
    │   │ ■ LoRA adapters:         0.1 GB      │       │   Paged optimizer states         │  │
    │   │ ■ Paged opt. states:     0.08 GB     │       │   (backup — mostly idle)         │  │
    │   │ ■ Layer norm (FP32):     0.5 GB      │       │   0.08 GB                        │  │
    │   │ ■ Activations:           4.0 GB      │       │                                  │  │
    │   │ ■ LoRA gradients:        0.15 GB     │       │                                  │  │
    │   │                                      │       │                                  │  │
    │   │   Used: ~8.3 GB  / 24 GB             │       │                                  │  │
    │   └──────────────────────────────────────┘       └──────────────────────────────────┘  │
    │                                                                                        │
    │                                                                                        │
    │   Spike batch (seq_len=2048, gradient checkpointing recompute):                        │
    │                                                                                        │
    │   GPU VRAM [24 GB]                               CPU RAM [64 GB]                       │
    │   ┌──────────────────────────────────────┐       ┌──────────────────────────────────┐  │
    │   │ ■ NF4 base weights:      3.5 GB      │       │                                  │  │
    │   │ ■ LoRA adapters:         0.1 GB      │       │  ← PAGE OUT: optimizer states    │  │
    │   │   Paged opt. (evicted)   0.00 GB  ←──┼───────┼──ₒᵤₜ 0.08 GB moved here           │  │
    │   │ ■ Layer norm (FP32):     0.5 GB      │       │   GPU freed 0.08 GB              │  │
    │   │ ■ Activations (spike):   18.0 GB     │       │                                  │  │
    │   │ ■ LoRA gradients:        0.15 GB     │       │                                  │  │
    │   │                                      │       │                                  │  │
    │   │   Used: ~22.3 GB / 24 GB             │       │                                  │  │
    │   │   No OOM! ✓                          │       │                                  │  │
    │   └──────────────────────────────────────┘       └──────────────────────────────────┘  │
    │                                                                                        │
    │   After spike: optimizer states page back to GPU VRAM automatically.                   │
    │   Training continues without interruption.                                             │
    │                                                                                        │
    └────────────────────────────────────────────────────────────────────────────────────────┘


    ──────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 6: QLORA COMPLETE SYSTEM — ALL FOUR INNOVATIONS TOGETHER
    ──────────────────────────────────────────────────────────────────────────────────────────

    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                       │
    │                       QLoRA COMPLETE SYSTEM OVERVIEW                                  │
    │                                                                                       │
    │                                                                                       │
    │    ┌─────────────────────────────────────────────────────────────────────────────┐    │
    │    │                         WHAT LIVES IN GPU VRAM                              │    │
    │    │                                                                             │    │
    │    │   ┌───────────────────────┐   ┌─────────────────┐   ┌────────────────────┐  │    │
    │    │   │  Base Model (NF4)     │   │  LoRA Adapters  │   │  Layer Norms (FP32)│  │    │
    │    │   │                       │   │  (BF16)         │   │                    │  │    │
    │    │   │  7B weights × 4 bits  │   │  A and B mats   │   │  Cast to FP32 for  │  │    │
    │    │   │  = ~3.5 GB            │   │  ~0.1 GB        │   │  numerical stable  │  │    │
    │    │   │                       │   │  requires_grad  │   │  normalization     │  │    │
    │    │   │  Block-wise scales    │   │  = True         │   │  ~0.5 GB           │  │    │
    │    │   │  (INT8 + FP32)        │   │                 │   │                    │  │    │
    │    │   │  ~0.1 GB (after DQ)   │   │                 │   │                    │  │    │
    │    │   │                       │   │                 │   │                    │  │    │
    │    │   │  requires_grad=False  │   │                 │   │  requires_grad     │  │    │
    │    │   │                       │   │                 │   │  = False           │  │    │
    │    │   └───────────────────────┘   └─────────────────┘   └────────────────────┘  │    │
    │    │                                                                             │    │
    │    │   ┌──────────────────────────────────────────────────────────────────────┐  │    │
    │    │   │  Paged Optimizer States (in CUDA unified memory)                     │  │    │
    │    │   │  INT8 quantized momentum + variance for LoRA params                  │  │    │
    │    │   │  ~0.08 GB  (pages to CPU RAM during memory pressure)                 │  │    │
    │    │   └──────────────────────────────────────────────────────────────────────┘  │    │
    │    │                                                                             │    │
    │    │   ┌──────────────────────────────────────────────────────────────────────┐  │    │
    │    │   │  Activations (with gradient checkpointing)                           │  │    │
    │    │   │  ~3-6 GB  (depends on sequence length and batch size)                │  │    │
    │    │   └──────────────────────────────────────────────────────────────────────┘  │    │
    │    │                                                                             │    │
    │    └─────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                       │
    │    TOTAL: ~7-12 GB  (for 7B model)   vs.  89-109 GB  (full fine-tuning)               │
    │                                                                                       │
    │                                                                                       │
    │                         FORWARD PASS DATA FLOW                                        │
    │                                                                                       │
    │    Input tokens                                                                       │
    │         │                                                                             │
    │         ▼                                                                             │
    │    Embeddings (NF4 → dequant BF16 → embedding lookup → BF16 output)                   │
    │         │                                                                             │
    │         ▼  ×32 layers                                                                 │
    │    ┌────────────────────────────────────────────────────────────────┐                 │
    │    │  Layer Norm: FP32 computation, BF16 input/output               │                 │
    │    │  Self-Attention:                                               │                 │
    │    │    Q = (dequant(W_q_nf4) × x) + (α/r) × B_q(A_q × x)           │                 │
    │    │    K = (dequant(W_k_nf4) × x) + (α/r) × B_k(A_k × x)           │                 │
    │    │    V = (dequant(W_v_nf4) × x) + (α/r) × B_v(A_v × x)           │                 │
    │    │    O = (dequant(W_o_nf4) × attn) + (α/r) × B_o(A_o × attn)     │                 │
    │    │  Layer Norm: FP32                                              │                 │
    │    │  MLP: (dequant NF4 × x) + LoRA (if target_modules=all-linear)  │                 │
    │    └────────────────────────────────────────────────────────────────┘                 │
    │         │                                                                             │
    │         ▼                                                                             │
    │    LM Head (NF4 → dequant BF16 → logits)                                              │
    │         │                                                                             │
    │         ▼                                                                             │
    │    Loss → Backward → Update LoRA A and B only                                         │
    │                                                                                       │
    └───────────────────────────────────────────────────────────────────────────────────────┘


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                     QLORA HYPERPARAMETERS IN DEPTH
                   (What to Set, Why, and What Breaks If You Get It Wrong)
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### BitsAndBytes Quantization Config

    bnb_4bit_quant_type: "nf4" vs "fp4"

        "nf4":  NormalFloat4 — use this. Optimal for normally-distributed weights.
                Quantile-based, information-theoretically superior to uniform quantization.
                The standard for QLoRA fine-tuning.

        "fp4":  FP4 (floating-point 4-bit) — a floating-point format with tiny mantissa.
                Used more in inference scenarios. Slightly lower quality for training.
                Avoid unless you have a specific reason to use it.


    bnb_4bit_compute_dtype: torch.bfloat16 vs torch.float16

        BF16:   Wider dynamic range (same exponent bits as FP32), lower precision.
                Standard for training on Ampere (A100, RTX 3090) and newer.
                Handles gradient explosions better.

        FP16:   Higher precision, smaller range. Can overflow during training.
                Use BF16 unless your hardware doesn't support it (pre-Ampere).


    bnb_4bit_use_double_quant: True vs False

        True:   Enable double quantization — saves ~375 MB on a 7B model.
                Recommended: almost always worth it, negligible quality impact.

        False:  Disable — marginally simpler dequantization, saves a tiny compute cost.
                Only disable if you're debugging or have plenty of GPU memory.


### LoRA Config for QLoRA

    Rank (r): Same considerations as standard LoRA, but start with higher ranks
              in QLoRA because the quantized base has more noise to overcome.

        QLoRA recommended ranges:
        r=16:   Good starting point for instruction tuning on 7B-13B models
        r=32:   Better for larger domain gaps or models with more capacity
        r=64:   Use when r=32 still underperforms; high memory cost
        r=8:    Only for very simple tasks or extreme memory constraints

    Alpha (α): Set equal to rank (α=r) or double (α=2r).
               Common for QLoRA: lora_alpha = 2 × r


    target_modules:
        "all-linear":                   Most expressive, highest memory cost
                                        Recommended for QLoRA on 7B (still fits easily)
        ["q_proj","v_proj"]:            Minimal — good for memory-constrained setups
        ["q_proj","k_proj","v_proj","o_proj"]:  Standard — good balance


### Learning Rate

    QLoRA uses the same learning rates as LoRA (higher than full fine-tuning):
        Typical range: 1e-4 to 3e-4
        Common default: 2e-4

    Because QLoRA's base model has quantization error, gradients may be noisier.
    Some practitioners use slightly lower learning rates (1e-4) for QLoRA vs LoRA.
    But this is a minor effect — start with 2e-4 and tune if needed.


### Batch Size and Gradient Accumulation

    QLoRA enables smaller per-device batch sizes due to memory savings:

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   BATCH SIZE COMPARISON (7B model, RTX 3090 24GB, seq_len=512)                   │
    │                                                                                  │
    │   Method              Max batch/GPU   Grad Accum   Effective Batch               │
    │   ──────              ─────────────   ──────────   ───────────────               │
    │   Full FT (BF16)      N/A (OOM)       N/A          N/A                           │
    │   LoRA (BF16)         2-4             8            16-32                         │
    │   QLoRA (NF4)         4-8             4-8          16-64                         │
    │                                                                                  │
    │   QLoRA's smaller base model footprint enables larger effective batches          │
    │   or longer sequences at the same GPU tier.                                      │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘


### Gradient Checkpointing

    Always enable gradient checkpointing with QLoRA:

    model.gradient_checkpointing_enable()

    Without it, activations for all 32 layers must reside in memory simultaneously.
    At sequence length 512 with batch size 4, this can be 8-15 GB of activation memory alone.

    With gradient checkpointing:
        Activations stored: every N layers (default: every layer, recomputing N-1 layers)
        Memory cost: ~3-5 GB instead of 8-15 GB
        Compute cost: ~30-35% more compute per step

    For QLoRA, the recomputation includes re-running the NF4 dequantization.
    This is deterministic and correct — no numerical issues.


---

{{QLORA_IMAGE}}

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                  QLORA QUALITY — WHAT'S LOST, AND WHEN IT MATTERS
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### The Quality Stack

    Full Fine-Tuning on BF16 base         →  Baseline (100% quality)
    LoRA on BF16 base                     →  95-99% of full FT
    QLoRA (NF4 base + BF16 LoRA)         →  94-99% of full FT
                                              (~1-3% below LoRA in most benchmarks)

The quality gap between LoRA and QLoRA is small — typically 1-4 points on standard benchmarks
depending on model size, task, and rank. The original QLoRA paper showed their Guanaco models
trained with QLoRA matched or exceeded ChatGPT on Vicuna benchmarks, despite running on
a fraction of the hardware.


### Where QLoRA Closes the Gap

    - Larger models: 70B QLoRA ≈ 70B LoRA (quantization error is proportionally smaller
      relative to the model's representational capacity)
    - Higher ranks: r=64 QLoRA compensates more for quantization noise
    - More training data: longer training allows adapters to compensate for base model noise
    - Lower-stake tasks: instruction following, summarization, classification
      (where exact precision matters less)


### Where the Gap Widens

    - Smaller models (7B): quantization error is larger relative to model capacity
    - Mathematical reasoning: requires exact weight precision for multi-step chains
    - Coding tasks with strict syntax: quantization errors can subtly corrupt code logic
    - Very low ranks (r=4): adapters have too few parameters to compensate for base noise
    - Short fine-tuning runs: not enough gradient steps for adapters to adapt to NF4 noise


### Practical Decision Guide

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   USE QLoRA WHEN:                        USE LoRA (BF16) WHEN:                   │
    │   ─────────────                          ─────────────────────                   │
    │   GPU < 24 GB                            GPU ≥ 24 GB for 7B model                │
    │   Model > 13B                            Quality difference matters              │
    │   Consumer GPU (3090, 4090)              Mathematical/coding tasks               │
    │   Budget constraints                     Production deployment                   │
    │   Experimenting / prototyping            Rank < 8 (less buffer for noise)        │
    │   Instruction tuning / chat tasks                                                │
    │   70B model on single A100 80G                                                   │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                  QLORA FAILURE MODES & COMMON PITFALLS
    ═══════════════════════════════════════════════════════════════════════════════════════════════


**1. Using FP16 compute dtype instead of BF16**

    bnb_4bit_compute_dtype=torch.float16    ← can cause NaN/Inf during training

    FP16 has a narrow dynamic range (max ~65,504). Gradients in LLM training can exceed
    this range. BF16 shares FP32's exponent width and handles this safely.
    Always use BF16 as compute dtype with QLoRA.


**2. Forgetting prepare_model_for_kbit_training()**

    Skipping this step means:
    - Layer norms stay in BF16 (numerical instability)
    - Gradient checkpointing not properly configured for quantized layers
    - Potential NaN losses or very slow convergence

    Always call prepare_model_for_kbit_training() before applying LoRA.


**3. Training with Flash Attention disabled on long sequences**

    Flash Attention is especially important for QLoRA because:
    - Long sequences are often the reason you need QLoRA (memory constraints)
    - Standard attention materializes the full attention matrix [seq × seq] in memory
    - Flash Attention avoids this entirely, enabling much longer sequences

    model = AutoModelForCausalLM.from_pretrained(
        ...,
        attn_implementation="flash_attention_2",    # Requires flash-attn package
    )


**4. Merging the adapter while the base is still 4-bit**

    # WRONG — merges adapter into the quantized base (poor quality):
    model.merge_and_unload()   ← when model is still in NF4

    # CORRECT — reload base in BF16, then merge:
    base = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, "./adapter/")
    model = model.merge_and_unload()   # Now merges cleanly in BF16

    Merging into a quantized base produces degraded results because the merge
    operation W = W₀ + (α/r)·BA runs on quantized W₀ values, amplifying errors.


**5. Wrong bnb_4bit_quant_type**

    Using "fp4" instead of "nf4" for training is a common mistake.
    NF4 was specifically designed and shown to be superior for neural network weight
    distributions. FP4 is mainly an inference format. Use NF4 for fine-tuning.


**6. Not enabling gradient checkpointing**

    Without gradient checkpointing, activations accumulate across all 32 layers.
    This often causes OOM even with QLoRA because activation memory is not reduced
    by quantization (activations are always in BF16).

    model.gradient_checkpointing_enable()   # Always include this


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                          SUMMARY MENTAL MODEL
    ═══════════════════════════════════════════════════════════════════════════════════════════════


    FULL FINE-TUNING          LoRA (BF16 BASE)          QLoRA (NF4 BASE + BF16 LoRA)

    ┌───────────────────┐     ┌─────────────────────┐     ┌────────────────────────────┐
    │ ██ ALL 7B weights │     │ ░░ 7B FROZEN (BF16) │     │ ◻◻ 7B FROZEN (NF4, 4-bit) │
    │ ██ updated (BF16) │     │ ░░                  │     │ ◻◻ (4× smaller)           │
    │ ██                │     │ ░░ + ██ tiny adapts │     │ ◻◻ + ██ tiny LoRA (BF16)  │
    │ ██                │     │ ░░ + ██ (BF16, ~0.1%│     │ ◻◻ + ██ (~0.1%)           │
    └───────────────────┘     └─────────────────────┘     └────────────────────────────┘

    VRAM:    89-109 GB         16-24 GB                    7-12 GB
    Quality: ████████ 100%     ██████▓  95-99%             ██████░  94-99%
    Hardware: 8×A100           1-2×A100                    1×RTX 3090 / 1×A100 80G
    (7B model)


The Four Things QLoRA Does:

    1. NF4 quantization: pack 7B weights into 3.5 GB instead of 14 GB
                         (information-optimal: more quantization levels near zero,
                          where most weights live)

    2. Double quantization: quantize the block-scale factors themselves
                            (saves an extra ~375 MB, marginal quality impact)

    3. Paged optimizers: store optimizer states in unified memory
                         (prevents OOM during memory spikes, enables longer sequences)

    4. BF16 LoRA adapters on top: train normally in BF16
                                   (dequantize NF4 → BF16 on-the-fly for compute,
                                    gradients only for tiny A and B matrices)


QLoRA didn't invent new model architectures or new training objectives.
It took existing tools — quantization, LoRA, paging — and combined them in exactly the right way
to push the frontier of what's trainable on a single GPU.

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect                       | Full Fine-Tuning         | LoRA (BF16 base)         | QLoRA (NF4 base)          |
|------------------------------|--------------------------|--------------------------|---------------------------|
| Trainable Parameters         | 100% (~7B)               | ~0.1-1% (~8M)            | ~0.1-1% (~8M)             |
| Base Model Storage Dtype     | BF16 (16-bit)            | BF16 (16-bit)            | NF4 (4-bit)               |
| Compute Dtype                | BF16                     | BF16                     | BF16 (dequant on-the-fly) |
| Base Model Memory (7B)       | 14 GB                    | 14 GB                    | ~3.5 GB ✓                 |
| Total Training Memory (7B)   | 89-109 GB                | 16-24 GB                 | 7-12 GB ✓                 |
| Total Training Memory (70B)  | 860+ GB                  | ~160 GB                  | ~46 GB ✓                  |
| Required Hardware (7B)       | 8× A100                  | 1-2× A100                | 1× RTX 3090/4090 ✓        |
| Required Hardware (70B)      | 16-20× A100              | 2× A100 (80G)            | 1× A100 80G ✓             |
| Optimizer States             | 56 GB (FP32 Adam)        | ~67 MB (LoRA only)       | ~17 MB (paged INT8) ✓     |
| Gradient Memory              | ~14 GB                   | ~33-300 MB               | ~33-300 MB                |
| Quantization Error           | None                     | None                     | Small (NF4 bounded)       |
| Training Speed               | Slowest                  | Fast                     | ~30-50% slower than LoRA  |
| Inference Overhead           | None                     | None (after merge)       | None (after merge)        |
| Checkpoint Size              | 14-140 GB                | 20-300 MB                | 20-300 MB                 |
| Quality vs. Full FT          | Baseline (100%)          | 95-99%                   | 94-99%                    |
| Merge to BF16 Possible?      | N/A                      | Yes (direct)             | Yes (reload BF16 + merge) |
| Key Extra Dependencies       | —                        | peft                     | peft, bitsandbytes        |
| Key Hyperparameters          | lr, epochs, batch        | + rank, alpha, targets   | + quant_type, double_quant, paged_opt |
| Typical Learning Rate        | 1e-6 to 5e-5             | 1e-4 to 3e-4             | 1e-4 to 2e-4              |
| Block-wise Quantization      | N/A                      | N/A                      | 64 weights/block          |
| Double Quantization Savings  | N/A                      | N/A                      | ~375 MB on 7B model       |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "QLoRA Full Setup — 4-Bit Base + LoRA Adapters": {
        "description": "Complete QLoRA configuration: NF4 quantization, double quant, and LoRA adapters",
        "runnable": False,
        "code": '''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# ─── Step 1: Configure NF4 Quantization ───────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Enable 4-bit storage
    bnb_4bit_quant_type="nf4",                  # NormalFloat4 (not INT4, not FP4)
    bnb_4bit_compute_dtype=torch.bfloat16,      # Compute in BF16 (not FP16!)
    bnb_4bit_use_double_quant=True,             # Quantize scale factors too (~375 MB saved)
)

# ─── Step 2: Load Model in 4-Bit ──────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                          # Assign layers to GPU(s) automatically
    attn_implementation="flash_attention_2",    # Recommended: saves activation memory
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# ─── Step 3: Prepare for k-bit Training ───────────────────────────────────────
# Casts layer norms to FP32, enables gradient checkpointing for quantized layers
model = prepare_model_for_kbit_training(model)

# ─── Step 4: Apply LoRA ───────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                                       # Rank — higher than standard LoRA for NF4 noise
    lora_alpha=32,                              # Scaling factor (= 2 × rank)
    target_modules="all-linear",               # All linear layers — recommended for QLoRA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~83M || all params: ~6.8B || trainable%: ~1.22%
'''
    },

    "Paged Optimizer Setup": {
        "description": "Configure paged AdamW optimizer to handle memory spikes without OOM",
        "runnable": False,
        "code": '''import bitsandbytes as bnb

# ─── Paged AdamW — INT8 optimizer states in CUDA unified memory ───────────────
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
)
# Optimizer states:
#   - Stored in CUDA unified memory (pages to CPU RAM during GPU pressure)
#   - Quantized to INT8 (4× smaller than FP32 Adam states)
#   - For 83M LoRA params: ~17 MB total (vs ~670 MB FP32 Adam)

# Alternative: PagedAdamW32bit (full FP32 precision, but larger, still paged)
optimizer_fp32 = bnb.optim.PagedAdamW32bit(
    model.parameters(),
    lr=2e-4,
)
# Use 32-bit version if you see training instability with 8-bit optimizer
'''
    },

    "QLoRA Training with SFTTrainer (Recommended)": {
        "description": "Full training loop using TRL's SFTTrainer — handles all QLoRA details automatically",
        "runnable": False,
        "code": '''from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl", split="train")

training_args = SFTConfig(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,              # Can be higher than full FT / standard LoRA
    gradient_accumulation_steps=4,             # Effective batch = 4 × 4 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,                                  # BF16 compute (matches bnb_4bit_compute_dtype)
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    gradient_checkpointing=True,               # Essential for QLoRA — reduces activation memory
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Recommended for newer PyTorch
    max_seq_length=2048,
    packing=False,                             # Set True for short examples (efficiency)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./qlora-output/final")  # Saves ONLY the adapter (~33-300 MB)
'''
    },

    "Dequantize + Merge Adapter to BF16": {
        "description": "Correctly merge a QLoRA adapter back into a full-precision base model",
        "runnable": False,
        "code": '''from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

# ─── Step 1: Reload base model in BF16 (NOT 4-bit) ───────────────────────────
# Important: merge must happen in full precision, not on the quantized base
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,        # Full precision for merge
    device_map="auto",
)

# ─── Step 2: Load adapter on top of full-precision base ───────────────────────
model = PeftModel.from_pretrained(base_model, "./qlora-output/final")

# ─── Step 3: Merge — computes W_merged = W₀_bf16 + (α/r) · B × A ─────────────
model = model.merge_and_unload()
# model is now a standard model — no PEFT or bitsandbytes dependency at inference

# ─── Step 4: Save merged model ────────────────────────────────────────────────
model.save_pretrained("./merged-model-bf16/")
tokenizer.save_pretrained("./merged-model-bf16/")
# This is a standard 14 GB BF16 model — can be loaded with any inference stack

# ─── Optional Step 5: Re-quantize for efficient inference ─────────────────────
# After merging, you can re-quantize the merged model for efficient inference:
# - Convert to GGUF (llama.cpp): supports INT4, INT5, INT8, etc.
# - Apply GPTQ or AWQ for GPU-efficient inference
# - Keep as BF16 if you have sufficient GPU memory
'''
    },

    "Inspect QLoRA Model Structure": {
        "description": "Examine what QLoRA actually does to the model — which params are quantized, which are trainable",
        "runnable": False,
        "code": '''# After loading QLoRA model, inspect parameter dtypes and trainability

print("=== TRAINABLE PARAMETERS (LoRA adapters, BF16) ===")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  TRAINABLE: {name:70s} dtype={str(param.dtype):15s} shape={str(list(param.shape)):25s}")

# Example output:
# TRAINABLE: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight  dtype=torch.bfloat16  shape=[16, 4096]
# TRAINABLE: base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight  dtype=torch.bfloat16  shape=[4096, 16]
# ... (× all target modules × all 32 layers)

print("\\n=== FROZEN QUANTIZED PARAMETERS (NF4 4-bit) ===")
for name, param in model.named_parameters():
    if not param.requires_grad and hasattr(param, "quant_type"):
        print(f"  FROZEN NF4: {name:70s} quant_type={param.quant_type}")

# Check quantization config
print("\\n=== QUANTIZATION CONFIG ===")
for name, module in model.named_modules():
    if hasattr(module, "weight") and hasattr(module.weight, "quant_type"):
        print(f"  {name}: quant_type={module.weight.quant_type}, "
              f"blocksize={module.weight.blocksize}")

# Check total memory usage
import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved() / 1e9
print(f"\\n=== GPU MEMORY ===")
print(f"  Allocated: {allocated:.2f} GB")
print(f"  Reserved:  {reserved:.2f} GB")
'''
    },

    "Manual NF4 Quantization Demonstration": {
        "description": "Demonstrate the NF4 quantization/dequantization process manually on a small matrix",
        "runnable": False,
        "code": '''import torch
import bitsandbytes as bnb
from bitsandbytes.functional import quantize_4bit, dequantize_4bit

# Create a small test weight matrix (simulating a real weight distribution)
torch.manual_seed(42)
W_original = torch.randn(256, 256).cuda()  # Normal distribution, like real weights
print(f"Original W: shape={W_original.shape}, dtype={W_original.dtype}")
print(f"Original W: min={W_original.min():.4f}, max={W_original.max():.4f}")
print(f"Original W memory: {W_original.nbytes / 1e6:.2f} MB")

# ─── Quantize to NF4 ──────────────────────────────────────────────────────────
W_quantized, quant_state = quantize_4bit(
    W_original,
    blocksize=64,          # 64 weights per quantization block
    quant_type="nf4",      # NormalFloat4
    compress_statistics=True,  # Enable double quantization
)
print(f"\\nQuantized W: shape={W_quantized.shape}, dtype={W_quantized.dtype}")
print(f"Quantized W memory: {W_quantized.nbytes / 1e6:.4f} MB")
print(f"Compression ratio: {W_original.nbytes / W_quantized.nbytes:.1f}x (codes only)")

# ─── Dequantize back to BF16 ─────────────────────────────────────────────────
W_dequantized = dequantize_4bit(W_quantized, quant_state)
print(f"\\nDequantized W: shape={W_dequantized.shape}, dtype={W_dequantized.dtype}")

# ─── Measure quantization error ───────────────────────────────────────────────
error = (W_original.float() - W_dequantized.float()).abs()
print(f"\\nQuantization error statistics:")
print(f"  Mean absolute error: {error.mean():.6f}")
print(f"  Max absolute error:  {error.max():.6f}")
print(f"  Relative error:      {(error / W_original.float().abs().clamp(min=1e-8)).mean():.4f}")

# ─── Compare with INT4 quantization ──────────────────────────────────────────
W_int4, quant_state_int4 = quantize_4bit(
    W_original,
    blocksize=64,
    quant_type="fp4",      # Compare with FP4 (similar to INT4)
)
W_int4_deq = dequantize_4bit(W_int4, quant_state_int4)
error_int4 = (W_original.float() - W_int4_deq.float()).abs()
print(f"\\nFP4 quantization error (for comparison):")
print(f"  Mean absolute error: {error_int4.mean():.6f}")
print(f"  (NF4 should have lower error for normally-distributed weights)")
'''
    },

    "QLoRA Inference — Two Options": {
        "description": "Run inference with QLoRA: either keep adapter separate (low VRAM) or merge (zero overhead)",
        "runnable": False,
        "code": '''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# ─── Option A: Keep base in 4-bit + adapter separate ─────────────────────────
# Use when: minimal VRAM, want to swap adapters
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./qlora-output/final")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("[INST] Explain QLoRA in simple terms. [/INST]", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ─── Option B: Merge and use as standard model ───────────────────────────────
# Use when: production deployment, maximum inference speed
base_bf16 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
merged_model = PeftModel.from_pretrained(base_bf16, "./qlora-output/final")
merged_model = merged_model.merge_and_unload()
merged_model.eval()

# Merged model = standard LLM, no PEFT/bitsandbytes dependency at runtime
with torch.no_grad():
    outputs = merged_model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''
    },

    "Multi-GPU QLoRA with device_map": {
        "description": "Distribute a large QLoRA model across multiple GPUs using automatic device mapping",
        "runnable": False,
        "code": '''from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# For 70B model across 2× A100 80GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# device_map="auto" distributes layers evenly across all available GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",          # Uses accelerate to split layers across GPUs
)

# Check how layers were distributed
from accelerate import infer_auto_device_map
device_map = {}
for name, param in model.named_parameters():
    device_map[name] = str(param.device)

# Print first few entries to see distribution:
for i, (name, device) in enumerate(device_map.items()):
    if i < 10:
        print(f"  {name[:50]:50s} → {device}")

# 70B in NF4:  ~35 GB total
# 2× A100 80G: 160 GB available
# Leaves 125 GB for activations, adapters, optimizer states
# Easily fits! (Would require ~16× A100s for full fine-tuning)
'''
    },

    "Validate QLoRA Setup Before Long Training Run": {
        "description": "Quick sanity checks to verify QLoRA is configured correctly before committing to a full run",
        "runnable": False,
        "code": '''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ─── Setup (abbreviated) ──────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
    target_modules="all-linear", task_type="CAUSAL_LM"))

# ─── Check 1: Trainable parameters ───────────────────────────────────────────
model.print_trainable_parameters()
# Should show ~0.1-3% trainable, depending on rank and target_modules

# ─── Check 2: Verify LoRA layers are BF16, base is NF4 ───────────────────────
lora_dtypes  = set(p.dtype for n,p in model.named_parameters() if "lora" in n)
base_dtypes  = set(p.dtype for n,p in model.named_parameters() if "lora" not in n and p.requires_grad == False)
print(f"LoRA adapter dtypes: {lora_dtypes}")    # Should be {torch.bfloat16}
print(f"Base model dtypes:   {base_dtypes}")    # Should include torch.uint8 (NF4 packed)

# ─── Check 3: Run a tiny forward+backward to verify no errors ─────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(["Hello world"], return_tensors="pt").to("cuda")
inputs["labels"] = inputs["input_ids"].clone()

outputs = model(**inputs)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")    # Should be a reasonable float, not NaN/Inf
loss.backward()
print("Backward pass: OK")

# ─── Check 4: Verify gradients exist only for LoRA ───────────────────────────
has_grad  = [(n, p.grad is not None) for n,p in model.named_parameters() if p.requires_grad]
print(f"Parameters with gradients: {sum(1 for _,g in has_grad if g)} / {len(has_grad)}")
# Should be all LoRA parameters

# ─── Check 5: GPU memory usage ────────────────────────────────────────────────
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
# For 7B QLoRA: should be ~4-7 GB at this point (before activations accumulate)

print("\\n✓ All checks passed — ready for full training run.")
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
#
# def get_content():
#     """Return all content for the QLoRA topic module."""
#     return {
#         "theory": THEORY,
#         "theory_raw": THEORY,
#         "interactive_components": [],  # No external visual components for this module
#         "complexity": COMPLEXITY,
#         "operations": OPERATIONS,
#         "render_operations": None,
#     }


def get_content():
    """Return all content for this topic module."""
    from Required_Images.qlora_visual import  QLORA_VISUAL_HEIGHT, QLORA_VISUAL_HTML


    # Interactive component data — app.py splits theory_raw at each
    # placeholder and injects an st.components.v1.html() iframe instead.
    interactive_components = [

        {
            "placeholder": "{{QLORA_IMAGE}}",
            "html": QLORA_VISUAL_HTML,
            "height": QLORA_VISUAL_HEIGHT,
        },

    ]
    return {
        "theory": THEORY,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
        "render_operations": None, # render_operations,
    }