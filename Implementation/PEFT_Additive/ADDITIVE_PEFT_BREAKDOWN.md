# Additive PEFT — Complete Breakdown
## Bottleneck Adapters & (IA)³

---

```
════════════════════════════════════════════════════════════════════════════════
                   WHERE ADDITIVE SITS IN THE PEFT TAXONOMY
════════════════════════════════════════════════════════════════════════════════

                              ┌───────────────────┐
                              │    PEFT METHODS   │
                              └─────────┬─────────┘
                                        │
           ┌──────────────┬─────────────┼─────────────┬──────────────┐
           │              │             │             │              │
           ▼              ▼             ▼             ▼              ▼
    ┌───────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │ ██ ADDITIVE ██│ │Reparame- │ │Selective │ │ Hybrid   │ │Prompt-Based  │
    │               │ │terization│ │          │ │          │ │              │
    │ Insert NEW    │ │          │ │          │ │          │ │              │
    │ modules into  │ │LoRA,DoRA │ │ BitFit   │ │ QLoRA    │ │Prefix Tuning │
    │ the model     │ │          │ │          │ │          │ │              │
    │               │ │          │ │          │ │          │ │              │
    │ • Bottleneck  │ │          │ │          │ │          │ │              │
    │   Adapters *  │ │          │ │          │ │          │ │              │
    │ • (IA)³    *  │ │          │ │          │ │          │ │              │
    └───────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
          │
          THIS MODULE

```

---

## The Core Idea — What Makes Additive Different

Every PEFT method freezes the base model and trains something tiny.
The question is *what* you add and *where* you put it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   REPARAMETERIZATION (LoRA):                                                │
│   Adds matrices PARALLEL to existing weights — merges and disappears        │
│                                                                             │
│       x ──┬──── W₀ (frozen) ──────────────────────┐                         │
│           └──── A → B (trainable) × (α/r) ────────┤ + → h                   │
│                                                   │                         │
│   Result: ZERO inference overhead (after merge)                             │
│                                                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│   ADDITIVE (Bottleneck Adapters):                                           │
│   Inserts NEW modules IN SERIES — permanently stays in the forward path     │
│                                                                             │
│       x → [Frozen Transformer Layer] → [NEW Adapter Module] → h             │
│                                              ↑                              │
│                                         stays here forever                  │
│                                                                             │
│   Result: PERMANENT inference overhead — every token passes through it      │
│                                                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│   ADDITIVE (IA)³:                                                           │
│   Inserts learned SCALING VECTORS — multiplied element-wise in-place        │
│                                                                             │
│       x → [Frozen Layer] → activations × learned_vector → h                 │
│                                                  ↑                          │
│                                         rescales existing activations       │
│                                                                             │
│   Result: minimal inference overhead (just a multiply — nearly free)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The defining trade-off of Additive methods vs LoRA:

| Property             | LoRA                   | Bottleneck Adapters             | (IA)³                             |
|----------------------|------------------------|---------------------------------|-----------------------------------|
| Inference overhead   | **Zero** (after merge) | Permanent (extra modules)       | ~Zero (just elementwise multiply) |
| Architecture change  | None (parallel bypass) | **Yes** (new serial layers)     | Minimal (scale vectors)           |
| Expressiveness       | High                   | **Highest** (has non-linearity) | Lowest                            |
| Parameters           | ~0.1-1%                | ~0.5-3%                         | **~0.01-0.1%**                    |
| Non-linearity        | No                     | **Yes** (ReLU/GELU in bottle)   | No                                |
| Mergeable            | Yes                    | **No**                          | Yes (into weight matrices)        |

---

---

# PART 1: BOTTLENECK ADAPTERS

---

## What Problem Bottleneck Adapters Solve

Houlsby et al. (2019) — the original adapter paper — made a simple observation:

> "What if we could add a small trainable module after each transformer layer,
> leave everything else frozen, and teach only that module the task?"

The module had to be small (so it doesn't eat all VRAM) but expressive enough
to capture complex task-specific transformations. The solution: a bottleneck.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   THE BOTTLENECK DESIGN                                                      │
│                                                                              │
│   Input h [4096]                                                             │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────┐                                    │
│   │  DOWN PROJECT:  W_down [4096 → 64]  │  ← compress to small dimension     │
│   └────────────────────┬────────────────┘                                    │
│                        │                                                     │
│                        ▼                                                     │
│   ┌────────────────────────────────────┐                                     │
│   │  NON-LINEARITY:  ReLU / GELU       │  ← key difference from LoRA!        │
│   └────────────────────┬───────────────┘    LoRA is purely linear.           │
│                        │                    Adapters can learn non-linear    │
│                        │                    task-specific transformations.   │
│                        ▼                                                     │
│   ┌────────────────────────────────────┐                                     │
│   │  UP PROJECT:    W_up [64 → 4096]   │  ← expand back to original dim      │
│   └────────────────────┬───────────────┘                                     │
│                        │                                                     │
│                        ▼                                                     │
│   ┌────────────────────────────────────┐                                     │
│   │  RESIDUAL ADD:  output = h + x     │  ← add original input back in       │
│   └────────────────────┬───────────────┘    ensures adapter starts as        │
│                        │                    identity (safe initialization)   │
│                        ▼                                                     │
│   Output h' [4096]                                                           │
│                                                                              │
│   Total parameters:  (4096×64) + 64 + (64×4096) + 4096  =  524,352           │
│   vs. transformer layer: ~50M+ params  →  ~1% overhead per placement         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

The bottleneck dimension (64 in the example) is the key hyperparameter —
called `reduction_factor` or `bottleneck_dim`. Smaller = fewer params, less expressive.

---

## The Math

For a single adapter module applied to hidden state h:

```
h' = h  +  W_up · activation( W_down · h  +  b_down )  +  b_up
     ↑            ↑                 ↑
  residual      expand          compress + non-linearity
  (original)    back up         through bottleneck
```

Where:
- `W_down`: [hidden_dim × bottleneck_dim]   e.g. [4096 × 64]
- `W_up`:   [bottleneck_dim × hidden_dim]   e.g. [64 × 4096]
- `b_down`, `b_up`: bias vectors
- `activation`: ReLU or GELU

**Initialization (the identity trick):**
```
W_up  = zeros    →    W_up · anything = 0
W_down = random        →    b_up + W_up · act(W_down · h) = 0

Therefore at step 0:   h' = h + 0 = h    ← perfect identity function
```

Same principle as LoRA's B=0 init — the adapter starts transparent and
learns to deviate from identity only as training progresses.

---

## Where Adapters Are Placed in the Transformer

This is where Additive differs fundamentally from LoRA.
LoRA adds a parallel bypass to individual weight matrices.
Adapters are inserted as entire new sequential modules *between* existing layers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ORIGINAL TRANSFORMER LAYER (no adapters):                                 │
│                                                                             │
│   Input h                                                                   │
│      │                                                                      │
│      ▼                                                                      │
│   [Self-Attention: Q, K, V, O projections]                                  │
│      │                                                                      │
│      ▼                                                                      │
│   [Add & LayerNorm]                                                         │
│      │                                                                      │
│      ▼                                                                      │
│   [Feed-Forward MLP: gate, up, down]                                        │
│      │                                                                      │
│      ▼                                                                      │
│   [Add & LayerNorm]                                                         │
│      │                                                                      │
│      ▼                                                                      │
│   Output h'                                                                 │
│                                                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│   HOULSBY (2019) — 2 adapters per layer:                                    │
│                                                                             │
│   Input h                                                                   │
│      │                                                                      │
│      ▼                                                                      │
│   [Self-Attention]                                                          │
│      │                                                                      │
│      ▼                                                                      │
│   [Add & LayerNorm]                                                         │
│      │                                                                      │
│      ▼                                                                      │
│   ╔══════════════╗  ← ADAPTER 1 (after attention)   ← TRAINABLE             │
│   ║ Down→Act→Up  ║    + residual                                            │
│   ║ + residual   ║                                                          │
│   ╚══════════════╝                                                          │
│      │                                                                      │
│      ▼                                                                      │
│   [Feed-Forward MLP]                                                        │
│      │                                                                      │
│      ▼                                                                      │
│   [Add & LayerNorm]                                                         │
│      │                                                                      │
│      ▼                                                                      │
│   ╔══════════════╗  ← ADAPTER 2 (after FFN)   ← TRAINABLE                   │
│   ║ Down→Act→Up  ║    + residual                                            │
│   ║ + residual   ║                                                          │
│   ╚══════════════╝                                                          │
│      │                                                                      │
│      ▼                                                                      │
│   Output h'                                                                 │
│                                                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│   PFEIFFER (2020) — 1 adapter per layer (more efficient):                   │
│                                                                             │
│   Input h                                                                   │
│      │                                                                      │
│      ▼                                                                      │
│   [Self-Attention]                                                          │
│      │                                                                      │
│      ▼                                                                      │
│   [Add & LayerNorm]                                                         │
│      │                                                                      │
│      ▼                                                                      │
│   [Feed-Forward MLP]                                                        │
│      │                                                                      │
│      ▼                                                                      │
│   [Add & LayerNorm]                                                         │
│      │                                                                      │
│      ▼                                                                      │
│   ╔══════════════╗  ← ADAPTER (only after FFN)   ← TRAINABLE                │
│   ║ Down→Act→Up  ║    + residual                                            │
│   ║ + residual   ║                                                          │
│   ╚══════════════╝                                                          │
│      │                                                                      │
│      ▼                                                                      │
│   Output h'                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Houlsby** = 2 adapters per layer, higher capacity, more parameters.
**Pfeiffer** = 1 adapter per layer, more efficient, nearly same performance.
Pfeiffer is the standard today.

---

## The Complete Forward Pass

```
┌──────────────────────────────────────────────────────────────────────────────┐
│   FULL FORWARD PASS (Pfeiffer, 1 adapter per layer)                          │
│                                                                              │
│   Input tokens: [batch=4, seq=512]                                           │
│        │                                                                     │
│        ▼                                                                     │
│   Embedding layer (FROZEN) → [4, 512, 4096]                                  │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  LAYER 0  (of 32)                                                    │   │
│   │                                                                      │   │
│   │  ┌────────────────────────────────────────────────────────────────┐  │   │
│   │  │  SELF-ATTENTION (ALL FROZEN)                                   │  │   │
│   │  │                                                                │  │   │
│   │  │  Q = W_q · h    [4096 → 4096]   frozen                         │  │   │
│   │  │  K = W_k · h    [4096 → 256]    frozen (GQA)                   │  │   │
│   │  │  V = W_v · h    [4096 → 256]    frozen                         │  │   │
│   │  │  attn = softmax(Q·K^T / √d) · V                                │  │   │
│   │  │  out = W_o · attn [4096 → 4096] frozen                         │  │   │
│   │  └──────────────────────────────────┬─────────────────────────────┘  │   │
│   │                                     │                                │   │
│   │                         Add & LayerNorm (FROZEN)                     │   │
│   │                                     │                                │   │
│   │  ┌────────────────────────────────────────────────────────────────┐  │   │
│   │  │  FEED-FORWARD MLP (ALL FROZEN)                                 │  │   │
│   │  │                                                                │  │   │
│   │  │  gate = W_gate · h  [4096 → 8192]  frozen                      │  │   │
│   │  │  up   = W_up · h    [4096 → 8192]  frozen                      │  │   │
│   │  │  mlp  = W_down · (SiLU(gate) × up) [8192 → 4096]               │  │   │
│   │  └──────────────────────────────────┬─────────────────────────────┘  │   │
│   │                                     │                                │   │
│   │                         Add & LayerNorm (FROZEN)                     │   │
│   │                                     │                                │   │
│   │                                     ▼                                │   │
│   │  ╔════════════════════════════════════════════════════════════════╗  │   │
│   │  ║  BOTTLENECK ADAPTER  (TRAINABLE — ~524K params)                ║  │   │
│   │  ║                                                                ║  │   │
│   │  ║  h_in   [4, 512, 4096]   ← input (also saved as residual)      ║  │   │
│   │  ║     │                                                          ║  │   │
│   │  ║     ▼                                                          ║  │   │
│   │  ║  W_down [4096 → 64]      → [4, 512, 64]   compress             ║  │   │
│   │  ║     │                                                          ║  │   │
│   │  ║     ▼                                                          ║  │   │
│   │  ║  GELU activation         → [4, 512, 64]   non-linearity        ║  │   │
│   │  ║     │                                                          ║  │   │
│   │  ║     ▼                                                          ║  │   │
│   │  ║  W_up   [64 → 4096]      → [4, 512, 4096]  expand              ║  │   │
│   │  ║     │                                                          ║  │   │
│   │  ║     ▼                                                          ║  │   │
│   │  ║  h_out = h_in + W_up(GELU(W_down(h_in)))   residual add        ║  │   │
│   │  ║                                                                ║  │   │
│   │  ╚════════════════════════════════════════════════════════════════╝  │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   × 32 layers = 32 adapter modules, each with their OWN W_down and W_up      │
│                                                                              │
│        │                                                                     │
│        ▼                                                                     │
│   LM Head (FROZEN) → logits [4, 512, 128256]                                 │
│        │                                                                     │
│        ▼                                                                     │
│   CrossEntropy Loss (only where labels ≠ -100)                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Backward Pass — Where the Gradients Go

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Loss (scalar)                                                              │
│     │                                                                        │
│     │  loss.backward()                                                       │
│     ▼                                                                        │
│                                                                              │
│   LM Head (FROZEN):                                                          │
│     ∂Loss/∂W_lm_head  → computed for chain rule, NOT STORED                  │
│     ∂Loss/∂h_final    → passed back through layers                           │
│                                                                              │
│   At each layer (working backward):                                          │
│                                                                              │
│   1. Add & LayerNorm (FROZEN)   → gradients flow through, not stored         │
│                                                                              │
│   2. ╔═══════════════════════════════════════════╗                           │
│      ║  ADAPTER (TRAINABLE):                     ║                           │
│      ║                                           ║                           │
│      ║  ∂Loss/∂W_up   → COMPUTED and STORED      ║  64 × 4096 = 262K floats  │
│      ║  ∂Loss/∂W_down → COMPUTED and STORED      ║  4096 × 64 = 262K floats  │
│      ║  ∂Loss/∂b_up   → COMPUTED and STORED      ║  4096 floats              │
│      ║  ∂Loss/∂b_down → COMPUTED and STORED      ║  64 floats                │
│      ║                                           ║                           │
│      ╚═══════════════════════════════════════════╝                           │
│                                                                              │
│   3. MLP (FROZEN)               → gradients flow through, not stored         │
│   4. Self-Attention (FROZEN)    → gradients flow through, not stored         │
│                                                                              │
│                                                                              │
│   GRADIENT MEMORY:                                                           │
│   Adapter params only: 32 layers × 524K params × 2 bytes = ~34 MB            │
│   vs. full fine-tuning: 1.24B × 2 bytes = ~2.5 GB (for 1B model)             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Count

```
┌──────────────────────────────────────────────────────────────────────────────┐
│   PARAMETER COUNT (Pfeiffer, 1B model, bottleneck=64)                        │
│                                                                              │
│   Per adapter module:                                                        │
│     W_down: [4096 × 64]    = 262,144 params                                  │
│     b_down: [64]           =      64 params                                  │
│     W_up:   [64 × 4096]    = 262,144 params                                  │
│     b_up:   [4096]         =   4,096 params                                  │
│     ────────────────────────────────────                                     │
│     Per adapter:             528,448 params                                  │
│                                                                              │
│   × 16 layers (1B model) = 8,455,168 params                                  │
│                                                                              │
│   vs. base model: 1,240,000,000 params                                       │
│   Trainable: 0.68%  — comparable to LoRA at r=16                             │
│                                                                              │
│                                                                              │
│   Effect of bottleneck size:                                                 │
│                                                                              │
│   Bottleneck   Params/adapter   Total (16 layers)   % of 1B model            │
│   ─────────    ─────────────    ─────────────────   ─────────────            │
│        8        66,624           1,065,984           0.09%                   │
│       16       131,200           2,099,200           0.17%                   │
│       32       266,240           4,259,840           0.34%                   │
│       64       528,448           8,455,168           0.68%  *                │
│      128     1,052,928          16,846,848           1.36%                   │
│      256     2,101,248          33,619,968           2.71%                   │
│                                                                              │
│   * = good default starting point                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Adapters Can't Merge (the key LoRA advantage they lack)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   LoRA merge:                                                                │
│     W_merged = W₀ + (α/r) · B · A                                            │
│                                                                              │
│     This works because LoRA adds a LINEAR transformation in parallel.        │
│     Two linear transforms can always be summed into one.                     │
│     W_merged is the same shape as W₀.  Architecture unchanged.               │
│                                                                              │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│   Adapter "merge" attempt:                                                   │
│     h' = h + W_up · GELU( W_down · h )                                       │
│                            ↑                                                 │
│                        NON-LINEAR                                            │
│                                                                              │
│   You CANNOT absorb this into any existing weight matrix because GELU        │
│   depends on the VALUE of h, not just its linear transformation.             │
│   There is no single matrix W_merged such that W_merged · h = h'.            │
│                                                                              │
│   The adapter must stay in the computational graph forever.                  │
│   Every token at inference time runs through:                                │
│                                                                              │
│     ... → FFN → LayerNorm → W_down → GELU → W_up → residual add → ...        │
│                                                                              │
│   This adds latency proportional to the bottleneck size at every layer.      │
│   For seq_len=512, batch=1, 32 layers: ~5-15% extra inference time.          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Saving and Loading Adapters

```
adapter_config.json:
{
    "adapter_type": "bottleneck",
    "base_model_name_or_path": "unsloth/Llama-3.2-1B-Instruct",
    "bottleneck_dim": 64,
    "non_linearity": "gelu",
    "adapter_placement": "after_ffn",    // "after_attn_and_ffn" or "after_ffn"
    "num_layers": 16
}

Saved weights (per adapter, per layer):
    W_down:   [4096, 64]    ~2 MB per layer
    b_down:   [64]
    W_up:     [64, 4096]    ~2 MB per layer
    b_up:     [4096]

Total file: ~16 layers × ~4 MB = ~64 MB
vs. full model: 2.5 GB
vs. LoRA at r=16: ~15 MB  ← LoRA is smaller because no biases, no non-linearity
```

---

---

# PART 2: (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

---

## What (IA)³ Is

Liu et al. (2022). The paper's title is the abbreviation: *Infused Adapter by
Inhibiting and Amplifying Inner Activations.*

The insight is radical simplicity: **don't add new modules. Don't add new weight
matrices. Just learn vectors that rescale the existing activations.**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Bottleneck Adapter:                                                        │
│   h' = h  +  W_up · GELU( W_down · h )                                       │
│              ←───── new module, 528K params per layer ─────→                 │
│                                                                              │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│   LoRA:                                                                      │
│   h' = W₀·h  +  (α/r) · B · A · h                                            │
│                  ←────── 65K extra params per target layer ──────→           │
│                                                                              │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│   (IA)³:                                                                     │
│   h' = (l ⊙ h)                                                               │
│         ↑                                                                    │
│         l is a learned vector, same shape as h                               │
│         ⊙ = element-wise multiplication (Hadamard product)                   │
│                                                                              │
│   That's it. No new layers. No matrix multiplications beyond the existing    │
│   ones. Just rescale each dimension of the existing activations.             │
│                                                                              │
│   Parameters: just |h| floats = 4096 per rescaling point                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The Math

(IA)³ learns three sets of vectors — `l_k`, `l_v`, and `l_ff`:

```
Attention:
    K = (l_k ⊙ W_k) · x        ← rescale KEY vectors element-wise
    V = (l_v ⊙ W_v) · x        ← rescale VALUE vectors element-wise

    The l vectors gate which features in K and V are amplified or suppressed.
    High values in l → that dimension matters more for this task.
    Low values in l → that dimension is inhibited.

Feed-Forward:
    output = (l_ff ⊙ GELU(gate)) × up    ← rescale gate activations
```

Initialization:
```
l_k  = ones   →   l_k ⊙ W_k = 1 × W_k = W_k   (identity — unchanged)
l_v  = ones   →   l_v ⊙ W_v = 1 × W_v = W_v   (identity — unchanged)
l_ff = ones   →   l_ff ⊙ act = 1 × act = act   (identity — unchanged)
```

Same principle as B=0 in LoRA and W_up=0 in Adapters:
model starts as identical to base, deviations learned over training.

---

## Where (IA)³ Applies Scaling

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   ONE TRANSFORMER LAYER WITH (IA)³                                           │
│                                                                              │
│   Input h                                                                    │
│      │                                                                       │
│      ▼                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │  SELF-ATTENTION                                                       │  │
│   │                                                                       │  │
│   │  Q = W_q · h                  (Q is NOT rescaled — attention is       │  │
│   │                                about WHAT to attend to, so K and V    │  │
│   │                                are the important ones to gate)        │  │
│   │                                                                       │  │
│   │  K = (l_k ⊙ W_k) · h         ← rescale key projections                │  │
│   │       ↑                                                               │  │
│   │   l_k: [4096] learned vector — initialized to 1.0 everywhere          │  │
│   │   The full W_k stays frozen. l_k just scales its output column-wise.  │  │
│   │                                                                       │  │
│   │  V = (l_v ⊙ W_v) · h         ← rescale value projections              │  │
│   │       ↑                                                               │  │
│   │   l_v: [4096] learned vector — initialized to 1.0 everywhere          │  │
│   │                                                                       │  │
│   │  attn = softmax(Q · K^T / √d) · V                                     │  │
│   │  out  = W_o · attn            (O is NOT rescaled)                     │  │
│   │                                                                       │  │
│   └────────────────────────────────────────┬──────────────────────────────┘  │
│                                            │                                 │
│                                Add & LayerNorm                               │
│                                            │                                 │
│   ┌────────────────────────────────────────▼──────────────────────────────┐  │
│   │  FEED-FORWARD MLP                                                     │  │
│   │                                                                       │  │
│   │  gate = W_gate · h                                                    │  │
│   │  up   = W_up · h                                                      │  │
│   │                                                                       │  │
│   │  out = W_down · ( l_ff ⊙ GELU(gate) ) × up                            │  │
│   │                    ↑                                                  │  │
│   │              l_ff: [8192] learned vector — scales the gate activations│  │
│   │              Note: shape is [intermediate_dim] not [hidden_dim]       │  │
│   │                    because it gates the FFN's internal activations    │  │
│   │                                                                       │  │
│   └───────────────────────────────────────────────────────────────────────┘  │
│                                            │                                 │
│                                Add & LayerNorm                               │
│                                            │                                 │
│                                       Output h'                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Count — Remarkably Tiny

```
┌──────────────────────────────────────────────────────────────────────────────┐
│   PARAMETER COUNT (IA)³ on 1B model (Llama-3.2-1B: 16 layers, d=2048,        │
│                                      kv_d=256, ffn=8192)                     │
│                                                                              │
│   Per layer:                                                                 │
│     l_k:  [kv_d] = [256]   ← key head dimension (GQA)                        │
│     l_v:  [kv_d] = [256]   ← value head dimension                            │
│     l_ff: [ffn]  = [8192]  ← FFN intermediate dimension                      │
│     ──────────────────────────                                               │
│     Per layer: 256 + 256 + 8192 = 8,704 params                               │
│                                                                              │
│   × 16 layers = 139,264 total trainable parameters                           │
│                                                                              │
│   vs. base model: 1,240,000,000 params                                       │
│   Trainable: 0.011%  ← this is ~10x fewer than LoRA r=16                     │
│                      ← ~60x fewer than Bottleneck Adapters                   │
│                                                                              │
│                                                                              │
│   COMPARISON TABLE (1B model):                                               │
│                                                                              │
│   Method              Trainable Params    % of Model    File Size            │
│   ──────              ────────────────    ──────────    ─────────            │
│   (IA)³               ~139K               0.011%        ~0.5 MB              │
│   LoRA (r=4)          ~4.2M               0.34%         ~16 MB               │
│   LoRA (r=16)         ~10.2M              0.82%         ~40 MB               │
│   Bottleneck (b=64)   ~8.5M               0.68%         ~64 MB               │
│   Full Fine-Tuning    1,240M              100%          ~2.5 GB              │
│                                                                              │
│   (IA)³ is the most parameter-efficient method in all of PEFT.               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Can (IA)³ Merge?

Yes — and this is a surprise given it's Additive. Because the rescaling is
LINEAR (element-wise multiply has no non-linearity), it CAN be folded in:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   (IA)³ key rescaling:                                                       │
│     K = (l_k ⊙ W_k) · h                                                      │
│                                                                              │
│   This is equivalent to:                                                     │
│     K = W_k_merged · h      where  W_k_merged = diag(l_k) · W_k              │
│                                                                              │
│   Or more efficiently (no explicit diag):                                    │
│     W_k_merged[i, :] = l_k[i] × W_k[i, :]    (scale each row by l_k[i])      │
│                                                                              │
│   After merge: W_k becomes W_k_merged, l_k is discarded.                     │
│   Zero inference overhead. Same as LoRA.                                     │
│                                                                              │
│   CONTRAST with Bottleneck Adapters:                                         │
│     h' = h + W_up · GELU( W_down · h )   ← GELU makes this non-mergeable     │
│                                                                              │
│   (IA)³ is the only Additive method that is also mergeable.                  │
│   It sits at the intersection of Additive and Reparameterization.            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## (IA)³ Visualised — What the l Vectors Actually Do

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Think of each dimension of h as a "feature channel."                       │
│   l is a learned dial for each channel — amplify or suppress it.             │
│                                                                              │
│   Key vector K [dim=8]:                                                      │
│                                                                              │
│   Before rescaling:   K = [ 0.8,  0.2, -0.5,  0.9, -0.1,  0.3,  0.7, -0.4]   │
│   l_k (learned):          [ 2.1,  0.1,  1.8,  0.05, 1.5, 0.02,  2.3, 1.9 ]   │
│   After rescaling:    K = [ 1.68, 0.02, -0.9, 0.045,-0.15, 0.006, 1.61,-0.76]│
│                             ↑           ↑                    ↑               │
│                         amplified    amplified           amplified           │
│                             by 2.1   by 1.8              by 2.3              │
│                                  ↑               ↑                           │
│                              suppressed      suppressed                      │
│                               (×0.1)         (×0.02)                         │
│                                                                              │
│   The model learns: for THIS task, features 0, 2, 6 are important            │
│   (amplify them), features 1, 5 are noise for this task (suppress them).     │
│                                                                              │
│   This is essentially learned feature selection — which dimensions of        │
│   the pretrained representations matter for the target task.                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

---

# PART 3: HEAD-TO-HEAD COMPARISON

---

```
    ════════════════════════════════════════════════════════════════════════════════
                BOTTLENECK ADAPTERS vs (IA)³ vs LoRA
    ════════════════════════════════════════════════════════════════════════════════

    ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    Description                          | Bottleneck Adapters     | (IA)³                | LoRA (r=16)
    ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    MECHANISM
    ------------------------------------------------------------------------------------------------------------
    Operation type                        | Non-linear              | Linear               | Linear             |
    Where applied                         | Serial (in sequence)    | In-place scaling     | Parallel bypass    |
    Non-linearity                         | Yes (GELU/ReLU)         | No                   | No                 |
    Residual connection                   | Yes                     | No                   | No (bypass itself) |
    
    PARAMETERS (1B model)
    ------------------------------------------------------------------------------------------------------------
    Trainable params                      | ~8.5M (0.68%)           | ~139K (0.011%)       | ~10.2M (0.82%)     |
    File size                             | ~64 MB                  | ~0.5 MB              | ~40 MB             |
    Per-layer overhead                    | 528K                    | 8.7K                 | 640K               |
    
    EXPRESSIVENESS
    ------------------------------------------------------------------------------------------------------------
    Capture non-linear transformations?   | YES                     | No                   | No                 |
    Best for                              | Complex domain shift    | Lightweight task     | General-purpose    |
                                          | tasks                   | steering             | fine-tuning        |
    Feature selection                     | Implicit (bottleneck)   | Explicit (l vectors) | Implicit (rank)    |
    
    INFERENCE
    -------------------------------------------------------------------------------------------------------------
    Extra compute                         | Yes (permanent)         | ~None                | None (after merge) |
    Can merge into base model?            | NO                      | YES                  | YES                |
    Latency overhead                      | 5–15%                   | <1%                  | 0% (after merge)   |
    Deployment format                     | Adapter + base          | Merge or keep        | Merge or keep      |
    
    TRAINING
    -------------------------------------------------------------------------------------------------------------
    VRAM overhead                         | Slightly more           | Minimal              | Comparable         |
                                          | (extra activations)     |                      |                    |
    Typical learning rate                 | 1e-4 to 1e-3            | 1e-3 to 3e-3         | 1e-4 to 3e-4       |
                                          |                         | (higher OK —         |                    |
                                          |                         | very few params)     |                    |

WHEN TO CHOOSE EACH:
  Choose Bottleneck if:   Task requires complex non-linear transformation
                          Domain shift is large (e.g., English → chemistry)
                          Expressiveness matters more than inference speed
                          You're OK with permanent adapter overhead

  Choose (IA)³ if:        Extreme parameter efficiency required
                          Many tasks, tiny storage budget
                          Hardware is very constrained
                          Task steering rather than deep domain adaptation

  Choose LoRA if:         Best general-purpose choice for most tasks
                          Zero inference overhead required after deployment
                          Good balance of expressiveness vs. parameter count
                          Adapter swapping needed (before merge)
```

---

---

# PART 4: COMPLETE DATA FLOW (both methods)

---

```
════════════════════════════════════════════════════════════════════════════════
   DATA PIPELINE — identical to LoRA up to the forward pass
════════════════════════════════════════════════════════════════════════════════

STAGE 1: Raw JSONL  →  exactly the same as LoRA / full fine-tuning
STAGE 2: Template formatting  →  exactly the same
STAGE 3: Tokenization  →  exactly the same
STAGE 4: Labels + loss mask (-100 for instruction tokens)  →  same
STAGE 5: Padding + attention mask  →  same
STAGE 6: Move to GPU + Embedding lookup  →  same

STAGE 7: Forward pass — THIS is where they diverge:

  LoRA:
    At each target weight: h = W₀·x + (α/r)·B·A·x   (parallel path added)
    Adapter modules: NOT present — no extra modules exist

  Bottleneck Adapters:
    At each target weight: h = W₀·x                  (normal frozen path)
    After each FFN block:  h = h + W_up·GELU(W_down·h) (extra module applied)

  (IA)³:
    At key projection:     K = (l_k ⊙ W_k)·x          (rescaled in-place)
    At value projection:   V = (l_v ⊙ W_v)·x          (rescaled in-place)
    At FFN gate:           out = W_down·(l_ff ⊙ GELU(gate))×up

STAGE 8: Loss  →  exactly the same (CrossEntropy, -100 masking)
STAGE 9: Backward pass  →  gradients stored ONLY for adapter/l params
STAGE 10: Optimizer update  →  updates ONLY adapter/l params
STAGE 11: Save  →  adapter weights / l vectors only
```

---

---

# PART 5: MEMORY LAYOUT — ALL THREE METHODS

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│   VRAM DURING TRAINING (1B model, BF16, batch=4, seq=512)                    │
│                                                                              │
│   FULL FINE-TUNING:                                                          │
│     Weights:           2.5 GB  ██████████████████████████                    │
│     Gradients:         2.5 GB  ██████████████████████████                    │
│     Optimizer (Adam):  10 GB   ████████████████████████████████████████████  │
│     Activations:       2-4 GB  ████████████████                              │
│     ─────────────────────────────────                                        │
│     TOTAL:            ~17-19 GB                                              │
│                                                                              │
│   LoRA (r=16):                                                               │
│     Frozen weights:    2.5 GB  ██████████████████████████                    │
│     LoRA weights:      ~40 MB  ▌                                             │
│     LoRA grads:        ~40 MB  ▌                                             │
│     Optimizer:         ~160 MB ▌                                             │
│     Activations:       1-3 GB  ████████                                      │
│     ─────────────────────────────────                                        │
│     TOTAL:            ~6-9 GB                                                │
│                                                                              │
│   Bottleneck Adapters (b=64):                                                │
│     Frozen weights:    2.5 GB  ██████████████████████████                    │
│     Adapter weights:   ~64 MB  ▌                                             │
│     Adapter grads:     ~64 MB  ▌                                             │
│     Optimizer:         ~256 MB █                                             │
│     Activations:       1-4 GB  ████████  (slightly more — extra modules)     │
│     ─────────────────────────────────                                        │
│     TOTAL:            ~7-10 GB                                               │
│                                                                              │
│   (IA)³:                                                                     │
│     Frozen weights:    2.5 GB  ██████████████████████████                    │
│     l vectors:         ~0.5 MB ▏                                             │
│     l gradients:       ~0.5 MB ▏                                             │
│     Optimizer:         ~2 MB   ▏                                             │
│     Activations:       1-3 GB  ████████                                      │
│     ─────────────────────────────────                                        │
│     TOTAL:            ~5-7 GB  (lightest training footprint)                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

---

# PART 6: PRACTICAL DECISION GUIDE

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   START HERE: What matters most to you?                                      │
│                                                                              │
│   Inference latency critical?                                                │
│   ├── YES:  Must be zero overhead?                                           │
│   │         ├── YES: Use LoRA (merge after training) or (IA)³ (merge-able)   │
│   │         └── NO:  (IA)³ (<1% overhead) or LoRA                            │
│   └── NO: Continue...                                                        │
│                                                                              │
│   Parameter budget extremely tight? (< 1M params, tiny storage)              │
│   ├── YES: Use (IA)³  (~139K params for 1B model)                            │
│   └── NO: Continue...                                                        │
│                                                                              │
│   Large domain shift? (English → medical, general → code, etc.)              │
│   ├── YES: Consider Bottleneck Adapters (non-linearity helps capture         │
│   │        complex transformations) or LoRA with larger rank                 │
│   └── NO: LoRA r=8-16 is the default best choice                             │
│                                                                              │
│   Serving many tasks from one base model?                                    │
│   ├── YES: LoRA (adapter swap) or (IA)³ (tiny files, easy swap)              │
│   └── NO: Any method works                                                   │
│                                                                              │
│   Simple recommendation for most cases:                                      │
│                                                                              │
│   LoRA (r=16) > (IA)³ > Bottleneck Adapters                                  │
│                                                                              │
│   LoRA wins in practice for general fine-tuning.                             │
│   (IA)³ wins when you need extreme efficiency.                               │
│   Bottleneck wins when expressiveness > inference cost.                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## One Final Analogy

Think of adapting a musician to play a new genre:

**Full Fine-Tuning**: Retrain them completely — they learn the new genre but may forget old skills.

**LoRA**: Give them a cheat sheet that adds parallel notes to their existing playing. The cheat sheet can be folded into their memory (merged) and they play naturally.

**Bottleneck Adapters**: Give them a small filter pedal that runs their sound through a compression/expansion effect after they play. The pedal always stays in the signal chain — you can't remove it without also removing the effect.

**(IA)³**: Give them a tiny set of dials that turn certain aspects of their playing up or down — more treble here, less vibrato there. Minimal gear, just learned adjustments. The dials can be permanently set and removed once calibrated.
