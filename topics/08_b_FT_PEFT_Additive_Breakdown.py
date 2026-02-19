"""
Fine Tuning - PEFT (Parameter-Efficient Fine-Tuning) Detailed Breakdown
========================================================================

A comprehensive deep dive into PEFT methods — LoRA, QLoRA, Adapters,
Prompt Tuning, and more. Covers the math, the memory, the data flow,
and the practical trade-offs that make PEFT the dominant fine-tuning
paradigm for modern LLMs.
"""

TOPIC_NAME = "PEFT_Additive_IA3_Breakdown"

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


    ══════════════════════════════════════════════════════════════════
                    PEFT (Parameter-Efficient Fine-Tuning)
    ══════════════════════════════════════════════════════════════════
            Goal: Adapt large models by training only a small
                  number of additional or selected parameters
    ══════════════════════════════════════════════════════════════════


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
                                        ADDITIVE - PEFT
═══════════════════════════════════════════════════════════════════════════════════════════════

Build new small modules and insert them into the model. The original architecture gets new components bolted on. 

Bottleneck Adapters are the classic example. 
Downside: those new modules stay in the model forever, adding inference latency.

---

You have a pre-trained transformer. Every layer in it follows this flow:

            Input → Self-Attention → Add & Norm → Feed-Forward (MLP) → Add & Norm → Output

Additive PEFT physically inserts new small modules into this pipeline that didn't exist before. 
The original layers are all frozen — you're literally adding new trainable components into the architecture.

The Main Additive Method: Bottleneck Adapters - step-by-step breakdown of what happens.

---

Step 1: Freeze the entire pre-trained model

Every single parameter in the original model gets requires_grad = False. 

No gradients will be computed for them, no optimizer states stored. They become read-only.

    for param in model.parameters():
        param.requires_grad = False    # 7 billion parameters → all frozen

---

Step 2: Insert adapter modules into every transformer layer

Two small adapter modules are inserted into each layer — one after self-attention, one after the feed-forward network:


        BEFORE (standard transformer layer):
        
            Input
              ↓
            Self-Attention
              ↓
            Add & LayerNorm
              ↓
            Feed-Forward (MLP)
              ↓
            Add & LayerNorm
              ↓
            Output
        
        
        AFTER (with adapters inserted):
        
            Input
              ↓
            Self-Attention          ← FROZEN
              ↓
            ██ ADAPTER MODULE 1 ██  ← NEW, trainable
              ↓
            Add & LayerNorm
              ↓
            Feed-Forward (MLP)      ← FROZEN
              ↓
            ██ ADAPTER MODULE 2 ██  ← NEW, trainable
              ↓
            Add & LayerNorm
              ↓
            Output

The adapter is now literally in the data path. Every token's representation must flow through it.

---

Step 3: Understand what's inside each adapter module
Each adapter is a tiny feed-forward network with a bottleneck — it squeezes the data down to a small dimension and expands it back. 
Here's exactly what happens to a single token's hidden state vector as it passes through:


        Input: h  (shape: [4096])      ← the token's hidden state coming from self-attention
               │
               │
               ▼
        Down-projection:  W_down × h   (W_down shape: [4096 × 64])
               │
               │               h is now compressed: [4096] → [64]
               │               This forces the adapter to learn a COMPRESSED
               │               representation of whatever adjustment is needed.
               │               It can't just memorize — it must generalize.
               ▼
        Non-linearity:    ReLU(compressed_h)
               │
               │               The non-linearity is important — without it,
               │               down-project then up-project is just a single
               │               linear transformation (matrix multiplication
               │               collapses). ReLU gives the adapter the ability
               │               to learn non-linear transformations.
               ▼
        Up-projection:    W_up × activated_h   (W_up shape: [4096 × 64])
               │
               │               Expanded back: [64] → [4096]
               │               Same dimensionality as the original hidden state.
               ▼
        Residual add:     output = adapter_output + h  (original input added back)
               │
               │               THIS IS CRITICAL. The residual connection means:
               │               - If the adapter learns nothing useful → output ≈ h
               │                 (model behaves like the pre-trained version)
               │               - If the adapter learns something → output = h + adjustment
               │                 (model gets a task-specific nudge)
               │
               ▼
        Output: h_adapted  (shape: [4096])    ← continues to the next part of the layer

---

Step 4: Count the trainable parameters

For a single adapter module with bottleneck dimension 64 and hidden dimension 4096:

        W_down:  [64 × 4096]    =   262,144 parameters
        b_down:  [64]           =        64 parameters
        W_up:    [4096 × 64]    =   262,144 parameters  
        b_up:    [4096]         =     4,096 parameters
        ──────────────────────────────────────────────
        Total per adapter:         ~524,448 parameters
        
        Per transformer layer:    2 adapters × 524,448 = ~1,048,896
        
        For 32 layers:            32 × 1,048,896 = ~33.6 million
        
        vs. 7 billion total model parameters   →   ~0.48% trainable

The bottleneck dimension (64 in this example) is the key hyperparameter. 
It's analogous to LoRA's rank — smaller means fewer parameters but less capacity.

---

Step 5: Training — what actually happens during a forward pass

Let's trace a single training example through a model with adapters:

    
    "Classify sentiment: This movie was great" → "positive"

Tokenize:  [1, 518, 25580, ..., 6374, 2]

Forward pass through Layer 0:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Token embeddings → Self-Attention (FROZEN)                 │
    │                        ↓                                    │
    │                     h = [4096-dim vector for each token]    │
    │                        ↓                                    │
    │                     Adapter 1:                              │
    │                        h_down = W_down × h    → [64-dim]    │
    │                        h_act  = ReLU(h_down)                │
    │                        h_up   = W_up × h_act  → [4096-dim]  │
    │                        h_out  = h_up + h      (residual)    │
    │                        ↓                                    │
    │                     Add & LayerNorm                         │
    │                        ↓                                    │
    │                     Feed-Forward MLP (FROZEN)               │
    │                        ↓                                    │
    │                     Adapter 2:                              │
    │                        (same squeeze → activate → expand)   │
    │                        ↓                                    │
    │                     Add & LayerNorm                         │
    │                        ↓                                    │
    │                     → passes to Layer 1                     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    ... repeat through all 32 layers ...

    Final output → compute loss on "positive" tokens
    
---

Step 6: Backward pass — where it gets efficient

    Loss computed on output
        ↓
    Backpropagation begins (working backward through the network)
        ↓
    At each frozen layer:
        - Gradients flow THROUGH the frozen weights (needed for chain rule)
        - But NO gradient is STORED for frozen weights
        - NO optimizer state maintained for frozen weights
        ↓
    At each adapter:
        - Gradients ARE computed and STORED for W_down, W_up, b_down, b_up
        - Optimizer (Adam) maintains momentum + variance for these params only
        ↓
    Weight update:
        - ONLY adapter parameters get updated
        - Frozen model: untouched
        

The memory savings come from the optimizer states. 
Adam stores 2 extra values per trainable parameter (momentum and variance). 
For 7B frozen parameters, that's 56 GB you never allocate. For 33M adapter parameters, it's ~260 MB.

---

Step 7: Saving the result

After training, you save ONLY the adapter weights:

Full fine-tuning checkpoint:          Adapter checkpoint:
──────────────────────────           ──────────────────────
model.safetensors    14 GB           adapter_weights.pt   ~130 MB
                                     adapter_config.json
                                     (references base model by name)
                                     
---

Why the Bottleneck Shape Matters

The squeeze-and-expand isn't arbitrary. It enforces an information bottleneck:

    4096 dimensions of information
        ↓
    FORCED through 64 dimensions     ← can't pass everything through
        ↓
    Back to 4096 dimensions
    
    The adapter MUST learn which 64 dimensions of variation
    are most important for your task. It's forced to prioritize.
    
    Bottleneck too small (e.g., 8)    :   Not enough capacity. Adapter can't
                                          capture the task's complexity.
                                          
    Bottleneck too large (e.g., 2048) :   Too much capacity. Approaches full
                                          fine-tuning cost. Defeats the purpose.
                                          
    Sweet spot (32-128)               :   Enough to capture task-specific
                                          adjustments without excess.
                                          
---

The Three Additive Sub-Methods

Bottleneck Adapters are the most important, but the additive category also includes:

(IA)³ — the most extreme version. Instead of inserting full modules, 
        it just learns three scaling vectors (one each for keys, values, and feed-forward outputs). 
        Each vector element-wise multiplies the activations — amplifying some dimensions and suppressing others. 
        Only ~0.01% of parameters. Extremely lightweight, less expressive.

Soft Prompts — sometimes classified as additive because you're adding new trainable embedding vectors to the input sequence. 
               These are covered more thoroughly under the "Prompt-Based" category, 
               but conceptually they're additive — new parameters that didn't exist before.
               
---

Why Adapters Lost to LoRA

The fundamental problem: adapters can't be removed after training. 
They sit in the forward pass permanently, adding latency to every inference call. 
LoRA's parallel bypass (h = W₀x + BAx) merges back into the weight matrix after training (W_merged = W₀ + BA), leaving zero trace. 
That single property — mergeability — is why LoRA became the dominant PEFT method and adapters faded into historical importance.


---

Additive PEFT — Complete Visual Diagram Breakdown
===================================================

Textual diagrams covering every aspect of Additive PEFT:
architecture, data flow, bottleneck mechanics, training loop,
memory layout, and comparison with other PEFT approaches.



    ══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    ADDITIVE PEFT — COMPLETE VISUAL BREAKDOWN
    ══════════════════════════════════════════════════════════════════════════════════════════════════════



    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 1:  WHERE ADDITIVE SITS IN THE PEFT TAXONOMY
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    ┌───────────────────┐
    │    PEFT METHODS   │
    └─────────┬─────────┘
              │
              │                
              ▼               
    ┌════════════════════┐ 
    ║                    ║ 
    ║   ██ ADDITIVE ██   ║ 
    ║                    ║ 
    ║ INSERT new modules ║ 
    ║ into the model     ║ 
    ║                    ║ 
    ║ • Bottleneck       ║ 
    ║   Adapters    *    ║ 
    ║ • (IA)³            ║ 
    ║                    ║ 
    ╚════════════════════╝ 
              │
              │  ★ This diagram breaks down everything inside this box
              ▼

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 2:  THE CORE IDEA — WHAT "ADDITIVE" MEANS VISUALLY
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    The core idea: frozen model + new modules bolted on
        
        
        ORIGINAL PRE-TRAINED MODEL                               MODEL WITH ADDITIVE PEFT
        (nothing changes here)                                   (new modules bolted on)

        ┌──────────────────────────┐                             ┌──────────────────────────┐
        │                          │                             │                          │
        │   ┌──────────────────┐   │                             │   ┌───────────────────┐  │
        │   │  Layer 31        │   │                             │   │  Layer 31 FROZEN  │──── ██ Adapter ██
        │   └──────────────────┘   │                             │   └───────────────────┘  │
        │   ┌──────────────────┐   │                             │   ┌───────────────────┐  │
        │   │  Layer 30        │   │                             │   │  Layer 30 FROZEN  │──── ██ Adapter ██
        │   └──────────────────┘   │                             │   └───────────────────┘  │
        │          ...             │                             │          ...             │
        │   ┌──────────────────┐   │                             │   ┌───────────────────┐  │
        │   │  Layer 1         │   │                             │   │  Layer 1  FROZEN  │──── ██ Adapter ██
        │   └──────────────────┘   │                             │   └───────────────────┘  │
        │   ┌──────────────────┐   │                             │   ┌───────────────────┐  │
        │   │  Layer 0         │   │                             │   │  Layer 0  FROZEN  │──── ██ Adapter ██
        │   └──────────────────┘   │                             │   └───────────────────┘  │
        │   ┌──────────────────┐   │                             │   ┌───────────────────┐  │
        │   │  Embedding       │   │                             │   │  Embedding FROZEN │  │
        │   └──────────────────┘   │                             │   └───────────────────┘  │
        └──────────────────────────┘                             └──────────────────────────┘   

        ALL params trainable (7B)                                ONLY adapters trainable (~33M)
        120+ GB VRAM                                             ~20 GB VRAM
        Full copy per task (14 GB)                               Adapter file per task (~130 MB)

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 3:  STANDARD TRANSFORMER LAYER vs. LAYER WITH ADAPTERS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Standard transformer layer vs. layer with adapters (side by side)
    
         STANDARD TRANSFORMER LAYER                    TRANSFORMER LAYER WITH ADAPTERS
         (Full Fine-Tuning: all trainable)             (Additive PEFT: only adapters trainable)

              ┌──────────┐                                  ┌──────────┐
              │  Input h │                                  │  Input h │
              └────┬─────┘                                  └────┬─────┘
                   │                                              │
                   ▼                                              ▼
         ┌─────────────────────┐                        ┌─────────────────────┐
         │                     │                        │                     │
         │   Self-Attention    │  ◄── TRAINABLE         │   Self-Attention    │  ◄── FROZEN
         │   (Wq, Wk, Wv, Wo)  │                        │   (Wq, Wk, Wv, Wo)  │
         │                     │                        │                     │
         └──────────┬──────────┘                        └──────────┬──────────┘
                    │                                               │
                    │                                               ▼
                    │                                    ╔══════════════════════╗
                    │                                    ║                      ║
                    │                                    ║   ██ ADAPTER 1 ██    ║  ◄── TRAINABLE
                    │                                    ║   (down → ReLU →     ║
                    │                                    ║    up → residual)    ║
                    │                                    ║                      ║
                    │                                    ╚══════════╤═══════════╝
                    │                                               │
                    ▼                                               ▼
         ┌─────────────────────┐                        ┌──────────────────────┐
         │   Residual Add      │                        │    Residual Add      │
         │   + LayerNorm       │                        │    + LayerNorm       │
         └──────────┬──────────┘                        └───────────┬──────────┘
                    │                                               │
                    ▼                                               ▼
         ┌─────────────────────┐                        ┌──────────────────────┐
         │                     │                        │                      │
         │   Feed-Forward      │  ◄── TRAINABLE         │    Feed-Forward      │  ◄── FROZEN
         │   (MLP)             │                        │    (MLP)             │
         │                     │                        │                      │
         └──────────┬──────────┘                        └───────────┬──────────┘
                    │                                               │
                    │                                               ▼
                    │                                    ╔══════════════════════╗
                    │                                    ║                      ║
                    │                                    ║   ██ ADAPTER 2 ██    ║  ◄── TRAINABLE
                    │                                    ║   (down → ReLU →     ║
                    │                                    ║    up → residual)    ║
                    │                                    ║                      ║
                    │                                    ╚══════════╤═══════════╝
                    │                                               │
                    ▼                                               ▼
         ┌─────────────────────┐                        ┌──────────────────────┐
         │   Residual Add      │                        │    Residual Add      │
         │   + LayerNorm       │                        │    + LayerNorm       │
         └──────────┬──────────┘                        └───────────┬──────────┘
                    │                                               │
                    ▼                                               ▼
              ┌──────────┐                                  ┌──────────┐
              │ Output h │                                  │ Output h │
              └──────────┘                                  └──────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 4:  INSIDE A SINGLE ADAPTER MODULE — THE BOTTLENECK
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Inside a single adapter module: the full bottleneck breakdown (down-project → ReLU → up-project → residual), 
                                    with parameter counts and annotations on why each component exists
        
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                   ║
    ║                              INSIDE ONE ADAPTER MODULE                                            ║
    ║                                                                                                   ║
    ║    Input: h                                                                                       ║
    ║    shape: [batch, seq_len, 4096]                                                                  ║
    ║      │                                                                                            ║
    ║      │                                                                                            ║
    ║      ├───────────────────────────────────────────────┐  (skip connection / residual)              ║
    ║      │                                               │                                            ║
    ║      ▼                                               │                                            ║
    ║    ┌─────────────────────────────────────────┐       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   DOWN-PROJECTION (W_down)              │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   [4096] ──────────────────────▶ [64]   │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   W_down: [64 × 4096] = 262,144 params  │       │                                            ║
    ║    │   b_down: [64]        =      64 params  │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   SQUEEZE: 4096 dims compressed to 64   │       │                                            ║
    ║    │   Forces adapter to learn WHAT MATTERS  │       │                                            ║
    ║    │   for this task — can't pass everything │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    └────────────────┬────────────────────────┘       │                                            ║
    ║                     │                                │                                            ║
    ║                     ▼                                │                                            ║
    ║    ┌─────────────────────────────────────────┐       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   NON-LINEARITY (ReLU or GELU)          │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   Without this: down × up = one linear  │       │                                            ║
    ║    │   transform (matrices collapse).        │       │                                            ║
    ║    │   ReLU lets adapter learn NON-LINEAR    │       │                                            ║
    ║    │   task-specific transformations.        │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    └────────────────┬────────────────────────┘       │                                            ║
    ║                     │                                │                                            ║
    ║                     ▼                                │                                            ║
    ║    ┌─────────────────────────────────────────┐       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   UP-PROJECTION (W_up)                  │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   [64] ──────────────────────▶ [4096]   │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   W_up: [4096 × 64] = 262,144 params    │       │                                            ║
    ║    │   b_up: [4096]      =   4,096 params    │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   EXPAND: back to original dimension    │       │                                            ║
    ║    │   so output is compatible with the      │       │                                            ║
    ║    │   rest of the transformer layer         │       │                                            ║
    ║    │                                         │       │                                            ║
    ║    └────────────────┬────────────────────────┘       │                                            ║
    ║                     │                                │                                            ║
    ║                     ▼                                │                                            ║
    ║    ┌─────────────────────────────────────────┐       │                                            ║
    ║    │                                         │       │                                            ║
    ║    │   RESIDUAL ADD                         ◄├───────┘                                            ║
    ║    │                                         │                                                    ║
    ║    │   output = adapter_out + h (original)   │                                                    ║
    ║    │                                         │                                                    ║
    ║    │   WHY THIS MATTERS:                     │                                                    ║
    ║    │   • At init, adapter ≈ 0 → output ≈ h   │                                                    ║
    ║    │     (model starts as pre-trained)       │                                                    ║
    ║    │   • After training: output = h + Δ      │                                                    ║
    ║    │     (model gets task-specific nudge)    │                                                    ║
    ║    │   • Adapter can NEVER destroy h — it    │                                                    ║
    ║    │     can only ADD to it                  │                                                    ║
    ║    │                                         │                                                    ║
    ║    └────────────────┬────────────────────────┘                                                    ║
    ║                     │                                                                             ║
    ║                     ▼                                                                             ║
    ║    Output: h_adapted                                                                              ║
    ║    shape: [batch, seq_len, 4096]   (same as input — transparent to rest of model)                 ║
    ║                                                                                                   ║
    ║                                                                                                   ║
    ║    TOTAL PARAMS PER ADAPTER:  262,144 + 64 + 262,144 + 4,096 = 528,448  (~0.5M)                   ║
    ║                                                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 5:  THE BOTTLENECK SHAPE — WHY IT WORKS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────


    INFORMATION FLOW THROUGH THE BOTTLENECK
        
    The bottleneck shape visualized as an information funnel, plus the size trade-off table
        

        4096 dimensions                       64 dimensions                      4096 dimensions
        (rich, full                           (compressed,                       (restored,
         representation)                       essential info only)               task-adapted)

        ║║║║║║║║║║║║║║║║║║                         ║║║║                        ║║║║║║║║║║║║║║║║║║
        ║║║║║║║║║║║║║║║║║║                         ║║║║                        ║║║║║║║║║║║║║║║║║║
        ║║║║║║║║║║║║║║║║║║    ─── W_down ───▶      ║║║║     ─── W_up ───▶      ║║║║║║║║║║║║║║║║║║
        ║║║║║║║║║║║║║║║║║║        SQUEEZE          ║║║║         EXPAND         ║║║║║║║║║║║║║║║║║║
        ║║║║║║║║║║║║║║║║║║                         ║║║║                        ║║║║║║║║║║║║║║║║║║
        ║║║║║║║║║║║║║║║║║║                         ║║║║                        ║║║║║║║║║║║║║║║║║║

                                              ▲
                                              │
                                    This is the BOTTLENECK
                                    Only 64 dims can pass through
                                    Adapter must learn which
                                    aspects of the input matter
                                    most for the task


        ┌──────────────────────────────────────────────────────────────────────────────────────┐
        │                                                                                      │
        │   BOTTLENECK SIZE TRADE-OFF                                                          │
        │                                                                                      │
        │   Bottleneck     Params/Adapter    Total (32 layers)    Capacity     Risk            │
        │   Dim                              (2 adapters/layer)                                │
        │   ─────────      ─────────────     ────────────────     ────────     ────            │
        │      8            ~65K              ~4.2M                Very low     Underfitting   │
        │     32            ~262K             ~16.8M               Low          Good balance   │
        │     64            ~525K             ~33.6M               Medium       Sweet spot  *  │
        │    128            ~1.05M            ~67.1M               High         Good balance   │
        │    256            ~2.1M             ~134M                Very high    Diminishing    │
        │   2048            ~16.8M            ~1.07B               Excessive    Defeats purpose│
        │                                                                                      │
        │   * = common default                                                                 │
        │                                                                                      │
        └──────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 6:  DATA FLOW THROUGH ENTIRE MODEL — FORWARD PASS WITH ADAPTERS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────


    Input: "Classify sentiment: This movie was great" → "positive"
    
    Complete forward pass data flow: tokenize → embed → through all 32 layers with both adapters → logits → loss

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   STEP 1: TOKENIZE + EMBED (same as full fine-tuning, nothing changes here)                      │
    │                                                                                                  │
    │   "Classify sentiment: This movie was great"                                                     │
    │       ↓ tokenizer                                                                                │
    │   [1, 518, 25580, 29962, 4134, 1598, ..., 2]     ← integer IDs                                   │
    │       ↓ embedding layer (FROZEN)                                                                 │
    │   [batch=1, seq_len=20, hidden=4096]               ← dense vectors                               │
    │                                                                                                  │
    └────────────────────────────────────┬─────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   STEP 2: FLOW THROUGH LAYER 0                                                                   │
    │                                                                                                  │
    │   h = [1, 20, 4096]                                                                              │
    │     │                                                                                            │
    │     ▼                                                                                            │
    │   ┌─────────────────────────────┐                                                                │
    │   │ Self-Attention (FROZEN)     │   Wq, Wk, Wv, Wo all frozen                                    │
    │   │ Computes Q, K, V, output    │   Gradients pass THROUGH but are NOT STORED                    │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │                  ▼                                                                               │
    │   ╔═══════════════════════════════════════╗                                                      │
    │   ║  ADAPTER 1 (TRAINABLE)                ║                                                      │
    │   ║                                       ║                                                      │
    │   ║  h_attn ──┬──▶ W_down ──▶ ReLU        ║                                                      │
    │   ║           │    [4096→64]              ║                                                      │
    │   ║           │        │                  ║                                                      │
    │   ║           │        ▼                  ║                                                      │
    │   ║           │    W_up ──▶ adapter_out   ║                                                      │
    │   ║           │    [64→4096]     │        ║                                                      │
    │   ║           │                  │        ║                                                      │
    │   ║           └──────── + ◄──────┘        ║   ← residual connection                              │
    │   ║                     │                 ║                                                      │
    │   ╚═════════════════════╪═════════════════╝                                                      │
    │                         │                                                                        │
    │                         ▼                                                                        │
    │   ┌─────────────────────────────┐                                                                │
    │   │ Add & LayerNorm             │                                                                │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │                  ▼                                                                               │
    │   ┌─────────────────────────────┐                                                                │
    │   │ Feed-Forward MLP (FROZEN)   │   W_gate, W_up, W_down all frozen                              │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │                  ▼                                                                               │
    │   ╔══════════════════════════════════════╗                                                       │
    │   ║  ADAPTER 2 (TRAINABLE)               ║                                                       │
    │   ║                                      ║                                                       │
    │   ║  h_ffn ──┬──▶ W_down ──▶ ReLU        ║                                                       │
    │   ║          │    [4096→64]              ║                                                       │
    │   ║          │        │                  ║                                                       │
    │   ║          │        ▼                  ║                                                       │
    │   ║          │    W_up ──▶ adapter_out   ║                                                       │
    │   ║          │    [64→4096]     │        ║                                                       │
    │   ║          │                  │        ║                                                       │
    │   ║          └──────── + ◄──────┘        ║   ← residual connection                               │
    │   ║                    │                 ║                                                       │
    │   ╚════════════════════╪═════════════════╝                                                       │
    │                        │                                                                         │
    │                        ▼                                                                         │
    │   ┌─────────────────────────────┐                                                                │
    │   │ Add & LayerNorm             │                                                                │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │                  ▼                                                                               │
    │   Output of Layer 0: h' = [1, 20, 4096]  → passes to Layer 1                                     │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         │    (repeat for layers 1-31, each with its own 2 adapters)
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   STEP 3: FINAL OUTPUT + LOSS                                                                    │
    │                                                                                                  │
    │   Output of Layer 31: h_final = [1, 20, 4096]                                                    │
    │       ↓ LM Head (FROZEN)                                                                         │
    │   logits = [1, 20, 32000]         ← probability over entire vocabulary                           │
    │       ↓                                                                                          │
    │   Loss = CrossEntropy(logits for output positions, target="positive")                            │
    │       ↓                                                                                          │
    │   Only positions where labels ≠ -100 contribute to loss (loss masking)                           │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 7:  BACKWARD PASS — WHERE GRADIENTS FLOW (AND DON'T)
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Backward pass showing exactly where gradients flow through (frozen layers) vs. where they're stored (adapters only), 
    plus the optimizer update

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   Loss                                                                                           │
    │     │                                                                                            │
    │     │  ∂Loss/∂logits                                                                             │
    │     ▼                                                                                            │
    │   ┌─────────────────────────────┐                                                                │
    │   │ LM Head (FROZEN)            │                                                                │
    │   │                             │                                                                │
    │   │ Gradients flow THROUGH ──▶  │   Gradients pass through for chain rule                        │
    │   │ but NOT STORED for update   │   but NO gradient tensor allocated for these weights           │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │     ▼ ▼ ▼ ▼ ▼ ▼ (back through layers 31 → 0)                                                     │
    │                                                                                                  │
    │   ┌─────────────────────────────┐                                                                │
    │   │ Feed-Forward (FROZEN)       │   Gradients flow through, NOT stored                           │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │   ╔══════════════╧══════════════════════════════════════════════════════════════════╗            │
    │   ║  ADAPTER 2                                                                      ║            │
    │   ║                                                                                 ║            │
    │   ║  Gradients:                                                                     ║            │
    │   ║    ∂Loss/∂W_up    →  COMPUTED and STORED  →  Used to update W_up                ║            │
    │   ║    ∂Loss/∂b_up    →  COMPUTED and STORED  →  Used to update b_up                ║            │
    │   ║    ∂Loss/∂W_down  →  COMPUTED and STORED  →  Used to update W_down              ║            │
    │   ║    ∂Loss/∂b_down  →  COMPUTED and STORED  →  Used to update b_down              ║            │
    │   ║                                                                                 ║            │
    │   ║  These are the ONLY gradients that get stored in this part of the network       ║            │
    │   ║                                                                                 ║            │
    │   ╚══════════════╤══════════════════════════════════════════════════════════════════╝            │
    │                  │                                                                               │
    │   ┌──────────────┴──────────────┐                                                                │
    │   │ Self-Attention (FROZEN)     │   Gradients flow through, NOT stored                           │
    │   └──────────────┬──────────────┘                                                                │
    │                  │                                                                               │
    │   ╔══════════════╧══════════════════════════════════════════════════════════════════╗            │
    │   ║  ADAPTER 1                                                                      ║            │
    │   ║                                                                                 ║            │
    │   ║  ∂Loss/∂W_up, ∂Loss/∂W_down, etc. → ALL STORED for update                       ║            │
    │   ║                                                                                 ║            │
    │   ╚══════════════╤══════════════════════════════════════════════════════════════════╝            │
    │                  │                                                                               │
    │                  ▼                                                                               │
    │                                                                                                  │
    │   OPTIMIZER UPDATE:                                                                              │
    │                                                                                                  │
    │       Adam updates ONLY adapter params:                                                          │
    │                                                                                                  │
    │       for param in adapter_parameters:                                                           │
    │           momentum[param]  = β₁ × momentum[param]  + (1-β₁) × gradient                           │
    │           variance[param]  = β₂ × variance[param]  + (1-β₂) × gradient²                          │
    │           param           -= lr × momentum / (√variance + ε)                                     │
    │                                                                                                  │
    │       Frozen params: NO optimizer states, NO updates, NO memory allocated                        │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 8:  MEMORY LAYOUT — WHAT LIVES WHERE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Memory layout comparison: Full FT (~94-114 GB) vs. Additive PEFT (~19-25 GB), with bar-style visualization showing where the savings come from

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   FULL FINE-TUNING MEMORY (7B model, BF16 + Adam)                                                │
    │                                                                                                  │
    │   GPU VRAM:                                                                                      │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │ Model Weights (BF16)        ██████████████  14 GB                                        │   │
    │   │ Gradients (BF16)            ██████████████  14 GB                                        │   │
    │   │ Optimizer States (FP32)     ████████████████████████████████████████████████████  56 GB  │   │
    │   │ Activations                 ██████████  10-30 GB                                         │   │
    │   │                                                                                          │   │
    │   │ TOTAL: ~94-114 GB                                                                        │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    │                                                                                                  │
    │   ADDITIVE PEFT MEMORY (7B model, BF16 base + adapters)                                          │
    │                                                                                                  │
    │   GPU VRAM:                                                                                      │
    │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
    │   │ Frozen Weights (BF16)       ██████████████  14 GB     (forward only, no grad/optimizer)  │   │
    │   │ Adapter Weights (BF16)      ▌  ~130 MB                                                   │   │
    │   │ Adapter Gradients (BF16)    ▌  ~130 MB                                                   │   │
    │   │ Adapter Optimizer (FP32)    █  ~520 MB      (Adam: momentum + variance for adapters)     │   │
    │   │ Activations                 ██████  4-10 GB  (reduced — fewer backward paths)            │   │
    │   │                                                                                          │   │
    │   │ TOTAL: ~19-25 GB                                                                         │   │
    │   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                  │
    │                                                                                                  │
    │   WHERE THE SAVINGS COME FROM:                                                                   │
    │                                                                                                  │
    │       Gradients saved:          14 GB    → ~130 MB         (eliminated for frozen params)        │
    │       Optimizer states saved:   56 GB    → ~520 MB         (eliminated for frozen params)        │
    │       Activation savings:       10-30 GB → 4-10 GB         (fewer backprop paths needed)         │
    │       ─────────────────────────────────────────────                                              │
    │       Total saved:              ~70-90 GB                                                        │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 9:  PARAMETER COUNT BREAKDOWN ACROSS THE MODEL
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Full parameter count breakdown: every component in a 7B model, which are frozen, which are trainable

    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                  │
    │   7B MODEL WITH BOTTLENECK ADAPTERS (bottleneck_dim=64, 32 layers, 2 adapters/layer)             │
    │                                                                                                  │
    │                                                                                                  │
    │   Component                  Params              Trainable?      Notes                           │
    │   ─────────                  ──────              ──────────      ─────                           │
    │   Embedding layer            131M                FROZEN          [32000 vocab × 4096]            │
    │                                                                                                  │
    │   Per transformer layer:                                                                         │
    │     Self-Attention                                                                               │
    │       W_q                    16.8M               FROZEN          [4096 × 4096]                   │
    │       W_k                    16.8M               FROZEN          [4096 × 4096]                   │
    │       W_v                    16.8M               FROZEN          [4096 × 4096]                   │
    │       W_o                    16.8M               FROZEN          [4096 × 4096]                   │
    │     ██ Adapter 1 ██          ~0.53M              ★ TRAINABLE     [4096→64→4096] + biases         │
    │     LayerNorm                8K                  FROZEN          [4096] × 2                      │
    │     Feed-Forward MLP                                                                             │
    │       W_gate                 45.1M               FROZEN          [4096 × 11008]                  │
    │       W_up                   45.1M               FROZEN          [4096 × 11008]                  │
    │       W_down                 45.1M               FROZEN          [11008 × 4096]                  │
    │     ██ Adapter 2 ██          ~0.53M              ★ TRAINABLE     [4096→64→4096] + biases         │
    │     LayerNorm                8K                  FROZEN          [4096] × 2                      │
    │                                                                                                  │
    │   LM Head                    131M                FROZEN          [4096 × 32000]                  │
    │                                                                                                  │
    │   ─────────────────────────────────────────────────────────────────────────────────              │
    │   FROZEN:     ~6,738M  (99.5%)                                                                   │
    │   TRAINABLE:  ~33.6M   (0.5%)    ← just the adapters                                             │
    │   ─────────────────────────────────────────────────────────────────────────────────              │
    │                                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 10:  TRAINING LOOP — ADDITIVE PEFT END-TO-END
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    End-to-end training loop: 
    
        freeze → insert adapters → data prep → forward → loss → backward → update → save


        ┌──────────────────┐             ┌──────────────────────┐
        │  Pre-trained     │             │  Task-Specific Data  │
        │  Model           │             │  (JSONL / Parquet)   │
        │  (e.g. LLaMA-7B) │             │                      │
        └──────┬───────────┘             └──────────┬───────────┘
               │                                    │
               ▼                                    │
        ┌───────────────────────────────┐           │
        │  STEP 1: FREEZE ALL PARAMS    │           │
        │                               │           │
        │  for p in model.parameters(): │           │
        │      p.requires_grad = False  │           │
        └──────────────┬────────────────┘           │
                       │                            │
                       ▼                            │
        ┌───────────────────────────────┐           │
        │  STEP 2: INSERT ADAPTERS      │           │
        │                               │           │
        │  For each of the 32 layers:   │           │
        │    Insert Adapter after       │           │
        │    self-attention             │           │
        │    Insert Adapter after       │           │
        │    feed-forward               │           │
        │                               │           │
        │  New params: requires_grad    │           │
        │  = True (trainable)           │           │
        └──────────────┬────────────────┘           │
                       │                            │
                       ▼                            ▼
        ┌──────────────────────────────────────────────────┐
        │  STEP 3: DATA PREPARATION (identical to Full FT) │
        │                                                  │
        │  Raw text → Chat template → Tokenize →           │
        │  Loss mask → Pad → Attention mask → Batch        │
        │                                                  │
        │  batch = {                                       │
        │    "input_ids":      [B, seq_len],               │
        │    "attention_mask": [B, seq_len],               │
        │    "labels":         [B, seq_len]  (-100 mask)   │
        │  }                                               │
        └─────────────────────────┬────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │  STEP 4: TRAINING LOOP                                               │
        │                                                                      │
        │  for epoch in range(num_epochs):                                     │
        │    for batch in dataloader:                                          │
        │                                                                      │
        │      ┌────────────────────────────────────────────────────────┐      │
        │      │  A) FORWARD PASS                                       │      │
        │      │                                                        │      │
        │      │  Input → Embed(FROZEN) → Layer 0:                      │      │
        │      │    Attn(FROZEN) → Adapter1(TRAIN) → Norm →             │      │
        │      │    FFN(FROZEN)  → Adapter2(TRAIN) → Norm               │      │
        │      │  → Layer 1 ... → Layer 31 → LM Head(FROZEN)            │      │
        │      │  → logits                                              │      │
        │      └──────────────────────────┬─────────────────────────────┘      │
        │                                 │                                    │
        │      ┌──────────────────────────▼─────────────────────────────┐      │
        │      │  B) LOSS = CrossEntropy(logits, labels)                │      │
        │      │     (only where labels ≠ -100)                         │      │
        │      └──────────────────────────┬─────────────────────────────┘      │
        │                                 │                                    │
        │      ┌──────────────────────────▼─────────────────────────────┐      │
        │      │  C) BACKWARD PASS                                      │      │
        │      │                                                        │      │
        │      │  loss.backward()                                       │      │
        │      │                                                        │      │
        │      │  Gradients flow through frozen layers (chain rule)     │      │
        │      │  Gradients STORED only for adapter W_down, W_up, b's   │      │
        │      │  (~33M params, ~130 MB of gradient storage)            │      │
        │      └──────────────────────────┬─────────────────────────────┘      │
        │                                 │                                    │
        │      ┌──────────────────────────▼─────────────────────────────┐      │
        │      │  D) OPTIMIZER STEP                                     │      │
        │      │                                                        │      │
        │      │  optimizer.step()  →  updates ONLY adapter params      │      │
        │      │  optimizer.zero_grad()                                 │      │
        │      │                                                        │      │
        │      │  Frozen 7B params: completely untouched                │      │
        │      └────────────────────────────────────────────────────────┘      │
        │                                                                      │
        │  Repeat for all batches, all epochs                                  │
        └─────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │  STEP 5: SAVE ADAPTERS ONLY                                          │
        │                                                                      │
        │  Saved files:                                                        │
        │    adapter_weights.pt     ~130 MB   (just the adapter parameters)    │
        │    adapter_config.json    ~1 KB     (bottleneck dim, base model)     │
        │                                                                      │
        │  Base model: NOT saved (referenced by name, loaded from Hub)         │
        │                                                                      │
        │  Compare full fine-tuning: model.safetensors = 14 GB                 │
        └──────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 11:  ADDITIVE vs REPARAMETERIZATION (LoRA) — ARCHITECTURAL DIFFERENCE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Adapters (in-series) vs. LoRA (in-parallel) architectural comparison, highlighting why LoRA won (mergeability)
    
    
        ADDITIVE (Adapters)                              REPARAMETERIZATION (LoRA)
        Modules inserted IN SERIES                       Bypass added IN PARALLEL

        ┌───────────────────┐                            ┌───────────────────┐
        │  Input x          │                            │  Input x          │
        └─────────┬─────────┘                            └────┬──────────┬───┘
                  │                                           │          │
                  ▼                                           │          │
        ┌───────────────────┐                                 │          │
        │  W₀ (FROZEN)      │                                 ▼          ▼
        │  Self-Attention   │                       ┌──────────────┐  ┌─────┐
        └─────────┬─────────┘                       │  W₀ (FROZEN) │  │  A  │
                  │                                 │              │  │(down│
                  ▼                                 └──────┬───────┘  └──┬──┘
        ╔═════════════════════╗                            │             │
        ║  ADAPTER (TRAIN)    ║                            │             ▼
        ║  4096 → 64 → 4096   ║                            │          ┌─────┐
        ║  + residual         ║                            │          │  B  │
        ╚═════════╤═══════════╝                            │          │(up) │
                  │                                        │          └──┬──┘
                  ▼                                        │             │
        ┌───────────────────┐                              ▼             ▼
        │  Continue...      │                          ┌────────┐
        └───────────────────┘                          │   +    │───▶ h
                                                       └────────┘

        Data flows THROUGH                              Data flows through W₀
        the adapter sequentially.                       AND through A→B in parallel.
        Adapter is always present                       After training, merge: W = W₀ + BA
        at inference time.                              LoRA disappears. Zero overhead.

        ┌──────────────────────────────────────────────────────────────────────────────┐
        │                                                                              │
        │  KEY DIFFERENCE:                                                             │
        │                                                                              │
        │  Adapters:  CANNOT be merged.  Extra latency at inference.  PERMANENT.       │
        │  LoRA:      CAN be merged.     Zero latency at inference.   REMOVABLE.       │
        │                                                                              │
        │  This single difference is why LoRA replaced Adapters as the standard.       │
        │                                                                              │
        └──────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 12:  ALL THREE ADDITIVE SUB-METHODS — SIDE BY SIDE
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    All three additive sub-methods side by side: Bottleneck Adapters, (IA)³, and Soft Prompts, with expressiveness vs. efficiency bars
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                      │
    │                                   ADDITIVE PEFT METHODS                                              │
    │                                                                                                      │
    │   ┌────────────────────────────┐  ┌────────────────────────────┐  ┌────────────────────────────────┐ │
    │   │   BOTTLENECK ADAPTERS      │  │   (IA)³                    │  │   SOFT PROMPTS                 │ │
    │   │                            │  │                            │  │   (sometimes classified here)  │ │
    │   │   Insert small FFN         │  │   Learn 3 rescaling        │  │                                │ │
    │   │   modules between layers   │  │   vectors per layer        │  │   Learn k continuous vectors   │ │
    │   │                            │  │                            │  │   prepended to input           │ │
    │   │                            │  │                            │  │                                │ │
    │   │   ┌──────┐                 │  │   K activations:           │  │   [v₁, v₂, ..., v_k, tokens]   │ │
    │   │   │ 4096 │ → 64 → 4096     │  │     K' = l_k ⊙ K           │  │    ↑ trainable    ↑ frozen     │ │
    │   │   │      │   ↑             │  │   V activations:           │  │                                │ │
    │   │   │      │   bottleneck    │  │     V' = l_v ⊙ V           │  │   v₁..v_k are free-floating    │ │
    │   │   └──────┘                 │  │   FFN activations:         │  │   vectors in embedding space   │ │
    │   │                            │  │     FFN' = l_ff ⊙ FFN(x)   │  │   — no real words correspond   │ │
    │   │   Has non-linearity (ReLU) │  │                            │  │                                │ │
    │   │   Has residual connection  │  │   ⊙ = element-wise mult    │  │   No architectural change      │ │
    │   │   Inserted in-series       │  │   Just scaling, no new     │  │   Just extra input tokens      │ │
    │   │                            │  │   layers or modules        │  │                                │ │
    │   │                            │  │                            │  │                                │ │
    │   │   Params: ~0.5-3% of model │  │   Params: ~0.01% of model  │  │   Params: ~0.001% of model     │ │
    │   │   Quality: Good            │  │   Quality: Moderate        │  │   Quality: Moderate (at scale) │ │
    │   │   Inference: Slower        │  │   Inference: Minimal cost  │  │   Inference: Minimal cost      │ │
    │   │   (extra layers in path)   │  │   (just 3 multiplications) │  │   (just extra tokens)          │ │
    │   │                            │  │                            │  │                                │ │
    │   └────────────────────────────┘  └────────────────────────────┘  └────────────────────────────────┘ │
    │                                                                                                      │
    │   EXPRESSIVENESS:   Adapters  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░     High (non-linear, sequential)                 │
    │                     (IA)³     ▓▓▓▓▓▓▓▓░░░░░░░░░░░░     Moderate (linear rescaling only)              │
    │                     Soft P.   ▓▓▓▓▓░░░░░░░░░░░░░░░     Lower (input-level only, no depth)            │ 
    │                                                                                                      │
    │   EFFICIENCY:       Adapters  ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░     Good (0.5-3% params)                          │
    │                     (IA)³     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░     Excellent (0.01% params)                      │
    │                     Soft P.   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     Best (0.001% params)                          │
    │                                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────┘

---

    ──────────────────────────────────────────────────────────────────────────────────────────────────────
    DIAGRAM 13:  INFERENCE — THE PERMANENT COST OF ADAPTERS
    ──────────────────────────────────────────────────────────────────────────────────────────────────────

    Inference cost: showing the permanent latency penalty of adapters vs. zero overhead after LoRA merge
        
                INFERENCE (serving predictions to users)

        FULL FINE-TUNING:                    ADDITIVE (Adapters):                 LoRA (after merge):

        ┌────────────────────┐               ┌────────────────────┐               ┌────────────────────┐
        │  Input             │               │  Input             │               │  Input             │
        │    ↓               │               │    ↓               │               │    ↓               │
        │  Attention         │               │  Attention         │               │  Attention         │
        │    ↓               │               │    ↓               │               │    ↓               │
        │  Norm              │               │  ██ Adapter 1 ██   │               │  Norm              │
        │    ↓               │               │    ↓               │               │    ↓               │
        │  FFN               │               │  Norm              │               │  FFN               │
        │    ↓               │               │    ↓               │               │    ↓               │
        │  Norm              │               │  FFN               │               │  Norm              │
        │    ↓               │               │    ↓               │               │    ↓               │
        │  Output            │               │  ██ Adapter 2 ██   │               │  Output            │
        │                    │               │    ↓               │               │                    │
        │  6 operations      │               │  Norm              │               │  6 operations      │
        │                    │               │    ↓               │               │  (same as original)│
        │                    │               │  Output            │               │                    │
        │                    │               │                    │               │  No adapters.      │
        │                    │               │  8 operations      │               │  Merged into W.    │
        │                    │               │  (+33% more work)  │               │  Zero overhead.    │
        └────────────────────┘               └────────────────────┘               └────────────────────┘

        Speed: Baseline                     Speed: ~10-30% slower               Speed: Baseline
                                            (extra adapter compute               (adapters gone)
                                             at every layer)

---
========================================================================================================================
---

            ════════════════════════════════════════════════════════════════════════════════
                                    Bottleneck Adapters & (IA)³
            ════════════════════════════════════════════════════════════════════════════════

The Core Idea — What Makes Additive Different
Every PEFT method freezes the base model and trains something tiny. The question is what you add and where you put it.
    
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
    
---
    
The Defining Trade-Off: Additive Methods vs LoRA

    | Property                      | **LoRA**               | **Bottleneck Adapters**          | **(IA)³**                         |
    | ----------------------------- | ---------------------- | -------------------------------- | --------------------------------- |
    | **Inference Overhead**        | Zero (after merge)     | Permanent (extra modules remain) | ~Zero (just elementwise multiply) |
    | **Architecture Change**       | None (parallel bypass) | Yes (new serial layers added)    | Minimal (scale vectors applied)   |
    | **Expressiveness**            | High                   | Highest (has non-linearity)      | Lowest                            |
    | **Parameters Added**          | ~0.1–1%                | ~0.5–3%                          | ~0.01–0.1%                        |
    | **Non-linearity**             | No                     | Yes (ReLU/GELU in bottleneck)    | No                                |
    | **Mergeable into Base Model** | Yes                    | No                               | Yes (into weight matrices)        |

    
---

##### PART 1: BOTTLENECK ADAPTERS

What Problem Bottleneck Adapters Solve
    Houlsby et al. (2019) — the original adapter paper — made a simple observation:

"What if we could add a small trainable module after each transformer layer,
leave everything else frozen, and teach only that module the task ?"

The module had to be small (so it doesn't eat all VRAM) but expressive enough
to capture complex task-specific transformations. 

**The solution: a bottleneck.**


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



**The bottleneck dimension (64 in the example) is the key hyperparameter — called reduction_factor or bottleneck_dim. 
Smaller = fewer params, less expressive.**

---

### The Math

For a single adapter module applied to hidden state h:

        h' = h  +  W_up · activation( W_down · h  +  b_down )  +  b_up
         ↑            ↑                 ↑
      residual      expand          compress + non-linearity
      (original)    back up         through bottleneck
          
Where:
    - W_down: [hidden_dim × bottleneck_dim]   e.g. [4096 × 64]
    - W_up:   [bottleneck_dim × hidden_dim]   e.g. [64 × 4096]
    - b_down, b_up: bias vectors
    - activation: ReLU or GELU


Initialization (the identity trick):

    W_up  = zeros          →    W_up · anything = 0
    W_down = random        →    b_up + W_up · act(W_down · h) = 0
    
    Therefore at step 0:   h' = h + 0 = h    ← perfect identity function

Same principle as LoRA's B=0 init — the adapter starts transparent and learns to deviate from identity only as training progresses.

---

### Where Adapters Are Placed in the Transformer

This is where Additive differs fundamentally from LoRA. LoRA adds a parallel bypass to individual weight matrices.
Adapters are inserted as entire new sequential modules between existing layers.

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │   ORIGINAL TRANSFORMER LAYER (no adapters):                                 │
    │                                                                             │
    │                            Input h                                          │
    │                              │                                              │
    │                              ▼                                              │
    │             [Self-Attention: Q, K, V, O projections]                        │
    │                              │                                              │
    │                              ▼                                              │
    │                     [Add & LayerNorm]                                       │
    │                              │                                              │
    │                              ▼                                              │
    │                [Feed-Forward MLP: gate, up, down]                           │
    │                              │                                              │
    │                              ▼                                              │
    │                       [Add & LayerNorm]                                     │
    │                              │                                              │
    │                              ▼                                              │
    │                           Output h'                                         │
    │                                                                             │
    │ ─────────────────────────────────────────────────────────────────────────── │
    │                                                                             │
    │   HOULSBY (2019) — 2 adapters per layer:                                    │
    │                                                                             │
    │                           Input h                                           │
    │                             │                                               │
    │                             ▼                                               │
    │                      [Self-Attention]                                       │
    │                             │                                               │
    │                             ▼                                               │
    │                      [Add & LayerNorm]                                      │
    │                             │                                               │
    │                             ▼                                               │
    │          ╔══════════════╗  ← ADAPTER 1 (after attention)   ← TRAINABLE      │
    │          ║ Down→Act→Up  ║    + residual                                     │
    │          ║ + residual   ║                                                   │
    │          ╚══════════════╝                                                   │
    │                             │                                               │
    │                             ▼                                               │
    │                      [Feed-Forward MLP]                                     │
    │                             │                                               │
    │                             ▼                                               │
    │                      [Add & LayerNorm]                                      │
    │                             │                                               │
    │                             ▼                                               │
    │          ╔══════════════╗  ← ADAPTER 2 (after FFN)   ← TRAINABLE            │
    │          ║ Down→Act→Up  ║    + residual                                     │
    │          ║ + residual   ║                                                   │
    │          ╚══════════════╝                                                   │
    │               │                                                             │
    │               ▼                                                             │
    │          Output h'                                                          │
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

Houlsby = 2 adapters per layer, higher capacity, more parameters.
Pfeiffer = 1 adapter per layer, more efficient, nearly same performance.
Pfeiffer is the standard today.

    Houlsby refers to the adapter method introduced in:
    Neil Houlsby et al., 2019
    Paper: “Parameter-Efficient Transfer Learning for NLP”

---

### The Complete Forward Pass

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

---

### Backward Pass — Where the Gradients Go

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

---

##### Parameter Count

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

---

### Why Adapters Can't Merge (the key LoRA advantage they lack)

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

---

### Saving and Loading Adapters

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

---

##### PART 2: (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

What (IA)³ Is
Liu et al. (2022). The paper's title is the abbreviation: Infused Adapter by
Inhibiting and Amplifying Inner Activations.

The insight is radical simplicity: don't add new modules. Don't add new weight matrices. 
Just learn vectors that rescale the existing activations.

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

---

### The Math

(IA)³ learns three sets of vectors — l_k, l_v, and l_ff:

Attention:

    K = (l_k ⊙ W_k) · x        ← rescale KEY vectors element-wise
    V = (l_v ⊙ W_v) · x        ← rescale VALUE vectors element-wise

    The l vectors gate which features in K and V are amplified or suppressed.
    High values in l → that dimension matters more for this task.
    Low values in l → that dimension is inhibited.

Feed-Forward:
    
    output = (l_ff ⊙ GELU(gate)) × up    ← rescale gate activations

Initialization:
    
    l_k  = ones   →   l_k  ⊙  W_k = 1 × W_k = W_k   (identity — unchanged)
    l_v  = ones   →   l_v  ⊙  W_v = 1 × W_v = W_v   (identity — unchanged)
    l_ff = ones   →   l_ff ⊙  act = 1 × act = act   (identity — unchanged)
    
Same principle as B=0 in LoRA and W_up=0 in Adapters:
model starts as identical to base, deviations learned over training.

---

### Where (IA)³ Applies Scaling

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
    │                                   Add & LayerNorm                            │
    │                                            │                                 │
    │   ┌────────────────────────────────────────▼──────────────────────────────┐  │
    │   │  FEED-FORWARD MLP                                                     │  │
    │   │                                                                       │  │
    │   │  gate = W_gate · h                                                    │  │
    │   │  up   = W_up · h                                                      │  │
    │   │                                                                       │  │
    │   │  out = W_down · ( l_ff ⊙ GELU(gate) ) ⊙ up                            │  │
    │   │                    ↑                                                  │  │
    │   │              l_ff: [8192] learned vector — scales the gate activations│  │
    │   │              Note: shape is [intermediate_dim] not [hidden_dim]       │  │
    │   │                    because it gates the FFN's internal activations    │  │
    │   │                                                                       │  │
    │   └───────────────────────────────────────────────────────────────────────┘  │
    │                                            │                                 │
    │                                     Add & LayerNorm                          │
    │                                            │                                 │
    │                                         Output h'                            │
    │                                                                              │
    └──────────────────────────────────────────────────────────────────────────────┘

---

### Parameter Count — Remarkably Tiny

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

---

### Can (IA)³ Merge?

Yes — and this is a surprise given it's Additive. Because the rescaling is
LINEAR (element-wise multiply has no non-linearity), it CAN be folded in:

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

### (IA)³ Visualised — What the l Vectors Actually Do

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

---

##### PART 3: HEAD-TO-HEAD COMPARISON

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

---

### PART 4: COMPLETE DATA FLOW (both methods)

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

---

### PART 5: MEMORY LAYOUT — ALL THREE METHODS

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

---

### PART 6: PRACTICAL DECISION GUIDE

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

---

One Final Analogy

Think of adapting a musician to play a new genre:

<u>Full Fine-Tuning</u>: Retrain them completely — they learn the new genre but may forget old skills.

<u>LoRA</u>: Give them a cheat sheet that adds parallel notes to their existing playing. 
The cheat sheet can be folded into their memory (merged) and they play naturally.

<u>Bottleneck Adapters</u>: Give them a small filter pedal that runs their sound through a compression/expansion effect after they play. 
The pedal always stays in the signal chain — you can't remove it without also removing the effect.

<u>(IA)³</u>: Give them a tiny set of dials that turn certain aspects of their playing up or down — more treble here, 
less vibrato there. Minimal gear, just learned adjustments. The dials can be permanently set and removed once calibrated.

---
### Additive Visual
{{PEFT_ADDITIVE_IMAGE}}
----
### IA3 Visual
{{IA3_IMAGE}}
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

# ─────────────────────────────────────────────────────────────────────────────
# NEW OPERATIONS — Paste these into the OPERATIONS dict in
# 08_b_FT_PEFT_Additive_Breakdown.py
# ─────────────────────────────────────────────────────────────────────────────

    # ── BOTTLENECK ADAPTERS ────────────────────────────────────────────────────

    "Bottleneck Adapter — Manual Implementation": {
        "description": "Hand-rolled bottleneck adapter module showing exactly what PEFT inserts into the model",
        "runnable": True,
        "code": '''import torch
import torch.nn as nn

class BottleneckAdapter(nn.Module):
    """
    One adapter module:  h → W_down → activation → W_up → + h (residual)

    Shapes:
        hidden_dim    : transformer hidden size  (e.g. 4096 for LLaMA-7B)
        bottleneck_dim: compression dimension    (e.g. 64  — the key hyperparameter)
    """
    def __init__(self, hidden_dim: int = 4096, bottleneck_dim: int = 64):
        super().__init__()

        # Down-project: compress to bottleneck
        self.W_down = nn.Linear(hidden_dim, bottleneck_dim, bias=True)

        # Non-linearity — the key difference from LoRA (which is purely linear)
        self.activation = nn.GELU()

        # Up-project: expand back to hidden dim
        self.W_up = nn.Linear(bottleneck_dim, hidden_dim, bias=True)

        # Identity initialization: W_up = 0 → adapter outputs 0 at step 0
        # So h' = h + 0 = h  (model starts as original pre-trained model)
        nn.init.zeros_(self.W_up.weight)
        nn.init.zeros_(self.W_up.bias)

        # W_down: small random init (standard)
        nn.init.normal_(self.W_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_down.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h shape: [batch, seq_len, hidden_dim]
        residual = h                              # save original for skip connection
        x = self.W_down(h)                        # [batch, seq_len, bottleneck_dim]
        x = self.activation(x)                    # non-linearity
        x = self.W_up(x)                          # [batch, seq_len, hidden_dim]
        return residual + x                       # residual add — never destroys h

# ── Quick demo ─────────────────────────────────────────────────────────────
adapter = BottleneckAdapter(hidden_dim=4096, bottleneck_dim=64)

# Count parameters
total_params = sum(p.numel() for p in adapter.parameters())
print(f"Adapter parameters: {total_params:,}")
# Expected: (4096×64 + 64) + (64×4096 + 4096) = 262,208 + 266,240 = 528,448

# Verify identity initialization: output ≈ input before any training
h = torch.randn(2, 10, 4096)         # [batch=2, seq=10, hidden=4096]
h_out = adapter(h)
print(f"Input == Output at init: {torch.allclose(h, h_out)}")   # True
print(f"Max deviation: {(h - h_out).abs().max().item():.2e}")    # ≈ 0.0
''',
    },

    "Bottleneck Adapter — Insert into Frozen Transformer": {
        "description": "Freeze a HuggingFace model and inject adapter modules into every layer (Pfeiffer style — 1 per layer, after FFN)",
        "runnable": False,
        "code": '''import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load base model ────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# ── Step 2: Freeze ALL base model parameters ───────────────────────────────
for param in model.parameters():
    param.requires_grad = False
print(f"Frozen params: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 3: Define bottleneck adapter ─────────────────────────────────────
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        super().__init__()
        self.W_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act    = nn.GELU()
        self.W_up   = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.W_up.weight)   # identity init
        nn.init.zeros_(self.W_up.bias)

    def forward(self, h):
        return h + self.W_up(self.act(self.W_down(h)))

# ── Step 4: Wrap each transformer layer with Pfeiffer-style adapter ───────
# Pfeiffer = single adapter inserted AFTER the FFN (not after attention)
hidden_dim = model.config.hidden_size   # 4096 for LLaMA-7B

class LayerWithAdapter(nn.Module):
    """Wraps an existing frozen transformer layer and appends one adapter."""
    def __init__(self, original_layer, adapter):
        super().__init__()
        self.original_layer = original_layer   # frozen
        self.adapter = adapter                 # trainable

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)
        hidden_state = outputs[0]              # shape: [batch, seq, hidden]
        hidden_state = self.adapter(hidden_state)
        return (hidden_state,) + outputs[1:]  # pass through other outputs unchanged

# Inject adapters into every layer
for i, layer in enumerate(model.model.layers):
    adapter = BottleneckAdapter(hidden_dim=hidden_dim, bottleneck_dim=64)
    model.model.layers[i] = LayerWithAdapter(layer, adapter)

# ── Step 5: Verify only adapter params are trainable ──────────────────────
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,}  ({100 * trainable / total:.4f}%)")
# Expected: ~33.6M trainable out of ~7B total (~0.48%)
''',
    },

    "Bottleneck Adapter — Training Loop": {
        "description": "Minimal training loop for bottleneck adapters — only adapter params get gradients and optimizer states",
        "runnable": False,
        "code": '''import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Assumes: model already has adapters injected and base weights frozen
# Assumes: train_dataset is a HuggingFace Dataset with input_ids + labels

# ── Optimizer: only adapter parameters ────────────────────────────────────
# This is critical — frozen params are excluded entirely
adapter_params = [p for p in model.parameters() if p.requires_grad]
print(f"Optimizer managing: {sum(p.numel() for p in adapter_params):,} params")

optimizer = AdamW(
    adapter_params,
    lr=1e-4,           # slightly higher than full FT is fine — adapter is tiny
    weight_decay=0.01,
)

# ── DataLoader ─────────────────────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# ── Training loop ──────────────────────────────────────────────────────────
model.train()
for epoch in range(3):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(model.device)
        labels    = batch["labels"].to(model.device)    # -100 masks instruction tokens

        # Forward: flows through frozen layers + adapter modules
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward: gradients computed only for adapter params
        # Frozen params: gradients flow THROUGH them but are NOT stored
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ── Save: only the adapter weights ────────────────────────────────────────
adapter_state = {
    name: param
    for name, param in model.state_dict().items()
    if "adapter" in name          # only adapter weights, not the frozen base
}
torch.save(adapter_state, "adapter_weights.pt")
print(f"Saved {len(adapter_state)} adapter tensors")
# ~130 MB  vs  14 GB for full model checkpoint
''',
    },

    "Bottleneck Adapter — PEFT Library (AdapterHub style)": {
        "description": "Using the adapters library (successor to AdapterHub) for production-grade bottleneck adapters",
        "runnable": False,
        "code": '''# pip install adapters
# The 'adapters' library is the maintained successor to adapter-transformers / AdapterHub

from adapters import AutoAdapterModel, AdapterConfig
from transformers import AutoTokenizer

# ── Load model with adapter support ───────────────────────────────────────
model = AutoAdapterModel.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# ── Configure bottleneck adapter ──────────────────────────────────────────
config = AdapterConfig(
    mh_adapter=False,          # No adapter after multi-head attention (Pfeiffer style)
    output_adapter=True,       # Adapter after FFN output ← Pfeiffer placement
    reduction_factor=64,       # bottleneck_dim = hidden_dim / reduction_factor
                               # LLaMA-7B: 4096 / 64 = 64-dim bottleneck
    non_linearity="gelu",      # Activation inside the bottleneck
    residual_before_ln=True,   # Residual added before LayerNorm
)

# ── Add and activate adapter ──────────────────────────────────────────────
model.add_adapter("sentiment_task", config=config)
model.train_adapter("sentiment_task")       # Freezes base model, activates adapter
model.set_active_adapters("sentiment_task")

# Verify
model.print_trainable_parameters()
# Output: trainable params: ~33M / 7B total (~0.48%)

# ── Save and reload ───────────────────────────────────────────────────────
model.save_adapter("./adapters/sentiment_task", "sentiment_task")
# Saves: adapter_model.bin (~130 MB) + adapter_config.json

# Load on any copy of the same base model
model2 = AutoAdapterModel.from_pretrained("meta-llama/Llama-2-7b-hf")
model2.load_adapter("./adapters/sentiment_task")
model2.set_active_adapters("sentiment_task")
''',
    },

    "Bottleneck Adapter — Inspect What Was Added": {
        "description": "Examine adapter structure, parameter counts per layer, and confirm frozen vs trainable split",
        "runnable": False,
        "code": '''# After injecting adapters, inspect what changed

print("=" * 70)
print("TRAINABLE (adapter) PARAMETERS")
print("=" * 70)
trainable_total = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  TRAIN  {name:60s}  {str(list(param.shape)):20s}  {param.numel():>10,}")
        trainable_total += param.numel()

print()
print("=" * 70)
print("FROZEN (base model) PARAMETERS — first 5 only")
print("=" * 70)
frozen_total = 0
frozen_shown = 0
for name, param in model.named_parameters():
    if not param.requires_grad:
        frozen_total += param.numel()
        if frozen_shown < 5:
            print(f"  FROZEN {name:60s}  {str(list(param.shape)):20s}  {param.numel():>10,}")
            frozen_shown += 1
print(f"  ... ({frozen_total:,} frozen params total — not shown)")

print()
print(f"Trainable : {trainable_total:>12,}  ({100*trainable_total/(trainable_total+frozen_total):.4f}%)")
print(f"Frozen    : {frozen_total:>12,}  ({100*frozen_total/(trainable_total+frozen_total):.4f}%)")
print(f"Total     : {trainable_total+frozen_total:>12,}")

# Example output (7B, bottleneck=64, 32 layers, 2 adapters/layer):
# TRAIN  model.layers.0.adapter.W_down.weight   [64, 4096]    262,144
# TRAIN  model.layers.0.adapter.W_down.bias     [64]               64
# TRAIN  model.layers.0.adapter.W_up.weight     [4096, 64]    262,144
# TRAIN  model.layers.0.adapter.W_up.bias       [4096]          4,096
# ... repeated for all 32 layers
# Trainable:   33,554,432  (0.4975%)
# Frozen:   6,706,958,336  (99.5025%)
''',
    },

    # ── (IA)³ ─────────────────────────────────────────────────────────────────

    "IA3 — Manual Implementation": {
        "description": "Hand-rolled (IA)³ showing the three learned rescaling vectors and how they gate keys, values, and FFN activations",
        "runnable": True,
        "code": '''import torch
import torch.nn as nn

class IA3Layer(nn.Module):
    """
    (IA)³ for a single transformer layer.

    Learns three tiny vectors:
        l_k  [kv_dim]  — rescales Key   activations in attention
        l_v  [kv_dim]  — rescales Value activations in attention
        l_ff [ffn_dim] — rescales FFN   intermediate activations

    All initialized to 1.0 → identity at step 0 (no change to model output).
    """
    def __init__(self, hidden_dim: int, kv_dim: int, ffn_dim: int):
        super().__init__()
        # One learned scalar per feature dimension — initialized to 1 (identity)
        self.l_k  = nn.Parameter(torch.ones(kv_dim))    # gates Key features
        self.l_v  = nn.Parameter(torch.ones(kv_dim))    # gates Value features
        self.l_ff = nn.Parameter(torch.ones(ffn_dim))   # gates FFN activations

    def rescale_keys(self, K: torch.Tensor) -> torch.Tensor:
        # K: [batch, heads, seq, kv_dim]  →  scale last dim by l_k
        return K * self.l_k                # broadcasts over batch, heads, seq

    def rescale_values(self, V: torch.Tensor) -> torch.Tensor:
        return V * self.l_v

    def rescale_ffn(self, ffn_activations: torch.Tensor) -> torch.Tensor:
        # ffn_activations: [batch, seq, ffn_dim]
        return ffn_activations * self.l_ff

# ── Demo ───────────────────────────────────────────────────────────────────
hidden_dim = 4096
kv_dim     = 128    # per-head key/value dim (varies by model)
ffn_dim    = 8192   # intermediate FFN dim  (typically 4× hidden)

ia3 = IA3Layer(hidden_dim=hidden_dim, kv_dim=kv_dim, ffn_dim=ffn_dim)

total_params = sum(p.numel() for p in ia3.parameters())
print(f"(IA)³ parameters per layer: {total_params:,}")
# Expected: 128 + 128 + 8192 = 8,448
# vs Bottleneck Adapter: 528,448 per layer  (~63x more)

# Verify identity at init: rescaling by 1.0 → no change
K = torch.randn(2, 32, 10, kv_dim)
print(f"Keys unchanged at init: {torch.allclose(K, ia3.rescale_keys(K))}")   # True

ffn_act = torch.randn(2, 10, ffn_dim)
print(f"FFN unchanged at init:  {torch.allclose(ffn_act, ia3.rescale_ffn(ffn_act))}")  # True
''',
    },

    "IA3 — PEFT Library Setup": {
        "description": "Apply (IA)³ to a HuggingFace model using the PEFT library — the production way",
        "runnable": False,
        "code": '''from peft import IA3Config, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Load base model ────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# ── Configure (IA)³ ────────────────────────────────────────────────────────
ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,

    # Which modules receive the learned rescaling vectors:
    target_modules=["k_proj", "v_proj", "down_proj"],
    #   k_proj   ← l_k scales Key   projections
    #   v_proj   ← l_v scales Value projections
    #   down_proj← l_ff scales FFN output (down-projection in SwiGLU FFN)

    # Which modules are feedforward (their activations get l_ff rescaling)
    feedforward_modules=["down_proj"],
)

# ── Apply (IA)³ ────────────────────────────────────────────────────────────
model = get_peft_model(model, ia3_config)
model.print_trainable_parameters()
# Output: trainable params: 151,552 || all params: 6,738,808,832 || trainable%: 0.0022
#                           ~152K                                              ~0.002%
# Compare LoRA r=8: ~8M params  →  (IA)³ is ~53x smaller

# ── All base weights are frozen automatically ──────────────────────────────
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Frozen parameters: {frozen:,}")
''',
    },

    "IA3 — Training and Saving": {
        "description": "Training loop for (IA)³ — note the unusually high learning rate and tiny checkpoint size",
        "runnable": False,
        "code": '''from peft import IA3Config, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
model = get_peft_model(model, ia3_config)

dataset = load_dataset("json", data_files="train.jsonl", split="train")

training_args = SFTConfig(
    output_dir="./ia3-output",
    num_train_epochs=3,
    per_device_train_batch_size=8,    # larger batch OK — very few params to update
    gradient_accumulation_steps=4,
    learning_rate=3e-3,               # (IA)³ uses MUCH higher LR than LoRA (1e-4)
                                      # because l vectors are tiny — larger steps needed
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)
trainer.train()

# Save ONLY the l vectors — incredibly small checkpoint
trainer.save_model("./ia3-output/final")
# Saved files:
#   adapter_model.safetensors   ~0.5 MB   ← just 151K floats × 2 bytes
#   adapter_config.json         ~1 KB
#
# Compare:
#   LoRA checkpoint:            ~33 MB
#   Bottleneck checkpoint:      ~130 MB
#   Full fine-tuned model:      14 GB
''',
    },

    "IA3 — Merge into Base Weights": {
        "description": "Fold learned l vectors into base weight matrices for zero inference overhead",
        "runnable": False,
        "code": '''from peft import PeftModel
from transformers import AutoModelForCausalLM

# (IA)³ rescaling is linear: K = l_k ⊙ (W_k · h)
# This is equivalent to:    K = (diag(l_k) · W_k) · h
# So we can absorb l_k into W_k by scaling each row:
#   W_k_merged[i, :] = l_k[i] × W_k[i, :]

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)

# Load (IA)³ adapter on top
model = PeftModel.from_pretrained(base_model, "./ia3-output/final")

# Merge l vectors into weight matrices — same API as LoRA
# Under the hood: for each target module, multiplies W rows by the learned l scalar
model = model.merge_and_unload()

# Save merged model — standard HuggingFace format, no PEFT dependency
model.save_pretrained("./ia3-merged")
# Result: normal model checkpoint (14 GB), zero inference overhead
# l vectors no longer exist — they're baked into the weight matrices

# Load for inference — no PEFT import needed
from transformers import pipeline
pipe = pipeline("text-generation", model="./ia3-merged", tokenizer=tokenizer)
output = pipe("Classify sentiment: This movie was great →", max_new_tokens=5)
print(output[0]["generated_text"])
''',
    },

    "IA3 — Compare Parameter Counts vs Other Methods": {
        "description": "Side-by-side parameter count comparison: (IA)³ vs LoRA vs Bottleneck Adapters for a 7B model",
        "runnable": True,
        "code": '''# Parameter count calculator — no model loading needed

def count_lora_params(hidden_dim, num_layers, rank,
                      target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    # Each target module: A [r × d_in] + B [d_out × r]
    # For square projections: d_in = d_out = hidden_dim
    params_per_module = 2 * hidden_dim * rank
    return params_per_module * len(target_modules) * num_layers

def count_adapter_params(hidden_dim, num_layers, bottleneck_dim, adapters_per_layer=1):
    # W_down [hidden × bottleneck] + b_down [bottleneck]
    # W_up   [bottleneck × hidden] + b_up   [hidden]
    params_per_adapter = (hidden_dim * bottleneck_dim + bottleneck_dim +
                          bottleneck_dim * hidden_dim + hidden_dim)
    return params_per_adapter * adapters_per_layer * num_layers

def count_ia3_params(kv_dim, ffn_dim, num_layers):
    # l_k [kv_dim] + l_v [kv_dim] + l_ff [ffn_dim]  per layer
    return (2 * kv_dim + ffn_dim) * num_layers

# ── LLaMA-2-7B architecture constants ─────────────────────────────────────
HIDDEN = 4096
KV_DIM = 128    # per-head key/value dim (32 heads, GQA-style)
FFN    = 11008  # LLaMA-2-7B intermediate FFN dim
LAYERS = 32
TOTAL  = 6_738_415_616   # total base model params

configs = {
    "LoRA  r=4 ":  count_lora_params(HIDDEN, LAYERS, rank=4),
    "LoRA  r=8 ":  count_lora_params(HIDDEN, LAYERS, rank=8),
    "LoRA  r=16":  count_lora_params(HIDDEN, LAYERS, rank=16),
    "Adapters b=64 ": count_adapter_params(HIDDEN, LAYERS, bottleneck_dim=64),
    "Adapters b=128": count_adapter_params(HIDDEN, LAYERS, bottleneck_dim=128),
    "(IA)³      ":  count_ia3_params(KV_DIM, FFN, LAYERS),
}

print(f"{'Method':<22}  {'Trainable':>14}  {'% of Model':>12}  {'File Size':>12}")
print("-" * 68)
for name, params in configs.items():
    pct       = 100 * params / TOTAL
    size_mb   = params * 2 / (1024 ** 2)   # BF16 = 2 bytes per param
    print(f"{name:<22}  {params:>14,}  {pct:>11.4f}%  {size_mb:>10.1f} MB")

# Expected output:
# Method                    Trainable    % of Model     File Size
# ────────────────────────────────────────────────────────────────
# LoRA  r=4               8,388,608        0.1245%        16.0 MB
# LoRA  r=8              16,777,216        0.2491%        32.0 MB
# LoRA  r=16             33,554,432        0.4981%        64.0 MB
# Adapters b=64          33,685,504        0.5001%       128.0 MB
# Adapters b=128         67,371,008        1.0002%       128.0 MB  (approx)
# (IA)³                     360,448        0.0053%         0.7 MB
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

# def get_content():
#     """Return all content for this topic module."""
#     return {
#         "theory": THEORY,
#         "complexity": COMPLEXITY,
#         "operations": OPERATIONS,
#     }


def get_content():
    """Return all content for this topic module."""
    from Required_Images.peft_additive_visual import PEFT_ADDITIVE_HEIGHT, PEFT_ADDITIVE_HTML
    from Required_Images.peft_ia3_visual import IA3_VISUAL_HEIGHT, IA3_VISUAL_HTML


    # Interactive component data — app.py splits theory_raw at each
    # placeholder and injects an st.components.v1.html() iframe instead.
    interactive_components = [

        {
            "placeholder": "{{PEFT_ADDITIVE_IMAGE}}",
            "html": PEFT_ADDITIVE_HTML,
            "height": PEFT_ADDITIVE_HEIGHT,
        },
        {
            "placeholder": "{{IA3_IMAGE}}",
            "html": IA3_VISUAL_HTML,
            "height": IA3_VISUAL_HEIGHT,
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