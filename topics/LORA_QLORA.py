"""



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

### The Math — Step by Step

**In full fine-tuning:**

    For a weight matrix W₀ of shape [d_out × d_in] (e.g., [4096 × 4096]):

    W_new = W₀ + ΔW

    ΔW is also [4096 × 4096] = 16,777,216 parameters to learn. That's the problem.


**In LoRA:**

    Instead of learning the full ΔW, decompose it:

    ΔW ≈ B × A

    Where:
        A is [r × d_in]    →  e.g., [8 × 4096] = 32,768 parameters
        B is [d_out × r]   →  e.g., [4096 × 8] = 32,768 parameters

    Total LoRA parameters: 32,768 + 32,768 = 65,536
    vs. full ΔW:           16,777,216

    That's a 256× reduction for this single weight matrix.


    r is the "rank" — the key hyperparameter. Typical values: 4, 8, 16, 32, 64.


**The forward pass becomes:**

    h = W₀x + (B × A)x     (where x is the input)
      = W₀x + BAx           (just matrix multiplication)

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                 │
    │                          LoRA FORWARD PASS                                      │
    │                                                                                 │
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
    │                                                                                 │
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

---

### Which Layers Get LoRA? — Target Modules

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

    Minimal (original paper):      W_q, W_v only
    Standard (most popular):       W_q, W_k, W_v, W_o
    Aggressive (best performance): W_q, W_k, W_v, W_o, W_gate, W_up, W_down  (all linear layers)

    More targets = more trainable parameters = better performance but more memory.

    In HuggingFace PEFT config, this looks like:
        target_modules=["q_proj", "v_proj"]                       # minimal
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]   # standard
        target_modules="all-linear"                               # aggressive

    Each target module in EACH transformer layer gets its own A and B matrices.
    For a 32-layer model with 4 attention targets and rank 8:
        32 layers × 4 targets × 2 matrices (A, B) × (8 × 4096) = ~8.4M parameters
        vs. 7B total = ~0.12% of the model

---

### The Rank (r) — How to Choose It

    r = 1:    Extremely compressed. Each adapter captures only 1 direction of variation.
              Works for very simple tasks or when data is tiny.

    r = 4:    Very lean. Good for straightforward classification or single-skill tasks.

    r = 8:    The default sweet spot. Works well for most instruction-tuning and chat tasks.
              This is where most practitioners start.

    r = 16:   Better for complex tasks requiring nuanced adaptation.
              Moderate memory increase.

    r = 32:   Approaching diminishing returns for many tasks.
              Good for very different target domains (e.g., English model → code model).

    r = 64+:  Rarely needed. If you need this much capacity, consider whether
              full fine-tuning or a larger rank is actually buying you anything.


    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                    │
    │    RANK vs. PERFORMANCE vs. MEMORY  (for a 7B model, 4 attention targets)          │
    │                                                                                    │
    │    Rank    Trainable Params     % of Model     Adapter Size     Quality            │
    │    ────    ────────────────     ──────────     ────────────     ───────            │
    │      4       ~4.2M                0.06%          ~16 MB         Good               │
    │      8       ~8.4M                0.12%          ~33 MB         Very Good   ★      │
    │     16      ~16.8M                0.24%          ~67 MB         Excellent          │
    │     32      ~33.5M                0.48%         ~134 MB         Excellent          │
    │     64      ~67.1M                0.96%         ~268 MB         Marginal gain      │
    │                                                                                    │
    │    ★ = recommended starting point                                                  │
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

Step 0: Understand What Problem LoRA Is Solving

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

---

Step 2: Attach LoRA Matrices to Target Weight Matrices

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

---

Step 3: Initialize A and B — This Is Critical

LoRA uses a very specific initialization:

    Matrix A:  Random Gaussian (small random values, like normal init)

    Matrix B:  ALL ZEROS

Why does B start at zero? Because at training step 0:

    ΔW = B × A = 0 × A = 0     (zero matrix)

    So: W_effective = W₀ + ΔW = W₀ + 0 = W₀

The model starts exactly as the pre-trained model. The LoRA adapters contribute nothing initially.
Training begins from a known-good state — the pre-trained model's behavior is perfectly preserved at step 0.

If both A and B were randomly initialized, B × A would produce random noise added to every weight matrix,
potentially destroying the pre-trained knowledge immediately.

As training progresses, B learns non-zero values and the product B × A gradually steers the model toward the new task.

---

Step 4: Prepare the Data (Identical to Full Fine-Tuning)

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

Step 5: Forward Pass — Where LoRA Changes Things

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

Step 6: Loss Computation (Identical to Full Fine-Tuning)

    logits = model output: [batch_size, seq_len, 32000]    (32000 = vocab size)
    labels = [-100, -100, ..., -100, 6374, 2]              (-100 = ignore)

    loss = CrossEntropyLoss(logits, labels)

    Only positions where labels ≠ -100 contribute.
    The model is graded ONLY on predicting the response tokens.

Nothing changes here between LoRA and full fine-tuning. Same loss function, same masking.

---

Step 7: Backward Pass — Where the Memory Savings Happen

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

---

Step 8: Optimizer Update — Only Adapters Move

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

Step 9: Repeat the Loop

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

Step 10: Save — Only the Adapter

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

Step 11: Deploy — Merge or Swap

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

---

LORA Diagram



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
    │          │                         ┌──────┴───────┐                              │
    │          │                         │ ██ ADAPTER ██│ ← NEW, trainable             │
    │          │                         │ ██ down+up ██│                              │
    │          │                         └──────┬───────┘                              │
    │          │                                │                                      │
    │   ┌──────┴────────┐                ┌──────┴───────┐                              │
    │   │   Add & Norm  │                │   Add & Norm │                              │
    │   └──────┬────────┘                └──────┬───────┘                              │
    │          │                                │                                      │
    │   ┌──────┴───────┐                 ┌──────┴───────┐                              │
    │   │ Feed-Forward │                 │ Feed-Forward │ (frozen)                     │
    │   └──────┬───────┘                 └──────┬───────┘                              │
    │          │                                │                                      │
    │          │                         ┌──────┴───────┐                              │
    │          │                         │ ██ ADAPTER ██│ ← NEW, trainable             │
    │          │                         │ ██ down+up ██│                              │
    │          │                         └──────┬───────┘                              │
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

"""


"""

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


"""