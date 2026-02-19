"""
Fine Tuning - PEFT (Parameter-Efficient Fine-Tuning) Deep Dive: Prompt Tuning
==============================================================================

A comprehensive deep dive into Prompt Tuning — the most widely studied and
utilized method in the prompt-based PEFT family.

Covers the intuition, the math, the full forward pass, every hyperparameter,
comparisons with Prefix Tuning and P-Tuning v2, and complete HuggingFace
implementation from scratch.

Paper: "The Power of Scale for Parameter-Efficient Prompt Tuning"
       Lester, Al-Rfou & Constant (Google Research, 2021)
"""

TOPIC_NAME = "Fine_Tuning_PEFT_Prompt_Tuning"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

Prompt Tuning's core insight: **you don't need to change any model weights at all —
just prepend a few learnable token vectors to every input, and gradient descent
will figure out what those vectors should mean.**

Before PEFT methods existed, "prompting" meant carefully writing text to steer
a model toward a desired behaviour. GPT-3 showed this worked surprisingly well —
but it was fragile, human-intensive, and hit a hard ceiling on quality.

Prompt Tuning (Lester et al., 2021) asks: what if instead of hand-crafting text
prompts, we let gradient descent find the *optimal continuous* prompt in the
model's embedding space? The result is "soft prompts" — a small number of
floating-point vectors that are prepended to every input but never interpreted
as real words. They are not constrained to lie near any word embedding; they
are free parameters that gradient descent can place anywhere in the embedding
space to maximally steer the frozen model.

Think of it this way:
    Hard prompting is like giving verbal instructions to a very knowledgeable
    consultant — you're constrained to use real words, and the consultant may
    interpret your words imperfectly.

    Prompt Tuning is like having a direct neural interface with that consultant:
    instead of words, you inject a precise activation pattern directly into their
    first layer of thought. The pattern is not a sentence — it's a learned
    continuous signal that steers their reasoning more directly than any words could.

---


            ══════════════════════════════════════════════════════════════════════════════
                              THE PROMPT-BASED PEFT FAMILY — OVERVIEW
            ══════════════════════════════════════════════════════════════════════════════


    There are four main prompt-based PEFT methods. They differ in WHERE the
    soft tokens are inserted and HOW they interact with the transformer.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │                        PROMPT-BASED PEFT FAMILY TREE                             │
    │                                                                                  │
    │  Hard Prompting            (no training — just carefully written text prefixes)  │
    │        │                                                                         │
    │        │ ← add continuous parameters at the input embedding layer                │
    │        ▼                                                                         │
    │  Prompt Tuning ◄───── THIS MODULE (Lester et al., 2021)                          │
    │  Soft tokens at INPUT ONLY. Frozen model. Fewest parameters.                     │
    │        │                                                                         │
    │        │ ← extend soft tokens to every transformer layer                         │
    │        ▼                                                                         │
    │  Prefix Tuning                              (Li & Lam, 2021)                     │
    │  Soft tokens prepended to K and V at EVERY attention layer.                      │
    │  More parameters, more steering power.                                           │
    │        │                                                                         │
    │        │ ← add reparameterization MLP to stabilize training                      │
    │        ▼                                                                         │
    │  P-Tuning v1                                (Liu et al., 2021)                   │
    │  LSTM/MLP-parameterized prompts, can be inserted in MIDDLE of input.             │
    │        │                                                                         │
    │        │ ← unify with prefix tuning, apply to NLU not just NLG                   │
    │        ▼                                                                         │
    │  P-Tuning v2                                (Liu et al., 2022)                   │
    │  Deep prefix applied to ALL layers, layer-specific prefix parameters.            │
    │  The most powerful prompt-based method but approaches LoRA in complexity.        │
    │                                                                                  │
    │  VERDICT: Prompt Tuning is the most studied, simplest, and most widely used      │
    │  in research and in HuggingFace PEFT. At large model scales (≥11B), it           │
    │  matches full fine-tuning quality with 0.001% of the parameters.                 │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                WHAT IS PROMPT TUNING? — THE CORE MECHANISM
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Hard Prompts vs. Soft Prompts — The Fundamental Shift

In standard language model inference, every input goes through this pipeline:

    Input text:      "Classify the sentiment of this review: Great movie!"
                          │
                          │  tokenize
                          ▼
    Token IDs:       [9226, 262, 15598, 286, 428, 2423, 25, 3878, 3807, 0]
                          │
                          │  embedding lookup  (frozen E ∈ ℝ^{V × d})
                          ▼
    Embeddings:      [e₁, e₂, e₃, e₄, e₅, e₆, e₇, e₈, e₉, e₁₀]   each in ℝ^d
                          │
                          │  transformer layers (frozen)
                          ▼
    Output logits:   [...]

A "hard prompt" just changes the input text:
    "Decide if this is positive or negative. Review: Great movie!"
    → different token IDs → different embeddings → hopefully better output

The hard prompt is constrained: every token must correspond to a real vocabulary item.
The embedding of each token is FIXED in the embedding table — you can't move it.

**Soft Prompt Tuning replaces this constraint entirely.**

Instead of prepending real tokens to the text, you prepend a matrix of FREE parameters:

    Soft prompt P = [p₁, p₂, ..., p_k]   where each pᵢ ∈ ℝ^d

    These pᵢ vectors are NOT indexed from the embedding table.
    They are NOT constrained to correspond to any real word.
    They live at arbitrary locations in ℝ^d.
    They are the ONLY trainable parameters.

The full input to the transformer becomes:

    [p₁, p₂, ..., p_k, e₁, e₂, ..., e_n]

where pᵢ are the soft prompt vectors (trainable) and eⱼ are the regular token
embeddings (frozen, from the embedding table).

The transformer processes all of them with full self-attention — the soft tokens
can attend to the real tokens and vice versa. But only the soft prompt parameters
receive gradient updates during training.

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                PARAMETER MATHEMATICS — HOW FEW IS FEW?
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Counting Soft Prompt Parameters

The soft prompt is a single matrix:

    P ∈ ℝ^{k × d}

where:
    k = number of soft tokens (a hyperparameter, typically 1 to 150)
    d = model hidden dimension (determined by model architecture)

That's it. That's all the trainable parameters.

Let's compute the numbers for real models:

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   SOFT PROMPT PARAMETER COUNTS                                                       │
    │                                                                                      │
    │   Model           d_model    k=10 tokens    k=100 tokens    Model total params       │
    │   ─────           ───────    ───────────    ────────────    ──────────────────       │
    │   GPT-2 small     768        7,680          76,800          117,000,000              │
    │   GPT-2 XL        1,600      16,000         160,000         1,500,000,000            │
    │   T5-Small        512        5,120          51,200          60,000,000               │
    │   T5-Large        1,024      10,240         102,400         770,000,000              │
    │   T5-XL           2,048      20,480         204,800         3,000,000,000            │
    │   T5-XXL          4,096      40,960         409,600         11,000,000,000           │
    │   LLaMA-2-7B      4,096      40,960         409,600         7,000,000,000            │
    │   LLaMA-2-13B     5,120      51,200         512,000         13,000,000,000           │
    │                                                                                      │
    │   PERCENTAGE TRAINABLE (k=100):                                                      │
    │   GPT-2 small:  76,800 / 117M   =  0.066%                                            │
    │   T5-Large:    102,400 / 770M   =  0.013%                                            │
    │   T5-XXL:      409,600 / 11B    =  0.0037%                                           │
    │   LLaMA-2-7B:  409,600 / 7B     =  0.0059%                                           │
    │                                                                                      │
    │   Compare: LoRA (r=16) on LLaMA-2-7B ≈ 0.12% (20× more than Prompt Tuning)           │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

Prompt Tuning is by far the most parameter-efficient PEFT method — often 10-100× fewer
trainable parameters than LoRA. The tradeoff is that it requires larger base models
to match fine-tuning quality (discussed in depth in the "Scale Effect" section below).

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                         THE COMPLETE FORWARD PASS — TRACING A BATCH THROUGH PROMPT TUNING
    ═══════════════════════════════════════════════════════════════════════════════════════════════


Let's trace a concrete batch through a prompt-tuned T5-Large model.
Task: text classification. Prompt length k=20. d_model=1024. Batch size B=4. Seq len=128.

**STEP 0: INITIALIZATION**

    The soft prompt matrix P is initialized (discussed in detail later):
    P ∈ ℝ^{20 × 1024}   (20 tokens × 1024 dimensions)
    requires_grad = True
    All other model parameters: requires_grad = False

**STEP 1: TOKENIZE THE INPUT BATCH**

    Input texts (batch of 4):
        [0]: "Review: This laptop is incredibly fast and the battery lasts forever."
        [1]: "Review: Arrived broken, packaging was damaged, total waste of money."
        [2]: "Review: Decent product, works as expected, nothing special."
        [3]: "Review: Absolutely blown away by the quality and customer service!"

    After tokenization (each padded/truncated to 128 tokens):
        token_ids ∈ ℤ^{4 × 128}   (batch × seq_len)

**STEP 2: EMBED THE TOKEN IDS**

    token_embeddings = E[token_ids]   where E ∈ ℝ^{V × d} is the frozen embedding table
    token_embeddings ∈ ℝ^{4 × 128 × 1024}   (batch × seq_len × d_model)

    Memory: 4 × 128 × 1024 × 2 bytes (BF16) = 1.05 MB

**STEP 3: PREPEND THE SOFT PROMPT**

    Expand P to match the batch:
    P_expanded = P.unsqueeze(0).expand(4, 20, 1024)   ∈ ℝ^{4 × 20 × 1024}

    This is a BROADCAST — not a copy. P is shared across all items in the batch.
    The SAME soft prompt steers every example in the batch identically.

    Concatenate along sequence dimension:
    input_embeds = concat([P_expanded, token_embeddings], dim=1)
    input_embeds ∈ ℝ^{4 × 148 × 1024}   (batch × (20 soft + 128 token) × d_model)

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                 │
    │   SEQUENCE LAYOUT AFTER PREPENDING SOFT PROMPT                                  │
    │                                                                                 │
    │   Position:  [ 0   1   2  ...  19 | 20  21  22  ...  147 ]                      │
    │              [─────────────────── | ─────────────────────]                      │
    │              [  SOFT PROMPT       |  REAL TOKEN EMBEDDINGS ]                    │
    │              [ p₁  p₂  p₃ ... p₂₀| e₁  e₂  e₃ ...  e₁₂₈ ]                       │
    │              [                    |                         ]                   │
    │              [  trainable         |  frozen (from emb table)]                   │
    │              [  ℝ^1024 each       |  ℝ^1024 each            ]                   │
    │              [  not real words    |  real words              ]                  │
    │                                                                                 │
    │   Attention can flow FREELY between all 148 positions.                          │
    │   Soft prompt tokens CAN attend to real tokens and vice versa.                  │
    │                                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘

**STEP 4: PASS THROUGH THE FROZEN TRANSFORMER**

    The model processes input_embeds normally — no architecture changes:

    Layer 0:
        Q = input_embeds @ W_Q    [4 × 148 × 1024]  (all from frozen W_Q)
        K = input_embeds @ W_K    [4 × 148 × 1024]
        V = input_embeds @ W_V    [4 × 148 × 1024]
        A = softmax(QK^T / √d_k)  [4 × 148 × 148]   ← full attention over all 148 tokens
        h = AV                     [4 × 148 × 1024]
        ... + FFN, layer norm, residual

    The soft token positions (0-19) participate FULLY in attention — they can:
        - Gather information from real tokens (attend to positions 20-147)
        - Broadcast information to real tokens (real tokens attend to positions 0-19)
        - Interact with each other (attend within positions 0-19)

    After all 24 transformer layers:
    output ∈ ℝ^{4 × 148 × 1024}

    For classification, we use the output at position 20 (first REAL token = [CLS] equiv.)
    or pool over the real token positions. The soft prompt positions are discarded from output.

**STEP 5: COMPUTE LOSS AND BACKPROPAGATE**

    logits = output[:, 20:, :] @ W_lm_head   → classification logits
    loss = cross_entropy(logits, labels)

    loss.backward()   →   gradient flows back through:
        - The frozen transformer layers (gradients computed but NOT stored in params)
        - input_embeds (gradient computed here)

    At input_embeds, the gradient is split:
        - For positions 20-147 (token embeddings): gradient flows to embedding table,
          but embedding table is frozen → gradient discarded (requires_grad=False)
        - For positions 0-19 (soft prompt): gradient flows to P → P is updated!

    Update:
        P ← P - lr × ∇_P(loss)

    That's the ENTIRE training step. Only P changes. Nothing else.

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                              SOFT PROMPT INITIALIZATION — THREE STRATEGIES
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Why Initialization Matters

Soft prompt parameters P ∈ ℝ^{k × d} start from some initial values and are
updated by gradient descent. Unlike LoRA (where B=0 ensures the adapter starts as
an identity at t=0), there's no such constraint for soft prompts. The initialization
point strongly affects:
    - How quickly training converges
    - Whether training is stable
    - Final quality ceiling

The Lester et al. paper compared three initialization strategies:

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   INITIALIZATION STRATEGY COMPARISON                                                 │
    │                                                                                      │
    │   Strategy 1: RANDOM SAMPLING                                                        │
    │   ─────────────────────────────                                                      │
    │   pᵢ ~ Uniform(-0.5, 0.5)   or   pᵢ ~ N(0, 0.02)                                     │
    │                                                                                      │
    │   Pros: Simple, no domain knowledge needed                                           │
    │   Cons: Starts far from any meaningful region of embedding space.                    │
    │         Slower convergence. Often lowest final quality.                              │
    │                                                                                      │
    │   When to use: Very large models (≥11B) where any init works fine.                   │
    │                Baseline for ablation studies.                                        │
    │                                                                                      │
    │   ─────────────────────────────────────────────────────────────────────────────      │
    │                                                                                      │
    │   Strategy 2: VOCABULARY INITIALIZATION (Most commonly used)                         │
    │   ─────────────────────────────────────────────────────────────────────────────      │
    │   Sample k tokens from the vocabulary, use their embeddings as initial pᵢ.           │
    │   The k tokens are often sampled from the 5000 most frequent tokens.                 │
    │                                                                                      │
    │   pᵢ = E[random_token_id]   where E is the frozen embedding table                    │
    │                                                                                      │
    │   Pros: Starts in a region of ℝ^d where the model "understands" inputs.              │
    │         Converges faster. Better quality at small model scales.                      │
    │   Cons: The selected tokens are random — no semantic connection to the task.         │
    │                                                                                      │
    │   Why it works: Pre-trained embedding space is highly structured. Starting           │
    │   near an existing embedding gives gradient descent a better landscape to            │
    │   navigate than an arbitrary point in high-dimensional space.                        │
    │                                                                                      │
    │   ─────────────────────────────────────────────────────────────────────────────      │
    │                                                                                      │
    │   Strategy 3: CLASS LABEL INITIALIZATION (Best for classification tasks)             │
    │   ─────────────────────────────────────────────────────────────────────────────      │
    │   Use the embeddings of words that describe the task's output classes.               │
    │                                                                                      │
    │   Example for sentiment classification (k=4):                                        │
    │       p₁ = E["positive"]                                                             │
    │       p₂ = E["negative"]                                                             │
    │       p₃ = E["sentiment"]                                                            │
    │       p₄ = E["classify"]                                                             │
    │                                                                                      │
    │   Pros: Starts with semantically meaningful signal. Best quality on smaller          │
    │         models. Fastest convergence. Strong inductive bias toward the task.          │
    │   Cons: Requires knowing the class labels upfront. Only applies to classification.   │
    │         Extra tokens (beyond # of classes) still need vocab initialization.          │
    │                                                                                      │
    │   ─────────────────────────────────────────────────────────────────────────────      │
    │                                                                                      │
    │   QUALITY COMPARISON (SuperGLUE benchmark, T5-Large):                                │
    │                                                                                      │
    │   Initialization    SuperGLUE score    Relative to model tuning                      │
    │   ──────────────    ───────────────    ───────────────────────                       │
    │   Random             71.4               77.2% of model tuning                        │
    │   Vocabulary         74.6               80.7% of model tuning                        │
    │   Class labels       75.1 *             81.2% of model tuning                        │
    │   Fine-tuning        92.5              100% (upper bound)                            │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

**Key insight from the paper:**
At scales below 1B parameters, initialization strategy has a significant effect (~4 points
on SuperGLUE). At 11B+ parameters, all three strategies converge to similar quality —
the model is powerful enough to overcome a poor initialization.


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                       PROMPT LENGTH — THE MOST IMPORTANT HYPERPARAMETER
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### How Many Soft Tokens Do You Need?

The prompt length k is the single most important hyperparameter in Prompt Tuning.
More tokens → more expressive capacity → better potential quality, but more parameters
and a longer effective sequence (which affects memory and computation).

From the Lester et al. ablation study (T5 models, SuperGLUE):

    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                    │
    │   PROMPT LENGTH ABLATION (SuperGLUE score, T5-Large 770M)                          │
    │                                                                                    │
    │   k = 1          68.1   ████████████████████████████░░░░░░░░░░░░                   │
    │   k = 5          70.2   █████████████████████████████████░░░░░░                    │
    │   k = 10         73.0   ███████████████████████████████████░░░░                    │
    │   k = 20         74.9   ████████████████████████████████████░░░                    │
    │   k = 50         75.3 * █████████████████████████████████████░░                    │
    │   k = 100        75.1   ████████████████████████████████████░░░  (diminishing)     │
    │   k = 150        74.9   ████████████████████████████████████░░░  (no gain)         │
    │                                                                                    │
    │   Full fine-tuning: 92.5  (theoretical ceiling)                                    │
    │                                                                                    │
    │   FINDING: Sweet spot is k=20–100. Beyond k=100, quality plateaus or slightly      │
    │   drops (more tokens = harder optimization landscape). Default recommendation: k=20│
    │                                                                                    │
    │   MEMORY COST of longer prompts:                                                   │
    │   k=20 adds 20 tokens to every sequence. For seq_len=512, batch_size=8, d=1024:    │
    │   Extra memory = 20 × 8 × 1024 × 2 bytes = 327 KB (negligible)                     │
    │   Attention cost scales as O((n+k)²) — for small k, impact is minimal              │
    │                                                                                    │
    └────────────────────────────────────────────────────────────────────────────────────┘

**Rule of thumb:** Start with k=20. If quality is insufficient:
    - Increase to k=50 or k=100 before trying other changes.
    - Beyond k=100, try Prefix Tuning instead (more powerful method).

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                           THE SCALE EFFECT — WHY BIGGER MODELS CHANGE EVERYTHING
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### The Most Important Finding in the Lester et al. Paper

The headline result of the Prompt Tuning paper is not about prompt initialization,
length, or architecture — it's about MODEL SCALE.

At small scales, Prompt Tuning underperforms fine-tuning significantly.
At large scales (≥11B), Prompt Tuning MATCHES fine-tuning with 0.001% of the parameters.

    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                    │
    │   QUALITY vs. MODEL SCALE (SuperGLUE, T5 family)                                   │
    │                                                                                    │
    │   Model Size    Prompt Tuning    Model Tuning    Gap (ΔPT)                         │
    │   ──────────    ─────────────    ────────────    ─────────                         │
    │   T5-Small (60M)     56.3             74.0         -17.7  ████████████████░░       │
    │   T5-Base  (250M)    66.4             80.4         -14.0  ████████████░░░░░        │
    │   T5-Large (770M)    75.1             84.2          -9.1  ███████░░░░░░░░░░        │
    │   T5-XL    (3B)      82.3             87.7          -5.4  ████░░░░░░░░░░░░░        │
    │   T5-XXL   (11B)     91.9 *           92.5          -0.6  ░░░░░░░░░░░░░░░░░        │
    │                                                                                    │
    │   Fine-tuning baseline (T5-XXL):  92.5                                             │
    │   Prompt Tuning (T5-XXL):         91.9   ← WITHIN 0.6 POINTS OF FULL FINE-TUNING   │
    │   Parameters trained:             409,600 out of 11,000,000,000 = 0.0037%          │
    │                                                                                    │
    │   THE CROSSOVER POINT IS ~11B PARAMETERS.                                          │
    │   Below this, Prompt Tuning leaves significant quality on the table.               │
    │   Above this, it is competitive with full fine-tuning.                             │
    │                                                                                    │
    └────────────────────────────────────────────────────────────────────────────────────┘

**WHY does scale close the gap?**

At small scales, the model needs architectural freedom to adapt — the frozen
transformer layers lack the capacity to map the soft prompt signal into useful
task-specific representations across all layers. The soft tokens at layer 0 can
only steer the model through one degree of freedom (the embedding space).

At large scales, the model has:
    1. More heads and layers to "process" the soft prompt signal
    2. A richer embedding space where soft tokens can express more complex steering
    3. More inherent task knowledge that the soft prompt can "unlock" rather than create

Think of it as a key (soft prompt) and a lock (frozen model):
    - A small, simple lock has few tumblers — a crude key works but imprecisely
    - A large, complex lock has many tumblers — but a precisely shaped key opens it perfectly
    - Gradient descent finds the precise key shape for the lock at hand

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                          PROMPT TUNING vs. PREFIX TUNING — A DETAILED COMPARISON
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Where the Soft Tokens Live — The Critical Difference

This is the most important architectural distinction between the two methods.

**Prompt Tuning:**

    Layer 0 input:    [p₁ ... p_k | e₁ ... e_n]  ← soft tokens here ONLY
                       ↑
                    ONLY PLACE where soft tokens appear

    The soft tokens are processed by layer 0, and their representations evolve
    through the transformer layers like any other token. But after layer 0, the
    model has already "mixed" the soft tokens with real tokens. By layer 5 or 10,
    the positions that started as soft tokens may have representations dominated
    by attended information from real tokens.

    The soft prompt influences the model INDIRECTLY — by contributing to the
    initial representations that flow through all layers.


**Prefix Tuning:**

    Layer 0 K, V:     [pk₁ ... pk_k | K(e₁) ... K(e_n)]   ← prefix appended to K
                       [pv₁ ... pv_k | V(e₁) ... V(e_n)]   ← prefix appended to V

    Layer 1 K, V:     [pk₁' ... pk_k' | K(h₁) ... K(h_n)]  ← DIFFERENT prefix at each layer
                       [pv₁' ... pv_k' | V(h₁) ... V(h_n)]

    Layer L K, V:     [pk₁ᴸ ... pk_kᴸ | K(hᴸ₋₁) ...  ]    ← layer-specific prefix
                       [pv₁ᴸ ... pv_kᴸ | V(hᴸ₋₁) ...  ]

    Total parameters: 2 × L × k × d   (2 for K and V, L layers, k prefix tokens, d dim)
    For T5-Large: 2 × 24 × 100 × 1024 = 4,915,200 parameters (vs. 102,400 for Prompt Tuning)

    Prefix Tuning provides DIRECT steering at every attention layer.
    The prefix can inject task-relevant information at each level of abstraction.

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   PROMPT TUNING vs. PREFIX TUNING — ARCHITECTURAL DIAGRAM                            │
    │                                                                                      │
    │                                                                                      │
    │   PROMPT TUNING                              PREFIX TUNING                           │
    │   ─────────────                              ─────────────                           │
    │                                                                                      │
    │   Input:                                     Input:                                  │
    │   [p₁..p_k | e₁..e_n]                        [e₁..e_n]  (no change to input)         │
    │         │                                           │                                │
    │         ▼                                           ▼                                │
    │   ┌──────────────┐                          ┌──────────────┐                         │
    │   │   Layer 0    │                          │   Layer 0    │◄── [pk⁰,pv⁰] injected   │
    │   │   (standard) │                          │   (standard) │    into K and V         │
    │   └──────┬───────┘                          └──────┬───────┘                         │
    │          │                                         │                                 │
    │          ▼                                         ▼                                 │
    │   ┌──────────────┐                          ┌──────────────┐                         │
    │   │   Layer 1    │                          │   Layer 1    │◄── [pk¹,pv¹] injected   │
    │   │   (standard) │                          │   (standard) │    into K and V         │
    │   └──────┬───────┘                          └──────┬───────┘                         │
    │          │                                         │                                 │
    │         ...                                       ...                                │
    │          │                                         │                                 │
    │          ▼                                         ▼                                 │
    │   ┌──────────────┐                          ┌──────────────┐                         │
    │   │   Layer L    │                          │   Layer L    │◄── [pk^L,pv^L]          │
    │   │   (standard) │                          │   (standard) │                         │
    │   └──────────────┘                          └──────────────┘                         │ 
    │                                                                                      │
    │   Soft tokens: 1 layer only                  Soft tokens: EVERY layer                │
    │   Trainable params (T5-L, k=100): 102K        Trainable params (T5-L, k=100): 4.9M   │
    │   Quality (T5-Large SuperGLUE): 75.1          Quality (T5-Large SuperGLUE): ~79.8    │
    │   Quality (T5-XXL SuperGLUE):   91.9          Quality (T5-XXL SuperGLUE):   ~92.7    │
    │                                                                                      │
    │   WHEN TO USE EACH:                                                                  │
    │   Prompt Tuning: Large models (≥3B), extreme parameter budget constraints,           │
    │                  serving multiple tasks simultaneously from one base model           │
    │   Prefix Tuning: Smaller models (<3B), better quality needed, or NLG tasks           │
    │                  (Prefix Tuning was designed with summarization/dialogue in mind)    │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                      MULTI-TASK SERVING — THE KILLER USE CASE FOR PROMPT TUNING
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### One Frozen Model, Many Soft Prompts

Prompt Tuning's most compelling real-world application is serving many tasks
simultaneously from a single frozen model. This is impossible with LoRA
(which modifies model weights) but trivially easy with Prompt Tuning.

The deployment architecture looks like this:

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │   SINGLE BASE MODEL — MULTIPLE TASK PROMPTS                                         │
    │                                                                                     │
    │                    ┌─────────────────────────────────────────┐                      │
    │                    │      FROZEN BASE MODEL (11B params)     │                      │
    │                    │      Loaded once, never changes         │                      │
    │                    │      Resides in GPU VRAM (shared)       │                      │
    │                    └─────────────────────────────────────────┘                      │
    │                              ▲           ▲           ▲                              │
    │              ┌───────────────┘           │           └──────────────┐               │
    │              │                           │                          │               │
    │   ┌────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐           │
    │   │ P_sentiment        │  │ P_summarization     │  │ P_translation_de   │           │
    │   │ (409,600 params)   │  │ (409,600 params)    │  │ (409,600 params)   │           │
    │   │ 1.6 MB             │  │ 1.6 MB              │  │ 1.6 MB             │           │
    │   └────────────────────┘  └─────────────────────┘  └────────────────────┘           │
    │                                                                                     │
    │   ← These are just small files on disk, swapped in per request!                     │
    │                                                                                     │
    │   TOTAL MEMORY:                                                                     │
    │   Traditional (3 separate fine-tuned models): 3 × 22 GB = 66 GB                     │
    │   Prompt Tuning (1 base + 3 prompts):         22 GB + (3 × 1.6 MB) ≈ 22 GB          │
    │                                                                                     │
    │   This is why Google uses Prompt Tuning at scale: one T5-XXL instance               │
    │   can serve thousands of different tasks simultaneously.                            │
    │                                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘

**The Business Case:**
    - Save 95-99% GPU memory vs. fine-tuning separate models per task
    - Instant task switching at inference time (swap the soft prompt matrix, not the model)
    - New task training takes minutes, not hours (only 409K parameters to optimize)
    - The frozen model can be updated once and all task prompts benefit automatically


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                               THE REPARAMETERIZATION TRICK — PROMPT TUNING TRAINING STABILITY
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Why Training Soft Prompts Can Be Unstable

Naively, the soft prompt is just a weight matrix. Why would training it be unstable?

The problem is the scale mismatch between the soft prompt and the frozen model.

When we update P via gradient descent, we're modifying embeddings that must
remain compatible with the frozen transformer's normalization expectations.
Transformer layers (especially Layer Norm) operate on inputs with specific
statistical properties. If the soft prompt vectors deviate too far from the
distribution of normal token embeddings, the frozen transformer processes
them poorly.

    Normal token embeddings (T5-Large):
        Mean:  0.003 (near zero)
        Std:   0.027
        Range: roughly [-0.15, 0.15]

    After 1000 gradient steps without careful LR, soft prompt vectors can drift to:
        Mean:  0.48 (far from normal input distribution)
        Std:   1.2
        Range: [-6.3, 7.1]  ← catastrophically out of distribution for layer norm

    When the transformer's first layer norm sees input vectors of magnitude ±6,
    all its learned normalization assumptions break down.

### The Fix: Learning Rate and Initialization

Prompt Tuning avoids this by using:

    1. Small initialization:      Use vocabulary embeddings (already in the expected range)
    2. Small learning rate:       lr = 0.3 (much higher than typical, but with AdaFactor/Adam
                                  this maps to effective updates that are small in practice)
    3. Separate prompt LR:        In HuggingFace PEFT, you can set a separate, smaller LR
                                  for the prompt parameters vs. any trainable head parameters

    In practice, Lester et al. use:
        Optimizer: Adafactor (adaptive — naturally controls update magnitude)
        Learning rate: 0.3 (with Adafactor's normalization, this is safe)
        Warm-up: 0 (Adafactor doesn't need warm-up)

### The Prefix Tuning Reparameterization (for reference)

Prefix Tuning introduced a separate reparameterization to handle this problem:
instead of learning P directly, they learn a smaller matrix P' and a MLP:

    P = MLP_θ(P')

During training, only P' and θ are updated. After training, the MLP is
discarded and P is computed once and frozen. This prevents the prefix vectors
from drifting out of distribution because the MLP acts as a regularizer.

Prompt Tuning does NOT use this trick — it trains P directly. This works
at large scales where the soft prompt has enough signal, but is one reason
Prefix Tuning is more stable at small scales.


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                     ATTENTION MASK — THE HIDDEN DETAIL
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Attention Masks Need Special Handling

When you prepend soft tokens to the input, you must also update the attention mask.
The attention mask tells the model which positions to attend to. Padding tokens get
mask value 0 (do not attend). Real tokens get mask value 1.

Without adjustment:
    input_ids attention_mask:    [1 1 1 1 1 1 1 1 0 0 0 ...]    (1=real, 0=padding)

After prepending k=20 soft tokens:
    new_attention_mask MUST be:  [1 1 1 1 1 ... 1 | 1 1 1 1 1 1 1 1 0 0 0 ...]
                                  ← 20 ones →   |  ← original mask →

If you forget to prepend 1s for the soft tokens, the transformer will mask out
the soft tokens — treating them as padding! This is a very common implementation bug.

    CORRECT:
        prefix_mask = torch.ones(batch_size, k, dtype=torch.long)
        full_mask = torch.cat([prefix_mask, input_attention_mask], dim=1)

    WRONG (masks out soft tokens):
        # Just using the original attention mask — DO NOT DO THIS
        outputs = model(inputs_embeds=input_embeds, attention_mask=original_mask)

The HuggingFace PEFT library handles this automatically. If you implement from
scratch, this is the most common place to introduce a silent bug.


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                          TRAINING DETAILS — WHAT ACTUALLY WORKS IN PRACTICE
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Optimizer Choice

**Adafactor (strongly recommended for Prompt Tuning):**

    Adafactor is a memory-efficient adaptive optimizer. It stores only a factored
    approximation of the second moment, saving memory compared to Adam.

    For Prompt Tuning:
        optimizer = Adafactor(
            [P],
            lr=0.3,
            relative_step=False,   # use explicit LR, not adaptive schedule
            warmup_init=False,
        )

    Why Adafactor works well for Prompt Tuning:
        - Adaptive step sizes per parameter prevent large updates to any single dimension
        - Much lower memory: O(n+m) vs Adam's O(n×m) for n×m matrices
        - Less sensitive to LR choice (adaptive step normalizes magnitudes)

**AdamW (also works, common in HuggingFace pipelines):**

    optimizer = AdamW([P], lr=1e-3, weight_decay=0.01)

    With AdamW, use a smaller LR than with Adafactor.
    Typical range: 1e-3 to 5e-3 for AdamW on soft prompts.
    Use separate param group with higher LR for the soft prompt vs. any classification head.


### Training Duration and Convergence

Soft prompt training converges faster than full fine-tuning in terms of wall-clock time
(fewer parameters to update) but may require more steps in terms of gradient updates:

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   CONVERGENCE COMPARISON (T5-Large, SST-2 sentiment classification)                  │
    │                                                                                      │
    │   Method              Steps to convergence    Wall-clock time    Final Accuracy      │
    │   ──────              ────────────────────    ───────────────    ──────────────      │
    │   Full fine-tuning    ~1,000 steps             ~8 hours          95.2%               │
    │   Prompt Tuning       ~10,000-30,000 steps     ~40 min           93.8%               │
    │   (k=100)                                                                            │
    │                                                                                      │
    │   Why more gradient steps?                                                           │
    │   - Each gradient update changes only 102,400 params vs 770,000,000                  │
    │   - The optimization landscape for soft prompts is harder (non-convex in d=1024)     │
    │   - The frozen model provides less feedback per step than a fully tunable model      │
    │                                                                                      │
    │   Why less wall-clock time despite more steps?                                       │
    │   - Each step is much faster (tiny gradient computation for 102K params vs 770M)     │
    │   - Smaller model during forward pass (no gradient through backbone)                 │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘


### Prompt Token Length and Sequence Budget

Every soft token you prepend increases the effective sequence length.
This has real costs for attention (quadratic in sequence length):

    Attention computation:  O((n + k)²)  per layer

    For n=512 (real tokens), k=100 (soft tokens):
        Attention over: 612 positions
        Cost increase: (612/512)² ≈ 1.43×  → 43% more attention compute

    For n=512, k=20 (recommended default):
        Cost increase: (532/512)² ≈ 1.08×  → only 8% more compute

    This is still much cheaper than Prefix Tuning (which adds to every layer's K and V)
    or than increasing batch size.

The practical memory cost of the trainable parameters themselves is negligible:
    k=100, d=4096 (LLaMA-2-7B): 409,600 × 2 bytes = 819 KB = 0.8 MB


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                         TASK SUITABILITY — WHERE PROMPT TUNING SHINES AND WHERE IT FAILS
    ═══════════════════════════════════════════════════════════════════════════════════════════════


### Tasks Where Prompt Tuning Works Well

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                      │
    │   TASK SUITABILITY MATRIX                                                            │
    │                                                                                      │
    │   Task Type                   Prompt Tuning Suitability     Notes                    │
    │   ─────────────────────────   ─────────────────────────     ─────────────────────    │
    │   Classification              ★★★★★  Excellent              Original use case       │
    │   Named Entity Recognition    ★★★★☆  Very Good              Good on large models    │
    │   Natural Language Inference  ★★★★★  Excellent              Benchmark-tested        │
    │   Question Answering          ★★★☆☆  Good (large models)    Needs ≥3B params        │
    │   Summarization               ★★★☆☆  Moderate               Prefix Tuning better    │
    │   Dialogue / Chatbot          ★★☆☆☆  Weak                   LoRA or Full FT better  │
    │   Code Generation             ★★☆☆☆  Weak                   Needs deep steering     │
    │   Domain Adaptation           ★★★☆☆  Good (large models)    Scale-dependent         │
    │   Multi-Task Serving          ★★★★★  Excellent              The killer use case     │
    │   Instruction Following       ★★★☆☆  Moderate               LoRA preferred          │
    │                                                                                      │
    │   VERDICT: Prompt Tuning excels at classification and NLU tasks on large models.     │
    │   For NLG (generation-heavy) tasks or smaller models, Prefix Tuning or LoRA are      │
    │   more appropriate.                                                                  │
    │                                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────┘

### The Frozen Model Constraint

The hardest limitation of Prompt Tuning is that the base model is completely frozen.
This means:

    1. Knowledge cannot be updated: if the base model has outdated or incorrect
       domain knowledge, the soft prompt cannot fix this — it can only steer
       what the model already knows.

    2. Vocabulary is fixed: the model's tokenizer and embedding table don't change.
       For highly specialized domains with novel terminology, the model may not have
       the right token representations at all.

    3. Structural task changes are hard: if the task requires a fundamentally different
       output format (e.g., generating structured JSON with strict schema), the frozen
       decoder may resist this more than a fine-tuned one.

    These are the cases where LoRA or full fine-tuning should be preferred.


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                          COMPLETE VISUAL SUMMARY — PROMPT TUNING FROM INPUT TO OUTPUT
    ═══════════════════════════════════════════════════════════════════════════════════════════════


    ┌───────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                           │
    │                         PROMPT TUNING — COMPLETE DATA FLOW                                │
    │                                                                                           │
    │                                                                                           │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐    │
    │  │  TRAINING TIME                                                                    │    │
    │  │                                                                                   │    │
    │  │  Input text: "Classify: Excellent product!"                                       │    │
    │  │      │                                                                            │    │
    │  │      │ tokenize + embed (frozen E)                                                │    │
    │  │      ▼                                                                            │    │
    │  │  Token embeddings [e₁, e₂, e₃, e₄]  ∈ ℝ^{4 × d}                                   │    │
    │  │      │                                                                            │    │
    │  │      │ prepend soft prompt P ∈ ℝ^{k × d}   ← TRAINABLE                            │    │
    │  │      ▼                                                                            │    │
    │  │  Combined input [p₁..p_k, e₁..e₄]  ∈ ℝ^{(k+4) × d}                                │    │
    │  │      │                                                                            │    │
    │  │      │  pass through FROZEN transformer (32 layers, no weight updates)            │    │
    │  │      ▼                                                                            │    │
    │  │  Output logits                                                                    │    │
    │  │      │                                                                            │    │
    │  │      │  loss.backward()                                                           │    │
    │  │      ▼                                                                            │    │
    │  │  ∇_P(loss) ← gradient flows back to P ONLY                                        │    │
    │  │      │                                                                            │    │
    │  │      │  optimizer.step()                                                          │    │
    │  │      ▼                                                                            │    │
    │  │  P ← P - lr × ∇_P                                                                 │    │
    │  │                                                                                   │    │
    │  └───────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                           │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐    │
    │  │  INFERENCE TIME                                                                   │    │
    │  │                                                                                   │    │
    │  │  Input text: "Classify: Terrible, broke on day 1."                                │    │
    │  │      │ tokenize + embed (frozen)                                                  │    │
    │  │      │ prepend TRAINED P (frozen after training)                                  │    │
    │  │      │ pass through frozen transformer                                            │    │
    │  │      ▼                                                                            │    │
    │  │  Output: "negative"   ← steered by the learned soft prompt                        │    │
    │  │                                                                                   │    │
    │  │  Same frozen base model, same GPU, just a different P matrix for each task!       │    │
    │  │                                                                                   │    │
    │  └───────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                           │
    └───────────────────────────────────────────────────────────────────────────────────────────┘


---


    ═══════════════════════════════════════════════════════════════════════════════════════════════
                                          SUMMARY MENTAL MODEL
    ═══════════════════════════════════════════════════════════════════════════════════════════════


    FULL FINE-TUNING          PREFIX TUNING                PROMPT TUNING

    ┌──────────────────┐      ┌───────────────────────┐     ┌───────────────────────┐
    │ ██ ALL weights   │      │ ░░ FROZEN model       │     │ ░░ FROZEN model       │
    │ ██ updated       │      │ ░░                    │     │ ░░                    │
    │ ██ (BF16)        │      │ ██ + prefix at every  │     │ ██ + prefix at        │
    │ ██               │      │ ██   K & V            │     │ ██   INPUT ONLY       │
    └──────────────────┘      └───────────────────────┘     └───────────────────────┘

    VRAM:     89-109 GB (7B)  VRAM:  14 GB (7B)            VRAM:  14 GB (7B)
    Params:   100%            Params: ~0.1-0.5%             Params: ~0.006%
    Quality:  100%            Quality: ~90-98%              Quality: ~75-95% (scale-dep)
    Hardware: 8×A100          Hardware: 1×A100 40G          Hardware: 1×RTX 3090


The Single Sentence Summary:

    Prompt Tuning learns a small set of free-floating embedding vectors that are
    prepended to every input, steering the completely frozen model toward a target
    task — changing nothing in the model itself, only what the model sees first.

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect                          | Full Fine-Tuning          | Prefix Tuning             | Prompt Tuning              | P-Tuning v2               |
|---------------------------------|---------------------------|---------------------------|----------------------------|---------------------------|
| Trainable Params                | 100% (~7B for LLaMA-7B)   | ~0.1-0.5% (all layers)    | ~0.001-0.006%              | ~0.1% (all layer prefix)  |
| Where Soft Tokens Live          | N/A                       | Every K & V in every layer| Input embedding only       | K & V in every layer      |
| Model Weights Changed?          | Yes (all)                 | No (frozen)               | No (frozen)                | No (frozen)               |
| Architecture Changes?           | None                      | Adds prefix to each layer | Adds prefix to embedding   | Adds prefix to each layer |
| Trainable Params (T5-Large)     | 770M                      | 4.9M (k=100)              | 102K (k=100)               | ~3.6M                     |
| Trainable Params (LLaMA-2-7B)   | 7B                        | ~25M (k=100)              | 410K (k=100)               | ~23M                      |
| Trainable Param % (LLaMA-7B)    | 100%                      | 0.36%                     | 0.006%                     | 0.33%                     |
| Checkpoint Size (LLaMA-7B, k=100)| 14 GB                    | ~50 MB                    | ~0.8 MB                    | ~46 MB                    |
| GPU Memory Required (7B)        | 89-109 GB                 | ~14-18 GB                 | ~14-16 GB                  | ~14-18 GB                 |
| Quality at Small Scale (<1B)    | Baseline 100%             | 80-90%                    | 55-75%                     | 78-88%                    |
| Quality at Large Scale (≥11B)   | Baseline 100%             | 97-99%                    | 99-100% ★                  | 99-100%                   |
| Best Task Type                  | All                       | NLG, summarization        | Classification, NLU        | NLU, NER, QA              |
| Multi-Task Serving              | Poor (one model/task)     | Good                      | Excellent ★ (tiny files)   | Good                      |
| Requires Large Model (≥3B)?     | No                        | No                        | Yes (for best quality)     | No                        |
| Reparameterization Used?        | N/A                       | MLP (during training only)| No (direct)                | No (direct)               |
| Compatible with Quantized Base? | N/A                       | Yes                       | Yes                        | Yes                       |
| Initialization Strategy         | N/A                       | Vocab embeddings          | Vocab or class labels      | Vocab embeddings          |
| Recommended Optimizer           | AdamW                     | AdamW or Adafactor        | Adafactor (lr=0.3)         | AdamW or Adafactor        |
| Typical LR                      | 1e-6 to 5e-5              | 1e-4 to 5e-4              | 0.3 (Adafactor), 1e-3 (Adam)| 1e-4 to 5e-4             |
| HuggingFace PEFT Support        | N/A                       | PromptEncoderConfig       | PromptTuningConfig ★       | PrefixTuningConfig        |
| PEFT Config Class               | N/A                       | PromptEncoderConfig       | PromptTuningConfig         | PrefixTuningConfig        |
| Key Hyperparameters             | lr, epochs, batch         | k, LR, init_text          | k, LR, init_text, init_method| k, LR, init_text         |
| Key Limitation                  | Memory, compute           | More params than PT       | Scale-dependent quality    | More params than PT       |
| Paper                           | —                         | Li & Lam 2021             | Lester et al. 2021         | Liu et al. 2022           |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Prompt Tuning — Minimal Setup with HuggingFace PEFT": {
        "description": "The simplest way to set up Prompt Tuning using the PEFT library",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

# ─── Step 1: Load the frozen base model ───────────────────────────────────────
# Prompt Tuning works best with encoder-decoder (T5) or decoder-only (GPT/LLaMA)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# ─── Step 2: Configure Prompt Tuning ──────────────────────────────────────────
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,          # T5 encoder-decoder

    # Prompt length: most important hyperparameter
    # Rule of thumb: start with 20, increase to 100 if quality insufficient
    num_virtual_tokens=20,

    # Initialization strategy: VOCAB (random vocab embeddings, good default)
    # Alternative: TEXT (use a custom init string — best if you have a hard prompt)
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the sentiment of the following text:",

    # Tokenizer needed for TEXT initialization (to convert init string to embeddings)
    tokenizer_name_or_path="google/flan-t5-large",
)

# ─── Step 3: Apply config — only adds a soft prompt matrix ────────────────────
model = get_peft_model(model, peft_config)

# Verify: only the soft prompt parameters are trainable
model.print_trainable_parameters()
# Output: trainable params: 20,480 || all params: 783,175,680 || trainable%: 0.0026%
# (20 tokens × 1024 d_model = 20,480 parameters — that's it!)

# ─── Check parameter structure ────────────────────────────────────────────────
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"TRAINABLE: {name}  shape={list(param.shape)}")
# Should print ONLY:
# TRAINABLE: prompt_encoder.default.embedding.weight  shape=[20, 1024]
''',
    },

    "Prompt Tuning — Full Training Loop (T5, Classification)": {
        "description": "Complete training pipeline for text classification using Prompt Tuning on T5",
        "runnable": False,
        "code": '''from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from datasets import load_dataset
import torch

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME   = "google/flan-t5-large"     # 770M params: good scale for prompt tuning
NUM_TOKENS   = 20                          # Soft prompt length (k)
INIT_TEXT    = "Classify the sentiment of this review as positive or negative:"
MAX_LENGTH   = 256
BATCH_SIZE   = 16
EPOCHS       = 30                          # More epochs needed than full fine-tuning
LR           = 3e-1                        # High LR — Adafactor normalizes this down

# ─── Load model and tokenizer ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

# ─── Prompt Tuning config ─────────────────────────────────────────────────────
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=NUM_TOKENS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text=INIT_TEXT,          # Best init: use your hard prompt
    tokenizer_name_or_path=MODEL_NAME,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ─── Dataset ──────────────────────────────────────────────────────────────────
dataset = load_dataset("sst2")

def preprocess(examples):
    # T5 is seq2seq: input → output as text
    inputs  = [f"sentence: {s}" for s in examples["sentence"]]
    targets = ["positive" if l == 1 else "negative" for l in examples["label"]]

    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=8, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    # Replace padding token id in labels with -100 (ignore in loss)
    model_inputs["labels"] = [
        [(-100 if t == tokenizer.pad_token_id else t) for t in label]
        for label in model_inputs["labels"]
    ]
    return model_inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# ─── Training args ────────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir="./prompt-tuning-output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    # CRITICAL: Use Adafactor for Prompt Tuning — prevents soft prompt drift
    optim="adafactor",
    lr_scheduler_type="constant",              # Adafactor doesn't need warm-up
    predict_with_generate=True,
    generation_max_length=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    logging_steps=50,
    # Gradient checkpointing: saves memory (activations NOT stored for frozen backbone)
    gradient_checkpointing=False,              # Set True for large models
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

trainer.train()
# Saves ONLY the soft prompt matrix (~40 KB) — not the full model
trainer.save_model("./prompt-tuning-output/best")
''',
    },

    "Prompt Tuning — Decoder-Only LLM (LLaMA / GPT-style)": {
        "description": "Apply Prompt Tuning to a causal (decoder-only) LLM like LLaMA-2",
        "runnable": False,
        "code": '''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
import torch

# ─── Load decoder-only model ──────────────────────────────────────────────────
# For causal LMs, Prompt Tuning is less common than LoRA but valid
# Best results at ≥7B; try 13B if 7B underperforms
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # IMPORTANT for decoder-only: pad on left

# ─── Prompt Tuning config for CAUSAL_LM ───────────────────────────────────────
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,            # <-- Changed from SEQ_2_SEQ_LM
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.RANDOM,  # or TEXT
    # For CAUSAL_LM, TEXT init doesn't work as cleanly (no encoder to anchor meaning)
    # Use RANDOM initialization and rely on scale (7B should be sufficient)
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 81,920 || all params: 6,738,476,032 || trainable%: 0.0012%
# (20 tokens × 4,096 d_model = 81,920 parameters)

# ─── Inspect what Prompt Tuning added ─────────────────────────────────────────
# The only change to the model: a soft prompt embedding table
print(model.prompt_encoder)
# PromptEmbedding(
#   (embedding): Embedding(20, 4096)   ← 20 soft tokens × 4096 dimensions
# )

# ─── Key difference for CAUSAL_LM: left-padding is essential ──────────────────
# For decoder-only models, the soft prompt is prepended to the BEGINNING of input.
# Padding must be on the LEFT so soft tokens are always at position 0.
# Right-padding would push soft tokens away from the generation start position.

# ─── Verify the effective input shape during forward pass ─────────────────────
sample = tokenizer(["Hello world"], return_tensors="pt").to("cuda")
# Without soft prompt: input shape = [1, 2]  (batch, seq_len)
# With soft prompt:    effective input = [1, 22]  (2 real + 20 soft)
# (The PEFT model handles this prepending automatically in its forward method)
outputs = model(**sample)
print(f"Output shape: {outputs.logits.shape}")
# [1, 22, vocab_size]  — notice 22 tokens (20 soft + 2 real)
''',
    },

    "Comparing Initialization Strategies": {
        "description": "Test all three initialization strategies and compare convergence speed",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
import torch

MODEL_NAME = "google/flan-t5-large"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── Strategy 1: Random Initialization ────────────────────────────────────────
config_random = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.RANDOM,
    # No init_text needed
)
model_random = get_peft_model(
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME), config_random
)
# Check initial prompt values — should be near uniform(-0.5, 0.5)
P_random = model_random.prompt_encoder.default.embedding.weight
print(f"Random init: mean={P_random.mean():.4f}, std={P_random.std():.4f}")
# Expected: mean≈0.0, std≈0.28

# ─── Strategy 2: Vocabulary Initialization ────────────────────────────────────
# PEFT library handles this internally when prompt_tuning_init=TEXT:
# It tokenizes the init_text and uses those token embeddings
config_vocab = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the following text:",  # Should be ≤ num_virtual_tokens words
    tokenizer_name_or_path=MODEL_NAME,
)
model_vocab = get_peft_model(
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME), config_vocab
)
P_vocab = model_vocab.prompt_encoder.default.embedding.weight
print(f"Vocab init:  mean={P_vocab.mean():.4f}, std={P_vocab.std():.4f}")
# Expected: mean≈0.003, std≈0.027 (same distribution as real token embeddings)

# ─── Difference in initial embedding space distribution ───────────────────────
# Normal token embeddings for this model live near:
sample_tok_ids = torch.randint(0, tokenizer.vocab_size, (20,))
sample_embs    = model_vocab.base_model.shared(sample_tok_ids)
print(f"Normal token emb: mean={sample_embs.mean():.4f}, std={sample_embs.std():.4f}")

# Vocab init (mean≈0.003, std≈0.027) matches normal tokens → smoother optimization landscape
# Random init (mean≈0.0, std≈0.28) is 10× wider → farther from model's operating range

# ─── Practical recommendation ─────────────────────────────────────────────────
# If you have a task description: use TEXT init with your hard prompt text
# If you have class labels:       use TEXT init with class label words  
# If you have neither:            use RANDOM — at large model scales it converges
''',
    },

    "Multi-Task Serving — One Model, Multiple Prompts": {
        "description": "Deploy one frozen base model and swap soft prompts for different tasks",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
import torch

MODEL_NAME = "google/flan-t5-xxl"  # 11B — where prompt tuning matches fine-tuning quality

# ─── Load the base model ONCE ─────────────────────────────────────────────────
# This model is frozen and shared across all tasks
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── Train separate soft prompts for each task ────────────────────────────────
# (In production, these would be pre-trained and saved to disk)
# Here we show how to load and use them

def load_task_prompt(task_prompt_path: str, base_model):
    """Load a saved PEFT soft prompt adapter on top of the frozen base."""
    return PeftModel.from_pretrained(base_model, task_prompt_path)

# ─── Inference: swap prompts without reloading the model ──────────────────────
# Each task prompt is just a tiny file (< 2 MB for T5-XXL with k=100)

# Task 1: Sentiment classification
model_sentiment = load_task_prompt("./prompts/sentiment", base_model)
model_sentiment.eval()

# Task 2: News summarization
model_summarize = load_task_prompt("./prompts/summarization", base_model)
model_summarize.eval()

# Task 3: German translation
model_translate = load_task_prompt("./prompts/translation_de", base_model)
model_translate.eval()

# ─── Example inference ────────────────────────────────────────────────────────
def run_inference(model, text, max_new_tokens=64):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(run_inference(model_sentiment, "This product is absolutely amazing!"))
# "positive"

print(run_inference(model_summarize, "Article: Scientists have discovered..."))
# "Summary: ..."

print(run_inference(model_translate, "Hello, how are you today?"))
# "Hallo, wie geht es Ihnen heute?"

# ─── Memory comparison ────────────────────────────────────────────────────────
# Base model (T5-XXL, BF16):              ~22 GB VRAM (loaded once)
# Each soft prompt (k=100, d=4096):       ~0.8 MB on disk, ~0.8 MB extra VRAM
# 1000 tasks × 0.8 MB = 800 MB of prompts for 1000 different tasks
# vs. 1000 × 22 GB = 22 TB if we fine-tuned separate models per task
''',
    },

    "Prefix Tuning — Setup and Comparison": {
        "description": "Configure Prefix Tuning via PEFT for comparison with Prompt Tuning",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM
from peft import PrefixTuningConfig, get_peft_model, TaskType

model_name = "google/flan-t5-large"
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ─── Prefix Tuning config ─────────────────────────────────────────────────────
prefix_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,

    # Number of virtual tokens per layer (same k concept as Prompt Tuning)
    # But unlike Prompt Tuning, these exist at EVERY transformer layer's K and V
    num_virtual_tokens=100,

    # Encoder-specific prefix length (T5 has encoder+decoder)
    # num_virtual_tokens applies to both by default
    encoder_hidden_size=None,           # Use model's d_model automatically

    # Prefix Tuning uses a reparameterization MLP during training:
    # P = MLP(P')   where P' is the real learned parameter
    # This prevents prefix vectors from drifting out of distribution
    # The MLP is discarded after training (prefix_projection=True during training)
    prefix_projection=True,
    inference_mode=False,               # True at inference (MLP removed)
)

model = get_peft_model(model, prefix_config)
model.print_trainable_parameters()
# trainable params: 4,915,200 || all params: 788,090,880 || trainable%: 0.62%
# (100 tokens × 24 layers × 2 (K+V) × 1024 dim = 4,915,200)

# ─── COMPARISON: Prompt Tuning vs Prefix Tuning (same k=100, T5-Large) ────────
# Prompt Tuning:  102,400 params   (0.013%)
# Prefix Tuning:  4,915,200 params (0.62%)
# → Prefix Tuning has 48× MORE trainable parameters per soft token
# → But it steers the model at every layer vs. just the input
# → Quality gap: SuperGLUE ~75.1 (PT) vs ~79.8 (PrefixT) at T5-Large scale

# ─── When to choose Prefix Tuning over Prompt Tuning ─────────────────────────
# Use Prefix Tuning when:
#   - Model is <3B parameters (Prompt Tuning underperforms at small scale)
#   - Task is NLG (generation, summarization, dialogue — Prefix Tuning excels)
#   - You need higher quality and can afford 48× more params
#
# Use Prompt Tuning when:
#   - Model is ≥3B (ideally ≥11B for full quality match)
#   - Task is classification or NLU
#   - Multi-task serving is required (tinier prompt files)
#   - Extreme parameter budget constraints
''',
    },

    "Saving and Loading Soft Prompts": {
        "description": "Save only the soft prompt (not the full model), then reload for inference",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
import os

MODEL_NAME = "google/flan-t5-large"

# ─── SAVING (after training) ──────────────────────────────────────────────────
# After training with PEFT, save only the soft prompt adapter
# model.save_pretrained() saves ONLY the trainable parameters (the soft prompt)
# NOT the full 770M parameter model — just the tiny soft prompt matrix

save_path = "./my-sentiment-prompt"
model.save_pretrained(save_path)

# What gets saved:
# ./my-sentiment-prompt/
#   adapter_config.json         ← PEFT config (PromptTuningConfig params)
#   adapter_model.safetensors   ← The soft prompt weights ONLY (< 1 MB!)

print("Files saved:")
for f in os.listdir(save_path):
    size_bytes = os.path.getsize(os.path.join(save_path, f))
    print(f"  {f}: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
# adapter_config.json:         351 bytes (0.3 KB)
# adapter_model.safetensors:   40,990 bytes (40.0 KB)  ← the entire prompt!

# ─── LOADING (at inference time) ──────────────────────────────────────────────
# Step 1: Load the frozen base model (shared across tasks)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

# Step 2: Load the soft prompt on top of the frozen base
model = PeftModel.from_pretrained(base_model, save_path)
model.eval()  # Important: puts the model in inference mode

# The base model weights are NOT stored in save_path.
# PeftModel combines: frozen base (loaded from hub) + soft prompt (from save_path)

# ─── Inference ────────────────────────────────────────────────────────────────
import torch
inputs = tokenizer(
    "sentence: The acting was brilliant and the plot was gripping.",
    return_tensors="pt"
)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# "positive"

# ─── Switching between tasks at inference time ────────────────────────────────
# base_model stays in memory — just load different prompts
model_neg = PeftModel.from_pretrained(base_model, "./my-negative-prompt")
model_sum = PeftModel.from_pretrained(base_model, "./my-summarize-prompt")
# Each swap: loads only ~40 KB from disk — essentially instant
''',
    },

    "Inspect Soft Prompt Embeddings — What Did Training Learn?": {
        "description": "Visualize and analyze what the soft prompt tokens learned",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
import torch.nn.functional as F

MODEL_NAME = "google/flan-t5-large"

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model      = PeftModel.from_pretrained(base_model, "./my-sentiment-prompt")

# ─── Extract the learned soft prompt matrix ───────────────────────────────────
P = model.prompt_encoder.default.embedding.weight  # [k × d_model]
print(f"Soft prompt shape: {P.shape}")             # e.g., [20, 1024]
print(f"Soft prompt stats: mean={P.mean():.4f}, std={P.std():.4f}")

# ─── Find the closest real vocabulary tokens to each soft prompt vector ────────
# This tells us what "semantic region" of embedding space each soft token landed in
# NOTE: soft prompt vectors are NOT real words — this is just an approximation

embedding_table = base_model.shared.weight  # [vocab_size × d_model]

for i, soft_token in enumerate(P):
    # Cosine similarity to all vocabulary embeddings
    soft_token_normalized = F.normalize(soft_token.unsqueeze(0), dim=-1)
    emb_normalized        = F.normalize(embedding_table, dim=-1)
    similarities          = (soft_token_normalized @ emb_normalized.T).squeeze(0)

    # Top-3 closest vocabulary tokens
    top3_scores, top3_ids = similarities.topk(3)
    top3_tokens = [tokenizer.decode([tid]) for tid in top3_ids]

    print(f"Soft token {i+1:2d}: closest vocab tokens = {top3_tokens} "
          f"(similarities: {top3_scores.tolist()})")

# Example output (sentiment prompt after training):
# Soft token  1: closest vocab tokens = ['▁positive', '▁negative', '▁sentiment'] (0.82, 0.79, 0.71)
# Soft token  2: closest vocab tokens = ['▁classify', '▁determine', '▁identify'] (0.78, 0.74, 0.69)
# Soft token  3: closest vocab tokens = ['▁review', '▁text', '▁sentence']       (0.71, 0.68, 0.65)
#
# The soft prompt has "learned" to occupy regions near task-relevant words,
# even though it was never constrained to match any specific vocabulary item.
# This is emergent task specialization through gradient descent.

# ─── Measure drift from initialization ────────────────────────────────────────
# If you saved the initial prompt, compare how far it drifted
# P_init = model at epoch 0
# drift  = (P - P_init).norm(dim=-1)
# print(f"Per-token drift from init: {drift.tolist()}")
''',
    },

    "Prompt Tuning Quality Benchmark — Ablation Template": {
        "description": "Template for running prompt length and init strategy ablation studies",
        "runnable": False,
        "code": '''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from datasets import load_dataset
import torch

MODEL_NAME = "google/flan-t5-large"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── Ablation grid ────────────────────────────────────────────────────────────
prompt_lengths   = [1, 5, 10, 20, 50, 100]
init_strategies  = [
    (PromptTuningInit.RANDOM, None),
    (PromptTuningInit.TEXT,   "Classify the sentiment:"),
]

results = {}

for num_tokens in prompt_lengths:
    for init_method, init_text in init_strategies:
        key = f"k={num_tokens}_{init_method.name}"

        config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=num_tokens,
            prompt_tuning_init=init_method,
            prompt_tuning_init_text=init_text if init_text else None,
            tokenizer_name_or_path=MODEL_NAME if init_text else None,
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model = get_peft_model(base_model, config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ... train here for N epochs on your dataset ...
        # accuracy = evaluate(model, eval_dataset)

        results[key] = {
            "num_tokens":  num_tokens,
            "init":        init_method.name,
            "n_params":    n_params,
            # "accuracy": accuracy,
        }

        print(f"Config {key}: {n_params:,} trainable params")

# ─── Expected results (T5-Large, SST-2 sentiment, 30 epochs) ──────────────────
#
# k=1  RANDOM:   n_params=1,024   accuracy≈87.5%
# k=1  TEXT:     n_params=1,024   accuracy≈89.2%
# k=10 RANDOM:   n_params=10,240  accuracy≈90.6%
# k=10 TEXT:     n_params=10,240  accuracy≈91.8%
# k=20 RANDOM:   n_params=20,480  accuracy≈91.4%
# k=20 TEXT:     n_params=20,480  accuracy≈93.1%   ← good default
# k=50 RANDOM:   n_params=51,200  accuracy≈92.8%
# k=50 TEXT:     n_params=51,200  accuracy≈93.6%   ← sweet spot
# k=100 RANDOM:  n_params=102,400 accuracy≈92.9%
# k=100 TEXT:    n_params=102,400 accuracy≈93.4%   ← diminishing returns
#
# Full fine-tuning T5-Large: ~95.2% (theoretical ceiling for this model)
''',
    },

    "Prompt Tuning from Scratch — No PEFT Library": {
        "description": "Implement Prompt Tuning manually without HuggingFace PEFT — understand every detail",
        "runnable": False,
        "code": '''import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Manual Prompt Tuning implementation ──────────────────────────────────────
class PromptTunedModel(nn.Module):
    """
    Wraps a frozen LLM with a trainable soft prompt prepended to every input.
    This is exactly what HuggingFace PEFT's PromptTuningConfig does internally.
    """

    def __init__(self, base_model_name: str, num_virtual_tokens: int = 20):
        super().__init__()

        # ── Load and FREEZE the base model ────────────────────────────────────
        self.model     = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze ALL base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # ── Create the TRAINABLE soft prompt ──────────────────────────────────
        d_model = self.model.config.hidden_size      # e.g., 4096 for LLaMA-7B
        self.num_virtual_tokens = num_virtual_tokens

        # The soft prompt: num_virtual_tokens × d_model
        # This is the ONLY trainable parameter
        self.soft_prompt = nn.Embedding(num_virtual_tokens, d_model)

        # Initialize from vocabulary embeddings (Strategy 2 — recommended)
        with torch.no_grad():
            # Sample random token IDs from the top 5000 most frequent tokens
            random_token_ids = torch.randint(0, 5000, (num_virtual_tokens,))
            # Copy those embeddings into the soft prompt
            self.soft_prompt.weight.data = (
                self.model.model.embed_tokens(random_token_ids).clone()
            )

        print(f"Base model params (frozen):   {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Soft prompt params (trainable): {self.soft_prompt.weight.numel():,}")
        print(f"Trainable %: {self.soft_prompt.weight.numel() / sum(p.numel() for p in self.model.parameters()) * 100:.6f}%")

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        seq_len    = input_ids.shape[1]

        # ── Step 1: Embed the real input tokens ───────────────────────────────
        # Get the embedding layer from the base model
        token_embeddings = self.model.model.embed_tokens(input_ids)
        # token_embeddings: [batch, seq_len, d_model]

        # ── Step 2: Get the soft prompt and expand to batch ───────────────────
        # Soft prompt indices: [0, 1, 2, ..., k-1]
        prompt_indices = torch.arange(self.num_virtual_tokens).to(input_ids.device)
        prompt_embeds  = self.soft_prompt(prompt_indices)
        # prompt_embeds: [num_virtual_tokens, d_model]

        # Expand to batch dimension (SAME prompt for all batch items)
        prompt_embeds  = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        # prompt_embeds: [batch, num_virtual_tokens, d_model]

        # ── Step 3: Concatenate soft prompt + real tokens ─────────────────────
        input_embeds = torch.cat([prompt_embeds, token_embeddings], dim=1)
        # input_embeds: [batch, num_virtual_tokens + seq_len, d_model]

        # ── Step 4: Update attention mask ─────────────────────────────────────
        # CRITICAL: prepend 1s for the soft prompt positions
        if attention_mask is not None:
            prefix_mask    = torch.ones(batch_size, self.num_virtual_tokens,
                                        dtype=attention_mask.dtype,
                                        device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # ── Step 5: Update labels ──────────────────────────────────────────────
        # For causal LM, labels must also be extended — mask out the soft prompt positions
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, self.num_virtual_tokens), fill_value=-100,
                dtype=labels.dtype, device=labels.device
            )
            # -100 tells the loss function to ignore these positions
            labels = torch.cat([prefix_labels, labels], dim=1)

        # ── Step 6: Forward pass through the FROZEN model ─────────────────────
        # Pass inputs_embeds instead of input_ids (since we already embedded + prepended)
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

# ─── Usage ────────────────────────────────────────────────────────────────────
model = PromptTunedModel("gpt2", num_virtual_tokens=20)

# Verify only soft prompt is trainable
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable parameters:")
for name, shape in trainable:
    print(f"  {name}: {list(shape)}")
# Should ONLY show: soft_prompt.weight: [20, 768]

# ─── Training step ────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)

tokenizer = model.tokenizer
sample    = tokenizer("Hello world, this is a test.", return_tensors="pt")
inputs    = {k: v for k, v in sample.items()}
inputs["labels"] = inputs["input_ids"].clone()

outputs = model(**inputs)
print(f"Loss: {outputs.loss.item():.4f}")
outputs.loss.backward()
optimizer.step()
optimizer.zero_grad()
print("One training step complete. Only soft_prompt.weight was updated.")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for the Prompt Tuning topic module."""
    return {
        "theory": THEORY,
        "theory_raw": THEORY,
        "interactive_components": [],
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
        "render_operations": None,
    }
