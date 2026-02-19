"""
Transformers - Intelligence in Disguise
============================================

"""


import math
import numpy as np
import base64
import os
import streamlit as st

TOPIC_NAME = "Transformer Architecture Model and Explanation"

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data.
    This allows images to render inside st.markdown() with unsafe_allow_html=True.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'




# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
##### Transformer Model Architecture

---
Even LSTM/GRU with their gates have two fundamental limitations:

*   SEQUENTIAL PROCESSING
*   LONG-RANGE DEPENDENCIES
    
---

    PROBLEM 1: SEQUENTIAL PROCESSING (Slow!)
    ═══════════════════════════════════════════
    
    LSTM processes one word at a time, in order:
    
      "The"  → wait → "cat" → wait → "sat" → wait → "on" → wait → "the" → wait → "mat"
       t=1             t=2             t=3             t=4             t=5            t=6
    
      Each step DEPENDS on the previous step's hidden state
      Cannot parallelize! Must wait for h(t-1) before computing h(t)
    
      For a 1000-word document: 1000 sequential steps
      GPUs are designed for PARALLEL computation — this wastes them
    
    
    PROBLEM 2: LONG-RANGE DEPENDENCIES (Still hard!)
    ═══════════════════════════════════════════════════
    
      "The cat that sat on the mat next to the dog that barked at the mailman was ___"
    
       word 1                                                                  word 16
       "The cat"                                                               "was ___"
          │                                                                       │
          └──── 14 steps apart ───────────────────────────────────────────────────┘
    
      Even LSTM: information must pass through 14 cells
      Each cell can slightly corrupt or dilute the information
      The connection between "cat" and "was" is INDIRECT
    
      What if every word could DIRECTLY look at every other word?
---

##### The Transformer's Big Idea: ATTENTION

    LSTM approach:
    ─────────────
      "The" → "cat" → "sat" → "on" → "the" → "mat" → predict
       Each word only sees the previous hidden state
       Information flows like a CHAIN
       
    Transformer approach:
    ─────────────────────
        "The"  "cat"  "sat"  "on"  "the"  "mat"  → predict
          ↕      ↕      ↕      ↕     ↕      ↕
        Every word can DIRECTLY attend to every other word
        Information flows like a WEB
        
             "The" ←──────────────────────── "mat"
               ↕                               ↕
             "cat" ←────── "on" ──────────► "the"
               ↕             ↕
             "sat" ←────────►↕
        
        No chain! No sequential processing!
        "cat" can directly look at "mat" in ONE step
---

##### Self-Attention — The Core Mechanism

    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │  Q = QUERY:   "What am I looking for?"                    │
    │                                                           │
    │  K = KEY:     "What do I contain?"                        │
    │                                                           │
    │  V = VALUE:   "What information do I give if selected?"   │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
    
    Analogy: searching a library
    
        You walk in with a QUERY   :      "I need books about cats"
        Each book has a KEY (title):      "Cat Biology", "Dog Training", "Cat Behavior"
        Each book has a VALUE      :      (the actual content inside)
        
        You MATCH your query against every key
        → "Cat Biology"   = strong match   → read a lot of this
        → "Dog Training"  = weak match     → mostly ignore
        → "Cat Behavior"  = strong match   → read a lot of this
        
        Your final knowledge = weighted combination of all values,
                             weighted by how well each key matched your query
                             
         
### Step-by-Step With Real Words

    Sentence: "The cat sat"

    Each word gets transformed into Q, K, V vectors using learned weight matrices:
    
      Word        Q (what I seek)       K (what I offer)      V (my content)
      ─────       ──────────────────    ──────────────────    ───────────────
      "The"       Q_the = [0.1, 0.2]    K_the = [0.3, 0.1]    V_the = [1, 0]
      "cat"       Q_cat = [0.9, 0.1]    K_cat = [0.8, 0.2]    V_cat = [0, 1]
      "sat"       Q_sat = [0.5, 0.7]    K_sat = [0.4, 0.9]    V_sat = [1, 1]
    
    These are computed as:
      Q = X × W_Q    (input × query weight matrix)
      K = X × W_K    (input × key weight matrix)
      V = X × W_V    (input × value weight matrix)
    
    W_Q, W_K, W_V are the LEARNED parameters (like filter weights in CNNs)
    

    Computing Attention for "cat"
    
    "cat" wants to know: which other words should I pay attention to?

    Step 1: SCORE — "cat"'s Query vs every word's Key (dot product)
    ══════════════════════════════════════════════════════════════════
    
        score("cat", "The") = Q_cat · K_the = (0.9×0.3) + (0.1×0.1) = 0.28
        score("cat", "cat") = Q_cat · K_cat = (0.9×0.8) + (0.1×0.2) = 0.74  ← highest!
        score("cat", "sat") = Q_cat · K_sat = (0.9×0.4) + (0.1×0.9) = 0.45
        
        "cat" finds itself most relevant (0.74), then "sat" (0.45), then "The" (0.28)
    
    
    Step 2: SCALE — divide by √(dimension) to keep numbers stable
    ══════════════════════════════════════════════════════════════════
    
        dimension = 2 (our vector size), √2 = 1.41
        
        scaled scores = [0.28/1.41, 0.74/1.41, 0.45/1.41]
                    = [0.20,      0.52,       0.32]
    
    
    Step 3: SOFTMAX — convert scores to probabilities (sum to 1)
    ══════════════════════════════════════════════════════════════════
    
        attention_weights = softmax([0.20, 0.52, 0.32])
                        = [0.24,  0.41,  0.35]
                           ▲      ▲      ▲
                          "The"  "cat"  "sat"
        
        "cat" will pay:  24% attention to "The"
                       41% attention to "cat" (itself)
                       35% attention to "sat"
    
    
    Step 4: WEIGHTED SUM of Values
    ══════════════════════════════════════════════════════════════════
    
        output_cat = 0.24 × V_the  +  0.41 × V_cat  +  0.35 × V_sat
                 = 0.24 × [1,0]  +  0.41 × [0,1]  +  0.35 × [1,1]
                 = [0.24, 0]     +  [0, 0.41]      +  [0.35, 0.35]
                 = [0.59, 0.76]
        
        This new vector for "cat" now CONTAINS information
        from all the other words, weighted by relevance!
  
    The Full Picture for All Words
    
                        Attention Weights Matrix
                        ──────────────────────────
                        "The"    "cat"    "sat"     ← Keys
                       ┌────────┬────────┬────────┐
        "The" (Query) →│  0.38  │  0.31  │  0.31  │  "The" mostly looks at itself
                       ├────────┼────────┼────────┤
        "cat" (Query) →│  0.24  │  0.41  │  0.35  │  "cat" looks at itself + "sat"
                       ├────────┼────────┼────────┤
        "sat" (Query) →│  0.21  │  0.33  │  0.46  │  "sat" looks at itself + "cat"
                       └────────┴────────┴────────┘
        
        Every word gets a NEW representation that's a blend
        of ALL other words, weighted by relevance.
        ALL of this happens in ONE step, in PARALLEL!

---

##### Multi-Head Attention

One attention mechanism might only capture one kind of relationship. 
Transformers use multiple heads — each one looks for different patterns:

        Head 1 might learn: "Who is the subject?"  (cat → sat)
        Head 2 might learn: "What's the article?"  (The → cat)
        Head 3 might learn: "What's the tense?"    (sat → past)
        Head 4 might learn: "What's nearby?"       (cat → sat, The → cat)
        
        ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Head 1  │   │  Head 2  │   │  Head 3  │   │  Head 4  │
        │  Q₁K₁V₁  │   │  Q₂K₂V₂  │   │  Q₃K₃V₃  │   │  Q₄K₄V₄  │
        └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │              │
             └──────┬───────┴──────┬───────┘              │
                    │              │                      │
                    ▼              ▼                      ▼
             ┌───────────────────────────────────────────────┐
             │           CONCATENATE all heads               │
             │    then multiply by W_output to combine       │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                            Final representation
                            (captures ALL types of relationships)
    
      All heads run in PARALLEL — no speed penalty!
      This is why Transformers love GPUs

##### Positional Encoding — Solving the Order Problem

    Problem: Attention treats all words equally — it has NO concept of order!
    
        "The cat sat on the mat"    ← attention sees this
        "mat the on sat cat The"    ← and this as IDENTICAL
    
        Because it's just matching Q, K, V — no sequential processing.
    
    
    Solution: ADD position information to each word BEFORE attention
    
        "The" = word_embedding + position_0_encoding
        "cat" = word_embedding + position_1_encoding
        "sat" = word_embedding + position_2_encoding
    
      Position encodings use sine and cosine waves at different frequencies:
    
      Position 0: [sin(0), cos(0), sin(0), cos(0)...]       = [0, 1, 0, 1...]
      Position 1: [sin(1), cos(1), sin(0.01), cos(0.01)...] = [0.84, 0.54, 0.01, 0.99...]
      Position 2: [sin(2), cos(2), sin(0.02), cos(0.02)...] = [0.91, -0.42, 0.02, 0.99...]
    
        Each position gets a UNIQUE fingerprint
        Nearby positions have SIMILAR patterns
      
      The network can learn "word at position 2 is close to position 3"

##### The Residual Connection (Add) — Why It Matters

    Without residual:    Input ──► [Attention] ──► [FFN] ──► Output
                     Information can get distorted through many layers

    With residual:       Input ──► [Attention] ──► [FFN] ──► Output
                            │                                   ▲
                            └───────────── ADD ─────────────────┘

                     Output = Attention(Input) + Input
                                                  ▲
                                                  │
                                      Original info preserved!

    Same idea as LSTM's cell state highway — gradient can flow
    directly through the skip connection without vanishing!

---

##### Full Transformer Architecture

        Input: "The cat sat on the ___"

                    ┌────────────────────────────────┐
                    │     INPUT EMBEDDING            │
                    │  word vectors + position info  │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │                              │
            ┌───────┤   TRANSFORMER BLOCK ×N       │  ◄── Stack N of these
            │       │                              │      (GPT-3 uses 96!)
            │       │  ┌────────────────────────┐  │
            │       │  │  Multi-Head Attention  │  │  ◄── Every word attends
            │       │  │  (8 or more heads)     │  │      to every word
            │       │  └───────────┬────────────┘  │
            │       │              │               │
         Repeat     │  ┌───────────▼────────────┐  │
         N times    │  │  Add & Layer Normalize │  │  ◄── Residual connection
            │       │  └───────────┬────────────┘  │      (skip connection)
            │       │              │               │
            │       │  ┌───────────▼────────────┐  │
            │       │  │  Feed-Forward Network  │  │  ◄── Regular dense layer
            │       │  │  (2 layers, ReLU)      │  │      applied to each word
            │       │  └───────────┬────────────┘  │
            │       │              │               │
            │       │  ┌───────────▼────────────┐  │
            │       │  │  Add & Layer Normalize │  │  ◄── Another residual
            │       │  └───────────┬────────────┘  │
            │       │              │               │
            └───────┤──────────────┘               │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     OUTPUT LAYER             │
                    │  Linear + Softmax            │
                    │  → probability for each word │
                    │     in vocabulary            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    Prediction: "mat" (highest probability)
                    
                    
##### Architecture: Core Modules

* The Transformer is built from a set of reusable modules. 
Below is a structured breakdown of every component you’ll find in the encoder–decoder and their supporting layers.


## Input Components
* Token Embeddings
Maps each token ID to a learnable vector of size dmodel.

* Positional Encodings
Adds fixed (sinusoidal) or learned vectors to inject token order information.

* (Optional) Segment/Type Embeddings
In models like BERT, distinguishes different segments (e.g., sentence A vs. B).

## Encoder Layer (repeated N times)
* Multi-Head Self-Attention
    • Scaled dot-product attention over queries, keys, and values from the same layer.
    • Splits d_model into h heads, computes attention in parallel, then concatenates.

* Add & LayerNorm
    • Residual connection: adds the attention output to the layer’s input.
    • Applies layer normalization.

* Position-Wise Feed-Forward Network (FFN)
    • Two linear transformations with ReLU (or GELU) in between.
    • Operates independently on each position.

* Add & LayerNorm
    • Residual connection around the FFN.
    • Layer normalization again.

* Dropout
    • Applied after attention, FFN, and sometimes on embeddings for regularization.

## Decoder Layer (repeated N times)
* Masked Multi-Head Self-Attention
    • Prevents attending to future positions via a causal (look-ahead) mask.

* Add & LayerNorm
    • Residual + layer normalization.

* Encoder–Decoder Multi-Head Attention
    • Queries from decoder; keys/values from encoder outputs.
    • Enables decoder to focus on relevant encoder representations.

* Add & LayerNorm
    • Residual + layer normalization.
    
* Position-Wise Feed-Forward Network
    • Same structure as in the encoder.

* Add & LayerNorm
    • Final residual + layer normalization.

* Dropout
    • Applied after each sub-layer for stability and regularization.

## Output Components

* Linear Projection: Maps decoder hidden states (size d_model) to vocabulary logits.

* Softmax : Converts logits to probability distribution over tokens.

---

##### Supporting Mechanisms

Scaled Dot-Product Attention - Computes attention weights :
             
    - Attention(Q, K, V) = softmax( (Q K^T) / √d_k ) V
             
First — What Are Q, K, V?

Every word in your input gets transformed into THREE different vectors:

    Input word: "cat" → word embedding → [0.2, 0.5, 0.8, 0.1]
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    │                     │                      │
                    ▼                     ▼                      ▼
               × W_Q matrix          × W_K matrix           × W_V matrix
               (learned)             (learned)              (learned)
                    │                     │                      │
                    ▼                     ▼                      ▼
               Q = [0.9, 0.1]        K = [0.8, 0.2]        V = [0.3, 0.7]
                    │                     │                      │
                    ▼                     ▼                      ▼
               "What am I            "What do I             "What info
                looking for?"         contain?"              do I carry?"


## The Library Analogy (Expanded)

    Imagine you're in a library:
    
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  Q = QUERY (Question)                                            │
    │  ─────────────────────                                           │
    │  "What information am I searching for?"                          │
    │                                                                  │
    │  You walk into a library and say:                                │
    │  "I need information about animals"                              │
    │  That's your QUERY                                               │
    │                                                                  │
    │                                                                  │
    │  K = KEY (Label on each book)                                    │
    │  ─────────────────────────────                                   │
    │  "What information does this word contain/offer?"                │
    │                                                                  │
    │  Each book on the shelf has a title:                             │
    │  Book 1: "Animal Biology"     ← KEY of book 1                    │
    │  Book 2: "French Cooking"     ← KEY of book 2                    │
    │  Book 3: "Animal Behavior"    ← KEY of book 3                    │
    │                                                                  │
    │  You MATCH your query against each key:                          │
    │  "animals" vs "Animal Biology"  → HIGH match                     │
    │  "animals" vs "French Cooking"  → LOW match                      │
    │  "animals" vs "Animal Behavior" → HIGH match                     │
    │                                                                  │
    │                                                                  │
    │  V = VALUE (Actual content inside each book)                     │
    │  ────────────────────────────────────────────                    │
    │  "What information do I actually give you?"                      │
    │                                                                  │
    │  After matching, you READ the books with high match:             │
    │  Book 1 content: [detailed biology facts]     ← VALUE of book 1  │
    │  Book 2 content: [recipes]                    ← VALUE of book 2  │
    │  Book 3 content: [behavioral studies]         ← VALUE of book 3  │
    │                                                                  │
    │  Your final knowledge = mostly Book 1 & 3 content (high match)   │
    │                        + barely any Book 2 content (low match)   │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    
    
    Why not just use the word itself for everything?
    
    Because a word NEEDS different things vs OFFERS different things!
    
    Example: "The cat sat"
    
      "sat" as a QUERY asks   :     "Who did the sitting?" (looking for a subject)
      "sat" as a KEY offers   :     "I'm a past-tense verb/action"
      "sat" as a VALUE carries:     actual semantic meaning of sitting
    
      "cat" as a QUERY asks   :     "What did I do?" (looking for a verb)
      "cat" as a KEY offers   :     "I'm a noun/subject/animal"
      "cat" as a VALUE carries:     actual semantic meaning of cat
    
      Different roles need different representations!
    
    
    Now — The Formula Piece by Piece
    
    Attention(Q, K, V) = softmax( (Q × K^T) / √d_k ) × V

    Let me number each operation:
    
    Step 1  :  K^T           ← transpose K
    Step 2  :  Q × K^T       ← multiply to get match scores
    Step 3  :  / √d_k        ← scale down to keep numbers stable
    Step 4  :  softmax(...)  ← convert scores to probabilities
    Step 5  :  × V           ← weighted sum of values
    
    
    Working Through With Real Numbers
    
    Sentence: "I love cats"   (3 words)
    Each word embedded as dimension d_k = 4
    
    After multiplying by W_Q, W_K, W_V (let's say output dimension = 2):
        
                 Q (what I seek)     K (what I offer)     V (my content)
        "I"      [1, 0]              [0, 1]               [1, 0]
        "love"   [0, 1]              [1, 1]               [0, 1]
        "cats"   [1, 1]              [1, 0]               [1, 1]
        
        Written as matrices:
        
              ┌       ┐           ┌       ┐           ┌       ┐
              │ 1   0 │           │ 0   1 │           │ 1   0 │
        Q  =  │ 0   1 │     K  =  │ 1   1 │     V  =  │ 0   1 │
              │ 1   1 │           │ 1   0 │           │ 1   1 │
              └       ┘           └       ┘           └       ┘
             (3 × 2)             (3 × 2)             (3 × 2)
             3 words,            3 words,            3 words,
             2 dimensions        2 dimensions        2 dimensions


    Step 1: K^T (K Transpose)
    
    "Transpose" = flip rows and columns

          ┌       ┐                ┌           ┐
          │ 0   1 │                │ 0   1   1 │
    K  =  │ 1   1 │    K^T  =      │ 1   1   0 │
          │ 1   0 │                └           ┘
          └       ┘                  (2 × 3)
           (3 × 2)
    
    Rows become columns:
      Row 1 [0, 1] → Column 1
      Row 2 [1, 1] → Column 2
      Row 3 [1, 0] → Column 3
    
    WHY transpose?
      Q is (3 × 2) — 3 words, each asking with 2-dim query
      K is (3 × 2) — 3 words, each offering 2-dim key
    
      To compute "how well does each query match each key"
      we need matrix multiplication: (3×2) × (2×3) = (3×3)
      That gives us a score for EVERY word pair!
      K must be transposed to make the dimensions line up.

    Step 2: Q × K^T (Match Scores)
    
        ┌       ┐     ┌           ┐       ┌                           ┐
        │ 1   0 │     │ 0   1   1 │       │ 1×0+0×1  1×1+0×1  1×1+0×0 │
        │ 0   1 │  ×  │ 1   1   0 │   =   │ 0×0+1×1  0×1+1×1  0×1+1×0 │
        │ 1   1 │     └           ┘       │ 1×0+1×1  1×1+1×1  1×1+1×0 │
        └       ┘                         └                           ┘
        
           ┌            ┐
           │  0   1   1 │  ← "I" scores   :   0 with "I", 1 with "love", 1 with "cats"
         = │  1   1   0 │  ← "love" scores:   1 with "I", 1 with "love", 0 with "cats"
           │  1   2   1 │  ← "cats" scores:   1 with "I", 2 with "love", 1 with "cats"
           └            ┘
              (3 × 3)
        
        This is the ATTENTION SCORE MATRIX!
        Every word has a match score with every other word.
        
        "cats" has highest score (2) with "love" — makes sense!
        "love" has zero score with "cats" — interesting asymmetry
        (what "love" seeks ≠ what "cats" offers, but what "cats" seeks ≈ what "love" offers)


    Step 3: Divide by √d_k (Scale)
    
        d_k = 2 (dimension of our Q and K vectors)
        √d_k = √2 = 1.414
        
               ┌            ┐                ┌                     ┐
               │  0   1   1 │                │  0.00   0.71   0.71 │
               │  1   1   0 │  / 1.414  =    │  0.71   0.71   0.00 │
               │  1   2   1 │                │  0.71   1.41   0.71 │
               └            ┘                └                     ┘
        
        WHY divide by √d_k?
        
          Without scaling, larger dimensions = larger dot products:
        
          d_k = 2:    dot product might be    0 to 2
          d_k = 512:  dot product might be    0 to 512!
        
          Large numbers going into softmax create EXTREME probabilities:
        
          softmax([1, 2])     = [0.27, 0.73]    ← nice spread
          softmax([100, 200]) = [0.00, 1.00]    ← one word gets ALL attention!
        
          Dividing by √d_k brings the numbers back to a reasonable range
          regardless of how big d_k is.
        
          Think of it as: "normalize for dimension size so the model
          behaves consistently whether d_k = 2 or d_k = 512"

    Step 4: Softmax (Convert to Probabilities)
    
        Softmax applied to EACH ROW (each word's attention distribution):

        softmax([0.00, 0.71, 0.71]) = [0.20, 0.40, 0.40]   ← "I" pays 20% to self,
                                                              40% to "love", 40% to "cats"
        
        softmax([0.71, 0.71, 0.00]) = [0.37, 0.37, 0.26]   ← "love" pays roughly equal
                                                              to "I" and self
        
        softmax([0.71, 1.41, 0.71]) = [0.24, 0.48, 0.28]   ← "cats" pays 48% to "love"!
                                                              highest attention
        
        
        Attention Weight Matrix:
        
                        "I"     "love"   "cats"     ← being attended TO (Keys)
                        ┌────────┬────────┬────────┐
        "I"       →     │  0.20  │  0.40  │  0.40  │  sum = 1.0 ✓
                        ├────────┼────────┼────────┤
        "love"    →     │  0.37  │  0.37  │  0.26  │  sum = 1.0 ✓
                        ├────────┼────────┼────────┤
        "cats"    →     │  0.24  │  0.48  │  0.28  │  sum = 1.0 ✓
                        └────────┴────────┴────────┘
                        ▲
                        attending FROM (Queries)
        
        Each row sums to 1 — it's a probability distribution!
        "Of all the words, how much attention should I pay to each?"

    Step 5: Multiply by V (Weighted Sum of Values)
    
        ┌                   ┐       ┌       ┐
        │ 0.20  0.40  0.40  │       │ 1   0 │    V for "I"
        │ 0.37  0.37  0.26  │   ×   │ 0   1 │    V for "love"
        │ 0.24  0.48  0.28  │       │ 1   1 │    V for "cats"
        └                   ┘       └       ┘
          attention weights           values
        
        
        For "I" (first row):
        = 0.20 × [1,0]  +  0.40 × [0,1]  +  0.40 × [1,1]
        = [0.20, 0]     +  [0, 0.40]     +  [0.40, 0.40]
        = [0.60, 0.80]
        
        For "love" (second row):
        = 0.37 × [1,0]  +  0.37 × [0,1]  +  0.26 × [1,1]
        = [0.37, 0]     +  [0, 0.37]     +  [0.26, 0.26]
        = [0.63, 0.63]
        
        For "cats" (third row):
        = 0.24 × [1,0]  +  0.48 × [0,1]  +  0.28 × [1,1]
        = [0.24, 0]     +  [0, 0.48]     +  [0.28, 0.28]
        = [0.52, 0.76]
        
        
        Final Output:
           ┌             ┐
           │ 0.60   0.80 │  ← new representation of "I"
           │ 0.63   0.63 │  ← new representation of "love"
           │ 0.52   0.76 │  ← new representation of "cats"
           └             ┘
        
        Each word is now a BLEND of all other words' values,
        weighted by how relevant they are!
        
        "cats" [0.52, 0.76] is heavily influenced by "love" (48% attention)
        Its new representation CONTAINS information about "love"

---   
##### The Complete Formula — All Together
    
    Attention(Q, K, V) = softmax( (Q × K^T) / √d_k ) × V

    Step by step:
    
        Q × K^T           "How well does each word match every other word?"
                         → raw match scores (can be any number)
        
        / √d_k            "Scale down so softmax doesn't go extreme"
                         → stable scores
        
        softmax(...)      "Convert to probabilities (0-1, sum to 1)"
                         → attention weights
        
        × V               "Blend each word's value using those weights"
                         → context-aware word representations
    
---   
##### Quick Reference Card

    ┌──────────┬───────────────────┬─────────────────────────────────┐
    │ Symbol   │ Name              │ Meaning                         │
    ├──────────┼───────────────────┼─────────────────────────────────┤
    │ Q        │ Query matrix      │ "What is each word looking for?"│
    │ K        │ Key matrix        │ "What does each word offer?"    │
    │ V        │ Value matrix      │ "What info does each word carry"│
    │ K^T      │ K transposed      │ K flipped (rows↔cols) so we     │
    │          │                   │ can multiply Q × K              │
    │ d_k      │ Key dimension     │ Size of each Q/K vector         │
    │ √d_k     │ Scale factor      │ Prevents extreme softmax values │
    │ Q × K^T  │ Attention scores  │ Match score for every word pair │
    │ softmax  │ Normalization     │ Converts scores to probabilities│
    │ × V      │ Weighted sum      │ Blend values by attention weight│
    └──────────┴───────────────────┴─────────────────────────────────┘


    The beautiful result:
    
      INPUT:  each word knows only about ITSELF
      OUTPUT: each word knows about ALL relevant words
    
      And this happens in ONE matrix multiplication — fully PARALLEL!
    
--- 

##### Multi-Head Attention — Why One Head Isn't Enough

    Think about reading this sentence:

        "The cat that chased the mouse sat on the mat"
    
    A SINGLE attention head might focus on ONE type of relationship:
    
        Head focuses on: "Who did the action?"
        "sat" attends strongly to → "cat" (subject)
      
        But it MISSES:
        - "cat" also relates to "chased" (what cat did)
        - "chased" relates to "mouse" (object)
        - "sat" relates to "mat" (location)
        - "the" relates to "cat" (article-noun pair)
    
        One head can only learn ONE attention pattern!
        We need MULTIPLE heads looking for DIFFERENT patterns.

---

The Concept

    SINGLE HEAD ATTENTION:
    ══════════════════════

        Input ──► ONE set of (Q, K, V) ──► ONE attention pattern ──► Output
        
        Like reading a book with ONE question in mind:
        "Who is the main character?" — you miss plot, setting, themes


    MULTI-HEAD ATTENTION (say 4 heads):
    ════════════════════════════════════

        Input ──┬──► Head 1: (Q₁, K₁, V₁) ──► pattern 1 (grammar)
                ├──► Head 2: (Q₂, K₂, V₂) ──► pattern 2 (meaning)
                ├──► Head 3: (Q₃, K₃, V₃) ──► pattern 3 (position)
                └──► Head 4: (Q₄, K₄, V₄) ──► pattern 4 (reference)
                          │
                          ▼
                    CONCATENATE all patterns
                          │
                          ▼
                    × W_O (output projection)
                          │
                          ▼
                    Final output (combines ALL relationship types)
        
        Like reading a book with FOUR questions simultaneously:
        "Who is the main character?"
        "What's the emotional tone?"
        "Where does it take place?"
        "What's the cause and effect?"

---  

The Dimension Split

    This is the clever part — multi-head attention is NOT more expensive!

    Say your model dimension d_model = 8
    
    SINGLE HEAD:
        Q, K, V each have dimension 8
        Attention computation: (3 × 8) × (8 × 3) = (3 × 3) scores
        Cost: expensive with dimension 8
    
    MULTI-HEAD (4 heads):
        SPLIT the 8 dimensions among 4 heads
        Each head gets d_k = 8 / 4 = 2 dimensions
        
        Full embedding for "cat":  [0.2, 0.5, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4]
                                     ▲              ▲              ▲         ▲
                                     │              │              │         │
                                    Head 1         Head 2         Head 3    Head 4
                                    [0.2, 0.5]    [0.8, 0.1]    [0.9, 0.3] [0.7, 0.4]
        
        Each head does attention with SMALLER vectors (dimension 2 instead of 8)
        4 heads × (dimension 2) = same total cost as 1 head × (dimension 8)
        BUT we get 4 DIFFERENT attention patterns!

---     
   
Full Numeric Example

    Sentence: "I love cats"    (3 words)
    d_model = 4                (total embedding size)
    num_heads = 2              (two attention heads)
    d_k = d_model / num_heads = 4 / 2 = 2   (each head gets 2 dimensions)
    
    Word embeddings (d_model = 4):
      "I"    = [1.0, 0.0, 0.5, 0.5]
      "love" = [0.0, 1.0, 1.0, 0.0]
      "cats" = [0.5, 0.5, 0.0, 1.0]

---

Each Head Has Its Own W_Q, W_K, W_V

    HEAD 1: focuses on dimensions [0,1] — learns grammatical relationships
    ═══════════════════════════════════════════════════════════════════════
    
        W_Q₁ (4×2 matrix):        W_K₁ (4×2 matrix):        W_V₁ (4×2 matrix):
        ┌              ┐           ┌              ┐           ┌              ┐
        │  1.0    0.0  │           │  0.0    1.0  │           │  1.0    0.0  │
        │  0.0    1.0  │           │  1.0    0.0  │           │  0.0    1.0  │
        │  0.0    0.0  │           │  0.0    0.0  │           │  0.0    0.0  │
        │  0.0    0.0  │           │  0.0    0.0  │           │  0.0    0.0  │
        └              ┘           └              ┘           └              ┘
        
        (Simplified weight matrices — in reality these are learned)
        
        Computing Q₁, K₁, V₁ for each word:
        
        Q₁("I")    = [1,0,0.5,0.5] × W_Q₁ = [1.0, 0.0]
        Q₁("love") = [0,1,1.0,0.0] × W_Q₁ = [0.0, 1.0]
        Q₁("cats") = [0.5,0.5,0,1] × W_Q₁ = [0.5, 0.5]
        
        K₁("I")    = [1,0,0.5,0.5] × W_K₁ = [0.0, 1.0]
        K₁("love") = [0,1,1.0,0.0] × W_K₁ = [1.0, 0.0]
        K₁("cats") = [0.5,0.5,0,1] × W_K₁ = [0.5, 0.5]
        
        V₁("I")    = [1,0,0.5,0.5] × W_V₁ = [1.0, 0.0]
        V₁("love") = [0,1,1.0,0.0] × W_V₁ = [0.0, 1.0]
        V₁("cats") = [0.5,0.5,0,1] × W_V₁ = [0.5, 0.5]
    
    
        Now run attention exactly as before:
        
        Q₁ × K₁^T:
        ┌              ┐     ┌              ┐       ┌              ┐
        │ 1.0    0.0   │     │ 0.0  1.0  0.5│       │ 0.0  1.0  0.5│
        │ 0.0    1.0   │  ×  │ 1.0  0.0  0.5│   =   │ 1.0  0.0  0.5│
        │ 0.5    0.5   │     └              ┘       │ 0.5  0.5  0.5│
        └              ┘                            └              ┘
        
        / √d_k = / √2 = / 1.414:
        ┌                    ┐
        │ 0.00   0.71   0.35 │
        │ 0.71   0.00   0.35 │
        │ 0.35   0.35   0.35 │
        └                    ┘
        
        softmax (each row):
        ┌                    ┐
        │ 0.26   0.42   0.32 │  ← "I" attends most to "love"
        │ 0.42   0.26   0.32 │  ← "love" attends most to "I"
        │ 0.33   0.33   0.34 │  ← "cats" attends roughly equally
        └                    ┘
        
        × V₁:
        ┌                    ┐     ┌              ┐       ┌              ┐
        │ 0.26  0.42  0.32   │     │ 1.0    0.0   │       │ 0.42   0.58  │
        │ 0.42  0.26  0.32   │  ×  │ 0.0    1.0   │   =   │ 0.58   0.42  │
        │ 0.33  0.33  0.34   │     │ 0.5    0.5   │       │ 0.50   0.50  │
        └                    ┘     └              ┘       └              ┘
        
        Head 1 Output:
        "I"    = [0.42, 0.58]  ← blended with "love" info
        "love" = [0.58, 0.42]  ← blended with "I" info
        "cats" = [0.50, 0.50]  ← balanced mix
    
    
    HEAD 2: focuses on dimensions [2,3] — learns semantic relationships
    ═══════════════════════════════════════════════════════════════════════
    
        (Different W_Q₂, W_K₂, W_V₂ matrices — learning different patterns)
        
        Let's say after the same process, Head 2 produces:
        
        Head 2 Output:
        "I"    = [0.30, 0.70]  ← different pattern!
        "love" = [0.60, 0.40]  ← "love" attends more to "cats" here
        "cats" = [0.55, 0.65]  ← "cats" attends more to "love" here

---

Concatenate + Project

    CONCATENATE the outputs from all heads:

    Head 1 output         Head 2 output    Concatenated
    ─────────────         ─────────────    ────────────────────
    "I":    [0.42, 0.58]  [0.30, 0.70]  →  [0.42, 0.58, 0.30, 0.70]
    "love": [0.58, 0.42]  [0.60, 0.40]  →  [0.58, 0.42, 0.60, 0.40]
    "cats": [0.50, 0.50]  [0.55, 0.65]  →  [0.50, 0.50, 0.55, 0.65]
    
    Size: back to d_model = 4 ✓  (2 from head 1 + 2 from head 2)
    
    
    MULTIPLY by W_O (output projection matrix, 4×4):
    
    This learned matrix COMBINES the insights from all heads:
    
    Final("I")    = [0.42, 0.58, 0.30, 0.70] × W_O = [new 4-dim vector]
    Final("love") = [0.58, 0.42, 0.60, 0.40] × W_O = [new 4-dim vector]
    Final("cats") = [0.50, 0.50, 0.55, 0.65] × W_O = [new 4-dim vector]
    
    Each word now carries information from MULTIPLE types of relationships!

---

What Each Head Learns (In Real Transformers)

    Researchers have visualized what different heads learn in trained models:
    
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  Head 1: "Who is the SUBJECT of this verb?"             │
    │                                                         │
    │    "The cat that chased the mouse sat on the mat"       │
    │          ████═══════════════════════►████               │
    │          cat ─────────────────────► sat                 │
    │                                                         │
    │  Head 2: "What is the OBJECT?"                          │
    │                                                         │
    │    "The cat that chased the mouse sat on the mat"       │
    │                   ████══►████                           │
    │                   chased → mouse                        │
    │                                                         │
    │  Head 3: "What is the NEXT word?"                       │
    │                                                         │
    │    "The cat that chased the mouse sat on the mat"       │
    │     ██►██  ██►██    ██►██   ██►██ ██►██ ██►██           │
    │     The→cat cat→that ...                                │
    │                                                         │
    │  Head 4: "What ARTICLE goes with which NOUN?"           │
    │                                                         │
    │    "The cat that chased the mouse sat on the mat"       │
    │     ███══►███            ███══════════════►███          │
    │     The → cat            the ────────────► mat          │
    │                                                         │
    │  Head 5: "What are RELATED CONCEPTS?"                   │
    │                                                         │
    │    "The cat that chased the mouse sat on the mat"       │
    │          ████══════════►████                            │
    │          cat ──────────► mouse (both animals)           │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    
    Nobody programs these patterns — the heads DISCOVER them during training!

---

##### How W_Q, W_K, W_V Are Learned
    The key question: how does the model figure out the RIGHT
    W_Q, W_K, W_V matrices?
    
    Answer: the EXACT same backpropagation process you already learned!

---

##### The Training Loop

    STEP 1: Initialize all weights RANDOMLY
    ═══════════════════════════════════════════
        W_Q = random small numbers    e.g., [[0.01, -0.03], [0.02, 0.01], ...]
        W_K = random small numbers
        W_V = random small numbers
        W_O = random small numbers
        
        At this point, attention is MEANINGLESS — just random weights
        looking at random things.

    
    STEP 2: Forward pass with training data
    ═══════════════════════════════════════════
        Training example:
            Input:  "The cat sat on the ___"
            Target: "mat"
        
        Input ──► Embedding ──► Q,K,V ──► Attention ──► Output Layer ──► Prediction
                                  ▲
                             Using random
                             W_Q, W_K, W_V
        
        Prediction: "banana"  (random weights = random prediction!)
        Target    : "mat"
        Loss      : HUGE
        

    STEP 3: Backward pass — gradients flow back to W_Q, W_K, W_V
    ═══════════════════════════════════════════════════════════════
    
        Remember the chain rule from our earlier example?
        Same thing, just a longer chain:
        
        ∂Loss         ∂Loss      ∂Output    ∂Attention    ∂(softmax)    ∂(QK^T)     ∂Q
        ─────────  =  ─────── ×  ─────── ×  ────────── ×  ────────── ×  ─────── ×  ─────
        ∂W_Q          ∂Output    ∂Attn_out  ∂(softmax)    ∂(QK^T)       ∂Q         ∂W_Q
        
                        ▲           ▲            ▲            ▲            ▲          ▲
                        │           │            │            │            │          │
                      output     attention    softmax       score        query     finally
                      layer      weighted     function      compute      compute   reaches
                              
                              sum W_Q!
        
        Each piece in this chain can be calculated because every
        operation (matrix multiply, softmax, etc.) has a known derivative.
    
    
    STEP 4: Update weights
    ═══════════════════════════════════════════════════════════════
    
        W_Q_new = W_Q_old - learning_rate × ∂Loss/∂W_Q
        W_K_new = W_K_old - learning_rate × ∂Loss/∂W_K
        W_V_new = W_V_old - learning_rate × ∂Loss/∂W_V
        W_O_new = W_O_old - learning_rate × ∂Loss/∂W_O
    
    
    STEP 5: Repeat millions of times
    ═══════════════════════════════════════════════════════════════
    
        Iteration 1:      Loss = 15.2    (terrible — random guessing)
        Iteration 100:    Loss = 8.4     (starting to learn basic patterns)
        Iteration 10000:  Loss = 2.1     (learning grammar, common phrases)
        Iteration 100000: Loss = 0.5     (understanding context well)
        Iteration 1M+:    Loss = 0.1     (near-human language understanding)
    
---

## What Happens Inside W_Q, W_K, W_V During Training

    EARLY TRAINING (random weights):
    ════════════════════════════════
    
        "The cat sat on the mat"
    
        Attention pattern — basically random:
    
                "The"  "cat" "sat" "on" "the"  "mat"
        "The"  [ 0.17  0.15  0.18  0.16  0.17  0.17 ]  ← looking everywhere equally
        "cat"  [ 0.16  0.18  0.15  0.17  0.17  0.17 ]  ← no meaningful pattern
        "sat"  [ 0.17  0.16  0.17  0.16  0.18  0.16 ]
      ...
    
      W_Q hasn't learned what to SEARCH for
      W_K hasn't learned what to ADVERTISE
      Result: uniform attention = "I don't know what's important"
    
    
    MID TRAINING (partially learned):
    ══════════════════════════════════
    
        "The cat sat on the mat"
        
        Attention is starting to show structure:
        
                "The"  "cat" "sat" "on" "the"  "mat"
        "The"  [ 0.25  0.30  0.15  0.10  0.10  0.10 ]  ← starting to look at nearby
        "cat"  [ 0.20  0.25  0.30  0.10  0.05  0.10 ]  ← "cat" notices "sat"
        "sat"  [ 0.10  0.35  0.20  0.15  0.05  0.15 ]  ← "sat" notices "cat"
        ...
        
        W_Q is learning: "verbs should look for their subjects"
        W_K is learning: "nouns should advertise they're subjects"
    
    
    FULLY TRAINED:
    ══════════════
    
        "The cat sat on the mat"
    
        Attention shows clear linguistic understanding:
        
                "The" "cat" "sat"  "on" "the"  "mat"
        "The"  [ 0.10  0.70  0.05  0.05  0.05  0.05 ]  ← "The" → "cat" (my noun!)
        "cat"  [ 0.10  0.15  0.55  0.05  0.05  0.10 ]  ← "cat" → "sat" (my verb!)
        "sat"  [ 0.05  0.50  0.10  0.10  0.05  0.20 ]  ← "sat" → "cat" (who sat?)
        "on"   [ 0.05  0.05  0.30  0.10  0.05  0.45 ]  ← "on"  → "mat" (on what?)
        ...
        
        W_Q learned: subjects search for verbs, prepositions search for objects
        W_K learned: verbs advertise to subjects, nouns advertise to prepositions
        W_V learned: what semantic content to pass along when attended to

---

{{TransformerModel_IMAGE}}

---
##### The Gradient Path Visualized
    For a specific weight in W_Q, say w_q[0][0]:

          Loss = 3.5
            ▲
            │ ∂Loss/∂prediction
            │
          prediction = softmax(output_logits)
            ▲
            │ ∂prediction/∂output_logits
            │
          output_logits = W_output × attention_output
            ▲
            │ ∂output_logits/∂attention_output
            │
          attention_output = softmax(scores) × V         ← Step 5 of attention
            ▲
            │ ∂attention_output/∂scores
            │
          scores = Q × K^T / √d_k                       ← Step 2-3 of attention
            ▲
            │ ∂scores/∂Q
            │
          Q = Input × W_Q                                ← Step 0 of attention
            ▲
            │ ∂Q/∂W_Q = Input
            │
          W_Q [0][0]  ← THIS is the weight we're updating!
            │
            ▼
          w_q[0][0]_new = w_q[0][0]_old - lr × (product of ALL the above)
            
          The gradient is the PRODUCT of all these partial derivatives
          multiplied together via the chain rule — just like we computed
          for the simple perceptron, but a longer chain!
---



---
##### Why Transformers DON'T Have Vanishing Gradients

    RNN gradient path to early time steps:

        Loss → t=100 → t=99 → t=98 → ... → t=2 → t=1 → W
               ×W_h    ×W_h   ×W_h          ×W_h   ×W_h
             
        99 multiplications → gradient vanishes!


* Transformer gradient path to ANY word:

        Loss → Output Layer → Attention Output → W_Q
                                  ▲
                                  │
             Attention connects DIRECTLY to every word
             Plus RESIDUAL connections (skip connections):
        
        Loss → Layer N ────────────────────────┐
             │                                 │ (skip)
             ▼                                 │
             Layer N-1 ────────────────────┐   │
             │                             │   │
             ▼                             │   │
             Layer N-2                     │   │
             ...                           │   │
             ▼                             │   │
             Layer 1 ◄─────────────────────┘   │
             ▼                                 │
             Input ◄───────────────────────────┘
        
        Gradient can take the SHORTCUT through residual connections!
        Never has to pass through more than a few multiplications.
  
        That's why Transformers can be 96 layers deep (GPT-3)
        without vanishing gradients!
---

##### Complete Summary — All the Learned Parameters in a Transformer

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  For EACH transformer block (and there are N blocks):           │
    │                                                                 │
    │  Multi-Head Attention:                                          │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │  Per head (× num_heads):                                  │  │
    │  │    W_Q  (d_model × d_k)    "how to create queries"        │  │
    │  │    W_K  (d_model × d_k)    "how to create keys"           │  │
    │  │    W_V  (d_model × d_k)    "how to create values"         │  │
    │  │                                                           │  │
    │  │  Shared across heads:                                     │  │
    │  │    W_O  (d_model × d_model) "how to combine head outputs" │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  Feed-Forward Network:                                          │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │    W_1  (d_model × d_ff)    "expand to higher dimension"  │  │
    │  │    b_1  (d_ff)              "bias 1"                      │  │
    │  │    W_2  (d_ff × d_model)    "compress back down"          │  │
    │  │    b_2  (d_model)           "bias 2"                      │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  Layer Norms:                                                   │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │    γ, β  (d_model each)     "scale and shift"             │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  ALL of these are learned through backpropagation!              │
    │  GPT-3 has 96 blocks × all these parameters = 175 BILLION       │
    │  weights total, all trained by the same gradient descent        │
    │  process we traced with a single perceptron!                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
---

##### Congratulations 
You've now traced the complete picture from a single weight update 
in a perceptron all the way to how a 175-billion parameter Transformer learns. 
The core principle never changed 
— it's always 

    forward pass → measure error → chain rule backward → update weights → repeat.


##### Why Transformers Beat RNNs

                    RNN/LSTM                    Transformer
                   ═══════════                 ═══════════════
    
    Processing:    Sequential                  PARALLEL
                   word by word                all words at once
    
                   "The"→"cat"→"sat"           "The" "cat" "sat"
                   must wait for each           processed simultaneously
    
    Speed:         Slow (sequential)           Fast (parallel)
                   Can't fully use GPUs         Perfect for GPUs
    
    Long-range:    Indirect path               DIRECT path
                   "cat"→→→→→→→"was"           "cat"↔"was"
                   through 14 cells             one attention step
    
                   ┌───┐ ┌───┐ ┌───┐           ┌───┐     ┌───┐
                   │cat│→│...│→│was│           │cat│←───→│was│
                   └───┘ └───┘ └───┘           └───┘     └───┘
                   14 hops = signal decay      1 hop = full signal
    
    Gradient:      Vanishes over time          Flows through residual
                                               connections directly
    
    Context:       Fixed-size hidden state     Grows with sequence
                   h = [0.3, 0.1, ...]        Attention can look at
                   must compress EVERYTHING    ANY word at ANY distance
                   into one small vector
    
    Training:      Slow                        Much faster
                   (sequential = no parallel)  (parallel = GPU heaven)


---
##### The Two Types of Transformers

    1. ENCODER-ONLY (e.g., BERT)
    ─────────────────────────
    Every word attends to EVERY other word (bidirectional)
    
    "The cat [MASK] on the mat"
     ↕   ↕    ↕    ↕  ↕   ↕
    All words see all words — figures out [MASK] = "sat"
    
    Used for: understanding text (classification, Q&A, search)
    
    
    2. DECODER-ONLY (e.g., GPT, Claude)
    ─────────────────────────────────
    
    Each word can ONLY attend to words BEFORE it (causal/left-to-right)
    
    "The cat sat on the ___"
    ← ← ← ← ← ←
    "The" sees: only itself
    "cat" sees: "The", "cat"
    "sat" sees: "The", "cat", "sat"
    "___" sees: all previous words → predicts "mat"
    
    This is done using a MASK in the attention matrix:
    
                "The"   "cat"  "sat"    "on"    "the"   "___"
    "The"    →  [ ✓     ✗       ✗       ✗       ✗       ✗  ]
    "cat"    →  [ ✓     ✓       ✗       ✗       ✗       ✗  ]
    "sat"    →  [ ✓     ✓       ✓       ✗       ✗       ✗  ]
    "on"     →  [ ✓     ✓       ✓       ✓       ✗       ✗  ]
    "the"    →  [ ✓     ✓       ✓       ✓       ✓       ✗  ]
    "___"    →  [ ✓     ✓       ✓       ✓       ✓       ✓  ]
    
    ✗ = masked (set to -infinity before softmax → 0 attention)
    
    Used for: generating text (chatbots, writing, code)
    
    
    3. ENCODER-DECODER (e.g., T5, original Transformer)
        ─────────────────────────────────────────────────
    Encoder reads input (bidirectional)
    Decoder generates output (left-to-right)
    Decoder also attends to encoder output (cross-attention)
    
    Used for: translation, summarization
---

##### Complete Evelution - Full Picture for Intelligence

    Perceptron (1958)
    │  "I can learn simple patterns"
    │  Limitation: only linear boundaries
    │
    ▼
    Multi-Layer Perceptron (1986)
    │  "I can learn complex patterns with backpropagation"
    │  Limitation: no spatial/temporal understanding
    │
    ├────────────────────────────────┐
    ▼                                ▼
    CNN (1989)                      RNN (1986)
    "I understand SPACE"            "I understand TIME"
    slides filters over images      loops over sequences
    │                               │
    │                               │  Limitation: vanishing gradient
    │                               ▼
    │                              LSTM (1997) / GRU (2014)
    │                              "I can remember long-term"
    │                              gates control memory
    │                               │
    │                               │  Limitation: still sequential,
    │                               │  still indirect connections
    │                               ▼
    └────────────────────────────── Transformer (2017)
                                    "I see EVERYTHING at once"
                                    attention replaces recurrence
                                    parallel processing
                                    direct word-to-word connections
                                    │
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
                      BERT        GPT        T5
                      (2018)      (2018)     (2019)
                      encoder     decoder    enc-dec
                      understand  generate   both
                        │           │
                        ▼           ▼
                      Search      ChatGPT, Claude, Gemini
                      engines     Copilot, Midjourney (text part)
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect          | Detail          |
|-----------------|-----------------|
| Parameters      |                 |
| Training Time   |                 |
| Inference Time  |                 |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Example Snippet": {
        "description": "[What this code demonstrates]",
        "runnable": True,
        "code": '''# Your code here
pass
'''
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    from Required_Images.transformer_visual import  TRANSFORMER_VISUAL_HTML, TRANSFORMER_VISUAL_HEIGHT

    # Fallback static image
    transformer_img = _image_to_html(
        "Required_Images/Transformer.png",
        alt="Transformer Model Explanation",
        width="50%"
    )
    theory_with_images = THEORY.replace("{{TransformerModel_IMAGE}}", transformer_img)

    # Interactive component data
    interactive_components = [
        {
            "placeholder": "{{TransformerModel_IMAGE}}",
            "html": TRANSFORMER_VISUAL_HTML,
            "height": TRANSFORMER_VISUAL_HEIGHT,
        }
    ]

    return {
        "theory": theory_with_images,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }


