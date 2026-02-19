"""
Learning Path — AI / Generative AI Roadmap
============================================
A structured path from foundational concepts to modern LLMs.

This module serves as both a guide and the first topic in the app.
"""

TOPIC_NAME = "Learning Path: Perceptron to LLMs"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## The Road from Perceptron to Large Language Models

This application walks you through the **complete evolution of AI**, starting from
the simplest computational neuron and building all the way to modern generative
models. Each topic in the sidebar corresponds to a milestone on this journey.

---

### Phase 1 — Mathematical Foundations
Before touching any neural network, you need comfort with:
- **Linear Algebra**: vectors, matrices, dot products, matrix multiplication
- **Calculus**: derivatives, partial derivatives, chain rule
- **Probability & Statistics**: distributions, Bayes' theorem, expectation
- **Optimization**: gradient descent, learning rate, convergence

> *You don't need a PhD — you need intuition for what these tools do and why.*

---


### Phase 2 — The Perceptron & Single Neurons
The **perceptron** (Rosenblatt, 1958) is where it all begins:
- A single unit that takes inputs, multiplies by weights, adds a bias, and applies a step function
- Can learn linearly separable patterns (AND, OR) but fails on XOR
- Introduces the core loop: **forward pass → compute error → update weights**

Key concepts: *weights, bias, activation function, decision boundary, learning rule*

---

### Phase 3 — Multi-Layer Networks & Backpropagation
Stacking perceptrons into layers solves the XOR problem and much more:
- **Feedforward Neural Networks** (MLPs): input → hidden layers → output
- **Activation functions**: sigmoid, tanh, ReLU and why they matter
- **Backpropagation** (Rumelhart et al., 1986): the chain rule applied layer by layer
- **Loss functions**: MSE for regression, cross-entropy for classification

Key concepts: *hidden layers, vanishing gradients, weight initialization, epochs, batches*

---

### Phase 4 — Convolutional Neural Networks (CNNs)
Designed for spatial data (images, grids):
- **Convolution**: sliding a filter/kernel across input to detect features
- **Pooling**: reducing spatial dimensions while keeping important info
- **Architectures**: LeNet → AlexNet → VGG → ResNet → EfficientNet
- Concepts: *feature maps, stride, padding, receptive field, skip connections*

---

### Phase 5 — Recurrent Neural Networks (RNNs)
Designed for sequential data (text, time series):
- **Vanilla RNN**: hidden state passed from step to step — suffers from vanishing gradients
- **LSTM** (Hochreiter & Schmidhuber, 1997): gates (forget, input, output) to control memory
- **GRU**: simplified LSTM with fewer parameters
- **Bidirectional RNNs**: reading sequences both forward and backward

Key concepts: *hidden state, sequence-to-sequence, teacher forcing, beam search*

---

### Phase 6 — The Attention Mechanism
The bridge between RNNs and Transformers:
- **Problem**: RNNs compress entire sequences into a fixed-size vector (bottleneck)
- **Solution**: let the decoder *attend* to all encoder states, weighted by relevance
- **Bahdanau Attention** (2014): additive attention over encoder states
- **Luong Attention** (2015): dot-product variants

Key concepts: *query, key, value, attention weights, context vector*

---

### Phase 7 — The Transformer
"Attention Is All You Need" (Vaswani et al., 2017) — the architecture that changed everything:
- **Self-Attention**: every token attends to every other token in the sequence
- **Multi-Head Attention**: run attention in parallel across multiple representation subspaces
- **Positional Encoding**: inject sequence order since there's no recurrence
- **Encoder-Decoder structure**: encoder reads input, decoder generates output

Key concepts: *scaled dot-product attention, layer normalization, residual connections, feed-forward layers*

---

### Phase 8 — Pre-trained Language Models
Taking Transformers and training them on massive text corpora:
- **BERT** (2018): encoder-only, masked language modeling, bidirectional
- **GPT** (2018–2024): decoder-only, autoregressive, next-token prediction
- **T5** (2019): encoder-decoder, text-to-text framework
- **Scaling laws**: more parameters + more data = emergent capabilities

Key concepts: *pre-training, fine-tuning, tokenization (BPE, WordPiece), transfer learning*

---

### Phase 9 — Generative AI & LLMs
The current frontier:
- **In-Context Learning**: few-shot prompting without gradient updates
- **RLHF**: reinforcement learning from human feedback for alignment
- **Instruction Tuning**: training models to follow instructions
- **RAG** (Retrieval-Augmented Generation): grounding LLMs with external knowledge
- **Prompt Engineering**: crafting inputs to get optimal outputs
- **Agents & Tool Use**: LLMs that can call APIs, write code, browse the web

Key concepts: *temperature, top-k/top-p sampling, context window, hallucination, grounding*

---

### Phase 10 — Beyond Text
- **Diffusion Models**: Stable Diffusion, DALL-E — image generation via iterative denoising
- **Multimodal Models**: GPT-4V, Gemini — processing text + images + audio
- **Video Generation**: Sora, Runway — temporal coherence in generated content
- **Speech**: Whisper (ASR), TTS models
- **Mixture of Experts (MoE)**: activating only a subset of parameters per input

---

### Suggested Study Order in This App

| # | Topic | Builds On |
|---|-------|-----------|
| 1 | Perceptron | — |
| 2 | Activation Functions | Perceptron |
| 3 | Loss Functions | Activation Functions |
| 4 | Gradient Descent | Loss Functions, Calculus |
| 5 | Backpropagation | Gradient Descent |
| 6 | Feedforward Neural Networks | All above |
| 7 | CNNs | Feedforward NN |
| 8 | RNNs / LSTM / GRU | Feedforward NN |
| 9 | Attention Mechanism | RNNs |
| 10 | Transformer Architecture | Attention |
| 11 | BERT & Encoder Models | Transformer |
| 12 | GPT & Decoder Models | Transformer |
| 13 | Fine-Tuning & Transfer Learning | Pre-trained Models |
| 14 | RLHF & Alignment | GPT |
| 15 | RAG | LLMs |
| 16 | Prompt Engineering | LLMs |
| 17 | Diffusion Models | Neural Nets, Probability |
| 18 | Multimodal Models | Transformer, Vision |
| 19 | Agents & Tool Use | LLMs, RAG |

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
### Model Comparison at a Glance

| Model / Architecture | Year | Type | Params (landmark) | Key Innovation |
|----------------------|------|------|--------------------|----------------|
| Perceptron | 1958 | Single neuron | ~tens | Linear decision boundary |
| MLP + Backprop | 1986 | Feedforward | ~thousands | Multi-layer learning via chain rule |
| LeNet (CNN) | 1998 | Convolutional | ~60K | Spatial feature extraction |
| AlexNet | 2012 | CNN | ~60M | Deep CNN + GPU training + ReLU |
| LSTM | 1997 | Recurrent | ~millions | Gated memory for sequences |
| Attention (Bahdanau) | 2014 | Seq2Seq addon | — | Dynamic context vectors |
| Transformer | 2017 | Self-attention | ~65M (base) | Parallel attention, no recurrence |
| BERT | 2018 | Encoder-only | 110M / 340M | Masked LM, bidirectional |
| GPT-2 | 2019 | Decoder-only | 1.5B | Autoregressive generation at scale |
| GPT-3 | 2020 | Decoder-only | 175B | In-context learning, few-shot |
| GPT-4 | 2023 | Multimodal | ~1.8T (rumored MoE) | Multimodal, reasoning |
| Stable Diffusion | 2022 | Diffusion | ~890M | Latent-space image generation |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Perceptron — Forward Pass": {
        "description": "The simplest building block: weighted sum + step activation",
        "runnable": False,
        "code": '''

import numpy as np

class Perceptron:
    def __init__(self, n_features, lr=0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = lr

    def forward(self, x):
        """Compute output: step(w · x + b)"""
        z = np.dot(self.weights, x) + self.bias
        return 1 if z >= 0 else 0

    def train_step(self, x, y_true):
        """Perceptron learning rule"""
        y_pred = self.forward(x)
        error = y_true - y_pred
        self.weights += self.lr * error * x
        self.bias += self.lr * error
'''
    },

    "Gradient Descent — Core Loop": {
        "description": "The optimization algorithm behind all neural network training",
        "runnable": False,
        "code": '''import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=100):
    """Simple gradient descent for linear regression."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ weights + bias

        # Compute gradients
        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update parameters
        weights -= lr * dw
        bias -= lr * db

        # Track loss
        loss = np.mean((y_pred - y) ** 2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return weights, bias
'''
    },

    "Self-Attention — Scaled Dot-Product": {
        "description": "The core mechanism of the Transformer architecture",
        "runnable": False,
        "code": '''import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Scaled dot-product attention.
    Q, K, V: (seq_len, d_k)

    Returns:
        output: (seq_len, d_k)
        attention_weights: (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)     # (seq_len, seq_len)

    # Softmax to get attention weights
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum of values
    output = attention_weights @ V       # (seq_len, d_k)

    return output, attention_weights

# Example
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

out, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {out.shape}")
print(f"Attention weights:\\n{weights.round(3)}")
'''
    },

    "Tokenization — Byte-Pair Encoding (BPE) Concept": {
        "description": "How modern LLMs break text into subword tokens",
        "runnable": False,
        "code": '''def simple_bpe_demo(text, num_merges=10):
    """
    Simplified BPE demonstration.
    Real BPE (used in GPT) operates on byte-level,
    this version shows the merge logic on characters.
    """
    # Start: each character is its own token
    tokens = list(text)
    print(f"Initial tokens ({len(tokens)}): {tokens[:20]}...")

    for i in range(num_merges):
        # Count all adjacent pairs
        pairs = {}
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pairs[pair] = pairs.get(pair, 0) + 1

        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merged = best_pair[0] + best_pair[1]

        # Merge all occurrences
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                new_tokens.append(merged)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1

        tokens = new_tokens
        print(f"Merge {i+1}: {best_pair} -> '{merged}' | tokens: {len(tokens)}")

    return tokens

# Example
text = "the cat sat on the mat the cat ate the rat"
final_tokens = simple_bpe_demo(text, num_merges=8)
print(f"\\nFinal vocabulary sample: {list(set(final_tokens))[:15]}")
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
