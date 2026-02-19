"""
Activation Functions — Architecture of AI
==========================================
Comprehensive guide to activation functions in neural networks.
From the original step function to modern GELU and Swish.

Covers: Step Function, Sigmoid, ReLU, Leaky ReLU, PReLU, ELU, GELU, Swish,
        Softmax, Tanh — with theory, math, backpropagation integration,
        visual intuition, and runnable Python implementations.

"""
import math
import numpy as np
import base64
import os

TOPIC_NAME = "Activation_Function_Explanation"
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
# THEORY - CORE CONCEPTS (Original Text Preserved + Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
### What Is an Activation Function?

An activation function is a mathematical rule that each neuron applies to its weighted sum 
before passing the result to the next layer. Without activation functions, a neural network 
would just be a series of linear transformations stacked together — no matter how many layers 
you add, the whole network would collapse into a single linear function. Activation functions 
introduce **non-linearity**, which is what gives neural networks the power to learn complex 
patterns like curves, boundaries, and hierarchies.

Every neuron does the same two-step process:
1. **Compute the weighted sum**: z = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + bias
2. **Apply the activation function**: output = f(z)

The activation function `f` is the entire subject of this module. The choice of `f` determines 
how the neuron responds to its input, how gradients flow during backpropagation, and ultimately 
how well the network can learn.

---

### Why Non-Linearity Matters

Consider what happens without activation functions. Each layer computes:

    Layer 1: h₁ = W₁·x + b₁
    Layer 2: h₂ = W₂·h₁ + b₂ = W₂·(W₁·x + b₁) + b₂ = (W₂·W₁)·x + (W₂·b₁ + b₂)

This is just another linear function: h₂ = W_combined · x + b_combined

No matter how many layers you stack, without activation functions the entire network is 
equivalent to a single-layer linear transformation. You get no benefit from depth. A 100-layer 
linear network has the exact same expressive power as a 1-layer linear network.

Activation functions break this linearity. They bend, clip, squash, or reshape the output at 
each layer, allowing the network to model curves, decision boundaries, and arbitrarily complex 
functions. This is the **Universal Approximation Theorem** in action — a neural network with 
at least one hidden layer and a non-linear activation function can approximate any continuous 
function to arbitrary precision (given enough neurons).

---

### The Three Foundational Activation Functions

---

#### Step Function (Original Perceptron — 1958)

```
f(z) = 1   if z ≥ 0
f(z) = 0   if z < 0
```

Takes the weighted sum. If it's zero or positive, output 1. If it's negative, output 0. 
Binary decision. No in-between. Derivative is 0 everywhere, so backpropagation can't use it.

**Output**: 0 or 1 (binary)
**Derivative**: 0 everywhere (useless for backpropagation)
**Used in**: The original 1958 perceptron, nowhere in modern deep learning

The step function gives a clean yes/no decision, but its derivative is zero, so 
backpropagation gets no information about which direction to adjust weights. It's like 
asking "am I getting warmer or colder?" and always getting the answer "I don't know."

**Historical Context**: Frank Rosenblatt's 1958 perceptron used this function. It could 
learn linearly separable patterns (like simple OR and AND gates) but Minsky and Papert 
proved in 1969 that single-layer perceptrons with step functions couldn't solve non-linearly 
separable problems like XOR. This contributed to the first "AI Winter." The step function 
was abandoned for training purposes but the concept of thresholding lives on in the final 
decision layer of binary classifiers (just not with backpropagation through it).

---

#### Sigmoid

```
f(z) = 1 / (1 + e^(-z))
```

Takes the weighted sum. Squashes it into a smooth value between 0 and 1. 
Large positive z → output near 1. Large negative z → output near 0. z = 0 → output exactly 0.5. 
Derivative maxes out at 0.25, which causes gradients to shrink through many layers.

**Output**: 0.0 to 1.0 (smooth probability)
**Derivative**: max 0.25, usually much less
**Used in**: Output layer for binary classification, historically in hidden layers

Sigmoid is smooth and differentiable, which made backpropagation possible. But its derivative 
is always less than 1, which causes gradients to shrink through many layers. It also has a 
second problem: its outputs are always positive (between 0 and 1), which means all weights 
in the next layer receive gradients of the same sign, causing them to all increase or all 
decrease together. This slows down learning because the weights can't move independently.

**The sigmoid derivative has an elegant form**:
```
σ'(z) = σ(z) × (1 - σ(z))
```
This means you can compute the derivative directly from the output itself — no need to 
recompute the original input z. If the output is 0.5 (maximum uncertainty), the derivative 
is 0.5 × 0.5 = 0.25 (maximum sensitivity). If the output is 0.99 (very confident), the 
derivative is 0.99 × 0.01 = 0.0099 (nearly flat, resistant to change). This elegant 
relationship made sigmoid computationally convenient despite its other limitations.

---

#### ReLU — Rectified Linear Unit

```
f(z) = max(0, z)
```

Takes the weighted sum. If positive, pass it through unchanged. If negative, output 0. 
Derivative is 1 for positive inputs (gradients flow at full strength) and 0 for negative 
inputs (neuron goes silent). Fast to compute, no exponentials involved.

ReLU is an activation function. That's it. It's a rule that each neuron applies to its 
weighted sum before passing the result to the next layer. We've already discussed two 
activation functions — the step function and sigmoid. ReLU is the third, and it's the one 
that dominates modern deep learning.

The rule is almost embarrassingly simple:
    If z is positive → output z as-is
    If z is negative → output 0

Or mathematically: f(z) = max(0, z)

Everything negative gets flattened to zero. Everything positive passes through unchanged. 
That's the entire function.

**Output**: 0 to infinity (unbounded positive)
**Derivative**: 0 or 1 (clean, strong signal)
**Used in**: Hidden layers of virtually all modern deep networks

ReLU lets gradients flow without shrinking (derivative = 1 for positive inputs). It's also 
computationally trivial — just a comparison and possibly setting to zero. No exponentials, 
no division. This makes it significantly faster than sigmoid on hardware.


---

### Side by Side Comparison

| Property            | Step         | Sigmoid           | ReLU              |
|---------------------|-------------|-------------------|-------------------|
| Formula             | 0 or 1      | 1/(1+e⁻ᶻ)        | max(0, z)         |
| Output range        | {0, 1}      | (0, 1)            | [0, ∞)            |
| Derivative          | 0           | max 0.25          | 0 or 1            |
| Vanishing gradient? | Total       | Yes               | No (when active)  |
| Used where          | Nowhere modern | Output layer   | Hidden layers     |
| Speed               | Fast        | Slow (exponential)| Fastest           |

z is the weighted sum that a neuron computes before applying the activation function:
```
z = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + bias
```
{{RELU_IMAGE}}
---

### Why ReLU Replaced Sigmoid

To understand why ReLU matters, you need to understand what was wrong with sigmoid.

#### The Vanishing Gradient Problem with Sigmoid

Remember how backpropagation works: the error signal flows backward through the network, 
and at each layer it gets multiplied by the sigmoid derivative. The sigmoid derivative has 
a maximum value of 0.25 (when the output is 0.5), and it gets smaller as the output 
approaches 0 or 1.

In a 10-layer network, the gradient gets multiplied by the sigmoid derivative 10 times:

```
Layer 10 (output):  gradient = 0.20
Layer 9:            gradient = 0.20 × 0.25 = 0.05
Layer 8:            gradient = 0.05 × 0.25 = 0.0125
Layer 7:            gradient = 0.0125 × 0.25 = 0.003
Layer 6:            gradient = 0.003 × 0.25 = 0.0008
Layer 5:            gradient = 0.0008 × 0.25 = 0.0002
...
Layer 1:            gradient ≈ 0.0000001
```

By the time the error signal reaches the early layers, it's essentially zero. Those layers 
receive no useful learning signal. They stop learning. The network is 10 layers deep but 
only the last few layers are actually training. This is the vanishing gradient problem, and 
it was the main reason deep networks didn't work well for decades.

#### How ReLU Fixes This

ReLU's derivative is:
```
If z > 0 → derivative = 1
If z < 0 → derivative = 0
If z = 0 → technically undefined, in practice treated as 0
```

That derivative of 1 is the key. When a neuron is active (z > 0), the gradient passes 
through completely unchanged. No shrinking. No multiplication by 0.25. The gradient at 
layer 1 can be just as strong as the gradient at layer 10.

```
Sigmoid through 10 layers:  0.25 × 0.25 × 0.25 × ... = ~0.0000001
ReLU through 10 layers:     1 × 1 × 1 × 1 × ... = 1
```

This is why ReLU unlocked deep learning. Networks could suddenly have 50, 100, even 1000 
layers and still train successfully because the gradient didn't vanish.

---

### What ReLU Actually Does to the Data

#### The Geometric Interpretation

Each neuron in a network computes a weighted sum and then applies ReLU. The weighted sum 
defines a hyperplane (a line in 2D, a plane in 3D, etc.) that divides the input space. 
ReLU then zeroes out everything on one side of that hyperplane.

A single ReLU neuron literally says: "I care about this region of the input space. 
Everything outside that region, I output zero and ignore."

When you have many ReLU neurons in a layer, each one defines its own hyperplane and zeros 
out a different region. Together, they carve the input space into polygonal regions — like 
a stained glass window. The network can assign different behaviors to different regions.

With sigmoid, the transitions between regions are soft and gradual. With ReLU, they're 
sharp — either you're active or you're zero. These sharp transitions make ReLU networks 
surprisingly effective at approximating complex functions, and the sharpness is actually an 
advantage because it creates cleaner, more distinct feature representations.

#### Sparsity

At any given input, roughly half the ReLU neurons in a layer output zero (because roughly 
half will have negative weighted sums). This means the network's representation at each layer 
is sparse — mostly zeros with some active values. Sparse representations are efficient and 
have been shown to often produce better features. The network essentially activates different 
subsets of neurons for different inputs, creating specialized pathways without being 
explicitly told to.

---

### The Dead Neuron Problem

ReLU has one significant flaw. If a neuron's weights shift during training such that its 
weighted sum is negative for every input in the training set, it will always output zero. A 
zero output means a zero gradient, which means zero weight updates. The neuron can never 
recover. It's permanently dead.

This can happen if:
- The learning rate is too high, causing a large weight update that pushes the neuron into 
  permanently negative territory
- The initial weights are unlucky
- The bias becomes very negative

In a large network, losing a few neurons isn't catastrophic. But if too many die, the 
network's capacity shrinks and performance degrades.

---

### ReLU Variants (Fixing the Dead Neuron Problem)

#### Leaky ReLU

```
If z > 0 → output = z
If z ≤ 0 → output = 0.01 × z
```

Instead of outputting exactly 0 for negative inputs, it outputs a tiny fraction of the 
input (typically 0.01x). This means the derivative for negative inputs is 0.01 instead of 
0. The neuron can still receive gradient signal even when its input is negative, so it can 
never fully die.

The leak is small enough that it doesn't interfere with the network's behavior, but large 
enough to keep the gradient flowing.

#### Parametric ReLU (PReLU)

```
If z > 0 → output = z
If z ≤ 0 → output = α × z    (α is LEARNED during training)
```

Same as Leaky ReLU, but instead of fixing the negative slope at 0.01, the network learns 
the optimal slope as a trainable parameter. Each neuron can learn its own ideal leak amount. 
If the data benefits from more leakage, α will be larger. If less is better, α will approach 
zero (becoming regular ReLU).

#### ELU (Exponential Linear Unit)

```
If z > 0 → output = z
If z ≤ 0 → output = α × (e^z - 1)
```

For negative inputs, ELU smoothly curves toward -α instead of going linearly negative. This 
produces outputs that are centered closer to zero on average (unlike ReLU, which has a mean 
greater than zero), which helps with training dynamics. The smooth curve also means the 
function is differentiable everywhere, including at z = 0.

#### GELU (Gaussian Error Linear Unit)

```
GELU(z) = z × Φ(z)    where Φ is the standard normal cumulative distribution
```

GELU doesn't have a hard cutoff at zero. Instead, it smoothly transitions, with the 
probability of passing the input through being based on how large it is. Small negative 
values get slightly reduced. Large negative values get pushed toward zero. Large positive 
values pass through nearly unchanged.

This is the activation function used in GPT, BERT, and most modern transformers. It was 
chosen because the smooth, probabilistic gating produces slightly better results than hard 
ReLU on language tasks.

**Why GELU works well for transformers**: In attention-based architectures, the smooth 
gating acts as a form of stochastic regularization. Rather than a hard on/off decision 
at zero, GELU gives inputs near zero a probabilistic chance of passing through. This 
nuanced behavior helps transformers maintain richer gradient signals in the dense 
feed-forward layers that sit between attention blocks.

#### Swish

```
Swish(z) = z × sigmoid(z)
```

Discovered through automated search by Google researchers. Like GELU, it's smooth and 
non-monotonic (it dips slightly below zero for small negative inputs before rising back). 
It has been shown to outperform ReLU in deeper networks.

**Why Swish was discovered by machines, not humans**: Google Brain used automated architecture 
search to test thousands of candidate activation functions. Swish (originally called SiL — 
Sigmoid Linear Unit) emerged as the best performer across multiple tasks. It's a reminder 
that not all good ideas in deep learning come from human intuition — sometimes brute-force 
search finds solutions humans wouldn't have designed.

---

### Two Additional Activation Functions Worth Knowing

#### Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Tanh is essentially a rescaled sigmoid: tanh(z) = 2·sigmoid(2z) - 1

**Output**: -1.0 to 1.0 (zero-centered)
**Derivative**: max 1.0 (at z = 0), approaches 0 at extremes
**Used in**: RNN/LSTM hidden states, some normalization contexts

Tanh fixes sigmoid's non-zero-centered problem — its outputs range from -1 to 1, meaning 
the mean output is closer to zero. This allows gradients to flow in both positive and 
negative directions, enabling weights to update independently. However, tanh still suffers 
from vanishing gradients at extreme values (just less severely than sigmoid because its 
derivative peaks at 1.0 instead of 0.25).

Tanh remains important in LSTM and GRU gating mechanisms, where its zero-centered output 
is used to modulate information flow. In those architectures, a tanh output of -1 means 
"strongly negative signal," 0 means "no signal," and 1 means "strongly positive signal."

#### Softmax (Multi-Class Output)

```
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```

Softmax is not applied per-neuron like the others. It's applied across an entire layer's 
outputs, converting a vector of raw scores into a probability distribution that sums to 1.

**Output**: (0, 1) for each element, and all elements sum to 1.0
**Used in**: Output layer for multi-class classification (choosing 1 of N categories)
**Relationship**: Sigmoid is actually a special case of softmax for 2 classes

If a network needs to classify an image as cat, dog, or bird, the output layer has 3 neurons 
producing raw scores (logits) like [2.0, 1.0, 0.5]. Softmax converts these to probabilities 
like [0.59, 0.24, 0.17]. The highest probability is the network's prediction. Softmax 
amplifies differences — the largest logit gets a disproportionately large probability — while 
ensuring the outputs form a valid probability distribution.

**Temperature scaling**: Softmax is often modified with a temperature parameter T:
softmax(zᵢ/T). Higher T → more uniform distribution (softer choices). Lower T → more 
peaked distribution (harder choices). T → 0 approaches argmax (hard selection). This is 
used in language models to control creativity vs. determinism in text generation.

---

### Which One Should You Use?

For most practical purposes:

**ReLU** — the default starting point. Use it unless you have a reason not to. It's fast, 
well-understood, and works well for most architectures (CNNs, standard feedforward networks). 
If you're just getting started, use ReLU.

**Leaky ReLU** — use when you're experiencing dead neurons. It's a safe drop-in replacement 
for ReLU with minimal overhead.

**GELU** — use for transformer architectures and NLP tasks. This is what the major language 
models use.

**Sigmoid** — only at the output layer for binary classification (to produce a probability). 
Almost never in hidden layers anymore.

**Tanh** — use in LSTM/GRU gates and when you need zero-centered outputs. Rarely as a 
general hidden layer activation.

**Softmax** — output layer for multi-class classification. Converts logits to probabilities.

**Swish/ELU** — worth experimenting with for very deep networks if ReLU isn't performing well.

---

### The Bigger Picture

ReLU's importance goes beyond being "just" an activation function. It was one of the key 
innovations that made deep learning practical. Before ReLU, training networks deeper than 
3-4 layers was extremely difficult because of vanishing gradients. After ReLU, researchers 
could train 10, 50, 100+ layer networks. This depth is what gave neural networks the capacity 
to learn the complex hierarchical representations needed for tasks like image recognition, 
language understanding, and game playing.

The jump from sigmoid to ReLU wasn't just a minor technical improvement. It was one of the 
unlocks that triggered the deep learning revolution starting around 2012, alongside larger 
datasets (ImageNet), better hardware (GPUs), and improved initialization (He initialization — 
which, not coincidentally, was specifically designed for ReLU networks).

---

### How Activation Functions Integrate with Backpropagation

This section connects activation functions to the weight update mechanism. Understanding this 
connection is essential — the activation function's derivative is the bridge between "how 
wrong was the output?" and "how should each weight change?"

#### The Rule: Every Adjustment Answers One Question

"How much did THIS specific weight contribute to the final error?"

If a weight contributed a lot to the mistake → big adjustment. If it barely contributed → 
tiny adjustment. If it actually helped → adjust it the opposite direction. The calculation 
figures out exactly how much blame each weight deserves.

#### How Blame Is Calculated: Three Factors

Every weight update is the product of exactly three things:
```
adjustment = learning_rate × delta × input_to_this_neuron
```

**Factor 1 — Learning Rate** — set by you. Controls the overall step size. Same for every 
weight. Think of this as "how cautious are we being."

**Factor 2 — Delta** — calculated by the network. This is the heart of it. It answers "how 
much was the neuron that this weight feeds into responsible for the error?" This is different 
for every neuron and changes every training step.

**Factor 3 — Input** — the value that flowed through this specific connection during the 
forward pass. If the input was large, this weight had a big influence on the output, so it 
gets a bigger adjustment. If the input was zero, this weight had no influence, so the 
adjustment is zero. You don't adjust something that had no effect.

#### How Delta Itself Is Calculated

Delta is calculated differently depending on where the neuron sits in the network.

**For an Output Neuron** (directly produces the prediction):
```
delta = (expected - actual) × activation_derivative(actual)
```
This has two parts:
- **(expected - actual)** — the raw error. Was the output too high or too low, and by how much?
- **activation_derivative(actual)** — how sensitive is this neuron right now? This is where the 
  activation function directly enters the equation. For sigmoid, this is σ(z)×(1-σ(z)). For 
  ReLU, this is simply 1 (if active) or 0 (if dead).

**For a Hidden Neuron** (doesn't directly produce the output):
```
delta = (sum of blame from next layer) × activation_derivative(this neuron's output)
```
The hidden neuron doesn't know what the "right answer" was for itself. So instead of using 
(expected - actual), it receives blame from the neurons it feeds into:
```
blame = Σ (next_neuron_delta × weight_connecting_them)
```
The logic: if the next neuron had a large delta (it was very wrong) and the weight connecting 
them is large (this hidden neuron had a strong influence on it), then this hidden neuron 
deserves a lot of blame.

This blame flows backward through every layer — this is backpropagation.

#### A Concrete Trace

Let's say we have a simple network: 1 hidden neuron → 1 output neuron.

State after forward pass:
```
Input to hidden: 0.8
Hidden neuron output: 0.6
Hidden-to-output weight: 0.7
Output neuron output: 0.65
Expected: 1.0
```

**Step 1: Output neuron delta**
```
error = 1.0 - 0.65 = 0.35
sigmoid_derivative(0.65) = 0.65 × (1 - 0.65) = 0.2275
delta_output = 0.35 × 0.2275 = 0.0796
```

**Step 2: Hidden neuron delta (blame flows backward)**
```
blame = delta_output × connecting_weight = 0.0796 × 0.7 = 0.0557
sigmoid_derivative(0.6) = 0.6 × (1 - 0.6) = 0.24
delta_hidden = 0.0557 × 0.24 = 0.0134
```

**Step 3: Update the hidden-to-output weight**
```
adjustment = learning_rate × delta_output × input_to_output_neuron
adjustment = 0.1 × 0.0796 × 0.6 = 0.00478
new_weight = 0.7 + 0.00478 = 0.70478
```

**Step 4: Update the input-to-hidden weight**
```
adjustment = learning_rate × delta_hidden × input_to_hidden_neuron
adjustment = 0.1 × 0.0134 × 0.8 = 0.00107
new_weight = old_weight + 0.00107
```

Nothing is random. Every number is deterministically computed from the error, the current 
weights, the current activations, and the derivatives.

#### Why This Works

The mathematical foundation is the **chain rule from calculus**. It says: if A influences B, 
and B influences C, then you can calculate how A influences C by multiplying "how A 
influences B" by "how B influences C."

In a neural network:
```
weight in layer 1 → affects → neuron in layer 1 → affects → neuron in layer 2 → affects → output → affects → loss
```

The chain rule lets us multiply through this entire chain to get: "how does this specific 
weight in layer 1 affect the final loss?" That's exactly what the delta × input calculation 
computes.

So every adjustment is:
- **Directed** — it knows whether to increase or decrease (from the sign of the delta)
- **Proportional** — weights that caused more error get adjusted more (from the magnitude of the delta)
- **Input-aware** — connections that carried strong signals get adjusted more than connections that carried weak signals
- **Sensitivity-aware** — neurons that are in a sensitive state (uncertain) get adjusted more than neurons that are saturated (confident)

There is no randomness, no trial and error, no guessing. Every adjustment is the 
mathematically optimal direction to reduce the error by the largest amount for the smallest 
change in weights. That's what gradient descent means — follow the steepest downhill path.

#### How the Activation Function Choice Changes the Backward Pass

Here's the same backpropagation trace but comparing what happens when you swap the 
activation function:

**With Sigmoid** (δ = error × sigmoid_derivative):
```
Layer 10 delta: 0.35 × 0.2275 = 0.0796
Layer 5 delta:  already shrunk to ~0.0002  (each layer multiplies by ≤ 0.25)
Layer 1 delta:  ~0.0000001                 (effectively zero — learning stopped)
```

**With ReLU** (δ = error × relu_derivative, which is 1 for active neurons):
```
Layer 10 delta: 0.35 × 1.0 = 0.35
Layer 5 delta:  0.35 still propagating at full strength (each layer multiplies by 1)
Layer 1 delta:  0.35 × (product of weights)  (gradient maintains meaningful magnitude)
```

This is the same math shown earlier in the vanishing gradient section, but now you can 
see exactly WHERE in the backpropagation calculation the activation function's derivative 
appears — it's the second factor in every delta computation. That's why a derivative of 
0.25 (sigmoid) vs 1.0 (ReLU) makes such a dramatic difference over many layers.





"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
### Complete Activation Function Reference

| Function    | Formula                      | Output Range | Derivative (Max)      | Vanishing Gradient? | Differentiable at 0? | Compute Cost |
|-------------|------------------------------|--------------|----------------------|---------------------|-----------------------|-------------|
| Step        | 0 or 1                       | {0, 1}       | 0 everywhere         | Total               | No                    | Trivial     |
| Sigmoid     | 1/(1+e⁻ᶻ)                     | (0, 1)       | 0.25                 | Yes                 | Yes                   | Moderate    |
| Tanh        | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)              | (-1, 1)       | 1.0                 | Yes (less severe)   | Yes                   | Moderate    |
| ReLU        | max(0, z)                    | [0, ∞)       | 1                    | No (when active)    | No                    | Trivial     |
| Leaky ReLU  | max(0.01z, z)                | (-∞, ∞)      | 1                    | No                  | No                    | Trivial     |
| PReLU       | max(αz, z)                   | (-∞, ∞)      | 1                    | No                  | No                    | Trivial     |
| ELU         | z if z>0, α(eᶻ-1) if z≤0     | (-α, ∞)       | 1                   | No                  | Yes                   | Moderate    |
| GELU        | z·Φ(z)                       | ≈(-0.17, ∞)  | ≈1.08                | No                  | Yes                   | Moderate    |
| Swish       | z·σ(z)                       | ≈(-0.28, ∞)  | ≈1.10                | No                  | Yes                   | Moderate    |
| Softmax     | eᶻⁱ / Σeᶻʲ                    | (0, 1) sum=1  | Varies               | No                  | Yes                   | Moderate    |

### Derivative Quick Reference

| Function    | Derivative                                   | Key Property                                |
|-------------|----------------------------------------------|---------------------------------------------|
| Step        | 0                                            | No gradient — cannot learn                  |
| Sigmoid     | σ(z) × (1 - σ(z))                            | Max 0.25 — gradients shrink                 |
| Tanh        | 1 - tanh²(z)                                 | Max 1.0 — better than sigmoid               |
| ReLU        | 1 if z > 0, else 0                           | Perfect passthrough when active             |
| Leaky ReLU  | 1 if z > 0, else 0.01                        | Always some gradient — no dead neurons      |
| PReLU       | 1 if z > 0, else α (learned)                 | Adaptive leak — network decides             |
| ELU         | 1 if z > 0, else α·eᶻ                         | Smooth transition at z = 0                 |
| GELU        | Φ(z) + z·φ(z)                                | Smooth, probabilistic gating                |
| Swish       | σ(z) + z·σ(z)·(1-σ(z)) = swish(z) + σ(z)·(1-swish(z)) | Non-monotonic, smooth              |

### When to Use Which — Architecture Guide

| Architecture          | Recommended Activation        | Why                                           |
|-----------------------|-------------------------------|-----------------------------------------------|
| Feedforward (MLP)     | ReLU / Leaky ReLU             | Fast, effective, well-understood              |
| CNN (ConvNets)        | ReLU                          | Sparsity helps feature detection              |
| Transformer           | GELU                          | Smooth gating suits attention mechanism       |
| LSTM / GRU            | Tanh + Sigmoid                | Tanh for state, Sigmoid for gates             |
| Binary Output         | Sigmoid                       | Produces probability in [0, 1]                |
| Multi-class Output    | Softmax                       | Produces probability distribution summing to 1|
| GAN Generator         | ReLU (hidden) + Tanh (output) | Tanh output maps to [-1, 1] image range       |
| GAN Discriminator     | Leaky ReLU                    | Prevents dead neurons with sparse gradients   |
| Very Deep Networks    | Swish / ELU                   | Better gradient flow in 100+ layer networks   |
| ResNets               | ReLU                          | Works great with skip connections             |
"""


# =============================================================================
# SECTIONS
# =============================================================================

SECTIONS = {
    "📐 Foundational Activation Functions": [
        "Step Function",
        "Sigmoid",
        "Tanh",
    ],
    "⚡ ReLU Family": [
        "ReLU",
        "Leaky ReLU",
        "Parametric ReLU (PReLU)",
        "ELU",
    ],
    "🧠 Modern Activations (Transformer Era)": [
        "GELU",
        "Swish",
    ],
    "🎯 Output Layer Activations": [
        "Sigmoid (Binary Classification)",
        "Softmax (Multi-Class Classification)",
    ],
    "🔬 Demonstrations & Comparisons": [
        "Vanishing Gradient Demo",
        "Dead Neuron Demo",
        "All Activations Visual Comparison",
        "Backpropagation with Different Activations",
    ],
    "🧪 Complete Network Simulations": [
        "Forward and Backward Pass (Sigmoid)",
        "Forward and Backward Pass (ReLU)",
        "Multi-Layer Gradient Flow Comparison",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # =========================================================================
    # FOUNDATIONAL ACTIVATION FUNCTIONS
    # =========================================================================

    "Step Function": {
        "description": "The original perceptron activation (1958). Binary output, zero derivative — cannot be used with backpropagation.",
        "code":
            '''import numpy as np

def step_function(z):
    """
    Step activation function.
    f(z) = 1 if z >= 0, else 0

    Used in: Original Perceptron (1958)
    Output range: {0, 1}
    Derivative: 0 everywhere (useless for gradient-based learning)
    """
    return np.where(z >= 0, 1, 0)

def step_derivative(z):
    """
    Derivative of step function.
    Always 0 — this is WHY the original perceptron couldn't use 
    gradient descent. Backpropagation was impossible.
    """
    return np.zeros_like(z)

# ── Demo ──────────────────────────────────────────────────
z_values = np.array([-3, -1, -0.01, 0, 0.01, 1, 3])

print("Step Function Demo")
print("=" * 50)
print(f"{'z':>8} | {'f(z)':>6} | {'f\\'(z)':>6}")
print("-" * 30)
for z in z_values:
    print(f"{z:8.2f} | {step_function(z):6.0f} | {step_derivative(z):6.0f}")

print()
print("Key insight: derivative is 0 everywhere.")
print("Backpropagation asks 'which direction should I adjust?'")
print("Step function always answers: 'I don't know' — learning is impossible.")            '''
    },

    "Sigmoid": {
        "description": "Smooth probability output between 0 and 1. Made backpropagation possible but causes vanishing gradients in deep networks.",
        "code":
            '''import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    f(z) = 1 / (1 + e^(-z))

    Used in: Output layer for binary classification
    Output range: (0, 1) — interpretable as probability
    Derivative: σ(z) × (1 - σ(z)), max = 0.25

    Problems:
      1. Vanishing gradient: derivative ≤ 0.25, shrinks through layers
      2. Not zero-centered: outputs always positive, causes zig-zag updates
      3. Expensive: requires exponential computation
    """
    # Numerically stable sigmoid
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def sigmoid_derivative(z):
    """
    Elegant property: derivative computable from output alone.
    σ'(z) = σ(z) × (1 - σ(z))

    Maximum value: 0.25 (when σ(z) = 0.5, i.e., z = 0)
    This max of 0.25 is WHY gradients vanish in deep sigmoid networks.
    """
    s = sigmoid(z)
    return s * (1 - s)

# ── Demo ──────────────────────────────────────────────────
z_values = np.array([-5, -2, -1, 0, 1, 2, 5])

print("Sigmoid Function Demo")
print("=" * 55)
print(f"{'z':>6} | {'σ(z)':>8} | {'σ\\'(z)':>8} | Interpretation")
print("-" * 55)
for z in z_values:
    s = sigmoid(z)
    d = sigmoid_derivative(z)
    if s > 0.8:
        interp = "Confident YES"
    elif s < 0.2:
        interp = "Confident NO"
    else:
        interp = "Uncertain"
    print(f"{z:6.1f} | {s:8.4f} | {d:8.4f} | {interp}")

print()
print("Notice: derivative peaks at 0.25 when z=0 (maximum uncertainty)")
print("         derivative → 0 when output is near 0 or 1 (saturated)")
print()

# Show vanishing gradient across layers
print("Vanishing Gradient Through 10 Layers (best case: derivative = 0.25)")
print("-" * 50)
gradient = 1.0
for layer in range(1, 11):
    gradient *= 0.25
    bar = "█" * max(1, int(gradient * 200))
    print(f"Layer {layer:2d}: gradient = {gradient:.10f}  {bar}")            '''
    },

    "Tanh": {
        "description": "Zero-centered sigmoid variant. Output range [-1, 1]. Used in LSTM/GRU gates.",
        "code":
            '''import numpy as np

def tanh(z):
    """
    Hyperbolic Tangent activation function.
    f(z) = (e^z - e^(-z)) / (e^z + e^(-z))

    Equivalent to: 2 × sigmoid(2z) - 1

    Used in: LSTM/GRU hidden states, some normalization layers
    Output range: (-1, 1) — zero-centered (advantage over sigmoid)
    Derivative: 1 - tanh²(z), max = 1.0 at z = 0

    Advantages over sigmoid:
      - Zero-centered output → gradients can be positive or negative
      - Stronger gradients (max derivative = 1.0 vs sigmoid's 0.25)

    Still has vanishing gradients for extreme values, just less severely.
    """
    return np.tanh(z)

def tanh_derivative(z):
    """
    tanh'(z) = 1 - tanh²(z)

    Like sigmoid, computable from output alone.
    Max value: 1.0 (at z = 0) — 4× stronger than sigmoid's max.
    Still approaches 0 for large |z| → vanishing gradient persists.
    """
    t = np.tanh(z)
    return 1 - t ** 2

# ── Demo ──────────────────────────────────────────────────
z_values = np.array([-5, -2, -1, 0, 1, 2, 5])

print("Tanh vs Sigmoid Comparison")
print("=" * 70)
print(f"{'z':>6} | {'tanh(z)':>8} | {'tanh\\'(z)':>9} | {'σ(z)':>8} | {'σ\\'(z)':>8}")
print("-" * 70)

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

for z in z_values:
    t = tanh(z)
    td = tanh_derivative(z)
    s = sigmoid(z)
    sd = s * (1 - s)
    print(f"{z:6.1f} | {t:8.4f} | {td:9.4f}   | {s:8.4f} | {sd:8.4f}")

print()
print("Key differences:")
print("  Tanh output is zero-centered: range (-1, 1) vs sigmoid (0, 1)")
print("  Tanh derivative peaks at 1.0 vs sigmoid's 0.25")
print("  Both still vanish at extremes, but tanh is 4× stronger")
print()
print("In LSTM/GRU:")
print("  tanh → hidden state modulation (needs +/- range)")
print("  sigmoid → gate values (needs 0-to-1 probability)")            '''
    },

    # =========================================================================
    # RELU FAMILY
    # =========================================================================

    "ReLU": {
        "description": "The activation that unlocked deep learning. max(0, z). Fast, effective, dominant in modern networks.",
        "code":
            '''import numpy as np

def relu(z):
    """
    Rectified Linear Unit.
    f(z) = max(0, z)

    If positive → pass through unchanged
    If negative → output 0

    Used in: Hidden layers of virtually all modern deep networks
    Output range: [0, ∞) — unbounded positive
    Derivative: 1 (active) or 0 (dead)

    Why it dominates:
      1. Solves vanishing gradient: derivative = 1 for active neurons
      2. Computationally trivial: just a comparison, no exponentials
      3. Induces sparsity: ~50% of neurons output zero
      4. Biological plausibility: neurons either fire or don't
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    ReLU derivative:
      z > 0 → 1  (gradient passes through at full strength)
      z < 0 → 0  (gradient is blocked — neuron contributes nothing)
      z = 0 → technically undefined, treated as 0 in practice

    This derivative of 1 is THE reason deep learning works.
    Through 100 layers: 1 × 1 × 1 × ... = 1 (no vanishing!)
    """
    return np.where(z > 0, 1.0, 0.0)

# ── Demo ──────────────────────────────────────────────────
z_values = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])

print("ReLU Function Demo")
print("=" * 50)
print(f"{'z':>6} | {'ReLU(z)':>8} | {'ReLU\\'(z)':>9} | Status")
print("-" * 50)
for z in z_values:
    r = relu(z)
    d = relu_derivative(z)
    status = "ACTIVE (gradient flows)" if z > 0 else "DEAD (gradient blocked)"
    print(f"{z:6.1f} | {r:8.2f} | {d:9.1f}   | {status}")

print()
print("Gradient Flow Comparison Through 10 Layers:")
print("-" * 50)
sigmoid_grad = 1.0
relu_grad = 1.0
for layer in range(1, 11):
    sigmoid_grad *= 0.25
    relu_grad *= 1.0  # assuming active neurons
    print(f"Layer {layer:2d}:  Sigmoid = {sigmoid_grad:.10f}  |  ReLU = {relu_grad:.1f}")

print()
print("After 10 layers:")
print(f"  Sigmoid gradient: {0.25**10:.10f} (essentially dead)")
print(f"  ReLU gradient:    {1.0:.1f} (full strength)")

# Sparsity demo
print()
random_inputs = np.random.randn(1000)
outputs = relu(random_inputs)
zero_count = np.sum(outputs == 0)
print(f"Sparsity: {zero_count}/1000 neurons output zero ({zero_count/10:.1f}%)")
print("This natural sparsity is actually a feature, not a bug.")            '''
    },

    "Leaky ReLU": {
        "description": "ReLU variant that prevents dead neurons by allowing a small negative slope (0.01).",
        "code":
            '''import numpy as np

def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU — fixes the dead neuron problem.
    f(z) = z      if z > 0
    f(z) = α × z  if z ≤ 0    (typically α = 0.01)

    Instead of outputting exactly 0 for negative inputs, it outputs 
    a tiny fraction (0.01x). This keeps the gradient flowing:
      - Derivative for z > 0: 1 (same as ReLU)
      - Derivative for z ≤ 0: α = 0.01 (small but non-zero!)

    The leak is small enough to not interfere with behavior, 
    but large enough to prevent neurons from dying permanently.
    """
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    """
    Derivative:
      z > 0 → 1
      z ≤ 0 → α (small but non-zero — neuron can still learn!)
    """
    return np.where(z > 0, 1.0, alpha)

# ── Demo: ReLU vs Leaky ReLU ─────────────────────────────
z_values = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])

print("ReLU vs Leaky ReLU Comparison")
print("=" * 70)
print(f"{'z':>6} | {'ReLU':>6} | {'ReLU\\'':>6} | {'Leaky':>8} | {'Leaky\\'':>7} | Dead?")
print("-" * 70)

def relu(z): return np.maximum(0, z)
def relu_d(z): return np.where(z > 0, 1.0, 0.0)

for z in z_values:
    r = relu(z)
    rd = relu_d(z)
    lr = leaky_relu(z)
    lrd = leaky_relu_derivative(z)
    dead = "DEAD!" if z < 0 and rd == 0 else ("LEAK" if z < 0 else "OK")
    print(f"{z:6.1f} | {r:6.2f} | {rd:6.2f} | {lr:8.4f} | {lrd:7.2f} | {dead}")

print()
print("Key insight: For z = -5:")
print(f"  ReLU output: {relu(-5.0):.2f}, derivative: {relu_d(-5.0):.2f} → completely dead")
print(f"  Leaky output: {leaky_relu(-5.0):.4f}, derivative: {leaky_relu_derivative(-5.0):.2f} → still learning!")            '''
    },

    "Parametric ReLU (PReLU)": {
        "description": "Like Leaky ReLU, but the negative slope α is a learnable parameter — the network decides the optimal leak.",
        "code":
            '''import numpy as np

class PReLU:
    """
    Parametric ReLU — network LEARNS the optimal leak.
    f(z) = z      if z > 0
    f(z) = α × z  if z ≤ 0    (α is a trainable parameter)

    Key difference from Leaky ReLU:
      - Leaky ReLU: α is fixed at 0.01 (a hyperparameter YOU choose)
      - PReLU: α starts at some value and is LEARNED during training

    Each neuron can learn its own α. Some might learn α ≈ 0 (becoming 
    regular ReLU). Others might learn α ≈ 0.2 (significant negative slope).
    The data decides what's best.
    """
    def __init__(self, alpha_init=0.25):
        self.alpha = alpha_init  # learnable parameter

    def forward(self, z):
        """Forward pass: compute PReLU output."""
        self.z = z  # save for backward pass
        return np.where(z > 0, z, self.alpha * z)

    def derivative(self, z):
        """Derivative with respect to z (for backprop through layers)."""
        return np.where(z > 0, 1.0, self.alpha)

    def alpha_gradient(self, z):
        """
        Gradient with respect to α (for learning the leak).
        dL/dα = dL/dout × dout/dα
        dout/dα = z  (when z ≤ 0, since output = α × z)
        dout/dα = 0  (when z > 0, since output = z, no α involved)
        """
        return np.where(z > 0, 0, z)

    def update_alpha(self, z, upstream_gradient, learning_rate=0.01):
        """Update α using gradient descent, just like any other parameter."""
        grad = np.mean(upstream_gradient * self.alpha_gradient(z))
        self.alpha -= learning_rate * grad

# ── Demo ──────────────────────────────────────────────────
prelu = PReLU(alpha_init=0.25)
z_values = np.array([-3, -1, 0, 1, 3])

print("PReLU Demo (α = 0.25)")
print("=" * 55)
print(f"{'z':>6} | {'output':>8} | {'dz':>6} | {'dα':>6}")
print("-" * 35)
for z in z_values:
    out = prelu.forward(z)
    dz = prelu.derivative(z)
    da = prelu.alpha_gradient(z)
    print(f"{z:6.1f} | {out:8.4f} | {dz:6.2f} | {da:6.2f}")

print()
print("PReLU adapts: if the network benefits from more leak, α grows.")
print("If less leak is better, α shrinks toward 0 (becoming standard ReLU).")
print("The data decides — you don't have to pick the right hyperparameter.")            '''
    },

    "ELU": {
        "description": "Exponential Linear Unit. Smooth curve for negative inputs, zero-centered outputs, differentiable everywhere.",
        "code":
            '''import numpy as np

def elu(z, alpha=1.0):
    """
    Exponential Linear Unit.
    f(z) = z                if z > 0
    f(z) = α × (e^z - 1)   if z ≤ 0

    Key properties:
      - Smooth curve for negatives (not a hard kink like Leaky ReLU)
      - Approaches -α for very negative z (bounded below)
      - Zero-centered outputs (mean closer to 0 than ReLU)
      - Differentiable everywhere, including z = 0
      - More expensive than ReLU (exponential computation for negatives)
    """
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_derivative(z, alpha=1.0):
    """
    ELU derivative:
      z > 0 → 1
      z ≤ 0 → α × e^z = f(z) + α  (smooth, never exactly zero)

    At z = 0: left derivative = α, right derivative = 1
    When α = 1, these match → fully smooth transition.
    """
    return np.where(z > 0, 1.0, alpha * np.exp(z))

# ── Demo ──────────────────────────────────────────────────
z_values = np.array([-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5])

print("ELU Demo (α = 1.0)")
print("=" * 55)
print(f"{'z':>6} | {'ELU(z)':>8} | {'ELU\\'(z)':>8} | Note")
print("-" * 55)
for z in z_values:
    e = elu(z)
    d = elu_derivative(z)
    note = ""
    if z < -3: note = "saturates near -α"
    elif z < 0: note = "smooth exponential curve"
    elif z == 0: note = "smooth transition point"
    else: note = "linear (same as ReLU)"
    print(f"{z:6.1f} | {e:8.4f} | {d:8.4f} | {note}")

print()
print("ELU vs ReLU for zero-centering:")
random_z = np.random.randn(10000)
relu_mean = np.mean(np.maximum(0, random_z))
elu_mean = np.mean(elu(random_z))
print(f"  Mean ReLU output: {relu_mean:.4f}  (biased positive)")
print(f"  Mean ELU output:  {elu_mean:.4f}  (closer to zero)")            '''
    },

    # =========================================================================
    # MODERN ACTIVATIONS
    # =========================================================================

    "GELU": {
        "description": "Gaussian Error Linear Unit — the activation behind GPT, BERT, and modern transformers. Smooth probabilistic gating.",
        "code":
            '''import numpy as np
from scipy import special  # for erf (error function)

def gelu_exact(z):
    """
    GELU — Gaussian Error Linear Unit (exact form).
    GELU(z) = z × Φ(z)
    where Φ is the standard normal CDF.

    Equivalent to: z × 0.5 × (1 + erf(z / √2))

    Intuition: Instead of a hard 0/1 gate (ReLU), GELU applies a 
    PROBABILISTIC gate. The probability of letting z through is 
    based on how large z is relative to other inputs.

      - Large positive z → Φ(z) ≈ 1 → passes through fully
      - Large negative z → Φ(z) ≈ 0 → blocked (like ReLU)
      - Near zero → Φ(z) ≈ 0.5 → half-passed (smooth transition!)

    Used in: GPT, BERT, ViT, and most modern transformers.
    """
    return z * 0.5 * (1 + special.erf(z / np.sqrt(2)))

def gelu_approx(z):
    """
    Fast approximation used in practice (avoids computing erf).
    GELU(z) ≈ 0.5 × z × (1 + tanh(√(2/π) × (z + 0.044715 × z³)))

    This is what most frameworks actually compute — it's faster 
    and the error is negligible.
    """
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

def gelu_derivative_approx(z):
    """
    GELU derivative (using the tanh approximation).
    """
    tanh_arg = np.sqrt(2/np.pi) * (z + 0.044715 * z**3)
    tanh_val = np.tanh(tanh_arg)
    sech2 = 1 - tanh_val**2
    dtanh = np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * z**2)
    return 0.5 * (1 + tanh_val) + 0.5 * z * sech2 * dtanh

# ── Demo: GELU vs ReLU ───────────────────────────────────
z_values = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])

print("GELU vs ReLU Comparison")
print("=" * 65)
print(f"{'z':>6} | {'GELU':>8} | {'ReLU':>6} | {'Diff':>8} | Observation")
print("-" * 65)

for z in z_values:
    g = gelu_approx(z)
    r = max(0, z)
    diff = g - r
    obs = ""
    if z < -1: obs = "both ≈ 0"
    elif z < 0: obs = "GELU slightly negative (soft gate)"
    elif z == 0: obs = "GELU = 0, ReLU = 0"
    elif z < 1: obs = "GELU slightly less than ReLU"
    else: obs = "nearly identical"
    print(f"{z:6.1f} | {g:8.4f} | {r:6.2f} | {diff:8.4f} | {obs}")

print()
print("Key insight: GELU is smoother than ReLU around z = 0.")
print("Small negative values get slightly reduced (not hard-zeroed).")
print("This probabilistic gating helps transformers maintain richer gradients.")
print()

# Verify approximation accuracy
if 'special' in dir():
    max_error = np.max(np.abs(gelu_exact(z_values) - gelu_approx(z_values)))
    print(f"Max approximation error: {max_error:.8f} (negligible)")            '''
    },

    "Swish": {
        "description": "Discovered by Google's automated search. Smooth, non-monotonic. Outperforms ReLU in very deep networks.",
        "code":
            '''import numpy as np

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def swish(z, beta=1.0):
    """
    Swish activation — discovered by machine search, not human intuition.
    Swish(z) = z × σ(βz)

    When β = 1 (default): Swish(z) = z × sigmoid(z)
    When β → ∞: approaches ReLU
    When β = 0: becomes linear function f(z) = z/2

    Properties:
      - Smooth everywhere (no kink at z = 0)
      - Non-monotonic: dips slightly below 0 for small negatives
      - Unbounded above, bounded below (≈ -0.28 at minimum)
      - Self-gated: the input z gates itself through sigmoid

    Discovered by Google Brain using automated architecture search 
    over thousands of candidate activation functions. It consistently 
    outperformed ReLU on ImageNet and other benchmarks.
    """
    return z * sigmoid(beta * z)

def swish_derivative(z, beta=1.0):
    """
    Swish'(z) = σ(βz) + βz × σ(βz) × (1 - σ(βz))
             = σ(βz) × (1 + βz × (1 - σ(βz)))
             = swish(z)/z + σ(βz) × (1 - swish(z)/z)  [for z ≠ 0]
    """
    s = sigmoid(beta * z)
    return s + beta * z * s * (1 - s)

# ── Demo ──────────────────────────────────────────────────
z_values = np.linspace(-5, 5, 21)

print("Swish Function Demo")
print("=" * 55)
print(f"{'z':>6} | {'Swish':>8} | {'ReLU':>6} | {'Swish\\'':>8}")
print("-" * 40)
for z in z_values:
    sw = swish(z)
    r = max(0, z)
    sd = swish_derivative(z)
    print(f"{z:6.1f} | {sw:8.4f} | {r:6.2f} | {sd:8.4f}")

print()
# Find the non-monotonic dip
z_neg = np.linspace(-2, 0, 100)
sw_neg = swish(z_neg)
min_val = np.min(sw_neg)
min_z = z_neg[np.argmin(sw_neg)]
print(f"Non-monotonic dip: Swish({min_z:.2f}) = {min_val:.4f}")
print("This small dip below zero for negative inputs is unique to Swish.")
print("It was NOT designed by humans — automated search found it optimal.")            '''
    },

    # =========================================================================
    # OUTPUT LAYER ACTIVATIONS
    # =========================================================================

    "Sigmoid (Binary Classification)": {
        "description": "Sigmoid at the output layer: converts any real number to a probability between 0 and 1 for binary yes/no decisions.",
        "code":
            '''import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def binary_classification_output(logit, threshold=0.5):
    """
    Complete binary classification output pipeline.

    1. Network produces a raw score (logit) — can be any real number
    2. Sigmoid squashes it to a probability — between 0 and 1
    3. Threshold converts probability to a decision — class 0 or 1

    Example: Email spam classifier
      logit = 2.5  →  sigmoid = 0.924  →  "SPAM" (> 0.5)
      logit = -1.0 →  sigmoid = 0.269  →  "NOT SPAM" (< 0.5)
    """
    probability = sigmoid(logit)
    prediction = 1 if probability >= threshold else 0
    return probability, prediction

# ── Demo ──────────────────────────────────────────────────
print("Binary Classification with Sigmoid Output")
print("=" * 60)
print("Scenario: Email Spam Classifier")
print()

test_cases = [
    ("Contains 'FREE MONEY!!!'", 4.2),
    ("Meeting at 3pm tomorrow", -2.5),
    ("Click here for prize", 1.8),
    ("Quarterly report attached", -3.1),
    ("Borderline email", 0.1),
]

print(f"{'Email':>30} | {'Logit':>6} | {'P(spam)':>8} | Prediction")
print("-" * 65)
for desc, logit in test_cases:
    prob, pred = binary_classification_output(logit)
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"{desc:>30} | {logit:6.1f} | {prob:8.4f} | {label}")

print()
print("Why sigmoid for binary output (and not ReLU)?")
print("  - Sigmoid guarantees output in (0, 1) → valid probability")
print("  - ReLU output is unbounded → not interpretable as probability")
print("  - Sigmoid + binary cross-entropy loss = clean gradient formula")            '''
    },

    "Softmax (Multi-Class Classification)": {
        "description": "Converts a vector of raw scores (logits) into a probability distribution. Used for choosing 1 of N classes.",
        "code":
            '''import numpy as np

def softmax(logits):
    """
    Softmax — converts raw scores to probability distribution.
    softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)

    NOT applied per-neuron. Applied across an entire layer.
    Output: each element is in (0, 1), and all elements sum to 1.0

    Properties:
      - Amplifies differences: largest logit gets disproportionately 
        large probability
      - Translation invariant: adding a constant to all logits 
        doesn't change the output
      - Sigmoid is softmax for 2 classes

    Numerical stability trick: subtract max(logits) before exp
    to prevent overflow (doesn't change the result due to 
    translation invariance).
    """
    # Subtract max for numerical stability
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)

def softmax_with_temperature(logits, temperature=1.0):
    """
    Temperature-scaled softmax for controlling output sharpness.
    softmax(zᵢ / T)

    T = 1.0  → standard softmax
    T > 1.0  → softer (more uniform) distribution
    T < 1.0  → sharper (more peaked) distribution
    T → 0    → approaches argmax (one-hot)
    T → ∞    → approaches uniform distribution

    Used in language models to control creativity:
      Low temperature  → more deterministic, "safe" outputs
      High temperature → more creative, diverse outputs
    """
    scaled = logits / temperature
    return softmax(scaled)

# ── Demo ──────────────────────────────────────────────────
print("Softmax Demo: Image Classifier")
print("=" * 60)

# Simulated raw scores from a network classifying an image
classes = ["Cat", "Dog", "Bird", "Fish", "Horse"]
logits = np.array([2.0, 1.0, 0.5, -1.0, -0.5])

probs = softmax(logits)

print(f"Raw logits: {logits}")
print()
print(f"{'Class':>8} | {'Logit':>6} | {'P(class)':>8} | Bar")
print("-" * 50)
for cls, logit, prob in zip(classes, logits, probs):
    bar = "█" * int(prob * 40)
    print(f"{cls:>8} | {logit:6.1f} | {prob:8.4f} | {bar}")
print(f"{'Sum':>8} |        | {np.sum(probs):8.4f} |")

print()
print("Temperature Scaling Effect:")
print("-" * 60)
for temp in [0.5, 1.0, 2.0, 5.0]:
    probs_t = softmax_with_temperature(logits, temp)
    top_prob = np.max(probs_t)
    entropy = -np.sum(probs_t * np.log(probs_t + 1e-10))
    print(f"  T={temp:4.1f}: probs={np.round(probs_t, 3)}  "
          f"(top={top_prob:.3f}, entropy={entropy:.3f})")

print()
print("Lower temperature → more confident (peaked)")
print("Higher temperature → more uncertain (uniform)")
print("Language models use this to control creativity vs. safety")            '''
    },

    # =========================================================================
    # DEMONSTRATIONS & COMPARISONS
    # =========================================================================

    "Vanishing Gradient Demo": {
        "description": "Demonstrates why sigmoid kills learning in deep networks and how ReLU fixes it.",
        "code":
            '''import numpy as np

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

print("═" * 65)
print("VANISHING GRADIENT PROBLEM — FULL DEMONSTRATION")
print("═" * 65)
print()
print("Setup: 10-layer network, initial gradient = 1.0")
print("At each layer, gradient is multiplied by the activation derivative.")
print()

# Track gradients through 10 layers for each activation
activations = {
    "Sigmoid":   {"deriv": 0.25, "grad": 1.0},  # max sigmoid derivative
    "Tanh":      {"deriv": 0.65, "grad": 1.0},  # typical tanh derivative
    "ReLU":      {"deriv": 1.00, "grad": 1.0},  # active neuron
    "Leaky ReLU":{"deriv": 1.00, "grad": 1.0},  # active neuron (same as ReLU)
}

print(f"{'Layer':>7} | {'Sigmoid':>12} | {'Tanh':>12} | {'ReLU':>12} | {'Leaky ReLU':>12}")
print("-" * 65)

for layer in range(1, 11):
    row = f"Layer {layer:2d} |"
    for name, data in activations.items():
        data["grad"] *= data["deriv"]
        row += f" {data['grad']:12.8f} |"
    print(row)

print()
print("Summary after 10 layers:")
print(f"  Sigmoid gradient:    {0.25**10:.12f}  ← VANISHED")
print(f"  Tanh gradient:       {0.65**10:.12f}  ← weak but present")
print(f"  ReLU gradient:       {1.0**10:.1f}             ← FULL STRENGTH")
print()
print("This is why deep learning was stuck for decades.")
print("Sigmoid networks deeper than ~4 layers couldn't learn.")
print("ReLU removed this barrier entirely.")
print()

# Extended: what happens at 50 and 100 layers?
print("At extreme depths:")
for depth in [20, 50, 100]:
    sig = 0.25 ** depth
    print(f"  {depth} layers — Sigmoid: {sig:.2e} | ReLU: 1.0")            '''
    },

    "Dead Neuron Demo": {
        "description": "Shows how neurons permanently die with ReLU and how Leaky ReLU prevents it.",
        "code":
            '''import numpy as np
np.random.seed(42)

print("═" * 65)
print("DEAD NEURON PROBLEM — DEMONSTRATION")
print("═" * 65)
print()

def relu(z): return np.maximum(0, z)
def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)

# Simulate a neuron with weights that gradually become negative
n_inputs = 5
weights = np.array([0.2, -0.1, 0.3, -0.2, 0.1])
bias = 0.1

# Generate 10 random inputs
inputs = np.random.randn(10, n_inputs)

print("Phase 1: Healthy neuron (bias = 0.1)")
print("-" * 50)
alive_count = 0
for i, x in enumerate(inputs):
    z = np.dot(x, weights) + bias
    out = relu(z)
    alive = "ACTIVE" if out > 0 else "dead"
    if out > 0: alive_count += 1
    print(f"  Input {i+1}: z = {z:7.4f}, ReLU = {out:7.4f}  [{alive}]")
print(f"  Active: {alive_count}/10")

# Simulate what happens after a large weight update pushes bias negative
print()
print("Phase 2: After large weight update (bias shifts to -5.0)")
print("-" * 50)
bad_bias = -5.0
alive_relu = 0
alive_leaky = 0

for i, x in enumerate(inputs):
    z = np.dot(x, weights) + bad_bias
    out_relu = relu(z)
    out_leaky = leaky_relu(z)

    if out_relu > 0: alive_relu += 1
    if True: alive_leaky += 1  # leaky is always "alive"

    status = "DEAD FOREVER" if out_relu == 0 else "active"
    print(f"  Input {i+1}: z = {z:7.4f}, "
          f"ReLU = {out_relu:7.4f} [{status}], "
          f"Leaky = {out_leaky:7.4f}")

print()
print(f"ReLU:  {alive_relu}/10 active  — "
      f"{'PERMANENTLY DEAD (gradient = 0, cannot recover)' if alive_relu == 0 else 'some survive'}")
print(f"Leaky: 10/10 active — gradient always flows (0.01), recovery possible")
print()

print("Why this matters:")
print("  Dead neuron: output = 0 → gradient = 0 → weight update = 0")
print("  Zero update means the neuron can NEVER change its weights.")
print("  It's stuck outputting zero forever. Capacity is permanently lost.")
print()
print("Prevention strategies:")
print("  1. Use Leaky ReLU (simplest fix)")
print("  2. Lower the learning rate (prevents overshooting)")
print("  3. Use He initialization (designed for ReLU networks)")
print("  4. Monitor percentage of dead neurons during training")            '''
    },

    "All Activations Visual Comparison": {
        "description": "Side-by-side numerical comparison of all activation functions and their derivatives across the same input range.",
        "code":
            '''import numpy as np

# ── Define all activation functions ──────────────────────
def step(z): return np.where(z >= 0, 1.0, 0.0)
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def tanh_act(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def leaky_relu(z): return np.where(z > 0, z, 0.01 * z)
def elu(z, a=1.0): return np.where(z > 0, z, a * (np.exp(z) - 1))
def gelu(z): return 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))
def swish(z): return z * sigmoid(z)

# ── Define all derivatives ───────────────────────────────
def step_d(z): return np.zeros_like(z)
def sigmoid_d(z): s = sigmoid(z); return s*(1-s)
def tanh_d(z): return 1 - np.tanh(z)**2
def relu_d(z): return np.where(z > 0, 1.0, 0.0)
def leaky_d(z): return np.where(z > 0, 1.0, 0.01)
def elu_d(z, a=1.0): return np.where(z > 0, 1.0, a*np.exp(z))
def swish_d(z): s = sigmoid(z); return s + z*s*(1-s)

z_values = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])

print("═" * 90)
print("ALL ACTIVATION FUNCTIONS — OUTPUT VALUES")
print("═" * 90)
print(f"{'z':>5} | {'Step':>5} | {'σ':>6} | {'tanh':>6} | "
      f"{'ReLU':>5} | {'Leaky':>6} | {'ELU':>6} | {'GELU':>6} | {'Swish':>6}")
print("-" * 90)
for z in z_values:
    print(f"{z:5.1f} | {step(z):5.1f} | {sigmoid(z):6.3f} | "
          f"{tanh_act(z):6.3f} | {relu(z):5.2f} | {leaky_relu(z):6.3f} | "
          f"{elu(z):6.3f} | {gelu(z):6.3f} | {swish(z):6.3f}")

print()
print("═" * 90)
print("ALL ACTIVATION FUNCTIONS — DERIVATIVE VALUES")
print("═" * 90)
print(f"{'z':>5} | {'Step':>5} | {'σ\\'':>6} | {'tanh\\'':>6} | "
      f"{'ReLU\\'':>6} | {'Leaky\\'':>7} | {'ELU\\'':>6} | {'Swish\\'':>7}")
print("-" * 80)
for z in z_values:
    print(f"{z:5.1f} | {step_d(z):5.1f} | {sigmoid_d(z):6.4f} | "
          f"{tanh_d(z):6.4f} | {relu_d(z):6.1f} | {leaky_d(z):7.2f} | "
          f"{elu_d(z):6.4f} | {swish_d(z):7.4f}")

print()
print("Key observations:")
print("  Step:    binary output, zero derivative — useless for learning")
print("  Sigmoid: squashed to (0,1), max derivative 0.25 — vanishing gradient")
print("  Tanh:    zero-centered, max derivative 1.0 — better than sigmoid")
print("  ReLU:    simple, derivative = 1 when active — deep learning enabler")
print("  Leaky:   like ReLU but never fully dead — safe default")
print("  ELU:     smooth negative side, zero-centered — good for deep nets")
print("  GELU:    smooth probabilistic gate — transformer standard")
print("  Swish:   smooth, non-monotonic — found by machine search")            '''
    },

    "Backpropagation with Different Activations": {
        "description": "Full forward + backward pass traced step-by-step, comparing sigmoid vs ReLU networks side by side.",
        "code":
            '''import numpy as np

print("═" * 70)
print("BACKPROPAGATION: SIGMOID vs RELU — SIDE BY SIDE")
print("═" * 70)
print()
print("Network: Input → Hidden (1 neuron) → Output (1 neuron)")
print("Task: Predict 1.0 from input 0.8")
print()

# ── Activation functions ─────────────────────────────────
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(output): return output * (1 - output)
def relu(z): return max(0, z)
def relu_deriv(z): return 1.0 if z > 0 else 0.0

# ── Shared initial state ─────────────────────────────────
input_val = 0.8
w_hidden = 0.5        # input → hidden weight
b_hidden = 0.1        # hidden bias
w_output = 0.7        # hidden → output weight
b_output = 0.2        # output bias
expected = 1.0
lr = 0.1

for act_name in ["SIGMOID", "RELU"]:
    print(f"{'─' * 30} {act_name} {'─' * 30}")

    # ── FORWARD PASS ─────────────────────────────────────
    z_hidden = input_val * w_hidden + b_hidden

    if act_name == "SIGMOID":
        h_out = sigmoid(z_hidden)
    else:
        h_out = relu(z_hidden)

    z_output = h_out * w_output + b_output

    if act_name == "SIGMOID":
        final_out = sigmoid(z_output)
    else:
        final_out = relu(z_output)  # for demo; output would normally use sigmoid

    print(f"  Forward Pass:")
    print(f"    z_hidden = {input_val} × {w_hidden} + {b_hidden} = {z_hidden:.4f}")
    print(f"    hidden_out = {act_name.lower()}({z_hidden:.4f}) = {h_out:.4f}")
    print(f"    z_output = {h_out:.4f} × {w_output} + {b_output} = {z_output:.4f}")
    print(f"    final_out = {act_name.lower()}({z_output:.4f}) = {final_out:.4f}")
    print(f"    error = {expected} - {final_out:.4f} = {expected - final_out:.4f}")
    print()

    # ── BACKWARD PASS ────────────────────────────────────
    error = expected - final_out

    if act_name == "SIGMOID":
        deriv_out = sigmoid_deriv(final_out)
        deriv_hid = sigmoid_deriv(h_out)
    else:
        deriv_out = relu_deriv(z_output)
        deriv_hid = relu_deriv(z_hidden)

    delta_output = error * deriv_out
    blame_hidden = delta_output * w_output
    delta_hidden = blame_hidden * deriv_hid

    # Weight updates
    dw_output = lr * delta_output * h_out
    dw_hidden = lr * delta_hidden * input_val

    print(f"  Backward Pass:")
    print(f"    {act_name.lower()}_derivative(output) = {deriv_out:.4f}")
    print(f"    delta_output = {error:.4f} × {deriv_out:.4f} = {delta_output:.4f}")
    print(f"    blame to hidden = {delta_output:.4f} × {w_output} = {blame_hidden:.4f}")
    print(f"    {act_name.lower()}_derivative(hidden) = {deriv_hid:.4f}")
    print(f"    delta_hidden = {blame_hidden:.4f} × {deriv_hid:.4f} = {delta_hidden:.4f}")
    print()
    print(f"  Weight Updates (lr = {lr}):")
    print(f"    Δw_output = {lr} × {delta_output:.4f} × {h_out:.4f} = {dw_output:.6f}")
    print(f"    Δw_hidden = {lr} × {delta_hidden:.4f} × {input_val} = {dw_hidden:.6f}")
    print(f"    new w_output = {w_output} + {dw_output:.6f} = {w_output + dw_output:.6f}")
    print(f"    new w_hidden = {w_hidden} + {dw_hidden:.6f} = {w_hidden + dw_hidden:.6f}")
    print()

print("═" * 70)
print("Notice how the sigmoid derivative (≤ 0.25) shrinks the delta,")
print("while the ReLU derivative (= 1.0) passes it through at full strength.")
print("In deeper networks, this difference compounds exponentially.")            '''
    },

    # =========================================================================
    # COMPLETE NETWORK SIMULATIONS
    # =========================================================================

    "Forward and Backward Pass (Sigmoid)": {
        "description": "Complete trace of a sigmoid network: forward pass, error calculation, backpropagation, and weight update.",
        "code":
            '''import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("═" * 65)
print("COMPLETE SIGMOID NETWORK — FORWARD + BACKWARD PASS")
print("═" * 65)
print()
print("Architecture: 1 input → 1 hidden neuron → 1 output neuron")
print()

# ── Initial state ────────────────────────────────────────
x = 0.8           # input
w1 = 0.5          # input-to-hidden weight
b1 = 0.1          # hidden bias
w2 = 0.7          # hidden-to-output weight
b2 = 0.2          # output bias
target = 1.0      # expected output
lr = 0.1          # learning rate

print("Initial weights: w1=0.5, b1=0.1, w2=0.7, b2=0.2")
print(f"Input: {x}, Target: {target}, Learning rate: {lr}")
print()

# ── FORWARD PASS ─────────────────────────────────────────
print("FORWARD PASS")
print("-" * 40)

z1 = x * w1 + b1
h = sigmoid(z1)
print(f"  Hidden: z = {x} × {w1} + {b1} = {z1}")
print(f"  Hidden: output = sigmoid({z1}) = {h:.6f}")

z2 = h * w2 + b2
y = sigmoid(z2)
print(f"  Output: z = {h:.6f} × {w2} + {b2} = {z2:.6f}")
print(f"  Output: prediction = sigmoid({z2:.6f}) = {y:.6f}")

error = target - y
print(f"  Error: {target} - {y:.6f} = {error:.6f}")
print()

# ── BACKWARD PASS ────────────────────────────────────────
print("BACKWARD PASS (Backpropagation)")
print("-" * 40)

# Output layer delta
d_out = y * (1 - y)  # sigmoid derivative
delta2 = error * d_out
print(f"  Output sigmoid_derivative = {y:.6f} × (1 - {y:.6f}) = {d_out:.6f}")
print(f"  Output delta = {error:.6f} × {d_out:.6f} = {delta2:.6f}")
print()

# Hidden layer delta (blame flows backward)
blame = delta2 * w2
d_hid = h * (1 - h)  # sigmoid derivative
delta1 = blame * d_hid
print(f"  Blame to hidden = {delta2:.6f} × {w2} = {blame:.6f}")
print(f"  Hidden sigmoid_derivative = {h:.6f} × (1 - {h:.6f}) = {d_hid:.6f}")
print(f"  Hidden delta = {blame:.6f} × {d_hid:.6f} = {delta1:.6f}")
print()

# ── WEIGHT UPDATES ───────────────────────────────────────
print("WEIGHT UPDATES")
print("-" * 40)
print("  Formula: new_weight = old_weight + lr × delta × input")
print()

dw2 = lr * delta2 * h
db2 = lr * delta2
dw1 = lr * delta1 * x
db1 = lr * delta1

print(f"  Δw2 = {lr} × {delta2:.6f} × {h:.6f} = {dw2:.8f}")
print(f"  Δb2 = {lr} × {delta2:.6f} = {db2:.8f}")
print(f"  Δw1 = {lr} × {delta1:.6f} × {x} = {dw1:.8f}")
print(f"  Δb1 = {lr} × {delta1:.6f} = {db1:.8f}")
print()

new_w2 = w2 + dw2
new_b2 = b2 + db2
new_w1 = w1 + dw1
new_b1 = b1 + db1

print(f"  w2: {w2} → {new_w2:.8f}")
print(f"  b2: {b2} → {new_b2:.8f}")
print(f"  w1: {w1} → {new_w1:.8f}")
print(f"  b1: {b1} → {new_b1:.8f}")
print()

# Verify: forward pass with new weights
z1_new = x * new_w1 + new_b1
h_new = sigmoid(z1_new)
z2_new = h_new * new_w2 + new_b2
y_new = sigmoid(z2_new)
new_error = target - y_new
print(f"  Verification — new prediction: {y_new:.6f} (was {y:.6f})")
print(f"  New error: {new_error:.6f} (was {error:.6f})")
print(f"  Error reduced by: {abs(error) - abs(new_error):.6f}")
print(f"  Direction: {'CORRECT ✓' if abs(new_error) < abs(error) else 'WRONG ✗'}")            '''
    },

    "Forward and Backward Pass (ReLU)": {
        "description": "Same network architecture as the sigmoid version, but with ReLU in hidden layer. Compare the gradient magnitudes.",
        "code":
            '''import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return max(0, z)

def relu_deriv(z):
    return 1.0 if z > 0 else 0.0

print("═" * 65)
print("COMPLETE ReLU NETWORK — FORWARD + BACKWARD PASS")
print("═" * 65)
print()
print("Architecture: 1 input → 1 hidden (ReLU) → 1 output (Sigmoid)")
print("Note: Output uses sigmoid for probability, hidden uses ReLU")
print()

# ── Initial state (same as sigmoid demo) ─────────────────
x = 0.8
w1 = 0.5
b1 = 0.1
w2 = 0.7
b2 = 0.2
target = 1.0
lr = 0.1

print(f"Initial weights: w1={w1}, b1={b1}, w2={w2}, b2={b2}")
print(f"Input: {x}, Target: {target}")
print()

# ── FORWARD PASS ─────────────────────────────────────────
print("FORWARD PASS")
print("-" * 40)

z1 = x * w1 + b1
h = relu(z1)
print(f"  Hidden: z = {x} × {w1} + {b1} = {z1}")
print(f"  Hidden: output = ReLU({z1}) = {h:.6f}")

z2 = h * w2 + b2
y = sigmoid(z2)
print(f"  Output: z = {h:.6f} × {w2} + {b2} = {z2:.6f}")
print(f"  Output: prediction = sigmoid({z2:.6f}) = {y:.6f}")

error = target - y
print(f"  Error: {target} - {y:.6f} = {error:.6f}")
print()

# ── BACKWARD PASS ────────────────────────────────────────
print("BACKWARD PASS (Backpropagation)")
print("-" * 40)

# Output layer: still sigmoid (for probability output)
d_out = y * (1 - y)
delta2 = error * d_out
print(f"  Output sigmoid_deriv = {d_out:.6f}")
print(f"  Output delta = {error:.6f} × {d_out:.6f} = {delta2:.6f}")
print()

# Hidden layer: ReLU derivative
blame = delta2 * w2
d_hid = relu_deriv(z1)
delta1 = blame * d_hid
print(f"  Blame to hidden = {delta2:.6f} × {w2} = {blame:.6f}")
print(f"  ReLU_derivative({z1}) = {d_hid:.1f}  ← {'FULL passthrough!' if d_hid == 1 else 'BLOCKED (dead neuron)'}")
print(f"  Hidden delta = {blame:.6f} × {d_hid:.1f} = {delta1:.6f}")
print()

# ── WEIGHT UPDATES ───────────────────────────────────────
print("WEIGHT UPDATES")
print("-" * 40)

dw2 = lr * delta2 * h
db2 = lr * delta2
dw1 = lr * delta1 * x
db1 = lr * delta1

print(f"  Δw2 = {dw2:.8f}")
print(f"  Δw1 = {dw1:.8f}")
print()

# ── Compare with sigmoid network ─────────────────────────
print("═" * 65)
print("COMPARISON: Hidden Layer Weight Update")
print("═" * 65)

# Sigmoid version (from previous demo)
z1_sig = x * w1 + b1
h_sig = sigmoid(z1_sig)
z2_sig = h_sig * w2 + b2
y_sig = sigmoid(z2_sig)
error_sig = target - y_sig
delta2_sig = error_sig * y_sig * (1 - y_sig)
blame_sig = delta2_sig * w2
delta1_sig = blame_sig * h_sig * (1 - h_sig)
dw1_sig = lr * delta1_sig * x

print(f"  Sigmoid hidden delta: {delta1_sig:.8f}  (derivative ≈ 0.25 shrank it)")
print(f"  ReLU hidden delta:    {delta1:.8f}  (derivative = 1.0 preserved it)")
print(f"  Ratio: ReLU delta is {abs(delta1/delta1_sig):.1f}× larger")
print()
print(f"  Sigmoid Δw1: {dw1_sig:.8f}")
print(f"  ReLU Δw1:    {dw1:.8f}")
print(f"  ReLU updates {abs(dw1/dw1_sig):.1f}× faster on this hidden weight")
print()
print("In deeper networks, this difference compounds exponentially.")            '''
    },

    "Multi-Layer Gradient Flow Comparison": {
        "description": "Tracks gradient magnitude through networks of increasing depth (5, 10, 20, 50 layers) for all activation functions.",
        "code":
            '''import numpy as np

print("═" * 70)
print("GRADIENT FLOW THROUGH DEEP NETWORKS — ALL ACTIVATIONS")
print("═" * 70)
print()
print("Tracking gradient magnitude through networks of various depths.")
print("Assumes best-case derivatives at each layer.")
print()

# Best-case derivative for each activation
activations = {
    "Step":       0.0,     # always 0
    "Sigmoid":    0.25,    # max derivative
    "Tanh":       1.0,     # max derivative (at z=0)
    "ReLU":       1.0,     # active neuron
    "Leaky ReLU": 1.0,     # active neuron (worst case: 0.01)
    "ELU":        1.0,     # positive side
    "GELU":       1.08,    # can slightly exceed 1
    "Swish":      1.10,    # can slightly exceed 1
}

depths = [1, 5, 10, 20, 50, 100]

print(f"{'Activation':<12} |", end="")
for d in depths:
    print(f" {'L='+str(d):>10} |", end="")
print()
print("-" * 85)

for name, deriv in activations.items():
    print(f"{name:<12} |", end="")
    for d in depths:
        if deriv == 0:
            grad = 0.0
            s = "0"
        else:
            grad = deriv ** d
            if grad > 1e6:
                s = f"{grad:.1e}"
            elif grad < 1e-6:
                s = f"{grad:.1e}"
            else:
                s = f"{grad:.6f}"
        print(f" {s:>10} |", end="")
    print()

print()
print("Analysis:")
print("-" * 70)
print("Step:     Gradient is always 0. Learning is impossible at any depth.")
print("Sigmoid:  Gradient vanishes exponentially. At 10 layers: ~10⁻⁶.")
print("Tanh:     Best case matches ReLU, but in practice derivative < 1")
print("          so it behaves more like sigmoid in deep networks.")
print("ReLU:     Perfect gradient passthrough. No loss at any depth.")
print("Leaky:    Same as ReLU when active. Worst case: 0.01^n (still")
print("          much better than sigmoid's 0.25^n).")
print("GELU/Swish: Derivatives slightly > 1 — risk of gradient EXPLOSION")
print("            in theory, but in practice the average is ≈ 1.")
print()

# Realistic simulation with random derivatives
print("═" * 70)
print("REALISTIC SIMULATION (random inputs, 1000 neurons per layer)")
print("═" * 70)
print()

np.random.seed(42)

for depth in [10, 50]:
    print(f"Network depth: {depth} layers")
    print("-" * 50)

    # Sigmoid: derivative = σ(z)×(1-σ(z)) for random z
    z_random = np.random.randn(1000)
    sig_grad = 1.0
    for _ in range(depth):
        s = 1 / (1 + np.exp(-np.random.randn(1000)))
        derivs = s * (1 - s)
        sig_grad *= np.mean(derivs)

    # ReLU: derivative = 1 for z>0, 0 for z<0 (roughly 50% active)
    relu_grad = 1.0
    for _ in range(depth):
        z = np.random.randn(1000)
        active_frac = np.mean(z > 0)
        relu_grad *= active_frac  # average derivative

    print(f"  Sigmoid avg gradient: {sig_grad:.2e}")
    print(f"  ReLU avg gradient:    {relu_grad:.2e}")
    print(f"  ReLU / Sigmoid ratio: {relu_grad/sig_grad:.2e}×")
    print()            '''
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    from Required_Images.relu_visual import RELU_VISUAL_HTML, RELU_VISUAL_HEIGHT

    # Fallback static image
    relu_img = _image_to_html(
        "Required_Images/relu_visual.png",
        alt="ReLU Activation Function",
        width="50%"
    )
    theory_with_images = THEORY.replace("{{RELU_IMAGE}}", relu_img)

    # Interactive component data
    interactive_components = [
        {
            "placeholder": "{{RELU_IMAGE}}",
            "html": RELU_VISUAL_HTML,
            "height": RELU_VISUAL_HEIGHT,
        }
    ]

    return {
        "theory": theory_with_images,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }

