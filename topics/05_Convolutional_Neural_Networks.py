"""
Convolutional Neural Networks — Architecture of AI
====================================================
Comprehensive guide to CNNs: from the problem they solve to the
hierarchy of learned features.

Covers: The Convolution Operation, Filters & Feature Maps, Weight Sharing,
        Pooling (Max, Average, Global Average), Stride & Padding,
        Fully Connected Layers, Backpropagation through CNNs,
        and the Feature Hierarchy — with theory, math, visual intuition,
        and runnable Python implementations.

"""
import math
import numpy as np
import base64
import os

TOPIC_NAME = "Convolutional_Neural_Networks"
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

# THEORY = """
#
# """
THEORY = """
## Part 1: The Problem That CNNs Solve

Remember how a single perceptron takes inputs, multiplies by weights, adds bias, and passes through
an activation? Now imagine you want to recognize whether a photo contains a cat. A tiny 28×28
grayscale image has 784 pixels. If you fed that into a regular neural network (called a "fully
connected" or "dense" network), every neuron in the first layer would connect to all 784 pixels.

That means one neuron = 784 weights + 1 bias = 785 parameters. A layer with 100 neurons = 78,500
parameters. And that's just a tiny image. A 1080p color image has over 6 million pixels. The
parameter count explodes, the network is slow, and worse — it doesn't understand spatial structure
at all. It treats pixel (0,0) and pixel (27,27) as equally related, even though nearby pixels are
far more relevant to each other.

**CNNs solve this with one elegant idea:** instead of looking at the entire image at once, slide a
small window across the image and look at local patches.

### Why Fully Connected Networks Fail on Images

To make the scale problem concrete, consider what happens with realistic image sizes:

| Image Size | Pixels | FC Weights (100 neurons) | CNN 3×3 Filter |
|:---|---:|---:|---:|
| 28×28 grayscale | 784 | 78,500 | 9 + 1 = **10** |
| 224×224 RGB | 150,528 | 15,052,800 | 27 + 1 = **28** |
| 1080p RGB | 6,220,800 | 622,080,000 | 27 + 1 = **28** |

A single CNN filter has the same number of parameters regardless of image size. That's the power
of **weight sharing** — the same small filter scans the entire image. A fully connected layer needs
a separate weight for every pixel-to-neuron connection.

But the parameter explosion isn't even the worst part. A fully connected network has no concept of
"nearby." Pixel (0,0) in the top-left corner is connected to the same neuron as pixel (27,27) in
the bottom-right corner with equal weight. The network has to *learn from scratch* that nearby
pixels matter more — and that's an enormous amount of structure to discover purely from data.
CNNs build this spatial understanding directly into the architecture.

---

## Part 2: The Core Concepts

### The Convolution Operation

This is where the name comes from. Instead of a neuron connecting to every input, a CNN uses a
**filter** (also called a **kernel**) — a small grid of weights that slides across the image.

Imagine your input is a simple 5×5 image (25 pixels), and your filter is 3×3 (9 weights):

```
Input Image (5×5)              Filter (3×3 Kernel)

┌───┬───┬───┬───┬───┐         ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │         │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤         ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │         │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤         ├───┼───┼───┤
│ 1 │ 1 │ 1 │ 0 │ 0 │         │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤         └───┴───┴───┘
│ 0 │ 0 │ 1 │ 1 │ 0 │
├───┼───┼───┼───┼───┤
│ 1 │ 0 │ 0 │ 1 │ 1 │
└───┴───┴───┴───┴───┘
```

The filter starts at the top-left corner, covers a 3×3 patch of the image, and does exactly what
a perceptron does — element-wise multiply and sum:

**Position (0,0)** — filter over top-left patch:

```
Image patch:      Filter:        Multiply:
1  0  1           1   0  -1      (1×1) + (0×0) + (1×-1)
0  1  0    ×      1   0  -1   =  (0×1) + (1×0) + (0×-1)
1  1  1           1   0  -1      (1×1) + (1×0) + (1×-1)

= (1 + 0 + -1) + (0 + 0 + 0) + (1 + 0 + -1) = 0
```

Then the filter slides one pixel to the right and does it again. Then again. When it reaches the
right edge, it drops down one row and starts sliding right again. Each position produces one
number. The result is a new, smaller grid called a **feature map** (or **activation map**).

For a 5×5 input with a 3×3 filter (no padding, stride of 1), the output is a 3×3 feature map.

```
Output Feature Map (3×3)

┌────┬────┬────┐
│  0 │  0 │ -2 │
├────┼────┼────┤
│  1 │  0 │ -1 │
├────┼────┼────┤
│  2 │  1 │ -1 │
└────┴────┴────┘
```

Each number in this feature map represents how much that patch of the image matched the filter's
pattern. This particular filter (with 1s on the left, -1s on the right) is detecting **vertical
edges** — places where the left side is bright and the right side is dark.

### The Convolution as Dot Product

If you unroll the math, each position of the convolution is just a **dot product** — the same
operation a perceptron does. The 3×3 image patch is flattened to a 9-element vector, the 3×3
filter is flattened to a 9-element vector, and you compute their dot product. A high dot product
means the patch closely matches the filter pattern. A low or negative dot product means it
doesn't match.

```
Patch flattened : [1, 0, 1, 0, 1, 0, 1, 1, 1]
Filter flattened: [1, 0, -1, 1, 0, -1, 1, 0, -1]

Dot product: (1)(1) + (0)(0) + (1)(-1) + (0)(1) + (1)(0) + (0)(-1)
           + (1)(1) + (1)(0) + (1)(-1) = 0
```

This is why convolution can be thought of as **"template matching"** — the filter is the template,
and the dot product measures how well each patch matches it.

---

### Key Insight: Weight Sharing

Here's what makes CNNs efficient. That 3×3 filter has only **9 weights + 1 bias = 10 parameters**.
And those same 10 parameters are reused at every position across the entire image. A fully
connected approach would need 784 weights per neuron. A CNN filter needs 9. This is called
**weight sharing**, and it's why CNNs can handle large images.

Weight sharing also encodes a powerful assumption: **translation equivariance**. If a vertical
edge exists in the top-left corner, the same filter can detect it in the bottom-right corner.
The filter doesn't need to learn "vertical edge at position (3,5)" and "vertical edge at
position (20,15)" separately. It learns "vertical edge" once and applies it everywhere.

This assumption is almost always correct for vision tasks. A cat's ear looks the same whether
it appears on the left side or the right side of the image. Weight sharing lets CNNs exploit
this regularity, dramatically reducing the number of parameters needed.

---

### What Does a Filter Detect?

Different filters detect different features:

```
Vertical edge:    Horizontal edge:    Corner:
 1  0 -1           1   1   1          0  1  1
 1  0 -1           0   0   0          0  0  1
 1  0 -1          -1  -1  -1          0  0  0
```

In the first layer of a CNN, filters learn to detect simple things like edges, gradients, and
color blobs. In deeper layers, they combine those simple features into complex ones:

> **Edges → Textures → Shapes → Faces**

#### More Filter Examples

Here are additional filters that CNNs commonly learn in their early layers:

```
Sharpen:         Blur (average):    Emboss:          Sobel X:
 0  -1  0        1/9 1/9 1/9        -2 -1  0         1  0 -1
-1   5 -1        1/9 1/9 1/9        -1  1  1         2  0 -2
 0  -1  0        1/9 1/9 1/9         0  1  2         1  0 -1
```

The network doesn't use these exact filters — it *learns* filters through backpropagation. But
the filters it learns often closely resemble these classical image processing operators. The
first-layer filters of AlexNet (2012), when visualized, show clear edge detectors, color
gradient detectors, and Gabor-like texture filters.

---

### Multiple Filters = Multiple Feature Maps

A single filter detects a single pattern. But images contain many different patterns — horizontal
edges, vertical edges, corners, curves, color gradients, and more. So a convolutional layer
uses **multiple filters**, each producing its own feature map.

If a layer has 32 filters, the output is 32 feature maps stacked together. Each map highlights
where a different pattern was found in the input. The next layer then takes these 32 maps as
input and applies its own filters to detect combinations of those patterns.

This is how **hierarchy** emerges. Layer 1's 32 feature maps detect 32 types of simple patterns.
Layer 2's filters combine those patterns into more complex ones. Each layer builds on the
representations discovered by the previous layer.

---

### 1×1 Convolutions — The Surprising Workhorse

This one confuses people at first. A 1×1 filter seems pointless — it only looks at a single
pixel, so how can it detect any spatial pattern? The answer: **it operates across channels,
not across space.**

Suppose the input to a layer has 256 feature maps (channels). A 1×1 convolution with 64
filters takes the 256 values at each spatial position and compresses them down to 64 values
through a learned weighted combination. It's essentially a tiny fully connected network applied
independently at every pixel location.

```
Input: 256 channels at each position
1×1 conv with 64 filters:

At position (i, j):
    input  = [ch1, ch2, ch3, ..., ch256]    (256 values)
    output = W × input + bias               (W is 64 × 256)
    result = [out1, out2, ..., out64]        (64 values)

This happens at EVERY spatial position independently.
```

**Why this matters:**

**Channel compression** — Going from 256 channels to 64 channels drastically reduces the number
of parameters that the next 3×3 convolutional layer needs. A 3×3 filter on 256 channels has
3×3×256 = 2,304 weights. On 64 channels it has 3×3×64 = 576 weights. That's a **4× reduction**.

**Adding non-linearity** — A 1×1 convolution followed by ReLU lets the network learn non-linear
combinations of existing features at each position without changing the spatial dimensions.

**Bottleneck design** — This is the core idea behind ResNet's bottleneck blocks and GoogLeNet's
Inception modules. Instead of running an expensive 3×3 convolution on 256 channels directly,
you first compress to 64 channels with a 1×1 conv, then apply the 3×3 conv on the smaller
representation, then expand back with another 1×1 conv:

```
256 channels → 1×1 conv → 64 channels → 3×3 conv → 64 channels → 1×1 conv → 256 channels
  (compress)                 (process)                  (expand)
```

This "bottleneck" pattern saves enormous computation while maintaining representational power
and is one of the key innovations that allowed networks to go from tens of layers to hundreds.

---

### Stride and Padding

Two parameters control how the filter slides across the input:

**Stride** — how many pixels the filter jumps between positions. Stride 1 means the filter
moves one pixel at a time (maximum overlap between positions). Stride 2 means it jumps two
pixels, producing a feature map half the size in each dimension.

```
Stride 1 on 5×5 with 3×3 filter → 3×3 output  (9 positions)
Stride 2 on 5×5 with 3×3 filter → 2×2 output  (4 positions)
```

**Padding** — adding zeros around the border of the input so the filter can be centered on edge
pixels. Without padding ("valid" convolution), the output is smaller than the input. With "same"
padding, enough zeros are added to keep the output the same size as the input.

```
Output size formula:
  output = floor((input_size - filter_size + 2 × padding) / stride) + 1

Examples (input = 28×28, filter = 3×3):
  No padding, stride 1:  (28 - 3 + 0) / 1 + 1 = 26×26
  Padding 1,  stride 1:  (28 - 3 + 2) / 1 + 1 = 28×28  ← same size!
  No padding, stride 2:  (28 - 3 + 0) / 2 + 1 = 13×13  ← halved!
```

Using stride > 1 is an alternative to pooling for downsampling — some modern architectures
(like the all-convolutional net) replace pooling layers entirely with strided convolutions.

---

### Convolutions on Multi-Channel Inputs (RGB Images)

A grayscale image has 1 channel. An RGB image has 3 channels (red, green, blue). When the input
has multiple channels, each filter also has multiple channels — one set of weights per input
channel.

For an RGB input, a 3×3 filter actually has shape 3×3×3 = 27 weights (plus 1 bias = 28
parameters). The filter performs element-wise multiplication and summation across **ALL** channels
simultaneously, producing a single value per position.

```
Filter shape: (height, width, input_channels)

    3×3 on grayscale:   3 × 3 × 1  =   9 weights
    3×3 on RGB:          3 × 3 × 3  =  27 weights
    3×3 on 64-channel:  3 × 3 × 64 = 576 weights
```

Each filter produces **ONE** feature map, regardless of how many input channels there are. If the
layer has 32 filters on an RGB input, the total parameters are:
32 × (3×3×3 + 1) = 32 × 28 = **896**.

---

## Part 3: Building a Small 3-Layer CNN

Let's build a concrete CNN that classifies a 5×5 grayscale image as either an **"X"** pattern or
an **"O"** pattern. Everything is kept tiny so you can trace every number.

### The Architecture

```
Layer 1: Convolutional Layer (2 filters, 3×3)
    ↓
Layer 2: Pooling Layer (2×2 max pooling)
    ↓
Layer 3: Fully Connected Layer (output: 2 neurons → "X" or "O")
```

### Input: A 5×5 "X" Pattern

```
1  0  0  0  1
0  1  0  1  0
0  0  1  0  0
0  1  0  1  0
1  0  0  0  1
```

---

### Layer 1: Convolutional Layer

We use 2 filters (this means this layer has 2 neurons, in CNN terms — each filter is like one
neuron that scans the whole image).

**Filter A** (learns to detect diagonal `\\` lines):
```
 1  0  0
 0  1  0
 0  0  1
```

**Filter B** (learns to detect diagonal `/` lines):
```
 0  0  1
 0  1  0
 1  0  0
```

Each filter slides across the 5×5 input and produces a 3×3 feature map.

#### Filter A applied (detects `\\` diagonals):

```
Position (0,0):                    Position (0,1):
1  0  0      1  0  0               0  0  0     1  0  0
0  1  0  ×   0  1  0     = 3       1  0  1  ×  0  1  0   = 1
0  0  1      0  0  1               0  1  0     0  0  1
```

After sliding through all 9 positions:

```
Feature Map A (3×3):
3  1  1      ← High value at (0,0): strong "\\" match there
1  3  1      ← High value at (1,1): strong "\\" match in center
1  1  3      ← High value at (2,2): strong "\\" match bottom-right
```

The diagonal of 3s means the filter found strong `\\` diagonals running through the image. That
makes sense — our "X" has a `\\` diagonal.

#### Feature Map B (from Filter B, detecting `/` diagonals):

```
Feature Map B (3×3):
1  1  3
1  3  1
3  1  1
```

The anti-diagonal of 3s shows it found `/` diagonals. Our "X" has both, so both feature maps
light up.

> **After ReLU activation** (same as you learned — max(0, z)):
> All values are already positive, so nothing changes here. But if any were negative, they'd
> become 0.

---

### Layer 2: Pooling Layer (2×2 Max Pooling)

Pooling is a **downsampling** step. It shrinks the feature map to reduce computation and make the
network care less about exact pixel positions (this is called **translation invariance** — a cat is
still a cat whether it's shifted 2 pixels left).

Max pooling takes a 2×2 window and keeps only the maximum value:

```
Feature Map A (3×3):         Pooled A (2×2):
                             After 2×2 Max Pool:
3  1  1
1  3  1          →           3  1
1  1  3                      1  3
```

*(With a 3×3 input pooled by 2×2 with stride 2, we get a 2×2 output. The bottom row and right
column that don't fit a full 2×2 window get handled depending on implementation — here we're
keeping it simple with a 2×2 output by only using positions where the full window fits, and
adjusting slightly for illustration. In practice, padding or ceiling mode would handle this.)*

Same for Feature Map B:

```
Feature Map B (3×3):         Pooled B (2×2):

1  1  3
1  3  1          →           3  3
3  1  1                      3  1
```

Now we have two 2×2 grids = **8 total values**.

---

### What Pooling Actually Does

Pooling takes a feature map and shrinks it by summarizing small regions into single values. It's
like looking at a photo and saying *"I don't need to know every pixel — just tell me the general
idea of each area."*

There are three main types of pooling, each with different math:

#### Type 1: Max Pooling *(Most Common)*

**Rule:** From each window, keep only the largest value.
That's it. No weights, no bias, no activation function. Just pick the max.

```
Input Feature Map (4×4)              2×2 Max Pool, Stride 2

┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 0 │                   Window 1: max(1, 3, 5, 7) = 7
├───┼───┼───┼───┤                   Window 2: max(2, 0, 1, 4) = 4
│ 5 │ 7 │ 1 │ 4 │                   Window 3: max(2, 0, 3, 8) = 8
├───┼───┼───┼───┤                   Window 4: max(6, 1, 9, 2) = 9
│ 2 │ 0 │ 6 │ 1 │
├───┼───┼───┼───┤                   Output (2×2):
│ 3 │ 8 │ 9 │ 2 │                   ┌───┬───┐
└───┴───┴───┴───┘                   │ 7 │ 4 │
                                     ├───┼───┤
                                     │ 8 │ 9 │
                                     └───┴───┘
```

**How the windows are selected:**

```
Window 1 (rows 0-1, cols 0-1):      Window 2 (rows 0-1, cols 2-3):
┌───┬───┐                           ┌───┬───┐
│ 1 │ 3 │                           │ 2 │ 0 │
├───┼───┤  → max = 7                ├───┼───┤  → max = 4
│ 5 │ 7 │                           │ 1 │ 4 │
└───┴───┘                           └───┴───┘

Window 3 (rows 2-3, cols 0-1):      Window 4 (rows 2-3, cols 2-3):
┌───┬───┐                           ┌───┬───┐
│ 2 │ 0 │                           │ 6 │ 1 │
├───┼───┤  → max = 8                ├───┼───┤  → max = 9
│ 3 │ 8 │                           │ 9 │ 2 │
└───┴───┘                           └───┴───┘
```

The math formula:

```
output(i, j) = max( input[i*s : i*s+p,  j*s : j*s+p] )

Where:  s = stride (how far the window jumps each step)
        p = pool size (width/height of the window)
```

#### Type 2: Average Pooling

**Rule:** From each window, compute the average (mean) of all values.

```
Input Feature Map (4×4)              2×2 Average Pool, Stride 2

┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 0 │                   Window 1: (1+3+5+7)/4  = 16/4 = 4.0
├───┼───┼───┼───┤                   Window 2: (2+0+1+4)/4  =  7/4 = 1.75
│ 5 │ 7 │ 1 │ 4 │                   Window 3: (2+0+3+8)/4  = 13/4 = 3.25
├───┼───┼───┼───┤                   Window 4: (6+1+9+2)/4  = 18/4 = 4.5
│ 2 │ 0 │ 6 │ 1 │
├───┼───┼───┼───┤                   Output (2×2):
│ 3 │ 8 │ 9 │ 2 │                   ┌──────┬──────┐
└───┴───┴───┴───┘                   │ 4.00 │ 1.75 │
                                     ├──────┼──────┤
                                     │ 3.25 │ 4.50 │
                                     └──────┴──────┘
```

The math formula:

```
                    1
output(i, j) = ───────── × Σ input[i*s + m, j*s + n]
                 p × p

Where m goes from 0 to p-1, n goes from 0 to p-1.
For a 2×2 pool: you sum all 4 values and divide by 4.
```

#### Type 3: Global Average Pooling

**Rule:** Take the average of the **ENTIRE** feature map, producing a single number.

```
Input Feature Map (4×4)

┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 0 │
├───┼───┼───┼───┤
│ 5 │ 7 │ 1 │ 4 │    →  (1+3+2+0+5+7+1+4+2+0+6+1+3+8+9+2) / 16
├───┼───┼───┼───┤    =  59 / 16
│ 2 │ 0 │ 6 │ 1 │    =  3.6875
├───┼───┼───┼───┤
│ 3 │ 8 │ 9 │ 2 │    Output: a single value → 3.6875
└───┴───┴───┴───┘
```

This is used near the end of modern CNNs (like ResNet, GoogLeNet) to collapse each feature map
into one number before the final classification layer.

---

### Understanding Stride in Pooling

Stride controls how far the window jumps between positions. This is what determines the output size.

```
Stride 2 (non-overlapping):         Stride 1 (overlapping):

Step 1:  Step 2:                     Step 1:  Step 2:  Step 3:
[█ █]· ·  · ·[█ █]                   [█ █]· ·  ·[█ █]·  · ·[█ █]
[█ █]· ·  · ·[█ █]                   [█ █]· ·  ·[█ █]·  · ·[█ █]
· · · ·  · · · ·                     · · · ·  · · · ·  · · · ·

Jump = 2 pixels                      Jump = 1 pixel
No overlap                           Windows overlap
Output is smaller                    Output is larger
```

**Output size formula:**

```
output_size = floor((input_size - pool_size) / stride) + 1

Examples with a 4×4 input:
  Pool 2×2, stride 2:  (4 - 2) / 2 + 1 = 2×2 output
  Pool 2×2, stride 1:  (4 - 2) / 1 + 1 = 3×3 output
  Pool 3×3, stride 1:  (4 - 3) / 1 + 1 = 2×2 output
```

---

### Why Max Pooling Wins Over Average Pooling

Consider a feature map where a filter detected an edge:

```
┌─────┬─────┐
│  0  │  0  │
├─────┼─────┤
│  0  │  9  │
└─────┴─────┘

Max pool → 9      (preserves the detection: "an edge was found here")
Avg pool → 2.25   (dilutes the signal: the strong 9 gets averaged down)
```

```
0  0  0  0
0  8  7  0
0  0  0  0
0  0  0  0
```

Max pooling says *"I only care that this feature was detected somewhere in this region, and I want
to keep the strongest signal."* Average pooling dilutes strong activations with surrounding zeros.
That's why max pooling is standard for most of the network, and average pooling is mainly used at
the very end (global average pooling) to summarize entire feature maps.

---

### The Critical Thing: Pooling Has ZERO Learnable Parameters

This is important to remember. Compare to the other layers:

| Layer | Has Weights & Biases? | Learns During Training? |
|:---|:---:|:---:|
| Convolutional layer | ✅ Yes | ✅ Yes |
| Fully connected layer | ✅ Yes | ✅ Yes |
| Pooling layer | ❌ No | ❌ No |

Pooling is a fixed mathematical operation. It doesn't get smarter during training. It's purely
mechanical shrinking.

---

### What About Backpropagation Through Pooling?

You might wonder — during the backward pass, how does the error flow back through a max pool
layer? Since max pooling only kept one value from each window, the gradient only flows back to
the position that had the maximum value. All other positions get a gradient of zero.

```
Forward pass:                    Backward pass:
┌─────┬─────┐                    Gradient from next layer: [δ]
│  2  │  5  │
├─────┼─────┤  → max = 8        ┌─────┬─────┐
│  8  │  1  │                    │  0  │  0  │  ← zero (wasn't the max)
└─────┴─────┘                    ├─────┼─────┤
                                 │  δ  │  0  │  ← gets the gradient (was the max)
                                 └─────┴─────┘
```

Only the value that "won" (the max) receives the error signal. The rest contributed nothing to
the output, so they receive no gradient. This is sometimes called a **"routing"** of the gradient —
the max pool layer remembers which position had the max during the forward pass and routes the
gradient back to only that position.

For average pooling, the gradient is distributed equally: each position gets δ / (p × p).

```
Max pool backward:               Average pool backward:
  gradient δ at output             gradient δ at output

  Input was:  [2, 8]              Input was:  [2, 8]
              [1, 3]                          [1, 3]
  Max was 8                       Average was 3.5

  Gradient:   [0, δ]              Gradient:   [δ/4, δ/4]
              [0, 0]                          [δ/4, δ/4]
                                    ↑ gradient split equally among all 4 positions
```

---

### Layer 3: Fully Connected Layer

This is where we go back to what you already know — regular perceptrons.

We flatten the two 2×2 pooled maps into a single vector:

```
From Pooled A: [3, 1, 1, 3]
From Pooled B: [3, 3, 3, 1]

Flattened: [3, 1, 1, 3, 3, 3, 3, 1]  (8 values)
```

This connects to 2 output neurons (one for "X", one for "O"). Each neuron does exactly what a
perceptron does — weighted sum + bias + activation:

```
Neuron "X":  z = w₁(3) + w₂(1) + w₃(1) + w₄(3) + w₅(3) + w₆(3) + w₇(3) + w₈(1) + bias
Neuron "O":  z = w₁(3) + w₂(1) + w₃(1) + w₄(3) + w₅(3) + w₆(3) + w₇(3) + w₈(1) + bias

(Each neuron has its own set of 8 weights + 1 bias.)
```

After training, the "X" neuron would learn weights that respond strongly when both diagonal
feature maps are active, while the "O" neuron would learn different weights.

The final output typically uses **softmax** to convert to probabilities:

```
Output: [0.92, 0.08]  →  92% "X",  8% "O"  →  Prediction: X ✓
```

---

## Part 4: The Full Picture

Here's the complete flow of our small CNN:

```
Input (5×5)
    ↓
Conv Layer: 2 filters × (3×3) = 18 weights + 2 biases = 20 parameters
    ↓
2 Feature Maps (3×3 each)
    ↓
ReLU Activation
    ↓
Max Pooling (2×2)
    ↓
2 Pooled Maps (2×2 each) → Flatten → 8 values
    ↓
Fully Connected: 2 neurons × (8 weights + 1 bias) = 18 parameters
    ↓
Softmax → [P("X"), P("O")]
```

```
5×5 Input Image
       │
       ▼
┌─────────────────────────┐
│  CONV LAYER (2 filters) │  ← 2 filters × (3×3 weights + 1 bias) = 20 parameters
│  3×3 filters, ReLU      │
│  Output: 2 × 3×3        │
└─────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  POOLING LAYER       │  ← 0 parameters (just takes max, nothing to learn)
│  2×2 max pool        │
│  Output: 2 × 2×2     │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  FLATTEN             │  ← reshape 2×2×2 into vector of 8
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  FULLY CONNECTED     │  ← 2 neurons × (8 weights + 1 bias) = 18 parameters
│  2 output neurons    │
│  Softmax activation  │
└──────────────────────┘
       │
       ▼
   [0.92, 0.08]
   Prediction: "X"

Total trainable parameters: 20 + 18 = 38
```

A fully connected network on the same 5×5 image with a hidden layer of equivalent capacity
would need far more parameters and wouldn't understand spatial patterns.

---

## Part 5: How It Learns (Training)

Training works exactly the same way you learned with perceptrons:

1. **Forward pass** — image flows through all 3 layers, produces a prediction (weights frozen)
2. **Calculate loss** — compare prediction to the correct label (e.g., cross-entropy loss)
3. **Backward pass (backpropagation)** — error flows backward through the fully connected layer,
   through the pooling layer, through the convolutional layer, updating all weights
4. **Repeat** with the next image

The key difference: in the conv layer, backpropagation updates the filter weights. Since the
same filter was applied at every position, the gradients from every position get combined to
update those shared weights. The filter learns the pattern that's most useful across the entire
image.

### Backpropagation Through Convolution — The Details

When the error signal reaches a convolutional layer during the backward pass, two things need
to happen:

1. **Compute the gradient for the filter weights** — At every position where the filter was
   applied during the forward pass, the gradient tells us "how should this filter weight change
   to reduce the error at this position?" Since the same filter was used at all positions, we
   sum up all these gradients. The filter update is the average contribution across the entire
   image.

2. **Pass the gradient backward to the previous layer** — The previous layer needs to know how
   its outputs contributed to the error. This is computed by correlating the upstream gradient
   with the (rotated) filter — essentially a convolution with the filter flipped 180°.

```
Forward:   output     = input ★ filter               (convolution)
Backward:  ∂L/∂filter = input ★ ∂L/∂output           (gradient for filter)
           ∂L/∂input  = ∂L/∂output ★ rot180(filter)  (gradient for input)
```

The gradient accumulation across positions is what makes weight sharing work during training.
The filter gets feedback from every position simultaneously, so it learns to detect patterns
that are useful everywhere in the image — not just at one location.

---

### Batch Normalization — Stabilizing the Layers

In practice, there's a critical component that sits between the convolution and the activation
in almost every modern CNN: **batch normalization** (or **"BatchNorm"**). Our toy 3-layer example
didn't need it, but any real CNN (AlexNet onward, and especially from ResNet onward) uses it
extensively.

**The problem BatchNorm solves:** as the network trains, the output distribution of each layer
shifts constantly because the weights of earlier layers keep changing. Layer 3's inputs might
have a mean of 5.0 in one training step and a mean of -2.0 in the next. This makes it hard for
deeper layers to learn — they're constantly chasing a moving target. This phenomenon is called
**internal covariate shift**.

BatchNorm fixes this by normalizing the output of each layer to have a mean of 0 and a standard
deviation of 1, then applying a learned scale and shift.

Here's the math for a single channel of a feature map across a mini-batch:

```
Step 1: Compute the mean of the batch
    μ = (1/m) × Σ xᵢ           (m = number of examples in the mini-batch)

Step 2: Compute the variance of the batch
    σ² = (1/m) × Σ (xᵢ - μ)²

Step 3: Normalize
    x̂ᵢ = (xᵢ - μ) / √(σ² + ε)    (ε ≈ 1e-5, avoids division by zero)

Step 4: Scale and shift (LEARNABLE parameters)
    yᵢ = γ × x̂ᵢ + β
```

Steps 1–3 are fixed math — they just standardize the data. **Step 4 is the key:** γ (gamma) and β
(beta) are learnable parameters that the network adjusts through backpropagation. This lets
the network undo the normalization if it wants to. If the network learns γ = σ and β = μ, it
gets back the original values. So BatchNorm never hurts — at worst it's a no-op, and in practice
it helps enormously.

**A concrete example** with a mini-batch of 4 values at one position in one feature map:

> Inputs from batch: **[2.0, 4.0, 6.0, 8.0]**

```
Step 1: μ = (2 + 4 + 6 + 8) / 4 = 5.0

Step 2: σ² = ((2-5)² + (4-5)² + (6-5)² + (8-5)²) / 4
           = (9 + 1 + 1 + 9) / 4 = 5.0

Step 3: Normalize (ε = 0.00001):
    x̂₁ = (2 - 5) / √5.00001 ≈ -1.342
    x̂₂ = (4 - 5) / √5.00001 ≈ -0.447
    x̂₃ = (6 - 5) / √5.00001 ≈  0.447
    x̂₄ = (8 - 5) / √5.00001 ≈  1.342

Step 4: If γ = 1, β = 0 (initial values):
    Output: [-1.342, -0.447, 0.447, 1.342]  (centered at 0, spread of ~1)
```

**Where BatchNorm sits in the layer order:**

```
Convolution → BatchNorm → ReLU → Pooling
```

This is the most common pattern. Some architectures put BatchNorm after ReLU,
but before is standard and what the original paper recommends.

**Why it matters in practice:**

- **Allows much higher learning rates** — without BatchNorm, high learning rates cause the
  training to diverge. With it, you can train 5–10× faster.
- **Acts as a regularizer** — the noise from computing statistics on mini-batches (rather than
  the full dataset) has a slight regularizing effect, sometimes reducing the need for dropout.
- **Reduces sensitivity to weight initialization** — bad initial weights matter less because
  BatchNorm corrects the distribution anyway.

BatchNorm adds 2 learnable parameters per feature map (γ and β). For a layer with 64 filters,
that's 128 extra parameters — negligible compared to the filter weights.

---

### Dropout — Preventing the Network From Memorizing

Dropout is the most widely used regularization technique in CNNs. **Regularization** means
anything that prevents the network from memorizing the training data (overfitting) and
encourages it to learn patterns that generalize to new data.

The idea is absurdly simple: **during training, randomly set some neuron outputs to zero.**

```
Without dropout:                    With dropout (p=0.5):
┌─────────────┐                     ┌─────────────┐
│  0.8   1.2  │                     │  0.8   0.0  │  ← killed
│  0.3   0.5  │                     │  0.0   0.5  │  ← killed
│  0.9   0.1  │                     │  0.9   0.1  │
└─────────────┘                     └─────────────┘
```

Each neuron has a probability *p* of being "dropped" (set to zero).
The surviving neurons are scaled up by 1/(1-p) to compensate.

**Why does randomly destroying information help?**

Because it forces the network to be **redundant**. No single neuron can afford to become a critical
specialist that the network depends on entirely. Every neuron must learn to be useful even when
its neighbors are randomly absent. This prevents **co-adaptation** — where neurons develop complex,
fragile dependencies on each other that work perfectly on the training data but break on anything
new.

Think of it like a team project where members are randomly absent each day. The team can't rely
on any one person, so everyone develops broader skills and the team becomes more robust.

**The dropout rate** (how many neurons to zero out) is a hyperparameter:

| Location | Typical Dropout Rate | Why |
|:---|:---:|:---|
| Fully connected layers | p = 0.5 | Many parameters → high overfitting risk |
| Convolutional layers | p = 0.1 – 0.3 | Weight sharing already regularizes |
| During inference | p = 0 | All neurons active at test time |

- p = 0.5 is the most common default for fully connected layers (kill half the neurons)
- p = 0.1 to 0.3 is typical for convolutional layers (they have fewer parameters due to weight sharing, so they overfit less and need less dropout)
- p = 0 means no dropout at all (used during testing/inference)

**Critically:** dropout is only active during training. During inference (actually using the
model), all neurons are active. This means the model at test time is actually an **ensemble** — an
average of all the possible "thinned" networks seen during training.

```
Training:
  Forward pass 1: neurons [A, _, C, _, E] active  (B, D dropped)
  Forward pass 2: neurons [_, B, C, D, _] active  (A, E dropped)
  Forward pass 3: neurons [A, B, _, D, E] active  (C dropped)
  ... each pass trains a different "sub-network"

Inference:
  All neurons [A, B, C, D, E] active (scaled by 1-p)
  = approximate average of all sub-networks
```

**Where dropout sits in a CNN:**

```
Conv → BatchNorm → ReLU → Pool → [Dropout is rarely used here]
                                        ↓
                                   Flatten
                                        ↓
                        Fully Connected → Dropout → ReLU
                                        ↓
                        Fully Connected → Output
```

Dropout is most commonly applied after fully connected layers, since those have the most
parameters and are the most prone to overfitting. In convolutional layers, the weight sharing
already provides strong regularization, so dropout is less common there (though a variant called
**spatial dropout** drops entire feature maps instead of individual neurons and is sometimes
used in conv layers).

---

### Other Regularization Techniques in CNNs

Dropout isn't the only weapon against overfitting. Here are others you'll encounter:

**Data augmentation** — Instead of changing the network, change the training data. Randomly flip
images horizontally, rotate them slightly, crop them, adjust brightness, and so on. This
artificially expands the training set and forces the network to learn features that are invariant
to these transformations. This is arguably the **single most effective** regularization technique for
CNNs.

```
Original image → [Random flip, Random crop, Color jitter, Random rotation]
               → 4 different training examples from 1 image
```

**Weight decay (L2 regularization)** — Add a penalty to the loss function proportional to the
sum of squared weights. This discourages any single weight from becoming too large, keeping the
model simpler.

```
Loss = CrossEntropy(prediction, label) + λ × Σ wᵢ²
                                          ↑
                        penalty for large weights (λ is a small number like 0.0001)
```

**Early stopping** — Monitor the loss on a validation set (data the network doesn't train on).
When validation loss starts increasing while training loss continues decreasing, stop training.
That divergence point is where the network starts memorizing rather than learning.

```
Training loss:    ↘ ↘ ↘ ↘ ↘ ↘ ↘ ↘  (keeps going down)
Validation loss:  ↘ ↘ ↘ ↘ ↗ ↗ ↗ ↗  (starts going up here → stop!)
                          ↑
                    Best model checkpoint
```

---

## Part 6: Why This Works — The Hierarchy

This is the beautiful insight of CNNs. In a real deep CNN (say for face recognition):

```
Layer 1 filters learn:  edges, gradients, color blobs
Layer 2 filters learn:  corners, curves, textures (by combining Layer 1 features)
Layer 3 filters learn:  eyes, noses, mouths (by combining Layer 2 features)
Layer 4 filters learn:  faces (by combining Layer 3 features)
```

Each layer builds on the previous one, going from simple to complex — just like how the human
visual cortex works.

This hierarchy is **not designed by hand**. The network discovers it through backpropagation. We
just provide the architecture (conv layers stacked with pooling) and the training data. The
filters self-organize to build a feature hierarchy that solves the task.

### Receptive Field — Why Depth Creates Abstraction

A key concept that explains the hierarchy is the **receptive field** — the region of the
original input image that influences a particular neuron's output.

- A neuron in **Layer 1** (3×3 filter) sees a **3×3** patch of the input.
- A neuron in **Layer 2** (3×3 filter on Layer 1's output) sees a **5×5** patch of the original input.
- A neuron in **Layer 3** sees a **7×7** patch. And so on.

Each successive layer "sees" a larger region of the original image. Early layers have small
receptive fields and can only detect local features (edges, textures). Deeper layers have large
receptive fields and can combine those local features into global patterns (faces, objects).

With pooling layers interspersed, the receptive field grows even faster:

```
Conv(3×3) → Pool(2×2) → Conv(3×3) → Pool(2×2) → Conv(3×3)

Receptive field: 3 → 6 → 10 → 14 → 18  (grows rapidly)
```

This is why deeper CNNs can recognize increasingly complex and abstract features.

---

## Part 7: Putting It All Together — What Makes CNNs Special

CNNs work because of three key properties:

**1. Local connectivity** — Each neuron only looks at a small patch of the input, not the entire
image. This captures the fact that useful visual features are local (an edge only involves nearby
pixels).

**2. Weight sharing** — The same filter is applied at every position. This captures the fact
that useful features can appear anywhere in the image (translation equivariance).

**3. Hierarchical composition** — Multiple layers build features on top of features. Edges
become textures, textures become parts, parts become objects. This captures the compositional
nature of visual scenes.

These three properties — **locality, sharing, and hierarchy** — are what make CNNs so much more
efficient and effective than fully connected networks for visual data. They encode our knowledge
about the structure of images directly into the network architecture, so the network doesn't
have to learn this structure from scratch.

This idea — encoding known structure into the architecture — is called an **inductive bias**,
and it's one of the most important concepts in machine learning. CNNs have a spatial inductive
bias. Transformers have an attention-based inductive bias. The right inductive bias for the
right problem is often more important than the size of the model.

---

## Part 8: The Evolution — From Toy Networks to ImageNet Champions

Understanding where CNNs came from and how they evolved gives you a roadmap of which
architectures to study next and why each innovation mattered.

### LeNet-5 (1998) — Where It All Started

Yann LeCun's LeNet-5 was the first CNN to be widely deployed in the real world — AT&T used it
to read handwritten digits on bank checks. It's almost identical in structure to the toy CNN we
built in Part 3, just slightly bigger:

```
Architecture:
Input (32×32 grayscale)
  → Conv (6 filters, 5×5) → Pool (2×2 avg)
  → Conv (16 filters, 5×5) → Pool (2×2 avg)
  → FC (120) → FC (84) → Output (10 digits)

Total parameters: ~60,000
```

LeNet proved the concept — local filters, weight sharing, and stacking conv+pool layers works.
But computers were too slow and datasets too small for the idea to take off. CNNs went quiet
for over a decade.

### AlexNet (2012) — The Deep Learning Big Bang

AlexNet shattered the ImageNet competition by such a large margin that it single-handedly
reignited interest in neural networks. It wasn't architecturally revolutionary — it was
essentially a bigger LeNet. What changed was the combination of three things: a massive dataset
(ImageNet, 1.2 million images), GPU training, and ReLU activations instead of sigmoid/tanh.

```
Architecture:
Input (224×224×3 RGB)
  → Conv (96 filters, 11×11, stride 4) → Pool → Norm
  → Conv (256 filters, 5×5) → Pool → Norm
  → Conv (384, 3×3) → Conv (384, 3×3) → Conv (256, 3×3) → Pool
  → FC (4096) → Dropout → FC (4096) → Dropout → Output (1000 classes)

Total parameters: ~60 million
Key innovations: ReLU activation, dropout regularization, GPU training
```

Notice the large 11×11 filters in the first layer. This was needed because the network was
relatively shallow — it needed large filters to capture enough spatial context early on. Later
architectures would discover that many small filters are better than a few large ones.

### VGGNet (2014) — The Power of Depth and Simplicity

VGG asked a simple question: what if we used only 3×3 filters and just made the network much
deeper? The insight: two stacked 3×3 convolutions have the same receptive field as one 5×5
convolution, but with fewer parameters and more non-linearities (two ReLU activations instead
of one).

```
Two 3×3 convs:  2 × (3×3×C×C) = 18C² parameters,  receptive field = 5×5
One 5×5 conv:   1 × (5×5×C×C) = 25C² parameters,  receptive field = 5×5

→ Same receptive field, 28% fewer parameters, and an extra ReLU for free.
```

```
VGG-16 Architecture (simplified):
Input (224×224×3)
  → [Conv 3×3, 64] × 2 → Pool
  → [Conv 3×3, 128] × 2 → Pool
  → [Conv 3×3, 256] × 3 → Pool
  → [Conv 3×3, 512] × 3 → Pool
  → [Conv 3×3, 512] × 3 → Pool
  → FC (4096) → FC (4096) → Output (1000)

Total parameters: ~138 million (mostly in the FC layers)
Key insight: depth with small filters beats shallow with large filters
```

VGG is beautifully simple — just 3×3 convs and 2×2 pools stacked uniformly — which makes it
a great architecture to study. But 138 million parameters is expensive, and most of them sit in
the fully connected layers.

### GoogLeNet / Inception (2014) — Going Wider, Not Just Deeper

GoogLeNet introduced the **Inception module**, which asks: why choose a single filter size when
you can use several in parallel? Each Inception module applies 1×1, 3×3, and 5×5 convolutions
simultaneously and concatenates the results.

```
                Input
                  │
    ┌─────────┬───┴───┬─────────┐
    │         │       │         │
1×1 conv  1×1 conv  1×1 conv  3×3 pool
    │         │       │         │
    │     3×3 conv  5×5 conv  1×1 conv
    │         │       │         │
    └─────────┴───┬───┴─────────┘
                  │
             Concatenate
```

Those 1×1 convolutions before the 3×3 and 5×5 filters? That's the bottleneck trick from our
1×1 convolution section — they reduce the channel count to keep computation manageable.

```
Total parameters: ~6.8 million (23× fewer than VGG!)
Key innovation: parallel multi-scale filters, 1×1 bottlenecks, global average pooling
    (replacing the huge FC layers)
```

### ResNet (2015) — The Skip Connection Revolution

ResNet solved the **degradation problem**: counterintuitively, simply stacking more layers
eventually makes the network *worse*, not better. Beyond a certain depth, training accuracy
actually decreases. This isn't overfitting — even the training performance degrades.

The solution is the **residual connection** (or **skip connection**). Instead of learning a direct
mapping H(x), each block learns the residual F(x) = H(x) - x, so the block output is F(x) + x:

```
Standard block:              Residual block:
    Input                        Input ─────────────┐
      │                            │                │
    Conv                         Conv               │
      │                            │                │
    Conv                         Conv               │
      │                            │                │
    Output                       + ← ───────────────┘  (add the input back)
                                   │
                                 Output
```

> If the block needs to learn the **identity function** (pass input through unchanged):
> - **Standard:** must learn weights that perfectly reproduce the input → *hard*
> - **Residual:** just learn F(x) = 0 (all weights → 0) → *easy!*

This simple change allowed networks to go from 16–19 layers (VGG) to 152 layers (and beyond —
researchers have trained ResNets with over 1000 layers). The skip connections ensure that
gradients can flow directly backward through the identity path, solving the vanishing gradient
problem in very deep networks.

```
ResNet-50 Architecture (simplified):
Input (224×224×3)
  → Conv 7×7, 64, stride 2 → Pool
  → [1×1, 64  → 3×3, 64  → 1×1, 256]  × 3    (bottleneck blocks!)
  → [1×1, 128 → 3×3, 128 → 1×1, 512]  × 4
  → [1×1, 256 → 3×3, 256 → 1×1, 1024] × 6
  → [1×1, 512 → 3×3, 512 → 1×1, 2048] × 3
  → Global Average Pooling → Output (1000)

Total parameters: ~25.6 million
Key innovation: skip connections, bottleneck blocks with 1×1 convolutions
```

### The Progression at a Glance

| Year | Model | Depth | Parameters | Top-5 Error | Key Idea |
|:---:|:---|:---:|---:|:---:|:---|
| 1998 | LeNet-5 | 7 | 60K | — | First working CNN |
| 2012 | AlexNet | 8 | 60M | 16.4% | GPU + ReLU + Dropout |
| 2014 | VGGNet | 19 | 138M | 7.3% | Depth with 3×3 filters |
| 2014 | GoogLeNet | 22 | 6.8M | 6.7% | Inception modules, 1×1 conv |
| 2015 | ResNet | 152 | 25.6M | 3.6% | Skip connections |

Notice the trend: deeper networks, fewer parameters (after VGG learned that lesson), lower
error. ResNet with 152 layers has fewer parameters than VGG with 19 layers, thanks to
bottleneck blocks and global average pooling replacing fully connected layers.

Each architecture introduced a specific innovation — ReLU, small filters, parallel branches,
skip connections — that subsequent architectures adopted and built upon.

Understanding this lineage gives you a framework for evaluating any CNN architecture you
encounter: what inductive biases does it encode, and what training problems does it solve?

---
{{CNN_IMAGE}}
---
## Part 9: Transfer Learning — Standing on the Shoulders of Giants

Here's a practical reality: **almost nobody trains a CNN from scratch anymore.** Training a ResNet-50
on ImageNet from random weights takes days on expensive GPUs and requires millions of labeled
images. Most real-world problems don't have millions of labeled images. You might have 500 photos
of defective products, or 2,000 X-rays labeled by a radiologist.

**Transfer learning** solves this by reusing a CNN that was already trained on a large dataset
(almost always ImageNet) and adapting it to your specific task.

### Why Transfer Learning Works

Remember the hierarchy from Part 6:

```
Layer 1: edges, gradients, color blobs
Layer 2: corners, curves, textures
Layer 3: eyes, noses, mouths
Layer 4: faces
```

Here's the key insight: **the early layers are universal.** Edges, textures, and corners appear
in almost every image — whether you're classifying cats, tumors, or satellite photos. Only the
later layers become task-specific. So you can reuse the early layers as-is and only retrain the
later ones.

### The Two Strategies

**Strategy 1: Feature Extraction** *(freeze everything, replace the head)*

Take a pretrained ResNet-50, remove its final classification layer (which outputs 1000 ImageNet
classes), and replace it with a new layer that outputs your number of classes. Freeze all the
convolutional layers — their weights don't change at all. Only the new final layer trains.

```
Pretrained ResNet-50:
[Conv layers — FROZEN] → [Global Avg Pool] → [FC: 1000 classes — REMOVED]

Your model:
[Conv layers — FROZEN] → [Global Avg Pool] → [FC: 3 classes — TRAINABLE]

Trainable parameters: ~6,000 (instead of 25.6 million)
Training time: minutes instead of days
Required data: can work with as few as 100–500 images per class
```

The frozen conv layers act as a fixed feature extractor. They convert your image into a rich
feature vector, and the new FC layer just learns which features matter for your task.

**Strategy 2: Fine-Tuning** *(unfreeze some layers, train with a small learning rate)*

Same setup, but you also unfreeze some of the later convolutional layers and train them with a
very small learning rate. This lets the network adapt its higher-level features to your specific
domain while keeping the low-level features (edges, textures) intact.

```
[Conv layers 1-40 — FROZEN] → [Conv layers 41-50 — TRAINABLE, small lr] → [FC — TRAINABLE]

Common recipe:
  1. Freeze everything, train FC head for a few epochs
  2. Unfreeze last few conv blocks, reduce learning rate by 10×
  3. Train for more epochs
  4. Optionally unfreeze more layers, reduce lr again
```

The rule of thumb: **the more data you have, the more layers you can afford to fine-tune.** With
500 images, only train the head. With 50,000 images, fine-tune the last several blocks. With
millions, you could fine-tune everything (but you'd still benefit from starting with pretrained
weights rather than random initialization).

### When Transfer Learning Breaks Down

Transfer learning assumes your data looks at least somewhat like ImageNet (natural photographs).
If your domain is radically different — say, spectrograms, microscopy images, or radar scans —
the early features (edges and textures) are still somewhat useful, but the benefit shrinks. In
these cases, fine-tuning more layers or even training from scratch with a good architecture can
sometimes outperform transfer learning.

---

## Part 10: What Came After — CNNs Meet Transformers

CNNs dominated computer vision from 2012 to roughly 2020. Then something unexpected happened:
**Vision Transformers (ViT)** showed that the Transformer architecture — originally designed
for language — could match or beat CNNs on image tasks.

### The Core Difference

CNNs build in spatial inductive biases: locality (small filters), translation equivariance
(weight sharing), and hierarchy (stacking layers). These biases are correct for images, which
is why CNNs work so well with limited data.

Transformers have almost no spatial bias. They split the image into patches (e.g., 16×16 pixel
patches), treat each patch like a "word," and use self-attention to let every patch attend to
every other patch. There's no concept of "nearby" — patch (0,0) and patch (14,14) interact
just as easily as adjacent patches.

```
CNN approach:                          ViT approach:
Image → [small local filters]         Image → [split into 16×16 patches]
     → [hierarchy of features]             → [treat patches like tokens]
     → [classification]                    → [self-attention across all patches]
                                            → [classification]

CNN sees: local patterns first,        ViT sees: all patches simultaneously,
  builds up to global                    learns which relationships matter
```

### The Trade-Off

CNNs learn efficiently with limited data because their biases are correct — you don't need
millions of images to learn that nearby pixels matter. Transformers need much more data to
discover spatial structure from scratch (ViT was trained on 300 million images), but once
they have enough data, their flexibility lets them discover patterns that CNN architectures
can't express.

In practice, the field has converged toward **hybrid architectures** that combine both:
convolutional stems for early feature extraction (where the spatial inductive bias helps most)
with transformer blocks for later layers (where global attention is more valuable).
Architectures like ConvNeXt showed that pure CNNs can match ViT when modernized with
techniques borrowed from Transformers, and architectures like CoAtNet blend both approaches.

> **The lesson:** CNNs aren't obsolete. They remain the right choice when data is limited,
> computation is constrained, or the task is clearly spatial. Understanding CNNs deeply — as you
> now do — gives you the foundation to understand why Transformers work differently and when each
> is appropriate.

---

## Part 11: Practical Checklist — Building Your First Real CNN

Now that you understand every component, here's the practical recipe that most practitioners
follow when tackling an image classification problem:

### Step 1: Don't Train From Scratch
Start with a pretrained model (ResNet-50 is a safe default).
Use transfer learning (Strategy 1 first, then Strategy 2 if needed).

### Step 2: Prepare Your Data
- Resize all images to the same size (224×224 is standard for ImageNet-pretrained models)
- Normalize pixel values (subtract ImageNet mean, divide by ImageNet std)
- Apply data augmentation: random horizontal flip, random crop, color jitter

### Step 3: Set Up Training
- Loss function: cross-entropy for classification
- Optimizer: Adam or SGD with momentum
- Learning rate: start with 1e-3 for the head, 1e-5 for fine-tuned layers
- Batch size: as large as your GPU memory allows (32–128 is common)
- Use a learning rate scheduler (reduce lr when validation loss plateaus)

### Step 4: Regularize
- Data augmentation (most important)
- Dropout on FC layers (p=0.5)
- Weight decay (1e-4)
- Early stopping based on validation loss
- BatchNorm is already in the pretrained model

### Step 5: Monitor and Iterate
- Track training loss AND validation loss every epoch
- If training loss >> validation loss: **underfitting** → train longer, use bigger model
- If training loss << validation loss: **overfitting** → more augmentation, more dropout
- If both are high: model is too small or learning rate is wrong

### Common Failure Modes and Fixes

| Problem | Likely Cause | Fix |
|:---|:---|:---|
| Loss doesn't decrease | Learning rate too high | Reduce lr by 10× |
| Loss decreases then explodes | Learning rate too high | Reduce lr, add gradient clipping |
| Training acc high, val acc low | Overfitting | More augmentation, dropout, weight decay |
| Both accuracies low | Underfitting | Bigger model, train longer, check data |
| Training is very slow | Batch size too small | Increase batch size, use mixed precision |
| NaN in loss | Numerical instability | Check for bad data, reduce lr, add BatchNorm |

This checklist isn't specific to CNNs — it applies to most deep learning. But understanding *why*
each step works requires understanding the components we've covered: how convolutions extract
features, how pooling provides invariance, how BatchNorm stabilizes training, how dropout prevents
overfitting, and how the hierarchical structure of CNNs builds abstract representations from
simple patterns.
"""
# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
### CNN Layer Reference

| Layer Type          | Parameters                                 | Output Shape                                           | Learnable? |
|---------------------|--------------------------------------------|--------------------------------------------------------|------------|
| Convolution         | F × (K×K×C_in + 1)                         | ((W-K+2P)/S + 1) × ((H-K+2P)/S + 1) × F              | Yes        |
| Max Pooling         | 0                                          | (W/S) × (H/S) × C                                     | No         |
| Average Pooling     | 0                                          | (W/S) × (H/S) × C                                     | No         |
| Global Avg Pooling  | 0                                          | 1 × 1 × C                                             | No         |
| Flatten             | 0                                          | W × H × C  (1D vector)                                | No         |
| Fully Connected     | C_in × C_out + C_out                       | C_out                                                  | Yes        |
| Batch Normalization | 2 × C (γ and β)                            | Same as input                                          | Yes        |
| Dropout             | 0                                          | Same as input                                          | No         |

Where: F = number of filters, K = kernel size, C_in = input channels, C = channels,
       W = width, H = height, P = padding, S = stride

### Convolution Output Size Quick Reference

| Input   | Kernel | Padding | Stride | Output  |
|---------|--------|---------|--------|---------|
| 28×28   | 3×3    | 0       | 1      | 26×26   |
| 28×28   | 3×3    | 1       | 1      | 28×28   |
| 28×28   | 5×5    | 0       | 1      | 24×24   |
| 28×28   | 5×5    | 2       | 1      | 28×28   |
| 28×28   | 3×3    | 0       | 2      | 13×13   |
| 224×224 | 7×7    | 3       | 2      | 112×112 |
| 224×224 | 3×3    | 1       | 1      | 224×224 |

### Pooling Gradient Routing

| Pooling Type    | Forward Rule          | Backward Rule                                  |
|-----------------|-----------------------|------------------------------------------------|
| Max Pooling     | Keep maximum value    | Gradient → position of max only (others get 0) |
| Average Pooling | Compute mean          | Gradient ÷ (p×p) to each position equally      |
| Global Average  | Mean of entire map    | Gradient ÷ (W×H) to every position             |

### CNN Architecture Milestones

| Architecture   | Year | Depth    | Key Innovation                              | Top-5 Error (ImageNet) |
|---------------|------|----------|---------------------------------------------|------------------------|
| LeNet-5       | 1998 | 5 layers | First practical CNN (digit recognition)     | N/A                    |
| AlexNet       | 2012 | 8 layers | ReLU, dropout, GPU training                 | 15.3%                  |
| VGGNet        | 2014 | 19 layers| Small 3×3 filters only                      | 7.3%                   |
| GoogLeNet     | 2014 | 22 layers| Inception modules, global avg pooling       | 6.7%                   |
| ResNet        | 2015 | 152 layers| Skip connections (residual learning)        | 3.6%                   |
| EfficientNet  | 2019 | variable | Compound scaling (width × depth × resolution)| 2.9%                  |
"""


# =============================================================================
# SECTIONS
# =============================================================================

SECTIONS = {
    "🧩 Core Convolution Concepts": [
        "Convolution Operation",
        "Multiple Filters & Feature Maps",
        "Stride and Padding",
    ],
    "🔽 Pooling Layers": [
        "Max Pooling",
        "Average Pooling",
        "Max vs Average Pooling Comparison",
    ],
    "🏗️ Building a Complete CNN": [
        "Small CNN (X vs O Classifier)",
        "Parameter Counting",
    ],
    "🔄 Training & Backpropagation": [
        "Forward Pass Through CNN",
        "Backpropagation Through Pooling",
        "Backpropagation Through Convolution",
    ],
    "👁️ Feature Hierarchy & Receptive Field": [
        "Receptive Field Growth",
        "Feature Hierarchy Simulation",
    ],
    "📊 Full Network Demonstrations": [
        "Complete CNN Forward + Backward Pass",
        "CNN vs Fully Connected Comparison",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Runnable code snippets for each concept
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # =========================================================================
    # CORE CONVOLUTION CONCEPTS
    # =========================================================================

    "Convolution Operation": {
        "description": "The fundamental CNN operation — a filter slides across an image computing dot products at each position to produce a feature map.",
        "code":
            '''import numpy as np

def convolve2d(image, kernel, stride=1):
    """
    2D convolution (no padding).

    Slides the kernel across the image, computing element-wise
    multiply-and-sum at each position — exactly what a perceptron does,
    but applied locally and repeatedly.

    Parameters:
        image:  2D numpy array (H × W)
        kernel: 2D numpy array (K × K)
        stride: how many pixels the filter jumps between positions

    Returns:
        feature_map: 2D numpy array with convolution results
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    out_h = (img_h - k_h) // stride + 1
    out_w = (img_w - k_w) // stride + 1

    feature_map = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row = i * stride
            col = j * stride
            patch = image[row:row+k_h, col:col+k_w]
            feature_map[i, j] = np.sum(patch * kernel)

    return feature_map


# ── Demo ──────────────────────────────────────────────────
print("CONVOLUTION OPERATION — STEP BY STEP")
print("=" * 60)
print()

# 5×5 input image
image = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
])

# Vertical edge detector
kernel = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
])

print("Input Image (5×5):")
print(image)
print()
print("Filter (3×3 vertical edge detector):")
print(kernel)
print()

# Show the convolution step by step
print("Sliding the filter across the image:")
print("-" * 60)

img_h, img_w = image.shape
k_h, k_w = kernel.shape
out_h = img_h - k_h + 1
out_w = img_w - k_w + 1

for i in range(out_h):
    for j in range(out_w):
        patch = image[i:i+k_h, j:j+k_w]
        result = np.sum(patch * kernel)
        products = patch * kernel
        print(f"  Position ({i},{j}): patch={patch.flatten().tolist()}")
        print(f"    × filter = {products.flatten().tolist()}")
        print(f"    sum = {result}")
        print()

feature_map = convolve2d(image, kernel)
print("Resulting Feature Map (3×3):")
print(feature_map)
print()
print("Each value represents how much that patch matches")
print("the vertical edge pattern. 0 means no edge detected.")            '''
    },

    "Multiple Filters & Feature Maps": {
        "description": "A convolutional layer uses multiple filters, each producing its own feature map. Together they detect different patterns.",
        "code":
            '''import numpy as np

def convolve2d(image, kernel):
    """2D convolution without padding."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            result[i, j] = np.sum(image[i:i+k_h, j:j+k_w] * kernel)
    return result

# ── Demo ──────────────────────────────────────────────────
print("MULTIPLE FILTERS — DETECTING DIFFERENT FEATURES")
print("=" * 60)
print()

# Image with a clear vertical edge on the left
image = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
], dtype=float)

print("Input Image (bright left, dark right):")
print(image.astype(int))
print()

# Multiple filters
filters = {
    "Vertical Edge": np.array([[ 1,  0, -1],
                                [ 1,  0, -1],
                                [ 1,  0, -1]]),

    "Horizontal Edge": np.array([[ 1,  1,  1],
                                  [ 0,  0,  0],
                                  [-1, -1, -1]]),

    "Diagonal \\\\ ": np.array([[ 1,  0,  0],
                                 [ 0,  1,  0],
                                 [ 0,  0,  1]]),

    "Blur (average)":  np.ones((3, 3)) / 9.0,
}

for name, kernel in filters.items():
    fmap = convolve2d(image, kernel)
    max_activation = np.max(np.abs(fmap))
    print(f"Filter: {name}")
    print(f"  Kernel: {kernel.flatten().tolist()}")
    print(f"  Feature Map:")
    for row in fmap:
        print(f"    [{', '.join(f'{v:6.2f}' for v in row)}]")
    print(f"  Max activation: {max_activation:.2f}")
    if max_activation > 2:
        print(f"  ⚡ Strong response — this filter detects a pattern present in the image")
    else:
        print(f"  ○ Weak response — this pattern is not prominent in the image")
    print()

print("Each filter produces one feature map.")
print(f"This layer with {len(filters)} filters produces {len(filters)} feature maps.")
print("The next layer takes ALL these maps as input and builds on them.")            '''
    },

    "Stride and Padding": {
        "description": "Stride controls how far the filter jumps. Padding adds zeros around borders. Together they control output size.",
        "code":
            '''import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    """2D convolution with stride and zero-padding."""
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = (img_h - k_h) // stride + 1
    out_w = (img_w - k_w) // stride + 1
    result = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row = i * stride
            col = j * stride
            result[i, j] = np.sum(image[row:row+k_h, col:col+k_w] * kernel)
    return result

def output_size(input_size, kernel_size, padding, stride):
    """Compute convolution output size."""
    return (input_size - kernel_size + 2 * padding) // stride + 1

# ── Demo ──────────────────────────────────────────────────
print("STRIDE AND PADDING — CONTROLLING OUTPUT SIZE")
print("=" * 60)
print()

image = np.random.randint(0, 10, (7, 7)).astype(float)
kernel = np.ones((3, 3)) / 9.0  # 3×3 average filter

print("Input Image (7×7):")
print(image.astype(int))
print()

configs = [
    {"stride": 1, "padding": 0, "label": "No padding, stride 1 (valid)"},
    {"stride": 1, "padding": 1, "label": "Padding 1, stride 1 (same)"},
    {"stride": 2, "padding": 0, "label": "No padding, stride 2 (downsampled)"},
    {"stride": 2, "padding": 1, "label": "Padding 1, stride 2"},
]

for cfg in configs:
    s, p = cfg["stride"], cfg["padding"]
    result = convolve2d(image, kernel, stride=s, padding=p)
    expected = output_size(7, 3, p, s)
    print(f"{cfg['label']}:")
    print(f"  Formula: ({7} - {3} + 2×{p}) / {s} + 1 = {expected}")
    print(f"  Output shape: {result.shape[0]}×{result.shape[1]}")
    print()

print("Output Size Formula:")
print("  output = floor((input - kernel + 2 × padding) / stride) + 1")
print()
print("Key patterns:")
print("  padding = (kernel-1)/2, stride 1  →  output = input  (same size)")
print("  padding = 0, stride 2             →  output ≈ input/2  (halved)")            '''
    },

    # =========================================================================
    # POOLING LAYERS
    # =========================================================================

    "Max Pooling": {
        "description": "Downsampling by keeping only the maximum value in each window. Zero learnable parameters — just picks the max.",
        "code":
            '''import numpy as np

def max_pool(feature_map, pool_size=2, stride=2):
    """
    Max pooling operation.

    Rule: From each window, keep only the largest value.
    No weights, no bias, no activation. Just pick the max.

    Also returns the mask showing WHERE the max was found
    (needed for backpropagation — gradient only flows to max positions).
    """
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))
    mask = np.zeros_like(feature_map)  # for backprop

    for i in range(out_h):
        for j in range(out_w):
            r = i * stride
            c = j * stride
            window = feature_map[r:r+pool_size, c:c+pool_size]
            max_val = np.max(window)
            output[i, j] = max_val

            # Record which position had the max (for backprop)
            max_pos = np.unravel_index(np.argmax(window), window.shape)
            mask[r + max_pos[0], c + max_pos[1]] = 1

    return output, mask

# ── Demo ──────────────────────────────────────────────────
print("MAX POOLING — STEP BY STEP")
print("=" * 60)
print()

feature_map = np.array([
    [1, 3, 2, 4],
    [5, 8, 1, 3],
    [2, 1, 7, 6],
    [4, 3, 2, 9]
], dtype=float)

print("Input Feature Map (4×4):")
print(feature_map.astype(int))
print()

print("Applying 2×2 Max Pooling (stride 2):")
print("-" * 40)

pooled, mask = max_pool(feature_map, pool_size=2, stride=2)

# Show each window
windows = [
    ((0,0), "Top-left"),
    ((0,1), "Top-right"),
    ((1,0), "Bottom-left"),
    ((1,1), "Bottom-right"),
]

for (oi, oj), label in windows:
    r, c = oi * 2, oj * 2
    window = feature_map[r:r+2, c:c+2]
    print(f"  {label} window: {window.flatten().tolist()}  →  max = {np.max(window):.0f}")

print()
print("Pooled Output (2×2):")
print(pooled.astype(int))
print()

print("Max Position Mask (for backpropagation):")
print(mask.astype(int))
print("  1 = gradient flows here, 0 = gradient blocked")
print()

# Size reduction
print(f"Size reduction: {feature_map.shape} → {pooled.shape}")
print(f"Values: {feature_map.size} → {pooled.size} ({100*(1-pooled.size/feature_map.size):.0f}% reduction)")
print(f"Parameters: 0 (pooling has NOTHING to learn)")            '''
    },

    "Average Pooling": {
        "description": "Downsampling by computing the mean of each window. Used at the end of modern networks (global average pooling).",
        "code":
            '''import numpy as np

def avg_pool(feature_map, pool_size=2, stride=2):
    """
    Average pooling operation.
    Rule: From each window, compute the mean of all values.
    """
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r = i * stride
            c = j * stride
            window = feature_map[r:r+pool_size, c:c+pool_size]
            output[i, j] = np.mean(window)

    return output

def global_avg_pool(feature_map):
    """
    Global Average Pooling — collapses entire feature map to one number.
    Used in modern architectures (ResNet, GoogLeNet) to replace
    the fully connected layer before the output.
    """
    return np.mean(feature_map)

# ── Demo ──────────────────────────────────────────────────
print("AVERAGE POOLING & GLOBAL AVERAGE POOLING")
print("=" * 60)
print()

feature_map = np.array([
    [1, 3, 2, 4],
    [5, 8, 1, 3],
    [2, 1, 7, 6],
    [4, 3, 2, 9]
], dtype=float)

print("Input Feature Map (4×4):")
print(feature_map.astype(int))
print()

# 2×2 average pooling
avg_result = avg_pool(feature_map)
print("2×2 Average Pooling:")
for i in range(2):
    for j in range(2):
        r, c = i*2, j*2
        window = feature_map[r:r+2, c:c+2]
        print(f"  Window ({i},{j}): {window.flatten().tolist()}")
        print(f"    mean = ({' + '.join(str(int(v)) for v in window.flatten())}) / 4 = {np.mean(window):.2f}")
print()
print("Average Pooled Output:")
print(avg_result)
print()

# Global average pooling
gap = global_avg_pool(feature_map)
print(f"Global Average Pooling: {gap:.2f}")
print(f"  (Mean of ALL {feature_map.size} values in the feature map)")
print(f"  Reduces {feature_map.shape} → single scalar")
print()

# Backprop comparison
delta = 1.0
print("Backpropagation gradient distribution:")
print(f"  Average pool: each position gets δ/{2*2} = {delta/(2*2):.4f}")
print(f"  Global avg:   each position gets δ/{feature_map.size} = {delta/feature_map.size:.4f}")            '''
    },

    "Max vs Average Pooling Comparison": {
        "description": "Side-by-side comparison showing why max pooling preserves strong signals while average pooling dilutes them.",
        "code":
            '''import numpy as np

def max_pool(fm, p=2, s=2):
    h, w = fm.shape
    oh, ow = (h-p)//s+1, (w-p)//s+1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i,j] = np.max(fm[i*s:i*s+p, j*s:j*s+p])
    return out

def avg_pool(fm, p=2, s=2):
    h, w = fm.shape
    oh, ow = (h-p)//s+1, (w-p)//s+1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i,j] = np.mean(fm[i*s:i*s+p, j*s:j*s+p])
    return out

# ── Demo ──────────────────────────────────────────────────
print("MAX POOLING vs AVERAGE POOLING")
print("=" * 60)
print()

# Case 1: Sparse activation (edge detected in one spot)
print("Case 1: Sparse Feature Map (edge detected at one location)")
print("-" * 60)
sparse = np.array([
    [0, 0, 0, 0],
    [0, 9, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=float)

print("Feature Map:")
print(sparse.astype(int))
max_r = max_pool(sparse)
avg_r = avg_pool(sparse)
print(f"  Max pooled: {max_r.flatten().tolist()}  — strong signal PRESERVED")
print(f"  Avg pooled: {avg_r.flatten().tolist()}  — strong signal DILUTED")
print()

# Case 2: Uniform activation
print("Case 2: Uniform Feature Map (pattern everywhere)")
print("-" * 60)
uniform = np.array([
    [5, 5, 5, 5],
    [5, 5, 5, 5],
    [5, 5, 5, 5],
    [5, 5, 5, 5]
], dtype=float)

print("Feature Map:")
print(uniform.astype(int))
max_r2 = max_pool(uniform)
avg_r2 = avg_pool(uniform)
print(f"  Max pooled: {max_r2.flatten().tolist()}  — same result")
print(f"  Avg pooled: {avg_r2.flatten().tolist()}  — same result")
print()

# Case 3: Mixed
print("Case 3: Realistic Feature Map")
print("-" * 60)
realistic = np.array([
    [2, 8, 1, 0],
    [1, 3, 0, 0],
    [0, 0, 7, 1],
    [0, 0, 2, 4]
], dtype=float)

print("Feature Map:")
print(realistic.astype(int))
max_r3 = max_pool(realistic)
avg_r3 = avg_pool(realistic)
print(f"  Max pooled: {max_r3.flatten().tolist()}")
print(f"  Avg pooled: {avg_r3.flatten().tolist()}")
print()

print("Summary:")
print("  Max pooling: 'Was this feature detected ANYWHERE in this region?'")
print("  Avg pooling: 'How much of this feature is present ON AVERAGE?'")
print()
print("  Max pooling → standard for hidden layers (preserves strong signals)")
print("  Global avg  → used at the end (summarizes entire feature maps)")            '''
    },

    # =========================================================================
    # BUILDING A COMPLETE CNN
    # =========================================================================

    "Small CNN (X vs O Classifier)": {
        "description": "Complete trace of a tiny CNN classifying a 5×5 image as 'X' or 'O'. Every number is shown.",
        "code":
            '''import numpy as np

def convolve2d(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            result[i, j] = np.sum(image[i:i+k_h, j:j+k_w] * kernel)
    return result

def relu(x):
    return np.maximum(0, x)

def max_pool(fm, p=2, s=2):
    h, w = fm.shape
    oh, ow = (h-p)//s+1, (w-p)//s+1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i,j] = np.max(fm[i*s:i*s+p, j*s:j*s+p])
    return out

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

# ── Build the CNN ─────────────────────────────────────────
print("═" * 65)
print("COMPLETE CNN — X vs O CLASSIFIER")
print("═" * 65)
print()

# Input: 5×5 "X" pattern
X_image = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
], dtype=float)

print("Input: 5×5 'X' Pattern")
print(X_image.astype(int))
print()

# ── Layer 1: Convolution (2 filters) ─────────────────────
print("LAYER 1: Convolution (2 filters, 3×3)")
print("-" * 50)

filter_A = np.array([[ 1, 0, 0],
                      [ 0, 1, 0],
                      [ 0, 0, 1]], dtype=float)  # detects \\

filter_B = np.array([[ 0, 0, 1],
                      [ 0, 1, 0],
                      [ 1, 0, 0]], dtype=float)  # detects /

fmap_A = convolve2d(X_image, filter_A)
fmap_B = convolve2d(X_image, filter_B)

print("Filter A (detects \\\\ diagonals):")
print(filter_A.astype(int))
print(f"Feature Map A (3×3):  ← diagonal of 3s = strong \\\\ detection")
print(fmap_A.astype(int))
print()

print("Filter B (detects / diagonals):")
print(filter_B.astype(int))
print(f"Feature Map B (3×3):  ← anti-diagonal of 3s = strong / detection")
print(fmap_B.astype(int))
print()

# ReLU
fmap_A_relu = relu(fmap_A)
fmap_B_relu = relu(fmap_B)
print("After ReLU: (all values already positive, no change)")
print()

# ── Layer 2: Max Pooling ─────────────────────────────────
print("LAYER 2: Max Pooling (2×2, stride 2)")
print("-" * 50)

pooled_A = max_pool(fmap_A_relu, p=2, s=1)
pooled_B = max_pool(fmap_B_relu, p=2, s=1)

print(f"Pooled A ({pooled_A.shape[0]}×{pooled_A.shape[1]}):")
print(pooled_A.astype(int))
print(f"Pooled B ({pooled_B.shape[0]}×{pooled_B.shape[1]}):")
print(pooled_B.astype(int))
print()

# ── Layer 3: Fully Connected ─────────────────────────────
print("LAYER 3: Fully Connected (2 output neurons)")
print("-" * 50)

# Flatten
flat = np.concatenate([pooled_A.flatten(), pooled_B.flatten()])
print(f"Flattened input: {flat.tolist()}  ({len(flat)} values)")
print()

# Trained weights (simulated — these would be learned)
w_x = np.array([0.3, -0.1, -0.1, 0.3, 0.3, 0.3, 0.3, -0.1])
w_o = np.array([-0.2, 0.3, 0.3, -0.2, -0.2, -0.2, -0.2, 0.3])
b_x, b_o = 0.1, 0.1

z_x = np.dot(flat, w_x) + b_x
z_o = np.dot(flat, w_o) + b_o
probs = softmax(np.array([z_x, z_o]))

print(f"Neuron 'X': z = {z_x:.4f}")
print(f"Neuron 'O': z = {z_o:.4f}")
print(f"Softmax: [P(X)={probs[0]:.4f}, P(O)={probs[1]:.4f}]")
print(f"Prediction: {'X ✓' if probs[0] > probs[1] else 'O ✗'}")
print()

# ── Parameter count ───────────────────────────────────────
conv_params = 2 * (3*3 + 1)
fc_params = 2 * (len(flat) + 1)
print(f"Total parameters:")
print(f"  Conv layer: 2 filters × (9 weights + 1 bias) = {conv_params}")
print(f"  FC layer:   2 neurons × ({len(flat)} weights + 1 bias) = {fc_params}")
print(f"  Total: {conv_params + fc_params}")            '''
    },

    "Parameter Counting": {
        "description": "Compare parameter counts between CNN and fully connected architectures on the same input.",
        "code":
            '''import numpy as np

print("═" * 65)
print("PARAMETER COUNTING — CNN vs FULLY CONNECTED")
print("═" * 65)
print()

# ── Case 1: Tiny 5×5 image ──────────────────────────────
print("Case 1: 5×5 Grayscale Image")
print("-" * 50)
input_pixels = 5 * 5

# CNN approach
cnn_conv1 = 2 * (3*3*1 + 1)   # 2 filters, 3×3, 1 channel
cnn_fc = 2 * (8 + 1)           # 2 outputs from 8 flattened values

# FC approach (same number of hidden neurons as CNN has filters)
fc_hidden = 100 * (input_pixels + 1)  # 100 hidden neurons
fc_output = 2 * (100 + 1)

print(f"  CNN:  Conv = {cnn_conv1}, FC = {cnn_fc}, Total = {cnn_conv1 + cnn_fc}")
print(f"  FC:   Hidden = {fc_hidden}, Output = {fc_output}, Total = {fc_hidden + fc_output}")
print(f"  Ratio: FC uses {(fc_hidden + fc_output) / (cnn_conv1 + cnn_fc):.1f}× more parameters")
print()

# ── Case 2: MNIST 28×28 ─────────────────────────────────
print("Case 2: MNIST 28×28 Grayscale Image")
print("-" * 50)
input_pixels = 28 * 28

# CNN: Conv(32 filters, 3×3) → Pool → Conv(64, 3×3) → Pool → FC(128) → FC(10)
cnn_c1 = 32 * (3*3*1 + 1)        # 320
cnn_c2 = 64 * (3*3*32 + 1)       # 18,496
# After two 2×2 pools: 28→13→5, with 64 channels: 5*5*64 = 1600
cnn_fc1 = 128 * (1600 + 1)       # 204,928
cnn_fc2 = 10 * (128 + 1)         # 1,290
cnn_total = cnn_c1 + cnn_c2 + cnn_fc1 + cnn_fc2

# FC: 784 → 512 → 256 → 10
fc_l1 = 512 * (784 + 1)          # 401,920
fc_l2 = 256 * (512 + 1)          # 131,328
fc_l3 = 10 * (256 + 1)           # 2,570
fc_total = fc_l1 + fc_l2 + fc_l3

print(f"  CNN breakdown:")
print(f"    Conv1 (32 filters, 3×3):      {cnn_c1:>10,}")
print(f"    Conv2 (64 filters, 3×3):      {cnn_c2:>10,}")
print(f"    FC1 (1600 → 128):             {cnn_fc1:>10,}")
print(f"    FC2 (128 → 10):               {cnn_fc2:>10,}")
print(f"    Total:                         {cnn_total:>10,}")
print()
print(f"  FC breakdown:")
print(f"    Layer 1 (784 → 512):          {fc_l1:>10,}")
print(f"    Layer 2 (512 → 256):          {fc_l2:>10,}")
print(f"    Layer 3 (256 → 10):           {fc_l3:>10,}")
print(f"    Total:                         {fc_total:>10,}")
print()
print(f"  CNN uses {cnn_total/fc_total*100:.1f}% of FC parameters")
print()

# ── Case 3: ImageNet 224×224×3 ───────────────────────────
print("Case 3: ImageNet 224×224 RGB Image")
print("-" * 50)
input_pixels = 224 * 224 * 3

# First FC layer alone
fc_first = 1024 * (input_pixels + 1)
# First CNN conv layer
cnn_first = 64 * (7*7*3 + 1)

print(f"  Just the FIRST layer:")
print(f"    FC (→1024 neurons):     {fc_first:>15,} parameters")
print(f"    CNN (64 filters, 7×7):  {cnn_first:>15,} parameters")
print(f"    Ratio: {fc_first / cnn_first:,.0f}×")
print()
print("Weight sharing makes CNNs feasible on real images.")
print("Fully connected networks simply cannot scale to high-resolution input.")            '''
    },

    # =========================================================================
    # TRAINING & BACKPROPAGATION
    # =========================================================================

    "Forward Pass Through CNN": {
        "description": "Complete forward pass tracing data flow through convolution → ReLU → pooling → fully connected → softmax.",
        "code":
            '''import numpy as np

def convolve2d(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i,j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return out

def relu(x): return np.maximum(0, x)

def max_pool_with_mask(fm, p=2, s=2):
    h, w = fm.shape
    oh, ow = (h-p)//s+1, (w-p)//s+1
    out = np.zeros((oh, ow))
    mask = np.zeros_like(fm)
    for i in range(oh):
        for j in range(ow):
            r, c = i*s, j*s
            window = fm[r:r+p, c:c+p]
            out[i,j] = np.max(window)
            pos = np.unravel_index(np.argmax(window), window.shape)
            mask[r+pos[0], c+pos[1]] = 1
    return out, mask

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

# ── Forward Pass ──────────────────────────────────────────
print("═" * 65)
print("COMPLETE FORWARD PASS — DATA FLOW THROUGH CNN")
print("═" * 65)
print()

# Input
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
], dtype=float)  # Diamond/O-like pattern

print("Step 0: Input Image (5×5)")
print(image.astype(int))
print(f"  Shape: {image.shape}")
print()

# Step 1: Convolution
print("Step 1: CONVOLUTION (1 filter, 3×3)")
print("-" * 50)
kernel = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]], dtype=float)
print(f"  Filter: {kernel.flatten().tolist()}")

conv_out = convolve2d(image, kernel)
bias = 0.5
conv_out_biased = conv_out + bias
print(f"  Convolution result + bias ({bias}):")
print(f"  {conv_out_biased}")
print(f"  Shape: {conv_out_biased.shape}")
print()

# Step 2: ReLU
print("Step 2: ReLU ACTIVATION")
print("-" * 50)
relu_out = relu(conv_out_biased)
n_zeros = np.sum(relu_out == 0)
print(f"  After ReLU: {relu_out}")
print(f"  {n_zeros} values zeroed out (were negative)")
print(f"  Shape: {relu_out.shape}")
print()

# Step 3: Max Pooling
print("Step 3: MAX POOLING (2×2, stride 2)")
print("-" * 50)
pooled, mask = max_pool_with_mask(relu_out, p=2, s=2)
print(f"  Pooled: {pooled}")
print(f"  Max positions saved for backprop")
print(f"  Shape: {relu_out.shape} → {pooled.shape}")
print()

# Step 4: Flatten
print("Step 4: FLATTEN")
print("-" * 50)
flat = pooled.flatten()
print(f"  Flattened: {flat.tolist()}")
print(f"  Shape: {pooled.shape} → ({len(flat)},)")
print()

# Step 5: Fully Connected + Softmax
print("Step 5: FULLY CONNECTED + SOFTMAX")
print("-" * 50)
np.random.seed(42)
n_classes = 3
W = np.random.randn(n_classes, len(flat)) * 0.5
b = np.zeros(n_classes)
logits = W @ flat + b
probs = softmax(logits)
classes = ["Circle", "X", "Diamond"]

print(f"  Logits: {np.round(logits, 4).tolist()}")
print(f"  Softmax probabilities:")
for cls, prob in zip(classes, probs):
    bar = "█" * int(prob * 30)
    print(f"    {cls:>8}: {prob:.4f}  {bar}")
print(f"  Prediction: {classes[np.argmax(probs)]}")            '''
    },

    "Backpropagation Through Pooling": {
        "description": "Shows how gradients route through max pooling (only to max positions) vs average pooling (distributed equally).",
        "code":
            '''import numpy as np

print("═" * 65)
print("BACKPROPAGATION THROUGH POOLING LAYERS")
print("═" * 65)
print()

# ── Max Pooling Backward Pass ────────────────────────────
print("MAX POOLING — GRADIENT ROUTING")
print("-" * 50)
print()

feature_map = np.array([
    [2, 8, 1, 0],
    [1, 3, 0, 5],
    [4, 1, 7, 2],
    [0, 6, 3, 9]
], dtype=float)

print("Forward pass — Feature Map (4×4):")
print(feature_map.astype(int))
print()

# Forward: max pool
pooled = np.zeros((2, 2))
masks = np.zeros_like(feature_map)
for i in range(2):
    for j in range(2):
        r, c = i*2, j*2
        window = feature_map[r:r+2, c:c+2]
        max_val = np.max(window)
        pooled[i, j] = max_val
        pos = np.unravel_index(np.argmax(window), (2, 2))
        masks[r+pos[0], c+pos[1]] = 1
        print(f"  Window ({i},{j}): {window.flatten().tolist()} → max = {max_val:.0f} at local pos {pos}")

print()
print(f"Pooled Output: {pooled.flatten().tolist()}")
print(f"Max Position Mask:")
print(masks.astype(int))
print()

# Backward: suppose upstream gradient is
upstream = np.array([[1.0, -0.5],
                      [0.3,  2.0]])

print("Backward pass — Upstream gradient from next layer:")
print(upstream)
print()

# Route gradient to max positions only
grad_input = np.zeros_like(feature_map)
for i in range(2):
    for j in range(2):
        r, c = i*2, j*2
        window = feature_map[r:r+2, c:c+2]
        pos = np.unravel_index(np.argmax(window), (2, 2))
        grad_input[r+pos[0], c+pos[1]] = upstream[i, j]

print("Gradient routed back to feature map (max pool):")
print(grad_input)
print("  Only the MAX positions receive gradient. Others get 0.")
print()

# ── Average Pooling Backward Pass ────────────────────────
print("AVERAGE POOLING — GRADIENT DISTRIBUTION")
print("-" * 50)
print()

grad_avg = np.zeros_like(feature_map)
for i in range(2):
    for j in range(2):
        r, c = i*2, j*2
        grad_avg[r:r+2, c:c+2] = upstream[i, j] / 4.0

print(f"Upstream gradient: {upstream.flatten().tolist()}")
print(f"Each value divided by pool_size² = 4:")
print(grad_avg)
print("  Every position gets an equal share of the gradient.")
print()

print("Key difference:")
print("  Max pool:  gradient → ONLY to the position that won (sparse)")
print("  Avg pool:  gradient → distributed equally to all positions (dense)")            '''
    },

    "Backpropagation Through Convolution": {
        "description": "Shows how gradients flow backward through a convolutional layer — computing filter weight gradients and input gradients.",
        "code":
            '''import numpy as np

def convolve2d(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i,j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return out

print("═" * 65)
print("BACKPROPAGATION THROUGH CONVOLUTION")
print("═" * 65)
print()

# ── Forward pass ──────────────────────────────────────────
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=float)

output = convolve2d(image, kernel)

print("Forward Pass:")
print(f"  Input (3×3): {image.flatten().tolist()}")
print(f"  Kernel (2×2): {kernel.flatten().tolist()}")
print(f"  Output (2×2): {output.flatten().tolist()}")
print()

# ── Backward pass ─────────────────────────────────────────
# Suppose the loss gradient w.r.t. output is:
dL_dout = np.array([
    [1.0, 0.5],
    [0.2, 0.8]
])

print("Backward Pass:")
print(f"  Upstream gradient (dL/dOutput): {dL_dout.flatten().tolist()}")
print()

# 1. Gradient for kernel weights
#    dL/dK[m,n] = sum over all positions where K[m,n] was used
print("Step 1: KERNEL GRADIENT (dL/dKernel)")
print("-" * 50)
dL_dkernel = np.zeros_like(kernel)
kh, kw = kernel.shape
oh, ow = dL_dout.shape

for m in range(kh):
    for n in range(kw):
        # This kernel weight was multiplied by input[i+m, j+n]
        # at each output position (i, j)
        grad = 0
        for i in range(oh):
            for j in range(ow):
                grad += dL_dout[i, j] * image[i+m, j+n]
        dL_dkernel[m, n] = grad
        print(f"  dL/dK[{m},{n}] = Σ(upstream × input patches) = {grad:.2f}")

print(f"  Kernel gradient: {dL_dkernel.flatten().tolist()}")
print(f"  This tells us how to update the filter weights.")
print()

# 2. Gradient for input (needed to continue backprop to earlier layers)
print("Step 2: INPUT GRADIENT (dL/dInput)")
print("-" * 50)
# Full convolution of upstream gradient with rotated kernel
rotated_kernel = np.rot90(kernel, 2)  # rotate 180°
print(f"  Rotated kernel (180°): {rotated_kernel.flatten().tolist()}")

# Pad the upstream gradient
padded = np.pad(dL_dout, kh-1, mode='constant')
dL_dinput = convolve2d(padded, rotated_kernel)

print(f"  Input gradient (dL/dInput):")
print(dL_dinput)
print(f"  This gradient flows to the previous layer.")
print()

# Weight update
lr = 0.01
new_kernel = kernel - lr * dL_dkernel
print("Step 3: WEIGHT UPDATE")
print("-" * 50)
print(f"  Learning rate: {lr}")
print(f"  Old kernel: {kernel.flatten().tolist()}")
print(f"  Gradient:   {dL_dkernel.flatten().tolist()}")
print(f"  New kernel: {np.round(new_kernel, 4).flatten().tolist()}")
print()
print("Key insight: The filter gradient is the SUM of contributions")
print("from ALL positions where the filter was applied. Weight sharing")
print("means one filter gets feedback from the entire image.")            '''
    },

    # =========================================================================
    # FEATURE HIERARCHY & RECEPTIVE FIELD
    # =========================================================================

    "Receptive Field Growth": {
        "description": "Demonstrates how each successive layer 'sees' a larger region of the original input, enabling increasingly abstract features.",
        "code":
            '''import numpy as np

print("═" * 65)
print("RECEPTIVE FIELD — WHY DEEPER LAYERS SEE MORE")
print("═" * 65)
print()

def receptive_field_size(layers):
    """
    Compute receptive field size for a stack of conv/pool layers.

    Each layer is a dict with:
        type: 'conv' or 'pool'
        kernel: kernel size
        stride: stride
    """
    rf = 1   # receptive field starts at 1 pixel
    jump = 1  # cumulative stride

    for layer in layers:
        k = layer['kernel']
        s = layer['stride']
        rf = rf + (k - 1) * jump
        jump = jump * s

    return rf

print("Architecture 1: Three Conv Layers (3×3, stride 1)")
print("-" * 50)
layers_conv = [
    {'type': 'conv', 'kernel': 3, 'stride': 1},
    {'type': 'conv', 'kernel': 3, 'stride': 1},
    {'type': 'conv', 'kernel': 3, 'stride': 1},
]

rf = 1
jump = 1
for i, layer in enumerate(layers_conv):
    rf = rf + (layer['kernel'] - 1) * jump
    jump *= layer['stride']
    print(f"  After Conv {i+1}: receptive field = {rf}×{rf}")

print(f"  3 conv layers: each neuron sees a {rf}×{rf} region of the input")
print()

print("Architecture 2: Conv + Pool alternating")
print("-" * 50)
layers_pool = [
    {'type': 'conv', 'kernel': 3, 'stride': 1},
    {'type': 'pool', 'kernel': 2, 'stride': 2},
    {'type': 'conv', 'kernel': 3, 'stride': 1},
    {'type': 'pool', 'kernel': 2, 'stride': 2},
    {'type': 'conv', 'kernel': 3, 'stride': 1},
]

rf = 1
jump = 1
for i, layer in enumerate(layers_pool):
    rf = rf + (layer['kernel'] - 1) * jump
    jump *= layer['stride']
    name = f"{layer['type'].capitalize():>4}({layer['kernel']}×{layer['kernel']}, s={layer['stride']})"
    print(f"  Layer {i+1} {name}: receptive field = {rf}×{rf}")

print(f"  Pooling rapidly EXPANDS the receptive field")
print()

print("Architecture 3: VGG-style deep stack")
print("-" * 50)
vgg_layers = []
for block in range(5):
    n_convs = 2 if block < 2 else 3
    for _ in range(n_convs):
        vgg_layers.append({'type': 'conv', 'kernel': 3, 'stride': 1})
    vgg_layers.append({'type': 'pool', 'kernel': 2, 'stride': 2})

rf = 1
jump = 1
block_end = [2, 5, 9, 13, 17]  # layer indices where each block ends
for i, layer in enumerate(vgg_layers):
    rf = rf + (layer['kernel'] - 1) * jump
    jump *= layer['stride']
    if i in [b - 1 for b in block_end]:
        block_num = block_end.index(i + 1) + 1
        print(f"  After Block {block_num}: receptive field = {rf}×{rf}")

print()
print("This is why deep CNNs work:")
print("  Early layers: small RF → detect edges, textures (local features)")
print("  Mid layers:   medium RF → detect parts, shapes (regional features)")
print("  Deep layers:  large RF → detect objects, scenes (global features)")            '''
    },

    "Feature Hierarchy Simulation": {
        "description": "Simulates how simple edge detectors in Layer 1 combine into complex pattern detectors in deeper layers.",
        "code":
            '''import numpy as np

def convolve2d(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape
    out = np.zeros((ih-kh+1, iw-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out

def relu(x): return np.maximum(0, x)

print("═" * 65)
print("FEATURE HIERARCHY — FROM EDGES TO OBJECTS")
print("═" * 65)
print()

# Create a small image with a clear structure
# A simple "house" shape
image = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
], dtype=float)

print("Input Image (7×7) — a simple house shape:")
for row in image:
    print("  " + " ".join("█" if v > 0 else "·" for v in row))
print()

# ── Layer 1: Edge Detection ──────────────────────────────
print("LAYER 1: Edge Detection (basic features)")
print("-" * 50)

filters_l1 = {
    "Vertical Edge":   np.array([[ 1, 0, -1],
                                  [ 1, 0, -1],
                                  [ 1, 0, -1]]),
    "Horizontal Edge": np.array([[ 1,  1,  1],
                                  [ 0,  0,  0],
                                  [-1, -1, -1]]),
}

l1_outputs = {}
for name, f in filters_l1.items():
    fmap = relu(convolve2d(image, f))
    l1_outputs[name] = fmap
    total_activation = np.sum(fmap)
    print(f"  {name}: total activation = {total_activation:.1f}")
    for row in fmap:
        print(f"    [{', '.join(f'{v:5.1f}' for v in row)}]")
    print()

print("Layer 1 detects WHERE edges exist — but doesn't know what they form.")
print()

# ── Layer 2: Combining features ──────────────────────────
print("LAYER 2: Combining Layer 1 Features")
print("-" * 50)

# In a real CNN, Layer 2 filters operate on the STACK of Layer 1 feature maps.
# Here we simulate by checking how Layer 1 outputs combine.

v_map = l1_outputs["Vertical Edge"]
h_map = l1_outputs["Horizontal Edge"]

# A corner detector responds when both vertical AND horizontal edges are present
print("  Corner detection = Vertical edges + Horizontal edges overlap")
# Find positions where both features are active
corner_map = np.minimum(v_map[:h_map.shape[0], :h_map.shape[1]],
                         h_map[:v_map.shape[0], :v_map.shape[1]])
print(f"  Corner activations (where both edges meet):")
for row in corner_map:
    print(f"    [{', '.join(f'{v:5.1f}' for v in row)}]")
n_corners = np.sum(corner_map > 0)
print(f"  Found {n_corners} corner-like regions")
print()

print("Layer 2 combines simple features into parts (corners, T-junctions, curves).")
print("Layer 3 would combine parts into objects (roof, wall, door).")
print("Layer 4 would combine objects into scenes (house).")
print()
print("This hierarchy is NOT hand-designed. The network discovers it")
print("through backpropagation. We just stack the layers.")            '''
    },

    # =========================================================================
    # FULL NETWORK DEMONSTRATIONS
    # =========================================================================

    "Complete CNN Forward + Backward Pass": {
        "description": "Full forward and backward pass through a tiny CNN, showing weight updates at every layer.",
        "code":
            '''import numpy as np
np.random.seed(42)

print("═" * 70)
print("COMPLETE CNN — FORWARD + BACKWARD PASS WITH WEIGHT UPDATES")
print("═" * 70)
print()

# ── Tiny CNN: Conv(1 filter, 2×2) → ReLU → FC(2) → Softmax ──

# Input: 3×3 image
image = np.array([[1, 2, 0],
                   [0, 1, 3],
                   [2, 0, 1]], dtype=float)

# Conv filter (2×2)
W_conv = np.array([[1.0, 0.0],
                    [0.0, -1.0]])
b_conv = 0.0

# FC weights: 4 inputs (2×2 feature map) → 2 classes
W_fc = np.array([[ 0.5, -0.3,  0.2,  0.1],
                  [-0.2,  0.4, -0.1,  0.3]])
b_fc = np.array([0.0, 0.0])

target = 0  # class 0
lr = 0.1

print("Network: Input(3×3) → Conv(2×2) → ReLU → FC(4→2) → Softmax")
print(f"Target class: {target}")
print()

# ═══ FORWARD PASS ═══════════════════════════════════════
print("FORWARD PASS")
print("=" * 50)

# Step 1: Convolution
conv_out = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        conv_out[i,j] = np.sum(image[i:i+2, j:j+2] * W_conv) + b_conv

print(f"  Conv output (before ReLU):")
print(f"    {conv_out}")

# Step 2: ReLU
relu_out = np.maximum(0, conv_out)
relu_mask = (conv_out > 0).astype(float)
print(f"  After ReLU: {relu_out}")
print(f"  ReLU mask:  {relu_mask.astype(int)}")

# Step 3: Flatten
flat = relu_out.flatten()
print(f"  Flattened: {flat.tolist()}")

# Step 4: FC + Softmax
logits = W_fc @ flat + b_fc
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)
print(f"  Logits: {np.round(logits, 4).tolist()}")
print(f"  Softmax: {np.round(probs, 4).tolist()}")
print(f"  Loss (cross-entropy): {-np.log(probs[target]):.4f}")
print()

# ═══ BACKWARD PASS ═════════════════════════════════════
print("BACKWARD PASS")
print("=" * 50)

# Step 1: Softmax + cross-entropy gradient
dL_dlogits = probs.copy()
dL_dlogits[target] -= 1  # gradient of CE + softmax
print(f"  dL/dLogits: {np.round(dL_dlogits, 4).tolist()}")

# Step 2: FC layer gradient
dL_dW_fc = np.outer(dL_dlogits, flat)
dL_db_fc = dL_dlogits.copy()
dL_dflat = W_fc.T @ dL_dlogits
print(f"  dL/dW_fc:")
for row in dL_dW_fc:
    print(f"    {np.round(row, 4).tolist()}")
print(f"  dL/dFlat: {np.round(dL_dflat, 4).tolist()}")

# Step 3: Unflatten + ReLU backward
dL_drelu = dL_dflat.reshape(2, 2)
dL_dconv = dL_drelu * relu_mask  # ReLU backward: zero out where input was ≤ 0
print(f"  dL/dConv (after ReLU backward):")
print(f"    {np.round(dL_dconv, 4)}")

# Step 4: Conv layer gradient
dL_dW_conv = np.zeros_like(W_conv)
for m in range(2):
    for n in range(2):
        grad = 0
        for i in range(2):
            for j in range(2):
                grad += dL_dconv[i, j] * image[i+m, j+n]
        dL_dW_conv[m, n] = grad

dL_db_conv = np.sum(dL_dconv)
print(f"  dL/dW_conv: {np.round(dL_dW_conv, 4).flatten().tolist()}")
print(f"  dL/db_conv: {dL_db_conv:.4f}")
print()

# ═══ WEIGHT UPDATES ════════════════════════════════════
print("WEIGHT UPDATES (lr = {})".format(lr))
print("=" * 50)

new_W_fc = W_fc - lr * dL_dW_fc
new_b_fc = b_fc - lr * dL_db_fc
new_W_conv = W_conv - lr * dL_dW_conv
new_b_conv = b_conv - lr * dL_db_conv

print(f"  Conv filter: {W_conv.flatten().tolist()} → {np.round(new_W_conv, 4).flatten().tolist()}")
print(f"  Conv bias:   {b_conv} → {np.round(new_b_conv, 4)}")
print(f"  FC weights updated (showing first row):")
print(f"    {np.round(W_fc[0], 4).tolist()} → {np.round(new_W_fc[0], 4).tolist()}")
print()

# Verify: forward pass with new weights
conv_new = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        conv_new[i,j] = np.sum(image[i:i+2, j:j+2] * new_W_conv) + new_b_conv
relu_new = np.maximum(0, conv_new)
flat_new = relu_new.flatten()
logits_new = new_W_fc @ flat_new + new_b_fc
exp_new = np.exp(logits_new - np.max(logits_new))
probs_new = exp_new / np.sum(exp_new)

print(f"  Verification — new prediction:")
print(f"    Old P(correct): {probs[target]:.4f}")
print(f"    New P(correct): {probs_new[target]:.4f}")
print(f"    Improvement: {'YES ✓' if probs_new[target] > probs[target] else 'NO ✗'}")
print(f"    Old loss: {-np.log(probs[target]):.4f}")
print(f"    New loss: {-np.log(probs_new[target]):.4f}")            '''
    },

    "CNN vs Fully Connected Comparison": {
        "description": "Trains both a CNN and a fully connected network on the same pattern recognition task, comparing parameter efficiency and performance.",
        "code":
            '''import numpy as np
np.random.seed(42)

def convolve2d(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape
    out = np.zeros((ih-kh+1, iw-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out

def relu(x): return np.maximum(0, x)
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

# ── Create dataset ────────────────────────────────────────
print("═" * 65)
print("CNN vs FULLY CONNECTED — PATTERN RECOGNITION COMPARISON")
print("═" * 65)
print()

# Pattern A: vertical line in center
pattern_A = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
], dtype=float)

# Pattern B: horizontal line in center
pattern_B = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# Shifted versions (translation test)
pattern_A_shifted = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0]
], dtype=float)

pattern_B_shifted = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
], dtype=float)

patterns = [
    ("Vertical (center)", pattern_A, 0),
    ("Horizontal (center)", pattern_B, 1),
    ("Vertical (shifted)", pattern_A_shifted, 0),
    ("Horizontal (shifted)", pattern_B_shifted, 1),
]

print("Patterns (0 = Vertical, 1 = Horizontal):")
for name, pat, label in patterns:
    print(f"  {name} → class {label}")
print()

# ── CNN approach ──────────────────────────────────────────
print("CNN APPROACH")
print("-" * 50)

# Vertical edge detector
vert_filter = np.array([[ 1, 0, -1],
                          [ 1, 0, -1],
                          [ 1, 0, -1]], dtype=float)

# Horizontal edge detector
horiz_filter = np.array([[ 1,  1,  1],
                           [ 0,  0,  0],
                           [-1, -1, -1]], dtype=float)

cnn_params = 2 * (9 + 1)  # 2 filters × (9 weights + 1 bias)
print(f"  Conv parameters: {cnn_params}")

for name, pat, label in patterns:
    v_map = relu(convolve2d(pat, vert_filter))
    h_map = relu(convolve2d(pat, horiz_filter))
    v_score = np.sum(v_map)
    h_score = np.sum(h_map)
    pred = 0 if v_score > h_score else 1
    correct = "✓" if pred == label else "✗"
    print(f"  {name:>25}: V={v_score:5.1f}, H={h_score:5.1f} → class {pred} {correct}")

print()
print("  CNN correctly identifies patterns even when SHIFTED,")
print("  because the same filter slides over all positions.")
print()

# ── FC approach ───────────────────────────────────────────
print("FULLY CONNECTED APPROACH")
print("-" * 50)

fc_params = 25 * 10 + 10 + 10 * 2 + 2  # input→hidden→output
print(f"  FC parameters: {fc_params}")

# Train weights that memorize the center patterns
# FC learns pixel positions, not spatial patterns
W1 = np.random.randn(10, 25) * 0.1
b1 = np.zeros(10)
W2 = np.random.randn(2, 10) * 0.1
b2 = np.zeros(2)

# Simple training loop on center patterns only
for epoch in range(200):
    for name, pat, label in patterns[:2]:  # train on center only
        flat = pat.flatten()
        h = relu(W1 @ flat + b1)
        logits = W2 @ h + b2
        probs = softmax(logits)

        # Gradient
        dL = probs.copy()
        dL[label] -= 1
        dW2 = np.outer(dL, h)
        db2 = dL
        dh = W2.T @ dL
        dh *= (h > 0)
        dW1 = np.outer(dh, flat)
        db1 = dh

        W2 -= 0.1 * dW2
        b2 -= 0.1 * db2
        W1 -= 0.1 * dW1
        b1 -= 0.1 * db1

print("  After training on CENTER patterns:")
for name, pat, label in patterns:
    flat = pat.flatten()
    h = relu(W1 @ flat + b1)
    logits = W2 @ h + b2
    probs = softmax(logits)
    pred = np.argmax(probs)
    correct = "✓" if pred == label else "✗"
    conf = probs[pred] * 100
    print(f"  {name:>25}: pred={pred}, conf={conf:5.1f}% {correct}")

print()
print("Summary:")
print(f"  CNN:  {cnn_params} params, handles shifts naturally (weight sharing)")
print(f"  FC:   {fc_params} params, struggles with shifted patterns")
print("  CNNs build spatial understanding into the architecture.")
print("  FC networks must learn it from scratch — and often fail.")            '''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────
# def get_content():
#     """Return all content for this topic module."""
#     return {
#         "theory": THEORY,
#         "theory_raw": THEORY,
#         "interactive_components": [],
#         "complexity": COMPLEXITY,
#         "operations": OPERATIONS,
#     }


def get_content():
    """Return all content for this topic module."""
    from Required_Images.cnn_visual import CNN_VISUAL_HTML , CNN_VISUAL_HEIGHT

    # Fallback static image
    cnn_img = _image_to_html(
        "Required_Images/cnn_visual.png",
        alt="CNN Explanation",
        width="50%"
    )
    theory_with_images = THEORY.replace("{{CNN_IMAGE}}", cnn_img)

    # Interactive component data
    interactive_components = [
        {
            "placeholder": "{{CNN_IMAGE}}",
            "html": CNN_VISUAL_HTML,
            "height": CNN_VISUAL_HEIGHT,
        }
    ]

    return {
        "theory": theory_with_images,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }
