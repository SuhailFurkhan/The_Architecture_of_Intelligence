from Required_Images import *
"""
Multi Level Perceptron (MLP) - The simplest neural network unit
============================================

A Multi Layer Neural Network unit.

"""
import base64
import os

TOPIC_NAME = "MultiLayer Perceptron"

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
### Multi-Layer Perceptron (MLP) — Complete Conceptual Breakdown ?

## **Part 1: Why Do We Need Multiple Layers ?**

The Single Perceptron's Fatal Flaw
As we demonstrated, a single perceptron can only draw a single straight line to separate data into two classes. 
This works for AND, OR, and NAND because in each case, you can draw one line that puts the 0s on one side and 1s on the other.

But XOR breaks this. Plot the XOR truth table on a graph:

      x2
      |
      |
    1 |  (0,1)=1      (1,1)=0
      |
    0 |  (0,0)=0      (1,0)=1
      +------------------------
         0              1     x1

The 1's are at top-left and bottom-right. The 0s are at bottom-left and top-right. 
No single straight line can separate them. Try any line you want — it will always misclassify at least one point.

This isn't just an academic curiosity. 
Most real-world problems are non-linear: recognizing faces, understanding language, predicting stock movements, diagnosing diseases. 
If neural networks could only draw straight lines, they'd be useless for almost everything interesting.

The insight is that while one line can't solve XOR, two lines can.
If you draw two lines across the graph, you can carve out a region that isolates the 1s from the 0s. 
But a single perceptron can only draw one line. So what if we use multiple perceptrons working together?



### Why XOR Fails — The Geometric Impossibility

But a single perceptron CANNOT learn XOR (exclusive or), because XOR is not linearly separable. 
This was famously pointed out by Minsky and Papert in 1969, and it temporarily halted neural network research for years (the so-called "AI Winter"). 
XOR Truth Table:

                                XOR Gate
                                (Exactly ONE input must be 1)

                                x₁   x₂   →   Output
                                --------------------
                                0    0        0
                                0    1        1
                                1    0        1
                                1    1        0

To see why this is impossible, plot the four points on a 2D grid:

                                x₂
                                |
                                |
                                |
                                |
                                1 |  ●(0,1)=1      ○(1,1)=0
                                |
                                0 |  ○(0,0)=0      ●(1,0)=1
                                |___________________________
                                0              1         x₁


The "1" outputs (●) sit at (0,1) and (1,0) — diagonally opposite each other. The "0" outputs (○) sit at (0,0) and (1,1) — also diagonally opposite. 

Try to draw a single straight line that puts all the ●'s on one side and all the ○'s on the other. 
You can't. No matter how you orient or position the line, it will always misclassify at least one point. 
This is the fundamental limitation of a single perceptron, and it's the geometric proof that XOR is not linearly separable.


### Simplified AND and XOR Breakdown    

I get that a single perceptron can solve and and or logic but why does it fail to XOR 

I am having a hard time understanding 
please simplify it for me so i can understand what was the need for

3:49 PM

Synthesized perceptron limitations to demystify XOR problem

Great question! Let me break this down as simply as possible.

What a perceptron actually does: It draws a single straight line to separate two groups (0s and 1s).

Think of it like this — plot the inputs on a 2D grid:

    AND:
        Input	Output
        (0,0)	0
        (0,1)	0
        (1,0)	0
        (1,1)	1

Only one corner is "1." You can easily draw a single line that puts (1,1) on one side and everything else on the other. ✅

    OR:
        Input	Output
        (0,0)	0
        (0,1)	1
        (1,0)	1
        (1,1)	1
Only one corner is "0." Again, one straight line separates it. ✅

    XOR:
        Input	Output
        (0,0)	0
        (0,1)	1
        (1,0)	1
        (1,1)	0
Now the 1s are at opposite corners and the 0s are at the other opposite corners. 
Try drawing one straight line that puts (0,1) and (1,0) on one side, and (0,0) and (1,1) on the other — you can't. They're diagonally mixed up. 

This is called being not linearly separable.

So what's the fix? You need more than one line, which means more than one perceptron. 
That's exactly what a multi-layer perceptron (MLP) gives you — a hidden layer that can draw multiple lines and then combine them. 
Essentially, the hidden layer breaks XOR into sub-problems that are linearly separable (like AND and OR), and the output layer combines those results.

This was actually a huge deal historically — Minsky & Papert pointed out this XOR limitation in 1969, which temporarily killed interest in neural networks. 
It wasn't until backpropagation and multi-layer networks became practical that the field took off again.

XOR doesn't train one perceptron "for" specific points directly. Instead, it decomposes XOR into two gates whose combined result gives XOR:
XOR = AND ( OR(x₁,x₂), NAND(x₁,x₂) )

XOR via OR, NAND, AND

                ┌───────┬───────┬──────────────┬────────────────┬──────────────┐
                │ Input │ Input │ OR (Neuron 1)│ NAND (Neuron 2)│ AND (Output) │
                │   A   │   B   │              │                │   = XOR      │
                ├───────┼───────┼──────────────┼────────────────┼──────────────┤
                │   0   │   0   │      0       │      1         │      0       │
                │   0   │   1   │      1       │      1         │      1       │
                │   1   │   0   │      1       │      1         │      1       │
                │   1   │   1   │      1       │      0         │      0       │
                └───────┴───────┴──────────────┴────────────────┴──────────────┘

So the hidden layer has two neurons:

Neuron 1 (OR)   : draws a line that filters out (0,0) — it says "at least one input must be 1"
Neuron 2 (NAND) : draws a line that filters out (1,1) — it says "they can't both be 1"

Then the output neuron (AND) combines them: "both conditions must be true." Only (0,1) and (1,0) pass both filters.

Here's how it flows:

{{MLP_IMAGE}}



The Solution: Stacking Layers
The insight is that while one line can't solve XOR, two lines can. If you draw two lines across the graph, 
you can carve out a region that isolates the 1s from the 0s. But a single perceptron can only draw one line. 

So what if we use multiple perceptrons working together?
This is the Multi-Layer Perceptron. Instead of one neuron making the final decision alone, we have a team:
•	A hidden layer of neurons that each draw their own line, creating intermediate signals
•	An output layer that combines those intermediate signals to make the final decision
The hidden layer doesn't solve the problem directly. 
It transforms the data into a new representation where the problem becomes linearly separable. 
Then the output layer draws a single line in that transformed space.


Part 2: Architecture
The Three Types of Layers
             
             
            INPUT LAYER          HIDDEN LAYER           OUTPUT LAYER
         (not real neurons)   (computation happens)    (final result)
                                
                                ┌────────┐
             x1     ───────▶    │  H1    │
                                ├────────┤
             x2     ───────▶    │  H2    │  ───────▶   ┌────────┐
                                ├────────┤             │  O1    │ ───▶ output
             x3     ───────▶    │  H3    │  ───────▶   └────────┘
                                └────────┘             
                                            
             x1 ─────────┐    ┌────┐
                         ├────┤ H1 ├────┐
             x2 ─────────┤    └────┘    │    ┌────┐
                         ├────┤ H2 ├────┼────┤ O1 ├──── output
             x3 ─────────┤    └────┘    │    └────┘
                         ├────┤ H3 ├────┘
                         │    └────┘

Input Layer — This isn't actually a layer of neurons. It's just your raw data being fed in. 
If you're classifying a 28×28 pixel image, the input layer has 784 values (one per pixel). 
If you're looking at two features like in XOR, it has 2 values. The input layer does no computation at all.

Hidden Layer(s) — These are the real workhorses.
Each neuron in a hidden layer receives input from every neuron in the previous layer (this is called "fully connected" or "dense"). 
It computes a weighted sum, adds its bias, and passes it through an activation function — exactly like a single perceptron. 
The word "hidden" simply means we don't directly observe these neurons' outputs. They're internal to the network.
We don't tell them what to output. They figure out useful intermediate representations on their own during training.

Output Layer — The final layer that produces the network's answer. 
For binary classification (yes/no), this is typically one neuron with sigmoid activation (outputting a probability between 0 and 1). 
For multi-class classification (cat/dog/bird), you'd have one neuron per class, usually with softmax activation.

"Fully Connected" Means Everything Connects to Everything
In an MLP, every neuron in one layer connects to every neuron in the next layer. 
If you have 3 inputs and 4 hidden neurons, that's 3 × 4 = 12 connections (weights), plus 4 biases — 16 learnable parameters just for that one layer.

This scales quickly. A network with architecture [784, 256, 128, 10] (typical for digit recognition) has:
•	Layer 1: 784 × 256 = 200,704 weights + 256 biases
•	Layer 2: 256 × 128 = 32,768 weights + 128 biases
•	Layer 3: 128 × 10 = 1,280 weights + 10 biases
•	Total: 235,146 learnable parameters

And that's a small network. GPT-scale models have billions.

Depth vs Width
Width = how many neurons per layer. More width means each layer can detect more features simultaneously.
Depth = how many layers. More depth means the network can build increasingly abstract representations — each layer refines the work of the previous one.
For example, in an image recognition network:
•	Layer 1 might detect edges and simple textures
•	Layer 2 combines edges into shapes (circles, corners, curves)
•	Layer 3 combines shapes into parts (eyes, noses, wheels)
•	Layer 4 combines parts into objects (faces, cars, dogs)
Each layer builds on the previous layer's output to create a more abstract, more useful representation. This hierarchical feature learning is what makes deep networks so powerful.

Part 3: The Forward Pass (Making a Prediction)
What Happens When Data Enters the Network
The forward pass is conceptually simple: data flows through the network, layer by layer, from input to output. No weights change. Nothing is learned. It's pure computation.
Let's trace a concrete example through a [2, 2, 1] network (2 inputs, 2 hidden neurons, 1 output neuron):
Input: [1, 0]

STEP 1: Input reaches Hidden Layer
    Every hidden neuron receives ALL inputs.

    Hidden Neuron 1:
        z = (1 × w1) + (0 × w2) + bias
        z = (1 × 0.3) + (0 × -0.2) + 0.1 = 0.4
        output = sigmoid(0.4) = 0.598

    Hidden Neuron 2:
        z = (1 × w3) + (0 × w4) + bias
        z = (1 × 0.5) + (0 × 0.4) + (-0.3) = 0.2
        output = sigmoid(0.2) = 0.550

STEP 2: Hidden Layer outputs reach Output Layer
    The output neuron receives the HIDDEN LAYER'S outputs, not the original inputs.

    Output Neuron:
        z = (0.598 × w5) + (0.550 × w6) + bias
        z = (0.598 × 0.7) + (0.550 × -0.6) + 0.2 = 0.289
        output = sigmoid(0.289) = 0.572

FINAL OUTPUT: 0.572
Notice something critical: the output neuron never sees the original input [1, 0]. It only sees [0.598, 0.550], 
which is the hidden layer's transformation of that input. The hidden layer has created a new representation, 
and the output layer works with that new representation.

This is the entire mechanism by which the network handles non-linearity. 
The hidden layer warps, bends, and transforms the input space. The output layer then makes a simple linear decision in the warped space.

Why Sigmoid and Not Step Function?

In the single perceptron, we used a step function (output is either 0 or 1). 
In an MLP, we use sigmoid (output is a smooth value between 0 and 1). The reason is purely about what happens in the backward pass.
Training uses calculus (specifically, derivatives) to figure out how to adjust weights.

A derivative answers the question: 

"If I nudge this weight slightly, how much does the output change?" 
The step function has a derivative of 0 everywhere (the output doesn't change for tiny nudges) except at exactly z = 0 (where it jumps infinitely). 
This gives the training algorithm no useful information.

Sigmoid is smooth and has a meaningful derivative everywhere. 
A small nudge to any weight produces a small, measurable change in the output. 
This gives the training algorithm a clear signal about which direction to adjust each weight.

### Part 4: The Backward Pass (Backpropagation)

**The Central Problem**

After the forward pass, we compare the prediction to the expected output and get an error.

**We know the output neuron was wrong. But how do we fix the hidden neurons?**

We didn't tell the hidden neurons what their output should be. We have no "expected output" for them. 
We only know what the final output should have been. So how does a hidden neuron know whether to increase or decrease its weights?
This is the problem that backpropagation solves, and it's the most important algorithm in all of deep learning.

**The Core Idea: Blame Flows Backward**
    Backpropagation is based on the chain rule from calculus. The intuition is:
    If neuron A feeds into neuron B, and neuron B made a mistake, then neuron A is partially to blame — in proportion to how much it influenced B's output.
    More specifically, A's blame depends on two things:
    1.	The strength of the connection between A and B (the weight). If the weight is large, A had a big influence on B, so A deserves more blame.
    2.	How much B itself was to blame (B's delta). If B was heavily responsible for the error, and A influenced B, then A shares in that responsibility.
    
    This logic chains backward through every layer: output layer neurons know their error directly, then they send blame backward to the hidden layer, weighted by the connecting weights.

**The Two Different Delta Calculations**

* Output Layer neurons — they can compute their error directly:
* delta = (expected - actual_output) × sigmoid_derivative(actual_output)

The first term is simply how wrong they were. 
The second term (sigmoid derivative) scales the correction based on the neuron's confidence. 
If the neuron was very confident (output near 0 or 1), the derivative is small, meaning small corrections. 
If the neuron was uncertain (output near 0.5), the derivative is larger, meaning bigger corrections. 
This makes intuitive sense — confident predictions need small tweaks, uncertain predictions need larger adjustments.

Hidden Layer neurons — they can't compute their error directly because they have no expected value. Instead, they receive blame from the next layer:

    downstream_error = Σ (next_layer_neuron_delta × weight_connecting_to_this_neuron)
    delta = downstream_error × sigmoid_derivative(this_neuron_output)

Each neuron in the next layer sends back a portion of its delta, scaled by the weight of the connection. 
If hidden neuron H1 connects to output neuron O1 with a weight of 0.7, and O1's delta is 0.1, then the blame flowing back to H1 through that connection is 0.1 × 0.7 = 0.07.
If H1 connects to multiple neurons in the next layer, all the blame signals are summed up.
Then this sum is multiplied by H1's own sigmoid derivative to get H1's final delta.

**Weight Updates (Same Formula Everywhere)**

Once every neuron has its delta, the weight update is identical to the single perceptron — just using delta instead of raw error:

                new_weight = old_weight + (learning_rate × neuron_delta × input_to_this_neuron)
                new_bias = old_bias + (learning_rate × neuron_delta)
                
This formula is the same for every weight in every layer. 
The only thing that differs is how the delta was calculated (directly for output neurons, propagated backward for hidden neurons).

    The Complete Cycle
    1. FORWARD PASS:
       Data flows input → hidden → output
       All weights are FROZEN
       Each neuron caches its output (needed later)
    
    2. ERROR COMPUTATION:
       Compare output to expected value
       Calculate loss (e.g., mean squared error)
    
    3. BACKWARD PASS:
       Compute output layer deltas (from error directly)
       Compute hidden layer deltas (from downstream blame)
       Update ALL weights and biases using deltas
    
    4. REPEAT for the next training example
    One complete pass through all training examples is called an epoch. Training typically runs for hundreds or thousands of epochs until the loss stops decreasing.

### Part 5: What the Hidden Layer Actually Learns

**The Transformation Insight**

This is perhaps the most profound concept in all of deep learning.
The hidden layer doesn't learn "rules" or "if-then conditions." It learns a coordinate transformation — it warps the input space so that previously inseparable data becomes separable.

For XOR, the original input space looks like this:
            
            
            Original Space:               After Hidden Layer:
              x2                            h2
               |                            |
               |                            |
               |                            | 
             1 |  1      0                1 |  (0,1) and (1,0) are now
               |                            |  on one side
             0 |  0      1                0 |  (0,0) and (1,1) are now
               +----------------→ x1        |  on the other side
               0         1                  +--------------------→ h1

            Can't draw one line               CAN draw one line!

The hidden layer has remapped the four points so that the two classes are now on opposite sides of a line. 
The output neuron then just draws that line — the same thing a single perceptron does. 
The hidden layer made the problem simple enough for a single perceptron to solve.

**In Deeper Networks, This Happens Repeatedly**
Each hidden layer transforms the output of the previous layer into a progressively more useful representation:
                
                Raw pixels → edges → shapes → parts → objects
                
Or for text:
                Characters → words → phrases → meaning → sentiment
                
Each transformation makes the data a little more organized, a little more structured, a little more useful for the final prediction. 
By the time the data reaches the output layer, the problem has been transformed from something impossibly complex (millions of raw pixel values) 
into something almost trivially simple (a handful of high-level features that clearly indicate the answer).

This is often described as representation learning or feature learning — the network doesn't need hand-crafted features. 
It learns to create its own features, layer by layer, from raw data

### Part 6: Loss Functions — Measuring "How Wrong"

**Why We Need a Loss Function**

In the single perceptron, the error was simple: expected - predicted, which was either -1, 0, or 1. 
In an MLP with sigmoid outputs, the prediction is a continuous value like 0.572 when it should be 1. 
We need a way to quantify exactly how wrong the network is — that's what a loss function does.

**Mean Squared Error (MSE)**
The simplest and most intuitive loss function:
            
            Loss = (1/n) × Σ(expected - predicted)²
            
If the expected output is 1 and the network predicts 0.572:

            Loss = (1 - 0.572)² = (0.428)² = 0.183

Squaring serves two purposes. It makes all errors positive (being wrong by +0.5 and -0.5 are equally bad). 
And it penalizes large errors disproportionately — an error of 0.8 is not twice as bad as 0.4, it's four times as bad (0.64 vs 0.16). 
This pushes the network to fix its worst predictions first.

**Binary Cross-Entropy (Log Loss)**
For classification problems, this is more commonly used:
            
            Loss = -(expected × log(predicted) + (1-expected) × log(1-predicted))
            
This punishes confident wrong predictions extremely harshly. If the network says "I'm 99% sure this is a 1" and it's actually a 0, the loss is enormous. 
If it says "I'm 51% sure this is a 1" and it's wrong, the loss is small. 
This drives the network toward calibrated confidence — it learns not just to get the right answer, but to know how sure it is.

**The Loss Landscape**
Imagine the loss as a landscape in high-dimensional space, where each dimension is one weight in the network. 
The height at any point is the loss at that particular combination of weights. 
Training is like dropping a ball onto this landscape and letting it roll downhill. 
The ball (the network's weights) is trying to find the lowest valley (the combination of weights that minimizes the loss).

This landscape is incredibly complex for real networks — with millions of dimensions, full of hills, valleys, saddle points, and flat plateaus. 
The art and science of training neural networks is largely about navigating this landscape efficiently.


### Part 7: Gradient Descent — Navigating the Loss Landscape

**The Core Idea**

We want to adjust the weights to reduce the loss. But which direction should we move them? The answer comes from the gradient — the derivative of the loss with respect to each weight.
The gradient tells you: "If I increase this weight by a tiny amount, does the loss go up or down, and by how much?"
•	If the gradient is positive, increasing the weight increases the loss → move the weight down
•	If the gradient is negative, increasing the weight decreases the loss → move the weight up
•	If the gradient is large, the loss is very sensitive to this weight → take a bigger step
•	If the gradient is small, the loss barely changes → take a smaller step

The weight update rule is:
        
        weight = weight - learning_rate × gradient

The negative sign ensures we move opposite to the gradient (downhill, not uphill). This is gradient descent.


### Backpropagation IS Gradient Computation

Backpropagation is not a separate algorithm from gradient descent. 
Backpropagation is the method by which we efficiently compute the gradient for every weight in the network. 
Gradient descent then uses those gradients to update the weights.

The delta values we computed earlier? Those ARE the gradients (technically, they're the key intermediate values used to compute the gradients). 
The weight update formula weight += learning_rate × delta × input is gradient descent applied to each weight.

**Stochastic Gradient Descent (SGD)**
In our script, we update weights after every single training example. 
This is called Stochastic Gradient Descent — the gradient is computed from one example at a time, which is noisy but fast.

**Alternatives include:**
    Batch Gradient Descent — compute the gradient over ALL training examples, average them, then make one update. More stable but slow and memory-intensive.
    Mini-Batch Gradient Descent — compute the gradient over a small batch (e.g., 32 or 64 examples), then update. This is what virtually all modern deep learning uses. It balances stability and speed.

**The Learning Rate Problem**
The learning rate controls step size. This single hyperparameter has an enormous impact:
    
    Too high — the network overshoots, bouncing around the loss landscape without settling into a valley. Loss may oscillate or even increase.
    Too low — the network makes tiny steps, taking forever to converge. It might also get stuck in shallow local minima instead of finding deeper, better valleys.
    Just right — the network descends smoothly and converges to a good solution. Finding this sweet spot is one of the most common practical challenges.

Modern optimizers like Adam, RMSProp, and AdaGrad adaptively adjust the learning rate per-weight during training, reducing the need to manually tune this hyperparameter.
They essentially give each weight its own learning rate that changes over time based on the history of gradients for that weight.

### Part 8: Common Problems and Their Solutions

**The Vanishing Gradient Problem**
In deep networks with sigmoid activation, each layer multiplies the gradient by the sigmoid derivative (maximum value: 0.25). 
After 10 layers, the gradient has been multiplied by 0.25 ten times: 0.25¹⁰ ≈ 0.000001. 
By the time the error signal reaches the first hidden layer, it's essentially zero. The early layers stop learning.

**Solutions:**
•	ReLU activation — its derivative is either 0 or 1, so gradients don't shrink as they propagate backward
•	Residual connections (ResNets) — shortcut connections that let gradients skip layers entirely
•	Batch normalization — normalizes activations within each layer, keeping gradients healthy
•	Better initialization (He, Xavier) — starts the network in a state where gradients flow well

**The Exploding Gradient Problem:**
The opposite: gradients grow exponentially as they propagate backward, causing massive weight updates that destabilize training. Weights shoot to extreme values, outputs become NaN.

Solutions:
•	Gradient clipping — if the gradient exceeds a threshold, scale it down
•	Proper initialization — prevents the conditions that cause explosion
•	Batch normalization — keeps intermediate values in a reasonable range

**Overfitting**
The network memorizes the training data instead of learning general patterns. It performs perfectly on training data but poorly on new data.

Solutions:
•	More training data — harder to memorize a larger dataset
•	Dropout — randomly disable neurons during training, forcing the network to not rely on any single neuron
•	Regularization (L1/L2) — add a penalty term to the loss that discourages large weights
•	Early stopping — stop training when performance on a validation set stops improving
•	Data augmentation — artificially expand the dataset with transformed versions of existing data

**Dead Neurons (ReLU-specific)**
ReLU outputs 0 for any negative input. If a neuron's weights shift such that its input is always negative, it permanently outputs 0 and stops learning (its gradient is permanently 0). 
The neuron is "dead."

Solutions:
•	Leaky ReLU — outputs a small value (e.g., 0.01x) instead of 0 for negative inputs
•	Parametric ReLU (PReLU) — the slope for negative values is a learnable parameter
•	GELU, Swish — smooth approximations of ReLU that don't have the hard zero cutoff

### Part 9: From MLP to Modern Deep Learning

**What the MLP Established**
Every concept we've covered — weighted sums, activation functions, forward pass, backpropagation, gradient descent, 
loss functions, hidden representations — is present in every modern neural network architecture. The MLP established the blueprint.

**What Changed in Modern Architectures**
Modern architectures didn't replace the MLP's principles. They added structural assumptions about the data to make learning more efficient:
    * Convolutional Neural Networks (CNNs) — assume spatial structure in images. 
    Instead of connecting every neuron to every input, neurons only look at small local patches. 
    This dramatically reduces parameters and forces the network to learn spatial features like edges and textures.
    
    * Recurrent Neural Networks (RNNs) — assume sequential structure in data (text, time series). 
    The same weights are reused at each time step, and hidden states carry information forward through the sequence.
    
    * Transformers — use attention mechanisms to let every element in a sequence directly interact with every other element, regardless of distance. 
    This solved the long-range dependency problem that plagued RNNs and is the architecture behind GPT, BERT, and every modern large language model.
    
But inside every one of these architectures, at every layer, the basic operation is still: weighted sum → activation → output. The perceptron's DNA runs through all of modern AI.




### NEURAL NETWORK COMPARISON
    |------------------------------|----------------------------------------------------|
    |SINGLE PERCEPTRON             |             MULTI-LAYER PERCEPTRON (MLP)           |
    |------------------------------|----------------------------------------------------|
    |Structure:                    |            Structure:                              |
    |x → [ Perceptron ] → y        |            x → [ Hidden Layer(s) ] → [ Output ]    |
    |------------------------------|----------------------------------------------------|
    |Layers:                       |                                                    |
    |• 1 layer (input → output)    |            • 2+ layers (input → hidden → output)   |
    |------------------------------|----------------------------------------------------|
    |Activation:                   |                                                    |
    |• Step function               |            • Sigmoid, ReLU, Tanh, Softmax          |
    |------------------------------|----------------------------------------------------|
    |Learning:                     |                                                    |
    |• Simple error update         |            • Backpropagation (chain rule)          |
    |  Δw = η · error · x          |            ∂L/∂w flows backward                    |
    |------------------------------|----------------------------------------------------|
    |Patterns learned:             |                                                    |
    |• Linear only                 |            • Linear + Non-linear                   |
    |------------------------------|----------------------------------------------------|
    |Error flow:                   |                                                    |
    |• Output only                 |            • Output → Hidden → Input               |
    |------------------------------|----------------------------------------------------|
    |Hidden representations:       |                                                    |        
    |• None                        |            • Learned feature representations       |
    |------------------------------|----------------------------------------------------|
    |Parameters:                   |                                                    |
    |• One set of weights + bias   |            • Weights + bias per neuron per layer   |
    |------------------------------|----------------------------------------------------|
    |Limitation:                   |                                                    |
    |• Cannot solve XOR            |           • Can solve XOR and complex patterns     |
    |------------------------------|----------------------------------------------------|
    |Modern relevance:             |                                                    |
    |• Conceptual foundation       |            • Backbone of Deep Learning             |
    |------------------------------|----------------------------------------------------|
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """

| Concept                | Single Perceptron         | Multi-Layer Perceptron                         |
|------------------------|---------------------------|------------------------------------------------|
| Layers                 | 1 (input → output)        | 2+ (input → hidden(s) → output)                |
| Activation             | Step function             | Sigmoid, ReLU, etc.                            |
| Learning               | Simple error rule         | Backpropagation (chain rule)                   |
| Can learn              | Linear patterns only      | Non-linear patterns                            |
| Error flow             | Direct (output only)      | Backward through all layers                    |
| Hidden representations | None                      | Learned automatically                          |
| Parameters             | Weights + bias            | Weights + bias (per neuron, per layer)         |
| Limitation             | Can't solve XOR           | Can solve XOR and far more complex problems    |
| Modern relevance       | Conceptual foundation     | Direct ancestor of all deep learning           |

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Multi-Layer Perceptron (MLP) ": {
        "description": "Multi Percepton Implementation",
        "runnable": True,
        "code":

r'''
"""
================================================================================
MULTI-LAYER PERCEPTRON (MLP) IMPLEMENTATION FROM SCRATCH
================================================================================

The single perceptron could NOT learn XOR because XOR is not linearly separable.
The solution? Stack multiple perceptrons into layers. This is the Multi-Layer
Perceptron — the foundation of deep learning.

Architecture (2 inputs → 2 hidden neurons → 1 output):

    INPUT LAYER          HIDDEN LAYER           OUTPUT LAYER
    (raw data)      (learns intermediate         (final
                       representations)          prediction)

                      ┌───────────┐
     x1 ───────────┬──┤ Neuron H1 ├──┬
                   ╳  └───────────┘  ╳  ┌───────────┐
                   ╳  ┌───────────┐  ╳──┤ Neuron O1 ├──► output
     x2 ───────────┴──┤ Neuron H2 ├──┘  └───────────┘
                      └───────────┘

    The ╳ means EVERY neuron in one layer connects to EVERY neuron in the
    next layer. This is called a "fully connected" or "dense" layer.

Key Differences from Single Perceptron:
    1. Multiple layers of neurons
    2. Uses SIGMOID instead of step function (needed for backpropagation)
    3. Learns via BACKPROPAGATION (chain rule of calculus) not simple error rule
    4. CAN learn non-linear patterns like XOR

This script demonstrates:
    1. Forward pass through multiple layers
    2. Backpropagation (how error flows backward to update weights)
    3. Solving XOR (impossible for a single perceptron)
    4. Step-by-step walkthrough of one complete training cycle
================================================================================
"""

import random
import math


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sigmoid(z):
    """
    Sigmoid Activation Function: σ(z) = 1 / (1 + e^(-z))

    WHY SIGMOID INSTEAD OF STEP FUNCTION?
    The step function has a flat slope everywhere (0) except at the boundary
    (undefined). Backpropagation needs to compute "how much does a small
    change in the weight affect the output?" — that's a derivative. The
    step function's derivative is 0 almost everywhere, so backpropagation
    gets no useful signal.

    Sigmoid is smooth and differentiable everywhere, so we can always
    compute meaningful gradients.

    Output range: 0.0 to 1.0 (acts like a probability)
    """
    # Clip to avoid overflow in exp()
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(output):
    """
    Derivative of sigmoid: σ'(z) = σ(z) × (1 - σ(z))

    CRITICAL INSIGHT: We can compute the derivative from the OUTPUT alone.
    We don't need the original z. This is a convenient property of sigmoid.

    This derivative tells us: "How sensitive is the neuron's output to
    changes in its input?" It's highest (0.25) when output = 0.5, and
    approaches 0 when output is near 0 or 1.

    This is important because:
        - When the neuron is "confident" (output near 0 or 1), the
          derivative is small → small weight updates → "I'm sure, don't
          change me much"
        - When the neuron is "uncertain" (output near 0.5), the derivative
          is larger → bigger weight updates → "I'm unsure, adjust me more"

    Args:
        output: The neuron's output (already passed through sigmoid)

    Returns:
        The derivative value
    """
    return output * (1.0 - output)


# ==============================================================================
# NEURON CLASS
# ==============================================================================

class Neuron:
    """
    A single neuron in the MLP.

    Each neuron stores:
        - weights: one per input connection
        - bias: one constant shift value
        - output: the last computed output (cached for backpropagation)
        - delta: the error signal for this neuron (computed during backprop)
    """

    def __init__(self, num_inputs):
        """
        Initialize one neuron with random weights and zero bias.

        WEIGHT INITIALIZATION STRATEGY:
        We use random values in a small range [-0.5, 0.5]. In production
        deep learning, you'd use He or Xavier initialization, but for this
        educational example, small random values work fine for 2-3 layers.

        Args:
            num_inputs: Number of connections coming into this neuron
        """
        # Random weights — breaks symmetry so neurons learn different things
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]

        # Bias starts at 0 — standard convention
        self.bias = 0.0

        # Cache these during forward pass — needed later in backward pass
        self.output = 0.0

        # Error signal — computed during backpropagation
        # "How much was THIS neuron responsible for the final error?"
        self.delta = 0.0

    def forward(self, inputs):
        """
        Forward pass: compute this neuron's output.

        z = Σ(input_i × weight_i) + bias
        output = sigmoid(z)

        Args:
            inputs: List of input values from previous layer

        Returns:
            The neuron's output (between 0 and 1)
        """
        # Weighted sum
        z = self.bias
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]

        # Activation
        self.output = sigmoid(z)
        return self.output


# ==============================================================================
# LAYER CLASS
# ==============================================================================

class Layer:
    """
    A layer of neurons.

    Each layer is a collection of neurons that all receive the same inputs
    (from the previous layer) and produce outputs (for the next layer).

    Example: A hidden layer with 4 neurons receiving input from 3 features
    has 4 Neuron objects, each with 3 weights + 1 bias = 16 learnable parameters.
    """

    def __init__(self, num_neurons, num_inputs_per_neuron):
        """
        Create a layer of neurons.

        Args:
            num_neurons: How many neurons in this layer
            num_inputs_per_neuron: How many inputs each neuron receives
                                  (= number of neurons in previous layer,
                                   or number of input features for first layer)
        """
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        """
        Forward pass for the entire layer.

        Every neuron in this layer receives the SAME inputs and computes
        its own output independently.

        Args:
            inputs: Output from the previous layer (or raw input data)

        Returns:
            List of outputs, one per neuron in this layer
        """
        return [neuron.forward(inputs) for neuron in self.neurons]


# ==============================================================================
# MULTI-LAYER PERCEPTRON (MLP) CLASS
# ==============================================================================

class MLP:
    """
    A Multi-Layer Perceptron neural network.

    Architecture is defined by a list of layer sizes. For example:
        [2, 4, 1] means:
            - 2 input features
            - 4 neurons in the hidden layer
            - 1 neuron in the output layer
            - Total connections: (2×4) + (4×1) = 12 weights + 5 biases = 17 parameters

        [2, 8, 4, 1] means:
            - 2 input features
            - 8 neurons in hidden layer 1
            - 4 neurons in hidden layer 2
            - 1 neuron in the output layer
            - Total: (2×8) + (8×4) + (4×1) = 52 weights + 13 biases = 65 parameters

    The more layers and neurons, the more complex patterns the network can learn,
    but also the more data and compute it needs.
    """

    def __init__(self, layer_sizes, learning_rate=0.5):
        """
        Build the MLP.

        Args:
            layer_sizes: List of integers defining the architecture.
                         First element = number of inputs (not a real layer).
                         Remaining elements = number of neurons per layer.
            learning_rate: How much to adjust weights each update step.
        """
        self.learning_rate = learning_rate
        self.layers = []

        # =====================================================================
        # BUILD THE NETWORK
        # =====================================================================
        # We create Layer objects for everything EXCEPT the input "layer".
        # The input layer isn't a real layer — it's just the raw data.
        #
        # Example: layer_sizes = [2, 4, 1]
        #   - Layer 0 (hidden): 4 neurons, each with 2 inputs (from input data)
        #   - Layer 1 (output): 1 neuron, with 4 inputs (from hidden layer)
        # =====================================================================
        for i in range(1, len(layer_sizes)):
            num_neurons = layer_sizes[i]
            num_inputs = layer_sizes[i - 1]
            self.layers.append(Layer(num_neurons, num_inputs))

        # Store architecture for display
        self.layer_sizes = layer_sizes

    def forward(self, inputs):
        """
        Forward pass: push data through the entire network.

        Data flows left to right:
            Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output

        Each layer's output becomes the next layer's input.

        Args:
            inputs: Raw input data

        Returns:
            List of outputs from the final layer
        """
        current_input = inputs

        for layer in self.layers:
            current_input = layer.forward(current_input)

        # current_input now holds the output layer's values
        return current_input

    def backward(self, expected):
        """
        Backward pass (BACKPROPAGATION): compute how much each neuron
        contributed to the error, then update weights accordingly.

        =====================================================================
        THIS IS THE HEART OF DEEP LEARNING
        =====================================================================

        The key idea: we know the error at the OUTPUT. But how do we fix
        neurons in HIDDEN layers that don't directly produce the output?

        Answer: the CHAIN RULE from calculus. We propagate the error
        backward, layer by layer, distributing blame to each neuron
        proportionally to how much it influenced the final result.

        Two different calculations depending on layer type:

        OUTPUT LAYER neurons:
            delta = (expected - output) × sigmoid_derivative(output)
            "How wrong was I?" × "How sensitive am I to changes?"

        HIDDEN LAYER neurons:
            delta = (Σ of downstream deltas × connecting weights) × sigmoid_derivative(output)
            "How much blame reaches me from the next layer?" × "How sensitive am I?"

        Then ALL neurons update the same way:
            weight += learning_rate × delta × input_to_this_neuron
            bias   += learning_rate × delta

        =====================================================================

        Args:
            expected: List of expected output values
        """

        # =====================================================================
        # STEP 1: COMPUTE DELTAS (error signals) FOR EACH NEURON
        # =====================================================================
        # We go BACKWARD — from the output layer to the first hidden layer.
        # This is why it's called BACK-propagation.
        # =====================================================================

        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]

            if layer_index == len(self.layers) - 1:
                # ----- OUTPUT LAYER -----
                # We can directly compute the error because we know the expected values
                for i, neuron in enumerate(layer.neurons):
                    error = expected[i] - neuron.output
                    # delta = error × derivative
                    # derivative tells us how much the output changes with input
                    # High confidence (output near 0 or 1) → small derivative → small update
                    # Low confidence (output near 0.5) → large derivative → larger update
                    neuron.delta = error * sigmoid_derivative(neuron.output)
            else:
                # ----- HIDDEN LAYER -----
                # We don't know the "expected" value for hidden neurons.
                # Instead, we distribute blame from the NEXT layer backward.
                next_layer = self.layers[layer_index + 1]
                for i, neuron in enumerate(layer.neurons):
                    # Sum up all the error that flows back to this neuron
                    # Each neuron in the next layer sends back:
                    #   its delta × the weight connecting it to our neuron
                    downstream_error = 0.0
                    for next_neuron in next_layer.neurons:
                        downstream_error += next_neuron.delta * next_neuron.weights[i]

                    neuron.delta = downstream_error * sigmoid_derivative(neuron.output)

        # =====================================================================
        # STEP 2: UPDATE WEIGHTS AND BIASES USING THE COMPUTED DELTAS
        # =====================================================================
        # Now every neuron knows its delta (how responsible it was for the error).
        # We use this to update weights.
        # =====================================================================

        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]

            # Determine what the inputs to this layer were
            if layer_index == 0:
                # First hidden layer — inputs come from the raw data
                # We need to get the raw input, which was cached during forward()
                # Since we don't cache it in the MLP, we'll get it from context
                layer_inputs = self._last_input
            else:
                # Other layers — inputs come from the previous layer's outputs
                layer_inputs = [n.output for n in self.layers[layer_index - 1].neurons]

            # Update each neuron's weights and bias
            for neuron in layer.neurons:
                for j in range(len(neuron.weights)):
                    # Same logic as the single perceptron, but delta replaces error:
                    # weight += learning_rate × delta × input
                    neuron.weights[j] += self.learning_rate * neuron.delta * layer_inputs[j]

                # Bias update (bias input is always 1)
                neuron.bias += self.learning_rate * neuron.delta

    def train_one_example(self, inputs, expected):
        """
        One complete training cycle for one example:
            1. Forward pass  → compute prediction
            2. Backward pass → compute deltas and update weights

        Args:
            inputs: Input data
            expected: Expected output (as a list, e.g., [1] or [0])

        Returns:
            The network's prediction (before weight update)
        """
        # Cache the input for use in backward()
        self._last_input = inputs

        # Forward pass — data flows through the network
        outputs = self.forward(inputs)

        # Backward pass — error flows back, weights update
        self.backward(expected)

        return outputs

    def train(self, training_data, labels, epochs=1000, verbose=True):
        """
        Train the network on a dataset for multiple epochs.

        Each epoch = one complete pass through ALL training examples.

        Args:
            training_data: List of input examples
            labels: List of expected outputs (each is a list, e.g., [[0], [1], [1], [0]])
            epochs: Number of complete passes through the data
            verbose: Whether to print progress

        Returns:
            List of average loss per epoch (for tracking convergence)
        """
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for inputs, expected in zip(training_data, labels):
                # Run one training cycle
                outputs = self.train_one_example(inputs, expected)

                # ---- Calculate loss (Mean Squared Error) ----
                # This measures HOW wrong the network is — not just right/wrong,
                # but by how much. The network's goal is to minimize this number.
                #
                # MSE = (1/n) × Σ(expected - predicted)²
                #
                # Squaring serves two purposes:
                #   1. Makes all errors positive (wrong in either direction counts)
                #   2. Penalizes large errors more than small errors
                for i in range(len(expected)):
                    total_loss += (expected[i] - outputs[i]) ** 2

            avg_loss = total_loss / len(training_data)
            loss_history.append(avg_loss)

            # Print progress at intervals
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"  Epoch {epoch + 1:>5}/{epochs}  |  Loss: {avg_loss:.6f}")

        return loss_history

    def predict(self, inputs):
        """
        Make a prediction (without updating weights).

        Args:
            inputs: Input data

        Returns:
            Network output(s)
        """
        return self.forward(inputs)

    def display_network(self):
        """Print the complete state of every neuron in the network."""
        print(f"\n  Network Architecture: {self.layer_sizes}")
        print(f"  Total layers: {len(self.layers)} (excluding input)")
        print()

        for i, layer in enumerate(self.layers):
            layer_type = "OUTPUT" if i == len(self.layers) - 1 else f"HIDDEN {i + 1}"
            print(f"  ── {layer_type} LAYER ({len(layer.neurons)} neurons) ──")
            for j, neuron in enumerate(layer.neurons):
                weights_str = [round(w, 4) for w in neuron.weights]
                print(f"     Neuron {j + 1}: weights={weights_str}, bias={round(neuron.bias, 4)}")
            print()


# ==============================================================================
# ==============================================================================
#                           DEMONSTRATIONS
# ==============================================================================
# ==============================================================================


def demonstrate_xor():
    """
    Train an MLP to solve XOR — the problem a single perceptron CANNOT solve.

    XOR Truth Table:
        [0, 0] → 0
        [0, 1] → 1
        [1, 0] → 1
        [1, 1] → 0

    WHY XOR IS INTERESTING:
    XOR requires understanding that "exactly one input is 1". This is a
    non-linear relationship. The hidden layer learns to create intermediate
    representations that make XOR linearly separable for the output neuron.

    Essentially:
        Hidden neuron 1 might learn something like OR  (is at least one input 1?)
        Hidden neuron 2 might learn something like AND (are both inputs 1?)
        Output neuron combines them: OR and NOT AND = XOR
    """
    print(f"\n{'=' * 60}")
    print(f"  SOLVING XOR WITH A MULTI-LAYER PERCEPTRON")
    print(f"{'=' * 60}")
    print(f"""
  Remember: A single perceptron FAILED at this.
  Now we use 2 inputs → 2 hidden neurons → 1 output.
  The hidden layer creates intermediate representations that
  make XOR solvable.
    """)

    # Training data
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[0],    [1],    [1],    [0]]     # XOR outputs (as lists)

    # Create MLP: 2 inputs → 2 hidden neurons → 1 output
    # Architecture: [2, 2, 1]
    random.seed(42)  # For reproducible results
    mlp = MLP(layer_sizes=[2, 2, 1], learning_rate=2.0)

    # ---- Show state before training ----
    print(f"  --- Before Training ---")
    mlp.display_network()

    print(f"  Predictions before training:")
    for inp, exp in zip(inputs, labels):
        output = mlp.predict(inp)
        print(f"    Input: {inp}  →  Output: {output[0]:.4f}  (expected: {exp[0]})")

    # ---- Train ----
    print(f"\n  --- Training for 10000 epochs ---\n")
    loss_history = mlp.train(inputs, labels, epochs=10000, verbose=True)

    # ---- Show state after training ----
    print(f"\n  --- After Training ---")
    mlp.display_network()

    # ---- Test results ----
    print(f"  --- Results ---")
    print(f"  {'Input':<12} {'Expected':<12} {'Raw Output':<15} {'Rounded':<10} {'Correct?'}")
    print(f"  {'-' * 58}")

    all_correct = True
    for inp, exp in zip(inputs, labels):
        output = mlp.predict(inp)
        raw = output[0]
        rounded = round(raw)
        correct = "✓" if rounded == exp[0] else "✗"
        if rounded != exp[0]:
            all_correct = False
        print(f"  {str(inp):<12} {exp[0]:<12} {raw:<15.4f} {rounded:<10} {correct}")

    if all_correct:
        print(f"\n  MLP successfully learned XOR!")
        print(f"    A single perceptron could never do this.")
    else:
        print(f"\n  MLP did not converge. Try different learning rate or more epochs.")

    # ---- Show loss curve ----
    print(f"\n  --- Loss Curve (how error decreased over training) ---")
    milestones = [0, 99, 499, 999, 2499, 4999, 7499, 9999]
    for i in milestones:
        if i < len(loss_history):
            bar_length = int(loss_history[i] * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  Epoch {i + 1:>5}: {bar} {loss_history[i]:.4f}")

    return mlp


def demonstrate_what_hidden_layer_learns(mlp):
    """
    Peek inside the hidden layer to see what intermediate representations
    it learned. This is key to understanding WHY the MLP can solve XOR.
    """
    print(f"\n{'=' * 60}")
    print(f"  WHAT DID THE HIDDEN LAYER LEARN?")
    print(f"{'=' * 60}")
    print(f"""
  The hidden layer transforms the inputs into a new representation
  where XOR becomes linearly separable. Let's see what each hidden
  neuron outputs for each input combination.
    """)

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_labels = [0, 1, 1, 0]

    print(f"  {'Input':<10} {'H1 Output':<12} {'H2 Output':<12} {'Final Out':<12} {'XOR'}")
    print(f"  {'-' * 55}")

    for inp, expected in zip(inputs, xor_labels):
        # Run forward pass
        mlp.forward(inp)

        # Extract hidden neuron outputs
        h1_out = mlp.layers[0].neurons[0].output
        h2_out = mlp.layers[0].neurons[1].output
        final = mlp.layers[1].neurons[0].output

        print(f"  {str(inp):<10} {h1_out:<12.4f} {h2_out:<12.4f} {final:<12.4f} {expected}")

    print(f"""
  Look at the hidden layer outputs (H1 and H2). The hidden layer has
  TRANSFORMED the original 2D input into a new 2D representation where
  the XOR pattern IS linearly separable!

  The original XOR problem:          After hidden layer:
      (0,0)→0  (0,1)→1              The four points are rearranged
      (1,0)→1  (1,1)→0              so a single line can separate
      (no single line works)          the 0s from the 1s

  This is the fundamental power of hidden layers — they learn to
  TRANSFORM data into representations where the problem becomes solvable.
  Deep learning is essentially learning better and better representations.
    """)


def demonstrate_backpropagation_step_by_step():
    """
    Walk through one COMPLETE training cycle with actual numbers,
    showing exactly what happens in forward pass and backward pass.
    """
    print(f"\n{'=' * 60}")
    print(f"  STEP-BY-STEP: ONE COMPLETE TRAINING CYCLE")
    print(f"{'=' * 60}")
    print(f"""
  Let's trace EXACTLY what happens for one input through a tiny network.

  Network: 2 inputs → 2 hidden neurons → 1 output
  Input: [1, 0], Expected output: [1] (this is from XOR: 1 XOR 0 = 1)
  Learning rate: 0.5
    """)

    # Create a tiny network with FIXED weights so we can trace everything
    random.seed(0)
    mlp = MLP(layer_sizes=[2, 2, 1], learning_rate=0.5)

    # Set specific weights for clarity
    mlp.layers[0].neurons[0].weights = [0.3, -0.2]
    mlp.layers[0].neurons[0].bias = 0.1
    mlp.layers[0].neurons[1].weights = [0.5, 0.4]
    mlp.layers[0].neurons[1].bias = -0.3
    mlp.layers[1].neurons[0].weights = [0.7, -0.6]
    mlp.layers[1].neurons[0].bias = 0.2

    inputs = [1, 0]
    expected = [1]

    print(f"  Initial Weights:")
    print(f"    Hidden Neuron 1: weights=[0.3, -0.2], bias=0.1")
    print(f"    Hidden Neuron 2: weights=[0.5, 0.4],  bias=-0.3")
    print(f"    Output Neuron:   weights=[0.7, -0.6], bias=0.2")

    # ===== FORWARD PASS =====
    print(f"\n  {'─' * 50}")
    print(f"  FORWARD PASS (data flows left → right)")
    print(f"  {'─' * 50}")

    # Hidden neuron 1
    z_h1 = (1 * 0.3) + (0 * -0.2) + 0.1
    o_h1 = sigmoid(z_h1)
    print(f"\n  Hidden Neuron 1:")
    print(f"    z = (1 × 0.3) + (0 × -0.2) + 0.1 = {z_h1}")
    print(f"    output = sigmoid({z_h1}) = {o_h1:.4f}")

    # Hidden neuron 2
    z_h2 = (1 * 0.5) + (0 * 0.4) + (-0.3)
    o_h2 = sigmoid(z_h2)
    print(f"\n  Hidden Neuron 2:")
    print(f"    z = (1 × 0.5) + (0 × 0.4) + (-0.3) = {z_h2}")
    print(f"    output = sigmoid({z_h2}) = {o_h2:.4f}")

    # Output neuron
    z_o = (o_h1 * 0.7) + (o_h2 * -0.6) + 0.2
    o_o = sigmoid(z_o)
    print(f"\n  Output Neuron:")
    print(f"    z = ({o_h1:.4f} × 0.7) + ({o_h2:.4f} × -0.6) + 0.2 = {z_o:.4f}")
    print(f"    output = sigmoid({z_o:.4f}) = {o_o:.4f}")

    print(f"\n  PREDICTION: {o_o:.4f}")
    print(f"  EXPECTED:   {expected[0]}")
    print(f"  ERROR:      {expected[0] - o_o:.4f}")
    print(f"  LOSS (MSE): {(expected[0] - o_o) ** 2:.4f}")

    # ===== BACKWARD PASS =====
    print(f"\n  {'─' * 50}")
    print(f"  BACKWARD PASS (error flows right → left)")
    print(f"  {'─' * 50}")

    # Output neuron delta
    error_o = expected[0] - o_o
    delta_o = error_o * sigmoid_derivative(o_o)
    print(f"\n  Output Neuron Delta:")
    print(f"    error = expected - output = {expected[0]} - {o_o:.4f} = {error_o:.4f}")
    print(f"    sigmoid_derivative({o_o:.4f}) = {o_o:.4f} × (1 - {o_o:.4f}) = {sigmoid_derivative(o_o):.4f}")
    print(f"    delta = error × derivative = {error_o:.4f} × {sigmoid_derivative(o_o):.4f} = {delta_o:.4f}")
    print(f"    Meaning: This is how much the output neuron needs to change")

    # Hidden neuron 1 delta
    downstream_error_h1 = delta_o * 0.7  # output neuron's delta × connecting weight
    delta_h1 = downstream_error_h1 * sigmoid_derivative(o_h1)
    print(f"\n  Hidden Neuron 1 Delta:")
    print(f"    downstream_error = output_delta × connecting_weight")
    print(f"    downstream_error = {delta_o:.4f} × 0.7 = {downstream_error_h1:.4f}")
    print(f"    delta = {downstream_error_h1:.4f} × sigmoid_derivative({o_h1:.4f}) = {delta_h1:.4f}")
    print(f"    Meaning: This is H1's share of the blame for the final error")

    # Hidden neuron 2 delta
    downstream_error_h2 = delta_o * (-0.6)
    delta_h2 = downstream_error_h2 * sigmoid_derivative(o_h2)
    print(f"\n  Hidden Neuron 2 Delta:")
    print(f"    downstream_error = output_delta × connecting_weight")
    print(f"    downstream_error = {delta_o:.4f} × (-0.6) = {downstream_error_h2:.4f}")
    print(f"    delta = {downstream_error_h2:.4f} × sigmoid_derivative({o_h2:.4f}) = {delta_h2:.4f}")
    print(f"    Meaning: This is H2's share of the blame for the final error")

    # ===== WEIGHT UPDATES =====
    print(f"\n  {'─' * 50}")
    print(f"  WEIGHT UPDATES")
    print(f"  {'─' * 50}")
    print(f"  Rule: new_weight = old_weight + learning_rate × delta × input")

    lr = 0.5

    print(f"\n  Output Neuron weights:")
    new_w1_o = 0.7 + lr * delta_o * o_h1
    new_w2_o = -0.6 + lr * delta_o * o_h2
    new_b_o = 0.2 + lr * delta_o
    print(f"    w1: 0.7 + ({lr} × {delta_o:.4f} × {o_h1:.4f}) = {new_w1_o:.4f}")
    print(f"    w2: -0.6 + ({lr} × {delta_o:.4f} × {o_h2:.4f}) = {new_w2_o:.4f}")
    print(f"    bias: 0.2 + ({lr} × {delta_o:.4f}) = {new_b_o:.4f}")

    print(f"\n  Hidden Neuron 1 weights:")
    new_w1_h1 = 0.3 + lr * delta_h1 * inputs[0]
    new_w2_h1 = -0.2 + lr * delta_h1 * inputs[1]
    new_b_h1 = 0.1 + lr * delta_h1
    print(f"    w1: 0.3 + ({lr} × {delta_h1:.4f} × {inputs[0]}) = {new_w1_h1:.4f}")
    print(f"    w2: -0.2 + ({lr} × {delta_h1:.4f} × {inputs[1]}) = {new_w2_h1:.4f}")
    print(f"    bias: 0.1 + ({lr} × {delta_h1:.4f}) = {new_b_h1:.4f}")

    print(f"\n  Hidden Neuron 2 weights:")
    new_w1_h2 = 0.5 + lr * delta_h2 * inputs[0]
    new_w2_h2 = 0.4 + lr * delta_h2 * inputs[1]
    new_b_h2 = -0.3 + lr * delta_h2
    print(f"    w1: 0.5 + ({lr} × {delta_h2:.4f} × {inputs[0]}) = {new_w1_h2:.4f}")
    print(f"    w2: 0.4 + ({lr} × {delta_h2:.4f} × {inputs[1]}) = {new_w2_h2:.4f}")
    print(f"    bias: -0.3 + ({lr} × {delta_h2:.4f}) = {new_b_h2:.4f}")

    # ===== SUMMARY =====
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY OF CHANGES")
    print(f"  {'─' * 50}")
    print(f"""
  Before → After:

    Hidden Neuron 1:
      weights: [0.3, -0.2]  → [{new_w1_h1:.4f}, {new_w2_h1:.4f}]
      bias:    0.1           → {new_b_h1:.4f}

    Hidden Neuron 2:
      weights: [0.5, 0.4]   → [{new_w1_h2:.4f}, {new_w2_h2:.4f}]
      bias:    -0.3          → {new_b_h2:.4f}

    Output Neuron:
      weights: [0.7, -0.6]  → [{new_w1_o:.4f}, {new_w2_o:.4f}]
      bias:    0.2           → {new_b_o:.4f}

  Every single weight and bias in the network was adjusted by a small
  amount. This is ONE step. Training repeats this thousands of times
  across all training examples until the network converges.
    """)


def demonstrate_deeper_network():
    """
    Show that you can add more layers for more complex problems.
    Uses a 3-layer network on XOR to show the concept of depth.
    """
    print(f"\n{'=' * 60}")
    print(f"  BONUS: DEEPER NETWORK (3 LAYERS)")
    print(f"{'=' * 60}")
    print(f"""
  Architecture: 2 inputs → 4 hidden → 3 hidden → 1 output
  This is "deeper" but overkill for XOR. It demonstrates that
  you can stack as many layers as needed for harder problems.
    """)

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[0],    [1],    [1],    [0]]

    random.seed(42)
    deep_mlp = MLP(layer_sizes=[2, 4, 3, 1], learning_rate=1.0)

    # Count parameters
    total_params = 0
    for layer in deep_mlp.layers:
        for neuron in layer.neurons:
            total_params += len(neuron.weights) + 1  # weights + bias
    print(f"  Total learnable parameters: {total_params}")

    print(f"\n  --- Training ---\n")
    loss_history = deep_mlp.train(inputs, labels, epochs=10000, verbose=True)

    print(f"\n  --- Results ---")
    print(f"  {'Input':<12} {'Expected':<12} {'Output':<15} {'Rounded':<10} {'Correct?'}")
    print(f"  {'-' * 58}")

    all_correct = True
    for inp, exp in zip(inputs, labels):
        output = deep_mlp.predict(inp)
        raw = output[0]
        rounded = round(raw)
        correct = "✓" if rounded == exp[0] else "✗"
        if rounded != exp[0]:
            all_correct = False
        print(f"  {str(inp):<12} {exp[0]:<12} {raw:<15.6f} {rounded:<10} {correct}")

    if all_correct:
        print(f"\n  Deeper network also learns XOR!")
    print(f"\n  Final loss: {loss_history[-1]:.6f}")


# ==============================================================================
# RUN ALL DEMONSTRATIONS
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  MULTI-LAYER PERCEPTRON: SOLVING WHAT ONE NEURON CAN'T")
    print("=" * 60)
    print("""
  A single perceptron failed at XOR because XOR is not linearly separable.
  By adding a hidden layer, we give the network the ability to learn
  intermediate representations that transform the problem into one that
  IS linearly separable.

  This script covers:
    1. Solving XOR with an MLP
    2. What the hidden layer actually learns
    3. Step-by-step backpropagation walkthrough
    4. A deeper network example
    """)

    # Demo 1: Solve XOR
    mlp = demonstrate_xor()

    # Demo 2: Peek inside the hidden layer
    demonstrate_what_hidden_layer_learns(mlp)

    # Demo 3: Step-by-step backpropagation
    demonstrate_backpropagation_step_by_step()

    # Demo 4: Deeper network
    demonstrate_deeper_network()

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print(f"""
  1. HIDDEN LAYERS transform data into new representations where
     previously unsolvable problems become solvable.

  2. BACKPROPAGATION distributes blame from the output error backward
     through every layer, so even hidden neurons know how to adjust.

  3. The CHAIN RULE from calculus is what makes this work — it lets us
     compute how each weight in any layer affects the final output.

  4. FORWARD PASS computes the prediction (weights are frozen).
     BACKWARD PASS computes the error and updates all weights.
     These two phases alternate during training.

  5. DEEPER networks can learn more complex representations, but need
     more data, compute, and careful tuning to train successfully.

  6. Every modern neural network — CNNs, RNNs, Transformers, GPT —
     is built on these exact same principles, just with more layers,
     more neurons, and specialized architectures.
    """)
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    from Required_Images.mlp_xor_component import MLP_XOR_HTML, MLP_XOR_HEIGHT

    # Fallback static image (still used if interactive component fails)
    mlp_img = _image_to_html(
        "Required_Images/MultiLayerPreceptron_Breakdown.png",
        alt="Multilayer Perceptron Architecture",
        width="50%"
    )
    theory_with_images = THEORY.replace("{{MLP_IMAGE}}", mlp_img)

    # Interactive component data — app.py splits theory_raw at each
    # placeholder and injects an st.components.v1.html() iframe instead.
    interactive_components = [
        {
            "placeholder": "{{MLP_IMAGE}}",
            "html": MLP_XOR_HTML,
            "height": MLP_XOR_HEIGHT,
        }
    ]

    return {
        "theory": theory_with_images,  # fallback (static image)
        "theory_raw": THEORY,  # raw template with placeholders
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }