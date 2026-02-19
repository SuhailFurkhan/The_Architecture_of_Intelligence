"""
Recurrent Neural Network(RNN)
============================================

[Optional: longer overview paragraph you can fill in later]
"""

import math
import numpy as np
import base64
import os

TOPIC_NAME = "Recurrent Neural Network(RNN)"

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

### Recurrent Neural Network(RNN)

## The Problem That RNNs Solve

Remember how CNNs solve the spatial problem — understanding that nearby pixels are related ? 
RNNs solve a different problem: sequential data, where the order matters.

Think about this sentence: "The cat sat on the ___"

To predict the next word, you need to remember the words that came before. 
A regular neural network (or a CNN) sees all inputs independently — it has no concept of "what came before." 

**An RNN introduces memory.**

---

#### How an RNN Works

In a regular perceptron, you learned:
    
        z = W·X + b → activation(z) → output

An RNN adds one critical ingredient — it feeds its previous output back into itself:
    
        h(t) = activation(W_input · X(t) + W_hidden · h(t-1) + b)

Where:

•	X(t) = the current input (e.g., the current word)

•	h(t-1) = the hidden state from the previous time step (the "memory")

•	h(t) = the new hidden state (updated memory + current input)

---

#### A Tiny RNN: Predicting Sentiment

Imagine we're processing the sentence "not good" one word at a time to determine if it's positive or negative.

    Time Step 1: Input = "not"        Time Step 2: Input = "good"
    
        ┌──────────┐                     ┌──────────┐
        │          │── h(1) ────────────►│          │── h(2) ──► Output
        │  RNN     │                     │  RNN     │
        │  Neuron  │                     │  Neuron  │
        │          │◄── h(0)=0           │          │
        └──────────┘                     └──────────┘
             ▲                                ▲
             │                                │
          X(1)="not"                      X(2)="good"

It's the same neuron reused at each step, but it carries forward a hidden state. This is called "unrolling" the RNN across time.

#### Walking Through With Real Numbers
Let's say words are encoded as simple numbers: "not" = -1, "good" = 0.8
Weights (the network learns these):

* W_input = 0.5 (how much to weight the current word)
* W_hidden = 0.9 (how much to weight the memory)
* b = 0
* Activation = tanh

---
Time step 1 — Processing "not":

    h(0) = 0  (no memory yet)
    
    h(1) = tanh(0.5 × (-1) + 0.9 × 0 + 0)
    
         = tanh(-0.5)
         
         = -0.46
     
The hidden state is now -0.46 — the network has a "negative" memory.

---

Time step 2 — Processing "good":

    h(2) = tanh(0.5 × 0.8 + 0.9 × (-0.46) + 0)
    
         = tanh(0.4 + (-0.414))
         
         = tanh(-0.014)
         
         = -0.014
         
Even though "good" is positive (0.8), the memory of "not" dragged the result slightly negative (-0.014). 
The network understood that "not good" ≠ "good" because it remembered the context.
If we had just fed "good" alone without memory: 

    tanh(0.5 × 0.8) 
    
    = tanh(0.4) 
    
    = 0.38 — positive! 

The memory changed the outcome.


#### RNN Architecture (3 Layers)

Just like we did with your CNN, here's a small 3-layer RNN:

---
Layer 1: RNN Layer (2 neurons) — processes sequence, builds memory

Layer 2: RNN Layer (2 neurons) — processes Layer 1's output sequence, builds higher-level patterns

Layer 3: Dense Layer (1 neuron) — takes final hidden state → output prediction

---

Input sequence: ["not", "very", "good"]

---
    Time ──────────────────────────────►

          t=1("not")    t=2("very")     t=3("good")
           │              │                │
     ┌─────▼─────┐ ┌──────▼─────┐   ┌──────▼─────┐
     │ RNN L1    │→│ RNN L1     │ → │ RNN L1     │→ h1(3)
     │ (2 neur.) │ │ (2 neur .) │   │ (2 neur.)  │
     └─────┬─────┘ └──────┬─────┘   └──────┬─────┘
           │              │                │
     ┌─────▼─────┐ ┌──────▼─────┐   ┌──────▼─────┐
     │ RNN L2    │→│ RNN L2     │ → │ RNN L2     │→ h2(3)
     │ (2 neur.) │ │ (2 neur.)  │   │ (2 neur.)  │
     └───────────┘ └────────────┘   └──────┬─────┘
                                           │
                                     ┌─────▼─────┐
                                     │ Dense L3  │→ Output
                                     │ (1 neuron)│  (positive/
                                     └───────────┘   negative)
---
      
Each RNN layer passes its hidden state horizontally (through time) and its output vertically (to the next layer). 
Layer 2 learns higher-level sequential patterns from Layer 1's outputs.                      
            
---
CNN Vs RNN: 
                             
    Aspect            CNN                                   RNN
    ---------------------------------------------------------------------------
    Designed for      Spatial data (images, grids)           Sequential data (text, audio, time series)
    
    Core operation    Filter slides across space             Neuron loops across time
    
    Memory            None — each patch processed            Yes — hidden state carries info
                      independently                          from past steps
    
    Parameter sharing Same filter reused at every position   Same weights reused at every time step
    
    Input shape       Fixed size (e.g., 28×28 image)         Variable length (e.g., sentences of any length)
    
    What it learns    Spatial patterns                       Temporal patterns
                      (edges → shapes → objects)             (word context → phrase meaning → sentiment)
    
    Key weakness      Can't handle order/sequence            Struggles with very long sequences
                                                             (vanishing gradient)
---

CNN  vs  RNN  (Textual Diagram)

    ┌───────────────────────────────┬────────────────────────────────────────┐
    │              CNN              │                  RNN                   │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Designed for                  │ Designed for                           │
    │ Spatial data (images, grids)  │ Sequential data (text, audio, time     │
    │                               │ series)                                │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Core operation                │ Core operation                         │
    │ Filter slides across space    │ Neuron loops across time               │
    │ (local window scans image)    │ (processes step-by-step)               │
    │   [ ]→[ ]→[ ]                 │   x₁ → x₂ → x₃ → ...                   │
    │   [ ]→[ ]→[ ]                 │     │    │    │                        │
    │   (space)                     │    h₁   h₂   h₃   ... (hidden state)   │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Memory                        │ Memory                                 │
    │ None — each patch processed   │ Yes — hidden state carries info from   │
    │ independently                 │ past steps                             │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Parameter sharing             │ Parameter sharing                      │
    │ Same filter reused at every   │ Same weights reused at every time step │
    │ position                      │                                        │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Input shape                   │ Input shape                            │
    │ Fixed size                    │ Variable length                        │
    │ (e.g., 28×28 image)           │ (e.g., sentences of any length)        │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ What it learns                │ What it learns                         │
    │ Spatial patterns              │ Temporal patterns                      │
    │ (edges → shapes → objects)    │ (word context → phrase meaning →       │
    │                               │ sentiment)                             │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Key weakness                  │ Key weakness                           │
    │ Can't handle order/sequence   │ Struggles with very long sequences     │
    │                               │ (vanishing gradient)                   │
    └───────────────────────────────┴────────────────────────────────────────┘

---

#### The Intuitive Summary

Think of it this way, using what you already know:

    * A regular perceptron looks at all inputs at once with no structure — like reading all the pixels or all the words jumbled together
    
    * A CNN says "let me look at small local neighborhoods and slide across" — perfect for images where nearby pixels matter
    
    * An RNN says "let me look at one thing at a time and remember what came before" — perfect for sequences where order matters


### Activation and Gradient 

**Activation: “Activation is the actual computed value in the forward pass”**
    
In an RNN at time t:

* pre-activation (raw value):
                
                a_t=W_x x_t+W_h h_(t-1)+b

* activation / hidden state (after nonlinearity):
                
                h_t=ϕ(a_t)"(e.g.,tanh or ReLU)" 

So the “activation” is a forward-pass computed value, but it’s more precise to say:

##### **Activation = output after nonlinearity (e.g., h_t), not the raw sum a_t.**

**“Gradient = how much each weight caused the error”**

The gradient tells you how sensitive the loss is to a weight.

For a weight w:
                
                ∂L/∂w

Interpretation:
	
	If ∂L/∂wis large (in magnitude), small changes in wchange the loss a lot → that weight strongly affects the error.
	
	Sign matters:
    - Positive gradient: increasing wincreases loss (bad), so gradient descent decreases w.
    - Negative gradient: increasing wdecreases loss, so gradient descent increases w.

Because weights are shared across time, the gradient for a weight is the sum of its contributions across all time steps (Backprop Through Time).

**Clean interview-ready phrasing**

##### Activation: “Forward-pass output after the nonlinearity (e.g., hidden state).”

##### Gradient: “Derivative of the loss w.r.t. a weight — measures how much that weight contributed to the error (sensitivity), summed over time in RNNs.”


    
---

## Part 2 — Training an RNN: Backpropagation Through Time (BPTT)

You already know how backpropagation works in a regular network: compute the loss, then send gradients 
backward through each layer to update the weights.

An RNN does the same thing, but since it's the **same neuron reused across time steps**, the gradients 
must flow **backward through time** — from the last time step all the way back to the first.

This is called **Backpropagation Through Time (BPTT)**.

---

#### Unrolling the RNN for Training

When we train, we "unroll" the RNN — treat each time step as if it were a separate layer:

---
    Forward Pass (left to right):
    
    X(1) → [RNN] → h(1) → [RNN] → h(2) → [RNN] → h(3) → Loss
              ↑               ↑               ↑
            h(0)=0          h(1)            h(2)
    
    Backward Pass (right to left — BPTT):
    
    Loss → ∂L/∂h(3) → ∂L/∂h(2) → ∂L/∂h(1) → update W_input, W_hidden, b
---

Because the **same weights** (W_input, W_hidden) are used at every step, the gradient for each weight 
is the **sum of gradients from all time steps**:

    ∂L/∂W = ∂L/∂W at t=3  +  ∂L/∂W at t=2  +  ∂L/∂W at t=1

This is what makes RNN training unique — and what causes the vanishing gradient problem.

---
Backpropagation Through Time (BPTT):

    REGULAR NETWORK — Backprop through LAYERS only:
    ────────────────────────────────────────────────

       Layer 1      Layer 2      Layer 3
       ┌─────┐     ┌─────┐     ┌─────┐
       │     │────►│     │────►│     │──► Output
       │     │     │     │     │     │
       └─────┘     └─────┘     └─────┘

       Backward:
       ◄────────── ◄────────── ◄──────── Error
       Direction: through LAYERS (vertical depth)
       Steps back: 3 (one per layer)



    RNN — Backprop through LAYERS *and* TIME:
    ────────────────────────────────────────────────

       The network is "unrolled" across time:

             t=1            t=2           t=3           t=4
           "The"          "cat"         "sat"         "on"
             │              │             │             │
             ▼              ▼             ▼             ▼
           ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
       L1  │ RNN │── h ──│ RNN │── h ──│ RNN │── h ──│ RNN │
           └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘
              │             │             │             │
              ▼             ▼             ▼             ▼
           ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
       L2  │ RNN │── h ──│ RNN │── h ──│ RNN │── h ──│ RNN │──► Output
           └─────┘       └─────┘       └─────┘       └─────┘

       Backward must go BOTH directions:

           ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
       L1  │     │◄─ h ──│     │◄─ h ──│     │◄─ h ──│     │ ◄── through TIME
           └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘
              ▲             ▲             ▲             ▲    ◄── through LAYERS
           ┌──┴──┐       ┌──┴──┐       ┌──┴──┐       ┌──┴──┐
       L2  │     │◄─ h ──│     │◄─ h ──│     │◄─ h ──│     │ ◄── through TIME
           └─────┘       └─────┘       └─────┘       └─────┘
                                                         ▲
                                                       Error
---

#### The Math Behind BPTT

For a simple RNN with:

    h(t) = tanh(W_hh · h(t-1) + W_xh · x(t) + b_h)
    y(t) = W_hy · h(t) + b_y

The gradient of the loss L with respect to W_hh at time step t involves a **chain of derivatives**:

    ∂L/∂W_hh = Σ (from t=1 to T)  ∂L/∂y(t) · ∂y(t)/∂h(t) · [ Π (from k=2 to t) ∂h(k)/∂h(k-1) ] · ∂h(1)/∂W_hh

That product term  Π ∂h(k)/∂h(k-1)  is the critical piece — it's a chain of multiplications.

Since   ∂h(t)/∂h(t-1) 
        
        = W_hh^T · diag(1 - tanh²(z(t))):

* Each factor includes multiplying by W_hh
* Each factor also includes the tanh derivative, which is always between 0 and 1

Over many time steps, you're multiplying many numbers < 1 together → **gradients shrink to zero**.


**Core Difference**

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  REGULAR NETWORK                  RNN                            │
    │  ────────────────                 ────────────────               │
    │                                                                  │
    │  Forward:                         Forward:                       │
    │  All inputs enter at once         Inputs enter ONE AT A TIME     │
    │  "see everything together"        "see one word, then next..."   │
    │                                                                  │
    │  Backward:                        Backward:                      │
    │  Go back through LAYERS           Go back through LAYERS         │
    │  (3 layers = 3 steps back)        *AND* through TIME             │
    │                                   (3 layers × 4 time steps       │
    │                                    = up to 12 steps back!)       │
    │                                                                  │
    │  When does it happen?             When does it happen?           │
    │  After one input produces         After the ENTIRE SEQUENCE      │
    │  one output                       has been processed             │
    │                                                                  │
    │  What gets updated?               What gets updated?             │
    │  Each layer has its OWN           The SAME weights are shared    │
    │  separate weights                 across all time steps, so      │
    │                                   gradients from ALL time steps  │
    │                                   are COMBINED to update them    │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    
---

    Sentence: "The cat sat" → Predict next word
                                        Expected: "on"
                                        Predicted: "ran"
                                        Error!
    
    STEP 1: Process entire sequence forward (all time steps)
    ──────────────────────────────────────────────────────────
    
      "The"──►[RNN]──h(1)──►[RNN]──h(2)──►[RNN]──h(3)──► "ran" (wrong!)
               t=1    │      t=2    │       t=3
                      │             │         │
               Same W_hidden  Same W_hidden   Same W_hidden
               Same W_input   Same W_input    Same W_input
               (ALL shared — it's the same neuron reused!)
    
    
    STEP 2: Calculate error
    ──────────────────────────────────────────────────────────
    
      Loss = how far "ran" is from "on"
    
    
    STEP 3: Backpropagate through TIME (this is the special part)
    ──────────────────────────────────────────────────────────
    
      Error at t=3: "At time step 3, how much did W_hidden
                     contribute to the error?"
                     → gradient_3 = 1.0
    
      Error at t=2: "At time step 2, how much did W_hidden
                     contribute to the error?"
                     → gradient_2 = gradient_3 × W_hidden = 0.5
                                    ▲
                                    │
                        THIS repeated multiplication
                        is what causes vanishing!
    
      Error at t=1: "At time step 1, how much did W_hidden
                     contribute to the error?"
                     → gradient_1 = gradient_2 × W_hidden = 0.25
    
    
    STEP 4: COMBINE all gradients to update the SHARED weights
    ──────────────────────────────────────────────────────────
    
      Total gradient for W_hidden = gradient_1 + gradient_2 + gradient_3
                                   = 0.25     + 0.5       + 1.0
                                   = 1.75
    
      W_hidden_new = W_hidden_old - learning_rate × 1.75
    
      Notice: ALL time steps contribute to ONE weight update
      because it's the SAME weight reused at every step!

---

    Regular NN:   Forward (all at once) → Error → Backward (through layers)
                  ✓ Your understanding is exactly right
    
    RNN:          Forward (one step at a time, across the full sequence)
                  → Error (only computed after full sequence)
                  → Backward (through layers AND back through every time step)
                  → All time step gradients COMBINED into one weight update
    
    The difference: RNN backprop has an EXTRA dimension — TIME
                   which means more multiplications
                   which means gradients vanish faster
                   which means early words get forgotten
               
---

## Part 3 — The Vanishing Gradient Problem

This is the **fundamental weakness** of vanilla RNNs and the reason LSTMs and GRUs were invented.

---

#### What Happens

When the gradient flows backward through many time steps, it must pass through a chain of multiplications:

    ∂h(t)/∂h(1) = ∂h(t)/∂h(t-1) × ∂h(t-1)/∂h(t-2) × ... × ∂h(2)/∂h(1)

Each factor contains:
* The **tanh derivative** (always between 0 and 1)
* The **weight matrix W_hh**

---

**Vanishing**: If W_hh has small values (eigenvalues < 1):

    0.5 × 0.5 × 0.5 × 0.5 × 0.5 = 0.03125   (5 steps)
    0.5^20 = 0.00000095                         (20 steps — practically zero!)
    
    The gradient disappears. Early time steps get NO learning signal.
    The network "forgets" the beginning of the sequence.
    
**Exploding**: If W_hh has large values (eigenvalues > 1):

    2.0 × 2.0 × 2.0 × 2.0 × 2.0 = 32          (5 steps)
    2.0^20 = 1,048,576                           (20 steps — overflow!)
    
    Weights blow up to infinity. Training becomes unstable (NaN loss).
    
---

#### Visualizing the Problem

Imagine passing a message through a long chain of people whispering:

---
    [Person 1] →whisper→ [Person 2] →whisper→ ... →whisper→ [Person 20]
    
    Vanishing: Each person forgets a little → Person 20 hears nothing
    Exploding: Each person exaggerates a little → Person 20 hears screaming
---

For text: "The cat, which was sitting on the mat in the garden behind the house near the old oak tree, **was** ___"

A vanilla RNN processing this needs the gradient from "was" to reach back to "cat" (the subject). 
That's ~15 time steps of multiplication — the gradient vanishes and the network can't learn 
that "cat" connects to "was."

In an RNN, backpropagation goes backward through time (called Backpropagation Through Time — BPTT). 
If the sequence is 100 words long, gradients must flow back 100 steps. Each step multiplies by W_hidden, 
and if that value is less than 1, the gradient shrinks exponentially — the network forgets early words.

This is why variants like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) were invented — 
they add gates that control what to remember and what to forget, solving the long-range memory problem.

---

##### The Vanishing Gradient Problem Breakdown 

In an RNN, backpropagation goes backward through time (called Backpropagation Through Time — BPTT). 
If the sequence is 100 words long, gradients must flow back 100 steps. Each step multiplies by W_hidden, 
and if that value is less than 1, the gradient shrinks exponentially — the network forgets early words.

This is why variants like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) were invented — 
they add gates that control what to remember and what to forget, solving the long-range memory problem.

##### vanishing gradient problem visualized:

**Flow of Data** 

    Regular Neural Network 
    
    FORWARD: Signal flows through ALL paths at once
    ─────────────────────────────────────────────────
    
    Input ──► [Neuron A] ──► [Neuron C] ──► [Neuron E] ──► Output = 0.6
          ──► [Neuron B] ──► [Neuron D] ──►                Expected = 1.0
                                                            Error = 0.4
    
    All neurons fire at once. One complete pass. Done.
    
    
    BACKWARD: Go back and adjust EVERY connection
    ─────────────────────────────────────────────────
    
    Input ◄── [Neuron A] ◄── [Neuron C] ◄── [Neuron E] ◄── Error = 0.4
          ◄── [Neuron B] ◄── [Neuron D] ◄──
    
    "Hey every connection, here's how much you messed up. Adjust."
    
    All weights updated. One complete backward pass. Done.

---

#### Explanation and Breakdown of Vanishing Gradient Problem with forward and backward pass 

    FORWARD PASS (left to right):
    ─────────────────────────────
    
    What flows: ACTIVATIONS (h) — the actual computed values
    Purpose:    Making a prediction
    When:       First
    
        Input          h(1)=0.7      h(2)=0.3      h(3)=0.8
          │               │              │              │
          ▼               ▼              ▼              ▼
       ┌──────┐  0.7  ┌──────┐  0.3  ┌──────┐  0.8  ┌──────┐──► Prediction
       │ t=1  │──────►│ t=2  │──────►│ t=3  │──────►│ t=4  │    = 0.6
       └──────┘       └──────┘       └──────┘       └──────┘
       
        •  Hidden activations: h(1),h(2),h(3),h(4)
        
        •  Output logit: z
        
        •  Prediction (output activation): y ̂
    
       These values are computed as:
       h(t) = tanh(W_input · X(t) + W_hidden · h(t-1) + b)
    
       This is what you asked about ▲ — YES these flow forward
       
    Example: Forward Pass — Processing "I love watching movies on weekends"
    
        Input:    "I"        "love"   "watching"   "movies"     "on"     "weekends"
                   │           │          │           │           │           │
                   ▼           ▼          ▼           ▼           ▼           ▼
                ┌──────┐   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
        h(0)=0 →│ RNN  │──→│ RNN  │───→│ RNN  │───→│ RNN  │───→│ RNN  │───→│ RNN  │──→ Output
                │ t=1  │   │ t=2  │    │ t=3  │    │ t=4  │    │ t=5  │    │ t=6  │   (Prediction)
                └──────┘   └──────┘    └──────┘    └──────┘    └──────┘    └──────┘
                  h(1)       h(2)        h(3)        h(4)        h(5)        h(6)
        
        ─────────────────── Signal flows FORWARD through time ──────────────────────►
    
        
    BACKWARD PASS (right to left):
    ──────────────────────────────
    
    What flows: GRADIENTS (∂Loss/∂W) — how much each weight caused the error
    Purpose:    Fixing the weights so next prediction is better
    When:       After forward pass, after calculating the error
    
        ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐◄── Loss
        │ t=1  │◄──────│ t=2  │◄──────│ t=3  │◄──────│ t=4  │    (error)
        └──────┘       └──────┘       └──────┘       └──────┘
          ▲               ▲              ▲              ▲
          │               │              │              │
        grad=0.03       grad=0.1       grad=0.4       grad=1.0
        
        These tell each neuron: "here's how much YOU
        contributed to the final error — adjust yourself"
        
    Backward Pass — Gradients Flow Backward
    
                 ┌──────┐   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
                 │ t=1  │◄──│ t=2  │◄───│ t=3  │◄───│ t=4  │◄───│ t=5  │◄───│ t=6  │◄── Loss
                 └──────┘   └──────┘    └──────┘    └──────┘    └──────┘    └──────┘
        
        ◄──────────────── Gradients flow BACKWARD through time ─────────────────────
        
        Gradient
        strength:  ░░░        ░░░░       ▒▒▒▒▒      ▓▓▓▓▓▓     ████████   ██████████
                   TINY!      very       small       medium      strong      FULL
                   ≈0.001     small                                         GRADIENT
                              ≈0.01      ≈0.08       ≈0.3        ≈0.7       = 1.0
    
    
    The Key Difference    
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   ACTIVATION (forward)        GRADIENT (backward)        │
    │   ─────────────────────       ──────────────────         │
    │   "What is my output?"        "How wrong was I?"         │
    │                                                          │
    │   Computed from inputs         Computed from the error   │
    │   Flows LEFT → RIGHT          Flows RIGHT → LEFT         │
    │   Used to MAKE prediction     Used to FIX weights        │
    │                                                          │
    │   Example: h = 0.7            Example: grad = 0.03       │
    │   "I computed 0.7"            "You were 0.03 responsible │
    │                                for the mistake"          │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    
    **Why It Vanishes — The Math**
    
    At each step backward, gradient is MULTIPLIED by W_hidden:
    
    Say W_hidden = 0.5
    
    Step 6 (start):  gradient = 1.0
                                × 0.5
    Step 5:          gradient = 0.5
                                × 0.5
    Step 4:          gradient = 0.25
                                × 0.5
    Step 3:          gradient = 0.125
                                × 0.5
    Step 2:          gradient = 0.0625
                                × 0.5
    Step 1:          gradient = 0.03125   ◄── Almost ZERO!
    
    
    Gradient
      ▲
      │
    1 │ █
      │ █
      │ █
      │ █  █
      │ █  █
      │ █  █
      │ █  █  █
      │ █  █  █
      │ █  █  █  █
      │ █  █  █  █  █
      │ █  █  █  █  █  █
      └──┴──┴──┴──┴──┴──────►
      t=6 t=5 t=4 t=3 t=2 t=1
      (recent)         (early)
    
    The Consequence
    
    Sentence: "The movie that my friend who lives in Paris recommended was ____"
    
         ◄─────────── 10 words apart ───────────────►
    
       "movie"                                      "was ____"
         │                                              │
    IMPORTANT!                                    Prediction
    (subject)                                     happens here
         │                                              │
    But gradient                                  Gradient is
    is ≈ 0 here                                   strong here
         │                                              │
         ▼                                              ▼
    Weights DON'T                                 Weights update
    update for this                               normally for
    connection                                    recent words
         │                                              │
         └──────► Network FORGETS "movie" ◄─────────────┘
                  and can't connect it to
                  the prediction!

So the Vanishing Gradient Problem is NOT about the forward signal dying.
The forward pass works fine — activations flow through just fine.
**The problem is only during the backward pass when we try to update weights:**

---

---
#### Solutions to Vanishing/Exploding Gradients

---
    Problem              Solution                          How It Helps
    ─────────────────────────────────────────────────────────────────────────
    Exploding gradients  Gradient clipping                 Cap gradient norm to a max value
    
    Vanishing gradients  LSTM / GRU                        Gates control information flow, 
                                                           creating "gradient highways"
    
    Vanishing gradients  Skip / Residual connections       Add shortcuts so gradient doesn't 
                                                           have to pass through every step
    
    Exploding gradients  Weight initialization             Careful init (orthogonal) keeps
                                                           eigenvalues near 1
    
    Both                 Gradient norm monitoring           Detect problems early in training
---

Gradient clipping is simple — if the gradient norm exceeds a threshold, scale it down:

    if ||gradient|| > max_norm:
        gradient = gradient × (max_norm / ||gradient||)

But the real breakthrough for vanishing gradients was the LSTM.

---

## Part 4 — LSTM (Long Short-Term Memory)

The LSTM was invented by Hochreiter & Schmidhuber (1997) specifically to solve the vanishing gradient problem.

The key idea: instead of one hidden state that gets squashed by tanh at every step, add a **cell state** — 
a conveyor belt of information that can flow across time steps with **minimal modification**.

LSTM adds a second path called the cell state — think of it as a separate long-term memory that runs alongside the hidden state:

    Regular RNN:
    h(0) ──×W──×W──×W──×W──×W──×W──► h(6)     Multiply, multiply, multiply
                                                 Signal DECAYS
    
    LSTM (adds a "highway" for memory):
    
         ┌────────────────────────────────────────────┐
         │        CELL STATE (memory highway)         │  ◄── Direct path!
         │  c(0)───────────────────────────────►c(6)  │      No repeated
         └────┬───────┬───────┬───────┬───────┬───────┘      multiplication
              │       │       │       │       │
            GATE    GATE    GATE    GATE    GATE    ◄── Gates CHOOSE
           forget  forget  forget  forget  forget       what to remember
            /add    /add    /add    /add    /add        and what to forget
              │       │       │       │       │
            ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
            │t=1│   │t=2│   │t=3│   │t=4│   │t=5│
            └───┘   └───┘   └───┘   └───┘   └───┘
    
    Gradient can flow DIRECTLY along the highway
    without being multiplied at every step!

---

    Regular RNN — one path (short-term memory only):

      h(0)──►[RNN]──h(1)──►[RNN]──h(2)──►[RNN]──h(3)──►
              Overwritten    Overwritten    Overwritten
              every step     every step     every step
    
    
    LSTM — two paths:
    
      c(0)════════════════════════════════════════════════►     CELL STATE
             ║            ║             ║                       (long-term memory)
             ║            ║             ║                       Protected highway!
             ║            ║             ║  
      h(0)──►[LSTM]──h(1)──►[LSTM]──h(2)──►[LSTM]──h(3)──► HIDDEN STATE
                                                                (short-term memory / output)
    
      ═══ = cell state (long-term memory, barely modified)
      ─── = hidden state (short-term memory, changes a lot)

---

##### **The Three Gates**
    An LSTM has three gates that control the flow of information. Each gate is a small neural network itself (sigmoid activation, outputs 0 to 1).
    Think of them as security guards at a building:
    
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  GATE 1: FORGET GATE         "What should I throw away?" │
    │  ─────────────────                                       │
    │  Looks at: current input + previous hidden state         │
    │  Outputs: 0 to 1 for each cell state value               │
    │           0 = completely forget this                     │
    │           1 = completely keep this                       │
    │                                                          │
    │  Example: New sentence started → forget old subject      │
    │                                                          │
    │                                                          │
    │  GATE 2: INPUT GATE          "What new info should I     │
    │  ──────────────────           store?"                    │
    │  Looks at: current input + previous hidden state         │
    │  Outputs: 0 to 1 for each new candidate value            │
    │           0 = don't store this                           │
    │           1 = store this                                 │
    │                                                          │
    │  Example: New subject "movie" → store it!                │
    │                                                          │
    │                                                          │
    │  GATE 3: OUTPUT GATE         "What should I output       │
    │  ──────────────────           right now?"                │
    │  Looks at: current input + previous hidden state         │
    │  Outputs: 0 to 1 for each cell state value               │
    │           0 = don't output this now                      │
    │           1 = output this now                            │
    │                                                          │
    │  Example: Predicting verb → output the stored subject    │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    

#### The Core Intuition

Think of the cell state as a **notebook**:

* The **forget gate** decides what to erase from the notebook
* The **input gate** decides what new information to write in
* The **output gate** decides what to read from the notebook to produce the current output

---
    
    Vanilla RNN:    h(t) = tanh(W·[h(t-1), x(t)] + b)        ← everything squashed every step
    
    LSTM:           C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)       ← cell state flows with minimal change
                    h(t) = o(t) ⊙ tanh(C(t))                 ← output is a filtered view of cell

---

#### LSTM Architecture — The 4 Components
---

                            Cell State C(t) — the "conveyor belt"
         ┌──────────────────────────────────────────────────────────┐
         │                                                          │
    C(t-1)──►[  × forget  ]──►[ + ]──────────────────────────────►C(t)
         │       gate        ↑                                      │
         │                   │                                      │
         │           [  × input gate  ]                             │
         │                   ↑                                      │
         │              [ tanh: C̃(t) ]                              │
         │                   ↑                                      │
         │         ┌─────────┴─────────┐                            │
         │         │  [h(t-1), x(t)]   │                   [ tanh ]
         │         │  concatenated     │                      │
         │         └─────────┬─────────┘               [ × output gate ]
         │                   │                                │
    h(t-1)──►───────────────►├─────────────────────────────►h(t)──►
                             │
                           x(t)
---
    
Inside One LSTM Cell — Full Diagram

    c(t-1) (previous cell state / long-term memory)
                      ║
          ┌───────────╨──────────────────────────────────────┐
          │           ║                                      │
          │     ┌─────╨─────┐                                │
          │     │  ×  ×  ×  │◄── FORGET gate (f)             │
          │     │  multiply │    "erase these memories"      │
          │     └─────╥─────┘                                │
          │           ║                                      │
          │     ┌─────╨─────┐                                │
          │     │  +  +  +  │◄── INPUT gate (i) × candidate  │
          │     │    add    │    "write new memories"        │
          │     └─────╥─────┘                                │
          │           ║                                      │
          │           ╠════════════════════════════════►c(t) │
          │           ║                        (new cell     │
          │     ┌─────╨─────┐                   state)       │
          │     │   tanh    │                                │
          │     └─────╥─────┘                                │
          │     ┌─────╨─────┐                                │
          │     │  ×  ×  ×  │◄── OUTPUT gate (o)             │
          │     │  multiply │    "read these memories"       │
          │     └─────╥─────┘                                │
          │           ║                                      │
          └───────────╨──────────────────────────────────────┘
                      ║
                    h(t) ──────────────────────────────► output
               (new hidden state)
    
    
    Where do the gates get their values from?
    
              ┌─────────────────────┐
              │    h(t-1)           │  previous hidden state
              │      +              │
              │    X(t)             │  current input
              │      │              │
              │      ▼              │
              │  ┌────────┐         │
              │  │sigmoid │──► f    │  forget gate  (values 0-1)
              │  ├────────┤         │
              │  │sigmoid │──► i    │  input gate   (values 0-1)
              │  ├────────┤         │
              │  │sigmoid │──► o    │  output gate  (values 0-1)
              │  ├────────┤         │
              │  │  tanh  │──► c̃    │  candidate    (values -1 to 1)
              │  └────────┘         │
              └─────────────────────┘
    
    Each gate has its OWN set of weights — so an LSTM
    has 4× more parameters than a regular RNN

---

**Step by step:**

**1. Forget Gate** — "What should I erase from memory?"

    f(t) = σ(W_f · [h(t-1), x(t)] + b_f)
    
    Output: vector of values between 0 and 1
    * 0 = completely forget this memory
    * 1 = completely keep this memory
    
    Example: When you see a new subject in a sentence, forget the old subject.

**2. Input Gate** — "What new information should I store?"

    i(t) = σ(W_i · [h(t-1), x(t)] + b_i)        ← how much to write (0-1)
    C̃(t) = tanh(W_C · [h(t-1), x(t)] + b_C)     ← what to write (-1 to 1)
    
    The input gate has two parts:
    * i(t) controls HOW MUCH to write (sigmoid → 0 to 1)
    * C̃(t) is WHAT to write (tanh → -1 to 1)

**3. Cell State Update** — "Update the notebook"

    C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)
    
    ⊙ means element-wise multiplication.
    * First term: old memories, selectively forgotten
    * Second term: new information, selectively written
    
    This is the magic! The cell state can flow unchanged if f(t)=1 and i(t)=0.
    That means gradients can flow backward through time WITHOUT vanishing.

**4. Output Gate** — "What should I output right now?"

    o(t) = σ(W_o · [h(t-1), x(t)] + b_o)
    h(t) = o(t) ⊙ tanh(C(t))
    
    The cell state holds the full memory, but the output is a filtered version.
    Not everything in memory is relevant to the current output.

---

#### Why LSTM Solves the Vanishing Gradient

In a vanilla RNN, the gradient must pass through tanh at every time step → vanishes.

In an LSTM, the cell state update is:

    C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)

The gradient of C(t) with respect to C(t-1) is simply **f(t)** — the forget gate.

    ∂C(t)/∂C(t-1) = f(t)

If the forget gate is close to 1, the gradient flows through **undiminished**. 
This creates a "gradient highway" — information (and gradients) can travel across 
many time steps without decaying.

---
    Vanilla RNN gradient path:   tanh' × W × tanh' × W × tanh' × W × ...  → vanishes
    
    LSTM gradient path:          f(t) × f(t-1) × f(t-2) × ...             → stays near 1 if gates allow it
---

---

**The Math (Simple Version)**

    Given: X(t) = current input, h(t-1) = previous hidden state, c(t-1) = previous cell state
    
    Step 1 — FORGET gate:    f = sigmoid(W_f · [h(t-1), X(t)] + b_f)
    Step 2 — INPUT gate:     i = sigmoid(W_i · [h(t-1), X(t)] + b_i)
    Step 3 — CANDIDATE:      c̃ = tanh(W_c · [h(t-1), X(t)] + b_c)
    Step 4 — UPDATE cell:    c(t) = f × c(t-1)  +  i × c̃
                                     ▲                ▲
                                     │                │
                              forget old      add new info
                              memories
    
    Step 5 — OUTPUT gate:    o = sigmoid(W_o · [h(t-1), X(t)] + b_o)
    Step 6 — HIDDEN state:   h(t) = o × tanh(c(t))
    
    
    **Numeric Walkthrough**
    
    Processing: "not" → "good"    (sentiment analysis)
    
    Encoding: "not" = [-1, 0.5],  "good" = [0.8, 0.9]
    Initial:  h(0) = [0, 0],  c(0) = [0, 0]
    
    Let's simplify to 1 dimension and say the gates compute:
    
    ═══════════════════════════════════════════════════
    TIME STEP 1: Processing "not"
    ═══════════════════════════════════════════════════
    
      Forget gate:    f = sigmoid(0.1)  = 0.52
                      "Forget about half of previous memory"
                      (not much to forget, memory was 0)
    
      Input gate:     i = sigmoid(0.9)  = 0.71
                      "This word is important — store 71% of it"
    
      Candidate:      c̃ = tanh(-0.8)   = -0.66
                      "The content to potentially store is negative"
    
      Cell state:     c(1) = 0.52 × 0    +   0.71 × (-0.66)
                           = 0           +   (-0.47)
                           = -0.47
                      "Long-term memory now stores: NEGATIVE sentiment"
    
      Output gate:    o = sigmoid(0.6)  = 0.65
      Hidden state:   h(1) = 0.65 × tanh(-0.47)
                           = 0.65 × (-0.44)
                           = -0.29
    
    ═══════════════════════════════════════════════════
    TIME STEP 2: Processing "good"
    ═══════════════════════════════════════════════════
    
      Forget gate:    f = sigmoid(0.3)  = 0.57
                      "Keep 57% of the memory of 'not'"
                      (doesn't fully erase it!)
    
      Input gate:     i = sigmoid(0.7)  = 0.67
                      "Store 67% of this new word"
    
      Candidate:      c̃ = tanh(0.6)    = 0.54
                      "The word 'good' is positive"
    
      Cell state:     c(2) = 0.57 × (-0.47)   +   0.67 × 0.54
                           = -0.27            +   0.36
                           = 0.09
                                ▲                    ▲
                                │                    │
                        Memory of "not"       New info "good"
                        still pulling         pulling positive
                        negative!
    
                      "Net memory is SLIGHTLY positive (0.09)"
                      "The negativity from 'not' partially survived!"
    
      Output gate:    o = sigmoid(0.5)  = 0.62
      Hidden state:   h(2) = 0.62 × tanh(0.09)
                           = 0.62 × 0.09
                           = 0.056
    
      Final output ≈ 0.056 (barely positive, near neutral — "not good" ≈ mixed/negative sentiment)


**Why This Solves Vanishing Gradients**
    
    Regular RNN cell state update:
      h(t) = tanh(W × h(t-1) + ...)
             ▲
             └── gradient must pass through tanh AND multiply by W
                 at EVERY step → vanishes!
    
    
    LSTM cell state update:
      c(t) = f × c(t-1)  +  i × c̃
             ▲
             └── gradient for c(t-1) is just f (the forget gate)
                 which is between 0 and 1
                 
                 If f ≈ 1, gradient flows through PERFECTLY
                 No tanh squashing! No W multiplication!
    
    
**Gradient flow comparison:**
    
    Regular RNN:    grad × W × tanh' × W × tanh' × W × tanh' → ≈ 0
                          ▲             ▲             ▲
                      shrinks!      shrinks!      shrinks!
                      
    
    LSTM:           grad × f    ×    f    ×    f           → preserved!
                         ≈0.9       ≈0.9      ≈0.9
                         
                    After 10 steps: 0.9^10 = 0.35  (still meaningful!)
                    vs RNN:         0.5^10 = 0.001 (dead!)
    
#### LSTM Walking Example

Let's process: "The **cat** sat on the mat and **it** was happy"

The network needs to remember that "it" refers to "cat":

---
    t=1 "The":    forget gate ≈ 1 (nothing to forget yet)
                  input gate  ≈ low (just an article, not important)
                  cell state  ≈ 0 (almost nothing stored)
    
    
    t=2 "cat":    forget gate ≈ 1 (keep everything)
                  input gate  ≈ HIGH (subject! important!)
                  cell state  ← stores "subject = cat" strongly
    
    
    t=3-7:        forget gate ≈ 1 (keep remembering the subject)
    "sat on       input gate  ≈ low/medium
    the mat       cell state  retains "cat" across all these steps
    and"
    
    
    t=8 "it":     forget gate ≈ 1 (still need the subject)
                  output gate ≈ HIGH (now the stored subject is relevant!)
                  h(t) reads "cat" from cell → correctly resolves "it" = "cat"
---

#### GRU vs LSTM

---

    LSTM                                    GRU
    ─────────────────────────────────────────────────────────────
    3 gates (forget, input, output)         2 gates (reset, update)
    2 states (cell state + hidden state)    1 state (hidden state only)
    More parameters                         Fewer parameters (~25% less)
    Better on very long sequences           Better on smaller datasets
    More expressive                         Faster to train
    
---

#### GRU Equations

---
    Reset gate:     r(t) = σ(W_r · [h(t-1), x(t)] + b_r)
    
    Update gate:    z(t) = σ(W_z · [h(t-1), x(t)] + b_z)
    
    Candidate:      h̃(t) = tanh(W · [r(t) ⊙ h(t-1), x(t)] + b)
    
    New hidden:     h(t) = (1 - z(t)) ⊙ h(t-1) + z(t) ⊙ h̃(t)
---

The **update gate z(t)** does double duty:

* z(t) close to 0 → keep old hidden state (like LSTM forget gate = 1)
* z(t) close to 1 → replace with new candidate (like LSTM input gate = 1)

The **reset gate r(t)** controls how much of the previous hidden state 
to use when computing the new candidate. If r(t) ≈ 0, the GRU ignores 
the previous hidden state entirely — like starting fresh.

---

#### GRU Architecture Diagram

---
    h(t-1) ──────────────────────┬──────────[ × (1-z) ]──────────►[ + ]──► h(t)
         │                       │                                   ↑
         │                       │                              [ × z(t) ]
         │                       │                                   ↑
         │                  [ × r(t) ]                          [ tanh: h̃ ]
         │                       │                                   ↑
         ├──► [ σ: reset gate ] ─┘                              [h(t-1)·r, x(t)]
         │         ↑
         ├──► [ σ: update gate z(t) ]──────────────────────────────►│
         │         ↑                                                │
         │    [h(t-1), x(t)]                                        │
         │                                                          │
    x(t) ─────────────────────────────────────────────────────────────
---

**Inside One GRU Cell**

    h(t-1) (previous hidden state)
             │
             ├──────────────────────────────┐
             │                              │
             ▼                              ▼
       ┌────────────┐                ┌────────────┐
       │  sigmoid   │                │  sigmoid   │
       │  RESET (r) │                │ UPDATE (z) │
       └─────┬──────┘                └──────┬─────┘
             │                              │
             ▼                              │
       r × h(t-1)                           │
             │                              │
             ▼                              │
       ┌────────────┐                       │
       │   tanh     │                       │
       │ candidate  │                       │
       │   h̃(t)     │                       │
       └─────┬──────┘                       │
             │                              │
             ▼                              ▼
       ┌─────────────────────────────────────────┐
       │                                         │
       │  h(t) = (1 - z) × h(t-1) + z × h̃(t)     │
       │              ▲                  ▲       │
       │              │                  │       │
       │         keep old               add new  │
       │         memory             i   nfo      │
       │                                         │
       └──────────────────────────────┬──────────┘
                                      │
                                      ▼
                                    h(t) ──► output

---

##### The Math

    Step 1 — RESET gate:     r = sigmoid(W_r · [h(t-1), X(t)] + b_r)
    Step 2 — UPDATE gate:    z = sigmoid(W_z · [h(t-1), X(t)] + b_z)
    Step 3 — CANDIDATE:      h̃ = tanh(W_h · [r × h(t-1), X(t)] + b_h)
                                             ▲
                                             │
                                       reset gate controls
                                       how much past to use
                                       when making candidate
    
    Step 4 — FINAL STATE:    h(t) = (1 - z) × h(t-1)  +  z × h̃      
                                         ▲                 ▲
                                         │                 │
                                    keep old               add new
                                    (if z=0.8,             (remaining 0.2
                                    keep 80%)              goes to new)

---
##### LSTM vs GRU — Side by Side

                        LSTM                           GRU
                  ══════════════                 ══════════════
    
    States:       cell (c) + hidden (h)         hidden (h) only
    
    Gates:        forget ─┐                     update (z) ──── does BOTH
                  input  ─┘                                     forget + input
                  output                        reset (r)
    
    Cell update:  c(t) = f×c(t-1) + i×c̃        h(t) = (1-z)×h(t-1) + z×h̃
                         ▲           ▲                  ▲              ▲
                      forget      input              keep old       add new
                      (independent)                  (linked! z and 1-z)
    
    Key diff:     forget and input gates         update gate forces a tradeoff:
                  are INDEPENDENT                more old = less new (always)
                  Can forget a lot AND           Can't keep everything AND
                  add a lot simultaneously       add everything simultaneously
    
    Parameters:   4 weight matrices              3 weight matrices
                  (more expensive)               (cheaper, faster)
    
    Performance:  Better on complex tasks        Often comparable
                  with long dependencies         Preferred when speed matters
                  and large datasets             or data is limited
    
    
    When to use what?
    
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Use LSTM when:           Use GRU when:             │
    │  ─────────────            ─────────────             │
    │  • Very long sequences    • Shorter sequences       │
    │  • Complex dependencies   • Need faster training    │
    │  • Plenty of data         • Limited data            │
    │  • Accuracy is priority   • Speed is priority       │
    │  • Default safe choice    • Quick experiments       │
    │                                                     │
    │  In practice: try both, see which works better      │
    │  for YOUR specific problem                          │
    │                                                     │
    └─────────────────────────────────────────────────────┘

---

{{RNN_IMAGE}}

---

##### The Full Evolution Timeline

    Simple RNN (1986)     "I have memory but I forget quickly"
      │
      │  Problem: vanishing gradients
      ▼
    LSTM (1997)           "I have gates to control what to remember"
          │                3 gates, 2 states, powerful but complex
          │
          │  Can we simplify?
          ▼
    GRU (2014)            "Same idea, fewer gates, nearly as good"
          │                2 gates, 1 state, faster
          │
          │  But even these struggle with VERY long sequences...
          ▼
    Transformer (2017)    "Forget recurrence entirely —
                           let every word look at every other word
                           directly using ATTENTION"
                           (This is what GPT, BERT, and I (Claude) use!)


"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect              | Vanilla RNN                          | LSTM                                        | GRU                                        |
|---------------------|--------------------------------------|---------------------------------------------|--------------------------------------------|
| **Parameters**      | O(h² + h·d)  — ~1x baseline          | O(4·(h² + h·d))  — ~4x vanilla RNN          | O(3·(h² + h·d))  — ~3x vanilla RNN         |
| **Forward Pass**    | O(T · h²)  per sequence              | O(T · 4h²)  per sequence                    | O(T · 3h²)  per sequence                   |
| **BPTT (Training)** | O(T · h²) — but gradients vanish     | O(T · 4h²) — gradients preserved            | O(T · 3h²) — gradients preserved           |
| **Inference Time**  | O(T · h²) — sequential (no parallel) | O(T · 4h²) — sequential (no parallel)       | O(T · 3h²) — sequential (no parallel)      |
| **Memory (train)**  | O(T · h) — store all hidden states   | O(T · 2h) — store h(t) + C(t) per step      | O(T · h) — store h(t) per step             |
| **Long-range deps** | Poor (vanishing gradient)            | Excellent (cell state highway)              | Good (update gate highway)                 |

Where: **T** = sequence length, **h** = hidden size, **d** = input dimension

**Key insight:** All RNNs are **sequential** — each time step depends on the previous one, so they 
**cannot be parallelized** across time steps. This is the main reason Transformers replaced them — 
Transformers process all positions in parallel using attention.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Vanilla RNN Forward Pass (NumPy)": {
        "description": "A from-scratch vanilla RNN forward pass. Processes a sequence step-by-step, "
                       "carrying a hidden state. Shows exactly what h(t) = tanh(W_xh·x(t) + W_hh·h(t-1) + b) looks like in code.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

# --- Dimensions ---
input_size  = 3   # size of each input vector x(t)
hidden_size = 4   # size of hidden state h(t)
seq_length  = 5   # number of time steps T

# --- Weight matrices (normally learned, here random) ---
W_xh = np.random.randn(hidden_size, input_size)  * 0.5   # input → hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5   # hidden → hidden
b_h  = np.zeros(hidden_size)                              # bias

# --- Random input sequence: T vectors of size input_size ---
X = np.random.randn(seq_length, input_size)

# --- Forward pass ---
h = np.zeros(hidden_size)  # h(0) = zero vector
all_hidden = []

print("=== Vanilla RNN Forward Pass ===\\n")
for t in range(seq_length):
    z = W_xh @ X[t] + W_hh @ h + b_h    # linear combination
    h = np.tanh(z)                         # activation
    all_hidden.append(h.copy())
    print(f"t={t+1}: h = [{', '.join(f'{v:+.4f}' for v in h)}]")

print(f"\\nFinal hidden state h({seq_length}):")
print(f"  {h.round(4)}")
print(f"\\nThis final hidden state encodes the entire sequence.")
'''
    },

    "Vanilla RNN Sentiment Demo": {
        "description": "The 'not good' sentiment example from the theory section, implemented in code. "
                       "Shows how memory of 'not' drags the output of 'good' negative.",
        "runnable": True,
        "code": '''import numpy as np

# --- Simple word encodings ---
words = {"not": -1.0, "good": 0.8, "very": 0.3, "bad": -0.9, "great": 0.9}

# --- Tiny RNN weights (hand-picked to illustrate the concept) ---
W_input  = 0.5    # how much to weight the current word
W_hidden = 0.9    # how much to weight the memory
b        = 0.0

def rnn_step(x, h_prev):
    """One step of a single-neuron RNN."""
    z = W_input * x + W_hidden * h_prev + b
    h = np.tanh(z)
    return h

def process_sentence(sentence_words):
    """Process a sentence word by word through the RNN."""
    h = 0.0   # h(0) = 0
    print(f'Processing: "{" ".join(sentence_words)}"')
    print(f"  h(0) = {h:.4f}  (no memory yet)\\n")
    
    for t, word in enumerate(sentence_words, 1):
        x = words[word]
        h_prev = h
        h = rnn_step(x, h_prev)
        z = W_input * x + W_hidden * h_prev + b
        print(f"  t={t} '{word}' (x={x:+.1f}):")
        print(f"    z = {W_input}×{x:+.1f} + {W_hidden}×{h_prev:+.4f} = {z:+.4f}")
        print(f"    h({t}) = tanh({z:+.4f}) = {h:+.4f}")
        print()
    
    sentiment = "POSITIVE ✓" if h > 0 else "NEGATIVE ✗"
    print(f"  Final h = {h:+.4f} → {sentiment}")
    print("-" * 50)
    return h

# --- Test sentences ---
print("=" * 50)
print("RNN Sentiment Demo: Memory Changes Meaning")
print("=" * 50 + "\\n")

process_sentence(["good"])              # Just "good" → positive
process_sentence(["not", "good"])       # "not good"  → negative (memory of "not" flips it)
process_sentence(["not", "very", "good"])  # "not very good" → still negative
process_sentence(["very", "good"])      # "very good" → strong positive
process_sentence(["not", "bad"])        # "not bad"   → slightly positive!
'''
    },

    # =========================================================================
    # VANISHING GRADIENT
    # =========================================================================

    "Vanishing Gradient Demonstration": {
        "description": "Demonstrates how gradients vanish in a vanilla RNN vs. stay healthy in an LSTM. "
                       "Shows the product of Jacobians shrinking exponentially over time steps.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

def demonstrate_vanishing_gradient(hidden_size=5, seq_lengths=[5, 10, 20, 50]):
    """Show how gradient magnitude decays over increasing sequence lengths."""
    
    # Random weight matrix with eigenvalues < 1 (typical after training)
    W = np.random.randn(hidden_size, hidden_size) * 0.5
    
    print("=" * 60)
    print("Vanishing Gradient in Vanilla RNN")
    print("=" * 60)
    print(f"\\nHidden size: {hidden_size}")
    print(f"Max eigenvalue of W_hh: {max(abs(np.linalg.eigvals(W))):.4f}")
    print(f"\\nGradient magnitude = ||∂h(T)/∂h(1)|| (product of Jacobians)")
    print("-" * 60)
    
    for T in seq_lengths:
        # Product of Jacobians: ∂h(t)/∂h(t-1) ≈ W_hh^T · diag(1-tanh²)
        # Simplified: just multiply W repeatedly (upper bound)
        gradient_product = np.eye(hidden_size)
        
        for t in range(T):
            # Each tanh derivative is between 0 and 1, use 0.65 as typical
            tanh_deriv = np.diag(np.random.uniform(0.3, 0.8, hidden_size))
            jacobian = W.T @ tanh_deriv
            gradient_product = jacobian @ gradient_product
        
        grad_norm = np.linalg.norm(gradient_product)
        bar = "█" * max(1, int(np.log10(grad_norm + 1e-30) + 30))
        print(f"  T = {T:3d} steps:  ||gradient|| = {grad_norm:.2e}  {bar}")
    
    print(f"\\n→ After 50 steps, gradient is essentially ZERO.")
    print(f"  The network CANNOT learn long-range dependencies!")
    print()
    
    # Now show what gradient clipping does for exploding
    print("=" * 60)
    print("Gradient Clipping (for Exploding Gradients)")
    print("=" * 60)
    
    grad = np.random.randn(hidden_size) * 100  # exploding gradient
    max_norm = 5.0
    
    print(f"\\n  Original gradient norm: {np.linalg.norm(grad):.2f}")
    
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    
    print(f"  After clipping (max_norm={max_norm}): {np.linalg.norm(grad):.2f}")
    print(f"  Direction preserved, magnitude capped ✓")

demonstrate_vanishing_gradient()
'''
    },

    # =========================================================================
    # LSTM
    # =========================================================================

    "LSTM Forward Pass (NumPy)": {
        "description": "A complete LSTM forward pass from scratch in NumPy. Shows all 4 gates "
                       "(forget, input, output, cell update) operating step by step. Compare with the "
                       "vanilla RNN to see how the cell state creates a gradient highway.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# --- Dimensions ---
input_size  = 3
hidden_size = 4
seq_length  = 5

# --- LSTM has 4 sets of weights (forget, input, cell candidate, output) ---
# Combined for efficiency: W has shape (4*hidden_size, input_size + hidden_size)
combined_size = input_size + hidden_size
W = np.random.randn(4 * hidden_size, combined_size) * 0.3
b = np.zeros(4 * hidden_size)

# Initialize forget gate biases to 1 (common practice — start by remembering)
b[0:hidden_size] = 1.0

# --- Random input sequence ---
X = np.random.randn(seq_length, input_size)

# --- Forward pass ---
h = np.zeros(hidden_size)   # hidden state
C = np.zeros(hidden_size)   # cell state (THE key LSTM addition)

print("=== LSTM Forward Pass ===\\n")

for t in range(seq_length):
    # Concatenate input and previous hidden state
    combined = np.concatenate([X[t], h])
    
    # All 4 gates in one matrix multiply (efficient)
    gates = W @ combined + b
    
    # Split into the 4 gates
    f_gate = sigmoid(gates[0*hidden_size : 1*hidden_size])   # forget gate
    i_gate = sigmoid(gates[1*hidden_size : 2*hidden_size])   # input gate
    C_cand = np.tanh(gates[2*hidden_size : 3*hidden_size])   # cell candidate
    o_gate = sigmoid(gates[3*hidden_size : 4*hidden_size])   # output gate
    
    # Cell state update (THE gradient highway)
    C = f_gate * C + i_gate * C_cand
    
    # Hidden state output
    h = o_gate * np.tanh(C)
    
    print(f"t={t+1}:")
    print(f"  Forget gate : [{', '.join(f'{v:.3f}' for v in f_gate)}]")
    print(f"  Input gate  : [{', '.join(f'{v:.3f}' for v in i_gate)}]")
    print(f"  Output gate : [{', '.join(f'{v:.3f}' for v in o_gate)}]")
    print(f"  Cell state  : [{', '.join(f'{v:+.3f}' for v in C)}]")
    print(f"  Hidden state: [{', '.join(f'{v:+.3f}' for v in h)}]")
    print()

print("Key observation: The cell state C accumulates information over time.")
print("The forget gate (close to 1 here due to bias=1) lets gradients flow!")
'''
    },

    # =========================================================================
    # GRU
    # =========================================================================

    "GRU Forward Pass (NumPy)": {
        "description": "A complete GRU forward pass from scratch. Simpler than LSTM with only 2 gates "
                       "(reset, update). Notice how the update gate z does double duty — controlling "
                       "both forgetting and inputting.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# --- Dimensions ---
input_size  = 3
hidden_size = 4
seq_length  = 5

# --- GRU has 3 sets of weights (reset, update, candidate) ---
combined_size = input_size + hidden_size
W_r = np.random.randn(hidden_size, combined_size) * 0.3   # reset gate
b_r = np.zeros(hidden_size)
W_z = np.random.randn(hidden_size, combined_size) * 0.3   # update gate
b_z = np.zeros(hidden_size)
W_h = np.random.randn(hidden_size, combined_size) * 0.3   # candidate
b_h = np.zeros(hidden_size)

# --- Random input sequence ---
X = np.random.randn(seq_length, input_size)

# --- Forward pass ---
h = np.zeros(hidden_size)

print("=== GRU Forward Pass ===\\n")

for t in range(seq_length):
    combined = np.concatenate([X[t], h])
    
    # Reset gate: how much of previous h to use for candidate
    r = sigmoid(W_r @ combined + b_r)
    
    # Update gate: interpolation between old h and new candidate
    z = sigmoid(W_z @ combined + b_z)
    
    # Candidate hidden state (with reset gate applied)
    combined_reset = np.concatenate([X[t], r * h])
    h_candidate = np.tanh(W_h @ combined_reset + b_h)
    
    # Final hidden state: interpolate
    h = (1 - z) * h + z * h_candidate
    
    print(f"t={t+1}:")
    print(f"  Reset gate  r: [{', '.join(f'{v:.3f}' for v in r)}]")
    print(f"  Update gate z: [{', '.join(f'{v:.3f}' for v in z)}]")
    print(f"  Hidden state : [{', '.join(f'{v:+.3f}' for v in h)}]")
    print()

print("Notice: z close to 0 → keep old h (memory preserved)")
print("        z close to 1 → use new candidate (memory updated)")
print("\\nGRU has ~25% fewer parameters than LSTM (2 gates vs 3)")
'''
    },

    # =========================================================================
    # COMPARISONS
    # =========================================================================

    "RNN vs LSTM vs GRU — Parameter Count": {
        "description": "Calculate and compare the exact number of trainable parameters in each architecture. "
                       "Shows why LSTMs are ~4x heavier and GRUs are ~3x heavier than vanilla RNNs.",
        "runnable": True,
        "code": '''import numpy as np

def count_params(name, input_size, hidden_size):
    """Count trainable parameters for RNN variants."""
    
    if name == "Vanilla RNN":
        # W_xh: hidden × input,  W_hh: hidden × hidden,  b: hidden
        w_xh = hidden_size * input_size
        w_hh = hidden_size * hidden_size
        bias = hidden_size
        total = w_xh + w_hh + bias
        gates = 1
        
    elif name == "LSTM":
        # 4 gates, each with W_xh, W_hh, b
        w_xh = 4 * hidden_size * input_size
        w_hh = 4 * hidden_size * hidden_size
        bias = 4 * hidden_size
        total = w_xh + w_hh + bias
        gates = 4
        
    elif name == "GRU":
        # 3 gates, each with W_xh, W_hh, b
        w_xh = 3 * hidden_size * input_size
        w_hh = 3 * hidden_size * hidden_size
        bias = 3 * hidden_size
        total = w_xh + w_hh + bias
        gates = 3
    
    return total, gates, w_xh, w_hh, bias

# --- Compare ---
input_size  = 128   # e.g., word embedding dimension
hidden_size = 256   # typical hidden size

print("=" * 65)
print("Parameter Count Comparison: RNN vs LSTM vs GRU")
print(f"Input size: {input_size},  Hidden size: {hidden_size}")
print("=" * 65)

results = []
for name in ["Vanilla RNN", "LSTM", "GRU"]:
    total, gates, w_xh, w_hh, bias = count_params(name, input_size, hidden_size)
    results.append((name, total, gates))
    print(f"\\n{name} ({gates} gate{'s' if gates > 1 else ''}):")
    print(f"  W_xh weights : {w_xh:>10,}")
    print(f"  W_hh weights : {w_hh:>10,}")
    print(f"  Biases       : {bias:>10,}")
    print(f"  TOTAL        : {total:>10,}")

baseline = results[0][1]
print(f"\\n{'─' * 65}")
print("Relative sizes:")
for name, total, _ in results:
    ratio = total / baseline
    bar = "█" * int(ratio * 10)
    print(f"  {name:<15} {total:>10,} params  ({ratio:.1f}x)  {bar}")
'''
    },

    "Sequence Length vs Memory Capacity": {
        "description": "Demonstrates how a vanilla RNN loses information over long sequences "
                       "while an LSTM-style approach preserves it. Uses a simple 'remember the first input' task.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def test_memory_retention(seq_length, hidden_size=10):
    """
    Test: can the network remember the FIRST input after many steps?
    Feed one meaningful input, then noise. Check if h still contains the signal.
    """
    input_size = 5
    
    # --- Vanilla RNN ---
    W_xh = np.random.randn(hidden_size, input_size) * 0.4
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.4
    
    # Meaningful first input
    signal = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    h_rnn = np.zeros(hidden_size)
    h_rnn = np.tanh(W_xh @ signal + W_hh @ h_rnn)
    h_after_signal_rnn = h_rnn.copy()
    
    # Feed noise for remaining steps
    for t in range(seq_length - 1):
        noise = np.random.randn(input_size) * 0.1
        h_rnn = np.tanh(W_xh @ noise + W_hh @ h_rnn)
    
    rnn_retention = np.corrcoef(h_after_signal_rnn, h_rnn)[0, 1]
    
    # --- LSTM-like (simplified: forget gate near 1) ---
    h_lstm = np.zeros(hidden_size)
    C_lstm = np.zeros(hidden_size)
    
    # Process signal
    f = np.ones(hidden_size) * 0.95   # high forget gate = strong memory
    i = np.ones(hidden_size) * 0.8
    C_lstm = f * C_lstm + i * np.tanh(W_xh @ signal)
    h_lstm = np.tanh(C_lstm)
    h_after_signal_lstm = h_lstm.copy()
    
    # Feed noise
    for t in range(seq_length - 1):
        noise = np.random.randn(input_size) * 0.1
        f = np.ones(hidden_size) * 0.95
        i = np.ones(hidden_size) * 0.1   # low input gate for noise
        C_lstm = f * C_lstm + i * np.tanh(W_xh @ noise)
        h_lstm = np.tanh(C_lstm)
    
    lstm_retention = np.corrcoef(h_after_signal_lstm, h_lstm)[0, 1]
    
    return rnn_retention, lstm_retention

# --- Test across different sequence lengths ---
print("=" * 65)
print("Memory Retention Test: Remember First Input After N Noise Steps")
print("=" * 65)
print(f"\\n{'Seq Length':<12} {'Vanilla RNN':<20} {'LSTM-style':<20}")
print("-" * 52)

for length in [5, 10, 20, 50, 100, 200]:
    rnn_r, lstm_r = test_memory_retention(length)
    rnn_bar  = "█" * max(0, int(abs(rnn_r)  * 20))
    lstm_bar = "█" * max(0, int(abs(lstm_r) * 20))
    print(f"  T={length:<6}  RNN: {rnn_r:>+.4f} {rnn_bar:<20}  LSTM: {lstm_r:>+.4f} {lstm_bar}")

print(f"\\n→ Vanilla RNN correlation drops to ~0 (memory gone)")
print(f"  LSTM retains high correlation (cell state preserves memory)")
'''
    },

    # =========================================================================
    # BPTT
    # =========================================================================

    "Backpropagation Through Time (Manual)": {
        "description": "Implements full BPTT for a vanilla RNN on a tiny sequence. Shows the forward pass, "
                       "loss computation, and backward pass with gradient accumulation across time steps.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

# --- Tiny RNN for BPTT demonstration ---
input_size  = 2
hidden_size = 3
output_size = 1
seq_length  = 4
lr = 0.01

# Weights
W_xh = np.random.randn(hidden_size, input_size)  * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5
W_hy = np.random.randn(output_size, hidden_size) * 0.5
b_h  = np.zeros(hidden_size)
b_y  = np.zeros(output_size)

# Toy data: predict sum of sequence
X = np.random.randn(seq_length, input_size)
target = np.array([np.sum(X)])  # target = sum of all inputs

print("=== Backpropagation Through Time (BPTT) ===\\n")

# ---- FORWARD PASS ----
print("--- Forward Pass ---")
h_states = [np.zeros(hidden_size)]  # h(0) = 0
z_states = []

for t in range(seq_length):
    z = W_xh @ X[t] + W_hh @ h_states[t] + b_h
    h = np.tanh(z)
    z_states.append(z)
    h_states.append(h)
    print(f"  t={t+1}: h = [{', '.join(f'{v:+.4f}' for v in h)}]")

# Output from final hidden state
y_pred = W_hy @ h_states[-1] + b_y
loss = 0.5 * np.sum((y_pred - target) ** 2)
print(f"\\n  Prediction: {y_pred[0]:.4f}")
print(f"  Target:     {target[0]:.4f}")
print(f"  Loss (MSE): {loss:.4f}")

# ---- BACKWARD PASS (BPTT) ----
print(f"\\n--- Backward Pass (BPTT) ---")

# Gradient of loss w.r.t. output
dL_dy = y_pred - target

# Gradient w.r.t. output weights
dL_dW_hy = dL_dy.reshape(-1, 1) @ h_states[-1].reshape(1, -1)
dL_db_y  = dL_dy

# Gradient flowing into the last hidden state
dL_dh = W_hy.T @ dL_dy

# Accumulate gradients over time steps
dL_dW_xh = np.zeros_like(W_xh)
dL_dW_hh = np.zeros_like(W_hh)
dL_db_h  = np.zeros_like(b_h)

print(f"\\n  Gradients flowing backward through time:")
for t in reversed(range(seq_length)):
    # tanh derivative: 1 - tanh²(z)
    dtanh = 1 - np.tanh(z_states[t]) ** 2
    
    # Gradient through tanh
    dL_dz = dL_dh * dtanh
    
    # Accumulate weight gradients
    dL_dW_xh += np.outer(dL_dz, X[t])
    dL_dW_hh += np.outer(dL_dz, h_states[t])
    dL_db_h  += dL_dz
    
    grad_norm = np.linalg.norm(dL_dh)
    print(f"  t={t+1}: ||∂L/∂h|| = {grad_norm:.6f}")
    
    # Propagate gradient to previous hidden state
    dL_dh = W_hh.T @ dL_dz

print(f"\\n  Gradient norms of accumulated weight gradients:")
print(f"    ||∂L/∂W_xh|| = {np.linalg.norm(dL_dW_xh):.6f}")
print(f"    ||∂L/∂W_hh|| = {np.linalg.norm(dL_dW_hh):.6f}")
print(f"    ||∂L/∂b_h||  = {np.linalg.norm(dL_db_h):.6f}")

# ---- WEIGHT UPDATE ----
W_xh -= lr * dL_dW_xh
W_hh -= lr * dL_dW_hh
b_h  -= lr * dL_db_h
W_hy -= lr * dL_dW_hy
b_y  -= lr * dL_db_y

# Check new prediction
h = np.zeros(hidden_size)
for t in range(seq_length):
    h = np.tanh(W_xh @ X[t] + W_hh @ h + b_h)
y_new = W_hy @ h + b_y
loss_new = 0.5 * np.sum((y_new - target) ** 2)

print(f"\\n--- After 1 BPTT Update ---")
print(f"  New prediction: {y_new[0]:.4f}")
print(f"  New loss:       {loss_new:.4f} (was {loss:.4f})")
print(f"  Loss decreased: {loss_new < loss} ✓" if loss_new < loss else f"  Loss: {loss_new:.4f}")
'''
    },

    # =========================================================================
    # BIDIRECTIONAL RNN
    # =========================================================================

    "Bidirectional RNN (NumPy)": {
        "description": "A bidirectional RNN that processes the sequence both forward and backward, "
                       "then concatenates the hidden states. Useful when full sequence context is available.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

input_size  = 3
hidden_size = 4
seq_length  = 5

# Forward RNN weights
W_xh_f = np.random.randn(hidden_size, input_size)  * 0.5
W_hh_f = np.random.randn(hidden_size, hidden_size) * 0.5
b_f    = np.zeros(hidden_size)

# Backward RNN weights (separate set)
W_xh_b = np.random.randn(hidden_size, input_size)  * 0.5
W_hh_b = np.random.randn(hidden_size, hidden_size) * 0.5
b_b    = np.zeros(hidden_size)

# Input sequence
X = np.random.randn(seq_length, input_size)

print("=== Bidirectional RNN ===\\n")

# --- Forward pass (left → right) ---
h_forward = []
h = np.zeros(hidden_size)
for t in range(seq_length):
    h = np.tanh(W_xh_f @ X[t] + W_hh_f @ h + b_f)
    h_forward.append(h.copy())

# --- Backward pass (right → left) ---
h_backward = []
h = np.zeros(hidden_size)
for t in range(seq_length - 1, -1, -1):
    h = np.tanh(W_xh_b @ X[t] + W_hh_b @ h + b_b)
    h_backward.insert(0, h.copy())

# --- Combine: concatenate forward and backward at each step ---
print("Hidden states at each time step:")
print(f"{'Step':<6} {'Forward (size 4)':<35} {'Backward (size 4)':<35} {'Combined (size 8)'}")
print("-" * 115)

for t in range(seq_length):
    combined = np.concatenate([h_forward[t], h_backward[t]])
    fwd_str = f"[{', '.join(f'{v:+.3f}' for v in h_forward[t])}]"
    bwd_str = f"[{', '.join(f'{v:+.3f}' for v in h_backward[t])}]"
    print(f"  t={t+1}  {fwd_str:<35} {bwd_str:<35} size={combined.shape[0]}")

print(f"\\nEach time step now has context from BOTH directions.")
print(f"Output dimension: {hidden_size} (forward) + {hidden_size} (backward) = {2 * hidden_size}")
print(f"\\nUse case: 'The ___ barked loudly' — backward RNN knows 'barked' when filling blank.")
'''
    },

    # =========================================================================
    # FULL TRAINING LOOP
    # =========================================================================

    "Mini RNN Training Loop (Character Prediction)": {
        "description": "A complete mini training loop: trains a vanilla RNN to predict the next character "
                       "in a short repeating pattern. Shows loss decreasing over epochs.",
        "runnable": True,
        "code": '''import numpy as np

np.random.seed(42)

# --- Character-level RNN on a tiny pattern ---
text = "abcabcabcabc"
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)

print(f"Vocab: {chars},  Text: '{text}'\\n")

# One-hot encode
def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1.0
    return v

# --- Model parameters ---
hidden_size = 10
W_xh = np.random.randn(hidden_size, vocab_size)  * 0.3
W_hh = np.random.randn(hidden_size, hidden_size) * 0.3
W_hy = np.random.randn(vocab_size, hidden_size)  * 0.3
b_h  = np.zeros(hidden_size)
b_y  = np.zeros(vocab_size)
lr   = 0.1

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# --- Training ---
print("=== Training RNN to predict next character ===\\n")

for epoch in range(201):
    # Forward pass
    h_states = [np.zeros(hidden_size)]
    z_states = []
    probs_list = []
    loss = 0
    
    for t in range(len(text) - 1):
        x = one_hot(char_to_idx[text[t]], vocab_size)
        target_idx = char_to_idx[text[t + 1]]
        
        z = W_xh @ x + W_hh @ h_states[t] + b_h
        h = np.tanh(z)
        z_states.append(z)
        h_states.append(h)
        
        logits = W_hy @ h + b_y
        p = softmax(logits)
        probs_list.append(p)
        loss -= np.log(p[target_idx] + 1e-10)
    
    # Backward pass (BPTT)
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    dW_hy = np.zeros_like(W_hy)
    db_h  = np.zeros_like(b_h)
    db_y  = np.zeros_like(b_y)
    dh_next = np.zeros(hidden_size)
    
    for t in reversed(range(len(text) - 1)):
        x = one_hot(char_to_idx[text[t]], vocab_size)
        target_idx = char_to_idx[text[t + 1]]
        
        dy = probs_list[t].copy()
        dy[target_idx] -= 1  # cross-entropy gradient
        
        dW_hy += np.outer(dy, h_states[t + 1])
        db_y  += dy
        
        dh = W_hy.T @ dy + dh_next
        dtanh = (1 - np.tanh(z_states[t]) ** 2)
        dz = dh * dtanh
        
        dW_xh += np.outer(dz, x)
        dW_hh += np.outer(dz, h_states[t])
        db_h  += dz
        dh_next = W_hh.T @ dz
    
    # Gradient clipping
    for param in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
        np.clip(param, -5, 5, out=param)
    
    # Update
    W_xh -= lr * dW_xh
    W_hh -= lr * dW_hh
    W_hy -= lr * dW_hy
    b_h  -= lr * db_h
    b_y  -= lr * db_y
    
    if epoch % 40 == 0:
        # Generate a prediction
        h = np.zeros(hidden_size)
        generated = text[0]
        x = one_hot(char_to_idx[text[0]], vocab_size)
        
        for _ in range(11):
            h = np.tanh(W_xh @ x + W_hh @ h + b_h)
            p = softmax(W_hy @ h + b_y)
            idx = np.argmax(p)
            generated += idx_to_char[idx]
            x = one_hot(idx, vocab_size)
        
        bar = "█" * max(1, int(30 - loss))
        print(f"  Epoch {epoch:>3}: loss={loss:.3f}  generated='{generated}'  {bar}")

print(f"\\n✓ The RNN learned the repeating 'abc' pattern!")
'''
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    from Required_Images.rnn_visual import  RNN_VISUAL_HTML, RNN_VISUAL_HEIGHT

    # Fallback static image
    rnn_img = _image_to_html(
        "Required_Images/rnn_preview.png",
        alt="RNN Explanation",
        width="50%"
    )
    theory_with_images = THEORY.replace("{{RNN_IMAGE}}", rnn_img)

    # Interactive component data
    interactive_components = [
        {
            "placeholder": "{{RNN_IMAGE}}",
            "html": RNN_VISUAL_HTML,
            "height": RNN_VISUAL_HEIGHT,
        }
    ]

    return {
        "theory": theory_with_images,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }

