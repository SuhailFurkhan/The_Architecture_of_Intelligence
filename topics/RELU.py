"""


### ReLU — Rectified Linear Unit

**What It Is ?**

ReLU is an activation function. That's it. It's a rule that each neuron applies to its weighted sum before passing the result to the next layer.
We've already discussed two activation functions — the step function and sigmoid. ReLU is the third, and it's the one that dominates modern deep learning.

The rule is almost embarrassingly simple:

                If z is positive → output z as-is
                If z is negative → output 0

Or mathematically: f(z) = max(0, z)

            Output
              ↑                          /
              |                        /
            4 |                      /
            3 |                    /
            2 |                  /
            1 |                /
            0 |───────────────┼──────────────→  z
            -1|
            -2|
            -3|
            -4|
                 -3  -2  -1   0   1   2   3   4

Everything negative gets flattened to zero. Everything positive passes through unchanged. That's the entire function.


**Why It Replaced Sigmoid**

To understand why ReLU matters, you need to understand what was wrong with sigmoid.

**The Vanishing Gradient Problem with Sigmoid**

Remember how backpropagation works: the error signal flows backward through the network, and at each layer it gets multiplied by the sigmoid derivative.
The sigmoid derivative has a maximum value of 0.25 (when the output is 0.5), and it gets smaller as the output approaches 0 or 1.

In a 10-layer network, the gradient gets multiplied by the sigmoid derivative 10 times:

Layer 10 (output):  gradient = 0.20
Layer 9:            gradient = 0.20 × 0.25 = 0.05
Layer 8:            gradient = 0.05 × 0.25 = 0.0125
Layer 7:            gradient = 0.0125 × 0.25 = 0.003
Layer 6:            gradient = 0.003 × 0.25 = 0.0008
Layer 5:            gradient = 0.0008 × 0.25 = 0.0002
...
Layer 1:            gradient ≈ 0.0000001

By the time the error signal reaches the early layers, it's essentially zero.
Those layers receive no useful learning signal. They stop learning. The network is 10 layers deep but only the last few layers are actually training.
This is the vanishing gradient problem, and it was the main reason deep networks didn't work well for decades.


How ReLU Fixes This
ReLU's derivative is:
If z > 0 → derivative = 1
If z < 0 → derivative = 0
If z = 0 → technically undefined, in practice treated as 0
That derivative of 1 is the key. When a neuron is active (z > 0), the gradient passes through completely unchanged. No shrinking. No multiplication by 0.25. The gradient at layer 1 can be just as strong as the gradient at layer 10.
Sigmoid through 10 layers:  0.25 × 0.25 × 0.25 × ... = ~0.0000001
ReLU through 10 layers:     1 × 1 × 1 × 1 × ... = 1
This is why ReLU unlocked deep learning. Networks could suddenly have 50, 100, even 1000 layers and still train successfully because the gradient didn't vanish.
"""