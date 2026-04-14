# Neural Network from Scratch: Complete Mathematical Guide

A step-by-step mathematical walkthrough of every operation in our 2-layer neural network, from initialization to trained model.

---

## Network Architecture

```
Input (784)  →  [W1, b1]  →  ReLU  →  Hidden (10)  →  [W2, b2]  →  Softmax  →  Output (10)
```

We classify 28×28 grayscale images of handwritten digits (MNIST) into 10 classes (digits 0-9).

**Notation used throughout:**

| Symbol | Meaning |
|--------|---------|
| m | Number of training samples |
| n | Number of input features (784 = 28×28) |
| X | Input matrix, shape (784, m) — each column is one sample |
| Y | True labels as one-hot vectors, shape (10, m) |
| W[l] | Weight matrix for layer l |
| b[l] | Bias vector for layer l |
| Z[l] | Pre-activation (linear output) at layer l |
| A[l] | Post-activation output at layer l |
| alpha | Learning rate |

---

## 1. Parameter Initialization

### The Problem: Symmetry Breaking

If all weights are initialized to the same value (e.g., zero), every neuron in a layer computes the same output. During backpropagation, they all receive the same gradient and update identically. The network is stuck — it can never learn different features. Random initialization breaks this symmetry.

### He Initialization (Recommended for ReLU)

For a layer with `n_in` input connections:

```
W ~ Normal(mean=0, std=sqrt(2 / n_in))
b = 0
```

**Why this specific scaling?**

Consider a single neuron computing `z = w1*x1 + w2*x2 + ... + wn*xn`. If inputs have variance `Var(x)` and weights have variance `Var(w)`:

```
Var(z) = n_in * Var(w) * Var(x)
```

To keep `Var(z) = Var(x)` (so signals don't explode or vanish across layers), we need:

```
Var(w) = 1 / n_in
```

ReLU zeroes out roughly half the values, cutting variance in half. To compensate:

```
Var(w) = 2 / n_in    →    std(w) = sqrt(2 / n_in)
```

### In Our Network

```
W1 ~ Normal(0, sqrt(2/784))   shape: (10, 784)    — 7,840 weights
b1 = 0                         shape: (10, 1)      — 10 biases

W2 ~ Normal(0, sqrt(2/10))    shape: (10, 10)     — 100 weights
b2 = 0                         shape: (10, 1)      — 10 biases

Total trainable parameters: 7,840 + 10 + 100 + 10 = 7,960
```

---

## 2. Forward Propagation

Forward propagation computes the network's prediction by passing data through each layer.

### 2.1 Hidden Layer — Linear Transform

```
Z[1] = W[1] . X + b[1]
```

Matrix dimensions: `(10, 784) . (784, m) + (10, 1) = (10, m)`

Each of the 10 hidden neurons computes a weighted sum of all 784 input features plus its bias. The bias `b[1]` with shape (10, 1) is broadcast across all m sample columns.

**Concrete example** (single neuron, single sample):

```
z_1 = w_{1,1}*x_1 + w_{1,2}*x_2 + ... + w_{1,784}*x_{784} + b_1
```

This is a dot product: `z = w^T * x + b`, computing a linear decision boundary in 784-dimensional space.

### 2.2 Hidden Layer — ReLU Activation

```
A[1] = ReLU(Z[1]) = max(0, Z[1])     (element-wise)
```

**What ReLU does:**
- Positive values pass through unchanged: `ReLU(2.5) = 2.5`
- Negative values become zero: `ReLU(-1.3) = 0`

**Why we need non-linearity:**

Without activation functions, the network computes:
```
Z[2] = W[2] . (W[1] . X + b[1]) + b[2]
     = (W[2] . W[1]) . X + (W[2] . b[1] + b[2])
     = W_combined . X + b_combined
```
This is just another linear function! No matter how many layers, the result is always linear. Non-linear activation functions let the network learn curved, complex decision boundaries.

**Why ReLU specifically?**
1. Simple: just clamp negatives to zero
2. Fast gradient: derivative is 1 (positive) or 0 (negative) — no expensive computation
3. No vanishing gradient for positive values (unlike sigmoid/tanh that saturate)
4. Induces sparsity: inactive neurons don't contribute, creating efficient representations

### 2.3 Output Layer — Linear Transform

```
Z[2] = W[2] . A[1] + b[2]
```

Dimensions: `(10, 10) . (10, m) + (10, 1) = (10, m)`

Each of the 10 output neurons computes a weighted sum of the 10 hidden activations. These raw scores (logits) represent unnormalized evidence for each class.

### 2.4 Output Layer — Softmax Activation

Softmax converts logits to a probability distribution:

```
A[2]_i = exp(Z[2]_i) / sum_j(exp(Z[2]_j))     for each column (sample)
```

**Properties:**
- Every output is in (0, 1)
- All outputs in a column sum to exactly 1
- Larger logits get exponentially larger probabilities (amplifies differences)

**Example:** If logits for one sample are `Z = [2.0, 1.0, 0.1]`:
```
exp(Z) = [7.39, 2.72, 1.11]
sum = 11.22
softmax = [0.659, 0.242, 0.099]     ← probabilities summing to 1
```

### Numerical Stability in Softmax

Raw softmax can overflow: `exp(1000) = infinity`. The fix: subtract the column maximum before exponentiating.

```
Z_stable = Z - max(Z)     (per column)
A_i = exp(Z_stable_i) / sum_j(exp(Z_stable_j))
```

**Why this works mathematically:**
```
exp(z_i - c) / sum(exp(z_j - c))
= exp(z_i) * exp(-c) / (sum(exp(z_j)) * exp(-c))
= exp(z_i) / sum(exp(z_j))
```

The constant `c` cancels. The largest exponent becomes `exp(0) = 1`, preventing overflow.

---

## 3. Loss Function: Categorical Cross-Entropy

### 3.1 One-Hot Encoding

Convert integer labels to vectors where the correct class has value 1, all others 0:

```
Label 3 → Y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Label 7 → Y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

This gives us the "target probability distribution" — all probability mass on the correct class.

### 3.2 Cross-Entropy Loss (Single Sample)

For true label vector `y` and predicted probabilities `a`:

```
L = -sum_k( y_k * log(a_k) )
```

Since `y` is one-hot (only one `y_k = 1`), this simplifies to:

```
L = -log(a_c)     where c is the correct class index
```

**Intuition through examples:**

| Predicted prob for correct class | Loss |
|----------------------------------|------|
| 0.99 | -log(0.99) = 0.01 (very low — confident and correct) |
| 0.50 | -log(0.50) = 0.69 (moderate — uncertain) |
| 0.01 | -log(0.01) = 4.61 (very high — confident and WRONG) |

Cross-entropy heavily penalizes confident wrong predictions, which is exactly what we want.

### 3.3 Average Loss (Cost Function)

Over all m training samples:

```
J = (1/m) * sum_{i=1}^{m} L_i = -(1/m) * sum_{i=1}^{m} log(A2[c_i, i])
```

where `c_i` is the correct class for sample `i`. This single number `J` measures overall model performance — gradient descent minimizes it.

---

## 4. Backward Propagation (Gradient Computation)

Backpropagation answers: "How much does each parameter contribute to the error?" It computes partial derivatives using the chain rule, working backwards from the loss to each parameter.

### 4.1 Output Layer Gradient: dZ[2]

We need `dJ/dZ[2]` — how the loss changes with respect to the output layer's pre-activation values.

**Starting point:** Loss for sample i is `L = -log(a_c)` where `a_c = softmax(z)_c`.

**The softmax derivative** has two cases:

For the j-th output with respect to the k-th logit:
```
da_j/dz_k = a_j * (1{j=k} - a_k)
```

Where `1{j=k}` is 1 when j=k, 0 otherwise.

**Combining with cross-entropy:**

```
dL/dz_k = -sum_j( y_j * (1/a_j) * da_j/dz_k )
        = -sum_j( y_j * (1/a_j) * a_j * (1{j=k} - a_k) )
        = -sum_j( y_j * (1{j=k} - a_k) )
        = -(y_k - a_k * sum_j(y_j))
        = -(y_k - a_k * 1)           (since y sums to 1 for one-hot)
        = a_k - y_k
```

This gives the beautifully simple result:

```
dZ[2] = A[2] - Y          shape: (10, m)
```

Each element tells us: "the predicted probability minus what it should be." If we predicted 0.8 for a class that should be 1.0, the gradient is -0.2 (push it up). If we predicted 0.3 for a class that should be 0.0, the gradient is +0.3 (push it down).

### 4.2 Weight and Bias Gradients for Layer 2

Since `Z[2] = W[2] . A[1] + b[2]`, by the chain rule:

**Weight gradient:**
```
dW[2] = (1/m) * dZ[2] . A[1]^T     shape: (10, 10)
```

Each element `dW[2]_{i,j}` tells us how much weight `W[2]_{i,j}` should change. It's the correlation between the output error (dZ[2]) and the hidden activations (A[1]) — if a hidden neuron is active when the output error is large, the weight connecting them gets a large gradient.

**Bias gradient:**
```
db[2] = (1/m) * sum(dZ[2], axis=1, keepdims=True)     shape: (10, 1)
```

Sum over all m samples (axis=1) because the bias affects every sample equally. We keep dimensions as (10, 1) to match `b[2]`'s shape.

### 4.3 Propagating Error to Hidden Layer: dZ[1]

To update layer 1's parameters, we need the error at the hidden layer:

**Step 1: Propagate through weights**
```
W[2]^T . dZ[2]     shape: (10, m)
```

This distributes the output error back to each hidden neuron, proportional to how strongly it's connected to erroneous outputs. A hidden neuron connected by large weights to outputs with large errors gets a large share of the blame.

**Step 2: Propagate through ReLU**
```
dZ[1] = (W[2]^T . dZ[2]) * ReLU'(Z[1])     shape: (10, m)
```

The ReLU derivative is:
```
ReLU'(z) = 1    if z > 0     (neuron was active → pass gradient through)
         = 0    if z <= 0    (neuron was inactive → block gradient)
```

This is the "gradient gate" — only neurons that fired during forward propagation receive gradient updates. Dead neurons (where Z <= 0) get zero gradient and don't change.

The `*` here is element-wise multiplication (Hadamard product), not matrix multiplication.

### 4.4 Weight and Bias Gradients for Layer 1

```
dW[1] = (1/m) * dZ[1] . X^T          shape: (10, 784)
db[1] = (1/m) * sum(dZ[1], axis=1, keepdims=True)    shape: (10, 1)
```

Same logic as layer 2: weight gradient is the correlation of layer error with layer input, bias gradient is the average error across samples.

### 4.5 Complete Gradient Summary

| Step | Formula | Shape | What It Means |
|------|---------|-------|---------------|
| 1 | dZ[2] = A[2] - Y | (10, m) | Output error: predicted minus true |
| 2 | dW[2] = (1/m) * dZ[2] . A[1]^T | (10, 10) | How much each output weight caused the error |
| 3 | db[2] = (1/m) * sum(dZ[2], axis=1, keepdims=True) | (10, 1) | Average output bias error |
| 4 | dZ[1] = W[2]^T . dZ[2] * ReLU'(Z[1]) | (10, m) | Hidden layer error (gated by ReLU) |
| 5 | dW[1] = (1/m) * dZ[1] . X^T | (10, 784) | How much each hidden weight caused the error |
| 6 | db[1] = (1/m) * sum(dZ[1], axis=1, keepdims=True) | (10, 1) | Average hidden bias error |

---

## 5. Gradient Descent: Parameter Updates

With all gradients computed, we update each parameter by stepping in the direction that reduces the loss:

```
W[1] := W[1] - alpha * dW[1]
b[1] := b[1] - alpha * db[1]
W[2] := W[2] - alpha * dW[2]
b[2] := b[2] - alpha * db[2]
```

### The Learning Rate (alpha)

The learning rate controls step size:

- **Too large** (e.g., alpha=10): Steps overshoot the minimum. Loss oscillates wildly or diverges to infinity. The network never converges.
- **Too small** (e.g., alpha=0.0001): Steps are tiny. Training is extremely slow. May get trapped in a poor local minimum.
- **Just right** (e.g., alpha=0.1 for this problem): Steady convergence toward a good solution.

### Why It Works: The Geometry

The loss function `J(W1, b1, W2, b2)` defines a surface in high-dimensional parameter space. The gradient `dJ/dW` points in the direction of steepest ascent. By subtracting `alpha * gradient`, we move in the direction of steepest *descent* — toward lower loss.

Each iteration:
1. Forward pass → compute predictions
2. Compute loss → measure how wrong we are
3. Backward pass → compute direction to improve
4. Update parameters → take a step in that direction

After many iterations, the loss converges and the network has learned to classify digits.

---

## 6. Putting It All Together

The complete training loop:

```
Initialize W1, b1, W2, b2 (random weights, zero biases)

For each iteration:
    # Forward propagation
    Z1 = W1 . X + b1
    A1 = ReLU(Z1)
    Z2 = W2 . A1 + b2
    A2 = softmax(Z2)

    # Backward propagation
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 . A1^T
    db2 = (1/m) * sum(dZ2, axis=1)
    dZ1 = W2^T . dZ2 * ReLU'(Z1)
    dW1 = (1/m) * dZ1 . X^T
    db1 = (1/m) * sum(dZ1, axis=1)

    # Update
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
```

With 784→10→10 architecture, learning rate 0.10, and 500 iterations on MNIST, this achieves ~85% accuracy — demonstrating that the math works.
