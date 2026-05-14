# Neural Network from Scratch — Complete Mathematical Guide

A step-by-step mathematical walkthrough of every operation in our 2-layer neural network, from initialization to a trained model. All equations are written in LaTeX so they render natively on GitHub.

---

## Network Architecture

$$
\underbrace{\mathbf{X}}_{\text{Input } (784)} \;\xrightarrow{\;W^{[1]},\,b^{[1]}\;}\; Z^{[1]} \;\xrightarrow{\;\text{ReLU}\;}\; A^{[1]} \;\xrightarrow{\;W^{[2]},\,b^{[2]}\;}\; Z^{[2]} \;\xrightarrow{\;\text{softmax}\;}\; \underbrace{A^{[2]}}_{\text{Output } (10)}
$$

We classify $28\times 28$ grayscale images of handwritten digits (MNIST) into 10 classes (digits 0–9).

### Notation

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of training samples |
| $n_x = 784$ | Number of input features ($28\times 28$) |
| $\mathbf{X}\in\mathbb{R}^{784\times m}$ | Input matrix — each column is one sample |
| $\mathbf{Y}\in\{0,1\}^{10\times m}$ | One-hot encoded labels |
| $W^{[\ell]}$ | Weight matrix of layer $\ell$ |
| $b^{[\ell]}$ | Bias vector of layer $\ell$ |
| $Z^{[\ell]}$ | Pre-activation at layer $\ell$ |
| $A^{[\ell]}$ | Post-activation at layer $\ell$ |
| $\alpha$ | Learning rate |
| $J$ | Cost (average loss) |

---

## 1. Parameter Initialization

### Symmetry Breaking

If every weight starts at the same value, every neuron in a layer computes the same output. Backpropagation then delivers identical gradients to all of them, so they update identically and the network can never learn distinct features. Random initialization breaks this symmetry.

### He Initialization (for ReLU networks)

For a layer with $n_{\text{in}}$ input connections:

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}}}\right), \qquad b = \mathbf{0}
$$

#### Why this specific scaling?

A neuron computes $z = \sum_{j=1}^{n_{\text{in}}} w_j x_j$. Assuming inputs and weights are zero-mean and independent,

$$
\operatorname{Var}(z) \;=\; n_{\text{in}}\,\operatorname{Var}(w)\,\operatorname{Var}(x).
$$

To preserve activation variance across layers ($\operatorname{Var}(z)=\operatorname{Var}(x)$), we need $\operatorname{Var}(w) = 1/n_{\text{in}}$. Since ReLU zeros out roughly half of the pre-activations (negative half), the variance is again halved on the way through the non-linearity. He init compensates by doubling:

$$
\operatorname{Var}(w) \;=\; \frac{2}{n_{\text{in}}} \quad\Longrightarrow\quad \sigma(w) \;=\; \sqrt{\frac{2}{n_{\text{in}}}}.
$$

### In Our Network

$$
\begin{aligned}
W^{[1]} &\sim \mathcal{N}\!\left(0, \tfrac{2}{784}\right), & W^{[1]} &\in \mathbb{R}^{10\times 784} && (7{,}840 \text{ weights})\\
b^{[1]} &= \mathbf{0}, & b^{[1]} &\in \mathbb{R}^{10\times 1} && (10 \text{ biases})\\
W^{[2]} &\sim \mathcal{N}\!\left(0, \tfrac{2}{10}\right), & W^{[2]} &\in \mathbb{R}^{10\times 10} && (100 \text{ weights})\\
b^{[2]} &= \mathbf{0}, & b^{[2]} &\in \mathbb{R}^{10\times 1} && (10 \text{ biases})
\end{aligned}
$$

**Total trainable parameters:** $7{,}840 + 10 + 100 + 10 = 7{,}960$.

---

## 2. Forward Propagation

### 2.1 Hidden Layer — Linear Transform

$$
Z^{[1]} \;=\; W^{[1]}\mathbf{X} + b^{[1]}
$$

Dimensions: $(10,784)\cdot(784,m) + (10,1) \;\to\; (10, m)$. The bias broadcasts across all $m$ sample columns.

### 2.2 Hidden Layer — ReLU Activation

$$
A^{[1]} \;=\; \operatorname{ReLU}\!\bigl(Z^{[1]}\bigr) \;=\; \max\!\bigl(0,\, Z^{[1]}\bigr) \quad \text{(element-wise)}
$$

#### Why non-linearity?

Without an activation function the composition

$$
Z^{[2]} = W^{[2]}\bigl(W^{[1]}\mathbf{X}+b^{[1]}\bigr)+b^{[2]} = \underbrace{W^{[2]}W^{[1]}}_{W_{\text{eff}}}\mathbf{X}+\underbrace{W^{[2]}b^{[1]}+b^{[2]}}_{b_{\text{eff}}}
$$

collapses to a single linear map. Non-linearities are what let the network bend decision boundaries.

#### Why ReLU?

1. Cheap: a single comparison.
2. Sparse: roughly half the neurons stay inactive, which acts as implicit regularization.
3. No saturation for positive inputs — gradients of magnitude 1 keep flowing.

### 2.3 Output Layer — Linear Transform

$$
Z^{[2]} \;=\; W^{[2]} A^{[1]} + b^{[2]}, \qquad Z^{[2]} \in \mathbb{R}^{10\times m}.
$$

### 2.4 Output Layer — Softmax

For each column (sample) of $Z^{[2]}$:

$$
A^{[2]}_{i} \;=\; \frac{e^{Z^{[2]}_{i}}}{\sum_{j=1}^{10} e^{Z^{[2]}_{j}}}
$$

Each column of $A^{[2]}$ is a probability distribution over the 10 classes.

#### Numerical Stability

Raw $e^{z_i}$ can overflow for large $z_i$. Subtract the column maximum $c=\max_j z_j$ before exponentiating:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
\;=\; \frac{e^{z_i}\,e^{-c}}{e^{-c}\sum_j e^{z_j}}
\;=\; \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

The constant $c$ cancels, and the largest exponent becomes $e^{0}=1$, eliminating overflow.

---

## 3. Loss Function — Categorical Cross-Entropy

### 3.1 One-Hot Encoding

For a true class $c$, the label vector $\mathbf{y}\in\{0,1\}^{10}$ has $y_c=1$ and $y_k=0$ for $k\ne c$.

### 3.2 Per-Sample Loss

$$
\mathcal{L} \;=\; -\sum_{k=1}^{10} y_k \log a_k \;=\; -\log a_c
$$

where the simplification uses $\mathbf{y}$ being one-hot.

| Predicted prob. for the correct class | $-\log a_c$ |
|---|---|
| $0.99$ | $0.010$ — confident and correct |
| $0.50$ | $0.693$ — uncertain |
| $0.01$ | $4.605$ — confident and wrong |

### 3.3 Cost Over the Mini-Batch

$$
J \;=\; \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}^{(i)} \;=\; -\frac{1}{m}\sum_{i=1}^{m}\log A^{[2]}_{\,c_i,\,i}.
$$

---

## 4. Backward Propagation

### 4.1 Output Layer — $dZ^{[2]}$

Using $a_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ the softmax Jacobian is

$$
\frac{\partial a_j}{\partial z_k} \;=\; a_j\bigl(\mathbb{1}_{\{j=k\}} - a_k\bigr).
$$

Combining with $\mathcal{L} = -\sum_j y_j \log a_j$:

$$
\frac{\partial \mathcal{L}}{\partial z_k}
= -\sum_j \frac{y_j}{a_j}\,\frac{\partial a_j}{\partial z_k}
= -\sum_j y_j\bigl(\mathbb{1}_{\{j=k\}} - a_k\bigr)
= a_k\sum_j y_j - y_k
= a_k - y_k,
$$

using $\sum_j y_j = 1$. Vectorized across the batch:

$$
\boxed{\; dZ^{[2]} \;=\; A^{[2]} - \mathbf{Y} \;}
$$

Softmax and cross-entropy cancel into a single elegant residual.

### 4.2 Parameter Gradients — Layer 2

By the chain rule applied to $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$:

$$
dW^{[2]} \;=\; \frac{1}{m}\, dZ^{[2]} \bigl(A^{[1]}\bigr)^{\!\top}, \qquad
db^{[2]} \;=\; \frac{1}{m}\sum_{i=1}^{m} dZ^{[2]}_{:,i}.
$$

### 4.3 Hidden Layer — $dZ^{[1]}$

Propagate through the second weight matrix, then gate by the ReLU derivative:

$$
dZ^{[1]} \;=\; \bigl(W^{[2]}\bigr)^{\!\top} dZ^{[2]} \;\odot\; \operatorname{ReLU}'\!\bigl(Z^{[1]}\bigr),
$$

where $\odot$ denotes element-wise (Hadamard) product and

$$
\operatorname{ReLU}'(z) \;=\; \begin{cases} 1, & z > 0 \\ 0, & z \le 0.\end{cases}
$$

Neurons that were inactive on the forward pass contribute zero gradient.

### 4.4 Parameter Gradients — Layer 1

$$
dW^{[1]} \;=\; \frac{1}{m}\, dZ^{[1]} \mathbf{X}^{\top}, \qquad
db^{[1]} \;=\; \frac{1}{m}\sum_{i=1}^{m} dZ^{[1]}_{:,i}.
$$

### 4.5 Gradient Summary

| # | Quantity | Formula | Shape |
|---|----------|---------|-------|
| 1 | $dZ^{[2]}$ | $A^{[2]} - \mathbf{Y}$ | $(10, m)$ |
| 2 | $dW^{[2]}$ | $\tfrac{1}{m}\, dZ^{[2]}\,{A^{[1]}}^{\top}$ | $(10, 10)$ |
| 3 | $db^{[2]}$ | $\tfrac{1}{m}\sum_i dZ^{[2]}_{:,i}$ | $(10, 1)$ |
| 4 | $dZ^{[1]}$ | $\bigl({W^{[2]}}^{\top} dZ^{[2]}\bigr)\odot \operatorname{ReLU}'(Z^{[1]})$ | $(10, m)$ |
| 5 | $dW^{[1]}$ | $\tfrac{1}{m}\, dZ^{[1]}\mathbf{X}^{\top}$ | $(10, 784)$ |
| 6 | $db^{[1]}$ | $\tfrac{1}{m}\sum_i dZ^{[1]}_{:,i}$ | $(10, 1)$ |

---

## 5. Gradient Descent — Parameter Updates

$$
\begin{aligned}
W^{[1]} &\leftarrow W^{[1]} - \alpha\, dW^{[1]} & b^{[1]} &\leftarrow b^{[1]} - \alpha\, db^{[1]} \\
W^{[2]} &\leftarrow W^{[2]} - \alpha\, dW^{[2]} & b^{[2]} &\leftarrow b^{[2]} - \alpha\, db^{[2]}
\end{aligned}
$$

### The Learning Rate

- $\alpha$ **too large** — steps overshoot the minimum, the loss oscillates or diverges.
- $\alpha$ **too small** — convergence is glacial; the model may stall in a poor basin.
- $\alpha \approx 0.1$ works well for this MNIST setup.

### Geometric Intuition

The cost $J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]})$ defines a surface in parameter space. The gradient $\nabla J$ points uphill; subtracting $\alpha\nabla J$ slides us toward lower loss.

---

## 6. Putting It All Together

```
Initialize W1, b1, W2, b2  (He init for W, zero for b)

For iteration = 1 .. T:
    # Forward
    Z1 = W1 @ X  + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    # Backward
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * dZ2.sum(axis=1, keepdims=True)
    dZ1 = (W2.T @ dZ2) * (Z1 > 0)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * dZ1.sum(axis=1, keepdims=True)

    # Update
    W1 -= alpha * dW1; b1 -= alpha * db1
    W2 -= alpha * dW2; b2 -= alpha * db2
```

With the $784 \to 10 \to 10$ architecture, $\alpha = 0.10$, and $T = 500$ iterations on MNIST, this converges to **~85% cross-validation accuracy** — see [README.md](README.md) for the verified training curve.
