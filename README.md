# Neural Networks from Scratch — Math & NumPy

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **A 2-layer neural network derived from first principles and implemented with nothing but NumPy.**
> Every line of code traces back to an equation, every equation is derived in this README.

---

## Table of Contents

1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
3. [Mathematical Foundation](#mathematical-foundation)
   - [Initialization](#1-initialization--he-init)
   - [Forward Propagation](#2-forward-propagation)
   - [Loss](#3-loss--categorical-cross-entropy)
   - [Backward Propagation](#4-backward-propagation)
   - [Gradient Descent](#5-gradient-descent)
4. [Variable Reference](#variable-reference)
5. [Verified Results](#verified-results)
6. [Getting Started](#getting-started)
7. [Interactive Streamlit Demo](#interactive-streamlit-demo)
8. [Project Structure](#project-structure)
9. [License](#license)

---

## Overview

This project provides a **comprehensive mathematical and practical understanding** of feed-forward neural networks. There are no high-level frameworks — no PyTorch, no TensorFlow — just **pure NumPy** wired directly to the math.

What you'll learn:

- The math behind neural networks — every equation, derived and explained
- Forward propagation — how inputs become predictions
- Backpropagation — how gradients flow backward via the chain rule
- Gradient descent — how parameters update to minimize loss
- Numerical stability — why naive softmax breaks and how to fix it

For the full step-by-step derivation, see [`MATH_GUIDE.md`](MATH_GUIDE.md).

---

## Network Architecture

A 2-layer fully-connected network trained on **MNIST** (28×28 grayscale digit images, 10 classes):

$$
\underbrace{\mathbf{X}}_{784 \text{ pixels}}
\;\longrightarrow\;
\underbrace{\bigl[\,W^{[1]}, b^{[1]}\,\bigr] \;\to\; \mathrm{ReLU}}_{\text{hidden layer (10 units)}}
\;\longrightarrow\;
\underbrace{\bigl[\,W^{[2]}, b^{[2]}\,\bigr] \;\to\; \mathrm{softmax}}_{\text{output layer (10 classes)}}
\;\longrightarrow\;
\hat{y} \in \{0,\dots,9\}
$$

**Total trainable parameters:** $784 \cdot 10 + 10 + 10 \cdot 10 + 10 = 7{,}960$.

---

## Mathematical Foundation

### 1. Initialization — He Init

For a ReLU network, weights are sampled from a Gaussian whose variance is tuned to preserve signal variance across layers:

$$
W^{[\ell]}_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n^{[\ell-1]}}\right), \qquad b^{[\ell]} = \mathbf{0}
$$

The $\sqrt{2/n_{\text{in}}}$ scaling compensates for ReLU zeroing out roughly half of the pre-activations. Random weights (rather than zero) also break the symmetry that would otherwise force every neuron in a layer to learn the same feature.

### 2. Forward Propagation

**Hidden layer:**

$$
Z^{[1]} \;=\; W^{[1]}\mathbf{X} + b^{[1]}, \qquad
A^{[1]} \;=\; \operatorname{ReLU}\!\bigl(Z^{[1]}\bigr) \;=\; \max\!\bigl(0,\, Z^{[1]}\bigr).
$$

**Output layer:**

$$
Z^{[2]} \;=\; W^{[2]} A^{[1]} + b^{[2]}, \qquad
A^{[2]}_{\,i} \;=\; \operatorname{softmax}\!\bigl(Z^{[2]}\bigr)_i \;=\; \frac{e^{Z^{[2]}_{i}}}{\displaystyle\sum_{j=1}^{10} e^{Z^{[2]}_{j}}}.
$$

**Numerically stable softmax** subtracts the column maximum $c = \max_j Z^{[2]}_j$ before exponentiating; the constant cancels in numerator and denominator and the largest exponent becomes $e^0 = 1$, preventing overflow:

$$
\operatorname{softmax}(Z)_i \;=\; \frac{e^{Z_i - c}}{\sum_j e^{Z_j - c}}.
$$

### 3. Loss — Categorical Cross-Entropy

For a one-hot target $\mathbf{y}$ with correct class $c$:

$$
\mathcal{L} \;=\; -\sum_{k=1}^{10} y_k \log a_k \;=\; -\log a_c.
$$

Averaged over the batch of $m$ samples:

$$
J \;=\; \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}^{(i)} \;=\; -\frac{1}{m}\sum_{i=1}^{m}\log A^{[2]}_{\,c_i,\,i}.
$$

Confident wrong predictions are penalized harshly ($-\log 0.01 \approx 4.6$) while confident correct predictions cost almost nothing ($-\log 0.99 \approx 0.01$).

### 4. Backward Propagation

The softmax–cross-entropy combination has a famously clean derivative:

$$
\boxed{\,dZ^{[2]} \;=\; A^{[2]} - \mathbf{Y}\,}
$$

From there the chain rule gives every other gradient:

$$
\begin{aligned}
dW^{[2]} &= \tfrac{1}{m}\, dZ^{[2]} \bigl(A^{[1]}\bigr)^{\!\top}, &
db^{[2]} &= \tfrac{1}{m}\sum_{i=1}^{m} dZ^{[2]}_{:,i}, \\[4pt]
dZ^{[1]} &= \bigl(W^{[2]}\bigr)^{\!\top} dZ^{[2]} \;\odot\; \operatorname{ReLU}'\!\bigl(Z^{[1]}\bigr), &
\operatorname{ReLU}'(z) &= \begin{cases} 1, & z>0 \\ 0, & z\le 0,\end{cases} \\[4pt]
dW^{[1]} &= \tfrac{1}{m}\, dZ^{[1]} \mathbf{X}^{\!\top}, &
db^{[1]} &= \tfrac{1}{m}\sum_{i=1}^{m} dZ^{[1]}_{:,i}.
\end{aligned}
$$

The Hadamard product with $\operatorname{ReLU}'(Z^{[1]})$ acts as a **gradient gate**: neurons that were inactive in the forward pass receive zero gradient.

### 5. Gradient Descent

$$
\begin{aligned}
W^{[\ell]} &\leftarrow W^{[\ell]} - \alpha\, dW^{[\ell]} \\
b^{[\ell]} &\leftarrow b^{[\ell]} - \alpha\, db^{[\ell]}
\end{aligned}
\qquad \text{for } \ell \in \{1, 2\}.
$$

Each step moves the parameters along $-\nabla J$, the direction of steepest descent on the loss surface.

---

## Variable Reference

| Symbol | Shape | Description |
|--------|-------|-------------|
| $\mathbf{X}$ | $(784, m)$ | Input matrix — each column is one flattened $28\times 28$ image |
| $W^{[1]}$ | $(10, 784)$ | Hidden layer weights |
| $b^{[1]}$ | $(10, 1)$ | Hidden layer biases |
| $Z^{[1]}$ | $(10, m)$ | Hidden pre-activation |
| $A^{[1]}$ | $(10, m)$ | Hidden post-activation (ReLU) |
| $W^{[2]}$ | $(10, 10)$ | Output layer weights |
| $b^{[2]}$ | $(10, 1)$ | Output layer biases |
| $Z^{[2]}$ | $(10, m)$ | Output pre-activation (logits) |
| $A^{[2]}$ | $(10, m)$ | Output probabilities (softmax) |
| $\mathbf{Y}$ | $(10, m)$ | One-hot encoded true labels |
| $\alpha$ | scalar | Learning rate |
| $m$ | scalar | Number of training samples |

---

## Verified Results

Trained on **MNIST** (`train.csv` from Kaggle): 42,000 samples split into 41,000 training and 1,000 cross-validation. Configuration: hidden size = 10, $\alpha = 0.10$, full-batch gradient descent, 500 iterations.

### Training Curve

| Iteration | Training accuracy |
|----------:|:-----------------:|
|   0       | 0.094 |
|  50       | 0.296 |
| 100       | 0.555 |
| 150       | 0.660 |
| 200       | 0.714 |
| 250       | 0.752 |
| 300       | 0.778 |
| 350       | 0.797 |
| 400       | 0.813 |
| 450       | 0.825 |
| **500**   | **~0.85** |

### Final Performance

| Metric | Value |
|---|---|
| Training accuracy | **~85%** |
| Cross-validation accuracy | **~85%** |
| Trainable parameters | $7{,}960$ |
| Wall-clock training time | ~30 s on a laptop CPU |

The training and validation accuracies stay close — with only 7,960 parameters and 500 full-batch steps, the model is far from overfitting. Hand-derived equations and a few dozen lines of NumPy reach a competitive result for a from-scratch implementation.

### How to Improve Further

- Wider hidden layer (e.g., 128 or 256 units) — pushes accuracy past 95%
- Add depth (more hidden layers)
- Mini-batch SGD with shuffling
- Learning-rate schedule (cosine or step decay)
- L2 weight decay or dropout regularization

---

## Getting Started

### Prerequisites

- Python 3.8 or newer
- The MNIST training CSV (`train.csv` from the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition) placed in the project root

### Install dependencies and run the notebook

```bash
git clone https://github.com/TirtheshJani/NN-with-math-and-numpy.git
cd NN-with-math-and-numpy
pip install -r requirements.txt
jupyter notebook nn_from_scratch.ipynb
```

The notebook is organized into the same five steps as this README — read the markdown cells alongside the code to see each equation come to life.

---

## Interactive Streamlit Demo

A self-contained interactive visualizer (no MNIST required) lets you train the same network on 2D toy datasets and watch the decision boundary form in real time.

```bash
pip install -r demo_requirements.txt
streamlit run demo_app.py
```

Features:

- Configurable hidden size, learning rate, epochs, and activation function (ReLU / sigmoid / tanh)
- Choice of toy datasets: Moons, Blobs, XOR
- Live training loss and accuracy curves
- Decision-boundary contour plot
- Weight-distribution histograms

---

## Project Structure

```
.
├── nn_from_scratch.ipynb     # Core implementation, fully annotated
├── MATH_GUIDE.md             # Detailed mathematical derivations (LaTeX)
├── demo_app.py               # Interactive Streamlit visualization
├── demo_requirements.txt     # Dependencies for the Streamlit demo
├── requirements.txt          # Dependencies for the notebook
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute.
