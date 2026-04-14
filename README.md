# Neural Networks from Scratch with Math & NumPy

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> **Understanding Neural Networks at the Mathematical Level**
> A deep dive into neural network fundamentals, implementing forward and backward propagation from scratch using only NumPy and mathematical equations.

---

## Project Overview

This project provides a **comprehensive mathematical and practical understanding** of neural networks by implementing them from scratch. No high-level frameworks — just **pure NumPy** and **mathematical equations**.

### What You'll Learn
- The math behind neural networks — every equation, derived and explained
- Forward propagation: how inputs become predictions
- Backpropagation: how gradients flow backwards through the chain rule
- Gradient descent: how parameters update to minimize loss
- Numerical stability: why naive implementations break and how to fix them

---

## Tech Stack

- **Core:** Python 3.8+, NumPy
- **Visualization:** Matplotlib, Plotly
- **Interactive Demo:** Streamlit
- **Environment:** Jupyter Notebook

---

## Network Architecture

```
Input (784)  →  Hidden Layer (10, ReLU)  →  Output Layer (10, Softmax)  →  Prediction
```

A 2-layer neural network trained on MNIST handwritten digits (28x28 grayscale images, 10 classes).

---

## Mathematical Foundation

### 1. Initialization — He Init for ReLU

```
W ~ Normal(0, sqrt(2 / n_in))
b = 0
```

Random weights break symmetry (so neurons learn different features). The `sqrt(2/n_in)` scaling keeps signal variance stable across layers when using ReLU.

### 2. Forward Propagation

**Hidden Layer:**
```
Z[1] = W[1] . X + b[1]          Linear transform
A[1] = max(0, Z[1])             ReLU activation
```

**Output Layer:**
```
Z[2] = W[2] . A[1] + b[2]      Linear transform
A[2] = softmax(Z[2])            Probability distribution
```

Where softmax converts raw scores to probabilities:
```
softmax(z_i) = exp(z_i) / sum_j(exp(z_j))
```

### 3. Loss — Categorical Cross-Entropy

```
J = -(1/m) * sum_i( log(A2[c_i, i]) )
```

Measures the gap between predicted probabilities and true labels. Penalizes confident wrong predictions heavily.

### 4. Backward Propagation (Gradients)

```
dZ[2] = A[2] - Y                                          Output error
dW[2] = (1/m) * dZ[2] . A[1]^T                            Output weight gradient
db[2] = (1/m) * sum(dZ[2], axis=1, keepdims=True)         Output bias gradient

dZ[1] = (W[2]^T . dZ[2]) * ReLU'(Z[1])                   Hidden error (gated)
dW[1] = (1/m) * dZ[1] . X^T                               Hidden weight gradient
db[1] = (1/m) * sum(dZ[1], axis=1, keepdims=True)         Hidden bias gradient
```

Key insight: `dZ[2] = A[2] - Y` comes from the elegant cancellation of softmax and cross-entropy derivatives.

### 5. Parameter Updates (Gradient Descent)

```
W := W - alpha * dW
b := b - alpha * db
```

Each step moves parameters in the direction that reduces the loss.

---

## Variable Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `X` | (784, m) | Input matrix — each column is one flattened 28x28 image |
| `W[1]` | (10, 784) | Hidden layer weights |
| `b[1]` | (10, 1) | Hidden layer biases |
| `Z[1]` | (10, m) | Hidden layer pre-activation |
| `A[1]` | (10, m) | Hidden layer post-activation (ReLU output) |
| `W[2]` | (10, 10) | Output layer weights |
| `b[2]` | (10, 1) | Output layer biases |
| `Z[2]` | (10, m) | Output layer pre-activation (logits) |
| `A[2]` | (10, m) | Output probabilities (softmax output) |
| `Y` | (10, m) | One-hot encoded true labels |
| `alpha` | scalar | Learning rate |
| `m` | scalar | Number of training samples |

---

## Getting Started

### Jupyter Notebook (Core Implementation)

```bash
pip install numpy matplotlib pandas jupyter
git clone https://github.com/TirtheshJani/NN-with-math-and-numpy.git
cd NN-with-math-and-numpy
jupyter notebook "NN from scratch wip .ipynb"
```

### Interactive Streamlit Demo

```bash
pip install -r demo_requirements.txt
streamlit run demo_app.py
```

The demo lets you configure the network (hidden neurons, learning rate, activation function), train on synthetic datasets, and visualize decision boundaries in real time.

---

## Project Structure

```
├── NN from scratch wip .ipynb    # Core implementation with math explanations
├── test read.md                  # Detailed mathematical guide (full derivations)
├── demo_app.py                   # Interactive Streamlit visualization
├── demo_requirements.txt         # Dependencies for the demo
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## License

MIT License — see [LICENSE](LICENSE) file.
