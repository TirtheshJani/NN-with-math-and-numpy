# 🧠 Neural Networks from Scratch with Math & NumPy

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> **Understanding Neural Networks at the Mathematical Level**  
> A deep dive into neural network fundamentals, implementing forward and backward propagation from scratch using only NumPy and mathematical equations.

---

## 📊 Project Overview

This project provides a **comprehensive mathematical and practical understanding** of neural networks by implementing them from scratch. No high-level frameworks—just **pure NumPy** and **mathematical equations**.

### Learning Objectives
- 🎯 Understand the math behind neural networks
- 📝 Implement forward propagation manually
- 🔄 Derive and code backpropagation
- 🧮 Master gradient descent optimization

---

## 🛠️ Tech Stack

- **Core:** Python 3.8+, NumPy
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook

---

## 📐 Mathematical Foundation

### Forward Propagation

#### Hidden Layer
```
Z[1] = W[1] · X + b[1]
A[1] = ReLU(Z[1]) = max(0, Z[1])
```

#### Output Layer
```
Z[2] = W[2] · A[1] + b[2]
A[2] = Softmax(Z[2])
```

### Backward Propagation (Gradients)

```
dZ[2] = A[2] - Y
dW[2] = (1/m) · dZ[2] · A[1]T
db[2] = (1/m) · Σ dZ[2]

dZ[1] = (W[2]T · dZ[2]) ⊙ ReLU'(Z[1])
dW[1] = (1/m) · dZ[1] · X.T
db[1] = (1/m) · Σ dZ[1]
```

### Parameter Updates
```
W := W - α · dW
b := b - α · db
```

---

## 🚀 Getting Started

```bash
pip install numpy matplotlib jupyter

# Clone repository
git clone https://github.com/TirtheshJani/NN-with-math-and-numpy.git
cd NN-with-math-and-numpy

jupyter notebook "NN from scratch wip.ipynb"
```

---

## 📊 Variable Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `X` | (784, m) | Input matrix (flattened 28x28 images) |
| `W[1]` | (10, 784) | Hidden layer weights |
| `b[1]` | (10, 1) | Hidden layer biases |
| `A[1]` | (10, m) | Hidden layer activated output |
| `W[2]` | (10, 10) | Output layer weights |
| `b[2]` | (10, 1) | Output layer biases |
| `A[2]` | (10, m) | Final predictions |
| `Y` | (10, m) | One-hot encoded labels |

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">
  <i>Master the fundamentals, build the future 🧠💻</i>
</p>
