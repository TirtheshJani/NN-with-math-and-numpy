# NN-with-math-and-numpy

## Forward Propagation:

### Linear Transformation for the Hidden Layer:

$Z^{[1]} = W^{[1]} X + b^{[1]}$
$W^{[1]}$: Weight matrix of shape (10, 784)
$X$: Input matrix of shape (784, m) where $m$ is the number of examples
$b^{[1]}$: Bias vector of shape (10, 1)
$Z^{[1]}$: Matrix of shape (10, m) containing the linear transformations for the hidden layer
### Activation of the Hidden Layer:

$A^{[1]} = g_{\text{ReLU}}(Z^{[1]})$
$g_{\text{ReLU}}$: ReLU activation function applied element-wise to $Z^{[1]}$
$A^{[1]}$: Matrix of shape (10, m) containing the activations of the hidden layer
### Linear Transformation for the Output Layer:

$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
$W^{[2]}$: Weight matrix of shape (10, 10)
$A^{[1]}$: Matrix of shape (10, m)
$b^{[2]}$: Bias vector of shape (10, 1)
$Z^{[2]}$: Matrix of shape (10, m) containing the linear transformations for the output layer
### Activation of the Output Layer:

$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$
$g_{\text{softmax}}$: Softmax activation function applied element-wise to $Z^{[2]}$
$A^{[2]}$: Matrix of shape (10, m) containing the probabilities for each class
## Backward Propagation:

### Gradient Calculation for the Output Layer:

$dZ^{[2]} = A^{[2]} - Y$
$Y$: One-hot encoded labels of shape (10, m)
$dZ^{[2]}$: Matrix of shape (10, m) containing the gradients of the output layer activations with respect to $Z^{[2]}$
### Gradient Calculation for the Output Layer Parameters:

$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$
$dW^{[2]}$: Gradient of the weight matrix $W^{[2]}$ of shape (10, 10)
$A^{[1]T}$: Transpose of $A^{[1]}$ of shape (m, 10)
$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$
$db^{[2]}$: Gradient of the bias vector $b^{[2]}$ of shape (10, 1)
### Gradient Calculation for the Hidden Layer:

$dZ^{[1]} = W^{[2]T} dZ^{[2]} * g_{\text{ReLU}}'(Z^{[1]})$
$g_{\text{ReLU}}'$: Derivative of the ReLU activation function applied element-wise to $Z^{[1]}$
$dZ^{[1]}$: Matrix of shape (10, m) containing the gradients of the hidden layer activations with respect to $Z^{[1]}$
### Gradient Calculation for the Hidden Layer Parameters:

$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$
$dW^{[1]}$: Gradient of the weight matrix $W^{[1]}$ of shape (10, 784)
$A^{[0]T}$: Transpose of the input matrix $A^{[0]} = X$ of shape (m, 784)
$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$
$db^{[1]}$: Gradient of the bias vector $b^{[1]}$ of shape (10, 1)
### Parameter Updates:

$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$: Update the weight matrix $W^{[2]}$ for the output layer
$b^{[2]} := b^{[2]} - \alpha db^{[2]}$: Update the bias vector $b^{[2]}$ for the output layer
$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$: Update the weight matrix $W^{[1]}$ for the hidden layer
$b^{[1]} := b^{[1]} - \alpha db^{[1]}$: Update the bias vector $b^{[1]}$ for the hidden layer
### Variable and Shape Summary:

$A^{[0]} = X$: Input matrix of shape (784, m)

$Z^{[1]} \sim A^{[1]}$: Hidden layer matrix of shape (10, m)

$W^{[1]}$: Weight matrix of shape (10, 784)

$b^{[1]}$: Bias vector of shape (10, 1)

$Z^{[2]} \sim A^{[2]}$: Output layer matrix of shape (10, m)

$W^{[2]}$: Weight matrix of shape (10, 10)

$b^{[2]}$: Bias vector of shape (10, 1)

$dZ^{[2]}$: Gradients of the output layer activations of shape (10, m)

$dW^{[2]}$: Gradients of the weight matrix $W^{[2]}$ of shape (10, 10)

$db^{[2]}$: Gradients of the bias vector $b^{[2]}$ of shape (10, 1)

$dZ^{[1]}$: Gradients of the hidden layer activations of shape (10, m)

$dW^{[1]}$: Gradients of the weight matrix $W^{[1]}$ of shape (10, 784)

$db^{[1]}$: Gradients of the bias vector $b^{[1]}$ of shape (10, 1)
