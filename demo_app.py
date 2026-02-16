"""
Interactive Neural Network Visualization Demo
Based on: NN-with-math-and-numpy by Tirthesh Jani
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

st.set_page_config(page_title="NN from Scratch", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .formula-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        font-family: monospace;
        border-left: 4px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Neural Network from Scratch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Understanding neural networks at the mathematical level with NumPy</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Network Configuration")
n_hidden = st.sidebar.slider("Hidden Layer Neurons", 5, 50, 10)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1, 0.001)
n_epochs = st.sidebar.slider("Training Epochs", 10, 500, 100, 10)
activation_func = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
dataset_choice = st.sidebar.selectbox("Dataset", ["Moons", "Blobs", "XOR"])

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        # Xavier initialization
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.activation = activation
        self.history = {'loss': [], 'accuracy': []}
        
    def _activate(self, Z):
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
    
    def _activate_derivative(self, Z):
        if self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
            return A * (1 - A)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2
    
    def forward(self, X):
        """Forward propagation"""
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self._activate(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        # Softmax for output
        exp_scores = np.exp(self.Z2 - np.max(self.Z2, axis=0, keepdims=True))
        self.A2 = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        return self.A2
    
    def backward(self, X, Y, learning_rate):
        """Backward propagation"""
        m = X.shape[1]
        
        # Output layer gradient
        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden layer gradient
        dZ1 = np.dot(self.W2.T, dZ2) * self._activate_derivative(self.Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def compute_loss(self, Y):
        """Cross-entropy loss"""
        m = Y.shape[1]
        log_probs = -np.log(self.A2[Y.argmax(axis=0), range(m)] + 1e-8)
        return np.sum(log_probs) / m
    
    def compute_accuracy(self, X, Y):
        """Compute accuracy"""
        predictions = self.forward(X)
        pred_labels = np.argmax(predictions, axis=0)
        true_labels = np.argmax(Y, axis=0)
        return np.mean(pred_labels == true_labels)
    
    def train(self, X, Y, epochs, lr):
        """Train the network"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute metrics
            loss = self.compute_loss(Y)
            acc = self.compute_accuracy(X, Y)
            
            self.history['loss'].append(loss)
            self.history['accuracy'].append(acc)
            
            # Backward pass
            self.backward(X, Y, lr)
        
        return self.history

# Generate dataset
def generate_dataset(dataset_type, n_samples=500):
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset_type == "Blobs":
        X, y = make_classification(n_samples=n_samples, n_features=2, 
                                   n_classes=2, n_redundant=0, 
                                   n_clusters_per_class=1, random_state=42)
    else:  # XOR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * (n_samples // 4))
        y = np.array([0, 1, 1, 0] * (n_samples // 4))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Prepare data
X, y = generate_dataset(dataset_choice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode labels
def one_hot(y, num_classes):
    return np.eye(num_classes)[y].T

Y_train = one_hot(y_train, 2)
Y_test = one_hot(y_test, 2)

# Transpose for our network (features x samples)
X_train_t = X_train.T
X_test_t = X_test.T

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📚 Math", "🔬 Architecture", "▶️ Training"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Visualization")
        
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str),
                        labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'},
                        title=f"{dataset_choice} Dataset",
                        color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Info")
        st.write(f"**Samples:** {len(X)}")
        st.write(f"**Features:** 2")
        st.write(f"**Classes:** 2")
        st.write(f"**Train/Test Split:** 80/20")
        
        st.divider()
        
        st.subheader("Class Distribution")
        class_counts = pd.Series(y).value_counts().sort_index()
        fig_pie = px.pie(values=class_counts.values, names=[f"Class {i}" for i in class_counts.index],
                        color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("🧮 Mathematical Foundation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Forward Propagation**")
        st.markdown("""
        <div class="formula-box">
        <b>Hidden Layer:</b><br>
        Z<sup>[1]</sup> = W<sup>[1]</sup> · X + b<sup>[1]</sup><br>
        A<sup>[1]</sup> = activation(Z<sup>[1]</sup>)<br><br>
        
        <b>Output Layer:</b><br>
        Z<sup>[2]</sup> = W<sup>[2]</sup> · A<sup>[1]</sup> + b<sup>[2]</sup><br>
        A<sup>[2]</sup> = softmax(Z<sup>[2]</sup>)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Backward Propagation**")
        st.markdown("""
        <div class="formula-box">
        <b>Output Gradients:</b><br>
        dZ<sup>[2]</sup> = A<sup>[2]</sup> - Y<br>
        dW<sup>[2]</sup> = (1/m) · dZ<sup>[2]</sup> · A<sup>[1]T</sup><br><br>
        
        <b>Hidden Gradients:</b><br>
        dZ<sup>[1]</sup> = (W<sup>[2]T</sup> · dZ<sup>[2]</sup>) ⊙ activation'(Z<sup>[1]</sup>)<br>
        dW<sup>[1]</sup> = (1/m) · dZ<sup>[1]</sup> · X<sup>T</sup>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🔄 Parameter Update (Gradient Descent)")
    st.markdown("""
    <div class="formula-box" style="text-align: center; font-size: 1.2rem;">
    W := W - α · dW<br>
    b := b - α · db
    </div>
    <p style="text-align: center;">where α (alpha) is the learning rate</p>
    """, unsafe_allow_html=True)

with tab3:
    st.subheader("🏗️ Network Architecture")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Visual network representation
        fig = go.Figure()
        
        # Layer positions
        layer_x = [0, 1, 2]
        layer_sizes = [2, n_hidden, 2]
        layer_names = ['Input\n(2 features)', f'Hidden\n({n_hidden} neurons)', 'Output\n(2 classes)']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        max_neurons = max(layer_sizes)
        
        # Draw neurons
        for i, (x, size, name, color) in enumerate(zip(layer_x, layer_sizes, layer_names, colors)):
            y_positions = np.linspace(-(size-1)/2, (size-1)/2, size) if size > 1 else [0]
            
            for y in y_positions:
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=30, color=color, line=dict(width=2, color='white')),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Layer label
            fig.add_annotation(x=x, y=max_neurons/2 + 0.8, text=name, 
                             showarrow=False, font=dict(size=14))
        
        # Draw connections (sample for visibility)
        y_input = np.linspace(-0.5, 0.5, 2)
        y_hidden = np.linspace(-(n_hidden-1)/2, (n_hidden-1)/2, n_hidden) if n_hidden > 1 else [0]
        y_output = np.linspace(-0.5, 0.5, 2)
        
        # Connect input to hidden (sample)
        for yi in y_input:
            for yh in y_hidden[:min(5, len(y_hidden))]:
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[yi, yh],
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Connect hidden to output (sample)
        for yh in y_hidden[:min(5, len(y_hidden))]:
            for yo in y_output:
                fig.add_trace(go.Scatter(
                    x=[1, 2], y=[yh, yo],
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title="Neural Network Architecture",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter counts
        n_params = (2 * n_hidden + n_hidden) + (n_hidden * 2 + 2)
        st.info(f"📊 **Total Trainable Parameters:** {n_params:,}")
        st.write(f"- W1: {2 * n_hidden} weights + {n_hidden} biases")
        st.write(f"- W2: {n_hidden * 2} weights + 2 biases")

with tab4:
    st.subheader("▶️ Train the Neural Network")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize and train network
        nn = NeuralNetwork(input_size=2, hidden_size=n_hidden, 
                          output_size=2, activation=activation_func.lower())
        
        # Progress callback
        class ProgressCallback:
            def __init__(self, total):
                self.total = total
            def update(self, epoch):
                progress = (epoch + 1) / self.total
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Training... Epoch {epoch + 1}/{self.total}")
        
        callback = ProgressCallback(n_epochs)
        
        # Manual training loop with progress
        history = {'loss': [], 'accuracy': []}
        for epoch in range(n_epochs):
            output = nn.forward(X_train_t)
            loss = nn.compute_loss(Y_train)
            acc = nn.compute_accuracy(X_train_t, Y_train)
            nn.backward(X_train_t, Y_train, learning_rate)
            history['loss'].append(loss)
            history['accuracy'].append(acc)
            callback.update(epoch)
        
        status_text.text("✅ Training Complete!")
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Loss", f"{history['loss'][-1]:.4f}")
        with col2:
            st.metric("Train Accuracy", f"{history['accuracy'][-1]*100:.1f}%")
        with col3:
            test_acc = nn.compute_accuracy(X_test_t, Y_test)
            st.metric("Test Accuracy", f"{test_acc*100:.1f}%")
        
        # Training curves
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history['loss'], mode='lines', 
                                         name='Loss', line=dict(color='#e74c3c')))
            fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch",
                                  yaxis_title="Loss", height=300)
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=[a*100 for a in history['accuracy']], 
                                        mode='lines', name='Accuracy',
                                        line=dict(color='#2ecc71')))
            fig_acc.update_layout(title="Training Accuracy", xaxis_title="Epoch",
                                 yaxis_title="Accuracy (%)", height=300)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Decision boundary visualization
        st.subheader("🎯 Decision Boundary")
        
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_input = np.c_[xx.ravel(), yy.ravel()].T
        Z = nn.forward(mesh_input)
        Z = np.argmax(Z, axis=0).reshape(xx.shape)
        
        fig_boundary = go.Figure()
        
        # Decision boundary
        fig_boundary.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale=[[0, 'rgba(52, 152, 219, 0.3)'], 
                       [1, 'rgba(231, 76, 60, 0.3)']],
            showscale=False,
            contours=dict(start=0, end=1, size=0.5),
            name='Decision Boundary'
        ))
        
        # Data points
        for i, (label, color) in enumerate(zip([0, 1], ['#3498db', '#e74c3c'])):
            mask = y == i
            fig_boundary.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Class {i}',
                marker=dict(size=10, color=color, line=dict(width=2, color='white'))
            ))
        
        fig_boundary.update_layout(
            title="Learned Decision Boundary",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=500
        )
        
        st.plotly_chart(fig_boundary, use_container_width=True)
        
        # Weight visualization
        st.subheader("🔍 Weight Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_w1 = px.histogram(nn.W1.flatten(), nbins=30,
                                 title="W1 (Input → Hidden) Distribution",
                                 color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_w1, use_container_width=True)
        
        with col2:
            fig_w2 = px.histogram(nn.W2.flatten(), nbins=30,
                                 title="W2 (Hidden → Output) Distribution",
                                 color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig_w2, use_container_width=True)

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎓 Based on <a href="https://github.com/TirtheshJani/NN-with-math-and-numpy">NN-with-math-and-numpy</a> by Tirthesh Jani</p>
    <p>Built with Streamlit ❤️</p>
</div>
""", unsafe_allow_html=True)
