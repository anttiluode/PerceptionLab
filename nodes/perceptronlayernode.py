"""
Perceptron Layer Node - A trainable layer that can learn XOR
Uses gradient descent, not just connection pruning
"""

import numpy as np
from PyQt6 import QtGui
import __main__

BaseNode = __main__.BaseNode

class PerceptronLayerNode(BaseNode):
    """
    A simple trainable perceptron layer.
    This can ACTUALLY learn XOR with proper training.
    """
    NODE_CATEGORY = "Learning"
    NODE_COLOR = QtGui.QColor(180, 60, 180)  # Purple - Learning
    
    def __init__(self, hidden_units=3, learning_rate=0.1):
        super().__init__()
        self.node_title = "Perceptron Layer"
        
        self.inputs = {
            'input_a': 'signal',
            'input_b': 'signal',
            'target': 'signal',      # For supervised learning
            'train_signal': 'signal'  # When >0.5, update weights
        }
        
        self.outputs = {
            'prediction': 'signal',
            'error': 'signal'
        }
        
        self.hidden_units = int(hidden_units)
        self.learning_rate = float(learning_rate)
        
        # Initialize weights randomly (small values)
        # Hidden layer: 2 inputs → hidden_units neurons
        self.W1 = np.random.randn(2, self.hidden_units) * 0.5
        self.b1 = np.zeros(self.hidden_units)
        
        # Output layer: hidden_units → 1 output
        self.W2 = np.random.randn(self.hidden_units, 1) * 0.5
        self.b2 = np.zeros(1)
        
        self.prediction = 0.0
        self.error = 0.0
        
        # For visualization
        self.weight_img = np.zeros((64, 64, 3), dtype=np.uint8)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, a, b):
        """Forward pass through the network"""
        # Input layer
        X = np.array([[a, b]])
        
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        h1 = np.tanh(z1)  # Tanh activation for hidden
        
        # Output layer
        z2 = h1 @ self.W2 + self.b2
        output = self.sigmoid(z2[0, 0])  # Sigmoid for output (0-1 range)
        
        return output, h1, X

    def backward(self, a, b, target):
        """Backward pass - gradient descent"""
        # Forward pass first
        output, h1, X = self.forward(a, b)
        
        # Calculate error
        error = target - output
        
        # Output layer gradients
        delta_out = error * self.sigmoid_derivative(output)
        
        # Hidden layer gradients
        delta_hidden = (delta_out * self.W2.T) * (1 - h1**2)  # tanh derivative
        
        # Update weights (gradient ascent, since we want to minimize error)
        self.W2 += self.learning_rate * h1.T * delta_out
        self.b2 += self.learning_rate * delta_out
        
        self.W1 += self.learning_rate * X.T @ delta_hidden
        self.b1 += self.learning_rate * delta_hidden[0]
        
        return error

    def step(self):
        # Get inputs
        a = self.get_blended_input('input_a', 'sum') or 0.0
        b = self.get_blended_input('input_b', 'sum') or 0.0
        target = self.get_blended_input('target', 'sum') or 0.0
        train_sig = self.get_blended_input('train_signal', 'sum') or 1.0  # Default: always train
        
        # Forward pass
        self.prediction, _, _ = self.forward(a, b)
        
        # Backward pass if training enabled
        if train_sig > 0.5:
            self.error = self.backward(a, b, target)
        else:
            self.error = target - self.prediction
        
        # Update visualization
        self._update_visualization()

    def _update_visualization(self):
        """Visualize the learned weights as a heatmap"""
        # Combine W1 and W2 into a single visualization
        # Show the strength of connections
        w_combined = np.abs(self.W1).mean(axis=1)  # Average hidden weights
        w_out = np.abs(self.W2).mean()
        
        self.weight_img.fill(20)
        
        # Simple bar chart of weight magnitudes
        h, w, _ = self.weight_img.shape
        for i, weight in enumerate(w_combined):
            bar_height = int(min(weight * 20, h - 5))
            x_pos = 10 + i * 20
            self.weight_img[h - bar_height:, x_pos:x_pos+15] = (0, int(weight*100), 0)

    def get_output(self, port_name):
        if port_name == 'prediction':
            return self.prediction
        elif port_name == 'error':
            return abs(self.error)
        return None
        
    def get_display_image(self):
        return QtGui.QImage(
            self.weight_img.data, 64, 64, 64*3, 
            QtGui.QImage.Format.Format_RGB888
        )
    
    def get_config_options(self):
        return [
            ("Hidden Units", "hidden_units", self.hidden_units, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None)
        ]
    
    def randomize(self):
        """Reset weights to random values"""
        self.W1 = np.random.randn(2, self.hidden_units) * 0.5
        self.b1 = np.zeros(self.hidden_units)
        self.W2 = np.random.randn(self.hidden_units, 1) * 0.5
        self.b2 = np.zeros(1)