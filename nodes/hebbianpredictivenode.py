"""
Hebbian Predictive Node
-----------------------
A memory node that generates active predictions.
It learns the statistical structure of the input (Latent Vector)
and attempts to reconstruct it.

The difference between Input and Prediction is "Surprise".

Inputs:
- latent_in (spectrum): The data to learn (from VAE).
- learning_rate (signal): How fast to update (from Observer's Plasticity).

Outputs:
- prediction (spectrum): The reconstructed vector (To Observer).
- error (signal): Magnitude of reconstruction error.
- weights (image): Visualization of the learned patterns.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class HebbianPredictiveNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 140, 0) # Dark Orange

    def __init__(self, latent_dim=16, learning_rate=0.01):
        super().__init__()
        self.node_title = "Hebbian Predictive Memory"
        
        self.inputs = {
            'latent_in': 'spectrum',      # Input vector
            'learning_rate': 'signal'     # Plasticity modulation
        }
        
        self.outputs = {
            'prediction': 'spectrum',     # The reconstruction (Connect to Observer)
            'error': 'signal',            # Local error metric
            'weights': 'image'            # View the memory
        }
        
        self.latent_dim = int(latent_dim)
        self.base_lr = float(learning_rate)
        
        # Initialize Weights (Identity + Noise to start)
        # We use a simple auto-associative matrix (dim x dim)
        # or a feature dictionary. Let's use a single layer auto-associator (W)
        # Prediction y = W * x
        # But standard Oja is for principal components.
        # Let's use a simple "Leaky Integrator" for the mean prediction (Expectation)
        # AND a covariance learner.
        # ACTUALLY, for the Observer loop, the best "Prediction" is the 
        # Reconstructed Input from the learned Manifold.
        
        # We will use a single-layer linear autoencoder trained via Hebbian rule.
        # y = Wx (Encode) -> x_hat = W.T y (Decode)
        # But W is orthonormalized via Oja's rule.
        
        self.weights = np.random.randn(self.latent_dim, self.latent_dim).astype(np.float32) * 0.1
        
        # Internal state
        self.prediction_val = np.zeros(self.latent_dim, dtype=np.float32)
        self.error_val = 0.0
        self.weight_vis = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        # 1. Get Input
        x = self.get_blended_input('latent_in', 'first')
        mod_lr = self.get_blended_input('learning_rate', 'sum')
        
        if x is None:
            return

        # Ensure dimensions match
        if len(x) != self.latent_dim:
            # Resize or pad
            new_x = np.zeros(self.latent_dim, dtype=np.float32)
            n = min(len(x), self.latent_dim)
            new_x[:n] = x[:n]
            x = new_x
            
        # Determine Learning Rate (Base * Modulation)
        # If mod_lr is None (not connected), use base. 
        # If connected (from Observer), it acts as a multiplier/gate.
        eta = self.base_lr
        if mod_lr is not None:
            eta *= np.clip(mod_lr, 0.0, 10.0) # Allow boosting up to 10x

        # 2. Forward Pass (Prediction)
        # In a linearized Hebbian PCA network (Sanger's Rule context):
        # Activation y = W @ x
        y = np.dot(self.weights, x)
        
        # Reconstruction (Prediction) x_hat = W.T @ y
        # This projects the input onto the learned "valid" subspace
        x_hat = np.dot(self.weights.T, y)
        
        self.prediction_val = x_hat
        
        # 3. Hebbian Update (Learning)
        # Generalized Hebbian Algorithm (Sanger's Rule) or Simple Oja
        # dW = eta * (y * (x - W.T*y).T) 
        # But element-wise for efficiency in numpy:
        # Residual = x - x_hat
        residual = x - x_hat
        self.error_val = np.mean(residual**2)
        
        # Update weights: W += eta * y * residual
        # We need to reshape for outer product
        # dW[i, j] = eta * y[i] * residual[j]
        dW = eta * np.outer(y, residual)
        
        self.weights += dW
        
        # Normalization (prevent explosion)
        # Oja's rule inherently normalizes, but explicit check helps stability
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True) + 1e-9
        self.weights /= norms

        # 4. Visualization (Weights)
        # Normalize weights to 0-255
        w_min, w_max = self.weights.min(), self.weights.max()
        w_norm = (self.weights - w_min) / (w_max - w_min + 1e-9)
        
        vis_size = 128
        w_img = cv2.resize(w_norm, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        self.weight_vis = cv2.applyColorMap((w_img * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Overlay Error
        cv2.putText(self.weight_vis, f"Err: {self.error_val:.4f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def get_output(self, port_name):
        if port_name == 'prediction':
            return self.prediction_val
        elif port_name == 'error':
            return float(self.error_val)
        elif port_name == 'weights':
            return self.weight_vis.astype(np.float32) / 255.0
        return None

    def get_display_image(self):
        return QtGui.QImage(self.weight_vis.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None),
            ("Base Learning Rate", "base_lr", self.base_lr, None)
        ]