"""
IHT Attractor W-Matrix Node - The learned holographic decoder
Implements trainable complex linear mapping W that projects
high-dimensional quantum states onto stable classical attractors.

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class IHTAttractorWNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(200, 50, 150)  # Magenta for attractor
    
    def __init__(self, hidden_dim=128, mapping_type='Learned'):
        super().__init__()
        self.node_title = "IHT W-Matrix"
        
        self.inputs = {
            'phase_field': 'image',     # Input quantum state
            'train_signal': 'signal'    # Trigger training steps
        }
        
        self.outputs = {
            'projected_field': 'image',     # W * ψ
            'attractor_image': 'image',     # Visualization of W structure
            'projection_quality': 'signal'   # How well it projects
        }
        
        self.hidden_dim = int(hidden_dim)
        self.mapping_type = mapping_type
        
        # The W matrix (complex)
        self.W = None
        self.last_input_shape = None
        
        # Training state
        self.training_mode = False
        self.learning_rate = 0.001
        self.loss_history = []
        
        # Outputs
        self.projected = None
        self.quality = 0.0
        
    def _init_W(self, input_size):
        """Initialize W matrix based on mapping type"""
        if self.mapping_type == 'Identity':
            # Baseline: just pass through
            self.W = np.eye(input_size, dtype=np.complex64)
            
        elif self.mapping_type == 'Random':
            # Random orthonormal (delocalized)
            real_part = np.random.randn(input_size, input_size)
            imag_part = np.random.randn(input_size, input_size)
            W_rand = real_part + 1j * imag_part
            
            # Orthonormalize via QR decomposition
            Q, R = np.linalg.qr(W_rand)
            self.W = Q.astype(np.complex64)
            
        elif self.mapping_type == 'Learned':
            # Start with identity + small noise
            self.W = np.eye(input_size, dtype=np.complex64)
            noise_scale = 0.01
            self.W += (np.random.randn(input_size, input_size) + 
                      1j * np.random.randn(input_size, input_size)) * noise_scale
            
    def _apply_W(self, psi_flat):
        """Apply W matrix to flattened complex field"""
        if self.W is None or self.W.shape[0] != len(psi_flat):
            self._init_W(len(psi_flat))
            
        return np.dot(self.W, psi_flat)
        
    def _compute_loss(self, psi_projected, psi_original):
        """Loss = negative coherence of projection"""
        # We want high coherence (phase alignment)
        coherence = np.abs(np.sum(psi_projected)) / (np.sum(np.abs(psi_projected)) + 1e-9)
        return -coherence  # Maximize coherence = minimize negative coherence
        
    def _gradient_step(self, psi_flat):
        """Simple gradient descent on W"""
        # Forward pass
        projected = self._apply_W(psi_flat)
        loss = self._compute_loss(projected, psi_flat)
        
        # Numerical gradient (finite differences)
        epsilon = 1e-5
        grad_W = np.zeros_like(self.W)
        
        # Only update a small random subset for speed
        n_samples = min(100, self.W.size)
        idx_i = np.random.randint(0, self.W.shape[0], n_samples)
        idx_j = np.random.randint(0, self.W.shape[1], n_samples)
        
        for i, j in zip(idx_i, idx_j):
            # Real part
            self.W[i, j] += epsilon
            proj_plus = self._apply_W(psi_flat)
            loss_plus = self._compute_loss(proj_plus, psi_flat)
            self.W[i, j] -= epsilon
            
            grad_W[i, j] = (loss_plus - loss) / epsilon
            
        # Update W
        self.W -= self.learning_rate * grad_W
        
        # Normalize rows to maintain stability
        for i in range(self.W.shape[0]):
            norm = np.linalg.norm(self.W[i, :])
            if norm > 1e-9:
                self.W[i, :] /= norm
                
        self.loss_history.append(float(loss))
        
    def step(self):
        phase_field = self.get_blended_input('phase_field', 'mean')
        train_signal = self.get_blended_input('train_signal', 'sum')
        
        if phase_field is None:
            return
            
        # Convert RGB phase field back to complex
        # (This is a simplification - in real use, we'd pass complex directly)
        if phase_field.ndim == 3:
            # Assume grayscale for now
            amp = np.mean(phase_field, axis=2)
        else:
            amp = phase_field
            
        h, w = amp.shape
        
        # Create complex field (amplitude only for now)
        psi_2d = amp.astype(np.complex64)
        psi_flat = psi_2d.flatten()
        
        # Training mode
        if train_signal is not None and train_signal > 0.5:
            self.training_mode = True
            self._gradient_step(psi_flat)
        else:
            self.training_mode = False
            
        # Apply W
        projected_flat = self._apply_W(psi_flat)
        self.projected = projected_flat.reshape(h, w)
        
        # Compute quality metric
        coherence = np.abs(np.sum(projected_flat)) / (np.sum(np.abs(projected_flat)) + 1e-9)
        self.quality = float(coherence)
        
    def get_output(self, port_name):
        if port_name == 'projected_field':
            if self.projected is None:
                return None
            # Return amplitude as image
            amp = np.abs(self.projected)
            amp_norm = amp / (amp.max() + 1e-9)
            return amp_norm.astype(np.float32)
            
        elif port_name == 'attractor_image':
            # Visualize W structure (first few rows)
            if self.W is None:
                return np.zeros((64, 64), dtype=np.float32)
                
            # Take a square subset
            n = min(64, self.W.shape[0])
            W_sub = self.W[:n, :n]
            
            # Show amplitude
            amp = np.abs(W_sub)
            amp_norm = amp / (amp.max() + 1e-9)
            return amp_norm.astype(np.float32)
            
        elif port_name == 'projection_quality':
            return self.quality
            
        return None
        
    def get_display_image(self):
        w_vis = self.get_output('attractor_image')
        if w_vis is None:
            return None
            
        img_u8 = (w_vis * 255).astype(np.uint8)
        
        # Add training indicator
        if self.training_mode:
            img_u8[:5, :] = 255  # White bar at top
            
        img_u8 = np.ascontiguousarray(img_u8)
        h, w = img_u8.shape
        return QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        
    def get_config_options(self):
        return [
            ("Mapping Type", "mapping_type", self.mapping_type, [
                ("Identity (Baseline)", "Identity"),
                ("Random (Delocalized)", "Random"),
                ("Learned (Optimized)", "Learned")
            ]),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
        ]