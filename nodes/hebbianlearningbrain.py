"""
Hebbian Learner Node - A "Latent Brain"
This node models a simple brain that learns from a stream of
latent vectors. It has an internal W-Matrix (its memory/structure)
that it updates using a Hebbian learning rule (outer product).

It "learns" the long-term correlation structure of its inputs.
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

class HebbianLearnerNode(BaseNode):
    """
    Takes a 1D latent vector and slowly accumulates its
    outer product into a stable W-Matrix.
    """
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 120, 40) # Learned Orange

    def __init__(self, learning_rate=0.01, decay=0.995):
        super().__init__()
        self.node_title = "Hebbian Learner (Brain)"
        
        self.inputs = {
            'latent_in': 'spectrum',
            'learning_rate': 'signal',
            'decay': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'w_matrix_out': 'image',        # The learned 2D matrix
            'eigenvalues_out': 'spectrum'   # The matrix's patterns
        }
        
        # Configurable defaults
        self.base_learning_rate = float(learning_rate)
        self.base_decay = float(decay)

        # Internal state
        self.w_matrix = None
        self.eigenvalues = None
        self.current_dim = 0
        self.last_reset = 0.0

    def step(self):
        # 1. Get Inputs
        latent_in = self.get_blended_input('latent_in', 'first')
        reset_sig = self.get_blended_input('reset', 'sum') or 0.0
        
        # Get dynamic learning/decay rates
        lr_sig = self.get_blended_input('learning_rate', 'sum')
        decay_sig = self.get_blended_input('decay', 'sum')
        
        # Use signal if provided, else use config default
        lr = lr_sig if lr_sig is not None else self.base_learning_rate
        decay = decay_sig if decay_sig is not None else self.base_decay
        
        # Clamp to safe values
        lr = np.clip(lr, 0.0, 1.0)
        decay = np.clip(decay, 0.8, 1.0)

        # 2. Handle Reset
        if reset_sig > 0.5 and self.last_reset <= 0.5:
            self.w_matrix = None
            self.eigenvalues = None
            self.current_dim = 0
        self.last_reset = reset_sig

        if latent_in is None:
            if self.w_matrix is not None:
                self.w_matrix *= decay # Slowly forget if no input
            return

        # 3. Initialize or Resize W-Matrix
        dim = len(latent_in)
        if self.w_matrix is None or self.current_dim != dim:
            self.current_dim = dim
            self.w_matrix = np.zeros((dim, dim), dtype=np.float32)
            self.eigenvalues = np.zeros(dim, dtype=np.float32)

        # 4. The Hebbian Learning Rule (Leaky Accumulator)
        # W_new = W_old * decay + (V ⊗ V) * learning_rate
        
        # Calculate the "instantaneous" W-Matrix for this frame
        current_w = np.outer(latent_in, latent_in)
        
        # Accumulate it into the long-term memory matrix
        self.w_matrix = (self.w_matrix * decay) + (current_w * lr)
        
        # 5. Symmetrize and Analyze
        self.w_matrix = (self.w_matrix + self.w_matrix.T) / 2.0
        try:
            self.eigenvalues = np.linalg.eigvalsh(self.w_matrix)
        except np.linalg.LinAlgError:
            self.eigenvalues.fill(0.0)

    def get_output(self, port_name):
        if port_name == 'w_matrix_out':
            if self.w_matrix is None:
                return None
            
            # Normalize for image output
            mat_min = self.w_matrix.min()
            mat_max = self.w_matrix.max()
            range_val = mat_max - mat_min
            
            if range_val < 1e-9:
                return np.zeros_like(self.w_matrix)
            
            return (self.w_matrix - mat_min) / range_val
        
        elif port_name == 'eigenvalues_out':
            return self.eigenvalues.astype(np.float32) if self.eigenvalues is not None else None
        
        return None

    def get_display_image(self):
        w_vis = self.get_output('w_matrix_out')
        if w_vis is None:
            img = np.zeros((96, 96, 3), dtype=np.uint8)
            cv2.putText(img, "Waiting...", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

        w_vis_u8 = (np.clip(w_vis, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap
        img_color = cv2.applyColorMap(w_vis_u8, cv2.COLORMAP_VIRIDIS)
        
        cv2.putText(img_color, f"Dim: {self.current_dim}x{self.current_dim}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Resize for display
        img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_NEAREST)
        img_resized = np.ascontiguousarray(img_resized)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Learning Rate", "base_learning_rate", self.base_learning_rate, None),
            ("Decay (0.8-1.0)", "base_decay", self.base_decay, None),
        ]