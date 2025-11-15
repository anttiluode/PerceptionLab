"""
Hebbian Learner 2 - Error-Driven Learning
------------------------------------------
Enhanced version with dynamic learning rate input.

Learning rate is modulated by external signal (e.g., prediction error/fractal dimension).
This implements the paper's prediction: learning = error × prediction

When error is HIGH → learning rate HIGH → rapid adaptation
When error is LOW → learning rate LOW → maintain structure
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class HebbianLearner2Node(BaseNode):
    """
    Hebbian learner with dynamic learning rate driven by external signal.
    Implements error-modulated plasticity from predictive coding literature.
    """
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 140, 60)  # Brighter orange
    
    def __init__(self, base_learning_rate=0.005, decay=0.995):
        super().__init__()
        self.node_title = "Hebbian Learner 2 (Error-Driven)"
        
        self.inputs = {
            'latent_in': 'spectrum',
            'learning_rate': 'signal',  # NEW: Dynamic learning rate input
            'decay': 'signal',          # Optional dynamic decay
            'reset': 'signal'
        }
        self.outputs = {
            'w_matrix_out': 'image',
            'eigenvalues_out': 'spectrum',
            'current_lr': 'signal',     # NEW: Output the actual learning rate being used
        }
        
        # Configurable defaults
        self.base_learning_rate = float(base_learning_rate)
        self.base_decay = float(decay)
        
        # Internal state
        self.w_matrix = None
        self.eigenvalues = None
        self.current_dim = 0
        self.last_reset = 0.0
        self.actual_learning_rate = self.base_learning_rate  # Track what we're actually using
    
    def step(self):
        # 1. Get Inputs
        latent_in = self.get_blended_input('latent_in', 'first')
        reset_sig = self.get_blended_input('reset', 'sum') or 0.0
        
        # Get dynamic learning rate from signal input
        lr_signal = self.get_blended_input('learning_rate', 'sum')
        decay_sig = self.get_blended_input('decay', 'sum')
        
        # Use signal if provided, else use config default
        if lr_signal is not None and lr_signal > 0:
            lr = lr_signal
        else:
            lr = self.base_learning_rate
            
        if decay_sig is not None:
            decay = decay_sig
        else:
            decay = self.base_decay
        
        # Clamp to safe values
        lr = np.clip(lr, 0.0, 1.0)
        decay = np.clip(decay, 0.8, 1.0)
        
        # Store for output
        self.actual_learning_rate = lr
        
        # 2. Handle Reset
        if reset_sig > 0.5 and self.last_reset <= 0.5:
            self.w_matrix = None
            self.eigenvalues = None
            self.current_dim = 0
        self.last_reset = reset_sig
        
        if latent_in is None:
            if self.w_matrix is not None:
                self.w_matrix *= decay  # Slowly forget if no input
            return
        
        # 3. Initialize or Resize W-Matrix
        dim = len(latent_in)
        if self.w_matrix is None or self.current_dim != dim:
            self.current_dim = dim
            self.w_matrix = np.zeros((dim, dim), dtype=np.float32)
            self.eigenvalues = np.zeros(dim, dtype=np.float32)
        
        # 4. The Hebbian Learning Rule with Dynamic Learning Rate
        # W_new = W_old * decay + (V ⊗ V) * learning_rate
        
        # Calculate the "instantaneous" W-Matrix for this frame
        current_w = np.outer(latent_in, latent_in)
        
        # Accumulate it with DYNAMIC learning rate
        # This is where error-driven learning happens!
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
        
        elif port_name == 'current_lr':
            return self.actual_learning_rate
        
        return None
    
    def get_display_image(self):
        w_vis = self.get_output('w_matrix_out')
        if w_vis is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.putText(img, "Waiting...", (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
        
        w_vis_u8 = (np.clip(w_vis, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap
        img_color = cv2.applyColorMap(w_vis_u8, cv2.COLORMAP_VIRIDIS)
        
        # Add info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_color, f"Dim: {self.current_dim}x{self.current_dim}", (5, 15),
                    font, 0.4, (255, 255, 255), 1)
        cv2.putText(img_color, f"LR: {self.actual_learning_rate:.5f}", (5, 35),
                    font, 0.4, (0, 255, 255), 1)
        
        # Show max eigenvalue (strength of learned pattern)
        if self.eigenvalues is not None and len(self.eigenvalues) > 0:
            max_eig = np.max(np.abs(self.eigenvalues))
            cv2.putText(img_color, f"Max Eig: {max_eig:.3f}", (5, 55),
                       font, 0.4, (255, 255, 0), 1)
        
        # Learning rate indicator bar
        lr_bar_w = int(self.actual_learning_rate / 0.05 * img_color.shape[1])  # Scale assuming max ~0.05
        cv2.rectangle(img_color, (0, img_color.shape[0] - 10), 
                     (lr_bar_w, img_color.shape[0]), (0, 255, 255), -1)
        
        # Resize for display
        img_resized = cv2.resize(img_color, (128, 128), interpolation=cv2.INTER_NEAREST)
        img_resized = np.ascontiguousarray(img_resized)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Base Learning Rate", "base_learning_rate", self.base_learning_rate, None),
            ("Decay (0.8-1.0)", "base_decay", self.base_decay, None),
        ]