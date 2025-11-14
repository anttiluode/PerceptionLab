"""
EEG Processor Node
Assembles all separate EEG band signals into a single, boosted
latent vector. Also provides individual boosted outputs.

This node is designed to:
1.  Collect all 6 outputs from an EEG node.
2.  Amplify them with a 'Base Scale' and a 'Scale Mod' input.
3.  Bundle them into a 6-dimensional 'latent_out' (spectrum) vector
    for use in VAEs, W-Matrix, or other latent-space nodes.
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

class EEGProcessorNode(BaseNode):
    """
    Assembles EEG signals into a single, scaled latent vector.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(60, 140, 160) # EEG Blue

    def __init__(self, base_scale=1.0):
        super().__init__()
        self.node_title = "EEG Processor"
        self.base_scale = float(base_scale)

        self.inputs = {
            'delta_in': 'signal',
            'theta_in': 'signal',
            'alpha_in': 'signal',
            'beta_in': 'signal',
            'gamma_in': 'signal',
            'raw_in': 'signal',
            'scale_mod': 'signal' # To dynamically change the boost
        }
        self.outputs = {
            'latent_out': 'spectrum', # The 6D boosted vector
            'delta_out': 'signal',
            'theta_out': 'signal',
            'alpha_out': 'signal',
            'beta_out': 'signal',
            'gamma_out': 'signal',
            'raw_out': 'signal'
        }

        # Internal state
        self.latent_vector = np.zeros(6, dtype=np.float32)

    def step(self):
        # 1. Get total scale
        # Use the base_scale from config, multiplied by the signal input
        scale_mod = self.get_blended_input('scale_mod', 'sum')
        if scale_mod is None:
            total_scale = self.base_scale
        else:
            # We add 1.0 so a 0.0 signal input means 1x scale
            total_scale = self.base_scale * (1.0 + scale_mod)

        # 2. Get and scale all inputs
        d = (self.get_blended_input('delta_in', 'sum') or 0.0) * total_scale
        t = (self.get_blended_input('theta_in', 'sum') or 0.0) * total_scale
        a = (self.get_blended_input('alpha_in', 'sum') or 0.0) * total_scale
        b = (self.get_blended_input('beta_in', 'sum') or 0.0) * total_scale
        g = (self.get_blended_input('gamma_in', 'sum') or 0.0) * total_scale
        r = (self.get_blended_input('raw_in', 'sum') or 0.0) * total_scale

        # 3. Assemble the latent vector
        self.latent_vector[0] = d
        self.latent_vector[1] = t
        self.latent_vector[2] = a
        self.latent_vector[3] = b
        self.latent_vector[4] = g
        self.latent_vector[5] = r

    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_vector.astype(np.float32)
        elif port_name == 'delta_out':
            return float(self.latent_vector[0])
        elif port_name == 'theta_out':
            return float(self.latent_vector[1])
        elif port_name == 'alpha_out':
            return float(self.latent_vector[2])
        elif port_name == 'beta_out':
            return float(self.latent_vector[3])
        elif port_name == 'gamma_out':
            return float(self.latent_vector[4])
        elif port_name == 'raw_out':
            return float(self.latent_vector[5])
        return None

    def get_display_image(self):
        """Visualize the 6-dimensional latent vector"""
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        bar_width = max(1, w // 6)
        
        # Normalize for display
        val_max = np.abs(self.latent_vector).max()
        if val_max < 1e-6: 
            val_max = 1.0
        
        labels = ["Del", "The", "Alp", "Beta", "Gam", "Raw"]
        
        for i, val in enumerate(self.latent_vector):
            x = i * bar_width
            norm_val = val / val_max
            bar_h = int(abs(norm_val) * (h/2 - 10))
            y_base = h // 2
            
            if val >= 0:
                color = (0, int(255 * abs(norm_val)), 0) # Green
                cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color, -1)
            else:
                color = (0, 0, int(255 * abs(norm_val))) # Red
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color, -1)
            
            # Draw label
            cv2.putText(img, labels[i], (x + 5, h - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Baseline
        cv2.line(img, (0, h//2), (w, h//2), (100,100,100), 1)
        
        scale_mod = self.get_blended_input('scale_mod', 'sum')
        total_scale = self.base_scale * (1.0 + scale_mod) if scale_mod is not None else self.base_scale
        
        cv2.putText(img, f"Boost: {total_scale:.2f}x", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Base Scale (Boost)", "base_scale", self.base_scale, None)
        ]