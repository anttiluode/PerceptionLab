"""
Latent Assembler Node (v2 - Corrected)
Collects individual signal inputs and assembles them into a latent vector (spectrum).
This node ONLY assembles. It does NOT decode.

The 'latent_out' port (orange) should be connected back to the 'latent_in'
port of the RealVAENode to be decoded by the TRAINED model.
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

class LatentAssemblerNode(BaseNode):
    """
    Assembles multiple signal inputs into a single latent vector (spectrum).
    Can also passthrough a spectrum and modify specific components.
    """
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(150, 150, 150)
    
    def __init__(self, latent_dim=16):
        super().__init__()
        self.node_title = "Latent Assembler"
        
        self.latent_dim = int(latent_dim)
        
        # Create inputs: one for each latent dimension
        self.inputs = {
            'latent_base': 'spectrum',  # Optional base
        }
        for i in range(self.latent_dim):
            self.inputs[f'in_{i}'] = 'signal'
        
        self.outputs = {
            'latent_out': 'spectrum',
            # --- REMOVED 'image_out' ---
        }
        
        self.latent_vector = np.zeros(self.latent_dim, dtype=np.float32)

    def step(self):
        # Start with base latent if provided
        base = self.get_blended_input('latent_base', 'first')
        
        if base is not None:
            # Use base as starting point
            if len(base) >= self.latent_dim:
                self.latent_vector = base[:self.latent_dim].astype(np.float32)
            else:
                # Pad if base is too short
                self.latent_vector = np.zeros(self.latent_dim, dtype=np.float32)
                self.latent_vector[:len(base)] = base.astype(np.float32)
        else:
            # Start from zeros
            self.latent_vector = np.zeros(self.latent_dim, dtype=np.float32)
        
        # Override with individual signal inputs (if connected)
        for i in range(self.latent_dim):
            signal_val = self.get_blended_input(f'in_{i}', 'sum')
            if signal_val is not None:
                self.latent_vector[i] = float(signal_val)
    
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_vector
        return None
    
    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        bar_width = max(1, w // self.latent_dim)
        
        # Normalize for display
        val_max = np.abs(self.latent_vector).max()
        if val_max < 1e-6: 
            val_max = 1.0
        
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
            
            # Label every 4th
            if i % 4 == 0:
                cv2.putText(img, str(i), (x+2, h-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Baseline
        cv2.line(img, (0, h//2), (w, h//2), (100,100,100), 1)
        
        # Status
        cv2.putText(img, f"Dim: {self.latent_dim}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None)
        ]