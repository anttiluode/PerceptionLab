"""
Latent Explorer Node - Manipulate individual PCA coefficients
Explore what each principal component controls in your visual space
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class LatentExplorerNode(BaseNode):
    """
    Interactive manipulation of PCA latent codes.
    Add/subtract individual principal components to see what they control.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(220, 120, 180)
    
    def __init__(self, num_controls=8):
        super().__init__()
        self.node_title = "Latent Explorer"
        
        self.inputs = {
            'latent_in': 'spectrum',
            'pc0_mod': 'signal',  # Modulation for PC0
            'pc1_mod': 'signal',
            'pc2_mod': 'signal',
            'pc3_mod': 'signal',
            'pc4_mod': 'signal',
            'pc5_mod': 'signal',
            'pc6_mod': 'signal',
            'pc7_mod': 'signal',
            'global_scale': 'signal',  # Scale all modifications
            'reset': 'signal'  # Reset to original
        }
        self.outputs = {
            'latent_out': 'spectrum',
            'delta': 'spectrum',  # The modification vector
            'magnitude': 'signal'  # How much we've changed
        }
        
        self.num_controls = int(num_controls)
        
        # State
        self.latent_original = None
        self.latent_modified = None
        self.delta_vector = None
        self.magnitude = 0.0
        
        # Internal modulation values (for display when no signal input)
        self.internal_mods = np.zeros(8)
        
    def step(self):
        # Get inputs
        latent_in = self.get_blended_input('latent_in', 'first')
        global_scale = self.get_blended_input('global_scale', 'sum')
        if global_scale is None:
            global_scale = 1.0
            
        reset_signal = self.get_blended_input('reset', 'sum') or 0.0
        
        if latent_in is None:
            return
            
        # Store original
        if self.latent_original is None or reset_signal > 0.5:
            self.latent_original = latent_in.copy()
            
        # Get modulation values for each PC
        mods = []
        for i in range(min(self.num_controls, len(latent_in))):
            mod_signal = self.get_blended_input(f'pc{i}_mod', 'sum')
            if mod_signal is not None:
                mods.append(mod_signal * global_scale)
                self.internal_mods[i] = mod_signal
            else:
                mods.append(0.0)
                
        # Create delta vector
        self.delta_vector = np.zeros_like(latent_in)
        for i, mod in enumerate(mods):
            self.delta_vector[i] = mod * 2.0  # Scale for visibility
            
        # Apply modifications
        self.latent_modified = self.latent_original + self.delta_vector
        
        # Calculate magnitude of change
        self.magnitude = np.linalg.norm(self.delta_vector)
        
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_modified
        elif port_name == 'delta':
            return self.delta_vector
        elif port_name == 'magnitude':
            return self.magnitude
        return None
        
    def get_display_image(self):
        """
        Visualize:
        - Top: Original latent code (gray)
        - Middle: Delta vector (colored by +/-)
        - Bottom: Modified latent code
        """
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        if self.latent_original is None:
            cv2.putText(img, "Waiting for input...", (10, 128), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return QtGui.QImage(img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)
            
        latent_dim = len(self.latent_original)
        bar_width = max(1, 256 // latent_dim)
        
        # Helper function to draw latent code
        def draw_code(code, y_offset, color_fn):
            code_norm = code.copy()
            code_max = np.abs(code_norm).max()
            if code_max > 1e-6:
                code_norm = code_norm / code_max
                
            for i, val in enumerate(code_norm):
                x = i * bar_width
                h = int(abs(val) * 64)
                y_base = y_offset + 64
                
                if val >= 0:
                    y_start = y_base - h
                    y_end = y_base
                else:
                    y_start = y_base
                    y_end = y_base + h
                    
                color = color_fn(i, val)
                cv2.rectangle(img, (x, y_start), (x+bar_width-1, y_end), color, -1)
                
            # Draw baseline
            cv2.line(img, (0, y_offset+64), (256, y_offset+64), (100,100,100), 1)
            
        # Draw original (top section)
        draw_code(self.latent_original, 0, lambda i, v: (150, 150, 150))
        
        # Draw delta (middle section) - colored by sign
        def delta_color(i, val):
            if i < self.num_controls:
                # Controlled PCs: red for negative, green for positive
                if val > 0:
                    return (0, int(255 * abs(val)), 0)
                else:
                    return (0, 0, int(255 * abs(val)))
            else:
                return (100, 100, 100)  # Uncontrolled PCs
                
        draw_code(self.delta_vector, 64, delta_color)
        
        # Draw modified (bottom section) - highlight active PCs
        def modified_color(i, val):
            if i < self.num_controls and abs(self.delta_vector[i]) > 0.01:
                # Active PC: bright cyan
                return (255, 255, 0)
            else:
                # Inactive: white
                return (200, 200, 200)
                
        draw_code(self.latent_modified, 128, modified_color)
        
        # Labels
        cv2.putText(img, "ORIG", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, "DELTA", (5, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, "MOD", (5, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Magnitude indicator
        mag_text = f"||Δ||={self.magnitude:.3f}"
        cv2.putText(img, mag_text, (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        return QtGui.QImage(img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Num Controls", "num_controls", self.num_controls, None)
        ]