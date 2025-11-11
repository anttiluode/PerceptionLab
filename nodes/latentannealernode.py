"""
LatentAnnealerNode - Applies diffusion (noise) and an external force vector
to a latent code.

** THIS FILE HAS BEEN FIXED TO BE COMPATIBLE WITH perception_lab_host.py **
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

class LatentAnnealerNode(BaseNode):
    
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(200, 100, 100) # Annealing Red

    def __init__(self, diffusion=0.2, seed=1234.0):
        super().__init__() 
        self.node_title = "Latent Annealer"
        
        # CORRECT API for inputs/outputs
        self.inputs = {
            "latent_in": "spectrum",  # From VAE
            "force_in": "spectrum",   # From an attractor
            "diffusion": "signal",    # Control diffusion via port
            "seed": "signal"          # Control seed via port
        }
        self.outputs = {
            "latent_out": "spectrum"
        }
        
        # Parameters from config
        self.current_diffusion = float(diffusion)
        self.current_seed = float(seed)
        np.random.seed(int(self.current_seed))
        
        # Internal state
        self.latent_output = None # Start as None

    # CORRECT API for config
    def get_config_options(self):
        return [
            ("Diffusion/Noise", "current_diffusion", self.current_diffusion, None),
            ("Random Seed", "current_seed", self.current_seed, None)
        ]

    # CORRECT API for main logic
    def step(self):
        # Update params from ports if connected
        diffusion_signal = self.get_blended_input("diffusion", "sum")
        if diffusion_signal is not None:
            # Map signal [0, 1] to a [0, 5] range
            self.current_diffusion = diffusion_signal * 5.0 
        
        seed_signal = self.get_blended_input("seed", "sum")
        if seed_signal is not None and int(seed_signal) != int(self.current_seed):
            self.current_seed = int(seed_signal)
            np.random.seed(self.current_seed)

        # Get data
        latent_in_np = self.get_blended_input("latent_in", "first")
        if latent_in_np is None:
            self.latent_output = None
            return
        
        # 1. Annealing (Adding Gaussian Noise for Exploration)
        noise = np.random.normal(0.0, self.current_diffusion, size=latent_in_np.shape).astype(np.float32)
        latent_annealed = latent_in_np + noise
        
        # 2. Attractor Stabilization (Adding Force Vector)
        force_np = self.get_blended_input("force_in", "first")
        if force_np is not None and force_np.shape == latent_annealed.shape:
            # Add the force vector (pulling the state towards the attractor)
            latent_annealed += force_np

        self.latent_output = latent_annealed

    # CORRECT API for output
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_output
        return None

    # Add a simple display
    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.latent_output is not None:
            # Draw latent vector as a bar graph
            latent_dim = len(self.latent_output)
            bar_width = max(1, w // latent_dim)
            
            # Normalize for display
            val_max = np.abs(self.latent_output).max()
            if val_max < 1e-6: val_max = 1.0
            
            for i, val in enumerate(self.latent_output):
                x = i * bar_width
                norm_val = val / val_max
                bar_h = int(abs(norm_val) * (h/2 - 10))
                y_base = h // 2
                
                if val >= 0:
                    color = (0, int(255 * abs(norm_val)), 0)
                    cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color, -1)
                else:
                    color = (0, 0, int(255 * abs(norm_val)))
                    cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color, -1)
            
            cv2.line(img, (0, h//2), (w, h//2), (100,100,100), 1)

        cv2.putText(img, f"Diffusion: {self.current_diffusion:.2f}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        img_contig = np.ascontiguousarray(img)
        return QtGui.QImage(img_contig.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
