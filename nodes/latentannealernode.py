"""
LatentAnnealerNode (Physics in Latent Space)

Applies iterative physics (diffusion/blur + noise) directly to a
latent vector (e.g., from a PCA node).

This simulates the generative process from 'crystal.py' or 'codeofages.py'
within the AI's "conceptual space" to find a stable "conceptual crystal".
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d # For 1D blur

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class LatentAnnealerNode(BaseNode):
    """
    Evolves a latent vector using simple physics rules.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 220, 100) # Physics Green

    def __init__(self, latent_dim=16):
        super().__init__()
        self.node_title = "Latent Annealer"
        
        self.inputs = {
            'latent_in': 'spectrum', # "Seed" vector from VAE/PCA
            'diffusion': 'signal',   # "Blur" / "Coupling"
            'noise': 'signal',       # "Fluctuation" / "Temperature"
            'reset': 'signal'        # 0->1 signal to re-seed
        }
        self.outputs = {
            'latent_out': 'spectrum' # The evolved vector
        }
        
        self.latent_dim = int(latent_dim)
        
        # Internal state
        self.latent_vector = np.random.rand(self.latent_dim).astype(np.float32)
        self.last_reset_val = 0.0

    def step(self):
        # --- 1. Get Inputs ---
        reset_signal = self.get_blended_input('reset', 'sum') or 0.0
        
        # Check for a "rising edge" on the reset signal to seed the simulation
        if reset_signal > 0.5 and self.last_reset_val <= 0.5:
            seed_vector = self.get_blended_input('latent_in', 'first')
            if seed_vector is not None and len(seed_vector) == self.latent_dim:
                print("Latent Annealer: Seeding from PCA input.")
                self.latent_vector = seed_vector.astype(np.float32)
        
        self.last_reset_val = reset_signal
        
        # --- 2. Apply Physics (inspired by crystal.py / codeofages.py) ---
        
        # Get physics parameters
        # 'diffusion' is the sigma for the 1D blur. Higher = more coupling
        diff_val = self.get_blended_input('diffusion', 'sum') or 0.5
        # 'noise' is the "temperature" of the system
        noise_val = self.get_blended_input('noise', 'sum') or 0.01
        
        # a) Diffusion (Gaussian Blur)
        # This makes adjacent PCA components "talk" to each other
        self.latent_vector = gaussian_filter1d(self.latent_vector, 
                                               sigma=diff_val, 
                                               mode='wrap') # 'wrap' = periodic boundary
        
        # b) Fluctuation (Noise)
        self.latent_vector += (np.random.randn(self.latent_dim) * noise_val)
        
        # c) Renormalize (Confinement)
        # Keep the vector in the [0, 1] range
        self.latent_vector = (self.latent_vector - self.latent_vector.min()) / (self.latent_vector.max() - self.latent_vector.min() + 1e-9)

    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_vector
        return None

    def get_display_image(self):
        # Visualize the latent vector as a bar graph
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        bar_width = w / self.latent_dim
        
        for i, val in enumerate(self.latent_vector):
            x = int(i * bar_width)
            bar_h = int(np.clip(val, 0, 1) * (h - 10))
            color_val = int(val * 255)
            cv2.rectangle(img, (x, h - bar_h), (x + int(bar_width) - 2, h), 
                          (color_val, 100, 255 - color_val), -1) # BGR
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None)
        ]