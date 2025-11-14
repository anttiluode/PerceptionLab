"""
Latent To W-Matrix Node - Creates a W-Matrix from a latent vector.

This node performs an outer product on a latent vector (psi),
creating a symmetric W-Matrix (psi ⊗ psi). This is a direct
implementation of Hebbian learning ("neurons that fire together,
wire together") and creates a "memory" or "structure" from a
single "state" or "thought."
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

class LatentToWMatrixNode(BaseNode):
    """
    Takes a 1D latent vector and computes its outer product
    to create a 2D W-Matrix (image).
    """
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(220, 180, 40) # Learned Gold

    def __init__(self):
        super().__init__()
        self.node_title = "Latent to W-Matrix"
        self.inputs = {'latent_in': 'spectrum'}
        self.outputs = {
            'w_matrix_out': 'image',        # The 2D matrix as an image
            'eigenvalues_out': 'spectrum'   # The 1D spectrum of the matrix
        }
        
        # Internal state
        self.w_matrix = np.zeros((16, 16), dtype=np.float32)
        self.eigenvalues = np.zeros(16, dtype=np.float32)
        self.current_dim = 16

    def step(self):
        latent_in = self.get_blended_input('latent_in', 'first')

        if latent_in is None:
            self.w_matrix *= 0.95 # Decay if no input
            return

        # --- 1. Dynamically resize to input vector ---
        self.current_dim = len(latent_in)
        if self.w_matrix.shape[0] != self.current_dim:
            self.w_matrix = np.zeros((self.current_dim, self.current_dim), dtype=np.float32)
            self.eigenvalues = np.zeros(self.current_dim, dtype=np.float32)

        # --- 2. The Core Logic: Outer Product (Hebbian Learning) ---
        # W = psi ⊗ psi
        self.w_matrix = np.outer(latent_in, latent_in)
        
        # --- 3. Symmetrize (like in HumanAttractorNode) ---
        self.w_matrix = (self.w_matrix + self.w_matrix.T) / 2.0
        
        # --- 4. Analyze the matrix's properties ---
        try:
            # Eigenvalues represent the "strength" of its principal patterns
            self.eigenvalues = np.linalg.eigvalsh(self.w_matrix)
        except np.linalg.LinAlgError:
            self.eigenvalues.fill(0.0)

    def get_output(self, port_name):
        if port_name == 'w_matrix_out':
            # Normalize matrix to [0, 1] for image output
            mat_min = self.w_matrix.min()
            mat_max = self.w_matrix.max()
            range_val = mat_max - mat_min
            
            if range_val < 1e-9:
                return np.zeros_like(self.w_matrix)
            
            return (self.w_matrix - mat_min) / range_val
        
        elif port_name == 'eigenvalues_out':
            # Output the "energy" of the matrix's patterns
            return self.eigenvalues.astype(np.float32)
        
        return None

    def get_display_image(self):
        # Get the normalized W-Matrix
        w_vis = self.get_output('w_matrix_out')
        w_vis_u8 = (np.clip(w_vis, 0, 1) * 255).astype(np.uint8)
        
        # Apply a colormap (Viridis is good for this)
        img_color = cv2.applyColorMap(w_vis_u8, cv2.COLORMAP_VIRIDIS)
        
        # Add dimension text
        cv2.putText(img_color, f"Dim: {self.current_dim}x{self.current_dim}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Resize for display
        img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_NEAREST)
        img_resized = np.ascontiguousarray(img_resized)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        # This node is fully dynamic based on input, so no config is needed.
        return []