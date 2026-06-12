"""
Skew Operator Node (The Entropy Engine)
---------------------------------------
Computes the Lag Covariance of a vector stream and isolates the Skew (A) matrix.
Outputs the "Skew Flux" (Frobenius norm of A), representing broken detailed balance,
directed time, and thermodynamic entropy.
"""

import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SkewOperatorNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(100, 50, 150) # Deep Purple

    def __init__(self, tau=5, dim=16):
        super().__init__()
        self.node_title = "Skew Operator (Lag-Cov)"
        self.inputs = {'vector_in': 'spectrum'}
        self.outputs = {
            'skew_matrix': 'image', 
            'skew_flux': 'signal'
        }
        
        self.tau = int(tau)
        self.dim = int(dim)
        self.history = deque(maxlen=self.tau + 1)
        
        self.C = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.skew_matrix = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.skew_flux = 0.0
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        vec = self.get_blended_input('vector_in', 'first')
        if vec is None:
            return
            
        # Ensure correct dimension
        v = np.array(vec, dtype=np.float32)
        if len(v) > self.dim: v = v[:self.dim]
        elif len(v) < self.dim: v = np.pad(v, (0, self.dim - len(v)))

        self.history.append(v)
        
        # We need at least tau+1 frames to compute lag covariance
        if len(self.history) >= self.tau + 1:
            v_t = self.history[-1]
            v_tau = self.history[0]
            
            # Rank-1 update to lag covariance (Exponential moving average)
            C_update = np.outer(v_t, v_tau)
            self.C = 0.95 * self.C + 0.05 * C_update

            # A = (C - C.T) / 2
            self.skew_matrix = 0.5 * (self.C - self.C.T)
            
            # Entropy Flux: Approximated here as the Frobenius norm of A 
            # (magnitude of rotational energy)
            self.skew_flux = float(np.linalg.norm(self.skew_matrix, 'fro'))
        
        # Render the Skew Matrix A as a Jet heatmap for the UI
        # Scale up slightly for visual clarity
        norm_A = np.clip(self.skew_matrix * 15.0 + 0.5, 0, 1) 
        img_u8 = (norm_A * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        self.display_img = cv2.resize(img_color, (128, 128), interpolation=cv2.INTER_NEAREST)

        # Print the Wattage/Flux on the node
        cv2.putText(self.display_img, f"Flux: {self.skew_flux:.4f}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def get_output(self, port_name):
        if port_name == 'skew_matrix': 
            # Output as float32 image for downstream visualizer if needed
            return np.stack([self.skew_matrix]*3, axis=-1)
        if port_name == 'skew_flux': 
            return self.skew_flux
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Lag (Tau)", "tau", self.tau, None),
            ("Vector Dim", "dim", self.dim, None)
        ]