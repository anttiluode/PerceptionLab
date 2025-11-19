"""
Cabbage Scanner Node (Safe)
---------------------------
[FIX] Changed error metric to Mean Absolute Error (MAE) to prevent square-overflow.
[FIX] Added input sanitization.
"""
import numpy as np
import cv2
from scipy.special import jn, jn_zeros
import json
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class CabbageScannerNode(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(0, 255, 128) 

    def __init__(self):
        super().__init__()
        self.node_title = "Cabbage Scanner"
        
        self.inputs = {
            'target_image': 'image',
        }
        
        self.outputs = {
            'dna_55': 'spectrum', 
            'reconstruction': 'image',
            'error': 'signal'
        }
        
        self.resolution = 128
        self.max_n = 5
        self.max_m = 5
        
        self.basis_functions = []
        self.coefficients = np.zeros(55, dtype=np.float32)
        self.reconstruction_img = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.error_val = 0.0
        
        self._precompute_basis()

    def _precompute_basis(self):
        h, w = self.resolution, self.resolution
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        x_norm = (x - cx) / (w / 2)
        y_norm = (y - cy) / (h / 2)
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-9
        theta = np.arctan2(y_norm, x_norm)
        mask = (r <= 1.0).astype(np.float32)
        
        self.basis_functions = []
        
        for n in range(1, self.max_n + 1):
            for m in range(0, self.max_m + 1):
                if m == 0:
                    zeros = jn_zeros(0, n)
                    k = zeros[-1]
                    radial = jn(0, k * r)
                    mode = radial * mask
                    mode /= (np.linalg.norm(mode) + 1e-9)
                    self.basis_functions.append(mode)
                else:
                    zeros = jn_zeros(m, n)
                    k = zeros[-1]
                    radial = jn(m, k * r)
                    mode_c = radial * np.cos(m * theta) * mask
                    mode_c /= (np.linalg.norm(mode_c) + 1e-9)
                    self.basis_functions.append(mode_c)
                    mode_s = radial * np.sin(m * theta) * mask
                    mode_s /= (np.linalg.norm(mode_s) + 1e-9)
                    self.basis_functions.append(mode_s)

    def step(self):
        target = self.get_blended_input('target_image', 'mean')
        if target is None: return

        # Safety Formatting
        if not np.all(np.isfinite(target)): return # Skip bad frames
        
        if target.dtype == np.float64: target = target.astype(np.float32)
        if len(target.shape) == 3: target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        # Robust Normalization
        t_max = target.max()
        if t_max > 1.0: target /= 255.0
        elif t_max > 0: target /= t_max # Auto-gain
        
        if target.shape[:2] != (self.resolution, self.resolution):
            target = cv2.resize(target, (self.resolution, self.resolution))

        # Decompose
        coeffs = []
        recon = np.zeros_like(target)
        
        for mode in self.basis_functions:
            w = np.sum(target * mode)
            coeffs.append(w)
            recon += w * mode
            
        self.coefficients = np.array(coeffs, dtype=np.float32)
        self.reconstruction_img = np.clip(recon, 0, 1)
        
        # Safe Error Calculation (MAE instead of MSE to prevent overflow)
        self.error_val = np.mean(np.abs(target - self.reconstruction_img))

    def get_output(self, port_name):
        if port_name == 'dna_55': return self.coefficients
        if port_name == 'reconstruction': return self.reconstruction_img
        if port_name == 'error': return float(self.error_val)
        return None

    def get_display_image(self):
        img = (self.reconstruction_img * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        cv2.putText(img, f"DNA Len: {len(self.coefficients)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)