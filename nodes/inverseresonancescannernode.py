"""
Inverse Resonance Node (The Soul Scanner) - Robust Fix
------------------------------------------------------
Performs "Inverse Morphogenesis."
It takes a visual input (Target Shape) and decomposes it into its
fundamental Eigenmode Coefficients (The "Address" or "DNA").

[FIXES]
- Handles float64 image inputs (OpenCV crash).
- Handles list vs numpy array initialization (AttributeError crash).
"""

import numpy as np
import cv2
from scipy.special import jn, jn_zeros
import json
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class InverseResonanceNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(0, 255, 255) # Cyan

    def __init__(self, resolution=128, max_n=5, max_m=5):
        super().__init__()
        self.node_title = "Inverse Resonance Scanner"
        
        self.inputs = {
            'target_image': 'image',    # The physical object to scan
            'scan_trigger': 'signal'    # > 0.5 to capture/save
        }
        
        self.outputs = {
            'dna_spectrum': 'spectrum', # The extracted address
            'reconstruction': 'image',  # The mathematical shadow
            'scan_error': 'signal'
        }
        
        self.resolution = int(resolution)
        self.max_n = int(max_n)
        self.max_m = int(max_m)
        
        # State - Initialize as Arrays to prevent Type Errors
        self.basis_functions = []
        self.basis_indices = []
        self.coefficients = np.array([], dtype=np.float32) 
        self.reconstruction_img = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.error_val = 0.0
        
        # Precompute the "Library of Forms" (Basis Set)
        self._precompute_basis()

    def _create_ellipsoidal_mask(self):
        h, w = self.resolution, self.resolution
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        mask = ((x - cx)**2 + (y - cy)**2) <= (h // 2)**2
        return mask.astype(np.float32)

    def _precompute_basis(self):
        self.basis_functions = []
        self.basis_indices = []
        
        h, w = self.resolution, self.resolution
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        
        x_norm = (x - cx) / (w / 2)
        y_norm = (y - cy) / (h / 2)
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-9
        theta = np.arctan2(y_norm, x_norm)
        
        mask = self._create_ellipsoidal_mask()
        
        for n in range(1, self.max_n + 1):
            for m in range(0, self.max_m + 1):
                if m == 0:
                    zeros = jn_zeros(0, n)
                    k = zeros[-1]
                    radial = jn(0, k * r)
                    angular_cos = 1.0
                    angular_sin = 0.0
                else:
                    zeros = jn_zeros(m, n)
                    k = zeros[-1]
                    radial = jn(m, k * r)
                    angular_cos = np.cos(m * theta)
                    angular_sin = np.sin(m * theta)
                
                # Real Component (Cosine)
                if m == 0:
                    mode = radial * mask
                    mode /= (np.linalg.norm(mode) + 1e-9)
                    self.basis_functions.append(mode)
                    self.basis_indices.append((n, m, 'cos'))
                else:
                    # Cosine Mode
                    mode_c = radial * angular_cos * mask
                    mode_c /= (np.linalg.norm(mode_c) + 1e-9)
                    self.basis_functions.append(mode_c)
                    self.basis_indices.append((n, m, 'cos'))
                    
                    # Sine Mode
                    mode_s = radial * angular_sin * mask
                    mode_s /= (np.linalg.norm(mode_s) + 1e-9)
                    self.basis_functions.append(mode_s)
                    self.basis_indices.append((n, m, 'sin'))

    def step(self):
        target = self.get_blended_input('target_image', 'mean')
        trigger = self.get_blended_input('scan_trigger', 'sum')
        
        if target is None:
            return

        # --- Robust Input Handling ---
        # 1. Handle float64 -> float32
        if target.dtype == np.float64:
            target = target.astype(np.float32)
            
        # 2. Handle 3-channel -> 1-channel
        if len(target.shape) == 3 and target.shape[2] == 3:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            
        # 3. Handle ranges (0-255 -> 0-1)
        if target.max() > 1.0:
             target = target / 255.0
             
        # Resize
        if target.shape[:2] != (self.resolution, self.resolution):
            target = cv2.resize(target, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # Apply Mask
        mask = self._create_ellipsoidal_mask()
        target = target * mask
        
        # Decomposition
        coeffs = []
        reconstruction = np.zeros_like(target)
        
        for i, mode in enumerate(self.basis_functions):
            weight = np.sum(target * mode)
            coeffs.append(weight)
            reconstruction += weight * mode
            
        self.coefficients = np.array(coeffs, dtype=np.float32)
        self.reconstruction_img = np.clip(reconstruction, 0, 1)
        
        # Error Calc
        diff = target - self.reconstruction_img
        self.error_val = np.mean(diff**2)
        
        if trigger is not None and trigger > 0.5:
            self.save_dna()

    def save_dna(self):
        dna_packet = {
            "name": "Scanned Object",
            "error": float(self.error_val),
            "modes": []
        }
        for i, val in enumerate(self.coefficients):
            if abs(val) > 0.01:
                n, m, type_ = self.basis_indices[i]
                dna_packet["modes"].append({
                    "n": n, "m": m, "type": type_, "amplitude": float(val)
                })
        print(json.dumps(dna_packet, indent=2))
        
    def get_output(self, port_name):
        if port_name == 'dna_spectrum':
            # SAFE CONVERSION: Handle list or array
            return np.array(self.coefficients, dtype=np.float32)
        elif port_name == 'reconstruction':
            return self.reconstruction_img
        elif port_name == 'scan_error':
            return float(self.error_val)
        return None

    def get_display_image(self):
        img = np.zeros((self.resolution, self.resolution * 2, 3), dtype=np.uint8)
        
        # Reconstruction
        rec_u8 = (np.clip(self.reconstruction_img,0,1) * 255).astype(np.uint8)
        img[:, :self.resolution] = cv2.applyColorMap(rec_u8, cv2.COLORMAP_VIRIDIS)
        
        cv2.putText(img, f"ERR: {self.error_val:.4f}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Barcode
        if len(self.coefficients) > 0:
            roi = img[:, self.resolution:]
            roi[:] = 20
            max_val = np.max(np.abs(self.coefficients)) + 1e-9
            bar_w = max(1, self.resolution // len(self.coefficients))
            
            for i, val in enumerate(self.coefficients):
                h = int((abs(val) / max_val) * (self.resolution - 10))
                x = i * bar_w
                color = (0, 255, 0) if val > 0 else (0, 0, 255)
                cv2.rectangle(roi, (x, self.resolution), (x + bar_w - 1, self.resolution - h), color, -1)
                
        return QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Max N", "max_n", self.max_n, None),
            ("Max M", "max_m", self.max_m, None)
        ]