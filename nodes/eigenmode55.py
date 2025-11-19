# eigenmode55node.py
"""
Eigenmode55Node - Direct 55D Address to Spatial Pattern Mapping.
Feeds the full Observer's perception directly into morphogenesis.
"""

import numpy as np
import cv2
from scipy.special import jn, jn_zeros
from scipy.ndimage import gaussian_filter
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class Eigenmode55Node(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(80, 60, 140) 

    def __init__(self, resolution=256, max_n=5, max_m=5):
        super().__init__()
        self.node_title = "Eigenmode 55 (Neural Modes)"
        
        self.inputs = {'dna_55': 'spectrum'} 
        
        self.outputs = {
            'lobe_activation_map': 'image', 
            'dominant_mode_power': 'signal',
            'dominant_mode_n': 'signal' # Output declaration
        }
        
        self.resolution = int(resolution)
        self.max_n = int(max_n)
        self.max_m = int(max_m)
        self.num_modes = 55 

        self.basis_functions = []
        self.basis_indices = []
        self._precompute_basis()
        
        self.lobe_activation_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # === THE FIX IS HERE ===
        self.dominant_mode_power = 0.0
        self.dominant_mode_n = 0.0 # Initialized to 0.0
        # =======================

    def _precompute_basis(self):
        h, w = self.resolution, self.resolution
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        x_norm = (x - cx) / (w / 2)
        y_norm = (y - cy) / (h / 2)
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-9
        theta = np.arctan2(y_norm, x_norm)
        mask = (r <= 1.0).astype(np.float32)

        for n in range(1, self.max_n + 1):
            for m in range(0, self.max_m + 1):
                try:
                    zeros = jn_zeros(m, n)
                    k = zeros[-1]
                except ValueError:
                    continue 

                radial = jn(m, k * r)
                
                if m == 0:
                    mode = radial * mask
                    mode /= (np.linalg.norm(mode) + 1e-9)
                    self.basis_functions.append(mode)
                    self.basis_indices.append((n, m, 'cos'))
                else:
                    mode_c = radial * np.cos(m * theta) * mask
                    mode_c /= (np.linalg.norm(mode_c) + 1e-9)
                    self.basis_functions.append(mode_c)
                    self.basis_indices.append((n, m, 'cos'))
                    
                    mode_s = radial * np.sin(m * theta) * mask
                    mode_s /= (np.linalg.norm(mode_s) + 1e-9)
                    self.basis_functions.append(mode_s)
                    self.basis_indices.append((n, m, 'sin'))

    def step(self):
        coeffs = self.get_blended_input('dna_55', 'first')
        
        if coeffs is None:
            self.lobe_activation_map *= 0.95
            return

        if isinstance(coeffs, list):
            coeffs = np.array(coeffs, dtype=np.float32)
        
        if len(coeffs) > self.num_modes:
            coeffs = coeffs[:self.num_modes]
        
        new_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        total_power = 0.0
        max_power = 0.0
        self.dominant_mode_n = 0.0 # Reset dominant mode for the frame
        
        for i in range(min(len(coeffs), len(self.basis_functions))):
            weight = coeffs[i] 
            mode = self.basis_functions[i]
            
            new_map += weight * mode
            total_power += weight ** 2
            
            if (weight ** 2) > max_power:
                 max_power = weight ** 2
                 self.dominant_mode_n = self.basis_indices[i][0]
        
        map_min, map_max = new_map.min(), new_map.max()
        range_val = map_max - map_min
        
        if range_val > 1e-9:
             new_map = (new_map - map_min) / range_val 

        self.lobe_activation_map = np.clip(np.tanh(new_map * 5.0), 0, 1)
        self.lobe_activation_map = gaussian_filter(self.lobe_activation_map, sigma=1.0)
        
        self.dominant_mode_power = float(np.sqrt(max_power))
        self.dominant_mode_n = float(self.dominant_mode_n)

    def get_output(self, port_name):
        if port_name == 'lobe_activation_map':
            return self.lobe_activation_map
        if port_name == 'dominant_mode_power':
            return self.dominant_mode_power
        if port_name == 'dominant_mode_n':
            return self.dominant_mode_n
        return None

    def get_display_image(self):
        img_u8 = (np.clip(self.lobe_activation_map, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        cv2.putText(img_color, f"Power: {self.dominant_mode_power:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_color, f"Mode N: {self.dominant_mode_n:.0f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(img_color.data, self.resolution, self.resolution, self.resolution * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
        ]