"""
V1 Retinotopic Transform Node
-----------------------------
Uses the actual Schwartz conformal mapping that V1 uses,
not just log-polar approximation.

w = k * log(z + a)

where z is complex visual field position, w is cortical position,
k controls magnification, a controls foveal representation.
"""

import numpy as np
import cv2

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class V1RetinotopicNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_TITLE = "V1 Retinotopic (Schwartz)"
    NODE_COLOR = QtGui.QColor(200, 100, 255)
    
    def __init__(self, k=15.0, a=0.7, foveal_bias=1.0):
        super().__init__()
        
        self.inputs = {
            'retinal_image': 'image',
            'k_mod': 'signal',      # Magnification modulation
            'a_mod': 'signal',      # Foveal size modulation
        }
        
        self.outputs = {
            'v1_cortex': 'image',
            'inverse_map': 'image',  # What V1 "sees" mapped back
        }
        
        # Schwartz parameters
        self.k = float(k)        # Cortical magnification
        self.a = float(a)        # Foveal constant (mm)
        self.foveal_bias = float(foveal_bias)
        
        self.size = 128
        self.last_v1 = None
        self.last_inverse = None
        
        # Precompute mapping
        self._build_maps()
    
    def _build_maps(self):
        """Build the Schwartz conformal mapping"""
        # Visual field coordinates (retinal)
        # Center is fovea, edges are periphery
        # Use complex number grid directly
        y, x = np.mgrid[-1:1:self.size*1j, -1:1:self.size*1j]
        
        # Complex visual field position
        z = x + 1j * y
        
        # Add small offset to avoid log(0)
        z_offset = z + self.a
        
        # Schwartz mapping: w = k * log(z + a)
        # This maps visual field to cortical coordinates
        w = self.k * np.log(np.abs(z_offset) + 1e-9) + 1j * np.angle(z_offset)
        
        # Extract real (eccentricity) and imaginary (polar angle) parts
        # These become x,y in cortical space
        cortical_x = np.real(w)
        cortical_y = np.imag(w)
        
        # --- FIX FOR NUMPY 2.0 ---
        # Replaced .ptp() with np.ptp()
        range_x = np.ptp(cortical_x) + 1e-9
        range_y = np.ptp(cortical_y) + 1e-9
        
        cortical_x = (cortical_x - cortical_x.min()) / range_x
        cortical_y = (cortical_y - cortical_y.min()) / range_y
        
        self.map_x = (cortical_x * (self.size - 1)).astype(np.float32)
        self.map_y = (cortical_y * (self.size - 1)).astype(np.float32)
        
        # Build inverse map (cortex -> visual field)
        # For visualization of "what V1 sees"
        self._build_inverse_map()
    
    def _build_inverse_map(self):
        """Build inverse mapping from cortex to visual field"""
        # Cortical coordinates
        cy, cx = np.mgrid[0:self.size, 0:self.size]
        
        # Normalize to w-space
        w_real = cx / self.size * (self.k * np.log(2 + self.a))  # Range of log mapping
        w_imag = (cy / self.size - 0.5) * 2 * np.pi  # -pi to pi
        
        w = w_real + 1j * w_imag
        
        # Inverse Schwartz: z = exp(w/k) - a
        z = np.exp(w / self.k) - self.a
        
        # Convert to image coordinates
        visual_x = (np.real(z) + 1) / 2 * (self.size - 1)
        visual_y = (np.imag(z) + 1) / 2 * (self.size - 1)
        
        self.inv_map_x = np.clip(visual_x, 0, self.size - 1).astype(np.float32)
        self.inv_map_y = np.clip(visual_y, 0, self.size - 1).astype(np.float32)
    
    def step(self):
        img = self.get_blended_input('retinal_image', 'first')
        k_mod = self.get_blended_input('k_mod', 'sum') or 0.0
        a_mod = self.get_blended_input('a_mod', 'sum') or 0.0
        
        if img is None:
            return
        
        # Rebuild maps if parameters changed significantly
        # (Optimization: Only rebuild if change is large enough to matter)
        if abs(k_mod) > 0.01 or abs(a_mod) > 0.01:
            self.k = self.k * (1 + k_mod * 0.1) # Damped modulation
            self.a = self.a * (1 + a_mod * 0.1)
            self._build_maps()

        # Ensure float32
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0
        
        # Resize if needed
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size))
        
        # Apply Schwartz mapping (visual field -> cortex)
        self.last_v1 = cv2.remap(img, self.map_x, self.map_y, 
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Apply inverse mapping (cortex -> visual field)
        self.last_inverse = cv2.remap(self.last_v1, self.inv_map_x, self.inv_map_y,
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def get_output(self, port_name):
        if port_name == 'v1_cortex':
            return self.last_v1
        elif port_name == 'inverse_map':
            return self.last_inverse
        return None
    
    def get_display_image(self):
        if self.last_v1 is None:
            return None
        
        # Side by side: V1 cortex and inverse map
        h, w = self.size, self.size * 2
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # V1 cortex (left)
        v1_vis = (np.clip(self.last_v1, 0, 1) * 255).astype(np.uint8)
        v1_color = cv2.applyColorMap(v1_vis, cv2.COLORMAP_INFERNO)
        display[:, :self.size] = v1_color
        
        # Inverse (right)
        if self.last_inverse is not None:
            inv_vis = (np.clip(self.last_inverse, 0, 1) * 255).astype(np.uint8)
            inv_color = cv2.applyColorMap(inv_vis, cv2.COLORMAP_JET)
            display[:, self.size:] = inv_color
        
        # Labels
        cv2.putText(display, "V1 Schwartz Map", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Retinal Reconstruction", (self.size + 5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Magnification (k)", "k", self.k, None),
            ("Foveal constant (a)", "a", self.a, None),
        ]