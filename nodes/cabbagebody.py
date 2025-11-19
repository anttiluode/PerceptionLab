"""
Cabbage Body Node (Clamped & Stable)
------------------------------------
The physical simulation engine.
[FIX] Added hard clamping to prevent infinite growth.
[FIX] Added Auto-Reset if physics explodes (NaN detection).
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class CabbageBodyNode(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(0, 200, 100)

    def __init__(self):
        super().__init__()
        self.node_title = "Cabbage Body"
        
        self.inputs = {
            'lobe_activation': 'image',
            'growth_rate': 'signal'
        }
        self.outputs = {
            'structure_3d': 'image',
            'thickness': 'image'
        }
        
        self.res = 512
        self._reset_state()

    def _reset_state(self):
        self.thickness = np.ones((self.res, self.res), dtype=np.float32)
        self.height = np.zeros_like(self.thickness)
        
    def step(self):
        # 1. Safety Check: Did we explode?
        if not np.all(np.isfinite(self.thickness)):
            print("CabbageBody: Physics exploded (NaN). Resetting.")
            self._reset_state()
            
        if self.thickness.shape[0] != self.res:
            self._reset_state()
            
        act = self.get_blended_input('lobe_activation', 'mean')
        rate = self.get_blended_input('growth_rate', 'sum') or 0.005
        
        # Limit growth rate to prevent instant explosion
        rate = np.clip(rate, 0.0, 1.0)
        
        if act is None: return
        
        if act.shape[:2] != (self.res, self.res):
            act = cv2.resize(act, (self.res, self.res))
            
        # 2. Physics with Clamping
        # Growth
        self.thickness += act * rate * 0.1
        
        # HARD CLAMP: Biological tissue cannot be infinitely thick
        self.thickness = np.clip(self.thickness, 0.1, 50.0)
        
        # Folding
        pressure = np.clip(self.thickness - 2.5, 0, None)**2
        pressure = np.clip(pressure, 0, 100.0) # Clamp pressure
        
        lap = cv2.Laplacian(self.thickness, cv2.CV_32F)
        
        # Update height with damping
        self.height += -lap * pressure * 0.1
        self.height *= 0.99 # Friction/Damping (Prevents runaway vibration)
        
        # Smooth
        self.thickness = gaussian_filter(self.thickness, 0.5)
        self.height = gaussian_filter(self.height, 0.5)

    def get_output(self, port_name):
        if port_name == 'structure_3d': return self.height
        if port_name == 'thickness': return self.thickness
        return None

    def get_display_image(self):
        # Safe normalization for display
        norm = cv2.normalize(self.height, None, 0, 255, cv2.NORM_MINMAX)
        if norm is None: return QtGui.QImage()
        
        norm = norm.astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        return QtGui.QImage(color.data, self.res, self.res, self.res*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [("Resolution", "res", self.res, None)]