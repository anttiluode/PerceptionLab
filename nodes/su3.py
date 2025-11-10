"""
SU3FieldNode (Strong Force Metaphor)

Simulates an SU(3) "color" force with confinement.
Pure colors (R, G, B) are "far" from neutral gray and are
"pulled" back strongly, creating a vibrating/jiggling effect.

[FIXED] Initialized self.field_out in __init__ to prevent AttributeError.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class SU3FieldNode(BaseNode):
    """
    Simulates "color confinement" by pulling colors to neutral.
    """
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(100, 220, 100) # Green

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "SU(3) Field (Strong)"
        
        self.inputs = {
            'field_in': 'image',           # Color charge field
            'confinement': 'signal'      # 0-1, strength of confinement
        }
        self.outputs = {'field_out': 'image'}
        
        self.size = int(size)
        
        # Internal buffer for jiggling
        self.dx = np.zeros((self.size, self.size), dtype=np.float32)
        self.dy = np.zeros((self.size, self.size), dtype=np.float32)
        
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.grid_x = x.astype(np.float32)
        self.grid_y = y.astype(np.float32)
        
        # --- START FIX ---
        # Initialize the output variable to prevent race condition
        self.field_out = np.zeros((self.size, self.size, 3), dtype=np.float32)
        # --- END FIX ---
        
    def _prepare_image(self, img):
        if img is None:
            return np.full((self.size, self.size, 3), 0.5, dtype=np.float32)
        
        if img.dtype != np.float32: img = img.astype(np.float32)
        if img.max() > 1.0: img /= 255.0
            
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        if img_resized.ndim == 2:
            return cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        return img_resized

    def step(self):
        # --- 1. Get Inputs ---
        field = self._prepare_image(self.get_blended_input('field_in', 'first'))
        confinement = (self.get_blended_input('confinement', 'sum') or 0.2) * 20.0
        
        # --- 2. Calculate "Color Purity" ---
        # Find the mean color (neutral gray point)
        mean_color = np.mean(field, axis=(0, 1))
        
        # Find distance from neutral for each pixel
        # This is our "confinement force" map
        color_diff = field - mean_color
        force_map = np.linalg.norm(color_diff, axis=2) # (H, W)
        
        # --- 3. Simulate "Gluon Jiggle" ---
        # Apply force to a simple oscillator (our displacement map)
        self.dx = self.dx * 0.9 + (np.random.randn(self.size, self.size) * force_map * confinement)
        self.dy = self.dy * 0.9 + (np.random.randn(self.size, self.size) * force_map * confinement)
        
        # Clamp displacement
        self.dx = np.clip(self.dx, -10.0, 10.0)
        self.dy = np.clip(self.dy, -10.0, 10.0)
        
        # --- 4. Apply Confinement Warp ---
        map_x = (self.grid_x + self.dx).astype(np.float32)
        map_y = (self.grid_y + self.dy).astype(np.float32)
        
        self.field_out = cv2.remap(
            field, map_x, map_y, 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # --- 5. Apply "Color Rotation" (Gluon Exchange) ---
        # We also slowly pull the colors toward the mean
        self.field_out = self.field_out * 0.99 + mean_color * 0.01
        self.field_out = np.clip(self.field_out, 0, 1) # Add clip for safety

    def get_output(self, port_name):
        if port_name == 'field_out':
            return self.field_out
        return None

    def get_display_image(self):
        # self.field_out is guaranteed to exist now
        return self.field_out
