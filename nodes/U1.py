"""
U1FieldNode (Electromagnetism Metaphor)

Simulates a U(1) gauge force, like electromagnetism.
It takes a grayscale "charge density" map and calculates
the resulting force field (like an E-field).

[FIXED] Initialized self.potential in __init__ and
saved potential to self.potential in step().
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class U1FieldNode(BaseNode):
    """
    Generates a U(1) force field from a charge density map.
    """
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 150, 220) # Blue

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "U(1) Field (E/M)"
        
        self.inputs = {
            'charge_in': 'image',    # Grayscale image (0-1)
            'strength': 'signal'     # 0-1, force strength
        }
        self.outputs = {
            'potential_out': 'image', # The scalar potential (blurred charge)
            'field_viz': 'image'      # Vector field visualization
        }
        
        self.size = int(size)
        
        # --- START FIX ---
        # Initialize the output variables to prevent AttributeError
        self.viz = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.potential = np.zeros((self.size, self.size), dtype=np.float32)
        # --- END FIX ---

    def _prepare_image(self, img):
        if img is None:
            return np.full((self.size, self.size), 0.5, dtype=np.float32)
        
        if img.dtype != np.float32: img = img.astype(np.float32)
        if img.max() > 1.0: img /= 255.0
            
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        if img_resized.ndim == 3:
            return cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        return img_resized

    def step(self):
        # --- 1. Get Charge Density ---
        # Map input [0, 1] to charge [-1, 1]
        charge_density = (self._prepare_image(
            self.get_blended_input('charge_in', 'first')
        ) * 2.0) - 1.0
        
        strength = self.get_blended_input('strength', 'sum') or 1.0
        
        # --- 2. Calculate Potential ---
        # Simulate long-range 1/r potential by blurring
        # A large blur kernel simulates the 1/r falloff
        ksize = self.size // 4 * 2 + 1 # Must be odd
        
        self.potential = cv2.GaussianBlur(charge_density, (ksize, ksize), 0)
        
        # --- 3. Calculate Force Field (E-Field) ---
        # E = -∇V (Force is the negative gradient of potential)
        grad_x = -cv2.Sobel(self.potential, cv2.CV_32F, 1, 0, ksize=3) * strength
        grad_y = -cv2.Sobel(self.potential, cv2.CV_32F, 0, 1, ksize=3) * strength
        
        # --- 4. Create Visualization ---
        self.viz = np.zeros((self.size, self.size, 3), dtype=np.float32)
        step = 8 # Draw an arrow every 8 pixels
        for y in range(0, self.size, step):
            for x in range(0, self.size, step):
                vx = grad_x[y, x] * 20 # Scale for viz
                vy = grad_y[y, x] * 20
                
                pt1 = (x, y)
                pt2 = (int(np.clip(x + vx, 0, self.size-1)), 
                       int(np.clip(y + vy, 0, self.size-1)))
                
                # Color based on direction
                angle = np.arctan2(vy, vx) + np.pi
                hue = int(angle / (2 * np.pi) * 179) # 0-179 for OpenCV HSV
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
                color_float = color_rgb.astype(np.float32) / 255.0
                
                # --- START FIX ---
                # Convert numpy.float32 to standard Python floats for OpenCV
                color_tuple = (float(color_float[0]), float(color_float[1]), float(color_float[2]))
                cv2.arrowedLine(self.viz, pt1, pt2, color_tuple, 1, cv2.LINE_AA)
                # --- END FIX ---

    def get_output(self, port_name):
        if port_name == 'potential_out':
            # Normalize potential [-max, +max] to [0, 1]
            p_max = np.max(np.abs(self.potential))
            if p_max == 0: return np.full((self.size, self.size), 0.5, dtype=np.float32)
            return (self.potential / (2 * p_max)) + 0.5
            
        elif port_name == 'field_viz':
            return self.viz
        return None

    def get_display_image(self):
        return self.viz