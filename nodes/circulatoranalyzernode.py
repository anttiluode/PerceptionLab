"""
CirculationAnalyzerNode

Analyzes the "total circulation cost" of a vector field
by calculating its 2D curl (vorticity).
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class CirculationAnalyzerNode(BaseNode):
    """
    Calculates the 2D curl (vorticity) of an input vector field.
    """
    NODE_CATEGORY = "Analyzer"
    NODE_COLOR = QtGui.QColor(220, 100, 100) # Red

    def __init__(self, size=64):
        super().__init__()
        self.node_title = "Circulation Analyzer"
        
        self.inputs = {
            'vector_field_in': 'image' # From CirculationFieldNode
        }
        self.outputs = {
            'total_circulation': 'signal', # "Total circulation cost"
            'vorticity_map': 'image'       # Visualization of curl
        }
        
        self.size = int(size)
        
        # Internal state
        self.total_circulation = 0.0
        self.vorticity_map = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def step(self):
        # --- 1. Get and Prepare Field ---
        field = self.get_blended_input('vector_field_in', 'first')
        if field is None:
            return

        # Ensure float32
        if field.dtype != np.float32:
            field = field.astype(np.float32)
        if field.max() > 1.0: # (Assumes 0-255 if not 0-1)
            field = field / 255.0
            
        field_resized = cv2.resize(field, (self.size, self.size), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Convert from [0, 1] (R,G) to [-1, 1] (vx, vy)
        vx = (field_resized[..., 0] * 2.0) - 1.0
        vy = (field_resized[..., 1] * 2.0) - 1.0
        
        # --- 2. Calculate Vorticity (Curl) ---
        # curl(F) = (dVy/dx - dVx/dy)
        
        # Must use CV_32F to handle negative numbers
        dvx_dy = cv2.Sobel(vx, cv2.CV_32F, 0, 1, ksize=3)
        dvy_dx = cv2.Sobel(vy, cv2.CV_32F, 1, 0, ksize=3)
        
        curl = dvy_dx - dvx_dy
        
        # --- 3. Calculate Outputs ---
        
        # "Total circulation cost" = average absolute vorticity
        self.total_circulation = np.mean(np.abs(curl))
        
        # --- 4. Create Visualization ---
        # Normalize curl from [-max, +max] to [0, 1]
        max_curl = np.max(np.abs(curl))
        if max_curl == 0:
            norm_curl = np.zeros((self.size, self.size), dtype=np.float32)
        else:
            norm_curl = (curl + max_curl) / (2 * max_curl)
        
        img_u8 = (norm_curl * 255).astype(np.uint8)
        self.vorticity_map = cv2.applyColorMap(img_u8, cv2.COLORMAP_BONE)
        self.vorticity_map = self.vorticity_map.astype(np.float32) / 255.0

    def get_output(self, port_name):
        if port_name == 'total_circulation':
            return self.total_circulation
        elif port_name == 'vorticity_map':
            return self.vorticity_map
        return None

    def get_display_image(self):
        return self.vorticity_map