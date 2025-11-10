"""
SU2FieldNode (Weak Force Metaphor)

Simulates an SU(2) gauge force with 3 components.
Treats the input image's RGB channels as a 3D "flavor space"
and rotates this space, simulating flavor change.

[FIXED] Initialized self.field_out in __init__ to prevent AttributeError.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class SU2FieldNode(BaseNode):
    """
    Rotates the RGB "flavor" space of an image.
    """
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(220, 100, 100) # Red

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "SU(2) Field (Weak)"
        
        self.inputs = {
            'field_in': 'image',   # Color image (flavor field)
            'rot_X': 'signal',     # 'W+' (R <-> G)
            'rot_Y': 'signal',     # 'W-' (G <-> B)
            'rot_Z': 'signal'      # 'Z0' (B <-> R)
        }
        self.outputs = {'field_out': 'image'}
        
        self.size = int(size)
        self.t = 0.0 # Internal time
        
        # --- START FIX ---
        # Initialize the output variable to prevent race condition
        self.field_out = np.zeros((self.size, self.size, 3), dtype=np.float32)
        # --- END FIX ---
        
    def _prepare_image(self, img):
        if img is None:
            return np.zeros((self.size, self.size, 3), dtype=np.float32)
        
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
        
        # Get rotation angles
        angle_x = (self.get_blended_input('rot_X', 'sum') or 0.0) * 0.1
        angle_y = (self.get_blended_input('rot_Y', 'sum') or 0.0) * 0.1
        angle_z = (self.get_blended_input('rot_Z', 'sum') or 0.0) * 0.1
        
        # --- 2. Build Rotation Matrices ---
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        
        # Note: OpenCV uses BGR, so we'll treat B=X, G=Y, R=Z
        
        # Z-axis rotation (R <-> G)
        R_z = np.float32([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [ 0,   0, 1]
        ])
        
        # X-axis rotation (G <-> B)
        R_x = np.float32([
            [1,  0,   0],
            [0, cx, -sx],
            [0, sx,  cx]
        ])
        
        # Y-axis rotation (B <-> R)
        R_y = np.float32([
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0,  cy]
        ])
        
        # Combine all rotations
        R_total = R_z @ R_y @ R_x
        
        # --- 3. Apply SU(2) Flavor Rotation ---
        # Reshape image for matrix multiplication
        h, w, c = field.shape
        field_flat = field.reshape((-1, 3))
        
        # Apply transformation
        # (field_flat @ R_total.T) is the same as (R_total @ field_flat.T).T
        rotated_field_flat = field_flat @ R_total.T
        
        # Reshape back to image
        self.field_out = rotated_field_flat.reshape((h, w, 3))
        
        # Clip to maintain valid color range
        self.field_out = np.clip(self.field_out, 0.0, 1.0)

    def get_output(self, port_name):
        if port_name == 'field_out':
            return self.field_out
        return None

    def get_display_image(self):
        # self.field_out is guaranteed to exist now
        return self.field_out