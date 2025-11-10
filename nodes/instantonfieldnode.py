"""
InstantonFieldNode

Simulates a continuous field dynamic based on the "action integral"
S[φ] = ∫ d⁴x [½(∂μφ)² + V(φ)]. It accumulates a field 'φ' based
on an input potential 'V(φ)' and a beta-field catalyst.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class InstantonFieldNode(BaseNode):
    """
    Generates 'instantons' by accumulating a field in a potential.
    """
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 200, 250) # Sky Blue

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Instanton Field"
        
        self.inputs = {
            'potential_in': 'image', # V(φ) - The landscape
            'beta_field': 'image',   # β-parameter field (catalyst)
            'diffusion': 'signal',   # (∂μφ)² - Smoothing/kinetic term
            'decay': 'signal'        # 0-1, how fast the field fades
        }
        self.outputs = {
            'field_out': 'image',      # The raw, continuous field φ
            'instanton_viz': 'image'   # Thresholded "instantons"
        }
        
        self.size = int(size)
        
        # The field φ, initialized as float32 for safety
        self.field = np.zeros((self.size, self.size), dtype=np.float32)

    def _prepare_image(self, img, default_val=0.0):
        """Helper to resize, format, and handle missing images."""
        if img is None:
            return np.full((self.size, self.size), default_val, dtype=np.float32)
        
        # Ensure float32 in 0-1 range
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
            
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        if img_resized.ndim == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_resized
            
        return np.clip(img_gray, 0, 1)

    def step(self):
        # --- 1. Get Inputs ---
        
        # [FIX 1: Logic Error] Use 'potential_in', not 'image_in'
        potential = 1.0 - self._prepare_image(
            self.get_blended_input('potential_in', 'first'), 
            default_val=1.0
        )
        
        beta_field = self._prepare_image(
            self.get_blended_input('beta_field', 'first'), 
            default_val=1.0
        )
        
        # Get standard Python floats (which are 64-bit)
        diffusion = self.get_blended_input('diffusion', 'sum') or 0.1
        decay = self.get_blended_input('decay', 'sum') or 0.05
        
        # --- 2. Simulate the Field ---
        
        # [FIX 2: Crash Fix]
        # Force self.field to be float32 *before* passing to OpenCV.
        # This fixes the crash if it was upcast to float64 on the previous frame.
        laplacian = cv2.Laplacian(self.field.astype(np.float32), cv2.CV_32F, ksize=3)
        
        # S[φ] = ∫ d⁴x [½(∂μφ)² + V(φ)]
        # All math here will be upcast to float64, which is fine
        new_field = (self.field * (1.0 - np.clip(decay, 0, 1))) + \
                     (laplacian * np.clip(diffusion, 0, 1)) + \
                     (potential * beta_field * 0.1) # 0.1 is a 'learning rate'
                     
        # Clamp to prevent runaway values
        new_field = np.clip(new_field, 0, 1)
        
        # [FIX 3: Prevent Future Crashes]
        # Store the result as float32, so it's correct for the *next* frame
        self.field = new_field.astype(np.float32)

    def get_output(self, port_name):
        if port_name == 'field_out':
            return self.field # Return the raw 0-1 float field
            
        elif port_name == 'instanton_viz':
            # Threshold the field to see the "instantons"
            _ , binary = cv2.threshold(self.field, 0.5, 1.0, cv2.THRESH_BINARY)
            
            # Apply colormap to make it look cool
            img_u8 = (binary * 255).astype(np.uint8)
            img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
            return img_color.astype(np.float32) / 255.0
            
        return None

    def get_display_image(self):
        # By default, display the 'instanton_viz' output
        return self.get_output('instanton_viz')