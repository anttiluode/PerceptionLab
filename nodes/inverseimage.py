"""
Image Invert & Blend Node
=========================
Takes an image, creates its negative (inverse), and blends between
the original and the inverse based on a percentage signal.

Inputs:
    - image_in: The source image.
    - blend_pct: 0.0 = Original, 1.0 (or 100) = Fully Inverted, 0.5 = 50% Gray mix.
"""

import cv2
import numpy as np
from PyQt6 import QtGui

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs={}; self.outputs={}
            self._output_values = {}
        def get_blended_input(self, name, mode): return None
        def set_output(self, name, val): pass

class ImageInvertBlendNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_TITLE = "Invert Blend"
    NODE_COLOR = QtGui.QColor(0, 180, 200) # Cyan/Teal
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',
            'blend_pct': 'signal' # Range 0.0 to 1.0 (e.g. 0.5 for 50%)
        }
        
        self.outputs = {
            'image_out': 'image'
        }
        
        # State
        self.cached_image = None
        self._output_values = {}
        self.display_text = "WAITING"
        
        # Standby Texture
        self.empty_img = np.zeros((128, 128, 3), dtype=np.uint8) + 50 # Dark gray

    def step(self):
        # 1. Get Inputs
        img = self.get_blended_input('image_in', 'mean')
        pct_in = self.get_blended_input('blend_pct', 'mean')
        
        if img is None or not isinstance(img, np.ndarray):
            self.cached_image = self.empty_img
            self.display_text = "NO IMAGE"
            self.set_output('image_out', None)
            return

        # Handle percentage input (allow 0-1 or 0-100 ranges, clamp to 0-1)
        if pct_in is None: pct_in = 0.0
        pct = float(pct_in)
        if pct > 1.0 and pct <= 100.0:
            pct /= 100.0
        pct = np.clip(pct, 0.0, 1.0)

        # 2. Process
        # Ensure work image is standard uint8 for reliable math
        work_img = img.copy()
        is_float = False
        if work_img.dtype != np.uint8:
            # If it's float 0-1 range
            if work_img.max() <= 1.05:
                is_float = True
            else:
                # Just clamp and convert
                work_img = np.clip(work_img, 0, 255).astype(np.uint8)

        if is_float:
            # FLOAT PATH (0.0 - 1.0)
            inverted = 1.0 - work_img
            # Linear Interpolation: A*(1-t) + B*t
            blended = work_img * (1.0 - pct) + inverted * pct
            self.cached_image = blended
        else:
            # UINT8 PATH (0 - 255)
            inverted = 255 - work_img
            # cv2.addWeighted is fast and handles clipping automatically
            # blended = img * alpha + inverted * beta + gamma
            blended = cv2.addWeighted(work_img, 1.0 - pct, inverted, pct, 0)
            self.cached_image = blended

        # Update display text percentage
        self.display_text = f"BLEND: {int(pct*100)}%"

        # 3. Output
        self.set_output('image_out', self.cached_image)

    # --- BOILERPLATE ---
    def get_output(self, name):
        return self._output_values.get(name)
    def set_output(self, name, val):
        self._output_values[name] = val
        
    def get_display_image(self):
        if self.cached_image is None: return None
        
        disp = self.cached_image.copy()
        
        # 1. Normalize for display if float
        if disp.dtype != np.uint8:
            disp = np.clip(disp * 255, 0, 255).astype(np.uint8)

        # 2. Resize for thumbnail if huge
        h, w = disp.shape[:2]
        if w > 256:
            disp = cv2.resize(disp, (256, int(h*(256/w))))
            
        # 3. Color Fix (ensure RGB)
        if len(disp.shape) == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        elif disp.shape[2] == 3:
            # Assuming input is BGR (standard opencv), convert to RGB for Qt
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        elif disp.shape[2] == 4:
            disp = cv2.cvtColor(disp, cv2.COLOR_BGRA2RGB)
            
        # 4. Text Overlay
        cv2.putText(disp, self.display_text, (5, disp.shape[0]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if self.display_text != "NO IMAGE" else (100,100,100), 1)

        # 5. QImage Construction
        disp = np.ascontiguousarray(disp)
        h, w, c = disp.shape
        qimg = QtGui.QImage(disp.data, w, h, w*c, QtGui.QImage.Format.Format_RGB888).copy()
        return qimg