"""
Feedback / Delay Node V3 - Universal Adapter
============================================
Handles both Grayscale (2D) and Holographic (3D) inputs seamlessly.
Auto-promotes Grayscale to Color to prevent 'Broadcast Error'.
"""

import numpy as np
import cv2
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class FeedbackNode(BaseNode):
    NODE_CATEGORY = "Utility"
    NODE_TITLE = "Feedback (Universal)"
    NODE_COLOR = QtGui.QColor(120, 120, 120)
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'image_in': 'image',
            'decay': 'signal',
            'gain': 'signal'
        }
        self.outputs = {
            'image_out': 'image'
        }
        
        self.memory = None
        self.w, self.h = 64, 64

    def _ensure_color(self, img):
        """Promotes 2D Grayscale to 3D BGR if needed."""
        if img is None: return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def step(self):
        # 1. READ PARAMS
        decay = self.get_blended_input('decay', 'mean')
        if decay is None: decay = 0.05
        
        gain = self.get_blended_input('gain', 'mean')
        if gain is None: gain = 1.05
        
        # 2. READ INPUT
        raw_input = self.get_blended_input('image_in', 'mean')
        
        # Filter bad data
        if raw_input is not None:
            if isinstance(raw_input, (str, np.str_)): raw_input = None
            elif raw_input.dtype.kind in {'U', 'S'}: raw_input = None

        # 3. NORMALIZE TO COLOR (The Fix)
        # We force everything to be 3-channel BGR. 
        # This prevents the (H,W) vs (H,W,3) crash.
        current_input = self._ensure_color(raw_input)

        # 4. INITIALIZE MEMORY
        if self.memory is None:
            if current_input is not None:
                # Inherit shape from first valid input
                self.memory = np.zeros_like(current_input, dtype=np.float32)
            else:
                # Default to 64x64 Color
                self.memory = np.zeros((64, 64, 3), dtype=np.float32)

        # 5. DYNAMIC RESIZE / RESHAPE
        # If input switches dimension (e.g. from Gray to Color stream), reset memory
        if current_input is not None:
            if current_input.shape != self.memory.shape:
                # Try to resize spatial dims first
                if current_input.shape[:2] != self.memory.shape[:2]:
                    self.memory = cv2.resize(self.memory, (current_input.shape[1], current_input.shape[0]))
                
                # Check channels again
                if current_input.shape != self.memory.shape:
                    # If channels mismatch (e.g. Memory is Gray, Input is Color), promote Memory
                    self.memory = self._ensure_color(self.memory)
            
            # BLEND
            self.memory = (self.memory * (1.0 - decay)) + (current_input * 0.5)
        else:
            self.memory = self.memory * (1.0 - decay)

        # 6. GAIN & CLIP
        self.memory *= gain
        self.memory = np.clip(self.memory, 0, 10.0) # Allow HDR headroom

    def get_output(self, name):
        return self.memory

    def get_display_image(self):
        if self.memory is None: return None
        # Display safely
        disp = np.clip(self.memory, 0, 1)
        if disp.ndim == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        return (disp * 255).astype(np.uint8)