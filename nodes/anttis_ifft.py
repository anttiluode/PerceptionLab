"""
iFFT Cochlea Node - Reconstructs an image from a complex spectrum.
Based on the hardwired iFFTCochleaNode from anttis_perception_laboratory.py
Requires: pip install scipy
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

# --- !! CRITICAL IMPORT BLOCK !! ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------------

try:
    from scipy.fft import irfft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: iFFTCochleaNode requires 'scipy'.")
    print("Please run: pip install scipy")


class iFFTCochleaNode(BaseNode):
    """
    Performs an Inverse Real FFT on a complex spectrum (from FFTCochleaNode)
    to reconstruct a 2D image.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(200, 100, 60)
    
    def __init__(self, height=120, width=160):
        super().__init__()
        self.node_title = "iFFT Cochlea"
        self.inputs = {'complex_spectrum': 'complex_spectrum'}
        self.outputs = {'image': 'image'}
        
        self.h, self.w = height, width
        self.reconstructed_img = np.zeros((self.h, self.w), dtype=np.float32)

    def step(self):
        if not SCIPY_AVAILABLE:
            return

        complex_spec = self.get_blended_input('complex_spectrum', 'mean')
        
        if complex_spec is not None and complex_spec.ndim == 2:
            try:
                # Perform inverse real FFT
                img = irfft(complex_spec, axis=1).astype(np.float32)
                
                # Resize to target output size (just in case)
                self.reconstructed_img = cv2.resize(img, (self.w, self.h))
                
                # Normalize for viewing (0-1)
                min_v, max_v = np.min(self.reconstructed_img), np.max(self.reconstructed_img)
                if (max_v - min_v) > 1e-6:
                    self.reconstructed_img = (self.reconstructed_img - min_v) / (max_v - min_v)
                else:
                    self.reconstructed_img.fill(0.5)
                    
            except Exception as e:
                print(f"iFFT Error: {e}")
                self.reconstructed_img.fill(0.0)
        else:
            # Fade to black if no input
            self.reconstructed_img *= 0.9 
            
    def get_output(self, port_name):
        if port_name == 'image':
            return self.reconstructed_img
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.reconstructed_img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Height", "height", self.h, None),
            ("Width", "width", self.w, None)
        ]