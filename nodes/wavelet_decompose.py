"""
Wavelet Decompose Node - Decomposes an image into DWT sub-bands (LL, LH, HL, HH)
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: WaveletDecomposeNode requires 'PyWavelets'.")
    print("Please run: pip install PyWavelets")

class WaveletDecomposeNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, wavelet_name='haar', size=128):
        super().__init__()
        self.node_title = "Wavelet Decompose (DWT)"
        
        self.inputs = {'image': 'image'}
        self.outputs = {
            'LL': 'image', # Approximation
            'LH': 'image', # Horizontal Detail
            'HL': 'image', # Vertical Detail
            'HH': 'image'  # Diagonal Detail
        }
        
        self.wavelet_name = str(wavelet_name)
        self.size = int(size)
        
        # Internal storage for output components
        h, w = self.size // 2, self.size // 2
        self.ll_out = np.zeros((h, w), dtype=np.float32)
        self.lh_out = np.zeros((h, w), dtype=np.float32)
        self.hl_out = np.zeros((h, w), dtype=np.float32)
        self.hh_out = np.zeros((h, w), dtype=np.float32)
        
        self.display_tiled = np.zeros((self.size, self.size), dtype=np.float32)
        
        if not PYWT_AVAILABLE:
            self.node_title = "DWT (No PyWT!)"

    def _normalize(self, arr):
        """Normalize an array to [0, 1] for visualization."""
        arr_min, arr_max = arr.min(), arr.max()
        if (arr_max - arr_min) > 1e-9:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr - arr_min # Return zero array

    def step(self):
        if not PYWT_AVAILABLE:
            return

        input_img = self.get_blended_input('image', 'mean')
        
        if input_img is None:
            # Fade all outputs
            self.ll_out *= 0.95
            self.lh_out *= 0.95
            self.hl_out *= 0.95
            self.hh_out *= 0.95
            return
            
        try:
            # Resize image to a square power-of-2-like size
            img_resized = cv2.resize(input_img, (self.size, self.size), 
                                     interpolation=cv2.INTER_AREA)
            
            # Perform 2D Discrete Wavelet Transform
            coeffs = pywt.dwt2(img_resized, self.wavelet_name)
            LL, (LH, HL, HH) = coeffs
            
            # Store normalized components for output
            self.ll_out = self._normalize(LL)
            self.lh_out = self._normalize(LH)
            self.hl_out = self._normalize(HL)
            self.hh_out = self._normalize(HH)
            
        except Exception as e:
            print(f"DWT Error: {e}")

    def get_output(self, port_name):
        if port_name == 'LL':
            return self.ll_out
        elif port_name == 'LH':
            return self.lh_out
        elif port_name == 'HL':
            return self.hl_out
        elif port_name == 'HH':
            return self.hh_out
        return None
        
    def get_display_image(self):
        # Create a tiled image for the node's display
        h, w = self.size // 2, self.size // 2
        
        self.display_tiled[:h, :w] = self.ll_out # Top-Left
        self.display_tiled[:h, w:] = self.lh_out # Top-Right
        self.display_tiled[h:, :w] = self.hl_out # Bottom-Left
        self.display_tiled[h:, w:] = self.hh_out # Bottom-Right
        
        img_u8 = (np.clip(self.display_tiled, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.size, self.size, self.size, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        # Get common wavelets
        wavelet_options = [
            ("Haar ('haar')", "haar"),
            ("Daubechies 1 ('db1')", "db1"),
            ("Daubechies 4 ('db4')", "db4"),
            ("Symlet 2 ('sym2')", "sym2"),
            ("Coiflet 1 ('coif1')", "coif1"),
        ]
        
        return [
            ("Wavelet", "wavelet_name", self.wavelet_name, wavelet_options),
            ("Resolution", "size", self.size, None),
        ]