"""
Scalogram Analyzer Node - Computes a CWT scalogram from an image's center slice
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
    print("Warning: ScalogramAnalyzerNode requires 'PyWavelets'.")
    print("Please run: pip install PyWavelets")

class ScalogramAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(60, 180, 160) # A teal/aqua color
    
    def __init__(self, num_scales=64, wavelet_name='morl'):
        super().__init__()
        self.node_title = "Scalogram Analyzer"
        
        self.inputs = {'image': 'image'}
        self.outputs = {'image': 'image'}
        
        self.num_scales = int(num_scales)
        self.wavelet_name = str(wavelet_name)
        
        self.output_image = np.zeros((self.num_scales, 128), dtype=np.float32)
        
        if not PYWT_AVAILABLE:
            self.node_title = "Scalogram (No PyWT!)"

    def step(self):
        if not PYWT_AVAILABLE:
            return

        input_img = self.get_blended_input('image', 'mean')
        
        if input_img is None:
            self.output_image *= 0.95 # Fade to black
            return
            
        try:
            # Extract the middle row as a 1D signal
            h, w = input_img.shape
            signal_1d = input_img[h // 2, :]
            
            # Define the scales to analyze
            # We use a logarithmic space for scales, which is common
            scales = np.geomspace(1, w / 2, self.num_scales)
            
            # Compute the Continuous Wavelet Transform (CWT)
            cfs, freqs = pywt.cwt(signal_1d, scales, self.wavelet_name)
            
            # The result is the scalogram (magnitude of coefficients)
            scalogram = np.abs(cfs)
            
            # Normalize for visualization
            s_min, s_max = scalogram.min(), scalogram.max()
            if (s_max - s_min) > 1e-9:
                scalogram = (scalogram - s_min) / (s_max - s_min)
                
            # Resize to fit a standard display aspect
            self.output_image = cv2.resize(scalogram, (w, self.num_scales),
                                           interpolation=cv2.INTER_LINEAR)
                                           
        except Exception as e:
            print(f"Scalogram Error: {e}")
            self.output_image *= 0.95

    def get_output(self, port_name):
        if port_name == 'image':
            return self.output_image
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.output_image, 0, 1) * 255).astype(np.uint8)
        
        # Apply a colormap for better visibility
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        # Common wavelets for CWT
        wavelet_options = [
            ("Morlet ('morl')", "morl"),
            ("Mexican Hat ('mexh')", "mexh"),
            ("Gaussian 1 ('gaus1')", "gaus1"),
            ("Complex Morlet ('cmor1.5-1.0')", "cmor1.5-1.0")
        ]
        
        return [
            ("Wavelet", "wavelet_name", self.wavelet_name, wavelet_options),
            ("Number of Scales", "num_scales", self.num_scales, None),
        ]