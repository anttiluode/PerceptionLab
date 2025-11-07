"""
Spectrum Analyzer Node - Splits an FFT spectrum into discrete bands
Place this file in the 'nodes/ folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class SpectrumAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, low_split=0.1, high_split=0.5):
        super().__init__()
        self.node_title = "Spectrum Analyzer"
        
        self.inputs = {'spectrum_in': 'spectrum'}
        self.outputs = {
            'bass': 'signal',
            'mids': 'signal',
            'high': 'signal'
        }
        
        self.low_split = float(low_split)  # 10% mark
        self.high_split = float(high_split) # 50% mark
        
        self.bass = 0.0
        self.mids = 0.0
        self.high = 0.0
        
        self.vis_img = np.zeros((64, 64, 3), dtype=np.uint8)

    def step(self):
        # get_blended_input will use 'mean' for array types like 'spectrum'
        spectrum = self.get_blended_input('spectrum_in', 'mean') 
        
        if spectrum is None or len(spectrum) == 0:
            self.bass *= 0.9
            self.mids *= 0.9
            self.high *= 0.9
            return
            
        spec_len = len(spectrum)
        low_idx = int(spec_len * self.low_split)
        high_idx = int(spec_len * self.high_split)
        
        # Calculate mean power in each band
        self.bass = np.mean(spectrum[0 : low_idx])
        self.mids = np.mean(spectrum[low_idx : high_idx])
        self.high = np.mean(spectrum[high_idx :])
        
        # Normalize (signals are often very small)
        total = self.bass + self.mids + self.high + 1e-9
        self.bass /= total
        self.mids /= total
        self.high /= total
        
        # Update visualization
        self.vis_img.fill(0)
        cv2.rectangle(self.vis_img, (0, 63 - int(self.bass * 63)), (20, 63), (0, 0, 255), -1)
        cv2.rectangle(self.vis_img, (22, 63 - int(self.mids * 63)), (42, 63), (0, 255, 0), -1)
        cv2.rectangle(self.vis_img, (44, 63 - int(self.high * 63)), (63, 63), (255, 0, 0), -1)

    def get_output(self, port_name):
        if port_name == 'bass':
            return self.bass
        elif port_name == 'mids':
            return self.mids
        elif port_name == 'high':
            return self.high
        return None

    # --- THIS IS THE FIX ---
    def get_display_image(self):
    # --- END FIX ---
        img = np.ascontiguousarray(self.vis_img)
        return QtGui.QImage(img.data, 64, 64, 64*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Bass/Mid Split (0-1)", "low_split", self.low_split, None),
            ("Mid/High Split (0-1)", "high_split", self.high_split, None),
        ]
