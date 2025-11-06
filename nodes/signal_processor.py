"""
Signal Processor Node - Applies various filters to a signal
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

class SignalProcessorNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, processing_mode='smoothing', factor=0.1):
        super().__init__()
        self.node_title = "Signal Processor"
        self.inputs = {'input_signal': 'signal'}
        self.outputs = {'output_signal': 'signal'}
        
        self.processing_mode = processing_mode
        self.factor = float(factor)
        self.last_input = 0.0
        self.integrated_state = 0.0
        self.processed_output = 0.0
        
    def step(self):
        u = self.get_blended_input('input_signal', 'sum') or 0.0
        
        output = u
        
        if self.processing_mode == 'smoothing':
            alpha = np.clip(self.factor, 0.0, 1.0) # Smoothing factor
            self.processed_output = self.processed_output * (1.0 - alpha) + u * alpha
            output = self.processed_output
            
        elif self.processing_mode == 'differentiation':
            # Factor acts as sensitivity (1/dt)
            output = (u - self.last_input) * (1.0 / max(self.factor, 1e-6)) 
            self.processed_output = output
            
        elif self.processing_mode == 'integration':
            # Factor acts as decay speed
            decay = np.clip(1.0 - self.factor * 0.1, 0.9, 1.0) 
            self.integrated_state = self.integrated_state * decay + u * 0.05
            output = self.integrated_state
            self.processed_output = output
            
        elif self.processing_mode == 'high_pass':
            # 1st order IIR high-pass. Factor is (1-alpha)
            alpha = np.clip(1.0 - self.factor, 0.01, 0.99)
            self.processed_output = alpha * (self.processed_output + u - self.last_input)
            output = self.processed_output

        elif self.processing_mode == 'full_wave_rectify':
            # Factor is unused
            output = np.abs(u)
            self.processed_output = output

        elif self.processing_mode == 'tanh_distortion':
            # Factor acts as gain/drive
            gain = max(self.factor, 1e-6)
            output = np.tanh(u * gain)
            self.processed_output = output

        self.last_input = u
        
    def get_output(self, port_name):
        if port_name == 'output_signal':
            return self.processed_output
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Simple bar display of the processed output
        v = np.clip(self.processed_output, -1.0, 1.0)
        bar_height = int((v + 1.0) / 2.0 * h)
        
        img[h - bar_height:, w//2 - 2 : w//2 + 2] = 255
        img[h//2 - 1 : h//2 + 1, :] = 80 # Center line

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Mode", "processing_mode", self.processing_mode, [
                ("Smoothing (EMA)", "smoothing"), 
                ("Differentiation", "differentiation"),
                ("Integration (Decay)", "integration"),
                ("High-Pass Filter", "high_pass"),
                ("Full Wave Rectify", "full_wave_rectify"),
                ("Tanh Distortion", "tanh_distortion")
            ]),
            ("Factor", "factor", self.factor, None)
        ]