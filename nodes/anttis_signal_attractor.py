"""
Signal Attractor Node - Generates a 2D chaotic pattern from two signals
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

class SignalAttractorNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 80, 180) # Attractor Purple
    
    def __init__(self, width=128, height=128, param_c=1.0, param_d=0.7):
        super().__init__()
        self.node_title = "Signal Attractor"
        self.inputs = {
            'signal_a': 'signal',
            'signal_b': 'signal'
        }
        self.outputs = {'image': 'image', 'x_out': 'signal', 'y_out': 'signal'}
        
        self.w, self.h = int(width), int(height)
        
        # Attractor state
        self.x, self.y = 0.1, 0.1
        
        # Parameters (a & b are controlled by input, c & d are configurable)
        self.param_c = float(param_c)
        self.param_d = float(param_d)
        
        # For visualization
        self.points = np.zeros((self.h, self.w), dtype=np.float32)
        self.img = np.zeros((self.h, self.w), dtype=np.float32)

    def step(self):
        # Get signals, map from [-1, 1] to [-2, 2]
        param_a = (self.get_blended_input('signal_a', 'sum') or 0.0) * 2.0
        param_b = (self.get_blended_input('signal_b', 'sum') or 0.0) * 2.0
        
        # Iterate the attractor equations 500 times per frame
        for _ in range(500):
            # Clifford Attractor equations
            x_new = np.sin(param_a * self.y) + self.param_c * np.cos(param_a * self.x)
            y_new = np.sin(param_b * self.x) + self.param_d * np.cos(param_b * self.y)
            
            self.x, self.y = x_new, y_new
            
            # Scale from [-2, 2] range to image coordinates
            px = int((self.x + 2.0) / 4.0 * self.w)
            py = int((self.y + 2.0) / 4.0 * self.h)
            
            if 0 <= px < self.w and 0 <= py < self.h:
                self.points[py, px] += 0.1 # Add energy
        
        # Apply decay to the image so it fades
        self.points *= 0.97
        self.points = np.clip(self.points, 0, 1.0)
        
        # Blur for a "glowing" effect
        self.img = cv2.GaussianBlur(self.points, (3, 3), 0)
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.img
        elif port_name == 'x_out':
            return self.x / 2.0 # Normalize to [-1, 1]
        elif port_name == 'y_out':
            return self.y / 2.0 # Normalize to [-1, 1]
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Param C", "param_c", self.param_c, None),
            ("Param D", "param_d", self.param_d, None),
            ("Width", "w", self.w, None),
            ("Height", "h", self.h, None),
        ]