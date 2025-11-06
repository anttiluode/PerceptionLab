"""
Strange Attractor Node - Generates chaotic 2D patterns
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

class StrangeAttractorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 140, 100) # A generative green
    
    def __init__(self, width=160, height=120):
        super().__init__()
        self.node_title = "Strange Attractor"
        self.inputs = {
            'param_a': 'signal',
            'param_b': 'signal',
            'param_c': 'signal',
            'param_d': 'signal'
        }
        self.outputs = {'image': 'image', 'x_signal': 'signal', 'y_signal': 'signal'}
        
        self.w, self.h = width, height
        self.img = np.zeros((self.h, self.w), dtype=np.float32)
        
        # Attractor state
        self.x, self.y = 0.1, 0.1
        
        # Default parameters for a "Clifford" attractor
        self.a = -1.4
        self.b = 1.6
        self.c = 1.0
        self.d = 0.7
        
        # For visualization
        self.points = np.zeros((self.h, self.w), dtype=np.float32)

    def step(self):
        # Update parameters from inputs, or use internal values
        self.a = self.get_blended_input('param_a', 'sum') or self.a
        self.b = self.get_blended_input('param_b', 'sum') or self.b
        self.c = self.get_blended_input('param_c', 'sum') or self.c
        self.d = self.get_blended_input('param_d', 'sum') or self.d
        
        # Iterate the attractor equations 500 times per frame for a dense plot
        for _ in range(500):
            # Clifford Attractor equations
            x_new = np.sin(self.a * self.y) + self.c * np.cos(self.a * self.x)
            y_new = np.sin(self.b * self.x) + self.d * np.cos(self.b * self.y)
            
            self.x, self.y = x_new, y_new
            
            # Scale from [-2, 2] range to image coordinates [0, w] and [0, h]
            px = int((self.x + 2.0) / 4.0 * self.w)
            py = int((self.y + 2.0) / 4.0 * self.h)
            
            # Plot the point
            if 0 <= px < self.w and 0 <= py < self.h:
                self.points[py, px] += 0.1 # Add energy to this pixel
        
        # Apply decay to the image so it fades
        self.points *= 0.98
        self.points = np.clip(self.points, 0, 1.0)
        
        # Blur the image slightly for a "glowing" effect
        self.img = cv2.GaussianBlur(self.points, (3, 3), 0)
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.img
        elif port_name == 'x_signal':
            return self.x / 2.0 # Normalize to [-1, 1]
        elif port_name == 'y_signal':
            return self.y / 2.0 # Normalize to [-1, 1]
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Param A", "a", self.a, None),
            ("Param B", "b", self.b, None),
            ("Param C", "c", self.c, None),
            ("Param D", "d", self.d, None),
        ]

    def randomize(self):
        # Add a randomize button
        self.a = np.random.uniform(-2.0, 2.0)
        self.b = np.random.uniform(-2.0, 2.0)
        self.c = np