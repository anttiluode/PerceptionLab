"""
Display Nodes - Image viewer and signal plotter
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

class ImageDisplayNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120) # Output Purple
    
    def __init__(self, width=160, height=120):
        super().__init__()
        self.node_title = "Image Display"
        self.inputs = {'image': 'image'}
        self.w, self.h = width, height
        self.img = np.zeros((self.h, self.w), dtype=np.float32)
        
    def step(self):
        img = self.get_blended_input('image', 'first')
        if img is not None:
            if img.shape != (self.h, self.w):
                # Use cv2.resize for robustness
                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            self.img = img
        else:
            self.img *= 0.95 # Fade to black
            
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

class SignalMonitorNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120) # Output Purple
    
    def __init__(self, history_len=500):
        super().__init__()
        self.node_title = "Signal Monitor"
        self.inputs = {'signal': 'signal'}
        self.history = deque(maxlen=history_len)
        self.history_len = history_len
        
    def step(self):
        val = self.get_blended_input('signal', 'sum') or 0.0
        
        # Handle potential arrays from mean blending
        if isinstance(val, np.ndarray):
            val = val.mean()
            
        self.history.append(float(val))
            
    def get_display_image(self):
        w, h = 64, 32 # Small preview
        img = np.zeros((h, w), dtype=np.uint8)
        if len(self.history) > 1:
            # Use last w samples
            history_array = np.array(list(self.history))
            if len(history_array) > w:
                history_array = history_array[-w:]
            
            min_val, max_val = np.min(history_array), np.max(history_array)
            range_val = max_val - min_val
            
            if range_val > 1e-6:
                vis_history = (history_array - min_val) / range_val
            else:
                vis_history = np.full_like(history_array, 0.5) 
            
            for i in range(len(vis_history) - 1):
                val1 = vis_history[i]
                y1 = int((1 - val1) * (h-1)) 
                x1 = int(i * (w / len(vis_history)))
                y1 = np.clip(y1, 0, h-1)
                img[y1, x1] = 255

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)