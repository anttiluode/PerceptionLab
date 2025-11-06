"""
Neural Field Node - Generates a 2D field from frequency-band signals
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

class NeuralFieldNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(200, 80, 200) # A generative purple
    
    def __init__(self, width=128, height=96):
        super().__init__()
        self.node_title = "Neural Field"
        self.inputs = {
            'theta': 'signal',  # 4-8 Hz (coarse structure)
            'alpha': 'signal',  # 8-13 Hz (intermediate)
            'beta': 'signal',   # 13-30 Hz (fine structure)
            'gamma': 'signal'   # 30-100 Hz (finest details)
        }
        self.outputs = {'image': 'image', 'signal': 'signal'}
        
        self.w, self.h = width, height
        
        # Pre-generate noise patterns at different scales
        self.noise_layers = [
            (cv2.resize(np.random.rand(self.h // 8, self.w // 8).astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_CUBIC)),
            (cv2.resize(np.random.rand(self.h // 4, self.w // 4).astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_CUBIC)),
            (cv2.resize(np.random.rand(self.h // 2, self.w // 2).astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_LINEAR)),
            (np.random.rand(self.h, self.w).astype(np.float32))
        ]
        
        # Normalize noise layers
        self.noise_layers = [(layer - layer.min()) / (layer.max() - layer.min() + 1e-9) for layer in self.noise_layers]
        
        self.field = np.zeros((self.h, self.w), dtype=np.float32)

    def step(self):
        # Get blended power from each band (normalize from [-1, 1] to [0, 1])
        theta_power = (self.get_blended_input('theta', 'sum') or 0.0 + 1.0) / 2.0
        alpha_power = (self.get_blended_input('alpha', 'sum') or 0.0 + 1.0) / 2.0
        beta_power  = (self.get_blended_input('beta', 'sum') or 0.0 + 1.0) / 2.0
        gamma_power = (self.get_blended_input('gamma', 'sum') or 0.0 + 1.0) / 2.0
        
        powers = [theta_power, alpha_power, beta_power, gamma_power]
        total_power = sum(powers) + 1e-9
        
        # Combine noise layers based on weighted average of powers
        self.field.fill(0.0)
        for i, layer in enumerate(self.noise_layers):
            self.field += layer * (powers[i] / total_power)
            
        # Add a slow "scrolling" effect to the noise
        self.noise_layers = [np.roll(layer, (1, 1), axis=(0, 1)) for layer in self.noise_layers]
        
        # Final normalization
        self.field = (self.field - self.field.min()) / (self.field.max() - self.field.min() + 1e-9)
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.field
        elif port_name == 'signal':
            return np.mean(self.field) * 2.0 - 1.0 # Remap to [-1, 1]
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.field, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Width", "w", self.w, None),
            ("Height", "h", self.h, None),
        ]