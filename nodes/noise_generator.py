"""
Noise Generator Node - Generates various noise types
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

class NoiseGeneratorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80) # Source Green
    
    def __init__(self, width=160, height=120, noise_type='white', speed=0.1):
        super().__init__()
        self.node_title = "Noise Gen"
        self.outputs = {'image': 'image', 'signal': 'signal'} 
        self.w, self.h = int(width), int(height)
        self.noise_type = noise_type 
        self.speed = float(speed)
        
        self._init_arrays()
        
    def _init_arrays(self):
        """Initialize or reinitialize arrays based on current w, h"""
        self.img = np.random.rand(self.h, self.w).astype(np.float32)
        self.signal_value = 0.0 
        self.brown_state = np.zeros((self.h, self.w), dtype=np.float32)
        self.perlin_phase = np.random.rand(2) * 100

    def _generate_noise_step(self, shape):
        """Generates a noise array based on the selected type."""
        if self.noise_type == 'white':
            return np.random.rand(*shape)
        
        elif self.noise_type == 'brown':
            # Ensure brown_state matches current shape
            if self.brown_state.shape != shape:
                self.brown_state = np.zeros(shape, dtype=np.float32)
            
            rand_step = np.random.randn(*shape) * 0.05 * self.speed
            self.brown_state = self.brown_state + rand_step
            self.brown_state = np.clip(self.brown_state, -1.0, 1.0)
            return (self.brown_state + 1.0) / 2.0
        
        elif self.noise_type == 'perlin':
            X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            self.perlin_phase += self.speed * 0.1 
            
            noise_val = (
                np.sin(X * 0.1 + self.perlin_phase[0]) + 
                np.sin(Y * 0.05 + self.perlin_phase[1] * 0.5)
            )
            noise_val = (noise_val - noise_val.min()) / (noise_val.max() - noise_val.min() + 1e-9)
            noise_val += np.random.rand(*shape) * 0.01 
            return np.clip(noise_val, 0, 1)
            
        elif self.noise_type == 'quantum':
            noise = np.random.rand(*shape)
            if np.random.rand() < 0.02 * self.speed * 10: 
                 noise += np.random.rand(*shape) * 0.5 * self.speed
            return np.clip(noise, 0, 1)
            
        return np.random.rand(*shape)

    def step(self):
        # Check if dimensions changed (from config update)
        if self.img.shape != (self.h, self.w):
            self._init_arrays()
        
        new_noise = self._generate_noise_step((self.h, self.w))
        
        self.img = self.img * (1.0 - self.speed) + new_noise * self.speed
        
        center_y, center_x = self.h // 2, self.w // 2
        window_size = 10
        y_start = max(0, center_y - window_size//2)
        y_end = min(self.h, center_y + window_size//2)
        x_start = max(0, center_x - window_size//2)
        x_end = min(self.w, center_x + window_size//2)
        
        center_patch = self.img[y_start:y_end, x_start:x_end]
        
        if center_patch.size > 0:
            self.signal_value = np.mean(center_patch) * 2.0 - 1.0
        else:
            self.signal_value = 0.0
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.img
        elif port_name == 'signal':
            return self.signal_value
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Noise Type", "noise_type", self.noise_type, [
                ("White (Uniform)", "white"), 
                ("Brown (Coherent)", "brown"),
                ("Perlin (Pattern)", "perlin"), 
                ("Quantum (Spikes)", "quantum")
            ]),
            ("Speed (Blend Factor)", "speed", self.speed, None),
        ]