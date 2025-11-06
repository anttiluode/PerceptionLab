"""
Reaction-Diffusion Node - Simulates Gray-Scott pattern formation
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

class ReactionDiffusionNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(200, 80, 200) # A wild, purple-ish color
    
    def __init__(self, width=128, height=96):
        super().__init__()
        self.node_title = "Reaction-Diffusion"
        self.inputs = {
            'seed_image': 'image',
            'feed_rate': 'signal',
            'kill_rate': 'signal'
        }
        self.outputs = {'image': 'image', 'signal': 'signal'}
        
        self.w, self.h = width, height
        
        # Gray-Scott parameters
        self.f = 0.055  # Feed rate
        self.k = 0.062  # Kill rate
        self.dA = 1.0   # Diffusion rate A
        self.dB = 0.5   # Diffusion rate B
        
        # Chemical concentrations
        # A (U) is the "substrate", B (V) is the "reactant"
        self.A = np.ones((self.h, self.w), dtype=np.float32)
        self.B = np.zeros((self.h, self.w), dtype=np.float32)
        
        # Seed the reaction
        self.seed_chemicals(self.w//2, self.h//2, 10)
        
        # Laplacian kernel for diffusion
        self.laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                                          [0.2, -1.0, 0.2],
                                          [0.05, 0.2, 0.05]], dtype=np.float32)

    def seed_chemicals(self, x, y, size):
        self.B[y-size:y+size, x-size:x+size] = 1.0

    def step(self):
        # Get parameters from inputs, or use defaults
        # Map signal range [0, 1] to a good parameter range
        f_in = self.get_blended_input('feed_rate', 'sum')
        k_in = self.get_blended_input('kill_rate', 'sum')
        
        if f_in is not None:
            self.f = np.clip(0.01 + f_in * 0.09, 0.01, 0.1) # map [0,1] to [0.01, 0.1]
        if k_in is not None:
            self.k = np.clip(0.045 + k_in * 0.025, 0.045, 0.07) # map [0,1] to [0.045, 0.07]

        # Use an input image to "paint" chemical B
        img_in = self.get_blended_input('seed_image', 'mean')
        if img_in is not None:
            img_resized = cv2.resize(img_in, (self.w, self.h))
            self.B[img_resized > 0.5] = 1.0
            
        # Run 5 simulation steps per frame for speed
        for _ in range(5):
            # Calculate diffusion using convolution
            laplace_A = cv2.filter2D(self.A, -1, self.laplacian_kernel)
            laplace_B = cv2.filter2D(self.B, -1, self.laplacian_kernel)
            
            # The reaction part
            reaction = self.A * self.B**2
            
            # Gray-Scott equations
            delta_A = (self.dA * laplace_A) - reaction + (self.f * (1 - self.A))
            delta_B = (self.dB * laplace_B) + reaction - ((self.k + self.f) * self.B)
            
            # Update chemicals
            self.A += delta_A
            self.B += delta_B
            
            # Clamp values
            self.A = np.clip(self.A, 0.0, 1.0)
            self.B = np.clip(self.B, 0.0, 1.0)

    def get_output(self, port_name):
        if port_name == 'image':
            # We visualize chemical B, which forms the patterns
            return self.B
        elif port_name == 'signal':
            # Output the mean concentration of B
            return np.mean(self.B)
        return None
        
    def get_display_image(self):
        # Display chemical B
        img_u8 = (np.clip(self.B, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Feed Rate (f)", "f", self.f, None),
            ("Kill Rate (k)", "k", self.k, None),
        ]