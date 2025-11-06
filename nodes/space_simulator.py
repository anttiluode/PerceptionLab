"""
Space Simulator Node - Simulates a 2D particle universe
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

class SpaceSimulatorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(50, 80, 160) # Deep space blue
    
    def __init__(self, particle_count=200, width=160, height=120):
        super().__init__()
        self.node_title = "Space Simulator"
        self.outputs = {'image': 'image', 'signal': 'signal'}
        
        self.w, self.h = width, height
        self.particle_count = int(particle_count)
        
        # Particle state
        self.positions = np.random.rand(self.particle_count, 2).astype(np.float32) * [self.w, self.h]
        self.velocities = (np.random.rand(self.particle_count, 2).astype(np.float32) - 0.5) * 2.0
        
        # The "density" image
        self.space = np.zeros((self.h, self.w), dtype=np.float32)
        
        self.time = 0.0

    def step(self):
        self.time += 0.01
        
        # Central attractor
        attractor_pos = np.array([
            self.w / 2 + np.sin(self.time * 0.5) * self.w * 0.3,
            self.h / 2 + np.cos(self.time * 0.3) * self.h * 0.3
        ])
        
        # Calculate forces (simple gravity)
        to_attractor = attractor_pos - self.positions
        dist_sq = np.sum(to_attractor**2, axis=1, keepdims=True) + 1e-3
        force = to_attractor / dist_sq * 5.0 # Gravity strength
        
        # Update velocities
        self.velocities += force * 0.1 # dt
        self.velocities *= 0.98 # Damping
        
        # Update positions
        self.positions += self.velocities
        
        # Bounce off walls
        mask_x_low = self.positions[:, 0] < 0
        mask_x_high = self.positions[:, 0] >= self.w
        mask_y_low = self.positions[:, 1] < 0
        mask_y_high = self.positions[:, 1] >= self.h
        
        self.positions[mask_x_low, 0] = 0
        self.positions[mask_x_high, 0] = self.w - 1
        self.positions[mask_y_low, 1] = 0
        self.positions[mask_y_high, 1] = self.h - 1
        
        self.velocities[mask_x_low | mask_x_high, 0] *= -0.5
        self.velocities[mask_y_low | mask_y_high, 1] *= -0.5

        # Update the density image
        self.space *= 0.9 # Fade old trails
        
        # Get integer positions
        int_pos = self.positions.astype(int)
        
        # Valid coordinates
        valid = (int_pos[:, 0] >= 0) & (int_pos[:, 0] < self.w) & \
                (int_pos[:, 1] >= 0) & (int_pos[:, 1] < self.h)
        
        valid_pos = int_pos[valid]
        
        # "Splat" particles onto the image
        if valid_pos.shape[0] > 0:
            self.space[valid_pos[:, 1], valid_pos[:, 0]] = 1.0 # Bright points
        
        # Blur to make it look like a density field
        display_img = cv2.GaussianBlur(self.space, (5, 5), 0)
        self.display_img = display_img

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_img
        elif port_name == 'signal':
            # Output mean velocity as a signal
            return np.mean(np.linalg.norm(self.velocities, axis=1))
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.display_img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Particle Count", "particle_count", self.particle_count, None)
        ]