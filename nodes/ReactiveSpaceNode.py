"""
Reactive Space Node - A simplified, audio-reactive version of the
earth19.py particle simulation.
Does not use Pygame, Torch, or OpenGL.
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
QtGui = __main__.QtGui
# ------------------------------------

# --- Color Map Dictionary ---
# Maps string names to OpenCV colormap constants
CMAP_DICT = {
    "gray": None, # Special case for no colormap
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "hot": cv2.COLORMAP_HOT,
    "jet": cv2.COLORMAP_JET
}


class ReactiveSpaceNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(50, 80, 160) # Deep space blue
    
    def __init__(self, particle_count=200, width=160, height=120, color_scheme='plasma'):
        super().__init__()
        self.node_title = "Reactive Space"
        
        # --- MODIFIED: Inputs for audio-reactivity ---
        self.inputs = {
            'bass_in': 'signal',  # Controls Sun/Attractor
            'highs_in': 'signal'  # Controls Stars/Particles
        }
        self.outputs = {'image': 'image', 'signal': 'signal'}
        
        self.w, self.h = width, height
        self.particle_count = int(particle_count)
        
        # --- ADDED: Color scheme ---
        self.color_scheme = str(color_scheme)
        
        # Particle state
        self.positions = np.random.rand(self.particle_count, 2).astype(np.float32) * [self.w, self.h]
        self.velocities = (np.random.rand(self.particle_count, 2).astype(np.float32) - 0.5) * 2.0
        
        # The "density" image
        self.space = np.zeros((self.h, self.w), dtype=np.float32)
        
        self.time = 0.0

    def step(self):
        self.time += 0.01
        
        # --- Get audio-reactive signals ---
        # Get 0-1 signals from (e.g.) SpectrumAnalyzer
        bass_energy = self.get_blended_input('bass_in', 'sum') or 0.0
        highs_energy = self.get_blended_input('highs_in', 'sum') or 0.0

        # Central attractor
        attractor_pos = np.array([
            self.w / 2 + np.sin(self.time * 0.5) * self.w * 0.3,
            self.h / 2 + np.cos(self.time * 0.3) * self.h * 0.3
        ])
        
        # Calculate forces (simple gravity)
        to_attractor = attractor_pos - self.positions
        dist_sq = np.sum(to_attractor**2, axis=1, keepdims=True) + 1e-3
        
        # --- MODIFIED: Bass controls gravity strength ---
        base_gravity = 5.0
        sun_pulse_strength = 1.0 + (bass_energy * 5.0) # Bass makes the "Sun" pulse
        force = to_attractor / dist_sq * (base_gravity * sun_pulse_strength)
        
        # Update velocities
        self.velocities += force * 0.1 # dt
        
        # --- MODIFIED: Highs add "energy" (jiggle) to stars ---
        star_jiggle = (np.random.rand(self.particle_count, 2) - 0.5) * (highs_energy * 0.5)
        self.velocities += star_jiggle
        
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
        # --- MODIFIED: Apply color map ---
        img_u8 = (np.clip(self.display_img, 0, 1) * 255).astype(np.uint8)
        
        cmap_cv2 = CMAP_DICT.get(self.color_scheme)
        
        if cmap_cv2 is not None:
            # Apply CV2 colormap
            img_color = cv2.applyColorMap(img_u8, cmap_cv2)
            img_color = np.ascontiguousarray(img_color)
            h, w = img_color.shape[:2]
            return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
        else:
            # Just return grayscale
            img_u8 = np.ascontiguousarray(img_u8)
            h, w = img_u8.shape
            return QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        # Create color scheme options for the dropdown
        color_options = [(name.title(), name) for name in CMAP_DICT.keys()]
        
        return [
            ("Particle Count", "particle_count", self.particle_count, None),
            ("Color Scheme", "color_scheme", self.color_scheme, color_options),
        ]