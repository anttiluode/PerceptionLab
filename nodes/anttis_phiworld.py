"""
Antti's PhiWorld Node - A TADS-like particle field simulation
Driven by an energy signal and perturbed by an image.
Based on the physics from phiworld2.py.
Requires: pip install scipy
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

try:
    from scipy.signal import convolve2d
    from scipy.ndimage import maximum_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: PhiWorldNode requires 'scipy'.")
    print("Please run: pip install scipy")

class PhiWorldNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, grid_size=96):
        super().__init__()
        self.node_title = "Antti's PhiWorld"
        
        self.inputs = {
            'energy_in': 'signal', # Drives the simulation
            'perturb_in': 'image'  # Pushes the field
        }
        self.outputs = {
            'field': 'image',       # The raw phi field
            'particles': 'image',   # Just the detected particles
            'count': 'signal'       # Number of particles
        }
        
        self.grid_size = int(grid_size)
        
        # --- Parameters from phiworld2.py ---
        self.dt = 0.08
        self.damping = 0.005 # Increased damping for stability in node
        self.base_c_sq = 1.0
        self.tension_factor = 5.0
        self.potential_lin = 1.0
        self.potential_cub = 0.2
        self.biharmonic_gamma = 0.02
        self.particle_threshold = 0.5
        
        # --- Internal State ---
        self.phi = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        self.phi_old = np.zeros_like(self.phi)
        
        # Optimized Laplacian Kernel
        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1]], dtype=np.float64)
        
        # Outputs
        self.particle_image = np.zeros_like(self.phi, dtype=np.float32)
        self.particle_count = 0.0

        if not SCIPY_AVAILABLE:
            self.node_title = "PhiWorld (No SciPy!)"

    # --- Physics methods adapted from phiworld2.py ---
    
    def _laplacian(self, f):
        # Using np.roll is faster than convolve2d for this kernel
        lap_x = np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
        lap_y = np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)
        return lap_x + lap_y

    def _biharmonic(self, f):
        lap_f = self._laplacian(f)
        return self._laplacian(lap_f) # Laplacian of the Laplacian

    def _potential_deriv(self, phi):
        return (-self.potential_lin * phi
                + self.potential_cub * (phi**3))

    def _local_speed_sq(self, phi):
        intensity = phi**2
        return self.base_c_sq / (1.0 + self.tension_factor * intensity + 1e-9)

    def _track_particles(self, field):
        """Optimized particle tracking using scipy.ndimage."""
        # Find local maxima using a 3x3 filter
        maxima_mask = (field == maximum_filter(field, size=3))
        # Find points above threshold
        threshold_mask = (field > self.particle_threshold)
        
        # Combine masks
        particle_mask = (maxima_mask & threshold_mask)
        
        # Update outputs
        self.particle_image = particle_mask.astype(np.float32)
        self.particle_count = np.sum(particle_mask)

    def step(self):
        if not SCIPY_AVAILABLE:
            return

        # Get inputs
        energy = self.get_blended_input('energy_in', 'sum') or 0.0
        perturb_img = self.get_blended_input('perturb_in', 'mean')
        
        if energy <= 0.01:
            # If no energy, dampen the field
            self.phi *= (1.0 - (self.damping * 10)) # Faster damping
            self.phi_old = self.phi.copy()
            self.particle_image *= 0.9
            self.particle_count = 0
            return

        # --- Run simulation step (from phiworld2.py) ---
        
        # Calculate forces
        lap_phi = self._laplacian(self.phi)
        biharm_phi = self._biharmonic(self.phi)
        c2 = self._local_speed_sq(self.phi)
        V_prime = self._potential_deriv(self.phi)
        
        # Scale acceleration by energy input
        acceleration = energy * ( (c2 * lap_phi) - V_prime - (self.biharmonic_gamma * biharm_phi) )

        # Update field (Verlet integration)
        velocity = self.phi - self.phi_old
        phi_new = self.phi + (1.0 - self.damping * self.dt) * velocity + (self.dt**2) * acceleration

        # --- Add Image Perturbation ---
        if perturb_img is not None:
            # Resize image to grid
            img_resized = cv2.resize(perturb_img, (self.grid_size, self.grid_size),
                                     interpolation=cv2.INTER_AREA)
            # "Push" the field with the image, scaled by energy
            phi_new += (img_resized - 0.5) * 0.1 * energy # (Image is 0-1, so map to -0.5 to 0.5)

        self.phi_old = self.phi.copy()
        self.phi = phi_new
        
        # Clamp to prevent instability
        self.phi = np.clip(self.phi, -10.0, 10.0)

        # Track particles on the new field
        self._track_particles(np.abs(self.phi))

    def get_output(self, port_name):
        if port_name == 'field':
            # Normalize field for output [-2, 2] -> [0, 1]
            return np.clip(self.phi * 0.25 + 0.5, 0.0, 1.0)
        elif port_name == 'particles':
            return self.particle_image
        elif port_name == 'count':
            return self.particle_count
        return None
        
    def get_display_image(self):
        # Normalize field for display
        img_norm = np.clip(self.phi * 0.25 + 0.5, 0.0, 1.0)
        
        img_u8 = (img_norm * 255).astype(np.uint8)
        
        # Apply a colormap
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        # Overlay particles in bright red
        img_color[self.particle_image > 0] = (0, 0, 255) # BGR for red
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Particle Thresh", "particle_threshold", self.particle_threshold, None),
            ("Damping", "damping", self.damping, None),
            ("Tension", "tension_factor", self.tension_factor, None),
            ("Linear Pot.", "potential_lin", self.potential_lin, None),
            ("Cubic Pot.", "potential_cub", self.potential_cub, None),
            ("Biharmonic (g)", "biharmonic_gamma", self.biharmonic_gamma, None),
        ]