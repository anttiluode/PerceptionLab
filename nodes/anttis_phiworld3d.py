"""
Antti's PhiWorld 3D Node - A 3D particle field simulation
Driven by an energy signal and perturbed by an image slice.
Physics adapted from phiworld2.py.
3D logic inspired by best.py.
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
    from scipy.ndimage import maximum_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: PhiWorld3DNode requires 'scipy'.")
    print("Please run: pip install scipy")

class PhiWorld3DNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, grid_size=48):
        super().__init__()
        self.node_title = "Antti's PhiWorld 3D"
        
        self.inputs = {
            'energy_in': 'signal', # Drives the simulation
            'perturb_in': 'image', # 2D image to "push" the field
            'z_slice': 'signal'    # Controls which Z-slice to push (range -1 to 1)
        }
        self.outputs = {
            'field_slice': 'image',   # A 2D slice of the 3D field (for display)
            'particles_slice': 'image', # A 2D slice of detected particles
            'count': 'signal'         # Total 3D particle count
        }
        
        self.grid_size = int(grid_size)
        
        # --- Parameters from phiworld2.py ---
        self.dt = 0.08
        self.damping = 0.005
        self.base_c_sq = 1.0
        self.tension_factor = 5.0
        self.potential_lin = 1.0
        self.potential_cub = 0.2
        self.biharmonic_gamma = 0.02
        self.particle_threshold = 0.5
        
        # --- Internal 3D State ---
        shape = (self.grid_size, self.grid_size, self.grid_size)
        self.phi = np.zeros(shape, dtype=np.float64)
        self.phi_old = np.zeros_like(self.phi)
        
        # Outputs
        self.particle_image = np.zeros_like(self.phi, dtype=np.float32)
        self.particle_count = 0.0

        if not SCIPY_AVAILABLE:
            self.node_title = "PhiWorld 3D (No SciPy!)"

    # --- 3D Physics methods adapted from phiworld2.py ---
    
    def _laplacian_3d(self, f):
        """A 3D Laplacian using numpy.roll (inspired by 2D version)"""
        lap_x = np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)
        lap_y = np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
        lap_z = np.roll(f, -1, axis=2) - 2 * f + np.roll(f, 1, axis=2)
        return lap_x + lap_y + lap_z

    def _biharmonic(self, f):
        """3D Biharmonic is the Laplacian of the Laplacian"""
        lap_f = self._laplacian_3d(f)
        return self._laplacian_3d(lap_f)

    def _potential_deriv(self, phi):
        """Element-wise potential, works in 3D"""
        return (-self.potential_lin * phi
                + self.potential_cub * (phi**3))

    def _local_speed_sq(self, phi):
        """Element-wise speed, works in 3D"""
        intensity = phi**2
        return self.base_c_sq / (1.0 + self.tension_factor * intensity + 1e-9)

    def _track_particles(self, field):
        """3D particle tracking using scipy.ndimage.maximum_filter"""
        # Find local maxima using a 3x3x3 filter
        maxima_mask = (field == maximum_filter(field, size=(3, 3, 3)))
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
        z_slice_signal = self.get_blended_input('z_slice', 'sum') or 0.0
        
        if energy <= 0.01:
            # If no energy, dampen the field
            self.phi *= (1.0 - (self.damping * 10)) # Faster damping
            self.phi_old = self.phi.copy()
            self.particle_image *= 0.9
            self.particle_count = 0
            return

        # --- Run 3D simulation step (adapted from phiworld2.py) ---
        
        # Calculate 3D forces
        lap_phi = self._laplacian_3d(self.phi)
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
            # Determine which Z-slice to push
            # Map signal [-1, 1] to [0, grid_size-1]
            z_index = int(np.clip((z_slice_signal + 1.0) / 2.0 * (self.grid_size - 1), 0, self.grid_size - 1))
            
            # Resize image to grid slice
            img_resized = cv2.resize(perturb_img, (self.grid_size, self.grid_size),
                                     interpolation=cv2.INTER_AREA)
                                     
            # "Push" the field at that slice
            push_force = (img_resized - 0.5) * 0.1 * energy # Map [0,1] to [-0.05, 0.05] * energy
            phi_new[z_index, :, :] += push_force

        self.phi_old = self.phi.copy()
        self.phi = phi_new
        
        # Clamp to prevent instability
        self.phi = np.clip(self.phi, -10.0, 10.0)

        # Track particles on the new 3D field
        self._track_particles(np.abs(self.phi))

    def get_output(self, port_name):
        # Output the middle slice for visualization
        z_mid = self.grid_size // 2
        
        if port_name == 'field_slice':
            # Normalize field slice for output [-2, 2] -> [0, 1]
            field_slice = self.phi[z_mid, :, :]
            return np.clip(field_slice * 0.25 + 0.5, 0.0, 1.0)
        
        elif port_name == 'particles_slice':
            return self.particle_image[z_mid, :, :]
            
        elif port_name == 'count':
            # Output the total 3D particle count
            return self.particle_count
        return None
        
    def get_display_image(self):
        # Get the middle slice for the node's display
        z_mid = self.grid_size // 2
        field_slice = self.phi[z_mid, :, :]
        particles_slice = self.particle_image[z_mid, :, :]
        
        # Normalize field for display
        img_norm = np.clip(field_slice * 0.25 + 0.5, 0.0, 1.0)
        img_u8 = (img_norm * 255).astype(np.uint8)
        
        # Apply a colormap
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        # Overlay particles in bright red
        img_color[particles_slice > 0] = (0, 0, 255) # BGR for red
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size (3D)", "grid_size", self.grid_size, None),
            ("Particle Thresh", "particle_threshold", self.particle_threshold, None),
            ("Damping", "damping", self.damping, None),
            ("Tension", "tension_factor", self.tension_factor, None),
            ("Linear Pot.", "potential_lin", self.potential_lin, None),
            ("Cubic Pot.", "potential_cub", self.potential_cub, None),
            ("Biharmonic (g)", "biharmonic_gamma", self.biharmonic_gamma, None),
        ]