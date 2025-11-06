"""
Antti's CrystalMaker Node - A 3D polyrhythmic field generator
Based on the PolyrhythmicSea class from crystal_kingdom.py
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
    from scipy.signal import convolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: CrystalMakerNode requires 'scipy'.")
    print("Please run: pip install scipy")

class CrystalMakerNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 200, 250) # Crystalline blue
    
    def __init__(self, grid_size=32, num_fields=10):
        super().__init__()
        self.node_title = "Antti's CrystalMaker"
        
        self.inputs = {
            'tension': 'signal',
            'damping': 'signal',
            'nonlinearity_a': 'signal',
            'nonlinearity_b': 'signal'
        }
        self.outputs = {
            'field_slice': 'image', # 2D slice of the 3D field
            'total_energy': 'signal'
        }
        
        self.N = int(grid_size)
        self.num_fields = int(num_fields)
        
        # --- Physics Parameters from crystal_kingdom.py ---
        self.dt = 0.05
        self.polyrhythm_coupling = 0.1
        self.nonlinearity_A = 1.0
        self.nonlinearity_B = 1.0
        self.damping_factor = 0.005
        self.tension = 5.0
        self.base_frequencies_min = 0.5
        self.base_frequencies_max = 2.5
        self.diffusion_coeffs_min = 0.05
        self.diffusion_coeffs_max = 0.1
        
        self.total_energy = 0.0
        
        # --- Internal 3D State ---
        self._initialize_fields_and_params()
        
        # 3D Laplacian Kernel
        self.kern = np.zeros((3,3,3), np.float32)
        self.kern[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
            self.kern[dx,dy,dz] = 1
            
        if not SCIPY_AVAILABLE:
            self.node_title = "CrystalMaker (No SciPy!)"

    def _initialize_fields_and_params(self):
        """Initializes or re-initializes fields."""
        shape = (self.N, self.N, self.N)
        self.phi_fields = [(np.random.rand(*shape).astype(np.float32) - 0.5) * 0.5
                           for _ in range(self.num_fields)]
        self.phi_o_fields = [np.copy(phi) for phi in self.phi_fields]
        
        self.base_frequencies = np.linspace(self.base_frequencies_min, self.base_frequencies_max, self.num_fields)
        self.diffusion_coeffs = np.linspace(self.diffusion_coeffs_max, self.diffusion_coeffs_min, self.num_fields)
        self.field_phases = np.random.uniform(0, 2 * np.pi, self.num_fields)

        self.phi = np.zeros(shape, dtype=np.float32)
        self.phi_o = np.zeros(shape, dtype=np.float32)
        self._update_summed_fields()

    def _update_summed_fields(self):
        """Update the main summed field from individual phi fields"""
        self.phi = np.sum(self.phi_fields, axis=0) / max(1, len(self.phi_fields))
        self.phi_o = np.sum(self.phi_o_fields, axis=0) / max(1, len(self.phi_fields))

    def _potential_deriv(self, field_k):
        """Calculate the derivative of the potential function for a field"""
        return -self.nonlinearity_A * field_k + self.nonlinearity_B * (field_k**3)
        
    def _laplacian(self, f):
        """3D Laplacian using convolution"""
        return convolve(f, self.kern, mode='wrap')

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # --- Update parameters from inputs ---
        # Map signals [-1, 1] to a useful range
        self.tension = (self.get_blended_input('tension', 'sum') or 0.0) * 10.0 + 10.0 # Range [0, 20]
        self.damping_factor = (self.get_blended_input('damping', 'sum') or 0.0) * 0.02 + 0.02 # Range [0, 0.04]
        self.nonlinearity_A = (self.get_blended_input('nonlinearity_a', 'sum') or 0.0) + 1.0 # Range [0, 2]
        self.nonlinearity_B = (self.get_blended_input('nonlinearity_b', 'sum') or 0.0) + 1.0 # Range [0, 2]

        # --- Run simulation step (from crystal_kingdom.py) ---
        new_phi_list = []

        self.field_phases += self.base_frequencies * self.dt
        self.field_phases %= (2 * np.pi)

        for k in range(self.num_fields):
            phi_k = self.phi_fields[k]
            phi_o_k = self.phi_o_fields[k]

            vel_k = phi_k - phi_o_k
            lap_k = self._laplacian(phi_k)
            potential_deriv_k = self._potential_deriv(phi_k)

            other_fields_sum = (np.sum(self.phi_fields, axis=0) - phi_k)
            coupling_force = self.polyrhythm_coupling * other_fields_sum / max(1, self.num_fields - 1)

            driving_force_k = 0.005 * np.sin(self.field_phases[k])
            c2 = 1.0 / (1.0 + self.tension * phi_k**2 + 1e-6)

            acc = (c2 * self.diffusion_coeffs[k] * lap_k -
                   potential_deriv_k +
                   coupling_force +
                   driving_force_k)

            new_phi_k = phi_k + (1 - self.damping_factor * self.dt) * vel_k + self.dt**2 * acc
            new_phi_list.append(new_phi_k)

        self.phi_fields = new_phi_list
        self._update_summed_fields()
        
        # Calculate total energy (simplified)
        self.total_energy = np.mean(self.phi**2)

    def get_output(self, port_name):
        if port_name == 'field_slice':
            # Output the middle slice
            z_mid = self.N // 2
            field_slice = self.phi[z_mid, :, :]
            
            # Normalize field for output
            vmax = np.abs(field_slice).max() + 1e-9
            return (field_slice / (2 * vmax)) + 0.5 # map [-v, v] to [0, 1]
            
        elif port_name == 'total_energy':
            return self.total_energy
        return None
        
    def get_display_image(self):
        # Get the middle slice for the node's display
        z_mid = self.N // 2
        field_slice = self.phi[z_mid, :, :]
        
        # Normalize field for display
        vmax = np.abs(field_slice).max() + 1e-9
        img_norm = np.clip((field_slice / (2 * vmax)) + 0.5, 0.0, 1.0)
        
        img_u8 = (img_norm * 255).astype(np.uint8)
        
        # Apply a colormap
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size (3D)", "N", self.N, None),
            ("Num Fields", "num_fields", self.num_fields, None),
        ]

    def close(self):
        # Clear large arrays on close
        self.phi_fields = []
        self.phi_o_fields = []
        self.phi = None
        self.phi_o = None
        super().close()