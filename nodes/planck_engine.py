"""
Revolving Bit Simulator (Planck Engine) Node
Implements the core mathematics of the Revolving Bit Theory from bit-theory.py
- Fundamental Bits (Spinors `S`)
- Lagging Manifest Fields (Complex Scalar `Φ`)
- Emergent motion and forces
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

# --- Pauli Matrices (NumPy version) ---
SIGMA_1 = np.array([[0, 1], [1, 0]], dtype=np.complex64) # sigma_x
SIGMA_2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex64) # sigma_y
SIGMA_3 = np.array([[1, 0], [0, -1]], dtype=np.complex64) # sigma_z

class RevolvingBit:
    """ Represents a single fundamental Bit (Spinor S) """
    def __init__(self, initial_pos, omega_0, k1, k2, tau_dt, grid_size):
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.S = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex64)
        self.tau = 0.0
        self.grid_size = grid_size
        
        # Store constants
        self.omega_0 = omega_0
        self.k1 = k1
        self.k2 = k2
        self.tau_dt = tau_dt

    def normalize_S(self):
        norm_S_sq = np.sum(np.abs(self.S)**2)
        if norm_S_sq > 1e-9:
            self.S /= np.sqrt(norm_S_sq)

    def revolve_step(self, external_phi_field_at_pos):
        """ Intrinsic revolution + interaction with external Phi field """
        V_spinor = (self.k1 * external_phi_field_at_pos.real * SIGMA_1 +
                    self.k2 * external_phi_field_at_pos.imag * SIGMA_2)
        
        H_spinor = self.omega_0 * SIGMA_3 + V_spinor
        
        dS = -1j * np.dot(H_spinor, self.S) * self.tau_dt
        self.S += dS
        self.normalize_S()
        self.tau += self.tau_dt

    def move_step(self, force_gradient, attraction_gamma, dt):
        """ Move based on gradients in the total Phi field """
        acceleration = attraction_gamma * force_gradient
        self.velocity += acceleration * dt
        self.velocity *= 0.98 # Damping
        self.pos += self.velocity * dt
        self.pos %= self.grid_size # Wrap around grid

class RevolvingBitNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 150, 200) # A "quantum" blue
    
    def __init__(self, grid_size=64, num_bits=3):
        super().__init__()
        self.node_title = "Planck Engine"
        
        self.inputs = {'coupling': 'signal', 'attraction': 'signal'}
        self.outputs = {
            'field_amp': 'image', 
            'field_phase': 'image', 
            'avg_amp': 'signal'
        }
        
        self.N = int(grid_size)
        
        # --- Physics Parameters from bit-theory.py ---
        self.DT = 0.01
        self.TAU_DT = 0.05
        self.C_SUBSTRATE = 1.0
        self.FIELD_MASS = 0.1
        self.OMEGA_0 = 1.0
        
        # Controllable params
        self.bit_field_coupling_g = 0.5
        self.spinor_potential_k1 = 0.1
        self.spinor_potential_k2 = 0.1
        self.attraction_gamma = 0.2
        
        # --- Internal State ---
        self.phi = np.zeros((self.N, self.N), dtype=np.complex64)
        self.phi_prev = self.phi.copy()
        
        self.bits = []
        for i in range(int(num_bits)):
            pos = np.random.uniform(self.N * 0.2, self.N * 0.8, 2)
            self.bits.append(RevolvingBit(
                pos, self.OMEGA_0, self.spinor_potential_k1, 
                self.spinor_potential_k2, self.TAU_DT, self.N
            ))
            
        # Precompute grid for field generation
        x_coords = np.arange(self.N, dtype=np.float32)
        self.X_grid, self.Y_grid = np.meshgrid(x_coords, x_coords, indexing='ij')

    def _laplacian_2d(self, grid):
        return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid)

    def _get_field_at_pos(self, bit, field_to_sample):
        """ Interpolate field value at a Bit's continuous position """
        x_idx = int(round(bit.pos[0])) % self.N
        y_idx = int(round(bit.pos[1])) % self.N
        return field_to_sample[x_idx, y_idx]

    def _get_gradient_at_pos(self, bit, field_mag):
        """ Estimate gradient of field magnitude at Bit's position """
        x = int(round(bit.pos[0]))
        y = int(round(bit.pos[1]))

        grad_x = (field_mag[(x + 1) % self.N, y % self.N] - 
                    field_mag[(x - 1) % self.N, y % self.N]) / 2.0
        grad_y = (field_mag[x % self.N, (y + 1) % self.N] - 
                    field_mag[x % self.N, (y - 1) % self.N]) / 2.0
        return np.array([grad_x, grad_y], dtype=np.float32)

    def step(self):
        # Update params from inputs
        self.bit_field_coupling_g = (self.get_blended_input('coupling', 'sum') or 0.0) * 0.5 + 0.5 # [0, 1]
        self.attraction_gamma = (self.get_blended_input('attraction', 'sum') or 0.0) * 0.2 + 0.2 # [0, 0.4]
        
        # --- 1. Evolve each Bit's internal spinor state S ---
        for bit in self.bits:
            phi_ext = self._get_field_at_pos(bit, self.phi)
            bit.revolve_step(phi_ext)

        # --- 2. Update the Manifest Field Phi based on ALL Bits ---
        source_term = np.zeros_like(self.phi)
        
        for bit in self.bits:
            dist_sq = (self.X_grid - bit.pos[0])**2 + (self.Y_grid - bit.pos[1])**2
            source_spread_sigma_sq = 4.0 # 2.0**2
            bit_source_profile = np.exp(-dist_sq / (2 * source_spread_sigma_sq))
            
            # Source is the complex spinor component S[0]
            source_term += self.bit_field_coupling_g * bit.S[0] * bit_source_profile

        # Evolve Phi field (Klein-Gordon)
        lap_phi = self._laplacian_2d(self.phi)
        
        phi_new = (2 * self.phi - self.phi_prev +
                   self.C_SUBSTRATE**2 * self.DT**2 * (lap_phi - self.FIELD_MASS**2 * self.phi + source_term))
        
        self.phi_prev = self.phi.copy()
        self.phi = phi_new
        
        # --- 3. Move each Bit based on the TOTAL Phi field ---
        phi_magnitude_field = np.abs(self.phi)
        for bit in self.bits:
            grad_phi_mag_at_pos = self._get_gradient_at_pos(bit, phi_magnitude_field)
            bit.move_step(grad_phi_mag_at_pos, self.attraction_gamma, self.DT)

    def get_output(self, port_name):
        mag = np.abs(self.phi)
        vmax = mag.max() + 1e-9
        
        if port_name == 'field_amp':
            return mag / vmax
        elif port_name == 'field_phase':
            return (np.angle(self.phi) + np.pi) / (2 * np.pi) # [0, 1]
        elif port_name == 'avg_amp':
            return np.mean(mag)
        return None
        
    def get_display_image(self):
        mag = np.abs(self.phi)
        vmax = mag.max() + 1e-9
        
        # Normalize amplitude and apply MAGMA colormap
        img_norm = (mag / vmax * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_MAGMA)
        
        # Draw bits
        for bit in self.bits:
            # (y, x) for cv2 drawing
            x_pos = int(round(bit.pos[1])) % self.N 
            y_pos = int(round(bit.pos[0])) % self.N
            cv2.circle(img_color, (x_pos, y_pos), 3, (0, 255, 255), -1) # Cyan bits
            
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size", "N", self.N, None),
            ("Num Bits", "num_bits", len(self.bits), None),
        ]