"""
Antti's Superfluid Node - Simulates a 1D complex field with knots
Physics based on the 1D NLSE from knotiverse_interactive_viewer.py
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
    from scipy.signal import hilbert
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: AnttiSuperfluidNode requires 'scipy'.")
    print("Please run: pip install scipy")

class AnttiSuperfluidNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 80, 180) # Superfluid purple
    
    def __init__(self, grid_size=512, coupling=0.5, nonlinear=0.8, damping=0.005):
        super().__init__()
        self.node_title = "Antti's Superfluid"
        
        self.inputs = {
            'signal_in': 'signal',
            'coupling': 'signal',
            'nonlinearity': 'signal',
            'damping': 'signal'
        }
        self.outputs = {
            'field_image': 'image',
            'angular_momentum': 'signal',
            'knot_count': 'signal'
        }
        
        # --- Parameters from knotiverse_interactive_viewer.py ---
        self.L = int(grid_size)
        self.dt = 0.05
        self.detect_threshold = 0.5
        self.saturation_threshold = 2.0
        self.max_amplitude_clip = 1e3
        
        # Default physics values (will be overridden by signals)
        self.coupling = coupling
        self.nonlinear = nonlinear
        self.damping = damping
        
        # --- Internal State ---
        rng = np.random.default_rng()
        self.psi = (rng.standard_normal(self.L) + 1j * rng.standard_normal(self.L)) * 0.01
        
        # Seed with a pulse
        x = np.arange(self.L)
        p = self.L // 2
        gauss = 1.0 * np.exp(-((x - p)**2) / (2 * 4**2))
        self.psi += gauss * np.exp(1j * 2.0 * np.pi * rng.random())
        
        self.knots = np.array([], dtype=int)
        self.angular_momentum_out = 0.0
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Superfluid (No SciPy!)"

    def laplacian_1d(self, arr):
        """Discrete laplacian with periodic boundary."""
        return np.roll(arr, -1) - 2*arr + np.roll(arr, 1)

    def step(self):
        if not SCIPY_AVAILABLE:
            return

        # --- Get inputs ---
        signal_in = self.get_blended_input('signal_in', 'sum') or 0.0
        coupling = self.get_blended_input('coupling', 'sum')
        nonlinear = self.get_blended_input('nonlinearity', 'sum')
        damping = self.get_blended_input('damping', 'sum')
        
        # Use signal if connected, else use internal value
        c = coupling if coupling is not None else self.coupling
        n = nonlinear if nonlinear is not None else self.nonlinear
        d = damping if damping is not None else self.damping

        # --- Physics Step (from knotiverse_interactive_viewer.py) ---
        lap = self.laplacian_1d(self.psi)
        coupling_term = 1j * c * lap
        
        amp = np.abs(self.psi)
        sat = np.tanh(amp / self.saturation_threshold)
        nonlin_term = -1j * n * (sat**2) * self.psi
        
        damping_term = -d * self.psi
        
        self.psi = self.psi + self.dt * (coupling_term + nonlin_term + damping_term)
        
        # --- Resonance from input signal ---
        # "Pluck" the center of the string
        self.psi[self.L // 2] += signal_in * 0.5 # Scale input
        
        # Stability checks
        self.psi = np.nan_to_num(self.psi, nan=0.0, posinf=0.0, neginf=0.0)
        amp_new = np.abs(self.psi)
        over = amp_new > self.max_amplitude_clip
        if np.any(over):
            self.psi[over] = self.psi[over] * (self.max_amplitude_clip / amp_new[over])
        
        amp_now = np.abs(self.psi)
        
        # --- Knot Detection ---
        left = np.roll(amp_now, 1)
        right = np.roll(amp_now, -1)
        mask_thresh = amp_now > self.detect_threshold
        mask_local_max = (amp_now >= left) & (amp_now >= right)
        self.knots = np.where(mask_thresh & mask_local_max)[0]
        self.knot_count_out = len(self.knots)
        
        # --- Angular Momentum ---
        grad_psi = np.roll(self.psi, -1) - np.roll(self.psi, 1)
        moment_density = np.imag(np.conj(self.psi) * grad_psi)
        self.angular_momentum_out = float(np.sum(moment_density))

    def get_output(self, port_name):
        if port_name == 'field_image':
            return self._draw_field_image(as_float=True)
        elif port_name == 'angular_momentum':
            return self.angular_momentum_out
        elif port_name == 'knot_count':
            return self.knot_count_out
        return None
        
    def _draw_field_image(self, as_float=False):
        h, w = 64, self.L
        img_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get field data
        amp_now = np.abs(self.psi)
        phase_now = np.angle(hilbert(self.psi.real))
        
        # Normalize
        amp_norm = np.clip(amp_now / self.saturation_threshold, 0, 1)
        phase_norm = (phase_now + np.pi) / (2 * np.pi)
        
        # Draw amplitude (top half) and phase (bottom half)
        h_half = h // 2
        for x in range(w):
            # Amplitude (Cyan)
            y_amp = int((h_half - 1) - amp_norm[x] * (h_half - 1))
            img_color[y_amp, x] = (255, 255, 0) # BGR for Cyan
            
            # Phase (Magenta)
            y_phase = int(h_half + (h_half - 1) - phase_norm[x] * (h_half - 1))
            img_color[y_phase, x] = (255, 0, 255) # BGR for Magenta
            
        # Draw center line
        cv2.line(img_color, (0, h // 2), (w, h // 2), (50, 50, 50), 1)
        
        # Draw knots (Red)
        for kx in self.knots:
            ky = int((h_half - 1) - amp_norm[kx] * (h_half - 1))
            cv2.circle(img_color, (kx, ky), 3, (0, 0, 255), -1) # BGR for Red
            
        if as_float:
            return img_color.astype(np.float32) / 255.0
            
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
        
    def get_display_image(self):
        return self._draw_field_image(as_float=False)

    def get_config_options(self):
        return [
            ("Grid Size", "L", self.L, None),
            ("Knot Threshold", "detect_threshold", self.detect_threshold, None),
            ("Coupling", "coupling", self.coupling, None),
            ("Nonlinearity", "nonlinear", self.nonlinear, None),
            ("Damping", "damping", self.damping, None),
        ]