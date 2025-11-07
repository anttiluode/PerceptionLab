"""
Bloch Qubit Node - Simulates a single qubit and plots its state on the Bloch Sphere.
Takes rotation angles as signal inputs.
Ported from qbit.py logic.
Requires: pip install numpy scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from scipy.linalg import expm # For matrix exponentiation

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.linalg import expm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: BlochQubitNode requires 'scipy'.")
    
# --- Pauli and Gate Matrices ---
# Simplified Hamiltonians for rotation gates (H_i = -i/2 * sigma_i)
H_X = np.array([[0, 1], [1, 0]], dtype=complex) * 0.5
H_Y = np.array([[0, -1j], [1j, 0]], dtype=complex) * 0.5
H_Z = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5
H_Hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

class BlochQubitNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(100, 100, 255) # Blue Qubit
    
    def __init__(self, hbar=1.0):
        super().__init__()
        self.node_title = "Bloch Qubit"
        
        self.inputs = {
            'rx_angle': 'signal',      # X-axis rotation angle
            'ry_angle': 'signal',      # Y-axis rotation angle
            'rz_angle': 'signal',      # Z-axis rotation angle
            'hadamard_trigger': 'signal' # Trigger Hadamard gate
        }
        self.outputs = {
            'bloch_x': 'signal',
            'bloch_y': 'signal',
            'bloch_z': 'signal',
            'prob_0': 'signal'
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Qubit (No SciPy!)"
            return

        self.hbar = float(hbar)
        
        # Initialize state: |0> (North Pole)
        self.state = np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
        
        # History for visualization
        self.bloch_coords_history = np.zeros((100, 3), dtype=np.float32)
        self.bloch_coords = self._get_bloch_coords(self.state)
        
        self.last_hadamard_trigger = 0.0

    def _get_bloch_coords(self, state):
        """Convert the qubit state to Bloch sphere coordinates (x,y,z)"""
        a, b = state
        x = 2 * np.real(a * np.conj(b))
        y = 2 * np.imag(a * np.conj(b))
        z = abs(a)**2 - abs(b)**2
        return np.array([x, y, z])

    def _apply_gate_hamiltonian(self, H, angle):
        """Evolve the state under Hamiltonian H for time proportional to angle."""
        if angle == 0.0: return
        
        # U = exp(-i H_matrix * angle / hbar)
        # We define H_matrix = H_i * 0.5 (from Qubit class example)
        # So U = expm(-1j * H * (angle / self.hbar))
        U = expm(-1j * H * (angle / self.hbar))
        self.state = U.dot(self.state)
        
        # Re-normalize (essential for numerical stability)
        norm = np.linalg.norm(self.state)
        if norm > 1e-9:
            self.state /= norm
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # 1. Get angle inputs (mapped from standard [-1, 1] signal)
        rx_angle = (self.get_blended_input('rx_angle', 'sum') or 0.0) * np.pi
        ry_angle = (self.get_blended_input('ry_angle', 'sum') or 0.0) * np.pi
        rz_angle = (self.get_blended_input('rz_angle', 'sum') or 0.0) * np.pi
        hadamard_trigger = self.get_blended_input('hadamard_trigger', 'sum') or 0.0
        
        # 2. Apply rotations (simulated via Hamiltonians from qbit.py)
        # These are applied sequentially in the simulation timestep
        self._apply_gate_hamiltonian(H_X, rx_angle)
        self._apply_gate_hamiltonian(H_Y, ry_angle)
        self._apply_gate_hamiltonian(H_Z, rz_angle)
        
        # 3. Handle Discrete Gates (Hadamard on rising edge)
        if hadamard_trigger > 0.5 and self.last_hadamard_trigger <= 0.5:
            self.state = H_Hadamard.dot(self.state)
            norm = np.linalg.norm(self.state)
            if norm > 1e-9:
                self.state /= norm

        self.last_hadamard_trigger = hadamard_trigger
        
        # 4. Update coordinates and history
        self.bloch_coords = self._get_bloch_coords(self.state)
        self.bloch_coords_history[:-1] = self.bloch_coords_history[1:]
        self.bloch_coords_history[-1] = self.bloch_coords
        

    def get_output(self, port_name):
        if port_name == 'bloch_x':
            return self.bloch_coords[0]
        elif port_name == 'bloch_y':
            return self.bloch_coords[1]
        elif port_name == 'bloch_z':
            return self.bloch_coords[2]
        elif port_name == 'prob_0':
            # Probability of measuring |0>
            return abs(self.state[0])**2
        return None
        
    def get_display_image(self):
        w, h = 96, 96
        img = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Draw central cross (Bloch sphere axes visualization)
        cv2.line(img, (0, h//2), (w, h//2), 50, 1) # X-axis projection
        cv2.line(img, (w//2, 0), (w//2, h), 50, 1) # Z-axis projection
        
        # 2. Plot history trajectory
        # Scale coordinates from [-1, 1] to [4, 92] range
        scale = (w - 8) / 2
        offset = w // 2
        
        # Convert 3D Bloch coordinates to 2D screen projection (simple isometric view)
        x_proj = (self.bloch_coords_history[:, 0] - self.bloch_coords_history[:, 1] * 0.5) * scale * 0.8 + offset
        y_proj = (self.bloch_coords_history[:, 2] + self.bloch_coords_history[:, 1] * 0.5) * scale * -0.8 + offset
        
        # Draw trajectory
        for i in range(1, len(x_proj)):
            pt1 = (int(x_proj[i-1]), int(y_proj[i-1]))
            pt2 = (int(x_proj[i]), int(y_proj[i]))
            
            # Fade old points
            color = 120 + int(i / len(x_proj) * 135)
            cv2.line(img, pt1, pt2, color, 1)
            
        # 3. Plot current state (bright dot)
        cx, cy, cz = self.bloch_coords
        
        current_x_proj = int((cx - cy * 0.5) * scale * 0.8 + offset)
        current_y_proj = int((cz + cy * 0.5) * scale * -0.8 + offset)
        
        cv2.circle(img, (current_x_proj, current_y_proj), 3, 255, -1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Effective ħ (hbar)", "hbar", self.hbar, None),
        ]