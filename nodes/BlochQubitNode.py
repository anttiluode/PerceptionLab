"""
Bloch Qubit Node - The Quantum Core
-----------------------------------
Simulates a qubit.
Outputs X, Y, Z coordinates explicitly for wiring into other nodes.
"""

import numpy as np
import cv2
from scipy.linalg import expm 

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

# Pauli Matrices
H_Y = np.array([[0, -1j], [1j, 0]], dtype=complex) * 0.5
H_Z = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5

class BlochQubitNode(BaseNode):
    NODE_CATEGORY = "Quantum"
    NODE_COLOR = QtGui.QColor(100, 0, 255)

    def __init__(self):
        super().__init__()
        self.node_title = "Bloch Qubit"
        
        self.inputs = {
            'ry_angle': 'signal', # Driven by Brain Error
            'rz_angle': 'signal'
        }
        
        self.outputs = {
            'bloch_x': 'signal', # The Superposition Signal
            'bloch_y': 'signal',
            'bloch_z': 'signal',
            'qubit_state': 'spectrum'
        }
        
        self.state = np.array([1, 0], dtype=complex)
        self.coords = (0.0, 0.0, 1.0)

    def step(self):
        # 1. Get Inputs
        theta_y = self.get_blended_input('ry_angle', 'sum')
        if theta_y is None: theta_y = 0.0
        
        # 2. Rotate |0>
        # Ry rotation moves state in X-Z plane
        U_y = expm(-1j * theta_y * H_Y)
        basis = np.array([1, 0], dtype=complex)
        self.state = U_y @ basis
        
        # 3. Calculate Coordinates
        # alpha, beta
        a, b = self.state[0], self.state[1]
        
        # Bloch Sphere mapping
        # FIX: Used np.conj instead of np.conjug
        x = 2 * (a * np.conj(b)).real
        y = 2 * (a * np.conj(b)).imag
        z = (np.abs(a)**2 - np.abs(b)**2)
        
        self.coords = (float(x), float(y), float(z))

    def get_output(self, port_name):
        if port_name == 'bloch_x': return self.coords[0]
        if port_name == 'bloch_y': return self.coords[1]
        if port_name == 'bloch_z': return self.coords[2]
        return None

    def get_display_image(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        c, r = (100, 100), 80
        
        # Draw Sphere
        cv2.circle(img, c, r, (50, 50, 50), 1)
        
        # Draw Vector
        x, y, z = self.coords
        px = int(c[0] + x * r)
        py = int(c[1] - z * r)
        
        color = (0, 255, 0)
        if abs(x) > 0.5: color = (0, 255, 255) # Yellow = Superposition
        
        cv2.line(img, c, (px, py), color, 2)
        cv2.putText(img, f"X: {x:.2f}", (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        return QtGui.QImage(img.data, 200, 200, 200*3, QtGui.QImage.Format.Format_RGB888)