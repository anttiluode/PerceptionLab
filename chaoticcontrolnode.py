"""
Chaotic Control Node - Simulates the Lorenz Attractor, a classic chaotic system.
It includes an input port ('control_nudge') to subtly influence the chaotic evolution,
testing if external signals can control the attractor's trajectory.
Ported from chaos_control_simulator (1).html.
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class ChaoticControlNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(150, 50, 50) # Chaotic Red
    
    def __init__(self, dt=0.01):
        super().__init__()
        self.node_title = "Chaotic Control (Lorenz)"
        
        self.inputs = {
            'control_nudge': 'signal',   # Input signal to influence the system
            'reset': 'signal'
        }
        self.outputs = {
            'chaos_x': 'signal',
            'chaos_y': 'signal',
            'phase_image': 'image',
        }
        
        # Lorenz Attractor parameters (standard values)
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8/3
        self.dt = float(dt)
        
        # System state (X, Y, Z)
        self.state = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
        # History for Phase Space Plot (X vs Y)
        self.history_len = 1000
        self.history_x = np.zeros(self.history_len, dtype=np.float64)
        self.history_y = np.zeros(self.history_len, dtype=np.float64)
        
        self.output_x = 0.0
        self.output_y = 0.0

    def _lorenz_derivative(self, state, nudge):
        """Lorenz system derivative with external nudge applied to dx/dt"""
        x, y, z = state
        sigma, rho, beta = self.sigma, self.rho, self.beta
        
        dx_dt = sigma * (y - x) + nudge # <--- CONTROL POINT
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])

    def _runge_kutta_4(self, state, nudge):
        """Standard RK4 numerical integration for the Lorenz system"""
        
        k1 = self._lorenz_derivative(state, nudge)
        
        state2 = state + 0.5 * self.dt * k1
        k2 = self._lorenz_derivative(state2, nudge)
        
        state3 = state + 0.5 * self.dt * k2
        k3 = self._lorenz_derivative(state3, nudge)
        
        state4 = state + self.dt * k3
        k4 = self._lorenz_derivative(state4, nudge)
        
        return state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def randomize(self):
        """Reset the system state to initial chaotic values"""
        self.state = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.history_x.fill(0.0)
        self.history_y.fill(0.0)
        
    def step(self):
        # 1. Get inputs
        control_nudge_in = self.get_blended_input('control_nudge', 'sum') or 0.0
        reset_sig = self.get_blended_input('reset', 'sum')
        
        if reset_sig is not None and reset_sig > 0.5:
            self.randomize()
            return

        # Map input signal [-1, 1] to a subtle control range [-0.5, 0.5]
        nudge_force = control_nudge_in * 0.5 
        
        # 2. Integrate the system
        self.state = self._runge_kutta_4(self.state, nudge_force)
        
        # 3. Update outputs and history
        self.output_x = self.state[0]
        self.output_y = self.state[1]
        
        self.history_x[:-1] = self.history_x[1:]
        self.history_x[-1] = self.output_x
        
        self.history_y[:-1] = self.history_y[1:]
        self.history_y[-1] = self.output_y

    def get_output(self, port_name):
        if port_name == 'chaos_x':
            return self.output_x
        elif port_name == 'chaos_y':
            return self.output_y
        elif port_name == 'phase_image':
            # This output is generated in get_display_image for efficiency
            # We return a dummy value so the port is active, or use get_display_image directly
            return np.zeros((64, 64), dtype=np.float32) 
        return None
        
    def get_display_image(self):
        w, h = 96, 96
        img = np.zeros((h, w), dtype=np.uint8)
        
        if self.history_x.max() == 0:
            return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

        # 1. Normalize and Scale History for Plotting
        # Find bounds for scaling
        min_val_x, max_val_x = self.history_x.min(), self.history_x.max()
        range_x = max_val_x - min_val_x
        
        min_val_y, max_val_y = self.history_y.min(), self.history_y.max()
        range_y = max_val_y - min_val_y

        # Define plotting area margins
        margin = 8
        scale_x = (w - 2 * margin) / (range_x + 1e-9)
        scale_y = (h - 2 * margin) / (range_y + 1e-9)

        # Map trajectory points to screen coordinates
        x_coords = ((self.history_x - min_val_x) * scale_x + margin).astype(int)
        # Flip Y-axis (top is 0)
        y_coords = (h - margin - (self.history_y - min_val_y) * scale_y).astype(int)
        
        # 2. Draw Trajectory (X vs Y Phase Space)
        for i in range(1, self.history_len):
            pt1 = (x_coords[i-1], y_coords[i-1])
            pt2 = (x_coords[i], y_coords[i])
            
            # Draw faded line
            color = 50 + int(i / self.history_len * 200)
            cv2.line(img, pt1, pt2, color, 1)

        # 3. Draw current point (Attractor)
        if self.history_len > 0:
            current_pt = (x_coords[-1], y_coords[-1])
            cv2.circle(img, current_pt, 2, 255, -1)
            
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Integration dt", "dt", self.dt, None),
            ("Sigma (σ)", "sigma", self.sigma, None),
            ("Rho (ρ)", "rho", self.rho, None),
            ("Beta (β)", "beta", self.beta, None),
        ]