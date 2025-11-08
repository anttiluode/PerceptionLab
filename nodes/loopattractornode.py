"""
Loop Attractor Node - A chaotic system with self-sustaining oscillations
Place this file in the 'nodes' folder as 'loopattractornode.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class LoopAttractorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(180, 60, 120)
    
    def __init__(self, dt=0.01, a=10.0, b=8/3, c=28.0):
        super().__init__()
        self.node_title = "Loop Attractor"
        
        self.inputs = {
            'perturbation': 'signal',
            'parameter_a': 'signal',
            'parameter_c': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'x_out': 'signal',
            'y_out': 'signal',
            'z_out': 'signal',
            'phase_image': 'image',
            'energy': 'signal'
        }
        
        self.dt = float(dt)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        
        self.x = 1.0
        self.y = 1.0
        self.z = 1.0
        
        self.history_len = 500
        self.history_x = np.zeros(self.history_len, dtype=np.float32)
        self.history_y = np.zeros(self.history_len, dtype=np.float32)
        self.history_z = np.zeros(self.history_len, dtype=np.float32)
        
        self.loop_phase = 0.0
        self.loop_amplitude = 1.0
        self.last_reset = 0.0
        
    def loop_dynamics(self, x, y, z, perturbation=0.0):
        dx = self.a * (y - x)
        dy = x * (self.c - z) - y
        dz = x * y - self.b * z
        
        loop_force = 0.5 * np.sin(self.loop_phase)
        dx += loop_force * y
        dy += loop_force * (-x)
        dx += perturbation
        
        self.loop_phase += 0.05 * np.sqrt(x*x + y*y + z*z + 0.01)
        self.loop_phase = self.loop_phase % (2 * np.pi)
        
        return dx, dy, dz
    
    def runge_kutta_4(self, x, y, z, perturbation=0.0):
        dx1, dy1, dz1 = self.loop_dynamics(x, y, z, perturbation)
        
        dx2, dy2, dz2 = self.loop_dynamics(
            x + 0.5*self.dt*dx1,
            y + 0.5*self.dt*dy1,
            z + 0.5*self.dt*dz1,
            perturbation
        )
        
        dx3, dy3, dz3 = self.loop_dynamics(
            x + 0.5*self.dt*dx2,
            y + 0.5*self.dt*dy2,
            z + 0.5*self.dt*dz2,
            perturbation
        )
        
        dx4, dy4, dz4 = self.loop_dynamics(
            x + self.dt*dx3,
            y + self.dt*dy3,
            z + self.dt*dz3,
            perturbation
        )
        
        new_x = x + (self.dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        new_y = y + (self.dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        new_z = z + (self.dt / 6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)
        
        return new_x, new_y, new_z
    
    def randomize(self):
        self.x = np.random.uniform(-5, 5)
        self.y = np.random.uniform(-5, 5)
        self.z = np.random.uniform(0, 30)
        self.loop_phase = np.random.uniform(0, 2*np.pi)
        self.history_x.fill(0)
        self.history_y.fill(0)
        self.history_z.fill(0)
        
    def step(self):
        perturbation = self.get_blended_input('perturbation', 'sum') or 0.0
        param_a = self.get_blended_input('parameter_a', 'sum')
        param_c = self.get_blended_input('parameter_c', 'sum')
        reset_sig = self.get_blended_input('reset', 'sum') or 0.0
        
        if reset_sig > 0.5 and self.last_reset <= 0.5:
            self.randomize()
        self.last_reset = reset_sig
        
        if param_a is not None:
            self.a = 10.0 + param_a * 5.0
        if param_c is not None:
            self.c = 30.0 + param_c * 10.0
        
        perturbation *= 5.0
        
        self.x, self.y, self.z = self.runge_kutta_4(self.x, self.y, self.z, perturbation)
        
        max_val = 100.0
        if abs(self.x) > max_val or abs(self.y) > max_val or abs(self.z) > max_val:
            self.randomize()
        
        self.history_x[:-1] = self.history_x[1:]
        self.history_x[-1] = self.x
        
        self.history_y[:-1] = self.history_y[1:]
        self.history_y[-1] = self.y
        
        self.history_z[:-1] = self.history_z[1:]
        self.history_z[-1] = self.z
        
    def get_output(self, port_name):
        if port_name == 'x_out':
            return np.tanh(self.x / 10.0)
        elif port_name == 'y_out':
            return np.tanh(self.y / 10.0)
        elif port_name == 'z_out':
            return np.tanh(self.z / 20.0)
        elif port_name == 'energy':
            return np.sqrt(self.x**2 + self.y**2 + self.z**2) / 30.0
        elif port_name == 'phase_image':
            return self.generate_phase_image()
        return None
    
    def generate_phase_image(self):
        w, h = 96, 96
        img = np.zeros((h, w), dtype=np.float32)
        
        if len(self.history_x) == 0:
            return img
        
        x_min, x_max = self.history_x.min(), self.history_x.max()
        y_min, y_max = self.history_y.min(), self.history_y.max()
        
        x_range = x_max - x_min + 1e-9
        y_range = y_max - y_min + 1e-9
        
        margin = 8
        x_coords = ((self.history_x - x_min) / x_range * (w - 2*margin) + margin).astype(int)
        y_coords = ((self.history_y - y_min) / y_range * (h - 2*margin) + margin).astype(int)
        
        y_coords = h - 1 - y_coords
        
        x_coords = np.clip(x_coords, 0, w-1)
        y_coords = np.clip(y_coords, 0, h-1)
        
        for i in range(1, len(x_coords)):
            intensity = i / len(x_coords)
            img[y_coords[i], x_coords[i]] = intensity
        
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
        
    def get_display_image(self):
        phase_img = self.generate_phase_image()
        
        img_u8 = (np.clip(phase_img, 0, 1) * 255).astype(np.uint8)
        
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_PLASMA)
        
        w, h = 96, 96
        x_min, x_max = self.history_x.min(), self.history_x.max()
        y_min, y_max = self.history_y.min(), self.history_y.max()
        x_range = x_max - x_min + 1e-9
        y_range = y_max - y_min + 1e-9
        
        margin = 8
        curr_x = int((self.x - x_min) / x_range * (w - 2*margin) + margin)
        curr_y = int((self.y - y_min) / y_range * (h - 2*margin) + margin)
        curr_y = h - 1 - curr_y
        
        curr_x = np.clip(curr_x, 0, w-1)
        curr_y = np.clip(curr_y, 0, h-1)
        
        cv2.circle(img_color, (curr_x, curr_y), 3, (255, 255, 255), -1)
        
        center = (w - 12, 12)
        radius = 8
        angle = int(np.degrees(self.loop_phase))
        cv2.ellipse(img_color, center, (radius, radius), 0, 0, angle, (0, 255, 255), 2)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Time Step (dt)", "dt", self.dt, None),
            ("Parameter A (speed)", "a", self.a, None),
            ("Parameter B (dissipation)", "b", self.b, None),
            ("Parameter C (size)", "c", self.c, None),
        ]