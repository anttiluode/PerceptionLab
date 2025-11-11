import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2
import time

class SpringNode(BaseNode):
    """
    Simulates a 1D damped spring.
    F = -k*x - c*v
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 180, 100)

    def __init__(self, mass=1.0, stiffness=0.1, damping=0.05):
        super().__init__()
        self.node_title = "Spring (1D)"
        
        self.inputs = {'target_pos': 'signal'}
        self.outputs = {'position': 'signal'}
        
        self.mass = float(mass)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        
        self.position = 0.0
        self.velocity = 0.0
        self.last_time = time.time()

    def step(self):
        # Calculate delta time
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0.1: # Clamp large timesteps (e.g., on load)
            dt = 0.1
        self.last_time = current_time

        # Get target
        target = self.get_blended_input('target_pos', 'sum') or 0.0
        
        # Calculate forces
        displacement = self.position - target
        spring_force = -self.stiffness * displacement
        damping_force = -self.damping * self.velocity
        total_force = spring_force + damping_force
        
        # Update physics (Euler integration)
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def get_output(self, port_name):
        if port_name == 'position':
            return self.position
        return None

    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw position
        pos_x = int(np.clip((self.position + 2) / 4.0, 0, 1) * w)
        cv2.circle(img, (pos_x, h//2), 10, (0, 255, 0), -1)
        
        # Draw target
        target = self.get_blended_input('target_pos', 'sum') or 0.0
        target_x = int(np.clip((target + 2) / 4.0, 0, 1) * w)
        cv2.circle(img, (target_x, h//2), 5, (0, 0, 255), -1)
        
        cv2.putText(img, f"Pos: {self.position:.2f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"Vel: {self.velocity:.2f}", (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Mass", "mass", self.mass, None),
            ("Stiffness", "stiffness", self.stiffness, None),
            ("Damping", "damping", self.damping, None)
        ]