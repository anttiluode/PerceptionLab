import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class FitzHughNagumoNode(BaseNode):
    """
    Simulates a FitzHugh-Nagumo neuron model.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 180, 180)

    def __init__(self, a=0.7, b=0.8, tau=12.5):
        super().__init__()
        self.node_title = "FHN Neuron"
        
        self.inputs = {'pain_stimulus': 'signal'}
        self.outputs = {
            'pain_out': 'signal',
            'stability_metric': 'signal'
        }
        
        self.a = float(a)
        self.b = float(b)
        self.tau = float(tau)
        
        self.v = 0.0  # Membrane potential ("pain")
        self.w = 0.0  # Recovery variable ("stability")
        self.dt = 0.1 # Simulation time step

    def step(self):
        # Get input current
        I = self.get_blended_input('pain_stimulus', 'sum') or 0.0
        
        # Model equations (Euler integration)
        dv = self.v - (self.v**3 / 3) - self.w + I
        dw = (self.v + self.a - self.b * self.w) / self.tau
        
        self.v += dv * self.dt
        self.w += dw * self.dt
        
        # Clamp values to prevent explosion
        self.v = np.clip(self.v, -5, 5)
        self.w = np.clip(self.w, -5, 5)

    def get_output(self, port_name):
        if port_name == 'pain_out':
            return self.v
        if port_name == 'stability_metric':
            return self.w
        return None

    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map v and w to screen
        v_y = int(h/2 - (self.v / 3.0) * (h/2))
        w_y = int(h/2 - (self.w / 3.0) * (h/2))
        
        cv2.circle(img, (w//2, v_y), 8, (0, 255, 255), -1) # 'v' (pain)
        cv2.circle(img, (w//2, w_y), 4, (255, 0, 0), -1)   # 'w' (stability)

        cv2.line(img, (0, h//2), (w, h//2), (50,50,50), 1)
        
        cv2.putText(img, f"Pain (v): {self.v:.2f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img, f"Stability (w): {self.w:.2f}", (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("a", "a", self.a, None),
            ("b", "b", self.b, None),
            ("tau", "tau", self.tau, None)
        ]