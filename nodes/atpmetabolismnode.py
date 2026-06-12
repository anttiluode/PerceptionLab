"""
ATP Metabolism Node (The Fatigue Oscillator)
============================================
Tracks energy demand from the cortex and maintains an ATP pool.
Creates a biological relaxation oscillator via hysteresis:
- High demand burns ATP.
- When ATP hits low_threshold, forces a REST state.
- During REST, ATP recovers, and demand drops until high_threshold is reached.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ATPMetabolismNode(BaseNode):
    NODE_CATEGORY = "Metabolism"
    NODE_COLOR = QtGui.QColor(200, 80, 40) # Mitochondria Red

    def __init__(self, burn_rate=0.08, recovery_rate=0.03):
        super().__init__()
        self.node_title = "ATP Metabolism"
        self.inputs = {'demand': 'signal'}
        self.outputs = {'atp_level': 'signal', 'resting': 'signal'}
        
        self.burn_rate = float(burn_rate)
        self.recovery_rate = float(recovery_rate)
        
        self.atp = 1.0
        self.resting = False
        self.display_img = np.zeros((64, 128, 3), dtype=np.uint8)

    def step(self):
        demand = self.get_blended_input('demand', 'sum') or 0.0

        # Hysteresis Logic
        if self.resting:
            self.atp += self.recovery_rate
            if self.atp >= 1.0:
                self.atp = 1.0
                self.resting = False # Wake up
        else:
            self.atp -= self.burn_rate * demand
            if self.atp <= 0.1:
                self.atp = 0.1
                self.resting = True # Force rest

        self._render_ui()

    def _render_ui(self):
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        
        # ATP Bar
        bar_w = int(self.atp * 120)
        color = (0, 0, 255) if self.resting else (0, 255, 0)
        cv2.rectangle(img, (4, 20), (4 + bar_w, 30), color, -1)
        cv2.rectangle(img, (4, 20), (124, 30), (100, 100, 100), 1)

        state_str = "RESTING" if self.resting else "RETRIEVING"
        cv2.putText(img, f"ATP: {self.atp:.2f} ({state_str})", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        self.display_img = img

    def get_output(self, port_name):
        if port_name == 'atp_level': return float(self.atp)
        if port_name == 'resting': return float(self.resting)
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 128, 64, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Burn Rate", "burn_rate", self.burn_rate, "float"),
            ("Recovery Rate", "recovery_rate", self.recovery_rate, "float")
        ]
    
    def set_config_options(self, options):
        if "burn_rate" in options: self.burn_rate = float(options["burn_rate"])
        if "recovery_rate" in options: self.recovery_rate = float(options["recovery_rate"])