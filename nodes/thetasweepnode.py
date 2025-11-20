# thetasweepnode.py
"""
Theta Sweep Node (The Alternator) - FIXED
---------------------------------
Implements the "Left-Right-Alternating" logic from Vollan et al. (2025).
Instead of summing inputs (which creates noise), it rapidly switches 
between them driven by a Theta Phase.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
import time

class ThetaSweepNode(BaseNode):
    NODE_CATEGORY = "Dynamics"
    NODE_COLOR = QtGui.QColor(200, 180, 50) # Theta Gold

    def __init__(self, theta_hz=8.0):
        super().__init__()
        self.node_title = "Theta Sweep (Alternator)"
        
        self.inputs = {
            'input_a': 'image',       # Reality A (e.g., Left Path)
            'input_b': 'image',       # Reality B (e.g., Right Path)
            'theta_drive': 'signal'   # Optional external Theta wave
        }
        
        self.outputs = {
            'swept_output': 'image',  # The Alternating Signal
            'current_phase': 'signal' # +1 for A, -1 for B
        }
        
        self.theta_hz = float(theta_hz)
        self.phase = 0.0
        self.last_time = None
        
        # Internal State
        self.current_output = None
        self.current_phase_val = 0.0
        self.active_channel = "Init"

    def step(self):
        # 1. Manage Time & Theta
        # Use standard time.time() for robustness
        if self.last_time is None:
            self.last_time = time.time()
            dt = 0.0
        else:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

        # Update Phase
        # We use the input signal if connected, otherwise internal clock
        ext_drive = self.get_blended_input('theta_drive', 'sum')
        
        if ext_drive is not None:
            # External drive (e.g. from Oscillator)
            # We check the sign: Positive = A, Negative = B
            val = ext_drive
        else:
            # Internal Clock (8Hz default)
            self.phase += self.theta_hz * 2 * np.pi * dt
            val = np.sin(self.phase)

        # 2. The Sweep Logic (The Commutator)
        img_a = self.get_blended_input('input_a', 'first')
        img_b = self.get_blended_input('input_b', 'first')
        
        # Handle missing inputs safely
        # If one is missing, replace it with zeros of the other's shape
        if img_a is None and img_b is None:
            self.current_output = None
            return
        
        if img_a is None: 
            img_a = np.zeros_like(img_b)
        if img_b is None: 
            img_b = np.zeros_like(img_a)

        # 3. The Hard Switch (Vollan et al. Logic)
        # The paper says it's not a blend; it's a discrete alternation.
        if val >= 0:
            self.current_output = img_a
            self.active_channel = "A (Positive)"
            self.current_phase_val = 1.0
        else:
            self.current_output = img_b
            self.active_channel = "B (Negative)"
            self.current_phase_val = -1.0

    def get_output(self, port_name):
        if port_name == 'swept_output':
            return self.current_output
        elif port_name == 'current_phase':
            return self.current_phase_val
        return None

    def get_display_image(self):
        if self.current_output is None: return None
        
        img_u8 = (np.clip(self.current_output, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        
        # Overlay Status
        cv2.putText(img_color, f"Active: {self.active_channel}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return QtGui.QImage(img_color.data, img_color.shape[1], img_color.shape[0], 
                           img_color.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
                           
    def get_config_options(self):
        return [("Theta Freq (Hz)", "theta_hz", self.theta_hz, 'float')]