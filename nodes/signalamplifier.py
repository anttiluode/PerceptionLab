"""
Signal Amplifier Node
---------------------
A simple utility to multiply an incoming signal by a gain factor.

This is perfect for "quiet" signals (like constraint_violation)
that need to be "louder" to be seen on a plotter
next to "loud" signals (like fractal_dimension).
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SignalAmplifierNode(BaseNode):
    NODE_CATEGORY = "Utilities"
    NODE_COLOR = QtGui.QColor(150, 150, 150)  # Gray
    
    def __init__(self, gain=10.0):
        super().__init__()
        self.node_title = "Signal Amplifier"
        
        self.inputs = {
            'signal_in': 'signal',
        }
        self.outputs = {
            'signal_out': 'signal',
        }
        
        self.gain = float(gain)
        self.output_value = 0.0
        
    def step(self):
        signal_in = self.get_blended_input('signal_in', 'sum')
        
        if signal_in is None:
            self.output_value = 0.0
        else:
            self.output_value = float(signal_in) * self.gain
            
    def get_output(self, port_name):
        if port_name == 'signal_out':
            return self.output_value
        return None

    def get_display_image(self):
        display = np.zeros((180, 200, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(display, f"In: {self.get_blended_input('signal_in', 'sum') or 0.0:.4f}", 
                   (10, 40), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(display, f"Gain: x{self.gain}", 
                   (10, 80), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(display, f"Out: {self.output_value:.4f}", 
                   (10, 130), font, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
        
        img_resized = np.ascontiguousarray(display)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Gain", "gain", self.gain, None),
        ]