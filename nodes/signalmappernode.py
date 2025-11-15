"""
Signal Mapper Node
------------------
Maps input signal from one range to another.
Useful for converting fractal dimension (1.0-2.0) to learning rate (0.001-0.01)
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class SignalMapperNode(BaseNode):
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(120, 120, 120)
    
    def __init__(self, input_min=1.0, input_max=2.0, output_min=0.001, output_max=0.01):
        super().__init__()
        self.node_title = "Signal Mapper"
        
        self.inputs = {
            'signal_in': 'signal',
        }
        
        self.outputs = {
            'signal_out': 'signal',
        }
        
        # Configurable mapping
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        
        self.output_value = 0.0
    
    def step(self):
        signal_in = self.get_blended_input('signal_in', 'sum')
        
        if signal_in is None:
            self.output_value = self.output_min
            return
        
        # Clamp to input range
        clamped = np.clip(signal_in, self.input_min, self.input_max)
        
        # Normalize to 0-1
        if self.input_max > self.input_min:
            normalized = (clamped - self.input_min) / (self.input_max - self.input_min)
        else:
            normalized = 0.5
        
        # Map to output range
        self.output_value = self.output_min + normalized * (self.output_max - self.output_min)
    
    def get_output(self, port_name):
        if port_name == 'signal_out':
            return self.output_value
        return None
    
    def get_display_image(self):
        w, h = 128, 96
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw input/output bars
        signal_in = self.get_blended_input('signal_in', 'sum') or 0.0
        
        # Input bar (top half)
        if self.input_max > self.input_min:
            in_normalized = (signal_in - self.input_min) / (self.input_max - self.input_min)
            in_normalized = np.clip(in_normalized, 0, 1)
        else:
            in_normalized = 0.5
            
        in_bar_w = int(in_normalized * w)
        cv2.rectangle(display, (0, 0), (in_bar_w, h//2 - 5), (255, 100, 0), -1)
        
        # Output bar (bottom half)
        if self.output_max > self.output_min:
            out_normalized = (self.output_value - self.output_min) / (self.output_max - self.output_min)
        else:
            out_normalized = 0.5
            
        out_bar_w = int(out_normalized * w)
        cv2.rectangle(display, (0, h//2 + 5), (out_bar_w, h), (0, 255, 100), -1)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, f'In: {signal_in:.3f}', (5, 15), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display, f'Out: {self.output_value:.4f}', (5, h - 5), font, 0.3, (255, 255, 255), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Input Min", "input_min", self.input_min, None),
            ("Input Max", "input_max", self.input_max, None),
            ("Output Min", "output_min", self.output_min, None),
            ("Output Max", "output_max", self.output_max, None),
        ]