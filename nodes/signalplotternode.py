"""
Signal Plotter Node
-------------------
Logs and plots multiple signal inputs over time.
Perfect for correlating "fractal_dimension" and "constraint_violation".
"""

import numpy as np
import cv2
from collections import deque
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SignalPlotterNode(BaseNode):
    NODE_CATEGORY = "Analyzers"
    NODE_COLOR = QtGui.QColor(200, 100, 0)  # Orange
    
    def __init__(self, history_size=500):
        super().__init__()
        self.node_title = "Signal Plotter"
        
        self.inputs = {
            'signal_A (Red)': 'signal',
            'signal_B (Green)': 'signal',
            'signal_C (Blue)': 'signal',
        }
        self.outputs = {
            'plot_image': 'image',
        }
        
        self.history_size = int(history_size)
        
        self.data = {
            'A': deque(maxlen=self.history_size),
            'B': deque(maxlen=self.history_size),
            'C': deque(maxlen=self.history_size),
        }
        
        self.plot_image = np.zeros((256, self.history_size, 3), dtype=np.uint8)
        self.colors = {
            'A': (255, 0, 0),  # Red
            'B': (0, 255, 0),  # Green
            'C': (0, 0, 255),  # Blue
        }
        
        self.min_val = 0.0
        self.max_val = 1.0

    def step(self):
        # Get data
        sig_a = self.get_blended_input('signal_A (Red)', 'sum')
        sig_b = self.get_blended_input('signal_B (Green)', 'sum')
        sig_c = self.get_blended_input('signal_C (Blue)', 'sum')
        
        # Store data
        if sig_a is not None:
            self.data['A'].append(sig_a)
        if sig_b is not None:
            self.data['B'].append(sig_b)
        if sig_c is not None:
            self.data['C'].append(sig_c)
            
        # Auto-range
        all_vals = list(self.data['A']) + list(self.data['B']) + list(self.data['C'])
        if all_vals:
            self.min_val = min(all_vals)
            self.max_val = max(all_vals)
            if self.max_val == self.min_val:
                self.max_val += 0.1

        # Draw plot
        self.plot_image.fill(0)
        h, w = self.plot_image.shape[:2]
        
        for key, color in self.colors.items():
            points = np.array(list(self.data[key]))
            if len(points) < 2:
                continue
            
            # Normalize points
            norm_points = (points - self.min_val) / (self.max_val - self.min_val + 1e-6)
            y_coords = h - 1 - (norm_points * (h - 1)).astype(int)
            x_coords = np.linspace(w - len(points), w - 1, len(points)).astype(int)
            
            pts = np.vstack((x_coords, y_coords)).T
            cv2.polylines(self.plot_image, [pts], isClosed=False, color=color, thickness=1)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.plot_image, f"Max: {self.max_val:.4f}", (10, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.plot_image, f"Min: {self.min_val:.4f}", (10, h - 10), font, 0.4, (255, 255, 255), 1)

    def get_output(self, port_name):
        if port_name == 'plot_image':
            return self.plot_image
        return None

    def get_display_image(self):
        img_resized = np.ascontiguousarray(self.plot_image)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("History Size", "history_size", self.history_size, None),
        ]