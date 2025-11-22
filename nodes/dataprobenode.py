"""
Data Probe Node - Visualizes signal data over time.
Acts as an oscilloscope to debug signal flows.
"""

import numpy as np
import cv2
from collections import deque
from PyQt6 import QtGui  # ✅ FIXED: Direct import instead of from __main__
import __main__

BaseNode = __main__.BaseNode

class DataProbeNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(50, 50, 200) # Probe Blue
    
    def __init__(self, history_length=100):
        super().__init__()
        self.node_title = "Data Probe"
        
        self.inputs = {
            'signal_in': 'signal'
        }
        
        self.outputs = {
            'visual_plot': 'image'
        }
        
        self.history_length = int(history_length)
        self.data_buffer = deque(maxlen=self.history_length)
        
        # Initialize buffer with zeros
        for _ in range(self.history_length):
            self.data_buffer.append(0.0)
            
        self.display_img = np.zeros((128, 256, 3), dtype=np.uint8)
        self.min_val = -1.0
        self.max_val = 1.0

    def step(self):
        # Get input signal
        val = self.get_blended_input('signal_in', 'sum')
        
        if val is None:
            val = 0.0
            
        self.data_buffer.append(float(val))
        
        # Render the plot
        self._render_plot()
        
    def _render_plot(self):
        # Clear image
        self.display_img.fill(20) # Dark gray background
        
        h, w, _ = self.display_img.shape
        
        # Convert buffer to numpy array
        data = np.array(self.data_buffer)
        
        # Dynamic scaling (optional, keeps the wave centered)
        current_min = np.min(data)
        current_max = np.max(data)
        
        # Smoothly adjust display range
        self.min_val = self.min_val * 0.95 + current_min * 0.05
        self.max_val = self.max_val * 0.95 + current_max * 0.05
        
        # Avoid division by zero
        if abs(self.max_val - self.min_val) < 0.001:
            scale = 1.0
        else:
            scale = (h - 20) / (self.max_val - self.min_val)
            
        # Map data to screen coordinates
        # Y-axis is inverted (0 is top)
        y_coords = h/2 - (data - (self.max_val + self.min_val)/2) * scale
        x_coords = np.linspace(0, w, len(data))
        
        # Create points for polylines
        points = np.column_stack((x_coords, y_coords)).astype(np.int32)
        
        # Draw the line
        cv2.polylines(self.display_img, [points], False, (0, 255, 255), 2)
        
        # Draw zero line
        zero_y = int(h/2 + (self.max_val + self.min_val)/2 * scale)
        if 0 <= zero_y < h:
            cv2.line(self.display_img, (0, zero_y), (w, zero_y), (100, 100, 100), 1)
            
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.display_img, f"Max: {self.max_val:.2f}", (5, 20), font, 0.5, (200, 200, 200), 1)
        cv2.putText(self.display_img, f"Min: {self.min_val:.2f}", (5, h-10), font, 0.5, (200, 200, 200), 1)
        cv2.putText(self.display_img, f"Cur: {data[-1]:.4f}", (w-100, 20), font, 0.5, (0, 255, 0), 1)

    def get_output(self, port_name):
        if port_name == 'visual_plot':
            return self.display_img.astype(np.float32) / 255.0
        return None
        
    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None)
        ]