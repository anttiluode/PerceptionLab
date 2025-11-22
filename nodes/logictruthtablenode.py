"""
Logic Truth Table Node
----------------------
Visualizes the learned logic of your network.
It monitors Input A and Input B, categorizes the state (00, 01, 10, 11),
and records the average 'Prediction' value for that state.

This stabilizes the view: instead of watching cycling numbers, you see
the stable "Logic Table" the network has learned.
"""

import numpy as np
import cv2
from PyQt6 import QtGui  # ✅ FIXED: Direct import instead of from __main__
import __main__

BaseNode = __main__.BaseNode

class LogicTruthTableNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(0, 150, 150) # Teal
    
    def __init__(self):
        super().__init__()
        self.node_title = "Logic Truth Table"
        
        self.inputs = {
            'input_a': 'signal',
            'input_b': 'signal',
            'prediction': 'signal'
        }
        self.outputs = {
            'table_image': 'image'
        }
        
        # Storage for the 4 states: [00, 01, 10, 11]
        # Format: [sum_values, count]
        self.states = {
            (0, 0): [0.0, 0],
            (0, 1): [0.0, 0],
            (1, 0): [0.0, 0],
            (1, 1): [0.0, 0]
        }
        
        self.display_img = np.zeros((256, 256, 3), dtype=np.uint8)
        self.reset_counter = 0

    def step(self):
        # Get signals
        a = self.get_blended_input('input_a', 'sum') or 0.0
        b = self.get_blended_input('input_b', 'sum') or 0.0
        pred = self.get_blended_input('prediction', 'sum') or 0.0
        
        # Quantize Inputs (Threshold at 0.5)
        state_a = 1 if a > 0.5 else 0
        state_b = 1 if b > 0.5 else 0
        key = (state_a, state_b)
        
        # Accumulate (EMA smoothing for stability)
        current_avg = 0.0
        if self.states[key][1] > 0:
            current_avg = self.states[key][0] / self.states[key][1]
            
        # Smooth update: 95% old + 5% new
        new_avg = current_avg * 0.95 + pred * 0.05
        
        # We store it back as (new_avg, 1) effectively resetting count to keep it moving
        self.states[key] = [new_avg, 1]
        
        self._render_table()
        
    def _render_table(self):
        # Draw 2x2 grid
        h, w, _ = self.display_img.shape
        half_w, half_h = w // 2, h // 2
        
        # Clear
        self.display_img.fill(0)
        
        # Define quadrants:
        # 0,0 (Top Left) | 0,1 (Top Right)
        # 1,0 (Bot Left) | 1,1 (Bot Right) -- Wait, usually tables are Input based
        # Let's do: A is Rows, B is Cols? 
        # Standard Logic Table:
        #      B=0   B=1
        # A=0 [0,0] [0,1]
        # A=1 [1,0] [1,1]
        
        positions = {
            (0, 0): (0, 0),
            (0, 1): (half_w, 0),
            (1, 0): (0, half_h),
            (1, 1): (half_w, half_h)
        }
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for key, (val_sum, count) in self.states.items():
            avg = val_sum / count if count > 0 else 0.0
            
            x, y = positions[key]
            
            # Draw Background Color based on Value (Black -> Green)
            brightness = int(np.clip(avg, 0, 1) * 255)
            color = (0, brightness, 0) # Green
            
            cv2.rectangle(self.display_img, (x, y), (x + half_w, y + half_h), color, -1)
            cv2.rectangle(self.display_img, (x, y), (x + half_w, y + half_h), (100, 100, 100), 1) # Border
            
            # Draw Text
            label = f"{key}: {avg:.2f}"
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            
            cv2.putText(self.display_img, label, (x + 10, y + half_h // 2), font, 0.6, text_color, 2)

    def get_output(self, port_name):
        if port_name == 'table_image':
            return self.display_img.astype(np.float32) / 255.0
        return None
        
    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return []