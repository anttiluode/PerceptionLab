"""
Debug Data Exporter
-------------------
Records signals to a CSV file for analysis.
Columns: Time, InputA, InputB, Target, Prediction
"""

import numpy as np
import cv2
import csv
import time
import os
from PyQt6 import QtGui  # ✅ FIXED
import __main__

BaseNode = __main__.BaseNode

class DebugDataExporterNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(200, 50, 50) # Red
    
    def __init__(self):
        super().__init__()
        self.node_title = "Data Exporter (CSV)"
        
        self.inputs = {
            'input_a': 'signal',
            'input_b': 'signal',
            'target': 'signal',
            'prediction': 'signal',
            'save_trigger': 'signal'
        }
        self.outputs = {}
        
        self.data_buffer = []
        self.max_buffer = 1000
        self.start_time = time.time()
        self.last_trigger = 0.0
        
        # Make file path explicit and visible
        self.output_path = os.path.join(os.getcwd(), "debug_data.csv")
        
    def step(self):
        # Collect
        a = self.get_blended_input('input_a', 'sum') or 0.0
        b = self.get_blended_input('input_b', 'sum') or 0.0
        tgt = self.get_blended_input('target', 'sum') or 0.0
        pred = self.get_blended_input('prediction', 'sum') or 0.0
        trig = self.get_blended_input('save_trigger', 'sum') or 0.0
        
        t = time.time() - self.start_time
        
        # Record row
        self.data_buffer.append([t, a, b, tgt, pred])
        if len(self.data_buffer) > self.max_buffer:
            self.data_buffer.pop(0)
            
        # Save on trigger (rising edge detection)
        if trig > 0.5 and self.last_trigger <= 0.5:
            self.save_to_csv()
            
        self.last_trigger = trig
        
    def save_to_csv(self):
        try:
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "InputA", "InputB", "Target", "Prediction"])
                writer.writerows(self.data_buffer)
            print(f"✅ Saved {len(self.data_buffer)} rows to: {self.output_path}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
            print(f"   Attempted path: {self.output_path}")
            
    def get_display_image(self):
        # Simple status display
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        msg = f"Rows: {len(self.data_buffer)}"
        cv2.putText(img, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return QtGui.QImage(img.data, 128, 64, 128*3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        """Allow user to see/change output path"""
        return [
            ("Output Path", "output_path", self.output_path, None)
        ]