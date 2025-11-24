"""
Debug Data Exporter - FIXED VERSION
------------------------------------
Records signals to a CSV file for analysis.
Columns: Time, InputA, InputB, Target, Prediction

USAGE:
1. Connect Brain.error -> input_a
2. Connect Qubit.velocity -> input_b
3. Connect ButtonNode -> save_trigger (or ConstantSignal=1.0 for auto-save)
"""

import numpy as np
import cv2
import csv
import time
import os
from PyQt6 import QtGui
import __main__

BaseNode = __main__.BaseNode

class DebugDataExporterNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(200, 50, 50) # Red
    
    def __init__(self):
        super().__init__()
        self.node_title = "CSV Logger"
        
        self.inputs = {
            'input_a': 'signal',
            'input_b': 'signal',
            'target': 'signal',
            'prediction': 'signal',
            'save_trigger': 'signal'
        }
        self.outputs = {}
        
        self.data_buffer = []
        self.max_buffer = 2000  # Increased buffer
        self.start_time = time.time()
        self.last_trigger = 0.0
        self.auto_save_counter = 0
        self.auto_save_interval = 100  # Auto-save every 100 frames if trigger connected
        
        # Make file path explicit
        self.output_path = os.path.join(os.getcwd(), "reservoir_quantum_data.csv")
        self.last_save_time = 0
        
    def step(self):
        # Collect data
        a = self.get_blended_input('input_a', 'sum')
        b = self.get_blended_input('input_b', 'sum')
        tgt = self.get_blended_input('target', 'sum')
        pred = self.get_blended_input('prediction', 'sum')
        trig = self.get_blended_input('save_trigger', 'sum')
        
        # Handle None values
        if a is None: a = 0.0
        if b is None: b = 0.0
        if tgt is None: tgt = 0.0
        if pred is None: pred = 0.0
        if trig is None: trig = 0.0
        
        # Convert to float if array
        if isinstance(a, (list, np.ndarray)): a = float(np.mean(a))
        if isinstance(b, (list, np.ndarray)): b = float(np.mean(b))
        if isinstance(tgt, (list, np.ndarray)): tgt = float(np.mean(tgt))
        if isinstance(pred, (list, np.ndarray)): pred = float(np.mean(pred))
        if isinstance(trig, (list, np.ndarray)): trig = float(np.mean(trig))
        
        t = time.time() - self.start_time
        
        # Record row
        self.data_buffer.append([t, a, b, tgt, pred])
        if len(self.data_buffer) > self.max_buffer:
            self.data_buffer.pop(0)
        
        # Auto-increment counter
        self.auto_save_counter += 1
        
        # Save on trigger (rising edge detection)
        if trig > 0.5 and self.last_trigger <= 0.5:
            self.save_to_csv()
            print(f"💾 Manual save triggered at {len(self.data_buffer)} rows")
        
        # Auto-save every N frames if trigger is constant high
        elif trig > 0.5 and self.auto_save_counter >= self.auto_save_interval:
            self.save_to_csv()
            self.auto_save_counter = 0
            print(f"💾 Auto-save at {len(self.data_buffer)} rows")
            
        self.last_trigger = trig
        
    def save_to_csv(self):
        if len(self.data_buffer) == 0:
            print("⚠️ No data to save yet!")
            return
            
        try:
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "InputA", "InputB", "Target", "Prediction"])
                writer.writerows(self.data_buffer)
            
            self.last_save_time = time.time()
            print(f"✅ Saved {len(self.data_buffer)} rows to: {self.output_path}")
            print(f"   Time range: {self.data_buffer[0][0]:.1f}s to {self.data_buffer[-1][0]:.1f}s")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            print(f"   Attempted path: {self.output_path}")
            import traceback
            traceback.print_exc()
            
    def get_display_image(self):
        # Status display
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        
        # Row count
        cv2.putText(img, f"Rows: {len(self.data_buffer)}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save indicator
        if time.time() - self.last_save_time < 1.0:
            cv2.putText(img, "SAVED!", (5, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return QtGui.QImage(img.data, 128, 64, 128*3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Output Path", "output_path", self.output_path, None),
            ("Max Buffer", "max_buffer", self.max_buffer, None)
        ]
    
    def set_config_options(self, options):
        if "output_path" in options:
            self.output_path = options["output_path"]
        if "max_buffer" in options:
            self.max_buffer = int(options["max_buffer"])