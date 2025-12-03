"""
Learner Logger Node (Fixed)
===========================
Logs W-Matrix training metrics. 
Includes an INTERNAL TRIGGER button in the config menu.

Captures:
- Coherence (Learning Progress)
- Loss (Error Signal)
- Overlap (Accuracy vs Stable Address)
"""

import numpy as np
import json
import time
import cv2
import os
from collections import deque

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class LearnerLoggerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Learner Logger"
    NODE_COLOR = QtGui.QColor(100, 50, 150)  # Deep Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'coherence': 'signal',
            'loss': 'signal',
            'overlap': 'signal',
            'learning_rate': 'signal',
            'trigger_input': 'signal' # Optional external trigger
        }
        
        self.outputs = {
            'step_count': 'signal',
            'save_status': 'signal'
        }
        
        # Internal State
        self.step_count = 0
        self.buffer = {
            'steps': [], 'coherence': [], 'loss': [], 'overlap': [], 'lr': []
        }
        
        # Config options
        self.save_now_button = False  # The internal button
        self.file_prefix = "w_matrix"
        
        self.last_save_msg = "Ready"
        self.flash_timer = 0

    def save_log(self):
        """Exports data to JSON"""
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.file_prefix}_{timestamp}.json"
            full_path = os.path.abspath(os.path.join(os.getcwd(), filename))
            
            export_data = {
                "meta": {"timestamp": timestamp, "total_steps": self.step_count},
                "metrics": self.buffer
            }
            
            # Safe Numpy encoder - Indentation Fixed
            def np_encoder(obj):
                if isinstance(obj, (np.generic, np.ndarray)):
                    return obj.tolist()
                return float(obj)

            with open(full_path, 'w') as f:
                # Fixed the 'default' argument syntax error
                json.dump(export_data, f, indent=2, default=np_encoder)
            
            self.last_save_msg = f"Saved: {filename}"
            self.flash_timer = 30
            print(f"LOG SAVED: {full_path}")
            
        except Exception as e:
            self.last_save_msg = f"Error: {str(e)[:15]}..."
            print(f"LOG ERROR: {e}")

    def step(self):
        self.step_count += 1
        if self.flash_timer > 0: self.flash_timer -= 1
        
        # 1. Handle Button Click (Config Menu)
        if self.save_now_button:
            self.save_log()
            self.save_now_button = False # Reset switch immediately
            
        # 2. Handle External Trigger
        trig = self.get_blended_input('trigger_input', 'sum')
        if trig is not None and trig > 0.5:
            if self.step_count % 10 == 0: # Prevent spamming
                 self.save_log()

        # 3. Record Data
        c = float(self.get_blended_input('coherence', 'sum') or 0.0)
        l = float(self.get_blended_input('loss', 'sum') or 0.0)
        o = float(self.get_blended_input('overlap', 'sum') or 0.0)
        lr = float(self.get_blended_input('learning_rate', 'sum') or 0.0)
        
        b = self.buffer
        b['steps'].append(self.step_count)
        b['coherence'].append(c)
        b['loss'].append(l)
        b['overlap'].append(o)
        b['lr'].append(lr)
        
        # RAM Limit
        if len(b['steps']) > 5000:
            for k in b: b[k].pop(0)

    def get_output(self, name):
        if name == 'step_count': return float(self.step_count)
        if name == 'save_status': return 1.0 if self.flash_timer > 0 else 0.0
        return 0.0

    def get_display_image(self):
        h, w = 60, 140
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Flash green on save
        if self.flash_timer > 0:
            img[:] = (50, 100, 50)
        else:
            img[:] = (40, 30, 50)
            
        # Text
        cv2.putText(img, "LEARNER LOG", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,255), 1)
        
        # Last Metric
        if self.buffer['coherence']:
            c = self.buffer['coherence'][-1]
            o = self.buffer['overlap'][-1]
            cv2.putText(img, f"Coh: {c:.3f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
            cv2.putText(img, f"Ovl: {o:.3f}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
            
        return __main__.numpy_to_qimage(img)

    def get_config_options(self):
        # This bool acts as a push button
        return [
            ("CLICK TO SAVE JSON", "save_now_button", self.save_now_button, 'bool'),
            ("File Prefix", "file_prefix", self.file_prefix, 'text'),
        ]