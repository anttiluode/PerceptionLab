"""
Phase-Lock Loop (PLL) Visualizer Node
-------------------------------------
Visualizes the synchronization between two phase fields (e.g., External vs. Internal).
This is the "Lag of Existence" visualizer.

Inputs:
- phase_a: External Phase (e.g., Webcam)
- phase_b: Internal Phase (e.g., Wave Mirror)

Outputs:
- lock_error: Signal representing the total phase difference (0 = Locked, 1 = Chaos)
- error_map: Image visualizing the local phase difference
"""

import numpy as np
from PyQt6 import QtGui
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------

class PhaseLockLoopNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(255, 100, 50)  # Alert Orange
    
    def __init__(self, sensitivity=1.0):
        super().__init__()
        self.node_title = "Phase-Lock Loop"
        
        self.inputs = {
            'phase_a': 'image', # External (Webcam)
            'phase_b': 'image'  # Internal (Wave Mirror)
        }
        
        self.outputs = {
            'lock_error': 'signal', # Global error signal for Optimizer
            'error_map': 'image'    # Visual feedback
        }
        
        self.sensitivity = float(sensitivity)
        self.error_metric = 1.0
        self.vis_img = np.zeros((128, 128, 3), dtype=np.uint8)
        
    def step(self):
        # 1. Get Phase Images
        # Expecting grayscale or single-channel float images representing phase (0..1)
        img_a = self.get_blended_input('phase_a', 'mean')
        img_b = self.get_blended_input('phase_b', 'mean')
        
        if img_a is None or img_b is None:
            return
            
        # Resize to match if needed (use smaller dimension for performance)
        h, w = img_a.shape[:2]
        if img_b.shape[:2] != (h, w):
            img_b = cv2.resize(img_b, (w, h))
            
        # 2. Calculate Phase Difference
        # Diff = Abs(A - B)
        # We handle the circular nature of phase (0 and 1 are the same)
        # Shortest distance on a circle: min(|a-b|, 1-|a-b|)
        
        diff = np.abs(img_a - img_b)
        diff = np.minimum(diff, 1.0 - diff) * 2.0 # Normalize 0..0.5 -> 0..1
        
        # Apply sensitivity
        diff = np.clip(diff * self.sensitivity, 0, 1)
        
        # 3. Calculate Global Error (For Optimizer)
        self.error_metric = np.mean(diff)
        
        # 4. Visualization
        # Map Error to Heatmap (Black=Locked, White/Red=Error)
        diff_u8 = (diff * 255).astype(np.uint8)
        self.vis_img = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
        
        # Invert for "Transparency" effect? 
        # Let's keep Heatmap: Dark = Good, Bright = Bad.
        
    def get_output(self, port_name):
        if port_name == 'lock_error':
            return float(self.error_metric)
        elif port_name == 'error_map':
            return self.vis_img.astype(np.float32) / 255.0
        return None

    def get_display_image(self):
        # Overlay Error Metric
        img = self.vis_img.copy()
        cv2.putText(img, f"Error: {self.error_metric:.3f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(img.data, img.shape[1], img.shape[0], 
                           img.shape[1]*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Sensitivity", "sensitivity", self.sensitivity, None)
        ]