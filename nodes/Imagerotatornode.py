"""
Image Rotator Node - Simple rotation by 90, 180, or -90 degrees
Works with PerceptionLab's get_blended_input protocol
"""

import cv2
import numpy as np
from PyQt6 import QtGui

import __main__
BaseNode = __main__.BaseNode

class ImageRotatorNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(255, 150, 50)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Image Rotator"
        
        self.inputs = {
            'image': 'image',
            'rot_90': 'signal',    # > 0.5 = rotate 90° clockwise
            'rot_180': 'signal',   # > 0.5 = rotate 180°
            'rot_neg90': 'signal'    # > 0.5 = rotate -90° 
        }
        
        self.outputs = {
            'image': 'image'
        }
        
        self.result_image = None
        self.current_mode = "PASS"
        
    def step(self):
        # Get image using PerceptionLab protocol
        img = self.get_blended_input('image', 'mean')
        
        if img is None:
            self.result_image = None
            self.current_mode = "NO INPUT"
            return
        
        # Get rotation signals
        s90 = self.get_blended_input('rot_90', 'max') or 0.0
        s180 = self.get_blended_input('rot_180', 'max') or 0.0
        s270 = self.get_blended_input('rot_neg90', 'max') or 0.0
        
        # Ensure we have a proper numpy array
        work = np.array(img)
        
        # Normalize to uint8 if needed
        if work.dtype != np.uint8:
            if work.max() <= 1.05:
                work = (work * 255).astype(np.uint8)
            else:
                work = np.clip(work, 0, 255).astype(np.uint8)
        
        # Handle grayscale -> BGR for consistency
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        
        # Apply rotation based on signal priority
        if float(s180) > 0.5:
            self.result_image = cv2.rotate(work, cv2.ROTATE_180)
            self.current_mode = "180°"
        elif float(s90) > 0.5:
            self.result_image = cv2.rotate(work, cv2.ROTATE_90_CLOCKWISE)
            self.current_mode = "90° CW"
        elif float(s270) > 0.5:
            self.result_image = cv2.rotate(work, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.current_mode = "-90° CCW"
        else:
            self.result_image = work
            self.current_mode = "PASS"
    
    def get_output(self, port_name):
        if port_name == 'image':
            if self.result_image is not None:
                # Return as float 0-1 for consistency with other nodes
                return self.result_image.astype(np.float32) / 255.0
            return None
        return None
    
    def get_display_image(self):
        if self.result_image is None:
            # Show placeholder
            placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.putText(placeholder, "NO", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(placeholder, "INPUT", (8, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            h, w = 64, 64
            return QtGui.QImage(placeholder.data, w, h, w * 3, 
                               QtGui.QImage.Format.Format_RGB888)
        
        # Resize for display if too large
        display = self.result_image.copy()
        h, w = display.shape[:2]
        max_size = 128
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            display = cv2.resize(display, (int(w * scale), int(h * scale)))
        
        # Add mode label
        cv2.putText(display, self.current_mode, (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Convert BGR to RGB for Qt
        if display.ndim == 3 and display.shape[2] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        
        h, w = display.shape[:2]
        ch = display.shape[2] if display.ndim == 3 else 1
        
        if ch == 1:
            return QtGui.QImage(display.data, w, h, w, 
                               QtGui.QImage.Format.Format_Grayscale8)
        else:
            return QtGui.QImage(display.data, w, h, w * ch, 
                               QtGui.QImage.Format.Format_RGB888).copy()