"""
Interactive Signal Node - Outputs a value that can be changed
with on-screen + and - buttons.

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
from PIL import Image, ImageDraw, ImageFont
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class InteractiveSignalNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80) # Source Green
    
    def __init__(self, value=1.0):
        super().__init__()
        self.node_title = "Interactive Signal"
        self.outputs = {'signal': 'signal'}
        
        # This attribute MUST be named 'zoom_factor'
        # for the host (perception_lab_host.py) to draw the +/ - buttons.
        self.zoom_factor = float(value)
        
        # Try to load a font for display
        try:
            self.font = ImageFont.load_default(size=14)
        except IOError:
            self.font = None

    def step(self):
        # The host application modifies self.zoom_factor directly
        # when the +/ - buttons are clicked.
        pass
        
    def get_output(self, port_name):
        if port_name == 'signal':
            # Output the current value
            return self.zoom_factor
        return None
        
    def get_display_image(self):
        w, h = 64, 32  # Small and wide
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Display the current value
        text = f"{self.zoom_factor:.3f}"
        
        if self.zoom_factor > 1.0:
            text_color = (100, 255, 100) # Green
        elif self.zoom_factor < 1.0:
            text_color = (255, 100, 100) # Red
        else:
            text_color = (200, 200, 200) # Gray
        
        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (w - text_w) / 2
            y = (h - text_h) / 2
        except Exception:
            x, y = 5, 5 # Fallback
            
        draw.text((x, y), text, fill=text_color, font=self.font)
        
        img_final = np.array(img_pil)
        img_final = np.ascontiguousarray(img_final)
        return QtGui.QImage(img_final.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        # Allow setting the initial value
        return [
            ("Initial Value", "zoom_factor", self.zoom_factor, None)
        ]