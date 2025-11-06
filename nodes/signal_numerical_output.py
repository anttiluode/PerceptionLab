"""
Signal Display Node - Displays a live numerical value
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
from PIL import Image, ImageDraw, ImageFont
import os

# --- !! CRITICAL IMPORT BLOCK !! ---
# This is the *only* correct way to import BaseNode and shared resources.
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------------

class SignalDisplayNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120) # Output Purple
    
    def __init__(self):
        super().__init__()
        self.node_title = "Signal Display"
        
        # Define ports
        self.inputs = {'signal': 'signal'}  # port_name: port_type
        self.outputs = {} # No outputs for this node
        
        # Internal state
        self.current_value = 0.0
        
        # Try to load a font
        try:
            self.font = ImageFont.load_default(size=14)
        except IOError:
            print("Warning: Default PIL font not found. Display text may be small.")
            self.font = None

    def step(self):
        """Called every frame - main processing logic"""
        # Get input data using 'sum' to handle multiple inputs
        input_val = self.get_blended_input('signal', 'sum')
        
        if input_val is not None:
            self.current_value = input_val
        else:
            # Gently decay to 0 if no signal is present
            self.current_value *= 0.95
        
    def get_output(self, port_name):
        """This node has no outputs"""
        return None
        
    def get_display_image(self):
        """Return a QImage for node preview"""
        w, h = 64, 32  # A smaller, wider display for text
        
        # Create a black background image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Format the text
        text = f"{self.current_value:.3f}"
        
        # Determine text color based on value
        if self.current_value > 0.01:
            text_color = (100, 255, 100) # Green
        elif self.current_value < -0.01:
            text_color = (255, 100, 100) # Red
        else:
            text_color = (200, 200, 200) # Gray
            
        # Calculate text position to center it
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) / 2
        y = (h - text_h) / 2
        
        # Draw the text
        draw.text((x, y), text, fill=text_color, font=self.font)
        
        # Convert back to QImage
        img_final = np.array(img_pil)
        img_final = np.ascontiguousarray(img_final)
        return QtGui.QImage(img_final.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        # No configuration options for this simple node
        return []