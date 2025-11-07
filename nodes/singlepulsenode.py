"""
Single Pulse Node - Outputs a signal of 1.0 for exactly one frame
when the user presses the R-button on the node.
"""

import numpy as np
from PyQt6 import QtGui
from PIL import Image, ImageDraw, ImageFont
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class SinglePulseNode(BaseNode):
    NODE_CATEGORY = "Source"  # Changed to Source because it now generates the input
    NODE_COLOR = QtGui.QColor(255, 120, 0) # Pulse Orange
    
    def __init__(self):
        super().__init__()
        self.node_title = "Pulse Trigger (R-Button)"
        
        # --- MODIFIED: No input port needed ---
        self.inputs = {}
        self.outputs = {'pulse_out': 'signal'}
        
        self.output_pulse = 0.0
        
        # Flag controlled by the manual R-button press
        self.manual_pulse_flag = False 
        self.frames_since_pulse = 0
        
        try:
            self.font = ImageFont.load_default()
        except IOError:
            self.font = None 

    def randomize(self):
        """
        This method is called when the user presses the 'R' button on the node.
        It sets the flag to trigger a pulse on the next step().
        """
        self.manual_pulse_flag = True
        
    def step(self):
        # 1. Check if the manual button was pressed (flag is True)
        if self.manual_pulse_flag:
            self.output_pulse = 1.0 # Send pulse for this frame
            self.frames_since_pulse = 0
            self.manual_pulse_flag = False # Reset the flag immediately
        
        # 2. If a pulse was sent last frame, ensure it returns to 0.0 now
        elif self.output_pulse > 0.0:
            self.output_pulse = 0.0
            self.frames_since_pulse += 1
        
        else:
            self.frames_since_pulse += 1

    def get_output(self, port_name):
        if port_name == 'pulse_out':
            return self.output_pulse
        return None
        
    def get_display_image(self):
        w, h = 96, 32 # Increased size for better text display
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Show pulse state
        if self.output_pulse == 1.0:
            img.fill(255)
            text = "PULSE!"
            fill_color = 0
        else:
            text = "Click R to Pulse"
            fill_color = 255
            
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        font_to_use = self.font if self.font else ImageFont.load_default()
            
        draw.text((w//8, h//4), text, fill=fill_color, font=font_to_use)
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return []
