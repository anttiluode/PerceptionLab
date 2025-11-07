"""
Single Pulse Node - Outputs a signal of 1.0 for exactly one frame
when triggered by a rising edge on its input, then immediately returns to 0.0.
Ideal for triggering events like 'Compute' or 'Reset'.
"""

import numpy as np
from PyQt6 import QtGui
import sys
import os

from PIL import Image, ImageDraw, ImageFont

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class SinglePulseNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(255, 120, 0) # Pulse Orange
    
    def __init__(self):
        super().__init__()
        self.node_title = "Single Pulse"
        
        self.inputs = {'trigger_in': 'signal'}
        self.outputs = {'pulse_out': 'signal'}
        
        self.output_pulse = 0.0
        self.last_input_val = 0.0
        self.frames_since_pulse = 2 # Starts off
        
        try:
            self.font = ImageFont.load_default()
        except IOError:
            self.font = None 

    def step(self):
        trigger_val = self.get_blended_input('trigger_in', 'sum') or 0.0

        if self.output_pulse > 0:
            self.output_pulse = 0.0
        
        if trigger_val > 0.5 and self.last_input_val <= 0.5:
            self.output_pulse = 1.0 # Send pulse for this frame
            self.frames_since_pulse = 0
        else:
            self.frames_since_pulse += 1

        self.last_input_val = trigger_val

    def get_output(self, port_name):
        if port_name == 'pulse_out':
            return self.output_pulse
        return None
        
    def get_display_image(self):
        w, h = 64, 32
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Show pulse state
        if self.output_pulse == 1.0:
            img.fill(255)
            text = "PULSE!"
        else:
            text = f"Idle ({self.frames_since_pulse})"
            
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        font_to_use = self.font if self.font else ImageFont.load_default()
        
        # --- FIX: Change (0, 0, 0) to 0 and (255, 255, 255) to 255 ---
        draw.text((w//4, h//4), text, fill=(0) if self.output_pulse == 1.0 else (255), font=font_to_use)
        # --- END FIX ---
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return []