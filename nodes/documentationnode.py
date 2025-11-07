"""
Documentation Node - Displays user-defined text for documenting a graph.
The text is saved with the graph file.
"""
import cv2
import numpy as np
from PyQt6 import QtGui
from PIL import Image, ImageDraw, ImageFont
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class DocumentationNode(BaseNode):
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(50, 50, 50) # Dark Gray for background utility
    
    def __init__(self, doc_text="[Graph Documentation]", width=200, height=100):
        super().__init__()
        self.node_title = "Documentation"
        
        # --- FIX: Use a simple output to force redraw ---
        self.outputs = {'refresh_flag': 'signal'}
        self.initial_refresh_counter = 5 # Pulse high for the first 5 frames
        # --- END FIX ---
        
        self.doc_text = str(doc_text)
        self.w, self.h = int(width), int(height)
        
        try:
            self.font = ImageFont.load_default()
        except IOError:
            self.font = None 

    def step(self):
        # Consume the initial refresh counter to force an update
        if self.initial_refresh_counter > 0:
            self.initial_refresh_counter -= 1
        pass

    def get_output(self, port_name):
        if port_name == 'refresh_flag':
            # Signal high for a few frames when first loading/running
            return 1.0 if self.initial_refresh_counter > 0 else 0.0
        return None
        
    def get_display_image(self):
        # Create a blank image
        img = np.zeros((self.h, self.w), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        text_lines = self.doc_text.split('\n')
        y_pos = 5
        
        font_to_use = self.font if self.font else ImageFont.load_default()

        try:
            for line in text_lines:
                draw.text((5, y_pos), line, fill=255, font=font_to_use)
                y_pos += 15
        except Exception:
            draw.text((5, 5), self.doc_text, fill=255, font=font_to_use)

        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        
        # Add border to distinguish it from the background
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), 100, 1)
        
        return QtGui.QImage(img.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Documentation Text", "doc_text", self.doc_text, None),
            ("Width", "w", self.w, None),
            ("Height", "h", self.h, None),
        ]