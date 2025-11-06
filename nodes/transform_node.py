"""
Spectral Memory Node - Applies temporal filtering in the frequency domain
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

class SpectralMemoryNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 150, 255) # A "complex" blue
    
    def __init__(self, decay=0.9, boost=0.1):
        super().__init__()
        self.node_title = "Spectral Memory"
        self.inputs = {
            'complex_spectrum': 'complex_spectrum',
            'decay': 'signal',
            'boost': 'signal'
        }
        self.outputs = {'complex_spectrum': 'complex_spectrum', 'image': 'image'}
        
        self.decay = float(decay)
        self.boost = float(boost)
        
        # The memory
        self.memory = None
        self.vis_img = np.zeros((64, 64), dtype=np.float32)

    def step(self):
        # Get parameters from inputs
        decay_in = self.get_blended_input('decay', 'sum')
        boost_in = self.get_blended_input('boost', 'sum')
        
        if decay_in is not None:
            self.decay = np.clip(decay_in, 0.8, 1.0) # Map [0,1] to [0.8, 1.0]
        if boost_in is not None:
            self.boost = np.clip(boost_in, 0.0, 0.2) # Map [0,1] to [0.0, 0.2]
            
        # Get the input spectrum
        spec_in = self.get_blended_input('complex_spectrum', 'mean')
        
        if spec_in is None:
            # If no input, just decay the memory
            if self.memory is not None:
                self.memory *= self.decay
            self.vis_img *= 0.95
            return
            
        # Initialize memory if this is the first frame
        if self.memory is None or self.memory.shape != spec_in.shape:
            self.memory = np.zeros_like(spec_in, dtype=np.complex128)
            
        # Apply the leaky integrator (memory)
        # memory = memory * decay + new_input * (1.0 - decay)
        self.memory = (self.memory * self.decay) + (spec_in * (1.0 - self.decay))
        
        # Apply boost (adds a bit of the raw signal back in)
        output_spec = self.memory + (spec_in * self.boost)

        # Update visualization (log magnitude)
        mag = np.log1p(np.abs(output_spec))
        if mag.max() > mag.min():
            mag = (mag - mag.min()) / (mag.max() - mag.min())
        
        self.vis_img = cv2.resize(mag, (64, 64)).astype(np.float32)

    def get_output(self, port_name):
        if port_name == 'complex_spectrum':
            return self.memory
        elif port_name == 'image':
            return self.vis_img
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.vis_img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, 64, 64, 64, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Decay (0.8-1.0)", "decay", self.decay, None),
            ("Boost (0.0-0.2)", "boost", self.boost, None),
        ]