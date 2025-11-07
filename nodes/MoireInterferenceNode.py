"""
Moiré Interference Node - Generates a 2D moiré pattern by interfering
two perpendicular sine waves. The frequencies of the waves are
controlled by the signal inputs.

Ported from moire_microscope.html
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class MoireInterferenceNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 180, 180) # Moiré Teal
    
    def __init__(self, size=128, base_phase_1=0.0, base_phase_2=0.0):
        super().__init__()
        self.node_title = "Moiré Interference"
        self.size = int(size)
        self.base_phase_1 = float(base_phase_1)
        self.base_phase_2 = float(base_phase_2)
        
        self.inputs = {
            'freq_1': 'signal', # Controls frequency of horizontal wave
            'freq_2': 'signal'  # Controls frequency of vertical wave
        }
        self.outputs = {'image': 'image'}
        
        # Pre-calculate coordinate grids
        self._init_grids()
        self.output_image = np.zeros((self.size, self.size), dtype=np.float32)

    def _init_grids(self):
        """Creates normalized coordinate grids [0, 1]"""
        if self.size == 0: self.size = 1 # Avoid division by zero
        u_vec = np.linspace(0, 1, self.size, dtype=np.float32)
        v_vec = np.linspace(0, 1, self.size, dtype=np.float32)
        # V (rows, 0->1), U (cols, 0->1)
        self.U, self.V = np.meshgrid(u_vec, v_vec) 
        self.output_image = np.zeros((self.size, self.size), dtype=np.float32)

    def step(self):
        # Check if size changed from config
        if self.U.shape[0] != self.size:
            self._init_grids()
            
        # 1. Get frequency inputs
        # We map the input signal (range -1 to 1) to a k-value (frequency)
        # e.g., mapping to a range of [5, 45]
        k1 = ((self.get_blended_input('freq_1', 'sum') or 0.0) + 1.0) * 20.0 + 5.0
        k2 = ((self.get_blended_input('freq_2', 'sum') or 0.0) + 1.0) * 20.0 + 5.0
        
        # 2. Port the core math from moire_microscope.html
        # const field1 = Math.sin(u * 20 * Math.PI + phase1);
        # const field2 = Math.cos(v * 20 * Math.PI + phase2);
        # const moireValue = Math.cos(field1 * Math.PI - field2 * Math.PI);
        
        # We use U (horizontal grid) for field 1 and V (vertical grid) for field 2
        field1 = np.sin(self.U * k1 * np.pi + self.base_phase_1)
        field2 = np.cos(self.V * k2 * np.pi + self.base_phase_2)
        
        # The interference pattern
        moire_value = np.cos(field1 * np.pi - field2 * np.pi)
        
        # 3. Normalize [-1, 1] to [0, 1] for image output
        self.output_image = (moire_value + 1.0) / 2.0

    def get_output(self, port_name):
        if port_name == 'image':
            return self.output_image
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.output_image, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.size, self.size, self.size, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Resolution", "size", self.size, None),
            ("Base Phase 1", "base_phase_1", self.base_phase_1, None),
            ("Base Phase 2", "base_phase_2", self.base_phase_2, None),
        ]