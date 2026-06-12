"""
Rotating Phase Field Node
-------------------------
A 2D spatial manifold (island generator) that rotates based on incoming Skew Flux.
If detailed balance holds (flux = 0), it is static. As flux rises, it forms 
chiral spirals representing emergent time and entropy.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class PhaseFieldNode(BaseNode):
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(50, 150, 150) # Teal

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Phase Field (Island)"
        self.inputs = {
            'skew_flux': 'signal', # Drives rotation (Arrow of Time)
            'scale': 'signal'      # Drives spatial frequency
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.phase = 0.0
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def step(self):
        flux = self.get_blended_input('skew_flux', 'sum') or 0.0
        scale_in = self.get_blended_input('scale', 'sum') or 0.5
        
        # 1. Integrate Flux into Phase (Entropy drives time forward)
        self.phase += float(flux) * 2.0 
        
        # 2. Generate Grid
        y, x = np.mgrid[-1:1:complex(self.size), -1:1:complex(self.size)]
        radius = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        
        # 3. Moiré / Spiral interference pattern
        freq = 3.0 + float(scale_in) * 15.0
        
        # The wave incorporates angular rotation tied directly to the skew flux
        wave = np.sin(freq * radius + angle * 3.0 - self.phase)
        
        # Normalize to 0..1 for image output
        wave_norm = (wave + 1.0) / 2.0
        
        # Cyan/Gold coloring for aesthetic differentiation
        self.display_image = np.stack([
            wave_norm * 0.9, 
            wave_norm * 0.8 + 0.1, 
            wave_norm * 0.5 + (1-wave_norm)*0.5
        ], axis=-1).astype(np.float32)

    def get_output(self, port_name):
        if port_name == 'image': return self.display_image
        return None

    def get_display_image(self):
        disp = (np.clip(self.display_image, 0, 1) * 255).astype(np.uint8)
        disp = cv2.resize(disp, (128, 128), interpolation=cv2.INTER_NEAREST)
        return QtGui.QImage(disp.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)