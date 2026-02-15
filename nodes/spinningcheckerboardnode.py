"""
RevolvingGridNode (Formerly Checkerboard)
=========================================
A geometric carrier wave that can SPIN.
This turns the grid into a Phase-Scanner.

Inputs:
- Scale: Frequency (Zoom)
- Angle: Rotation (The Spin)
- Shift: Phase (The Linear Slide)
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    # Standalone testing mocks
    class BaseNode:
        def get_blended_input(self, name, mode): return 0.0
    import PyQt6.QtGui as QtGui

class SpinningCheckerboardNode(BaseNode):
    NODE_CATEGORY = "Generator"
    NODE_TITLE = "Revolving Grid" # Renamed to reflect new power
    NODE_COLOR = QtGui.QColor(200, 200, 220) # Pale Blue

    def __init__(self, size=256):
        super().__init__()
        
        self.inputs = {
            'scale': 'signal',   # Frequency (Zoom)
            'angle': 'signal',   # ROTATION (The Spin Info)
            'phase': 'signal'    # SHIFT (The Twist Info)
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def step(self):
        # 1. GET INFORMATION ( The Signals driving the geometry )
        scale_in = self.get_blended_input('scale', 'sum')
        if scale_in is None: scale_in = 0.5
        
        angle_in = self.get_blended_input('angle', 'sum')
        if angle_in is None: angle_in = 0.0
        
        phase_in = self.get_blended_input('phase', 'sum')
        if phase_in is None: phase_in = 0.0
        
        # 2. MAP INPUTS TO PHYSICS
        # Scale: 0.0 -> Huge squares, 1.0 -> Tiny squares (High Freq)
        square_size = max(2, int(60 - scale_in * 50)) 
        
        # Angle: Input 0..1 maps to 0..180 degrees (Pi)
        theta = angle_in * np.pi 
        
        # Phase: Input 0..1 shifts the grid by one full cell width
        shift = phase_in * square_size
        
        # 3. GENERATE ROTATED COORDINATES
        y, x = np.mgrid[0:self.size, 0:self.size]
        
        # Center the rotation
        cx, cy = self.size // 2, self.size // 2
        x = x - cx
        y = y - cy
        
        # Rotation Matrix
        c, s = np.cos(theta), np.sin(theta)
        rx = x * c - y * s
        ry = x * s + y * c
        
        # Apply Phase Shift
        rx += shift
        ry += shift
        
        # 4. CREATE CHECKERBOARD (XOR Logic on Rotated Grid)
        # Using Sine waves instead of hard squares makes it a cleaner "Wave"
        # but hard squares give that "digital" look. Let's stick to hard squares 
        # for maximum contrast (Moiré works best with sharp edges).
        
        grid_x = np.floor(rx / square_size) % 2
        grid_y = np.floor(ry / square_size) % 2
        
        # XOR pattern: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0
        pattern = np.abs(grid_x - grid_y)
        
        # 5. RENDER
        # Convert to RGB (Gray)
        img = np.stack([pattern]*3, axis=-1).astype(np.float32)
        
        self.display_image = img

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_image
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.display_image, 0, 1) * 255).astype(np.uint8)
        return QtGui.QImage(img_u8.data, self.size, self.size, 
                           self.size*3, QtGui.QImage.Format.Format_RGB888)