"""
Cabbage Observer Node (Bulletproof)
-----------------------------------
The Homeostatic Controller.
[FIX] Added clamping to visualization coordinates to prevent Integer Overflow crashes.
[FIX] Added bounds checking for drawing rectangles.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class CabbageObserverNode(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(255, 215, 0)

    def __init__(self):
        super().__init__()
        self.node_title = "Cabbage Observer"
        
        self.inputs = {
            'reality_dna': 'spectrum',    # From Scanner A
            'dream_dna': 'spectrum'       # From Scanner B
        }
        self.outputs = {
            'growth_drive': 'signal',
            'attention_map': 'image'
        }
        
        self.latent_dim = 55 # Matches Scanner
        self.sensitivity = 50.0
        self.drive_val = 0.0
        
        # Visualization buffer (50 rows, 550 cols)
        self.h = 50
        self.w = 550
        self.att_map = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def step(self):
        real = self.get_blended_input('reality_dna', 'first')
        dream = self.get_blended_input('dream_dna', 'first')
        
        # Safety Zero
        if real is None: real = np.zeros(self.latent_dim)
        if dream is None: dream = np.zeros(self.latent_dim)
        
        # Safety Resize
        def fix(v):
            v = np.array(v, dtype=np.float32).flatten()
            if len(v) < self.latent_dim:
                return np.pad(v, (0, self.latent_dim - len(v)))
            return v[:self.latent_dim]
            
        real = fix(real)
        dream = fix(dream)
        
        # Compare
        error = np.abs(real - dream)
        total_error = np.mean(error)
        
        # Drive (Unclamped for physics)
        self.drive_val = total_error * self.sensitivity
        
        # Visualize (Clamped for display safety)
        self.att_map = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        for i, e in enumerate(error):
            if i * 10 >= self.w: break 
            
            # CLAMPING: Ensure height doesn't exceed image bounds
            # e is error. If e=1.0, h=100. Image is 50 high.
            # So we clip h to be at most 50.
            
            bar_height = int(e * 100)
            bar_height = max(0, min(bar_height, self.h)) # Clamp 0..50
            
            # Calculate Coordinates
            x1 = int(i * 10)
            x2 = int(i * 10 + 8)
            y1 = int(self.h)
            y2 = int(self.h - bar_height)
            
            # Safety check
            if x2 > self.w: x2 = self.w
            
            # Draw Red Bar
            cv2.rectangle(self.att_map, (x1, y1), (x2, y2), (0,0,255), -1)

    def get_output(self, port_name):
        if port_name == 'growth_drive': return float(self.drive_val)
        if port_name == 'attention_map': return self.att_map
        return None

    def get_display_image(self):
        h, w = self.att_map.shape[:2]
        return QtGui.QImage(self.att_map.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)