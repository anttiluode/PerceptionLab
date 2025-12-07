import numpy as np
import cv2
from collections import deque

# --- STRICT COMPATIBILITY BOILERPLATE ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0
        def step(self): pass
        def get_output(self, name): return None
        def get_display_image(self): return None

class PhaseSpaceNode2(BaseNode):
    """
    Phase Space Reconstruction (Aggressive Zoom)
    --------------------------------------------
    Now uses Standard Deviation scaling to visualize microscopic noise.
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Phase Space Geometry"
    NODE_COLOR = QtGui.QColor(100, 0, 150) # Deep Purple

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal', 
        }
        
        self.outputs = {
            'phase_plot': 'image'
        }
        
        # PARAMETERS
        self.history_len = 1000  
        self.delay = 15          
        
        # BUFFERS
        self.buffer = deque(maxlen=self.history_len + self.delay)
        
        self.image_size = 256
        self._output_image = None
        self._outs = {}

    def step(self):
        # 1. Get Input
        val = self.get_blended_input('signal_in', 'mean')
        if val is None: val = 0.0
        val = float(val)
        
        # 2. Store in Buffer
        self.buffer.append(val)
        
        # 3. Reconstruct Geometry
        if len(self.buffer) > self.delay:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            pts = []
            
            data = np.array(self.buffer)
            
            # --- AGGRESSIVE Z-SCORE ZOOM ---
            # We normalize based on Standard Deviation.
            # This guarantees the noise fills the screen.
            mean = np.mean(data)
            std = np.std(data)
            
            # If the signal is truly dead (std is 0), we can't zoom.
            if std < 1e-9: 
                std = 1.0 # Avoid divide by zero
            
            # Zoom factor: 3 standard deviations fits in the window
            scale_factor = (self.image_size / 2) / (3 * std)
            center_offset = self.image_size / 2

            for i in range(self.delay, len(self.buffer)):
                # Fade out old points
                age_factor = (i - self.delay) / (len(self.buffer) - self.delay)
                if age_factor < 0.2: continue 
                
                raw_x = self.buffer[i]
                raw_y = self.buffer[i - self.delay]
                
                # Z-Score Transform
                # (Value - Mean) / Std * Scale + Center
                px = int((raw_x - mean) * scale_factor + center_offset)
                py = int((raw_y - mean) * scale_factor + center_offset)
                
                # Clamp to be safe
                px = np.clip(px, 0, self.image_size-1)
                py = np.clip(py, 0, self.image_size-1)
                
                pts.append((px, py))

            if len(pts) > 1:
                for j in range(1, len(pts)):
                    pt1 = pts[j-1]
                    pt2 = pts[j]
                    
                    # Color: Cyan/Purple
                    color = (255, 200, 50) 
                    
                    cv2.line(img, pt1, pt2, color, 1)

            # Debug Overlay (Shows how tiny the signal is)
            cv2.putText(img, f"StdDev: {std:.6f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            self._output_image = img
            self._outs['phase_plot'] = img

    def get_output(self, name):
        return self._outs.get(name)

    def get_display_image(self):
        return self._output_image