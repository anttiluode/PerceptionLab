"""
Stroboscopic Attractor Node (The Poincaré Section)
--------------------------------------------------
Visualizes the 'Discrete Self'.

Logic:
1. Listens continuously to the signal (to maintain the 11ms delay buffer).
2. ONLY plots a point when the 'Trigger' (Gamma Strobe) fires.
3. Uses Aggressive Auto-Zoom so even weak signals create a constellation.
"""

import numpy as np
import cv2
from collections import deque

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None

class StroboscopicAttractorNode3(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Stroboscopic Attractor"
    NODE_COLOR = QtGui.QColor(100, 0, 150) # Deep Purple

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal', # The Gated Signal
            'trigger': 'signal'    # The Strobe (Dendritic Gate)
        }
        
        self.outputs = {
            'attractor_view': 'image'
        }
        
        # 1. CONTINUOUS BUFFER (Time History)
        # Must run every frame to keep t-delay accurate
        self.delay = 15 # ~11ms at 60fps is closer to 10-15 frames depending on sampling
        self.time_buffer = deque(maxlen=100) 
        
        # 2. DISCRETE STORAGE (The Constellation)
        # Stores (x, y) points captured at strobe moments
        self.points = deque(maxlen=2000)
        
        self.display = np.zeros((256, 256, 3), dtype=np.uint8)

    def step(self):
        # 1. ALWAYS READ SIGNAL
        val = self.get_blended_input('signal_in', 'mean')
        trigger = self.get_blended_input('trigger', 'mean')
        
        if val is None: val = 0.0
        val = float(val)
        
        # Update continuous history (The "Wire")
        self.time_buffer.append(val)
        
        # 2. CHECK STROBE (The "Shutter")
        # Only capture a point if Trigger is High AND we have enough history
        if trigger is not None and trigger > 0.5:
            if len(self.time_buffer) > self.delay:
                # X = Current Value
                # Y = Value 'delay' frames ago
                x = val
                y = self.time_buffer[-1 - self.delay]
                self.points.append((x, y))
        
        # 3. DRAW
        self._draw_constellation()
        self.set_output('attractor_view', self.display)

    def _draw_constellation(self):
        self.display.fill(10) # Dark background
        
        if len(self.points) < 2: 
            cv2.putText(self.display, "WAITING FOR STROBE...", (10, 128), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return
            
        # --- AGGRESSIVE AUTO-ZOOM ---
        # Calculate statistics of the points cloud
        pts_array = np.array(self.points)
        mean_x = np.mean(pts_array[:, 0])
        mean_y = np.mean(pts_array[:, 1])
        std_x = np.std(pts_array[:, 0])
        std_y = np.std(pts_array[:, 1])
        
        # Avoid div by zero
        if std_x < 1e-9: std_x = 1.0
        if std_y < 1e-9: std_y = 1.0
        
        # Scaling: Fit 4 standard deviations into the screen
        scale_x = (128.0) / (4 * std_x)
        scale_y = (128.0) / (4 * std_y)
        
        h, w = self.display.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Draw points
        for i, (x, y) in enumerate(self.points):
            # Z-Score Transform
            px = int(cx + (x - mean_x) * scale_x)
            py = int(cy - (y - mean_y) * scale_y) # Flip Y for graph
            
            # Clip to screen
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            
            # Color fades with age
            age = i / len(self.points)
            brightness = int(age * 255)
            
            # Glowing Dots
            if age > 0.8: # Newest points are bright yellow
                color = (200, 255, 255)
                size = 2
            else: # Old points are purple/fade
                color = (brightness, brightness//2, brightness)
                size = 1
                
            cv2.circle(self.display, (px, py), size, color, -1)
            
        # Stats
        cv2.putText(self.display, f"PTS: {len(self.points)}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name): return getattr(self, '_outs', {}).get(name)
    def set_output(self, name, val): 
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val