"""
Takens Embedding Node (The Geometry Engine)
-------------------------------------------
Reconstructs the Phase Space Attractor from a single 1D signal 
using Time-Delay Embedding.

Theory:
It plots the signal against itself in the past. 
If the brain is "Thinking" (Cyclic/Stable), this draws a Ring or Torus.
If the brain is "Lost" (Noise), this draws a messy Cloud.
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

class TakensEmbeddingNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Takens Geometry"
    NODE_COLOR = QtGui.QColor(100, 0, 150) # Deep Purple

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal',
            'delay_tau': 'signal' # Optional: Dynamic tuning of delay
        }
        
        self.outputs = {
            'attractor_view': 'image',
            'dimension_score': 'signal' # Metric of how "3D" the shape is
        }
        
        # Buffer for history (Needs to be long enough for 2*tau)
        self.history_len = 5000
        self.buffer = deque([0.0]*self.history_len, maxlen=self.history_len)
        
        # Visuals
        self.display = np.zeros((300, 300, 3), dtype=np.uint8)
        self.default_tau = 15 # ~11ms at 60fps

    def step(self):
        # 1. READ INPUT
        val = self.get_blended_input('signal_in', 'mean')
        if val is None: val = 0.0
        val = float(val)
        
        self.buffer.append(val)
        
        # 2. READ DELAY (TAU)
        tau_sig = self.get_blended_input('delay_tau', 'mean')
        if tau_sig is not None:
            tau = int(tau_sig)
        else:
            tau = self.default_tau
            
        # Clamp tau
        tau = max(1, min(tau, self.history_len // 3))
        
        # 3. CONSTRUCT VECTORS (Embedding)
        # We need enough history
        if len(self.buffer) < 2 * tau + 1:
            return

        # We will draw the last N points to show the "Tail"
        trail_length = 5000
        points = []
        
        # Retrieve history snapshot
        history = list(self.buffer)
        
        # Build trajectory
        # X = t, Y = t - tau
        for i in range(len(history) - trail_length, len(history)):
            if i < 2 * tau: continue
            
            x = history[i]
            y = history[i - tau]
            # z = history[i - 2*tau] # For 3D rotation logic if needed
            
            points.append((x, y))
            
        # 4. DRAW
        self._draw_attractor(points)
        
        # 5. OUTPUT
        self.set_output('attractor_view', self.display)

    def _draw_attractor(self, points):
        self.display.fill(20) # Dark Background
        
        if not points: return
        
        # Auto-Zoom / Normalization
        # Find min/max to fit screen
        pts_np = np.array(points)
        min_xy = pts_np.min(axis=0)
        max_xy = pts_np.max(axis=0)
        span = max_xy - min_xy
        
        # Avoid div zero
        span[span < 0.00001] = 1.0
        
        scale_x = 280.0 / span[0]
        scale_y = 280.0 / span[1]
        
        # Center offsets
        off_x = 10 - min_xy[0] * scale_x
        off_y = 10 - min_xy[1] * scale_y
        
        # Draw Trajectory
        screen_points = []
        for x, y in points:
            sx = int(x * scale_x + off_x)
            sy = int(y * scale_y + off_y)
            screen_points.append((sx, sy))
            
        # Draw Lines
        if len(screen_points) > 1:
            # Gradient Color (Old = Dark, New = Bright)
            for i in range(len(screen_points) - 1):
                alpha = i / len(screen_points)
                # Color Shift: Blue -> Cyan -> White
                b = 255
                g = int(255 * alpha)
                r = int(100 * alpha)
                
                cv2.line(self.display, screen_points[i], screen_points[i+1], (b, g, r), 1)
                
        # Draw Head (The "Now")
        if screen_points:
            cv2.circle(self.display, screen_points[-1], 4, (255, 255, 255), -1)

    def get_output(self, name): return getattr(self, '_outs', {}).get(name)
    def set_output(self, name, val): 
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val