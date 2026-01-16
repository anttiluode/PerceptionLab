"""
Fractal Dimension Node - "The Math of the Soul"
===============================================
Calculates the Minkowski-Bouligand (Box-Counting) Dimension of the structure.

- Dim ~ 1.0: Simple Lines (Euclidean Order)
- Dim ~ 1.7: Diffusion Limited Aggregation (Biological Growth)
- Dim ~ 2.0: Pure Noise (Chaos)

If this number stabilizes, you have found a constant of the system.
"""

import numpy as np
import cv2
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class FractalDimensionNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Fractal Dimension (Minkowski)"
    NODE_COLOR = QtGui.QColor(255, 100, 0) # Math Orange
    
    def __init__(self):
        super().__init__()
        self.inputs = {'structure_in': 'image'} # Connect Layer 3 Structure here
        self.outputs = {'plot': 'image', 'dimension': 'signal'}
        
        self.history = []
        self._last_display = None

    def step(self):
        img = self.get_blended_input('structure_in', 'mean')
        if img is None: return
        
        # Binarize the structure (White Riverbeds vs Black Rock)
        if img.ndim > 2: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = (img > 0.2).astype(np.uint8) # Threshold the "carvings"
        
        # Calculate Box Counting Dimension
        # We define pixel counts at different scales (s)
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        h, w = thresh.shape
        if np.sum(thresh) == 0:
            D = 0.0
        else:
            for s in scales:
                # Resize to scale (simulating grid boxes)
                resized = cv2.resize(thresh, (w//s, h//s), interpolation=cv2.INTER_NEAREST)
                non_zero = np.sum(resized > 0)
                if non_zero > 0:
                    counts.append(np.log(non_zero))
                else:
                    counts.append(0)
            
            # Linear Fit: log(N) = D * log(1/s) + c
            # Slope of the log-log plot is the Dimension
            x = np.log([1.0/s for s in scales])
            y = np.array(counts)
            
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                D = slope
            else:
                D = 0.0

        self.history.append(D)
        if len(self.history) > 100: self.history.pop(0)
        
        # Visualize
        self._render_plot(D)
        self.outputs['dimension'] = D

    def _render_plot(self, current_d):
        H, W = 200, 300
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(canvas, (0,0), (W,H), (30,30,30), -1)
        
        # Draw History
        if len(self.history) > 1:
            pts = []
            min_d, max_d = 0.0, 2.0
            for i, val in enumerate(self.history):
                px = int((i / 100.0) * W)
                norm = (val - min_d) / (max_d - min_d)
                py = int(H - (norm * H))
                pts.append((px, py))
            
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (0, 255, 255), 2)
            
        # Text
        cv2.putText(canvas, f"FRACTAL DIMENSION: {current_d:.4f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                   
        # Context
        if current_d < 1.1: txt = "LINE (1D)"
        elif current_d < 1.8: txt = "ORGANIC / DLA"
        else: txt = "NOISE (2D)"
        
        cv2.putText(canvas, txt, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        self._last_display = canvas
        self.outputs['plot'] = canvas

    def get_output(self, name):
        if name == 'plot': return self.outputs.get('plot')
        if name == 'dimension': return self.outputs.get('dimension')
        return None

    def get_display_image(self):
        if self._last_display is None: return None
        h, w = self._last_display.shape[:2]
        return QtGui.QImage(self._last_display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)