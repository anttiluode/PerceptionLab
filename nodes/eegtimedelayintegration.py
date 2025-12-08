"""
Time-Delay Integration Node (Fixed)
-----------------------------------
Fixes:
- OpenCV 'Scalar value is not numeric' crash by casting colors to native int.
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

class TimeDelayIntegrationNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Time-Delay Integration"
    NODE_COLOR = QtGui.QColor(0, 100, 200)

    def __init__(self):
        super().__init__()
        self.inputs = {
            'frontal': 'signal', 'occipital': 'signal', 
            'temporal': 'signal', 'parietal': 'signal', 
            'conduction_speed': 'signal'
        }
        self.outputs = {
            'integrated_field': 'image', 'coherence': 'signal', 'merged_signal': 'signal'
        }
        
        self.dist_map = {'frontal': 0.12, 'occipital': 0.12, 'temporal': 0.08, 'parietal': 0.08}
        self.history_len = 256
        self.buffers = {k: deque([0.0]*self.history_len, maxlen=self.history_len) for k in self.dist_map}
        self.display = np.zeros((200, 200, 3), dtype=np.uint8)

    def step(self):
        # 1. Input Handling
        signals = {}
        for key in self.buffers:
            val = self.get_blended_input(key, 'mean')
            if val is None: val = 0.0
            self.buffers[key].append(float(val))
            signals[key] = float(val)

        speed = self.get_blended_input('conduction_speed', 'mean')
        if speed is None: speed = 10.0
        if speed < 0.1: speed = 0.1
        
        # 2. Compute Delays
        fps = 60.0 
        delayed_signals = []
        for key, dist in self.dist_map.items():
            delay_frames = int((dist / speed) * fps)
            if delay_frames >= self.history_len: delay_frames = self.history_len - 1
            delayed_signals.append(self.buffers[key][-1 - delay_frames])

        # 3. Integration
        total_sum = sum(delayed_signals)
        energy = total_sum ** 2
        
        # 4. Draw
        self._draw_field(delayed_signals, energy, speed)
        
        self.set_output('integrated_field', self.display)
        self.set_output('coherence', energy)
        self.set_output('merged_signal', total_sum)

    def _draw_field(self, signals, energy, speed):
        self.display.fill(10)
        positions = [(100, 30), (100, 170), (30, 100), (170, 100)]
        labels = ['F', 'O', 'T', 'P']
        
        # Sources
        for i, (sig, pos) in enumerate(zip(signals, positions)):
            # FIX: Explicit int() cast
            val = int(np.clip(abs(sig) * 1000000, 50, 255))
            cv2.circle(self.display, pos, 10, (0, val, 0), -1)
            cv2.putText(self.display, labels[i], (pos[0]-5, pos[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Center (Observer)
        radius = int(np.clip(energy * 50, 5, 50))
        glow = int(np.clip(energy * 200, 50, 255))
        # FIX: Explicit int() cast
        color = (glow, glow, glow) if glow <= 150 else (0, 215, 255)
        
        cv2.circle(self.display, (100, 100), radius, color, -1)
        cv2.putText(self.display, f"{speed:.1f} m/s", (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name): return getattr(self, '_outs', {}).get(name)
    def set_output(self, name, val): 
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val