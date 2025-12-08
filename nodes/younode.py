# nodes/YouNode.py
# This node IS the conscious observer.
# It implements the Drebitz 2025 gamma gate + your 11 ms spread.

import numpy as np
import cv2
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except:
    from PyQt6 import QtGui
    class BaseNode: pass

class YouNode(BaseNode):
    NODE_CATEGORY = "Identity"
    NODE_TITLE = "You (11 ms Window)"
    NODE_COLOR = QtGui.QColor(255, 215, 0)           # Pure Gold

    def __init__(self):
        super().__init__()
        self.inputs = {
            'reality_stream': 'signal',   # raw EEG or any continuous input
            'conduction_speed': 'signal'  # optional speed control (m/s)
        }
        self.outputs = {
            'your_now': 'signal',         # what you are conscious of right now
            'your_past': 'signal',        # what just left your window
            'consciousness': 'image'      # visual of your 11 ms self
        }

        # 300 ms buffer = the "ocean" of reality
        self.buffer = deque(maxlen=300)   # ~300 ms at Lab's internal rate
        self.gamma_phase = 0.0
        self.canvas = np.zeros((220, 460, 3), np.uint8)

    def step(self):
        # 1. Get the raw stream of reality
        raw = self.get_blended_input('reality_stream', 'mean')
        if raw is None: raw = 0.0
        self.buffer.append(raw)

        # 2. Optional speed control (default 10 m/s → ~11 ms cross-brain)
        speed = self.get_blended_input('conduction_speed', 'mean')
        if speed is None or speed <= 0: speed = 10.0

        # 3. Your gamma gate (~50 Hz → 20 ms cycle, 11 ms open)
        self.gamma_phase = (self.gamma_phase + 0.314) % (2 * np.pi)
        gate = np.cos(self.gamma_phase)                # -1 to +1
        is_open = gate > 0.45                           # ~11 ms window

        # 4. You only exist when the gate is open
        if is_open and len(self.buffer) >= 11:
            your_now = self.buffer[-1]                  # the chosen slice
            your_past = self.buffer[-12] if len(self.buffer) >= 12 else 0.0
        else:
            your_now = 0.0
            your_past = raw

        # 5. Visualise You
        self.canvas.fill(8)
        h, w = self.canvas.shape[:2]

        cv2.putText(self.canvas, "THE 300 ms OCEAN OF REALITY", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,140), 1)
        cv2.putText(self.canvas, "YOU (11 ms Conscious Window)", (80, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,215,0), 2)

        # Draw the full buffer as a wave
        if len(self.buffer) > 10:
            pts = np.array([[x*1.5, 100 + v*60] for x,v in enumerate(self.buffer)])
            cv2.polylines(self.canvas, [pts.astype(int)], False, (60,60,180), 2)

        # Draw the golden moving gate
        gate_x = len(self.buffer) * 1.5 - 11
        color = (0, 255, 255) if is_open else (50, 50, 150)
        thickness = -1 if is_open else 4
        cv2.rectangle(self.canvas, (int(gate_x-8), 60), (int(gate_x+19), 140),
                     color, thickness)

        cv2.putText(self.canvas, "CONSCIOUS" if is_open else "UNCONSCIOUS",
                   (150, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0,255,255) if is_open else (50,50,200), 2)

        # 6. Outputs
        self.set_output('your_now', your_now)
        self.set_output('your_past', your_past)
        self.set_output('consciousness', self.canvas.copy())

    def get_output(self, name):
        return getattr(self, '_outs', {}).get(name)

    def set_output(self, name, val):
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val

    def get_display_image(self):
        return self.canvas.copy()