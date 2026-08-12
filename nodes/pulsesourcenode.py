"""
Pulse Source Node
-----------------

Emits a short pulse every `period` steps, so the Wave Field can be kicked
without hand-triggering anything.

    period = 0   -> single shot at start, then silence (best for ringdown)
    period > 0   -> repeating pulse

Keep `period` comfortably longer than the time a disturbance needs to cross
the aperture line, or successive pulses overlap and the lag measurement turns
into a mush of superposed wavefronts. With the Wave Field defaults that is a
few hundred steps.

Outputs:
    signal : the pulse
    gate   : 1.0 while pulsing, 0.0 otherwise (handy as a scope channel)
"""

from __future__ import annotations

import numpy as np
import cv2
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class PulseSourceNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80)

    def __init__(self, period=300, width=3, amplitude=1.0):
        super().__init__()
        self.node_title = "Pulse Source"

        self.inputs = {"trigger": "signal"}
        self.outputs = {"signal": "signal", "gate": "signal"}

        self.period = int(period)
        self.width = int(width)
        self.amplitude = float(amplitude)
        self.warmup = 60          # let the field settle before the first pulse

        self.t = 0
        self.value = 0.0
        self.gate_value = 0.0
        self.last_trigger = 0.0

        self.plot = np.zeros((60, 200, 3), dtype=np.uint8)
        self.hist = []

    def step(self):
        self.t += 1
        firing = False

        if self.t > self.warmup:
            if self.period <= 0:
                firing = (self.t - self.warmup) <= max(1, self.width)
            else:
                phase = (self.t - self.warmup) % max(1, self.period)
                firing = phase < max(1, self.width)

        # Optional external edge trigger.
        trig = self.get_blended_input("trigger", "sum")
        if trig is not None:
            tv = float(trig)
            if tv > 0.5 and self.last_trigger <= 0.5:
                firing = True
            self.last_trigger = tv

        self.gate_value = 1.0 if firing else 0.0
        self.value = self.amplitude if firing else 0.0

        self.hist.append(self.value)
        if len(self.hist) > 200:
            self.hist = self.hist[-200:]
        self._render()

    def _render(self):
        h, w = 60, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if len(self.hist) >= 2:
            a = np.asarray(self.hist, dtype=np.float32)
            m = max(1e-6, float(np.max(np.abs(a))))
            prev = None
            for i, v in enumerate(a):
                x = int(i / max(1, len(a) - 1) * (w - 1))
                y = int(h - 4 - (v / m) * (h - 10))
                if prev is not None:
                    cv2.line(img, prev, (x, y), (120, 255, 160), 1)
                prev = (x, y)
        cv2.putText(img, f"T={self.period} w={self.width}", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
        self.plot = np.ascontiguousarray(img)

    def get_output(self, port_name):
        if port_name == "signal":
            return float(self.value)
        if port_name == "gate":
            return float(self.gate_value)
        return None

    def get_display_image(self):
        h, w = self.plot.shape[:2]
        return QtGui.QImage(self.plot.data, w, h, w * 3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Period", "period", self.period, None),
            ("Width", "width", self.width, None),
            ("Amplitude", "amplitude", self.amplitude, "float"),
            ("Warmup", "warmup", self.warmup, None),
        ]

    def set_config_options(self, options):
        if "period" in options:
            self.period = int(float(options["period"]))
        if "width" in options:
            self.width = int(float(options["width"]))
        if "amplitude" in options:
            self.amplitude = float(options["amplitude"])
        if "warmup" in options:
            self.warmup = int(float(options["warmup"]))
