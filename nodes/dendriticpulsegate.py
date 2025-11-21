#!/usr/bin/env python3
"""
Dendritic Pulse Gate Node
-------------------------
Implements predictive dendritic gating based on:

- Phase-dependent excitability (Drebitz / gamma cycle gating)
- Stock-logic style sequence memory (pattern signatures)
- Gain-based gating (suppression vs amplification)

Behavior:
- Input passes only when phase is in "effective window"
- If matching a known historical pattern -> gain > 1
- Novel patterns suppressed until learned
"""

import numpy as np
from collections import deque
import cv2
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)


class DendriticPulseGateNode(BaseNode):
    NODE_CATEGORY = "Gating"
    NODE_COLOR = QtGui.QColor(180, 80, 200)

    def __init__(self, memory_size=15, threshold=0.5):
        super().__init__()
        self.node_title = "Dendritic Pulse Gate"

        # --- Node I/O ---
        self.inputs = {
            'signal': 'signal',
            'phase': 'signal',     # normalized 0-1
        }
        self.outputs = {
            'gated': 'signal',
            'confidence': 'signal',
            'gain': 'signal',
        }

        # --- Parameters ---
        self.threshold = float(threshold)
        self.history = deque(maxlen=memory_size)
        self.pattern_memory = {}

        # internal state
        self.prediction_confidence = 0.0
        self.last_gain = 0.0
        self.last_output = 0.0

        # display buffer
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)

    # ---------------------------------------------------------
    # Internal Pattern Logic
    # ---------------------------------------------------------

    def _get_pattern_signature(self):
        """Turns history into symbolic trend signature."""
        if len(self.history) < 3:
            return None

        vals = list(self.history)
        sig = []

        for i in range(1, len(vals)):
            diff = vals[i] - vals[i - 1]
            if diff > 0.01:
                sig.append('U')
            elif diff < -0.01:
                sig.append('D')
            else:
                sig.append('S')

        return "".join(sig)

    # ---------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------

    def step(self):
        signal = self.get_blended_input('signal', 'sum') or 0.0
        phase = self.get_blended_input('phase', 'sum') or 0.0

        # Store history first
        self.history.append(signal)

        # Default low gain
        gain = 0.1

        # Phase gating rule
        effective_phase = (phase < 0.15) or (phase > 0.85)

        if not effective_phase:
            # Suppressed if wrong phase
            self.last_output = 0.0
            self.last_gain = 0.0
            return

        # Sequence signature
        sig = self._get_pattern_signature()

        if sig:
            # Have we seen this pattern before?
            if sig in self.pattern_memory:
                count = self.pattern_memory[sig]

                # confidence = frequency of occurrence (scaled)
                self.prediction_confidence = min(1.0, count / 10.0)

                if self.prediction_confidence > self.threshold:
                    gain = 1.0 + self.prediction_confidence
            else:
                # new pattern
                self.pattern_memory[sig] = 0
                self.prediction_confidence *= 0.9

            # reinforce memory
            self.pattern_memory[sig] += 1

        # Output gated signal
        output = signal * gain

        self.last_output = output
        self.last_gain = gain

    # ---------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------

    def get_output(self, port_name):
        if port_name == 'gated':
            return float(self.last_output)
        if port_name == 'confidence':
            return float(self.prediction_confidence)
        if port_name == 'gain':
            return float(self.last_gain)
        return None

    # ---------------------------------------------------------
    # UI Preview
    # ---------------------------------------------------------

    def get_display_image(self):
        img = self.display_img.copy()
        img[:] = (40, 10, 60)

        text = [
            f"gain: {self.last_gain:.3f}",
            f"confidence: {self.prediction_confidence:.3f}",
            f"patterns: {len(self.pattern_memory)}"
        ]

        y = 15
        for t in text:
            cv2.putText(img, t, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 255), 1)
            y += 18

        return QtGui.QImage(
            img.data, 128, 128, 128 * 3, QtGui.QImage.Format.Format_RGB888
        )

    def get_config_options(self):
        return [
            ("Memory Size", "memory_size", len(self.history), None),
            ("Threshold", "threshold", self.threshold, None),
        ]
