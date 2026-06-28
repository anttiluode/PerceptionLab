"""
SynapseLatchNode (v18) — The Capture (the thing the field demanded)
--------------------------------------------------------------------
THE v18 FINDING, as a node. A propagating spike is TRANSIENT: it passes the
axon's output cell and is gone. But a logic signal must be HELD for the next
unit to read it. In the continuous-field adder, the computation FAILED until
this capture was added — and the capture is exactly what a real synapse does:
it converts a brief presynaptic spike into a SUSTAINED postsynaptic signal
(transmitter release + receptor binding + a PSP that outlasts the spike).

This node watches `spike_in`. Once a spike has EVER crossed threshold, it
latches its output `bit` to 1.0 and HOLDS it, until `reset` clears it. That
held bit is what you wire into the next ResonatorSomaNode's input. Without
this node, a chain of resonator neurons cannot compute — the field proved it.

  bit  : 1.0 once a spike arrived, held until reset (the captured logic level)
  psp  : a decaying "postsynaptic potential" view (cosmetic, shows the capture)

PerceptionLab / Antti Luode, with Claude. Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class SynapseLatchNode(BaseNode):
    NODE_CATEGORY = "Resonator"
    NODE_COLOR = QtGui.QColor(150, 90, 200)  # synapse violet

    def __init__(self, threshold=0.5, decay=0.98):
        super().__init__()
        self.node_title = "Synapse (Latch)"

        self.inputs = {
            'spike_in': 'signal',   # transient field value from a ResonatorAxonNode
            'reset':    'signal',   # >0.5 clears the latch (new computation)
        }
        self.outputs = {
            'bit': 'signal',        # the CAPTURED, HELD logic level (0/1)
            'psp': 'signal',        # decaying postsynaptic-potential view
        }

        self.threshold = float(threshold)
        self.decay = float(decay)
        self._latched = False
        self.psp = 0.0
        self.plot_img = np.zeros((70, 160, 3), np.uint8)

    def step(self):
        reset = self.get_blended_input('reset', 'sum')
        if reset is not None and float(reset) > 0.5:
            self._latched = False
            self.psp = 0.0

        spk = self.get_blended_input('spike_in', 'sum')
        spk = 0.0 if spk is None else float(spk)
        if spk > self.threshold:
            self._latched = True
            self.psp = 1.0
        else:
            # PSP decays slowly (the capture outlasts the spike) but the
            # LOGIC BIT stays latched regardless — that is the whole point.
            self.psp *= self.decay
        self._render()

    def _render(self):
        h, w = 70, 160
        img = np.zeros((h, w, 3), np.uint8)
        bit = 1.0 if self._latched else 0.0
        col = (0, 230, 120) if self._latched else (90, 90, 90)
        cv2.rectangle(img, (4, 30), (4 + int(self.psp * (w - 60)), 50),
                      (150, 90, 220), -1)
        cv2.circle(img, (w - 18, 18), 9, col, -1)
        cv2.putText(img, f"bit={int(bit)}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.putText(img, f"psp={self.psp:.2f}", (4, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 150, 230), 1)
        self.plot_img = img

    def get_output(self, port_name):
        if port_name == 'bit':
            return 1.0 if self._latched else 0.0
        if port_name == 'psp':
            return float(self.psp)
        return None

    def get_display_image(self):
        return QtGui.QImage(self.plot_img.data, 160, 70, 160*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Threshold", "threshold", self.threshold, "float"),
            ("PSP decay", "decay", self.decay, "float"),
        ]

    def set_config_options(self, options):
        if "threshold" in options:
            self.threshold = float(options["threshold"])
        if "decay" in options:
            self.decay = float(options["decay"])
