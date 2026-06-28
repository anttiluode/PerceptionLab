"""
ResonatorSomaNode (v18) — The Interference Soma
------------------------------------------------
The computing half of the v18 directional resonator neuron, as a PerceptionLab
node. This is the Berglund central cavity: it MIXES its input bits by phasor
interference and reads out a logical bit from the resonance amplitude band.

  encode:  bit 0 -> +1 (phase 0),  bit 1 -> -1 (phase pi)
  soma:    s = sum_j w_j * p_j + bias
  output:  fire = 1  iff  lo <= |s| <= hi      (an amplitude BAND)

|s| is constructive when inputs agree, destructive when they disagree. Because
|s| is a quadratic (nonlinear) function of the inputs, ONE soma computes XOR —
the linearly-inseparable function. Pick the gate kind to set (w, bias, lo, hi).

This node takes up to 3 scalar inputs (in_a, in_b, in_c), thresholds each at
0.5 to a bit, computes the gate, and outputs `fire` (1.0 / 0.0). Wire `fire`
into a ResonatorAxonNode to turn the decision into a directional spike.

PerceptionLab / Antti Luode, with Claude. Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


# gate configs: (weights, bias, lo, hi). Same table as the repo's gates.py.
GATE_TABLE = {
    "XOR":  ([1, 1], 0.0, -0.1, 1.0),
    "XNOR": ([1, 1], 0.0,  1.0, 99.0),
    "AND":  ([1, 1], -2.0, 3.0, 99.0),
    "OR":   ([1, 1], +2.0, -0.1, 3.0),
    "NAND": ([1, 1], -2.0, -0.1, 3.0),
    "NOR":  ([1, 1], +2.0, 3.0, 99.0),
    "NOT":  ([1],    +1.0, 1.0, 99.0),
}


class ResonatorSomaNode(BaseNode):
    NODE_CATEGORY = "Resonator"
    NODE_COLOR = QtGui.QColor(80, 130, 200)  # soma blue

    def __init__(self, gate="XOR"):
        super().__init__()
        self.node_title = "Resonator Soma"

        self.inputs = {
            'in_a': 'signal',
            'in_b': 'signal',
            'in_c': 'signal',   # used by 3-input gates / majority chains
        }
        self.outputs = {
            'fire':      'signal',   # 1.0 if the gate fires, else 0.0
            'amplitude': 'signal',   # |s|, the raw resonance amplitude
        }

        self.gate = str(gate)
        self.fire = 0.0
        self.amplitude = 0.0
        self._bits = (0, 0)
        self.plot_img = np.zeros((128, 200, 3), np.uint8)

    @staticmethod
    def _encode(b):
        return np.exp(1j * np.pi * int(b))

    def step(self):
        cfg = GATE_TABLE.get(self.gate, GATE_TABLE["XOR"])
        weights, bias, lo, hi = cfg
        n_in = len(weights)

        raw = []
        for port in ('in_a', 'in_b', 'in_c')[:max(n_in, 2)]:
            v = self.get_blended_input(port, 'sum')
            raw.append(0 if v is None else (1 if float(v) > 0.5 else 0))
        bits = raw[:n_in]
        self._bits = tuple(bits)

        p = np.array([self._encode(b) for b in bits], complex)
        w = np.asarray(weights, complex)
        s = np.sum(w * p) + complex(bias)
        amp = float(abs(s))
        self.amplitude = amp
        self.fire = 1.0 if (lo - 1e-9 <= amp <= hi + 1e-9) else 0.0
        self._render()

    def _render(self):
        h, w = 128, 200
        img = np.zeros((h, w, 3), np.uint8)
        # show the amplitude as a bar, the band as a green window, fire state
        lo = GATE_TABLE.get(self.gate, GATE_TABLE["XOR"])[2]
        hi = GATE_TABLE.get(self.gate, GATE_TABLE["XOR"])[3]
        scale = 40.0  # px per amplitude unit (|s| maxes ~4)
        y0 = h - 10
        # band window
        band_lo = int(np.clip(lo, 0, 4)) if lo > 0 else 0
        band_hi = int(np.clip(hi, 0, 4))
        cv2.rectangle(img, (0, y0 - int(band_hi*scale)),
                      (w, y0 - int(band_lo*scale)), (15, 45, 15), -1)
        # amplitude bar
        ah = int(np.clip(self.amplitude, 0, 4) * scale)
        col = (0, 230, 120) if self.fire > 0.5 else (90, 90, 200)
        cv2.rectangle(img, (w//2 - 18, y0 - ah), (w//2 + 18, y0), col, -1)
        cv2.putText(img, f"{self.gate}  bits={self._bits}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.putText(img, f"|s|={self.amplitude:.2f}  fire={int(self.fire)}",
                    (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (120, 220, 255), 1)
        self.plot_img = img

    def get_output(self, port_name):
        if port_name == 'fire':
            return float(self.fire)
        if port_name == 'amplitude':
            return float(self.amplitude)
        return None

    def get_display_image(self):
        return QtGui.QImage(self.plot_img.data, 200, 128, 200*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Gate", "gate", self.gate,
             [("XOR", "XOR"), ("XNOR", "XNOR"), ("AND", "AND"),
              ("OR", "OR"), ("NAND", "NAND"), ("NOR", "NOR"), ("NOT", "NOT")]),
        ]

    def set_config_options(self, options):
        if "gate" in options:
            self.gate = str(options["gate"])
