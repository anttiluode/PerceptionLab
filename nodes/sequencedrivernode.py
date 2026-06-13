"""
Sequence Driver Node (forward / reverse cycle source)
=====================================================
Drives the Predictive Cortex with a CYCLE of patterns in a chosen direction. The
whole point of the grain-of-the-arrow demo is that order matters: teach the cortex
the forward cycle (it carves a transition memory = the write-side A), then flip the
direction to "reverse" and watch the surprise — and therefore the wattage — jump,
because the reverse order fights the current the cortex just learned.

  query_vec : the current pattern in the cycle (optionally corrupted)
  query_img : same, as an image
  index     : which pattern is showing
  dir       : +1 forward, -1 reverse (also shown on the node)

Set Direction to "forward", let it learn for a while (forward surprise/wattage falls
as the discount appears), then set Direction to "reverse" and watch the cost rise.
At g=0 on the cortex there is no learned current and no difference — the control.

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class SequenceDriverNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(70, 120, 110)

    def __init__(self, num_patterns=6, side=20, corruption=0.15, dwell=1, direction=1):
        super().__init__()
        self.node_title = "Sequence Driver"
        self.inputs = {}
        self.outputs = {
            'query_vec': 'spectrum', 'query_img': 'image',
            'clean_vec': 'spectrum', 'index': 'signal', 'dir': 'signal',
        }
        self.num_patterns = int(num_patterns)
        self.side = int(side)
        self.corruption = float(corruption)
        self.dwell = int(dwell)
        self.direction = int(direction)
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.idx = 0
        self._build()
        self.query_vec = self.Praw[0].copy()
        self.clean_vec = self.Praw[0].copy()
        self.query_img = self._to_img(self.query_vec)

    def _build(self):
        # Disjoint blobs in a 2x3 grid -> orthogonal patterns (non-overlapping support),
        # so the cortex grows one clean node per pattern and the transition memory is crisp.
        S = self.side
        self.dim = S * S
        pats = []
        for k in range(self.num_patterns):
            im = np.zeros((S, S), np.float32)
            r = (k // 3) % 2          # row 0/1
            c = k % 3                 # col 0/1/2
            y0 = int((0.10 + 0.45 * r) * S); y1 = int((0.40 + 0.45 * r) * S)
            x0 = int((0.07 + 0.31 * c) * S); x1 = int((0.30 + 0.31 * c) * S)
            im[y0:y1, x0:x1] = 1.0
            pats.append(im.flatten())
        self.Praw = np.stack(pats).astype(np.float32)

    def _to_img(self, vec):
        S = self.side
        g = np.clip(vec.reshape(S, S), 0, None); g = g / (g.max() + 1e-9)
        return np.stack([g, g, g], axis=-1).astype(np.float32)

    def step(self):
        step_n = self.t // max(self.dwell, 1)
        self.idx = (self.direction * step_n) % self.num_patterns
        clean = self.Praw[self.idx]
        q = clean + self.corruption * self.rng.standard_normal(self.dim).astype(np.float32)
        self.clean_vec = clean.copy()
        self.query_vec = q
        self.query_img = self._to_img(np.clip(q, 0, None))
        self.t += 1

    def get_output(self, port_name):
        if port_name == 'query_vec': return self.query_vec
        if port_name == 'clean_vec': return self.clean_vec
        if port_name == 'query_img': return self.query_img
        if port_name == 'index':     return float(self.idx)
        if port_name == 'dir':       return float(self.direction)
        return None

    def get_display_image(self):
        S = self.side
        q = (np.clip(self.query_img, 0, 1) * 255).astype(np.uint8)
        big = cv2.resize(q, (128, 128), interpolation=cv2.INTER_NEAREST)
        tag = "FWD ->" if self.direction > 0 else "<- REV"
        cv2.putText(big, f"#{self.idx} {tag}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 220, 120) if self.direction > 0 else (90, 150, 255), 1)
        big = np.ascontiguousarray(big)
        return QtGui.QImage(big.data, 128, 128, 128 * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Num Patterns", "num_patterns", self.num_patterns, None),
            ("Side (px)", "side", self.side, None),
            ("Corruption", "corruption", self.corruption, "float"),
            ("Dwell (frames)", "dwell", self.dwell, None),
            ("Direction (1 fwd / -1 rev)", "direction", self.direction, None),
        ]

    def set_config_options(self, options):
        rebuild = False
        if "num_patterns" in options: self.num_patterns = int(options["num_patterns"]); rebuild = True
        if "side" in options: self.side = int(options["side"]); rebuild = True
        if "corruption" in options: self.corruption = float(options["corruption"])
        if "dwell" in options: self.dwell = int(options["dwell"])
        if "direction" in options:
            d = int(options["direction"]); self.direction = 1 if d >= 0 else -1
        if rebuild: self._build()
