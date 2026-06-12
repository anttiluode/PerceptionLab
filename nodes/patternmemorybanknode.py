"""
Pattern Memory Bank Node (The Engram Library)
=============================================
The "dataset" for the metabolic associative-memory workflow. Holds a small set of
recognizable shape templates, and each frame presents ONE of them, corrupted with
noise + occlusion, as a query. Emits both vectors (for the cortex to recall) and
images (for you to watch the noise get cleaned).

  query_vec  : corrupted pattern  -> Associative Cortex .recall
  clean_vec  : the ground-truth   -> Associative Cortex .teach   (one-shot storage)
  query_img / clean_img : same, as images, for the eye
  index      : which template is showing

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class PatternMemoryBankNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 130, 90)

    def __init__(self, num_patterns=5, side=20, corruption=0.5, dwell=50):
        super().__init__()
        self.node_title = "Engram Library"
        self.inputs = {'select': 'signal', 'corruption_mod': 'signal'}
        self.outputs = {
            'query_vec': 'spectrum', 'clean_vec': 'spectrum',
            'query_img': 'image', 'clean_img': 'image', 'index': 'signal',
        }
        self.num_patterns = int(num_patterns)
        self.side = int(side)
        self.corruption = float(corruption)
        self.dwell = int(dwell)
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.idx = 0
        self._build()
        self.query_vec = self.Praw[0].copy()
        self.clean_vec = self.Praw[0].copy()
        self.query_img = self._to_img(self.query_vec)
        self.clean_img = self._to_img(self.clean_vec)

    def _build(self):
        S = self.side
        def shp(kind):
            im = np.zeros((S, S), np.float32)
            a, b = int(S*0.25), int(S*0.75)
            if kind % 5 == 0: cv2.circle(im, (a, a), max(2, S//7), 1.0, -1)            # dot TL
            if kind % 5 == 1: cv2.rectangle(im, (b-3, a-3), (b+3, a+3), 1.0, -1)        # sq TR
            if kind % 5 == 2:
                pts = np.array([[a, b+3], [a-4, b-2], [a+4, b-2]]); cv2.fillPoly(im, [pts], 1.0)  # tri BL
            if kind % 5 == 3:
                cv2.line(im, (b, b-4), (b, b+4), 1, 2); cv2.line(im, (b-4, b), (b+4, b), 1, 2)     # plus BR
            if kind % 5 == 4:
                c = S//2; cv2.line(im, (c-3, c-3), (c+3, c+3), 1, 2); cv2.line(im, (c+3, c-3), (c-3, c+3), 1, 2)  # X
            return im.flatten()
        self.dim = S * S
        self.Praw = np.stack([shp(k) for k in range(self.num_patterns)]).astype(np.float32)

    def _to_img(self, vec):
        S = self.side
        g = np.clip(vec.reshape(S, S), 0, None)
        g = g / (g.max() + 1e-9)
        return np.stack([g, g, g], axis=-1).astype(np.float32)

    def step(self):
        sel = self.get_blended_input('select', 'sum')
        cmod = self.get_blended_input('corruption_mod', 'sum')
        if sel is not None:
            self.idx = int(abs(sel)) % self.num_patterns
        else:
            self.idx = (self.t // self.dwell) % self.num_patterns
        corr = self.corruption * (1.0 + (float(cmod) if cmod is not None else 0.0))

        clean = self.Praw[self.idx]
        q = clean + corr * self.rng.standard_normal(self.dim).astype(np.float32)
        if self.rng.random() < 0.5:                       # half the time, occlude a strip
            m = np.ones(self.dim, np.float32)
            s = self.rng.integers(0, max(1, self.dim - self.dim//3))
            m[s:s + self.dim//3] = 0.0
            q = q * m

        self.clean_vec = clean.copy()
        self.query_vec = q
        self.clean_img = self._to_img(clean)
        self.query_img = self._to_img(np.clip(q, 0, None))
        self.t += 1

    def get_output(self, port_name):
        if port_name == 'query_vec': return self.query_vec
        if port_name == 'clean_vec': return self.clean_vec
        if port_name == 'query_img': return self.query_img
        if port_name == 'clean_img': return self.clean_img
        if port_name == 'index':     return float(self.idx)
        return None

    def get_display_image(self):
        S = self.side
        q = (np.clip(self.query_img, 0, 1) * 255).astype(np.uint8)
        big = cv2.resize(q, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"#{self.idx} noisy", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 120), 1)
        big = np.ascontiguousarray(big)
        return QtGui.QImage(big.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Num Patterns", "num_patterns", self.num_patterns, None),
            ("Side (px)", "side", self.side, None),
            ("Corruption", "corruption", self.corruption, "float"),
            ("Dwell (frames)", "dwell", self.dwell, None),
        ]

    def set_config_options(self, options):
        rebuild = False
        if "num_patterns" in options: self.num_patterns = int(options["num_patterns"]); rebuild = True
        if "side" in options: self.side = int(options["side"]); rebuild = True
        if "corruption" in options: self.corruption = float(options["corruption"])
        if "dwell" in options: self.dwell = int(options["dwell"])
        if rebuild: self._build()
