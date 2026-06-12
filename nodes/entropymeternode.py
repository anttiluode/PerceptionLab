"""
Entropy Meter Node (Detailed-Balance Regime + Landauer Floor)
=============================================================
The Entropy Engine (SkewOperatorNode) outputs the skew flux ||A|| — a monotonic
indicator of broken detailed balance. This node reads it as thermodynamics:

  1. REGIME   near-equilibrium (flux ~ 0, time-reversible, detailed balance HOLDS)
              vs driven (flux high, detailed balance BROKEN, entropy produced).
              This is the rest-vs-task split Lynn et al. (2021) measured in the brain:
              the cortex nearly obeys detailed balance at rest and breaks it under load.

  2. ENTROPY-PRODUCTION RATE  sigma_hat ~ flux^2  (the rotational power of A).
              Labelled a PROXY, not calibrated nats: ||A|| is a lower-bound indicator
              of nonequilibrium (an asymmetric cross-correlation => broken DB).

  3. LANDAUER FLOOR  the *thermodynamic minimum* power to sustain that arrow of time,
              k_B T * sigma_hat (illustrative). 

  4. METABOLIC MARKUP  if the Metabolic Spike node's spikes are connected, how many
              k_B T the system actually burns per unit arrow-of-time. For a real
              neuron this is enormous (~1e10 k_B T per spike): brains buy speed and
              reliability, NOT thermodynamic efficiency. We show the gap honestly.

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

K_B = 1.380649e-23
T_BODY = 310.0
E_SPIKE_JOULES = 1.2e-10     # ~1e9 ATP/spike (Attwell & Laughlin 2001), order-of-magnitude


class EntropyMeterNode(BaseNode):
    NODE_CATEGORY = "Metabolic"
    NODE_COLOR = QtGui.QColor(150, 90, 200)  # entropy violet

    def __init__(self, db_threshold=0.04, smoothing=0.05):
        super().__init__()
        self.node_title = "Entropy Meter (Detailed Balance)"
        self.inputs = {
            'skew_flux': 'signal',   # ||A|| from the Entropy Engine
            'spikes':    'signal',   # optional: actual spikes, for the floor-vs-actual markup
        }
        self.outputs = {
            'entropy_rate':   'signal',   # sigma_hat ~ flux^2 (proxy)
            'regime':         'signal',   # 0 = detailed balance (HOLD), 1 = broken (SCAN)
            'landauer_floor': 'signal',   # k_B T * sigma_hat (illustrative, Joules-scaled)
            'meter':          'image',
        }
        self.db_threshold = float(db_threshold)
        self.smoothing = float(smoothing)

        self.flux_s = 0.0
        self.sigma = 0.0
        self.regime = 0.0
        self.floor = 0.0
        self.spike_s = 0.0
        self.peak_sigma = 1e-9
        self.flux_hist = deque(maxlen=200)
        self.display_img = np.zeros((140, 256, 3), dtype=np.uint8)

    def step(self):
        flux = self.get_blended_input('skew_flux', 'sum')
        spikes = self.get_blended_input('spikes', 'sum')
        flux = float(flux) if flux is not None else 0.0
        self.spike_s = (1 - self.smoothing) * self.spike_s + self.smoothing * (float(spikes) if spikes is not None else 0.0)

        # smooth the flux the way a meter reads it
        self.flux_s = (1 - self.smoothing) * self.flux_s + self.smoothing * flux
        self.flux_hist.append(self.flux_s)

        # entropy-production proxy: rotational power of A (>=0, zero iff detailed balance)
        self.sigma = self.flux_s ** 2
        self.peak_sigma = max(self.peak_sigma * 0.9995, self.sigma)

        # regime: near-equilibrium vs driven
        self.regime = float(self.flux_s > self.db_threshold)

        # Landauer thermodynamic floor (illustrative): k_B T per nat of arrow-of-time
        # scale sigma into a nominal nats/s by its running peak so the floor is readable
        nats_per_s = (self.sigma / max(self.peak_sigma, 1e-9))  # normalised 0..1 "arrow rate"
        self.floor = K_B * T_BODY * nats_per_s

        self._render(nats_per_s)

    def get_output(self, port_name):
        if port_name == 'entropy_rate':   return self.sigma
        if port_name == 'regime':         return self.regime
        if port_name == 'landauer_floor': return self.floor
        if port_name == 'meter':          return self.display_img
        return None

    def _render(self, nats):
        h, w = 140, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # flux trace
        H = np.array(self.flux_hist) if len(self.flux_hist) else np.zeros(2)
        mx = max(H.max(), self.db_threshold * 1.5, 1e-6)
        for i in range(1, len(H)):
            x0 = int((i - 1) / len(H) * w); x1 = int(i / len(H) * w)
            y0 = int(70 - H[i - 1] / mx * 60); y1 = int(70 - H[i] / mx * 60)
            cv2.line(img, (x0, y0), (x1, y1), (200, 140, 90), 1)
        # detailed-balance threshold line
        yt = int(70 - self.db_threshold / mx * 60)
        cv2.line(img, (0, yt), (w, yt), (70, 70, 90), 1)
        cv2.putText(img, "skew flux ||A||  (entropy production)", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 140, 90), 1)

        # regime banner
        if self.regime > 0.5:
            txt = "BROKEN DETAILED BALANCE  -> arrow of time, dissipating"
            col = (90, 130, 255)
        else:
            txt = "DETAILED BALANCE  -> reversible, near-free hold"
            col = (120, 200, 120)
        cv2.putText(img, txt, (4, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1)

        # floor vs actual markup
        cv2.putText(img, f"Landauer floor ~ {self.floor:.2e} W  (k_B T * arrow rate)",
                    (4, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 180), 1)
        if self.spike_s > 1e-6:
            actual_W = self.spike_s * E_SPIKE_JOULES * 60.0   # ~per second at 60 fps
            markup = actual_W / max(self.floor, 1e-30)
            cv2.putText(img, f"actual ~ {actual_W:.2e} W   markup x{markup:.1e} above floor",
                        (4, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 130, 255), 1)
        else:
            cv2.putText(img, "(connect spikes for the metabolic markup)",
                        (4, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 110, 120), 1)

        self.display_img = img

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 140, 256 * 3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("DB Threshold (flux)", "db_threshold", self.db_threshold, "float"),
            ("Smoothing", "smoothing", self.smoothing, "float"),
        ]

    def set_config_options(self, options):
        if "db_threshold" in options: self.db_threshold = float(options["db_threshold"])
        if "smoothing" in options: self.smoothing = float(options["smoothing"])
