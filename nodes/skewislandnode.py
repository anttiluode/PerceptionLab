"""
Skew Island Node (The Arrow Reader)
-----------------------------------
Antti's SpectralIslandsV3 result, live: split the lag-covariance of a
MULTICHANNEL field into its symmetric half (power, phase-blind) and its skew
half (rotation, direction). The skew half's conjugate eigenpairs ARE the
spectral islands — Koopman rotation modes — and the SIGN of each rate is that
island's chirality: its arrow of time. (spectral_islands.py: lag_covariance,
extract_islands, reused verbatim with attribution.)

  C_tau = E[ r(t) r(t-tau)^T ]
  S = (C+C^T)/2   symmetric : power, time-symmetric, direction-BLIND
  A = (C-C^T)/2   skew      : rotation, chirality, the arrow

WHAT IT GIVES:
  - frequency  : dominant island rotation rate (cyc/step)
  - chirality  : sign of that rate = arrow of time (+1 / -1)
  - persistence: how steady the dominant island is over the window
  - locked     : a phase-locked oscillation at the discovered rate (sin(phase))
  - plot       : island rates as bars + a chirality arrow

READ THIS BEFORE TRUSTING THE CHIRALITY (measured, June 2026):
  Chirality is meaningful ONLY because this node's input is MULTICHANNEL —
  distinct channels with directed coupling (pattern A leads B, not B->A).
  On a SINGLE scalar stream's delay embedding the skew half is time-SYMMETRIC
  to ~1 part in 1e4 (a 1-D autocorrelation is even — Wiener-Khinchin), so the
  arrow is INVISIBLE there. That is why this node wants a FIELD (e.g. the
  Image->Vector latent, several Fork channels), NOT the scalar Hankel state.
  The dendrite reads frequency+persistence; only the field carries direction.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class SkewIslandNode(BaseNode):
    NODE_CATEGORY = "Temporal"
    NODE_COLOR = QtGui.QColor(210, 110, 200)  # rotation magenta

    def __init__(self, lag=4, hist_len=300, max_channels=24):
        super().__init__()
        self.node_title = "Skew Island (Arrow)"

        self.inputs = {
            'field_in':  'spectrum',   # MULTICHANNEL — this is the point
            'scale_mod': 'signal',
        }
        self.outputs = {
            'frequency':   'signal',
            'chirality':   'signal',
            'persistence': 'signal',
            'locked':      'signal',
            'plot':        'image',
        }

        self.lag = int(lag)
        self.hist_len = int(hist_len)
        self.max_channels = int(max_channels)

        self.buf = deque(maxlen=self.hist_len)
        self.frequency = 0.0
        self.chirality = 0.0
        self.persistence = 0.0
        self.phase = 0.0
        self.locked = 0.0
        self.omegas = np.zeros(1)
        self.ref_plane = None
        self._tick = 0
        self.plot_img = np.zeros((128, 220, 3), np.uint8)

    # --- Antti's primitives (SpectralIslandsV3), reused ----------------------
    @staticmethod
    def _lag_covariance(R, tau):
        return R[tau:].T @ R[:-tau] / (len(R) - tau)

    @staticmethod
    def _extract_islands(C):
        A = 0.5 * (C - C.T)
        w, V = np.linalg.eig(A)
        om = w.imag
        keep = om > 1e-9
        om, V = om[keep], V[:, keep]
        order = np.argsort(-om)
        return om[order], V[:, order], A

    def step(self):
        vec = self.get_blended_input('field_in', 'first')
        if vec is None:
            return
        v = np.asarray(vec, float).ravel()
        # keep the most active channels (cheap, keeps the operator small)
        if len(v) > self.max_channels:
            idx = np.argsort(-np.abs(v - v.mean()))[:self.max_channels]
            v = v[np.sort(idx)]
        self.buf.append(v)
        self._tick += 1

        if len(self.buf) > self.lag + 8 and self._tick % 6 == 0:
            try:
                R = np.stack(self.buf)                       # (T, K) channel overlaps
                R = R - R.mean(0, keepdims=True)
                sd = R.std(0, keepdims=True); sd[sd < 1e-9] = 1.0
                R = R / sd
                C = self._lag_covariance(R, self.lag)
                om, V, A = self._extract_islands(C)
                if len(om):
                    self.omegas = om
                    w0 = om[0]
                    self.frequency = float(np.arcsin(np.clip(abs(w0), 0, 1)) / (2*np.pi*self.lag))
                    # CHIRALITY needs a CONSISTENT reference plane: a freshly extracted
                    # eigenvector can be either conjugate partner, so its raw sign floats.
                    # Lock the dominant plane once; read the signed angular momentum
                    # L = Im(z z*_lag) (SpectralIslandsV3.island_chirality) against it.
                    # Relative to that frame the arrow is stable and FLIPS if the stream's
                    # direction reverses. (sign(omega) alone cannot: eig keeps only +omega.)
                    if self.ref_plane is None or len(self.ref_plane) != V.shape[0]:
                        self.ref_plane = V[:, 0].copy()
                    z = R @ np.conj(self.ref_plane)
                    L = (z[self.lag:] * np.conj(z[:-self.lag])).imag
                    self.chirality = float(np.sign(L.mean())) if len(L) else 0.0
                    self.persistence = float(abs(w0) / (np.abs(om).sum() + 1e-9))
            except np.linalg.LinAlgError:
                pass

        # phase-locked oscillation at the discovered rate, with the discovered arrow
        self.phase += 2*np.pi * self.frequency * self.chirality
        self.locked = float(np.sin(self.phase))
        self._render()

    def _render(self):
        h, w = 128, 220
        img = np.zeros((h, w, 3), np.uint8)
        om = self.omegas[:8]
        if len(om):
            m = np.abs(om).max() + 1e-9
            bw = w / max(len(om), 1)
            for i, o in enumerate(om):
                ph = int(abs(o)/m * (h-30))
                x = int(i*bw)
                cv2.rectangle(img, (x, h-20-ph), (int(x+bw-2), h-20), (210, 110, 200), -1)
        arrow = "->  +" if self.chirality >= 0 else "<-  -"
        cv2.putText(img, f"f={self.frequency:.3f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.putText(img, f"arrow {arrow}", (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 220, 255), 1)
        self.plot_img = img

    def get_output(self, port_name):
        if port_name == 'frequency':   return float(self.frequency)
        if port_name == 'chirality':   return float(self.chirality)
        if port_name == 'persistence': return float(self.persistence)
        if port_name == 'locked':      return float(self.locked)
        if port_name == 'plot':        return self.plot_img.astype(np.float32)/255.0
        return None

    def get_display_image(self):
        return QtGui.QImage(self.plot_img.data, 220, 128, 220*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Lag (tau)", "lag", self.lag, None),
            ("History length", "hist_len", self.hist_len, None),
            ("Max channels", "max_channels", self.max_channels, None),
        ]

    def set_config_options(self, options):
        if "lag" in options:
            self.lag = int(options["lag"])
        if "hist_len" in options:
            self.hist_len = int(options["hist_len"]); self.buf = deque(maxlen=self.hist_len)
        if "max_channels" in options:
            self.max_channels = int(options["max_channels"])
