"""
Hankel Node (The Dendrite)
--------------------------
Takens delay-embedding as a first-class node — the move the whole Geometric
Neuron line has rested on since the accident: it is the dendrite stand-in, the
one way to get a STATE out of a single scalar stream when you can't model the
full cable. Stack the recent history of one number into a delay vector and the
trajectory becomes a navigable geometry (Takens 1981).

WHAT IT GIVES (honestly):
  - state    : the current delay vector  [x(t), x(t-tau), ..., x(t-(d-1)tau)]
  - geometry : x(t) vs x(t-lag) — the attractor you can see (the limit cycle)
  - frequency: dominant rotation rate of that orbit, read by a small DMD on the
               buffer (cyc/step). NOTHING is hand-set — it falls out of the operator.
  - persistence (rho): |lambda| of that mode. 1 = a sustained ring; <1 = damped.

WHAT IT CANNOT GIVE (measured, not assumed): the ARROW of time. A scalar
stream's second-order statistics are time-symmetric (its autocorrelation is even
— Wiener-Khinchin), so a delay embedding of ONE channel cannot tell forward from
backward. Chirality is a MULTICHANNEL property — feed a field (many channels) to
the Skew Island node for that. The dendrite reads frequency + persistence; the
field reads direction.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class HankelNode(BaseNode):
    NODE_CATEGORY = "Temporal"
    NODE_COLOR = QtGui.QColor(90, 160, 220)  # dendrite blue

    def __init__(self, dim=32, tau=4, buffer_len=600):
        super().__init__()
        self.node_title = "Hankel (Dendrite)"

        self.inputs = {'signal_in': 'signal'}
        self.outputs = {
            'state':       'spectrum',   # the delay vector (the embedded state)
            'frequency':   'signal',     # dominant rotation rate (cyc/step)
            'persistence': 'signal',     # |lambda| of that mode
            'geometry':    'image',      # the attractor you can see
        }

        self.dim = int(dim)
        self.tau = int(tau)
        self.buffer_len = int(buffer_len)

        self.buf = deque(maxlen=self.buffer_len)
        self.state_vec = np.zeros(self.dim, np.float32)
        self.frequency = 0.0
        self.persistence = 0.0
        self._tick = 0
        self.geo_img = np.zeros((128, 128, 3), np.uint8)

    # --- the delay embedding -------------------------------------------------
    def _embed(self, x):
        """build the (d, N) Hankel matrix from buffer x with spacing tau."""
        N = len(x) - (self.dim - 1) * self.tau
        if N <= 2:
            return None
        return np.stack([x[i*self.tau:i*self.tau+N] for i in range(self.dim)])

    # --- a small DMD on the buffer: frequency + persistence, nothing set -----
    def _dmd(self, H):
        X1, X2 = H[:, :-1], H[:, 1:]
        try:
            U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        r = min(16, int((S > S[0]*1e-10).sum())) if S[0] > 0 else 1
        r = max(r, 1)
        Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].conj().T
        At = Ur.conj().T @ X2 @ Vr @ np.diag(1.0/Sr)
        lam, Wv = np.linalg.eig(At)
        Phi = X2 @ Vr @ np.diag(1.0/Sr) @ Wv
        try:
            b = np.linalg.lstsq(Phi, X1[:, 0].astype(complex), rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        energy = np.abs(b) * np.linalg.norm(Phi, axis=0)
        f = np.abs(np.angle(lam)) / (2*np.pi)                 # cyc/step (columns advance 1 step)
        rho = np.abs(lam)
        min_f = 1.0 / ((self.dim - 1) * self.tau)             # >=1 cycle in window
        pos = f > min_f
        if not pos.any():
            return 0.0, 0.0
        i = np.argmax(energy[pos])
        return float(f[pos][i]), float(rho[pos][i])

    def step(self):
        s = self.get_blended_input('signal_in', 'sum')
        if s is None:
            return
        self.buf.append(float(s))
        self._tick += 1

        x = np.asarray(self.buf, float)
        if len(x) < (self.dim - 1) * self.tau + 4:
            return
        xc = x - x.mean()

        H = self._embed(xc)
        if H is not None:
            self.state_vec = H[:, -1].astype(np.float32)       # current delay vector
            if self._tick % 8 == 0:                            # DMD is the dear part; throttle
                self.frequency, self.persistence = self._dmd(H)
        self._render(xc)

    def _render(self, xc):
        h = w = 128
        img = np.zeros((h, w, 3), np.uint8)
        lag = max(int(round(1.0/self.frequency/4)), 1) if self.frequency > 1e-6 else self.tau
        lag = min(lag, len(xc)-1)
        a, b = xc[:-lag], xc[lag:]
        m = np.max(np.abs(xc)) + 1e-9
        ax = ((a/m*0.45 + 0.5) * (w-1)).astype(int)
        ay = ((1 - (b/m*0.45 + 0.5)) * (h-1)).astype(int)
        for i in range(1, len(ax)):
            cv2.line(img, (ax[i-1], ay[i-1]), (ax[i], ay[i]), (0, 220, 120), 1)
        cv2.putText(img, f"f={self.frequency:.3f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"rho={self.persistence:.3f}", (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1)
        self.geo_img = img

    def get_output(self, port_name):
        if port_name == 'state':
            return self.state_vec
        if port_name == 'frequency':
            return float(self.frequency)
        if port_name == 'persistence':
            return float(self.persistence)
        if port_name == 'geometry':
            return self.geo_img.astype(np.float32) / 255.0
        return None

    def get_display_image(self):
        return QtGui.QImage(self.geo_img.data, 128, 128, 128*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Embedding dim", "dim", self.dim, None),
            ("Tau (delay)", "tau", self.tau, None),
            ("Buffer length", "buffer_len", self.buffer_len, None),
        ]

    def set_config_options(self, options):
        if "dim" in options:
            self.dim = int(options["dim"]); self.state_vec = np.zeros(self.dim, np.float32)
        if "tau" in options:
            self.tau = int(options["tau"])
        if "buffer_len" in options:
            self.buffer_len = int(options["buffer_len"]); self.buf = deque(maxlen=self.buffer_len)
