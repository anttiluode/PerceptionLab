"""
SeptumPacemakerNode (v19) -- The Theta Pacemaker
-------------------------------------------------
The medial septum as a bank of coupled Kuramoto oscillators producing a
dynamical theta rhythm with phase reset (Vardalakis et al. 2024, eLife). This
is the pacemaker the resonator neuron's own latency findings kept demanding:
v16 (latency = logic depth, "why a brain needs a clock"), v17 (carry latency
linear in word width), v18 (the field demanded a capture). The clock those
results assumed is GENERATED here, by the same edge-of-chaos / coupled-
oscillator mechanism that produced the original PerceptionLab accident.

  Each oscillator i:
     dtheta_i = omega_i + (k/N) sum_j sin(theta_j - theta_i)
                        + G_reset * X(t) * Z(theta_i)
  ensemble theta read from the order parameter r = (1/N) sum exp(i theta_j);
  output theta = rectified cos(mean phase) scaled by coherence.

OUTPUTS:
  theta    : the generated theta drive (rectified cosine) -- wire into the
             reset/advance ports of resonator neurons to PACE them
  peak     : 1.0 at each theta peak (one per cycle) -- the stage-advance tick
  phase    : the ensemble mean phase (0..2pi)
  coherence: the order parameter A (0=incoherent, 1=synchronized)
  plot     : theta trace + coherence

INPUTS:
  reset_in : X(t), a downstream-firing feedback that phase-resets the clock
             (wire a neuron's spike/arrived here to close the loop -- exactly
             the accident's downstream->coupler feedback shape)
  f0_mod   : optional modulation of the center theta frequency

Drop this node next to the v18 ResonatorSoma/Axon/Synapse nodes: wire `peak`
or `theta` into the neurons' `reset` ports so each theta cycle re-arms them,
and wire a neuron's output back into `reset_in` to close the loop.

PerceptionLab / Antti Luode, with Claude. Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


def prc_Z(theta):
    """Phase response curve: -sin(theta). Zero at the peak, advances the phase
    toward the peak elsewhere (Lengyel 2005 / Akam 2012 style, as in
    Vardalakis 2024)."""
    return -np.sin(theta)


class SeptumPacemakerNode(BaseNode):
    NODE_CATEGORY = "Resonator"
    NODE_COLOR = QtGui.QColor(220, 180, 40)  # septum gold (pacemaker)

    def __init__(self, n=50, f0=7.0, k=6.0, g_reset=25.0):
        super().__init__()
        self.node_title = "Septum (Theta Pacemaker)"

        self.inputs = {
            'reset_in': 'signal',   # X(t): downstream firing fed back (loop)
            'f0_mod':   'signal',   # optional center-frequency modulation
        }
        self.outputs = {
            'theta':     'signal',   # generated theta drive (rectified cosine)
            'peak':      'signal',   # 1.0 at each theta peak (stage advance)
            'phase':     'signal',   # ensemble mean phase
            'coherence': 'signal',   # order parameter A
            'plot':      'image',
        }

        self.n = int(n)
        self.f0 = float(f0)
        self.k = float(k)
        self.g_reset = float(g_reset)
        self.sigma = 0.5
        self.dt = 1e-3
        self.g_theta = 1.0

        rng = np.random.default_rng(0)
        self.omega = 2*np.pi * rng.normal(self.f0, self.sigma, self.n)
        self.theta = rng.uniform(0, 2*np.pi, self.n)
        self.A = 0.0
        self.phi = 0.0
        self.I_theta = 0.0
        self.peak = 0.0
        self._last_phase = None

        self.theta_hist = deque(maxlen=400)
        self.coh_hist = deque(maxlen=400)
        self.plot_img = np.zeros((110, 260, 3), np.uint8)

    def _order_parameter(self):
        r = np.mean(np.exp(1j * self.theta))
        return abs(r), np.angle(r)

    def step(self):
        # optional frequency modulation
        f0_mod = self.get_blended_input('f0_mod', 'sum')
        if f0_mod is not None:
            # gentle shift of all natural frequencies
            self.omega += 2*np.pi * float(f0_mod) * 0.01

        reset_in = self.get_blended_input('reset_in', 'sum')
        reset_in = 0.0 if reset_in is None else float(reset_in)

        A, phi = self._order_parameter()
        self.A, self.phi = A, phi
        coupling = self.k * A * np.sin(phi - self.theta)
        reset = self.g_reset * reset_in * prc_Z(self.theta)
        self.theta = (self.theta + self.dt * (self.omega + coupling + reset)) % (2*np.pi)

        A2, phi2 = self._order_parameter()
        self.I_theta = self.g_theta * A2 * max(np.cos(phi2), 0.0)

        # peak detection: mean phase wraps through 0 (cos maximal)
        self.peak = 0.0
        if self._last_phase is not None and self._last_phase > 2.5 and phi2 < 0.5:
            self.peak = 1.0
        self._last_phase = phi2

        self.theta_hist.append(self.I_theta)
        self.coh_hist.append(A2)
        self._render()

    def _render(self):
        h, w = 110, 260
        img = np.zeros((h, w, 3), np.uint8)
        th = np.array(self.theta_hist)
        if len(th) > 2:
            m = th.max() + 1e-9
            for i in range(1, len(th)):
                x0 = int((i-1)/len(th)*w); x1 = int(i/len(th)*w)
                y0 = int(70 - th[i-1]/m*55); y1 = int(70 - th[i]/m*55)
                cv2.line(img, (x0, y0), (x1, y1), (220, 180, 40), 1)
        ch = np.array(self.coh_hist)
        if len(ch) > 2:
            for i in range(1, len(ch)):
                x0 = int((i-1)/len(ch)*w); x1 = int(i/len(ch)*w)
                y0 = int(105 - ch[i-1]*28); y1 = int(105 - ch[i]*28)
                cv2.line(img, (x0, y0), (x1, y1), (60, 160, 220), 1)
        cv2.putText(img, f"theta {self.f0:.1f}Hz  A={self.A:.2f}", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 200, 120), 1)
        if self.peak > 0.5:
            cv2.putText(img, "PEAK", (w-50, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 220, 255), 1)
        self.plot_img = img

    def get_output(self, port_name):
        if port_name == 'theta':     return float(self.I_theta)
        if port_name == 'peak':      return float(self.peak)
        if port_name == 'phase':     return float(self.phi)
        if port_name == 'coherence': return float(self.A)
        if port_name == 'plot':      return self.plot_img.astype(np.float32)/255.0
        return None

    def get_display_image(self):
        return QtGui.QImage(self.plot_img.data, 260, 110, 260*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Oscillators (N)", "n", self.n, None),
            ("Theta freq f0 (Hz)", "f0", self.f0, "float"),
            ("Coupling k", "k", self.k, "float"),
            ("Reset gain", "g_reset", self.g_reset, "float"),
        ]

    def set_config_options(self, options):
        if "n" in options:
            self.n = int(options["n"])
            rng = np.random.default_rng(0)
            self.omega = 2*np.pi * rng.normal(self.f0, self.sigma, self.n)
            self.theta = rng.uniform(0, 2*np.pi, self.n)
        if "f0" in options:
            self.f0 = float(options["f0"])
            rng = np.random.default_rng(0)
            self.omega = 2*np.pi * rng.normal(self.f0, self.sigma, self.n)
        if "k" in options:
            self.k = float(options["k"])
        if "g_reset" in options:
            self.g_reset = float(options["g_reset"])
