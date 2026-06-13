"""
Metabolic Spike Node (The Wattage Meter)  —  "the spike IS the wattage"
=======================================================================
Closes the thermodynamic loop of the Geometric-Neuron line. The Entropy Engine
(SkewOperatorNode) measures broken detailed balance — the arrow of time — as the
skew flux ||A||. This node makes the system PAY for it.

It runs the v5 / Chiral-Ear delta-code and bills two meters:

  P_field = c_field * |ds|          continuous cost of moving charge to change content
  P_spike = E_spike * (n spikes)    the action potential + pump-restoration cost
  P_total = P_field + P_spike

In a real neuron the spike dominates the budget (Attwell & Laughlin 2001:
~1e9 ATP per action potential ~ 1e-10 J). So holding a thought is nearly free;
thinking a new one costs spikes. That asymmetry — the delta-code — is the
sparse-coding energy economy of cortex, here as a live meter.

------------------------------------------------------------------------------
v2 ADDITION — billing the SURPRISE (the grain of the arrow):
  An optional `surprise` input (the prediction-error residual x - pred from the
  Predictive Cortex). When connected, the delta-code fires on the MAGNITUDE of the
  surprise rather than on the change in raw content. This is the key refinement from
  `the_unnatural_direction.md`, build (a): the cost of direction lives in the
  prediction error (Still et al. 2012's non-predictive information), not in the raw
  recall stream — billing the raw stream is direction-blind (forward and reverse
  cost the same). With surprise connected, going WITH the learned current is
  predicted and cheap; going AGAINST it is unpredicted and dear.
  If `surprise` is NOT connected, behaviour is exactly the original (back-compatible).
------------------------------------------------------------------------------

Verified (arrow_cost_proof.py): gap=0 at detailed balance (g=0); reverse costs more
for g>0; the forward/reverse cost gap correlates with ||A|| at r≈0.99. The gap is
mostly a growing DISCOUNT on the natural direction, not a tax on the reverse.

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
E_SPIKE_JOULES = 1.2e-10
FPS = 60.0


class MetabolicSpikeNode(BaseNode):
    NODE_CATEGORY = "Metabolic"
    NODE_COLOR = QtGui.QColor(200, 70, 50)

    def __init__(self, dim=16, E_spike=1.0, c_field=1.0, thr=0.5):
        super().__init__()
        self.node_title = "Metabolic Spike (Wattage)"
        self.inputs = {
            'vector_in': 'spectrum',   # the content being held / changed (legacy mode)
            'surprise':  'spectrum',   # NEW: prediction-error residual; if present, bill THIS
            'skew_flux': 'signal',
            'theta':     'signal',
        }
        self.outputs = {
            'spikes':  'signal',
            'wattage': 'signal',
            'energy':  'signal',
            'silence': 'signal',
            'meter':   'image',
        }
        self.dim = int(dim)
        self.E_spike = float(E_spike); self.c_field = float(c_field)
        self.thr = float(thr)
        self.f_theta = 8.0; self.m_theta = 0.8
        self.leak_v = 0.90; self.beta = 3.0; self.tau_a = 0.18
        self.sigma_JN = 0.03
        self.ema = 0.2
        self.drive_gain = 7.0; self.dz_scale = 25.0
        self.dt = 1.0 / FPS

        self.ov = np.zeros(self.dim); self.prev = np.zeros(self.dim)
        self.v = np.zeros(self.dim); self.a = np.zeros(self.dim)
        self.t = 0
        self.spikes = 0.0; self.ds = 0.0
        self.P_spike = 0.0; self.P_field = 0.0; self.P_total = 0.0
        self.energy_rel = 0.0; self.spike_count = 0
        self.peak_P = 1e-9; self.silence = 1.0; self.last_flux = 0.0
        self.mode = "content"
        self.raster = deque(maxlen=180); self.Phist = deque(maxlen=180)
        self.display_img = np.zeros((160, 256, 3), dtype=np.uint8)

    def _fit(self, x):
        x = np.array(x, dtype=np.float32)
        if len(x) > self.dim: x = x[:self.dim]
        elif len(x) < self.dim: x = np.pad(x, (0, self.dim - len(x)))
        return x

    def step(self):
        sup = self.get_blended_input('surprise', 'first')
        vec = self.get_blended_input('vector_in', 'first')
        flux = self.get_blended_input('skew_flux', 'sum')
        theta_ext = self.get_blended_input('theta', 'sum')
        self.last_flux = float(flux) if flux is not None else 0.0

        # theta gate (internal unless driven)
        if theta_ext is not None:
            g = 1.0 + self.m_theta * float(theta_ext)
        else:
            g = 1.0 + self.m_theta * np.cos(2 * np.pi * self.f_theta * self.t * self.dt)

        if sup is not None:
            # ---- SURPRISE MODE: fire on the magnitude of the prediction-error residual ----
            self.mode = "surprise"
            err = self._fit(sup)
            dz = np.abs(err)
            self.ds = float(np.linalg.norm(err))
            drive = self.drive_gain * np.tanh(self.dz_scale * dz)
        elif vec is not None:
            # ---- LEGACY CONTENT MODE: fire on the CHANGE in held content ----
            self.mode = "content"
            v_in = self._fit(vec)
            self.ov = (1 - self.ema) * self.ov + self.ema * v_in
            dz = np.abs(self.ov - self.prev)
            self.ds = float(np.linalg.norm(self.ov - self.prev))
            self.prev = self.ov.copy()
            drive = self.drive_gain * np.abs(self.ov) * np.tanh(self.dz_scale * dz)
        else:
            self.spikes = 0.0; self.ds = 0.0
            self.P_spike *= 0.9; self.P_field *= 0.9
            self.P_total = self.P_spike + self.P_field
            self.raster.append(np.zeros(self.dim)); self.Phist.append(self.P_total)
            self._render(); return

        net = g * drive - self.beta * self.a + self.sigma_JN * np.random.randn(self.dim)
        self.v = self.leak_v * self.v + net
        sp = (self.v > self.thr).astype(np.float32)
        self.v = self.v * (1 - sp)
        self.a = (self.a + sp) * np.exp(-self.dt / self.tau_a)
        self.spikes = float(sp.sum())

        self.P_spike = self.E_spike * self.spikes
        self.P_field = self.c_field * self.ds
        self.P_total = self.P_spike + self.P_field

        self.energy_rel += self.P_total
        self.spike_count += int(self.spikes)
        self.peak_P = max(self.peak_P * 0.9995, self.P_total)
        self.silence = self.peak_P / max(self.P_total, 1e-9)

        self.raster.append(sp.copy()); self.Phist.append(self.P_total)
        self.t += 1
        self._render()

    def get_output(self, port_name):
        if port_name == 'spikes':  return self.spikes
        if port_name == 'wattage': return self.P_total
        if port_name == 'energy':  return self.energy_rel
        if port_name == 'silence': return self.silence
        if port_name == 'meter':   return self.display_img
        return None

    def _render(self):
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        R = np.array(self.raster) if len(self.raster) else np.zeros((1, self.dim))
        nfr = R.shape[0]
        for fi in range(nfr):
            x = int(fi / max(nfr, 1) * w)
            for ch in np.where(R[fi] > 0)[0]:
                y = 6 + int(ch / max(self.dim, 1) * 60)
                cv2.line(img, (x, y), (x, y + 2), (80, 200, 255), 1)
        label = "spikes (fire on SURPRISE)" if self.mode == "surprise" else "spikes (fire on change)"
        cv2.putText(img, label, (4, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 160), 1)

        base = 96
        pf = np.clip(self.P_field / max(self.peak_P, 1e-9), 0, 1)
        ps = np.clip(self.P_spike / max(self.peak_P, 1e-9), 0, 1)
        cv2.rectangle(img, (60, base), (60 + int(pf * 180), base + 12), (90, 230, 160), -1)
        cv2.rectangle(img, (60, base + 16), (60 + int(ps * 180), base + 28), (60, 120, 255), -1)
        flbl = "surprise" if self.mode == "surprise" else "field |ds|"
        cv2.putText(img, flbl, (4, base + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 230, 160), 1)
        cv2.putText(img, "spike", (4, base + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 120, 255), 1)

        regime = "SCAN  (against grain)" if self.P_total > 0.2 * self.peak_P else "HOLD  (with grain)"
        rcol = (60, 120, 255) if "SCAN" in regime else (120, 200, 120)
        cv2.putText(img, regime, (4, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rcol, 1)
        joules = self.spike_count * E_SPIKE_JOULES
        cv2.putText(img, f"silence {self.silence:6.0f}x   E~{joules:.2e} J",
                    (4, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        self.display_img = img

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 160, 256 * 3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Vector Dim", "dim", self.dim, None),
            ("E_spike (cost/spike)", "E_spike", self.E_spike, "float"),
            ("c_field (cost/|ds|)", "c_field", self.c_field, "float"),
            ("Threshold", "thr", self.thr, "float"),
        ]

    def set_config_options(self, options):
        if "dim" in options:
            self.dim = int(options["dim"])
            self.ov = np.zeros(self.dim); self.prev = np.zeros(self.dim)
            self.v = np.zeros(self.dim); self.a = np.zeros(self.dim)
        if "E_spike" in options: self.E_spike = float(options["E_spike"])
        if "c_field" in options: self.c_field = float(options["c_field"])
        if "thr" in options: self.thr = float(options["thr"])
