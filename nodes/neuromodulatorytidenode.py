"""
Neuromodulatory Tide Node (The Infraslow Primer)
================================================
The slow primer the Mycelial Cortex was missing.

The ephaptic field is fast (ms, ~100 um) and carries DIRECTION. It cannot hold a
state for ten seconds. So the slow primer must be CHEMICAL — a neuromodulatory
tide (histamine is the exemplar; ACh/NA/5-HT do the same). It carries no pattern
and no direction. It is pure readiness: WHETHER the substrate is allowed to
resonate, not WHAT it recalls or WHICH WAY it turns.

Physically (established): histamine closes a resting K+ leak conductance via H1/H2
receptors (Haas, Sergeeva & Selbach 2008; McCormick & Williamson). That raises
membrane resistance Rm -> raises the length constant lambda -> raises the cable
attenuation alpha toward lossless (the orbit stays coherent), AND depolarizes the
cell toward threshold (excitability up). Downstream this becomes two engine dials:
alpha (cable coherence) and beta (recall sharpness). This node supplies the scalar
that turns them.

IMPORTANT — why this is NOT the ATP node:
  ATPMetabolismNode is REACTIVE: it burns on demand and forces rest (a relaxation
  oscillator driven by the work). The histaminergic signal Morishita et al. (Neuron
  2026) measured is SPONTANEOUS: it fluctuates on its own at infraslow 0.05-0.1 Hz,
  it PRECEDES the cue, and it PREDICTS the trial rather than being caused by it.
  So this is a free-running infraslow oscillator with slow correlated wander,
  optionally (weakly) perturbed by task events — not a demand-gated one.

Outputs:
  tide       0..1 readiness scalar (drive PrimedCortex beta / alpha gain)
  phase      0..1 phase of the slow oscillation
  high_state 1 when tide is above its recent high-percentile threshold
  cue_gate   1 on the rising edge into a high state (clean closed-loop trigger,
             the analogue of Morishita et al.'s closed-loop CS-US on high states)
  meter      the slow trace, threshold line, and HIGH/low banner

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

FPS = 60.0


class NeuromodulatoryTideNode(BaseNode):
    NODE_CATEGORY = "Metabolism"
    NODE_COLOR = QtGui.QColor(190, 130, 70)  # histamine amber

    def __init__(self, freq_hz=0.07, wander=0.25, high_pct=0.70, perturb_gain=0.05):
        super().__init__()
        self.node_title = "Neuromodulatory Tide (Infraslow Primer)"
        self.inputs = {
            'perturb': 'signal',   # OPTIONAL: task events transiently nudge the ongoing tide
        }
        self.outputs = {
            'tide':       'signal',
            'phase':      'signal',
            'high_state': 'signal',
            'cue_gate':   'signal',
            'meter':      'image',
        }
        self.freq_hz = float(freq_hz)          # 0.05-0.1 Hz infraslow band
        self.wander = float(wander)            # amplitude of slow correlated drift
        self.high_pct = float(high_pct)        # high-state threshold percentile
        self.perturb_gain = float(perturb_gain)
        self.dt = 1.0 / FPS

        self.phase = 0.0
        self.ou = 0.0                          # slow Ornstein-Uhlenbeck wander state
        self.tide = 0.5
        self.high_state = 0.0
        self.cue_gate = 0.0
        self.thr = 0.6
        self.was_high = False
        self.hist = deque(maxlen=900)          # ~15 s of memory at 60 fps
        self.rng = np.random.default_rng(0)
        self.display_img = np.zeros((120, 256, 3), dtype=np.uint8)

    def step(self):
        perturb = self.get_blended_input('perturb', 'sum') or 0.0

        # free-running infraslow phase (spontaneous, NOT demand-driven)
        self.phase = (self.phase + self.freq_hz * self.dt) % 1.0
        base = 0.5 + 0.5 * np.sin(2.0 * np.pi * self.phase)

        # slow correlated wander so the tide tracks an "integrated state", not a clean sine
        self.ou = 0.98 * self.ou + 0.02 * self.rng.standard_normal()
        drift = self.wander * np.tanh(self.ou)

        # task events only transiently perturb the ongoing tide (Morishita Fig S4)
        target = base + drift + self.perturb_gain * float(perturb)
        self.tide = float(np.clip(0.85 * self.tide + 0.15 * target, 0.0, 1.0))
        self.hist.append(self.tide)

        # high-state detection relative to the recent distribution (percentile threshold)
        H = np.array(self.hist)
        self.thr = float(np.quantile(H, self.high_pct)) if len(H) > 30 else 0.6
        is_high = self.tide >= self.thr
        self.cue_gate = 1.0 if (is_high and not self.was_high) else 0.0
        self.high_state = 1.0 if is_high else 0.0
        self.was_high = is_high

        self._render()

    def get_output(self, port_name):
        if port_name == 'tide':       return float(self.tide)
        if port_name == 'phase':      return float(self.phase)
        if port_name == 'high_state': return float(self.high_state)
        if port_name == 'cue_gate':   return float(self.cue_gate)
        if port_name == 'meter':      return self.display_img
        return None

    def _render(self):
        h, w = 120, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        H = np.array(self.hist) if len(self.hist) else np.zeros(2)

        # the slow tide trace
        for i in range(1, len(H)):
            x0 = int((i - 1) / len(H) * w); x1 = int(i / len(H) * w)
            y0 = int(86 - H[i - 1] * 64); y1 = int(86 - H[i] * 64)
            cv2.line(img, (x0, y0), (x1, y1), (70, 170, 230), 1)

        # high-state threshold line
        yt = int(86 - self.thr * 64)
        cv2.line(img, (0, yt), (w, yt), (80, 80, 110), 1)

        cv2.putText(img, "infraslow tide (readiness)", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 170, 230), 1)

        if self.high_state > 0.5:
            cv2.putText(img, "TIDE IN  ->  primed: alpha & beta up, can resonate",
                        (4, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 220, 140), 1)
        else:
            cv2.putText(img, "tide out  ->  leaky, sub-threshold, cannot resonate",
                        (4, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 140), 1)

        cv2.putText(img, f"tide {self.tide:.2f}", (190, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 210), 1)
        self.display_img = img

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 120, 256 * 3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Frequency (Hz, infraslow 0.05-0.1)", "freq_hz", self.freq_hz, "float"),
            ("Wander (slow drift amplitude)", "wander", self.wander, "float"),
            ("High-state percentile", "high_pct", self.high_pct, "float"),
            ("Perturb gain (task -> tide)", "perturb_gain", self.perturb_gain, "float"),
        ]

    def set_config_options(self, options):
        if "freq_hz" in options:      self.freq_hz = float(options["freq_hz"])
        if "wander" in options:       self.wander = float(options["wander"])
        if "high_pct" in options:     self.high_pct = float(options["high_pct"])
        if "perturb_gain" in options: self.perturb_gain = float(options["perturb_gain"])
