"""
Primed Cortex Node (Mycelial Cortex + the slow tide on the dial)
================================================================
The Mycelial Cortex, with the Neuromodulatory Tide gating WHETHER it can resonate.
The tide is a slow scalar readiness (the chemical primer). It carries no pattern
and no direction; it only sets two dials:

  beta_eff = beta * gain(tide)         the recall sharpness (excitability up: a
                                       fainter resonant match now wins). When the
                                       tide is out, beta collapses and recall blurs.

  express  = smoothstep(tide)          the EXPRESSION GATE. The recalled memory is
                                       ALWAYS computed (it is never lost), but its
                                       confidence and image are only EXPRESSED when
                                       the tide is in. Tide out => stored but
                                       inaccessible (Ryan & Frankland 2022). Tide in
                                       => the same query recalls cleanly.

This reproduces, as a behavioural analogue, Morishita et al. (Neuron 2026): the
SAME cue is recalled well in a high pre-cue state and fails in a low one — recall
gated by a slow, spontaneous, pre-arrival state, the input difficulty unchanged.

As a free consequence the arrow-of-time readout only sharpens on the high tide:
at low beta the winner traffic is too noisy to carry a stable skew circulation, so
readiness gates not just WHETHER recall happens but whether DIRECTION is legible.

NOT modelled here (honest): the alpha / cable-losslessness effect. This node has
no explicit Takens buffer (alpha^k) to modulate — its `leak` is field-persistence
in the recall iteration, not cable attenuation. Faking alpha onto `leak` would be a
lie. alpha stays a labelled prediction for the explicit-Takens node (see the paper).

Back-compatible: with no `tide` connected the tide defaults to 1.0 and this behaves
exactly like the Mycelial Cortex.

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


def _norm(v): return v / (np.linalg.norm(v) + 1e-9)
def _softmax(x, b):
    x = b * (x - x.max()); e = np.exp(x); return e / (e.sum() + 1e-12)


class PrimedCortexNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(110, 170, 130)  # primed mycelial green

    def __init__(self, beta=18.0, leak=0.25, familiar=0.30, novel=0.20,
                 alpha=0.20, budget=48, gain_floor=0.06, gate_lo=0.25, gate_hi=0.60):
        super().__init__()
        self.node_title = "Primed Cortex (tide-gated mesh)"
        self.inputs = {
            'query_vec':    'spectrum',
            'tide':         'signal',     # the slow readiness scalar (0..1)
            'inject_token': 'spectrum',
            'learn_enable': 'signal',
            'atp':          'signal',
        }
        self.outputs = {
            'recall_vec':    'spectrum',  # the cleaned memory (ALWAYS computed - never lost)
            'recall_img':    'image',     # raw recall
            'expressed_img': 'image',     # recall dimmed by the tide (what is ACCESSIBLE)
            'confidence':    'signal',    # raw recall confidence
            'expressed_conf':'signal',    # confidence * expression gate  <-- the headline readout
            'express':       'signal',    # the expression gate itself, 0..1
            'n_nodes':       'signal',
            'arrow':         'signal',    # sharpens only on the high tide
            'demand':        'signal',
            'viz':           'image',
        }
        self.beta = float(beta); self.leak = float(leak)
        self.familiar = float(familiar); self.novel = float(novel)
        self.alpha = float(alpha); self.budget = int(budget)
        self.gain_floor = float(gain_floor)
        self.gate_lo = float(gate_lo); self.gate_hi = float(gate_hi)

        self.P = []
        self.dim = None
        self.recall_vec = None
        self.confidence = 0.0; self.expressed_conf = 0.0; self.express = 1.0
        self.arrow = 0.0; self.demand = 0.0; self.tide = 1.0
        self.last_winner = -1
        self.C = np.zeros((self.budget, self.budget))
        self.prev_onehot = np.zeros(self.budget)
        self.display_img = np.zeros((160, 256, 3), dtype=np.uint8)
        self.recall_img = np.zeros((16, 16, 3), dtype=np.float32)
        self.expressed_img = np.zeros((16, 16, 3), dtype=np.float32)

    # -------------------------------------------------------------
    def _ensure(self, v):
        if self.dim is None:
            self.dim = len(v)
            self.recall_vec = np.zeros(self.dim, np.float32)

    def _smooth(self, x, lo, hi):
        t = float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def _recall(self, query, beta_eff, steps=12):
        s = _norm(query.copy()); P = np.stack(self.P)
        for _ in range(steps):
            o = P @ s; w = _softmax(o, beta_eff)
            s = _norm(self.leak * s + (w @ P))
        return s

    def _update_arrow(self, winner):
        oh = np.zeros(self.budget)
        if 0 <= winner < self.budget: oh[winner] = 1.0
        self.C = 0.97 * self.C + 0.03 * np.outer(oh, self.prev_onehot)
        self.prev_onehot = oh
        A = 0.5 * (self.C - self.C.T)
        n = min(len(self.P), self.budget)
        self.arrow = float(sum(A[k, (k + 1) % n] - A[(k + 1) % n, k] for k in range(n))) if n > 1 else 0.0

    # -------------------------------------------------------------
    def step(self):
        q = self.get_blended_input('query_vec', 'first')
        inj = self.get_blended_input('inject_token', 'first')
        learn = self.get_blended_input('learn_enable', 'sum')
        atp = self.get_blended_input('atp', 'sum')
        tide = self.get_blended_input('tide', 'sum')
        self.tide = 1.0 if tide is None else float(np.clip(tide, 0.0, 1.0))
        learn_on = True if learn is None else (float(learn) > 0.5)

        src = q if q is not None else inj
        if src is None:
            self._render(); return
        v = np.array(src, dtype=np.float32)
        self._ensure(v)
        if len(v) != self.dim:
            v = (v[:self.dim] if len(v) > self.dim else np.pad(v, (0, self.dim - len(v))))
        q = _norm(v)

        if len(self.P) == 0:
            self.P.append(q.copy())
        P = np.stack(self.P)
        ov = P @ q; best = int(np.argmax(ov)); bestov = float(ov[best])

        # --- the tide sets the dials: readiness raises beta (excitability up) ---
        beta_eff = max(1.0, self.beta * (self.gain_floor + (1.0 - self.gain_floor) * self.tide))

        # INNER side: converge the shared field -> recall (memory is ALWAYS computed)
        s = self._recall(q, beta_eff)
        self.recall_vec = s.astype(np.float32)
        self.confidence = float(_softmax(P @ s, beta_eff).max())

        # --- the EXPRESSION GATE: stored vs accessible ---
        self.express = self._smooth(self.tide, self.gate_lo, self.gate_hi)
        self.expressed_conf = self.confidence * self.express
        self.demand = self.expressed_conf if self.expressed_conf > 0.5 else 0.05

        # learning (consolidate / spawn), ATP-gated as before
        spawned = False
        atp_ok = (atp is None) or (float(atp) > 0.25)
        if learn_on:
            if bestov >= self.familiar:
                self.P[best] = _norm((1 - self.alpha) * self.P[best] + self.alpha * q)
            elif bestov < self.novel and len(self.P) < self.budget and atp_ok:
                self.P.append(q.copy()); best = len(self.P) - 1; spawned = True

        # OUTER side: winner traffic -> arrow (only legible when beta is high)
        self.last_winner = best
        self._update_arrow(best)

        # reconstruct recall as an image, and the expressed (gated) version
        side = int(np.sqrt(self.dim))
        if side * side == self.dim:
            g = self.recall_vec.reshape(side, side)
            g = (g - g.min()) / (np.ptp(g) + 1e-9)
            self.recall_img = np.stack([g, g, g], -1).astype(np.float32)
            self.expressed_img = (self.recall_img * self.express).astype(np.float32)

        self._render(spawned)

    # -------------------------------------------------------------
    def get_output(self, port_name):
        if port_name == 'recall_vec':     return self.recall_vec
        if port_name == 'recall_img':     return self.recall_img
        if port_name == 'expressed_img':  return self.expressed_img
        if port_name == 'confidence':     return self.confidence
        if port_name == 'expressed_conf': return self.expressed_conf
        if port_name == 'express':        return self.express
        if port_name == 'n_nodes':        return float(len(self.P))
        if port_name == 'arrow':          return self.arrow
        if port_name == 'demand':         return self.demand
        if port_name == 'viz':            return self.display_img
        return None

    def _render(self, spawned=False):
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # expressed recall thumbnail (what is actually accessible right now)
        r = (np.clip(self.expressed_img, 0, 1) * 255).astype(np.uint8)
        if r.shape[0] >= 2:
            r = cv2.resize(r, (84, 84), interpolation=cv2.INTER_NEAREST)
            img[22:106, 8:92] = r
        cv2.putText(img, "accessible recall", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 200, 170), 1)

        # tide / expression bar
        cv2.putText(img, f"tide {self.tide:.2f}", (104, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 170, 230), 1)
        ex = float(np.clip(self.express, 0, 1))
        cv2.rectangle(img, (104, 30), (104 + int(ex * 140), 42), (90, 220, 140), -1)
        cv2.rectangle(img, (104, 30), (244, 42), (80, 90, 90), 1)
        cv2.putText(img, f"expressed conf {self.expressed_conf:.2f}", (104, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 210), 1)

        # population ring
        cx, cy, R = 150, 96, 26
        n = len(self.P)
        for k in range(n):
            a = 2 * np.pi * k / max(n, 1)
            x = int(cx + R * np.cos(a)); y = int(cy + R * np.sin(a))
            c = (80, 230, 255) if k == self.last_winner else (70, 90, 80)
            cv2.circle(img, (x, y), 3, c, -1)
        cv2.putText(img, f"nodes {n}", (8, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 200, 170), 1)
        if spawned:
            cv2.putText(img, "+SPAWN", (190, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 255, 160), 1)

        state = "PRIMED" if self.express > 0.5 else "inaccessible (stored, not lost)"
        scol = (90, 220, 140) if self.express > 0.5 else (120, 120, 140)
        cv2.putText(img, f"{state}   arrow {self.arrow:+.3f}", (8, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, scol, 1)
        cv2.putText(img, "slow tide = whether | fast field = which way",
                    (8, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (120, 130, 140), 1)
        self.display_img = img

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 160, 256 * 3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Beta (recall sharpness)", "beta", self.beta, "float"),
            ("Field leak", "leak", self.leak, "float"),
            ("Familiar thr", "familiar", self.familiar, "float"),
            ("Novel thr", "novel", self.novel, "float"),
            ("Consolidation alpha", "alpha", self.alpha, "float"),
            ("Gain floor (beta at tide=0)", "gain_floor", self.gain_floor, "float"),
            ("Express gate lo", "gate_lo", self.gate_lo, "float"),
            ("Express gate hi", "gate_hi", self.gate_hi, "float"),
            ("Node budget", "budget", self.budget, None),
        ]

    def set_config_options(self, options):
        for k in ("beta", "leak", "familiar", "novel", "alpha", "gain_floor", "gate_lo", "gate_hi"):
            if k in options: setattr(self, k, float(options[k]))
        if "budget" in options:
            self.budget = int(options["budget"])
            self.C = np.zeros((self.budget, self.budget)); self.prev_onehot = np.zeros(self.budget)
