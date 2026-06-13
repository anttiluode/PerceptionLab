"""
Predictive Cortex Node (the live grain-of-the-arrow cortex)
===========================================================
The runnable, in-graph version of `arrow_cost_proof.py`. It extends the Mycelial
Cortex with one thing: a learned TRANSITION memory — the write-side skew operator
`A`, kept in node space — and uses it to PREDICT the next state. It then emits the
SURPRISE residual `x - pred` that the Metabolic Spike meter bills.

Why this exists (from `the_unnatural_direction.md`, build a):
  The cost of the arrow does NOT live in the raw recall stream — the change |ds|
  between consecutive patterns is the same forward or backward. It lives in the
  PREDICTION ERROR: the part of each input the learned current failed to anticipate
  (= Still et al. 2012's non-predictive information). Going WITH the carved current
  is predicted (small surprise, cheap); going AGAINST it is unpredicted (large
  surprise, dear). The forward direction earns a thermodynamic discount that grows
  with the asymmetry strength `g`; the reverse direction simply forfeits it.

How it predicts (critical detail — it must NOT peek at the current input):
  pred  = norm( P[prev_winner]            # symmetric "stay" (where we were)
              + g * Sum_j w_j P[j] )      # asymmetric "advance" (where the learned
                                          #   current expects to go), w = T[:,prev]/sum
  surprise_vec = recall - pred            # denoised percept vs prediction (predictive-coding
                                          #   residual; measured on the cleaned recall, not the
                                          #   raw noisy query, so corruption noise does not swamp
                                          #   the forward/reverse prediction difference)

Transition memory (the write-side A, node space):
  on each frame  T[winner, prev_winner] += 1   (decayed) — "winner followed prev".
  A_T = (T - T^T)/2 ;  ||A_T|| is the broken-detailed-balance flux of what it learned.
  `g` is how strongly that learned current shapes the prediction — the live ||A|| knob.
  At g = 0 the prediction ignores the current: detailed balance, no grain, no cost gap.

Verified standalone (arrow_cost_proof.py): gap=0 at g=0; reverse costs more for g>0;
gap tracks ||A|| at r≈0.99; the activity arrow flips sign on reversal. This node
reproduces that live and emits `surprise_vec` for the Metabolic Spike meter.

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


class PredictiveCortexNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(90, 150, 170)  # predictive teal

    def __init__(self, beta=18.0, leak=0.25, familiar=0.30, novel=0.20,
                 alpha=0.20, budget=48, g=1.0, trans_decay=0.99):
        super().__init__()
        self.node_title = "Predictive Cortex (emits surprise)"
        self.inputs = {
            'query_vec':    'spectrum',
            'inject_token': 'spectrum',
            'learn_enable': 'signal',
            'atp':          'signal',
        }
        self.outputs = {
            'recall_vec':   'spectrum',   # cleaned memory (symmetric recall)
            'recall_img':   'image',
            'pred_vec':     'spectrum',   # the learned current's guess for THIS input
            'surprise_vec': 'spectrum',   # query - pred  -> bill THIS with Metabolic Spike
            'surprise':     'signal',     # |surprise|, scalar (for a display / ATP gate)
            'n_nodes':      'signal',
            'confidence':   'signal',
            'arrow':        'signal',     # read-side skew circulation of winner traffic
            'Anorm':        'signal',     # write-side ||A_T|| of the learned transitions
            'demand':       'signal',
            'viz':          'image',
        }
        self.beta = float(beta); self.leak = float(leak)
        self.familiar = float(familiar); self.novel = float(novel)
        self.alpha = float(alpha); self.budget = int(budget)
        self.g = float(g); self.trans_decay = float(trans_decay)

        self.P = []
        self.dim = None
        self.T = np.zeros((self.budget, self.budget))   # transition memory (write-side A)
        self.recall_vec = None; self.pred_vec = None; self.surprise_vec = None
        self.surprise = 0.0; self.confidence = 0.0; self.arrow = 0.0
        self.Anorm = 0.0; self.demand = 0.0
        self.prev_winner = -1; self.last_winner = -1
        self.C = np.zeros((self.budget, self.budget))   # winner lag-cov (read-side arrow)
        self.prev_onehot = np.zeros(self.budget)
        self.display_img = np.zeros((160, 256, 3), dtype=np.uint8)
        self.recall_img = np.zeros((16, 16, 3), dtype=np.float32)

    # -------------------------------------------------------------
    def _ensure(self, v):
        if self.dim is None:
            self.dim = len(v)
            self.recall_vec = np.zeros(self.dim, np.float32)
            self.pred_vec = np.zeros(self.dim, np.float32)
            self.surprise_vec = np.zeros(self.dim, np.float32)

    def _recall(self, query, atp, steps=12):
        s = _norm(query.copy()); P = np.stack(self.P)
        beta = max(self.beta * (atp if atp is not None else 1.0), 1.0)
        for _ in range(steps):
            o = P @ s; w = _softmax(o, beta); s = _norm(self.leak * s + (w @ P))
        return s

    def _predict(self, P):
        """Prediction of the CURRENT input from the PREVIOUS winner only (no peeking)."""
        if self.prev_winner < 0 or self.prev_winner >= len(P):
            return np.zeros(self.dim, np.float32)
        stay = self.P[self.prev_winner]
        trow = self.T[:len(P), self.prev_winner]
        ssum = trow.sum()
        advance = ((trow / ssum) @ P) if ssum > 1e-9 else np.zeros(self.dim)
        return _norm(stay + self.g * advance).astype(np.float32)

    def _update_arrow(self, winner):
        oh = np.zeros(self.budget)
        if 0 <= winner < self.budget: oh[winner] = 1.0
        self.C = 0.97 * self.C + 0.03 * np.outer(oh, self.prev_onehot)
        self.prev_onehot = oh
        A = 0.5 * (self.C - self.C.T)
        n = min(len(self.P), self.budget)
        self.arrow = float(sum(A[k, (k+1) % n] - A[(k+1) % n, k] for k in range(n))) if n > 1 else 0.0

    # -------------------------------------------------------------
    def step(self):
        q = self.get_blended_input('query_vec', 'first')
        inj = self.get_blended_input('inject_token', 'first')
        learn = self.get_blended_input('learn_enable', 'sum')
        atp = self.get_blended_input('atp', 'sum')
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

        # --- PREDICTION (from previous frame only; never peeks at the current input) ---
        pred = self._predict(P)
        self.pred_vec = pred

        # --- RECALL (symmetric field denoises the noisy query) + confidence ---
        ov = P @ q; best = int(np.argmax(ov)); bestov = float(ov[best])
        s = self._recall(q, atp)
        self.recall_vec = s.astype(np.float32)
        self.confidence = float(_softmax(P @ s, self.beta).max())

        # --- SURPRISE = denoised percept vs prediction (predictive-coding residual) ---
        # measured on the cleaned recall, not the raw noisy query, so the corruption
        # noise floor does not swamp the forward/reverse prediction difference.
        self.surprise_vec = (self.recall_vec - pred).astype(np.float32) if np.any(pred) \
            else self.recall_vec.copy()
        self.surprise = float(np.linalg.norm(self.surprise_vec))
        # demand rises with both recall effort AND surprise (paying for the unnatural direction)
        self.demand = max(self.confidence if self.confidence > 0.5 else 0.05,
                          float(np.tanh(self.surprise)))

        # --- learning: templates (consolidate / spawn), ATP-gated ---
        atp_ok = (atp is None) or (float(atp) > 0.25)
        spawned = False
        if learn_on:
            if bestov >= self.familiar:
                self.P[best] = _norm((1 - self.alpha) * self.P[best] + self.alpha * q)
            elif bestov < self.novel and len(self.P) < self.budget and atp_ok:
                self.P.append(q.copy()); best = len(self.P) - 1; spawned = True

        # --- learning: the transition (write-side A): winner followed prev_winner ---
        # only genuine pattern->pattern steps; self-loops (winner repeats during a dwell)
        # would swamp the transition memory and erase the learned direction.
        if (learn_on and self.prev_winner >= 0 and best != self.prev_winner
                and best < self.budget and atp_ok):
            self.T *= self.trans_decay
            self.T[best, self.prev_winner] += 1.0
        n = len(self.P)
        if n > 1:
            Tn = self.T[:n, :n]; A = 0.5 * (Tn - Tn.T)
            self.Anorm = float(np.linalg.norm(A))

        # --- arrow (read side) + bookkeeping ---
        self.last_winner = best
        self._update_arrow(best)
        self.prev_winner = best

        # reconstruct recall + surprise as images
        side = int(np.sqrt(self.dim))
        if side * side == self.dim:
            g = self.recall_vec.reshape(side, side)
            g = (g - g.min()) / (np.ptp(g) + 1e-9)
            self.recall_img = np.stack([g, g, g], -1).astype(np.float32)

        self._render(spawned)

    # -------------------------------------------------------------
    def get_output(self, port_name):
        if port_name == 'recall_vec':   return self.recall_vec
        if port_name == 'recall_img':   return self.recall_img
        if port_name == 'pred_vec':     return self.pred_vec
        if port_name == 'surprise_vec': return self.surprise_vec
        if port_name == 'surprise':     return self.surprise
        if port_name == 'n_nodes':      return float(len(self.P))
        if port_name == 'confidence':   return self.confidence
        if port_name == 'arrow':        return self.arrow
        if port_name == 'Anorm':        return self.Anorm
        if port_name == 'demand':       return self.demand
        if port_name == 'viz':          return self.display_img
        return None

    def _render(self, spawned=False):
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        r = (np.clip(self.recall_img, 0, 1) * 255).astype(np.uint8)
        if r.shape[0] >= 2:
            r = cv2.resize(r, (84, 84), interpolation=cv2.INTER_NEAREST)
            img[20:104, 8:92] = r
        cv2.putText(img, "recall", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 170), 1)

        # surprise bar (the bill of the unnatural direction)
        sup = float(np.clip(self.surprise / 1.6, 0, 1))
        cv2.rectangle(img, (100, 30), (100 + int(sup * 150), 44), (60, 120, 255), -1)
        cv2.rectangle(img, (100, 30), (250, 44), (90, 90, 110), 1)
        cv2.putText(img, f"surprise {self.surprise:.2f}", (100, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 150, 255), 1)

        # population ring
        cx, cy, R = 150, 88, 26
        n = len(self.P)
        for k in range(n):
            a = 2 * np.pi * k / max(n, 1)
            x = int(cx + R * np.cos(a)); y = int(cy + R * np.sin(a))
            c = (80, 230, 255) if k == self.last_winner else (70, 90, 80)
            cv2.circle(img, (x, y), 3, c, -1)
        cv2.putText(img, f"nodes {n}", (8, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 200, 170), 1)
        if spawned:
            cv2.putText(img, "+SPAWN", (190, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 255, 160), 1)

        ardir = "->fwd" if self.arrow > 1e-4 else ("<-rev" if self.arrow < -1e-4 else " --")
        cv2.putText(img, f"arrow {self.arrow:+.3f} {ardir}   ||A|| {self.Anorm:.2f}  g={self.g:.1f}",
                    (8, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 210), 1)
        cv2.putText(img, "with current = predicted (cheap) | against = surprise (dear)",
                    (8, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 130, 140), 1)
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
            ("Asymmetry g (live ||A|| knob)", "g", self.g, "float"),
            ("Transition decay", "trans_decay", self.trans_decay, "float"),
            ("Node budget", "budget", self.budget, None),
        ]

    def set_config_options(self, options):
        for k in ("beta", "leak", "familiar", "novel", "alpha", "g", "trans_decay"):
            if k in options: setattr(self, k, float(options[k]))
        if "budget" in options:
            self.budget = int(options["budget"])
            self.T = np.zeros((self.budget, self.budget))
            self.C = np.zeros((self.budget, self.budget))
            self.prev_onehot = np.zeros(self.budget)
