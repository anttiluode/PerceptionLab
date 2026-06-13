"""
Mycelial Cortex Node (Distributed Holographic Spiking Mesh)
===========================================================
A self-contained population of geometric neurons sharing one field. It is the
distributed-brain primitive: it recalls, it GROWS a node when something novel
arrives, it consolidates noisy exposures into clean memories, it reads the arrow
of time of its own spike traffic, and it speaks one token protocol so anything
that emits a vector token can join it.

Anatomy of each internal node (the framework, literal):
  - dendrite  = a Takens delay buffer of the node's own overlap with the field
  - AIS       = its grating  p_k : simultaneously its RECEIVE filter and its
                TRANSMIT pattern -- the holographic antenna (reciprocal)
  - axon/spike= a sparse token broadcast when the node wins the field competition

Two sides, kept separate (Model A / Model B):
  - INNER (analog): the shared field s converges by softmax competition
        s <- norm( leak*s + Sum_k softmax(beta * <p_k,s>) p_k )
    = modern Hopfield / attention = collective content-addressable recall.
  - OUTER (sparse): a node spikes/emits a token when it wins -> the digital,
    event-driven broadcast other peers (and the arrow-of-time readout) see.

Growth (verified ~once per distinct novel pattern):
  - familiar  (best overlap >= familiar)  -> CONSOLIDATE: nudge that node's
              template toward the input (denoise over exposures)
  - novel     (best overlap <  novel)     -> SPAWN a node tuned to the input
  ATP (if connected) gates new growth: an exhausted mesh cannot form memories.

Arrow of time: skew circulation of the recent winner traffic (the population's
own lag covariance). Forward A->B->C and its reverse give opposite sign.

Federation: an 'inject_token' input lets an external peer (another machine, an
LLM, a sensor) push a token into the same field. If it resonates it is recalled;
if novel it spawns a node. The mesh lives with whatever connects.

VERIFIED standalone (mycelial_mesh_proof.py): recall cos 0.21->0.78 @ 79%
recognition; spawn-once + consolidation -> templates cos ~0.9 to truth; arrow
sign flips on reversal; foreign peer integrates at cos 1.0.

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


class MycelialCortexNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(70, 160, 120)  # mycelial green

    def __init__(self, beta=18.0, leak=0.25, familiar=0.30, novel=0.20,
                 alpha=0.20, budget=48):
        super().__init__()
        self.node_title = "Mycelial Cortex (Distributed Mesh)"
        self.inputs = {
            'query_vec':    'spectrum',   # what to recall / perceive
            'inject_token': 'spectrum',   # a peer's broadcast (federation)
            'learn_enable': 'signal',     # 1 = allow growth/consolidation (default on)
            'atp':          'signal',     # optional metabolic gate
        }
        self.outputs = {
            'recall_vec':  'spectrum',    # the cleaned-up memory
            'recall_img':  'image',
            'n_nodes':     'signal',      # population size (grows)
            'confidence':  'signal',
            'arrow':       'signal',      # skew circulation of winner traffic (time's arrow)
            'demand':      'signal',      # for the ATP pool
            'viz':         'image',
        }
        self.beta = float(beta); self.leak = float(leak)
        self.familiar = float(familiar); self.novel = float(novel)
        self.alpha = float(alpha); self.budget = int(budget)

        self.P = []                       # node templates (gratings)
        self.dim = None
        self.recall_vec = None
        self.confidence = 0.0; self.arrow = 0.0; self.demand = 0.0
        self.last_winner = -1
        self.win_hist = deque(maxlen=240)         # recent winner indices
        self.C = np.zeros((self.budget, self.budget))   # winner lag-covariance (fixed size)
        self.prev_onehot = np.zeros(self.budget)
        self.display_img = np.zeros((160, 256, 3), dtype=np.uint8)
        self.recall_img = np.zeros((16, 16, 3), dtype=np.float32)

    # -------------------------------------------------------------
    def _ensure(self, v):
        if self.dim is None:
            self.dim = len(v)
            self.recall_vec = np.zeros(self.dim, np.float32)

    def _recall(self, query, atp, steps=12):
        s = _norm(query.copy()); P = np.stack(self.P)
        beta = self.beta * (atp if atp is not None else 1.0)
        beta = max(beta, 1.0)
        win_steps = np.zeros(len(P))
        for t in range(steps):
            o = P @ s; w = _softmax(o, beta)
            s = _norm(self.leak * s + (w @ P))
            win_steps[np.argmax(w)] += 1
        return s, win_steps

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

        # a peer's token counts as an input too (federation)
        src = q if q is not None else inj
        if src is None:
            self._render(); return
        v = np.array(src, dtype=np.float32)
        self._ensure(v)
        if len(v) != self.dim:
            v = (v[:self.dim] if len(v) > self.dim else np.pad(v, (0, self.dim - len(v))))
        q = _norm(v)

        # seed the very first node from the first thing seen
        if len(self.P) == 0:
            self.P.append(q.copy())

        P = np.stack(self.P)
        ov = P @ q; best = int(np.argmax(ov)); bestov = float(ov[best])

        # INNER side: converge the shared field -> recall
        s, win_steps = self._recall(q, atp)
        self.recall_vec = s.astype(np.float32)
        self.confidence = float(_softmax(P @ s, self.beta).max())
        self.demand = self.confidence if self.confidence > 0.5 else 0.05

        # learning: consolidate (familiar) or spawn (novel) -- ATP gates growth
        spawned = False
        atp_ok = (atp is None) or (float(atp) > 0.25)
        if learn_on:
            if bestov >= self.familiar:
                self.P[best] = _norm((1 - self.alpha) * self.P[best] + self.alpha * q)
            elif bestov < self.novel and len(self.P) < self.budget and atp_ok:
                self.P.append(q.copy()); best = len(self.P) - 1; spawned = True

        # OUTER side: winner = the spike/token broadcast; arrow of time from traffic
        self.last_winner = best
        self.win_hist.append(best)
        self._update_arrow(best)

        # reconstruct recall as an image
        side = int(np.sqrt(self.dim))
        if side * side == self.dim:
            g = self.recall_vec.reshape(side, side)
            g = (g - g.min()) / (np.ptp(g) + 1e-9)
            self.recall_img = np.stack([g, g, g], -1).astype(np.float32)

        self._render(spawned)

    # -------------------------------------------------------------
    def get_output(self, port_name):
        if port_name == 'recall_vec': return self.recall_vec
        if port_name == 'recall_img': return self.recall_img
        if port_name == 'n_nodes':    return float(len(self.P))
        if port_name == 'confidence': return self.confidence
        if port_name == 'arrow':      return self.arrow
        if port_name == 'demand':     return self.demand
        if port_name == 'viz':        return self.display_img
        return None

    def _render(self, spawned=False):
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # recalled memory thumbnail
        r = (np.clip(self.recall_img, 0, 1) * 255).astype(np.uint8)
        if r.shape[0] >= 2:
            r = cv2.resize(r, (96, 96), interpolation=cv2.INTER_NEAREST)
            img[20:116, 8:104] = r
        cv2.putText(img, "recall", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 170), 1)

        # population ring: a dot per node, last winner lit
        cx, cy, R = 180, 60, 42
        n = len(self.P)
        for k in range(n):
            a = 2 * np.pi * k / max(n, 1)
            x = int(cx + R * np.cos(a)); y = int(cy + R * np.sin(a))
            c = (80, 230, 255) if k == self.last_winner else (70, 90, 80)
            cv2.circle(img, (x, y), 4, c, -1)
        cv2.putText(img, f"nodes {n}", (140, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 170), 1)
        if spawned:
            cv2.putText(img, "+SPAWN", (140, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 255, 160), 1)

        # readouts
        ardir = "->fwd" if self.arrow > 1e-4 else ("<-rev" if self.arrow < -1e-4 else " --")
        cv2.putText(img, f"conf {self.confidence:.2f}   arrow {self.arrow:+.3f} {ardir}",
                    (8, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 210), 1)
        cv2.putText(img, "inner=field recall   outer=spike/token",
                    (8, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 130, 140), 1)
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
            ("Node budget", "budget", self.budget, None),
        ]

    def set_config_options(self, options):
        for k in ("beta", "leak", "familiar", "novel", "alpha"):
            if k in options: setattr(self, k, float(options[k]))
        if "budget" in options:
            self.budget = int(options["budget"])
            self.C = np.zeros((self.budget, self.budget)); self.prev_onehot = np.zeros(self.budget)
