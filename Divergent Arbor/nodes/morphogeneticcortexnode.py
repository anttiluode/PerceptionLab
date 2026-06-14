"""
Morphogenetic Cortex Node (nodes that grow a shape from what they hear)
=======================================================================
A Mycelial Cortex whose nodes carry a grown MORPHOLOGY as well as content. Recall
runs on the content (robust); the morphology is a receptive FOOTPRINT m_k -- which
input dimensions the node has claimed -- grown by EMA on the dims active when it
wins, with LATERAL COMPETITION so nodes carve the input space into territories. The
footprint is the engine stand-in for a dendritic arbor's receptive coverage: shape
is the filter (cable theory), grown by activity (Grubb & Burrone 2010; Kuba 2010).

What the headless proof (shape_divergence_proof.py) established, and what this node
lets you watch live:
  - the SAME content stream carves a DIFFERENT organization per mesh -- different
    node count, different partition (which patterns fragment into redundant nodes),
    different footprint sizes -- while recall converges (knowledge federates, form
    does not);
  - but each unit's footprint tracks the STIMULUS it specialized on, not the mesh's
    history (per-node shape cos 0.98 between peers). The divergence lives in the
    CARVING, not the cell. This is degeneracy (Marder; Edelman & Gally), localized.

To SEE divergence in one graph: drop two of these nodes (different `seed`) reading
the same Engram stream. A small per-mesh `spawn_jitter` makes the novelty decision
stochastic, so the two meshes carve the same stream into different organizations
from internal noise alone -- watch their node counts and footprint galleries drift
apart while both still recall the shapes. (The federation setup -- different prior +
reordered stream over the token relay -- is the other way to get it, with the
verified numbers in the proof.)

Outputs:
  recall_vec / recall_img : the recalled content
  n_nodes                 : population size (diverges)
  mean_size               : mean footprint size in dims (diverges)
  confidence, demand
  viz                     : recall + a GALLERY of the grown node footprints (the shapes)

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


def _norm(v): return v / (np.linalg.norm(v) + 1e-9)
def _softmax(x, b):
    x = b * (x - x.max()); e = np.exp(x); return e / (e.sum() + 1e-12)


class MorphogeneticCortexNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(120, 150, 90)  # arbor green

    def __init__(self, beta=22.0, leak=0.22, familiar=0.62, novel=0.42,
                 alpha=0.22, morph=0.2, inhib=0.05, spawn_jitter=0.04,
                 budget=40, seed=0):
        super().__init__()
        self.node_title = "Morphogenetic Cortex (grows shapes)"
        self.inputs = {
            'query_vec':    'spectrum',
            'inject_token': 'spectrum',
            'learn_enable': 'signal',
            'atp':          'signal',
        }
        self.outputs = {
            'recall_vec': 'spectrum',
            'recall_img': 'image',
            'n_nodes':    'signal',
            'mean_size':  'signal',
            'confidence': 'signal',
            'demand':     'signal',
            'viz':        'image',
        }
        self.beta = float(beta); self.leak = float(leak)
        self.familiar = float(familiar); self.novel = float(novel)
        self.alpha = float(alpha); self.morph = float(morph); self.inhib = float(inhib)
        self.spawn_jitter = float(spawn_jitter); self.budget = int(budget)
        self.seed = int(seed)

        self.P = []; self.M = []
        self.dim = None
        self.rng = np.random.default_rng(self.seed)
        self.recall_vec = None
        self.confidence = 0.0; self.demand = 0.0; self.mean_size = 0.0
        self.last_winner = -1
        self.display_img = np.zeros((160, 256, 3), dtype=np.uint8)
        self.recall_img = np.zeros((16, 16, 3), dtype=np.float32)

    def _ensure(self, v):
        if self.dim is None:
            self.dim = len(v)
            self.recall_vec = np.zeros(self.dim, np.float32)

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
            a = np.abs(q); self.P.append(q.copy()); self.M.append(a / (a.max() + 1e-9))

        P = np.stack(self.P)
        ov = P @ q; best = int(np.argmax(ov)); bestov = float(ov[best])

        # recall on CONTENT (robust, footprint-independent)
        s = q.copy()
        for _ in range(14):
            o = P @ s; w = _softmax(o, self.beta); s = _norm(self.leak * s + (w @ P))
        self.recall_vec = s.astype(np.float32)
        self.confidence = float(_softmax(P @ s, self.beta).max())
        self.demand = self.confidence if self.confidence > 0.5 else 0.05

        # morphogenesis: grow / prune footprints, with a stochastic spawn decision
        atp_ok = (atp is None) or (float(atp) > 0.25)
        a = np.abs(q); aind = a / (a.max() + 1e-9)
        jitter = self.spawn_jitter * float(self.rng.standard_normal())
        if learn_on:
            if bestov >= self.familiar:
                self.P[best] = _norm((1 - self.alpha) * self.P[best] + self.alpha * q)
                self.M[best] = np.clip(self.M[best] + self.morph * (aind - self.M[best]), 0, 1)
                for j in range(len(self.M)):            # lateral competition for territory
                    if j != best:
                        self.M[j] = np.clip(self.M[j] - self.inhib * aind * self.M[j], 0, 1)
            elif bestov < (self.novel + jitter) and len(self.P) < self.budget and atp_ok:
                self.P.append(q.copy()); self.M.append(aind.copy())
                best = len(self.P) - 1

        self.last_winner = best
        self.mean_size = float(np.mean([m.sum() for m in self.M])) if self.M else 0.0

        side = int(np.sqrt(self.dim))
        if side * side == self.dim:
            g = self.recall_vec.reshape(side, side)
            g = (g - g.min()) / (np.ptp(g) + 1e-9)
            self.recall_img = np.stack([g, g, g], -1).astype(np.float32)

        self._render()

    def get_output(self, port_name):
        if port_name == 'recall_vec': return self.recall_vec
        if port_name == 'recall_img': return self.recall_img
        if port_name == 'n_nodes':    return float(len(self.P))
        if port_name == 'mean_size':  return self.mean_size
        if port_name == 'confidence': return self.confidence
        if port_name == 'demand':     return self.demand
        if port_name == 'viz':        return self.display_img
        return None

    def _render(self):
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # recalled content thumbnail (top-left)
        r = (np.clip(self.recall_img, 0, 1) * 255).astype(np.uint8)
        if r.shape[0] >= 2:
            r = cv2.resize(r, (52, 52), interpolation=cv2.INTER_NEAREST)
            img[22:74, 8:60] = r
        cv2.putText(img, "recall", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 200, 130), 1)

        # gallery of grown footprint SHAPES (each node's m reshaped)
        cv2.putText(img, "grown footprints (the shapes)", (70, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 200, 130), 1)
        side = int(np.sqrt(self.dim)) if self.dim else 0
        cols, thumb, pad = 6, 26, 3
        x0, y0 = 70, 22
        nshow = min(len(self.M), 12)
        for i in range(nshow):
            m = self.M[i]
            if side * side == self.dim:
                mm = m.reshape(side, side)
            else:
                k = int(np.ceil(np.sqrt(len(m)))); mm = np.pad(m, (0, k*k-len(m))).reshape(k, k)
            mm = (np.clip(mm, 0, 1) * 255).astype(np.uint8)
            tile = cv2.applyColorMap(cv2.resize(mm, (thumb, thumb),
                                                interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_SUMMER)
            cx = x0 + (i % cols) * (thumb + pad)
            cy = y0 + (i // cols) * (thumb + pad)
            if cy + thumb <= 92 and cx + thumb <= w:
                img[cy:cy+thumb, cx:cx+thumb] = tile
                if i == self.last_winner:
                    cv2.rectangle(img, (cx, cy), (cx+thumb-1, cy+thumb-1), (90, 255, 200), 1)

        # readouts
        cv2.putText(img, f"nodes {len(self.P)}   mean size {self.mean_size:.0f} dims",
                    (8, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 210, 180), 1)
        cv2.putText(img, f"conf {self.confidence:.2f}   seed {self.seed}",
                    (8, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 190, 170), 1)
        cv2.putText(img, "same stream -> own carving (count, partition, size)",
                    (8, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (120, 130, 110), 1)
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
            ("Footprint growth", "morph", self.morph, "float"),
            ("Lateral competition", "inhib", self.inhib, "float"),
            ("Spawn jitter (divergence noise)", "spawn_jitter", self.spawn_jitter, "float"),
            ("Seed (per-mesh identity)", "seed", self.seed, None),
            ("Node budget", "budget", self.budget, None),
        ]

    def set_config_options(self, options):
        for k in ("beta", "leak", "familiar", "novel", "alpha", "morph", "inhib", "spawn_jitter"):
            if k in options: setattr(self, k, float(options[k]))
        if "budget" in options: self.budget = int(options["budget"])
        if "seed" in options:
            self.seed = int(options["seed"]); self.rng = np.random.default_rng(self.seed)
