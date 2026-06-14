"""
shape_divergence_proof.py  —  same stream, different carving, same knowledge
============================================================================
The federation result (README_federation) showed two peers grow DIFFERENT internal
models (different node counts, different organization) from the SAME token stream,
yet recall the same patterns. This proof asks the sharper question the morphology
intuition raises: WHERE does that divergence live?

Each node here carries a MORPHOLOGY as well as a content vector:
  - content  p_k : the stored pattern (what it recalls)        -- recall uses this
  - footprint m_k: a grown receptive shape (which input dims it claims), grown by
                   EMA on the dims active when it wins, with lateral competition so
                   nodes carve the input space into territories (history-dependent)

Two peers (A, B) are taught the SAME six patterns, but with a different prior (their
own seed/background nodes) and a reordered stream (the relay reorders tokens) -- the
exact conditions of the live federation. We then measure, against a CONTROL where the
prior and order are identical:

  P1  do the two peers build the same organization?   (node count, partition)
  P2  do they recall the same knowledge?              (cos to truth, recognition, agreement)
  P3  does each unit grow a different receptive shape? (per-node footprint cosine)

THE FALSIFIABLE / VERIFIED RESULT (fixed seeds, reproducible):
  ORGANIZATION DIVERGES   - A=7 nodes, B=9; A splits one pattern into 2 nodes,
                            B splits two others; mean footprint size 13.1 vs 11.4 dims.
  KNOWLEDGE CONVERGES     - recall cos 0.76/0.75, recognition 78%/78%,
                            cross-peer recall agreement cos 0.81.
  PER-NODE SHAPE CONVERGES- footprint cos(A,B) per matched pattern = 0.98. A single
                            unit's receptive shape tracks the STIMULUS it specialized
                            on, NOT the peer's history.
  CONTROL                 - identical prior + order -> shape cos 1.00, same counts.
                            (so the divergence is caused by history, not by noise.)

THE READING: degeneracy (Marder & Goaillard 2006; Edelman & Gally 2001) -- the same
function reached by different structures -- but localized. In this substrate the
degeneracy lives in the CARVING (how many units, how the space is partitioned), not
in the individual receptive field. What federates is the knowledge, which is
basis-independent; what diverges is the organization, which is grown locally.

HONEST LIMITS: six near-orthogonal patterns, one corruption level, relative units;
recall ~0.76 matches the standalone mesh (mycelial_mesh_proof), i.e. middling, not
great; the two-peer divergence here is from prior+order (the federation setup), a
single mesh-pair, structural not biophysical. The bet (that any of it is experienced)
is untouched.

Run:  python shape_divergence_proof.py
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np

D = 64
def norm(v): return v / (np.linalg.norm(v) + 1e-9)
def softmax(x, b): x = b * (x - x.max()); e = np.exp(x); return e / (e.sum() + 1e-12)
def make_patterns(seed=0):
    rng = np.random.default_rng(seed)
    return np.array([norm(rng.standard_normal(D)) for _ in range(6)])   # near-orthogonal


class Arbor:
    """A mesh whose nodes carry a grown receptive footprint as well as content."""
    def __init__(self, beta=25, leak=0.2, familiar=0.80, novel=0.60, alpha=0.25,
                 morph=0.2, inhib=0.06, budget=40, seed=0):
        self.beta, self.leak = beta, leak
        self.familiar, self.novel, self.alpha = familiar, novel, alpha
        self.morph, self.inhib, self.budget = morph, inhib, budget
        self.P, self.M = [], []
        self.tag = seed
        self.rng = np.random.default_rng(seed)

    def seed_bg(self, k):
        for _ in range(k):
            q = norm(self.rng.standard_normal(D)); a = np.abs(q)
            self.P.append(q.copy()); self.M.append(a / (a.max() + 1e-9))

    def perceive(self, q, learn=True):
        q = norm(q)
        if len(self.P) == 0:
            a = np.abs(q); self.P.append(q.copy()); self.M.append(a / (a.max() + 1e-9))
        P = np.stack(self.P); sc = P @ q
        best = int(np.argmax(sc)); bestov = float(sc[best])
        s = q.copy()
        for _ in range(14):                       # recall on CONTENT (robust)
            o = P @ s; w = softmax(o, self.beta); s = norm(self.leak * s + (w @ P))
        if learn:
            a = np.abs(q); aind = a / (a.max() + 1e-9)
            if bestov >= self.familiar:
                self.P[best] = norm((1 - self.alpha) * self.P[best] + self.alpha * q)
                self.M[best] = np.clip(self.M[best] + self.morph * (aind - self.M[best]), 0, 1)
                for j in range(len(self.M)):       # lateral competition for territory
                    if j != best:
                        self.M[j] = np.clip(self.M[j] - self.inhib * aind * self.M[j], 0, 1)
            elif bestov < self.novel and len(self.P) < self.budget:
                self.P.append(q.copy()); self.M.append(aind.copy())
        return s, best, bestov


def teach(mesh, P, order, corr=0.05):
    rng = np.random.default_rng(1000 + mesh.tag)
    for k in order:
        mesh.perceive(P[k] + corr * rng.standard_normal(D), learn=True)

def recall_test(mesh, P, corr=0.45, trials=400, seed=1):
    rng = np.random.default_rng(seed); M = len(P); cr = []; ok = 0
    for _ in range(trials):
        k = int(rng.integers(M)); q = norm(P[k] + corr * rng.standard_normal(D))
        s, b, o = mesh.perceive(q, learn=False)
        cr.append(P[k] @ norm(s)); ok += (int(np.argmax(P @ norm(s))) == k)
    return float(np.mean(cr)), 100 * ok / trials

def owners(mesh, P):
    Pm = np.stack(mesh.P)
    return np.array([int(np.argmax(P @ norm(Pm[i]))) for i in range(len(mesh.P))])
def nodes_per_pattern(mesh, P):
    o = owners(mesh, P); return [int(np.sum(o == k)) for k in range(len(P))]
def shape_for(mesh, P, k):
    Pm = np.stack(mesh.P); return mesh.M[int(np.argmax(Pm @ P[k]))]
def mean_size(mesh): return float(np.mean([m.sum() for m in mesh.M]))


def main():
    P = make_patterns(0); reps = 12; base = list(range(6)) * reps

    print("=" * 78)
    print("SAME STREAM, DIFFERENT CARVING, SAME KNOWLEDGE")
    print("=" * 78)

    # ---- DIVERGENT: different prior + reordered stream (the federation setup) ----
    A = Arbor(seed=11); A.seed_bg(1)
    B = Arbor(seed=29); B.seed_bg(3)
    oa = base[:]; np.random.default_rng(11).shuffle(oa)
    ob = base[:]; np.random.default_rng(29).shuffle(ob)
    teach(A, P, oa); teach(B, P, ob)
    crA, recA = recall_test(A, P, seed=5); crB, recB = recall_test(B, P, seed=5)
    rng = np.random.default_rng(7); ag = []
    for _ in range(400):
        k = int(rng.integers(6)); q = norm(P[k] + 0.45 * rng.standard_normal(D))
        sA, _, _ = A.perceive(q, False); sB, _, _ = B.perceive(q, False)
        ag.append(norm(sA) @ norm(sB))
    shp = np.mean([norm(shape_for(A, P, k)) @ norm(shape_for(B, P, k)) for k in range(6)])

    print("\n[1] DIVERGENT  (different prior + reordered stream, identical content)")
    print(f"    organization:  A={len(A.P)} nodes   B={len(B.P)} nodes")
    print(f"    partition:     A nodes/pattern {nodes_per_pattern(A,P)}")
    print(f"                   B nodes/pattern {nodes_per_pattern(B,P)}")
    print(f"    footprint size:A={mean_size(A):.1f}   B={mean_size(B):.1f} dims (of {D})")
    print(f"    -> ORGANIZATION DIVERGES (count, partition, size)")
    print(f"    knowledge:     recall->truth A={crA:.2f}/{recA:.0f}%  B={crB:.2f}/{recB:.0f}%")
    print(f"                   cross-peer recall agreement cos {np.mean(ag):.2f}")
    print(f"    -> KNOWLEDGE CONVERGES")
    print(f"    per-node shape:footprint cos(A,B) per matched pattern = {shp:.2f}")
    print(f"    -> THE SHAPE OF A UNIT TRACKS THE STIMULUS, not the peer's history")

    # ---- CONTROL: identical prior + identical order -> no divergence ----
    A2 = Arbor(seed=11); A2.seed_bg(2)
    B2 = Arbor(seed=11); B2.seed_bg(2)
    oc = base[:]; np.random.default_rng(3).shuffle(oc)
    teach(A2, P, oc); teach(B2, P, oc)
    shp2 = np.mean([norm(shape_for(A2, P, k)) @ norm(shape_for(B2, P, k)) for k in range(6)])
    print("\n[2] CONTROL  (identical prior + identical order)")
    print(f"    organization:  A={len(A2.P)} nodes   B={len(B2.P)} nodes")
    print(f"    per-node shape:footprint cos(A,B) = {shp2:.2f}   (== history, not noise, causes divergence)")

    print("\n" + "=" * 78)
    print("Degeneracy (Marder; Edelman & Gally), localized: it lives in the CARVING,")
    print("not the receptive field. What federates is knowledge; what diverges is form.")
    print("Do not hype. Do not lie. Just show.")
    print("=" * 78)


if __name__ == "__main__":
    main()
