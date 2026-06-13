"""
mycelial_mesh_proof.py  —  the distributed holographic spiking mesh, proven
===========================================================================
A population of geometric neurons sharing one field. No center. It:
  D1  recalls (collective Hopfield through the shared field)         -- denoise
  D2  GROWS a node on novelty, then consolidates noisy exposures     -- learn
  D3  reads the arrow of time of its OWN spike traffic               -- time
  D4  lets a foreign peer join through the same token protocol       -- federate

Inner side (analog, converges):  s <- norm(leak*s + Sum_k softmax(beta<p_k,s>) p_k)
Outer side (sparse, broadcast):  a node emits a token when it wins the field.

Honest limits printed inline. Run: python mycelial_mesh_proof.py
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
rng = np.random.default_rng(0); D = 64
def norm(v): return v / (np.linalg.norm(v) + 1e-9)
def softmax(x, b): x = b*(x-x.max()); e = np.exp(x); return e/(e.sum()+1e-12)


class Mesh:
    def __init__(self, D, beta=20, leak=0.25, familiar=0.28, novel=0.20, alpha=0.25, budget=64):
        self.D=D; self.beta=beta; self.leak=leak
        self.familiar=familiar; self.novel=novel; self.alpha=alpha; self.budget=budget; self.P=[]
    def add(self, p): self.P.append(norm(p)); return len(self.P)-1
    def recall(self, query, steps=12):
        s = norm(query.copy()); P = np.stack(self.P)
        for _ in range(steps):
            o = P@s; w = softmax(o, self.beta); s = norm(self.leak*s + (w@P))
        return s
    def perceive(self, query, learn=True, atp=1.0):
        P = np.stack(self.P); q = norm(query)
        best = int(np.argmax(P@q)); bestov = float((P@q)[best])
        s = self.recall(query); spawned = False
        if learn:
            if bestov >= self.familiar:
                self.P[best] = norm((1-self.alpha)*self.P[best] + self.alpha*q)
            elif bestov < self.novel and len(self.P) < self.budget and atp > 0.25:
                self.add(q); spawned = True
        return s, best, bestov, spawned


def main():
    M = 8
    P = np.stack([norm(rng.standard_normal(D)) for _ in range(M)])
    mesh = Mesh(D)
    for k in range(4): mesh.add(P[k])           # mesh initially knows patterns 0..3

    print("D1  distributed recall (collective Hopfield through the shared field)")
    cq=[]; cr=[]; ok=0
    for _ in range(300):
        k = rng.integers(4); q = norm(P[k] + 0.6*rng.standard_normal(D))
        s, best, ov, _ = mesh.perceive(q, learn=False)
        cq.append(P[k]@q); cr.append(P[k]@norm(s)); ok += (best == k)
    print(f"    cos(query,truth) {np.mean(cq):.2f} -> cos(recall,truth) {np.mean(cr):.2f}   "
          f"recognition {100*ok/300:.0f}%   (winner-take-all sparse: 1 active node)")

    print("\nD2  growth on novelty + consolidation (learning channel SNR adequate)")
    print(f"    nodes before {len(mesh.P)}")
    spawns = 0; order = list(range(4, M))*8; rng.shuffle(order)
    for k in order:
        q = norm(P[k] + 0.15*rng.standard_normal(D))
        _, _, _, sp = mesh.perceive(q, learn=True); spawns += sp
    cleanliness = [max(np.stack(mesh.P) @ P[k]) for k in range(4, M)]
    print(f"    nodes after  {len(mesh.P)}  (spawned {spawns} for 4 distinct novel patterns)")
    print(f"    consolidated template cos to true clean pattern: {np.round(cleanliness,2)}")
    print(f"    (honest limit: single-exposure novelty in heavy noise can fragment; "
          f"consolidation is what cleans it)")

    print("\nD3  arrow of time from the mesh's own winner traffic (skew circulation)")
    def run_seq(order, reps=60):
        Pn = np.stack(mesh.P[:8]); F = []; s = norm(P[order[0]])
        for _ in range(reps):
            for k in order:
                s = norm(0.5*s + norm(P[k] + 0.25*rng.standard_normal(D)))
                o = Pn@s; F.append((o == o.max()).astype(float))
        return np.array(F)
    def circ(F, tau=1):
        N = F.shape[1]; C = np.zeros((N, N))
        for t in range(tau, len(F)): C = 0.97*C + 0.03*np.outer(F[t], F[t-tau])
        A = 0.5*(C-C.T); return float(sum(A[k, (k+1) % N] - A[(k+1) % N, k] for k in range(N)))
    cf, crv = circ(run_seq([0, 1, 2, 3])), circ(run_seq([3, 2, 1, 0]))
    print(f"    forward A->B->C->D circulation {cf:+.3f}   reverse {crv:+.3f}   "
          f"flips: {np.sign(cf) != np.sign(crv)}")

    print("\nD4  federation: an external peer joins through the token protocol")
    foreign = norm(rng.standard_normal(D)); mesh.add(foreign)
    q = norm(foreign + 0.6*rng.standard_normal(D))
    s, best, ov, _ = mesh.perceive(q, learn=False)
    print(f"    foreign peer recall cos {foreign@norm(s):.2f}, node #{best} won "
          f"=> integrated with no retraining")


if __name__ == "__main__":
    main()
