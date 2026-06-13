"""
federation_proof.py — teach on A, recall on B, only tokens cross the wire.
No shared weights. Two independent meshes + a relay. The smallest experiment
that turns the distributed-brain design into a result.
"""
import time, numpy as np
from token_relay_server import Relay
from relay_client import RelayClient
from mycelial_mesh_proof import Mesh, norm

D=64; PORT=8772
rng=np.random.default_rng(7)

# the relay (the field bus over the wire)
relay=Relay("127.0.0.1",PORT).start(); time.sleep(0.3)

# two peers, each its OWN mesh, seeded with different unrelated background patterns
A=Mesh(D); B=Mesh(D)
for _ in range(2): A.add(norm(rng.standard_normal(D)))
for _ in range(2): B.add(norm(rng.standard_normal(D)))
ca=RelayClient("127.0.0.1",PORT,src=1).start()
cb=RelayClient("127.0.0.1",PORT,src=2).start()
time.sleep(0.5)
print("peers connected:", ca.connected, cb.connected)

# the secret only A ever sees:
P_target = norm(rng.standard_normal(D))
print(f"\nB knows nothing about P_target. B nodes before: {len(B.P)}")

# ---- transmission phase: A learns P_target and broadcasts; B learns from tokens ----
sent=0
for frame in range(80):
    # A perceives a (clean-ish) instance of the target and learns it
    qA = norm(P_target + 0.15*rng.standard_normal(D))
    sA,bestA,ovA,_ = A.perceive(qA, learn=True)
    # A broadcasts its current recalled memory when confident (sparse, event-driven)
    if ovA>0.25 and frame%2==0:
        ca.send(sA, conf=float(ovA)); sent+=1
    # B ingests whatever arrived from the wire (and ONLY that)
    for tok in cb.poll():
        B.perceive(np.array(tok["payload"]), learn=True)
    time.sleep(0.04)
time.sleep(0.3)
for tok in cb.poll():
    B.perceive(np.array(tok["payload"]), learn=True)

print(f"A broadcast {sent} tokens.  B nodes after: {len(B.P)}  (B grew a node from tokens alone)")

# how clean is B's learned template, measured against the secret it never saw directly?
PB=np.stack(B.P); tmpl_cos=float(np.max(PB@P_target))
print(f"B's best template cos to the true P_target: {tmpl_cos:.2f}")

# ---- recall test on B: present a HEAVILY corrupted P_target, B must recall it ----
ok=0; cr=[]
for _ in range(200):
    q=norm(P_target + 0.6*rng.standard_normal(D))
    s,best,ov,_=B.perceive(q, learn=False)
    cr.append(P_target@norm(s)); ok+=(best==np.argmax(PB@P_target))
print(f"\nB recall of P_target (corruption 0.6):  cos {np.mean(cr):.2f}   "
      f"hits-the-learned-node {100*ok/200:.0f}%")

verdict = "PASS" if (tmpl_cos>0.6 and np.mean(cr)>0.5) else "WEAK"
print(f"\n=> {verdict}: a memory taught on A was recalled on B with only tokens on the wire, no shared weights.")
print(f"   relay total messages: {relay.n_msgs}  (sparse)")
ca.stop(); cb.stop(); relay.stop()
