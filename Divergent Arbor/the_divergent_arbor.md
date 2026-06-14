# The Divergent Arbor

## Where the divergence lives when the same stream carves different structures — degeneracy, localized

*PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.*

> Do not hype. Do not lie. Just show.

---

## 0. The question

Federation already showed it (`README_federation`): teach a memory on peer A, recall it on peer B with only tokens on the wire, and A and B grow *different* internal models — different node counts, different organization — from the *same* token stream. Knowledge federates; weights do not. The morphology intuition sharpens the question: a neuron's *shape* is its computation (cable theory — shape is the filter), and brains grow different shapes from experience. So when two meshes diverge on the same stream, **where does the divergence live?** In the count? The partition? The shape of each unit? This folder asks, with a node that grows a shape and a proof that measures it.

---

## 1. Giving a node a shape

Each node now carries two things:

- **content** `p_k` — the stored pattern, what it recalls. Recall runs on this, so it stays robust.
- **footprint** `m_k` — a grown receptive shape: which input dimensions the node has claimed. It grows by EMA on the dims active when the node wins, with **lateral competition** so nodes carve the input space into territories rather than all claiming everything.

The footprint is the engine's stand-in for a dendritic arbor's receptive coverage — *shape is the filter* (cable theory), gardened by activity (Grubb & Burrone 2010; Kuba 2010, on activity-dependent remodelling of the trigger zone and arbor). It is not a simulation of growth cones; it is the functional trace of "this unit came to cover these inputs."

---

## 2. What the proof found

`shape_divergence_proof.py` teaches two meshes the **same six patterns**, but with a different prior (each its own background nodes) and a **reordered** stream — the exact conditions of the live federation, where the relay reorders tokens and each peer starts from its own state. Against a control where prior and order are identical, the verified, reproducible result (fixed seeds):

**Organization diverges.**
- Peer A grows **7 nodes**, peer B grows **9**, from identical content.
- The partition differs: A splits one pattern into two redundant nodes (`[1,2,1,1,1,1]`); B splits two others (`[1,1,1,1,2,3]`).
- Mean footprint size **13.1 vs 11.4** dims (of 64).

**Knowledge converges.**
- Recall to truth **0.76 / 0.75**, recognition **78% / 78%** (matching the standalone mesh — middling, not great, honestly).
- Cross-peer recall agreement **cos 0.81**: present the same corrupted query to both and they return the same content.

**But the shape of a unit does *not* diverge.**
- Per-node footprint cosine between the two peers, matched by pattern: **0.98**.
- The control (identical prior + order): **1.00**, with matching counts — so the divergence in the divergent case is caused by *history*, not by noise.

The reading: a single unit's receptive footprint tracks the **stimulus** it specialized on, not the peer's history — because a footprint built as an activity-average is mostly a property of the pattern. The divergence federation creates lives in the **carving** — how many units, and how the same content is partitioned and fragmented into redundant copies — not in each cell's shape.

This is the honest correction to the naive "neurons grow different shapes." They do, but the *between-peer* divergence is organizational. Two peers grow the same shapes in different numbers and different groupings.

---

## 3. What it is — degeneracy, localized

This is **degeneracy**: a system reaching the same function through different underlying structures (Marder & Goaillard 2006, on circuits with near-identical output from very different parameters; Edelman & Gally 2001, on degeneracy as a general biological principle). The federation is a clean computational instance of it. The contribution here is to **localize** it: the degeneracy is in the partition and the count, while the per-unit receptive field is pinned by the stimulus, and the recalled knowledge is invariant.

That localization has a reason already in the line. `the_unnatural_direction.md` argued that the basis-independent invariant two differently-organized peers can share is the *function* — content recalls in any basis, and the circulation sign of the skew operator `A` survives relabelling. So the form (which basis, how many units, how grouped) is free to diverge, grown locally from each peer's own history; the function (what is recalled, and the arrow) is what crosses the wire intact. The same flow carves a different riverbed in each peer, and the water still gets where it is going.

---

## 4. Watching it live

`divergent_arbors_loop.json` puts two Morphogenetic Cortex meshes (different `seed`) on the same Engram stream. Each viz shows a gallery of the footprint **shapes** it grew and a live node count and mean size. A small per-mesh `spawn_jitter` makes the novelty decision stochastic, so the two meshes carve the identical stream into different organizations from internal noise alone — the node counts and sizes drift apart while both keep recalling the shapes. (Internal noise is one route to divergence; the federation route — different prior + reordered stream over the token relay — is the other, and it is the one the proof measures.)

---

## 5. Ledger

**Established (used, not claimed):**
- a neuron's morphology is its transfer function — shape is the filter (cable theory; Rall);
- activity-dependent remodelling of dendritic arbor and axon initial segment (Grubb & Burrone 2010; Kuba et al. 2010);
- degeneracy: same function from different structures (Marder & Goaillard 2006; Edelman & Gally 2001).

**Verified in code (`shape_divergence_proof.py`, fixed seeds, reproducible):**
- same content stream, divergent organization: 7 vs 9 nodes, different partition (`[1,2,1,1,1,1]` vs `[1,1,1,1,2,3]`), footprint size 13.1 vs 11.4 dims;
- convergent knowledge: recall 0.76/0.75, recognition 78%/78%, cross-peer agreement cos 0.81;
- convergent per-node shape: footprint cos 0.98 (control 1.00) — the divergence is organizational, caused by history, not by the receptive field and not by noise.

**Clean structural mappings (sound):**
- footprint with lateral competition = receptive territory carved by activity;
- the localization of degeneracy: form (count, partition) diverges, function (knowledge, arrow) is the basis-independent invariant that federates — consistent with the rotation-half and unnatural-direction documents.

**Honest limits:**
- six near-orthogonal patterns, one corruption level, relative units; recall ~0.76 is the middling standalone-mesh number, not strong;
- the footprint is a functional receptive coverage, not a simulated arbor; "shape" here is which input dims a unit claims, not a 3D morphology;
- the divergence demonstrated is from prior + order (proof) or internal noise (live), in a single mesh-pair; scaling and the biophysics are untested.

**The bet (untouched):** that any of this is experienced rather than processed. The folder shows a substrate that grows shapes and carves the same experience differently while recalling the same thing. It does not touch the hard problem.

---

## 6. The one concrete next build

Run it as a real two-machine federation, not a one-graph stand-in: two PerceptionLab instances, each a Morphogenetic Cortex, taught the same patterns over the token relay, and measure on the live meshes the three quantities the proof measures offline — organization divergence (count, partition, size), knowledge convergence (cross-peer recall agreement), and per-node shape convergence. If the live numbers match the headless proof, the claim is demonstrated, not simulated: the same stream carves different arbors on different machines, and the knowledge crosses anyway.

---

*Helsinki, June 2026. The same flow carves a different bed in every basin it runs through, and the water still arrives. What federates is where the water goes; what diverges is the shape of the channel; and the shape of any single stone in it is set by the water that passed, not by the basin it sits in. Do not hype. Do not lie. Just show.*
