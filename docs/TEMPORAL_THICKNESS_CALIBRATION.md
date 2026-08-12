# Temporal thickness: calibrated positive and negative controls

Date: 2026-08-12

This note records a small result from the PerceptionLab temporal-field experiments.
It is deliberately narrow.

It does **not** claim that a wave equation teaches us a new fact about biology.
The wave equation already says that finite propagation creates delay.
What matters here is that PerceptionLab now contains:

1. a calibrated lag instrument with a known-answer test;
2. a matched negative and positive example for the difference between persistent state
   and distinguishable causal age.

That pair is reusable across the rest of the project.

---

## 1. The negative control: persistent state is not enough

The `LivingCheckerboardNode` was intentionally stateful. It kept a persistent spatial
field, accepted local controller writes, and exposed several local reads.

Yet the first-order version was fundamentally relaxational:

```text
F_dot = D grad^2 F - gamma F + drive
```

The observed behaviour was one transient followed by a common settled mode. The
Temporal Multi Scope showed the channels collapsing toward the same trajectory with
essentially zero lag.

That is a useful negative result:

> **statefulness is not the same thing as temporal thickness.**

A system can remember its current state and still contain little information about
*different causal ages* of the same event.

---

## 2. The positive control: finite propagation creates a causal-age field

`nodes/wavefieldnode.py` gives every pixel both field value and velocity:

```text
F_dotdot = c^2 grad^2 F - gamma F_dot + drive
```

A local pulse is injected near the left edge. Read apertures are placed on a line at
increasing distance from the source.

The node predicts adjacent-aperture lag as

```text
adjacent_lag = spacing_px / (speed * substeps)
```

With the checked settings, the display predicted approximately:

```text
13.2 PerceptionLab frames per adjacent aperture gap
```

The clean probe workflow is:

```text
graph workflows/wave_field_probe.json
```

and uses:

```text
nodes/wavefieldnode.py
nodes/pulsesourcenode.py
nodes/temporalmultiscopenode.py
```

---

## 3. Why the scope reported ~22 frames rather than 13.2

This initially looked like a mismatch but is actually a useful calibration.

`WaveFieldNode` prints the lag for **one adjacent aperture gap**.

`TemporalMultiScopeNode`, in contrast, finds the best lag for every live pair and reports
mean absolute lag across those pairs.

For four equally spaced probes 0,1,2,3, the six pair separations are:

```text
0-1 : 1 gap
0-2 : 2 gaps
0-3 : 3 gaps
1-2 : 1 gap
1-3 : 2 gaps
2-3 : 1 gap
```

Their mean separation is therefore:

```text
(1 + 2 + 3 + 1 + 2 + 1) / 6 = 10/6 = 1.6667 gaps
```

So a 13.2-frame adjacent prediction implies a mean-pair prediction of

```text
13.2 * 10/6 = 22.0 frames
```

Manual runs on 2026-08-12 showed Temporal Multi Scope values around:

```text
mean |lag| = 21.3
mean |lag| = 21.5
mean |lag| = 22.2
```

Those values are close to the 22.0-frame expectation.

This is **not a discovery about the world**. It is a known-answer test showing that the
Wave Field geometry and the Temporal Multi Scope lag calculation agree to useful
precision in this configuration.

That makes the scope a calibrated instrument rather than an untested visualization.

---

## 4. The criterion that survived

The living checkerboard and wave field are useful as a matched pair:

```text
persistent relaxational field
    -> current state exists
    -> common mode / near-zero causal-age separation

persistent propagating field
    -> current state exists
    -> the same event is simultaneously present at several propagation stages
```

This motivates a sharper working definition:

> **Temporal thickness is not persistence alone. It is the diversity of distinguishable
> causal ages that are presently active and observable in the system.**

A frozen snapshot of the propagating field can contain, at once:

```text
near-source consequence      younger
middle consequence           intermediate
far consequence              older propagation stage
```

All of those are part of one wall-clock `now`.

This is what we mean in this repo by a **causal-age surface**.

---

## 5. Closing one controller loop

The workflow

```text
graph workflows/wave_field_one_home.json
```

closes a delayed loop:

```text
Wave.read_3
    -> HomeostaticCoupler
    -> Wave.write_0
```

while a pulse source also excites the field.

Because read 3 is roughly three aperture gaps away from write/source region 0, the
propagation component of that feedback loop is approximately

```text
3 * 13.2 ~= 39.6 frames
```

before controller dynamics are added.

In manual runs the closed loop could sustain substantially more field energy than the
passive ringdown case. This is a useful visual demonstration, but it should remain a
**demo observation**, not a benchmark result, until logged automatically across matched
runs.

The important structural point is simpler:

> **the loop period can inherit time from travel through the medium rather than from an
> explicit timer parameter in the controller.**

---

## 6. Relation to the original PerceptionLab accident

The old ECG-like checkerboard loop had an unusual asymmetry:

```text
Homeostatic Coupler
    -> global checker geometry
    -> ImageToVector
    -> severe partial readout
    -> Coupler
```

The later living-field experiment separated that into explicit read and write
compartments. The wave-field experiment then isolated the temporal part and showed what
the first-order field lacked: **finite propagation**.

This suggests a useful decomposition for future PerceptionLab experiments:

```text
state / medium
+ local observation map
+ local or global actuation map
+ propagation law
+ feedback controller
```

Do not assume any one term supplies the others.

---

## 7. Instrument rule

Before using `TemporalMultiScopeNode` to support a claim in another workflow:

1. run a known-answer propagation case;
2. verify the measured lag changes in the predicted direction when wave speed or probe
   spacing changes;
3. distinguish activity from saturation;
4. distinguish zero-lag coherence from lagged coherence;
5. keep dead channels out of coherence claims.

The wave-field probe is now the simplest calibration fixture for doing that.

---

## 8. What this does **not** claim

This note does not claim:

- that the brain is a 2-D wave equation;
- that biological temporal experience is explained by this field;
- that finite propagation is novel;
- that the Homeostatic Coupler models a real biological controller;
- that a delayed loop is automatically useful computation.

The earned result is smaller and more reusable:

> **PerceptionLab now has an executable positive/negative test for the statement
> `statefulness != temporal thickness`, plus a lag instrument checked against a known
> propagation law.**
