# AI-001 — an online temporal transit reader

Proposed from [PerceptionLabAI experiment 001](https://github.com/anttiluode/PerceptionLabAI/tree/main/experiments/001_causal_age), 15 November 2026.

## What this adds

A small, **read-only** `TemporalTransitPredictorNode` which estimates two
ordinary linear models using *only past paired samples*: one predicts far[t]
from near[t-D], the other from contemporaneous near[t]. It compares the
held-out current far[t] against both. The graph combines the existing
PulseSource, WaveField, TemporalMultiScope, and SignalPlotter nodes.
No changes are made to the WaveField physics or PerceptionLab host.

## What was verified, and what was not

The [AI workshop's CI run](https://github.com/anttiluode/PerceptionLabAI/actions/runs/43684369567)
passed **6 tests**, including actual upstream WaveField execution, a delayed
synthetic signal, a D=0 identity control, a flat-signal control, a reset check,
and a check that every workflow class and port exists in upstream code.

The **headless 128×128 run** (740 ticks, real upstream wave) predicted a
6.5829-frame transit, used D=7, and observed an average **+11.8858 dB**
delayed-prediction advantage across 370 latter-half valid frames (overlapping
windows, not independent replicates). Final online MSEs were **0.00664636**
for the delayed predictor and **0.01981033** for the simultaneous baseline.
The D=0 control was **exactly 0 dB**.

The saved GUI workflow uses **256×256** and its geometric prediction is
13.17 frames, rounded to D=13. That GUI run was **not** yet measured.
The result is a model-internal causal-age observation; it does not establish
plasticity, neural learning, biological memory, or generalization.

## Run in PerceptionLab

Load `graph workflows/ai_001_wave_transit_predictor.json`. PulseSource
drives WaveField; tap 0 and tap 1 drive the Transit Predictor; four taps
also drive TemporalMultiScope. SignalPlotter displays the advantage,
innovation, and multiscope lag. Both predictors train on earlier frames.
The same host tick delay is shared by both taps.

If modifying `speed`, `substeps`, `first_x`, `last_x`, or `size`,
recalculate D and run a matched D=0 control. The node resets its history
when you change its delay or window.

## Provenance and next checks

Source, runner, unit tests, and receipts remain at
[PerceptionLabAI](https://github.com/anttiluode/PerceptionLabAI).
Next checks: held-out speeds, reversed source/receiver geometry, and
multiple independent seeds or randomized stimuli. Preserve failures.
Do not hype. Do not lie. Just show.
