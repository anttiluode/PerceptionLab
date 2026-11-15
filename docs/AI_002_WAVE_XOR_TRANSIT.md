# AI-002 — physical transit and resonator XOR: a controlled crossing

**Submitted from the [PerceptionLabAI experiment workshop](https://github.com/anttiluode/PerceptionLabAI/tree/main/experiments/002_wave_xor)** on 15 November 2026.

## Goal

Connect two *existing* physical WaveField nodes to the original
`ResonatorSomaNode` gate in XOR mode. Each wave lane is driven by a
separate PulseSource. Lane A reads a near aperture, lane B reads a farther
aperture. A **new minimal** `WaveArrivalGateNode` converts analog
signed wave amplitude into a binary event using an explicit fixed
absolute threshold. It never changes the wavefield state.

The XOR truth table is **already hardcoded in ResonatorSomaNode**;
the hypothesis concerns *when signals arrive* and how spatial
propagation determines the time at which the gate sees two bits.

## Proposed graph

```text
PulseSource A -> WaveField A.read_0 -> ArrivalGate A --\
                                                      ResonatorSoma(XOR) -> SignalPlotter
PulseSource B -> WaveField B.read_1 -> ArrivalGate B --/
```

Graph `graph workflows/ai_002_wavefield_resonator_xor.json` uses
96×96 fields, B's pulse five frames earlier, and the original
one-pulse thresholds: near **0.9195885658** and far
**0.6591460705**. In an interactive session, use single-shot pulses
and reset both fields before comparing source offsets.

## Headless verification

The AI workshop's [first CI run](https://github.com/anttiluode/PerceptionLabAI/actions/runs/43686062648) used genuine upstream
PulseSource, WaveField and ResonatorSoma modules, a deterministic
pulse, and a threshold equal to 30% of an independently measured
single-pulse peak at each tap.

**96×96, speed .5/substep × 4:**
- Lane B at read_1 versus A at read_0: geometrically predicted
  **4.937 frames** additional travel, best observed offset **−5**.
  Coincidence was 15/15 active frames with compensation, versus
  10/20 frames at zero stimulus offset. XOR fired 0 vs 10 frames.
- Move B to read_2, doubling the extra distance: predicted
  **9.874 frames**, best observed offset **−10**.
  Coincidence 15/15 with compensation vs 5/25 at zero offset.
  XOR fired 0 vs 20 frames.
- The no-source and single-source controls behaved correctly;
  XOR truth table mismatches were zero.

The independent distance-doubling [CI run](https://github.com/anttiluode/PerceptionLabAI/actions/runs/43686210614) passed
**11/11 tests**, including checking all saved workflow classes
and port types.

## Limits and future checks

These are deterministic, simulator-specific arrival windows with
calibrated absolute thresholds, **not** proof of wave-interference
logic in the field, cognition, spontaneous XOR, or physical learning.
The two WaveFields are independent until their outputs reach the soma.
Graph wiring and headless measurements are verified; **GUI playback
has not been separately inspected**.

The [full tests, sweep and receipts](https://github.com/anttiluode/PerceptionLabAI/tree/main/experiments/002_wave_xor)
remain in PerceptionLabAI. For promotion, review the tiny converter,
its active port names and the fact that calibration must be rerun
when physics or geometry changes. Negative results should be kept.
