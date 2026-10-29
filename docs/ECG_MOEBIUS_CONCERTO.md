# ECG + MoebiusLoopMemory ("Concerto")

This is an **experimental combination**, not evidence of brain waves, neurons,
a biological ECG, or a new memory law. The historical ECG graph is preserved.

The baseline graph is [ECG](../graph%20workflows/ecg.json). See
[GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview)
for the historical explanation: 4 coarse image pixels return to a 50-step variance
controller, which modulates checkerboard square size. These are **not eigenmodes**.

## Run on Windows (same Python environment as Perception Lab)

```powershell
git pull
python3.13 perceptionlab12.py
```

Load via the Perception Lab graph-open command:

- [ecg_moebius_observer.json](../graph%20workflows/ecg_moebius_observer.json)
  leaves the original feedback edges intact and adds a **passive** memory observer.
- [ecg_moebius_feedback.json](../graph%20workflows/ecg_moebius_feedback.json)
  additionally connects a bounded memory scalar to coupler `gain_mod`
  (edge weight 0.5). The maximum new gain modulation is ±5% with current code.

The new node is `nodes/moebiusloopmemorynode.py`. It imports
`nodes/loopmemory.py`, copied from [MoebiusLoopMemory](https://github.com/anttiluode/MoebiusLoopMemory).
The recurrence is exactly the same form as in the tested webcam demo:
`s_next=tanh(leak*R@s + B@u + gate*(B@u)*tanh(A@R@s))`.

The node accepts the 256-value `ImageToVector.vector_out` spectrum,
projects this lossily to 16D, and displays a state-history image.
Outputs: `memory_vector` (16D), `memory_signal` (bounded scalar),
`input_novelty` (input-induced state increment), and `gate_drive`.
`change` mode by default has zero drive for an identical image;
`appearance` mode continuously feeds a static image.

### What must be established

1. Compare original, observer and feedback with identical initial settings and
   scheduler. Passive observer **must not** alter the old ECG rhythm.
2. Compare feedback to matched no-apical-gate and unsigned-cyclic controls.
   Changing the waveform is **not yet** making the system useful.
3. For practical functionality, train a **linear readout** on labeled
   temporal sequences or webcam gestures and test on held-out examples.
   Compare against raw frame, simple frame difference, a rolling average and
   the ECG controller's own 50-step variance signal.
4. Separate externally written memory from controller-generated oscillations:
   with zero external input, the controller must not be credited for invented
   sensory information. The `edge_of_chaos` label alone does not establish
   mathematical chaos.

At the pictured 5.6 processed frames/s, a 0.92 linear leak gives an approximate
8.3-step / 1.5-second small-signal half-life; the ECG 50-sample window is about
9 seconds *if* the scheduler advances at the same rate. Actual timescale
depends on the host and inputs. A future real-time version should use
elapsed-time-based decay.

**Verification:** The independent loop and bridge equations and zero-input
behavior were checked headlessly with synthetic vectors. Full PyQt graph
loading and the visual interaction still need a run inside the user's app.
