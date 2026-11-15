# The PerceptionLab collaboration cycle

**PerceptionLab** is the host and community library of interchangeable computational organs: sources, oscillators, propagating fields, memories, learners, controllers, and instruments.

**[PerceptionLabAI](https://github.com/anttiluode/PerceptionLabAI)** is the experimental workshop in which Antti, other humans, and AI agents work on those organs by constructing *saved node workflows* rather than isolated demo repositories.

## How we now operate

1. **Ask a narrow question.** State what result would contradict it. Start from [PerceptionLabNodes](https://anttiluode.github.io/PerceptionLabNodes/), which inventories the older modules and workflows, and reuse available code.
2. **Wire the experiment.** Save the runnable JSON graph in PerceptionLabAI. Create a new Python node only when necessary; verify exact input and output ports.
3. **Intervene and measure.** Use matched baselines, negative controls, seed/version tracking, headless runners, and independent readouts. A convincing visualization is not sufficient.
4. **Keep receipts and mistakes.** Commit test code, logs or result JSON, limitations, failed hypotheses, and corrections in the experiment folder.
5. **Propose upstream.** A reusable node and a PerceptionLab-compatible saved workflow can move to PerceptionLab via a reviewed pull request; experimental scaffolding remains in PerceptionLabAI.
6. **Repeat.** Other contributors can fork, branch, run alternatives, compare, and submit reproducible improvements. No contributor gets a privileged scientific claim by identity.

The goal is **compositional research**: connect previously independent mechanisms and ask where behavior, timing, memory, or learning actually lives.

## The first complete example

- [AI workshop experiment 001](https://github.com/anttiluode/PerceptionLabAI/tree/main/experiments/001_causal_age): WaveField + PulseSource + new temporal transit reader, with controls.
- [Original submitted pull request #1](https://github.com/anttiluode/PerceptionLab/pull/1) (merged 15 November 2026).
- [Reusable node](../nodes/temporal_transit_predictor_node.py) and [saved workflow](../graph%20workflows/ai_001_wave_transit_predictor.json) now live in this repository.
- [Measurement report and limitations](AI_001_TEMPORAL_TRANSIT_PREDICTOR.md): a delayed aperture measurement improved linear prediction in one headless simulator run. It is not evidence of learned biological memory.

## Different roles, shared standard

PerceptionLabAI is *not* a fork that replaces the main lab. It is a place for risky ideas and cross-strata combinations, including negative results. PerceptionLab is *not* meant to collect every speculative prototype. It is where reviewed reusable components and provenance are retained.

A newcomer can make a useful contribution without understanding all 466 node files: identify one pair of nodes that should interact, wire them, measure a known quantity and submit the reproducible result.

See the [workshop rules](https://github.com/anttiluode/PerceptionLabAI/blob/main/CONTRIBUTING.md) before contributing.
