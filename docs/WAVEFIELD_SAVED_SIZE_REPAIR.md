# WaveField saved-size repair

**Why this matters:** PerceptionLab's graph loader constructs each node with
`cls()`, then calls `set_config_options(config)`, then assigns any remaining
saved attributes. Before this change, `WaveFieldNode` did not process
`size` in that method. A graph saved with `size: 96` therefore left the
actual wave arrays, read apertures and injection stamps at **256×256**.
Only the `size` attribute changed afterwards. Graph behavior could look
convincing but not match headless experiments that construct
`WaveFieldNode(size=96)` directly.

**Fix:** Expose grid size as an editable property, apply it during configuration,
reallocate both fields and the display image, clear stale readout state, and
rebuild the wave source and aperture geometry. Changing size is explicitly a
reset of physical state, not a lossless resample.

**Regression test:** `python -m unittest discover -s tests -p 'test_wavefield_size_load.py' -v`
checks that the real upstream WaveField, when configured like a saved
PerceptionLab graph, ends up with arrays and geometry at the requested size.
It then injects a pulse and confirms that wave propagation works.

**Trigger:** [PerceptionLabAI experiment 003](https://github.com/anttiluode/PerceptionLabAI/tree/ai/003-fading-field-loop/experiments/003_fade_closed_loop);
the user observed a single fade-triggered spike followed by silence in the GUI,
despite the 96×96 headless loop producing 14 spikes. This issue explains the
scale and calibration mismatch. It does **not** by itself verify GUI operation
after the fix; the actual interactive loop should be retested.

**Scope:** No changes to the wave update equation, damping, physics, or signal
port names. Settings containing `size` will now genuinely resize the field.
All graph workflows that specify `size` should be checked for calibration
differences compared with the old GUI behavior.
