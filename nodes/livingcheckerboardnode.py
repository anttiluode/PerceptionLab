"""
Living Checkerboard v3 — temporal field / anti-saturation revision
------------------------------------------------------------------

This keeps the useful v2 idea (persistent field + small local read apertures), but fixes
two problems that make the first Stage-A experiment easy to misread:

1. WRITES ARE TARGET-RELAXATION, NOT DC INTEGRATION.
   v2 did:
       field += write_gain * (controller_output - 0.5)
   every frame. Any persistent non-zero controller output therefore pushes a tile toward
   a rail until template/leak happen to balance it.

   v3 interprets controller output as a LOCAL TARGET:
       field += write_gain * (target - field)
   so a steady controller can hold a state without necessarily saturating it.

2. THE FIELD CAN CARRY TEMPORAL PROPAGATION AT A USEFUL UI TIMESCALE.
   Several stable local diffusion substeps are allowed per PerceptionLab frame.
   This is still diffusion, not a claim of wave propagation.  It is only to make
   field-mediated lag measurable before template pull erases the perturbation.

The original checkerboard mechanism is still available through square_size.  For the
closest population generalisation of the ECG accident, connect ALL Home.signal_out ports
to square_size as a weak summed/global geometry drive, while retaining each Home's local
read/write loop.

Suggested first wiring:

    Living.read_i -> Home_i.signal_in
    Home_i.signal_out -> Living.write_i

Then, for the temporal/geometry condition:

    Home_0.signal_out --weak--> Living.square_size
    Home_1.signal_out --weak--> Living.square_size
    Home_2.signal_out --weak--> Living.square_size
    Home_3.signal_out --weak--> Living.square_size

Use edge weights or an averaging node so the summed geometry drive stays moderate.

This is an experiment node, not a biological claim.
"""

from __future__ import annotations

import math
import numpy as np
import cv2
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class LivingCheckerboardNode(BaseNode):
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(125, 150, 205)
    N_COMPARTMENTS = 8

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Living Checkerboard v3"

        self.size = int(size)

        self.inputs = {
            "square_size": "signal",
            "global_drive": "signal",
        }
        for i in range(self.N_COMPARTMENTS):
            self.inputs[f"write_{i}"] = "signal"

        self.outputs = {
            "image": "image",
            "field_vector": "spectrum",
            "read_vector": "spectrum",
        }
        for i in range(self.N_COMPARTMENTS):
            self.outputs[f"read_{i}"] = "signal"

        self.field = np.full((self.size, self.size), 0.5, dtype=np.float32)

        # Field dynamics.  diffusion is PER SUBSTEP and should remain <= ~0.24
        # for the explicit 2-D five-point update.
        self.template_pull = 0.004
        self.diffusion = 0.16
        self.diffusion_steps = 4
        self.leak = 0.0005

        # Local controller write: move the tile toward a bounded target rather than
        # integrating a DC offset forever.
        self.write_gain = 0.06
        self.write_center = 0.5
        self.write_sharpness = 1.5

        self.global_gain = 0.03

        # Smooth the geometry control before converting it to integer square width.
        # The final integer conversion deliberately preserves the boundary-crossing
        # discontinuity of the old CheckerboardNode.
        self.geometry_state = 0.10
        self.geometry_slew = 0.08

        # Small, spatially inequivalent read apertures.
        self.read_aperture = 0.15
        self.aperture_seed = 1234

        self.vector_dim = 256
        self.field_vector = np.full(self.vector_dim, 0.5, dtype=np.float32)
        self.read_values = np.full(self.N_COMPARTMENTS, 0.5, dtype=np.float32)

        self._build_compartment_masks()
        self._build_read_apertures()
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def _build_compartment_masks(self):
        n = self.N_COMPARTMENTS
        cols = int(math.ceil(math.sqrt(n * 2.0)))
        rows = int(math.ceil(n / cols))

        self.masks = []
        self.tiles = []
        for i in range(n):
            r, c = divmod(i, cols)
            y0 = int(round(r * self.size / rows))
            y1 = int(round((r + 1) * self.size / rows))
            x0 = int(round(c * self.size / cols))
            x1 = int(round((c + 1) * self.size / cols))

            mask = np.zeros((self.size, self.size), dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            self.masks.append(mask)
            self.tiles.append((y0, y1, x0, x1))

    def _build_read_apertures(self):
        rng = np.random.default_rng(int(self.aperture_seed))
        frac = float(np.clip(self.read_aperture, 0.01, 1.0))

        self.apertures = []
        for y0, y1, x0, x1 in self.tiles:
            h, w = y1 - y0, x1 - x0
            side = max(2, int(round(frac * min(h, w))))
            side = min(side, h, w)
            ry0 = y0 + int(rng.integers(0, max(1, h - side + 1)))
            rx0 = x0 + int(rng.integers(0, max(1, w - side + 1)))
            self.apertures.append((ry0, ry0 + side, rx0, rx0 + side))

    def _checkerboard(self, control):
        square_size = int(5 + float(control) * 50)
        square_size = max(1, min(self.size, square_size))
        y, x = np.mgrid[0:self.size, 0:self.size]
        return (((x // square_size) + (y // square_size)) % 2).astype(np.float32)

    @staticmethod
    def _laplacian(a):
        p = np.pad(a, 1, mode="edge")
        return (
            p[1:-1, 2:] + p[1:-1, :-2] +
            p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * a
        )

    def _bounded_write_target(self, value):
        # Controller outputs may extend outside 0..1.  Convert them smoothly to a
        # valid local field target while keeping 0.5 neutral.
        z = (float(value) - self.write_center) * self.write_sharpness
        return 0.5 + 0.5 * np.tanh(z)

    def step(self):
        size_in = self.get_blended_input("square_size", "sum")
        if size_in is not None:
            target_geometry = float(size_in)
            self.geometry_state += self.geometry_slew * (
                target_geometry - self.geometry_state
            )

        target = self._checkerboard(self.geometry_state)

        # Weak geometric attractor.
        self.field += self.template_pull * (target - self.field)

        # Local transport. Multiple stable substeps make propagation visible on the
        # timescale of an interactive patch.
        d = float(np.clip(self.diffusion, 0.0, 0.24))
        for _ in range(max(0, int(self.diffusion_steps))):
            self.field += d * self._laplacian(self.field)

        self.field += self.leak * (0.5 - self.field)

        global_drive = self.get_blended_input("global_drive", "sum")
        if global_drive is not None:
            gtarget = self._bounded_write_target(global_drive)
            self.field += self.global_gain * (gtarget - self.field)

        # Independent local writes as target relaxation.
        for i, mask in enumerate(self.masks):
            value = self.get_blended_input(f"write_{i}", "sum")
            if value is None:
                continue
            local_target = self._bounded_write_target(value)
            self.field += self.write_gain * mask * (local_target - self.field)

        self.field = np.clip(self.field, 0.0, 1.0)

        # Small local reads.
        for i, (ry0, ry1, rx0, rx1) in enumerate(self.apertures):
            self.read_values[i] = float(
                self.field[ry0:ry1, rx0:rx1].mean()
            )

        # Whole-field vector.
        side = int(math.ceil(math.sqrt(self.vector_dim)))
        tiny = cv2.resize(
            self.field, (side, side), interpolation=cv2.INTER_AREA
        ).flatten()
        self.field_vector[:] = 0.0
        n = min(len(tiny), self.vector_dim)
        self.field_vector[:n] = tiny[:n]

        self._render()

    def _render(self):
        gray = (np.clip(self.field, 0.0, 1.0) * 255).astype(np.uint8)
        display = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

        for y0, y1, x0, x1 in self.tiles:
            cv2.rectangle(
                display, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1
            )

        # Yellow boxes MUST be visible if v3/v2 aperture code is actually loaded.
        for i, (ry0, ry1, rx0, rx1) in enumerate(self.apertures):
            cv2.rectangle(
                display, (rx0, ry0), (rx1 - 1, ry1 - 1), (0, 255, 255), 2
            )
            cv2.putText(
                display, str(i), (rx0, max(11, ry0 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1
            )

        cv2.putText(
            display, f"geom {self.geometry_state:.3f}", (4, self.size - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1
        )
        self.display_image = np.ascontiguousarray(display)

    def get_output(self, port_name):
        if port_name == "image":
            return np.stack([self.field] * 3, axis=-1).astype(np.float32)
        if port_name == "field_vector":
            return self.field_vector
        if port_name == "read_vector":
            return self.read_values.astype(np.float32)
        if port_name.startswith("read_"):
            try:
                i = int(port_name.split("_", 1)[1])
            except Exception:
                return 0.0
            if 0 <= i < self.N_COMPARTMENTS:
                return float(self.read_values[i])
            return 0.0
        return None

    def get_display_image(self):
        h, w = self.display_image.shape[:2]
        return QtGui.QImage(
            self.display_image.data, w, h, w * 3,
            QtGui.QImage.Format.Format_RGB888
        )

    def get_config_options(self):
        return [
            ("Template Pull", "template_pull", self.template_pull, "float"),
            ("Diffusion", "diffusion", self.diffusion, "float"),
            ("Diffusion Steps", "diffusion_steps", self.diffusion_steps, None),
            ("Leak", "leak", self.leak, "float"),
            ("Write Gain", "write_gain", self.write_gain, "float"),
            ("Write Center", "write_center", self.write_center, "float"),
            ("Write Sharpness", "write_sharpness", self.write_sharpness, "float"),
            ("Global Gain", "global_gain", self.global_gain, "float"),
            ("Geometry Slew", "geometry_slew", self.geometry_slew, "float"),
            ("Read Aperture", "read_aperture", self.read_aperture, "float"),
            ("Aperture Seed", "aperture_seed", self.aperture_seed, "float"),
        ]

    def set_config_options(self, options):
        rebuild = False
        float_names = (
            "template_pull", "diffusion", "leak", "write_gain",
            "write_center", "write_sharpness", "global_gain",
            "geometry_slew",
        )
        for name in float_names:
            if name in options:
                setattr(self, name, float(options[name]))

        if "diffusion_steps" in options:
            self.diffusion_steps = int(float(options["diffusion_steps"]))
        if "read_aperture" in options:
            self.read_aperture = float(options["read_aperture"])
            rebuild = True
        if "aperture_seed" in options:
            self.aperture_seed = int(float(options["aperture_seed"]))
            rebuild = True
        if rebuild:
            self._build_read_apertures()
'''

scope = r'''"""
Temporal Multi Scope
--------------------

A replacement/companion for the first MultiScope.

Why:
- ordinary zero-lag coherence can call DEAD flat channels "coherent";
- field-mediated communication is expected to appear as a LAG, so zero-lag Pearson
  correlation can miss the interesting thing entirely;
- saturation must be distinguished from stable specialization.

Inputs:
    vector_in : arbitrary channel vector
    sig_0..3  : optional loose channels

Outputs:
    spread         : mean across-time channel spread
    coherence      : zero-lag mean pairwise correlation; 0 when <2 active channels
    activity       : mean temporal std across channels
    saturation     : fraction of samples close to 0 or 1
    lag_coherence  : mean best absolute cross-correlation across +/- max_lag
    mean_abs_lag   : mean lag, in steps, at that best correlation
    plot           : trace display

This is an instrument, not part of the dynamics unless you wire its scalar outputs back.
"""

from __future__ import annotations

import numpy as np
import cv2
from collections import deque
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class TemporalMultiScopeNode(BaseNode):
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(55, 85, 150)
    N_LOOSE = 4

    def __init__(self, history=400):
        super().__init__()
        self.node_title = "Temporal Multi Scope"

        self.inputs = {"vector_in": "spectrum"}
        for i in range(self.N_LOOSE):
            self.inputs[f"sig_{i}"] = "signal"

        self.outputs = {
            "plot": "image",
            "spread": "signal",
            "coherence": "signal",
            "activity": "signal",
            "saturation": "signal",
            "lag_coherence": "signal",
            "mean_abs_lag": "signal",
        }

        self.history_len = int(history)
        self.max_lag = 60
        self.live_threshold = 1e-5
        self.rail_epsilon = 0.02
        self.autoscale = 1.0

        self.buf = None
        self.n_ch = 0

        self.spread_value = 0.0
        self.coherence_value = 0.0
        self.activity_value = 0.0
        self.saturation_value = 0.0
        self.lag_coherence_value = 0.0
        self.mean_abs_lag_value = 0.0

        self.plot_img = np.zeros((240, 440, 3), dtype=np.uint8)
        self.palette = [
            (80, 220, 255), (120, 255, 120), (255, 140, 90), (255, 120, 220),
            (255, 230, 100), (160, 160, 255), (100, 255, 220), (220, 180, 255),
        ]

    def _gather(self):
        vals = []
        vec = self.get_blended_input("vector_in", "first")
        if vec is not None:
            vals.extend(np.asarray(vec, dtype=np.float32).ravel().tolist())

        for i in range(self.N_LOOSE):
            s = self.get_blended_input(f"sig_{i}", "sum")
            if s is not None:
                vals.append(float(s))

        return np.asarray(vals, dtype=np.float32) if vals else None

    @staticmethod
    def _corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        sa = float(a.std())
        sb = float(b.std())
        if sa < 1e-12 or sb < 1e-12:
            return None
        return float(np.mean((a / sa) * (b / sb)))

    def _best_lag(self, a, b):
        max_lag = min(int(self.max_lag), max(0, len(a) // 3))
        best_corr = None
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                aa, bb = a[-lag:], b[:len(b) + lag]
            elif lag > 0:
                aa, bb = a[:len(a) - lag], b[lag:]
            else:
                aa, bb = a, b

            if len(aa) < 12:
                continue
            c = self._corr(aa, bb)
            if c is None:
                continue
            if best_corr is None or abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag

        return best_corr, best_lag

    def step(self):
        vals = self._gather()
        if vals is None or vals.size == 0:
            return

        if self.buf is None or vals.size != self.n_ch:
            self.n_ch = int(vals.size)
            self.buf = deque(maxlen=self.history_len)

        self.buf.append(vals.copy())

        if len(self.buf) >= 2:
            H = np.stack(list(self.buf), axis=0)

            self.spread_value = float(np.mean(np.std(H, axis=1)))

            channel_sd = H.std(axis=0)
            self.activity_value = float(np.mean(channel_sd))

            self.saturation_value = float(
                np.mean((H <= self.rail_epsilon) | (H >= 1.0 - self.rail_epsilon))
            )

            live = channel_sd > float(self.live_threshold)

            # Zero-lag coherence. Crucial change from v1: dead channels are NOT
            # declared perfectly coherent.
            cors = []
            live_ids = np.where(live)[0]
            for ii in range(len(live_ids)):
                for jj in range(ii + 1, len(live_ids)):
                    c = self._corr(H[:, live_ids[ii]], H[:, live_ids[jj]])
                    if c is not None:
                        cors.append(c)
            self.coherence_value = float(np.mean(cors)) if cors else 0.0

            # Lag-aware pairwise statistic.
            lag_corrs = []
            lag_steps = []
            for ii in range(len(live_ids)):
                for jj in range(ii + 1, len(live_ids)):
                    c, lag = self._best_lag(
                        H[:, live_ids[ii]], H[:, live_ids[jj]]
                    )
                    if c is not None:
                        lag_corrs.append(abs(c))
                        lag_steps.append(abs(lag))

            self.lag_coherence_value = (
                float(np.mean(lag_corrs)) if lag_corrs else 0.0
            )
            self.mean_abs_lag_value = (
                float(np.mean(lag_steps)) if lag_steps else 0.0
            )

        self._render()

    def _render(self):
        h, w = 240, 440
        img = np.zeros((h, w, 3), dtype=np.uint8)

        if self.buf is None or len(self.buf) < 2:
            self.plot_img = img
            return

        H = np.stack(list(self.buf), axis=0)
        T, C = H.shape

        if self.autoscale >= 0.5:
            vmin, vmax = float(H.min()), float(H.max())
            pad = 0.05 * max(1e-6, vmax - vmin)
            vmin, vmax = vmin - pad, vmax + pad
        else:
            vmin, vmax = 0.0, 1.0
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        def y_of(v):
            # leave 58 px for text
            trace_h = h - 60
            return 58 + int(
                trace_h - 1 - (v - vmin) / (vmax - vmin) * (trace_h - 1)
            )

        for c in range(C):
            col = self.palette[c % len(self.palette)]
            prev = None
            for t in range(T):
                x = int(t / max(1, T - 1) * (w - 1))
                p = (x, y_of(float(H[t, c])))
                if prev is not None:
                    cv2.line(img, prev, p, col, 1)
                prev = p

        lines = [
            f"spread {self.spread_value:.4f}   activity {self.activity_value:.4f}",
            f"zero coher {self.coherence_value:+.3f}   saturation {self.saturation_value:.3f}",
            f"lag coher {self.lag_coherence_value:.3f}   mean |lag| {self.mean_abs_lag_value:.1f}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                img, text, (4, 15 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1
            )

        self.plot_img = np.ascontiguousarray(img)

    def get_output(self, port_name):
        if port_name == "plot":
            return self.plot_img.astype(np.float32) / 255.0
        if port_name == "spread":
            return float(self.spread_value)
        if port_name == "coherence":
            return float(self.coherence_value)
        if port_name == "activity":
            return float(self.activity_value)
        if port_name == "saturation":
            return float(self.saturation_value)
        if port_name == "lag_coherence":
            return float(self.lag_coherence_value)
        if port_name == "mean_abs_lag":
            return float(self.mean_abs_lag_value)
        return None

    def get_display_image(self):
        h, w = self.plot_img.shape[:2]
        return QtGui.QImage(
            self.plot_img.data, w, h, w * 3,
            QtGui.QImage.Format.Format_RGB888
        )

    def get_config_options(self):
        return [
            ("History", "history_len", self.history_len, None),
            ("Max Lag", "max_lag", self.max_lag, None),
            ("Live Threshold", "live_threshold", self.live_threshold, "float"),
            ("Rail Epsilon", "rail_epsilon", self.rail_epsilon, "float"),
            ("Autoscale", "autoscale", self.autoscale, "float"),
        ]

    def set_config_options(self, options):
        if "history_len" in options:
            self.history_len = int(float(options["history_len"]))
            self.buf = None
        if "max_lag" in options:
            self.max_lag = int(float(options["max_lag"]))
        if "live_threshold" in options:
            self.live_threshold = float(options["live_threshold"])
        if "rail_epsilon" in options:
            self.rail_epsilon = float(options["rail_epsilon"])
        if "autoscale" in options:
            self.autoscale = float(options["autoscale"])