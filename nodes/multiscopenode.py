"""
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