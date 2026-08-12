"""
Wave Field Node — the momentum version of the living checkerboard
------------------------------------------------------------------

The v3 living field was first order:

    F_dot = D grad^2 F - gamma F + drive

Every term says "move toward something". Nothing says "keep moving". So a kick
dies and the compartments settle into one common mode, which is what the scope
showed: activity falling, zero-lag coherence +1, mean lag 0.

This node is second order. Every pixel carries a value AND a velocity:

    F_dotdot = c^2 grad^2 F - gamma F_dot + drive

Now a local impulse travels. The single question this node exists to answer:

    Does one current spatial state contain the same event at several stages
    of propagation?

The apertures are laid out in a LINE at increasing distance from the source, so
the answer is quantitative rather than aesthetic. Expected lag between adjacent
apertures, in PerceptionLab frames:

    lag = spacing_px / (speed * substeps)

The node prints that prediction on its display. Compare it against the Temporal
Scope's "mean |lag|". If the measured lag matches the prediction and moves when
you change speed, the field is carrying the event. If mean lag stays ~0 while
coherence stays high, it is a common mode and the branch closes.

Stability: explicit 2-D wave update needs speed <= ~0.70 per substep. Default
0.50 with 4 substeps is comfortable.

Inputs:
    impulse       : drive injected at the source point
    global_drive  : drive spread over the whole field
    write_0..7    : drive injected at aperture i (for closing a Home loop)

Outputs:
    image, field_vector, read_vector, read_0..7, energy
"""

from __future__ import annotations

import math
import numpy as np
import cv2
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class WaveFieldNode(BaseNode):
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(90, 130, 200)

    N_APERTURES = 8

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Wave Field"

        self.size = int(size)

        self.inputs = {"impulse": "signal", "global_drive": "signal"}
        for i in range(self.N_APERTURES):
            self.inputs[f"write_{i}"] = "signal"

        self.outputs = {
            "image": "image",
            "field_vector": "spectrum",
            "read_vector": "spectrum",
            "energy": "signal",
        }
        for i in range(self.N_APERTURES):
            self.outputs[f"read_{i}"] = "signal"

        # --- wave parameters -------------------------------------------------
        self.speed = 0.50        # px per substep; keep <= 0.70 for stability
        self.damping = 0.010     # velocity damping per substep -> ringdown length
        self.substeps = 4        # substeps per PerceptionLab frame
        self.drive_gain = 0.60   # impulse -> velocity injection strength
        self.source_radius = 4.0 # px, gaussian sigma of the injection spot
        self.edge_absorb = 0.06  # boundary damping band; 0 = reflecting (echoes)

        # --- geometry --------------------------------------------------------
        self.read_aperture = 6   # px, side of each square read window
        self.line_y = 0.50       # aperture line height, fraction of size
        self.first_x = 0.20      # first aperture, fraction of size
        self.last_x = 0.92       # last aperture, fraction of size
        self.source_x = 0.08     # source point, fraction of size

        # --- state -----------------------------------------------------------
        self.field = np.zeros((self.size, self.size), dtype=np.float32)
        self.vel = np.zeros((self.size, self.size), dtype=np.float32)

        self.vector_dim = 256
        self.field_vector = np.zeros(self.vector_dim, dtype=np.float32)
        self.read_values = np.zeros(self.N_APERTURES, dtype=np.float32)
        self.energy_value = 0.0

        self._build_geometry()
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #

    def _build_geometry(self):
        s = self.size
        y = int(self.line_y * s)
        half = max(1, int(self.read_aperture) // 2)

        xs = np.linspace(self.first_x * s, self.last_x * s, self.N_APERTURES)
        self.apertures = []
        self.aperture_xy = []
        for x in xs:
            xi = int(round(x))
            self.apertures.append(
                (max(0, y - half), min(s, y + half + 1),
                 max(0, xi - half), min(s, xi + half + 1))
            )
            self.aperture_xy.append((xi, y))

        self.source_xy = (int(self.source_x * s), y)
        self.spacing_px = float(xs[1] - xs[0]) if self.N_APERTURES > 1 else 0.0

        # Gaussian injection stamps.
        self.source_stamp = self._stamp(*self.source_xy)
        self.write_stamps = [self._stamp(x, yy) for (x, yy) in self.aperture_xy]

        # Boundary absorption mask: 1 in the interior, < 1 near the edges.
        if self.edge_absorb > 0.0:
            band = max(2, int(self.edge_absorb * self.size))
            ramp = np.ones(self.size, dtype=np.float32)
            r = np.linspace(0.90, 1.0, band, dtype=np.float32)
            ramp[:band] = r
            ramp[-band:] = r[::-1]
            self.absorb = np.outer(ramp, ramp).astype(np.float32)
        else:
            self.absorb = None

    def _stamp(self, cx, cy):
        sig = max(0.8, float(self.source_radius))
        rad = int(math.ceil(3.0 * sig))
        y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        g = np.exp(-(x * x + y * y) / (2.0 * sig * sig)).astype(np.float32)
        g /= g.max()
        return (cx, cy, rad, g)

    def _inject(self, stamp, amount):
        cx, cy, rad, g = stamp
        s = self.size
        y0, y1 = cy - rad, cy + rad + 1
        x0, x1 = cx - rad, cx + rad + 1
        gy0 = max(0, -y0)
        gx0 = max(0, -x0)
        y0c, y1c = max(0, y0), min(s, y1)
        x0c, x1c = max(0, x0), min(s, x1)
        if y1c <= y0c or x1c <= x0c:
            return
        sub = g[gy0:gy0 + (y1c - y0c), gx0:gx0 + (x1c - x0c)]
        self.vel[y0c:y1c, x0c:x1c] += amount * sub

    @staticmethod
    def _laplacian(a):
        p = np.pad(a, 1, mode="edge")
        return (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * a)

    # ------------------------------------------------------------------ #

    def step(self):
        imp = self.get_blended_input("impulse", "sum")
        if imp is not None and abs(float(imp)) > 1e-9:
            self._inject(self.source_stamp, self.drive_gain * float(imp))

        for i in range(self.N_APERTURES):
            w = self.get_blended_input(f"write_{i}", "sum")
            if w is not None and abs(float(w)) > 1e-9:
                self._inject(self.write_stamps[i], self.drive_gain * float(w))

        gd = self.get_blended_input("global_drive", "sum")
        if gd is not None and abs(float(gd)) > 1e-9:
            self.vel += 0.02 * self.drive_gain * float(gd)

        c2 = float(self.speed) ** 2
        damp = float(np.clip(self.damping, 0.0, 0.5))

        for _ in range(max(1, int(self.substeps))):
            acc = c2 * self._laplacian(self.field)
            self.vel += acc
            self.vel *= (1.0 - damp)
            self.field += self.vel
            if self.absorb is not None:
                self.field *= self.absorb
                self.vel *= self.absorb

        np.clip(self.field, -4.0, 4.0, out=self.field)
        np.clip(self.vel, -4.0, 4.0, out=self.vel)

        for i, (y0, y1, x0, x1) in enumerate(self.apertures):
            self.read_values[i] = float(self.field[y0:y1, x0:x1].mean())

        self.energy_value = float(
            np.mean(self.field * self.field) + np.mean(self.vel * self.vel)
        )

        side = int(math.ceil(math.sqrt(self.vector_dim)))
        tiny = cv2.resize(self.field, (side, side),
                          interpolation=cv2.INTER_AREA).flatten()
        self.field_vector[:] = 0.0
        n = min(len(tiny), self.vector_dim)
        self.field_vector[:n] = tiny[:n]

        self._render()

    def _render(self):
        # Signed field: blue negative, red positive, dark at zero.
        v = np.clip(self.field * 0.5 + 0.5, 0.0, 1.0)
        u8 = (v * 255).astype(np.uint8)
        disp = cv2.applyColorMap(u8, cv2.COLORMAP_COOL)

        sx, sy = self.source_xy
        cv2.circle(disp, (sx, sy), 5, (255, 255, 255), 1)

        for i, (y0, y1, x0, x1) in enumerate(self.apertures):
            cv2.rectangle(disp, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
            cv2.putText(disp, str(i), (x0 - 2, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)

        pred = self.spacing_px / max(1e-6, self.speed * max(1, int(self.substeps)))
        cv2.putText(disp, f"predicted lag {pred:.1f} frames/gap",
                    (4, self.size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (240, 240, 240), 1)
        cv2.putText(disp, f"energy {self.energy_value:.4f}",
                    (4, self.size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (240, 240, 240), 1)

        self.display_image = np.ascontiguousarray(disp)

    # ------------------------------------------------------------------ #

    def get_output(self, port_name):
        if port_name == "image":
            v = np.clip(self.field * 0.5 + 0.5, 0.0, 1.0)
            return np.stack([v] * 3, axis=-1).astype(np.float32)
        if port_name == "field_vector":
            return self.field_vector
        if port_name == "read_vector":
            return self.read_values.astype(np.float32)
        if port_name == "energy":
            return float(self.energy_value)
        if port_name.startswith("read_"):
            try:
                i = int(port_name.split("_", 1)[1])
            except Exception:
                return 0.0
            if 0 <= i < self.N_APERTURES:
                return float(self.read_values[i])
            return 0.0
        return None

    def get_display_image(self):
        h, w = self.display_image.shape[:2]
        return QtGui.QImage(self.display_image.data, w, h, w * 3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Speed", "speed", self.speed, "float"),
            ("Damping", "damping", self.damping, "float"),
            ("Substeps", "substeps", self.substeps, None),
            ("Drive Gain", "drive_gain", self.drive_gain, "float"),
            ("Source Radius", "source_radius", self.source_radius, "float"),
            ("Edge Absorb", "edge_absorb", self.edge_absorb, "float"),
            ("Read Aperture", "read_aperture", self.read_aperture, None),
            ("Line Y", "line_y", self.line_y, "float"),
            ("First X", "first_x", self.first_x, "float"),
            ("Last X", "last_x", self.last_x, "float"),
            ("Source X", "source_x", self.source_x, "float"),
        ]

    def set_config_options(self, options):
        rebuild = False
        for name in ("speed", "damping", "drive_gain"):
            if name in options:
                setattr(self, name, float(options[name]))
        if "substeps" in options:
            self.substeps = int(float(options["substeps"]))
        for name in ("source_radius", "edge_absorb", "line_y",
                     "first_x", "last_x", "source_x"):
            if name in options:
                setattr(self, name, float(options[name]))
                rebuild = True
        if "read_aperture" in options:
            self.read_aperture = int(float(options["read_aperture"]))
            rebuild = True
        if rebuild:
            self._build_geometry()
