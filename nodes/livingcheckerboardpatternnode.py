"""
Living Checkerboard Field Node
------------------------------

A stateful version of the original CheckerboardNode.

Instead of regenerating an untouchable image every step, this node owns a persistent
2-D field. Eight spatial compartments can independently READ and WRITE that field.

Typical wiring:

    LivingField.read_0  -> HomeostaticCoupler_0.signal_in
    Coupler_0.signal_out -> LivingField.write_0

    LivingField.read_1  -> HomeostaticCoupler_1.signal_in
    Coupler_1.signal_out -> LivingField.write_1
    ...

Optional:
    Coupler_i.signal_out -> Coupler_j.signal_in

This creates two distinct interaction routes:

    1. direct wires between controllers
    2. indirect coupling through the shared persistent field

The field is pulled weakly toward a checkerboard template, diffuses spatially,
leaks slowly toward neutral 0.5, and receives local writes.  The point is not to
claim a biological model.  It is a clean PerceptionLab experiment for asking
whether compartmentalized homeostatic controllers differentiate, synchronize,
or launch propagating disturbances through a shared medium.
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
    NODE_COLOR = QtGui.QColor(130, 150, 190)

    N_COMPARTMENTS = 8

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Living Checkerboard"

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
        }
        for i in range(self.N_COMPARTMENTS):
            self.outputs[f"read_{i}"] = "signal"

        # Persistent field state, 0..1.
        self.field = np.full((self.size, self.size), 0.5, dtype=np.float32)

        # Dynamics.
        self.template_pull = 0.025   # attraction toward current checkerboard
        self.diffusion = 0.055       # communication across compartment boundaries
        self.leak = 0.002            # very slow drift toward neutral 0.5
        self.write_gain = 0.12       # strength of local controller writes
        self.global_gain = 0.04
        self.write_center = 0.5      # coupler setpoint-like neutral write value

        # For direct vector inspection: 16x16 = 256 current field values.
        self.vector_dim = 256
        self.field_vector = np.full(self.vector_dim, 0.5, dtype=np.float32)

        self.read_values = np.full(self.N_COMPARTMENTS, 0.5, dtype=np.float32)

        self._build_compartment_masks()
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def _build_compartment_masks(self):
        """Hard spatial compartments; diffusion is what couples their boundaries."""
        n = self.N_COMPARTMENTS
        cols = int(math.ceil(math.sqrt(n * 2.0)))
        rows = int(math.ceil(n / cols))

        self.masks = []
        for i in range(n):
            r = i // cols
            c = i % cols

            y0 = int(round(r * self.size / rows))
            y1 = int(round((r + 1) * self.size / rows))
            x0 = int(round(c * self.size / cols))
            x1 = int(round((c + 1) * self.size / cols))

            mask = np.zeros((self.size, self.size), dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            self.masks.append(mask)

    def _checkerboard(self, size_in):
        # Keep the original node's mapping as closely as possible.
        if size_in is None:
            size_in = 0.1
        square_size = int(5 + float(size_in) * 50)
        square_size = max(1, min(self.size, square_size))

        y, x = np.mgrid[0:self.size, 0:self.size]
        pattern = ((x // square_size) + (y // square_size)) % 2
        return pattern.astype(np.float32)

    def _laplacian(self, a):
        """Reflecting-boundary 5-point Laplacian."""
        p = np.pad(a, 1, mode="edge")
        return (
            p[1:-1, 2:]
            + p[1:-1, :-2]
            + p[2:, 1:-1]
            + p[:-2, 1:-1]
            - 4.0 * a
        )

    def step(self):
        size_in = self.get_blended_input("square_size", "sum")
        target = self._checkerboard(size_in)

        # 1. The old checkerboard remains a weak attractor, not a hard overwrite.
        self.field += self.template_pull * (target - self.field)

        # 2. The field has its own spatial dynamics.
        if self.diffusion != 0.0:
            self.field += self.diffusion * self._laplacian(self.field)

        # 3. Very slow leak prevents indefinite saturation.
        self.field += self.leak * (0.5 - self.field)

        # 4. Optional whole-field perturbation.
        global_drive = self.get_blended_input("global_drive", "sum")
        if global_drive is not None:
            self.field += self.global_gain * (float(global_drive) - self.write_center)

        # 5. Independent local writes.
        for i, mask in enumerate(self.masks):
            value = self.get_blended_input(f"write_{i}", "sum")
            if value is None:
                continue
            drive = float(value) - self.write_center
            self.field += self.write_gain * drive * mask

        # Soft bounding preserves more gradient than a hard binary rewrite.
        self.field = np.clip(self.field, 0.0, 1.0)

        # 6. Independent local reads.
        for i, mask in enumerate(self.masks):
            denom = float(mask.sum()) + 1e-9
            self.read_values[i] = float((self.field * mask).sum() / denom)

        # 7. 256-value snapshot for the old Image->Vector style experiments.
        side = int(math.ceil(math.sqrt(self.vector_dim)))
        tiny = cv2.resize(
            self.field, (side, side), interpolation=cv2.INTER_AREA
        ).flatten()
        self.field_vector[:] = 0.0
        self.field_vector[: min(len(tiny), self.vector_dim)] = tiny[: self.vector_dim]

        # Display: field plus compartment boundaries.
        gray = (np.clip(self.field, 0.0, 1.0) * 255).astype(np.uint8)
        display = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

        # Draw the actual read/write compartments.
        for mask in self.masks:
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cv2.rectangle(display, (x0, y0), (x1, y1), (220, 220, 220), 1)

        self.display_image = np.ascontiguousarray(display)

    def get_output(self, port_name):
        if port_name == "image":
            # PerceptionLab image convention in the original node is float 0..1 RGB.
            return np.stack([self.field] * 3, axis=-1).astype(np.float32)

        if port_name == "field_vector":
            return self.field_vector

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
            self.display_image.data,
            w,
            h,
            w * 3,
            QtGui.QImage.Format.Format_RGB888,
        )

    def get_config_options(self):
        return [
            ("Template Pull", "template_pull", self.template_pull, "float"),
            ("Diffusion", "diffusion", self.diffusion, "float"),
            ("Leak", "leak", self.leak, "float"),
            ("Write Gain", "write_gain", self.write_gain, "float"),
            ("Global Gain", "global_gain", self.global_gain, "float"),
            ("Write Center", "write_center", self.write_center, "float"),
        ]

    def set_config_options(self, options):
        for name in (
            "template_pull",
            "diffusion",
            "leak",
            "write_gain",
            "global_gain",
            "write_center",
        ):
            if name in options:
                setattr(self, name, float(options[name]))
