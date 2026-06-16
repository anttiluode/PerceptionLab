"""
PhaseFieldNode.py  —  PerceptionLab
-----------------------------------
A true 2D leaky diffusive field (the volume-conductor / screened-Poisson
operator), so you can inject templates as current sources and watch the medium
settle.

    phi <- phi + dt * ( D * laplacian(phi)  -  phi / tau  +  source )

This is the substrate from §9 of 'Mathematical Holography in the Brain'. On its
own it is a BLUR: its steady-state response to a static source s is G*s, the
source convolved with the field's Green's function. That is the §3 fracture made
visible — a diffusive field has no spatial phase, so it records by superposition
but reads back only a smeared envelope. To recover identity you still need a
projection read-out (a separate node); to recover ORDER you need the clock
(gate this node's write/read with your Moiré/theta coupler).

Ports
  inputs:
    source_in : image   - injected as a current source each step (the template)
    write_gate: signal   - >0.5 = inject source (write); <=0.5 = let field decay
    reset     : signal   - >0.5 = clear the field
  outputs:
    field_out : image    - the settled field (G * sources), 0..1
    energy    : signal    - mean |phi|, a scalar you can plot / feed the coupler

Config: grid size N, diffusion D, leak tau, dt, steps-per-frame.
Stability (explicit Euler): keep dt * (4*D + 1/tau) < 1.
"""
import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class PhaseFieldNode2(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(160, 100, 255)  # wave violet, matches ComplexInterference

    def __init__(self, N=48, D=0.18, tau=8.0, dt=0.2, steps_per_frame=4):
        super().__init__()
        self.node_title = "Phase Field (leaky Laplacian)"

        self.inputs = {
            'source_in': 'image',
            'write_gate': 'signal',
            'reset': 'signal',
        }
        self.outputs = {
            'field_out': 'image',
            'energy': 'signal',
        }

        self.N = int(N)
        self.D = float(D)
        self.tau = float(tau)
        self.dt = float(dt)
        self.steps_per_frame = int(steps_per_frame)

        self.phi = np.zeros((self.N, self.N), dtype=np.float32)
        self.energy_out = 0.0
        self._kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)

    # ---- field operator -------------------------------------------------
    def _lap(self, f):
        # 5-point Laplacian, replicate (Neumann) boundary
        return cv2.filter2D(f, -1, self._kernel, borderType=cv2.BORDER_REPLICATE)

    def _coerce_source(self, img):
        """Accept an incoming image of any size/type, return NxN float32."""
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim == 3:
            a = a.mean(axis=2)
        a = a.astype(np.float32)
        mx = np.abs(a).max()
        if mx > 1.5:            # looks like 0..255
            a = a / 255.0
        if a.shape != (self.N, self.N):
            a = cv2.resize(a, (self.N, self.N), interpolation=cv2.INTER_AREA)
        return a

    # ---- per-step -------------------------------------------------------
    def step(self):
        if (self.get_blended_input('reset', 'sum') or 0.0) > 0.5:
            self.phi[:] = 0.0

        gate = self.get_blended_input('write_gate', 'sum')
        writing = True if gate is None else (gate > 0.5)   # default: always write
        src = self._coerce_source(self.get_blended_input('source_in', 'first'))
        drive = src if (writing and src is not None) else 0.0

        for _ in range(self.steps_per_frame):
            self.phi += self.dt * (self.D * self._lap(self.phi)
                                   - self.phi / self.tau + drive)

        self.energy_out = float(np.mean(np.abs(self.phi)))

    # ---- outputs --------------------------------------------------------
    def get_output(self, port_name):
        if port_name == 'field_out':
            f = self.phi
            mn, mx = f.min(), f.max()
            norm = (f - mn) / (mx - mn + 1e-9)
            return norm.astype(np.float32)
        elif port_name == 'energy':
            return self.energy_out
        return None

    def get_display_image(self):
        f = self.phi
        mn, mx = f.min(), f.max()
        norm = ((f - mn) / (mx - mn + 1e-9) * 255).astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"PHASE FIELD  E={self.energy_out:.3f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"D={self.D} tau={self.tau}", (6, 248),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        return QtGui.QImage(img.data, 256, 256, 256 * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Grid N", "N", self.N, None),
            ("Diffusion D", "D", self.D, None),
            ("Leak tau", "tau", self.tau, None),
            ("dt", "dt", self.dt, None),
            ("Steps/frame", "steps_per_frame", self.steps_per_frame, None),
        ]
