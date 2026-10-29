"""Perception Lab adapter for the original MoebiusLoopMemory recurrence.
Requires nodes/loopmemory.py (copied from MoebiusLoopMemory).
The original ECG graph is preserved. This is a numerical state, not a wave.
"""
from collections import deque
import math
import os
import sys
import numpy as np
import cv2
import __main__

# Dynamic node loading does not otherwise expose the nodes/ directory.
node_dir = os.path.dirname(os.path.abspath(__file__))
if node_dir not in sys.path:
    sys.path.insert(0, node_dir)
from loopmemory import LoopConfig, LoopMemory

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class MoebiusTemporalMemoryNode(BaseNode):
    NODE_CATEGORY = "Memory"
    NODE_COLOR = QtGui.QColor(68, 160, 205)

    def __init__(self):
        super().__init__()
        self.node_title = "Moebius Temporal Memory"
        self.inputs = {"spectrum_in": "spectrum", "reset": "signal"}
        self.outputs = {
            "memory_vector": "spectrum",
            "memory_signal": "signal",
            "input_novelty": "signal",
            "gate_drive": "signal",
            "history_plot": "image",
        }
        self.dim = 16
        self.geometry = "signed_shift"
        self.leak = .92
        self.gate = 1.2
        self.projection_gain = 4.
        self.input_mode = "change"
        self.seed = 99
        self._build()

    def _build(self):
        self.net = LoopMemory(LoopConfig(dim=16, geometry=self.geometry,
            leak=self.leak, gate=self.gate, output_normalize=False), seed=self.seed)
        rng = np.random.default_rng(2026)
        self.projection = rng.normal(size=(256, 16)) / math.sqrt(256)
        self.state = np.zeros(16, dtype=np.float64)
        self.previous = None
        self.trace = deque(maxlen=200)
        self.novelty = 0.
        self.gate_drive = 0.
        self.plot = np.zeros((120, 256, 3), dtype=np.uint8)

    def step(self):
        reset = self.get_blended_input("reset", "sum")
        if reset is not None and float(reset) > .5:
            self._build()
        vec = self.get_blended_input("spectrum_in", "first")
        u = np.zeros(16)
        if vec is not None:
            a = np.asarray(vec, dtype=np.float64).ravel()
            if a.size:
                # Archived ECG uses 256 coarse image samples, max-normalized.
                a = np.nan_to_num(a[:256], nan=0., posinf=0., neginf=0.)
                if a.size < 256:
                    a = np.pad(a, (0, 256-a.size))
                a = np.clip(a, -1., 1.)
                if self.input_mode == "change":
                    delta = a-self.previous if self.previous is not None else np.zeros(256)
                else:
                    delta = a-float(a.mean())
                self.previous = a.copy()
                u = self.projection_gain*(delta@self.projection)
                length = np.linalg.norm(u)
                if length > 2.:
                    u *= 2./length
        zero = self.net.advance(self.state, np.zeros(16))
        updated = self.net.advance(self.state, u)
        self.novelty = float(np.linalg.norm(updated-zero))
        basal = self.net.B@u
        rotated = self.net.R@self.state
        self.gate_drive = float(np.linalg.norm(
            self.gate*basal*np.tanh(self.net.A@rotated)))
        self.state = updated
        self.trace.append(self.state.copy())
        self._render()

    def _render(self):
        self.plot[:] = 24
        if not self.trace:
            return
        data = np.asarray(self.trace).T
        a = np.clip(data, -1., 1.)
        rgb = np.empty((16, a.shape[1], 3), dtype=np.uint8)
        rgb[:,:,0] = 35+(np.maximum(-a,0)*180).astype(np.uint8)
        rgb[:,:,1] = 35+(np.maximum(a,0)*170).astype(np.uint8)
        rgb[:,:,2] = 35
        field = np.zeros((16,200,3),dtype=np.uint8)
        field[:] = 24
        field[:,-a.shape[1]:] = rgb
        self.plot = cv2.resize(field,(256,120),interpolation=cv2.INTER_NEAREST)

    def get_output(self, port_name):
        if port_name == "memory_vector":
            return self.state.copy()
        if port_name == "memory_signal":
            # Bounded state channel, for deliberately weak experimental feedback.
            return float(.2*np.tanh(self.state[0]))
        if port_name == "input_novelty":
            # Actual incremental change due to input, NOT state rotation alone.
            return float(np.tanh(self.novelty))
        if port_name == "gate_drive":
            return float(np.tanh(self.gate_drive))
        if port_name == "history_plot":
            return self.plot.copy()
        return None

    def get_display_image(self):
        h,w = self.plot.shape[:2]
        return QtGui.QImage(self.plot.data,w,h,3*w,
                    QtGui.QImage.Format.Format_RGB888).copy()

    def get_config_options(self):
        return [
            ("Input mode","input_mode",self.input_mode,
                [("Change","change"),("Appearance","appearance")]),
            ("Geometry","geometry",self.geometry,
                [("Signed shift","signed_shift"),("Cyclic","cyclic"),
                 ("Orthogonal","orthogonal"),("Identity","identity")]),
            ("Leak","leak",self.leak,"float"),
            ("Gate","gate",self.gate,"float"),
            ("Projection gain","projection_gain",self.projection_gain,"float"),
        ]

    def set_config_options(self,options):
        self.input_mode = str(options.get("input_mode",self.input_mode))
        self.geometry = str(options.get("geometry",self.geometry))
        self.leak = float(np.clip(float(options.get("leak",self.leak)),0,1))
        self.gate = float(np.clip(float(options.get("gate",self.gate)),0,3))
        self.projection_gain = float(np.clip(float(
            options.get("projection_gain",self.projection_gain)),.01,20))
        self._build()
