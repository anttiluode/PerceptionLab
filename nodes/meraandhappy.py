
"""
Holography / Tensor Network Nodes for PerceptionLab (Host v9.x)
==============================================================

This file fixes the exact crash you're seeing:

    'HappyHolographNode' object has no attribute 'get_input'
    'MeraRenormNode' object has no attribute 'get_input'

PerceptionLab's BaseNode expects nodes to:
  - declare self.inputs / self.outputs
  - implement step()
  - implement get_output(port_name)
  - implement get_display_image()  (used by NodeItem.update_display)

But your MERA/HaPPY prototypes were written in a "get_input / set_output" style.
So we add a tiny compatibility mixin (_PLCompatOutMixin) that provides:
  - get_input(...)   -> wraps BaseNode.get_blended_input(...)
  - set_output(...)
  - get_output(...)

Nodes included:
  1) MeraRenormNode          (MERA-like Haar disentangling + coarse-grain pyramid)
  2) HappyHolographNode      (AdS-ish hyperbolic disk tiling driven by boundary signal)
  3) RandomTensorNetworkNode (RTN-style random orthogonal contractions + entropy proxy)

No external deps beyond numpy/cv2/PyQt6.
"""

import __main__
import math
from collections import deque

import numpy as np
import cv2


# -----------------------------------------------------------------------------
# Host imports (PerceptionLab injects BaseNode + QtGui into __main__)
# -----------------------------------------------------------------------------
try:
    BaseNode = __main__.BaseNode
except Exception:
    # Standalone fallback (lets you run/pytest the file)
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}

        def pre_step(self):
            self.input_data = {k: [] for k in self.inputs}

        def set_input(self, port_name, value, port_type='signal', coupling=1.0):
            self.input_data.setdefault(port_name, []).append(value)

        def get_blended_input(self, port_name, blend_mode='sum'):
            vals = self.input_data.get(port_name, [])
            if not vals:
                return None
            return vals[0]


try:
    QtGui = __main__.QtGui
except Exception:
    try:
        from PyQt6 import QtGui  # type: ignore
    except Exception:
        QtGui = None


# -----------------------------------------------------------------------------
# Compatibility Mixin (fixes missing get_input / set_output)
# -----------------------------------------------------------------------------
class _PLCompatOutMixin:
    def _ensure_out(self):
        if not hasattr(self, "_out"):
            self._out = {}

    def set_output(self, name, value):
        self._ensure_out()
        self._out[name] = value

    def get_output(self, port_name):
        self._ensure_out()
        return self._out.get(port_name, None)

    def get_input(self, port_name, blend_mode="sum"):
        if hasattr(self, "get_blended_input"):
            try:
                return self.get_blended_input(port_name, blend_mode=blend_mode)
            except TypeError:
                return self.get_blended_input(port_name, blend_mode)
        return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _qimage_to_numpy(qimg):
    """Convert QImage -> np.uint8 RGB."""
    if qimg is None or QtGui is None:
        return None
    if not isinstance(qimg, QtGui.QImage):
        return None
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w = qimg.width()
    h = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return arr[..., :3].copy()


def _to_gray_f32(img):
    """Accept uint8/float images (2D/3D/QImage) and return grayscale float32 in [0,1]."""
    if img is None:
        return None

    if QtGui is not None and isinstance(img, QtGui.QImage):
        img = _qimage_to_numpy(img)
        if img is None:
            return None

    if not isinstance(img, np.ndarray):
        return None

    if img.ndim == 3:
        img = img[..., :3].astype(np.float32).mean(axis=2)
    else:
        img = img.astype(np.float32)

    mx = float(np.max(img)) if img.size else 0.0
    if mx > 1.5:
        img = img / 255.0

    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# =============================================================================
# 1) MERA Renormalizer (Haar disentanglers)
# =============================================================================
class MeraRenormNode(_PLCompatOutMixin, BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(160, 100, 200) if QtGui else None

    def __init__(self):
        super().__init__()
        self.node_title = "MERA Renormalizer"

        self.inputs = {"image_in": "image"}
        self.outputs = {
            "bulk_view": "image",
            "deep_tensor": "spectrum",
            "entropy_spectrum": "spectrum",
            "entropy_flow": "signal",
        }

        # Config as real attributes (so PerceptionLab's config dialog works)
        self.layers = 6
        self.display_size = 320
        self.slice_h = 52
        self.detail_keep = 0.15  # 0..1, leak a bit of detail into coarse

        self._display = None

    @staticmethod
    def _haar(blocks):
        """
        blocks: (H2, W2, 2, 2) float32
        Returns a, dh, dv, dd each (H2, W2)
        """
        b00 = blocks[..., 0, 0]
        b01 = blocks[..., 0, 1]
        b10 = blocks[..., 1, 0]
        b11 = blocks[..., 1, 1]
        a  = (b00 + b01 + b10 + b11) * 0.5
        dh = (b00 + b01 - b10 - b11) * 0.5
        dv = (b00 - b01 + b10 - b11) * 0.5
        dd = (b00 - b01 - b10 + b11) * 0.5
        return a, dh, dv, dd

    @staticmethod
    def _entropy(a, dh, dv, dd, eps=1e-12):
        p0 = a * a
        p1 = dh * dh
        p2 = dv * dv
        p3 = dd * dd
        s = p0 + p1 + p2 + p3 + eps
        p0 /= s; p1 /= s; p2 /= s; p3 /= s
        ent = -(p0*np.log(p0+eps) + p1*np.log(p1+eps) + p2*np.log(p2+eps) + p3*np.log(p3+eps))
        return ent

    def get_config_options(self):
        return [
            ("Layers", "layers", self.layers, None),
            ("Display size", "display_size", self.display_size, None),
            ("Slice height", "slice_h", self.slice_h, None),
            ("Detail keep (0..1)", "detail_keep", self.detail_keep, None),
        ]

    def set_config_options(self, options):
        if isinstance(options, dict):
            for k, v in options.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def step(self):
        img = _to_gray_f32(self.get_input("image_in", blend_mode="first"))
        if img is None:
            return

        layers = int(self.layers)
        disp_w = int(self.display_size)
        slice_h = int(self.slice_h)
        detail_keep = float(np.clip(float(self.detail_keep), 0.0, 1.0))

        current = img
        pyramid = []
        entropies = []

        for _ in range(max(1, layers)):
            h, w = current.shape[:2]
            if min(h, w) < 4:
                break

            h2 = (h // 2) * 2
            w2 = (w // 2) * 2
            cur = current[:h2, :w2]

            blocks = cur.reshape(h2 // 2, 2, w2 // 2, 2).transpose(0, 2, 1, 3)

            a, dh, dv, dd = self._haar(blocks)
            entropies.append(float(np.mean(self._entropy(a, dh, dv, dd))))

            coarse = a + detail_keep * (np.abs(dh) + np.abs(dv) + np.abs(dd)) / 3.0
            coarse = np.clip(coarse, 0.0, 1.0).astype(np.float32)

            vis = cv2.resize(cur, (disp_w, slice_h), interpolation=cv2.INTER_AREA)
            pyramid.append(vis)

            current = coarse

        if not pyramid:
            return

        bulk = np.vstack(pyramid[::-1])
        bulk = cv2.resize(bulk, (disp_w, disp_w), interpolation=cv2.INTER_AREA)
        self.set_output("bulk_view", bulk)
        self._display = bulk

        deep = current.flatten()
        out = np.zeros(256, dtype=np.float32)
        L = min(256, deep.size)
        out[:L] = deep[:L]
        self.set_output("deep_tensor", out)

        ent_vec = np.array(entropies, dtype=np.float32)
        self.set_output("entropy_spectrum", ent_vec)
        self.set_output("entropy_flow", float(np.mean(ent_vec)) if ent_vec.size else 0.0)

    def get_display_image(self):
        return self._display


# =============================================================================
# 2) HaPPY Code / AdS3-ish Hyperbolic Disk
# =============================================================================
class HappyHolographNode(_PLCompatOutMixin, BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(0, 150, 150) if QtGui else None

    def __init__(self):
        super().__init__()
        self.node_title = "HaPPY Code (AdS3)"

        self.inputs = {"boundary_signal": "signal"}
        self.outputs = {"hyperbolic_view": "image", "central_charge": "signal"}

        # Config as attributes
        self.resolution = 320
        self.bulk_depth = 7
        self.buffer_len = 512
        self.line_strength = 0.6

        self.buffer = deque(maxlen=int(self.buffer_len))
        self.tiles = []
        self.edges = []
        self._display = None

        self._last_resolution = None
        self._last_bulk_depth = None
        self._last_buffer_len = None

        self._rebuild()

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Bulk depth", "bulk_depth", self.bulk_depth, None),
            ("Buffer length", "buffer_len", self.buffer_len, None),
            ("Line strength (0..1)", "line_strength", self.line_strength, None),
        ]

    def set_config_options(self, options):
        if isinstance(options, dict):
            for k, v in options.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        # ensure rebuild happens after load
        self._rebuild()

    def _rebuild(self):
        res = int(max(64, self.resolution))
        depth = int(max(1, self.bulk_depth))
        blen = int(max(32, self.buffer_len))

        self.buffer = deque(self.buffer, maxlen=blen)

        cx = res // 2
        cy = res // 2
        R = res * 0.48

        tiles = []
        tiles.append({"depth": 0, "x": cx, "y": cy, "r": max(3, int(res * 0.045)), "parent": -1})

        idx_by_depth = {0: [0]}
        for d in range(1, depth + 1):
            n = 5 * (2 ** (d - 1))
            rr = math.tanh(0.55 * d) * R
            radius = max(2, int(res * (0.042 * (0.83 ** d))))
            indices = []
            phase = d * 0.3
            for k in range(n):
                ang = (2 * math.pi * k / n) + phase
                x = cx + rr * math.cos(ang)
                y = cy + rr * math.sin(ang)
                parent_list = idx_by_depth[d - 1]
                parent = parent_list[int(round(k * len(parent_list) / n)) % len(parent_list)]
                tiles.append({"depth": d, "x": x, "y": y, "r": radius, "parent": parent})
                indices.append(len(tiles) - 1)
            idx_by_depth[d] = indices

        edges = []
        for i, t in enumerate(tiles):
            p = t["parent"]
            if p >= 0:
                edges.append((i, p))
        for inds in idx_by_depth.values():
            if len(inds) > 1:
                for a, b in zip(inds, inds[1:] + inds[:1]):
                    edges.append((a, b))

        self.tiles = tiles
        self.edges = edges

        self._last_resolution = res
        self._last_bulk_depth = depth
        self._last_buffer_len = blen

    def step(self):
        # Hot-rebuild if user changed attributes via Configure dialog
        if (self._last_resolution != int(self.resolution) or
            self._last_bulk_depth != int(self.bulk_depth) or
            self._last_buffer_len != int(self.buffer_len)):
            self._rebuild()

        v = self.get_input("boundary_signal", blend_mode="sum")
        if v is not None:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    self.buffer.append(fv)
            except Exception:
                pass

        if len(self.buffer) < 32:
            return

        res = int(self._last_resolution)
        line_strength = float(np.clip(float(self.line_strength), 0.0, 1.0))

        sig = np.asarray(self.buffer, dtype=np.float32)
        sig = sig - float(np.mean(sig))
        sig = sig / (float(np.std(sig)) + 1e-6)

        canvas = np.zeros((res, res), dtype=np.float32)
        yy, xx = np.ogrid[:res, :res]
        cx = res / 2.0
        cy = res / 2.0
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (res * 0.48) ** 2

        for i, t in enumerate(self.tiles):
            val = sig[i % sig.size]
            c = 0.5 + 0.25 * np.tanh(val)
            cv2.circle(canvas, (int(t["x"]), int(t["y"])), int(t["r"]), float(c), -1, lineType=cv2.LINE_AA)

        edges_img = np.zeros_like(canvas)
        for a, b in self.edges:
            ta = self.tiles[a]; tb = self.tiles[b]
            cv2.line(edges_img, (int(ta["x"]), int(ta["y"])), (int(tb["x"]), int(tb["y"])),
                     1.0, 1, lineType=cv2.LINE_AA)

        canvas = canvas * (1.0 - line_strength) + edges_img * line_strength
        canvas *= disk.astype(np.float32)
        canvas = np.clip(canvas, 0.0, 1.0)

        self.set_output("hyperbolic_view", canvas)
        self._display = canvas

        center_val = float(np.mean(sig[-32:])) if sig.size >= 32 else float(np.mean(sig))
        self.set_output("central_charge", center_val)

    def get_display_image(self):
        return self._display


# =============================================================================
# 3) Random Tensor Network (RTN) Node
# =============================================================================
class RandomTensorNetworkNode(_PLCompatOutMixin, BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(200, 80, 80) if QtGui else None

    def __init__(self):
        super().__init__()
        self.node_title = "Random Tensor Network"

        self.inputs = {"boundary_in": "spectrum"}
        self.outputs = {"bulk_state": "spectrum", "rt_entropy": "signal", "network_view": "image"}

        # Config as attributes
        self.dim = 256
        self.layers = 5
        self.seed = 42
        self.display = 320

        self._Ws = []
        self._display_img = None

        self._last_dim = None
        self._last_layers = None
        self._last_seed = None

        self._rebuild()

    def get_config_options(self):
        return [
            ("Dim (state size)", "dim", self.dim, None),
            ("Layers", "layers", self.layers, None),
            ("Seed", "seed", self.seed, None),
            ("Display", "display", self.display, None),
        ]

    def set_config_options(self, options):
        if isinstance(options, dict):
            for k, v in options.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self._rebuild()

    def _rebuild(self):
        dim = int(max(16, self.dim))
        layers = int(max(1, self.layers))
        seed = int(self.seed)

        rng = np.random.default_rng(seed)
        Ws = []
        for _ in range(layers):
            A = rng.standard_normal((dim, dim), dtype=np.float32)
            Q, _ = np.linalg.qr(A)  # orthogonal-ish
            Ws.append(Q.astype(np.float32))

        self._Ws = Ws
        self._last_dim = dim
        self._last_layers = layers
        self._last_seed = seed

    def step(self):
        # hot rebuild if config changed via dialog (which sets attrs directly)
        if (self._last_dim != int(self.dim) or
            self._last_layers != int(self.layers) or
            self._last_seed != int(self.seed)):
            self._rebuild()

        boundary = self.get_input("boundary_in", blend_mode="mean")
        if boundary is None:
            return

        vec = np.asarray(boundary, dtype=np.float32).flatten()
        dim = int(self._last_dim)

        x = np.zeros(dim, dtype=np.float32)
        L = min(dim, vec.size)
        if L > 0:
            x[:L] = vec[:L]

        n = float(np.linalg.norm(x)) + 1e-8
        x = x / n

        for W in self._Ws:
            x = W @ x
            x = np.tanh(x)

        p = x * x
        p_sum = float(np.sum(p)) + 1e-12
        p = p / p_sum
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        self.set_output("bulk_state", x.astype(np.float32))
        self.set_output("rt_entropy", entropy)

        dim_for_vis = min(256, p.size)
        vis = p[:dim_for_vis]
        side = int(math.sqrt(dim_for_vis))
        if side * side != dim_for_vis:
            side = 16
            vis = p[:256]
        img = vis.reshape(side, side)
        img = img / (float(np.max(img)) + 1e-8)
        disp = int(max(64, self.display))
        img = cv2.resize(img.astype(np.float32), (disp, disp), interpolation=cv2.INTER_NEAREST)

        self.set_output("network_view", img)
        self._display_img = img

    def get_display_image(self):
        return self._display_img
