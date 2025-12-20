import numpy as np
import cv2
import mne
from pathlib import Path

# --- STRICT COMPATIBILITY BOILERPLATE ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode):
            return None

def _try_decimate(rr, tris, target_tris: int):
    """Decimate surface to ~target_tris triangles if available; else keep original."""
    try:
        from mne.surface import decimate_surface
        rr2, tris2 = decimate_surface(rr, tris, n_triangles=int(target_tris))
        return rr2, tris2
    except Exception:
        return rr, tris

def _build_edges_from_tris(tris: np.ndarray):
    """Return symmetric directed edge lists (I,J) from triangles."""
    t = tris.astype(np.int64)
    e01 = t[:, [0, 1]]
    e12 = t[:, [1, 2]]
    e20 = t[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    I = edges[:, 0]
    J = edges[:, 1]
    # add reverse edges
    I2 = np.concatenate([I, J])
    J2 = np.concatenate([J, I])
    return I2, J2

def _smooth_signal(vals: np.ndarray, I: np.ndarray, J: np.ndarray, n: int, steps: int):
    """Simple neighbor-averaging smoothing using directed edges (I->J)."""
    x = vals.astype(np.float32, copy=True)
    eps = 1e-8
    for _ in range(int(steps)):
        neigh_sum = np.bincount(I, weights=x[J], minlength=n).astype(np.float32)
        deg = np.bincount(I, minlength=n).astype(np.float32)
        neigh_mean = neigh_sum / (deg + eps)
        x = 0.5 * x + 0.5 * neigh_mean
    return x

def _hsv_to_bgr_u8(h, s, v):
    """h,s,v in [0..1] -> BGR uint8 image."""
    hsv = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (h * 179.0)          # OpenCV hue: 0..179
    hsv[..., 1] = (s * 255.0)
    hsv[..., 2] = (v * 255.0)
    hsv_u8 = np.clip(hsv, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)
    return bgr

class HoloCortex2DNode(BaseNode):
    """
    HoloCortex (2D)
    --------------
    CPU-only "cortex sheet" renderer driven by complex mode coefficients.

    Inputs:
      - complex_modes: complex_spectrum (length n_modes)
      - phase_coherence: signal (0..1) optional
      - modulation: signal optional (extra gain)

    Outputs:
      - image_out: image
      - cortex_image: image (alias)
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "HoloCortex (2D)"
    NODE_COLOR = QtGui.QColor(180, 50, 255)

    def __init__(self):
        super().__init__()

        self.inputs = {
            "complex_modes": "complex_spectrum",
            "phase_coherence": "signal",
            "modulation": "signal",
        }
        self.outputs = {
            "image_out": "image",
            "cortex_image": "image",
        }

        # ---- Config ----
        self.surface_type = "inflated"     # inflated / pial / white
        self.target_tris_per_hemi = 40000  # good balance; you already saw ~40004 verts
        self.n_modes = 10

        # Rendering
        self.W = 640
        self.H = 360
        self.render_mode = "holo"   # "holo" | "magnitude" | "phase"
        self.gamma = 0.65           # brightness shaping
        self.blur = 3               # post blur (0 disables)
        self.text = True

        # Stabilizers
        self._ema_mag = 0.15
        self._mag_scale = 1.0

        # ---- Load fsaverage and precompute mapping ----
        (self.rr_lh, self.tris_lh,
         self.rr_rh, self.tris_rh) = self._load_fsaverage()

        # Combine vertices for mode topo generation + field reconstruction
        self.rr = np.vstack([self.rr_lh, self.rr_rh]).astype(np.float32)
        self.nv_lh = int(self.rr_lh.shape[0])
        self.nv = int(self.rr.shape[0])

        # Build edges for smoothing (on combined mesh)
        tris_rh_off = self.tris_rh + self.nv_lh
        tris_all = np.vstack([self.tris_lh, tris_rh_off]).astype(np.int32)
        self.edge_I, self.edge_J = _build_edges_from_tris(tris_all)

        # Precompute 2D pixel coords (two-hemisphere layout)
        # Use (y,z) plane, normalize inside each hemi, pack LH left / RH right.
        self.px, self.py = self._precompute_pixel_coords()

        # Precompute "mode topographies" on vertices (smoothness varies by mode)
        self.mode_topos = self._make_smooth_mode_topographies()

        # Preview cache
        self._last = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        print(f"[HoloCortex2D] fsaverage {self.surface_type}: nv={self.nv} (LH={self.nv_lh}, RH={self.rr_rh.shape[0]})")

    def _load_fsaverage(self):
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        fs_dir = Path(fs_dir)
        subjects_dir = fs_dir.parent
        subject = "fsaverage"

        surf_lh = subjects_dir / subject / "surf" / f"lh.{self.surface_type}"
        surf_rh = subjects_dir / subject / "surf" / f"rh.{self.surface_type}"

        rr_lh, tris_lh = mne.read_surface(str(surf_lh))
        rr_rh, tris_rh = mne.read_surface(str(surf_rh))

        rr_lh, tris_lh = _try_decimate(rr_lh, tris_lh, self.target_tris_per_hemi)
        rr_rh, tris_rh = _try_decimate(rr_rh, tris_rh, self.target_tris_per_hemi)

        rr_lh = rr_lh.astype(np.float32)
        rr_rh = rr_rh.astype(np.float32)
        tris_lh = tris_lh.astype(np.int32)
        tris_rh = tris_rh.astype(np.int32)

        print(f"Loaded fsaverage {self.surface_type}: vertices={rr_lh.shape[0]+rr_rh.shape[0]}, faces={tris_lh.shape[0]+tris_rh.shape[0]}")
        return rr_lh, tris_lh, rr_rh, tris_rh

    def _precompute_pixel_coords(self):
        W, H = self.W, self.H

        def hemi_to_pixels(rr_hemi, x0, x1):
            yz = rr_hemi[:, [1, 2]].astype(np.float32)
            mn = yz.min(axis=0)
            mx = yz.max(axis=0)
            span = (mx - mn) + 1e-6
            uv = (yz - mn) / span  # 0..1
            # pad a bit away from borders
            pad = 0.05
            uv = pad + (1 - 2*pad) * uv
            px = (x0 + uv[:, 0] * (x1 - x0)).astype(np.int32)
            py = ((1.0 - uv[:, 1]) * (H - 1)).astype(np.int32)
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)
            return px, py

        px_lh, py_lh = hemi_to_pixels(self.rr_lh, 0, (W // 2) - 1)
        px_rh, py_rh = hemi_to_pixels(self.rr_rh, (W // 2), W - 1)

        px = np.concatenate([px_lh, px_rh], axis=0)
        py = np.concatenate([py_lh, py_rh], axis=0)
        return px, py

    def _make_smooth_mode_topographies(self):
        rng = np.random.default_rng(1234)
        topos = np.zeros((self.nv, self.n_modes), dtype=np.float32)

        for k in range(self.n_modes):
            base = rng.standard_normal(self.nv).astype(np.float32)

            # Low modes smoother, high modes sharper.
            # k=0 -> lots of smoothing, k=last -> little smoothing
            steps = int(np.interp(k, [0, self.n_modes - 1], [42, 6]))

            sm = _smooth_signal(base, self.edge_I, self.edge_J, self.nv, steps=steps)

            # Normalize
            sm -= sm.mean()
            sm /= (sm.std() + 1e-6)
            topos[:, k] = sm

        return topos

    def step(self):
        cm = self.get_blended_input("complex_modes", "mean")
        if cm is None:
            cm = np.zeros(self.n_modes, dtype=np.complex64)
        else:
            cm = np.asarray(cm, dtype=np.complex64).reshape(-1)
            if cm.size < self.n_modes:
                tmp = np.zeros(self.n_modes, dtype=np.complex64)
                tmp[:cm.size] = cm
                cm = tmp
            else:
                cm = cm[:self.n_modes]

        # Optional gain
        mod = self.get_blended_input("modulation", "mean")
        if mod is None:
            mod = 1.0
        mod = float(np.clip(mod, 0.0, 5.0))

        # Reconstruct complex field on vertices
        field = (self.mode_topos @ (cm * mod)).astype(np.complex64)

        # Vertex -> pixel splat (accumulate)
        W, H = self.W, self.H
        flat_idx = (self.py.astype(np.int64) * W + self.px.astype(np.int64))

        acc_r = np.zeros(W * H, dtype=np.float32)
        acc_i = np.zeros(W * H, dtype=np.float32)
        acc_w = np.zeros(W * H, dtype=np.float32)

        fr = field.real.astype(np.float32)
        fi = field.imag.astype(np.float32)

        np.add.at(acc_r, flat_idx, fr)
        np.add.at(acc_i, flat_idx, fi)
        np.add.at(acc_w, flat_idx, 1.0)

        acc_w = acc_w + 1e-6
        img_c = (acc_r / acc_w) + 1j * (acc_i / acc_w)
        img_c = img_c.reshape(H, W)

        mag = np.abs(img_c).astype(np.float32)
        ph = np.angle(img_c).astype(np.float32)  # -pi..pi

        # Stabilize scaling so it doesn’t flicker to black/white
        mag_peak = float(np.percentile(mag, 99.5))
        self._mag_scale = (1 - self._ema_mag) * self._mag_scale + self._ema_mag * (mag_peak + 1e-6)
        mag_n = mag / (self._mag_scale + 1e-6)
        mag_n = np.clip(mag_n, 0.0, 1.0)
        mag_n = mag_n ** self.gamma

        # Mask where we actually have vertices
        mask = (acc_w.reshape(H, W) > 1.001).astype(np.float32)
        if self.blur and self.blur > 0:
            k = int(self.blur) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            mag_n = cv2.GaussianBlur(mag_n, (k, k), 0)

        # Coherence controls contrast/brightness
        coh = self.get_blended_input("phase_coherence", "mean")
        if coh is None:
            coh = 0.75
        coh = float(np.clip(coh, 0.05, 1.0))

        if self.render_mode == "magnitude":
            g = (mag_n * 255.0 * coh).astype(np.uint8)
            bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        elif self.render_mode == "phase":
            ph01 = (ph + np.pi) / (2.0 * np.pi)
            v = np.clip(mag_n * (0.25 + 0.75 * coh), 0.0, 1.0)
            s = np.ones_like(v) * 1.0
            bgr = _hsv_to_bgr_u8(ph01, s, v)

        else:  # "holo"
            ph01 = (ph + np.pi) / (2.0 * np.pi)
            # Saturation up when coherence up; value from magnitude
            s = np.clip(0.35 + 0.65 * coh, 0.0, 1.0) * np.ones_like(mag_n)
            v = np.clip(mag_n * (0.35 + 0.65 * coh), 0.0, 1.0)
            bgr = _hsv_to_bgr_u8(ph01, s, v)

        # Apply mask & add subtle rim
        bgr = (bgr.astype(np.float32) * mask[..., None]).astype(np.uint8)

        # Edge/rim to make it “feel like cortex”
        rim = cv2.Canny((mask * 255).astype(np.uint8), 40, 120)
        rim = cv2.dilate(rim, np.ones((3, 3), np.uint8), iterations=1)
        bgr[rim > 0] = np.clip(bgr[rim > 0].astype(np.int16) + 35, 0, 255).astype(np.uint8)

        if self.text:
            cv2.putText(bgr, "HoloCortex (2D)", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

        self._last = bgr

    def get_output(self, port_name):
        if port_name in ("image_out", "cortex_image"):
            return self._last
        return None

    def get_display_image(self):
        return self._last
