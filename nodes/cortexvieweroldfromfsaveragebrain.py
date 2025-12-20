"""
Holographic Cortex Viewer (Perception Lab Node)
----------------------------------------------
Fixes:
- MNE fetch_fsaverage() returns Path (not dict)
- pyqtgraph GLMeshItem color handling: use MeshData.setVertexColors(Nx4)
- avoid GL crash / huge mesh overload via decimation
- avoid black output when modes are zero/unconnected
- always outputs an image (2D fallback if OpenGL unavailable)

Inputs:
- complex_modes: complex_spectrum (len>=n_modes)
- phase_coherence: signal (0..1) optional

Output:
- image_out: image (connect to Display node)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import cv2
import mne

# --- STRICT COMPATIBILITY BOILERPLATE ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

# Optional OpenGL viewer
_HAS_GL = True
try:
    from pyqtgraph.opengl import GLViewWidget, GLMeshItem, MeshData
except Exception:
    _HAS_GL = False


def _try_decimate(rr: np.ndarray, tris: np.ndarray, target_tris: int):
    """Decimate surface if possible; else return original."""
    try:
        # MNE provides decimation in recent versions
        from mne.surface import decimate_surface
        rr2, tris2 = decimate_surface(rr, tris, n_triangles=int(target_tris))
        return rr2, tris2
    except Exception:
        return rr, tris


class HolographicCortexViewerNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Holographic Cortex Viewer"
    NODE_COLOR = QtGui.QColor(180, 50, 255)

    def __init__(self):
        super().__init__()

        self.inputs = {
            "complex_modes": "complex_spectrum",
            "phase_coherence": "signal",
        }
        self.outputs = {
            "image_out": "image",
        }

        # ---- Config ----
        self.n_modes = 10
        self.surface_type = "inflated"      # inflated / pial / white
        self.target_tris_per_hemi = 40000   # lower = faster & less GL pain
        self.low_mode_glow = True

        # ---- Load fsaverage surface ----
        self.vertices, self.faces = self._load_fsaverage_surface()
        self.n_vertices = int(self.vertices.shape[0])

        # ---- Mode topographies placeholder ----
        # IMPORTANT:
        # Replace this with YOUR REAL topographies on the SAME vertex set.
        # If you don't yet have vertex-space topographies, you can:
        #   - start with random (visual sanity)
        #   - then implement interpolation from your mode_topo to vertices.
        rng = np.random.default_rng(0)
        self.mode_topographies = rng.standard_normal((self.n_vertices, self.n_modes)).astype(np.float32)

        # Preview image cache (always exists)
        self._last_preview = np.zeros((256, 256, 3), dtype=np.uint8)

        # Precompute a simple spherical UV mapping for 2D fallback texture
        self._uv = self._compute_uv(self.vertices)  # (n_vertices, 2) in [0,1]

        # ---- OpenGL viewer (optional) ----
        self.gl_view = None
        self.mesh_item = None
        self.meshdata = None

        if _HAS_GL:
            try:
                self.meshdata = MeshData(vertexes=self.vertices, faces=self.faces)

                # Start with uniform vertex colors (Nx4 RGBA float32)
                init_colors = np.ones((self.n_vertices, 4), dtype=np.float32) * 0.5
                init_colors[:, 3] = 1.0
                self.meshdata.setVertexColors(init_colors)

                self.gl_view = GLViewWidget()
                self.gl_view.opts["distance"] = 350
                self.gl_view.opts["fov"] = 60

                self.mesh_item = GLMeshItem(
                    meshdata=self.meshdata,
                    smooth=True,
                    drawEdges=False,
                    shader="shaded",
                )
                self.gl_view.addItem(self.mesh_item)

                # Do NOT force show() here; PerceptionLab may embed via get_custom_widget().
            except Exception as e:
                print("[HolographicCortexViewerNode] OpenGL init failed, using 2D fallback:", e)
                self.gl_view = None
                self.mesh_item = None
                self.meshdata = None

    def _load_fsaverage_surface(self):
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))  # .../mne_data/fsaverage
        subjects_dir = fs_dir.parent
        subject = "fsaverage"

        surf_lh = subjects_dir / subject / "surf" / f"lh.{self.surface_type}"
        surf_rh = subjects_dir / subject / "surf" / f"rh.{self.surface_type}"

        rr_lh, tris_lh = mne.read_surface(str(surf_lh))
        rr_rh, tris_rh = mne.read_surface(str(surf_rh))

        # Decimate (strongly recommended)
        rr_lh, tris_lh = _try_decimate(rr_lh, tris_lh, self.target_tris_per_hemi)
        rr_rh, tris_rh = _try_decimate(rr_rh, tris_rh, self.target_tris_per_hemi)

        # Offset RH face indices by LH vertex count
        tris_rh = tris_rh + rr_lh.shape[0]

        vertices = np.vstack([rr_lh, rr_rh]).astype(np.float32)
        faces = np.vstack([tris_lh, tris_rh]).astype(np.int32)

        print(f"Loaded fsaverage {self.surface_type}: vertices={vertices.shape[0]}, faces={faces.shape[0]}")
        return vertices, faces

    def _compute_uv(self, vertices: np.ndarray) -> np.ndarray:
        """
        Simple spherical UV:
        - center vertices
        - map lon/lat to [0,1]
        This gives a stable 2D texture even without OpenGL.
        """
        v = vertices.astype(np.float32)
        v = v - v.mean(axis=0, keepdims=True)
        r = np.linalg.norm(v, axis=1) + 1e-8
        x, y, z = v[:, 0] / r, v[:, 1] / r, v[:, 2] / r

        lon = np.arctan2(y, x)          # [-pi, pi]
        lat = np.arcsin(np.clip(z, -1, 1))  # [-pi/2, pi/2]

        u = (lon + np.pi) / (2 * np.pi)
        vv = (lat + (np.pi / 2)) / np.pi
        return np.stack([u, vv], axis=1).astype(np.float32)

    def _modes_to_vertex_colors(self, complex_modes: np.ndarray, coherence: float) -> np.ndarray:
        # field(v) = Σ mode_i * topo(v,i)
        field = (self.mode_topographies @ complex_modes.astype(np.complex64)).astype(np.complex64)

        mag = np.abs(field).astype(np.float32)
        ph = np.angle(field).astype(np.float32)

        mag_max = float(mag.max())
        if mag_max < 1e-8:
            # Not connected / all-zero modes -> show neutral purple-ish “alive” preview
            rgb = np.tile(np.array([[0.35, 0.15, 0.45]], dtype=np.float32), (self.n_vertices, 1))
            a = np.full((self.n_vertices, 1), float(np.clip(coherence, 0.2, 1.0)), dtype=np.float32)
            return np.concatenate([rgb, a], axis=1)

        mag = mag / (mag_max + 1e-8)

        # phase -> [0,1] -> colormap
        ph01 = (ph + np.pi) / (2.0 * np.pi)
        ph_u8 = (ph01 * 255.0).astype(np.uint8)

        cm = cv2.applyColorMap(ph_u8.reshape(-1, 1), cv2.COLORMAP_TWILIGHT_SHIFTED)
        rgb = cm.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB

        # brightness by magnitude
        rgb *= mag[:, None]

        # low-mode glow: make the “hum” visible
        if self.low_mode_glow:
            low_power = float(np.sum(np.abs(complex_modes[:3])))
            rgb[:, 2] = np.clip(rgb[:, 2] + 0.12 * low_power * mag, 0.0, 1.0)

        a = np.full((self.n_vertices, 1), float(np.clip(coherence, 0.05, 1.0)), dtype=np.float32)
        out = np.concatenate([rgb, a], axis=1).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    def _render_2d_fallback(self, vertex_colors_rgba: np.ndarray, size=(256, 256)) -> np.ndarray:
        """
        Create a 2D texture by splatting vertex colors into an equirectangular UV grid.
        Always returns a non-black image (unless all colors are black).
        """
        H, W = int(size[0]), int(size[1])
        img = np.zeros((H, W, 3), dtype=np.float32)
        cnt = np.zeros((H, W, 1), dtype=np.float32)

        uv = self._uv
        xs = np.clip((uv[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
        ys = np.clip(((1.0 - uv[:, 1]) * (H - 1)).astype(np.int32), 0, H - 1)

        rgb = vertex_colors_rgba[:, :3].astype(np.float32)
        # splat
        img[ys, xs] += rgb
        cnt[ys, xs] += 1.0

        img = img / (cnt + 1e-8)

        # light blur so it looks continuous
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # overlay label
        cv2.putText(img_u8, "HoloCortex (2D)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return img_u8

    def step(self):
        # Pull inputs
        cm = self.get_blended_input("complex_modes", "mean")
        if cm is None:
            complex_modes = np.zeros(self.n_modes, dtype=np.complex64)
        else:
            complex_modes = np.asarray(cm, dtype=np.complex64).ravel()
            if complex_modes.size < self.n_modes:
                tmp = np.zeros(self.n_modes, dtype=np.complex64)
                tmp[:complex_modes.size] = complex_modes
                complex_modes = tmp
            else:
                complex_modes = complex_modes[:self.n_modes]

        coh = self.get_blended_input("phase_coherence", "mean")
        if coh is None:
            coherence = 0.85
        else:
            coherence = float(np.clip(float(coh), 0.0, 1.0))

        # Compute per-vertex colors
        vcols = self._modes_to_vertex_colors(complex_modes, coherence)

        # Try OpenGL update (if available)
        if self.meshdata is not None and self.mesh_item is not None:
            try:
                self.meshdata.setVertexColors(vcols)
                self.mesh_item.setMeshData(meshdata=self.meshdata)
            except Exception as e:
                # Don’t die; just fall back to 2D
                print("[HolographicCortexViewerNode] OpenGL update failed, falling back to 2D:", e)

        # Update preview image cache (always)
        # Prefer GL render if it works; otherwise 2D fallback
        self._last_preview = self.get_display_image(fallback_colors=vcols)

    def get_custom_widget(self):
        # If PerceptionLab embeds custom widgets, you’ll get the 3D view.
        return self.gl_view

    def get_display_image(self, fallback_colors: np.ndarray | None = None):
        # Attempt GL screenshot
        if self.gl_view is not None:
            try:
                img = self.gl_view.renderToArray((256, 256))  # BGRA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                # If it rendered all-black, don’t trust it
                if img.mean() > 1.0:
                    return img
            except Exception:
                pass

        # 2D fallback texture (guaranteed)
        if fallback_colors is None:
            fallback_colors = np.ones((self.n_vertices, 4), dtype=np.float32) * 0.5
            fallback_colors[:, 3] = 1.0
        return self._render_2d_fallback(fallback_colors, size=(256, 256))

    def get_output(self, port_name):
        if port_name == "image_out":
            return self._last_preview
        return None
