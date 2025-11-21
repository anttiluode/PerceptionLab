# holographicreconstruction.py
"""
Holographic Reconstruction Node (patched)
-----------------------------------------
Performs an Optical Fourier Transform (2D FFT) on an interference/hologram
and extracts a magnitude (reconstruction) and phase map.

Fixes applied:
- Forces input arrays to float32 and normalizes them to 0..1 to avoid CV_64F errors.
- Uses np.ptp for NumPy 2.0 compatibility.
- Ensures outputs are float32 0..1 arrays and display conversion uses uint8.
- Adds safe guards for unexpected shapes / dtypes.
"""

import numpy as np
import cv2

# Host imports (safe retrieval from __main__ as the host provides BaseNode/QtGui)
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class HolographicReconstructionNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(100, 255, 200)  # Reconstructed Green

    def __init__(self, scale_factor=10.0):
        super().__init__()
        self.node_title = "Holographic Reconstruction"

        self.inputs = {
            'hologram': 'image'
        }

        self.outputs = {
            'reconstruction': 'image',  # The Magnitude (What is there?)
            'phase_content': 'image'    # The Phase (Where is it?)
        }

        self.scale_factor = float(scale_factor)
        # storage for visualizable images (float32 0..1)
        self.mag_img = np.zeros((128, 128), dtype=np.float32)
        self.phase_img = np.zeros((128, 128), dtype=np.float32)

    # -------------------------
    # core processing
    # -------------------------
    def step(self):
        # 1. Get the Hologram (Interference Pattern)
        hologram = self.get_blended_input('hologram', 'mean')
        if hologram is None:
            return

        # --- Ensure proper dtype and normalization to avoid CV_64F / cvtColor errors ---
        # Convert to numpy array if some host gives something array-like
        if not isinstance(hologram, np.ndarray):
            try:
                hologram = np.array(hologram)
            except Exception:
                # Can't convert — bail out gracefully
                return

        # Force float32 to satisfy OpenCV color operations and reduce memory for FFT
        hologram = hologram.astype(np.float32, copy=False)

        # If image has multiple channels, ensure shape is (H, W, C). If single channel, keep as-is.
        if hologram.ndim == 3 and hologram.shape[2] in (3, 4):
            # Normalize to 0..1 if values appear outside that range
            maxv = float(hologram.max()) if hologram.size else 0.0
            minv = float(hologram.min()) if hologram.size else 0.0
            if maxv > 1.0 or minv < 0.0:
                # Scale to 0..1
                hologram = (hologram - minv) / (maxv - minv + 1e-12)

            # Convert BGR/RGB to grayscale using OpenCV which supports float32 images
            try:
                gray = cv2.cvtColor(hologram, cv2.COLOR_BGR2GRAY)
            except Exception:
                # As a fallback, compute luminosity manually (safe)
                # assume channel order is BGR or RGB, use simple average-lum
                gray = np.mean(hologram[..., :3], axis=2)
        else:
            # Single-channel case: normalize to 0..1
            gray = hologram
            maxv = float(gray.max()) if gray.size else 0.0
            minv = float(gray.min()) if gray.size else 0.0
            if maxv > 1.0 or minv < 0.0:
                gray = (gray - minv) / (maxv - minv + 1e-12)

        # Ensure gray is float32 and finite
        gray = gray.astype(np.float32, copy=False)
        gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)

        # Optionally apply a small windowing to reduce spectral leakage (comment/uncomment as needed)
        # window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
        # gray = gray * window

        # 2. The Optical Transform (2D FFT)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)  # Move zero freq to center

        # 3. Extract Magnitude (The Virtual Image)
        magnitude = 20.0 * np.log(np.abs(f_shift) + 1e-9)  # log scale
        # Normalize magnitude to 0..1 using np.ptp for NumPy 2.0 safety
        mag_min = float(np.min(magnitude))
        mag_ptp = float(np.ptp(magnitude)) + 1e-12
        mag_norm = (magnitude - mag_min) / mag_ptp
        self.mag_img = mag_norm.astype(np.float32, copy=False)

        # 4. Extract Phase
        phase = np.angle(f_shift)  # range -pi..pi
        self.phase_img = ((phase + np.pi) / (2.0 * np.pi)).astype(np.float32, copy=False)

        # Resize outputs to reasonable display size if very small/large (optional)
        target_size = (128, 128)
        if self.mag_img.shape != target_size:
            try:
                self.mag_img = cv2.resize(self.mag_img, target_size, interpolation=cv2.INTER_LINEAR)
            except Exception:
                self.mag_img = cv2.resize(np.clip(self.mag_img, 0.0, 1.0), target_size, interpolation=cv2.INTER_LINEAR)
        if self.phase_img.shape != target_size:
            try:
                self.phase_img = cv2.resize(self.phase_img, target_size, interpolation=cv2.INTER_LINEAR)
            except Exception:
                self.phase_img = cv2.resize(np.clip(self.phase_img, 0.0, 1.0), target_size, interpolation=cv2.INTER_LINEAR)

    # -------------------------
    # host outputs
    # -------------------------
    def get_output(self, port_name):
        if port_name == 'reconstruction':
            # float32 0..1
            return self.mag_img
        elif port_name == 'phase_content':
            return self.phase_img
        return None

    # -------------------------
    # For UI display (QImage)
    # -------------------------
    def get_display_image(self):
        # Build a left/right visualization: magnitude | phase (both colorized)
        h, w = 128, 256  # height, width
        out = np.zeros((h, w, 3), dtype=np.uint8)

        # Magnitude: apply inferno colormap
        mag_u8 = (np.clip(self.mag_img, 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            mag_color = cv2.applyColorMap(mag_u8, cv2.COLORMAP_INFERNO)
        except Exception:
            # fallback: replicate grayscale to 3 channels
            mag_color = np.stack([mag_u8, mag_u8, mag_u8], axis=2)

        # Phase: apply twilight/other colormap
        phase_u8 = (np.clip(self.phase_img, 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            phase_color = cv2.applyColorMap(phase_u8, cv2.COLORMAP_TWILIGHT)
        except Exception:
            phase_color = np.stack([phase_u8, phase_u8, phase_u8], axis=2)

        # Place into output canvas (left: mag, right: phase)
        out[:, :128] = cv2.resize(mag_color, (128, 128), interpolation=cv2.INTER_NEAREST)
        out[:, 128:] = cv2.resize(phase_color, (128, 128), interpolation=cv2.INTER_NEAREST)

        # Add labels (white)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "VIRTUAL IMAGE", (6, 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "PHASE FIELD", (138, 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert to QImage for host display
        try:
            qimg = QtGui.QImage(out.data, w, h, out.strides[0], QtGui.QImage.Format.Format_RGB888)
            return qimg
        except Exception:
            # If QImage construction fails for some host, return raw array (some hosts accept this)
            return out

    def get_config_options(self):
        return [
            ("Scale Factor", "scale_factor", self.scale_factor, None)
        ]
