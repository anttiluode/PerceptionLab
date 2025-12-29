import os
import re
import numpy as np
import cv2

# --- HOST IMPORT BLOCK (PerceptionLab pattern) ---
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
# ------------------------------------------------


try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    mne = None
    MNE_AVAILABLE = False


class CorticalSheetNode(BaseNode):
    """
    Cortical Sheet Source (reliable PerceptionLab node)

    - Robust EDF loading (auto-load on config change via step check)
    - Robust channel name cleaning + mapping to 2D sheet
    - Stable QImage display (detached copy to avoid black screens)
    - Simple sheet dynamics: Izhikevich + Laplacian coupling + blurred EEG injection
    """

    NODE_NAME = "Cortical Sheet Source"
    NODE_TITLE = "Cortical Sheet"  # Added for framework compatibility
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(180, 50, 50) if QtGui else None

    def __init__(self):
        super().__init__()

        # --- IO ---
        self.inputs = {
            "coupling": "signal",      # diffusion strength
            "excitability": "signal",  # input gain
            "speed": "signal",         # samples per step
            "reset": "signal",         # pulse
        }
        self.outputs = {
            "cortex_view": "image",
            "lfp_signal": "spectrum",
            "sensor_data": "spectrum",
        }

        # --- Config ---
        self.edf_path = ""
        self._last_path = ""       # Backbone for robust reloading
        self.status_msg = "No file"
        self.is_loaded = False

        # --- Sheet params ---
        self.grid_size = 128
        self.dt = 0.5  # "simulation" dt, not EDF dt

        # Izhikevich (regular spiking-ish)
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0

        self.v = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * self.c
        self.u = self.v * self.b

        self.data_cache = None


        self.current_idx = 0

        # electrode mapping
        self.electrode_coords = []   # list of (r, c) on grid
        self.electrode_indices = []  # list of channel indices in data_cache

        # --- Display ---
        self.display_image = None
        self.last_stats = {"input_max_uV": 0.0, "mean_v_mV": float(self.c), "mapped": 0}

        # Basic 10–20-ish map in normalized sheet coords (0..1)
        self.standard_map = {
            "FP1": (0.30, 0.10), "FP2": (0.70, 0.10),
            "F7":  (0.10, 0.30), "F3":  (0.30, 0.30), "FZ": (0.50, 0.25), "F4": (0.70, 0.30), "F8": (0.90, 0.30),
            "T7":  (0.10, 0.50), "C3":  (0.30, 0.50), "CZ": (0.50, 0.50), "C4": (0.70, 0.50), "T8": (0.90, 0.50),
            "P7":  (0.10, 0.70), "P3":  (0.30, 0.70), "PZ": (0.50, 0.75), "P4": (0.70, 0.70), "P8": (0.90, 0.70),
            "O1":  (0.35, 0.90), "OZ":  (0.50, 0.90), "O2": (0.65, 0.90),
        }

        # create an initial status frame so the node face isn't blank
        self._update_display_buffer(force_status=True)

    # ---------- PerceptionLab config hooks ----------
    def get_config_options(self):
        return [("EDF File Path", "edf_path", self.edf_path, None)]

    def set_config_options(self, options):
        # Match GrammarGeometry: assume dict, set attrs if present
        # This is more reliable than custom parsing logic
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # We do NOT call load_edf() here. We let step() handle it via _maybe_reload()

    def _maybe_reload(self):
        """Checks if path changed and triggers load. Called every step."""
        path = str(self.edf_path or "").strip().strip('"').strip("'")
        
        # Simple sanitation
        path = path.replace("\\", "/")
        path = "".join(ch for ch in path if ord(ch) >= 32)
        
        if path != self._last_path:
            self._last_path = path
            self.edf_path = path # Update internal path to sanitized version
            if path:
                self.load_edf()
            else:
                self.is_loaded = False
                self.status_msg = "No file"
                self._update_display_buffer(force_status=True)

    # ---------- Small compatibility layer for inputs ----------
    def _read_input_scalar(self, name, default=0.0, mode="mean"):
        """
        Tries common BaseNode APIs, otherwise returns default.
        """
        # PerceptionLab often has get_blended_input(name, mode)
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, mode)
                if val is None:
                    return default
                return float(val)
            except Exception:
                return default

        # Some nodes use get_input(name)
        fn = getattr(self, "get_input", None)
        if callable(fn):
            try:
                val = fn(name)
                if val is None:
                    return default
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                    return float(np.mean(val))
                return float(val)
            except Exception:
                return default

        return default
    
    # ---------- Output Compatibility ----------
    def get_output(self, port_name):
        """Explicit getter for frameworks that don't read self.outputs directly."""
        if port_name == "cortex_view":
            return self.display_image
        if port_name == "lfp_signal":
            return self.outputs.get("lfp_signal", None)
        if port_name == "sensor_data":
            return self.outputs.get("sensor_data", None)
        return None

    # ---------- EDF loading ----------
    def load_edf(self):
        if not MNE_AVAILABLE:
            self.status_msg = "MNE not installed"
            self.is_loaded = False
            self._update_display_buffer(force_status=True)
            print("[CorticalSheet] MNE not available. Install: pip install mne")
            return False

        if not self.edf_path or not os.path.exists(self.edf_path):
            self.status_msg = "File not found"
            self.is_loaded = False
            self._update_display_buffer(force_status=True)
            print(f"[CorticalSheet] File not found: {self.edf_path}")
            return False

        try:
            print(f"[CorticalSheet] Loading EDF: {self.edf_path}")
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)

            # Prefer EEG channels only (if present)
            try:
                raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, emg=False, misc=False, stim=False)
            except Exception:
                pass

            # Optional: downsample for speed (keeps it snappy like your “fast” runs)
            try:
                if raw.info["sfreq"] > 256:
                    raw.resample(256, npad="auto", verbose=False)
            except Exception:
                pass

            self.raw = raw
            self.sfreq = float(raw.info["sfreq"])
            self.data_cache = raw.get_data()  # uV scale depends on EDF; we treat as “uV-ish”
            self.current_idx = 0

            self._map_electrodes()

            self.is_loaded = True
            self.status_msg = f"Loaded {os.path.basename(self.edf_path)} | sf={self.sfreq:.1f}Hz | mapped={len(self.electrode_coords)}"
            print(f"[CorticalSheet] OK: {self.status_msg}")

            self._update_display_buffer(force_status=False)
            return True

        except Exception as e:
            self.is_loaded = False
            self.status_msg = f"Load error: {str(e)[:60]}"
            self._update_display_buffer(force_status=True)
            print("[CorticalSheet] FAILED:", e)
            return False

    def _clean_ch_name(self, name: str) -> str:
        n = name.upper()
        n = n.replace("EEG", "")
        n = n.replace(" ", "")
        n = n.replace("-REF", "")
        n = n.replace("REF", "")
        # strip common suffixes like "-A1", "-A2", etc.
        n = re.sub(r"[-_](A1|A2|M1|M2|LE|RE)$", "", n)
        # keep only alnum
        n = re.sub(r"[^A-Z0-9]", "", n)
        return n

    def _map_electrodes(self):
        self.electrode_coords = []
        self.electrode_indices = []

        if self.raw is None:
            return

        names = [self._clean_ch_name(ch) for ch in self.raw.ch_names]
        margin = 10
        scale = self.grid_size - 2 * margin
        mapped = 0

        for i, cn in enumerate(names):
            pos = None
            # exact
            if cn in self.standard_map:
                pos = self.standard_map[cn]
            else:
                # partial match (handles stuff like "FP1A2" or "EEGFP1")
                for key, p in self.standard_map.items():
                    if key in cn:
                        pos = p
                        break

            if pos is None:
                continue

            c = int(pos[0] * scale + margin)
            r = int(pos[1] * scale + margin)
            r = int(np.clip(r, 0, self.grid_size - 1))
            c = int(np.clip(c, 0, self.grid_size - 1))

            self.electrode_coords.append((r, c))
            self.electrode_indices.append(i)
            mapped += 1

        self.last_stats["mapped"] = mapped
        print(f"[CorticalSheet] Mapped electrodes: {mapped}/{len(names)}")

    # ---------- Main simulation tick ----------
    def step(self):
        # 1. ALWAYS check for config changes first (Robust Pattern)
        self._maybe_reload()

        # If still not loaded, just update status display and exit
        if not self.is_loaded or self.data_cache is None:
            self._update_display_buffer(force_status=True)
            self.outputs["cortex_view"] = self.display_image
            self.outputs["lfp_signal"] = np.array([float(np.mean(self.v))], dtype=np.float32)
            self.outputs["sensor_data"] = np.zeros((0,), dtype=np.float32)
            return

        # Controls (safe defaults)
        coupling = self._read_input_scalar("coupling", default=0.15, mode="mean")
        gain = self._read_input_scalar("excitability", default=1.0, mode="mean")
        speed = self._read_input_scalar("speed", default=1.0, mode="mean")
        reset = self._read_input_scalar("reset", default=0.0, mode="pulse")

        coupling = float(np.clip(coupling, 0.0, 2.0))
        gain = float(np.clip(gain, 0.0, 500.0))

        steps = int(np.clip(round(speed), 1, 20))

        if reset > 0.5:
            self.v[:] = self.c
            self.u[:] = self.v * self.b
            self.current_idx = 0

        n_samples = self.data_cache.shape[1]
        vec_last = None
        input_max = 0.0

        for _ in range(steps):
            if self.current_idx >= n_samples:
                self.current_idx = 0

            # Build sparse input from electrode samples
            I = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

            if self.electrode_indices:
                vec = self.data_cache[self.electrode_indices, self.current_idx].astype(np.float32, copy=False)
                vec_last = vec

                # Track max for HUD
                if vec.size > 0:
                    input_max = float(np.max(np.abs(vec)))

                # place into grid
                for (r, c), vch in zip(self.electrode_coords, vec):
                    I[r, c] += vch

                # blur so it behaves like “field injection”, not pixel dots
                I = cv2.GaussianBlur(I, (15, 15), 5) * (gain * 0.02)


            # Laplacian coupling on v (diffusion / sheet conduction)
            v = self.v
            u = self.u

            v_pad = np.pad(v, 1, mode="edge")
            lap = (
                v_pad[0:-2, 1:-1] + v_pad[2:, 1:-1] +
                v_pad[1:-1, 0:-2] + v_pad[1:-1, 2:] -
                4.0 * v
            ).astype(np.float32, copy=False)

            # Izhikevich update
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I + (lap * coupling))
            du = self.a * (self.b * v - u)

            v = v + dv * self.dt
            u = u + du * self.dt

            spk = v >= 30.0
            v[spk] = self.c
            u[spk] = u[spk] + self.d

            self.v = v
            self.u = u

            self.current_idx += 1

        # Outputs
        self.last_stats["input_max_uV"] = input_max
        self.last_stats["mean_v_mV"] = float(np.mean(self.v))

        self.outputs["lfp_signal"] = np.array([self.last_stats["mean_v_mV"]], dtype=np.float32)
        self.outputs["sensor_data"] = vec_last if vec_last is not None else np.zeros((0,), dtype=np.float32)

        self._update_display_buffer(force_status=False)
        self.outputs["cortex_view"] = self.display_image

    def get_display_image(self):
        return self.display_image

    # ---------- Display ----------
    def _update_display_buffer(self, force_status: bool = False):
        h, w = 512, 512

        if (not self.is_loaded) or force_status:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            msg = self.status_msg or "No file"
            cv2.putText(img, "CORTICAL SHEET", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
            cv2.putText(img, msg, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 160, 255), 2)
            if not MNE_AVAILABLE:
                cv2.putText(img, "Install mne: pip install mne", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 160, 255), 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if QtGui:
                qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
                self.display_image = qimg
            return

        # Render membrane potential heatmap
        disp = np.clip(self.v, -90.0, 40.0)
        norm = ((disp + 90.0) / 130.0 * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        img = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)

        # Electrodes
        scale_r = h / self.grid_size
        scale_c = w / self.grid_size
        for r, c in self.electrode_coords:
            center = (int(c * scale_c), int(r * scale_r))
            cv2.circle(img, center, 4, (0, 255, 0), -1)

        # HUD
        hud = (255, 255, 255)
        cv2.putText(img, f"Sample: {self.current_idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud, 1)
        cv2.putText(img, f"InputMax: {self.last_stats['input_max_uV']:.2f}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 1)
        cv2.putText(img, f"MeanV: {self.last_stats['mean_v_mV']:.2f} mV", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 1)
        cv2.putText(img, f"Mapped: {self.last_stats['mapped']}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if QtGui:
            # IMPORTANT: .copy() detaches from numpy buffer so it won't go black later
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg