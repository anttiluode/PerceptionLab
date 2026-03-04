"""
Standalone Chronotopic Radar Node
=================================
Antti Luode (PerceptionLab) | 2026

Self-standing Perception Lab node for the Deerskin Time-Folded Surface.
Includes native EDF loading, Theta/Alpha phase extraction, and logarithmic
amplification. No external input nodes required.
"""

import os
import numpy as np
import cv2
import traceback

try:
    import mne
    from scipy.signal import butter, filtfilt, hilbert
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    try:
        from PyQt6 import QtGui
    except ImportError:
        class MockQtGui:
            @staticmethod
            def QColor(*args): return None
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}

# ── 19-Channel 2D Coordinates ──
ELEC_COORDS_2D = {
    'FP1': (-0.3, 0.7),  'FP2': (0.3, 0.7),
    'F7':  (-0.7, 0.4),  'F3':  (-0.4, 0.4), 'FZ': (0.0, 0.4), 'F4': (0.4, 0.4), 'F8': (0.7, 0.4),
    'T3':  (-0.8, 0.0),  'C3':  (-0.4, 0.0), 'CZ': (0.0, 0.0), 'C4': (0.4, 0.0), 'T4': (0.8, 0.0),
    'T5':  (-0.7, -0.4), 'P3':  (-0.4, -0.4), 'PZ': (0.0, -0.4), 'P4': (0.4, -0.4), 'T6': (0.7, -0.4),
    'O1':  (-0.3, -0.7), 'O2':  (0.3, -0.7)
}

class StandaloneChronotopicRadarNode(BaseNode):
    NODE_CATEGORY = "Deerskin Architecture"
    NODE_TITLE = "Standalone Chronotopic Radar"
    NODE_COLOR = QtGui.QColor(245, 158, 11) # Amber

    def __init__(self):
        super().__init__()
        
        # Self-standing: No inputs required!
        self.inputs = {}
        
        self.outputs = {
            'radar_view': 'image',
        }
        
        # Configuration parameters
        self.edf_path = ""
        self._last_path = ""  # Used to track UI changes continuously
        self.mode = 1         # 0 = Moiré, 1 = Chronotopic
        self.amp_slider = 0.0 # -10 to 70 (Logarithmic multiplier)
        self.blur = 0.85      # Temporal decay (0.0 to 0.99)
        
        # Internal State
        self.res = 200
        self.n_ch = 19
        self.sfreq = 250
        self.window_size_s = 1.0
        
        self.raw_data = None
        self.theta_phase = None
        self.alpha_phase = None
        
        self.current_frame = 0
        self.accum_color = np.zeros((self.res * self.res, 4), dtype=np.float32)
        self.cached_image = np.zeros((self.res, self.res, 3), dtype=np.uint8)
        
        self.status_msg = "No EDF. Using Synthetic."
        self.status_color = (0, 165, 255) # Orange (BGR)
        
        if MNE_AVAILABLE:
            self.b_low, self.a_low = butter(3, 2 / (self.sfreq / 2), btype='low')
            
        self._init_geometry()

    def _init_geometry(self):
        x = np.linspace(-1.1, 1.1, self.res)
        y = np.linspace(-1.1, 1.1, self.res)
        xx, yy = np.meshgrid(x, y, indexing='ij') 
        self.grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
        self.mask = (self.grid_coords[:, 0]**2 + self.grid_coords[:, 1]**2) <= 1.0

        self.elec_weights = np.zeros((len(self.grid_coords), self.n_ch))
        for i, (name, coord) in enumerate(ELEC_COORDS_2D.items()):
            dists = np.linalg.norm(self.grid_coords - np.array(coord), axis=1)
            self.elec_weights[:, i] = 1.0 / (dists**3 + 0.05)
            
        self.elec_weights /= self.elec_weights.sum(axis=1, keepdims=True)
        self.elec_weights[~self.mask] = 0.0

    def get_config_options(self):
        return [
            ("EDF File Path", "edf_path", self.edf_path, 'string'),
            ("Mode (0=Moiré, 1=Chrono)", "mode", self.mode, 'int'),
            ("Log Amp (-10 to 70)", "amp_slider", self.amp_slider, 'float'),
            ("Temporal Blur (0-0.99)", "blur", self.blur, 'float'),
        ]
        
    def set_config_options(self, options):
        if isinstance(options, dict):
            if 'mode' in options and options['mode'] != self.mode:
                self.accum_color.fill(0)
            for k, v in options.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def _load_edf(self, path):
        if not MNE_AVAILABLE:
            self.status_msg = "Error: MNE/SciPy not installed."
            self.status_color = (0, 0, 255) # Red in BGR
            return

        if not os.path.exists(path):
            self.status_msg = "File not found."
            self.status_color = (0, 0, 255)
            self.raw_data = None
            return

        try:
            self.status_msg = "Loading EDF... (Wait)"
            self.status_color = (0, 255, 255) # Yellow
            
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.filter(1.0, 45.0, verbose=False)
            if raw.info['sfreq'] != 250:
                raw.resample(250, verbose=False)
                
            ch_names_upper = [c.upper().replace(' ', '').replace('-', '').replace('.', '') for c in raw.ch_names]
            
            mapped_indices = []
            mapped_names = []
            for target in ELEC_COORDS_2D.keys():
                for i, name in enumerate(ch_names_upper):
                    if target in name:
                        mapped_indices.append(i)
                        mapped_names.append(target)
                        break
                        
            if len(mapped_indices) == 0:
                self.status_msg = "No 10-20 channels found!"
                self.status_color = (0, 0, 255)
                self.raw_data = None
                return

            # If some channels are missing, we still try to run with what we have
            if len(mapped_indices) < self.n_ch:
                print(f"Warning: Only found {len(mapped_indices)}/19 channels. Pad with zeros for missing.")
                self.raw_data = np.zeros((self.n_ch, raw.n_times))
                found_data = raw.get_data()[mapped_indices, :]
                for i, idx in enumerate(mapped_indices):
                    self.raw_data[i, :] = found_data[i, :]
            else:
                self.raw_data = raw.get_data()[mapped_indices, :]
            
            # Precompute Theta and Alpha phases for the entire file (massive optimization)
            b_t, a_t = butter(4, [4/(self.sfreq/2), 8/(self.sfreq/2)], btype='band')
            theta_data = np.array([filtfilt(b_t, a_t, ch) for ch in self.raw_data])
            self.theta_phase = np.angle(hilbert(theta_data, axis=1))
            
            b_a, a_a = butter(4, [8/(self.sfreq/2), 13/(self.sfreq/2)], btype='band')
            alpha_data = np.array([filtfilt(b_a, a_a, ch) for ch in self.raw_data])
            self.alpha_phase = np.angle(hilbert(alpha_data, axis=1))
            
            self.current_frame = 0
            self.accum_color.fill(0)
            
            filename = os.path.basename(path)
            self.status_msg = f"LIVE: {filename}"
            self.status_color = (0, 255, 0) # Green in BGR
            
        except Exception as e:
            self.status_msg = "Load Error!"
            self.status_color = (0, 0, 255)
            self.raw_data = None
            print(f"EDF Load Error: {str(e)}")
            traceback.print_exc()

    def _fast_takens(self, signal, tau):
        n = len(signal) - 2 * tau
        if n < 5: return 0.0
        X = np.column_stack([signal[2*tau:], signal[tau:tau+n], signal[:n]])
        std = X.std(axis=0) + 1e-10
        X = (X - X.mean(axis=0)) / std
        _, sv, _ = np.linalg.svd(X, full_matrices=False)
        sv = sv / (sv.sum() + 1e-10)
        return -np.sum(sv * np.log2(sv + 1e-10))

    def step(self):
        # --- FIX: Track path changes continuously just like the old node ---
        current_path = str(self.edf_path).strip().strip('\"').strip('\'')
        if current_path != self._last_path:
            self._last_path = current_path
            if current_path != "":
                self._load_edf(current_path)
            else:
                self.raw_data = None
                self.status_msg = "No EDF. Using Synthetic."
                self.status_color = (0, 165, 255)
        # -------------------------------------------------------------------

        win_samples = int(self.window_size_s * self.sfreq)
        
        # 1. Fetch 1-Second Window (Real or Synthetic)
        if self.raw_data is not None:
            if self.current_frame + win_samples > self.raw_data.shape[1]:
                self.current_frame = 0 # Loop back to start
                
            window_data = self.raw_data[:, self.current_frame : self.current_frame + win_samples]
            window_theta = self.theta_phase[:, self.current_frame : self.current_frame + win_samples]
            window_alpha = self.alpha_phase[:, self.current_frame : self.current_frame + win_samples]
            
            # Advance frame by ~50ms per step to simulate real-time playback
            self.current_frame += int(self.sfreq * 0.05) 
            time_sec = self.current_frame / self.sfreq
        else:
            # Synthetic Fractal Backup if no EDF loaded
            t = np.linspace(0, 1.0, win_samples) + (self.current_frame * 0.05)
            window_data = np.zeros((self.n_ch, win_samples))
            window_theta = np.zeros((self.n_ch, win_samples))
            window_alpha = np.zeros((self.n_ch, win_samples))
            
            for i in range(self.n_ch):
                window_data[i] = np.sin(2 * np.pi * 1.5 * t + np.random.rand()*0.1) + 0.5 * np.sin(2 * np.pi * 10.0 * t + i*0.3)
                window_theta[i] = np.angle(np.exp(1j * (2 * np.pi * 6.0 * t + np.random.rand()*0.2)))
                window_alpha[i] = np.angle(np.exp(1j * (2 * np.pi * 10.0 * t + np.random.rand()*0.2)))
                
            self.current_frame += 1
            time_sec = self.current_frame * 0.05

        global_phase = np.angle(np.mean(np.exp(1j * window_theta), axis=0))
        amp = 10.0 ** (self.amp_slider / 10.0)

        # 2. Calculate Metrics
        target_color = np.zeros((len(self.grid_coords), 4), dtype=np.float32)
        
        if self.mode == 1:
            # === CHRONOTOPIC MODE ===
            comp_fast = np.zeros(self.n_ch)
            comp_med = np.zeros(self.n_ch)
            comp_slow = np.zeros(self.n_ch)
            
            for i in range(self.n_ch):
                comp_fast[i] = self._fast_takens(window_data[i], tau=2)  # Blue (Fast)
                comp_med[i]  = self._fast_takens(window_data[i], tau=6)  # Green (Medium)
                comp_slow[i] = self._fast_takens(window_data[i], tau=12) # Red (Slow)
                
            norm_fast = np.clip(((comp_fast - 1.0) / 1.5) * amp, 0, 1)
            norm_med  = np.clip(((comp_med - 1.0) / 1.5) * amp, 0, 1)
            norm_slow = np.clip(((comp_slow - 1.0) / 1.5) * amp, 0, 1)
            
            target_color[:, 2] = np.dot(self.elec_weights, norm_fast)  # Blue
            target_color[:, 1] = np.dot(self.elec_weights, norm_med)   # Green
            target_color[:, 0] = np.dot(self.elec_weights, norm_slow)  # Red
            
        else:
            # === MOIRÉ MODE ===
            wire_metrics = np.zeros(self.n_ch)
            outer_metrics = np.zeros(self.n_ch)
            
            for i in range(self.n_ch):
                wire_metrics[i] = self._fast_takens(window_data[i], tau=5)
                
                interference = np.sin(window_theta[i]) * np.sin(window_alpha[i])
                if MNE_AVAILABLE:
                    envelope = filtfilt(self.b_low, self.a_low, np.abs(interference))
                    outer_metrics[i] = np.mean(envelope) + 0.4 * (np.max(envelope) - np.min(envelope))
                else:
                    outer_metrics[i] = np.abs(np.mean(interference))
                
            wire_norm = np.clip(((wire_metrics - 1.0) / 1.5) * amp, 0, 1) 
            outer_norm = np.clip((outer_metrics * 2.0) * amp, 0, 1)

            wire_pixel = np.dot(self.elec_weights, wire_norm)
            outer_pixel = np.dot(self.elec_weights, outer_norm)
            
            target_color[:, 0] += wire_pixel * 0.8
            target_color[:, 1] += wire_pixel * 0.4
            
            target_color[:, 0] += outer_pixel * 0.2
            target_color[:, 1] += outer_pixel * 0.9
            target_color[:, 2] += outer_pixel * 0.8
                
            sparks = np.clip(((wire_pixel * outer_pixel) ** 3) * 3.0, 0, 1)
            target_color[:, 0] += sparks
            target_color[:, 1] += sparks
            target_color[:, 2] += sparks

        target_color[:, 3] = 1.0
        
        # 3. Apply Temporal Blur
        self.accum_color = np.maximum(target_color, self.accum_color * self.blur)
        
        display_color = np.clip(self.accum_color, 0, 1)
        display_color[~self.mask] = 0.0
        
        # Convert to 8-bit BGR for CV2/PerceptionLab viewing
        img_bgr = (display_color[:, :3] * 255).astype(np.uint8)
        img_bgr = img_bgr.reshape((self.res, self.res, 3))
        
        # 4. Draw UI Overlays natively on the image
        cv2.circle(img_bgr, (self.res//2, self.res//2), int(self.res//2 * 0.95), (82, 61, 52), 2)
        
        amp_text = f"Amp: {amp:.1f}x" if amp < 1000 else f"Amp: {amp/1000:.1f}kx"
        cv2.putText(img_bgr, amp_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_bgr, f"Time: {time_sec:.1f}s", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_bgr, self.status_msg, (10, self.res - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.status_color, 1)
        
        self.cached_image = img_bgr

    def get_output(self, name):
        if name == 'radar_view':
            return self.cached_image
        return None

    def get_display_image(self):
        return self.cached_image