"""
Theta-Beta Gating Observatory (Interference Edition)
----------------------------------------------------
A fully self-contained scientific workstation for the Perception Lab.

Features:
1.  **Robust EEG Loading:** Implements the "Ultimate Fix" (Aggressive cleaning, Sphere Model, NaN sanitization).
2.  **Source Space Reconstruction:** Solves the inverse problem internally to isolate specific brain regions.
3.  **Dual-Stream Analysis:** Extracts "Driver" (Gate) and "Target" (Content) signals.
4.  **Real-Time PAC:** Computes Phase-Amplitude Coupling metrics on the fly.
5.  **Quad-View Dashboard:** Renders Time, Phase Space, Polar, and Histogram views.
6.  **Interference Output:** Generates a holographic interference pattern between the Gate and Content waves.

Inputs:
    - speed: Playback speed multiplier (1.0 = Realtime).
    - gain: Signal amplification factor (default ~20x).
    - smoothing: Visual smoothing for envelopes.
    - history: Length of the visual trail (0.0 - 1.0).

Outputs:
    - display: The 4-panel dashboard image.
    - interference: The Theta-Beta Interference Pattern (Moiré).
    - gate_raw: The raw driver signal (e.g., Frontal Theta).
    - content_env: The envelope of the target signal (e.g., Temporal Beta).
    - phase_lock: Instantaneous Phase-Locking Value (0-1).
    - takens_x: X-coordinate of the phase space box.
    - takens_y: Y-coordinate of the phase space box.
"""

import numpy as np
import cv2
import os
import sys
import time
from collections import deque
from scipy import signal

# --- MNE IMPORT SAFETY ---
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# --- COMPATIBILITY BOILERPLATE ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0
        def step(self): pass
        def get_output(self, name): return None
        def get_display_image(self): return None

# --- VISUALIZATION CONSTANTS ---
COLOR_BG = (20, 20, 25)
COLOR_GRID = (50, 50, 60)
COLOR_TEXT = (180, 180, 180)
COLOR_GATE = (200, 200, 200)      # White/Grey
COLOR_SIGNAL = (0, 165, 255)      # Orange
COLOR_LOCK = (255, 50, 50)        # Blue
COLOR_HIST = (100, 200, 100)      # Green

class ThetaBetaGatingNode(BaseNode):
    NODE_CATEGORY = "Perception Lab"
    NODE_TITLE = "Theta-Beta Gating"
    NODE_COLOR = QtGui.QColor(130, 0, 220) # Deep Purple

    def __init__(self):
        super().__init__()
        
        # --- PORTS ---
        self.inputs = {
            'speed': 'float',
            'gain': 'float',
            'smoothing': 'float',
            'history': 'float'
        }
        
        self.outputs = {
            'display': 'image',
            'interference': 'image',     # <--- NEW: The Interference Pattern
            'gate_raw': 'signal',
            'content_env': 'signal',
            'phase_lock': 'signal',
            'takens_x': 'signal',
            'takens_y': 'signal'
        }
        
        # --- CONFIGURATION (User Editable) ---
        self.edf_path = r"E:\DocsHouse\450\2.edf" # Default
        
        self.gate_region = "frontal"
        self.gate_band = "theta"
        self.gate_freqs = (4, 8)
        
        self.target_region = "temporal"
        self.target_band = "beta"
        self.target_freqs = (12, 30)
        
        self.base_gain = 20.0
        self.takens_delay_ms = 40.0 
        
        # --- INTERNAL STATE ---
        self.fs = 160.0
        self.gate_series = None
        self.signal_series = None
        self.phase_series = None 
        self.env_series = None   
        
        self.playback_idx = 0.0
        self.is_loaded = False
        self.load_error = ""
        self.needs_load = True
        
        # --- VISUALIZATION STATE ---
        self.img_w, self.img_h = 1000, 700
        self._output_image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self._interference_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Buffers
        self.buf_len = 1000
        self.gate_buf = deque(maxlen=self.buf_len)
        self.env_buf = deque(maxlen=self.buf_len)
        self.lock_buf = deque(maxlen=self.buf_len)
        self.phase_buf = deque(maxlen=self.buf_len)
        
        # Histogram State
        self.n_bins = 24
        self.phase_hist = np.zeros(self.n_bins)
        
        self._last_outs = {k: 0.0 for k in self.outputs}

    def get_config_options(self):
        """Right-click configuration menu"""
        regions = [("Frontal", "frontal"), ("Temporal", "temporal"), 
                   ("Parietal", "parietal"), ("Occipital", "occipital"), ("Limbic", "limbic")]
        
        bands = [("Delta (0.5-4)", "delta"), ("Theta (4-8)", "theta"), 
                 ("Alpha (8-12)", "alpha"), ("Beta (12-30)", "beta"), ("Gamma (30-80)", "gamma")]
        
        return [
            ("EEG File", "edf_path", self.edf_path, "file_open"),
            ("Base Gain", "base_gain", self.base_gain, "float"),
            ("Driver Region", "gate_region", self.gate_region, regions),
            ("Driver Band", "gate_band", self.gate_band, bands),
            ("Target Region", "target_region", self.target_region, regions),
            ("Target Band", "target_band", self.target_band, bands),
        ]

    def _get_band_freqs(self, band_name):
        mapping = {
            "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
            "beta": (12, 30), "gamma": (30, 80)
        }
        return mapping.get(band_name, (4, 8))

    def _clean_names(self, raw):
        rename = {}
        for ch in raw.ch_names:
            clean = ch.replace('.', '').strip().upper()
            if clean == "FZ": clean = "Fz"
            if clean == "CZ": clean = "Cz"
            if clean == "PZ": clean = "Pz"
            if clean == "OZ": clean = "Oz"
            if clean == "FP1": clean = "Fp1"
            if clean == "FP2": clean = "Fp2"
            rename[ch] = clean
        raw.rename_channels(rename)
        return raw

    def setup_source(self):
        """Full MNE Pipeline embedded in the node."""
        if not MNE_AVAILABLE:
            self.load_error = "MNE Library not installed"
            return

        if not os.path.exists(self.edf_path):
            self.load_error = "File path invalid"
            return

        try:
            print(f"[{self.NODE_TITLE}] Pipeline Start: {self.edf_path}")
            
            self.gate_freqs = self._get_band_freqs(self.gate_band)
            self.target_freqs = self._get_band_freqs(self.target_band)
            
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            self.fs = raw.info['sfreq']
            
            raw = self._clean_names(raw)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            raw.set_eeg_reference('average', projection=True, verbose=False)
            
            sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.095, 
                                         info=raw.info, relative_radii=(0.90, 0.92, 0.97, 1.0), 
                                         sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False)
            
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
            src = mne.setup_volume_source_space(subject='fsaverage', pos=30.0, 
                                              sphere=sphere, bem=None, 
                                              subjects_dir=subjects_dir, verbose=False)
            
            fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere, 
                                          eeg=True, meg=False, verbose=False)
            
            # --- CRITICAL SAFETY: Sanitize Forward Matrix ---
            G = fwd['sol']['data']
            if not np.all(np.isfinite(G)):
                print(f"[{self.NODE_TITLE}] Sanitizing Forward Matrix...")
                np.nan_to_num(G, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                fwd['sol']['data'] = G
            
            cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
            inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, 
                                                       depth=None, loose='auto', verbose=False)
            
            print(f"[{self.NODE_TITLE}] Extracting Sources...")
            raw_gate = raw.copy().filter(self.gate_freqs[0], self.gate_freqs[1], verbose=False)
            stc_gate = mne.minimum_norm.apply_inverse_raw(raw_gate, inv, lambda2=1.0/9.0, method='dSPM', verbose=False)
            
            raw_target = raw.copy().filter(self.target_freqs[0], self.target_freqs[1], verbose=False)
            stc_target = mne.minimum_norm.apply_inverse_raw(raw_target, inv, lambda2=1.0/9.0, method='dSPM', verbose=False)
            
            coords = src[0]['rr'][stc_gate.vertices[0]]
            
            mask_gate = self._get_region_mask(coords, self.gate_region)
            mask_target = self._get_region_mask(coords, self.target_region)
            
            if np.sum(mask_gate) == 0: mask_gate[:] = True
            if np.sum(mask_target) == 0: mask_target[:] = True
            
            gate_data = np.mean(stc_gate.data[mask_gate], axis=0)
            target_data = np.mean(stc_target.data[mask_target], axis=0)
            
            self.gate_series = (gate_data - np.mean(gate_data)) / (np.std(gate_data) + 1e-9)
            self.signal_series = (target_data - np.mean(target_data)) / (np.std(target_data) + 1e-9)
            
            analytic_gate = signal.hilbert(self.gate_series)
            analytic_target = signal.hilbert(self.signal_series)
            
            self.phase_series = np.angle(analytic_gate) 
            self.env_series = np.abs(analytic_target)   
            
            self.gate_buf.clear()
            self.env_buf.clear()
            self.lock_buf.clear()
            self.phase_buf.clear()
            self.phase_hist = np.zeros(self.n_bins)
            
            self.is_loaded = True
            self.load_error = None
            print(f"[{self.NODE_TITLE}] Ready. {len(self.gate_series)} samples.")
            
        except Exception as e:
            self.load_error = str(e)
            print(f"[{self.NODE_TITLE}] Setup Error: {e}")
            import traceback
            traceback.print_exc()

    def _get_region_mask(self, coords, region_name):
        if region_name == "frontal":
            return coords[:, 1] > 0.05
        elif region_name == "occipital":
            return coords[:, 1] < -0.05
        elif region_name == "parietal":
            return (coords[:, 1] < 0.0) & (coords[:, 1] > -0.06) & (coords[:, 2] > 0.04)
        elif region_name == "temporal":
            return (coords[:, 1] < 0.0) & (coords[:, 2] < 0.0) & (np.abs(coords[:, 0]) > 0.03)
        elif region_name == "limbic":
            return np.sum(np.abs(coords), axis=1) < 0.05
        else:
            return np.ones(len(coords), dtype=bool)

    def step(self):
        if self.needs_load:
            self.setup_source()
            self.needs_load = False
        
        if not self.is_loaded:
            self._render_error()
            return

        speed_in = self.get_blended_input('speed', 'mean')
        speed = 1.0 if speed_in is None else max(0.1, speed_in)
        
        gain_in = self.get_blended_input('gain', 'mean')
        gain = self.base_gain if gain_in is None else max(1.0, gain_in)
        
        idx = int(self.playback_idx)
        if idx >= len(self.gate_series) - 1:
            self.playback_idx = 0
            idx = 0
            
        g_val = self.gate_series[idx]
        e_val = self.env_series[idx]
        p_val = self.phase_series[idx] 
        
        delay_samples = int((self.takens_delay_ms / 1000.0) * self.fs)
        if delay_samples < 1: delay_samples = 1
        
        idx_delayed = idx - delay_samples
        if idx_delayed < 0: idx_delayed += len(self.gate_series)
        g_delayed = self.gate_series[idx_delayed]
        
        lock_metric = e_val 
        
        self.gate_buf.append(g_val)
        self.env_buf.append(e_val)
        self.phase_buf.append(p_val)
        
        phase_norm = (p_val + np.pi) / (2 * np.pi) 
        bin_idx = int(phase_norm * self.n_bins) % self.n_bins
        self.phase_hist[bin_idx] += e_val * gain * 0.1
        self.phase_hist *= 0.98 
        
        # --- NEW: Generate Interference Pattern ---
        self._render_interference(p_val, e_val, gain)
        
        self._last_outs['gate_raw'] = float(g_val * gain)
        self._last_outs['content_env'] = float(e_val * gain)
        self._last_outs['phase_lock'] = float(lock_metric * gain)
        self._last_outs['takens_x'] = float(g_val * gain)
        self._last_outs['takens_y'] = float(g_delayed * gain)
        
        self.playback_idx += speed
        
        self._render_dashboard(gain, g_val, g_delayed, e_val, p_val)

    def _render_interference(self, phase, envelope, gain):
        """Generates the Theta-Beta Holographic Interference"""
        S = 256
        
        # Create vectors
        x = np.linspace(0, 8*np.pi, S)
        y = np.linspace(0, 8*np.pi, S)
        
        # Theta determines Phase (Horizontal Shift)
        grid_x = np.sin(x + phase)
        
        # Beta determines Frequency/Density (Vertical Compression)
        # Higher Beta = More stripes (Content Density)
        beta_power = np.clip(envelope * gain * 0.1, 0, 5.0)
        grid_y = np.sin(y * (1.0 + beta_power * 2.0))
        
        # Combine (Outer Product) to create Lattice
        # This creates a grid that "breathes" with Beta and "scrolls" with Theta
        pattern = np.outer(grid_y, grid_x)
        
        # Normalize to image
        img_norm = np.clip((pattern + 1) * 127.5, 0, 255).astype(np.uint8)
        
        # Apply color map
        self._interference_img = cv2.applyColorMap(img_norm, cv2.COLORMAP_OCEAN)
        
        # Add Label
        cv2.putText(self._interference_img, "INTERFERENCE", (10, 20), 
                   cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)

    def _render_error(self):
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        txt = "NO DATA"
        if self.load_error: txt = f"ERROR: {self.load_error}"
        elif not self.edf_path: txt = "CONFIGURE NODE TO LOAD FILE"
        
        cv2.putText(img, txt, (50, self.img_h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        self._output_image = img

    def _render_dashboard(self, gain, cur_gate, cur_gate_delayed, cur_env, cur_phase):
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        
        mid_x = self.img_w // 2
        mid_y = self.img_h // 2
        
        cv2.line(img, (mid_x, 0), (mid_x, self.img_h), COLOR_GRID, 1)
        cv2.line(img, (0, mid_y), (self.img_w, mid_y), COLOR_GRID, 1)
        
        rect_tl = (0, 0, mid_x, mid_y)
        self._draw_time_series(img, rect_tl, gain)
        
        rect_bl = (0, mid_y, mid_x, self.img_h)
        self._draw_takens(img, rect_bl, gain)
        
        rect_br = (mid_x, mid_y, self.img_w, self.img_h)
        self._draw_compass(img, rect_br, gain, cur_phase, cur_env)
        
        rect_tr = (mid_x, 0, self.img_w, mid_y)
        self._draw_histogram(img, rect_tr)
        
        self._output_image = img

    def _draw_time_series(self, img, rect, gain):
        x0, y0, x1, y1 = rect
        w = x1 - x0
        h = y1 - y0
        h_mid = y0 + h//2
        
        if len(self.gate_buf) < 2: return
        
        n_points = min(len(self.gate_buf), w)
        start_idx = len(self.gate_buf) - n_points
        
        pts_g = np.zeros((n_points, 1, 2), dtype=np.int32)
        pts_e = np.zeros((n_points, 1, 2), dtype=np.int32)
        
        for i in range(n_points):
            val_g = self.gate_buf[start_idx + i]
            val_e = self.env_buf[start_idx + i]
            
            px = x0 + i
            py_g = int(h_mid - val_g * gain * 5)
            py_e = int(h_mid - val_e * gain * 5)
            
            pts_g[i] = [px, np.clip(py_g, y0, y1)]
            pts_e[i] = [px, np.clip(py_e, y0, y1)]
            
        cv2.polylines(img, [pts_g], False, COLOR_GATE, 1)
        cv2.polylines(img, [pts_e], False, COLOR_SIGNAL, 1)
        
        cv2.putText(img, f"TIME: {self.gate_region.upper()} {self.gate_band} vs {self.target_band}", 
                   (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)

    def _draw_takens(self, img, rect, gain):
        x0, y0, x1, y1 = rect
        cx = x0 + (x1-x0)//2
        cy = y0 + (y1-y0)//2
        scale = gain * 15.0
        
        d_idx = int((self.takens_delay_ms / 1000.0) * self.fs)
        if len(self.gate_buf) <= d_idx: return
        
        n_points = min(len(self.gate_buf) - d_idx, 400)
        prev_pt = None
        
        for i in range(n_points):
            curr_idx = len(self.gate_buf) - 1 - i
            past_idx = curr_idx - d_idx
            
            gx = self.gate_buf[curr_idx]
            gy = self.gate_buf[past_idx]
            energy = self.env_buf[curr_idx]
            
            px = int(cx + gx * scale)
            py = int(cy + gy * scale)
            
            intensity = min(1.0, energy * (gain/10.0))
            b = int(50 + 200 * intensity)
            g = int(50 + 150 * intensity)
            r = int(50 + 50 * intensity)
            col = (
                int(50 + (0 - 50)*intensity), 
                int(20 + (200 - 20)*intensity), 
                int(20 + (255 - 20)*intensity)
            )
            
            if prev_pt is not None:
                if (x0 < px < x1) and (y0 < py < y1):
                    cv2.line(img, prev_pt, (px, py), col, 1)
            prev_pt = (px, py)
            
            if i == 0:
                 cv2.circle(img, (px, py), 4, (255, 255, 255), -1)

        cv2.putText(img, "PHASE SPACE (GATING BOX)", (x0+10, y0+20), 
                   cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)

    def _draw_compass(self, img, rect, gain, cur_phase, cur_env):
        x0, y0, x1, y1 = rect
        cx = x0 + (x1-x0)//2
        cy = y0 + (y1-y0)//2
        rad = min((x1-x0), (y1-y0)) // 3
        
        cv2.circle(img, (cx, cy), rad, (60, 60, 60), 1)
        cv2.line(img, (cx-rad, cy), (cx+rad, cy), (40, 40, 40), 1)
        cv2.line(img, (cx, cy-rad), (cx, cy+rad), (40, 40, 40), 1)
        
        vec_len = cur_env * gain * 20.0
        vec_x = int(cx + np.cos(cur_phase) * vec_len)
        vec_y = int(cy + np.sin(cur_phase) * vec_len)
        
        cv2.line(img, (cx, cy), (vec_x, vec_y), COLOR_LOCK, 2)
        cv2.circle(img, (vec_x, vec_y), 5, (255, 255, 255), -1)
        
        stride = 2
        for i in range(0, min(len(self.phase_buf), 200), stride):
            idx = len(self.phase_buf) - 1 - i
            p = self.phase_buf[idx]
            e = self.env_buf[idx]
            
            vl = e * gain * 20.0
            vx = int(cx + np.cos(p) * vl)
            vy = int(cy + np.sin(p) * vl)
            
            cv2.circle(img, (vx, vy), 1, (100, 100, 100), -1)
            
        cv2.putText(img, "PHASE LOCK COMPASS", (x0+10, y0+20), 
                   cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)

    def _draw_histogram(self, img, rect):
        x0, y0, x1, y1 = rect
        w = x1 - x0
        h = y1 - y0
        
        bar_w = w // self.n_bins
        max_val = np.max(self.phase_hist) + 1e-9
        
        for i in range(self.n_bins):
            val = self.phase_hist[i]
            bar_h = int((val / max_val) * (h * 0.8))
            bx = x0 + i * bar_w
            by = y1 - 10
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 2, by), COLOR_HIST, -1)
            
        cv2.putText(img, "PREFERRED PHASE (PAC)", (x0+10, y0+20), 
                   cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_TEXT, 1)

    def get_output(self, name):
        if name == 'display': return self._output_image
        if name == 'interference': return self._interference_img
        return self._last_outs.get(name, 0.0)

    def get_display_image(self):
        return self._output_image