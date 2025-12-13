"""
Wavelet Token Engine - Brain -> Tokens (Keys & Values)
======================================================
The Heavy Node that extracts discrete tokens from continuous brain signals.

ARCHITECTURE:
1. Loads EEG internally (MNE source localization)
2. Extracts 4 brain regions (Frontal, Temporal, Parietal, Occipital)
3. Applies Continuous Wavelet Transform (CWT) to each region
4. Extracts "tokens" = discrete frequency-band bursts
5. Assigns Key (region + frequency) and Value (amplitude + phase)

TOKENS:
- Key: WHERE and WHAT FREQUENCY (e.g., "Frontal_Gamma_40Hz")
- Value: HOW MUCH and WHEN (amplitude + phase at that moment)

OUTPUTS:
- token_stream: Spectrum (N x 3: key, amp, phase)
- frontal_tokens: Spectrum
- temporal_tokens: Spectrum
- token_attention: Image (Active tokens heatmap)
- control_matrix: Complex Spectrum (Interference Field)
- display: Image (Dashboard)
"""

import numpy as np
import cv2
import os
from collections import deque
from scipy.signal import hilbert, butter, lfilter

# --- MANUAL MORLET FIX (Bypasses SciPy Import Error) ---
def local_morlet(M, s, w=5.0):
    """
    Complex Morlet wavelet, defined manually to avoid import errors.
    M: Length of the wavelet
    s: Scaling factor (width)
    w: Omega0 (frequency parameter)
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    # Psi(x) = pi**(-0.25) * exp(1j*w*x) * exp(-0.5*x**2)
    wavelet = np.pi**(-0.25) * np.exp(1j * w * x) * np.exp(-0.5 * x**2)
    return wavelet

# --- MNE IMPORT SAFETY ---
try:
    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
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

class WaveletTokenEngineNode(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Wavelet Token Engine"
    NODE_COLOR = QtGui.QColor(255, 140, 0) # Deep Orange
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "speed": "signal",
            "gain": "signal",
            "burst_threshold": "signal",
        }
        
        self.outputs = {
            'display': 'image',
            'token_stream': 'spectrum',        # All tokens (N x 3)
            'frontal_tokens': 'spectrum',      # Just frontal
            'temporal_tokens': 'spectrum',     # Just temporal
            'token_attention': 'image',        # Heatmap
            'control_matrix': 'spectrum'       # Complex FFT for interference
        }
        
        # Config
        self.edf_path = r"E:\DocsHouse\450\2.edf"
        self.fs = 160.0
        self.base_speed = 1.0
        self.base_threshold = 1.5  # Sigma multiplier
        
        # State
        self.is_loaded = False
        self.load_error = ""
        self.needs_load = True
        self.playback_idx = 0.0
        
        # Source data (full time series)
        self.source_series = {
            'frontal': None,
            'temporal': None,
            'parietal': None,
            'occipital': None
        }
        
        # Frequency Bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 70)
        }
        
        # Token vocabulary
        self.token_vocab = {}
        self.build_vocabulary()
        
        # Active State
        self.active_tokens = []  
        self.token_history = deque(maxlen=200)
        
        # Visualization
        self._display = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def build_vocabulary(self):
        """Builds the dictionary of possible tokens (Region x Band)"""
        regions = ['frontal', 'temporal', 'parietal', 'occipital']
        idx = 0
        for region in regions:
            for band_name in self.freq_bands.keys():
                self.token_vocab[(region, band_name)] = idx
                idx += 1
        print(f"[Tokens] Vocabulary size: {len(self.token_vocab)}")
    
    def get_config_options(self):
        return [
            ("EEG File", "edf_path", self.edf_path, "file_open"),
            ("Reload", "needs_load", True, "button")
        ]

    # --- MNE HELPER METHODS (Restored from PhaseGatingNode) ---
    
    def _clean_names(self, raw):
        """Standardizes channel names to 10-20 system"""
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

    def _get_region_mask(self, coords, region_name):
        """Returns boolean mask for 3D source coordinates"""
        if region_name == "frontal":
            return coords[:, 1] > 0.05
        elif region_name == "occipital":
            return coords[:, 1] < -0.05
        elif region_name == "parietal":
            return (coords[:, 1] < 0.0) & (coords[:, 1] > -0.06) & (coords[:, 2] > 0.04)
        elif region_name == "temporal":
            return (coords[:, 1] < 0.0) & (coords[:, 2] < 0.0) & (np.abs(coords[:, 0]) > 0.03)
        else:
            return np.ones(len(coords), dtype=bool)
    
    def setup_source(self):
        """Heavy MNE Pipeline: Load -> Forward -> Inverse -> Extract Regions"""
        if not MNE_AVAILABLE:
            self.load_error = "MNE not installed"
            return
        
        if not os.path.exists(self.edf_path):
            self.load_error = "File not found"
            return
        
        try:
            print(f"[Tokens] Loading: {self.edf_path}")
            
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            self.fs = raw.info['sfreq']
            
            # Clean names & Montage
            raw = self._clean_names(raw)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            raw.set_eeg_reference('average', projection=True, verbose=False)
            
            # Broadband Filter (Keep 0.5 - 70Hz to capture Delta through Gamma)
            high_freq = min(70, (self.fs / 2) - 1)
            raw.filter(0.5, high_freq, verbose=False)
            
            # --- SOURCE MODEL ---
            print("[Tokens] Building Source Model (Sphere + Volumetric)...")
            sphere = mne.make_sphere_model(
                r0=(0., 0., 0.), head_radius=0.095, 
                relative_radii=(0.90, 0.92, 0.97, 1.0),
                sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False
            )
            
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
            # Check for fsaverage, fetch if missing
            if not os.path.exists(os.path.join(subjects_dir, 'fsaverage')):
                 print("Fetching fsaverage...")
                 mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)

            src = mne.setup_volume_source_space(
                subject='fsaverage', pos=30.0,
                sphere=sphere, bem=None,
                subjects_dir=subjects_dir, verbose=False
            )
            
            fwd = mne.make_forward_solution(
                raw.info, trans=None, src=src, bem=sphere,
                eeg=True, meg=False, verbose=False
            )
            
            # Sanitize forward matrix (Fixes the Divide by Zero warnings)
            G = fwd['sol']['data']
            if not np.all(np.isfinite(G)):
                np.nan_to_num(G, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                fwd['sol']['data'] = G
            
            cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
            inv = mne.minimum_norm.make_inverse_operator(
                raw.info, fwd, cov, depth=None, loose='auto', verbose=False
            )
            
            print("[Tokens] Extracting Regional Time Series...")
            
            # Apply Inverse
            stc = mne.minimum_norm.apply_inverse_raw(
                raw, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            
            coords = src[0]['rr'][stc.vertices[0]]
            
            # Extract each region using the masks
            for region_name in ['frontal', 'temporal', 'parietal', 'occipital']:
                mask = self._get_region_mask(coords, region_name)
                
                if np.sum(mask) == 0:
                    print(f"Warning: Region {region_name} yielded no vertices. Using fallback.")
                    mask[:] = True 
                
                # Mean source activity for this region
                region_data = np.mean(stc.data[mask], axis=0)
                
                # Z-Score Normalize
                region_data = (region_data - np.mean(region_data)) / (np.std(region_data) + 1e-9)
                self.source_series[region_name] = region_data
            
            self.is_loaded = True
            self.load_error = ""
            print(f"[Tokens] Ready. {len(self.source_series['frontal'])} samples at {self.fs}Hz")
            
        except Exception as e:
            self.load_error = str(e)
            print(f"[Tokens] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def step(self):
        if self.needs_load:
            self.setup_source()
            self.needs_load = False
        
        if not self.is_loaded:
            self._render_error()
            return
        
        # --- FIX: SAFE INPUT HANDLING (NoneType Protection) ---
        
        # Speed
        speed_val = self.get_blended_input("speed", "mean")
        if speed_val is None: speed_val = 0.0  # <--- The Fix
        speed = self.base_speed * (speed_val if speed_val > 0 else 1.0)
        
        # Gain
        gain_val = self.get_blended_input("gain", "mean")
        if gain_val is None: gain_val = 0.0    # <--- The Fix
        gain = 1.0 * (gain_val if gain_val > 0 else 1.0)
        
        # Threshold
        thresh_val = self.get_blended_input("burst_threshold", "mean")
        if thresh_val is None: thresh_val = 0.0 # <--- The Fix
        threshold_mult = self.base_threshold * (thresh_val if thresh_val > 0 else 1.0)
        
        # Playback
        idx = int(self.playback_idx)
        total_len = len(self.source_series['frontal'])
        window_len = 256
        
        if idx + window_len >= total_len:
            self.playback_idx = 0
            idx = 0
        
        # === TOKEN EXTRACTION ===
        self.active_tokens = []
        
        for region_name, series in self.source_series.items():
            if series is None: continue
            
            # Extract window
            window = series[idx:idx + window_len] * gain
            
            # For each frequency band
            for band_name, (low, high) in self.freq_bands.items():
                
                # Check bounds
                nyq = self.fs / 2.0
                if high >= nyq: high = nyq - 0.1
                if low >= high: continue
                
                # Fast Bandpass
                b, a = butter(3, [low/nyq, high/nyq], btype='band')
                band_signal = lfilter(b, a, window)
                
                # Analytic Signal (Hilbert)
                analytic = hilbert(band_signal)
                envelope = np.abs(analytic)
                phase = np.angle(analytic)
                
                # Center value
                mid = window_len // 2
                amp = envelope[mid]
                phi = phase[mid]
                
                # Threshold logic
                local_mean = np.mean(envelope)
                local_std = np.std(envelope)
                thresh_val = local_mean + threshold_mult * local_std
                
                # Create Token if Burst detected
                if amp > thresh_val and amp > 0.1:
                    key_id = self.token_vocab.get((region_name, band_name), 0)
                    
                    self.active_tokens.append({
                        'key': float(key_id),
                        'region': region_name,
                        'band': band_name,
                        'amplitude': float(amp),
                        'phase': float(phi),
                        'frequency': float((low + high) / 2),
                        'time': self.playback_idx / self.fs
                    })
        
        self.token_history.append(list(self.active_tokens))
        
        self._update_output_ports()
        self._render_dashboard()
        self.playback_idx += speed
    
    def _update_output_ports(self):
        # 1. Main Stream
        if self.active_tokens:
            arr = np.array([[t['key'], t['amplitude'], t['phase']] for t in self.active_tokens], dtype=np.float32)
            self.outputs['token_stream'] = arr
        else:
            self.outputs['token_stream'] = np.zeros((1, 3), dtype=np.float32)
            
        # 2. Regional
        for rname in ['frontal', 'temporal']:
            subset = [t for t in self.active_tokens if t['region'] == rname]
            out_key = f"{rname}_tokens"
            if subset:
                self.outputs[out_key] = np.array([[t['key'], t['amplitude'], t['phase']] for t in subset], dtype=np.float32)
            else:
                self.outputs[out_key] = np.zeros((1, 3), dtype=np.float32)
        
        # 3. Control Matrix (FFT of Attention)
        attn_map = np.zeros((128, 128), dtype=np.float32)
        for t in self.active_tokens:
            k = int(t['key'])
            # Map key to position (grid layout)
            y = (k % 16) * 8
            x = (k // 16) * 8
            if x < 128 and y < 128:
                attn_map[y, x] = t['amplitude']
        
        fft_mat = np.fft.fftshift(np.fft.fft2(attn_map))
        self.outputs['control_matrix'] = fft_mat
        
        # 4. Attention Image
        norm_attn = cv2.normalize(attn_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.outputs['token_attention'] = cv2.applyColorMap(norm_attn, cv2.COLORMAP_INFERNO)

    def _render_dashboard(self):
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # Draw Stream History
        cell_w = max(1, w // 200)
        cell_h = 300 // len(self.token_vocab)
        
        hist_len = min(len(self.token_history), 200)
        
        for i in range(hist_len):
            x = i * cell_w
            tokens = self.token_history[-(hist_len - i)]
            for t in tokens:
                y = 40 + int(t['key']) * cell_h
                
                # Color
                col = (150, 150, 150)
                if t['region'] == 'frontal': col = (255, 100, 100)
                elif t['region'] == 'temporal': col = (100, 255, 100)
                elif t['region'] == 'parietal': col = (255, 255, 100)
                elif t['region'] == 'occipital': col = (100, 100, 255)
                
                # Intensity
                alpha = min(1.0, t['amplitude'] / 3.0)
                c_int = tuple(int(c * alpha) for c in col)
                
                cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), c_int, -1)
        
        # Draw Labels
        cv2.putText(img, f"ACTIVE TOKENS: {len(self.active_tokens)}", (10, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw Active Cards
        x_pos = 10
        y_pos = 380
        sorted_toks = sorted(self.active_tokens, key=lambda x: x['amplitude'], reverse=True)
        
        for t in sorted_toks[:8]:
            cv2.rectangle(img, (x_pos, y_pos), (x_pos + 140, y_pos + 80), (50, 50, 60), -1)
            
            c = (200, 200, 200)
            if t['region'] == 'frontal': c = (100, 100, 255) # BGR
            
            cv2.putText(img, f"{t['region'][:4].upper()} {t['band']}", (x_pos+5, y_pos+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            cv2.putText(img, f"Amp: {t['amplitude']:.2f}", (x_pos+5, y_pos+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            x_pos += 150
            if x_pos > w - 150: break

        self._display = img
    
    def _render_error(self):
        img = self._display
        img[:] = (20, 20, 25)
        if not self.load_error:
            cv2.putText(img, "LOADING MNE MODEL...", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        else:
            cv2.putText(img, f"ERROR: {self.load_error}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        self._display = img

    def get_display_image(self): return self._display
    def get_output(self, name): return self.outputs.get(name)