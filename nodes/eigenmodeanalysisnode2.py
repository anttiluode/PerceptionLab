"""
Eigenmode + EEG Analysis Node - Synchronized Spatial & Temporal Decomposition
==============================================================================

This node outputs BOTH:
- Traditional band powers (delta, theta, alpha, beta, gamma) 
- Eigenmode activations (modes 1-10 as individual signals)

All outputs are SYNCHRONIZED - computed from the same time window.
This allows downstream exploration of how temporal frequencies relate
to spatial network modes.

THEORY:
- Band powers = temporal frequency content (how fast neurons oscillate)
- Eigenmodes = spatial frequency content (how activity spreads across network)

By outputting both as synchronized signals, you can:
- Correlate alpha power with specific eigenmodes
- See if theta phase relates to mode switching
- Discover which modes carry which frequencies

OUTPUTS (all synchronized):
  Band Powers (signal):
    - delta_power, theta_power, alpha_power, beta_power, gamma_power
  
  Eigenmode Activations (signal):
    - mode_1 through mode_10 (individual mode strengths)
  
  Composite Outputs:
    - eigenmode_image: Visual of mode activations over time
    - mode_topo: Topographic reconstruction
    - band_spectrum: 5-dim vector of band powers (latent)
    - mode_spectrum: 10-dim vector of mode activations (latent)
    - full_spectrum: 15-dim combined vector (latent)
    - raw_signal: Amplified mean EEG

Created: December 2025
For: PerceptionLab v11
"""

import numpy as np
import cv2
import os
from collections import deque

# === SCIPY IMPORTS ===
try:
    from scipy import signal
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available.")

# === MNE IMPORT ===
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available.")

# === PERCEPTION LAB COMPATIBILITY ===
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name, mode): 
            return None
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}

# === CONSTANTS ===
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}

CHANNEL_POSITIONS_2D = {
    'FP1': (0.35, 0.95), 'FP2': (0.65, 0.95), 'FPZ': (0.5, 0.95),
    'F7': (0.15, 0.75), 'F3': (0.35, 0.75), 'FZ': (0.5, 0.75), 
    'F4': (0.65, 0.75), 'F8': (0.85, 0.75),
    'FC5': (0.2, 0.65), 'FC1': (0.4, 0.65), 'FC2': (0.6, 0.65), 'FC6': (0.8, 0.65),
    'T7': (0.1, 0.5), 'C3': (0.3, 0.5), 'CZ': (0.5, 0.5), 
    'C4': (0.7, 0.5), 'T8': (0.9, 0.5),
    'T3': (0.1, 0.5), 'T4': (0.9, 0.5),
    'CP5': (0.2, 0.35), 'CP1': (0.4, 0.35), 'CP2': (0.6, 0.35), 'CP6': (0.8, 0.35),
    'P7': (0.15, 0.25), 'P3': (0.35, 0.25), 'PZ': (0.5, 0.25), 
    'P4': (0.65, 0.25), 'P8': (0.85, 0.25),
    'PO7': (0.25, 0.15), 'PO3': (0.4, 0.15), 'POZ': (0.5, 0.15),
    'PO4': (0.6, 0.15), 'PO8': (0.75, 0.15),
    'O1': (0.35, 0.05), 'OZ': (0.5, 0.05), 'O2': (0.65, 0.05),
    'TP7': (0.1, 0.4), 'TP8': (0.9, 0.4),
    'FT7': (0.1, 0.65), 'FT8': (0.9, 0.65),
}


class EigenmodeEEGNode(BaseNode):
    """
    Synchronized Eigenmode + Band Power Analysis
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Eigenmode + EEG"
    NODE_COLOR = QtGui.QColor(180, 80, 220)  # Purple-pink
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'gain_mod': 'signal',
            'speed_mod': 'signal',
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # === BAND POWER SIGNALS (synchronized) ===
            'delta_power': 'signal',
            'theta_power': 'signal',
            'alpha_power': 'signal',
            'beta_power': 'signal',
            'gamma_power': 'signal',
            
            # === EIGENMODE SIGNALS (synchronized) ===
            'mode_1': 'signal',
            'mode_2': 'signal',
            'mode_3': 'signal',
            'mode_4': 'signal',
            'mode_5': 'signal',
            'mode_6': 'signal',
            'mode_7': 'signal',
            'mode_8': 'signal',
            'mode_9': 'signal',
            'mode_10': 'signal',
            
            # === COMPOSITE OUTPUTS ===
            'eigenmode_image': 'image',
            'mode_topo': 'image',
            'band_spectrum': 'spectrum',      # 5-dim
            'mode_spectrum': 'spectrum',      # 10-dim  
            'full_spectrum': 'spectrum',      # 15-dim combined
            'raw_signal': 'signal',
            
            # === DERIVED SIGNALS ===
            'dominant_mode': 'signal',
            'alpha_mode_ratio': 'signal',     # alpha / mode_2 correlation proxy
            'theta_mode_ratio': 'signal',     # theta / mode_1 correlation proxy
        }
        
        # === CONFIGURATION ===
        self.edf_path = ""
        self.time_window = 0.3
        self.amplification = 50.0
        self.band_amplification = 5.0
        self.update_every = 1
        self.target_fs = 128.0
        self.n_modes = 10
        
        # === INTERNAL STATE ===
        self.raw = None
        self.fs = 128.0
        self.n_channels = 0
        self.channel_names = []
        self.channel_positions = None
        
        # Eigenmode data
        self.eigenmodes = None
        self.eigenvalues = None
        self.modes_computed = False
        
        # Playback
        self.playback_idx = 0
        self.total_samples = 0
        self.frame_count = 0
        
        # Current outputs (all synchronized)
        self.current_band_powers = {b: 0.0 for b in BANDS.keys()}
        self.current_mode_activations = np.zeros(self.n_modes)
        self.current_raw = 0.0
        
        # Images
        self.current_mode_image = None
        self.current_topo_image = None
        
        # Loading state
        self.is_loaded = False
        self.load_error = ""
        self.needs_load = True
        self._last_path = ""
        
        # History for visualization
        self.mode_history = deque(maxlen=100)
        self.band_history = deque(maxlen=100)
        
    def _load_edf(self):
        """Load EDF and compute eigenmodes"""
        if not MNE_AVAILABLE:
            self.load_error = "MNE not installed"
            return False
            
        if not self.edf_path or not os.path.exists(self.edf_path):
            self.load_error = f"File not found: {self.edf_path}"
            return False
            
        try:
            print(f"[EigenmodeEEG] Loading: {self.edf_path}")
            
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            raw.pick_types(eeg=True, exclude=[])
            
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore')
            except:
                pass
            
            if raw.info['sfreq'] > self.target_fs * 1.5:
                raw.resample(self.target_fs, verbose=False)
                
            self.raw = raw
            self.fs = raw.info['sfreq']
            self.n_channels = len(raw.ch_names)
            self.channel_names = [ch.upper() for ch in raw.ch_names]
            self.total_samples = raw.n_times
            
            self._setup_channel_positions()
            self._compute_eigenmodes()
            
            self.is_loaded = True
            self.needs_load = False
            self._last_path = self.edf_path
            self.playback_idx = 0
            
            print(f"[EigenmodeEEG] Loaded: {self.n_channels} ch, "
                  f"{self.total_samples/self.fs:.1f}s @ {self.fs:.0f}Hz")
            
            return True
            
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            print(f"[EigenmodeEEG] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_channel_positions(self):
        """Get 2D positions for channels"""
        positions = []
        for ch in self.channel_names:
            ch_upper = ch.upper()
            if ch_upper in CHANNEL_POSITIONS_2D:
                positions.append(CHANNEL_POSITIONS_2D[ch_upper])
            else:
                positions.append((np.random.rand(), np.random.rand()))
        self.channel_positions = np.array(positions)
    
    def _compute_eigenmodes(self):
        """Compute graph Laplacian eigenmodes"""
        if self.raw is None:
            return
            
        n_samples_for_cov = min(int(5.0 * self.fs), self.total_samples)
        data, _ = self.raw[:, :n_samples_for_cov]
        
        # Correlation-based connectivity
        corr = np.corrcoef(data)
        corr = np.nan_to_num(corr, nan=0.0)
        
        A = np.abs(corr)
        np.fill_diagonal(A, 0)
        threshold = np.percentile(A, 50)
        A[A < threshold] = 0
        
        # Graph Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A + 1e-8 * np.eye(self.n_channels)
        
        n_modes = min(self.n_modes + 1, self.n_channels - 1)
        
        try:
            if self.n_channels > 50:
                L_sparse = csr_matrix(L.astype(np.float32))
                eigenvalues, eigenmodes = eigsh(L_sparse, k=n_modes, which='SM', 
                                                 tol=1e-4, maxiter=3000)
            else:
                eigenvalues, eigenmodes = np.linalg.eigh(L)
                eigenvalues = eigenvalues[:n_modes]
                eigenmodes = eigenmodes[:, :n_modes]
                
            idx = np.argsort(eigenvalues)
            self.eigenvalues = eigenvalues[idx]
            self.eigenmodes = eigenmodes[:, idx]
            self.modes_computed = True
            
            print(f"[EigenmodeEEG] Computed {n_modes} eigenmodes")
            
        except Exception as e:
            print(f"[EigenmodeEEG] Eigenmode error: {e}")
            self.modes_computed = False
    
    def step(self):
        """Main processing - compute synchronized band powers and mode activations"""
        if self.edf_path != self._last_path:
            self.needs_load = True
            
        if self.needs_load and self.edf_path:
            self._load_edf()
            
        if not self.is_loaded or not self.modes_computed:
            return
            
        self.frame_count += 1
        if self.frame_count % max(1, self.update_every) != 0:
            return
            
        # Get modulation inputs
        gain_mod = self.get_blended_input('gain_mod', 'sum') or 0.0
        speed_mod = self.get_blended_input('speed_mod', 'sum') or 0.0
        
        total_gain = self.amplification * (1.0 + gain_mod)
        speed = 1.0 + speed_mod * 0.5
        
        # Get current window
        window_samples = int(self.time_window * self.fs)
        start_idx = int(self.playback_idx)
        end_idx = start_idx + window_samples
        
        if end_idx >= self.total_samples:
            self.playback_idx = 0
            start_idx = 0
            end_idx = window_samples
            
        data, _ = self.raw[:, start_idx:end_idx]
        
        # ============================================
        # SYNCHRONIZED COMPUTATION
        # ============================================
        
        # === 1. BAND POWERS ===
        nyq = self.fs / 2.0
        for band_name, (low, high) in BANDS.items():
            if high >= nyq:
                high = nyq - 1
            if low < high:
                try:
                    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
                    band_data = np.zeros_like(data)
                    for ch in range(data.shape[0]):
                        band_data[ch] = signal.filtfilt(b, a, data[ch])
                    power = np.mean(band_data**2) * self.band_amplification
                    self.current_band_powers[band_name] = float(np.log1p(power))
                except:
                    self.current_band_powers[band_name] = 0.0
            else:
                self.current_band_powers[band_name] = 0.0
        
        # === 2. EIGENMODE ACTIVATIONS ===
        channel_activity = np.mean(data, axis=1) * total_gain
        self.current_raw = float(np.mean(channel_activity))
        
        # Project onto modes 1-10 (skip mode 0 which is constant)
        for i in range(self.n_modes):
            mode_idx = i + 1  # Skip mode 0
            if mode_idx < self.eigenmodes.shape[1]:
                mode_vector = self.eigenmodes[:, mode_idx]
                activation = np.dot(mode_vector, channel_activity)
                self.current_mode_activations[i] = activation
            else:
                self.current_mode_activations[i] = 0.0
        
        # Store history
        self.mode_history.append(self.current_mode_activations.copy())
        self.band_history.append([self.current_band_powers[b] for b in BANDS.keys()])
        
        # === 3. CREATE VISUALIZATIONS ===
        self._create_mode_image()
        self._create_topo_image(channel_activity)
        
        # Advance playback
        self.playback_idx += window_samples * 0.5 * speed
        
    def _create_mode_image(self):
        """Create visualization showing both modes and bands"""
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.float32)
        
        history = list(self.mode_history)
        band_hist = list(self.band_history)
        n_history = len(history)
        
        if n_history > 0:
            # Normalize
            all_modes = np.array(history)
            mode_max = np.abs(all_modes).max() + 1e-6
            
            all_bands = np.array(band_hist) if band_hist else np.zeros((1, 5))
            band_max = np.abs(all_bands).max() + 1e-6
            
            bar_width = max(1, w // n_history)
            
            # Top section: Modes (100 pixels)
            mode_height = 100 / self.n_modes
            for t, modes in enumerate(history):
                x = t * bar_width
                for m, val in enumerate(modes):
                    y = int(m * mode_height)
                    y_end = int((m + 1) * mode_height)
                    norm_val = val / mode_max
                    if norm_val > 0:
                        color = (0, norm_val * 0.8, norm_val)
                    else:
                        color = (-norm_val, 0, -norm_val * 0.3)
                    img[y:y_end, x:x+bar_width] = color
            
            # Separator line
            img[100:102, :] = (0.3, 0.3, 0.3)
            
            # Bottom section: Bands (58 pixels)
            band_height = 56 / 5
            band_colors = [
                (0.6, 0.2, 0.1),   # delta - brown
                (0.8, 0.2, 0.2),   # theta - red
                (0.2, 0.8, 0.2),   # alpha - green
                (0.8, 0.8, 0.2),   # beta - yellow
                (0.2, 0.2, 0.8),   # gamma - blue
            ]
            for t, bands in enumerate(band_hist):
                x = t * bar_width
                for b, val in enumerate(bands):
                    y = 102 + int(b * band_height)
                    y_end = 102 + int((b + 1) * band_height)
                    norm_val = val / band_max
                    base_color = np.array(band_colors[b])
                    img[y:y_end, x:x+bar_width] = base_color * norm_val
        
        # Labels
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.putText(img_u8, "Modes 1-10", (5, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(img_u8, "Bands", (5, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Band labels on right
        band_names = ['d', 't', 'a', 'b', 'g']
        for i, name in enumerate(band_names):
            y = 112 + int(i * 11)
            cv2.putText(img_u8, name, (w-12, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        self.current_mode_image = img_u8
    
    def _create_topo_image(self, channel_activity):
        """Create topographic map"""
        if self.channel_positions is None:
            return
            
        h, w = 128, 128
        img = np.zeros((h, w), dtype=np.float32)
        
        # Reconstruct from modes
        reconstructed = np.zeros(self.n_channels)
        for i in range(self.n_modes):
            mode_idx = i + 1
            if mode_idx < self.eigenmodes.shape[1]:
                act = self.current_mode_activations[i]
                reconstructed += act * self.eigenmodes[:, mode_idx] / self.n_modes
        
        # Interpolate to grid
        for ch_idx in range(self.n_channels):
            x, y = self.channel_positions[ch_idx]
            px, py = int(x * (w-1)), int((1-y) * (h-1))
            
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        dist = np.sqrt(dx**2 + dy**2)
                        weight = np.exp(-dist**2 / 20)
                        img[ny, nx] += weight * reconstructed[ch_idx]
        
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        self.current_topo_image = img_color
    
    def get_output(self, port_name):
        """Return synchronized outputs"""
        if not self.is_loaded:
            if port_name in ['delta_power', 'theta_power', 'alpha_power', 
                            'beta_power', 'gamma_power', 'raw_signal',
                            'dominant_mode', 'alpha_mode_ratio', 'theta_mode_ratio'] or \
               port_name.startswith('mode_'):
                return 0.0
            return None
        
        # === BAND POWER SIGNALS ===
        if port_name == 'delta_power':
            return self.current_band_powers['delta']
        elif port_name == 'theta_power':
            return self.current_band_powers['theta']
        elif port_name == 'alpha_power':
            return self.current_band_powers['alpha']
        elif port_name == 'beta_power':
            return self.current_band_powers['beta']
        elif port_name == 'gamma_power':
            return self.current_band_powers['gamma']
        
        # === EIGENMODE SIGNALS ===
        elif port_name == 'mode_1':
            return float(self.current_mode_activations[0])
        elif port_name == 'mode_2':
            return float(self.current_mode_activations[1])
        elif port_name == 'mode_3':
            return float(self.current_mode_activations[2])
        elif port_name == 'mode_4':
            return float(self.current_mode_activations[3])
        elif port_name == 'mode_5':
            return float(self.current_mode_activations[4])
        elif port_name == 'mode_6':
            return float(self.current_mode_activations[5])
        elif port_name == 'mode_7':
            return float(self.current_mode_activations[6])
        elif port_name == 'mode_8':
            return float(self.current_mode_activations[7])
        elif port_name == 'mode_9':
            return float(self.current_mode_activations[8])
        elif port_name == 'mode_10':
            return float(self.current_mode_activations[9])
        
        # === IMAGE OUTPUTS ===
        elif port_name == 'eigenmode_image':
            return self.current_mode_image
        elif port_name == 'mode_topo':
            return self.current_topo_image
        
        # === SPECTRUM OUTPUTS ===
        elif port_name == 'band_spectrum':
            return np.array([self.current_band_powers[b] for b in BANDS.keys()], 
                           dtype=np.float32)
        elif port_name == 'mode_spectrum':
            return self.current_mode_activations.astype(np.float32)
        elif port_name == 'full_spectrum':
            bands = np.array([self.current_band_powers[b] for b in BANDS.keys()])
            return np.concatenate([bands, self.current_mode_activations]).astype(np.float32)
        
        # === DERIVED SIGNALS ===
        elif port_name == 'raw_signal':
            return self.current_raw
        elif port_name == 'dominant_mode':
            return float(np.argmax(np.abs(self.current_mode_activations)) + 1)
        elif port_name == 'alpha_mode_ratio':
            # Alpha power / |mode_2| - tests if alpha correlates with mode 2
            alpha = self.current_band_powers['alpha']
            mode2 = abs(self.current_mode_activations[1]) + 1e-6
            return float(alpha / mode2)
        elif port_name == 'theta_mode_ratio':
            # Theta power / |mode_1| - tests if theta correlates with mode 1
            theta = self.current_band_powers['theta']
            mode1 = abs(self.current_mode_activations[0]) + 1e-6
            return float(theta / mode1)
            
        return None
    
    def get_display_image(self):
        """Return display for node preview"""
        if self.current_mode_image is not None:
            img = self.current_mode_image
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        else:
            w, h = 128, 64
            img = np.zeros((h, w, 3), dtype=np.uint8)
            msg = self.load_error[:20] if self.load_error else "No EDF"
            cv2.putText(img, msg, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            img = np.ascontiguousarray(img)
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Time Window (s)", "time_window", self.time_window, None),
            ("Mode Amplification", "amplification", self.amplification, None),
            ("Band Amplification", "band_amplification", self.band_amplification, None),
            ("Update Every N Frames", "update_every", self.update_every, None),
        ]
    
    def close(self):
        self.raw = None
        self.eigenmodes = None
        self.eigenvalues = None