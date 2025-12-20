"""
EEG Eigenmode Analysis Node - Raj-Style Graph Laplacian Brain Dynamics
=======================================================================

Based on: Wang, Owen, Mukherjee, Raj (2017) "Brain network eigenmodes provide 
a robust and compact representation of the structural connectome"

This node computes the graph Laplacian eigenmodes from EEG channel connectivity
and projects EEG signals onto these modes. Unlike full MNE source localization,
this operates directly on sensor space for speed.

THEORY:
The brain graph's Laplacian eigenmodes describe how activity "diffuses" through
the network. Low eigenmodes = slow, global patterns. High eigenmodes = fast,
local patterns. By projecting EEG onto these modes, we decompose brain activity
into its fundamental spatial frequencies.

OUTPUTS:
- eigenmode_image: 2D visualization of current mode activations
- mode_spectrum: Vector of all mode activations (latent)
- dominant_mode: Strongest mode index (signal)
- mode_energy: Total energy in selected mode range (signal)
- low_modes: Slow global activity (signal)
- high_modes: Fast local activity (signal)  
- mode_complex: Complex representation for holographic processing
- raw_signal: Amplified raw EEG for monitoring

SETTINGS:
- Time Window: How much EEG to analyze (0.1s to 2.0s)
- Mode Range: Which eigenmodes to focus on (start, end)
- Amplification: Signal boost for weak EEG
- Update Rate: How often to recompute (frames to skip)

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
    from scipy.sparse import coo_matrix, diags, csr_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. EigenmodeAnalysisNode will not function.")

# === MNE IMPORT (optional, for EDF loading) ===
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available. EDF loading will not work.")

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
# Standard 10-20 channel positions (approximate 2D layout for visualization)
CHANNEL_POSITIONS_2D = {
    'FP1': (0.35, 0.95), 'FP2': (0.65, 0.95), 'FPZ': (0.5, 0.95),
    'F7': (0.15, 0.75), 'F3': (0.35, 0.75), 'FZ': (0.5, 0.75), 
    'F4': (0.65, 0.75), 'F8': (0.85, 0.75),
    'FC5': (0.2, 0.65), 'FC1': (0.4, 0.65), 'FC2': (0.6, 0.65), 'FC6': (0.8, 0.65),
    'T7': (0.1, 0.5), 'C3': (0.3, 0.5), 'CZ': (0.5, 0.5), 
    'C4': (0.7, 0.5), 'T8': (0.9, 0.5),
    'T3': (0.1, 0.5), 'T4': (0.9, 0.5),  # Alternative names
    'CP5': (0.2, 0.35), 'CP1': (0.4, 0.35), 'CP2': (0.6, 0.35), 'CP6': (0.8, 0.35),
    'P7': (0.15, 0.25), 'P3': (0.35, 0.25), 'PZ': (0.5, 0.25), 
    'P4': (0.65, 0.25), 'P8': (0.85, 0.25),
    'PO7': (0.25, 0.15), 'PO3': (0.4, 0.15), 'POZ': (0.5, 0.15),
    'PO4': (0.6, 0.15), 'PO8': (0.75, 0.15),
    'O1': (0.35, 0.05), 'OZ': (0.5, 0.05), 'O2': (0.65, 0.05),
    # Temporal alternatives
    'TP7': (0.1, 0.4), 'TP8': (0.9, 0.4),
    'FT7': (0.1, 0.65), 'FT8': (0.9, 0.65),
}

# Frequency bands for optional band-specific eigenmode analysis
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}


class EigenmodeAnalysisNode(BaseNode):
    """
    EEG Eigenmode Analysis - Graph Laplacian decomposition of brain activity
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Eigenmode Analysis"
    NODE_COLOR = QtGui.QColor(150, 50, 200)  # Purple for eigenmode/spectral
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'gain_mod': 'signal',      # External gain modulation
            'mode_select': 'signal',   # External mode selection (0-1 maps to mode range)
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Image outputs
            'eigenmode_image': 'image',     # 2D visualization of mode activations
            'mode_topo': 'image',           # Topographic map of dominant mode
            
            # Latent/spectrum outputs
            'mode_spectrum': 'spectrum',     # Full vector of mode activations
            'mode_complex': 'complex_spectrum',  # Complex representation
            
            # Signal outputs
            'dominant_mode': 'signal',       # Index of strongest mode
            'mode_energy': 'signal',         # Total energy in mode range
            'low_modes': 'signal',           # Slow/global activity (modes 1-5)
            'high_modes': 'signal',          # Fast/local activity (modes 15+)
            'raw_signal': 'signal',          # Amplified mean EEG
            'eigenvalue_ratio': 'signal',    # λ2/λ1 ratio (connectivity measure)
        }
        
        # === CONFIGURATION ===
        self.edf_path = ""
        self.time_window = 0.3          # Seconds to analyze (short for speed!)
        self.mode_range_start = 1       # First mode to include (0 is constant)
        self.mode_range_end = 20        # Last mode to include
        self.n_modes_compute = 30       # Total modes to compute
        self.amplification = 50.0       # Signal amplification
        self.update_every = 1           # Frames to skip between updates
        self.target_fs = 128.0          # Resample rate
        self.band_filter = 'broadband'  # 'broadband', 'alpha', 'theta', etc.
        self.weight_scheme = 'uniform'  # 'uniform', 'linear_decay', 'exponential'
        
        # === INTERNAL STATE ===
        self.raw = None
        self.fs = 128.0
        self.n_channels = 0
        self.channel_names = []
        self.channel_positions = None   # Nx2 array
        
        # Eigenmode data
        self.laplacian = None
        self.eigenmodes = None          # n_channels x n_modes
        self.eigenvalues = None         # n_modes
        self.modes_computed = False
        
        # Playback state
        self.playback_idx = 0
        self.total_samples = 0
        self.frame_count = 0
        
        # Current outputs
        self.current_mode_activations = None
        self.current_mode_image = None
        self.current_topo_image = None
        self.current_raw = 0.0
        
        # Loading state
        self.is_loaded = False
        self.load_error = ""
        self.needs_load = True
        self._last_path = ""
        
        # History for visualization
        self.mode_history = deque(maxlen=100)
        
    def _load_edf(self):
        """Load EDF file and compute eigenmodes"""
        if not MNE_AVAILABLE:
            self.load_error = "MNE not installed"
            return False
            
        if not self.edf_path or not os.path.exists(self.edf_path):
            self.load_error = f"File not found: {self.edf_path}"
            return False
            
        try:
            print(f"[EigenmodeNode] Loading: {self.edf_path}")
            
            # Load with MNE
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            # Pick EEG channels only
            raw.pick_types(eeg=True, exclude=[])
            
            # Apply montage for positions
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore')
            except:
                pass
            
            # Resample for speed
            if raw.info['sfreq'] > self.target_fs * 1.5:
                raw.resample(self.target_fs, verbose=False)
                
            self.raw = raw
            self.fs = raw.info['sfreq']
            self.n_channels = len(raw.ch_names)
            self.channel_names = [ch.upper() for ch in raw.ch_names]
            self.total_samples = raw.n_times
            
            # Get channel positions
            self._setup_channel_positions()
            
            # Compute eigenmodes from connectivity
            self._compute_eigenmodes()
            
            self.is_loaded = True
            self.needs_load = False
            self._last_path = self.edf_path
            self.playback_idx = 0
            
            print(f"[EigenmodeNode] Loaded: {self.n_channels} channels, "
                  f"{self.total_samples/self.fs:.1f}s, {self.fs:.0f}Hz")
            print(f"[EigenmodeNode] Computed {self.n_modes_compute} eigenmodes")
            
            return True
            
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            print(f"[EigenmodeNode] Load error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_channel_positions(self):
        """Get 2D positions for channels"""
        positions = []
        valid_channels = []
        
        for i, ch in enumerate(self.channel_names):
            ch_upper = ch.upper()
            if ch_upper in CHANNEL_POSITIONS_2D:
                positions.append(CHANNEL_POSITIONS_2D[ch_upper])
                valid_channels.append(i)
            else:
                # Try without numbers
                ch_base = ''.join([c for c in ch_upper if not c.isdigit()])
                if ch_base in CHANNEL_POSITIONS_2D:
                    positions.append(CHANNEL_POSITIONS_2D[ch_base])
                    valid_channels.append(i)
                else:
                    # Assign random position
                    positions.append((np.random.rand(), np.random.rand()))
                    valid_channels.append(i)
        
        self.channel_positions = np.array(positions)
        print(f"[EigenmodeNode] Positioned {len(positions)} channels")
    
    def _compute_eigenmodes(self):
        """
        Compute graph Laplacian eigenmodes from EEG connectivity.
        
        We estimate connectivity from correlation of a short segment,
        then compute the Laplacian L = D - A and its eigenmodes.
        """
        if self.raw is None:
            return
            
        # Use first few seconds to estimate connectivity
        n_samples_for_cov = min(int(5.0 * self.fs), self.total_samples)
        data, _ = self.raw[:, :n_samples_for_cov]
        
        # Compute correlation matrix as connectivity proxy
        # (Real connectivity would use tractography, but we use functional proxy)
        corr = np.corrcoef(data)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Convert to adjacency (threshold small correlations)
        A = np.abs(corr)
        np.fill_diagonal(A, 0)
        
        # Optional: threshold weak connections
        threshold = np.percentile(A, 50)  # Keep top 50%
        A[A < threshold] = 0
        
        # Graph Laplacian: L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Regularize
        L = L + 1e-8 * np.eye(self.n_channels)
        
        # Compute eigenmodes (smallest eigenvalues = slowest modes)
        n_modes = min(self.n_modes_compute, self.n_channels - 1)
        
        try:
            # Use sparse solver for larger matrices
            if self.n_channels > 50:
                L_sparse = csr_matrix(L.astype(np.float32))
                eigenvalues, eigenmodes = eigsh(L_sparse, k=n_modes, which='SM', 
                                                 tol=1e-4, maxiter=3000)
            else:
                # Dense solver for small matrices
                eigenvalues, eigenmodes = np.linalg.eigh(L)
                eigenvalues = eigenvalues[:n_modes]
                eigenmodes = eigenmodes[:, :n_modes]
                
            # Sort by eigenvalue (should already be sorted, but ensure)
            idx = np.argsort(eigenvalues)
            self.eigenvalues = eigenvalues[idx]
            self.eigenmodes = eigenmodes[:, idx]
            
            self.laplacian = L
            self.modes_computed = True
            
            print(f"[EigenmodeNode] Eigenvalue range: {self.eigenvalues[1]:.4f} to {self.eigenvalues[-1]:.4f}")
            
        except Exception as e:
            print(f"[EigenmodeNode] Eigenmode computation failed: {e}")
            self.modes_computed = False
    
    def _get_mode_weights(self, n_modes):
        """Get weights for combining eigenmodes"""
        if self.weight_scheme == 'uniform':
            return np.ones(n_modes) / n_modes
        elif self.weight_scheme == 'linear_decay':
            weights = np.linspace(1.0, 0.1, n_modes)
            return weights / weights.sum()
        elif self.weight_scheme == 'exponential':
            weights = np.exp(-0.2 * np.arange(n_modes))
            return weights / weights.sum()
        elif self.weight_scheme == 'eigenvalue':
            # Weight by inverse eigenvalue (slower modes = more weight)
            if self.eigenvalues is not None:
                weights = 1.0 / (self.eigenvalues[1:n_modes+1] + 1e-6)
                return weights / weights.sum()
        return np.ones(n_modes) / n_modes
    
    def step(self):
        """Main processing step"""
        # Check if we need to reload
        if self.edf_path != self._last_path:
            self.needs_load = True
            
        if self.needs_load and self.edf_path:
            self._load_edf()
            
        if not self.is_loaded or not self.modes_computed:
            return
            
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % max(1, self.update_every) != 0:
            return
            
        # Get gain modulation
        gain_mod = self.get_blended_input('gain_mod', 'sum')
        if gain_mod is None:
            gain_mod = 0.0
        total_gain = self.amplification * (1.0 + gain_mod)
        
        # Get mode selection
        mode_sel = self.get_blended_input('mode_select', 'sum')
        if mode_sel is not None:
            # Map 0-1 to mode range
            mode_sel = np.clip(mode_sel, 0, 1)
            n_available = self.mode_range_end - self.mode_range_start
            selected_mode = self.mode_range_start + int(mode_sel * n_available)
        else:
            selected_mode = None
        
        # Get current window of EEG data
        window_samples = int(self.time_window * self.fs)
        start_idx = int(self.playback_idx)
        end_idx = start_idx + window_samples
        
        if end_idx >= self.total_samples:
            # Loop back
            self.playback_idx = 0
            start_idx = 0
            end_idx = window_samples
            
        data, _ = self.raw[:, start_idx:end_idx]
        
        # Optional band filtering
        if self.band_filter != 'broadband' and self.band_filter in BANDS:
            low, high = BANDS[self.band_filter]
            nyq = self.fs / 2.0
            if high < nyq:
                b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
                for ch in range(data.shape[0]):
                    data[ch] = signal.filtfilt(b, a, data[ch])
        
        # Compute mean activity per channel
        channel_activity = np.mean(data, axis=1) * total_gain
        
        # Store raw signal output
        self.current_raw = float(np.mean(channel_activity))
        
        # === PROJECT ONTO EIGENMODES ===
        # Each mode captures a different "spatial frequency" of brain activity
        mode_start = max(1, self.mode_range_start)  # Skip mode 0 (constant)
        mode_end = min(self.mode_range_end, self.eigenmodes.shape[1])
        n_modes = mode_end - mode_start
        
        if n_modes <= 0:
            return
        
        # Project: activation_i = mode_i^T * channel_activity
        mode_activations = np.zeros(n_modes)
        for i in range(n_modes):
            mode_idx = mode_start + i
            mode_vector = self.eigenmodes[:, mode_idx]
            mode_activations[i] = np.dot(mode_vector, channel_activity)
        
        self.current_mode_activations = mode_activations
        self.mode_history.append(mode_activations.copy())
        
        # === CREATE VISUALIZATIONS ===
        self._create_mode_image(mode_activations, selected_mode)
        self._create_topo_image(channel_activity, mode_activations)
        
        # Advance playback
        self.playback_idx += window_samples * 0.5  # 50% overlap
        
    def _create_mode_image(self, activations, selected_mode=None):
        """Create 2D visualization of mode activations over time"""
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.float32)
        
        # Draw mode history as vertical bars
        history = list(self.mode_history)
        n_history = len(history)
        n_modes = len(activations)
        
        if n_history > 0 and n_modes > 0:
            # Normalize activations
            all_acts = np.array(history)
            act_max = np.abs(all_acts).max() + 1e-6
            
            bar_width = max(1, w // n_history)
            mode_height = h / n_modes
            
            for t, act in enumerate(history):
                x = t * bar_width
                for m, val in enumerate(act):
                    y = int(m * mode_height)
                    y_end = int((m + 1) * mode_height)
                    
                    # Color by sign and magnitude
                    norm_val = val / act_max
                    if norm_val > 0:
                        color = (0, norm_val * 0.8, norm_val)  # Cyan for positive
                    else:
                        color = (-norm_val, 0, -norm_val * 0.3)  # Magenta for negative
                        
                    img[y:y_end, x:x+bar_width] = color
        
        # Highlight selected mode if any
        if selected_mode is not None and n_modes > 0:
            mode_idx = selected_mode - self.mode_range_start
            if 0 <= mode_idx < n_modes:
                y = int(mode_idx * (h / n_modes))
                img[y:y+2, :] = (1, 1, 0)  # Yellow line
        
        # Add labels
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.putText(img_u8, f"Modes {self.mode_range_start}-{self.mode_range_end}", 
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_u8, f"Win: {self.time_window:.2f}s", 
                    (5, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        self.current_mode_image = img_u8
    
    def _create_topo_image(self, channel_activity, mode_activations):
        """Create topographic map of projected activity"""
        if self.channel_positions is None:
            return
            
        h, w = 128, 128
        img = np.zeros((h, w), dtype=np.float32)
        
        # Weight channels by mode activations
        weights = self._get_mode_weights(len(mode_activations))
        
        # Reconstruct activity pattern from modes
        reconstructed = np.zeros(self.n_channels)
        for i, (act, weight) in enumerate(zip(mode_activations, weights)):
            mode_idx = self.mode_range_start + i
            if mode_idx < self.eigenmodes.shape[1]:
                reconstructed += weight * act * self.eigenmodes[:, mode_idx]
        
        # Interpolate to grid
        for ch_idx in range(self.n_channels):
            x, y = self.channel_positions[ch_idx]
            px, py = int(x * (w-1)), int((1-y) * (h-1))  # Flip y
            
            # Gaussian blob for each channel
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        dist = np.sqrt(dx**2 + dy**2)
                        weight = np.exp(-dist**2 / 20)
                        img[ny, nx] += weight * reconstructed[ch_idx]
        
        # Normalize
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        # Apply colormap
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        self.current_topo_image = img_color
    
    def get_output(self, port_name):
        """Return outputs"""
        if not self.is_loaded or self.current_mode_activations is None:
            if port_name in ['dominant_mode', 'mode_energy', 'low_modes', 
                            'high_modes', 'raw_signal', 'eigenvalue_ratio']:
                return 0.0
            return None
        
        acts = self.current_mode_activations
        n_modes = len(acts)
        
        if port_name == 'eigenmode_image':
            return self.current_mode_image
            
        elif port_name == 'mode_topo':
            return self.current_topo_image
            
        elif port_name == 'mode_spectrum':
            return acts.astype(np.float32)
            
        elif port_name == 'mode_complex':
            # Create complex representation for holographic processing
            # Phase encodes mode index, magnitude encodes activation
            n = len(acts)
            phases = np.linspace(0, 2*np.pi, n, endpoint=False)
            complex_spec = acts * np.exp(1j * phases)
            return complex_spec.astype(np.complex64)
            
        elif port_name == 'dominant_mode':
            return float(np.argmax(np.abs(acts)) + self.mode_range_start)
            
        elif port_name == 'mode_energy':
            return float(np.sum(acts**2))
            
        elif port_name == 'low_modes':
            # First 5 modes (slow/global)
            return float(np.sum(acts[:min(5, n)]**2))
            
        elif port_name == 'high_modes':
            # Last modes (fast/local)
            return float(np.sum(acts[max(0, n-5):]**2))
            
        elif port_name == 'raw_signal':
            return self.current_raw
            
        elif port_name == 'eigenvalue_ratio':
            # λ2/λ_max ratio - measure of network connectivity
            if self.eigenvalues is not None and len(self.eigenvalues) > 2:
                return float(self.eigenvalues[1] / (self.eigenvalues[-1] + 1e-6))
            return 0.0
            
        return None
    
    def get_display_image(self):
        """Return display for node preview"""
        if self.current_mode_image is not None:
            img = self.current_mode_image
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        else:
            # Show loading status
            w, h = 128, 64
            img = np.zeros((h, w, 3), dtype=np.uint8)
            
            if self.load_error:
                msg = "ERROR"
                color = (255, 100, 100)
            elif not self.edf_path:
                msg = "No EDF"
                color = (150, 150, 150)
            else:
                msg = "Loading..."
                color = (100, 200, 255)
                
            cv2.putText(img, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1)
            
            img = np.ascontiguousarray(img)
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        """Return configurable options"""
        band_options = [
            ('broadband', 'broadband'),
            ('delta', 'delta'),
            ('theta', 'theta'),
            ('alpha', 'alpha'),
            ('beta', 'beta'),
            ('gamma', 'gamma')
        ]
        
        weight_options = [
            ('uniform', 'uniform'),
            ('linear_decay', 'linear_decay'),
            ('exponential', 'exponential'),
            ('eigenvalue', 'eigenvalue')
        ]
        
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Time Window (s)", "time_window", self.time_window, None),
            ("Mode Range Start", "mode_range_start", self.mode_range_start, None),
            ("Mode Range End", "mode_range_end", self.mode_range_end, None),
            ("Amplification", "amplification", self.amplification, None),
            ("Update Every N Frames", "update_every", self.update_every, None),
            ("Band Filter", "band_filter", self.band_filter, band_options),
            ("Weight Scheme", "weight_scheme", self.weight_scheme, weight_options),
        ]
    
    def close(self):
        """Cleanup"""
        self.raw = None
        self.eigenmodes = None
        self.eigenvalues = None