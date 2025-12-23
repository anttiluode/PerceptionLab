"""
HolographicInterferenceNode - Pribram's Dream
==============================================

"The smallest unit is not the neuron. It's the interference fringe."

Karl Pribram proposed that the brain stores information holographically -
as interference patterns between waves at every scale. We've been stuck
on 4-5 frequency bands (delta, theta, alpha, beta, gamma) like they're
fundamental. They're not. They're just the peaks we named.

The actual EEG contains a CONTINUOUS spectrum. Every frequency interferes
with every other frequency. The number of interference patterns scales
as N*(N-1)/2 where N is the number of frequencies.

This node goes to town:
- Decomposes EEG into HUNDREDS of frequency bins (not 4, not 16, but 256+)
- Computes ALL pairwise interference patterns
- Renders the result as a massive holographic field (up to 4K/8K)
- The output IS the hologram - the interference pattern of your brain's waves

With 256 frequency bins, we get 32,640 unique interference pairs.
Each pair creates a beat frequency and a phase relationship.
Together they form a holographic encoding of the original signal.

WHY THIS MIGHT NOT BE RIDICULOUS:
- Holography requires reference + signal beam interference
- EEG bands naturally provide multiple "beams" at different frequencies
- The brain's dendritic trees ARE doing this computation spatially
- We just never look at it this way because we collapse to 5 bands

INPUTS:
- raw_eeg: Raw EEG signal (or use internal file)
- resolution_scale: How large to make the output (1=1K, 2=2K, 4=4K, 8=8K)
- freq_resolution: How many frequency bins (64, 128, 256, 512)
- interference_mode: How to compute interference ('beat', 'phase', 'complex')
- time_window: Analysis window in seconds

OUTPUTS:
- hologram: The massive interference pattern image
- spectrum: The frequency decomposition
- beat_matrix: The matrix of beat frequencies
- phase_matrix: The matrix of phase relationships
- dominant_interference: Strongest interference pair
- total_energy: Total spectral energy
- complexity: Measure of interference pattern complexity

Created: December 2025
For Antti's quest to find what's really in the signal
"""

import numpy as np
import cv2
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert, stft
from scipy.ndimage import gaussian_filter, zoom
import os

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


class HolographicInterferenceNode2(BaseNode):
    """
    Decomposes EEG into massive frequency spread and computes
    all pairwise interference patterns as a holographic field.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Holographic Interference"
    NODE_COLOR = QtGui.QColor(255, 0, 255)  # Magenta - beyond the visible
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'raw_eeg': 'signal',           # External EEG signal
            'resolution_scale': 'signal',   # 1=1K, 2=2K, 4=4K, 8=8K
            'freq_resolution': 'signal',    # Number of frequency bins
            'time_window': 'signal',        # Analysis window
            'reference_freq': 'signal',     # Reference beam frequency (0=auto)
            'reset': 'signal'
        }
        
        self.outputs = {
            # Main outputs
            'hologram': 'image',            # The massive interference field
            'hologram_small': 'image',      # Downsampled for preview
            'spectrum_image': 'image',      # Frequency decomposition visual
            
            # Matrices
            'beat_matrix': 'image',         # Beat frequencies between all pairs
            'phase_matrix': 'image',        # Phase relationships
            'coherence_matrix': 'image',    # Coherence between frequency pairs
            
            # Signals
            'dominant_beat': 'signal',      # Strongest beat frequency
            'total_energy': 'signal',       # Total spectral energy
            'complexity': 'signal',         # Entropy of interference pattern
            'peak_frequency': 'signal',     # Dominant frequency
            'n_interferences': 'signal',    # Number of interference pairs computed
            
            # Spectrum for downstream
            'full_spectrum': 'spectrum'     # All frequency bin powers
        }
        
        # === CONFIGURATION ===
        self.edf_path = ""
        self.selected_region = "All"
        
        # Resolution settings
        self.base_resolution = 1024        # Base output size (1K)
        self.resolution_scale = 1          # Multiplier (1, 2, 4, 8)
        self.output_resolution = 1024      # Actual output size
        
        # Frequency decomposition settings
        self.n_freq_bins = 256             # Number of frequency bins
        self.freq_min = 0.5                # Minimum frequency Hz
        self.freq_max = 100.0              # Maximum frequency Hz (beyond gamma!)
        self.freq_spacing = 'log'          # 'linear' or 'log' spacing
        
        # Interference computation
        self.interference_mode = 'complex' # 'beat', 'phase', 'complex'
        self.use_hilbert = True            # Use analytic signal for phase
        
        # Time parameters
        self.window_size = 2.0             # Seconds
        self.fs = 256.0                    # Sampling rate
        
        # === STATE ===
        # EEG data
        self.raw_mne = None
        self.eeg_buffer = np.zeros(int(self.fs * self.window_size))
        self.current_time = 0.0
        
        # Frequency analysis
        self.freq_bins = np.logspace(np.log10(self.freq_min), 
                                      np.log10(self.freq_max), 
                                      self.n_freq_bins)
        self.freq_amplitudes = np.zeros(self.n_freq_bins)
        self.freq_phases = np.zeros(self.n_freq_bins)
        
        # Interference matrices
        self.beat_matrix = np.zeros((self.n_freq_bins, self.n_freq_bins))
        self.phase_matrix = np.zeros((self.n_freq_bins, self.n_freq_bins))
        self.coherence_matrix = np.zeros((self.n_freq_bins, self.n_freq_bins))
        
        # Output fields
        self.hologram = np.zeros((self.output_resolution, self.output_resolution))
        self.hologram_small = np.zeros((256, 256))
        
        # Metrics
        self.dominant_beat = 0.0
        self.total_energy = 0.0
        self.complexity = 0.0
        self.peak_frequency = 0.0
        
        # Precompute interference basis
        self._precompute_interference_basis()
        
        self._last_path = ""
        self.t = 0
    
    def _precompute_interference_basis(self):
        """
        Precompute the spatial basis functions for interference patterns.
        Each frequency pair creates a specific spatial pattern.
        """
        # For efficiency, we compute patterns at lower resolution and upsample
        basis_res = min(256, self.output_resolution)
        
        # Coordinate grids
        x = np.linspace(-1, 1, basis_res)
        y = np.linspace(-1, 1, basis_res)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.THETA = np.arctan2(self.Y, self.X)
        
        # Precompute some common patterns
        # (full precomputation of all pairs would use too much memory)
        self.radial_basis = np.exp(-self.R**2 * 2)  # Gaussian envelope
    
    def _update_freq_bins(self):
        """Update frequency bin array based on settings."""
        if self.freq_spacing == 'log':
            self.freq_bins = np.logspace(
                np.log10(max(0.1, self.freq_min)), 
                np.log10(self.freq_max), 
                self.n_freq_bins
            )
        else:
            self.freq_bins = np.linspace(self.freq_min, self.freq_max, self.n_freq_bins)
        
        # Resize matrices
        self.freq_amplitudes = np.zeros(self.n_freq_bins)
        self.freq_phases = np.zeros(self.n_freq_bins)
        self.beat_matrix = np.zeros((self.n_freq_bins, self.n_freq_bins))
        self.phase_matrix = np.zeros((self.n_freq_bins, self.n_freq_bins))
        self.coherence_matrix = np.zeros((self.n_freq_bins, self.n_freq_bins))
    
    def _load_edf(self):
        """Load EEG file."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_path):
            return False
        
        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            raw.resample(self.fs, verbose=False)
            self.raw_mne = raw
            self._last_path = self.edf_path
            self.current_time = 0.0
            print(f"[Holographic] Loaded: {self.edf_path}")
            return True
        except Exception as e:
            print(f"[Holographic] Load error: {e}")
            return False
    
    def _get_eeg_window(self):
        """Get current EEG window from file or input."""
        if self.raw_mne is not None:
            start = int(self.current_time * self.fs)
            end = start + int(self.window_size * self.fs)
            
            if end >= self.raw_mne.n_times:
                self.current_time = 0.0
                start = 0
                end = int(self.window_size * self.fs)
            
            data, _ = self.raw_mne[:, start:end]
            
            # Average across channels
            if data.ndim > 1:
                data = np.mean(data, axis=0)
            
            # Advance time
            self.current_time += 1.0 / 30.0
            
            return data
        
        return self.eeg_buffer
    
    def _decompose_frequencies(self, signal):
        """
        Decompose signal into frequency bins using filter bank.
        Extract amplitude and phase for each bin.
        """
        n = len(signal)
        if n < 10:
            return
        
        nyq = self.fs / 2.0
        
        for i, freq in enumerate(self.freq_bins):
            # Define narrow bandpass around this frequency
            bw = max(0.5, freq * 0.1)  # 10% bandwidth, min 0.5 Hz
            low = max(0.1, freq - bw/2) / nyq
            high = min(0.99, (freq + bw/2) / nyq)
            
            if low >= high or low <= 0 or high >= 1:
                self.freq_amplitudes[i] = 0
                self.freq_phases[i] = 0
                continue
            
            try:
                b, a = butter(2, [low, high], btype='band')
                filtered = filtfilt(b, a, signal)
                
                # Get amplitude
                self.freq_amplitudes[i] = np.sqrt(np.mean(filtered**2))
                
                # Get phase using Hilbert transform
                if self.use_hilbert and len(filtered) > 10:
                    analytic = hilbert(filtered)
                    self.freq_phases[i] = np.angle(np.mean(analytic))
                else:
                    self.freq_phases[i] = 0.0
                    
            except Exception:
                self.freq_amplitudes[i] = 0
                self.freq_phases[i] = 0
    
    def _compute_interference_matrices(self):
        """
        Compute all pairwise interference patterns.
        This is O(N^2) but we're going for it.
        """
        n = self.n_freq_bins
        
        for i in range(n):
            for j in range(i+1, n):
                # Beat frequency
                beat = abs(self.freq_bins[i] - self.freq_bins[j])
                self.beat_matrix[i, j] = beat
                self.beat_matrix[j, i] = beat
                
                # Phase difference
                phase_diff = self.freq_phases[i] - self.freq_phases[j]
                self.phase_matrix[i, j] = phase_diff
                self.phase_matrix[j, i] = -phase_diff
                
                # Coherence (product of amplitudes, weighted by phase alignment)
                coherence = (self.freq_amplitudes[i] * self.freq_amplitudes[j] * 
                            np.cos(phase_diff / 2)**2)
                self.coherence_matrix[i, j] = coherence
                self.coherence_matrix[j, i] = coherence
    
    def _render_hologram(self):
        """
        Render the holographic interference pattern.
        Each frequency pair contributes a spatial pattern.
        """
        # Work at basis resolution, then upsample
        basis_res = min(256, self.output_resolution)
        hologram = np.zeros((basis_res, basis_res))
        
        n = self.n_freq_bins
        
        # Limit pairs for performance (top coherence pairs)
        flat_coherence = self.coherence_matrix.flatten()
        if np.max(flat_coherence) > 0:
            threshold = np.percentile(flat_coherence[flat_coherence > 0], 90)
        else:
            threshold = 0
        
        pair_count = 0
        max_pairs = 500  # Limit for performance
        
        for i in range(n):
            for j in range(i+1, n):
                if self.coherence_matrix[i, j] < threshold:
                    continue
                if pair_count >= max_pairs:
                    break
                
                # Get interference parameters
                f1, f2 = self.freq_bins[i], self.freq_bins[j]
                a1, a2 = self.freq_amplitudes[i], self.freq_amplitudes[j]
                p1, p2 = self.freq_phases[i], self.freq_phases[j]
                
                beat = abs(f1 - f2)
                phase_diff = p1 - p2
                amp = np.sqrt(a1 * a2)
                
                if amp < 1e-10:
                    continue
                
                # Create spatial interference pattern
                # This is where the holography happens:
                # Two "beams" at different frequencies create fringes
                
                # Spatial frequency of fringes (proportional to beat frequency)
                k = beat * 0.5  # Scale factor for visualization
                
                # Angle based on frequency ratio (creates different orientations)
                angle = (f1 / f2) * np.pi
                
                # The interference pattern
                pattern = amp * np.cos(
                    k * (self.X * np.cos(angle) + self.Y * np.sin(angle)) * 10 +
                    phase_diff +
                    (f1 + f2) * self.R * 0.5  # Radial component
                )
                
                # Apply envelope
                pattern *= self.radial_basis
                
                hologram += pattern
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        # Normalize
        if hologram.max() != hologram.min():
            hologram = (hologram - hologram.min()) / (hologram.max() - hologram.min())
        
        # Upsample to output resolution if needed
        if self.output_resolution > basis_res:
            scale = self.output_resolution / basis_res
            hologram = zoom(hologram, scale, order=1)
        
        self.hologram = hologram
        
        # Create small preview
        if self.output_resolution > 256:
            self.hologram_small = cv2.resize(hologram, (256, 256))
        else:
            self.hologram_small = hologram.copy()
    
    def _compute_metrics(self):
        """Compute output metrics."""
        # Find dominant beat frequency
        max_idx = np.unravel_index(np.argmax(self.coherence_matrix), 
                                    self.coherence_matrix.shape)
        if max_idx[0] != max_idx[1]:
            self.dominant_beat = abs(self.freq_bins[max_idx[0]] - 
                                     self.freq_bins[max_idx[1]])
        
        # Total spectral energy
        self.total_energy = np.sum(self.freq_amplitudes**2)
        
        # Peak frequency
        self.peak_frequency = self.freq_bins[np.argmax(self.freq_amplitudes)]
        
        # Complexity (entropy of hologram)
        hist, _ = np.histogram(self.hologram.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        self.complexity = -np.sum(hist * np.log2(hist + 1e-10))
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        raw_in = self.get_blended_input('raw_eeg', 'sum')
        res_scale = self.get_blended_input('resolution_scale', 'sum')
        freq_res = self.get_blended_input('freq_resolution', 'sum')
        time_win = self.get_blended_input('time_window', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.current_time = 0.0
            return
        
        # Update settings from inputs
        if res_scale is not None:
            new_scale = int(np.clip(res_scale, 1, 8))
            if new_scale != self.resolution_scale:
                self.resolution_scale = new_scale
                self.output_resolution = self.base_resolution * new_scale
                self._precompute_interference_basis()
        
        if freq_res is not None:
            new_n = int(np.clip(freq_res, 32, 512))
            if new_n != self.n_freq_bins:
                self.n_freq_bins = new_n
                self._update_freq_bins()
        
        # Load EEG file if path changed
        if self.edf_path and self.edf_path != self._last_path:
            self._load_edf()
        
        # Get EEG data
        if raw_in is not None:
            # External input - add to buffer
            self.eeg_buffer = np.roll(self.eeg_buffer, -1)
            self.eeg_buffer[-1] = raw_in
            signal = self.eeg_buffer
        else:
            signal = self._get_eeg_window()
        
        if signal is None or len(signal) < 10:
            return
        
        # === DECOMPOSE INTO FREQUENCIES ===
        self._decompose_frequencies(signal)
        
        # === COMPUTE INTERFERENCE ===
        self._compute_interference_matrices()
        
        # === RENDER HOLOGRAM ===
        self._render_hologram()
        
        # === COMPUTE METRICS ===
        self._compute_metrics()
    
    def get_output(self, port_name):
        if port_name == 'hologram':
            return (self.hologram * 255).astype(np.uint8)
        
        elif port_name == 'hologram_small':
            return (self.hologram_small * 255).astype(np.uint8)
        
        elif port_name == 'spectrum_image':
            # Visualize frequency decomposition
            h = 128
            w = min(512, self.n_freq_bins)
            img = np.zeros((h, w), dtype=np.uint8)
            
            if self.freq_amplitudes.max() > 0:
                amps_norm = self.freq_amplitudes / self.freq_amplitudes.max()
                # Resample if needed
                if len(amps_norm) != w:
                    amps_norm = np.interp(
                        np.linspace(0, len(amps_norm)-1, w),
                        np.arange(len(amps_norm)),
                        amps_norm
                    )
                
                for i, a in enumerate(amps_norm):
                    bar_h = int(a * (h - 10))
                    img[h-bar_h:h, i] = 200
            
            return img
        
        elif port_name == 'beat_matrix':
            mat = self.beat_matrix.copy()
            if mat.max() > 0:
                mat = mat / mat.max()
            # Resize for visibility
            mat_vis = cv2.resize(mat, (256, 256))
            return (mat_vis * 255).astype(np.uint8)
        
        elif port_name == 'phase_matrix':
            mat = (self.phase_matrix + np.pi) / (2 * np.pi)
            mat_vis = cv2.resize(mat, (256, 256))
            return (mat_vis * 255).astype(np.uint8)
        
        elif port_name == 'coherence_matrix':
            mat = self.coherence_matrix.copy()
            if mat.max() > 0:
                mat = mat / mat.max()
            mat_vis = cv2.resize(mat, (256, 256))
            return (mat_vis * 255).astype(np.uint8)
        
        elif port_name == 'dominant_beat':
            return float(self.dominant_beat)
        
        elif port_name == 'total_energy':
            return float(self.total_energy)
        
        elif port_name == 'complexity':
            return float(self.complexity)
        
        elif port_name == 'peak_frequency':
            return float(self.peak_frequency)
        
        elif port_name == 'n_interferences':
            n = self.n_freq_bins
            return float(n * (n - 1) / 2)
        
        elif port_name == 'full_spectrum':
            return self.freq_amplitudes.astype(np.float32)
        
        return None
    
    def get_display_image(self):
        # Composite display: hologram + spectrum + info
        h, w = 256, 384
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Hologram (left 256x256)
        holo_vis = (self.hologram_small * 255).astype(np.uint8)
        holo_color = cv2.applyColorMap(holo_vis, cv2.COLORMAP_TWILIGHT_SHIFTED)
        display[:256, :256] = holo_color
        
        # Spectrum (right panel)
        spec_h = 100
        if self.freq_amplitudes.max() > 0:
            amps_norm = self.freq_amplitudes / self.freq_amplitudes.max()
            bar_w = 128 // len(amps_norm) if len(amps_norm) < 128 else 1
            for i in range(min(128, len(amps_norm))):
                idx = int(i * len(amps_norm) / 128)
                bar_h = int(amps_norm[idx] * spec_h)
                x = 256 + i
                display[spec_h-bar_h:spec_h, x] = [100, 255, 100]
        
        # Info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, f"Bins: {self.n_freq_bins}", (260, 120), 
                   font, 0.35, (255,255,255), 1)
        cv2.putText(display, f"Pairs: {int(self.n_freq_bins*(self.n_freq_bins-1)/2)}", 
                   (260, 140), font, 0.35, (255,255,255), 1)
        cv2.putText(display, f"Peak: {self.peak_frequency:.1f}Hz", (260, 160), 
                   font, 0.35, (255,255,255), 1)
        cv2.putText(display, f"Beat: {self.dominant_beat:.1f}Hz", (260, 180), 
                   font, 0.35, (255,255,255), 1)
        cv2.putText(display, f"Complex: {self.complexity:.2f}", (260, 200), 
                   font, 0.35, (255,255,255), 1)
        cv2.putText(display, f"Res: {self.output_resolution}", (260, 220), 
                   font, 0.35, (200,200,200), 1)
        
        # Title
        cv2.putText(display, "HOLOGRAPHIC", (5, 15), font, 0.4, (255,0,255), 1)
        cv2.putText(display, "INTERFERENCE", (5, 30), font, 0.4, (255,0,255), 1)
        
        return QtGui.QImage(display.data, w, h, w * 3, 
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        spacing_opts = [('log', 'log'), ('linear', 'linear')]
        mode_opts = [('complex', 'complex'), ('beat', 'beat'), ('phase', 'phase')]
        region_opts = [
            ('All', 'All'),
            ('Occipital', 'Occipital'),
            ('Temporal', 'Temporal'),
            ('Parietal', 'Parietal'),
            ('Frontal', 'Frontal'),
            ('Central', 'Central')
        ]
        
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_opts),
            ("Frequency Bins", "n_freq_bins", self.n_freq_bins, None),
            ("Min Frequency (Hz)", "freq_min", self.freq_min, None),
            ("Max Frequency (Hz)", "freq_max", self.freq_max, None),
            ("Frequency Spacing", "freq_spacing", self.freq_spacing, spacing_opts),
            ("Resolution Scale (1-8)", "resolution_scale", self.resolution_scale, None),
            ("Window Size (s)", "window_size", self.window_size, None),
            ("Interference Mode", "interference_mode", self.interference_mode, mode_opts),
            ("Use Hilbert Phase", "use_hilbert", self.use_hilbert, 
             [('True', True), ('False', False)]),
        ]
    
    def set_config_options(self, options):
        rebuild = False
        for key, value in options.items():
            if key == 'n_freq_bins':
                new_n = int(value)
                if new_n != self.n_freq_bins:
                    self.n_freq_bins = new_n
                    rebuild = True
            elif key == 'resolution_scale':
                new_scale = int(np.clip(float(value), 1, 8))
                if new_scale != self.resolution_scale:
                    self.resolution_scale = new_scale
                    self.output_resolution = self.base_resolution * new_scale
                    self._precompute_interference_basis()
            elif key in ('freq_min', 'freq_max', 'freq_spacing'):
                old_val = getattr(self, key)
                if key in ('freq_min', 'freq_max'):
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)
                if old_val != getattr(self, key):
                    rebuild = True
            elif hasattr(self, key):
                if isinstance(getattr(self, key), bool):
                    setattr(self, key, value in (True, 'True', 'true', '1', 1))
                elif isinstance(getattr(self, key), float):
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)
        
        if rebuild:
            self._update_freq_bins()