"""
EEG Phi Analyzer Node
=====================
Loads real EEG data and analyzes its φ-structure in real-time.

This is the critical test: Does BIOLOGICAL signal - before heavy digital
processing - show golden ratio structure in its frequency content?

The EEG was recorded from a real brain. Yes, it passed through an ADC,
but at high sample rates the continuous biological dynamics should 
preserve their natural frequency relationships.

If EEG shows high φ-scores while synthetic digital signals don't,
we have evidence that φ-structure is biological, not artifactual.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import os

# --- PERCEPTION LAB INTEGRATION ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# ----------------------------------

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available. Install with: pip install mne")

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
PHI_POWERS = {i: PHI**i for i in range(-4, 6)}

# EEG frequency bands (the classical ones - note these are close to φ-ratios!)
EEG_BANDS = {
    'delta': (0.5, 4),    # ~4 Hz
    'theta': (4, 8),      # ~6 Hz (4 * 1.5 ≈ 4 * φ^0.58)
    'alpha': (8, 13),     # ~10 Hz (θ * φ ≈ 6 * 1.618 ≈ 10)
    'beta': (13, 30),     # ~20 Hz (α * φ ≈ 10 * 1.618 ≈ 16)
    'gamma': (30, 100),   # ~40 Hz (β * φ ≈ 20 * 1.618 ≈ 32)
}

# Brain regions
EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}


class EEGPhiAnalyzerNode(BaseNode):
    """
    Loads EEG and measures its φ-structure in real-time.
    
    Key insight: EEG frequency bands themselves approximate φ-ratios:
    - Delta ~4 Hz
    - Theta ~6-7 Hz (δ × φ^0.7)
    - Alpha ~10 Hz (θ × φ)  
    - Beta ~16 Hz (α × φ)
    - Gamma ~26 Hz (β × φ)
    
    This node measures whether the ACTUAL peak frequencies in the signal
    follow this pattern, or if it's just how we defined the bands.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "EEG Phi Analyzer"
    NODE_COLOR = QtGui.QColor(180, 120, 200)  # Purple - consciousness research
    
    def __init__(self, edf_file_path=""):
        super().__init__()
        self.node_title = "EEG φ-Analyzer"
        
        # Inputs for external modulation
        self.inputs = {
            'trigger': 'signal',  # External trigger for snapshot
        }
        
        # Outputs
        self.outputs = {
            # Standard band powers
            'delta': 'signal',
            'theta': 'signal', 
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            
            # Raw signal
            'raw_signal': 'signal',
            
            # Phi analysis outputs
            'phi_score': 'signal',        # Overall φ-structure score
            'phi_deviation': 'signal',    # Mean deviation from φ-powers
            'peak_count': 'signal',       # Number of spectral peaks found
            'dominant_freq': 'signal',    # Strongest frequency
            
            # Visualization
            'spectrum_view': 'image',     # Power spectrum with φ-lines
            'phi_history': 'image',       # φ-score over time
            'band_ratios': 'spectrum',    # Actual ratios between bands
        }
        
        # Configuration
        self.edf_file_path = edf_file_path
        self.selected_region = "Occipital"
        self.analysis_window = 2.0  # seconds
        self.phi_tolerance = 0.15   # 15% tolerance for φ-hit
        
        # State
        self._last_path = ""
        self._last_region = ""
        self.raw = None
        self.fs = 256.0  # Higher sample rate for better frequency resolution
        self.current_time = 0.0
        
        # Output values
        self.band_powers = {band: 0.0 for band in EEG_BANDS}
        self.phi_score = 0.0
        self.phi_deviation = 1.0
        self.peak_count = 0
        self.dominant_freq = 0.0
        self.raw_signal = 0.0
        
        # History for visualization
        self.phi_history = np.zeros(128)
        self.spectrum_data = None
        self.peak_frequencies = []
        self.peak_ratios = []
        
        # Display buffers
        self._spectrum_image = None
        self._history_image = None
        
        if not MNE_AVAILABLE:
            self.node_title = "EEG φ (MNE Required!)"
            
    def load_edf(self):
        """Load EEG file."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_file_path):
            self.raw = None
            self.node_title = "EEG φ (No File)"
            return
            
        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            
            # Normalize channel names
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            # Select region
            if self.selected_region != "All":
                region_channels = EEG_REGIONS.get(self.selected_region, [])
                available = [ch for ch in region_channels if ch in raw.ch_names]
                if available:
                    raw.pick_channels(available)
                else:
                    print(f"No channels for region {self.selected_region}, using all")
                    
            # Resample
            raw.resample(self.fs, verbose=False)
            
            self.raw = raw
            self.current_time = 0.0
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            
            duration = raw.n_times / self.fs
            self.node_title = f"EEG φ ({self.selected_region}) {duration:.0f}s"
            print(f"Loaded EEG: {self.edf_file_path}, {raw.ch_names}, {duration:.1f}s")
            
        except Exception as e:
            self.raw = None
            self.node_title = f"EEG φ (Error)"
            print(f"Error loading EEG: {e}")
            
    def compute_phi_score(self, peak_freqs):
        """
        Compute φ-score from peak frequencies.
        
        For each pair of peaks, compute ratio and check if near φ-power.
        """
        if len(peak_freqs) < 2:
            return 0.0, 1.0, []
            
        ratios = []
        for i in range(len(peak_freqs)):
            for j in range(i + 1, len(peak_freqs)):
                if peak_freqs[i] > 0 and peak_freqs[j] > 0:
                    r = max(peak_freqs[i], peak_freqs[j]) / min(peak_freqs[i], peak_freqs[j])
                    ratios.append(r)
                    
        if not ratios:
            return 0.0, 1.0, []
            
        # Check each ratio against φ-powers
        phi_hits = 0
        deviations = []
        
        for r in ratios:
            min_dev = float('inf')
            for phi_p in PHI_POWERS.values():
                if phi_p > 0:
                    dev = abs(r - phi_p) / phi_p
                    min_dev = min(min_dev, dev)
                    
            deviations.append(min_dev)
            if min_dev <= self.phi_tolerance:
                phi_hits += 1
                
        score = phi_hits / len(ratios)
        mean_dev = np.mean(deviations)
        
        return score, mean_dev, ratios
        
    def analyze_spectrum(self, data):
        """
        Compute power spectrum and find peaks.
        Returns: frequencies, powers, peak_indices
        """
        # Compute FFT
        n = len(data)
        freqs = np.fft.rfftfreq(n, 1/self.fs)
        fft = np.fft.rfft(data * np.hanning(n))
        powers = np.abs(fft) ** 2
        
        # Smooth for peak finding
        kernel = np.ones(5) / 5
        powers_smooth = np.convolve(powers, kernel, mode='same')
        
        # Find peaks in relevant range (0.5 - 60 Hz)
        freq_mask = (freqs >= 0.5) & (freqs <= 60)
        valid_powers = np.zeros_like(powers_smooth)
        valid_powers[freq_mask] = powers_smooth[freq_mask]
        
        # Normalize
        if valid_powers.max() > 0:
            valid_powers = valid_powers / valid_powers.max()
            
        # Find peaks
        peak_indices, properties = find_peaks(
            valid_powers,
            height=0.1,
            prominence=0.05,
            distance=int(0.5 * n / self.fs)  # Min 0.5 Hz apart
        )
        
        return freqs, powers, peak_indices
        
    def step(self):
        """Main processing step."""
        # Check for config changes
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()
            
        if self.raw is None:
            return
            
        # Get data window
        start_sample = int(self.current_time * self.fs)
        window_samples = int(self.analysis_window * self.fs)
        end_sample = start_sample + window_samples
        
        # Loop if at end
        if end_sample >= self.raw.n_times:
            self.current_time = 0.0
            start_sample = 0
            end_sample = window_samples
            
        # Get data
        data, _ = self.raw[:, start_sample:end_sample]
        
        # Average channels
        if data.ndim > 1:
            data = np.mean(data, axis=0)
            
        if data.size == 0:
            return
            
        # Raw signal output (normalized)
        self.raw_signal = float(np.mean(data) * 1e6)  # Convert to µV scale
        
        # === SPECTRAL ANALYSIS ===
        freqs, powers, peak_indices = self.analyze_spectrum(data)
        
        self.spectrum_data = (freqs, powers)
        self.peak_frequencies = freqs[peak_indices].tolist()
        self.peak_count = len(peak_indices)
        
        # Dominant frequency
        if len(peak_indices) > 0:
            dominant_idx = peak_indices[np.argmax(powers[peak_indices])]
            self.dominant_freq = freqs[dominant_idx]
        else:
            self.dominant_freq = 0.0
            
        # === BAND POWERS ===
        nyq = self.fs / 2.0
        for band, (low, high) in EEG_BANDS.items():
            try:
                # Bandpass filter
                b, a = signal.butter(4, [low/nyq, min(high/nyq, 0.99)], btype='band')
                filtered = signal.filtfilt(b, a, data)
                power = np.log1p(np.mean(filtered**2) * 1e12)  # Scale to reasonable range
                # Smooth
                self.band_powers[band] = self.band_powers[band] * 0.7 + power * 0.3
            except:
                pass
                
        # === PHI ANALYSIS ===
        self.phi_score, self.phi_deviation, self.peak_ratios = self.compute_phi_score(
            self.peak_frequencies
        )
        
        # Update history
        self.phi_history[:-1] = self.phi_history[1:]
        self.phi_history[-1] = self.phi_score
        
        # === GENERATE VISUALIZATIONS ===
        self._generate_spectrum_image(freqs, powers, peak_indices)
        self._generate_history_image()
        
        # Advance time
        self.current_time += 1.0 / 30.0  # ~30fps
        
    def _generate_spectrum_image(self, freqs, powers, peak_indices):
        """Generate spectrum visualization with φ-power lines."""
        w, h = 128, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Frequency range: 0-60 Hz mapped to width
        max_freq = 60.0
        
        # Normalize powers for display
        valid_mask = freqs <= max_freq
        display_powers = np.log1p(powers[valid_mask])
        if display_powers.max() > 0:
            display_powers = display_powers / display_powers.max()
            
        # Draw spectrum (cyan)
        for i, (f, p) in enumerate(zip(freqs[valid_mask], display_powers)):
            x = int(f / max_freq * (w - 1))
            y = int(p * (h - 1))
            if 0 <= x < w and y > 0:
                img[h-y:h, x, 1] = 200  # Green channel
                img[h-y:h, x, 2] = 255  # Blue channel (cyan)
                
        # Draw φ-power frequency lines (gold) - if dominant freq found
        if self.dominant_freq > 0:
            base = self.dominant_freq
            for p in range(-2, 4):
                phi_freq = base * (PHI ** p)
                if 0 < phi_freq < max_freq:
                    x = int(phi_freq / max_freq * (w - 1))
                    img[:, x, 0] = 200  # Red
                    img[:, x, 1] = 180  # Green (gold color)
                    
        # Mark peaks (yellow dots)
        for idx in peak_indices:
            if freqs[idx] <= max_freq:
                x = int(freqs[idx] / max_freq * (w - 1))
                y = int(display_powers[idx] * (h - 1)) if idx < len(display_powers) else 0
                # Draw marker
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if 0 <= x+dx < w and 0 <= h-1-y+dy < h:
                            img[h-1-y+dy, x+dx] = [0, 255, 255]  # Yellow
                            
        # Draw φ-score bar at bottom
        score_width = int(self.phi_score * w)
        img[h-3:h, :score_width, 1] = 255  # Green bar
        
        self._spectrum_image = img
        
    def _generate_history_image(self):
        """Generate φ-score history visualization."""
        w, h = 128, 32
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw history line
        for i in range(len(self.phi_history) - 1):
            x = int(i * w / len(self.phi_history))
            y = int(self.phi_history[i] * (h - 1))
            
            # Color based on score (red=low, green=high)
            r = int(255 * (1 - self.phi_history[i]))
            g = int(255 * self.phi_history[i])
            
            if y > 0:
                img[h-y:h, x, 0] = r
                img[h-y:h, x, 1] = g
                
        # Draw 0.5 threshold line
        mid_y = h // 2
        img[mid_y, :, :] = [50, 50, 50]
        
        self._history_image = img
        
    def get_output(self, port_name):
        """Return output values."""
        if port_name in self.band_powers:
            return self.band_powers[port_name]
        elif port_name == 'raw_signal':
            return self.raw_signal
        elif port_name == 'phi_score':
            return self.phi_score
        elif port_name == 'phi_deviation':
            return self.phi_deviation
        elif port_name == 'peak_count':
            return float(self.peak_count)
        elif port_name == 'dominant_freq':
            return self.dominant_freq
        elif port_name == 'spectrum_view':
            return self._spectrum_image
        elif port_name == 'phi_history':
            return self._history_image
        elif port_name == 'band_ratios':
            return np.array(self.peak_ratios) if self.peak_ratios else np.array([0.0])
        return None
        
    def get_display_image(self):
        """Return combined visualization for node face."""
        if self._spectrum_image is None:
            # Return placeholder
            img = np.zeros((64, 128, 3), dtype=np.uint8)
            return QtGui.QImage(img.data, 128, 64, 384, QtGui.QImage.Format.Format_RGB888)
            
        # Combine spectrum and history
        h1, w1 = self._spectrum_image.shape[:2]
        h2, w2 = self._history_image.shape[:2] if self._history_image is not None else (32, 128)
        
        combined = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        combined[:h1, :w1] = self._spectrum_image
        if self._history_image is not None:
            combined[h1:h1+h2, :w2] = self._history_image
            
        # Add text overlay for score
        # (Simple approach - draw score value in pixels)
        score_str = f"{self.phi_score:.2f}"
        
        combined = np.ascontiguousarray(combined)
        h, w = combined.shape[:2]
        
        qimg = QtGui.QImage(combined.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = combined  # Prevent garbage collection
        return qimg
        
    def get_config_options(self):
        """Configuration dialog options."""
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("Analysis Window (s)", "analysis_window", self.analysis_window, 'float'),
            ("φ Tolerance", "phi_tolerance", self.phi_tolerance, 'float'),
        ]
        
    def close(self):
        """Cleanup."""
        self.raw = None
        super().close()


# === STANDALONE TEST ===
if __name__ == "__main__":
    import numpy as np
    PHI = (1 + np.sqrt(5)) / 2
    PHI_POWERS = {i: PHI**i for i in range(-4, 6)}
    
    print("EEG Phi Analyzer Node")
    print("=" * 50)
    print(f"Golden Ratio φ = {PHI:.6f}")
    print()
    print("Classical EEG bands and their φ-relationships:")
    print()
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    centers = [2, 6, 10, 20, 40]  # Approximate center frequencies
    
    for i in range(len(bands) - 1):
        ratio = centers[i+1] / centers[i]
        closest_phi = min(PHI_POWERS.values(), key=lambda p: abs(p - ratio) if p > 0 else float('inf'))
        phi_power = [k for k, v in PHI_POWERS.items() if v == closest_phi][0]
        deviation = abs(ratio - closest_phi) / closest_phi * 100
        
        print(f"  {bands[i]:>6} → {bands[i+1]:<6}: {centers[i]:>2} → {centers[i+1]:>2} Hz")
        print(f"         Ratio: {ratio:.3f}, Nearest φ^{phi_power}: {closest_phi:.3f}, Dev: {deviation:.1f}%")
        print()
        
    print("If EEG bands evolved to follow φ-ratios, the brain is a φ-computer.")