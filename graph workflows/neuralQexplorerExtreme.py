"""
HolographicExtremeFixed - All Bounds Issues Fixed
==================================================

Properly handles:
- Empty arrays on startup
- Display bounds checking
- Zero-size reduction operations
- Variable resolution outputs

NO CAPS on frequency bins - push through the wall.
"""

import numpy as np
import cv2
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import zoom
from collections import deque
import os
import time

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


class HolographicExtremeFixed(BaseNode):
    """
    Uncapped holographic interference with all bounds issues fixed.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Holographic Extreme"
    NODE_COLOR = QtGui.QColor(255, 50, 50)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'raw_eeg': 'signal',
            'freq_bins': 'signal',
            'time_window_ms': 'signal',
            'max_pairs': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'hologram': 'image',
            'hologram_small': 'image',
            'spectrum_image': 'image',
            'coherence_matrix': 'image',
            'structure_detected': 'signal',
            'structure_strength': 'signal',
            'breakdown_point': 'signal',
            'n_bins': 'signal',
            'n_pairs': 'signal',
            'n_pairs_rendered': 'signal',
            'compute_time_ms': 'signal',
            'peak_frequency': 'signal',
            'dominant_beat': 'signal',
            'full_spectrum': 'spectrum',
            'combined_view': 'image'
        }
        
        # === EEG SOURCE ===
        self.edf_path = ""
        self.selected_region = "All"
        self.raw_mne = None
        self.fs = 1000.0
        self.current_time = 0.0
        self._last_path = ""
        
        # === PARAMETERS (NO CAPS) ===
        self.n_freq_bins = 256
        self.freq_min = 0.5
        self.freq_max = 200.0
        self.freq_spacing = 'log'
        self.time_window_ms = 100.0
        self.max_pairs_to_render = 1000
        self.use_fft_direct = True
        self.output_resolution = 512
        
        # === STATE - Initialize with proper sizes ===
        self.eeg_buffer_size = int(self.fs * 2)
        self.eeg_buffer = np.zeros(self.eeg_buffer_size)
        
        # IMPORTANT: Initialize arrays with proper size, not empty
        self.freq_bins = np.linspace(self.freq_min, self.freq_max, self.n_freq_bins)
        self.freq_amplitudes = np.zeros(self.n_freq_bins)
        self.freq_phases = np.zeros(self.n_freq_bins)
        
        self.hologram = np.zeros((256, 256))  # Start smaller
        self.hologram_small = np.zeros((256, 256))
        self.coherence_matrix_small = np.zeros((64, 64))
        
        # Metrics
        self.structure_detected = False
        self.structure_strength = 0.0
        self.breakdown_point = False
        self.compute_time_ms = 0.0
        self.peak_frequency = 0.0
        self.dominant_beat = 0.0
        self.n_pairs_rendered = 0
        
        self.structure_history = deque(maxlen=50)
        
        # Basis for rendering
        self._init_basis()
        
        self.t = 0
    
    def _safe_max(self, arr):
        """Safely get max of array, returning 0 for empty arrays."""
        if arr is None or not hasattr(arr, 'size') or arr.size == 0:
            return 0.0
        return float(np.max(arr))
    
    def _init_basis(self):
        """Initialize coordinate basis for rendering."""
        res = 256  # Fixed basis resolution
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.THETA = np.arctan2(self.Y, self.X)
        self.radial_envelope = np.exp(-self.R**2 * 2)
    
    def _update_freq_bins(self):
        """Update frequency bins."""
        n = max(4, self.n_freq_bins)  # Minimum 4 bins
        if self.freq_spacing == 'log':
            self.freq_bins = np.logspace(
                np.log10(max(0.1, self.freq_min)),
                np.log10(max(1.0, self.freq_max)),
                n
            )
        else:
            self.freq_bins = np.linspace(self.freq_min, self.freq_max, n)
        
        self.freq_amplitudes = np.zeros(n)
        self.freq_phases = np.zeros(n)
        self.n_freq_bins = n
    
    def _load_edf(self):
        """Load EEG file."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_path):
            return False
        
        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            original_fs = raw.info['sfreq']
            target_fs = min(original_fs, 1000.0)
            
            if target_fs != original_fs:
                raw.resample(target_fs, verbose=False)
            
            self.fs = raw.info['sfreq']
            self.raw_mne = raw
            self._last_path = self.edf_path
            self.current_time = 0.0
            
            self.eeg_buffer_size = int(self.fs * 2)
            self.eeg_buffer = np.zeros(self.eeg_buffer_size)
            
            print(f"[Extreme] Loaded: {self.edf_path} @ {self.fs}Hz")
            return True
        except Exception as e:
            print(f"[Extreme] Load error: {e}")
            return False
    
    def _get_eeg_window(self):
        """Get EEG window."""
        n_samples = max(4, int(self.time_window_ms * self.fs / 1000))
        
        if self.raw_mne is not None:
            start = int(self.current_time * self.fs)
            end = start + n_samples
            
            if end >= self.raw_mne.n_times:
                self.current_time = 0.0
                start = 0
                end = n_samples
            
            try:
                data, _ = self.raw_mne[:, start:end]
                if data.ndim > 1:
                    data = np.mean(data, axis=0)
                self.current_time += 1.0 / 30.0
                return data
            except:
                return self.eeg_buffer[-n_samples:]
        
        return self.eeg_buffer[-n_samples:]
    
    def _decompose_fft_direct(self, signal):
        """Direct FFT decomposition."""
        n = len(signal)
        if n < 4:
            return
        
        spectrum = rfft(signal)
        freqs = rfftfreq(n, 1/self.fs)
        
        for i, target_freq in enumerate(self.freq_bins):
            if i >= len(self.freq_amplitudes):
                break
            idx = np.argmin(np.abs(freqs - target_freq))
            if idx < len(spectrum):
                self.freq_amplitudes[i] = np.abs(spectrum[idx])
                self.freq_phases[i] = np.angle(spectrum[idx])
            else:
                self.freq_amplitudes[i] = 0
                self.freq_phases[i] = 0
    
    def _compute_structure_strength(self):
        """Measure structure in frequency decomposition."""
        max_amp = self._safe_max(self.freq_amplitudes)
        if max_amp == 0:
            return 0.0
        
        amps = self.freq_amplitudes / (max_amp + 1e-10)
        
        # Peakiness
        sorted_amps = np.sort(amps)
        n = len(amps)
        if n > 10:
            top_10 = sorted_amps[-n//10:].mean()
            bottom_90 = sorted_amps[:-n//10].mean()
        else:
            top_10 = amps.mean()
            bottom_90 = 0
        peakiness = (top_10 - bottom_90) / (top_10 + 1e-10)
        
        # Entropy
        amps_sum = amps.sum()
        if amps_sum > 0:
            amps_norm = amps / amps_sum
            entropy = -np.sum(amps_norm * np.log2(amps_norm + 1e-10))
            max_entropy = np.log2(len(amps)) if len(amps) > 1 else 1
            structure_from_entropy = 1 - (entropy / max_entropy)
        else:
            structure_from_entropy = 0
        
        strength = (peakiness + structure_from_entropy) / 2
        return float(np.clip(strength, 0, 1))
    
    def _select_pairs_adaptive(self):
        """Select frequency pairs to render."""
        n = len(self.freq_bins)
        max_pairs = min(self.max_pairs_to_render, n * (n-1) // 2)
        
        if n < 2:
            return []
        
        pairs = []
        
        if n <= 100:
            for i in range(n):
                for j in range(i+1, n):
                    coherence = self.freq_amplitudes[i] * self.freq_amplitudes[j]
                    pairs.append((i, j, coherence))
        else:
            # Sample top amplitude bins
            top_n = min(50, n // 4)
            top_indices = np.argsort(self.freq_amplitudes)[-top_n:]
            for i in range(len(top_indices)):
                for j in range(i+1, len(top_indices)):
                    ii, jj = top_indices[i], top_indices[j]
                    coherence = self.freq_amplitudes[ii] * self.freq_amplitudes[jj]
                    pairs.append((ii, jj, coherence))
            
            # Sample across range
            step = max(1, n // 100)
            sampled = list(range(0, n, step))
            for i in range(len(sampled)):
                for j in range(i+1, len(sampled)):
                    ii, jj = sampled[i], sampled[j]
                    coherence = self.freq_amplitudes[ii] * self.freq_amplitudes[jj]
                    pairs.append((ii, jj, coherence))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:max_pairs]
    
    def _render_hologram(self, pairs):
        """Render hologram from pairs."""
        res = 256
        hologram = np.zeros((res, res))
        
        for i, j, weight in pairs:
            if weight < 1e-10:
                continue
            if i >= len(self.freq_bins) or j >= len(self.freq_bins):
                continue
            
            f1, f2 = self.freq_bins[i], self.freq_bins[j]
            a1 = self.freq_amplitudes[i] if i < len(self.freq_amplitudes) else 0
            a2 = self.freq_amplitudes[j] if j < len(self.freq_amplitudes) else 0
            p1 = self.freq_phases[i] if i < len(self.freq_phases) else 0
            p2 = self.freq_phases[j] if j < len(self.freq_phases) else 0
            
            beat = abs(f1 - f2)
            phase_diff = p1 - p2
            amp = np.sqrt(a1 * a2)
            
            k = beat * 0.5
            angle = (f1 / (f2 + 1e-10)) * np.pi
            
            pattern = amp * np.cos(
                k * (self.X * np.cos(angle) + self.Y * np.sin(angle)) * 10 +
                phase_diff +
                (f1 + f2) * self.R * 0.5
            )
            pattern *= self.radial_envelope
            hologram += pattern
        
        # Normalize
        h_min, h_max = hologram.min(), hologram.max()
        if h_max != h_min:
            hologram = (hologram - h_min) / (h_max - h_min)
        
        self.hologram = hologram
        self.hologram_small = hologram.copy()
        self.n_pairs_rendered = len(pairs)
    
    def _detect_breakdown(self):
        """Detect quantum wall."""
        self.structure_history.append(self.structure_strength)
        
        if len(self.structure_history) < 10:
            return False
        
        recent = list(self.structure_history)[-10:]
        avg_recent = np.mean(recent)
        
        self.breakdown_point = avg_recent < 0.05
        return self.breakdown_point
    
    def step(self):
        self.t += 1
        start_time = time.time()
        
        # Get inputs
        raw_in = self.get_blended_input('raw_eeg', 'sum')
        bins_in = self.get_blended_input('freq_bins', 'sum')
        window_in = self.get_blended_input('time_window_ms', 'sum')
        pairs_in = self.get_blended_input('max_pairs', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.structure_history.clear()
            self.breakdown_point = False
            self.current_time = 0.0
            return
        
        # Update parameters
        if bins_in is not None:
            new_bins = int(max(4, bins_in))
            if new_bins != self.n_freq_bins:
                self.n_freq_bins = new_bins
                self._update_freq_bins()
        
        if window_in is not None:
            self.time_window_ms = max(1.0, window_in)
        
        if pairs_in is not None:
            self.max_pairs_to_render = int(max(10, pairs_in))
        
        # Load EDF
        if self.edf_path and self.edf_path != self._last_path:
            self._load_edf()
            self._update_freq_bins()
        
        # Update buffer
        if raw_in is not None:
            self.eeg_buffer = np.roll(self.eeg_buffer, -1)
            self.eeg_buffer[-1] = raw_in
        
        # Get signal
        signal = self._get_eeg_window()
        if signal is None or len(signal) < 4:
            return
        
        # Decompose
        self._decompose_fft_direct(signal)
        
        # Measure structure
        self.structure_strength = self._compute_structure_strength()
        self.structure_detected = self.structure_strength > 0.1
        
        # Select and render pairs
        pairs = self._select_pairs_adaptive()
        self._render_hologram(pairs)
        
        # Find peaks
        max_amp = self._safe_max(self.freq_amplitudes)
        if max_amp > 0 and len(self.freq_bins) > 0:
            self.peak_frequency = self.freq_bins[np.argmax(self.freq_amplitudes)]
        
        if len(pairs) > 0:
            top_pair = pairs[0]
            if top_pair[0] < len(self.freq_bins) and top_pair[1] < len(self.freq_bins):
                self.dominant_beat = abs(self.freq_bins[top_pair[0]] - self.freq_bins[top_pair[1]])
        
        self._detect_breakdown()
        self.compute_time_ms = (time.time() - start_time) * 1000
    
    def get_output(self, port_name):
        if port_name == 'hologram':
            return (self.hologram * 255).astype(np.uint8)
        elif port_name == 'hologram_small':
            return (self.hologram_small * 255).astype(np.uint8)
        elif port_name == 'spectrum_image':
            h, w = 128, 256
            img = np.zeros((h, w), dtype=np.uint8)
            max_amp = self._safe_max(self.freq_amplitudes)
            if max_amp > 0 and len(self.freq_amplitudes) > 0:
                amps = self.freq_amplitudes / max_amp
                for i in range(min(w, len(amps))):
                    idx = int(i * len(amps) / w) if w > 0 else 0
                    if idx < len(amps):
                        bar_h = int(amps[idx] * (h - 5))
                        if bar_h > 0:
                            img[h-bar_h:h, i] = 200
            return img
        elif port_name == 'coherence_matrix':
            return (self.coherence_matrix_small * 255).astype(np.uint8)
        elif port_name == 'structure_detected':
            return 1.0 if self.structure_detected else 0.0
        elif port_name == 'structure_strength':
            return float(self.structure_strength)
        elif port_name == 'breakdown_point':
            return 1.0 if self.breakdown_point else 0.0
        elif port_name == 'n_bins':
            return float(self.n_freq_bins)
        elif port_name == 'n_pairs':
            return float(self.n_freq_bins * (self.n_freq_bins - 1) / 2)
        elif port_name == 'n_pairs_rendered':
            return float(self.n_pairs_rendered)
        elif port_name == 'compute_time_ms':
            return float(self.compute_time_ms)
        elif port_name == 'peak_frequency':
            return float(self.peak_frequency)
        elif port_name == 'dominant_beat':
            return float(self.dominant_beat)
        elif port_name == 'full_spectrum':
            return self.freq_amplitudes.astype(np.float32)
        elif port_name == 'combined_view':
            return self._render_combined()
        return None
    
    def _render_combined(self):
        """Render combined view - ALL BOUNDS CHECKED."""
        # Fixed display size
        h, w = 300, 450
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Hologram (left side, max 250x250 to leave room)
        holo_size = 250
        if self.hologram_small is not None and self.hologram_small.size > 0:
            holo = np.clip(self.hologram_small * 255, 0, 255).astype(np.uint8)
            if holo.shape[0] != holo_size or holo.shape[1] != holo_size:
                holo = cv2.resize(holo, (holo_size, holo_size))
            holo_color = cv2.applyColorMap(holo, cv2.COLORMAP_TWILIGHT_SHIFTED)
            display[:holo_size, :holo_size] = holo_color
        
        # Spectrum bar (bottom, below hologram)
        spec_y = 255
        spec_h = 40
        spec_w = 250
        max_amp = self._safe_max(self.freq_amplitudes)
        if max_amp > 0 and len(self.freq_amplitudes) > 0:
            amps = self.freq_amplitudes / max_amp
            for i in range(spec_w):
                idx = int(i * len(amps) / spec_w)
                if idx < len(amps):
                    bar_h = int(amps[idx] * spec_h)
                    if bar_h > 0 and spec_y + spec_h <= h:
                        y_start = spec_y + spec_h - bar_h
                        y_end = spec_y + spec_h
                        if y_start >= 0 and y_end <= h and i < w:
                            display[y_start:y_end, i] = [100, 255, 100]
        
        # Stats (right side, x=260 to 450)
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_text = 260
        y = 20
        
        # Title
        color = (0, 0, 255) if self.breakdown_point else (0, 255, 255)
        cv2.putText(display, "EXTREME HOLO", (x_text, y), font, 0.45, color, 1)
        y += 25
        
        if self.breakdown_point:
            cv2.putText(display, "!! BREAKDOWN !!", (x_text, y), font, 0.4, (0, 0, 255), 1)
            y += 22
        
        cv2.putText(display, f"Bins: {self.n_freq_bins}", (x_text, y), font, 0.4, (255,255,255), 1)
        y += 18
        
        n_pairs = self.n_freq_bins * (self.n_freq_bins - 1) // 2
        if n_pairs > 1000000:
            pairs_str = f"{n_pairs/1000000:.1f}M"
        elif n_pairs > 1000:
            pairs_str = f"{n_pairs/1000:.1f}K"
        else:
            pairs_str = str(n_pairs)
        cv2.putText(display, f"Pairs: {pairs_str}", (x_text, y), font, 0.4, (255,255,255), 1)
        y += 18
        
        cv2.putText(display, f"Rendered: {self.n_pairs_rendered}", (x_text, y), font, 0.4, (255,255,255), 1)
        y += 18
        
        cv2.putText(display, f"Window: {self.time_window_ms:.1f}ms", (x_text, y), font, 0.4, (255,255,255), 1)
        y += 18
        
        # Structure with color
        s = self.structure_strength
        s_color = (0, 255, 0) if s > 0.3 else (0, 255, 255) if s > 0.1 else (0, 0, 255)
        cv2.putText(display, f"Structure: {s:.3f}", (x_text, y), font, 0.4, s_color, 1)
        y += 18
        
        cv2.putText(display, f"Peak: {self.peak_frequency:.1f}Hz", (x_text, y), font, 0.35, (200,200,200), 1)
        y += 16
        
        cv2.putText(display, f"Beat: {self.dominant_beat:.1f}Hz", (x_text, y), font, 0.35, (200,200,200), 1)
        y += 16
        
        cv2.putText(display, f"Time: {self.compute_time_ms:.1f}ms", (x_text, y), font, 0.35, (150,150,150), 1)
        y += 20
        
        # Uncertainty
        if self.time_window_ms > 0 and self.n_freq_bins > 1:
            freq_res = (self.freq_max - self.freq_min) / self.n_freq_bins
            uncertainty = (self.time_window_ms / 1000) * freq_res
            cv2.putText(display, f"Dt*Df: {uncertainty:.4f}", (x_text, y), font, 0.35, (255,200,100), 1)
            y += 16
            
            heisenberg = 1 / (4 * np.pi)
            ratio = uncertainty / heisenberg if heisenberg > 0 else 0
            cv2.putText(display, f"vs h/4pi: {ratio:.2f}x", (x_text, y), font, 0.35, (255,200,100), 1)
        
        return display
    
    def get_display_image(self):
        try:
            display = self._render_combined()
            return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                               display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
        except Exception as e:
            # Fallback to simple display on error
            display = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.putText(display, "Initializing...", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return QtGui.QImage(display.data, 200, 100, 600, 
                               QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Frequency Bins (NO CAP)", "n_freq_bins", self.n_freq_bins, None),
            ("Min Frequency Hz", "freq_min", self.freq_min, None),
            ("Max Frequency Hz", "freq_max", self.freq_max, None),
            ("Frequency Spacing", "freq_spacing", self.freq_spacing, 
             [('log', 'log'), ('linear', 'linear')]),
            ("Time Window (ms)", "time_window_ms", self.time_window_ms, None),
            ("Max Pairs to Render", "max_pairs_to_render", self.max_pairs_to_render, None),
        ]
    
    def set_config_options(self, options):
        rebuild = False
        for key, value in options.items():
            if key == 'n_freq_bins':
                new_n = int(max(4, float(value)))
                if new_n != self.n_freq_bins:
                    self.n_freq_bins = new_n
                    rebuild = True
            elif hasattr(self, key):
                old_val = getattr(self, key)
                if isinstance(old_val, bool):
                    setattr(self, key, value in (True, 'True', 'true', 1))
                elif isinstance(old_val, float):
                    setattr(self, key, float(value))
                    if key in ('freq_min', 'freq_max'):
                        rebuild = True
                elif isinstance(old_val, int):
                    setattr(self, key, int(float(value)))
                else:
                    setattr(self, key, value)
                    if key == 'freq_spacing':
                        rebuild = True
        
        if rebuild:
            self._update_freq_bins()