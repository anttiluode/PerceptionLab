"""
Spectral Swarm Node - BULLETPROOF VERSION
==========================================
"How many eyes does it take to see reality?"

This node tests the core hypothesis: if consciousness is tomographic reconstruction
from multiple aliased observers, then MORE frequency slices should produce
RICHER crystal structures in the combined field.

CREATED: December 2025
AUTHORS: Antti + Claude
"""

import numpy as np
import cv2
from collections import deque
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq

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
        def get_blended_input(self, name, mode):
            return None


class SpectralSwarmNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Spectral Swarm"
    NODE_COLOR = QtGui.QColor(255, 180, 50)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Spectral Swarm (N-Frequency Tomography)"
        
        self.inputs = {
            'eeg_signal': 'signal',
            'eeg_spectrum': 'spectrum',
            'token_stream': 'spectrum',
            'num_bands': 'signal',
            'freq_min': 'signal',
            'freq_max': 'signal',
            'global_coupling': 'signal',
            'adaptation_rate': 'signal',
            'lattice_zoom': 'signal',
            'lattice_freq': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'display': 'image',
            'combined_field': 'complex_spectrum',
            'band_powers': 'spectrum',
            'coupling_matrix': 'spectrum',
            'symmetry_score': 'signal',
            'anisotropy': 'signal',
            'criticality': 'signal',
            'phase_label': 'signal',
            'num_active_bands': 'signal',
        }
        
        # Parameters
        self.freq_min = 1.0
        self.freq_max = 45.0
        self.sample_rate = 160.0
        self.global_coupling = 0.5
        self.adaptation_rate = 0.01
        self.lattice_zoom = 1.0
        self.lattice_freq = 4.0
        self.field_size = 64
        self.buffer_size = 512
        
        # Signal buffer
        self.signal_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize with N=4
        self._current_N = 0  # Force initialization
        self._init_arrays(4)
        
        # Metrics
        self.symmetry_score = 0.0
        self.anisotropy = 0.0
        self.criticality = 0.0
        self.phase_label = 0
        self.epoch = 0
        
        # History
        self.symmetry_history = deque(maxlen=100)
        self.criticality_history = deque(maxlen=100)
        
        # Combined field
        self.combined_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        
        # Display
        self._display = np.zeros((750, 1100, 3), dtype=np.uint8)
    
    def _init_arrays(self, N):
        """Initialize all arrays for N bands - SINGLE SOURCE OF TRUTH"""
        N = max(2, min(64, int(N)))
        
        if N == self._current_N:
            return
        
        self._current_N = N
        self.bandwidth = (self.freq_max - self.freq_min) / max(1, N)
        
        # All arrays initialized together with same size
        self._band_freqs = np.linspace(self.freq_min, self.freq_max, N)
        self._band_powers = np.zeros(N)
        self._band_phases = np.zeros(N)
        self._states = np.ones(N, dtype=np.complex128)
        self._field_kappa = np.ones(N) * self.global_coupling
        self._inter_kappa = np.ones((N, N)) * 0.3
        np.fill_diagonal(self._inter_kappa, 0)
        self._individual_fields = [np.zeros((self.field_size, self.field_size), dtype=np.complex128) 
                                   for _ in range(N)]
    
    def _safe_get(self, arr, idx, default=0.0):
        """Safely get array element with bounds check"""
        try:
            if arr is None:
                return default
            if idx < 0 or idx >= len(arr):
                return default
            return arr[idx]
        except:
            return default
    
    def _safe_set(self, arr, idx, value):
        """Safely set array element with bounds check"""
        try:
            if arr is None:
                return
            if idx < 0 or idx >= len(arr):
                return
            arr[idx] = value
        except:
            pass
    
    def _parse_input(self, val):
        """Parse various input formats to float"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        if isinstance(val, np.ndarray):
            return float(np.mean(np.abs(val))) if val.size > 0 else 0.0
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return float(val[0]) if not hasattr(val[0], '__len__') else 0.0
        return 0.0
    
    def _extract_band_power(self, spectrum, freqs, center_freq, bandwidth):
        """Extract power in a frequency band from spectrum"""
        try:
            low = center_freq - bandwidth / 2
            high = center_freq + bandwidth / 2
            mask = (freqs >= low) & (freqs <= high)
            
            if not np.any(mask):
                return 0.0, 0.0
            
            band_spectrum = spectrum[mask]
            power = np.mean(np.abs(band_spectrum))
            phase = np.angle(np.mean(band_spectrum))
            return power, phase
        except:
            return 0.0, 0.0
    
    def _decompose_signal(self):
        """Decompose buffered signal into N frequency bands"""
        if len(self.signal_buffer) < self.buffer_size // 2:
            return False
        
        try:
            sig = np.array(list(self.signal_buffer))
            sig = sig - np.mean(sig)
            
            if np.std(sig) < 1e-10:
                return False
            
            window = np.hanning(len(sig))
            sig_windowed = sig * window
            
            spectrum = fft(sig_windowed)
            freqs = fftfreq(len(sig), 1.0 / self.sample_rate)
            
            pos_mask = freqs >= 0
            spectrum = spectrum[pos_mask]
            freqs = freqs[pos_mask]
            
            N = self._current_N
            for i in range(N):
                if i < len(self._band_freqs) and i < len(self._band_powers) and i < len(self._band_phases):
                    power, phase = self._extract_band_power(
                        spectrum, freqs,
                        self._band_freqs[i],
                        self.bandwidth
                    )
                    self._band_powers[i] = power
                    self._band_phases[i] = phase
                    
                    if i < len(self._states):
                        self._states[i] = power * np.exp(1j * phase)
            
            return True
        except Exception as e:
            return False
    
    def _create_attractor_field(self, idx):
        """Create lattice field for one attractor"""
        try:
            N = self._current_N
            if idx >= N or idx >= len(self._states) or idx >= len(self._band_freqs):
                return np.zeros((self.field_size, self.field_size), dtype=np.complex128)
            
            size = self.field_size
            span = np.pi * self.lattice_zoom
            
            x = np.linspace(-span, span, size)
            y = np.linspace(-span, span, size)
            X, Y = np.meshgrid(x, y)
            
            field = np.zeros((size, size), dtype=np.complex128)
            
            state = self._states[idx]
            amp = np.abs(state)
            phase = np.angle(state)
            
            projection_angle = idx * np.pi / max(1, N)
            freq_factor = self._band_freqs[idx] / max(1, self.freq_max)
            base_freq = self.lattice_freq * (0.5 + freq_factor)
            
            for i in range(6):
                wave_angle = i * np.pi / 3 + projection_angle + phase
                kx = base_freq * np.cos(wave_angle)
                ky = base_freq * np.sin(wave_angle)
                wave = amp * np.exp(1j * (kx * X + ky * Y))
                field += wave
            
            max_val = np.max(np.abs(field))
            if max_val > 1e-10:
                field = field / max_val
            
            return field
        except:
            return np.zeros((self.field_size, self.field_size), dtype=np.complex128)
    
    def _compute_symmetry_score(self, field):
        """Compute rotational symmetry score"""
        try:
            mag = np.abs(field)
            fft_field = np.fft.fftshift(np.fft.fft2(mag))
            power = np.abs(fft_field) ** 2
            
            center = self.field_size // 2
            radius = self.field_size // 4
            
            angles = np.arange(0, 360, 60) * np.pi / 180
            samples = []
            
            for angle in angles:
                xi = int(center + radius * np.cos(angle))
                yi = int(center + radius * np.sin(angle))
                if 0 <= xi < self.field_size and 0 <= yi < self.field_size:
                    samples.append(power[yi, xi])
            
            if len(samples) < 6:
                return 0.0
            
            samples = np.array(samples)
            mean_p = np.mean(samples)
            std_p = np.std(samples)
            
            if mean_p < 1e-10:
                return 0.0
            
            return max(0, min(1, 1.0 - std_p / mean_p))
        except:
            return 0.0
    
    def _compute_anisotropy(self, field):
        """Compute anisotropy (stripe detection)"""
        try:
            mag = np.abs(field)
            gx = np.abs(np.diff(mag, axis=1)).mean()
            gy = np.abs(np.diff(mag, axis=0)).mean()
            total = gx + gy + 1e-10
            return abs(gx - gy) / total
        except:
            return 0.0
    
    def _classify_phase(self):
        """Classify current phase: soup, critical, or stripes"""
        if self.anisotropy > 0.3:
            return 2
        elif self.symmetry_score > 0.3:
            return 1
        else:
            return 0
    
    def _adapt_optics(self):
        """Adapt inter-attractor coupling"""
        if self.adaptation_rate <= 0:
            return
        
        try:
            N = self._current_N
            reward = self.criticality
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if i < self._inter_kappa.shape[0] and j < self._inter_kappa.shape[1]:
                        if reward > 0.2:
                            self._inter_kappa[i, j] += self.adaptation_rate * 0.1
                        else:
                            self._inter_kappa[i, j] -= self.adaptation_rate * 0.05
                        self._inter_kappa[i, j] = np.clip(self._inter_kappa[i, j], 0.05, 1.0)
        except:
            pass
    
    def step(self):
        self.epoch += 1
        
        # Get inputs
        eeg = self._parse_input(self.get_blended_input('eeg_signal', 'sum'))
        num_bands = self._parse_input(self.get_blended_input('num_bands', 'sum'))
        freq_min = self._parse_input(self.get_blended_input('freq_min', 'sum'))
        freq_max = self._parse_input(self.get_blended_input('freq_max', 'sum'))
        global_coupling = self._parse_input(self.get_blended_input('global_coupling', 'sum'))
        adaptation = self._parse_input(self.get_blended_input('adaptation_rate', 'sum'))
        zoom = self._parse_input(self.get_blended_input('lattice_zoom', 'sum'))
        freq = self._parse_input(self.get_blended_input('lattice_freq', 'sum'))
        reset = self._parse_input(self.get_blended_input('reset', 'sum'))
        
        token_stream = self.get_blended_input('token_stream', 'sum')
        eeg_spectrum = self.get_blended_input('eeg_spectrum', 'sum')
        
        # Handle reset
        if reset > 0.5:
            self.signal_buffer.clear()
            self._current_N = 0
            self._init_arrays(4)
            self.symmetry_history.clear()
            self.criticality_history.clear()
            return
        
        # Update N - THIS MUST HAPPEN BEFORE ANY ARRAY ACCESS
        if num_bands >= 2:
            self._init_arrays(int(num_bands))
        
        # Update frequency range
        if freq_min > 0:
            self.freq_min = max(0.5, freq_min)
        if freq_max > self.freq_min:
            self.freq_max = min(100, freq_max)
        
        # Recompute band frequencies after freq range change
        N = self._current_N
        if N > 0:
            self.bandwidth = (self.freq_max - self.freq_min) / N
            self._band_freqs = np.linspace(self.freq_min, self.freq_max, N)
        
        # Update other parameters
        if global_coupling > 0:
            self.global_coupling = global_coupling
        if adaptation > 0:
            self.adaptation_rate = adaptation
        if zoom > 0:
            self.lattice_zoom = np.clip(zoom, 0.25, 8.0)
        if freq > 0:
            self.lattice_freq = np.clip(freq, 1.0, 16.0)
        
        # Buffer signal - priority order
        signal_added = False
        
        if eeg_spectrum is not None:
            try:
                if isinstance(eeg_spectrum, np.ndarray) and eeg_spectrum.size > 0:
                    self.signal_buffer.append(float(np.mean(np.abs(eeg_spectrum))))
                    signal_added = True
                elif isinstance(eeg_spectrum, (int, float)):
                    self.signal_buffer.append(float(eeg_spectrum))
                    signal_added = True
            except:
                pass
        
        if not signal_added and eeg != 0:
            self.signal_buffer.append(eeg)
            signal_added = True
        
        if not signal_added and token_stream is not None:
            try:
                if isinstance(token_stream, np.ndarray) and len(token_stream) > 0:
                    self.signal_buffer.append(float(np.mean(token_stream)))
                    signal_added = True
            except:
                pass
        
        if not signal_added:
            t = self.epoch / 160.0
            test_signal = (np.sin(2 * np.pi * 10 * t) +
                          0.5 * np.sin(2 * np.pi * 20 * t) +
                          0.3 * np.sin(2 * np.pi * 5 * t))
            self.signal_buffer.append(test_signal)
        
        # Decompose signal
        if not self._decompose_signal():
            return
        
        # Create individual fields
        N = self._current_N
        for i in range(N):
            if i < len(self._individual_fields):
                self._individual_fields[i] = self._create_attractor_field(i)
        
        # Combine fields
        self.combined_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        total_weight = 0
        
        for i in range(N):
            if i < len(self._band_powers) and i < len(self._individual_fields):
                weight = self._band_powers[i] + 0.01
                self.combined_field += weight * self._individual_fields[i]
                total_weight += weight
        
        if total_weight > 0:
            self.combined_field /= total_weight
        
        # Compute metrics
        self.symmetry_score = self._compute_symmetry_score(self.combined_field)
        self.anisotropy = self._compute_anisotropy(self.combined_field)
        self.criticality = self.symmetry_score * (1 - self.anisotropy)
        self.phase_label = self._classify_phase()
        
        self.symmetry_history.append(self.symmetry_score)
        self.criticality_history.append(self.criticality)
        
        # Adapt optics
        self._adapt_optics()
        
        # Update display
        self._update_display()
    
    def _update_display(self):
        """Create visualization"""
        img = np.zeros((750, 1100, 3), dtype=np.uint8)
        img[:] = (20, 25, 30)
        
        N = self._current_N
        
        # Title
        cv2.putText(img, f"SPECTRAL SWARM - N={N} Frequency Bands", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        cv2.putText(img, f"Epoch: {self.epoch} | Freq: {self.freq_min:.1f}-{self.freq_max:.1f} Hz | BW: {self.bandwidth:.1f} Hz/band",
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # Individual band panels (up to 8)
        max_display = min(N, 8)
        panel_size = 80
        panel_y = 80
        
        for i in range(max_display):
            panel_x = 20 + i * (panel_size + 15)
            
            # Band label
            freq_val = self._safe_get(self._band_freqs, i, 0.0)
            cv2.putText(img, f"{freq_val:.1f}Hz", (panel_x, panel_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 150), 1)
            
            # Field visualization
            if i < len(self._individual_fields):
                field = self._individual_fields[i]
                mag = np.abs(field)
                phase = np.angle(field)
                
                hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
                hsv[:, :, 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
                hsv[:, :, 1] = 200
                max_mag = mag.max()
                if max_mag > 1e-10:
                    hsv[:, :, 2] = (mag / max_mag * 255).astype(np.uint8)
                
                field_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                field_resized = cv2.resize(field_color, (panel_size, panel_size))
                img[panel_y:panel_y + panel_size, panel_x:panel_x + panel_size] = field_resized
            
            # Power indicator
            power = self._safe_get(self._band_powers, i, 0.0)
            power_norm = min(power * 10, 1.0)
            bar_height = int(power_norm * 30)
            cv2.rectangle(img, (panel_x, panel_y + panel_size + 5),
                         (panel_x + 20, panel_y + panel_size + 5 + bar_height),
                         (100, 200, 100), -1)
        
        if N > 8:
            cv2.putText(img, f"... +{N - 8} more bands", (20 + 8 * (panel_size + 15), panel_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Combined field
        combined_x, combined_y = 20, 220
        combined_size = 200
        
        cv2.putText(img, "COMBINED FIELD (Tomographic Reconstruction)", (combined_x, combined_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
        
        mag = np.abs(self.combined_field)
        phase = np.angle(self.combined_field)
        
        hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:, :, 1] = 220
        max_mag = mag.max()
        if max_mag > 1e-10:
            hsv[:, :, 2] = (mag / max_mag * 255).astype(np.uint8)
        
        combined_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined_resized = cv2.resize(combined_color, (combined_size, combined_size))
        img[combined_y:combined_y + combined_size, combined_x:combined_x + combined_size] = combined_resized
        
        # Metrics panel
        metrics_x, metrics_y = 250, 220
        
        cv2.putText(img, "METRICS", (metrics_x, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        
        cv2.putText(img, f"6-fold Symmetry: {self.symmetry_score:.3f}", (metrics_x, metrics_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        bar_w = int(self.symmetry_score * 150)
        cv2.rectangle(img, (metrics_x, metrics_y + 35), (metrics_x + bar_w, metrics_y + 45),
                     (100, 200, 100), -1)
        
        cv2.putText(img, f"Anisotropy: {self.anisotropy:.3f}", (metrics_x, metrics_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        bar_w = int(self.anisotropy * 150)
        cv2.rectangle(img, (metrics_x, metrics_y + 70), (metrics_x + bar_w, metrics_y + 80),
                     (200, 100, 100), -1)
        
        cv2.putText(img, f"CRITICALITY: {self.criticality:.3f}", (metrics_x, metrics_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        bar_w = int(self.criticality * 150)
        cv2.rectangle(img, (metrics_x, metrics_y + 105), (metrics_x + bar_w, metrics_y + 120),
                     (100, 200, 255), -1)
        
        phase_labels = ["SOUP", "CRITICAL", "STRIPES"]
        phase_colors = [(150, 150, 100), (100, 255, 100), (100, 100, 200)]
        cv2.putText(img, f"Phase: {phase_labels[self.phase_label]}", (metrics_x, metrics_y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_colors[self.phase_label], 2)
        
        # Band power spectrum
        spec_x, spec_y = 450, 220
        spec_w, spec_h = 300, 100
        
        cv2.putText(img, "BAND POWER SPECTRUM", (spec_x, spec_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        if N > 0:
            bar_width = max(2, spec_w // N - 1)
            max_power = max(np.max(self._band_powers), 1e-10)
            
            for i in range(N):
                power = self._safe_get(self._band_powers, i, 0.0)
                freq_val = self._safe_get(self._band_freqs, i, self.freq_min)
                
                x = spec_x + i * (bar_width + 1)
                height = int((power / max_power) * spec_h)
                
                freq_range = max(self.freq_max - self.freq_min, 1.0)
                hue = int((freq_val - self.freq_min) / freq_range * 120)
                color = cv2.cvtColor(np.array([[[hue, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
                
                cv2.rectangle(img, (x, spec_y + 10 + spec_h - height),
                             (x + bar_width, spec_y + 10 + spec_h),
                             tuple(int(c) for c in color), -1)
        
        # History plot
        hist_x, hist_y = 450, 350
        hist_w, hist_h = 300, 80
        
        cv2.putText(img, "CRITICALITY HISTORY", (hist_x, hist_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        if len(self.criticality_history) > 2:
            hist = np.array(list(self.criticality_history))
            for i in range(1, len(hist)):
                x1 = hist_x + int((i - 1) / len(hist) * hist_w)
                x2 = hist_x + int(i / len(hist) * hist_w)
                y1 = hist_y + 10 + hist_h - int(hist[i - 1] * hist_h)
                y2 = hist_y + 10 + hist_h - int(hist[i] * hist_h)
                cv2.line(img, (x1, y1), (x2, y2), (100, 200, 255), 1)
        
        # Coupling matrix (if small enough)
        if N <= 16 and N > 0:
            mat_x, mat_y = 800, 80
            cell_size = max(8, 120 // N)
            
            cv2.putText(img, "INTER-OPTICS", (mat_x, mat_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
            
            for i in range(N):
                for j in range(N):
                    x = mat_x + j * cell_size
                    y = mat_y + i * cell_size
                    
                    if i < self._inter_kappa.shape[0] and j < self._inter_kappa.shape[1]:
                        val = self._inter_kappa[i, j]
                    else:
                        val = 0
                    
                    intensity = int(val * 255)
                    color = (intensity, intensity // 2, 50) if i != j else (40, 40, 40)
                    cv2.rectangle(img, (x, y), (x + cell_size - 1, y + cell_size - 1), color, -1)
        
        # Theory box
        theory_y = 480
        cv2.putText(img, "TOMOGRAPHIC RECONSTRUCTION HYPOTHESIS:", (20, theory_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        cv2.putText(img, f"N={N} frequency bands = {N} projection angles through spectral space",
                   (20, theory_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1)
        cv2.putText(img, "Crowther Criterion: N projections resolve complexity ~N. More bands = richer crystals?",
                   (20, theory_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1)
        
        cv2.putText(img, f"Try varying N: currently N={N}", (20, theory_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        cv2.putText(img, "N=2: minimal | N=4: traditional | N=8: enhanced | N=16+: high-resolution",
                   (20, theory_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        cv2.putText(img, f"zoom={self.lattice_zoom:.1f} | freq={self.lattice_freq:.1f} | coupling={self.global_coupling:.2f}",
                   (20, 730), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        self._display = img
    
    def get_output(self, name):
        N = self._current_N
        if name == 'display':
            return self._display
        elif name == 'combined_field':
            return self.combined_field
        elif name == 'band_powers':
            return self._band_powers if self._band_powers is not None else np.zeros(4)
        elif name == 'coupling_matrix':
            return self._inter_kappa.flatten() if self._inter_kappa is not None else np.zeros(16)
        elif name == 'symmetry_score':
            return float(self.symmetry_score)
        elif name == 'anisotropy':
            return float(self.anisotropy)
        elif name == 'criticality':
            return float(self.criticality)
        elif name == 'phase_label':
            return float(self.phase_label)
        elif name == 'num_active_bands':
            return float(N)
        return None
    
    def get_display_image(self):
        h, w = self._display.shape[:2]
        return QtGui.QImage(self._display.data, w, h, w * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Number of Bands", "_current_N", self._current_N, None),
            ("Freq Min (Hz)", "freq_min", self.freq_min, None),
            ("Freq Max (Hz)", "freq_max", self.freq_max, None),
            ("Lattice Zoom", "lattice_zoom", self.lattice_zoom, None),
            ("Lattice Freq", "lattice_freq", self.lattice_freq, None),
        ]