"""
High Frequency Probe Node
=========================

The crystal chip was trained on EEG: 1-100 Hz.
But what resonances hide in the trained geometry?

This node probes the crystal at higher frequencies:
- 100 Hz - 10 kHz sweep
- Impulse response analysis
- Harmonic detection
- Resonance mapping

What patterns emerge when you excite trained neural
geometry at frequencies the brain never showed it?

The filters reveal themselves when you push past
their training distribution.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque

# --- HOST IMPORT BLOCK ---
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


class HighFrequencyProbeNode(BaseNode):
    """
    Probes crystal chip at frequencies beyond EEG training range.
    
    Generates:
    - Frequency sweeps (chirps)
    - Impulse trains at various rates
    - White noise bursts
    - Harmonic probes
    
    Analyzes:
    - Resonance peaks
    - Harmonic responses
    - Phase relationships
    - Hidden modes
    """
    
    NODE_NAME = "HF Probe"
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(200, 100, 50) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'crystal_response': 'signal',  # Response from crystal
            'crystal_image': 'image',       # Activity view from crystal
            'enable': 'signal',
            'freq_start': 'signal',
            'freq_end': 'signal',
            'probe_mode': 'signal'  # 0=sweep, 1=impulse, 2=noise, 3=harmonic
        }
        
        self.outputs = {
            'probe_signal': 'signal',       # Signal to inject into crystal
            'probe_image': 'image',         # Spatial probe pattern
            'spectrum_view': 'image',       # Frequency response
            'resonance_view': 'image',      # Detected resonances
            'peak_freq': 'signal',          # Dominant resonance frequency
            'harmonic_ratio': 'signal',     # Harmonic structure measure
            'q_factor': 'signal'            # Resonance sharpness
        }
        
        # Probe configuration
        self.freq_start = 100.0    # Hz
        self.freq_end = 5000.0     # Hz (5 kHz)
        self.sweep_duration = 500  # steps for full sweep
        self.probe_mode = 0        # 0=sweep, 1=impulse, 2=noise, 3=harmonic
        
        # Probe state
        self.step_count = 0
        self.current_freq = self.freq_start
        self.phase = 0.0
        
        # Response collection
        self.response_history = deque(maxlen=2048)
        self.probe_history = deque(maxlen=2048)
        self.frequency_log = deque(maxlen=2048)
        
        # Spatial probe pattern (64x64 to match crystal)
        self.grid_size = 64
        self.probe_pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Analysis results
        self.spectrum = np.zeros(1024, dtype=np.float32)
        self.resonances = []  # List of (freq, amplitude, q_factor)
        self.peak_frequency = 0.0
        self.harmonic_ratio = 0.0
        self.q_factor = 0.0
        
        # Display
        self.spectrum_display = None
        self.resonance_display = None
        
        # Impulse state
        self.impulse_interval = 50  # steps between impulses
        self.last_impulse = 0
        
        # Harmonic probe state
        self.base_freq = 100.0
        self.n_harmonics = 8
        
    def _read_input(self, name, default=None):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                return val if val is not None else default
            except:
                return default
        return default
    
    def _read_image_input(self, name):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first")
                if val is None:
                    return None
                if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                    return val
            except:
                pass
        return None
    
    def step(self):
        self.step_count += 1
        
        # Read configuration
        enable = self._read_input('enable', 1.0)
        f_start = self._read_input('freq_start', self.freq_start)
        f_end = self._read_input('freq_end', self.freq_end)
        mode = int(self._read_input('probe_mode', self.probe_mode))
        
        if f_start: self.freq_start = f_start
        if f_end: self.freq_end = f_end
        self.probe_mode = mode
        
        # Read crystal response
        response = self._read_input('crystal_response', 0.0)
        crystal_img = self._read_image_input('crystal_image')
        
        # Store response
        if response is not None:
            self.response_history.append(float(response))
        
        # Generate probe signal
        if enable and enable > 0.5:
            probe_value = self._generate_probe()
        else:
            probe_value = 0.0
        
        self.probe_history.append(probe_value)
        self.frequency_log.append(self.current_freq)
        
        # Generate spatial probe pattern
        self._update_probe_pattern()
        
        # Analyze response
        if self.step_count % 100 == 0 and len(self.response_history) > 256:
            self._analyze_response()
        
        # Update display
        if self.step_count % 10 == 0:
            self._update_display()
    
    def _generate_probe(self):
        """Generate probe signal based on current mode."""
        
        if self.probe_mode == 0:
            # Frequency sweep (chirp)
            return self._generate_sweep()
        elif self.probe_mode == 1:
            # Impulse train
            return self._generate_impulse()
        elif self.probe_mode == 2:
            # White noise burst
            return self._generate_noise()
        elif self.probe_mode == 3:
            # Harmonic probe
            return self._generate_harmonic()
        else:
            return 0.0
    
    def _generate_sweep(self):
        """Logarithmic frequency sweep."""
        # Progress through sweep
        progress = (self.step_count % self.sweep_duration) / self.sweep_duration
        
        # Logarithmic frequency
        log_start = np.log10(self.freq_start)
        log_end = np.log10(self.freq_end)
        log_freq = log_start + progress * (log_end - log_start)
        self.current_freq = 10 ** log_freq
        
        # Generate sine at current frequency
        # Note: step rate is ~100 Hz, so we're simulating higher frequencies
        # by phase accumulation
        dt = 0.01  # 10ms per step
        self.phase += 2 * np.pi * self.current_freq * dt
        self.phase = self.phase % (2 * np.pi)
        
        return np.sin(self.phase) * 10.0  # Amplitude 10
    
    def _generate_impulse(self):
        """Sharp impulse train - reveals impulse response."""
        self.current_freq = 1000.0 / self.impulse_interval  # Effective frequency
        
        if self.step_count - self.last_impulse >= self.impulse_interval:
            self.last_impulse = self.step_count
            return 50.0  # Sharp impulse
        return 0.0
    
    def _generate_noise(self):
        """Band-limited white noise."""
        self.current_freq = (self.freq_start + self.freq_end) / 2  # Nominal
        return np.random.randn() * 10.0
    
    def _generate_harmonic(self):
        """Sum of harmonics - probes harmonic response."""
        dt = 0.01
        self.phase += 2 * np.pi * self.base_freq * dt
        self.current_freq = self.base_freq
        
        # Sum of harmonics with decreasing amplitude
        signal = 0.0
        for n in range(1, self.n_harmonics + 1):
            signal += np.sin(self.phase * n) / n
        
        return signal * 5.0
    
    def _update_probe_pattern(self):
        """Create spatial probe pattern for injection."""
        # Different spatial patterns based on mode
        t = self.step_count * 0.1
        
        if self.probe_mode == 0:
            # Sweep: rotating spatial frequency
            kx = np.cos(t * 0.1) * self.current_freq / 500.0
            ky = np.sin(t * 0.1) * self.current_freq / 500.0
            
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            X, Y = np.meshgrid(x, y)
            
            self.probe_pattern = np.sin(kx * X + ky * Y + self.phase)
            
        elif self.probe_mode == 1:
            # Impulse: center point
            self.probe_pattern = np.zeros((self.grid_size, self.grid_size))
            if self.step_count == self.last_impulse:
                cx, cy = self.grid_size // 2, self.grid_size // 2
                self.probe_pattern[cy-2:cy+2, cx-2:cx+2] = 1.0
                
        elif self.probe_mode == 2:
            # Noise: random spatial pattern
            self.probe_pattern = np.random.randn(self.grid_size, self.grid_size) * 0.3
            
        elif self.probe_mode == 3:
            # Harmonic: concentric rings
            cx, cy = self.grid_size // 2, self.grid_size // 2
            x = np.arange(self.grid_size) - cx
            y = np.arange(self.grid_size) - cy
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            
            self.probe_pattern = np.sin(R * 0.5 + self.phase)
        
        # Normalize
        self.probe_pattern = np.clip(self.probe_pattern, -1, 1)
    
    def _analyze_response(self):
        """Analyze collected response for resonances."""
        if len(self.response_history) < 256:
            return
        
        response = np.array(list(self.response_history))
        
        # FFT of response
        n = len(response)
        fft = np.fft.rfft(response)
        magnitude = np.abs(fft)
        
        # Frequency axis (assuming 100 Hz sample rate for steps)
        freqs = np.fft.rfftfreq(n, d=0.01)
        
        # Store spectrum (resample to fixed size)
        if len(magnitude) > 10:
            self.spectrum = np.interp(
                np.linspace(0, len(magnitude)-1, 1024),
                np.arange(len(magnitude)),
                magnitude
            )
        
        # Find peaks (resonances)
        self.resonances = []
        
        # Simple peak detection
        for i in range(2, len(magnitude) - 2):
            if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                if magnitude[i] > magnitude[i-2] and magnitude[i] > magnitude[i+2]:
                    if magnitude[i] > np.mean(magnitude) * 2:  # Significant peak
                        freq = freqs[i] if i < len(freqs) else 0
                        amp = magnitude[i]
                        
                        # Estimate Q factor (width at half max)
                        half_max = magnitude[i] / 2
                        width = 1
                        for j in range(1, min(10, i, len(magnitude)-i)):
                            if magnitude[i-j] < half_max or magnitude[i+j] < half_max:
                                width = j
                                break
                        q = freq / (2 * width * (freqs[1] - freqs[0]) + 0.01)
                        
                        self.resonances.append((freq, amp, q))
        
        # Sort by amplitude
        self.resonances.sort(key=lambda x: x[1], reverse=True)
        
        # Top resonance
        if self.resonances:
            self.peak_frequency = self.resonances[0][0]
            self.q_factor = self.resonances[0][2]
        
        # Harmonic ratio: are peaks at harmonic intervals?
        if len(self.resonances) >= 2:
            f0 = self.resonances[0][0]
            if f0 > 0:
                ratios = [r[0] / f0 for r in self.resonances[1:5]]
                # How close to integers?
                harmonic_score = sum(1 - abs(r - round(r)) for r in ratios if r > 0)
                self.harmonic_ratio = harmonic_score / max(1, len(ratios))
    
    def _update_display(self):
        """Create visualizations."""
        size = 400
        
        # === Spectrum View ===
        spec_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        if len(self.spectrum) > 0 and np.max(self.spectrum) > 0:
            # Draw spectrum
            spec_norm = self.spectrum / (np.max(self.spectrum) + 0.01)
            
            for i in range(min(len(spec_norm), size)):
                height = int(spec_norm[i] * (size - 50))
                x = int(i * size / len(spec_norm))
                
                # Color by frequency
                hue = int(i * 180 / len(spec_norm))
                color = self._hsv_to_rgb(hue, 255, 200)
                
                cv2.line(spec_img, (x, size - 30), (x, size - 30 - height), color, 1)
        
        # Draw resonance markers
        for freq, amp, q in self.resonances[:5]:
            if freq > 0:
                x = int(freq / 50 * size / 10)  # Scale to display
                x = min(x, size - 1)
                cv2.line(spec_img, (x, 0), (x, 50), (0, 255, 255), 2)
        
        # Labels
        cv2.putText(spec_img, "HF SPECTRUM", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(spec_img, f"Mode: {['Sweep', 'Impulse', 'Noise', 'Harmonic'][self.probe_mode]}", 
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(spec_img, f"Freq: {self.current_freq:.1f} Hz", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(spec_img, f"Peak: {self.peak_frequency:.1f} Hz", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(spec_img, f"Q: {self.q_factor:.1f}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(spec_img, f"Harmonic: {self.harmonic_ratio:.2f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # Frequency axis labels
        for f in [100, 500, 1000, 2000, 5000]:
            x = int(f / 50 * size / 100)
            if x < size:
                cv2.putText(spec_img, f"{f}", (x, size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        self.spectrum_display = cv2.cvtColor(spec_img, cv2.COLOR_BGR2RGB)
        
        # === Resonance View ===
        res_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Draw probe pattern
        pattern_scaled = cv2.resize(
            ((self.probe_pattern + 1) * 127).astype(np.uint8),
            (size, size)
        )
        res_img[:, :, 1] = pattern_scaled  # Green channel
        
        # Draw resonance info
        cv2.putText(res_img, "PROBE PATTERN", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # List top resonances
        y = 50
        for i, (freq, amp, q) in enumerate(self.resonances[:8]):
            cv2.putText(res_img, f"{i+1}. {freq:.1f} Hz (Q={q:.1f})", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            y += 18
        
        # Draw time series of recent probe and response
        if len(self.probe_history) > 100:
            probe = np.array(list(self.probe_history)[-200:])
            resp = np.array(list(self.response_history)[-200:])
            
            probe_norm = probe / (np.max(np.abs(probe)) + 0.01)
            resp_norm = resp / (np.max(np.abs(resp)) + 0.01)
            
            for i in range(len(probe_norm) - 1):
                x1 = int(i * size / len(probe_norm))
                x2 = int((i+1) * size / len(probe_norm))
                
                # Probe (blue)
                y1 = int(size - 100 + probe_norm[i] * 30)
                y2 = int(size - 100 + probe_norm[i+1] * 30)
                cv2.line(res_img, (x1, y1), (x2, y2), (255, 100, 100), 1)
                
                # Response (yellow)
                if i < len(resp_norm) - 1:
                    y1 = int(size - 50 + resp_norm[i] * 30)
                    y2 = int(size - 50 + resp_norm[i+1] * 30)
                    cv2.line(res_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        cv2.putText(res_img, "Probe (blue) / Response (yellow)", (10, size - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        self.resonance_display = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB tuple."""
        h = h / 180.0
        s = s / 255.0
        v = v / 255.0
        
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i % 6 == 0:
            r, g, b = v, t, p
        elif i % 6 == 1:
            r, g, b = q, v, p
        elif i % 6 == 2:
            r, g, b = p, v, t
        elif i % 6 == 3:
            r, g, b = p, q, v
        elif i % 6 == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return (int(b * 255), int(g * 255), int(r * 255))
    
    def get_output(self, port_name):
        if port_name == 'probe_signal':
            if self.probe_history:
                return float(self.probe_history[-1])
            return 0.0
        elif port_name == 'probe_image':
            # Return probe pattern as image (scaled to 0-255)
            img = ((self.probe_pattern + 1) * 127).astype(np.uint8)
            img_rgb = cv2.cvtColor(cv2.applyColorMap(img, cv2.COLORMAP_TWILIGHT), cv2.COLOR_BGR2RGB)
            return img_rgb
        elif port_name == 'spectrum_view':
            return self.spectrum_display
        elif port_name == 'resonance_view':
            return self.resonance_display
        elif port_name == 'peak_freq':
            return float(self.peak_frequency)
        elif port_name == 'harmonic_ratio':
            return float(self.harmonic_ratio)
        elif port_name == 'q_factor':
            return float(self.q_factor)
        return None
    
    def get_display_image(self):
        if self.spectrum_display is not None and QtGui:
            h, w = self.spectrum_display.shape[:2]
            return QtGui.QImage(self.spectrum_display.data, w, h, w * 3,
                              QtGui.QImage.Format.Format_RGB888).copy()
        return None
    
    def get_config_options(self):
        return [
            ("Freq Start (Hz)", "freq_start", self.freq_start, None),
            ("Freq End (Hz)", "freq_end", self.freq_end, None),
            ("Sweep Duration", "sweep_duration", self.sweep_duration, None),
            ("Probe Mode (0-3)", "probe_mode", self.probe_mode, None),
            ("Impulse Interval", "impulse_interval", self.impulse_interval, None),
            ("Base Freq (harmonic)", "base_freq", self.base_freq, None),
            ("N Harmonics", "n_harmonics", self.n_harmonics, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)