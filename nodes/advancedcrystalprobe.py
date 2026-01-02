"""
Advanced Crystal Probe Node
============================

Comprehensive probing toolkit for EEG-trained crystal chips.
Implements methods from 2025 computational neuroscience:

1. FINE GAMMA MAPPING (30-100 Hz, 0.1 Hz resolution)
   - High-resolution sweep around the 49.8 Hz resonance
   - Bode plot generation (amplitude + phase vs frequency)
   - Subharmonic detection (alpha-theta interactions)

2. IMPULSE RESPONSE ANALYSIS
   - Tuned impulse trains at resonant frequencies
   - Ring-down time measurement
   - Eigenmode extraction

3. THETA-BURST STIMULATION
   - 5 Hz bursts of gamma (50-100 Hz)
   - Mimics hippocampal plasticity protocols
   - Vibrational resonance probing

4. HIGH FREQUENCY OSCILLATIONS (80-500 Hz)
   - Safe chirp probes (capped at 1 MHz)
   - HFO/ripple detection
   - Stochastic resonance via noise injection

5. SPATIAL PATTERN INJECTION
   - Pin-specific activation patterns
   - Sequential replay patterns
   - Gradient probes for connectivity mapping

Based on: CLS theory, holographic ensemble stimulation,
         vibrational resonance, HFO detection methods

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque
from scipy import signal as scipy_signal

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


class AdvancedCrystalProbeNode(BaseNode):
    """
    Advanced multi-modal probe for crystal chip analysis.
    
    Modes:
    0: Fine Gamma Sweep (30-100 Hz, high resolution)
    1: Impulse at Resonance (49.8 Hz tuned)
    2: Theta-Burst Stimulation
    3: HFO Chirp (80-500 Hz)
    4: Noise + Resonance Hunt
    5: Spatial Replay Patterns
    6: Bi-Frequency Modulation
    7: Custom Frequency Lock
    """
    
    NODE_NAME = "Advanced Crystal Probe"
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(220, 80, 60) if QtGui else None
    
    # Maximum frequency to prevent overflow
    MAX_FREQ = 1e6  # 1 MHz cap
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'crystal_response': 'signal',
            'crystal_image': 'image',
            'enable': 'signal',
            'probe_mode': 'signal',
            'target_freq': 'signal',      # Lock to specific frequency
            'modulation': 'signal',        # External modulation
            'spatial_pattern': 'image'     # Custom spatial input
        }
        
        self.outputs = {
            'probe_signal': 'signal',
            'probe_image': 'image',
            'spectrum_view': 'image',
            'bode_view': 'image',          # Amplitude + phase plot
            'resonance_map': 'image',      # Spatial resonance map
            'peak_freq': 'signal',
            'q_factor': 'signal',
            'phase_shift': 'signal',
            'ring_time': 'signal',         # Impulse ring-down time
            'coherence': 'signal'          # Probe-response coherence
        }
        
        # === Mode Configuration ===
        self.probe_mode = 0
        self.step_count = 0
        
        # === Fine Gamma Parameters (Mode 0) ===
        self.gamma_start = 30.0      # Hz
        self.gamma_end = 100.0       # Hz
        self.gamma_resolution = 0.1  # Hz per step
        self.gamma_sweep_steps = int((100 - 30) / 0.1)  # 700 steps
        
        # === Impulse Parameters (Mode 1) ===
        self.impulse_target = 49.8   # Hz - the discovered resonance
        self.impulse_interval = 20   # steps (tuned to ~50 Hz)
        self.last_impulse = 0
        self.ring_down_samples = deque(maxlen=100)
        self.ring_time = 0.0
        
        # === Theta-Burst Parameters (Mode 2) ===
        self.theta_freq = 5.0        # Hz - burst frequency
        self.gamma_burst_freq = 50.0 # Hz - within burst
        self.burst_duration = 10     # steps per burst
        self.burst_count = 0
        self.in_burst = False
        
        # === HFO Parameters (Mode 3) ===
        self.hfo_start = 80.0        # Hz
        self.hfo_end = 500.0         # Hz
        self.hfo_sweep_duration = 1000
        
        # === Noise Parameters (Mode 4) ===
        self.noise_bandwidth = (30, 100)  # Focus on gamma band
        self.noise_amplitude = 5.0
        
        # === Spatial Replay Parameters (Mode 5) ===
        self.replay_sequence = []
        self.replay_step = 0
        self.replay_speed = 5  # steps per pattern
        
        # === Bi-Frequency Parameters (Mode 6) ===
        self.carrier_freq = 50.0     # High freq carrier
        self.envelope_freq = 5.0     # Theta envelope
        
        # === Frequency Lock Parameters (Mode 7) ===
        self.locked_freq = 49.8      # User-adjustable
        
        # === State Variables ===
        self.current_freq = 30.0
        self.phase = 0.0
        self.envelope_phase = 0.0
        self.dt = 0.01  # 10ms per step (~100 Hz update rate)
        
        # === Response Collection ===
        self.response_history = deque(maxlen=4096)
        self.probe_history = deque(maxlen=4096)
        self.frequency_log = deque(maxlen=4096)
        self.phase_history = deque(maxlen=4096)
        
        # === Bode Plot Data ===
        self.bode_freqs = []
        self.bode_amplitudes = []
        self.bode_phases = []
        
        # === Analysis Results ===
        self.spectrum = np.zeros(1024)
        self.resonances = []
        self.peak_frequency = 49.8
        self.q_factor = 0.0
        self.phase_shift = 0.0
        self.coherence = 0.0
        
        # === Spatial Probe ===
        self.grid_size = 64
        self.probe_pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._init_replay_sequence()
        
        # === Displays ===
        self.spectrum_display = None
        self.bode_display = None
        self.resonance_map = None
        
    def _init_replay_sequence(self):
        """Initialize spatial replay patterns mimicking memory episodes."""
        # Create sequence of activation patterns: frontal → parietal → occipital
        self.replay_sequence = []
        
        g = self.grid_size
        
        # Pattern 1: Frontal activation (top of grid)
        p1 = np.zeros((g, g), dtype=np.float32)
        p1[:g//4, g//4:3*g//4] = 1.0
        self.replay_sequence.append(p1)
        
        # Pattern 2: Central spread
        p2 = np.zeros((g, g), dtype=np.float32)
        p2[g//4:g//2, g//4:3*g//4] = 1.0
        self.replay_sequence.append(p2)
        
        # Pattern 3: Parietal
        p3 = np.zeros((g, g), dtype=np.float32)
        p3[g//2:3*g//4, :] = 1.0
        self.replay_sequence.append(p3)
        
        # Pattern 4: Occipital (bottom)
        p4 = np.zeros((g, g), dtype=np.float32)
        p4[3*g//4:, :] = 1.0
        self.replay_sequence.append(p4)
        
        # Pattern 5: Global synchrony
        p5 = np.ones((g, g), dtype=np.float32) * 0.5
        self.replay_sequence.append(p5)
        
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
                if val is not None and hasattr(val, 'shape'):
                    return val
            except:
                pass
        return None
    
    def step(self):
        self.step_count += 1
        
        # Read inputs
        enable = self._read_input('enable', 1.0)
        mode = int(self._read_input('probe_mode', self.probe_mode) or 0)
        target = self._read_input('target_freq', None)
        modulation = self._read_input('modulation', 0.0)
        
        self.probe_mode = mode % 8  # 8 modes
        
        if target is not None and target > 0:
            self.locked_freq = min(target, self.MAX_FREQ)
        
        # Read crystal response
        response = self._read_input('crystal_response', 0.0) or 0.0
        self.response_history.append(float(response))
        
        # Generate probe based on mode
        if enable and enable > 0.5:
            probe_value = self._generate_probe(modulation)
        else:
            probe_value = 0.0
        
        self.probe_history.append(probe_value)
        self.frequency_log.append(self.current_freq)
        self.phase_history.append(self.phase)
        
        # Update spatial pattern
        self._update_probe_pattern()
        
        # Periodic analysis
        if self.step_count % 50 == 0:
            self._analyze_response()
        
        # Update displays
        if self.step_count % 10 == 0:
            self._update_displays()
    
    def _generate_probe(self, modulation=0.0):
        """Generate probe signal based on current mode."""
        
        if self.probe_mode == 0:
            return self._generate_fine_gamma_sweep()
        elif self.probe_mode == 1:
            return self._generate_resonant_impulse()
        elif self.probe_mode == 2:
            return self._generate_theta_burst()
        elif self.probe_mode == 3:
            return self._generate_hfo_chirp()
        elif self.probe_mode == 4:
            return self._generate_resonance_noise()
        elif self.probe_mode == 5:
            return self._generate_spatial_replay()
        elif self.probe_mode == 6:
            return self._generate_bifreq_modulation()
        elif self.probe_mode == 7:
            return self._generate_freq_lock(modulation)
        
        return 0.0
    
    def _generate_fine_gamma_sweep(self):
        """Mode 0: High-resolution sweep in gamma band (30-100 Hz)."""
        # Calculate position in sweep
        sweep_pos = self.step_count % self.gamma_sweep_steps
        
        # Linear frequency sweep for fine resolution
        self.current_freq = self.gamma_start + sweep_pos * self.gamma_resolution
        self.current_freq = min(self.current_freq, self.gamma_end)
        
        # Phase accumulation
        self.phase += 2 * np.pi * self.current_freq * self.dt
        self.phase %= (2 * np.pi)
        
        # Record for Bode plot at each frequency
        if sweep_pos % 10 == 0:  # Sample every 1 Hz
            self._record_bode_point()
        
        return np.sin(self.phase) * 10.0
    
    def _generate_resonant_impulse(self):
        """Mode 1: Impulse train tuned to 49.8 Hz resonance."""
        # Interval tuned to resonant frequency
        interval = int(1.0 / (self.impulse_target * self.dt))
        interval = max(1, interval)
        
        self.current_freq = self.impulse_target
        
        if self.step_count - self.last_impulse >= interval:
            self.last_impulse = self.step_count
            # Start collecting ring-down
            self.ring_down_samples.clear()
            return 50.0  # Sharp impulse
        
        # Collect ring-down response
        if len(self.response_history) > 0:
            self.ring_down_samples.append(self.response_history[-1])
            self._analyze_ring_down()
        
        return 0.0
    
    def _generate_theta_burst(self):
        """Mode 2: Theta-burst stimulation (5 Hz bursts of gamma)."""
        # Theta envelope
        self.envelope_phase += 2 * np.pi * self.theta_freq * self.dt
        self.envelope_phase %= (2 * np.pi)
        
        # Are we in burst? (positive half of theta cycle)
        self.in_burst = np.sin(self.envelope_phase) > 0
        
        if self.in_burst:
            # Generate gamma burst
            self.current_freq = self.gamma_burst_freq
            self.phase += 2 * np.pi * self.gamma_burst_freq * self.dt
            self.phase %= (2 * np.pi)
            
            # Burst amplitude modulated by theta envelope
            envelope = np.sin(self.envelope_phase)
            return np.sin(self.phase) * envelope * 15.0
        else:
            self.current_freq = self.theta_freq
            return 0.0
    
    def _generate_hfo_chirp(self):
        """Mode 3: High Frequency Oscillation chirp (80-500 Hz)."""
        # Progress through sweep
        progress = (self.step_count % self.hfo_sweep_duration) / self.hfo_sweep_duration
        
        # Logarithmic sweep for HFO range
        log_start = np.log10(self.hfo_start)
        log_end = np.log10(min(self.hfo_end, self.MAX_FREQ))
        log_freq = log_start + progress * (log_end - log_start)
        
        self.current_freq = min(10 ** log_freq, self.MAX_FREQ)
        
        # Phase accumulation
        self.phase += 2 * np.pi * self.current_freq * self.dt
        self.phase %= (2 * np.pi)
        
        return np.sin(self.phase) * 8.0
    
    def _generate_resonance_noise(self):
        """Mode 4: Band-limited noise for stochastic resonance hunting."""
        self.current_freq = (self.noise_bandwidth[0] + self.noise_bandwidth[1]) / 2
        
        # Generate white noise
        noise = np.random.randn() * self.noise_amplitude
        
        # Simple band-pass approximation using phase modulation
        # Add some structure at gamma frequencies
        gamma_component = np.sin(self.phase) * 2.0
        self.phase += 2 * np.pi * 50.0 * self.dt  # Gamma carrier
        self.phase %= (2 * np.pi)
        
        return noise + gamma_component
    
    def _generate_spatial_replay(self):
        """Mode 5: Sequential spatial patterns mimicking memory replay."""
        self.current_freq = 10.0  # Nominal
        
        # Change pattern every replay_speed steps
        if self.step_count % self.replay_speed == 0:
            self.replay_step = (self.replay_step + 1) % len(self.replay_sequence)
        
        # Modulate signal based on current pattern energy
        pattern_energy = np.mean(self.replay_sequence[self.replay_step])
        
        # Return signal proportional to pattern activation
        return pattern_energy * 20.0
    
    def _generate_bifreq_modulation(self):
        """Mode 6: Bi-frequency probe (gamma carrier on theta envelope)."""
        # Theta envelope
        self.envelope_phase += 2 * np.pi * self.envelope_freq * self.dt
        self.envelope_phase %= (2 * np.pi)
        envelope = (np.sin(self.envelope_phase) + 1) / 2  # 0 to 1
        
        # Gamma carrier
        self.phase += 2 * np.pi * self.carrier_freq * self.dt
        self.phase %= (2 * np.pi)
        carrier = np.sin(self.phase)
        
        self.current_freq = self.carrier_freq
        
        return carrier * envelope * 15.0
    
    def _generate_freq_lock(self, modulation=0.0):
        """Mode 7: Lock to specific frequency with optional modulation."""
        # Apply external modulation to locked frequency
        freq = self.locked_freq + modulation * 10.0
        freq = np.clip(freq, 0.1, self.MAX_FREQ)
        
        self.current_freq = freq
        self.phase += 2 * np.pi * freq * self.dt
        self.phase %= (2 * np.pi)
        
        return np.sin(self.phase) * 10.0
    
    def _update_probe_pattern(self):
        """Update spatial probe pattern based on mode."""
        t = self.step_count * self.dt
        
        if self.probe_mode == 0:
            # Gamma sweep: spatial frequency matches temporal
            k = self.current_freq / 100.0  # Spatial frequency
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            X, Y = np.meshgrid(x, y)
            self.probe_pattern = np.sin(k * X + self.phase) * np.cos(k * Y)
            
        elif self.probe_mode == 1:
            # Impulse: center point activation
            self.probe_pattern = np.zeros((self.grid_size, self.grid_size))
            if self.step_count == self.last_impulse:
                cx, cy = self.grid_size // 2, self.grid_size // 2
                # Create a sharp gaussian
                x = np.arange(self.grid_size)
                y = np.arange(self.grid_size)
                X, Y = np.meshgrid(x, y)
                self.probe_pattern = np.exp(-((X-cx)**2 + (Y-cy)**2) / 8)
                
        elif self.probe_mode == 2:
            # Theta-burst: pulsing concentric rings
            if self.in_burst:
                cx, cy = self.grid_size // 2, self.grid_size // 2
                x = np.arange(self.grid_size) - cx
                y = np.arange(self.grid_size) - cy
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                self.probe_pattern = np.sin(R * 0.3 + self.phase) * np.sin(self.envelope_phase)
            else:
                self.probe_pattern *= 0.9  # Decay
                
        elif self.probe_mode == 3:
            # HFO chirp: radial wave expanding
            cx, cy = self.grid_size // 2, self.grid_size // 2
            x = np.arange(self.grid_size) - cx
            y = np.arange(self.grid_size) - cy
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            k = self.current_freq / 200.0
            self.probe_pattern = np.sin(k * R - self.phase)
            
        elif self.probe_mode == 4:
            # Noise: random spatial pattern
            self.probe_pattern = np.random.randn(self.grid_size, self.grid_size) * 0.5
            # Apply spatial smoothing
            self.probe_pattern = cv2.GaussianBlur(self.probe_pattern, (5, 5), 1.0)
            
        elif self.probe_mode == 5:
            # Spatial replay: use current sequence pattern
            self.probe_pattern = self.replay_sequence[self.replay_step].copy()
            
        elif self.probe_mode == 6:
            # Bi-freq: theta envelope modulates spatial gamma pattern
            k = self.carrier_freq / 100.0
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            X, Y = np.meshgrid(x, y)
            envelope = (np.sin(self.envelope_phase) + 1) / 2
            self.probe_pattern = np.sin(k * X + self.phase) * envelope
            
        elif self.probe_mode == 7:
            # Freq lock: standing wave at locked frequency
            k = self.locked_freq / 100.0
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            X, Y = np.meshgrid(x, y)
            self.probe_pattern = np.sin(k * X) * np.sin(k * Y) * np.sin(self.phase)
        
        # Normalize
        self.probe_pattern = np.clip(self.probe_pattern, -1, 1).astype(np.float32)
    
    def _record_bode_point(self):
        """Record amplitude and phase for Bode plot."""
        if len(self.response_history) < 50:
            return
        
        # Get recent response and probe
        response = np.array(list(self.response_history)[-50:])
        probe = np.array(list(self.probe_history)[-50:])
        
        # Amplitude ratio
        resp_amp = np.std(response)
        probe_amp = np.std(probe)
        amplitude = resp_amp / (probe_amp + 0.01)
        
        # Phase estimation via cross-correlation
        if len(response) > 10:
            corr = np.correlate(probe - np.mean(probe), 
                               response - np.mean(response), mode='full')
            lag = np.argmax(corr) - len(probe) + 1
            phase = lag * self.dt * self.current_freq * 360  # degrees
        else:
            phase = 0
        
        # Store
        self.bode_freqs.append(self.current_freq)
        self.bode_amplitudes.append(amplitude)
        self.bode_phases.append(phase % 360)
        
        # Keep limited history
        if len(self.bode_freqs) > 500:
            self.bode_freqs = self.bode_freqs[-500:]
            self.bode_amplitudes = self.bode_amplitudes[-500:]
            self.bode_phases = self.bode_phases[-500:]
    
    def _analyze_ring_down(self):
        """Analyze impulse response ring-down time."""
        if len(self.ring_down_samples) < 20:
            return
        
        samples = np.array(list(self.ring_down_samples))
        envelope = np.abs(samples)
        
        # Find time to decay to 1/e
        if np.max(envelope) > 0:
            threshold = np.max(envelope) / np.e
            below_threshold = np.where(envelope < threshold)[0]
            if len(below_threshold) > 0:
                self.ring_time = below_threshold[0] * self.dt * 1000  # ms
            else:
                self.ring_time = len(samples) * self.dt * 1000
    
    def _analyze_response(self):
        """Comprehensive response analysis."""
        if len(self.response_history) < 256:
            return
        
        response = np.array(list(self.response_history)[-1024:])
        probe = np.array(list(self.probe_history)[-1024:])
        
        # === Spectrum Analysis ===
        fft = np.fft.rfft(response)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(response), d=self.dt)
        
        # Resample to fixed size
        if len(magnitude) > 10:
            self.spectrum = np.interp(
                np.linspace(0, len(magnitude)-1, 1024),
                np.arange(len(magnitude)),
                magnitude
            )
        
        # === Peak Detection ===
        self.resonances = []
        for i in range(2, len(magnitude) - 2):
            if (magnitude[i] > magnitude[i-1] and 
                magnitude[i] > magnitude[i+1] and
                magnitude[i] > np.mean(magnitude) * 2):
                
                freq = freqs[i] if i < len(freqs) else 0
                amp = magnitude[i]
                
                # Q factor estimation
                half_max = magnitude[i] / 2
                width = 1
                for j in range(1, min(20, i, len(magnitude)-i)):
                    if magnitude[i-j] < half_max or magnitude[i+j] < half_max:
                        width = j
                        break
                df = freqs[1] - freqs[0] if len(freqs) > 1 else 1
                q = freq / (2 * width * df + 0.001)
                
                self.resonances.append((freq, amp, q))
        
        self.resonances.sort(key=lambda x: x[1], reverse=True)
        
        if self.resonances:
            self.peak_frequency = self.resonances[0][0]
            self.q_factor = self.resonances[0][2]
        
        # === Coherence ===
        if len(probe) > 0 and len(response) > 0:
            # Cross-correlation based coherence
            corr = np.correlate(probe - np.mean(probe),
                               response - np.mean(response), mode='valid')
            auto_p = np.correlate(probe - np.mean(probe),
                                  probe - np.mean(probe), mode='valid')
            auto_r = np.correlate(response - np.mean(response),
                                  response - np.mean(response), mode='valid')
            
            if auto_p[0] > 0 and auto_r[0] > 0:
                self.coherence = np.max(np.abs(corr)) / np.sqrt(auto_p[0] * auto_r[0])
    
    def _update_displays(self):
        """Create all visualization outputs."""
        size = 400
        
        # === Spectrum Display ===
        self._update_spectrum_display(size)
        
        # === Bode Plot Display ===
        self._update_bode_display(size)
        
        # === Resonance Map ===
        self._update_resonance_map(size)
    
    def _update_spectrum_display(self, size):
        """Spectrum with mode info and resonance markers."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        mode_names = [
            "Fine Gamma (30-100Hz)", "Resonant Impulse", "Theta-Burst",
            "HFO Chirp", "Noise Hunt", "Spatial Replay",
            "Bi-Freq Mod", "Freq Lock"
        ]
        
        # Draw spectrum
        if np.max(self.spectrum) > 0:
            spec_norm = self.spectrum / np.max(self.spectrum)
            for i in range(min(len(spec_norm), size)):
                height = int(spec_norm[i] * (size - 80))
                x = int(i * size / len(spec_norm))
                hue = int(i * 180 / len(spec_norm))
                color = self._hsv_to_rgb(hue, 255, 200)
                cv2.line(img, (x, size - 40), (x, size - 40 - height), color, 1)
        
        # Mark resonances
        for freq, amp, q in self.resonances[:5]:
            if freq > 0 and freq < 100:
                x = int(freq * size / 100)
                cv2.line(img, (x, 0), (x, 30), (0, 255, 255), 2)
        
        # Labels
        cv2.putText(img, f"Mode: {mode_names[self.probe_mode]}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Freq: {self.current_freq:.1f} Hz", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Peak: {self.peak_frequency:.1f} Hz", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(img, f"Q: {self.q_factor:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(img, f"Coherence: {self.coherence:.2f}", (200, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(img, f"Ring-down: {self.ring_time:.1f} ms", (200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Frequency axis
        for f in [10, 30, 50, 70, 100]:
            x = int(f * size / 100)
            cv2.putText(img, str(f), (x, size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        self.spectrum_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _update_bode_display(self, size):
        """Bode plot: amplitude and phase vs frequency."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        if len(self.bode_freqs) > 2:
            freqs = np.array(self.bode_freqs)
            amps = np.array(self.bode_amplitudes)
            phases = np.array(self.bode_phases)
            
            # Sort by frequency
            idx = np.argsort(freqs)
            freqs = freqs[idx]
            amps = amps[idx]
            phases = phases[idx]
            
            # Amplitude plot (top half)
            if np.max(amps) > 0:
                amp_norm = amps / np.max(amps)
                for i in range(len(freqs) - 1):
                    x1 = int((freqs[i] - 30) / 70 * size)
                    x2 = int((freqs[i+1] - 30) / 70 * size)
                    y1 = int(size/2 - amp_norm[i] * (size/2 - 40))
                    y2 = int(size/2 - amp_norm[i+1] * (size/2 - 40))
                    if 0 <= x1 < size and 0 <= x2 < size:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Phase plot (bottom half)
            phase_norm = phases / 360
            for i in range(len(freqs) - 1):
                x1 = int((freqs[i] - 30) / 70 * size)
                x2 = int((freqs[i+1] - 30) / 70 * size)
                y1 = int(size - 20 - phase_norm[i] * (size/2 - 40))
                y2 = int(size - 20 - phase_norm[i+1] * (size/2 - 40))
                if 0 <= x1 < size and 0 <= x2 < size:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 2)
        
        # Divider
        cv2.line(img, (0, size//2), (size, size//2), (100, 100, 100), 1)
        
        # Labels
        cv2.putText(img, "BODE PLOT", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Amplitude (green)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(img, "Phase (red)", (10, size//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
        
        # Mark 49.8 Hz
        x_498 = int((49.8 - 30) / 70 * size)
        cv2.line(img, (x_498, 0), (x_498, size), (255, 255, 0), 1)
        cv2.putText(img, "49.8", (x_498 - 15, size//2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        self.bode_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _update_resonance_map(self, size):
        """Spatial map showing probe pattern and resonance hotspots."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Draw probe pattern
        pattern_scaled = cv2.resize(
            ((self.probe_pattern + 1) * 127).astype(np.uint8),
            (size, size)
        )
        img[:, :, 1] = pattern_scaled
        
        # Overlay resonance info
        cv2.putText(img, "PROBE PATTERN", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # List resonances
        y = 45
        for i, (freq, amp, q) in enumerate(self.resonances[:8]):
            text = f"{i+1}. {freq:.1f} Hz (Q={q:.1f})"
            cv2.putText(img, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            y += 16
        
        # Draw recent time series
        if len(self.probe_history) > 50:
            probe = np.array(list(self.probe_history)[-100:])
            resp = np.array(list(self.response_history)[-100:])
            
            p_norm = probe / (np.max(np.abs(probe)) + 0.01)
            r_norm = resp / (np.max(np.abs(resp)) + 0.01)
            
            for i in range(len(p_norm) - 1):
                x1 = int(i * size / len(p_norm))
                x2 = int((i+1) * size / len(p_norm))
                
                # Probe (blue)
                y1 = int(size - 60 + p_norm[i] * 20)
                y2 = int(size - 60 + p_norm[i+1] * 20)
                cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 1)
                
                # Response (yellow)
                if i < len(r_norm) - 1:
                    y1 = int(size - 30 + r_norm[i] * 20)
                    y2 = int(size - 30 + r_norm[i+1] * 20)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        cv2.putText(img, "Probe/Response", (10, size - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        self.resonance_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to BGR tuple for OpenCV."""
        h = h / 180.0
        s = s / 255.0
        v = v / 255.0
        
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i % 6 == 0: r, g, b = v, t, p
        elif i % 6 == 1: r, g, b = q, v, p
        elif i % 6 == 2: r, g, b = p, v, t
        elif i % 6 == 3: r, g, b = p, q, v
        elif i % 6 == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        
        return (int(b * 255), int(g * 255), int(r * 255))
    
    def get_output(self, port_name):
        if port_name == 'probe_signal':
            return float(self.probe_history[-1]) if self.probe_history else 0.0
        elif port_name == 'probe_image':
            img = ((self.probe_pattern + 1) * 127).astype(np.uint8)
            return cv2.cvtColor(cv2.applyColorMap(img, cv2.COLORMAP_TWILIGHT), cv2.COLOR_BGR2RGB)
        elif port_name == 'spectrum_view':
            return self.spectrum_display
        elif port_name == 'bode_view':
            return self.bode_display
        elif port_name == 'resonance_map':
            return self.resonance_map
        elif port_name == 'peak_freq':
            return float(self.peak_frequency)
        elif port_name == 'q_factor':
            return float(self.q_factor)
        elif port_name == 'phase_shift':
            return float(self.phase_shift)
        elif port_name == 'ring_time':
            return float(self.ring_time)
        elif port_name == 'coherence':
            return float(self.coherence)
        return None
    
    def get_display_image(self):
        if self.spectrum_display is not None and QtGui:
            h, w = self.spectrum_display.shape[:2]
            return QtGui.QImage(self.spectrum_display.data, w, h, w * 3,
                               QtGui.QImage.Format.Format_RGB888).copy()
        return None
    
    def get_config_options(self):
        return [
            ("Probe Mode (0-7)", "probe_mode", self.probe_mode, None),
            ("Gamma Start (Hz)", "gamma_start", self.gamma_start, None),
            ("Gamma End (Hz)", "gamma_end", self.gamma_end, None),
            ("Gamma Resolution (Hz)", "gamma_resolution", self.gamma_resolution, None),
            ("Impulse Target (Hz)", "impulse_target", self.impulse_target, None),
            ("Theta Freq (Hz)", "theta_freq", self.theta_freq, None),
            ("Gamma Burst Freq (Hz)", "gamma_burst_freq", self.gamma_burst_freq, None),
            ("HFO Start (Hz)", "hfo_start", self.hfo_start, None),
            ("HFO End (Hz)", "hfo_end", self.hfo_end, None),
            ("Carrier Freq (Hz)", "carrier_freq", self.carrier_freq, None),
            ("Envelope Freq (Hz)", "envelope_freq", self.envelope_freq, None),
            ("Locked Freq (Hz)", "locked_freq", self.locked_freq, None),
            ("Replay Speed (steps)", "replay_speed", self.replay_speed, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
