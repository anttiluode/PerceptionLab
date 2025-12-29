"""
Crystal Probe Node
==================

Interrogates a frozen crystal to discover what it "knows."

The crystal has learned a structure from EEG. That structure
encodes something - preferences, resonances, patterns it
recognizes. This node probes to find out what.

Methods:
1. Frequency sweep - which frequencies make it resonate?
2. Spatial patterns - which electrode patterns activate it most?
3. Impulse response - poke it and watch what happens
4. Resonance detection - find the crystal's natural frequencies

The crystal's response tells us what it learned.
What patterns does it "want" to produce?
What was it thinking about?
"""

import numpy as np
import cv2

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


class CrystalProbeNode(BaseNode):
    """
    Probes a frozen crystal to discover its learned structure.
    """
    
    NODE_NAME = "Crystal Probe"
    NODE_TITLE = "Crystal Probe"
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(100, 180, 100) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "lfp_signal": "signal",      # From crystal's LFP output
            "crystal_view": "image",      # Crystal's weight structure
            "trigger": "signal",          # Start a probe sequence
        }
        
        self.outputs = {
            "probe_signal": "signal",     # Signal to inject into crystal
            "resonance_map": "image",     # Which frequencies resonate
            "analysis_view": "image",     # Main display
            "dominant_freq": "signal",    # Crystal's preferred frequency
            "response_strength": "signal", # How strongly it responds
        }
        
        # Probe modes
        self.probe_modes = ["frequency_sweep", "impulse", "noise", "chirp", "pattern"]
        self.current_mode = "frequency_sweep"
        self.mode_index = 0
        
        # Frequency sweep parameters
        self.freq_min = 1.0    # Hz
        self.freq_max = 40.0   # Hz (covers delta through gamma)
        self.freq_current = 1.0
        self.freq_step = 0.5
        self.sweep_time = 0.0
        self.samples_per_freq = 100  # How long to test each frequency
        self.sample_count = 0
        
        # Response recording
        self.frequency_responses = {}  # freq -> average response amplitude
        self.current_responses = []    # Responses at current frequency
        
        # Impulse mode
        self.impulse_countdown = 0
        self.impulse_interval = 50  # Steps between impulses
        self.impulse_response_buffer = []
        
        # Analysis results
        self.dominant_frequency = 0.0
        self.peak_response = 0.0
        self.resonance_spectrum = np.zeros(80)  # 0.5 Hz bins from 0-40 Hz
        
        # Display
        self.display_image = None
        self.history_length = 200
        self.lfp_history = np.zeros(self.history_length)
        self.probe_history = np.zeros(self.history_length)
        
        # State
        self.step_count = 0
        self.is_probing = False
        self.probe_complete = False
        
        self._update_display()
    
    def get_config_options(self):
        mode_options = [(m, m) for m in self.probe_modes]
        return [
            ("Probe Mode", "current_mode", self.current_mode, mode_options),
            ("Min Frequency (Hz)", "freq_min", self.freq_min, None),
            ("Max Frequency (Hz)", "freq_max", self.freq_max, None),
            ("Samples per Frequency", "samples_per_freq", self.samples_per_freq, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _read_input(self, name, default=0.0):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                return float(val)
            except:
                return default
        return default
    
    def step(self):
        self.step_count += 1
        
        # Read crystal's response
        lfp = self._read_input("lfp_signal", 0.0)
        trigger = self._read_input("trigger", 0.0)
        
        # Update history
        self.lfp_history[:-1] = self.lfp_history[1:]
        self.lfp_history[-1] = lfp
        
        # Start probing on trigger
        if trigger > 0.5 and not self.is_probing:
            self._start_probe()
        
        # Generate probe signal based on mode
        probe_signal = self._generate_probe_signal()
        
        # Update probe history
        self.probe_history[:-1] = self.probe_history[1:]
        self.probe_history[-1] = probe_signal
        
        # Record response if probing
        if self.is_probing:
            self._record_response(lfp)
        
        # Store outputs
        self._output_values = {
            "probe_signal": probe_signal,
            "dominant_freq": self.dominant_frequency,
            "response_strength": self.peak_response,
        }
        
        self._update_display()
    
    def _start_probe(self):
        """Initialize a new probe sequence."""
        self.is_probing = True
        self.probe_complete = False
        self.freq_current = self.freq_min
        self.sample_count = 0
        self.frequency_responses = {}
        self.current_responses = []
        self.resonance_spectrum = np.zeros(80)
        print(f"[CrystalProbe] Starting {self.current_mode} probe...")
    
    def _generate_probe_signal(self):
        """Generate the probe signal based on current mode."""
        
        if self.current_mode == "frequency_sweep":
            # Sine wave at current frequency
            t = self.step_count * 0.01  # Assume ~100 steps/sec
            signal = np.sin(2 * np.pi * self.freq_current * t) * 50.0
            return signal
            
        elif self.current_mode == "impulse":
            # Brief impulse every N steps
            self.impulse_countdown -= 1
            if self.impulse_countdown <= 0:
                self.impulse_countdown = self.impulse_interval
                self.impulse_response_buffer = []
                return 100.0  # Strong impulse
            return 0.0
            
        elif self.current_mode == "noise":
            # White noise - tests all frequencies simultaneously
            return np.random.randn() * 30.0
            
        elif self.current_mode == "chirp":
            # Frequency increases over time
            t = self.step_count * 0.01
            freq = self.freq_min + (self.freq_max - self.freq_min) * (t % 10.0) / 10.0
            return np.sin(2 * np.pi * freq * t) * 50.0
            
        elif self.current_mode == "pattern":
            # Alpha-theta pattern (common brain rhythm)
            t = self.step_count * 0.01
            alpha = np.sin(2 * np.pi * 10.0 * t) * 30.0
            theta = np.sin(2 * np.pi * 6.0 * t) * 20.0
            return alpha + theta
        
        return 0.0
    
    def _record_response(self, lfp):
        """Record the crystal's response to current probe."""
        
        if self.current_mode == "frequency_sweep":
            self.current_responses.append(abs(lfp))
            self.sample_count += 1
            
            # Move to next frequency
            if self.sample_count >= self.samples_per_freq:
                # Calculate average response at this frequency
                avg_response = np.mean(self.current_responses) if self.current_responses else 0
                self.frequency_responses[self.freq_current] = avg_response
                
                # Update spectrum
                bin_idx = int((self.freq_current - 0.0) / 0.5)
                if 0 <= bin_idx < len(self.resonance_spectrum):
                    self.resonance_spectrum[bin_idx] = avg_response
                
                # Advance frequency
                self.freq_current += self.freq_step
                self.sample_count = 0
                self.current_responses = []
                
                # Check if sweep complete
                if self.freq_current > self.freq_max:
                    self._analyze_results()
                    self.is_probing = False
                    self.probe_complete = True
                    
        elif self.current_mode == "impulse":
            self.impulse_response_buffer.append(lfp)
            # Analyze impulse response after collecting enough samples
            if len(self.impulse_response_buffer) >= self.impulse_interval:
                self._analyze_impulse_response()
    
    def _analyze_results(self):
        """Analyze the frequency sweep results."""
        if not self.frequency_responses:
            return
        
        # Find dominant frequency
        freqs = list(self.frequency_responses.keys())
        responses = list(self.frequency_responses.values())
        
        if responses:
            max_idx = np.argmax(responses)
            self.dominant_frequency = freqs[max_idx]
            self.peak_response = responses[max_idx]
            
            print(f"[CrystalProbe] Analysis complete!")
            print(f"  Dominant frequency: {self.dominant_frequency:.1f} Hz")
            print(f"  Peak response: {self.peak_response:.2f}")
            
            # Identify frequency band
            if self.dominant_frequency < 4:
                band = "DELTA (deep sleep, unconscious)"
            elif self.dominant_frequency < 8:
                band = "THETA (drowsy, meditative)"
            elif self.dominant_frequency < 13:
                band = "ALPHA (relaxed, eyes closed)"
            elif self.dominant_frequency < 30:
                band = "BETA (alert, active thinking)"
            else:
                band = "GAMMA (high cognition, binding)"
            
            print(f"  Band: {band}")
    
    def _analyze_impulse_response(self):
        """Analyze impulse response to find natural frequencies."""
        if len(self.impulse_response_buffer) < 10:
            return
        
        # Simple FFT of impulse response
        response = np.array(self.impulse_response_buffer)
        fft = np.abs(np.fft.rfft(response))
        freqs = np.fft.rfftfreq(len(response), d=0.01)  # Assume 100 Hz sampling
        
        if len(fft) > 1:
            peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
            if peak_idx < len(freqs):
                self.dominant_frequency = freqs[peak_idx]
                self.peak_response = fft[peak_idx]
    
    def get_output(self, port_name):
        if port_name == "analysis_view":
            return self.display_image
        elif port_name == "resonance_map":
            return self._render_resonance_map()
        elif port_name == "probe_signal":
            return self._output_values.get("probe_signal", 0.0)
        elif port_name in ["dominant_freq", "response_strength"]:
            return self._output_values.get(port_name, 0.0)
        return None
    
    def _render_resonance_map(self):
        """Render the frequency resonance spectrum as an image."""
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if np.max(self.resonance_spectrum) > 0:
            spectrum_norm = self.resonance_spectrum / np.max(self.resonance_spectrum)
        else:
            spectrum_norm = self.resonance_spectrum
        
        bar_width = w // len(self.resonance_spectrum)
        
        for i, val in enumerate(spectrum_norm):
            x = i * bar_width
            bar_h = int(val * (h - 20))
            
            # Color by frequency band
            freq = i * 0.5
            if freq < 4:
                color = (255, 100, 100)  # Delta - red
            elif freq < 8:
                color = (255, 200, 100)  # Theta - orange
            elif freq < 13:
                color = (100, 255, 100)  # Alpha - green
            elif freq < 30:
                color = (100, 200, 255)  # Beta - cyan
            else:
                color = (200, 100, 255)  # Gamma - purple
            
            cv2.rectangle(img, (x, h - bar_h), (x + bar_width - 1, h), color, -1)
        
        # Labels
        cv2.putText(img, "D", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        cv2.putText(img, "T", (30, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(img, "A", (55, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(img, "B", (90, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        cv2.putText(img, "G", (160, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)
        
        return img
    
    def _update_display(self):
        """Create the main analysis display."""
        w, h = 400, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "CRYSTAL PROBE", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 100), 2)
        
        # Mode
        mode_color = (0, 255, 255) if self.is_probing else (150, 150, 150)
        cv2.putText(img, f"Mode: {self.current_mode}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # Status
        if self.is_probing:
            status = f"Probing... {self.freq_current:.1f} Hz"
            cv2.putText(img, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif self.probe_complete:
            cv2.putText(img, "Probe complete!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(img, "Send trigger to start", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Results
        if self.dominant_frequency > 0:
            cv2.putText(img, f"Dominant: {self.dominant_frequency:.1f} Hz", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)
            cv2.putText(img, f"Response: {self.peak_response:.1f}", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
            
            # Band interpretation
            if self.dominant_frequency < 4:
                band = "DELTA - Deep/Unconscious"
                color = (255, 100, 100)
            elif self.dominant_frequency < 8:
                band = "THETA - Meditative/Drowsy"
                color = (255, 200, 100)
            elif self.dominant_frequency < 13:
                band = "ALPHA - Relaxed/Visual"
                color = (100, 255, 100)
            elif self.dominant_frequency < 30:
                band = "BETA - Active/Alert"
                color = (100, 200, 255)
            else:
                band = "GAMMA - High Cognition"
                color = (200, 100, 255)
            
            cv2.putText(img, band, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw waveforms
        wave_y = 200
        wave_h = 80
        
        # LFP response (white)
        if np.max(np.abs(self.lfp_history)) > 0:
            lfp_norm = self.lfp_history / (np.max(np.abs(self.lfp_history)) + 1e-6)
        else:
            lfp_norm = self.lfp_history
        
        for i in range(len(lfp_norm) - 1):
            x1 = int(i * w / len(lfp_norm))
            x2 = int((i + 1) * w / len(lfp_norm))
            y1 = int(wave_y - lfp_norm[i] * wave_h / 2)
            y2 = int(wave_y - lfp_norm[i + 1] * wave_h / 2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Probe signal (green)
        if np.max(np.abs(self.probe_history)) > 0:
            probe_norm = self.probe_history / (np.max(np.abs(self.probe_history)) + 1e-6)
        else:
            probe_norm = self.probe_history
        
        for i in range(len(probe_norm) - 1):
            x1 = int(i * w / len(probe_norm))
            x2 = int((i + 1) * w / len(probe_norm))
            y1 = int(wave_y - probe_norm[i] * wave_h / 2)
            y2 = int(wave_y - probe_norm[i + 1] * wave_h / 2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Baseline
        cv2.line(img, (0, wave_y), (w, wave_y), (50, 50, 50), 1)
        
        # Legend
        cv2.putText(img, "Response", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, "Probe", (100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Mini spectrum
        if np.max(self.resonance_spectrum) > 0:
            spec_x = w - 100
            spec_w = 90
            spec_h = 40
            spec_y = 45
            
            spectrum_norm = self.resonance_spectrum / np.max(self.resonance_spectrum)
            bar_w = spec_w // len(spectrum_norm)
            
            for i, val in enumerate(spectrum_norm[:spec_w // max(bar_w, 1)]):
                x = spec_x + i * bar_w
                bar_h = int(val * spec_h)
                cv2.rectangle(img, (x, spec_y + spec_h - bar_h), (x + bar_w - 1, spec_y + spec_h), (100, 180, 100), -1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image