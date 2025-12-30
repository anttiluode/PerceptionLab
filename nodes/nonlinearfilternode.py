"""
Nonlinear Filter Analysis Node
==============================

Visualizes what a nonlinear system (like the crystal) does to a signal.

Shows:
1. INPUT SPECTRUM - What frequencies went in
2. OUTPUT SPECTRUM - What frequencies came out  
3. TRANSFER FUNCTION - Output/Input ratio (what's amplified/suppressed)
4. GENERATED FREQUENCIES - Frequencies in output that weren't in input
5. PHASE RELATIONSHIP - How output timing relates to input

This reveals the crystal as a nonlinear filter - not just passing frequencies
but creating new ones through its learned geometry.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque

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


class NonlinearFilterNode(BaseNode):
    """
    Analyzes the nonlinear filtering properties of a system.
    """
    
    NODE_NAME = "Nonlinear Filter"
    NODE_TITLE = "Nonlinear Filter"
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(200, 50, 150) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "signal_in": "signal",    # The input to the system (probe signal)
            "signal_out": "signal",   # The output from the system (crystal response)
        }
        
        self.outputs = {
            "analysis_view": "image",      # Main visualization
            "transfer_function": "image",  # Transfer function plot
            "generated_freqs": "signal",   # Amount of generated frequencies
            "nonlinearity": "signal",      # Nonlinearity measure (0=linear, 1=very nonlinear)
            "dominant_harmonic": "signal", # Strongest harmonic ratio
        }
        
        # Buffer settings
        self.buffer_size = 512
        self.input_buffer = deque([0.0] * self.buffer_size, maxlen=self.buffer_size)
        self.output_buffer = deque([0.0] * self.buffer_size, maxlen=self.buffer_size)
        
        # Analysis results
        self.input_spectrum = np.zeros(self.buffer_size // 2)
        self.output_spectrum = np.zeros(self.buffer_size // 2)
        self.transfer_function = np.ones(self.buffer_size // 2)
        self.generated_spectrum = np.zeros(self.buffer_size // 2)
        
        # Metrics
        self.nonlinearity_score = 0.0
        self.generated_power = 0.0
        self.dominant_harmonic = 1.0
        self.phase_lag = 0.0
        
        # Sample rate assumption
        self.sample_rate = 100.0  # Hz
        
        # Display
        self.step_count = 0
        self.display_image = None
        
        self._update_display()
    
    def _read_signal(self, name, default=0.0):
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
        
        # Read signals
        sig_in = self._read_signal("signal_in", 0.0)
        sig_out = self._read_signal("signal_out", 0.0)
        
        # Add to buffers
        self.input_buffer.append(sig_in)
        self.output_buffer.append(sig_out)
        
        # Analyze every 16 steps for efficiency
        if self.step_count % 16 == 0:
            self._analyze_signals()
        
        # Update display every 8 steps
        if self.step_count % 8 == 0:
            self._update_display()
    
    def _analyze_signals(self):
        """Compute spectral analysis of input vs output."""
        # Convert to arrays
        input_arr = np.array(self.input_buffer, dtype=np.float32)
        output_arr = np.array(self.output_buffer, dtype=np.float32)
        
        # Remove DC
        input_arr = input_arr - np.mean(input_arr)
        output_arr = output_arr - np.mean(output_arr)
        
        # Window
        window = np.hanning(self.buffer_size)
        input_windowed = input_arr * window
        output_windowed = output_arr * window
        
        # FFT
        input_fft = np.fft.rfft(input_windowed)
        output_fft = np.fft.rfft(output_windowed)
        
        # Power spectra
        self.input_spectrum = np.abs(input_fft) ** 2
        self.output_spectrum = np.abs(output_fft) ** 2
        
        # Avoid division by zero
        input_safe = np.maximum(self.input_spectrum, 1e-10)
        
        # Transfer function (output/input ratio)
        self.transfer_function = self.output_spectrum / input_safe
        
        # Generated frequencies: what's in output but not in input
        # Threshold input to find "silent" frequencies
        input_threshold = np.max(self.input_spectrum) * 0.01
        input_mask = self.input_spectrum < input_threshold
        
        # Generated = output power at frequencies where input is quiet
        self.generated_spectrum = self.output_spectrum * input_mask
        
        # Metrics
        total_output_power = np.sum(self.output_spectrum) + 1e-10
        generated_power = np.sum(self.generated_spectrum)
        
        self.generated_power = generated_power
        self.nonlinearity_score = generated_power / total_output_power
        self.nonlinearity_score = np.clip(self.nonlinearity_score, 0, 1)
        
        # Find dominant harmonic
        # Look for peaks in transfer function
        if np.max(self.transfer_function) > 0:
            peak_idx = np.argmax(self.transfer_function[1:]) + 1  # Skip DC
            input_peak_idx = np.argmax(self.input_spectrum[1:]) + 1
            if input_peak_idx > 0:
                self.dominant_harmonic = peak_idx / input_peak_idx
            else:
                self.dominant_harmonic = 1.0
        
        # Phase analysis (cross-correlation)
        correlation = np.correlate(output_arr, input_arr, mode='full')
        lag_idx = np.argmax(correlation) - (self.buffer_size - 1)
        self.phase_lag = lag_idx / self.sample_rate * 1000  # in ms
    
    def get_output(self, port_name):
        if port_name == "analysis_view":
            return self.display_image
        elif port_name == "transfer_function":
            return self._render_transfer_function()
        elif port_name == "generated_freqs":
            return self.generated_power
        elif port_name == "nonlinearity":
            return self.nonlinearity_score
        elif port_name == "dominant_harmonic":
            return self.dominant_harmonic
        return None
    
    def _render_transfer_function(self):
        """Render just the transfer function as an image."""
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Normalize transfer function for display
        tf = self.transfer_function[:w]
        tf_log = np.log10(tf + 1e-10)
        tf_norm = (tf_log - tf_log.min()) / (tf_log.max() - tf_log.min() + 1e-10)
        
        for i in range(min(len(tf_norm), w)):
            bar_h = int(tf_norm[i] * (h - 10))
            color = (100, 255, 200)  # Cyan-ish
            cv2.line(img, (i, h - 5), (i, h - 5 - bar_h), color, 1)
        
        return img
    
    def _update_display(self):
        """Create main visualization."""
        w, h = 600, 500
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "NONLINEAR FILTER ANALYSIS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 50, 150), 2)
        
        # Frequency axis (for all plots)
        freqs = np.fft.rfftfreq(self.buffer_size, 1.0 / self.sample_rate)
        max_freq_idx = min(len(freqs), 200)  # Limit to ~50 Hz display
        
        plot_w = 280
        plot_h = 80
        
        # --- INPUT SPECTRUM ---
        self._draw_spectrum(img, 10, 50, plot_w, plot_h, 
                           self.input_spectrum[:max_freq_idx],
                           "INPUT SPECTRUM", (0, 255, 0))
        
        # --- OUTPUT SPECTRUM ---
        self._draw_spectrum(img, 310, 50, plot_w, plot_h,
                           self.output_spectrum[:max_freq_idx],
                           "OUTPUT SPECTRUM", (0, 200, 255))
        
        # --- TRANSFER FUNCTION ---
        self._draw_spectrum(img, 10, 160, plot_w, plot_h,
                           self.transfer_function[:max_freq_idx],
                           "TRANSFER FUNCTION (Out/In)", (255, 200, 100),
                           log_scale=True)
        
        # --- GENERATED FREQUENCIES ---
        self._draw_spectrum(img, 310, 160, plot_w, plot_h,
                           self.generated_spectrum[:max_freq_idx],
                           "GENERATED (not in input)", (255, 50, 255))
        
        # --- TIME DOMAIN COMPARISON ---
        self._draw_time_signals(img, 10, 280, w - 20, 100)
        
        # --- METRICS ---
        metrics_y = 410
        
        # Nonlinearity meter
        cv2.putText(img, "Nonlinearity:", (10, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        bar_w = int(self.nonlinearity_score * 200)
        cv2.rectangle(img, (120, metrics_y - 15), (120 + bar_w, metrics_y),
                     (255, 50, 255), -1)
        cv2.rectangle(img, (120, metrics_y - 15), (320, metrics_y),
                     (100, 100, 100), 1)
        cv2.putText(img, f"{self.nonlinearity_score:.1%}", (330, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 255), 1)
        
        # Phase lag
        cv2.putText(img, f"Phase Lag: {self.phase_lag:.1f} ms", (10, metrics_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Dominant harmonic
        cv2.putText(img, f"Dominant Harmonic: {self.dominant_harmonic:.2f}x", (200, metrics_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # Generated power
        cv2.putText(img, f"Generated Power: {self.generated_power:.1f}", (400, metrics_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 255), 1)
        
        # Interpretation
        interpretation = self._interpret_results()
        cv2.putText(img, interpretation, (10, metrics_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 200), 1)
        
        # Convert to QImage
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3,
                               QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def _draw_spectrum(self, img, x, y, w, h, spectrum, title, color, log_scale=False):
        """Draw a spectrum plot."""
        # Background
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
        
        # Title
        cv2.putText(img, title, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        if len(spectrum) == 0 or np.max(spectrum) == 0:
            return
        
        # Normalize
        if log_scale:
            spec = np.log10(spectrum + 1e-10)
            spec = spec - spec.min()
        else:
            spec = spectrum.copy()
        
        spec_max = np.max(spec)
        if spec_max > 0:
            spec_norm = spec / spec_max
        else:
            spec_norm = spec
        
        # Draw bars
        n_bins = min(len(spec_norm), w)
        bin_width = max(1, w // n_bins)
        
        for i in range(n_bins):
            bar_h = int(spec_norm[i] * (h - 10))
            bx = x + i * bin_width
            cv2.line(img, (bx, y + h - 5), (bx, y + h - 5 - bar_h), color, 1)
        
        # Frequency labels
        cv2.putText(img, "0", (x + 2, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
        cv2.putText(img, "50Hz", (x + w - 25, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
    
    def _draw_time_signals(self, img, x, y, w, h):
        """Draw input and output signals in time domain."""
        # Background
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
        
        cv2.putText(img, "TIME DOMAIN: Input (green) vs Output (orange)", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Get recent samples
        n_samples = min(200, len(self.input_buffer))
        input_arr = np.array(list(self.input_buffer)[-n_samples:])
        output_arr = np.array(list(self.output_buffer)[-n_samples:])
        
        if len(input_arr) < 2:
            return
        
        # Normalize both to same scale for comparison
        all_vals = np.concatenate([input_arr, output_arr])
        val_min, val_max = np.min(all_vals), np.max(all_vals)
        val_range = val_max - val_min + 1e-10
        
        input_norm = (input_arr - val_min) / val_range
        output_norm = (output_arr - val_min) / val_range
        
        # Draw center line
        center_y = y + h // 2
        cv2.line(img, (x, center_y), (x + w, center_y), (50, 50, 50), 1)
        
        # Draw signals
        for i in range(len(input_arr) - 1):
            px1 = x + int(i / len(input_arr) * w)
            px2 = x + int((i + 1) / len(input_arr) * w)
            
            # Input (green)
            py1_in = y + h - 5 - int(input_norm[i] * (h - 10))
            py2_in = y + h - 5 - int(input_norm[i + 1] * (h - 10))
            cv2.line(img, (px1, py1_in), (px2, py2_in), (0, 255, 0), 1)
            
            # Output (orange)
            py1_out = y + h - 5 - int(output_norm[i] * (h - 10))
            py2_out = y + h - 5 - int(output_norm[i + 1] * (h - 10))
            cv2.line(img, (px1, py1_out), (px2, py2_out), (0, 150, 255), 1)
    
    def _interpret_results(self):
        """Generate human-readable interpretation."""
        if self.nonlinearity_score < 0.1:
            return "Nearly LINEAR - output mostly follows input"
        elif self.nonlinearity_score < 0.3:
            return "MILDLY NONLINEAR - some frequency generation"
        elif self.nonlinearity_score < 0.6:
            return "MODERATELY NONLINEAR - significant transformation"
        else:
            return "HIGHLY NONLINEAR - crystal creating its own dynamics"
    
    def get_display_image(self):
        return self.display_image
    
    def get_config_options(self):
        return [
            ("Sample Rate", "sample_rate", self.sample_rate, None),
            ("Buffer Size", "buffer_size", self.buffer_size, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)