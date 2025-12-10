"""
Signal to Field Converter
=========================
Takes raw EEG signal (or any 1D time series) and converts it to a 2D spatial field.

This is the "content" side - you choose what signal to send in.
The Brain Sampler provides the "key" (timing + structure).
When you interfere them, you test: "Does this signal unlock this brain state?"

Methods:
1. Phase Space Embedding (Takens): signal(t) vs signal(t-τ)
2. Time-Frequency (Spectrogram): frequency vs time
3. Delay Matrix: Create 2D texture from 1D signal delays
"""

import numpy as np
import cv2
from scipy.signal import stft, hilbert
from scipy.ndimage import gaussian_filter

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None
    from PyQt6 import QtGui

class SignalToFieldNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Signal → Field (For Interference)"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Cyan
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal',        # Any 1D signal (theta, alpha, gamma, whatever)
            'method_select': 'signal',    # 0=Phase Space, 1=Spectrogram, 2=Delay Matrix
            'field_size': 'signal'        # Resolution (default 64)
        }
        
        self.outputs = {
            'field_visual': 'image',           # The 2D field visualization
            'complex_field': 'complex_spectrum', # FFT for interference
            'raw_field': 'image'               # Raw field data as image
        }
        
        # Config
        self.buffer_size = 512
        self.default_size = 64
        self.fs = 256.0  # Sampling rate assumption
        
        # State
        self.signal_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.current_field = None
        self.current_complex = None
        
    def step(self):
        # 1. Get Input
        val = self.get_blended_input('signal_in', 'sum')
        if val is None: val = 0.0
        
        # Roll buffer
        self.signal_buffer[:-1] = self.signal_buffer[1:]
        self.signal_buffer[-1] = val
        
        if np.all(self.signal_buffer == 0):
            return
        
        # 2. Get method and size
        method = int(self.get_blended_input('method_select', 'sum') or 0)
        field_size = int(self.get_blended_input('field_size', 'sum') or self.default_size)
        field_size = max(32, min(128, field_size))  # Clamp to reasonable range
        
        # 3. Convert signal to 2D field
        if method == 0:
            field = self._phase_space_field(field_size)
        elif method == 1:
            field = self._spectrogram_field(field_size)
        else:
            field = self._delay_matrix_field(field_size)
        
        self.current_field = field
        
        # 4. Compute FFT for interference
        # Center and normalize before FFT
        field_centered = field - np.mean(field)
        if np.std(field_centered) > 1e-9:
            field_centered /= np.std(field_centered)
        
        # Apply Gaussian window to reduce edge artifacts
        window = self._gaussian_window_2d(field_size)
        field_windowed = field_centered * window
        
        # FFT and shift
        complex_fft = np.fft.fftshift(np.fft.fft2(field_windowed))
        self.current_complex = complex_fft
    
    def _phase_space_field(self, size):
        """
        Method 0: Phase Space Embedding (Like the Box visualizer)
        Creates trajectory: signal(t) vs signal(t-delay)
        """
        delay = 15  # ~150ms at 100Hz
        
        # Get recent window
        window = self.signal_buffer[-256:]
        
        if len(window) <= delay:
            return np.zeros((size, size), dtype=np.float32)
        
        # Create trajectory
        x_traj = window[:-delay]
        y_traj = window[delay:]
        
        # Z-score normalize (zoom in on structure)
        x_norm = (x_traj - np.mean(x_traj)) / (np.std(x_traj) + 1e-9)
        y_norm = (y_traj - np.mean(y_traj)) / (np.std(y_traj) + 1e-9)
        
        # Clip to reasonable range (±3 sigma)
        x_norm = np.clip(x_norm, -3, 3)
        y_norm = np.clip(y_norm, -3, 3)
        
        # Create 2D density map
        field, _, _ = np.histogram2d(x_norm, y_norm, bins=size, 
                                      range=[[-3, 3], [-3, 3]])
        
        # Smooth slightly for continuous field
        field = gaussian_filter(field, sigma=1.0)
        
        # Normalize
        if field.max() > 0:
            field = field / field.max()
        
        return field.astype(np.float32)
    
    def _spectrogram_field(self, size):
        """
        Method 1: Time-Frequency Spectrogram
        Shows frequency content over time
        """
        # Use recent window
        window = self.signal_buffer[-256:]
        
        # Compute STFT
        f, t, Zxx = stft(window, fs=self.fs, nperseg=64, noverlap=48)
        
        # Take magnitude
        spec = np.abs(Zxx)
        
        # Resize to target size
        spec_resized = cv2.resize(spec, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # Log scale for better visualization
        spec_log = np.log1p(spec_resized)
        
        # Normalize
        if spec_log.max() > 0:
            spec_log = spec_log / spec_log.max()
        
        return spec_log.astype(np.float32)
    
    def _delay_matrix_field(self, size):
        """
        Method 2: Delay Embedding Matrix
        Creates 2D texture from multiple time delays
        Similar to building a "memory trace"
        """
        # Use recent window
        window = self.signal_buffer[-size*4:]  # Need enough samples
        
        if len(window) < size * 2:
            return np.zeros((size, size), dtype=np.float32)
        
        # Create matrix where each row is the signal at different delays
        field = np.zeros((size, size), dtype=np.float32)
        
        max_delay = size
        step = max(1, len(window) // size)
        
        for row in range(size):
            delay = row
            if delay < len(window):
                # Extract delayed signal segment
                segment = window[-(size + delay):len(window)-delay if delay > 0 else None]
                if len(segment) >= size:
                    # Resample to size
                    indices = np.linspace(0, len(segment)-1, size).astype(int)
                    field[row, :] = segment[indices]
        
        # Normalize per row (like different frequency bands)
        for row in range(size):
            row_data = field[row, :]
            if np.std(row_data) > 1e-9:
                field[row, :] = (row_data - np.mean(row_data)) / np.std(row_data)
        
        # Overall normalization
        field = np.clip(field, -3, 3)
        field = (field + 3) / 6  # Map to [0, 1]
        
        return field.astype(np.float32)
    
    def _gaussian_window_2d(self, size):
        """Create 2D Gaussian window to reduce edge effects in FFT"""
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        window = np.exp(-(X**2 + Y**2) / 2)
        return window
    
    def get_output(self, port_name):
        if port_name == 'field_visual':
            if self.current_field is not None:
                # Colorized visualization
                viz = (self.current_field * 255).astype(np.uint8)
                return cv2.applyColorMap(viz, cv2.COLORMAP_VIRIDIS)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        elif port_name == 'complex_field':
            if self.current_complex is not None:
                return self.current_complex
            return np.zeros((64, 64), dtype=np.complex128)
        
        elif port_name == 'raw_field':
            if self.current_field is not None:
                # Grayscale version
                viz = (self.current_field * 255).astype(np.uint8)
                return viz
            return np.zeros((64, 64), dtype=np.uint8)
        
        return None
    
    def get_display_image(self):
        if self.current_field is None:
            return np.zeros((256, 512, 3), dtype=np.uint8)
        
        # Create split view
        display = np.zeros((256, 512, 3), dtype=np.uint8)
        
        # LEFT: Spatial field (real space)
        field_viz = (self.current_field * 255).astype(np.uint8)
        field_color = cv2.applyColorMap(field_viz, cv2.COLORMAP_VIRIDIS)
        field_large = cv2.resize(field_color, (256, 256))
        display[:, :256] = field_large
        
        # RIGHT: FFT magnitude (frequency space)
        if self.current_complex is not None:
            fft_mag = np.abs(self.current_complex)
            fft_log = np.log1p(fft_mag)
            if fft_log.max() > 0:
                fft_log = fft_log / fft_log.max()
            fft_viz = (fft_log * 255).astype(np.uint8)
            fft_color = cv2.applyColorMap(fft_viz, cv2.COLORMAP_MAGMA)
            fft_large = cv2.resize(fft_color, (256, 256))
            display[:, 256:] = fft_large
        
        # Labels
        cv2.putText(display, "SPATIAL FIELD", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "FFT MAGNITUDE", (270, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Method indicator
        method_names = ["Phase Space", "Spectrogram", "Delay Matrix"]
        method = int(self.get_blended_input('method_select', 'sum') or 0)
        method_text = method_names[min(method, 2)]
        cv2.putText(display, f"Method: {method_text}", (10, 245), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return display
    
    def get_config_options(self):
        return [
            ("Method", "method_select", 0, ["Phase Space", "Spectrogram", "Delay Matrix"]),
            ("Field Size", "field_size", 64, "int"),
            ("Sampling Rate", "fs", 256.0, "float")
        ]