"""
Antti's Image-to-Moiré Node
Applies a signal-controlled band-pass filter in the frequency domain
to isolate specific spatial frequencies, creating Moiré-like patterns.
Inspired by the FFT->filter->IFFT logic in sigh_image.py.
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

try:
    from scipy.fft import fft2, ifft2, fftshift, fftfreq
    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False
    print("Warning: ImageMoireNode requires 'scipy'.")
    print("Please run: pip install scipy")

class ImageMoireNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, resolution=128):
        super().__init__()
        self.node_title = "Image to Moiré"
        
        self.inputs = {
            'image': 'image',
            'peak_freq': 'signal',  # Controls center of frequency band (0 to 1)
            'bandwidth': 'signal' # Controls width of frequency band (0 to 1)
        }
        self.outputs = {'image': 'image'}
        
        self.resolution = int(resolution)
        self.peak_freq = 0.1  # Default peak frequency
        self.bandwidth = 0.1  # Default bandwidth
        
        self.output_image = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Pre-calculate the frequency grid
        self._k_magnitude = self._create_frequency_grid(self.resolution)
        
        if not SCIPY_FFT_AVAILABLE:
            self.node_title = "Moiré (No SciPy!)"

    def _create_frequency_grid(self, n):
        """Creates a centered grid of frequency magnitudes."""
        freq_x = fftshift(fftfreq(n))
        freq_y = fftshift(fftfreq(n))
        fx, fy = np.meshgrid(freq_x, freq_y)
        k_magnitude = np.sqrt(fx**2 + fy**2)
        # Normalize from [0, 0.707] to [0, 1]
        return k_magnitude / 0.707

    def step(self):
        if not SCIPY_FFT_AVAILABLE:
            return

        input_img = self.get_blended_input('image', 'mean')
        
        # Get control signals, mapping from [-1, 1] to [0, 1]
        peak_signal = self.get_blended_input('peak_freq', 'sum')
        bw_signal = self.get_blended_input('bandwidth', 'sum')
        
        # Use signal if connected, else use internal config
        # Map signal from [-1, 1] to [0, 1], or use config [0, 1]
        peak = (peak_signal + 1.0) / 2.0 if peak_signal is not None else self.peak_freq
        bw = (bw_signal + 1.0) / 2.0 if bw_signal is not None else self.bandwidth
        
        if input_img is None:
            self.output_image *= 0.95 # Fade to black
            return
            
        try:
            # Resize image to target resolution
            img_resized = cv2.resize(input_img, (self.resolution, self.resolution),
                                     interpolation=cv2.INTER_AREA)
            
            # --- 1. FFT ---
            field_fft = fftshift(fft2(img_resized))
            
            # --- 2. Create Filter Mask ---
            # Map bandwidth from [0, 1] to a small, usable range
            bw_scaled = bw * 0.05 + 0.005 # e.g., 0.005 to 0.055
            
            # Create a Gaussian ring (band-pass filter)
            distance_from_peak = np.abs(self._k_magnitude - peak)
            filter_mask = np.exp(-(distance_from_peak**2) / (2 * bw_scaled**2))
            
            # --- 3. Apply Filter ---
            filtered_fft = field_fft * filter_mask
            
            # --- 4. IFFT ---
            result = ifft2(filtered_fft) # Already shifted
            result_real = np.abs(result) # Use magnitude
            
            # Normalize for output
            r_min, r_max = result_real.min(), result_real.max()
            if (r_max - r_min) > 1e-9:
                self.output_image = (result_real - r_min) / (r_max - r_min)
            else:
                self.output_image.fill(0.0)
                                           
        except Exception as e:
            print(f"Image Moiré Error: {e}")
            self.output_image *= 0.95

    def get_output(self, port_name):
        if port_name == 'image':
            return self.output_image
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.output_image, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.resolution, self.resolution, self.resolution, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Peak Freq (0-1)", "peak_freq", self.peak_freq, None),
            ("Bandwidth (0-1)", "bandwidth", self.bandwidth, None),
        ]