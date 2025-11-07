"""
Fractal Analyzer Node - Extracts fractal dimension and multi-scale energy
from an input image and outputs a synthesized fractal texture.
Ported from 'app.py' logic.

Outputs:
- fractal_beta: The estimated fractal exponent (slope)
- wavelet_energy: 1D image barcode of energy levels
- fractal_twin: Synthesized image (Random Phase)

Place this file in the 'nodes' folder
Requires: pip install numpy scipy pywt
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from scipy.fft import fft2, fftshift, fftfreq, ifft2 # <--- FIX: Added ifft2
from scipy.stats import linregress
import pywt

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    # --- FIX: Included ifft2 here as well ---
    from scipy.fft import fft2, fftshift, fftfreq, ifft2 
    from scipy.stats import linregress
    import pywt
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("Warning: FractalAnalyzerNode requires 'scipy' and 'pywt'.")
    print("Please run: pip install scipy pywt")


class FractalAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(220, 180, 40) # Golden Analysis Color
    
    def __init__(self, size=96, fit_range_min=5, levels=5):
        super().__init__()
        self.node_title = "Fractal Explorer"
        
        self.inputs = {'image_in': 'image'}
        self.outputs = {
            'fractal_beta': 'signal',         # The beta exponent (slope)
            'wavelet_energy': 'image',        # 1D image barcode of energy levels
            'fractal_twin': 'image'           # Synthesized image (Random Phase)
        }
        
        self.size = int(size)
        self.fit_range_min = int(fit_range_min)
        self.levels = int(levels)
        
        # Internal state
        self.beta_value = 0.0
        self.wavelet_energy_data = np.zeros(self.levels, dtype=np.float32)
        self.synthesized_img = np.zeros((self.size, self.size), dtype=np.float32)
        
        if not LIBS_AVAILABLE:
            self.node_title = "Fractal (Libs Missing!)"

    def _compute_power_spectrum(self, gray_img):
        """Computes 2D FFT Power Spectrum for fractal analysis."""
        
        # 1. FFT
        F = fftshift(fft2(gray_img))
        
        # 2. Magnitude and Power
        F_mag = np.abs(F)
        F_power = F_mag**2
        
        # 3. Frequency Radii (K-space)
        Ny, Nx = gray_img.shape
        kx = fftfreq(Nx)
        ky = fftfreq(Ny)
        KX, KY = np.meshgrid(kx, ky)
        R = np.sqrt(KX**2 + KY**2)
        
        # 4. Azimuthal averaging (Average power in rings)
        r_flat = R.flatten()
        p_flat = F_power.flatten()
        
        bins = np.linspace(0, np.max(r_flat), Nx // 4) 
        
        digitized = np.digitize(r_flat, bins)
        
        power_sums = np.array([p_flat[digitized == i].sum() for i in range(1, len(bins) + 1)])
        counts = np.array([p_flat[digitized == i].size for i in range(1, len(bins) + 1)])
        
        # Filter out empty bins and the DC component (first bin)
        valid_indices = (counts > 0) & (power_sums > 0)
        freqs = bins[valid_indices]
        powers = power_sums[valid_indices] / counts[valid_indices]
        
        return freqs, powers

    def _estimate_fractal_exponent(self, freqs, powers):
        """Fits a line to the log-log power spectrum: log(P) = -β * log(f) + I"""
        
        # Filter range: skip DC component and noisy high frequencies
        fit_mask = freqs > self.fit_range_min * (1/self.size) 
        
        if np.sum(fit_mask) < 5: # Need enough points for fit
            return 0.0, 0.0, 0.0
        
        log_freqs = np.log(freqs[fit_mask])
        log_powers = np.log(powers[fit_mask])
        
        slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_powers)
        
        # β is the negative of the slope
        return -slope, intercept, r_value**2
    
    def _synthesize_random_phase(self, gray_img):
        """Creates a synthesized image with the original amplitude spectrum
        but randomized phase (the 'Fractal Twin')."""
        
        # 1. Compute original spectrum (amplitude and phase)
        F_orig = fft2(gray_img)
        F_mag = np.abs(F_orig)
        
        # 2. Generate random phase
        np.random.seed(42) # Deterministic randomness
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(*F_orig.shape))
        
        # 3. New spectrum: Original Magnitude * Random Phase
        F_synth = F_mag * random_phase
        
        # 4. Inverse FFT
        img_synth = np.real(ifft2(F_synth))
        
        # 5. Normalize and clip to [0, 1]
        img_synth -= img_synth.min()
        img_synth /= (img_synth.max() + 1e-9)
        
        return img_synth.astype(np.float32)

    def _compute_wavelet_energies(self, gray_img):
        """Computes energy sum for DWT coefficients at each level."""
        
        coeffs = pywt.wavedec2(gray_img, wavelet='haar', level=self.levels)
        
        energies = []
        cA = coeffs[0]
        energies.append(np.sum(cA**2))
        
        for detail in coeffs[1:]:
            cH, cV, cD = detail
            level_energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
            energies.append(level_energy)
            
        energies = np.array(energies)
        total_energy = np.sum(energies)
        
        # Output is normalized energy distribution
        return energies / (total_energy + 1e-9)


    def step(self):
        if not LIBS_AVAILABLE:
            return
            
        img_in = self.get_blended_input('image_in', 'mean')
        
        if img_in is None:
            self.beta_value *= 0.9 # Decay signal output
            return
        
        # Ensure image is 2D and resize
        if img_in.ndim == 3:
             img_in = cv2.cvtColor((img_in * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
             
        # Resize to internal resolution
        gray_img = cv2.resize(img_in, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        # 1. Compute Fractal Beta (Slope)
        freqs, powers = self._compute_power_spectrum(gray_img)
        beta, intercept, r2 = self._estimate_fractal_exponent(freqs, powers)
        self.beta_value = beta
        
        # 2. Compute Wavelet Energies
        normalized_energies = self._compute_wavelet_energies(gray_img)
        
        # Create a 1D image (barcode) output for energies
        self.wavelet_energy_data = np.zeros(self.levels, dtype=np.float32)
        self.wavelet_energy_data[:len(normalized_energies)-1] = normalized_energies[1:]
        
        # 3. Synthesize Random-Phase Twin
        self.synthesized_img = self._synthesize_random_phase(gray_img)


    def get_output(self, port_name):
        if port_name == 'fractal_beta':
            # Map beta (which is usually around 1-3) to a simple signal scale (0-1)
            return np.clip(self.beta_value / 4.0, 0.0, 1.0)
            
        elif port_name == 'wavelet_energy':
            # Return 1D array as a 1D image/feature vector
            return self.wavelet_energy_data.reshape(1, -1)
            
        elif port_name == 'fractal_twin':
            return self.synthesized_img
            
        return None
        
    def get_display_image(self):
        # Visualize the Synthesized Image
        img_u8 = (np.clip(self.synthesized_img, 0, 1) * 255).astype(np.uint8)
        
        # Add beta value overlay
        h, w = img_u8.shape
        cv2.putText(img_u8, f"β: {self.beta_value:.2f}", (5, h - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)

        img_resized = cv2.resize(img_u8, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape
        return QtGui.QImage(img_resized.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
            ("Fit Range Start (k)", "fit_range_min", self.fit_range_min, None),
            ("Wavelet Levels", "levels", self.levels, None),
        ]