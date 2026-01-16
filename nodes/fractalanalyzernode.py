"""
Robust Fractal Analyzer Node - Measures scale-invariant structure
Computes fractal beta (power spectrum slope) with robust fallbacks.
Works with natural images, physics simulations, and extreme patterns.

Place this file in the 'nodes' folder as 'fractal_analyzer_robust.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import fft2, fftshift, ifft2, rfftfreq
    from scipy.stats import linregress
    import pywt
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("Warning: FractalAnalyzerNode requires 'scipy' and 'PyWavelets'.")
    print("Please run: pip install scipy pywavelets")


class FractalAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(220, 180, 40)  # Golden Analysis Color
    
    def __init__(self, size=96, fit_range_min=5, levels=5):
        super().__init__()
        self.node_title = "Fractal Analyzer (Robust)"
        
        self.inputs = {'image_in': 'image'}
        self.outputs = {
            'fractal_beta': 'signal',       # Primary: Power spectrum slope
            'complexity': 'signal',          # Fallback: Wavelet-based complexity
            'spectral_energy': 'signal',     # Total high-frequency energy
            'spectrum_image': 'image',       # Visualization of power spectrum
            'fractal_twin': 'image'          # Synthesized random-phase version
        }
        
        self.size = int(size)
        self.fit_range_min = int(fit_range_min)
        self.levels = int(levels)
        
        # Internal state
        self.fractal_beta = 0.0
        self.complexity_value = 0.0
        self.spectral_energy = 0.0
        self.last_power_spectrum = None
        self.synthesized_img = np.zeros((self.size, self.size), dtype=np.float32)
        self.measurement_method = "none"  # Track which method succeeded
        
        if not LIBS_AVAILABLE:
            self.node_title = "Fractal (Libs Missing!)"

    def _compute_radial_profile(self, power_2d):
        """
        Compute radially averaged power spectrum.
        Returns: (frequencies, radial_power)
        """
        h, w = power_2d.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radius map
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        # Radial binning
        r_max = min(center_x, center_y)
        radial_profile = np.zeros(r_max)
        radial_counts = np.zeros(r_max)
        
        for radius in range(r_max):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = np.mean(power_2d[mask])
                radial_counts[radius] = np.sum(mask)
        
        # Only return frequencies with sufficient samples
        valid = radial_counts > 0
        frequencies = np.arange(r_max)[valid]
        radial_power = radial_profile[valid]
        
        return frequencies, radial_power

    def _robust_fractal_beta(self, gray_img):
        """
        Primary method: Compute fractal beta from power spectrum slope.
        Returns: (beta, success_flag, method_name)
        """
        try:
            # 1. Compute 2D FFT
            F = fft2(gray_img)
            power_2d = np.abs(fftshift(F))**2
            
            # 2. Add epsilon to prevent log(0)
            power_2d += 1e-10
            
            # 3. Store for visualization
            self.last_power_spectrum = power_2d
            
            # 4. Compute radial average
            freqs, radial_power = self._compute_radial_profile(power_2d)
            
            # 5. Skip DC component and ensure we have enough points
            if len(freqs) < self.fit_range_min:
                return 0.0, False, "too_few_points"
            
            freqs = freqs[1:]  # Skip r=0 (DC)
            radial_power = radial_power[1:]
            
            # 6. Fit only in valid frequency range
            fit_start = max(1, self.fit_range_min)
            fit_end = len(freqs)
            
            if fit_end - fit_start < 3:
                return 0.0, False, "insufficient_range"
            
            log_freqs = np.log(freqs[fit_start:fit_end])
            log_power = np.log(radial_power[fit_start:fit_end])
            
            # 7. Check for valid values (no NaN, no Inf)
            valid_mask = np.isfinite(log_freqs) & np.isfinite(log_power)
            if np.sum(valid_mask) < 3:
                return 0.0, False, "invalid_values"
            
            log_freqs = log_freqs[valid_mask]
            log_power = log_power[valid_mask]
            
            # 8. Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_power)
            
            # 9. Sanity check: beta should be negative and reasonable
            if not np.isfinite(slope):
                return 0.0, False, "infinite_slope"
            
            if slope > 0:  # Physically impossible for power spectrum
                return 0.0, False, "positive_slope"
            
            if slope < -10:  # Probably numerical error
                return -10.0, True, "clamped_low"
            
            # 10. Success!
            return slope, True, "fractal_beta"
            
        except Exception as e:
            return 0.0, False, f"exception_{type(e).__name__}"

    def _wavelet_complexity(self, gray_img):
        """
        Fallback method 1: Wavelet-based complexity measure.
        Returns: (complexity, success_flag, method_name)
        """
        try:
            # Compute DWT
            coeffs = pywt.wavedec2(gray_img, wavelet='db4', level=self.levels)
            
            # Compute energy at each level
            energies = []
            
            # Approximation (low-frequency)
            cA = coeffs[0]
            low_freq_energy = np.sum(cA**2)
            energies.append(low_freq_energy)
            
            # Details (high-frequency)
            high_freq_energy = 0.0
            for detail in coeffs[1:]:
                cH, cV, cD = detail
                level_energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
                energies.append(level_energy)
                high_freq_energy += level_energy
            
            # Complexity = ratio of high-freq to low-freq energy
            total_energy = np.sum(energies)
            if total_energy < 1e-10:
                return 0.0, False, "zero_energy"
            
            complexity = high_freq_energy / total_energy
            
            # Convert to pseudo-beta (map [0,1] to [-3, -1])
            pseudo_beta = -3.0 + complexity * 2.0
            
            return pseudo_beta, True, "wavelet_fallback"
            
        except Exception as e:
            return 0.0, False, f"wavelet_exception_{type(e).__name__}"

    def _std_complexity(self, gray_img):
        """
        Fallback method 2: Simple standard deviation.
        Returns: (complexity, success_flag, method_name)
        """
        try:
            std = np.std(gray_img)
            
            # Convert to pseudo-beta (map std [0, 0.5] to [-3, -1])
            pseudo_beta = -3.0 + np.clip(std * 4.0, 0, 1) * 2.0
            
            return pseudo_beta, True, "std_fallback"
            
        except:
            return -2.0, True, "default_fallback"

    def _synthesize_random_phase(self, gray_img):
        """
        Create a 'fractal twin' with same amplitude spectrum but random phase.
        """
        try:
            F_orig = fft2(gray_img)
            F_mag = np.abs(F_orig)
            
            # Deterministic random phase
            np.random.seed(42)
            random_phase = np.exp(1j * 2 * np.pi * np.random.rand(*F_orig.shape))
            
            F_synth = F_mag * random_phase
            img_synth = np.real(ifft2(F_synth))
            
            # Normalize to [0, 1]
            img_synth -= img_synth.min()
            img_synth /= (img_synth.max() + 1e-9)
            
            return img_synth.astype(np.float32)
            
        except:
            return np.zeros_like(gray_img, dtype=np.float32)

    def _generate_spectrum_visualization(self):
        """
        Create a visual representation of the power spectrum.
        """
        if self.last_power_spectrum is None:
            return np.zeros((64, 64), dtype=np.float32)
        
        # Log scale for better visualization
        log_power = np.log(self.last_power_spectrum + 1e-10)
        
        # Normalize
        log_power -= log_power.min()
        log_power /= (log_power.max() + 1e-9)
        
        # Resize for display
        vis = cv2.resize(log_power, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        return vis.astype(np.float32)

    def step(self):
        if not LIBS_AVAILABLE:
            return
        
        # Get input image (use 'first' to avoid blending issues)
        img_in = self.get_blended_input('image_in', 'first')
        
        # --- FIX: Guard against Bad Data (Strings/Non-Arrays) ---
        if img_in is not None:
            if isinstance(img_in, (str, np.str_)):
                img_in = None
            elif not isinstance(img_in, np.ndarray):
                img_in = None
            elif not hasattr(img_in, 'ndim'):
                img_in = None
        # --------------------------------------------------------
        
        if img_in is None:
            # Decay outputs when no input
            self.fractal_beta *= 0.95
            self.complexity_value *= 0.95
            self.spectral_energy *= 0.95
            return
        
        # Ensure grayscale
        if img_in.ndim == 3:
            if img_in.shape[2] == 4:  # RGBA
                img_in = cv2.cvtColor(img_in.astype(np.float32), cv2.COLOR_RGBA2GRAY)
            else:  # RGB/BGR
                img_in = cv2.cvtColor(img_in.astype(np.float32), cv2.COLOR_BGR2GRAY)
        
        # Resize to working resolution
        gray_img = cv2.resize(img_in, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        if gray_img.max() > 1.0:
            gray_img = gray_img / 255.0
        
        # === Cascade of measurement methods ===
        
        # Method 1: Try fractal beta (primary)
        beta, success, method = self._robust_fractal_beta(gray_img)
        
        if success:
            self.fractal_beta = beta
            self.measurement_method = method
        else:
            # Method 2: Try wavelet complexity (fallback 1)
            beta, success, method = self._wavelet_complexity(gray_img)
            
            if success:
                self.fractal_beta = beta
                self.measurement_method = method
            else:
                # Method 3: Use std dev (fallback 2)
                beta, success, method = self._std_complexity(gray_img)
                self.fractal_beta = beta
                self.measurement_method = method
        
        # Compute spectral energy (total high-frequency content)
        if self.last_power_spectrum is not None:
            center = self.size // 2
            high_freq_mask = np.zeros_like(self.last_power_spectrum)
            y, x = np.ogrid[:self.size, :self.size]
            r = np.sqrt((x - center)**2 + (y - center)**2)
            high_freq_mask[r > center // 2] = 1.0
            self.spectral_energy = np.sum(self.last_power_spectrum * high_freq_mask)
            self.spectral_energy = np.log10(self.spectral_energy + 1.0) / 10.0  # Normalize
        
        # Compute wavelet-based complexity (always, for secondary output)
        _, wavelet_success, _ = self._wavelet_complexity(gray_img)
        if wavelet_success:
            # Store as 0-1 normalized complexity
            self.complexity_value = (self.fractal_beta + 3.0) / 2.0  # Map [-3,-1] to [0,1]
        
        # Synthesize fractal twin
        self.synthesized_img = self._synthesize_random_phase(gray_img)

    def get_output(self, port_name):
        if port_name == 'fractal_beta':
            return self.fractal_beta
        
        elif port_name == 'complexity':
            return self.complexity_value
        
        elif port_name == 'spectral_energy':
            return self.spectral_energy
        
        elif port_name == 'spectrum_image':
            return self._generate_spectrum_visualization()
        
        elif port_name == 'fractal_twin':
            return self.synthesized_img
        
        return None
    
    def get_display_image(self):
        if not LIBS_AVAILABLE:
            return None
        
        # Show the synthesized fractal twin
        img_u8 = (np.clip(self.synthesized_img, 0, 1) * 255).astype(np.uint8)
        
        # Overlay the fractal beta value and method
        h, w = img_u8.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Format beta with sign
        beta_text = f"β: {self.fractal_beta:.2f}"
        method_text = f"{self.measurement_method}"
        
        # Draw text with shadow for readability
        cv2.putText(img_u8, beta_text, (6, h - 16), font, 0.3, 0, 1, cv2.LINE_AA)
        cv2.putText(img_u8, beta_text, (5, h - 17), font, 0.3, 255, 1, cv2.LINE_AA)
        
        cv2.putText(img_u8, method_text, (6, h - 4), font, 0.25, 0, 1, cv2.LINE_AA)
        cv2.putText(img_u8, method_text, (5, h - 5), font, 0.25, 200, 1, cv2.LINE_AA)
        
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
            ("Fit Range Min", "fit_range_min", self.fit_range_min, None),
            ("Wavelet Levels", "levels", self.levels, None),
        ]