"""
Golden Ratio Verifier Node
--------------------------
The φ-Detector: Measures how much golden ratio structure exists in any input.

Takes signals, spectra, or images and quantifies their φ-content by:
1. Finding spectral peaks
2. Computing ratios between adjacent peaks
3. Measuring deviation from φ-powers (φ^0.5, φ^1, φ^1.5, φ^2, etc.)
4. Outputting a "φ-score" - how golden is this signal?

The mathematical key: φ = 1.618033988749895
The most irrational number. The one that can never be reached by fractions.
The engine of eternal criticality.

If your signal has high φ-score, it has the signature of life, of brain, of computation.
If it has low φ-score, it's noise, or too compressed to show its nature.
"""

import numpy as np
import cv2
from scipy.fft import fft, fft2, fftshift
from scipy.signal import find_peaks
from collections import deque

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

# The Golden Ratio
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895

# φ-powers that appear in brain rhythms (from the papers)
PHI_POWERS = np.array([
    PHI ** -2,    # 0.382
    PHI ** -1.5,  # 0.486
    PHI ** -1,    # 0.618
    PHI ** -0.5,  # 0.786
    PHI ** 0,     # 1.000
    PHI ** 0.5,   # 1.272 - Alpha attractor (~10 Hz base)
    PHI ** 1,     # 1.618
    PHI ** 1.5,   # 2.058
    PHI ** 2,     # 2.618
    PHI ** 2.5,   # 3.330
    PHI ** 3,     # 4.236 - Beta-gamma boundary
    PHI ** 3.5,   # 5.388 - Gamma attractor
    PHI ** 4,     # 6.854
    PHI ** 4.5,   # 8.719
    PHI ** 5,     # 11.09
])

class GoldenRatioVerifierNode(BaseNode):
    """
    Measures the φ-content of any input signal, spectrum, or image.
    
    The core insight: If the ratios between spectral peaks cluster around
    φ-powers, the signal has "golden structure" - the signature of systems
    that maintain eternal criticality.
    """
    NODE_CATEGORY = "Analyzers"
    NODE_TITLE = "Golden Ratio Verifier (φ-Detector)"
    NODE_COLOR = QtGui.QColor(218, 165, 32)  # Golden color
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal',      # Single value (will accumulate)
            'spectrum_in': 'spectrum',   # 1D frequency array
            'image_in': 'image',         # 2D pattern (will FFT)
            'reset': 'signal'
        }
        
        self.outputs = {
            'phi_score': 'signal',           # 0-1: How golden is this?
            'phi_deviation': 'signal',       # Average deviation from φ-powers
            'peak_count': 'signal',          # Number of peaks found
            'ratio_histogram': 'image',      # Visual: distribution of ratios
            'phi_map': 'image',              # Visual: φ-structure in 2D
            'analysis_view': 'image',        # Combined visualization
            'peak_ratios': 'spectrum',       # Raw ratios between peaks
            'best_phi_power': 'signal'       # Which φ^n is dominant
        }
        
        # Signal accumulator
        self.signal_buffer = deque(maxlen=1024)
        
        # Analysis results
        self.phi_score = 0.0
        self.phi_deviation = 1.0
        self.peak_count = 0
        self.peak_ratios = np.array([])
        self.best_phi_power = 0.0
        
        # Visualization buffers
        self.ratio_histogram = np.zeros((128, 256, 3), dtype=np.uint8)
        self.phi_map = np.zeros((128, 128), dtype=np.float32)
        self.analysis_view = np.zeros((256, 512, 3), dtype=np.uint8)
        
        # History for tracking
        self.phi_score_history = deque(maxlen=100)
        
        self.step_count = 0

    def find_phi_deviation(self, ratio):
        """Find minimum deviation from any φ-power"""
        deviations = np.abs(PHI_POWERS - ratio) / PHI_POWERS
        return np.min(deviations), np.argmin(deviations)

    def analyze_spectrum(self, spectrum):
        """
        Core analysis: Find peaks in spectrum, compute ratios, measure φ-content
        """
        if spectrum is None or len(spectrum) < 4:
            return
            
        spectrum = np.abs(np.array(spectrum, dtype=np.float64))
        
        # Normalize
        spec_max = np.max(spectrum)
        if spec_max < 1e-9:
            return
        spectrum = spectrum / spec_max
        
        # Find peaks (at least 10% of max, separated by at least 2 bins)
        peaks, properties = find_peaks(spectrum, height=0.1, distance=2, prominence=0.05)
        
        self.peak_count = len(peaks)
        
        if len(peaks) < 2:
            # Not enough peaks to compute ratios
            self.phi_score = 0.0
            self.phi_deviation = 1.0
            return
            
        # Sort peaks by magnitude (strongest first)
        peak_heights = spectrum[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        
        # Compute ratios between all pairs of peak frequencies
        ratios = []
        deviations = []
        phi_power_votes = np.zeros(len(PHI_POWERS))
        
        for i in range(len(sorted_peaks)):
            for j in range(i + 1, len(sorted_peaks)):
                f1 = sorted_peaks[i] + 1  # Avoid division by zero
                f2 = sorted_peaks[j] + 1
                
                # Compute ratio (always > 1)
                ratio = max(f1, f2) / min(f1, f2)
                
                if ratio > 0.1 and ratio < 20:  # Reasonable range
                    ratios.append(ratio)
                    dev, phi_idx = self.find_phi_deviation(ratio)
                    deviations.append(dev)
                    phi_power_votes[phi_idx] += 1
        
        if len(ratios) == 0:
            self.phi_score = 0.0
            self.phi_deviation = 1.0
            return
            
        self.peak_ratios = np.array(ratios)
        
        # φ-score: fraction of ratios within 10% of a φ-power
        close_to_phi = np.array(deviations) < 0.1
        self.phi_score = np.mean(close_to_phi)
        
        # Average deviation
        self.phi_deviation = np.mean(deviations)
        
        # Best φ-power
        self.best_phi_power = np.argmax(phi_power_votes) - 4  # Center around φ^0
        
        # Update history
        self.phi_score_history.append(self.phi_score)

    def analyze_image(self, img):
        """
        Analyze 2D pattern for φ-structure via radial FFT
        """
        if img is None:
            return
            
        # Convert to grayscale float
        if img.ndim == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        
        # Normalize
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            return
            
        # 2D FFT
        f = fft2(img)
        fshift = fftshift(f)
        magnitude = np.abs(fshift)
        
        # Compute radial profile (average magnitude at each radius)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        max_radius = min(cy, cx)
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
        
        for radius in range(max_radius):
            mask = r == radius
            if np.any(mask):
                radial_profile[radius] = np.mean(magnitude[mask])
                counts[radius] = np.sum(mask)
        
        # Analyze radial profile as 1D spectrum
        self.analyze_spectrum(radial_profile)
        
        # Create φ-map: highlight regions where local ratios match φ
        self.create_phi_map(magnitude)

    def create_phi_map(self, magnitude):
        """
        Create a visualization showing where φ-structure exists in the 2D FFT
        """
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Resize for display
        target_size = 128
        mag_resized = cv2.resize(magnitude, (target_size, target_size))
        
        # For each point, compute local φ-ness
        phi_map = np.zeros((target_size, target_size), dtype=np.float32)
        
        # Compute radial distance for each pixel
        y, x = np.ogrid[:target_size, :target_size]
        center = target_size // 2
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Check if radius ratios match φ-powers
        for phi_power in PHI_POWERS[4:10]:  # φ^0 to φ^5
            target_ratio = phi_power
            # Find rings at φ-ratio distances
            for base_r in range(5, center, 3):
                target_r = base_r * target_ratio
                if target_r < center:
                    ring_mask = np.abs(r - target_r) < 2
                    phi_map[ring_mask] += mag_resized[ring_mask] * 0.2
        
        # Normalize
        if phi_map.max() > 0:
            phi_map /= phi_map.max()
            
        self.phi_map = phi_map

    def create_ratio_histogram(self):
        """
        Create histogram of peak ratios with φ-powers marked
        """
        h, w = 128, 256
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.peak_ratios) == 0:
            self.ratio_histogram = hist_img
            return
            
        # Bin ratios from 0.3 to 7 (covers φ^-2 to φ^4)
        bins = np.linspace(0.3, 7, w)
        
        # Create histogram
        counts, _ = np.histogram(self.peak_ratios, bins=bins)
        if counts.max() > 0:
            counts = counts / counts.max()
        
        # Draw histogram bars
        for i in range(len(counts)):
            bar_height = int(counts[i] * (h - 20))
            if bar_height > 0:
                cv2.rectangle(hist_img, 
                             (i, h - bar_height), 
                             (i + 1, h),
                             (100, 200, 100), -1)
        
        # Mark φ-powers with vertical lines
        for i, phi_power in enumerate(PHI_POWERS):
            if 0.3 < phi_power < 7:
                x = int((phi_power - 0.3) / (7 - 0.3) * w)
                color = (0, 215, 255) if abs(i - 4) <= 2 else (100, 100, 255)  # Gold for central powers
                cv2.line(hist_img, (x, 0), (x, h), color, 1)
                
        # Add φ label at φ^1
        x_phi = int((PHI - 0.3) / (7 - 0.3) * w)
        cv2.putText(hist_img, "phi", (x_phi - 10, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1)
        
        self.ratio_histogram = hist_img

    def create_analysis_view(self):
        """
        Combined visualization showing all φ-analysis
        """
        view = np.zeros((256, 512, 3), dtype=np.uint8)
        
        # Top-left: Ratio histogram
        view[0:128, 0:256] = self.ratio_histogram
        
        # Top-right: φ-map
        phi_colored = cv2.applyColorMap((self.phi_map * 255).astype(np.uint8), 
                                        cv2.COLORMAP_HOT)
        view[0:128, 256:384] = cv2.resize(phi_colored, (128, 128))
        
        # Top-right corner: φ-score display
        score_display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Draw φ-score as arc
        center = (64, 80)
        radius = 50
        angle = int(self.phi_score * 180)  # 0-180 degrees
        cv2.ellipse(score_display, center, (radius, radius), 180, 0, 180, (50, 50, 50), 2)
        if angle > 0:
            color = (0, 255, 0) if self.phi_score > 0.5 else (0, 165, 255)
            cv2.ellipse(score_display, center, (radius, radius), 180, 0, angle, color, 4)
        
        # φ-score text
        cv2.putText(score_display, f"phi-SCORE", (15, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(score_display, f"{self.phi_score:.3f}", (30, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
        cv2.putText(score_display, f"peaks:{self.peak_count}", (30, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        view[0:128, 384:512] = score_display
        
        # Bottom: φ-score history
        history_view = np.zeros((128, 512, 3), dtype=np.uint8)
        
        if len(self.phi_score_history) > 1:
            history = np.array(self.phi_score_history)
            for i in range(1, len(history)):
                x1 = int((i - 1) * 512 / 100)
                x2 = int(i * 512 / 100)
                y1 = int((1 - history[i-1]) * 100) + 14
                y2 = int((1 - history[i]) * 100) + 14
                color = (0, 255, 0) if history[i] > 0.5 else (0, 165, 255)
                cv2.line(history_view, (x1, y1), (x2, y2), color, 2)
        
        # Reference lines
        cv2.line(history_view, (0, 64), (512, 64), (50, 50, 50), 1)  # 0.5 line
        cv2.putText(history_view, "0.5", (5, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(history_view, "phi-Score History", (200, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        view[128:256, :] = history_view
        
        # Golden border if high φ-score
        if self.phi_score > 0.6:
            cv2.rectangle(view, (0, 0), (511, 255), (0, 215, 255), 3)
            
        self.analysis_view = view

    def step(self):
        self.step_count += 1
        
        # Check reset
        reset = self.get_blended_input('reset', 'sum')
        if reset is not None and reset > 0.5:
            self.signal_buffer.clear()
            self.phi_score_history.clear()
            self.phi_score = 0.0
            self.phi_deviation = 1.0
            self.peak_count = 0
            return
        
        # Priority: Image > Spectrum > Signal
        img = self.get_blended_input('image_in', 'mean')
        spectrum = self.get_blended_input('spectrum_in', 'sum')
        signal = self.get_blended_input('signal_in', 'sum')
        
        if img is not None:
            self.analyze_image(img)
        elif spectrum is not None:
            if hasattr(spectrum, '__len__'):
                self.analyze_spectrum(spectrum)
        elif signal is not None:
            # Accumulate signal samples
            self.signal_buffer.append(signal)
            
            # Analyze accumulated buffer via FFT
            if len(self.signal_buffer) >= 64:
                buffer_array = np.array(self.signal_buffer)
                spectrum = np.abs(fft(buffer_array))[:len(buffer_array)//2]
                self.analyze_spectrum(spectrum)
        
        # Update visualizations
        self.create_ratio_histogram()
        self.create_analysis_view()

    def get_output(self, port_name):
        if port_name == 'phi_score':
            return float(self.phi_score)
        elif port_name == 'phi_deviation':
            return float(self.phi_deviation)
        elif port_name == 'peak_count':
            return float(self.peak_count)
        elif port_name == 'ratio_histogram':
            return self.ratio_histogram
        elif port_name == 'phi_map':
            return (self.phi_map * 255).astype(np.uint8)
        elif port_name == 'analysis_view':
            return self.analysis_view
        elif port_name == 'peak_ratios':
            return self.peak_ratios if len(self.peak_ratios) > 0 else np.array([1.0])
        elif port_name == 'best_phi_power':
            return float(self.best_phi_power)
        return None

    def get_display_image(self):
        # Return the combined analysis view
        img = np.ascontiguousarray(self.analysis_view)
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Signal Buffer Size", "signal_buffer_size", 1024, None),
        ]