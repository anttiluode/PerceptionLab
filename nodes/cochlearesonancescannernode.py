"""
COCHLEAR RESONANCE SCANNER
==========================
Analyzes the row-wise FFT (cochlear decomposition) of holographic patterns
to find standing wave structures, harmonic relationships, and bilateral symmetry.

THEORY:
The 1D FFT of each row reveals the FREQUENCY CONTENT at each spatial position.
Standing waves create specific patterns:
- NODES (nulls) where waves cancel
- ANTINODES (peaks) where waves reinforce
- HARMONIC SERIES if the pattern is resonant

When the brain's hologram "locks" onto a resonant mode, we should see:
- Clean harmonic ratios (1:2:3:4 or octaves)
- High bilateral symmetry
- Sharp peaks in the frequency profile

This node scans for these signatures and signals when resonance is found.

Author: Built for Antti's consciousness crystallography research
"""

import os
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
            self.input_data = {}
        def pre_step(self): 
            self.input_data = {name: [] for name in self.inputs}
        def get_blended_input(self, name, mode): 
            return None

try:
    from scipy.fft import rfft, fftshift
    from scipy.signal import find_peaks, savgol_filter
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CochlearResonanceScannerNode(BaseNode):
    """
    Scans holographic patterns for standing wave resonance and harmonic structure.
    """
    
    NODE_NAME = "Cochlear Resonance Scanner"
    NODE_TITLE = "Cochlear Resonance"
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(220, 120, 40) if QtGui else None  # Orange like FFT Cochlea
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'image_in': 'image',                  # Hologram image
            'complex_field': 'complex_spectrum',  # Or complex field directly
            'threshold': 'signal',                # Resonance detection threshold
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Visualizations
            'cochlea_view': 'image',        # Row-wise FFT visualization
            'resonance_map': 'image',       # Where resonance is strongest
            'symmetry_view': 'image',       # Bilateral symmetry map
            'harmonic_view': 'image',       # Harmonic structure visualization
            
            # Analysis
            'frequency_profile': 'spectrum',  # Average frequency content
            'harmonic_ratios': 'spectrum',    # Detected harmonic ratios
            
            # Signals
            'resonance_score': 'signal',     # Overall resonance strength
            'symmetry_score': 'signal',      # Bilateral symmetry
            'harmonic_score': 'signal',      # How harmonic the structure is
            'dominant_freq': 'signal',       # Dominant frequency bin
            'num_harmonics': 'signal',       # Number of detected harmonics
            'resonance_found': 'signal',     # 1.0 when strong resonance detected
        }
        
        # === CONFIG ===
        self.resonance_threshold = 0.4
        self.min_peaks = 3  # Minimum peaks for harmonic detection
        self.harmonic_tolerance = 0.15  # 15% tolerance for harmonic matching
        
        # === STATE ===
        self.cochlea_image = None
        self.frequency_profile = None
        self.harmonic_ratios = None
        
        self.resonance_score = 0.0
        self.symmetry_score = 0.0
        self.harmonic_score = 0.0
        self.dominant_freq = 0.0
        self.num_harmonics = 0
        self.resonance_found = False
        
        # Peak tracking
        self.detected_peaks = []
        self.peak_ratios = []
        
        # Display
        self.display_image = None
        self._init_display()
    
    def get_config_options(self):
        return [
            ("Resonance Threshold", "resonance_threshold", self.resonance_threshold, None),
            ("Min Peaks", "min_peaks", self.min_peaks, None),
            ("Harmonic Tolerance", "harmonic_tolerance", self.harmonic_tolerance, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _init_display(self):
        w, h = 400, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "COCHLEAR RESONANCE", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 120, 40), 2)
        cv2.putText(img, "Connect image_in or complex_field", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        self.display_image = img
    
    # === ANALYSIS METHODS ===
    
    def _compute_cochlea(self, image):
        """Compute row-wise FFT (cochlear decomposition)."""
        if image is None or image.size == 0:
            return None
        
        # Ensure grayscale float
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        if gray.max() > 1.0:
            gray = gray / 255.0
        
        # Row-wise FFT
        spectrum = np.abs(rfft(gray, axis=1))
        
        # Log scale for visualization
        log_spec = np.log1p(spectrum)
        
        # Normalize
        if log_spec.max() > log_spec.min():
            log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min())
        
        return spectrum, log_spec
    
    def _analyze_frequency_profile(self, spectrum):
        """Analyze the average frequency content."""
        if spectrum is None:
            return None, [], 0.0
        
        # Average across rows
        profile = np.mean(spectrum, axis=0)
        
        # Smooth
        if len(profile) > 11:
            smoothed = savgol_filter(profile, 11, 3)
        else:
            smoothed = profile
        
        # Find peaks (resonant frequencies)
        peaks, properties = find_peaks(smoothed, 
                                       height=np.mean(smoothed),
                                       distance=5,
                                       prominence=np.std(smoothed) * 0.5)
        
        # Dominant frequency
        if len(peaks) > 0:
            heights = smoothed[peaks]
            dominant_idx = peaks[np.argmax(heights)]
        else:
            dominant_idx = np.argmax(smoothed)
        
        return profile, peaks, dominant_idx
    
    def _analyze_harmonics(self, peaks, profile):
        """Check if peaks form a harmonic series."""
        if len(peaks) < self.min_peaks:
            return 0.0, [], []
        
        # Sort peaks by position
        peaks = np.sort(peaks)
        
        # Check for harmonic relationships
        # Harmonics: f, 2f, 3f, 4f...
        # Or octaves: f, 2f, 4f, 8f...
        
        fundamental_candidates = []
        
        for i, p1 in enumerate(peaks):
            if p1 < 5:  # Skip DC-adjacent
                continue
            
            # Check if other peaks are harmonics of this one
            harmonics_found = [p1]
            
            for harmonic in range(2, 10):
                expected = p1 * harmonic
                tolerance = expected * self.harmonic_tolerance
                
                # Find closest peak to expected harmonic
                distances = np.abs(peaks - expected)
                if np.min(distances) < tolerance:
                    closest = peaks[np.argmin(distances)]
                    if closest not in harmonics_found:
                        harmonics_found.append(closest)
            
            if len(harmonics_found) >= self.min_peaks:
                fundamental_candidates.append((p1, harmonics_found))
        
        if not fundamental_candidates:
            return 0.0, [], []
        
        # Find best fundamental (most harmonics)
        best = max(fundamental_candidates, key=lambda x: len(x[1]))
        fundamental, harmonics = best
        
        # Calculate harmonic score
        # Based on how well peaks match expected harmonics
        ratios = [h / fundamental for h in harmonics]
        expected_ratios = list(range(1, len(ratios) + 1))
        
        errors = [abs(r - e) / e for r, e in zip(sorted(ratios), expected_ratios)]
        harmonic_score = 1.0 - np.mean(errors)
        harmonic_score = np.clip(harmonic_score, 0, 1)
        
        return harmonic_score, harmonics, ratios
    
    def _analyze_symmetry(self, cochlea):
        """Analyze bilateral symmetry of the cochlear pattern."""
        if cochlea is None:
            return 0.0, None
        
        h, w = cochlea.shape
        
        # Split left and right
        left = cochlea[:, :w//2]
        right = np.fliplr(cochlea[:, w//2:])
        
        # Match dimensions
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        # Compute correlation
        if np.std(left) > 0 and np.std(right) > 0:
            corr = np.corrcoef(left.flatten(), right.flatten())[0, 1]
            symmetry_score = (corr + 1) / 2  # Map -1,1 to 0,1
        else:
            symmetry_score = 0.0
        
        # Create symmetry map
        diff = np.abs(left - right)
        symmetry_map = 1.0 - (diff / (diff.max() + 1e-6))
        
        return symmetry_score, symmetry_map
    
    def _compute_resonance_map(self, spectrum, peaks):
        """Create map showing where resonance is strongest."""
        if spectrum is None:
            return None
        
        h, w = spectrum.shape
        
        # For each row, check how strongly it contains the resonant frequencies
        resonance_map = np.zeros((h, w), dtype=np.float32)
        
        for peak in peaks:
            if peak < w:
                # Weight by peak strength
                col = spectrum[:, peak]
                resonance_map[:, peak] = col / (col.max() + 1e-6)
        
        # Smooth
        resonance_map = gaussian_filter1d(resonance_map, sigma=2, axis=1)
        
        return resonance_map
    
    # === MAIN STEP ===
    
    def step(self):
        """Main processing step."""
        
        # Get inputs
        image = self.get_blended_input('image_in', 'mean')
        field = self.get_blended_input('complex_field', 'mean')
        threshold = self.get_blended_input('threshold', 'sum')
        
        if threshold is not None:
            self.resonance_threshold = float(np.clip(threshold, 0.1, 0.9))
        
        # Get image from either input
        if field is not None and np.iscomplexobj(field):
            image = np.abs(field).astype(np.float32)
            if image.max() > 0:
                image = image / image.max()
        
        if image is None:
            return
        
        # === COMPUTE COCHLEA ===
        result = self._compute_cochlea(image)
        if result is None:
            return
        
        spectrum, log_spectrum = result
        self.cochlea_image = (log_spectrum * 255).astype(np.uint8)
        
        # === FREQUENCY ANALYSIS ===
        profile, peaks, dominant = self._analyze_frequency_profile(spectrum)
        self.frequency_profile = profile
        self.detected_peaks = peaks
        self.dominant_freq = float(dominant)
        
        # === HARMONIC ANALYSIS ===
        harm_score, harmonics, ratios = self._analyze_harmonics(peaks, profile)
        self.harmonic_score = harm_score
        self.harmonic_ratios = np.array(ratios) if ratios else np.array([])
        self.num_harmonics = len(harmonics)
        self.peak_ratios = ratios
        
        # === SYMMETRY ANALYSIS ===
        sym_score, sym_map = self._analyze_symmetry(log_spectrum)
        self.symmetry_score = sym_score
        
        # === RESONANCE MAP ===
        res_map = self._compute_resonance_map(spectrum, peaks)
        
        # === OVERALL RESONANCE SCORE ===
        # Combine harmonic, symmetry, and peak strength
        peak_strength = len(peaks) / 20.0  # Normalize by expected max peaks
        peak_strength = np.clip(peak_strength, 0, 1)
        
        self.resonance_score = (
            self.harmonic_score * 0.4 +
            self.symmetry_score * 0.3 +
            peak_strength * 0.3
        )
        
        self.resonance_found = self.resonance_score > self.resonance_threshold
        
        if self.resonance_found:
            print(f"[CochlearResonance] RESONANCE DETECTED! Score={self.resonance_score:.3f}")
            print(f"  Harmonics: {self.num_harmonics}, Symmetry: {self.symmetry_score:.3f}")
            print(f"  Peak ratios: {self.peak_ratios[:5]}")
        
        # === UPDATE DISPLAY ===
        self._update_display(log_spectrum, profile, peaks, sym_map, res_map)
    
    def _update_display(self, cochlea, profile, peaks, sym_map, res_map):
        """Create display visualization."""
        
        # Layout
        panel_w, panel_h = 180, 120
        margin = 5
        info_w = 180
        
        total_w = panel_w * 2 + info_w + margin * 4
        total_h = panel_h * 2 + margin * 3 + 40
        
        display = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        display[:] = 20
        
        # === COCHLEA VIEW (top-left) ===
        if cochlea is not None:
            coch_color = cv2.applyColorMap((cochlea * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            coch_resized = cv2.resize(coch_color, (panel_w, panel_h))
            display[margin+20:margin+20+panel_h, margin:margin+panel_w] = coch_resized
        cv2.putText(display, "Cochlea FFT", (margin, margin+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # === FREQUENCY PROFILE (top-right) ===
        if profile is not None:
            prof_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            
            # Normalize profile
            p_norm = profile / (profile.max() + 1e-6)
            
            # Draw profile
            for i in range(min(len(p_norm), panel_w)):
                h_val = int(p_norm[i] * (panel_h - 10))
                cv2.line(prof_img, (i, panel_h), (i, panel_h - h_val), (100, 100, 255), 1)
            
            # Mark peaks
            for peak in peaks:
                if peak < panel_w:
                    cv2.line(prof_img, (peak, 0), (peak, panel_h), (0, 255, 0), 1)
            
            display[margin+20:margin+20+panel_h, margin*2+panel_w:margin*2+panel_w*2] = prof_img
        cv2.putText(display, "Frequency Profile", (margin*2+panel_w, margin+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # === SYMMETRY MAP (bottom-left) ===
        if sym_map is not None:
            sym_color = cv2.applyColorMap((sym_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            sym_resized = cv2.resize(sym_color, (panel_w, panel_h))
            display[margin*2+panel_h+30:margin*2+panel_h*2+30, margin:margin+panel_w] = sym_resized
        cv2.putText(display, "Symmetry Map", (margin, margin*2+panel_h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # === RESONANCE MAP (bottom-right) ===
        if res_map is not None:
            res_norm = res_map / (res_map.max() + 1e-6)
            res_color = cv2.applyColorMap((res_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            res_resized = cv2.resize(res_color, (panel_w, panel_h))
            display[margin*2+panel_h+30:margin*2+panel_h*2+30, margin*2+panel_w:margin*2+panel_w*2] = res_resized
        cv2.putText(display, "Resonance Map", (margin*2+panel_w, margin*2+panel_h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # === INFO PANEL (right side) ===
        info_x = margin * 3 + panel_w * 2
        info_y = margin
        
        cv2.putText(display, "COCHLEAR", (info_x, info_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 120, 40), 2)
        cv2.putText(display, "RESONANCE", (info_x, info_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 120, 40), 2)
        
        # Status
        if self.resonance_found:
            status = "RESONANCE!"
            status_color = (0, 255, 0)
        else:
            status = "Scanning..."
            status_color = (150, 150, 150)
        
        cv2.putText(display, status, (info_x, info_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Metrics
        cv2.putText(display, f"Score: {self.resonance_score:.3f}", (info_x, info_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(display, f"Thresh: {self.resonance_threshold:.2f}", (info_x, info_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        cv2.putText(display, f"Harmonics: {self.num_harmonics}", (info_x, info_y + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(display, f"H-Score: {self.harmonic_score:.3f}", (info_x, info_y + 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        cv2.putText(display, f"Symmetry: {self.symmetry_score:.3f}", (info_x, info_y + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        
        cv2.putText(display, f"Peaks: {len(self.detected_peaks)}", (info_x, info_y + 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)
        cv2.putText(display, f"Dom.Freq: {self.dominant_freq:.0f}", (info_x, info_y + 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)
        
        # Resonance bar
        bar_x = info_x
        bar_y = info_y + 270
        bar_w = 150
        bar_h = 15
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(self.resonance_score * bar_w)
        color = (0, 255, 0) if self.resonance_found else (100, 100, 255)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        thresh_x = int(self.resonance_threshold * bar_w)
        cv2.line(display, (bar_x + thresh_x, bar_y), (bar_x + thresh_x, bar_y + bar_h), (0, 0, 255), 2)
        
        self.display_image = display
    
    # === OUTPUTS ===
    
    def get_output(self, port_name):
        if port_name == 'cochlea_view':
            if self.cochlea_image is not None:
                return cv2.applyColorMap(self.cochlea_image, cv2.COLORMAP_INFERNO)
            return None
        elif port_name == 'resonance_map':
            return self.display_image  # Full display for now
        elif port_name == 'symmetry_view':
            return self.display_image
        elif port_name == 'harmonic_view':
            return self.display_image
        elif port_name == 'frequency_profile':
            return self.frequency_profile
        elif port_name == 'harmonic_ratios':
            return self.harmonic_ratios
        elif port_name == 'resonance_score':
            return self.resonance_score
        elif port_name == 'symmetry_score':
            return self.symmetry_score
        elif port_name == 'harmonic_score':
            return self.harmonic_score
        elif port_name == 'dominant_freq':
            return self.dominant_freq
        elif port_name == 'num_harmonics':
            return float(self.num_harmonics)
        elif port_name == 'resonance_found':
            return 1.0 if self.resonance_found else 0.0
        return None
    
    def get_display_image(self):
        return self.display_image
    
    def close(self):
        pass