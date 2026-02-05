"""
FFT Ring Analyzer Node (FIXED VERSION)
=======================================
Detects and analyzes concentric ring structures in FFT magnitude images.

FIXES FROM ORIGINAL:
1. Handles UNSHIFTED FFT input (complex_spectrum from HolographicFFTNode)
2. Searches to the CORNERS (max_radius = diagonal), not just inscribed circle
3. Auto-detects if data is shifted vs unshifted
4. Lower default peak prominence for subtle rings
5. Better peak detection for edge-located rings

The double-ring structure represents two dominant spatial scales
in the phase field - this node quantifies that structure.

Author: Fixed for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    try:
        from PyQt6 import QtGui
    except ImportError:
        class MockQtGui:
            @staticmethod
            def QColor(*args): return None
            class QImage:
                Format_RGB888 = 0
                def __init__(self, *args): pass
                def copy(self): return self
        QtGui = MockQtGui()
    
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name, mode): 
            data = self.input_data.get(name, [None])
            return data[0] if data else None


# Known significant ratios
SIGNIFICANT_RATIOS = {
    1.618: "φ (Golden)",
    2.0: "Octave (2:1)",
    1.5: "Fifth (3:2)",
    1.333: "Fourth (4:3)",
    2.618: "φ² (Golden²)",
    1.414: "√2",
    1.732: "√3",
    2.236: "√5",
    3.0: "Triple (3:1)",
    2.5: "Fifth (5:2)",
}


class RingAnalyzerNode(BaseNode):
    """
    Analyzes concentric ring structures in FFT images.
    Detects dominant spatial frequencies and their relationships.
    
    FIXED: Now properly handles unshifted FFT and searches corners.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Ring Analyzer"
    NODE_COLOR = QtGui.QColor(50, 150, 255)  # Blue - frequency analysis
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'image_in': 'image',              # Image to analyze (will FFT if needed)
            'fft_magnitude': 'image',         # Pre-computed FFT magnitude
            'complex_spectrum': 'complex_spectrum',  # Complex FFT input
            'do_fft': 'signal',               # 1 = FFT the input, 0 = assume already FFT
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Visualizations
            'radial_profile_image': 'image',  # Graph of power vs radius
            'ring_overlay': 'image',          # FFT with detected rings marked
            'polar_view': 'image',            # Polar transform of FFT
            
            # Ring data
            'ring_1_radius': 'signal',        # Radius of strongest ring
            'ring_2_radius': 'signal',        # Radius of second ring
            'ring_3_radius': 'signal',        # Radius of third ring (if exists)
            'ring_ratio_1_2': 'signal',       # Ratio of ring 1 to ring 2
            'ring_ratio_2_3': 'signal',       # Ratio of ring 2 to ring 3
            
            # Analysis
            'num_rings': 'signal',            # Number of detected rings
            'ratio_type': 'signal',           # Encoded ratio classification
            'ring_sharpness': 'signal',       # How peaked vs diffuse
            'isotropy': 'signal',             # How circular vs elliptical
            
            # Raw data
            'radial_profile': 'signal',       # The actual profile array (as signal)
            'peak_radii': 'signal',           # All peak radii
        }
        
        # === PARAMETERS ===
        self.resolution = 128
        self.do_fft = True
        self.smoothing = 1.5          # CHANGED: Less smoothing to preserve edge peaks
        self.peak_prominence = 0.02   # CHANGED: Much lower - detect subtle rings
        self.peak_distance = 3        # CHANGED: Closer peaks allowed
        self.search_corners = True    # NEW: Search to corners, not just inscribed circle
        self.auto_shift = True        # NEW: Auto-detect and shift unshifted FFT
        
        # === STATE ===
        self._outputs = {}
        self._fft_mag = None
        self._fft_mag_shifted = None  # NEW: Always store shifted version for display
        self._radial_profile = None
        self._peaks = []
        self._ratio_name = "Unknown"
        
    def _is_shifted(self, fft_mag):
        """
        Detect if FFT is shifted (DC at center) or unshifted (DC at corners).
        Unshifted FFT has maximum at corners; shifted has maximum near center.
        """
        h, w = fft_mag.shape
        center_val = fft_mag[h//2, w//2]
        corner_val = max(
            fft_mag[0, 0],
            fft_mag[0, w-1],
            fft_mag[h-1, 0],
            fft_mag[h-1, w-1]
        )
        # If corners are much brighter than center, data is unshifted
        return center_val > corner_val * 0.5
    
    def _compute_fft_magnitude(self, img):
        """Compute FFT magnitude from image."""
        if img is None:
            return None, None
            
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        
        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        
        # Resize
        if img.shape[0] != self.resolution:
            img = cv2.resize(img, (self.resolution, self.resolution))
        
        # FFT - keep both unshifted and shifted
        fft_unshifted = fft2(img)
        fft_shifted = fftshift(fft_unshifted)
        
        mag_unshifted = np.abs(fft_unshifted)
        mag_shifted = np.abs(fft_shifted)
        
        # Log scale for better visualization
        mag_unshifted = np.log1p(mag_unshifted)
        mag_shifted = np.log1p(mag_shifted)
        
        return mag_shifted, mag_unshifted  # Return shifted for analysis
    
    def _compute_radial_profile(self, fft_mag):
        """
        Compute average power as function of radius from center.
        FIXED: Now searches to corners (diagonal), not just inscribed circle.
        """
        h, w = fft_mag.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radius map
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # FIXED: Maximum radius reaches corners
        if self.search_corners:
            max_radius = int(np.sqrt(center_x**2 + center_y**2))
        else:
            max_radius = min(center_x, center_y)
        
        # Bin by radius
        n_bins = max_radius
        radial_profile = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        for r in range(n_bins):
            mask = (R >= r) & (R < r + 1)
            if mask.sum() > 0:
                radial_profile[r] = fft_mag[mask].mean()
                counts[r] = mask.sum()
        
        # Smooth
        if self.smoothing > 0:
            radial_profile = gaussian_filter1d(radial_profile, self.smoothing)
        
        return radial_profile, max_radius
    
    def _detect_rings(self, radial_profile):
        """
        Find peaks in radial profile = ring radii.
        FIXED: Better handling of edge peaks and subtle features.
        """
        # Normalize
        profile = radial_profile.copy()
        pmax = profile.max()
        pmin = profile.min()
        if pmax > pmin:
            profile = (profile - pmin) / (pmax - pmin)
        
        # Find peaks with lower threshold
        peaks, properties = find_peaks(
            profile,
            prominence=self.peak_prominence,
            distance=self.peak_distance,
            height=0.05  # Minimum height threshold
        )
        
        # Sort by prominence (strength) if available
        if len(peaks) > 0 and 'prominences' in properties:
            order = np.argsort(properties['prominences'])[::-1]
            peaks = peaks[order]
        elif len(peaks) > 0:
            # Sort by height
            heights = profile[peaks]
            order = np.argsort(heights)[::-1]
            peaks = peaks[order]
        
        return peaks, profile
    
    def _compute_isotropy(self, fft_mag):
        """
        Measure how circular vs elliptical the power distribution is.
        1.0 = perfect circle, <1.0 = elliptical/asymmetric
        """
        h, w = fft_mag.shape
        center_y, center_x = h // 2, w // 2
        
        # Sample at multiple angles
        n_angles = 36
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        
        # For each angle, measure power at fixed radius
        test_radius = min(center_x, center_y) // 3
        powers = []
        
        for angle in angles:
            x = int(center_x + test_radius * np.cos(angle))
            y = int(center_y + test_radius * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                powers.append(fft_mag[y, x])
        
        if len(powers) < 2:
            return 1.0
        
        powers = np.array(powers)
        # Isotropy = 1 - (std/mean)
        isotropy = 1.0 - (powers.std() / (powers.mean() + 1e-9))
        return max(0, min(1, isotropy))
    
    def _identify_ratio(self, ratio):
        """
        Check if ratio matches known significant values.
        """
        if ratio is None or ratio <= 0:
            return "N/A"
        
        # Check against known ratios
        best_match = None
        best_diff = 0.1  # Tolerance
        
        for known_ratio, name in SIGNIFICANT_RATIOS.items():
            diff = abs(ratio - known_ratio)
            if diff < best_diff:
                best_diff = diff
                best_match = name
        
        if best_match:
            return best_match
        else:
            return f"{ratio:.3f}"
    
    def _create_radial_profile_image(self, profile, peaks, max_radius):
        """
        Create visualization of radial profile with peaks marked.
        """
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (20, 20, 30)  # Dark background
        
        if profile is None or len(profile) == 0:
            return img
        
        # Normalize profile
        profile_norm = profile.copy()
        pmax = profile_norm.max()
        pmin = profile_norm.min()
        if pmax > pmin:
            profile_norm = (profile_norm - pmin) / (pmax - pmin)
        
        # Draw profile
        x_scale = w / len(profile)
        y_scale = h * 0.8
        y_offset = h * 0.9
        
        points = []
        for i, val in enumerate(profile_norm):
            x = int(i * x_scale)
            y = int(y_offset - val * y_scale)
            points.append((x, y))
        
        # Draw line
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (0, 255, 255), 1)
        
        # Mark peaks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, peak in enumerate(peaks[:5]):
            if peak < len(profile_norm):
                x = int(peak * x_scale)
                y = int(y_offset - profile_norm[peak] * y_scale)
                color = colors[i % len(colors)]
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.putText(img, f"R{i+1}={peak}", (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Labels
        cv2.putText(img, "Radial Profile", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Max R={max_radius}", (w - 80, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(img, "Radius ->", (w - 60, h - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        return img
    
    def _create_ring_overlay(self, fft_mag, peaks):
        """
        Create FFT image with detected rings overlaid.
        """
        # Normalize and convert to color
        img = fft_mag.copy()
        img = img - img.min()
        img = img / (img.max() + 1e-9)
        img = (img * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Draw detected rings
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, peak in enumerate(peaks[:5]):
            color = colors[i % len(colors)]
            cv2.circle(img, center, int(peak), color, 1)
        
        # Draw corner indicator if we're searching corners
        if self.search_corners:
            max_r = int(np.sqrt((w//2)**2 + (h//2)**2))
            cv2.circle(img, center, max_r, (128, 128, 128), 1, cv2.LINE_AA)
        
        return img
    
    def _create_polar_view(self, fft_mag):
        """
        Transform FFT to polar coordinates.
        Rings become horizontal lines.
        """
        h, w = fft_mag.shape
        center = (w // 2, h // 2)
        
        # Use full diagonal for polar transform
        if self.search_corners:
            max_radius = np.sqrt(center[0]**2 + center[1]**2)
        else:
            max_radius = min(center[0], center[1])
        
        # Polar transform
        polar = cv2.linearPolar(fft_mag.astype(np.float32), 
                                (float(center[0]), float(center[1])), 
                                float(max_radius), 
                                cv2.WARP_FILL_OUTLIERS)
        
        # Normalize and colorize
        polar = polar - polar.min()
        polar = polar / (polar.max() + 1e-9)
        polar = (polar * 255).astype(np.uint8)
        polar = cv2.applyColorMap(polar, cv2.COLORMAP_VIRIDIS)
        
        return polar
    
    def step(self):
        """Main processing step."""
        # Get inputs
        img_in = self.get_blended_input('image_in', 'first')
        fft_in = self.get_blended_input('fft_magnitude', 'first')
        complex_in = self.get_blended_input('complex_spectrum', 'first')
        do_fft_sig = self.get_blended_input('do_fft', 'first')
        
        if do_fft_sig is not None:
            self.do_fft = bool(do_fft_sig)
        
        # Determine FFT magnitude
        if complex_in is not None:
            # Complex input - need to check if shifted
            magnitude = np.abs(complex_in)
            magnitude = np.log1p(magnitude)
            
            # Auto-shift if needed
            if self.auto_shift and not self._is_shifted(magnitude):
                self._fft_mag = np.abs(fftshift(complex_in))
                self._fft_mag = np.log1p(self._fft_mag)
            else:
                self._fft_mag = magnitude
                
        elif fft_in is not None and not self.do_fft:
            # Assume input is already FFT magnitude
            if fft_in.ndim == 3:
                fft_in = np.mean(fft_in, axis=2)
            self._fft_mag = fft_in.astype(np.float32)
            if self._fft_mag.max() > 1:
                self._fft_mag = self._fft_mag / 255.0
                
            # Auto-shift check
            if self.auto_shift and not self._is_shifted(self._fft_mag):
                self._fft_mag = fftshift(self._fft_mag)
                
        elif img_in is not None:
            # Compute FFT of input image
            self._fft_mag, _ = self._compute_fft_magnitude(img_in)
        else:
            return
        
        if self._fft_mag is None:
            return
        
        # Ensure correct size
        if self._fft_mag.shape[0] != self.resolution:
            self._fft_mag = cv2.resize(self._fft_mag, (self.resolution, self.resolution))
        
        # === COMPUTE RADIAL PROFILE ===
        self._radial_profile, max_radius = self._compute_radial_profile(self._fft_mag)
        
        # === DETECT RINGS ===
        self._peaks, profile_norm = self._detect_rings(self._radial_profile)
        num_rings = len(self._peaks)
        self._outputs['num_rings'] = float(num_rings)
        
        # === RING RADII ===
        for i in range(3):
            key = f'ring_{i+1}_radius'
            if i < num_rings:
                self._outputs[key] = float(self._peaks[i])
            else:
                self._outputs[key] = 0.0
        
        # === RING RATIOS ===
        if num_rings >= 2:
            r1, r2 = self._peaks[0], self._peaks[1]
            ratio_1_2 = max(r1, r2) / (min(r1, r2) + 1e-9)
            self._outputs['ring_ratio_1_2'] = float(ratio_1_2)
            self._ratio_name = self._identify_ratio(ratio_1_2)
        else:
            self._outputs['ring_ratio_1_2'] = 0.0
            self._ratio_name = "N/A"
        
        if num_rings >= 3:
            r2, r3 = self._peaks[1], self._peaks[2]
            ratio_2_3 = max(r2, r3) / (min(r2, r3) + 1e-9)
            self._outputs['ring_ratio_2_3'] = float(ratio_2_3)
        else:
            self._outputs['ring_ratio_2_3'] = 0.0
        
        # === RING SHARPNESS ===
        if num_rings > 0:
            peak_values = [self._radial_profile[p] for p in self._peaks[:3] 
                          if p < len(self._radial_profile)]
            mean_profile = self._radial_profile.mean()
            if mean_profile > 0:
                sharpness = np.mean(peak_values) / mean_profile
            else:
                sharpness = 0
            self._outputs['ring_sharpness'] = float(min(sharpness, 10))
        else:
            self._outputs['ring_sharpness'] = 0.0
        
        # === ISOTROPY ===
        self._outputs['isotropy'] = float(self._compute_isotropy(self._fft_mag))
        
        # === CREATE VISUALIZATIONS ===
        self._outputs['radial_profile_image'] = self._create_radial_profile_image(
            self._radial_profile, self._peaks, max_radius)
        self._outputs['ring_overlay'] = self._create_ring_overlay(
            self._fft_mag, self._peaks)
        self._outputs['polar_view'] = self._create_polar_view(self._fft_mag)
        
        # Raw data outputs
        self._outputs['radial_profile'] = float(self._radial_profile.mean()) if self._radial_profile is not None else 0
        self._outputs['peak_radii'] = float(self._peaks[0]) if len(self._peaks) > 0 else 0
        
    def get_output(self, port_name):
        """Return requested output."""
        return self._outputs.get(port_name, None)
    
    def get_display_image(self):
        """Create visualization for node face."""
        # Stack: Ring overlay on top, radial profile below
        overlay = self._outputs.get('ring_overlay')
        profile = self._outputs.get('radial_profile_image')
        
        if overlay is None:
            overlay = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        if profile is None:
            profile = np.zeros((64, self.resolution, 3), dtype=np.uint8)
        
        # Resize overlay to match width
        overlay = cv2.resize(overlay, (self.resolution, self.resolution))
        profile = cv2.resize(profile, (self.resolution, 64))
        
        # Stack vertically
        display = np.vstack([overlay, profile])
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        num_rings = int(self._outputs.get('num_rings', 0))
        ratio = self._outputs.get('ring_ratio_1_2', 0)
        
        cv2.putText(display, f"Rings: {num_rings}", (5, 15), font, 0.4, (255, 255, 255), 1)
        if num_rings >= 2:
            cv2.putText(display, f"Ratio: {ratio:.3f} ({self._ratio_name})", 
                       (5, 30), font, 0.35, (0, 255, 255), 1)
        
        # Ring radii
        for i in range(min(3, num_rings)):
            r = self._outputs.get(f'ring_{i+1}_radius', 0)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            cv2.putText(display, f"R{i+1}={r:.0f}", (self.resolution - 50, 15 + i*12), 
                       font, 0.3, colors[i], 1)
        
        display = np.ascontiguousarray(display)
        h, w = display.shape[:2]
        
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg.copy()
    
    def get_config_options(self):
        """Configuration options."""
        return [
            ("Resolution", "resolution", self.resolution, 'int'),
            ("Do FFT on input", "do_fft", self.do_fft, 'bool'),
            ("Profile Smoothing", "smoothing", self.smoothing, 'float'),
            ("Peak Prominence", "peak_prominence", self.peak_prominence, 'float'),
            ("Min Peak Distance", "peak_distance", self.peak_distance, 'int'),
            ("Search Corners", "search_corners", self.search_corners, 'bool'),
            ("Auto-Shift FFT", "auto_shift", self.auto_shift, 'bool'),
        ]
    
    def set_config_options(self, options):
        """Apply configuration."""
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("FFT Ring Analyzer Node (FIXED)")
    print("=" * 60)
    print()
    print("FIXES in this version:")
    print("  1. Handles UNSHIFTED FFT (complex_spectrum from HolographicFFT)")
    print("  2. Searches to CORNERS (full diagonal), not inscribed circle")
    print("  3. Auto-detects shifted vs unshifted data")
    print("  4. Lower prominence threshold for subtle rings")
    print()
    print("OUTPUTS:")
    print("  ring_1_radius, ring_2_radius, ring_3_radius - Ring positions")
    print("  ring_ratio_1_2, ring_ratio_2_3 - Ratios between rings")
    print("  num_rings - Number of detected rings")
    print()
    print("SIGNIFICANT RATIOS:")
    for ratio, name in sorted(SIGNIFICANT_RATIOS.items()):
        print(f"  {ratio:.3f} = {name}")
    print()
    
    # Test with synthetic double-ring pattern at EDGES (like you see)
    print("Testing with edge-located double-ring pattern...")
    
    res = 128
    center = res // 2
    Y, X = np.ogrid[:res, :res]
    R = np.sqrt((X - center)**2 + (Y - center)**2)
    
    # Create double-ring pattern at EDGES (high frequency)
    max_r = np.sqrt(center**2 + center**2)  # Diagonal
    ring1_radius = max_r * 0.7  # 70% to edge
    ring2_radius = max_r * 0.85  # 85% to edge
    ring_width = 3
    
    pattern = np.exp(-((R - ring1_radius)**2) / (2 * ring_width**2))
    pattern += 0.7 * np.exp(-((R - ring2_radius)**2) / (2 * ring_width**2))
    pattern += np.random.randn(res, res) * 0.05  # Less noise
    
    node = RingAnalyzerNode()
    node.do_fft = False  # Input is already "FFT-like"
    node.search_corners = True
    
    # Process
    node._fft_mag = pattern
    node._fft_mag = cv2.resize(node._fft_mag, (node.resolution, node.resolution))
    node._radial_profile, max_radius = node._compute_radial_profile(node._fft_mag)
    node._peaks, _ = node._detect_rings(node._radial_profile)
    
    print(f"Max search radius: {max_radius} (searching to corners)")
    print(f"Detected {len(node._peaks)} rings")
    if len(node._peaks) >= 2:
        r1, r2 = node._peaks[0], node._peaks[1]
        ratio = max(r1, r2) / min(r1, r2)
        print(f"Ring 1 radius: {r1:.1f}")
        print(f"Ring 2 radius: {r2:.1f}")
        print(f"Ratio: {ratio:.3f}")
        print(f"Ratio type: {node._identify_ratio(ratio)}")
    elif len(node._peaks) == 1:
        print(f"Only 1 ring detected at radius: {node._peaks[0]:.1f}")
    else:
        print("No rings detected - check peak_prominence setting")