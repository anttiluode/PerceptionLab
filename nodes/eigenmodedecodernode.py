"""
FFT Eigenmode Decoder Node
==========================
Decodes individual eigenmodes (spatial frequency components) from FFT output.

The Problem:
When you FFT a hologram, you get a pattern of bright spots - each spot
represents a specific spatial frequency (eigenmode) of the original field.
This node lets you:
1. Isolate individual spots (modes)
2. Reconstruct what each mode looks like in real space
3. Track which modes carry the most energy
4. Selectively filter to see "what the brain is saying" vs "geometry"

Theory:
In drumhead physics, each bright spot in the FFT corresponds to a 
resonant mode of the membrane. By inverse-transforming individual spots,
we can see the standing wave pattern each mode contributes.

The high-SNR patterns you're seeing (SNR 1275!) suggest strong
coherent structure - not noise. This decoder helps you see what
that structure actually looks like.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

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


class EigenmodeDecoderNode(BaseNode):
    """
    Decodes spatial frequency eigenmodes from FFT of holographic fields.
    
    Takes FFT magnitude/phase and allows selective reconstruction of
    individual frequency components to see what patterns they encode.
    """
    
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Eigenmode Decoder"
    NODE_COLOR = QtGui.QColor(255, 100, 50)  # Orange - decoder/analysis
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'fft_spectrum': 'complex_spectrum',  # Complex FFT from HolographicFFTNode
            'fft_magnitude': 'image',            # Magnitude image (fallback)
            'mode_select': 'signal',             # Which mode to isolate (0-N)
            'mask_radius': 'signal',             # Radius of isolation mask
            'threshold': 'signal',               # Peak detection threshold
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Reconstructed modes
            'mode_0_image': 'image',        # Strongest mode reconstructed
            'mode_1_image': 'image',        # Second strongest
            'mode_2_image': 'image',        # Third strongest
            'mode_3_image': 'image',        # Fourth strongest
            'selected_mode': 'image',       # User-selected mode
            
            # Combined/filtered
            'top_4_combined': 'image',      # Sum of top 4 modes
            'filtered_recon': 'image',      # Reconstruction with threshold filter
            
            # Analysis
            'peak_map': 'image',            # Detected peaks visualization
            'mode_spectrum': 'image',       # Which frequencies are active
            
            # Signals
            'num_modes': 'signal',          # How many significant modes detected
            'mode_0_power': 'signal',       # Power in strongest mode
            'mode_1_power': 'signal',
            'mode_2_power': 'signal',
            'mode_3_power': 'signal',
            'total_power': 'signal',
            'mode_0_freq': 'signal',        # Spatial frequency of mode 0
        }
        
        # === PARAMETERS ===
        self.resolution = 128
        self.mask_radius = 5           # Pixels around peak to include
        self.threshold = 0.1           # Fraction of max for peak detection
        self.selected_mode_idx = 0     # Which mode user wants to see
        self.max_modes = 20            # Maximum modes to track
        
        # === STATE ===
        self._fft_complex = None
        self._peaks = []               # List of (y, x, power) tuples
        self._mode_images = {}         # Cached reconstructions
        self._peak_map = None
        self._mode_spectrum = None
        
        # Output caches
        self._outputs = {}
        
    def _detect_peaks(self, magnitude):
        """
        Find the bright spots in FFT magnitude.
        These are the eigenmodes / spatial frequency components.
        """
        if magnitude is None:
            return []
        
        # Normalize
        mag_norm = magnitude.astype(np.float32)
        if mag_norm.max() > 0:
            mag_norm = mag_norm / mag_norm.max()
        
        # Apply threshold
        thresh_val = self.threshold
        binary = (mag_norm > thresh_val).astype(np.uint8) * 255
        
        # Find contours / connected components
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        peaks = []
        for contour in contours:
            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Get power at this peak (sum within contour region)
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                power = np.sum(magnitude[mask > 0])
                
                peaks.append((cy, cx, power))
        
        # Sort by power (strongest first)
        peaks.sort(key=lambda p: p[2], reverse=True)
        
        # Limit to max_modes
        return peaks[:self.max_modes]
    
    def _isolate_mode(self, fft_complex, peak_y, peak_x, radius):
        """
        Create a masked version of FFT with only one mode.
        Then IFFT to see what that mode looks like in real space.
        """
        h, w = fft_complex.shape
        
        # Create circular mask around peak
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - peak_y)**2 + (x_grid - peak_x)**2)
        mask = (dist <= radius).astype(np.float32)
        
        # Apply Gaussian falloff for smoother edges
        sigma = radius / 2
        mask = np.exp(-dist**2 / (2 * sigma**2))
        mask[dist > radius * 1.5] = 0
        
        # Apply mask to FFT
        masked_fft = fft_complex * mask
        
        # Inverse FFT to get real-space pattern
        # Need to un-shift before IFFT
        unshifted = ifftshift(masked_fft)
        reconstructed = ifft2(unshifted)
        
        # Return magnitude (the pattern)
        return np.abs(reconstructed)
    
    def _compute_spatial_frequency(self, peak_y, peak_x, h, w):
        """
        Convert peak position to spatial frequency.
        Center of FFT = DC (0 frequency)
        Distance from center = spatial frequency
        """
        center_y, center_x = h // 2, w // 2
        dy = peak_y - center_y
        dx = peak_x - center_x
        freq = np.sqrt(dy**2 + dx**2)
        return freq
    
    def step(self):
        """Main processing step."""
        # Get inputs
        fft_in = self.get_blended_input('fft_spectrum', 'first')
        mag_in = self.get_blended_input('fft_magnitude', 'first')
        mode_sel = self.get_blended_input('mode_select', 'first')
        radius_in = self.get_blended_input('mask_radius', 'first')
        thresh_in = self.get_blended_input('threshold', 'first')
        
        # Update parameters
        if mode_sel is not None:
            self.selected_mode_idx = int(mode_sel)
        if radius_in is not None:
            self.mask_radius = max(1, int(radius_in))
        if thresh_in is not None:
            self.threshold = float(thresh_in)
        
        # Get complex FFT
        if fft_in is not None and np.iscomplexobj(fft_in):
            self._fft_complex = fft_in.copy()
            magnitude = np.abs(fft_in)
        elif mag_in is not None:
            # Fallback: use magnitude image, assume zero phase
            if mag_in.ndim == 3:
                mag_in = np.mean(mag_in, axis=2)
            magnitude = mag_in.astype(np.float32)
            if magnitude.max() > 1:
                magnitude = magnitude / 255.0
            # Create pseudo-complex (magnitude only)
            self._fft_complex = magnitude.astype(np.complex64)
        else:
            return  # No input
        
        # Ensure correct resolution
        h, w = magnitude.shape
        self.resolution = h
        
        # === DETECT PEAKS ===
        self._peaks = self._detect_peaks(magnitude)
        num_peaks = len(self._peaks)
        self._outputs['num_modes'] = float(num_peaks)
        
        # === CREATE PEAK MAP ===
        peak_map = np.zeros((h, w, 3), dtype=np.uint8)
        # Background: dim magnitude
        mag_vis = (magnitude / (magnitude.max() + 1e-9) * 100).astype(np.uint8)
        peak_map[:, :, 0] = mag_vis
        peak_map[:, :, 1] = mag_vis
        peak_map[:, :, 2] = mag_vis
        
        # Draw peaks with colors
        colors = [
            (255, 0, 0),    # Red - mode 0
            (0, 255, 0),    # Green - mode 1
            (0, 0, 255),    # Blue - mode 2
            (255, 255, 0),  # Yellow - mode 3
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, (py, px, power) in enumerate(self._peaks[:6]):
            color = colors[i % len(colors)]
            cv2.circle(peak_map, (px, py), self.mask_radius, color, 2)
            cv2.putText(peak_map, str(i), (px + 5, py - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        self._peak_map = peak_map
        self._outputs['peak_map'] = peak_map
        
        # === ISOLATE AND RECONSTRUCT MODES ===
        total_power = 0
        combined_top4 = np.zeros((h, w), dtype=np.float32)
        
        for i in range(min(4, num_peaks)):
            py, px, power = self._peaks[i]
            
            # Reconstruct this mode
            mode_img = self._isolate_mode(self._fft_complex, py, px, self.mask_radius)
            
            # Normalize for display
            if mode_img.max() > 0:
                mode_img = mode_img / mode_img.max()
            
            # Store
            self._mode_images[i] = mode_img
            
            # Convert to displayable
            mode_u8 = (mode_img * 255).astype(np.uint8)
            mode_colored = cv2.applyColorMap(mode_u8, cv2.COLORMAP_INFERNO)
            
            self._outputs[f'mode_{i}_image'] = mode_colored
            self._outputs[f'mode_{i}_power'] = float(power)
            
            # Compute spatial frequency
            if i == 0:
                freq = self._compute_spatial_frequency(py, px, h, w)
                self._outputs['mode_0_freq'] = float(freq)
            
            # Add to combined
            combined_top4 += mode_img * (power / (self._peaks[0][2] + 1e-9))
            total_power += power
        
        # Fill remaining mode outputs if not enough peaks
        for i in range(num_peaks, 4):
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            self._outputs[f'mode_{i}_image'] = blank
            self._outputs[f'mode_{i}_power'] = 0.0
        
        self._outputs['total_power'] = float(total_power)
        
        # === COMBINED TOP 4 ===
        if combined_top4.max() > 0:
            combined_top4 = combined_top4 / combined_top4.max()
        combined_u8 = (combined_top4 * 255).astype(np.uint8)
        combined_colored = cv2.applyColorMap(combined_u8, cv2.COLORMAP_VIRIDIS)
        self._outputs['top_4_combined'] = combined_colored
        
        # === SELECTED MODE ===
        sel_idx = self.selected_mode_idx
        if sel_idx < num_peaks:
            py, px, power = self._peaks[sel_idx]
            sel_img = self._isolate_mode(self._fft_complex, py, px, self.mask_radius)
            if sel_img.max() > 0:
                sel_img = sel_img / sel_img.max()
            sel_u8 = (sel_img * 255).astype(np.uint8)
            sel_colored = cv2.applyColorMap(sel_u8, cv2.COLORMAP_PLASMA)
            self._outputs['selected_mode'] = sel_colored
        else:
            self._outputs['selected_mode'] = np.zeros((h, w, 3), dtype=np.uint8)
        
        # === FILTERED RECONSTRUCTION ===
        # Reconstruct using only peaks above threshold
        filtered_recon = np.zeros((h, w), dtype=np.float32)
        for py, px, power in self._peaks:
            mode_img = self._isolate_mode(self._fft_complex, py, px, self.mask_radius)
            filtered_recon += mode_img
        
        if filtered_recon.max() > 0:
            filtered_recon = filtered_recon / filtered_recon.max()
        filt_u8 = (filtered_recon * 255).astype(np.uint8)
        filt_colored = cv2.applyColorMap(filt_u8, cv2.COLORMAP_MAGMA)
        self._outputs['filtered_recon'] = filt_colored
        
        # === MODE SPECTRUM (which frequencies are active) ===
        mode_spectrum = np.zeros((h, w), dtype=np.float32)
        for py, px, power in self._peaks:
            # Draw weighted spot at each peak position
            cv2.circle(mode_spectrum, (px, py), 3, float(power), -1)
        
        if mode_spectrum.max() > 0:
            mode_spectrum = mode_spectrum / mode_spectrum.max()
        spec_u8 = (mode_spectrum * 255).astype(np.uint8)
        spec_colored = cv2.applyColorMap(spec_u8, cv2.COLORMAP_HOT)
        self._outputs['mode_spectrum'] = spec_colored
        
    def get_output(self, port_name):
        """Return requested output."""
        return self._outputs.get(port_name, None)
    
    def get_display_image(self):
        """Create visualization for node face."""
        res = self.resolution
        
        # 2x2 grid: Peak Map | Mode 0 | Mode 1 | Combined
        half = res // 2
        display = np.zeros((res, res, 3), dtype=np.uint8)
        
        def resize_img(img):
            if img is None:
                return np.zeros((half, half, 3), dtype=np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return cv2.resize(img, (half, half))
        
        # Top-left: Peak map
        if self._peak_map is not None:
            display[:half, :half] = resize_img(self._peak_map)
        
        # Top-right: Mode 0
        m0 = self._outputs.get('mode_0_image')
        if m0 is not None:
            display[:half, half:] = resize_img(m0)
        
        # Bottom-left: Mode 1
        m1 = self._outputs.get('mode_1_image')
        if m1 is not None:
            display[half:, :half] = resize_img(m1)
        
        # Bottom-right: Combined
        comb = self._outputs.get('top_4_combined')
        if comb is not None:
            display[half:, half:] = resize_img(comb)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "PEAKS", (3, 12), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display, "MODE0", (half + 3, 12), font, 0.3, (255, 100, 100), 1)
        cv2.putText(display, "MODE1", (3, half + 12), font, 0.3, (100, 255, 100), 1)
        cv2.putText(display, "COMBINED", (half + 3, half + 12), font, 0.3, (100, 200, 255), 1)
        
        # Stats
        num = self._outputs.get('num_modes', 0)
        cv2.putText(display, f"N={int(num)}", (res - 40, res - 5), font, 0.3, (200, 200, 200), 1)
        
        display = np.ascontiguousarray(display)
        h, w = display.shape[:2]
        
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg.copy()
    
    def get_config_options(self):
        """Configuration options."""
        return [
            ("Mask Radius", "mask_radius", self.mask_radius, 'int'),
            ("Detection Threshold", "threshold", self.threshold, 'float'),
            ("Selected Mode Index", "selected_mode_idx", self.selected_mode_idx, 'int'),
            ("Max Modes to Track", "max_modes", self.max_modes, 'int'),
        ]
    
    def set_config_options(self, options):
        """Apply configuration."""
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("FFT Eigenmode Decoder")
    print("=" * 50)
    print()
    print("This node decodes spatial frequency components from FFT output.")
    print()
    print("INPUTS:")
    print("  fft_spectrum  - Complex FFT from HolographicFFTNode")
    print("  fft_magnitude - Magnitude image (fallback)")
    print("  mode_select   - Which mode to view (0-N)")
    print("  mask_radius   - Size of isolation mask")
    print("  threshold     - Peak detection sensitivity")
    print()
    print("OUTPUTS:")
    print("  mode_0_image through mode_3_image - Top 4 modes reconstructed")
    print("  selected_mode - User-selected mode")
    print("  top_4_combined - Sum of strongest modes")
    print("  filtered_recon - Full filtered reconstruction")
    print("  peak_map - Visualization of detected peaks")
    print()
    print("THEORY:")
    print("  Each bright spot in the FFT is a spatial frequency eigenmode.")
    print("  By isolating and inverse-transforming each spot, we see what")
    print("  standing wave pattern it contributes to the hologram.")
    print()
    print("  High SNR (like your 1275!) means strong coherent structure.")
    print("  This decoder reveals what that structure actually looks like.")
    print()
    
    # Create test pattern
    print("Creating test pattern...")
    res = 128
    x = np.linspace(-np.pi, np.pi, res)
    y = np.linspace(-np.pi, np.pi, res)
    X, Y = np.meshgrid(x, y)
    
    # Sum of a few sinusoids (like eigenmodes)
    pattern = (np.sin(3*X) * np.cos(3*Y) + 
               0.5 * np.sin(7*X + 2*Y) + 
               0.3 * np.cos(5*X - 4*Y))
    
    # FFT
    from scipy.fft import fft2, fftshift
    fft_result = fftshift(fft2(pattern))
    
    # Test node
    node = EigenmodeDecoderNode()
    # Override get_blended_input for testing
    node._test_fft = fft_result
    original_get = node.get_blended_input
    def test_get(name, mode):
        if name == 'fft_spectrum':
            return node._test_fft
        return None
    node.get_blended_input = test_get
    node.threshold = 0.01  # Lower threshold for test
    node.step()
    
    print(f"FFT shape: {fft_result.shape}")
    print(f"FFT magnitude range: {np.abs(fft_result).min():.2f} - {np.abs(fft_result).max():.2f}")
    print(f"Detected {node._outputs.get('num_modes', 0):.0f} modes")
    print(f"Mode 0 power: {node._outputs.get('mode_0_power', 0):.2f}")
    print(f"Mode 0 spatial frequency: {node._outputs.get('mode_0_freq', 0):.2f}")
    print()
    print("Node ready! Connect HolographicFFTNode → EigenmodeDecoder")