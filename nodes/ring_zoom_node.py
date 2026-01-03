"""
Ring Zoom Node
==============
Navigate through frequency shells in FFT space.

The FFT organizes information by spatial frequency:
- Center: DC (average brightness)
- Inner rings: Low frequencies (large structures)
- Outer rings: High frequencies (fine details)

This node lets you:
1. Select a specific frequency band (ring)
2. Extract only that band
3. Reconstruct what spatial pattern creates it
4. Animate through scales to see the "frequency movie"

Like tuning a radio dial through the spectrum of spatial reality.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# --- PERCEPTION LAB INTEGRATION ---
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
            self.node_title = "Base"
        def get_blended_input(self, name, mode): 
            return None
# ----------------------------------

PHI = (1 + np.sqrt(5)) / 2


class RingZoomNode(BaseNode):
    """
    Frequency Ring Navigator.
    
    Extracts and visualizes specific frequency bands from images/FFTs.
    Allows "zooming" through scale-space to see what structure
    exists at each frequency shell.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Ring Zoom"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Cyan - frequency analysis
    
    def __init__(self, size=512, center_freq=0.3, bandwidth=0.1, auto_sweep=False):
        super().__init__()
        self.node_title = "Ring Zoom"
        
        self.inputs = {
            'image': 'image',              # Input image
            'fft_input': 'complex_spectrum', # Or direct FFT input
            'center_mod': 'signal',         # Modulate center frequency
            'bandwidth_mod': 'signal',      # Modulate bandwidth
            'sweep_speed': 'signal',        # Auto-sweep speed
        }
        
        self.outputs = {
            # Extracted views
            'ring_spatial': 'image',        # What this frequency band looks like in space
            'ring_fft': 'image',             # The extracted ring in FFT space
            'full_fft': 'image',             # Full FFT with ring highlighted
            
            # Multi-ring analysis
            'phi_rings_view': 'image',       # Multiple rings at φ-spaced frequencies
            'ring_stack': 'image',           # Stack of extracted rings
            
            # Signals
            'ring_energy': 'signal',         # Energy in current ring
            'center_freq': 'signal',         # Current center frequency
            'peak_freq': 'signal',           # Frequency with most energy
        }
        
        # Parameters
        self.size = int(size)
        self.center_freq = float(center_freq)  # 0-1, fraction of max frequency
        self.bandwidth = float(bandwidth)       # Width of the ring
        self.auto_sweep = bool(auto_sweep)
        self.sweep_phase = 0.0
        
        # Force Resolution Setting (The Fix for "Pretty" Images)
        self.force_resolution = 512 
        
        # State
        self._spectrum = None
        self._ring_spatial = None
        self._ring_fft = None
        self._full_fft = None
        self._phi_rings = None
        self._ring_stack = None
        
        self._ring_energy = 0.0
        self._peak_freq = 0.0
        
        # Precompute frequency grid
        self._build_freq_grid()
        
    def _build_freq_grid(self):
        """Build normalized frequency distance grid."""
        cy, cx = self.size // 2, self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        
        # Distance from center (DC)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        if max_dist == 0: max_dist = 1
        
        # Normalized 0-1
        self.freq_grid = dist / max_dist
        
    def _extract_ring(self, spectrum, center, width):
        """
        Extract a frequency ring from the spectrum.
        
        Returns: masked spectrum, ring mask
        """
        # Create ring mask with soft edges
        inner = center - width / 2
        outer = center + width / 2
        
        # Soft edges using sigmoid-like transition
        # Higher sharpness = cleaner cuts, lower = softer blend
        sharpness = 30.0  
        inner_mask = 1 / (1 + np.exp(-sharpness * (self.freq_grid - inner)))
        outer_mask = 1 / (1 + np.exp(-sharpness * (outer - self.freq_grid)))
        
        ring_mask = inner_mask * outer_mask
        
        # Apply mask
        masked = spectrum * ring_mask
        
        return masked, ring_mask
        
    def _spectrum_to_spatial(self, spectrum):
        """Convert frequency spectrum back to spatial domain."""
        # Unshift, inverse FFT
        unshifted = ifftshift(spectrum)
        spatial = ifft2(unshifted)
        return np.real(spatial)
        
    def _find_peak_frequency(self, spectrum):
        """Find the frequency with maximum energy."""
        magnitude = np.abs(spectrum)
        
        # Radial profile
        n_bins = 100
        bin_edges = np.linspace(0, 1, n_bins + 1)
        radial_energy = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (self.freq_grid >= bin_edges[i]) & (self.freq_grid < bin_edges[i+1])
            if np.sum(mask) > 0:
                radial_energy[i] = np.mean(magnitude[mask])
                
        # Find peak (skip DC)
        if len(radial_energy) > 1:
            peak_idx = np.argmax(radial_energy[1:]) + 1
            self._peak_freq = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        
        return self._peak_freq
    
    # Alias step to update if the system calls update
    def update(self, packet=None):
        self.step()

    def step(self):
        """Main processing step."""
        # Get inputs
        img = self.get_blended_input('image', 'first')
        fft_in = self.get_blended_input('fft_input', 'first')
        center_mod = self.get_blended_input('center_mod', 'sum')
        bw_mod = self.get_blended_input('bandwidth_mod', 'sum')
        sweep_mod = self.get_blended_input('sweep_speed', 'sum')
        
        # === DYNAMIC RESIZING FIX ===
        # Determine target size: Force Res > Input Res > Current Res
        target_size = self.size
        
        # If we have an image, we can adapt
        if img is not None:
             # If force_resolution is set, use that (Super-sampling)
            if self.force_resolution > 0:
                target_size = self.force_resolution
            else:
                # Otherwise match input resolution (Robustness)
                target_size = max(img.shape[0], img.shape[1])
        
        # If the size changed, REBUILD THE GRID immediately
        # This prevents the "boolean index did not match" error
        if target_size != self.size:
            self.size = target_size
            self._build_freq_grid()

        # Compute FFT if image provided
        if img is not None:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img /= 255.0
                    
            if img.ndim == 3:
                img = cv2.cvtColor((img * 255).astype(np.uint8), 
                                  cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # RESIZE IMAGE TO TARGET (Super-sampling logic)
            if img.shape[0] != self.size or img.shape[1] != self.size:
                img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
            
            self._spectrum = fftshift(fft2(img))
            
        elif fft_in is not None:
            # Use provided FFT
            if fft_in.shape[0] != self.size:
                # Resize FFT mag/phase if sizes don't match (Advanced)
                mag = np.abs(fft_in)
                phase = np.angle(fft_in)
                mag = cv2.resize(mag, (self.size, self.size))
                phase = cv2.resize(phase, (self.size, self.size))
                self._spectrum = fftshift(mag * np.exp(1j * phase))
            else:
                self._spectrum = fftshift(fft_in)
        else:
            return
            
        # Apply modulations
        if center_mod is not None:
            self.center_freq = np.clip(0.3 + center_mod * 0.1, 0.01, 0.99)
        if bw_mod is not None:
            self.bandwidth = np.clip(0.1 + bw_mod * 0.05, 0.02, 0.5)
            
        # Auto-sweep
        if self.auto_sweep or (sweep_mod is not None and sweep_mod > 0):
            speed = sweep_mod if sweep_mod else 0.01
            self.sweep_phase += speed
            # Oscillate center frequency
            self.center_freq = 0.1 + 0.4 * (np.sin(self.sweep_phase) + 1) / 2
            
        # Find peak frequency
        self._find_peak_frequency(self._spectrum)
        
        # === EXTRACT CURRENT RING ===
        ring_spectrum, ring_mask = self._extract_ring(
            self._spectrum, self.center_freq, self.bandwidth
        )
        
        # Ring energy
        total_energy = np.sum(np.abs(self._spectrum)**2) + 1e-10
        self._ring_energy = np.sum(np.abs(ring_spectrum)**2) / total_energy
        
        # Convert ring to spatial
        self._ring_spatial = self._spectrum_to_spatial(ring_spectrum)
        
        # Visualize ring FFT
        ring_mag = np.log1p(np.abs(ring_spectrum))
        rm_min, rm_max = ring_mag.min(), ring_mag.max()
        self._ring_fft = (ring_mag - rm_min) / (rm_max - rm_min + 1e-10)
        
        # Full FFT with ring highlighted
        full_mag = np.log1p(np.abs(self._spectrum))
        fm_min, fm_max = full_mag.min(), full_mag.max()
        full_norm = (full_mag - fm_min) / (fm_max - fm_min + 1e-10)
        
        # Create color image with ring highlighted
        full_color = np.zeros((self.size, self.size, 3), dtype=np.float32)
        full_color[:, :, 0] = full_norm * (1 - ring_mask * 0.5)  # Red: full minus ring
        full_color[:, :, 1] = full_norm * ring_mask               # Green: ring only
        full_color[:, :, 2] = full_norm                           # Blue: full
        self._full_fft = full_color
        
        # === PHI-SPACED RINGS ===
        self._build_phi_rings()
        
        # === RING STACK ===
        self._build_ring_stack()
        
    def _build_phi_rings(self):
        """Extract multiple rings at φ-spaced frequencies."""
        if self._spectrum is None:
            return
            
        h, w = self.size, self.size
        n_rings = 5
        
        # Create visualization
        phi_view = np.zeros((h, w, 3), dtype=np.float32)
        
        # Base frequency and φ-spaced multiples
        base_freq = 0.05
        
        for i in range(n_rings):
            freq = base_freq * (PHI ** i)
            if freq > 0.9:
                break
                
            ring_spec, ring_mask = self._extract_ring(self._spectrum, freq, 0.05)
            ring_spatial = self._spectrum_to_spatial(ring_spec)
            
            # Normalize
            ring_norm = np.abs(ring_spatial)
            if ring_norm.max() > 0:
                ring_norm = ring_norm / ring_norm.max()
                
            # Add to view with different colors for each ring
            hue = i / n_rings
            # Simple HSV-like coloring
            r = np.clip(1 - abs(hue - 0) * 3, 0, 1) + np.clip(1 - abs(hue - 1) * 3, 0, 1)
            g = np.clip(1 - abs(hue - 0.33) * 3, 0, 1)
            b = np.clip(1 - abs(hue - 0.66) * 3, 0, 1)
            
            phi_view[:, :, 0] += ring_norm * r * 0.5
            phi_view[:, :, 1] += ring_norm * g * 0.5
            phi_view[:, :, 2] += ring_norm * b * 0.5
            
        # Normalize final
        phi_view = np.clip(phi_view, 0, 1)
        self._phi_rings = phi_view
        
    def _build_ring_stack(self):
        """Build a stack of rings at different frequencies."""
        if self._spectrum is None:
            return
            
        n_rings = 8
        ring_size = self.size // 4
        
        # Create grid of extracted rings
        stack = np.zeros((ring_size * 2, ring_size * 4, 3), dtype=np.float32)
        
        freqs = np.linspace(0.05, 0.8, n_rings)
        
        for i, freq in enumerate(freqs):
            ring_spec, _ = self._extract_ring(self._spectrum, freq, 0.08)
            ring_spatial = np.abs(self._spectrum_to_spatial(ring_spec))
            
            if ring_spatial.max() > 0:
                ring_spatial = ring_spatial / ring_spatial.max()
                
            # Resize for stack
            ring_small = cv2.resize(ring_spatial, (ring_size, ring_size))
            
            # Position in grid
            row = i // 4
            col = i % 4
            
            # Color by frequency (low=red, high=blue)
            t = freq
            color = np.zeros((ring_size, ring_size, 3), dtype=np.float32)
            color[:, :, 0] = ring_small * (1 - t)  # Red for low freq
            color[:, :, 1] = ring_small * (1 - abs(t - 0.5) * 2)  # Green for mid
            color[:, :, 2] = ring_small * t  # Blue for high freq
            
            y1, y2 = row * ring_size, (row + 1) * ring_size
            x1, x2 = col * ring_size, (col + 1) * ring_size
            
            # Fix bounds if rounding errors occur
            h_s, w_s = color.shape[:2]
            stack[y1:y1+h_s, x1:x1+w_s] = color
            
        self._ring_stack = stack
        
    def get_output(self, port_name):
        """Return outputs."""
        if port_name == 'ring_spatial':
            if self._ring_spatial is not None:
                # Normalize for output
                rs = np.abs(self._ring_spatial)
                if rs.max() > 0:
                    rs = rs / rs.max()
                return rs.astype(np.float32)
            return None
        elif port_name == 'ring_fft':
            return self._ring_fft
        elif port_name == 'full_fft':
            return self._full_fft
        elif port_name == 'phi_rings_view':
            return self._phi_rings
        elif port_name == 'ring_stack':
            return self._ring_stack
        elif port_name == 'ring_energy':
            return self._ring_energy
        elif port_name == 'center_freq':
            return self.center_freq
        elif port_name == 'peak_freq':
            return self._peak_freq
        return None
        
    def get_display_image(self):
        """Combined visualization for node face."""
        if self._ring_spatial is None:
            return None
            
        h, w = self.size // 2, self.size
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Left: Ring spatial
        if self._ring_spatial is not None:
            rs = np.abs(self._ring_spatial)
            if rs.max() > 0:
                rs = rs / rs.max()
            rs_small = cv2.resize(rs, (w // 2, h))
            rs_color = cv2.applyColorMap((rs_small * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            display[:, :w//2] = rs_color
            
        # Right: Full FFT with ring
        if self._full_fft is not None:
            fft_small = cv2.resize(self._full_fft, (w // 2, h))
            display[:, w//2:] = (fft_small * 255).astype(np.uint8)
            
        # Labels
        cv2.putText(display, f"f={self.center_freq:.2f}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"E={self._ring_energy:.2f}", (5, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Render for QT
        display = np.ascontiguousarray(display)
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg
        
    def get_config_options(self):
        """Configuration."""
        return [
            ("Size", "size", self.size, 'int'),
            ("Force Res", "force_resolution", self.force_resolution, 'int'),
            ("Center Frequency", "center_freq", self.center_freq, 'float'),
            ("Bandwidth", "bandwidth", self.bandwidth, 'float'),
            ("Auto Sweep", "auto_sweep", self.auto_sweep, 'bool'),
        ]