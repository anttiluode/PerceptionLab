"""
Phi-Sigh Bridge Node
====================
The Synchronicity Test: Are frequency filtering and φ-sampling the same operation?

HYPOTHESIS (from Gemini):
- Sigh (frequency): Bandpass filter keeps mid-frequencies (structure)
- Phi (spatial): Sparse sampling keeps scale-invariant features (structure)
- They are THE SAME FILTER in different domains

PREDICTION:
If you pre-filter with Sigh (remove high-freq noise), then φ-sample,
the artifacts should vanish because both are now "seeing the same ghost."

This node chains:
1. FFT bandpass (Sigh) - keeps structure frequencies, removes noise
2. φ-spiral sampling (Phi) - samples at golden-angle positions
3. Reconstruction - from sparse φ-samples

If the synchronicity is real:
- Pre-filtered + φ-sampled should be CLEANER than either alone
- The residual energy should be minimal (both agree on what matters)

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

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
GOLDEN_ANGLE = 2 * np.pi / (PHI * PHI)  # ~137.5°


class PhiSighBridgeNode(BaseNode):
    """
    The Bridge: Frequency filtering + Spatial φ-sampling unified.
    
    Tests whether Sigh (FFT bandpass) and Phi (golden sampling)
    are the same operation in different domains.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Phi-Sigh Bridge"
    NODE_COLOR = QtGui.QColor(255, 180, 100)  # Golden orange
    
    def __init__(self, size=256, n_samples=500, low_cut=0.02, high_cut=0.4):
        super().__init__()
        self.node_title = "φ-Sigh Bridge"
        
        self.inputs = {
            'image': 'image',           # Input image (webcam etc)
            'low_cut_mod': 'signal',    # Modulate low frequency cutoff
            'high_cut_mod': 'signal',   # Modulate high frequency cutoff
            'density_mod': 'signal',    # Modulate φ-sample density
        }
        
        self.outputs = {
            # The pipeline stages
            'original': 'image',           # Input as-is
            'sigh_filtered': 'image',      # After FFT bandpass (Sigh)
            'phi_sampled': 'image',        # φ-sample points visualization
            'phi_reconstructed': 'image',  # Reconstructed from φ-samples
            'residual': 'image',           # What was lost
            
            # Combined view
            'bridge_view': 'image',        # Side-by-side comparison
            
            # Analysis outputs
            'sigh_energy': 'signal',       # Energy in structure band
            'phi_coverage': 'signal',      # φ-sampling coverage metric
            'sync_score': 'signal',        # How well Sigh and Phi agree
            'residual_energy': 'signal',   # Energy in residual (lower = better sync)
        }
        
        # Parameters
        self.size = int(size)
        self.n_samples = int(n_samples)
        self.low_cut = float(low_cut)    # Low frequency cutoff (fraction of max)
        self.high_cut = float(high_cut)  # High frequency cutoff (fraction of max)
        
        # State
        self._original = None
        self._sigh_filtered = None
        self._phi_samples = None
        self._phi_reconstructed = None
        self._residual = None
        self._bridge_view = None
        
        self._sigh_energy = 0.0
        self._phi_coverage = 0.0
        self._sync_score = 0.0
        self._residual_energy = 0.0
        
        # Build φ-spiral sample positions
        self._build_phi_spiral()
        
    def _build_phi_spiral(self):
        """Build golden-angle spiral sample positions."""
        self.sample_points = []
        
        for i in range(self.n_samples):
            # Radius: sqrt gives uniform density, but we want foveal
            t = i / self.n_samples
            r = t ** 0.6 * 0.95  # Slightly foveal (denser center)
            
            # Golden angle rotation
            theta = i * GOLDEN_ANGLE
            
            # Normalized coordinates (-1 to 1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            self.sample_points.append((x, y))
            
        self.sample_points = np.array(self.sample_points)
        
    def _apply_sigh_filter(self, image):
        """
        Apply FFT bandpass filter (the "Sigh" operation).
        Keeps mid-frequencies where structure lives.
        """
        # FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        # Create bandpass mask
        h, w = image.shape
        cy, cx = h // 2, w // 2
        
        # Distance from center (normalized 0-1)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        # Bandpass: smooth transitions
        # Low cut: attenuate frequencies below this
        # High cut: attenuate frequencies above this
        low_mask = 1 - np.exp(-(dist_norm / self.low_cut)**2) if self.low_cut > 0 else 1
        high_mask = np.exp(-(dist_norm / self.high_cut)**4) if self.high_cut < 1 else 1
        
        bandpass = low_mask * high_mask
        
        # Apply filter
        fshift_filtered = fshift * bandpass
        
        # Compute energy in passband
        total_energy = np.sum(np.abs(fshift)**2)
        pass_energy = np.sum(np.abs(fshift_filtered)**2)
        self._sigh_energy = pass_energy / (total_energy + 1e-10)
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        filtered = np.fft.ifft2(f_ishift)
        
        return np.real(filtered)
        
    def _sample_with_phi_spiral(self, image):
        """
        Sample image at golden-spiral positions.
        Returns sample values and visualization.
        """
        h, w = image.shape
        
        samples = []
        sample_vis = np.zeros((h, w), dtype=np.float32)
        
        for x_norm, y_norm in self.sample_points:
            # Convert normalized coords to image coords
            ix = int((x_norm + 1) / 2 * (w - 1))
            iy = int((y_norm + 1) / 2 * (h - 1))
            
            if 0 <= ix < w and 0 <= iy < h:
                val = image[iy, ix]
                samples.append((ix, iy, val))
                
                # Mark in visualization
                cv2.circle(sample_vis, (ix, iy), 2, float(val), -1)
                
        # Coverage metric: fraction of image "reached" by samples
        # (using Voronoi-like coverage estimation)
        covered = np.sum(sample_vis > 0)
        self._phi_coverage = covered / (h * w)
        
        return samples, sample_vis
        
    def _reconstruct_from_samples(self, samples, shape):
        """
        Reconstruct image from sparse φ-spiral samples.
        Uses natural neighbor / linear interpolation.
        """
        h, w = shape
        
        if len(samples) < 4:
            return np.zeros((h, w), dtype=np.float32)
            
        # Extract sample positions and values
        points = np.array([(s[0], s[1]) for s in samples])
        values = np.array([s[2] for s in samples])
        
        # Create grid for interpolation
        grid_x, grid_y = np.mgrid[0:w, 0:h]
        
        # Interpolate using linear (fast) or cubic (smooth)
        try:
            reconstructed = griddata(points, values, (grid_x, grid_y), 
                                     method='linear', fill_value=np.mean(values))
            reconstructed = reconstructed.T  # griddata returns transposed
        except:
            reconstructed = np.full((h, w), np.mean(values), dtype=np.float32)
            
        # Fill any remaining NaN with mean
        reconstructed = np.nan_to_num(reconstructed, nan=np.mean(values))
        
        return reconstructed.astype(np.float32)
        
    def _compute_sync_score(self, sigh_image, phi_recon):
        """
        Compute how well Sigh and Phi "agree".
        High sync = both see the same structure.
        """
        # Normalize both
        sigh_norm = (sigh_image - sigh_image.min()) / (sigh_image.max() - sigh_image.min() + 1e-10)
        phi_norm = (phi_recon - phi_recon.min()) / (phi_recon.max() - phi_recon.min() + 1e-10)
        
        # Correlation coefficient
        sigh_flat = sigh_norm.flatten()
        phi_flat = phi_norm.flatten()
        
        correlation = np.corrcoef(sigh_flat, phi_flat)[0, 1]
        
        # Sync score: 0 = no agreement, 1 = perfect agreement
        self._sync_score = max(0, correlation)
        
        return self._sync_score
        
    def step(self):
        """Main processing step."""
        # Get input
        img = self.get_blended_input('image', 'first')
        
        # Get modulations
        low_mod = self.get_blended_input('low_cut_mod', 'sum')
        high_mod = self.get_blended_input('high_cut_mod', 'sum')
        density_mod = self.get_blended_input('density_mod', 'sum')
        
        if img is None:
            return
            
        # Apply modulations
        if low_mod is not None:
            self.low_cut = np.clip(0.02 + low_mod * 0.05, 0.001, 0.2)
        if high_mod is not None:
            self.high_cut = np.clip(0.4 + high_mod * 0.1, 0.1, 0.9)
        if density_mod is not None:
            new_n = int(np.clip(500 + density_mod * 100, 100, 2000))
            if new_n != self.n_samples:
                self.n_samples = new_n
                self._build_phi_spiral()
                
        # Convert to grayscale float
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0
                
        if img.ndim == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), 
                               cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = img
            
        # Resize for processing
        gray = cv2.resize(gray, (self.size, self.size))
        self._original = gray.copy()
        
        # === STAGE 1: SIGH (Frequency Bandpass) ===
        self._sigh_filtered = self._apply_sigh_filter(gray)
        
        # Normalize sigh output
        self._sigh_filtered = (self._sigh_filtered - self._sigh_filtered.min()) / \
                              (self._sigh_filtered.max() - self._sigh_filtered.min() + 1e-10)
        
        # === STAGE 2: PHI (Golden Spiral Sampling) ===
        # Sample from the SIGH-FILTERED image (the bridge!)
        samples, self._phi_samples = self._sample_with_phi_spiral(self._sigh_filtered)
        
        # === STAGE 3: RECONSTRUCT ===
        self._phi_reconstructed = self._reconstruct_from_samples(samples, gray.shape)
        
        # === STAGE 4: RESIDUAL ===
        self._residual = np.abs(self._sigh_filtered - self._phi_reconstructed)
        self._residual_energy = np.mean(self._residual**2)
        
        # === COMPUTE SYNC SCORE ===
        self._compute_sync_score(self._sigh_filtered, self._phi_reconstructed)
        
        # === BUILD BRIDGE VIEW ===
        self._build_bridge_view()
        
    def _build_bridge_view(self):
        """Create combined visualization."""
        h, w = self.size, self.size
        
        # 2x2 grid: original | sigh | phi_recon | residual
        view = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        def to_color(img, cmap='gray'):
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
            img_uint8 = (img_norm * 255).astype(np.uint8)
            if cmap == 'inferno':
                return cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)
            elif cmap == 'hot':
                return cv2.applyColorMap(img_uint8, cv2.COLORMAP_HOT)
            else:
                return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        
        # Original (top-left)
        if self._original is not None:
            view[:h, :w] = to_color(self._original)
            cv2.putText(view, "Original", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Sigh filtered (top-right)
        if self._sigh_filtered is not None:
            view[:h, w:] = to_color(self._sigh_filtered, 'inferno')
            cv2.putText(view, f"Sigh (E={self._sigh_energy:.2f})", (w + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Phi reconstructed (bottom-left)
        if self._phi_reconstructed is not None:
            view[h:, :w] = to_color(self._phi_reconstructed, 'inferno')
            cv2.putText(view, f"Phi Recon (C={self._phi_coverage:.2f})", (5, h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Residual (bottom-right) - what Sigh and Phi disagree on
        if self._residual is not None:
            view[h:, w:] = to_color(self._residual, 'hot')
            cv2.putText(view, f"Residual (Sync={self._sync_score:.2f})", (w + 5, h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Sync score bar at bottom
        bar_width = int(self._sync_score * w * 2)
        cv2.rectangle(view, (0, h * 2 - 10), (bar_width, h * 2), (0, 255, 0), -1)
        
        self._bridge_view = view
        
    def get_output(self, port_name):
        """Return outputs."""
        if port_name == 'original':
            return self._original
        elif port_name == 'sigh_filtered':
            return self._sigh_filtered
        elif port_name == 'phi_sampled':
            return self._phi_samples
        elif port_name == 'phi_reconstructed':
            return self._phi_reconstructed
        elif port_name == 'residual':
            return self._residual
        elif port_name == 'bridge_view':
            return self._bridge_view
        elif port_name == 'sigh_energy':
            return self._sigh_energy
        elif port_name == 'phi_coverage':
            return self._phi_coverage
        elif port_name == 'sync_score':
            return self._sync_score
        elif port_name == 'residual_energy':
            return self._residual_energy
        return None
        
    def get_display_image(self):
        """Return bridge view for node face."""
        if self._bridge_view is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(img, "Phi-Sigh Bridge", (20, 128),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 100), 1)
            return QtGui.QImage(img.data, 256, 256, 768, QtGui.QImage.Format.Format_RGB888)
            
        h, w = self._bridge_view.shape[:2]
        view = np.ascontiguousarray(self._bridge_view)
        
        qimg = QtGui.QImage(view.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = view
        return qimg
        
    def get_config_options(self):
        """Configuration."""
        return [
            ("Size", "size", self.size, 'int'),
            ("Phi Samples", "n_samples", self.n_samples, 'int'),
            ("Low Freq Cut", "low_cut", self.low_cut, 'float'),
            ("High Freq Cut", "high_cut", self.high_cut, 'float'),
        ]


# === STANDALONE TEST ===
if __name__ == "__main__":
    import numpy as np
    PHI = (1 + np.sqrt(5)) / 2
    GOLDEN_ANGLE = 2 * np.pi / (PHI * PHI)
    
    print("Phi-Sigh Bridge: The Synchronicity Test")
    print("=" * 60)
    print()
    print("HYPOTHESIS:")
    print("  Sigh (FFT bandpass) and Phi (golden sampling)")
    print("  are the SAME FILTER expressed in different domains.")
    print()
    print("PREDICTION:")
    print("  If true, pre-filtering with Sigh should make")
    print("  Phi-sampling produce CLEANER results, because")
    print("  both are now 'seeing the same ghost.'")
    print()
    print("THE BRIDGE:")
    print("  Input → Sigh (frequency filter) → Phi (spatial sample) → Output")
    print()
    print("SYNC SCORE:")
    print("  High sync = Sigh and Phi agree on structure")
    print("  Low residual = Both captured the same information")
    print()
    print("If sync_score → 1.0 and residual_energy → 0.0,")
    print("the synchronicity is REAL.")
    print()
    print(f"Golden Angle: {np.degrees(GOLDEN_ANGLE):.2f}°")
    print(f"φ = {PHI:.6f}")
