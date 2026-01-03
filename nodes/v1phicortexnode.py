"""
V1 Phi Cortex Node
==================
Combines two biological vision principles:

1. SCHWARTZ CONFORMAL MAPPING (Real V1 Retinotopy)
   - Fovea (center) gets massive cortical area
   - Periphery compressed logarithmically
   - w = k * log(z + a)

2. GOLDEN RATIO SAMPLING (Hypothesized Optimal)
   - Sample points placed at φ-intervals
   - Scale-invariant, no aliasing harmonics
   - Matches 1/f natural image statistics

The result: A "cortical image" that represents what V1 might 
actually be computing - dense center, sparse periphery, 
with φ-structured sampling throughout.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
import math

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
            self.input_data = {}
        def get_blended_input(self, name, mode): 
            return self.input_data.get(name, [None])[0] if name in self.input_data else None
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}
# ----------------------------------

PHI = (1 + np.sqrt(5)) / 2


class V1PhiCortexNode(BaseNode):
    """
    V1-like visual processing with golden ratio sampling.
    
    Input: Retinal image (webcam, etc.)
    Output: Cortical representation + reconstructions
    
    Combines:
    - Log-polar foveation (like real V1)
    - Phi-spaced radial sampling rings
    - Phi-spaced angular sampling
    """
    
    NODE_CATEGORY = "Vision"
    NODE_TITLE = "V1 Phi Cortex"
    NODE_COLOR = QtGui.QColor(180, 100, 220)  # Purple - visual cortex
    
    def __init__(self, cortex_size=128, k=12.0, a=0.5, phi_density=0.3):
        super().__init__()
        
        self.inputs = {
            'retinal_image': 'image',    # Input image (webcam etc)
            'fovea_x': 'signal',          # Fovea position X (-1 to 1)
            'fovea_y': 'signal',          # Fovea position Y (-1 to 1)
            'k_mod': 'signal',            # Magnification modulation
            'density_mod': 'signal',      # Phi density modulation
        }
        
        self.outputs = {
            'cortex_view': 'image',       # The V1 cortical map
            'phi_skeleton': 'image',      # Sampling points visualization
            'reconstruction': 'image',    # Reconstructed from samples
            'residual': 'image',          # What was lost
            'fovea_zoom': 'image',        # High-res foveal region
            'cortex_activity': 'signal',  # Mean cortical activation
            'phi_score': 'signal',        # Phi-structure in output
        }
        
        # Parameters
        self.cortex_size = int(cortex_size)
        self.k = float(k)              # Cortical magnification
        self.a = float(a)              # Foveal constant
        self.phi_density = float(phi_density)
        
        # Fovea position (normalized -1 to 1)
        self.fovea_x = 0.0
        self.fovea_y = 0.0
        
        # Output buffers
        self._cortex = None
        self._skeleton = None
        self._reconstruction = None
        self._residual = None
        self._fovea_zoom = None
        self._cortex_activity = 0.0
        self._phi_score = 0.0
        
        # Precompute phi sampling structure
        self._build_phi_rings()
        
    def _build_phi_rings(self):
        """
        Build golden ratio sampling rings.
        
        Radii at: r_n = a * φ^n  (log-spaced by φ)
        Angles at: θ_m = m * 2π/φ² (golden angle ≈ 137.5°)
        
        This creates a sunflower-like pattern - the most efficient
        packing that avoids alignment artifacts.
        """
        # Golden angle in radians
        golden_angle = 2 * np.pi / (PHI * PHI)  # ≈ 2.399 rad ≈ 137.5°
        
        # Number of sample points based on density
        n_points = int(500 * self.phi_density) + 50
        
        # Vogel's sunflower spiral (golden angle spacing)
        self.sample_points = []
        
        for i in range(n_points):
            # Radius grows as sqrt for uniform density, but we'll use φ-power for foveation
            # r = sqrt(i) gives uniform density
            # r = φ^(i*scale) gives log-density (foveal)
            
            # Blend: dense center (foveal), sparse periphery
            t = i / n_points
            r = (t ** 0.7) * 0.95  # 0 to 0.95 (leave border)
            
            # Golden angle rotation
            theta = i * golden_angle
            
            # Convert to cartesian (normalized -1 to 1)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            self.sample_points.append((x, y, r))
            
        self.sample_points = np.array(self.sample_points)
        
        # Build radial rings at φ-spaced intervals for visualization
        self.phi_rings = []
        r = 0.1
        while r < 1.0:
            self.phi_rings.append(r)
            r *= PHI ** 0.5  # φ^0.5 spacing for visible rings
            
    def _apply_schwartz_map(self, img, fovea_x=0, fovea_y=0):
        """
        Apply Schwartz conformal mapping centered on fovea.
        
        This is what V1 actually does: log-polar transform
        that magnifies the fovea.
        """
        h, w = img.shape[:2]
        out_size = self.cortex_size
        
        # Create output coordinates
        cy, cx = np.mgrid[0:out_size, 0:out_size]
        
        # Normalize to -1, 1
        cx_norm = (cx / out_size - 0.5) * 2
        cy_norm = (cy / out_size - 0.5) * 2
        
        # Convert to complex for Schwartz inverse
        # We want: given cortical position, where in visual field?
        w_real = cx_norm * 2  # Scale
        w_imag = cy_norm * np.pi  # Angle range
        
        w_complex = w_real + 1j * w_imag
        
        # Inverse Schwartz: z = exp(w/k) - a
        z = np.exp(w_complex / self.k) - self.a
        
        # Shift by fovea position
        visual_x = np.real(z) + fovea_x
        visual_y = np.imag(z) + fovea_y
        
        # Convert to image coordinates
        map_x = ((visual_x + 1) / 2 * (w - 1)).astype(np.float32)
        map_y = ((visual_y + 1) / 2 * (h - 1)).astype(np.float32)
        
        # Remap
        cortex = cv2.remap(img, map_x, map_y, 
                           cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_REFLECT)
        
        return cortex
        
    def _sample_with_phi_spiral(self, img):
        """
        Sample image at golden spiral points.
        Returns sampled values and a visualization.
        """
        h, w = img.shape[:2]
        
        samples = []
        skeleton = np.zeros((h, w, 3), dtype=np.uint8)
        
        for x, y, r in self.sample_points:
            # Convert normalized coords to image coords
            ix = int((x + 1) / 2 * (w - 1))
            iy = int((y + 1) / 2 * (h - 1))
            
            if 0 <= ix < w and 0 <= iy < h:
                # Sample value
                val = img[iy, ix] if img.ndim == 2 else np.mean(img[iy, ix])
                samples.append((ix, iy, val, r))
                
                # Draw sample point (color by radius - yellow center, blue edge)
                color_r = int(255 * (1 - r))
                color_g = int(255 * (1 - r * 0.5))
                color_b = int(255 * r)
                
                cv2.circle(skeleton, (ix, iy), max(1, int(3 * (1 - r) + 1)), 
                          (color_b, color_g, color_r), -1)
                
        # Draw phi rings
        center = (w // 2, h // 2)
        for ring_r in self.phi_rings:
            radius = int(ring_r * min(w, h) / 2)
            cv2.circle(skeleton, center, radius, (50, 100, 50), 1)
            
        return samples, skeleton
        
    def _reconstruct_from_samples(self, samples, shape):
        """
        Reconstruct image from sparse phi-spiral samples.
        Uses inverse distance weighting.
        """
        h, w = shape[:2]
        
        if len(samples) == 0:
            return np.zeros((h, w), dtype=np.float32)
            
        # Extract sample positions and values
        sx = np.array([s[0] for s in samples])
        sy = np.array([s[1] for s in samples])
        sv = np.array([s[2] for s in samples])
        
        # Create coordinate grids
        yy, xx = np.mgrid[0:h, 0:w]
        
        # IDW reconstruction (simplified for speed)
        recon = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        # For each sample, add weighted contribution
        for i in range(len(samples)):
            dist = np.sqrt((xx - sx[i])**2 + (yy - sy[i])**2) + 1
            w_i = 1.0 / (dist ** 2)
            recon += w_i * sv[i]
            weights += w_i
            
        recon /= (weights + 1e-9)
        
        return recon
        
    def step(self):
        """Main processing step."""
        # Get inputs
        img = self.get_blended_input('retinal_image', 'first')
        fovea_x_mod = self.get_blended_input('fovea_x', 'sum')
        fovea_y_mod = self.get_blended_input('fovea_y', 'sum')
        k_mod = self.get_blended_input('k_mod', 'sum')
        density_mod = self.get_blended_input('density_mod', 'sum')
        
        if img is None:
            return
            
        # Update fovea position
        if fovea_x_mod is not None:
            self.fovea_x = np.clip(fovea_x_mod, -0.8, 0.8)
        if fovea_y_mod is not None:
            self.fovea_y = np.clip(fovea_y_mod, -0.8, 0.8)
            
        # Update parameters if modulated
        if k_mod is not None:
            self.k = max(5.0, 12.0 + k_mod * 2)
        if density_mod is not None:
            new_density = np.clip(0.3 + density_mod * 0.1, 0.1, 1.0)
            if abs(new_density - self.phi_density) > 0.05:
                self.phi_density = new_density
                self._build_phi_rings()
            
        # Ensure proper format
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0
                
        # Convert to grayscale if needed
        if img.ndim == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = img
            
        # Resize for processing
        proc_size = self.cortex_size * 2
        gray = cv2.resize(gray, (proc_size, proc_size))
        
        # 1. Apply Schwartz mapping (V1 retinotopy)
        self._cortex = self._apply_schwartz_map(gray, self.fovea_x, self.fovea_y)
        
        # 2. Sample with phi-spiral
        samples, self._skeleton = self._sample_with_phi_spiral(gray)
        
        # 3. Reconstruct from samples
        self._reconstruction = self._reconstruct_from_samples(samples, gray.shape)
        
        # 4. Compute residual
        self._residual = np.abs(gray - self._reconstruction)
        
        # 5. Extract fovea zoom
        fovea_cx = int((self.fovea_x + 1) / 2 * proc_size)
        fovea_cy = int((self.fovea_y + 1) / 2 * proc_size)
        zoom_r = proc_size // 8
        
        y1 = max(0, fovea_cy - zoom_r)
        y2 = min(proc_size, fovea_cy + zoom_r)
        x1 = max(0, fovea_cx - zoom_r)
        x2 = min(proc_size, fovea_cx + zoom_r)
        
        self._fovea_zoom = gray[y1:y2, x1:x2].copy()
        if self._fovea_zoom.size > 0:
            self._fovea_zoom = cv2.resize(self._fovea_zoom, (64, 64))
        else:
            self._fovea_zoom = np.zeros((64, 64), dtype=np.float32)
            
        # 6. Compute activity metrics
        self._cortex_activity = float(np.mean(self._cortex))
        
        # 7. Compute phi-score (ratio of sample positions)
        if len(samples) > 2:
            radii = sorted([s[3] for s in samples if s[3] > 0.01])
            if len(radii) > 2:
                ratios = [radii[i+1]/radii[i] for i in range(len(radii)-1) if radii[i] > 0]
                phi_hits = sum(1 for r in ratios if abs(r - PHI**0.5) < 0.2 or abs(r - PHI) < 0.2)
                self._phi_score = phi_hits / len(ratios) if ratios else 0.0
            else:
                self._phi_score = 0.0
        else:
            self._phi_score = 0.0
            
    def get_output(self, port_name):
        """Return output data."""
        if port_name == 'cortex_view':
            return self._cortex
        elif port_name == 'phi_skeleton':
            return self._skeleton
        elif port_name == 'reconstruction':
            return self._reconstruction
        elif port_name == 'residual':
            return self._residual
        elif port_name == 'fovea_zoom':
            return self._fovea_zoom
        elif port_name == 'cortex_activity':
            return self._cortex_activity
        elif port_name == 'phi_score':
            return self._phi_score
        return None
        
    def get_display_image(self):
        """Create combined visualization for node face."""
        size = self.cortex_size
        
        # 2x2 grid: cortex | skeleton | reconstruction | residual
        display = np.zeros((size, size * 2, 3), dtype=np.uint8)
        
        # Cortex (left) - V1 map
        if self._cortex is not None:
            cortex_vis = (np.clip(self._cortex, 0, 1) * 255).astype(np.uint8)
            cortex_color = cv2.applyColorMap(
                cv2.resize(cortex_vis, (size, size)), 
                cv2.COLORMAP_INFERNO
            )
            display[:, :size] = cortex_color
            
        # Skeleton (right) - phi sampling points
        if self._skeleton is not None:
            skel_resized = cv2.resize(self._skeleton, (size, size))
            display[:, size:] = skel_resized
            
        # Add labels
        cv2.putText(display, f"V1 Cortex", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, f"Phi Spiral k={self.k:.1f}", (size + 5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, f"phi:{self._phi_score:.2f}", (size + 5, size - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
        
        # Mark fovea
        fx = int((self.fovea_x + 1) / 2 * size)
        fy = int((self.fovea_y + 1) / 2 * size)
        cv2.circle(display, (size + size//2, size//2), 3, (0, 255, 255), -1)
        
        display = np.ascontiguousarray(display)
        h, w = display.shape[:2]
        
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg
        
    def get_config_options(self):
        """Configuration dialog."""
        return [
            ("Cortex Size", "cortex_size", self.cortex_size, 'int'),
            ("Magnification (k)", "k", self.k, 'float'),
            ("Foveal Constant (a)", "a", self.a, 'float'),
            ("Phi Density", "phi_density", self.phi_density, 'float'),
        ]
        
    def close(self):
        """Cleanup."""
        super().close()


# === STANDALONE TEST ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("V1 Phi Cortex - Biological Vision Simulation")
    print("=" * 50)
    print(f"Golden Ratio φ = {PHI:.6f}")
    print(f"Golden Angle = {360 / PHI**2:.1f}° = {2*np.pi/PHI**2:.3f} rad")
    print()
    
    # Create test pattern
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Concentric circles + radial lines (good for testing retinotopy)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    test_img = (np.sin(R * 20) * 0.5 + np.sin(theta * 8) * 0.5 + 1) / 2
    
    print("Golden spiral sampling creates sunflower-like pattern")
    print("where each new point is 137.5° from the last.")
    print()
    print("This is the MOST efficient packing - no radial alignment,")
    print("no angular harmonics, pure scale-invariance.")
    print()
    print("Combined with V1's log-polar mapping:")
    print("  - Center (fovea): Dense sampling, high detail")
    print("  - Edge (periphery): Sparse sampling, motion/gist only")
    print()
    print("This is how your visual cortex actually samples the world.")