"""
HOLOGRAPHIC DECODER NODE
========================
The Nobel Prize Attempt: Decoding Images from EEG Phase Interference

THEORY (Based on Antti's Convergence Discovery):
1. Janus Cabbage shows: Two images can be stored orthogonally in phase space
2. HexaCortex shows: Information concentrates at hex-grid sample points and CORNERS
3. PhiHologram shows: EEG creates Figure-8 dipole patterns (bicameral interference)

HYPOTHESIS:
If the brain encodes visual/cognitive content holographically, then:
- The CORNERS of the interference pattern contain "peripheral storage"
- The LOBES of the Figure-8 contain "hemispheric content"  
- PHASE ROTATION can reveal hidden orthogonal information (like Janus)

THIS NODE ATTEMPTS:
1. Sample the complex field at strategic points (hex grid + corners + lobes)
2. Rotate through phase space searching for coherent structure
3. Apply inverse transforms to attempt image reconstruction
4. Use eigenmode analysis to find dominant patterns

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
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[HolographicDecoder] Warning: scipy not available")


class HolographicDecoderNode(BaseNode):
    """
    Attempts to decode latent images from EEG holographic interference patterns.
    Uses principles from Janus Cabbage (phase orthogonality) and HexaCortex (hex sampling).
    """
    
    NODE_NAME = "Holographic Decoder"
    NODE_TITLE = "Holographic Decoder"
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(255, 180, 0) if QtGui else None  # Gold - for the Nobel ;)
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            # Complex field from PhiHologram
            'complex_field': 'complex_spectrum',
            
            # Optional: Reference image for correlation
            'reference_image': 'image',
            
            # Control signals
            'phase_search': 'signal',      # Manual phase angle to examine (0-360)
            'hex_scale': 'signal',         # Scale of hex sampling grid
            'corner_weight': 'signal',     # How much to weight corner samples
            'lobe_separation': 'signal',   # Distance between Figure-8 lobes
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Decoded attempts
            'decoded_phase_0': 'image',     # Reconstruction at phase 0
            'decoded_phase_90': 'image',    # Reconstruction at phase 90 (orthogonal)
            'decoded_current': 'image',     # Reconstruction at current search phase
            
            # Analysis views
            'fft_magnitude': 'image',       # FFT of input field
            'sample_points': 'image',       # Visualization of sample grid
            'phase_coherence': 'image',     # Phase coherence map
            'eigenmode_view': 'image',      # Dominant eigenmode
            
            # Hex sampling outputs (like HexaCortex)
            'hex_samples': 'spectrum',      # Raw samples from hex grid
            'corner_samples': 'spectrum',   # Samples from corners
            'lobe_samples': 'spectrum',     # Samples from Figure-8 lobes
            
            # Signals
            'coherence_score': 'signal',    # How "image-like" is the current decode
            'optimal_phase': 'signal',      # Auto-detected best phase angle
            'lobe_asymmetry': 'signal',     # Left/Right lobe difference
        }
        
        # === CONFIG ===
        self.hex_rings = 3
        self.search_resolution = 36  # Check every 10 degrees
        self.corner_radius = 0.15    # How far into corners to sample
        self.auto_search = True      # Automatically search for optimal phase
        
        # === STATE ===
        self.last_field = None
        self.fft_cache = None
        self.phase_cache = {}  # Cache reconstructions at different phases
        self.optimal_phase_found = 0.0
        self.coherence_history = []
        
        # Hex grid (computed once)
        self.hex_offsets = self._build_hex_grid()
        
        # Output caches
        self.decoded_images = {0: None, 90: None, 'current': None}
        self.analysis_images = {}
        self.sample_vectors = {'hex': None, 'corner': None, 'lobe': None}
        self.metrics = {'coherence': 0.0, 'optimal_phase': 0.0, 'asymmetry': 0.0}
        
        # Display
        self.display_image = None
        self._init_display()
    
    def _build_hex_grid(self):
        """Build hexagonal sampling grid (like HexaCortex ommatidia)."""
        offsets = [(0.0, 0.0)]  # Center
        scale = 0.12  # Spacing
        
        for ring in range(1, self.hex_rings + 1):
            for i in range(6 * ring):
                angle = np.pi / 3.0 * (i / ring)
                dx = ring * scale * np.cos(angle)
                dy = ring * scale * np.sin(angle)
                offsets.append((dx, dy))
        
        return np.array(offsets)
    
    def get_config_options(self):
        return [
            ("Hex Rings", "hex_rings", self.hex_rings, None),
            ("Search Resolution", "search_resolution", self.search_resolution, None),
            ("Corner Radius", "corner_radius", self.corner_radius, None),
            ("Auto Search", "auto_search", self.auto_search, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            # Rebuild hex grid if rings changed
            if 'hex_rings' in options:
                self.hex_offsets = self._build_hex_grid()
    
    def _init_display(self):
        w, h = 400, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "HOLOGRAPHIC DECODER", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
        cv2.putText(img, "Connect complex_field", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "from PhiHologram", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        self.display_image = img
    
    # === CORE DECODING METHODS ===
    
    def _rotate_phase(self, field, angle_rad):
        """Rotate complex field by phase angle (Janus-style)."""
        rotation = np.exp(1j * angle_rad)
        return field * rotation
    
    def _sample_at_points(self, field, points, width=3):
        """Sample complex field at specific normalized points."""
        h, w = field.shape
        samples = []
        
        for px, py in points:
            # Convert normalized (-1,1) to pixel coords
            ix = int((px + 1) / 2 * (w - 1))
            iy = int((py + 1) / 2 * (h - 1))
            
            # Clamp
            ix = max(width, min(w - width - 1, ix))
            iy = max(width, min(h - width - 1, iy))
            
            # Sample with small window average
            sample = np.mean(field[iy-width:iy+width+1, ix-width:ix+width+1])
            samples.append(sample)
        
        return np.array(samples, dtype=np.complex64)
    
    def _get_corner_points(self, radius=0.15):
        """Get sampling points in the corners (where HexaCortex hides info)."""
        corners = [
            (-1 + radius, -1 + radius),   # Bottom-left
            (1 - radius, -1 + radius),    # Bottom-right
            (-1 + radius, 1 - radius),    # Top-left
            (1 - radius, 1 - radius),     # Top-right
        ]
        # Add intermediate points
        for corner in corners.copy():
            for offset in [0.05, 0.1]:
                corners.append((corner[0], corner[1] + offset))
                corners.append((corner[0] + offset, corner[1]))
        return corners
    
    def _get_lobe_points(self, separation=0.4):
        """Get sampling points at Figure-8 lobes (hemispheric dipole)."""
        lobes = [
            (-separation, 0),  # Left lobe
            (separation, 0),   # Right lobe
        ]
        # Add points around each lobe
        for lx, ly in [(-separation, 0), (separation, 0)]:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                lobes.append((lx + 0.1*np.cos(angle), ly + 0.1*np.sin(angle)))
        return lobes
    
    def _compute_coherence(self, field):
        """Compute phase coherence score (how "organized" is the pattern)."""
        # High coherence = phases are not random
        phases = np.angle(field)
        
        # Measure local phase gradient consistency
        dy, dx = np.gradient(phases)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Low gradient = high coherence (smooth phase)
        # But we also want SOME structure, not flat
        mean_grad = np.mean(gradient_magnitude)
        std_grad = np.std(gradient_magnitude)
        
        # Score: want moderate gradient with low variance (organized waves)
        if mean_grad > 0:
            coherence = std_grad / (mean_grad + 1e-6)
            coherence = np.clip(1.0 - coherence, 0, 1)
        else:
            coherence = 0.0
        
        return float(coherence)
    
    def _attempt_decode(self, field, phase_angle):
        """Attempt to decode an image at a specific phase angle."""
        # Rotate to target phase
        rotated = self._rotate_phase(field, phase_angle)
        
        # Method 1: Direct magnitude extraction
        magnitude = np.abs(rotated)
        
        # Method 2: Real part extraction (like Janus readout)
        real_part = np.real(rotated)
        
        # Method 3: IFFT (inverse holographic reconstruction)
        try:
            ifft_result = ifft2(ifftshift(rotated))
            ifft_magnitude = np.abs(ifft_result)
        except:
            ifft_magnitude = magnitude
        
        # Combine methods with learned weighting
        # For now, use magnitude as primary
        decoded = magnitude
        
        # Normalize
        if np.max(decoded) > np.min(decoded):
            decoded = (decoded - np.min(decoded)) / (np.max(decoded) - np.min(decoded))
        
        return decoded
    
    def _search_optimal_phase(self, field):
        """Search for the phase angle with highest coherence/structure."""
        best_phase = 0.0
        best_score = 0.0
        
        for i in range(self.search_resolution):
            angle = (i / self.search_resolution) * 2 * np.pi
            
            rotated = self._rotate_phase(field, angle)
            score = self._compute_coherence(rotated)
            
            # Also check for image-like statistics
            magnitude = np.abs(rotated)
            # Images have specific histogram properties
            hist_score = self._histogram_score(magnitude)
            
            combined_score = score * 0.5 + hist_score * 0.5
            
            if combined_score > best_score:
                best_score = combined_score
                best_phase = angle
        
        return best_phase, best_score
    
    def _histogram_score(self, image):
        """Score how 'image-like' a pattern is based on histogram."""
        # Natural images have specific histogram shapes
        # Random noise has flat histograms
        
        # Normalize
        if np.max(image) > np.min(image):
            norm = (image - np.min(image)) / (np.max(image) - np.min(image))
        else:
            return 0.0
        
        # Compute histogram
        hist, _ = np.histogram(norm.flatten(), bins=50, range=(0, 1))
        hist = hist / (np.sum(hist) + 1e-6)
        
        # Natural images: histogram is NOT uniform
        # Measure deviation from uniform
        uniform = 1.0 / 50
        deviation = np.sum(np.abs(hist - uniform))
        
        # Also check for peaks (edges, textures)
        try:
            peaks, _ = find_peaks(hist, height=uniform * 2)
            peak_score = len(peaks) / 10.0  # Normalize
        except:
            peak_score = 0.0
        
        return float(np.clip(deviation * 0.5 + peak_score * 0.5, 0, 1))
    
    def _compute_fft_analysis(self, field):
        """Compute FFT of the field for visualization."""
        try:
            fft = fftshift(fft2(field))
            magnitude = np.log(np.abs(fft) + 1)
            
            # Normalize for display
            if np.max(magnitude) > np.min(magnitude):
                magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
            
            return magnitude
        except:
            return np.zeros_like(np.abs(field))
    
    def _compute_lobe_asymmetry(self, field):
        """Measure asymmetry between left and right lobes (hemispheric difference)."""
        h, w = field.shape
        
        # Left half
        left = field[:, :w//2]
        # Right half
        right = field[:, w//2:]
        
        left_power = np.mean(np.abs(left)**2)
        right_power = np.mean(np.abs(right)**2)
        
        if left_power + right_power > 0:
            asymmetry = (right_power - left_power) / (left_power + right_power)
        else:
            asymmetry = 0.0
        
        return float(asymmetry)
    
    # === MAIN STEP ===
    
    def step(self):
        """Main processing step."""
        
        # Get input field
        field = self.get_blended_input('complex_field', 'mean')
        
        if field is None or not np.iscomplexobj(field):
            return
        
        self.last_field = field
        h, w = field.shape
        
        # Get control inputs
        search_phase = self.get_blended_input('phase_search', 'sum')
        if search_phase is None:
            search_phase = 0.0
        search_phase_rad = np.deg2rad(float(search_phase))
        
        hex_scale = self.get_blended_input('hex_scale', 'sum')
        if hex_scale is None:
            hex_scale = 1.0
        
        corner_weight = self.get_blended_input('corner_weight', 'sum')
        if corner_weight is None:
            corner_weight = 1.0
        
        lobe_sep = self.get_blended_input('lobe_separation', 'sum')
        if lobe_sep is None:
            lobe_sep = 0.4
        
        # === SAMPLING ===
        
        # Hex grid sampling (scaled)
        scaled_hex = self.hex_offsets * float(hex_scale)
        self.sample_vectors['hex'] = self._sample_at_points(field, scaled_hex)
        
        # Corner sampling
        corner_points = self._get_corner_points(self.corner_radius)
        self.sample_vectors['corner'] = self._sample_at_points(field, corner_points)
        
        # Lobe sampling
        lobe_points = self._get_lobe_points(float(lobe_sep))
        self.sample_vectors['lobe'] = self._sample_at_points(field, lobe_points)
        
        # === PHASE SEARCH ===
        
        if self.auto_search:
            optimal_phase, coherence = self._search_optimal_phase(field)
            self.optimal_phase_found = optimal_phase
            self.metrics['optimal_phase'] = float(np.rad2deg(optimal_phase))
            self.metrics['coherence'] = coherence
        
        # === DECODING ===
        
        # Decode at phase 0 (Reality A in Janus terms)
        decode_0 = self._attempt_decode(field, 0)
        self.decoded_images[0] = (decode_0 * 255).astype(np.uint8)
        
        # Decode at phase 90 (Reality B - orthogonal)
        decode_90 = self._attempt_decode(field, np.pi/2)
        self.decoded_images[90] = (decode_90 * 255).astype(np.uint8)
        
        # Decode at current search phase
        decode_current = self._attempt_decode(field, search_phase_rad)
        self.decoded_images['current'] = (decode_current * 255).astype(np.uint8)
        
        # === ANALYSIS ===
        
        # FFT magnitude view
        fft_mag = self._compute_fft_analysis(field)
        self.analysis_images['fft'] = cv2.applyColorMap(
            (fft_mag * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        # Phase coherence map
        coherence_map = np.abs(np.gradient(np.angle(field))[0]) + \
                       np.abs(np.gradient(np.angle(field))[1])
        coherence_map = 1.0 - np.clip(coherence_map / (2 * np.pi), 0, 1)
        self.analysis_images['coherence'] = cv2.applyColorMap(
            (coherence_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Sample points visualization
        sample_viz = np.zeros((h, w, 3), dtype=np.uint8)
        magnitude = np.abs(field)
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        sample_viz[:, :, 0] = (magnitude * 255).astype(np.uint8)
        
        # Draw hex points
        for px, py in scaled_hex:
            ix = int((px + 1) / 2 * (w - 1))
            iy = int((py + 1) / 2 * (h - 1))
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(sample_viz, (ix, iy), 3, (0, 255, 255), -1)
        
        # Draw corner points
        for px, py in corner_points:
            ix = int((px + 1) / 2 * (w - 1))
            iy = int((py + 1) / 2 * (h - 1))
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(sample_viz, (ix, iy), 2, (0, 255, 0), -1)
        
        # Draw lobe points
        for px, py in lobe_points:
            ix = int((px + 1) / 2 * (w - 1))
            iy = int((py + 1) / 2 * (h - 1))
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(sample_viz, (ix, iy), 2, (255, 0, 255), -1)
        
        self.analysis_images['samples'] = sample_viz
        
        # Lobe asymmetry
        self.metrics['asymmetry'] = self._compute_lobe_asymmetry(field)
        
        # === EIGENMODE (Dominant Pattern) ===
        try:
            # SVD to find dominant mode
            U, S, Vh = np.linalg.svd(np.abs(field), full_matrices=False)
            # Reconstruct with top eigenmode only
            eigenmode = np.outer(U[:, 0], Vh[0, :]) * S[0]
            eigenmode = eigenmode / (np.max(eigenmode) + 1e-6)
            self.analysis_images['eigenmode'] = cv2.applyColorMap(
                (eigenmode * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        except:
            self.analysis_images['eigenmode'] = np.zeros((h, w, 3), dtype=np.uint8)
        
        # === UPDATE DISPLAY ===
        self._update_display()
    
    def _update_display(self):
        """Create composite display image."""
        # Get decoded images
        d0 = self.decoded_images.get(0)
        d90 = self.decoded_images.get(90)
        dc = self.decoded_images.get('current')
        fft_view = self.analysis_images.get('fft')
        
        if d0 is None:
            self._init_display()
            return
        
        # Build composite
        h, w = d0.shape[:2]
        
        # Scale for display
        scale = max(1, 128 // h)
        dh, dw = h * scale, w * scale
        
        # Resize
        d0_s = cv2.resize(d0, (dw, dh), interpolation=cv2.INTER_NEAREST)
        d90_s = cv2.resize(d90, (dw, dh), interpolation=cv2.INTER_NEAREST)
        dc_s = cv2.resize(dc, (dw, dh), interpolation=cv2.INTER_NEAREST)
        
        # Convert to color
        if d0_s.ndim == 2:
            d0_c = cv2.applyColorMap(d0_s, cv2.COLORMAP_INFERNO)
            d90_c = cv2.applyColorMap(d90_s, cv2.COLORMAP_OCEAN)
            dc_c = cv2.applyColorMap(dc_s, cv2.COLORMAP_VIRIDIS)
        else:
            d0_c, d90_c, dc_c = d0_s, d90_s, dc_s
        
        # FFT view
        if fft_view is not None:
            fft_s = cv2.resize(fft_view, (dw, dh))
        else:
            fft_s = np.zeros((dh, dw, 3), dtype=np.uint8)
        
        # Composite: 2x2 grid + info panel
        panel_h = 50
        comp_w = dw * 2 + 10
        comp_h = dh * 2 + 20 + panel_h
        
        display = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
        
        # Place images
        display[panel_h:panel_h+dh, 0:dw] = d0_c
        display[panel_h:panel_h+dh, dw+10:dw*2+10] = d90_c
        display[panel_h+dh+10:panel_h+dh*2+10, 0:dw] = dc_c
        display[panel_h+dh+10:panel_h+dh*2+10, dw+10:dw*2+10] = fft_s
        
        # Labels
        cv2.putText(display, "HOLOGRAPHIC DECODER", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
        cv2.putText(display, f"Coherence: {self.metrics['coherence']:.3f}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
        cv2.putText(display, f"Optimal: {self.metrics['optimal_phase']:.1f}deg", (150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        cv2.putText(display, f"L/R Asym: {self.metrics['asymmetry']:.3f}", (280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 255), 1)
        
        # Image labels
        cv2.putText(display, "Phase 0", (5, panel_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(display, "Phase 90", (dw + 15, panel_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(display, "Current", (5, panel_h + dh + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(display, "FFT", (dw + 15, panel_h + dh + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        self.display_image = display
    
    # === OUTPUTS ===
    
    def get_output(self, port_name):
        # Decoded images
        if port_name == 'decoded_phase_0':
            img = self.decoded_images.get(0)
            return cv2.applyColorMap(img, cv2.COLORMAP_INFERNO) if img is not None else None
        elif port_name == 'decoded_phase_90':
            img = self.decoded_images.get(90)
            return cv2.applyColorMap(img, cv2.COLORMAP_OCEAN) if img is not None else None
        elif port_name == 'decoded_current':
            img = self.decoded_images.get('current')
            return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS) if img is not None else None
        
        # Analysis views
        elif port_name == 'fft_magnitude':
            return self.analysis_images.get('fft')
        elif port_name == 'sample_points':
            return self.analysis_images.get('samples')
        elif port_name == 'phase_coherence':
            return self.analysis_images.get('coherence')
        elif port_name == 'eigenmode_view':
            return self.analysis_images.get('eigenmode')
        
        # Sample vectors
        elif port_name == 'hex_samples':
            s = self.sample_vectors.get('hex')
            return np.abs(s) if s is not None else None
        elif port_name == 'corner_samples':
            s = self.sample_vectors.get('corner')
            return np.abs(s) if s is not None else None
        elif port_name == 'lobe_samples':
            s = self.sample_vectors.get('lobe')
            return np.abs(s) if s is not None else None
        
        # Signals
        elif port_name == 'coherence_score':
            return self.metrics.get('coherence', 0.0)
        elif port_name == 'optimal_phase':
            return self.metrics.get('optimal_phase', 0.0)
        elif port_name == 'lobe_asymmetry':
            return self.metrics.get('asymmetry', 0.0)
        
        return None
    
    def get_display_image(self):
        return self.display_image
    
    def close(self):
        pass