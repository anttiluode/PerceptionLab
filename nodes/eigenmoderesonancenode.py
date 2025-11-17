"""
Eigenmode Resonance Node v3 - FIXED VERSION
--------------------------------------------
Takes EEG frequency bands and determines which brain eigenmodes are active

FIXES in v3:
- 100x stronger normalization (was killing signal)
- Temporal stability resonance (instead of spatial structure)
- Contrast enhancement (makes variations visible)
- Configurable sensitivity

Theory:
1. Different EEG frequencies correspond to different eigenmode numbers
2. Active eigenmodes create spatial activation patterns (lobes)
3. Resonance = temporal stability of eigenmode pattern
4. Output shows which brain regions should be active given the EEG
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.special import jn, jn_zeros
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class EigenmodeResonanceNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(80, 60, 140)  # Deep purple - consciousness analysis
    
    def __init__(self, aspect_ratio=2.0, resolution=256, resonance_threshold=0.3, 
                 sensitivity=1.0, contrast_boost=2.0):
        super().__init__()
        self.node_title = "EEG Eigenmode Analyzer v3"
        
        self.inputs = {
            'delta': 'signal',   # 1-4 Hz
            'theta': 'signal',   # 4-8 Hz
            'alpha': 'signal',   # 8-13 Hz
            'beta': 'signal',    # 13-30 Hz
            'gamma': 'signal',   # 30-45 Hz
            'raw_signal': 'signal',  # Optional total power
        }
        
        self.outputs = {
            'eigenmode_activation': 'image',  # Which modes are active
            'lobe_activation_map': 'image',   # Spatial activation pattern
            'resonance_score': 'signal',      # How stable (0-1)
            'dominant_mode_n': 'signal',      # Which radial mode is strongest
            'dominant_mode_m': 'signal',      # Which angular mode is strongest
            'total_activation': 'signal',     # Overall brain activity
        }
        
        # Configuration
        self.aspect_ratio = float(aspect_ratio)
        self.resolution = int(resolution)
        self.resonance_threshold = float(resonance_threshold)
        self.sensitivity = float(sensitivity)  # NEW: adjustable sensitivity
        self.contrast_boost = float(contrast_boost)  # NEW: contrast enhancement
        
        # Eigenmode-frequency mapping
        self.frequency_to_modes = {
            'delta': [(1, 0), (1, 1)],           # Slow, global modes
            'theta': [(2, 0), (2, 1)],           # Low-order modes
            'alpha': [(2, 2), (3, 1)],           # Classic "resting state" modes
            'beta': [(3, 2), (4, 1), (3, 3)],   # Active processing modes
            'gamma': [(4, 2), (5, 1), (4, 3)],  # High-frequency, local modes
        }
        
        # State
        self.eigenmode_activation = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.lobe_activation_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.previous_activation_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)  # NEW
        self.resonance_score = 0.0
        self.dominant_mode_n = 0
        self.dominant_mode_m = 0
        self.total_activation = 0.0
        self.eeg_bands = {'delta': 0.0, 'theta': 0.0, 'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0}
        
        # Precompute eigenmodes
        self.eigenmode_cache = {}
        self.mask = None
        self._precompute_eigenmodes()
        
    def _create_ellipsoidal_mask(self):
        """Create brain-shaped domain"""
        h, w = self.resolution, self.resolution
        cy, cx = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        
        a = cx * 0.9
        b = cy * 0.9 / self.aspect_ratio
        
        mask = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) <= 1.0
        
        return mask.astype(np.float32), a, b
    
    def _compute_eigenmode(self, n, m, a, b):
        """Compute specific (n,m) eigenmode on elliptical domain"""
        h, w = self.resolution, self.resolution
        cy, cx = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        x_norm = (x - cx) / a
        y_norm = (y - cy) / b
        
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)
        
        # Bessel function eigenmode
        if m == 0:
            zeros = jn_zeros(m, n + 1)
            k_nm = zeros[min(n, len(zeros) - 1)]
            radial = jn(m, k_nm * r)
            angular = np.ones_like(theta)
        else:
            zeros = jn_zeros(m, max(1, n))
            k_nm = zeros[min(n, len(zeros) - 1)]
            radial = jn(m, k_nm * r)
            angular = np.cos(m * theta)
        
        eigenmode = radial * angular
        
        # Normalize
        if eigenmode.max() > 0:
            eigenmode = eigenmode / eigenmode.max()
        
        return eigenmode
    
    def _precompute_eigenmodes(self):
        """Precompute all eigenmodes we'll need"""
        self.mask, a, b = self._create_ellipsoidal_mask()
        
        # Compute all modes referenced in frequency_to_modes
        for band, mode_list in self.frequency_to_modes.items():
            for n, m in mode_list:
                key = (n, m)
                if key not in self.eigenmode_cache:
                    eigenmode = self._compute_eigenmode(n, m, a, b)
                    eigenmode = eigenmode * self.mask
                    self.eigenmode_cache[key] = eigenmode
    
    def _compute_resonance(self, activation_map):
        """
        NEW RESONANCE METRIC: Temporal stability + single-mode dominance
        
        Old metric measured spatial structure (always high for eigenmodes)
        New metric measures:
        1. How stable the pattern is over time (temporal coherence)
        2. How much one mode dominates (vs mixed/noisy state)
        """
        # Method 1: Temporal stability (70%)
        # How similar is current map to previous frame?
        if self.previous_activation_map.max() > 0:
            # Normalize both to compare shape, not amplitude
            curr_norm = activation_map / (np.max(activation_map) + 1e-9)
            prev_norm = self.previous_activation_map / (np.max(self.previous_activation_map) + 1e-9)
            
            # Similarity = 1 - difference
            difference = np.mean(np.abs(curr_norm - prev_norm))
            temporal_stability = 1.0 - np.clip(difference, 0, 1)
        else:
            temporal_stability = 0.5  # Neutral on first frame
        
        # Method 2: Pattern strength (30%)
        # How strong is the activation vs noise?
        if activation_map.max() > 0:
            # Ratio of peak to mean (higher = more focused pattern)
            peak_to_mean = activation_map.max() / (np.mean(activation_map) + 1e-9)
            pattern_strength = np.clip(peak_to_mean / 10.0, 0, 1)  # Normalize
        else:
            pattern_strength = 0.0
        
        # Combine metrics
        resonance = (temporal_stability * 0.7 + pattern_strength * 0.3)
        resonance = np.clip(resonance, 0, 1)
        
        return resonance
    
    def step(self):
        # Get EEG inputs
        eeg_bands = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            value = self.get_blended_input(band, 'sum')
            eeg_bands[band] = value if value is not None else 0.0
        
        # Store for display debugging
        self.eeg_bands = eeg_bands
        
        # Initialize activation map
        activation_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        mode_activations = {}  # Track which modes are how active
        
        # For each frequency band, activate corresponding eigenmodes
        for band, power in eeg_bands.items():
            if power > 0.00001:  # Very low threshold to catch tiny signals
                mode_list = self.frequency_to_modes[band]
                
                for n, m in mode_list:
                    key = (n, m)
                    eigenmode = self.eigenmode_cache[key]
                    
                    # FIXED NORMALIZATION - 100x stronger!
                    # With 1B boost giving 0.42, this gives: 0.42 * 1.0 * sensitivity = 0.42
                    # Which is MUCH better than the old 0.42 * 0.01 = 0.004!
                    normalized_power = np.clip(power * 1.0 * self.sensitivity, 0, 2.0)
                    
                    activation_map += eigenmode * normalized_power
                    
                    # Track mode activation
                    if key not in mode_activations:
                        mode_activations[key] = 0.0
                    mode_activations[key] += normalized_power
        
        # Apply mask
        activation_map = activation_map * self.mask
        
        # CONTRAST ENHANCEMENT - makes variations visible!
        if activation_map.max() > 0:
            # Subtract minimum to remove baseline
            activation_map = activation_map - activation_map.min()
            
            # Apply contrast boost (power function)
            activation_map = np.power(activation_map / activation_map.max(), 1.0 / self.contrast_boost)
            
            # Renormalize
            activation_map = activation_map / (activation_map.max() + 1e-9)
        
        # Clip to ensure positive values (eigenmodes can be negative)
        activation_map = np.clip(activation_map, 0, 1)
        
        # Smooth activation (neural activity spreads)
        activation_map = ndimage.gaussian_filter(activation_map, sigma=2.0)
        
        # Store lobe activation map
        self.lobe_activation_map = activation_map
        
        # Find dominant mode
        if mode_activations:
            dominant_key = max(mode_activations, key=mode_activations.get)
            self.dominant_mode_n = dominant_key[0]
            self.dominant_mode_m = dominant_key[1]
        else:
            self.dominant_mode_n = 0
            self.dominant_mode_m = 0
        
        # Compute resonance score (NEW: temporal stability)
        self.resonance_score = self._compute_resonance(activation_map)
        
        # Store current as previous for next frame
        self.previous_activation_map = activation_map.copy()
        
        # Total activation (use absolute value to avoid negatives)
        self.total_activation = np.mean(np.abs(activation_map))
        
        # Create eigenmode activation visualization
        self.eigenmode_activation = self._create_mode_activation_viz(mode_activations)
        
    def _create_mode_activation_viz(self, mode_activations):
        """Create visualization showing which modes are active"""
        # Create a grid showing all possible modes
        max_n = 5
        max_m = 4
        
        cell_size = self.resolution // max(max_n, max_m)
        viz = np.zeros((max_n * cell_size, max_m * cell_size), dtype=np.float32)
        
        for (n, m), activation in mode_activations.items():
            if n < max_n and m < max_m:
                # Place activation value in grid
                y_start = n * cell_size
                x_start = m * cell_size
                
                # Fill cell with activation level
                viz[y_start:y_start+cell_size, x_start:x_start+cell_size] = activation
        
        return viz
    
    def get_output(self, port_name):
        if port_name == 'eigenmode_activation':
            return self.eigenmode_activation
        elif port_name == 'lobe_activation_map':
            return self.lobe_activation_map
        elif port_name == 'resonance_score':
            return self.resonance_score
        elif port_name == 'dominant_mode_n':
            return float(self.dominant_mode_n)
        elif port_name == 'dominant_mode_m':
            return float(self.dominant_mode_m)
        elif port_name == 'total_activation':
            return self.total_activation
        return None
    
    def get_display_image(self):
        display_w = 512
        display_h = 512
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        quad_size = display_w // 2
        
        # Top left: Lobe activation map (main output)
        lobe_u8 = (np.clip(self.lobe_activation_map, 0, 1) * 255).astype(np.uint8)
        lobe_color = cv2.applyColorMap(lobe_u8, cv2.COLORMAP_HOT)
        lobe_resized = cv2.resize(lobe_color, (quad_size, quad_size))
        display[:quad_size, :quad_size] = lobe_resized
        
        # Top right: Eigenmode activation grid
        if self.eigenmode_activation.max() > 0:
            mode_u8 = (self.eigenmode_activation * 255 / self.eigenmode_activation.max()).astype(np.uint8)
        else:
            mode_u8 = np.zeros_like(self.eigenmode_activation, dtype=np.uint8)
        mode_color = cv2.applyColorMap(mode_u8, cv2.COLORMAP_VIRIDIS)
        mode_resized = cv2.resize(mode_color, (quad_size, quad_size))
        display[:quad_size, quad_size:] = mode_resized
        
        # Bottom left: Dominant mode visualization
        if self.dominant_mode_n > 0 or self.dominant_mode_m > 0:
            key = (self.dominant_mode_n, self.dominant_mode_m)
            if key in self.eigenmode_cache:
                dominant = self.eigenmode_cache[key]
                # Clip to valid range before converting to uint8
                dominant_u8 = (np.clip((dominant + 1) * 127, 0, 255)).astype(np.uint8)
                dominant_color = cv2.applyColorMap(dominant_u8, cv2.COLORMAP_TWILIGHT)
                dominant_resized = cv2.resize(dominant_color, (quad_size, quad_size))
                display[quad_size:, :quad_size] = dominant_resized
        
        # Bottom right: Resonance indicator
        resonance_viz = np.zeros((quad_size, quad_size, 3), dtype=np.uint8)
        
        # Draw resonance meter
        bar_height = int(np.clip(self.resonance_score, 0, 1) * quad_size)
        resonance_viz[-bar_height:, :] = [0, 255, 0] if self.resonance_score > self.resonance_threshold else [255, 100, 0]
        
        # Add activation level as background (clip to valid uint8 range)
        activation_level = int(np.clip(self.total_activation * 255, 0, 255))
        resonance_viz[:, :, 2] = activation_level  # Blue channel shows total activation
        
        display[quad_size:, quad_size:] = resonance_viz
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'LOBE ACTIVATION', 
                   (10, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'MODE GRID', 
                   (quad_size + 10, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f'DOMINANT (n={self.dominant_mode_n},m={self.dominant_mode_m})', 
                   (10, quad_size + 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f'RESONANCE: {self.resonance_score:.3f}', 
                   (quad_size + 10, quad_size + 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Bottom info
        info_text = f'Total Act={self.total_activation:.3f} | Coherent: {"YES" if self.resonance_score > self.resonance_threshold else "NO"}'
        cv2.putText(display, info_text, 
                   (10, display_h - 30), font, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Debug: Show actual incoming values
        debug_text = f'IN: D={self.eeg_bands.get("delta", 0):.2f} T={self.eeg_bands.get("theta", 0):.2f} A={self.eeg_bands.get("alpha", 0):.2f} B={self.eeg_bands.get("beta", 0):.2f} G={self.eeg_bands.get("gamma", 0):.2f}'
        cv2.putText(display, debug_text,
                   (10, display_h - 10), font, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Aspect Ratio", "aspect_ratio", self.aspect_ratio, None),
            ("Resolution", "resolution", self.resolution, None),
            ("Resonance Threshold", "resonance_threshold", self.resonance_threshold, None),
            ("Sensitivity (0.1-10)", "sensitivity", self.sensitivity, None),
            ("Contrast Boost (1-5)", "contrast_boost", self.contrast_boost, None),
        ]