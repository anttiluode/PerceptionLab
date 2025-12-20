"""
Complex Field Viewer - Holographic Interference of Eigenmode Phases
====================================================================

Takes complex-valued eigenmode activations and renders them as a 2D
interference field. This is the "hologram" of brain state.

WHAT YOU SEE:
- Bright regions: Modes constructively interfering (in-phase)
- Dark regions: Modes destructively interfering (anti-phase)
- Swirling patterns: Traveling waves through mode space
- Stable patterns: Standing waves (attractor states)

Each eigenmode is assigned a spatial pattern (approximating its topology).
The complex amplitude (magnitude + phase) determines how it contributes
to the total field. The result is a real-time holographic representation
of brain dynamics.

INPUTS:
- complex_modes: Complex spectrum from ModePhaseAnalyzerNode (purple)
- modulation: Optional signal to modulate field intensity

OUTPUTS:
- interference_field: The main visualization (image)
- magnitude_field: Just the amplitude (image)
- phase_field: Just the phase (image)  
- field_energy: Total field energy (signal)
- field_entropy: Spatial entropy of field (signal)
- peak_x, peak_y: Location of maximum interference (signals)
- complex_field_out: Raw complex field for further processing

Created: December 2025
"""

import numpy as np
import cv2
from collections import deque

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
            return None
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}


class ComplexFieldViewerNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Complex Field Viewer"
    NODE_COLOR = QtGui.QColor(150, 50, 200)  # Deep purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_modes': 'complex_spectrum',  # From ModePhaseAnalyzerNode
            'modulation': 'signal',               # Optional intensity mod
        }
        
        self.outputs = {
            'interference_field': 'image',
            'magnitude_field': 'image',
            'phase_field': 'image',
            'field_energy': 'signal',
            'field_entropy': 'signal',
            'peak_x': 'signal',
            'peak_y': 'signal',
            'vortex_count': 'signal',
            'complex_field_out': 'complex_spectrum',
        }
        
        # Config
        self.field_size = 200
        self.n_modes = 10
        self.colormap = 'twilight'  # Good for phase
        self.display_mode = 'interference'  # 'interference', 'magnitude', 'phase'
        self.spatial_scale = 1.0
        self.temporal_smoothing = 0.3
        
        # Precompute spatial basis patterns
        self._init_spatial_basis()
        
        # State
        self.complex_field = None
        self.prev_field = None
        self.interference_image = None
        self.magnitude_image = None
        self.phase_image = None
        
        # Metrics
        self._energy = 0.0
        self._entropy = 0.0
        self._peak_x = 0.5
        self._peak_y = 0.5
        self._vortex_count = 0
        
        self.frame_count = 0
        
    def _init_spatial_basis(self):
        """
        Create spatial basis patterns for each eigenmode.
        These approximate the topology of graph Laplacian eigenmodes.
        
        Low modes = large-scale, smooth patterns (global)
        High modes = fine-scale, complex patterns (local)
        """
        size = self.field_size
        self.spatial_basis = []
        
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Convert to polar for some patterns
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        for i in range(self.n_modes):
            freq = (i + 1) * 0.4 * self.spatial_scale
            
            # Create diverse spatial patterns mimicking eigenmode topology
            if i == 0:
                # Mode 1: Global, smooth gradient (L-R hemispheric)
                pattern = np.cos(freq * X * 0.5)
            elif i == 1:
                # Mode 2: Superior-Inferior gradient
                pattern = np.cos(freq * Y * 0.5)
            elif i == 2:
                # Mode 3: Anterior-Posterior
                pattern = np.cos(freq * (X + Y) / np.sqrt(2) * 0.7)
            elif i == 3:
                # Mode 4: Radial (center-surround)
                pattern = np.cos(freq * R * 0.6)
            elif i == 4:
                # Mode 5: Angular (rotational)
                pattern = np.cos(2 * Theta)
            elif i == 5:
                # Mode 6: Checkerboard-like
                pattern = np.cos(freq * X) * np.cos(freq * Y)
            elif i == 6:
                # Mode 7: Higher frequency radial
                pattern = np.cos(freq * R)
            elif i == 7:
                # Mode 8: Spiral
                pattern = np.cos(freq * R + 2 * Theta)
            elif i == 8:
                # Mode 9: Fine grid
                pattern = np.cos(freq * X * 1.5) + np.cos(freq * Y * 1.5)
            else:
                # Mode 10: Complex interference
                pattern = np.cos(freq * X * 2) * np.cos(freq * Y * 1.5) + np.cos(freq * R)
            
            # Normalize
            pattern = pattern / (np.max(np.abs(pattern)) + 1e-6)
            self.spatial_basis.append(pattern)
    
    def step(self):
        self.frame_count += 1
        
        # Get complex modes
        complex_modes = self.get_blended_input('complex_modes', 'mean')
        
        if complex_modes is None:
            complex_modes = np.zeros(self.n_modes, dtype=np.complex64)
        else:
            complex_modes = np.array(complex_modes).flatten()
            if not np.iscomplexobj(complex_modes):
                complex_modes = complex_modes.astype(np.complex64)
            if len(complex_modes) < self.n_modes:
                complex_modes = np.pad(complex_modes, (0, self.n_modes - len(complex_modes)))
            elif len(complex_modes) > self.n_modes:
                complex_modes = complex_modes[:self.n_modes]
        
        # Get modulation
        mod = self.get_blended_input('modulation', 'sum')
        if mod is None:
            mod = 1.0
        else:
            mod = 1.0 + float(mod) * 0.5
        
        # Build complex field by superposing all modes
        field = np.zeros((self.field_size, self.field_size), dtype=np.complex64)
        
        for i in range(self.n_modes):
            # Each mode contributes its spatial pattern * complex amplitude
            field += complex_modes[i] * self.spatial_basis[i] * mod
        
        # Temporal smoothing for stability
        if self.prev_field is not None:
            field = self.temporal_smoothing * self.prev_field + (1 - self.temporal_smoothing) * field
        self.prev_field = field.copy()
        self.complex_field = field
        
        # Compute metrics
        self._compute_metrics(field)
        
        # Render visualizations
        if self.frame_count % 2 == 0:
            self._render_interference(field)
            self._render_magnitude(field)
            self._render_phase(field)
    
    def _compute_metrics(self, field):
        """Compute field statistics"""
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # Energy
        self._energy = float(np.sum(magnitude**2))
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        self._peak_y = peak_idx[0] / self.field_size
        self._peak_x = peak_idx[1] / self.field_size
        
        # Entropy (spatial distribution)
        mag_norm = magnitude / (np.sum(magnitude) + 1e-10)
        mag_flat = mag_norm.flatten()
        mag_flat = mag_flat[mag_flat > 1e-10]
        self._entropy = float(-np.sum(mag_flat * np.log(mag_flat)))
        
        # Vortex count (phase singularities)
        # Simple approximation: count rapid phase changes
        phase_dx = np.diff(phase, axis=1)
        phase_dy = np.diff(phase, axis=0)
        # Wrap phase differences
        phase_dx = np.mod(phase_dx + np.pi, 2*np.pi) - np.pi
        phase_dy = np.mod(phase_dy + np.pi, 2*np.pi) - np.pi
        # Count large jumps as vortex indicators
        self._vortex_count = int(np.sum(np.abs(phase_dx) > 2.5) + np.sum(np.abs(phase_dy) > 2.5))
    
    def _render_interference(self, field):
        """
        Render the interference pattern.
        Shows REAL part - this is what you'd see on a screen.
        """
        size = self.field_size
        
        # Real part shows interference fringes
        real_part = np.real(field)
        magnitude = np.abs(field)
        
        # Normalize
        real_max = np.max(np.abs(real_part)) + 1e-6
        real_norm = (real_part / real_max + 1) / 2  # Map to 0-1
        
        mag_max = np.max(magnitude) + 1e-6
        mag_norm = magnitude / mag_max
        
        # Create RGB image
        # Brightness from magnitude, hue from real part sign
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Option 1: Twilight-like coloring
        # Positive real = warm colors, negative = cool colors
        for y in range(size):
            for x in range(size):
                r = real_norm[y, x]
                m = mag_norm[y, x]
                
                # Warm-cool based on real part sign
                if real_part[y, x] > 0:
                    # Warm: yellow to red
                    img[y, x, 2] = int(255 * m * r)           # Red
                    img[y, x, 1] = int(180 * m * (1-r) * 0.5) # Green
                    img[y, x, 0] = int(50 * m * (1-r))        # Blue
                else:
                    # Cool: cyan to blue
                    img[y, x, 2] = int(50 * m * r)            # Red
                    img[y, x, 1] = int(180 * m * r * 0.5)     # Green  
                    img[y, x, 0] = int(255 * m * (1-r))       # Blue
        
        # Add interference fringes overlay
        fringes = ((real_norm * 10) % 1.0 * 30).astype(np.uint8)
        img[:, :, 1] = np.clip(img[:, :, 1] + fringes, 0, 255)
        
        # Mark peak location
        peak_x = int(self._peak_x * size)
        peak_y = int(self._peak_y * size)
        cv2.circle(img, (peak_x, peak_y), 5, (255, 255, 255), 1)
        
        # Title and stats
        cv2.putText(img, "Interference", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"E:{self._energy:.0f}", (5, size-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(img, f"V:{self._vortex_count}", (5, size-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        self.interference_image = img
    
    def _render_magnitude(self, field):
        """Render magnitude only"""
        magnitude = np.abs(field)
        mag_max = np.max(magnitude) + 1e-6
        mag_norm = (magnitude / mag_max * 255).astype(np.uint8)
        
        img = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
        
        cv2.putText(img, "Magnitude", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.magnitude_image = img
    
    def _render_phase(self, field):
        """Render phase only"""
        phase = np.angle(field)
        phase_norm = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        
        img = cv2.applyColorMap(phase_norm, cv2.COLORMAP_HSV)
        
        cv2.putText(img, "Phase", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.phase_image = img
    
    def get_output(self, port_name):
        if port_name == 'interference_field':
            return self.interference_image
        elif port_name == 'magnitude_field':
            return self.magnitude_image
        elif port_name == 'phase_field':
            return self.phase_image
        elif port_name == 'field_energy':
            return self._energy
        elif port_name == 'field_entropy':
            return self._entropy
        elif port_name == 'peak_x':
            return self._peak_x
        elif port_name == 'peak_y':
            return self._peak_y
        elif port_name == 'vortex_count':
            return float(self._vortex_count)
        elif port_name == 'complex_field_out':
            if self.complex_field is not None:
                # Downsample for output
                return self.complex_field[::4, ::4].flatten().astype(np.complex64)
            return None
        return None
    
    def get_display_image(self):
        if self.interference_image is not None:
            img = np.ascontiguousarray(self.interference_image)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        w, h = 100, 100
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "Waiting...", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Field Size", "field_size", self.field_size, None),
            ("Spatial Scale", "spatial_scale", self.spatial_scale, None),
            ("Temporal Smoothing", "temporal_smoothing", self.temporal_smoothing, None),
        ]