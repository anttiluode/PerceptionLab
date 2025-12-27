"""
Neural Field Perturbation Node
==============================

Converts EEG frequency bands (delta, theta, alpha, beta, gamma) into
a 2D spatial perturbation field suitable for the SHPF node.

The idea: Different frequency bands represent different spatial scales
of neural activity. Slow waves (delta) = large-scale coordination.
Fast waves (gamma) = local processing.

This node maps:
- Delta (0.5-4 Hz)  → Large, slow-moving blobs (global)
- Theta (4-8 Hz)    → Medium patterns (hippocampal-scale)
- Alpha (8-13 Hz)   → Posterior-dominant waves
- Beta (13-30 Hz)   → Smaller, faster patterns (motor/attention)
- Gamma (30-100 Hz) → Fine-grained local activity

The output is a 2D field that can perturb the SHPF's continuous dynamics,
letting real EEG rhythms drive the biological computation simulation.

Author: For Antti's PerceptionLab
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.special import jn

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    class QtGui:
        @staticmethod
        def QColor(*args): return None


class NeuralFieldPerturbNode(BaseNode):
    """
    Converts EEG band powers into a 2D spatial perturbation field.
    
    Each band creates patterns at different spatial scales,
    mimicking how different frequencies reflect different scales
    of neural organization.
    """
    
    NODE_CATEGORY = "Biological"
    NODE_TITLE = "Neural Field Perturb"
    NODE_COLOR = QtGui.QColor(100, 150, 180)  # Neural blue
    
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.field_size = 64  # Match SHPF default
        self.time = 0.0
        self.dt = 0.033  # ~30fps
        
        # Band weights (how much each band contributes)
        self.delta_weight = 1.0
        self.theta_weight = 1.0
        self.alpha_weight = 1.0
        self.beta_weight = 1.0
        self.gamma_weight = 1.0
        
        # Spatial scales for each band (sigma for gaussian, or mode number)
        # Delta = largest scale, Gamma = smallest
        self.delta_scale = 20.0   # Large blobs
        self.theta_scale = 12.0   # Medium
        self.alpha_scale = 8.0    # Medium-small
        self.beta_scale = 4.0     # Small
        self.gamma_scale = 2.0    # Fine
        
        # Temporal frequencies (how fast patterns move)
        self.delta_freq = 0.5     # Slow drift
        self.theta_freq = 2.0     # Theta rhythm
        self.alpha_freq = 5.0     # Alpha oscillation
        self.beta_freq = 10.0     # Beta
        self.gamma_freq = 20.0    # Fast gamma
        
        # Inputs
        self.inputs = {
            'delta': 'signal',      # Delta band power (0.5-4 Hz)
            'theta': 'signal',      # Theta band power (4-8 Hz)
            'alpha': 'signal',      # Alpha band power (8-13 Hz)
            'beta': 'signal',       # Beta band power (13-30 Hz)
            'gamma': 'signal',      # Gamma band power (30-100 Hz)
            'raw_spectrum': 'spectrum',  # Alternative: full spectrum input
        }
        
        # Outputs
        self.outputs = {
            'perturbation': 'image',     # 2D field for SHPF
            'field_energy': 'signal',    # Total field energy
            'band_balance': 'spectrum',  # 5-element showing band contributions
        }
        
        # State
        self.field = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        self.band_values = np.zeros(5, dtype=np.float32)  # delta, theta, alpha, beta, gamma
        
        # Precompute basis patterns for each band
        self._precompute_bases()
        
    def _precompute_bases(self):
        """Precompute spatial basis patterns for each frequency band"""
        size = self.field_size
        
        # Create coordinate grids
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        
        # Normalized coordinates
        x_norm = (x - cx) / (size / 2)
        y_norm = (y - cy) / (size / 2)
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)
        
        # Delta: Large-scale global pattern (low spatial frequency)
        # Using Bessel J0 for circular symmetry
        self.delta_base = np.cos(r * np.pi * 0.5).astype(np.float32)
        self.delta_base = gaussian_filter(self.delta_base, self.delta_scale / 4)
        
        # Theta: Hippocampal-like traveling waves
        # Asymmetric pattern that can "travel"
        self.theta_base = np.sin(x_norm * np.pi + y_norm * np.pi * 0.5).astype(np.float32)
        self.theta_base = gaussian_filter(self.theta_base, self.theta_scale / 4)
        
        # Alpha: Posterior-dominant, more structured
        # Multiple lobes like visual cortex organization
        self.alpha_base = (np.cos(x_norm * np.pi * 2) * np.cos(y_norm * np.pi * 2)).astype(np.float32)
        self.alpha_base = gaussian_filter(self.alpha_base, self.alpha_scale / 4)
        
        # Beta: Motor/attention - more distributed small patterns
        self.beta_base = np.sin(r * np.pi * 4 + theta * 2).astype(np.float32)
        self.beta_base = gaussian_filter(self.beta_base, self.beta_scale / 4)
        
        # Gamma: Fine-grained local - highest spatial frequency
        self.gamma_base = np.cos(x_norm * np.pi * 6) * np.cos(y_norm * np.pi * 6)
        self.gamma_base = self.gamma_base.astype(np.float32)
        self.gamma_base = gaussian_filter(self.gamma_base, self.gamma_scale / 4)
        
        # Normalize all bases
        for base in [self.delta_base, self.theta_base, self.alpha_base, 
                     self.beta_base, self.gamma_base]:
            base /= (np.abs(base).max() + 1e-9)
    
    def _create_band_field(self, base, amplitude, freq, phase_offset=0):
        """Create a time-varying field for one band"""
        # Temporal modulation
        temporal = np.sin(2 * np.pi * freq * self.time + phase_offset)
        
        # Spatial pattern modulated by amplitude and time
        field = base * amplitude * (0.5 + 0.5 * temporal)
        
        return field
    
    def step(self):
        """Generate the combined perturbation field from all bands"""
        
        # Get band inputs
        delta_in = self.get_blended_input('delta', 'sum')
        theta_in = self.get_blended_input('theta', 'sum')
        alpha_in = self.get_blended_input('alpha', 'sum')
        beta_in = self.get_blended_input('beta', 'sum')
        gamma_in = self.get_blended_input('gamma', 'sum')
        spectrum_in = self.get_blended_input('raw_spectrum', 'first')
        
        # If spectrum provided, extract bands from it
        if spectrum_in is not None and len(spectrum_in) >= 5:
            # Assume spectrum is ordered or use first 5 as bands
            if delta_in is None:
                delta_in = float(spectrum_in[0])
            if theta_in is None:
                theta_in = float(spectrum_in[1])
            if alpha_in is None:
                alpha_in = float(spectrum_in[2])
            if beta_in is None:
                beta_in = float(spectrum_in[3])
            if gamma_in is None:
                gamma_in = float(spectrum_in[4])
        
        # Default values if no input
        delta_val = float(delta_in) if delta_in is not None else 0.5
        theta_val = float(theta_in) if theta_in is not None else 0.3
        alpha_val = float(alpha_in) if alpha_in is not None else 0.4
        beta_val = float(beta_in) if beta_in is not None else 0.2
        gamma_val = float(gamma_in) if gamma_in is not None else 0.1
        
        # Normalize inputs to reasonable range
        delta_val = np.clip(delta_val, 0, 2)
        theta_val = np.clip(theta_val, 0, 2)
        alpha_val = np.clip(alpha_val, 0, 2)
        beta_val = np.clip(beta_val, 0, 2)
        gamma_val = np.clip(gamma_val, 0, 2)
        
        # Store for output
        self.band_values = np.array([delta_val, theta_val, alpha_val, 
                                      beta_val, gamma_val], dtype=np.float32)
        
        # Create time-varying fields for each band
        delta_field = self._create_band_field(
            self.delta_base, 
            delta_val * self.delta_weight,
            self.delta_freq,
            phase_offset=0
        )
        
        theta_field = self._create_band_field(
            self.theta_base,
            theta_val * self.theta_weight,
            self.theta_freq,
            phase_offset=np.pi/4
        )
        
        alpha_field = self._create_band_field(
            self.alpha_base,
            alpha_val * self.alpha_weight,
            self.alpha_freq,
            phase_offset=np.pi/2
        )
        
        beta_field = self._create_band_field(
            self.beta_base,
            beta_val * self.beta_weight,
            self.beta_freq,
            phase_offset=np.pi * 0.75
        )
        
        gamma_field = self._create_band_field(
            self.gamma_base,
            gamma_val * self.gamma_weight,
            self.gamma_freq,
            phase_offset=np.pi
        )
        
        # Combine all bands
        self.field = (delta_field + theta_field + alpha_field + 
                      beta_field + gamma_field)
        
        # Normalize to [-1, 1] range
        field_max = np.abs(self.field).max()
        if field_max > 1e-6:
            self.field = self.field / field_max
        
        # Add small noise for stochasticity
        self.field += np.random.randn(self.field_size, self.field_size).astype(np.float32) * 0.05
        
        # Advance time
        self.time += self.dt
    
    def get_output(self, port_name):
        if port_name == 'perturbation':
            # Return as image (will be converted appropriately)
            # Scale to 0-1 range for image output
            field_out = (self.field + 1) / 2  # [-1,1] -> [0,1]
            return field_out.astype(np.float32)
        elif port_name == 'field_energy':
            return float(np.sum(self.field**2))
        elif port_name == 'band_balance':
            return self.band_values
        return None
    
    def get_display_image(self):
        """Visualize the perturbation field and band contributions"""
        width = 300
        height = 250
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Main field visualization (top)
        field_norm = (self.field + 1) / 2  # Normalize to 0-1
        field_u8 = (field_norm * 255).astype(np.uint8)
        field_color = cv2.applyColorMap(field_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        field_big = cv2.resize(field_color, (150, 150))
        display[10:160, 10:160] = field_big
        
        # Band bars (right side)
        band_names = ['δ', 'θ', 'α', 'β', 'γ']
        band_colors = [
            (255, 100, 100),   # Delta - red
            (100, 255, 100),   # Theta - green  
            (100, 100, 255),   # Alpha - blue
            (255, 255, 100),   # Beta - yellow
            (255, 100, 255),   # Gamma - magenta
        ]
        
        bar_x = 170
        bar_w = 20
        max_bar_h = 120
        
        for i, (name, val, color) in enumerate(zip(band_names, self.band_values, band_colors)):
            x = bar_x + i * (bar_w + 5)
            bar_h = int(np.clip(val, 0, 2) / 2 * max_bar_h)
            
            # Bar
            cv2.rectangle(display, (x, 150 - bar_h), (x + bar_w, 150), color, -1)
            cv2.rectangle(display, (x, 30), (x + bar_w, 150), (80, 80, 80), 1)
            
            # Label
            cv2.putText(display, name, (x + 3, 165), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Value
            cv2.putText(display, f"{val:.1f}", (x, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Field energy
        energy = float(np.sum(self.field**2))
        cv2.putText(display, f"Field Energy: {energy:.2f}", (10, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Time
        cv2.putText(display, f"t = {self.time:.1f}s", (10, 205),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Title
        cv2.putText(display, "Neural Field Perturbation", (10, 235),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, width, height, width*3, 
                           QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Field Size", "field_size", self.field_size, None),
            ("Delta Weight", "delta_weight", self.delta_weight, None),
            ("Theta Weight", "theta_weight", self.theta_weight, None),
            ("Alpha Weight", "alpha_weight", self.alpha_weight, None),
            ("Beta Weight", "beta_weight", self.beta_weight, None),
            ("Gamma Weight", "gamma_weight", self.gamma_weight, None),
        ]
    
    def set_config_options(self, options):
        for key, value in options.items():
            if hasattr(self, key):
                old_val = getattr(self, key)
                setattr(self, key, type(old_val)(value))
        
        if 'field_size' in options:
            self._precompute_bases()