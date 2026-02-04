"""
Holographic Carrier Separator Node
==================================

Separates the CARRIER (electrode geometry eigenmode) from the SIGNAL 
(actual EEG phase information) in PhiHologram output.

The Problem:
When you create interference from 20 electrodes on a square canvas,
you get a characteristic "corner structure" that appears REGARDLESS
of what phases you feed in. This is the geometric eigenmode of the
electrode layout - the CARRIER WAVE.

The actual brain information MODULATES this carrier. To see what the
brain is doing, we need to separate or remove the carrier.

Methods:
1. BASELINE SUBTRACTION: Generate a "flat phase" hologram (all electrodes
   at phase 0) and subtract it from the real hologram
2. PHASE-ONLY ANALYSIS: The carrier affects magnitude more than phase
3. EIGENMODE DECOMPOSITION: Project out the dominant geometric modes
4. DIFFERENTIAL: Compare consecutive frames to see what changes (signal)
   vs what stays constant (carrier)

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift

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
        QtGui = MockQtGui()
    
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name, mode): 
            data = self.input_data.get(name, [None])
            return data[0] if data else None
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}


# Standard 10-20 electrode positions (same as PhiHologram)
ELECTRODE_POS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6), 'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'T7': (-0.9, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0, 0.0), 'C4': (0.4, 0.0), 'T8': (0.9, 0.0),
    'P7': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5), 'P4': (0.35, -0.5), 'P8': (0.7, -0.5),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85)
}


class HolographicCarrierSeparatorNode(BaseNode):
    """
    Separates electrode geometry eigenmode (carrier) from EEG signal.
    
    Theory:
    - CARRIER: The corner/quadrant structure from electrode layout geometry
    - SIGNAL: The actual brain-state information modulating the carrier
    
    The carrier is what you'd see if all electrodes had identical phase.
    The signal is the deviation from that baseline.
    """
    
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Carrier/Signal Separator"
    NODE_COLOR = QtGui.QColor(50, 180, 150)  # Teal - separation/analysis
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'hologram_field': 'complex_spectrum',  # From PhiHologram band_field
            'hologram_image': 'image',             # From PhiHologram band_hologram
            'spatial_k': 'signal',                 # Current k value (for baseline)
            'phase_rot': 'signal',                 # Current rotation (for baseline)
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Separated components
            'carrier_field': 'complex_spectrum',   # The geometric eigenmode
            'signal_field': 'complex_spectrum',    # The EEG information
            'carrier_image': 'image',              # Carrier visualization
            'signal_image': 'image',               # Signal visualization
            
            # Analysis outputs
            'phase_deviation': 'image',            # Where phase deviates from carrier
            'magnitude_ratio': 'image',            # Signal/Carrier magnitude ratio
            'corner_strength': 'image',            # How strong is corner structure
            
            # Metrics
            'carrier_power': 'signal',             # Total carrier energy
            'signal_power': 'signal',              # Total signal energy  
            'snr': 'signal',                       # Signal-to-noise (carrier) ratio
            'corner_index': 'signal',              # 0-1 how much is corner vs uniform
        }
        
        # === PARAMETERS ===
        self.resolution = 128
        self.current_k = 15.0
        self.current_rot = 0.0
        self.separation_mode = 0  # 0=baseline, 1=eigenmode, 2=differential
        
        # Temporal buffer for differential mode
        self.history_size = 10
        self.field_history = []
        
        # Cached outputs
        self._carrier_field = None
        self._signal_field = None
        self._carrier_image = None
        self._signal_image = None
        self._phase_deviation = None
        self._magnitude_ratio = None
        self._corner_strength = None
        
        self._carrier_power = 0.0
        self._signal_power = 0.0
        self._snr = 0.0
        self._corner_index = 0.0
        
        # Pre-compute distance maps for baseline generation
        self._init_geometry()
    
    def _init_geometry(self):
        """Pre-compute electrode distance maps."""
        res = self.resolution
        x = np.linspace(-1.5, 1.5, res).astype(np.float32)
        y = np.linspace(-1.5, 1.5, res).astype(np.float32)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.dist_maps = {}
        for name, (ex, ey) in ELECTRODE_POS.items():
            self.dist_maps[name] = np.sqrt((self.X - ex)**2 + (self.Y - ey)**2)
    
    def _generate_carrier_baseline(self, k, rot_deg):
        """
        Generate the CARRIER: what the hologram looks like with all
        electrodes at identical phase (phase=0).
        
        This reveals the pure geometric eigenmode of the electrode layout.
        """
        res = self.resolution
        field = np.zeros((res, res), dtype=np.complex64)
        rot_rad = np.deg2rad(rot_deg)
        
        for elec_name, dist in self.dist_maps.items():
            # All electrodes at phase 0 - only geometry matters
            theta = 0 - (k * dist) + rot_rad
            wave = np.cos(theta) + 1j * np.sin(theta)
            field += wave
        
        # Normalize
        field = field / len(self.dist_maps)
        
        return field
    
    def _generate_random_carrier(self, k, rot_deg, seed=42):
        """
        Generate carrier with random phases to see average structure.
        Multiple random samples averaged together reveal the eigenmode.
        """
        res = self.resolution
        np.random.seed(seed)
        
        n_samples = 20
        field_sum = np.zeros((res, res), dtype=np.complex64)
        
        for _ in range(n_samples):
            field = np.zeros((res, res), dtype=np.complex64)
            rot_rad = np.deg2rad(rot_deg)
            
            for elec_name, dist in self.dist_maps.items():
                # Random phase for each electrode
                phi = np.random.uniform(0, 2*np.pi)
                theta = phi - (k * dist) + rot_rad
                wave = np.cos(theta) + 1j * np.sin(theta)
                field += wave
            
            field = field / len(self.dist_maps)
            field_sum += np.abs(field)  # Sum magnitudes
        
        # Average magnitude (phase averages out)
        avg_magnitude = field_sum / n_samples
        
        # Return as complex with zero phase
        return avg_magnitude.astype(np.complex64)
    
    def _compute_corner_strength(self, field):
        """
        Compute how much energy is in the corners vs center.
        High corner_index = strong geometric eigenmode.
        """
        mag = np.abs(field)
        res = self.resolution
        
        # Define corner regions (outer 25%)
        quarter = res // 4
        corners = np.zeros_like(mag, dtype=bool)
        corners[:quarter, :quarter] = True
        corners[:quarter, -quarter:] = True
        corners[-quarter:, :quarter] = True
        corners[-quarter:, -quarter:] = True
        
        # Define center region (inner 50%)
        center = np.zeros_like(mag, dtype=bool)
        center[quarter:-quarter, quarter:-quarter] = True
        
        corner_energy = np.mean(mag[corners])
        center_energy = np.mean(mag[center])
        
        # Corner index: how much stronger are corners than center
        if center_energy > 0:
            ratio = corner_energy / center_energy
            # Normalize to 0-1 range (expect ratio 0.5-2.0 typically)
            corner_index = np.clip((ratio - 0.5) / 1.5, 0, 1)
        else:
            corner_index = 0.5
        
        # Create visualization
        corner_vis = np.zeros((res, res), dtype=np.float32)
        corner_vis[corners] = corner_energy
        corner_vis[center] = center_energy
        
        return corner_index, corner_vis
    
    def _separate_by_baseline(self, field, k, rot_deg):
        """
        Method 1: Baseline Subtraction
        
        Carrier = flat-phase hologram
        Signal = actual - carrier
        """
        carrier = self._generate_carrier_baseline(k, rot_deg)
        
        # Normalize carrier to same magnitude as input
        carrier_scale = np.mean(np.abs(field)) / (np.mean(np.abs(carrier)) + 1e-9)
        carrier = carrier * carrier_scale
        
        # Signal is the difference
        signal = field - carrier
        
        return carrier, signal
    
    def _separate_by_eigenmode(self, field):
        """
        Method 2: Eigenmode Decomposition
        
        Transform to frequency domain, identify dominant modes (carrier),
        and separate from the rest (signal).
        """
        # FFT of field
        F = fft2(field)
        F_shift = fftshift(F)
        
        # Magnitude spectrum
        mag = np.abs(F_shift)
        
        # Find dominant mode (DC + low frequencies = carrier)
        res = self.resolution
        center = res // 2
        
        # Create carrier mask: strong DC and immediate neighbors
        carrier_mask = np.zeros((res, res), dtype=np.float32)
        
        # DC component
        dc_radius = 5
        y, x = np.ogrid[:res, :res]
        dc_dist = np.sqrt((x - center)**2 + (y - center)**2)
        carrier_mask[dc_dist < dc_radius] = 1.0
        
        # Also grab the corner frequencies (the characteristic eigenmode)
        corner_radius = 10
        for cx, cy in [(0, 0), (0, res-1), (res-1, 0), (res-1, res-1)]:
            corner_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            carrier_mask[corner_dist < corner_radius] = 1.0
        
        # Smooth the mask
        carrier_mask = cv2.GaussianBlur(carrier_mask, (11, 11), 0)
        
        # Separate in frequency domain
        carrier_F = F_shift * carrier_mask
        signal_F = F_shift * (1 - carrier_mask)
        
        # Transform back
        carrier = ifft2(fftshift(carrier_F))
        signal = ifft2(fftshift(signal_F))
        
        return carrier.astype(np.complex64), signal.astype(np.complex64)
    
    def _separate_by_differential(self, field):
        """
        Method 3: Temporal Differential
        
        Carrier = what stays constant across time (geometric)
        Signal = what varies (brain dynamics)
        """
        # Add to history
        self.field_history.append(field.copy())
        if len(self.field_history) > self.history_size:
            self.field_history.pop(0)
        
        if len(self.field_history) < 3:
            # Not enough history, fall back to baseline
            return self._separate_by_baseline(field, self.current_k, self.current_rot)
        
        # Carrier = temporal mean magnitude (phase varies, magnitude is stable)
        mag_stack = np.array([np.abs(f) for f in self.field_history])
        carrier_mag = np.mean(mag_stack, axis=0)
        
        # Carrier phase = temporal circular mean
        phase_stack = np.array([np.angle(f) for f in self.field_history])
        carrier_phase = np.arctan2(
            np.mean(np.sin(phase_stack), axis=0),
            np.mean(np.cos(phase_stack), axis=0)
        )
        
        carrier = carrier_mag * np.exp(1j * carrier_phase)
        
        # Signal = deviation from carrier
        signal = field - carrier
        
        return carrier.astype(np.complex64), signal.astype(np.complex64)
    
    def step(self):
        """Main processing step."""
        # Get inputs
        field = self.get_blended_input('hologram_field', 'first')
        image = self.get_blended_input('hologram_image', 'first')
        k_in = self.get_blended_input('spatial_k', 'first')
        rot_in = self.get_blended_input('phase_rot', 'first')
        
        # Update parameters
        if k_in is not None:
            self.current_k = float(k_in)
        if rot_in is not None:
            self.current_rot = float(rot_in)
        
        # Need at least field or image
        if field is None and image is None:
            return
        
        # If only image, convert to pseudo-field
        if field is None and image is not None:
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if image.max() > 1:
                image /= 255.0
            # Resize
            image = cv2.resize(image, (self.resolution, self.resolution))
            # Convert to complex (magnitude only, zero phase)
            field = image.astype(np.complex64)
        
        # Ensure correct size
        if field.shape[0] != self.resolution:
            # Resize complex field
            mag = cv2.resize(np.abs(field), (self.resolution, self.resolution))
            phase = cv2.resize(np.angle(field), (self.resolution, self.resolution))
            field = mag * np.exp(1j * phase)
        
        # === SEPARATION ===
        if self.separation_mode == 0:
            carrier, signal = self._separate_by_baseline(field, self.current_k, self.current_rot)
        elif self.separation_mode == 1:
            carrier, signal = self._separate_by_eigenmode(field)
        else:
            carrier, signal = self._separate_by_differential(field)
        
        self._carrier_field = carrier
        self._signal_field = signal
        
        # === COMPUTE METRICS ===
        carrier_mag = np.abs(carrier)
        signal_mag = np.abs(signal)
        
        self._carrier_power = float(np.sum(carrier_mag**2))
        self._signal_power = float(np.sum(signal_mag**2))
        self._snr = self._signal_power / (self._carrier_power + 1e-9)
        
        self._corner_index, self._corner_strength = self._compute_corner_strength(field)
        
        # === PHASE DEVIATION ===
        # Where does the actual phase differ from carrier phase?
        carrier_phase = np.angle(carrier)
        field_phase = np.angle(field)
        phase_diff = np.abs(np.angle(np.exp(1j * (field_phase - carrier_phase))))
        self._phase_deviation = phase_diff / np.pi  # Normalize to 0-1
        
        # === MAGNITUDE RATIO ===
        # Signal strength relative to carrier
        self._magnitude_ratio = signal_mag / (carrier_mag + 1e-9)
        self._magnitude_ratio = np.clip(self._magnitude_ratio, 0, 2) / 2  # Normalize
        
        # === CREATE VISUALIZATIONS ===
        # Carrier image
        carrier_norm = carrier_mag / (carrier_mag.max() + 1e-9)
        carrier_u8 = (carrier_norm * 255).astype(np.uint8)
        self._carrier_image = cv2.applyColorMap(carrier_u8, cv2.COLORMAP_BONE)
        
        # Signal image
        signal_norm = signal_mag / (signal_mag.max() + 1e-9)
        signal_u8 = (signal_norm * 255).astype(np.uint8)
        self._signal_image = cv2.applyColorMap(signal_u8, cv2.COLORMAP_INFERNO)
    
    def get_output(self, port_name):
        """Return outputs."""
        outputs = {
            'carrier_field': self._carrier_field,
            'signal_field': self._signal_field,
            'carrier_image': self._carrier_image,
            'signal_image': self._signal_image,
            'phase_deviation': self._phase_deviation,
            'magnitude_ratio': self._magnitude_ratio,
            'corner_strength': self._corner_strength,
            'carrier_power': self._carrier_power,
            'signal_power': self._signal_power,
            'snr': self._snr,
            'corner_index': self._corner_index,
        }
        return outputs.get(port_name, None)
    
    def get_display_image(self):
        """Create visualization for node face."""
        res = self.resolution
        
        # Side by side: Carrier | Signal
        display = np.zeros((res, res * 2, 3), dtype=np.uint8)
        
        if self._carrier_image is not None:
            display[:, :res] = cv2.resize(self._carrier_image, (res, res))
        
        if self._signal_image is not None:
            display[:, res:] = cv2.resize(self._signal_image, (res, res))
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "CARRIER (geometry)", (5, 15), font, 0.35, (200, 200, 200), 1)
        cv2.putText(display, "SIGNAL (brain)", (res + 5, 15), font, 0.35, (255, 200, 100), 1)
        
        # Metrics
        mode_names = ["Baseline", "Eigenmode", "Differential"]
        mode_name = mode_names[self.separation_mode]
        cv2.putText(display, f"[{mode_name}]", (5, res - 25), font, 0.3, (150, 150, 150), 1)
        cv2.putText(display, f"Corner:{self._corner_index:.2f}", (5, res - 10), font, 0.3, (100, 200, 255), 1)
        cv2.putText(display, f"SNR:{self._snr:.3f}", (res + 5, res - 10), font, 0.3, (255, 200, 100), 1)
        
        display = np.ascontiguousarray(display)
        h, w = display.shape[:2]
        
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg
    
    def get_config_options(self):
        """Configuration options."""
        mode_options = [
            ("Baseline Subtraction", 0),
            ("Eigenmode Decomposition", 1),
            ("Temporal Differential", 2)
        ]
        return [
            ("Separation Mode", "separation_mode", self.separation_mode, mode_options),
            ("Resolution", "resolution", self.resolution, 'int'),
            ("History Size (for differential)", "history_size", self.history_size, 'int'),
        ]
    
    def set_config_options(self, options):
        """Apply configuration."""
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            if 'resolution' in options:
                self._init_geometry()


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("Holographic Carrier Separator")
    print("=" * 40)
    print()
    print("This node separates:")
    print("  CARRIER - The geometric eigenmode of electrode layout")
    print("           (the 'corner structure' you see)")
    print("  SIGNAL  - The actual EEG phase information")
    print("           (what the brain is doing)")
    print()
    print("Three separation methods:")
    print("  0: Baseline - subtract flat-phase hologram")
    print("  1: Eigenmode - frequency domain separation")
    print("  2: Differential - temporal averaging")
    print()
    
    # Quick test
    node = HolographicCarrierSeparatorNode()
    
    # Generate test carrier
    carrier = node._generate_carrier_baseline(k=15, rot_deg=0)
    print(f"Carrier baseline generated: {carrier.shape}")
    print(f"Carrier magnitude range: {np.abs(carrier).min():.3f} - {np.abs(carrier).max():.3f}")
    
    # Compute corner index
    corner_idx, corner_vis = node._compute_corner_strength(carrier)
    print(f"Corner index (pure carrier): {corner_idx:.3f}")
    print()
    print("High corner_index = strong geometric eigenmode (carrier)")
    print("Connect PhiHologram → this node to see the separation!")