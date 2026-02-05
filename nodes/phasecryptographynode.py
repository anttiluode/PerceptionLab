"""
Phase Cryptography Node
=======================
Extracts "hidden" phase encodings from EEG holograms.

Theory:
The brain encodes information in phase relationships that are invisible
to magnitude-only analysis. By performing specific operations that
cancel magnitude while preserving phase structure, we can reveal
the "encrypted" binding patterns.

Methods:
1. ORTHOGONAL RESIDUAL: What survives when two bands cancel
2. PHASE SKELETON: Pure phase field with magnitude normalized
3. DIFFERENTIAL PHASE: How phase changes between time steps
4. CROSS-FREQUENCY LOCK: Where slow phase predicts fast amplitude

This node takes PhiHologram outputs and reveals the hidden structure.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy import signal
from scipy.stats import entropy

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


class PhaseCryptographyNode(BaseNode):
    """
    Extracts hidden phase encodings from holographic fields.
    Reveals the "encrypted" binding patterns in EEG.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Phase Cryptography"
    NODE_COLOR = QtGui.QColor(200, 50, 200)  # Purple - hidden/cryptic
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'field_a': 'complex_spectrum',    # First frequency band field (e.g., delta)
            'field_b': 'complex_spectrum',    # Second frequency band field (e.g., gamma)
            'hologram_a': 'image',            # Fallback: image input
            'hologram_b': 'image',            # Fallback: image input
            'decode_angle': 'signal',         # Viewing angle for orthogonal decode
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Core extractions
            'phase_skeleton': 'image',        # Pure phase (magnitude = 1)
            'orthogonal_residual': 'image',   # What survives cancellation
            'binding_map': 'image',           # Where phases align
            'conflict_map': 'image',          # Where phases oppose
            
            # Cross-frequency analysis
            'cfc_map': 'image',               # Cross-frequency coupling strength
            'phase_gradient': 'image',        # Spatial derivative of phase
            'vortex_map': 'image',            # Phase singularities (vortices)
            
            # Decoded views
            'decoded_0': 'image',             # View at 0°
            'decoded_45': 'image',            # View at 45° (null/between)
            'decoded_90': 'image',            # View at 90°
            'decoded_custom': 'image',        # View at custom angle
            
            # Metrics
            'binding_strength': 'signal',     # Overall phase alignment
            'conflict_strength': 'signal',    # Overall phase opposition
            'vortex_count': 'signal',         # Number of singularities
            'phase_entropy': 'signal',        # Complexity of phase field
            'diagonal_entropy': 'signal',     # Null-space diagonal metric
        }
        
        # === PARAMETERS ===
        self.resolution = 128
        self.decode_angle = 45.0
        
        # === STATE ===
        self._outputs = {}
        self._Z = None  # Combined complex field
        self._phase_a = None
        self._phase_b = None
        
        # History for temporal analysis
        self._history = []
        self._history_size = 10
        
    def _image_to_field(self, img):
        """Convert image to pseudo-complex field."""
        if img is None:
            return None
        if np.iscomplexobj(img):
            return img
        
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        
        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
            
        # Resize if needed
        if img.shape[0] != self.resolution:
            img = cv2.resize(img, (self.resolution, self.resolution))
        
        # Convert to complex (magnitude only, random phase)
        # Actually, preserve phase from FFT of image
        fft = fftshift(fft2(img))
        return fft
    
    def _normalize_field(self, field):
        """Resize complex field to standard resolution."""
        if field is None:
            return None
        
        if field.shape[0] != self.resolution:
            mag = cv2.resize(np.abs(field), (self.resolution, self.resolution))
            phase = cv2.resize(np.angle(field), (self.resolution, self.resolution))
            field = mag * np.exp(1j * phase)
        
        return field
    
    def _extract_phase_skeleton(self, field):
        """
        Extract pure phase structure with unit magnitude.
        This reveals the "shape" of the field without amplitude masking.
        """
        phase = np.angle(field)
        skeleton = np.exp(1j * phase)
        return skeleton
    
    def _compute_binding_map(self, phase_a, phase_b):
        """
        Compute where phases align (binding) vs oppose (conflict).
        Binding = cos(phase_diff) close to 1
        Conflict = cos(phase_diff) close to -1
        """
        phase_diff = phase_a - phase_b
        
        # Binding: phases aligned
        binding = np.cos(phase_diff)
        binding = (binding + 1) / 2  # Map to 0-1
        
        # Conflict: phases opposed
        conflict = -np.cos(phase_diff)
        conflict = (conflict + 1) / 2
        
        return binding, conflict
    
    def _compute_cfc(self, field_slow, field_fast):
        """
        Cross-frequency coupling: where slow phase predicts fast amplitude.
        High CFC = gamma amplitude modulated by delta phase.
        """
        phase_slow = np.angle(field_slow)
        amp_fast = np.abs(field_fast)
        
        # Normalize amplitude
        amp_fast = amp_fast / (amp_fast.max() + 1e-9)
        
        # CFC: correlation between phase and amplitude
        # Simplified: modulation index approximation
        # Bin by slow phase, measure amp variance
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        cfc_map = np.zeros_like(amp_fast)
        
        for i in range(n_bins):
            mask = (phase_slow >= phase_bins[i]) & (phase_slow < phase_bins[i+1])
            if mask.sum() > 0:
                local_amp = amp_fast[mask].mean()
                cfc_map[mask] = local_amp
        
        return cfc_map
    
    def _compute_phase_gradient(self, phase):
        """
        Spatial gradient of phase field.
        High gradient = rapid phase change = boundary/edge.
        """
        # Handle wrap-around at ±π
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)
        
        grad_cos_y, grad_cos_x = np.gradient(phase_cos)
        grad_sin_y, grad_sin_x = np.gradient(phase_sin)
        
        # Magnitude of gradient
        grad_mag = np.sqrt(grad_cos_x**2 + grad_cos_y**2 + 
                          grad_sin_x**2 + grad_sin_y**2)
        
        return grad_mag
    
    def _detect_vortices(self, phase):
        """
        Detect phase singularities (vortices).
        These are points where phase wraps around completely.
        """
        # Compute phase winding around each pixel
        phase_pad = np.pad(phase, 1, mode='wrap')
        
        vortex_map = np.zeros_like(phase)
        
        for i in range(1, phase_pad.shape[0] - 1):
            for j in range(1, phase_pad.shape[1] - 1):
                # Get phases around this pixel (clockwise)
                neighbors = [
                    phase_pad[i-1, j],   # top
                    phase_pad[i-1, j+1], # top-right
                    phase_pad[i, j+1],   # right
                    phase_pad[i+1, j+1], # bottom-right
                    phase_pad[i+1, j],   # bottom
                    phase_pad[i+1, j-1], # bottom-left
                    phase_pad[i, j-1],   # left
                    phase_pad[i-1, j-1], # top-left
                    phase_pad[i-1, j],   # back to top (close loop)
                ]
                
                # Compute total phase winding
                winding = 0
                for k in range(len(neighbors) - 1):
                    diff = neighbors[k+1] - neighbors[k]
                    # Wrap to [-π, π]
                    while diff > np.pi: diff -= 2*np.pi
                    while diff < -np.pi: diff += 2*np.pi
                    winding += diff
                
                # Vortex if winding ≈ ±2π
                vortex_map[i-1, j-1] = abs(winding) / (2 * np.pi)
        
        return vortex_map
    
    def _orthogonal_decode(self, Z, angle_deg):
        """
        Decode the combined field at a specific phase angle.
        0° = Reality A, 90° = Reality B, 45° = null/between
        """
        angle_rad = np.deg2rad(angle_deg)
        rotator = np.exp(-1j * angle_rad)
        rotated = Z * rotator
        return np.real(rotated)
    
    def _analyze_diagonal_entropy(self, field):
        """
        Compute diagonal entropy of FFT (the key metric from our proof).
        """
        fft_mag = np.abs(fftshift(fft2(field)))
        
        h, w = fft_mag.shape
        center_y, center_x = h // 2, w // 2
        
        Y, X = np.ogrid[:h, :w]
        Y = Y - center_y
        X = X - center_x
        
        R = np.sqrt(X**2 + Y**2)
        R[center_y, center_x] = 1
        
        theta = np.arctan2(Y, X)
        
        # Diagonal mask (45° and 135°)
        diag_mask = (np.abs(np.abs(theta) - np.pi/4) < np.pi/8) | \
                    (np.abs(np.abs(theta) - 3*np.pi/4) < np.pi/8)
        
        center_mask = R > 3
        diag_mask = diag_mask & center_mask
        
        diag_values = fft_mag[diag_mask]
        diag_values = diag_values / (diag_values.sum() + 1e-9)
        diag_values = diag_values[diag_values > 0]
        
        return entropy(diag_values)
    
    def step(self):
        """Main processing step."""
        # Get inputs
        field_a = self.get_blended_input('field_a', 'first')
        field_b = self.get_blended_input('field_b', 'first')
        img_a = self.get_blended_input('hologram_a', 'first')
        img_b = self.get_blended_input('hologram_b', 'first')
        angle_in = self.get_blended_input('decode_angle', 'first')
        
        if angle_in is not None:
            self.decode_angle = float(angle_in)
        
        # Convert images to fields if needed
        if field_a is None and img_a is not None:
            field_a = self._image_to_field(img_a)
        if field_b is None and img_b is not None:
            field_b = self._image_to_field(img_b)
        
        if field_a is None:
            return
        
        # Normalize
        field_a = self._normalize_field(field_a)
        if field_b is not None:
            field_b = self._normalize_field(field_b)
        else:
            # Single input: use its own FFT as second field
            field_b = fftshift(fft2(np.abs(field_a)))
        
        # Extract phases
        self._phase_a = np.angle(field_a)
        self._phase_b = np.angle(field_b)
        
        # === ORTHOGONAL ENCODING ===
        # Normalize magnitudes
        mag_a = np.abs(field_a)
        mag_b = np.abs(field_b)
        mag_a = mag_a / (mag_a.max() + 1e-9)
        mag_b = mag_b / (mag_b.max() + 1e-9)
        
        # Combine with phase preservation
        self._Z = mag_a * np.exp(1j * self._phase_a) + \
                  1j * mag_b * np.exp(1j * self._phase_b)
        
        # === PHASE SKELETON ===
        skeleton = self._extract_phase_skeleton(self._Z)
        skeleton_vis = (np.angle(skeleton) + np.pi) / (2 * np.pi)
        skeleton_u8 = (skeleton_vis * 255).astype(np.uint8)
        self._outputs['phase_skeleton'] = cv2.applyColorMap(skeleton_u8, cv2.COLORMAP_HSV)
        
        # === BINDING/CONFLICT MAPS ===
        binding, conflict = self._compute_binding_map(self._phase_a, self._phase_b)
        
        binding_u8 = (binding * 255).astype(np.uint8)
        self._outputs['binding_map'] = cv2.applyColorMap(binding_u8, cv2.COLORMAP_VIRIDIS)
        self._outputs['binding_strength'] = float(binding.mean())
        
        conflict_u8 = (conflict * 255).astype(np.uint8)
        self._outputs['conflict_map'] = cv2.applyColorMap(conflict_u8, cv2.COLORMAP_HOT)
        self._outputs['conflict_strength'] = float(conflict.mean())
        
        # === ORTHOGONAL RESIDUAL ===
        # What survives when A and B cancel (the null space content)
        residual = self._orthogonal_decode(self._Z, 45)
        residual = residual - residual.min()
        residual = residual / (residual.max() + 1e-9)
        residual_u8 = (residual * 255).astype(np.uint8)
        self._outputs['orthogonal_residual'] = cv2.applyColorMap(residual_u8, cv2.COLORMAP_PLASMA)
        
        # === CROSS-FREQUENCY COUPLING ===
        cfc = self._compute_cfc(field_a, field_b)
        cfc_u8 = (cfc * 255).astype(np.uint8)
        self._outputs['cfc_map'] = cv2.applyColorMap(cfc_u8, cv2.COLORMAP_INFERNO)
        
        # === PHASE GRADIENT ===
        combined_phase = np.angle(self._Z)
        gradient = self._compute_phase_gradient(combined_phase)
        gradient = gradient / (gradient.max() + 1e-9)
        gradient_u8 = (gradient * 255).astype(np.uint8)
        self._outputs['phase_gradient'] = cv2.applyColorMap(gradient_u8, cv2.COLORMAP_BONE)
        
        # === VORTEX DETECTION ===
        vortices = self._detect_vortices(combined_phase)
        vortex_count = np.sum(vortices > 0.5)
        self._outputs['vortex_count'] = float(vortex_count)
        
        vortex_u8 = (np.clip(vortices, 0, 1) * 255).astype(np.uint8)
        self._outputs['vortex_map'] = cv2.applyColorMap(vortex_u8, cv2.COLORMAP_MAGMA)
        
        # === DECODED VIEWS ===
        for angle, name in [(0, 'decoded_0'), (45, 'decoded_45'), 
                            (90, 'decoded_90'), (self.decode_angle, 'decoded_custom')]:
            decoded = self._orthogonal_decode(self._Z, angle)
            decoded = decoded - decoded.min()
            decoded = decoded / (decoded.max() + 1e-9)
            decoded_u8 = (decoded * 255).astype(np.uint8)
            self._outputs[name] = cv2.applyColorMap(decoded_u8, cv2.COLORMAP_TWILIGHT)
        
        # === PHASE ENTROPY ===
        phase_hist, _ = np.histogram(combined_phase, bins=36, range=(-np.pi, np.pi))
        phase_hist = phase_hist / phase_hist.sum()
        phase_hist = phase_hist[phase_hist > 0]
        self._outputs['phase_entropy'] = float(entropy(phase_hist))
        
        # === DIAGONAL ENTROPY ===
        self._outputs['diagonal_entropy'] = float(self._analyze_diagonal_entropy(residual))
        
    def get_output(self, port_name):
        """Return requested output."""
        return self._outputs.get(port_name, None)
    
    def get_display_image(self):
        """Create visualization for node face."""
        res = self.resolution
        half = res // 2
        
        # 2x2 grid: Phase Skeleton | Binding | Residual | CFC
        display = np.zeros((res, res, 3), dtype=np.uint8)
        
        def get_resized(key):
            img = self._outputs.get(key)
            if img is None:
                return np.zeros((half, half, 3), dtype=np.uint8)
            return cv2.resize(img, (half, half))
        
        display[:half, :half] = get_resized('phase_skeleton')
        display[:half, half:] = get_resized('binding_map')
        display[half:, :half] = get_resized('orthogonal_residual')
        display[half:, half:] = get_resized('cfc_map')
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "PHASE", (3, 12), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display, "BIND", (half + 3, 12), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display, "NULL", (3, half + 12), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display, "CFC", (half + 3, half + 12), font, 0.3, (255, 255, 255), 1)
        
        # Metrics
        bind = self._outputs.get('binding_strength', 0)
        diag_ent = self._outputs.get('diagonal_entropy', 0)
        cv2.putText(display, f"B:{bind:.2f} E:{diag_ent:.2f}", (3, res - 5), 
                   font, 0.25, (200, 200, 200), 1)
        
        display = np.ascontiguousarray(display)
        h, w = display.shape[:2]
        
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg.copy()
    
    def get_config_options(self):
        """Configuration options."""
        return [
            ("Resolution", "resolution", self.resolution, 'int'),
            ("Decode Angle (°)", "decode_angle", self.decode_angle, 'float'),
        ]
    
    def set_config_options(self, options):
        """Apply configuration."""
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("Phase Cryptography Node")
    print("=" * 50)
    print()
    print("This node extracts hidden phase encodings from EEG holograms.")
    print()
    print("OUTPUTS:")
    print("  phase_skeleton    - Pure phase structure (magnitude = 1)")
    print("  binding_map       - Where phases align (coupling)")
    print("  conflict_map      - Where phases oppose (decoupling)")
    print("  orthogonal_residual - The null-space content (45° view)")
    print("  cfc_map           - Cross-frequency coupling strength")
    print("  phase_gradient    - Spatial phase changes (edges)")
    print("  vortex_map        - Phase singularities (special points)")
    print("  decoded_*         - Views at specific angles")
    print()
    print("METRICS:")
    print("  binding_strength  - Overall phase alignment")
    print("  diagonal_entropy  - Null-space complexity (THE key metric)")
    print("  vortex_count      - Number of singularities")
    print()
    print("Connect: PhiHologram.delta_field → field_a")
    print("         PhiHologram.gamma_field → field_b")
