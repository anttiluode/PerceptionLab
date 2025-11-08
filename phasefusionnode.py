"""
Phase Fusion Field Node - Merges two signals through quantum field dynamics
Creates coherent phase-locked oscillations from independent inputs via instanton-mediated coupling.
Place this file in the 'nodes' folder as 'phasefusionnode.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import fft, ifft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: PhaseFusionNode requires scipy")

class PhaseFusionNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(120, 80, 200)  # Purple for quantum coupling
    
    def __init__(self, field_size=256, coupling_strength=0.01):
        super().__init__()
        self.node_title = "Phase Fusion Field"
        
        self.inputs = {
            'signal_a': 'signal',      # First signal to fuse
            'signal_b': 'signal',      # Second signal to fuse
            'coupling': 'signal',      # Control fusion strength
            'damping': 'signal'        # Control field dissipation
        }
        
        self.outputs = {
            'fused_output': 'signal',     # Phase-locked merged signal
            'coherence': 'signal',        # Phase coherence measure
            'field_image': 'image',       # Field amplitude visualization
            'phase_diff': 'signal'        # Phase difference between inputs
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Phase Fusion (No SciPy!)"
            return
        
        self.field_size = int(field_size)
        self.alpha = float(coupling_strength)  # Non-linear coupling (instanton strength)
        
        # Complex field state (instantons live here)
        self.field = np.zeros(self.field_size, dtype=np.complex128)
        self.field_prev = np.zeros_like(self.field)
        
        # Time evolution parameters
        self.dt = 0.01
        self.damping = 0.98
        
        # Frequency space (for fast Laplacian)
        k = fftfreq(self.field_size, 1.0) * 2 * np.pi
        self.k2 = k**2
        
        # Injection points for the two signals
        self.inject_a_pos = self.field_size // 4
        self.inject_b_pos = 3 * self.field_size // 4
        
        # Instanton tracking (peaks in the field)
        self.instantons = []
        
    def inject_signals(self, signal_a, signal_b, coupling_strength):
        """
        Inject two signals at different positions in the field.
        They will create localized excitations (instantons) that interact.
        """
        # Scale signals for field injection
        amp_a = signal_a * 0.5
        amp_b = signal_b * 0.5
        
        # Create complex injection (amplitude + phase)
        # The imaginary part allows phase information to propagate
        inject_a = amp_a * (1 + 1j)
        inject_b = amp_b * (1 + 1j)
        
        # Apply coupling strength
        inject_a *= coupling_strength
        inject_b *= coupling_strength
        
        # Inject at specified positions with Gaussian spread
        spread = 10
        x = np.arange(self.field_size)
        
        gaussian_a = np.exp(-((x - self.inject_a_pos)**2) / (2 * spread**2))
        gaussian_b = np.exp(-((x - self.inject_b_pos)**2) / (2 * spread**2))
        
        self.field += inject_a * gaussian_a
        self.field += inject_b * gaussian_b
    
    def evolve_field(self):
        """
        Evolve the field using a non-linear wave equation.
        The instanton dynamics come from the non-linear term that depends on field intensity.
        """
        # Transform to frequency space for fast Laplacian
        F = fft(self.field)
        laplacian = ifft(-self.k2 * F)
        
        # Non-linear term (instanton coupling)
        # This creates localized, stable structures (instantons)
        intensity = np.abs(self.field)**2
        nonlinear_factor = 1.0 / (1.0 + self.alpha * intensity)
        
        # Wave equation: d²ψ/dt² = ∇²ψ / (1 + α|ψ|²)
        acceleration = laplacian * nonlinear_factor
        
        # Verlet integration
        new_field = 2 * self.field - self.field_prev + self.dt**2 * acceleration
        
        # Apply damping
        new_field *= self.damping
        
        # Update state
        self.field_prev[:] = self.field
        self.field[:] = new_field
    
    def detect_instantons(self):
        """
        Find peaks in the field amplitude (instantons are localized excitations)
        """
        amplitude = np.abs(self.field)
        
        # Simple peak detection
        peaks = []
        for i in range(1, len(amplitude) - 1):
            if amplitude[i] > amplitude[i-1] and amplitude[i] > amplitude[i+1]:
                if amplitude[i] > 0.1:  # Threshold
                    peaks.append(i)
        
        self.instantons = peaks
        return peaks
    
    def measure_coherence(self):
        """
        Measure phase coherence between the two injection regions.
        High coherence means the signals have phase-locked.
        """
        # Get phases at injection points
        phase_a = np.angle(self.field[self.inject_a_pos])
        phase_b = np.angle(self.field[self.inject_b_pos])
        
        # Phase difference
        phase_diff = np.abs(phase_a - phase_b)
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap to [0, π]
        
        # Coherence: 1 when in-phase, 0 when out-of-phase
        coherence = 1.0 - (phase_diff / np.pi)
        
        return coherence, phase_diff
    
    def get_fused_signal(self):
        """
        Extract the merged signal from the middle of the field.
        This is where the two signals have propagated and interfered.
        """
        middle = self.field_size // 2
        
        # Average over a small region
        region = slice(middle - 5, middle + 5)
        fused_amplitude = np.mean(np.abs(self.field[region]))
        fused_phase = np.angle(np.mean(self.field[region]))
        
        # Convert to real signal
        fused = fused_amplitude * np.cos(fused_phase)
        
        return fused
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        # Get inputs
        signal_a = self.get_blended_input('signal_a', 'sum') or 0.0
        signal_b = self.get_blended_input('signal_b', 'sum') or 0.0
        coupling_in = self.get_blended_input('coupling', 'sum')
        damping_in = self.get_blended_input('damping', 'sum')
        
        # Update parameters
        coupling_strength = coupling_in if coupling_in is not None else 1.0
        if coupling_in is not None:
            coupling_strength = 0.5 + coupling_in * 0.5  # Map to [0, 1]
        
        if damping_in is not None:
            self.damping = 0.95 + damping_in * 0.04  # Map to [0.95, 0.99]
        
        # Inject the two signals
        self.inject_signals(signal_a, signal_b, coupling_strength)
        
        # Evolve the field (instanton dynamics)
        self.evolve_field()
        
        # Detect instantons
        self.detect_instantons()
    
    def get_output(self, port_name):
        if port_name == 'fused_output':
            return self.get_fused_signal()
        
        elif port_name == 'coherence':
            coherence, _ = self.measure_coherence()
            return coherence
        
        elif port_name == 'phase_diff':
            _, phase_diff = self.measure_coherence()
            return phase_diff / np.pi  # Normalize to [0, 1]
        
        elif port_name == 'field_image':
            return self.generate_field_image()
        
        return None
    
    def generate_field_image(self):
        """Generate visualization of the field"""
        h = 64
        w = self.field_size
        
        # Create 2D image (amplitude and phase)
        amplitude = np.abs(self.field)
        phase = np.angle(self.field)
        
        # Normalize amplitude
        amp_norm = amplitude / (np.max(amplitude) + 1e-9)
        
        # Create image
        img = np.zeros((h, w), dtype=np.float32)
        
        # Draw amplitude as height
        for i in range(w):
            height = int(amp_norm[i] * (h - 1))
            img[h - height:, i] = amp_norm[i]
        
        return img
    
    def get_display_image(self):
        if not SCIPY_AVAILABLE:
            return None
        
        field_img = self.generate_field_image()
        img_u8 = (np.clip(field_img, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        h, w = img_color.shape[:2]
        
        # Mark injection points
        inject_a_x = self.inject_a_pos * w // self.field_size
        inject_b_x = self.inject_b_pos * w // self.field_size
        
        cv2.circle(img_color, (inject_a_x, h - 5), 3, (255, 0, 0), -1)  # Red
        cv2.circle(img_color, (inject_b_x, h - 5), 3, (0, 255, 0), -1)  # Green
        
        # Mark instantons (field peaks)
        for inst_pos in self.instantons:
            inst_x = inst_pos * w // self.field_size
            cv2.circle(img_color, (inst_x, 10), 2, (255, 255, 255), -1)  # White
        
        # Mark fusion point (center)
        center_x = w // 2
        cv2.line(img_color, (center_x, 0), (center_x, h), (255, 255, 0), 1)  # Yellow
        
        # Resize for display
        img_resized = cv2.resize(img_color, (128, 64), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Field Size", "field_size", self.field_size, None),
            ("Coupling Strength (α)", "alpha", self.alpha, None),
            ("Time Step (dt)", "dt", self.dt, None),
        ]