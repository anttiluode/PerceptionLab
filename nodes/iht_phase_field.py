"""
IHT Phase Field Node - The fundamental quantum substrate
Implements complex Bloch-sphere cellular automaton with:
- Unitary evolution (Division/branching)
- Dissipative coupling (Dilution/decoherence)
- Attractor alignment

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class IHTPhaseFieldNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 50, 200)  # Deep purple for quantum
    
    def __init__(self, grid_size=64):
        super().__init__()
        self.node_title = "IHT Phase Field"
        
        self.inputs = {
            'dilution': 'signal',      # γ parameter (0-1)
            'alignment': 'signal',      # η parameter for attractor
            'perturbation': 'image'     # External disturbance
        }
        
        self.outputs = {
            'phase_field': 'image',     # Complex field visualization
            'coherence': 'signal',      # Global phase coherence
            'constraint_density': 'image',  # ρ_C for gravity coupling
            'participation_ratio': 'signal'  # PR metric
        }
        
        self.N = int(grid_size)
        
        # Physics parameters
        self.alpha = 0.1  # Diffusion strength (division)
        self.gamma = 0.05  # Base dilution rate
        self.eta = 0.1     # Attractor alignment strength
        
        # Complex phase field (Bloch sphere states)
        self.psi = np.random.randn(self.N, self.N).astype(np.complex64)
        self.psi += 1j * np.random.randn(self.N, self.N).astype(np.complex64)
        
        # Normalize initially
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 1e-9:
            self.psi /= norm
            
        # Attractor state (will be learned or set)
        self.attractor = np.zeros_like(self.psi)
        self._init_simple_attractor()
        
        # Metrics
        self.coherence_value = 1.0
        self.constraint_density = np.zeros((self.N, self.N), dtype=np.float32)
        self.pr_value = 0.0
        
    def _init_simple_attractor(self):
        """Initialize a simple Gaussian attractor"""
        y, x = np.ogrid[-self.N//2:self.N//2, -self.N//2:self.N//2]
        r2 = x*x + y*y
        self.attractor = np.exp(-r2 / (2 * (self.N/8)**2)).astype(np.complex64)
        self.attractor /= np.sqrt(np.sum(np.abs(self.attractor)**2))
        
    def _unitary_step(self):
        """Division: Quantum branching via discrete Laplacian"""
        # FFT-based diffusion (periodic boundary)
        psi_fft = np.fft.fft2(self.psi)
        
        # Frequency coordinates
        kx = np.fft.fftfreq(self.N).reshape(-1, 1)
        ky = np.fft.fftfreq(self.N).reshape(1, -1)
        k2 = kx**2 + ky**2
        
        # Diffusion in Fourier space
        psi_fft *= np.exp(-self.alpha * k2)
        
        self.psi = np.fft.ifft2(psi_fft)
        
    def _dilution_step(self):
        """Dilution: Decoherence/normalization"""
        self.psi *= (1.0 - self.gamma)
        
    def _attractor_step(self):
        """Attractor alignment: Projection toward learned state"""
        # Spatial localization (Gaussian window around center)
        y, x = np.ogrid[-self.N//2:self.N//2, -self.N//2:self.N//2]
        r2 = x*x + y*y
        lambda_x = np.exp(-r2 / (2 * (self.N/4)**2))
        
        # Project toward attractor
        error = self.psi - self.attractor
        self.psi -= self.eta * lambda_x * error
        
    def _compute_metrics(self):
        """Compute coherence, PR, and constraint density"""
        # Global phase coherence
        total_amp = np.sum(np.abs(self.psi))
        phase_sum = np.sum(self.psi)
        self.coherence_value = np.abs(phase_sum) / (total_amp + 1e-9)
        
        # Participation Ratio
        amp2 = np.abs(self.psi)**2
        amp4 = amp2**2
        sum_amp2 = np.sum(amp2)
        sum_amp4 = np.sum(amp4)
        if sum_amp4 > 1e-12:
            self.pr_value = (sum_amp2**2) / sum_amp4
        else:
            self.pr_value = 0.0
            
        # Constraint density (where amplitude is localized)
        self.constraint_density = np.abs(self.psi)**2
        
    def step(self):
        # Get control parameters
        dilution_in = self.get_blended_input('dilution', 'sum')
        alignment_in = self.get_blended_input('alignment', 'sum')
        
        if dilution_in is not None:
            # Map from [-1,1] to [0, 0.2]
            self.gamma = np.clip((dilution_in + 1.0) / 2.0 * 0.2, 0.0, 0.2)
            
        if alignment_in is not None:
            # Map from [-1,1] to [0, 0.5]
            self.eta = np.clip((alignment_in + 1.0) / 2.0 * 0.5, 0.0, 0.5)
            
        # External perturbation
        perturb = self.get_blended_input('perturbation', 'mean')
        if perturb is not None:
            perturb_resized = cv2.resize(perturb, (self.N, self.N))
            # Add as phase modulation
            self.psi *= np.exp(1j * perturb_resized * np.pi)
            
        # Run physics steps
        self._unitary_step()
        self._dilution_step()
        self._attractor_step()
        
        # Periodic renormalization
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 1e-9:
            self.psi /= norm
            
        self._compute_metrics()
        
    def get_output(self, port_name):
        if port_name == 'phase_field':
            # Visualize as amplitude with phase hue
            amp = np.abs(self.psi)
            phase = np.angle(self.psi)
            
            # Normalize amplitude
            amp_norm = amp / (amp.max() + 1e-9)
            
            # Map phase to hue (0-180 for OpenCV HSV)
            hue = ((phase + np.pi) / (2*np.pi) * 180).astype(np.uint8)
            sat = (amp_norm * 255).astype(np.uint8)
            val = (amp_norm * 255).astype(np.uint8)
            
            hsv = np.stack([hue, sat, val], axis=-1)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            return rgb.astype(np.float32) / 255.0
            
        elif port_name == 'coherence':
            return self.coherence_value
            
        elif port_name == 'constraint_density':
            return self.constraint_density
            
        elif port_name == 'participation_ratio':
            return self.pr_value
            
        return None
        
    def get_display_image(self):
        # Show phase field
        rgb_out = self.get_output('phase_field')
        if rgb_out is None:
            return None
            
        rgb_u8 = (rgb_out * 255).astype(np.uint8)
        
        # Add coherence bar at bottom
        bar_h = 5
        coherence_color = int(self.coherence_value * 255)
        rgb_u8[-bar_h:, :] = [coherence_color, coherence_color, 0]
        
        rgb_u8 = np.ascontiguousarray(rgb_u8)
        h, w = rgb_u8.shape[:2]
        return QtGui.QImage(rgb_u8.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Grid Size", "N", self.N, None),
            ("Diffusion (α)", "alpha", self.alpha, None),
        ]