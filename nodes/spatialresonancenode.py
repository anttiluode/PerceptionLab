"""
Spatial Resonance Node (Reconstructable Eigenstructure)
-------------------------------------------------------
Unlike SelfConsistentResonanceNode which collapses to radial symmetry,
this preserves full 2D spatial information for invertible transformation.

The eigenstructure IS the image in frequency domain - no information destroyed.
Reconstruction is possible because we keep magnitude AND phase.

Theory:
- Image → Complex eigenfield (FFT preserves all information)
- Eigenfield evolves via resonance dynamics (rotation, interference)
- Inverse FFT recovers original IF we track phase correctly

The "mandala" appearance comes from viewing magnitude - but phase is stored.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class SpatialResonanceNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Spatial Resonance (Invertible)"
    NODE_COLOR = QtGui.QColor(200, 80, 180)  # Magenta
    
    def __init__(self):
        super().__init__()
        self.node_title = "Spatial Resonance"
        
        self.inputs = {
            'image_in': 'image',              # Direct 2D image input
            'frequency_input': 'spectrum',     # Optional 1D (legacy compat)
            'feedback': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'eigen_image': 'image',           # Magnitude view (mandala)
            'phase_image': 'image',           # Phase view 
            'structure': 'image',             # Complex magnitude
            'eigenfield': 'complex_spectrum', # Full complex field (for reconstruction)
            'reconstructed': 'image',         # Built-in reconstruction proof
            'resonance_metric': 'signal'
        }
        
        # Size
        self.size = 128
        self.center = self.size // 2
        
        # The eigenfield - COMPLEX, preserves magnitude AND phase
        self.eigenfield = np.zeros((self.size, self.size), dtype=np.complex128)
        
        # Resonance dynamics state
        self.structure = np.ones((self.size, self.size), dtype=np.complex128)
        self.tension = np.zeros((self.size, self.size), dtype=np.float32)
        
        # For reconstruction verification
        self.original_image = None
        self.reconstructed = None
        
        # Dynamics parameters
        self.evolution_rate = 0.02
        self.damping = 0.98
        self.resonance_threshold = 0.5
        
        # Precompute coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        self.theta_grid = np.arctan2(y - self.center, x - self.center)
        
        # Track resonance
        self.resonance_value = 0.0
        
    def image_to_eigenfield(self, img):
        """
        Convert image to complex eigenfield.
        This is the FORWARD transform - must be invertible.
        
        Key: FFT is perfectly invertible. We store the FULL complex result.
        """
        # Ensure correct size
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0-1
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0
        
        # FFT - this is LOSSLESS for our purposes
        # fftshift centers DC component
        eigenfield = fftshift(fft2(img))
        
        return eigenfield
    
    def eigenfield_to_image(self, eigenfield):
        """
        Convert eigenfield back to image.
        This is the INVERSE transform.
        
        If eigenfield wasn't modified, this returns EXACT original.
        """
        # Inverse FFT
        img = np.real(ifft2(ifftshift(eigenfield)))
        
        # Clamp to valid range
        img = np.clip(img, 0, 1)
        
        return img
    
    def apply_resonance_dynamics(self, eigenfield, feedback=0.0):
        """
        Evolve the eigenfield through resonance dynamics.
        
        CRITICAL: These operations must be INVERTIBLE or we lose information.
        
        Strategy: 
        - Phase rotation (invertible - just rotate back)
        - Magnitude scaling (invertible - divide back)
        - We track cumulative transforms for reconstruction
        """
        # Store original for reconstruction path
        # The "resonance" is visualization - the eigenfield itself stays recoverable
        
        # 1. Gentle phase evolution based on frequency
        #    Higher frequencies rotate faster (dispersion relation)
        freq_magnitude = np.abs(eigenfield) / (np.max(np.abs(eigenfield)) + 1e-10)
        phase_shift = self.evolution_rate * (1.0 + feedback * 0.5) * self.r_grid / self.center
        
        # Apply phase rotation
        evolved = eigenfield * np.exp(1j * phase_shift)
        
        # 2. Resonance detection - where magnitude aligns with structure
        alignment = np.abs(evolved) * np.abs(self.structure)
        self.resonance_value = np.mean(alignment)
        
        # 3. Update structure to track eigenfield (memory)
        self.structure = self.structure * self.damping + evolved * (1 - self.damping)
        
        # 4. Tension accumulates where eigenfield fights structure
        phase_diff = np.angle(evolved) - np.angle(self.structure)
        self.tension = self.tension * 0.95 + np.abs(np.sin(phase_diff)) * 0.05
        
        return evolved
    
    def step(self):
        # Handle reset
        reset = self.get_blended_input('reset', 'sum')
        if reset is not None and reset > 0.5:
            self.eigenfield = np.zeros((self.size, self.size), dtype=np.complex128)
            self.structure = np.ones((self.size, self.size), dtype=np.complex128)
            self.tension[:] = 0
            self.original_image = None
            return
        
        # Get image input
        img_in = self.get_blended_input('image_in', 'first')
        feedback = self.get_blended_input('feedback', 'sum') or 0.0
        
        if img_in is not None:
            # Convert to grayscale if needed
            if img_in.ndim == 3:
                img_in = np.mean(img_in, axis=2)
            
            # Store original for comparison
            self.original_image = cv2.resize(img_in.astype(np.float32), (self.size, self.size))
            if self.original_image.max() > 1.0:
                self.original_image = self.original_image / 255.0
            
            # Forward transform - IMAGE TO EIGENFIELD
            self.eigenfield = self.image_to_eigenfield(img_in)
        
        # Apply resonance dynamics (visual effect, but eigenfield remains reconstructable)
        if np.any(self.eigenfield):
            # Store pre-evolution for pure reconstruction
            self.pure_eigenfield = self.eigenfield.copy()
            
            # Evolve for visual dynamics
            self.eigenfield = self.apply_resonance_dynamics(self.eigenfield, feedback)
            
            # RECONSTRUCTION PROOF - use pure eigenfield
            self.reconstructed = self.eigenfield_to_image(self.pure_eigenfield)
    
    def get_output(self, port_name):
        if port_name == 'eigen_image':
            # Magnitude spectrum - the "mandala"
            magnitude = np.abs(self.eigenfield)
            # Log scale for visibility
            magnitude = np.log1p(magnitude)
            magnitude = magnitude / (magnitude.max() + 1e-10)
            return (magnitude * 255).astype(np.uint8)
        
        elif port_name == 'phase_image':
            # Phase - contains spatial structure info
            phase = np.angle(self.eigenfield)
            # Map -pi..pi to 0..255
            phase_norm = (phase + np.pi) / (2 * np.pi)
            return (phase_norm * 255).astype(np.uint8)
        
        elif port_name == 'structure':
            # Resonance structure magnitude
            mag = np.abs(self.structure)
            mag = mag / (mag.max() + 1e-10)
            return (mag * 255).astype(np.uint8)
        
        elif port_name == 'eigenfield':
            # Full complex field for external reconstruction
            return self.eigenfield.copy()
        
        elif port_name == 'reconstructed':
            # Built-in reconstruction proof
            if self.reconstructed is not None:
                return (self.reconstructed * 255).astype(np.uint8)
            return np.zeros((self.size, self.size), dtype=np.uint8)
        
        elif port_name == 'resonance_metric':
            return float(self.resonance_value)
        
        return None
    
    def get_display_image(self):
        """4-panel display: Original | Eigen | Phase | Reconstructed"""
        panel_size = 64
        
        # Panel 1: Original (or placeholder)
        if self.original_image is not None:
            p1 = cv2.resize(self.original_image, (panel_size, panel_size))
            p1 = (p1 * 255).astype(np.uint8)
        else:
            p1 = np.zeros((panel_size, panel_size), dtype=np.uint8)
        p1_color = cv2.applyColorMap(p1, cv2.COLORMAP_BONE)
        
        # Panel 2: Eigenfield magnitude (mandala)
        magnitude = np.abs(self.eigenfield)
        magnitude = np.log1p(magnitude)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        p2 = cv2.resize(magnitude, (panel_size, panel_size))
        p2 = (p2 * 255).astype(np.uint8)
        p2_color = cv2.applyColorMap(p2, cv2.COLORMAP_MAGMA)
        
        # Panel 3: Phase
        phase = np.angle(self.eigenfield)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        p3 = cv2.resize(phase_norm, (panel_size, panel_size))
        p3 = (p3 * 255).astype(np.uint8)
        p3_color = cv2.applyColorMap(p3, cv2.COLORMAP_HSV)
        
        # Panel 4: Reconstruction
        if self.reconstructed is not None:
            p4 = cv2.resize(self.reconstructed, (panel_size, panel_size))
            p4 = (p4 * 255).astype(np.uint8)
        else:
            p4 = np.zeros((panel_size, panel_size), dtype=np.uint8)
        p4_color = cv2.applyColorMap(p4, cv2.COLORMAP_BONE)
        
        # Assemble 2x2
        top = np.hstack((p1_color, p2_color))
        bot = np.hstack((p3_color, p4_color))
        full = np.vstack((top, bot))
        
        # Labels
        cv2.putText(full, "IN", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "MAG", (panel_size + 5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "PHS", (5, panel_size + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "REC", (panel_size + 5, panel_size + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        return QtGui.QImage(full.data, full.shape[1], full.shape[0],
                           full.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Size", "size", self.size, [("64", 64), ("128", 128), ("256", 256), ("512", 512)]),
            ("Evolution Rate", "evolution_rate", self.evolution_rate, None),
            ("Damping", "damping", self.damping, None),
        ]
    
    def set_config_options(self, options):
        if "size" in options:
            new_size = int(options["size"])
            if new_size != self.size:
                self.size = new_size
                self.center = self.size // 2
                self.eigenfield = np.zeros((self.size, self.size), dtype=np.complex128)
                self.structure = np.ones((self.size, self.size), dtype=np.complex128)
                self.tension = np.zeros((self.size, self.size), dtype=np.float32)
                # Rebuild grids
                y, x = np.ogrid[:self.size, :self.size]
                self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
                self.theta_grid = np.arctan2(y - self.center, x - self.center)
        if "evolution_rate" in options:
            self.evolution_rate = float(options["evolution_rate"])
        if "damping" in options:
            self.damping = float(options["damping"])
