"""
Eigenfield Reconstruct Node
---------------------------
Reconstructs image from complex eigenfield.

NO CHEATING - this node receives ONLY the eigenfield output.
It has no access to the original image.

The reconstruction works because:
1. FFT is mathematically invertible
2. We preserved both magnitude AND phase in the eigenfield
3. Inverse FFT recovers the spatial domain signal

If the eigenfield was modified (resonance dynamics), reconstruction
will show those modifications - proving information flow is bidirectional.
"""

import numpy as np
import cv2
from scipy.fft import ifft2, ifftshift

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class EigenfieldReconstructNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Eigenfield -> Image"
    NODE_COLOR = QtGui.QColor(100, 180, 200)  # Cyan - complement of magenta
    
    def __init__(self):
        super().__init__()
        self.node_title = "Eigen Reconstruct"
        
        self.inputs = {
            'eigenfield': 'complex_spectrum',  # Complex 2D field
            'eigen_image': 'image',            # Fallback: magnitude only (lossy)
            'phase_image': 'image',            # Fallback: separate phase
        }
        
        self.outputs = {
            'reconstructed': 'image',
            'error_map': 'image',              # If we have reference
            'quality_metric': 'signal'
        }
        
        self.size = 128
        self.reconstructed = np.zeros((self.size, self.size), dtype=np.float32)
        self.quality = 0.0
        
        # For error computation if eigenfield has reference
        self.error_map = np.zeros((self.size, self.size), dtype=np.float32)
    
    def step(self):
        # Priority 1: Full complex eigenfield (lossless reconstruction)
        eigenfield = self.get_blended_input('eigenfield', 'first')
        
        if eigenfield is not None and np.iscomplexobj(eigenfield):
            # INVERSE TRANSFORM - the core reconstruction
            self.reconstructed = self._reconstruct_from_complex(eigenfield)
            self.quality = 1.0  # Perfect reconstruction possible
            return
        
        # Priority 2: Magnitude + Phase images (nearly lossless)
        magnitude_img = self.get_blended_input('eigen_image', 'first')
        phase_img = self.get_blended_input('phase_image', 'first')
        
        if magnitude_img is not None and phase_img is not None:
            self.reconstructed = self._reconstruct_from_mag_phase(magnitude_img, phase_img)
            self.quality = 0.9  # Small quantization loss
            return
        
        # Priority 3: Magnitude only (lossy - needs phase estimation)
        if magnitude_img is not None:
            self.reconstructed = self._reconstruct_from_magnitude_only(magnitude_img)
            self.quality = 0.5  # Significant loss without phase
            return
    
    def _reconstruct_from_complex(self, eigenfield):
        """
        Perfect reconstruction from complex eigenfield.
        This is just inverse FFT - mathematically exact.
        """
        # Inverse shift and inverse FFT
        spatial = ifft2(ifftshift(eigenfield))
        
        # Take real part (imaginary should be ~0 for real images)
        img = np.real(spatial)
        
        # Normalize to 0-1
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        
        # Ensure correct size
        if img.shape[0] != self.size:
            img = cv2.resize(img.astype(np.float32), (self.size, self.size))
        
        return img.astype(np.float32)
    
    def _reconstruct_from_mag_phase(self, magnitude_img, phase_img):
        """
        Reconstruct from separate magnitude and phase images.
        Small loss from 8-bit quantization of phase.
        """
        # Handle input formats
        if magnitude_img.ndim == 3:
            magnitude_img = np.mean(magnitude_img, axis=2)
        if phase_img.ndim == 3:
            phase_img = np.mean(phase_img, axis=2)
        
        # Ensure same size
        target_size = max(magnitude_img.shape[0], phase_img.shape[0])
        magnitude_img = cv2.resize(magnitude_img.astype(np.float32), (target_size, target_size))
        phase_img = cv2.resize(phase_img.astype(np.float32), (target_size, target_size))
        
        # Magnitude was log-scaled for display, undo that
        magnitude_img = magnitude_img / 255.0 if magnitude_img.max() > 1 else magnitude_img
        magnitude = np.expm1(magnitude_img * 5)  # Approximate inverse of log1p scaling
        
        # Phase was mapped from [-pi, pi] to [0, 255], undo that
        phase_img = phase_img / 255.0 if phase_img.max() > 1 else phase_img
        phase = phase_img * 2 * np.pi - np.pi
        
        # Reconstruct complex field
        eigenfield = magnitude * np.exp(1j * phase)
        
        # Inverse transform
        return self._reconstruct_from_complex(eigenfield)
    
    def _reconstruct_from_magnitude_only(self, magnitude_img):
        """
        Reconstruct from magnitude only - LOSSY.
        Uses iterative phase retrieval (Gerchberg-Saxton variant).
        
        This is the hard case - phase is lost, must be estimated.
        """
        if magnitude_img.ndim == 3:
            magnitude_img = np.mean(magnitude_img, axis=2)
        
        magnitude_img = cv2.resize(magnitude_img.astype(np.float32), (self.size, self.size))
        
        # Undo log scaling
        magnitude_img = magnitude_img / 255.0 if magnitude_img.max() > 1 else magnitude_img
        magnitude = np.expm1(magnitude_img * 5)
        
        # Gerchberg-Saxton phase retrieval
        # Start with random phase
        phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        for iteration in range(50):  # More iterations = better estimate
            # Construct complex spectrum
            spectrum = magnitude * np.exp(1j * phase)
            
            # Inverse FFT to spatial domain
            spatial = ifft2(ifftshift(spectrum))
            
            # Apply spatial constraints:
            # - Must be real (or nearly real)
            # - Must be positive (for typical images)
            # - Could add support constraints if known
            spatial_constrained = np.abs(spatial)  # Force real and positive
            
            # Optional: additional spatial constraints
            # spatial_constrained = np.clip(spatial_constrained, 0, 1)
            
            # Forward FFT back to frequency domain
            new_spectrum = np.fft.fftshift(np.fft.fft2(spatial_constrained))
            
            # Keep original magnitude, take new phase
            phase = np.angle(new_spectrum)
        
        # Final reconstruction with estimated phase
        final_spectrum = magnitude * np.exp(1j * phase)
        spatial = ifft2(ifftshift(final_spectrum))
        img = np.abs(spatial)
        
        # Normalize
        if img.max() > 0:
            img = img / img.max()
        
        return img.astype(np.float32)
    
    def get_output(self, port_name):
        if port_name == 'reconstructed':
            return (self.reconstructed * 255).astype(np.uint8)
        
        elif port_name == 'error_map':
            return (self.error_map * 255).astype(np.uint8)
        
        elif port_name == 'quality_metric':
            return self.quality
        
        return None
    
    def get_display_image(self):
        """Show reconstruction with quality indicator"""
        img = (self.reconstructed * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        
        # Resize for display
        display = cv2.resize(img_color, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Quality bar
        bar_width = int(self.quality * 120)
        cv2.rectangle(display, (4, 118), (4 + bar_width, 124), (0, 255, 0), -1)
        cv2.rectangle(display, (4, 118), (124, 124), (255, 255, 255), 1)
        
        # Label
        cv2.putText(display, f"Q={self.quality:.1f}", (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(display.data, 128, 128, 128 * 3, 
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Output Size", "size", self.size, [("64", 64), ("128", 128), ("256", 256)]),
        ]
    
    def set_config_options(self, options):
        if "size" in options:
            self.size = int(options["size"])
            self.reconstructed = np.zeros((self.size, self.size), dtype=np.float32)
            self.error_map = np.zeros((self.size, self.size), dtype=np.float32)
