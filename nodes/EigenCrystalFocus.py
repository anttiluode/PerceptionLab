"""
Eigen Crystal Viewer Node
=========================
Displays eigenmode crystals from complex spectrum input.
Matches the Crystal Cave visualization style.

Takes complex spectrum and shows its eigenmode structure
through FFT-based crystal rendering.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class EigenCrystalViewerNode(BaseNode):
    """
    Eigen Crystal Viewer - Shows eigenmode crystals from complex spectra.
    
    Matches Crystal Cave's visualization by using the same eigenmode computation:
    eigenmode = abs(fftshift(fft2(complex_structure)))
    
    Takes interference output and renders beautiful crystal patterns.
    """
    
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Eigen Crystal Viewer"
    NODE_COLOR = QtGui.QColor(200, 100, 255)  # Crystal purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_spectrum_in': 'complex_spectrum',  # From interference
            'image_in': 'image',                        # Alternative input
            'settle_steps': 'signal',                   # Settling iterations
            'diffusion': 'signal',                      # Diffusion strength
            'phase_rate': 'signal',                     # Phase evolution rate
        }
        
        self.outputs = {
            'eigen_image': 'image',             # Main eigenmode image
            'structure_image': 'image',         # Structure field
            'spectrum_out': 'spectrum',         # Radial spectrum
            'coherence': 'signal',              # Phase coherence
        }
        
        # Layer sizes (like Crystal Cave)
        self.layer_sizes = [32, 64, 128]
        self.n_layers = 3
        
        # Physics parameters
        self.settle_steps = 20
        self.diffusion = 0.5
        self.phase_rate = 0.05
        self.tension_rate = 0.1
        self.threshold = 0.6
        
        # Initialize layers
        self.layers = []
        for size in self.layer_sizes:
            layer = {
                'size': size,
                'center': size // 2,
                'structure': self._init_structure(size),
                'tension': np.zeros((size, size), dtype=np.float32),
                'r_grid': self._make_r_grid(size)
            }
            self.layers.append(layer)
        
        # Output storage
        self.eigenmodes = [np.zeros((s, s), dtype=np.float32) for s in self.layer_sizes]
        self.structures = [np.zeros((s, s), dtype=np.float32) for s in self.layer_sizes]
        self.current_coherence = 0.0
        self.output_spectrum = np.zeros(64, dtype=np.float32)
        
    def _init_structure(self, size):
        """Initialize complex structure field with small noise."""
        structure = np.ones((size, size), dtype=np.complex128)
        structure += (np.random.randn(size, size) + 
                     1j * np.random.randn(size, size)) * 0.1
        return structure
    
    def _make_r_grid(self, size):
        """Create radial distance grid."""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        return np.sqrt((x - center)**2 + (y - center)**2)
    
    def compute_eigenmode(self, layer):
        """Compute eigenmode of layer - EXACTLY like Crystal Cave."""
        return np.abs(fftshift(fft2(layer['structure'])))
    
    def compute_coherence(self, layer):
        """Compute phase coherence."""
        phase = np.angle(layer['structure'])
        return float(np.abs(np.mean(np.exp(1j * phase))))
    
    def spectrum_to_chord(self, spectrum, n_harmonics=5):
        """Convert spectrum to harmonic chord."""
        if spectrum is None or len(spectrum) == 0:
            return np.ones(n_harmonics) * 0.5
        
        # Flatten if 2D
        if spectrum.ndim > 1:
            spectrum = np.mean(np.abs(spectrum), axis=0)
        
        spectrum = np.abs(spectrum)
        
        if len(spectrum) >= n_harmonics:
            band_size = len(spectrum) // n_harmonics
            chord = np.array([
                np.mean(spectrum[i*band_size:(i+1)*band_size])
                for i in range(n_harmonics)
            ], dtype=np.float32)
        else:
            chord = np.interp(
                np.linspace(0, len(spectrum)-1, n_harmonics),
                np.arange(len(spectrum)),
                spectrum
            ).astype(np.float32)
        
        if chord.max() > 1e-9:
            chord = chord / chord.max()
        
        return chord
    
    def project_chord_to_rings(self, layer, chord):
        """Project chord to concentric rings on layer grid."""
        size = layer['size']
        center = layer['center']
        r_grid = layer['r_grid']
        
        ring_width = center / len(chord)
        pattern = np.zeros((size, size), dtype=np.float32)
        
        for i, intensity in enumerate(chord):
            inner = i * ring_width
            outer = (i + 1) * ring_width
            mask = (r_grid >= inner) & (r_grid < outer)
            pattern[mask] = intensity
        
        return pattern
    
    def settle_layer(self, layer, chord):
        """Let layer settle under chord input."""
        size = layer['size']
        
        # Reset structure with small noise
        layer['structure'] = self._init_structure(size)
        layer['tension'][:] = 0
        
        for step in range(self.settle_steps):
            # Project chord to 2D input pattern
            input_2d = self.project_chord_to_rings(layer, chord)
            
            # Normalize
            if input_2d.max() > 1e-9:
                input_2d = input_2d / input_2d.max()
            
            # Current eigenmode
            eigen = self.compute_eigenmode(layer)
            eigen_norm = eigen / (eigen.max() + 1e-9)
            
            # Tension = where input doesn't match eigenmode
            resistance = input_2d * (1.0 - eigen_norm)
            layer['tension'] += resistance * self.tension_rate
            
            # Critical avalanche
            critical = layer['tension'] > self.threshold
            n_critical = np.sum(critical)
            
            if n_critical > 0:
                # Phase flip at critical points
                layer['structure'][critical] *= -1
                
                # Reset tension
                layer['tension'][critical] = 0
                
                # Diffusion
                layer['structure'] = (
                    gaussian_filter(np.real(layer['structure']), self.diffusion) +
                    1j * gaussian_filter(np.imag(layer['structure']), self.diffusion)
                )
            
            # Phase evolution
            layer['structure'] *= np.exp(1j * self.phase_rate)
            
            # Normalize magnitude
            mag = np.abs(layer['structure'])
            layer['structure'][mag > 1.0] /= mag[mag > 1.0]
        
        return self.compute_coherence(layer), self.compute_eigenmode(layer)
    
    def eigenmode_to_spectrum(self, eigenmode):
        """Convert 2D eigenmode to 1D radial spectrum."""
        size = eigenmode.shape[0]
        center = size // 2
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
        
        r_max = min(center, 64)
        spectrum = np.zeros(r_max, dtype=np.float32)
        
        for i in range(r_max):
            mask = (r == i)
            if np.any(mask):
                spectrum[i] = np.mean(eigenmode[mask])
        
        return spectrum
    
    def step(self):
        """Process input through settling layers."""
        # Get inputs
        complex_in = self.get_blended_input('complex_spectrum_in', 'first')
        image_in = self.get_blended_input('image_in', 'first')
        
        settle = self.get_blended_input('settle_steps', 'sum')
        diff = self.get_blended_input('diffusion', 'sum')
        phase = self.get_blended_input('phase_rate', 'sum')
        
        # Update parameters
        if settle is not None:
            self.settle_steps = int(np.clip(settle, 5, 100))
        if diff is not None:
            self.diffusion = float(np.clip(diff, 0.1, 2.0))
        if phase is not None:
            self.phase_rate = float(np.clip(phase, 0.01, 0.2))
        
        # Get input chord
        if complex_in is not None:
            chord = self.spectrum_to_chord(complex_in)
        elif image_in is not None:
            # Convert image to spectrum first
            if image_in.ndim == 3:
                gray = np.mean(image_in, axis=2)
            else:
                gray = image_in.copy()
            if gray.max() > 1.0:
                gray = gray / 255.0
            gray = cv2.resize(gray.astype(np.float32), (64, 64))
            spectrum_2d = np.abs(fftshift(fft2(gray)))
            chord = self.spectrum_to_chord(spectrum_2d.flatten())
        else:
            chord = np.ones(5, dtype=np.float32) * 0.5
        
        # Process through layers
        total_coherence = 0.0
        current_chord = chord.copy()
        
        for i, layer in enumerate(self.layers):
            coherence, eigenmode = self.settle_layer(layer, current_chord)
            total_coherence += coherence
            
            self.eigenmodes[i] = eigenmode
            self.structures[i] = np.abs(layer['structure'])
            
            # Extract spectrum for next layer
            spectrum = self.eigenmode_to_spectrum(eigenmode)
            current_chord = self.spectrum_to_chord(spectrum)
        
        self.current_coherence = total_coherence / self.n_layers
        self.output_spectrum = self.eigenmode_to_spectrum(self.eigenmodes[-1])
    
    def get_output(self, port_name):
        if port_name == 'eigen_image':
            eigen = self.eigenmodes[-1]
            if eigen.max() > 0:
                eigen = eigen / eigen.max()
            return (eigen * 255).astype(np.uint8)
        elif port_name == 'structure_image':
            struct = self.structures[-1]
            if struct.max() > 0:
                struct = struct / struct.max()
            return (struct * 255).astype(np.uint8)
        elif port_name == 'spectrum_out':
            return self.output_spectrum
        elif port_name == 'coherence':
            return self.current_coherence
        return None
    
    def get_display_image(self):
        """Create visualization matching Crystal Cave style."""
        # 3 rows (layers) x 3 columns (struct, blank, eigen)
        panel_size = 100
        margin = 2
        
        col_width = panel_size + margin
        width = col_width * 3
        height = col_width * 3 + 30  # Extra for status
        
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        for row, layer in enumerate(self.layers):
            y_start = row * col_width
            
            # Panel 1: Structure (magnitude)
            struct_mag = np.abs(layer['structure'])
            struct_mag = struct_mag / (struct_mag.max() + 1e-9)
            struct_img = cv2.resize(struct_mag.astype(np.float32), (panel_size, panel_size))
            struct_color = cv2.applyColorMap((struct_img * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
            display[y_start:y_start+panel_size, 0:panel_size] = struct_color
            
            # Panel 2: Empty (like Crystal Cave's scars, but we don't have training)
            # Leave black or show tension
            tension_img = cv2.resize(layer['tension'].astype(np.float32), (panel_size, panel_size))
            if tension_img.max() > 0:
                tension_img = tension_img / tension_img.max()
            tension_color = cv2.applyColorMap((tension_img * 255).astype(np.uint8), cv2.COLORMAP_BONE)
            display[y_start:y_start+panel_size, col_width:col_width+panel_size] = tension_color
            
            # Panel 3: Eigenmode (the crystal!)
            eigen = self.compute_eigenmode(layer)
            eigen_log = np.log(1 + eigen)
            eigen_norm = eigen_log / (eigen_log.max() + 1e-9)
            eigen_img = cv2.resize(eigen_norm.astype(np.float32), (panel_size, panel_size))
            eigen_color = cv2.applyColorMap((eigen_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            display[y_start:y_start+panel_size, col_width*2:col_width*2+panel_size] = eigen_color
        
        # Labels
        cv2.putText(display, "Struct", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(display, "Tension", (col_width + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(display, "Eigen", (col_width*2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Status
        status_y = height - 15
        cv2.putText(display, f"Coh: {self.current_coherence:.2f}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Steps: {self.settle_steps}", (100, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Diff: {self.diffusion:.2f}", (200, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, width, height, width * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Settle Steps", "settle_steps", self.settle_steps, None),
            ("Diffusion", "diffusion", self.diffusion, None),
            ("Phase Rate", "phase_rate", self.phase_rate, None),
            ("Tension Rate", "tension_rate", self.tension_rate, None),
            ("Threshold", "threshold", self.threshold, None),
        ]
    
    def set_config_options(self, options):
        for key, value in options.items():
            if hasattr(self, key):
                setattr(self, key, type(getattr(self, key))(value))