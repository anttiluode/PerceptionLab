"""
Eigen Crystal Viewer Node - V6 (High-Res Smoothing)
===================================================
Displays eigenmode crystals from complex spectrum input.

V6 FIX: 
- Changed Upscaling from "Nearest Neighbor" (Blocky) to "Bicubic" (Smooth).
  The Seed layer is only 32x32 pixels. Bicubic interpolation smooths
  the jagged edges into organic gradients, making it look "High Res".
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
    """
    
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Eigen Crystal Viewer"
    NODE_COLOR = QtGui.QColor(200, 100, 255)  # Crystal purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_spectrum_in': 'complex_spectrum',  
            'image_in': 'image',                        
            'settle_steps': 'signal',                   
            'diffusion': 'signal',                      
            'phase_rate': 'signal',                     
        }
        
        self.outputs = {
            'eigen_image': 'image',             # The Spectral Crystal
            'structure_image': 'image',         # The Spatial Rings
            'spectrum_out': 'spectrum',         
            'coherence': 'signal',              
        }
        
        # Layer sizes 
        self.layer_sizes = [32, 64, 128]
        self.n_layers = 3
        
        # Physics parameters
        self.settle_steps = 20
        self.diffusion = 0.5
        self.phase_rate = 0.05
        self.tension_rate = 0.1
        self.threshold = 0.6
        
        # Configuration
        self.output_layer_idx = 0  # 0=Seed(Small), 1=Growth(Med), 2=Field(Large)
        
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
        structure = np.ones((size, size), dtype=np.complex128)
        structure += (np.random.randn(size, size) + 
                     1j * np.random.randn(size, size)) * 0.1
        return structure
    
    def _make_r_grid(self, size):
        center = size // 2
        y, x = np.ogrid[:size, :size]
        return np.sqrt((x - center)**2 + (y - center)**2)
    
    def compute_eigenmode(self, layer):
        return np.abs(fftshift(fft2(layer['structure'])))
    
    def compute_coherence(self, layer):
        phase = np.angle(layer['structure'])
        return float(np.abs(np.mean(np.exp(1j * phase))))
    
    def spectrum_to_chord(self, spectrum, n_harmonics=5):
        if spectrum is None or len(spectrum) == 0:
            return np.ones(n_harmonics) * 0.5
        
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
        size = layer['size']
        
        layer['structure'] = self._init_structure(size)
        layer['tension'][:] = 0
        
        for step in range(self.settle_steps):
            input_2d = self.project_chord_to_rings(layer, chord)
            
            if input_2d.max() > 1e-9:
                input_2d = input_2d / input_2d.max()
            
            eigen = self.compute_eigenmode(layer)
            eigen_norm = eigen / (eigen.max() + 1e-9)
            
            resistance = input_2d * (1.0 - eigen_norm)
            layer['tension'] += resistance * self.tension_rate
            
            critical = layer['tension'] > self.threshold
            n_critical = np.sum(critical)
            
            if n_critical > 0:
                layer['structure'][critical] *= -1
                layer['tension'][critical] = 0
                layer['structure'] = (
                    gaussian_filter(np.real(layer['structure']), self.diffusion) +
                    1j * gaussian_filter(np.imag(layer['structure']), self.diffusion)
                )
            
            layer['structure'] *= np.exp(1j * self.phase_rate)
            
            mag = np.abs(layer['structure'])
            layer['structure'][mag > 1.0] /= mag[mag > 1.0]
        
        return self.compute_coherence(layer), self.compute_eigenmode(layer)
    
    def eigenmode_to_spectrum(self, eigenmode):
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
        complex_in = self.get_blended_input('complex_spectrum_in', 'first')
        image_in = self.get_blended_input('image_in', 'first')
        
        settle = self.get_blended_input('settle_steps', 'sum')
        diff = self.get_blended_input('diffusion', 'sum')
        phase = self.get_blended_input('phase_rate', 'sum')
        
        if settle is not None:
            self.settle_steps = int(np.clip(settle, 5, 100))
        if diff is not None:
            self.diffusion = float(np.clip(diff, 0.1, 2.0))
        if phase is not None:
            self.phase_rate = float(np.clip(phase, 0.01, 0.2))
        
        if complex_in is not None:
            chord = self.spectrum_to_chord(complex_in)
        elif image_in is not None:
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
        
        total_coherence = 0.0
        current_chord = chord.copy()
        
        for i, layer in enumerate(self.layers):
            coherence, eigenmode = self.settle_layer(layer, current_chord)
            total_coherence += coherence
            
            self.eigenmodes[i] = eigenmode
            self.structures[i] = np.abs(layer['structure'])
            
            spectrum = self.eigenmode_to_spectrum(eigenmode)
            current_chord = self.spectrum_to_chord(spectrum)
        
        self.current_coherence = total_coherence / self.n_layers
        self.output_spectrum = self.eigenmode_to_spectrum(self.eigenmodes[-1])
    
    def get_output(self, port_name):
        idx = self.output_layer_idx
        if idx >= len(self.layer_sizes): idx = 0
        
        if port_name == 'eigen_image':
            eigen = self.eigenmodes[idx]
            if eigen.max() > 0:
                eigen_log = np.log(1 + eigen)
                eigen_norm = eigen_log / (eigen_log.max() + 1e-9)
                
                # --- V6 FIX: High-Quality Bicubic Upscaling ---
                target_size = 256
                eigen_upscaled = cv2.resize(
                    eigen_norm.astype(np.float32), 
                    (target_size, target_size), 
                    interpolation=cv2.INTER_CUBIC
                )
                
                eigen_uint8 = (eigen_upscaled * 255).astype(np.uint8)
                return cv2.applyColorMap(eigen_uint8, cv2.COLORMAP_JET)
                
            return np.zeros((256, 256, 3), dtype=np.uint8)
            
        elif port_name == 'structure_image':
            struct = self.structures[idx]
            if struct.max() > 0:
                struct_norm = struct / (struct.max() + 1e-9)
                
                # --- V6 FIX: High-Quality Bicubic Upscaling ---
                target_size = 256
                struct_upscaled = cv2.resize(
                    struct_norm.astype(np.float32), 
                    (target_size, target_size), 
                    interpolation=cv2.INTER_CUBIC
                )
                
                struct_uint8 = (struct_upscaled * 255).astype(np.uint8)
                return cv2.applyColorMap(struct_uint8, cv2.COLORMAP_TWILIGHT)
            return np.zeros((256, 256, 3), dtype=np.uint8)
            
        elif port_name == 'spectrum_out':
            return self.output_spectrum
        elif port_name == 'coherence':
            return self.current_coherence
        return None
    
    def get_display_image(self):
        panel_size = 100
        margin = 2
        col_width = panel_size + margin
        width = col_width * 3
        height = col_width * 3 + 30 
        
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        for row, layer in enumerate(self.layers):
            y_start = row * col_width
            
            # Panel 1: Structure (Twilight)
            struct_mag = np.abs(layer['structure'])
            struct_mag = struct_mag / (struct_mag.max() + 1e-9)
            struct_img = cv2.resize(struct_mag.astype(np.float32), (panel_size, panel_size))
            struct_color = cv2.applyColorMap((struct_img * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
            display[y_start:y_start+panel_size, 0:panel_size] = struct_color
            
            # Panel 2: Tension (Bone)
            tension_img = cv2.resize(layer['tension'].astype(np.float32), (panel_size, panel_size))
            if tension_img.max() > 0:
                tension_img = tension_img / tension_img.max()
            tension_color = cv2.applyColorMap((tension_img * 255).astype(np.uint8), cv2.COLORMAP_BONE)
            display[y_start:y_start+panel_size, col_width:col_width+panel_size] = tension_color
            
            # Panel 3: Eigenmode (Jet)
            eigen = self.compute_eigenmode(layer)
            eigen_log = np.log(1 + eigen)
            eigen_norm = eigen_log / (eigen_log.max() + 1e-9)
            eigen_img = cv2.resize(eigen_norm.astype(np.float32), (panel_size, panel_size))
            eigen_color = cv2.applyColorMap((eigen_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            display[y_start:y_start+panel_size, col_width*2:col_width*2+panel_size] = eigen_color
            
            if row == self.output_layer_idx:
                cv2.rectangle(display, (0, y_start), (width, y_start+panel_size), (255, 255, 255), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Struct", (10, 15), font, 0.4, (150, 150, 150), 1)
        cv2.putText(display, "Tension", (col_width + 10, 15), font, 0.4, (150, 150, 150), 1)
        cv2.putText(display, "Eigen", (col_width*2 + 10, 15), font, 0.4, (150, 150, 150), 1)
        
        status_y = height - 15
        cv2.putText(display, f"Coh: {self.current_coherence:.2f}", (10, status_y), font, 0.4, (200, 200, 200), 1)
        
        layers = ["Seed", "Grow", "Field"]
        selected = layers[min(self.output_layer_idx, 2)]
        cv2.putText(display, f"Out: {selected}", (100, status_y), font, 0.4, (100, 255, 100), 1)
        
        return display 
    
    def get_config_options(self):
        layer_opts = [
            ('Seed (Small - 32px)', 0), 
            ('Growth (Med - 64px)', 1), 
            ('Field (Large - 128px)', 2)
        ]
        
        return [
            ("Output Layer", "output_layer_idx", self.output_layer_idx, layer_opts),
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