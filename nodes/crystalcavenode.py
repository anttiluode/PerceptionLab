"""
Crystal Cave Node - Resonance Intelligence
==========================================
A network of coupled resonance fields that learn through scarring.

NOT a neural network. NOT a VAE.
A dynamical system that settles into learned attractors.

Training: Images carve attractor basins through scar formation
Recall: Input settles toward nearest learned attractor
Output: The eigenmode signature of the settled state

"Crystals that scar together, resonate together."
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import os

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class CrystalCaveNode(BaseNode):
    """
    Resonance Intelligence - Crystal Cave
    
    A hierarchy of coupled resonance fields that learn through scarring.
    Each layer's eigenmode filters input to the next layer.
    
    Training: Present images -> system settles -> scars deepen
    Recall: Present partial/new input -> settles to learned attractor
    """
    
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Crystal Cave"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Crystal blue
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',        # Training/query image
            'spectrum_in': 'spectrum',  # Direct magnitude spectrum input
            'complex_spectrum_in': 'complex_spectrum', # NEW - Complex spectral input (purple wire)
            'train': 'signal',          # Training gate
            'reset': 'signal'           # Reset all scars
        }
        
        self.outputs = {
            'image_out': 'image',       # Settled eigenmode as image
            'spectrum_out': 'spectrum', # Eigenmode as 1D spectrum
            'complex_spectrum': 'complex_spectrum',  # Complex spectral output (purple wire)
            'resonance': 'signal',      # How well input resonates (0-1)
            'coherence': 'signal'       # Phase coherence of final state
        }
        
        # Architecture
        self.n_layers = 3
        self.layer_sizes = [32, 64, 128]
        self.n_harmonics = 5  # The magic number
        
        # Physics parameters
        self.settle_steps = 30
        self.scar_rate = 0.02
        self.tension_rate = 0.1
        self.threshold = 0.6
        self.diffusion = 0.5
        self.phase_rate = 0.05
        
        # State
        self.frozen = False
        self.training_count = 0
        self.current_resonance = 0.0
        self.current_coherence = 0.0
        
        # Initialize layers
        self.init_layers()
        
        # Output storage
        self.output_image = np.zeros((128, 128), dtype=np.float32)
        self.output_spectrum = np.zeros(64, dtype=np.float32)
        self.output_complex_spectrum = None 
    
    def init_layers(self):
        """Initialize resonance layers."""
        self.layers = []
        
        for size in self.layer_sizes:
            layer = {
                'size': size,
                'center': size // 2,
                'structure': self._init_structure(size),
                'tension': np.zeros((size, size), dtype=np.float32),
                'scars': np.ones((size, size), dtype=np.float32),  # Transfer function
                'r_grid': self._make_r_grid(size)
            }
            self.layers.append(layer)
    
    def _init_structure(self, size):
        """Initialize complex structure field."""
        structure = np.ones((size, size), dtype=np.complex128)
        structure += (np.random.randn(size, size) + 
                             1j * np.random.randn(size, size)) * 0.1
        return structure
    
    def _make_r_grid(self, size):
        """Create radial distance grid."""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        return np.sqrt((x - center)**2 + (y - center)**2)
    
    def reset_layer(self, layer):
        """Reset a layer's dynamic state (not scars)."""
        size = layer['size']
        layer['structure'] = self._init_structure(size)
        layer['tension'][:] = 0
    
    def reset_all(self):
        """Full reset including scars."""
        for layer in self.layers:
            size = layer['size']
            layer['structure'] = self._init_structure(size)
            layer['tension'][:] = 0
            layer['scars'][:] = 1.0  # Clear all scars
        self.training_count = 0
        print("CrystalCave: Full reset - all scars cleared")
    
    def image_to_chord(self, image):
        """
        Convert image to harmonic chord.
        Extracts the frequency signature, not the pixels.
        """
        if image is None:
            return np.ones(self.n_harmonics) * 0.5
        
        # Ensure grayscale
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()
        
        # Resize to standard
        gray = cv2.resize(gray.astype(np.float32), (64, 64))
        
        # Normalize
        if gray.max() > 1.0:
            gray = gray / 255.0
        
        # Get 2D FFT
        spectrum_2d = np.abs(fftshift(fft2(gray)))
        
        # Extract radial profile
        center = 32
        y, x = np.ogrid[:64, :64]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Average in radial bands
        max_r = center
        band_width = max_r / self.n_harmonics
        
        chord = np.zeros(self.n_harmonics, dtype=np.float32)
        for i in range(self.n_harmonics):
            inner = i * band_width
            outer = (i + 1) * band_width
            mask = (r >= inner) & (r < outer)
            if np.any(mask):
                chord[i] = np.mean(spectrum_2d[mask])
        
        # Normalize
        if chord.max() > 1e-9:
            chord = chord / chord.max()
        
        return chord
    
    def spectrum_to_chord(self, spectrum):
        """Convert 1D spectrum to harmonic chord."""
        if spectrum is None or len(spectrum) == 0:
            return np.ones(self.n_harmonics) * 0.5
        
        # Resample to n_harmonics
        if len(spectrum) >= self.n_harmonics:
            # Average into bands
            band_size = len(spectrum) // self.n_harmonics
            chord = np.array([
                np.mean(np.abs(spectrum[i*band_size:(i+1)*band_size]))
                for i in range(self.n_harmonics)
            ], dtype=np.float32)
        else:
            # Interpolate up
            chord = np.interp(
                np.linspace(0, len(spectrum)-1, self.n_harmonics),
                np.arange(len(spectrum)),
                np.abs(spectrum)
            ).astype(np.float32)
        
        # Normalize
        if chord.max() > 1e-9:
            chord = chord / chord.max()
        
        return chord
        
    def complex_spectrum_to_chord(self, complex_spectrum):
        """Convert complex 2D spectrum (from rfft) to harmonic chord."""
        if complex_spectrum is None or complex_spectrum.size == 0:
            return np.ones(self.n_harmonics) * 0.5
            
        # 1. Convert complex spectrum to a magnitude spectrum (2D)
        # Use np.abs on the complex data
        mag_spectrum = np.abs(complex_spectrum)
        
        # 2. Average the magnitude across rows (axis=0) to get 1D profile
        spectrum_1d = np.mean(mag_spectrum, axis=0)
        
        # 3. Use the existing magnitude-to-chord logic
        # Note: spectrum_1d from rfft is only half the length of the spatial domain
        # The spectrum_to_chord logic handles resizing/averaging.
        return self.spectrum_to_chord(spectrum_1d)

    
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
    
    def compute_eigenmode(self, layer):
        """Compute eigenmode of layer."""
        return np.abs(fftshift(fft2(layer['structure'])))
    
    def compute_coherence(self, layer):
        """Compute phase coherence."""
        phase = np.angle(layer['structure'])
        return float(np.abs(np.mean(np.exp(1j * phase))))
    
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
    
    def settle_layer(self, layer, chord, train=False):
        """
        Let layer settle under chord input.
        Returns coherence and eigenmode.
        """
        size = layer['size']
        
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
                
                # SCARRING - only if training and not frozen
                if train and not self.frozen:
                    layer['scars'][critical] *= (1.0 - self.scar_rate)
                
                # Reset tension
                layer['tension'][critical] = 0
                
                # Diffusion
                layer['structure'] = (
                    gaussian_filter(np.real(layer['structure']), self.diffusion) +
                    1j * gaussian_filter(np.imag(layer['structure']), self.diffusion)
                )
                
            # Phase evolution - modulated by scars!
            # Scarred regions evolve slower (more stable)
            layer['structure'] *= np.exp(1j * self.phase_rate * layer['scars'])
            
            # Normalize magnitude
            mag = np.abs(layer['structure'])
            layer['structure'][mag > 1.0] /= mag[mag > 1.0]
        
        # Final state
        coherence = self.compute_coherence(layer)
        eigenmode = self.compute_eigenmode(layer)
        
        return coherence, eigenmode
    
    def forward(self, chord, train=False):
        """
        Process chord through all layers.
        
        Each layer's eigenmode becomes a filter for the next layer.
        Returns final eigenmode and resonance measure.
        """
        current_chord = chord.copy()
        total_coherence = 0.0
        
        for i, layer in enumerate(self.layers):
            # Reset dynamic state (keep scars)
            self.reset_layer(layer)
            
            # Settle this layer
            coherence, eigenmode = self.settle_layer(layer, current_chord, train)
            total_coherence += coherence
            
            # Extract spectrum for next layer
            spectrum = self.eigenmode_to_spectrum(eigenmode)
            
            # Convert to chord for next layer
            current_chord = self.spectrum_to_chord(spectrum)
        
        # Final layer's eigenmode is the output
        final_eigen = self.compute_eigenmode(self.layers[-1])
        final_spectrum = self.eigenmode_to_spectrum(final_eigen)
        
        # Resonance = average coherence across layers
        resonance = total_coherence / self.n_layers
        
        # Final coherence
        final_coherence = self.compute_coherence(self.layers[-1])
        
        return final_eigen, final_spectrum, resonance, final_coherence
    
    def step(self):
        """Main processing step."""
        # Get inputs
        image_in = self.get_blended_input('image_in', 'first')
        spectrum_in = self.get_blended_input('spectrum_in', 'first')
        complex_spectrum_in = self.get_blended_input('complex_spectrum_in', 'first') # NEW INPUT
        train_signal = self.get_blended_input('train', 'sum') or 0.0
        reset_signal = self.get_blended_input('reset', 'sum') or 0.0
        
        # Reset check
        if reset_signal > 0.5:
            self.reset_all()
            return
        
        # Determine input chord (Priority: Complex Spec > Image > Magnitude Spec > None)
        if complex_spectrum_in is not None:
            # New highest priority input
            chord = self.complex_spectrum_to_chord(complex_spectrum_in)
        elif image_in is not None:
            chord = self.image_to_chord(image_in)
        elif spectrum_in is not None:
            chord = self.spectrum_to_chord(spectrum_in)
        else:
            # No input - maintain state with neutral chord
            chord = np.ones(self.n_harmonics, dtype=np.float32) * 0.5
        
        # Training mode?
        train = (train_signal > 0.5) and not self.frozen
        
        if train:
            self.training_count += 1
            if self.training_count % 100 == 0:
                print(f"CrystalCave: Training step {self.training_count}")
        
        # Process through network
        eigenmode, spectrum, resonance, coherence = self.forward(chord, train)
        
        # Store outputs
        self.output_image = eigenmode / (eigenmode.max() + 1e-9)
        self.output_spectrum = spectrum
        self.current_resonance = resonance
        self.current_coherence = coherence

        # --- COMPLEX SPECTRUM OUTPUT ---
        # The final structure holds the complex-valued field
        final_structure = self.layers[-1]['structure']
        
        # Use the REAL part of the structure for reconstruction
        structure_real = np.real(final_structure)

        # Row-wise rfft to match FFT Cochlea / iFFT Cochlea format
        self.output_complex_spectrum = np.fft.rfft(structure_real.astype(np.float64), axis=1)
        # -----------------------------
    
    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.output_image
        elif port_name == 'spectrum_out':
            return self.output_spectrum
        elif port_name == 'complex_spectrum': 
            return self.output_complex_spectrum
        elif port_name == 'resonance':
            return float(self.current_resonance)
        elif port_name == 'coherence':
            return float(self.current_coherence)
        return None
    
    def get_display_image(self):
        """Create visualization of all layers."""
        # Create display grid: 3 layers x 3 panels (structure, scars, eigen)
        panel_size = 86
        margin = 2
        
        # Calculate exact dimensions
        col_width = panel_size + margin
        width = col_width * 3 - margin  # Remove trailing margin
        height = col_width * 3 - margin + 40  # Extra for status
        
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        for row, layer in enumerate(self.layers):
            y_start = row * col_width
            
            # Panel 1: Structure (magnitude)
            struct_mag = np.abs(layer['structure'])
            struct_mag = struct_mag / (struct_mag.max() + 1e-9)
            struct_img = cv2.resize(struct_mag.astype(np.float32), (panel_size, panel_size))
            struct_color = cv2.applyColorMap((struct_img * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
            
            x1 = 0
            display[y_start:y_start+panel_size, x1:x1+panel_size] = struct_color
            
            # Panel 2: Scars (memory)
            scars_img = cv2.resize(layer['scars'].astype(np.float32), (panel_size, panel_size))
            scars_color = cv2.applyColorMap((scars_img * 255).astype(np.uint8), cv2.COLORMAP_BONE)
            
            x2 = col_width
            display[y_start:y_start+panel_size, x2:x2+panel_size] = scars_color
            
            # Panel 3: Eigenmode (the star)
            eigen = self.compute_eigenmode(layer)
            eigen_log = np.log(1 + eigen)  # Log scale to see structure
            eigen_norm = eigen_log / (eigen_log.max() + 1e-9)
            eigen_img = cv2.resize(eigen_norm.astype(np.float32), (panel_size, panel_size))
            eigen_color = cv2.applyColorMap((eigen_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            x3 = col_width * 2
            display[y_start:y_start+panel_size, x3:x3+panel_size] = eigen_color
        
        # Status bar
        status_y = height - 35
        
        # Training status
        mode = "FROZEN" if self.frozen else "LEARNING"
        color = (100, 100, 255) if self.frozen else (100, 255, 100)
        cv2.putText(display, mode, (5, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Metrics
        cv2.putText(display, f"Train: {self.training_count}", (5, status_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Res: {self.current_resonance:.2f}", (100, status_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Coh: {self.current_coherence:.2f}", (190, status_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Column labels
        cv2.putText(display, "Struct", (20, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(display, "Scars", (col_width + 20, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(display, "Eigen", (col_width * 2 + 20, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        return QtGui.QImage(display.data, width, height, width * 3, 
                            QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Frozen", "frozen", self.frozen, [(True, True), (False, False)]),
            ("Settle Steps", "settle_steps", self.settle_steps, None),
            ("Scar Rate", "scar_rate", self.scar_rate, None),
            ("N Harmonics", "n_harmonics", self.n_harmonics, None),
        ]
    
    def set_config_options(self, options):
        if "frozen" in options:
            self.frozen = bool(options["frozen"])
            print(f"CrystalCave: {'Frozen' if self.frozen else 'Learning'}")
        if "settle_steps" in options:
            self.settle_steps = int(options["settle_steps"])
        if "scar_rate" in options:
            self.scar_rate = float(options["scar_rate"])
        if "n_harmonics" in options:
            self.n_harmonics = int(options["n_harmonics"])
    
    # --- Persistence ---
    def save_custom_state(self, folder_path, node_id):
        """Save learned scars."""
        filename = f"node_{node_id}_crystal_cave.npz"
        filepath = os.path.join(folder_path, filename)
        
        # Save scars from all layers
        scars_dict = {f'scars_{i}': layer['scars'] for i, layer in enumerate(self.layers)}
        scars_dict['training_count'] = self.training_count
        scars_dict['frozen'] = self.frozen
        
        np.savez(filepath, **scars_dict)
        print(f"CrystalCave: Saved {self.training_count} training steps of scars")
        return filename
    
    def load_custom_state(self, filepath):
        """Load learned scars."""
        try:
            data = np.load(filepath)
            
            for i, layer in enumerate(self.layers):
                key = f'scars_{i}'
                if key in data:
                    # Resize if necessary
                    loaded_scars = data[key]
                    if loaded_scars.shape == layer['scars'].shape:
                        layer['scars'] = loaded_scars
                    else:
                        layer['scars'] = cv2.resize(loaded_scars, 
                                                     (layer['size'], layer['size']))
            
            self.training_count = int(data.get('training_count', 0))
            self.frozen = bool(data.get('frozen', True))
            
            print(f"CrystalCave: Loaded scars ({self.training_count} training steps)")
            
        except Exception as e:
            print(f"CrystalCave: Error loading state: {e}")