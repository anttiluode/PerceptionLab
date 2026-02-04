"""
Gabor Filter Bank Node - Testing Pribram's Holographic Brain Theory
====================================================================

Implements a V1-like Gabor filter bank to probe holographic properties
of EEG interference patterns from PhiHologramNode.

Pribram's Core Claims (testable here):
1. Information is DISTRIBUTED via interference (fragments reconstruct whole)
2. PHASE carries the information (not magnitude)
3. Multiple "memories" can be MULTIPLEXED in same field
4. Gabor functions are the natural basis (V1 simple cells)

Experiments enabled:
- Feed hologram → see if Gabor decomposition reveals structure
- Feed cropped hologram → test distributed reconstruction
- Feed phase-only vs magnitude-only → test phase dominance
- Feed superimposed holograms → test multiplexing separation

Author: Built for Antti's consciousness crystallography / Pribram testing
"""

import numpy as np
import cv2
from scipy import ndimage

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    # Fallback for standalone testing
    try:
        from PyQt6 import QtGui
    except ImportError:
        # Minimal mock for testing without PyQt6
        class MockQtGui:
            @staticmethod
            def QColor(*args):
                return None
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


class GaborFilterBankNode(BaseNode):
    """
    V1-like Gabor filter bank for probing holographic structure.
    
    Gabor filters are optimal for joint space-frequency localization,
    matching V1 simple cell receptive fields. Pribram explicitly linked
    these to holographic processing in dendritic microfields.
    """
    
    NODE_CATEGORY = "Vision"
    NODE_TITLE = "Gabor Bank (Pribram)"
    NODE_COLOR = QtGui.QColor(220, 150, 50)  # Orange-gold (Gabor wavelets)
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'image_in': 'image',                    # From hologram or any image
            'complex_field': 'complex_spectrum',    # Direct complex field input
            'orientation_mod': 'signal',            # Modulate preferred orientation
            'frequency_mod': 'signal',              # Modulate preferred frequency
            'phase_only': 'signal',                 # >0.5 = use phase only (Pribram test)
            'magnitude_only': 'signal',             # >0.5 = use magnitude only (control)
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Main visualizations
            'energy_map': 'image',           # Combined filter energy (max across all)
            'orientation_map': 'image',      # Dominant orientation at each pixel
            'frequency_map': 'image',        # Dominant frequency at each pixel
            'phase_map': 'image',            # Phase response visualization
            
            # Reconstruction (Pribram test: can we reconstruct from Gabor responses?)
            'reconstruction': 'image',       # Inverse from filter responses
            'residual': 'image',             # What's lost in reconstruction
            
            # Individual filter stack (for detailed analysis)
            'filter_stack': 'image',         # Tiled view of all filter responses
            
            # Metrics
            'total_energy': 'signal',        # Sum of all filter energies
            'orientation_coherence': 'signal', # How aligned are responses?
            'phase_coherence': 'signal',     # Phase consistency (Pribram key metric)
            'reconstruction_error': 'signal', # How much is lost?
            'holographic_score': 'signal',   # Composite Pribram-ness score
        }
        
        # === PARAMETERS ===
        # Orientation parameters
        self.num_orientations = 8           # 0°, 22.5°, 45°, ... 157.5°
        
        # Frequency parameters (cycles per image width)
        self.frequencies = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]  # Low to high
        
        # Gabor kernel parameters
        self.sigma = 3.0                    # Gaussian envelope width
        self.gamma = 0.5                    # Aspect ratio (elongation)
        self.psi = 0                        # Phase offset (0 = even, π/2 = odd)
        
        # Processing size
        self.process_size = 128
        
        # === CACHES ===
        self._energy_map = None
        self._orientation_map = None
        self._frequency_map = None
        self._phase_map = None
        self._reconstruction = None
        self._residual = None
        self._filter_stack = None
        
        self._total_energy = 0.0
        self._orientation_coherence = 0.0
        self._phase_coherence = 0.0
        self._reconstruction_error = 0.0
        self._holographic_score = 0.0
        
        # Pre-build filter bank
        self._filters = None
        self._build_filter_bank()
    
    def _build_gabor_kernel(self, size, frequency, orientation, sigma=None, gamma=None):
        """
        Build a single Gabor kernel.
        
        Gabor = Gaussian envelope × Complex sinusoid
        g(x,y) = exp(-x'²+γ²y'²/2σ²) × exp(i(2πfx' + ψ))
        
        where x' = x*cos(θ) + y*sin(θ)
              y' = -x*sin(θ) + y*cos(θ)
        """
        if sigma is None:
            sigma = self.sigma
        if gamma is None:
            gamma = self.gamma
            
        # Kernel size (odd)
        ksize = int(6 * sigma) | 1
        ksize = min(ksize, size // 2)
        
        # Coordinate grids
        x = np.linspace(-ksize//2, ksize//2, ksize)
        y = np.linspace(-ksize//2, ksize//2, ksize)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        theta = orientation
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gaussian envelope
        gaussian = np.exp(-(X_rot**2 + (gamma * Y_rot)**2) / (2 * sigma**2))
        
        # Complex sinusoid (wavelength = 1/frequency)
        wavelength = 1.0 / (frequency + 1e-6)
        sinusoid_real = np.cos(2 * np.pi * X_rot / wavelength + self.psi)
        sinusoid_imag = np.sin(2 * np.pi * X_rot / wavelength + self.psi)
        
        # Gabor = Gaussian × Sinusoid
        kernel_real = gaussian * sinusoid_real
        kernel_imag = gaussian * sinusoid_imag
        
        # Normalize
        kernel_real -= kernel_real.mean()
        kernel_imag -= kernel_imag.mean()
        
        norm = np.sqrt(np.sum(kernel_real**2) + np.sum(kernel_imag**2)) + 1e-9
        kernel_real /= norm
        kernel_imag /= norm
        
        return kernel_real.astype(np.float32), kernel_imag.astype(np.float32)
    
    def _build_filter_bank(self):
        """Pre-compute all Gabor filters."""
        self._filters = []
        
        orientations = np.linspace(0, np.pi, self.num_orientations, endpoint=False)
        
        for freq in self.frequencies:
            for ori in orientations:
                real, imag = self._build_gabor_kernel(
                    self.process_size, freq, ori
                )
                self._filters.append({
                    'real': real,
                    'imag': imag,
                    'freq': freq,
                    'ori': ori
                })
    
    def _apply_filter_bank(self, img):
        """
        Apply all Gabor filters to image.
        Returns energy, phase, and individual responses.
        """
        h, w = img.shape[:2]
        n_filters = len(self._filters)
        
        # Storage
        energies = np.zeros((n_filters, h, w), dtype=np.float32)
        phases = np.zeros((n_filters, h, w), dtype=np.float32)
        
        for i, filt in enumerate(self._filters):
            # Convolve with real and imaginary parts
            resp_real = cv2.filter2D(img, cv2.CV_32F, filt['real'])
            resp_imag = cv2.filter2D(img, cv2.CV_32F, filt['imag'])
            
            # Energy (magnitude squared)
            energies[i] = resp_real**2 + resp_imag**2
            
            # Phase
            phases[i] = np.arctan2(resp_imag, resp_real)
        
        return energies, phases
    
    def _compute_orientation_map(self, energies):
        """Find dominant orientation at each pixel."""
        h, w = energies.shape[1:]
        n_ori = self.num_orientations
        n_freq = len(self.frequencies)
        
        # Sum energies across frequencies for each orientation
        ori_energies = np.zeros((n_ori, h, w), dtype=np.float32)
        
        for i in range(n_ori):
            for j in range(n_freq):
                idx = j * n_ori + i
                if idx < len(energies):
                    ori_energies[i] += energies[idx]
        
        # Find dominant orientation
        dominant_ori = np.argmax(ori_energies, axis=0)
        
        # Normalize to 0-1 for visualization
        return dominant_ori.astype(np.float32) / (n_ori - 1)
    
    def _compute_frequency_map(self, energies):
        """Find dominant frequency at each pixel."""
        h, w = energies.shape[1:]
        n_ori = self.num_orientations
        n_freq = len(self.frequencies)
        
        # Sum energies across orientations for each frequency
        freq_energies = np.zeros((n_freq, h, w), dtype=np.float32)
        
        for j in range(n_freq):
            for i in range(n_ori):
                idx = j * n_ori + i
                if idx < len(energies):
                    freq_energies[j] += energies[idx]
        
        # Find dominant frequency
        dominant_freq = np.argmax(freq_energies, axis=0)
        
        # Normalize to 0-1
        return dominant_freq.astype(np.float32) / (n_freq - 1)
    
    def _compute_phase_coherence(self, phases, energies):
        """
        Compute phase coherence across filters.
        High coherence = holographic-like distributed structure.
        
        This is a key Pribram metric: in a hologram, phase relationships
        are consistent across space.
        """
        # Weight phases by energy
        weighted_sin = np.zeros(phases.shape[1:], dtype=np.float32)
        weighted_cos = np.zeros(phases.shape[1:], dtype=np.float32)
        total_energy = np.zeros(phases.shape[1:], dtype=np.float32)
        
        for i in range(len(phases)):
            e = energies[i]
            p = phases[i]
            weighted_sin += e * np.sin(p)
            weighted_cos += e * np.cos(p)
            total_energy += e
        
        total_energy = np.maximum(total_energy, 1e-9)
        
        # Resultant length (0 = random phases, 1 = perfect coherence)
        coherence = np.sqrt(weighted_sin**2 + weighted_cos**2) / total_energy
        
        return coherence
    
    def _reconstruct_from_responses(self, img, energies):
        """
        Attempt to reconstruct image from Gabor responses.
        
        Pribram test: If holographic, even partial responses
        should reconstruct meaningful structure.
        """
        h, w = img.shape[:2]
        reconstruction = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        for i, filt in enumerate(self._filters):
            # Use energy-weighted filter as reconstruction kernel
            energy = energies[i]
            
            # Convolve energy with transposed filter (pseudo-inverse)
            contrib = cv2.filter2D(energy, cv2.CV_32F, filt['real'].T)
            reconstruction += contrib
            weights += energy
        
        # Normalize
        weights = np.maximum(weights, 1e-9)
        reconstruction /= np.sqrt(weights)
        
        # Normalize to original range
        if img.max() > img.min():
            reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-9)
            reconstruction = reconstruction * (img.max() - img.min()) + img.min()
        
        return reconstruction
    
    def _compute_holographic_score(self, phase_coherence, reconstruction_error, orientation_coherence):
        """
        Composite score for "Pribram-ness" of the input.
        
        High score = behaves holographically:
        - High phase coherence (distributed phase structure)
        - Low reconstruction error (info preserved in Gabor decomposition)
        - Moderate orientation coherence (not random, not single-oriented)
        """
        # Phase coherence: higher = more holographic
        phase_score = np.mean(phase_coherence)
        
        # Reconstruction: lower error = more holographic
        recon_score = 1.0 - min(reconstruction_error, 1.0)
        
        # Orientation: peak around 0.5 (neither random nor trivial)
        ori_score = 1.0 - abs(orientation_coherence - 0.5) * 2
        
        # Weighted combination
        holographic = 0.5 * phase_score + 0.3 * recon_score + 0.2 * ori_score
        
        return float(np.clip(holographic, 0, 1))
    
    def step(self):
        """Main processing step."""
        # Get input image
        img = self.get_blended_input('image_in', 'mean')
        complex_field = self.get_blended_input('complex_field', 'mean')
        
        # Phase/magnitude mode switches (Pribram tests)
        phase_only = self.get_blended_input('phase_only', 'max') or 0.0
        magnitude_only = self.get_blended_input('magnitude_only', 'max') or 0.0
        
        # Handle complex field input
        if img is None and complex_field is not None:
            if float(phase_only) > 0.5:
                # PHASE ONLY TEST: Key Pribram experiment
                # Holographic info should be in phase
                img = np.angle(complex_field)
                img = (img + np.pi) / (2 * np.pi)  # Normalize to 0-1
            elif float(magnitude_only) > 0.5:
                # MAGNITUDE ONLY (control): Should lose structure
                img = np.abs(complex_field)
                img = img / (img.max() + 1e-9)
            else:
                # Default: use magnitude for visualization
                img = np.abs(complex_field)
                img = img / (img.max() + 1e-9)
        
        if img is None:
            return
        
        # Prepare image
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Convert to grayscale if color
        if img.ndim == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), 
                              cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Resize for processing
        h_orig, w_orig = img.shape[:2]
        img_proc = cv2.resize(img, (self.process_size, self.process_size))
        
        # Apply filter bank
        energies, phases = self._apply_filter_bank(img_proc)
        
        # Compute energy map (max across all filters)
        self._energy_map = np.max(energies, axis=0)
        
        # Compute orientation and frequency maps
        self._orientation_map = self._compute_orientation_map(energies)
        self._frequency_map = self._compute_frequency_map(energies)
        
        # Compute phase map (weighted average phase)
        self._phase_map = self._compute_phase_coherence(phases, energies)
        
        # Reconstruction (Pribram test)
        self._reconstruction = self._reconstruct_from_responses(img_proc, energies)
        
        # Residual
        self._residual = np.abs(img_proc - self._reconstruction)
        
        # Build filter stack visualization
        n_filters = len(self._filters)
        n_cols = self.num_orientations
        n_rows = len(self.frequencies)
        tile_size = 32
        
        stack = np.zeros((n_rows * tile_size, n_cols * tile_size), dtype=np.float32)
        for j in range(n_rows):
            for i in range(n_cols):
                idx = j * n_cols + i
                if idx < n_filters:
                    tile = cv2.resize(energies[idx], (tile_size, tile_size))
                    tile = tile / (tile.max() + 1e-9)
                    stack[j*tile_size:(j+1)*tile_size, 
                          i*tile_size:(i+1)*tile_size] = tile
        self._filter_stack = stack
        
        # Compute metrics
        self._total_energy = float(np.sum(self._energy_map))
        
        # Orientation coherence: how uniform is the orientation map?
        ori_hist, _ = np.histogram(self._orientation_map.flatten(), bins=self.num_orientations)
        ori_hist = ori_hist / ori_hist.sum()
        self._orientation_coherence = float(1.0 - (-np.sum(ori_hist * np.log(ori_hist + 1e-9)) / np.log(self.num_orientations)))
        
        # Phase coherence (mean)
        self._phase_coherence = float(np.mean(self._phase_map))
        
        # Reconstruction error
        self._reconstruction_error = float(np.mean(self._residual))
        
        # Holographic score
        self._holographic_score = self._compute_holographic_score(
            self._phase_map, 
            self._reconstruction_error,
            self._orientation_coherence
        )
    
    def get_output(self, port_name):
        """Return output data."""
        outputs = {
            'energy_map': self._energy_map,
            'orientation_map': self._orientation_map,
            'frequency_map': self._frequency_map,
            'phase_map': self._phase_map,
            'reconstruction': self._reconstruction,
            'residual': self._residual,
            'filter_stack': self._filter_stack,
            'total_energy': self._total_energy,
            'orientation_coherence': self._orientation_coherence,
            'phase_coherence': self._phase_coherence,
            'reconstruction_error': self._reconstruction_error,
            'holographic_score': self._holographic_score,
        }
        return outputs.get(port_name, None)
    
    def get_display_image(self):
        """Create visualization for node face."""
        size = self.process_size
        
        # 2x2 display: Energy | Orientation | Phase | Reconstruction
        display = np.zeros((size, size * 2, 3), dtype=np.uint8)
        half = size // 2
        
        # Top-left: Energy map (inferno colormap)
        if self._energy_map is not None:
            energy_norm = self._energy_map / (self._energy_map.max() + 1e-9)
            energy_u8 = (energy_norm * 255).astype(np.uint8)
            energy_color = cv2.applyColorMap(
                cv2.resize(energy_u8, (size, size)), 
                cv2.COLORMAP_INFERNO
            )
            display[:, :size] = energy_color
        
        # Top-right: Phase coherence (viridis)
        if self._phase_map is not None:
            phase_u8 = (np.clip(self._phase_map, 0, 1) * 255).astype(np.uint8)
            phase_color = cv2.applyColorMap(
                cv2.resize(phase_u8, (size, size)), 
                cv2.COLORMAP_VIRIDIS
            )
            display[:, size:] = phase_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Gabor Energy", (5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Phase Coherence", (size + 5, 15), font, 0.4, (255, 255, 255), 1)
        
        # Add holographic score
        cv2.putText(display, f"H:{self._holographic_score:.2f}", (5, size - 5), 
                   font, 0.4, (0, 255, 200), 1)
        cv2.putText(display, f"P:{self._phase_coherence:.2f}", (size + 5, size - 5), 
                   font, 0.4, (0, 255, 200), 1)
        
        display = np.ascontiguousarray(display)
        h, w = display.shape[:2]
        
        qimg = QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = display
        return qimg
    
    def get_config_options(self):
        """Configuration dialog options."""
        return [
            ("Num Orientations", "num_orientations", self.num_orientations, 'int'),
            ("Sigma (envelope)", "sigma", self.sigma, 'float'),
            ("Gamma (aspect)", "gamma", self.gamma, 'float'),
            ("Process Size", "process_size", self.process_size, 'int'),
        ]
    
    def set_config_options(self, options):
        """Apply configuration."""
        rebuild = False
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    old_val = getattr(self, key)
                    setattr(self, key, value)
                    if key in ['num_orientations', 'sigma', 'gamma', 'process_size']:
                        rebuild = True
        
        if rebuild:
            self._build_filter_bank()


# === STANDALONE TEST ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Gabor Filter Bank - Pribram Holographic Brain Test")
    print("=" * 55)
    print()
    print("Pribram's Holonomic Brain Theory claims:")
    print("1. Memory/perception = distributed interference patterns")
    print("2. Gabor functions are the natural basis (V1 simple cells)")
    print("3. Phase carries the information, not magnitude")
    print("4. Fragments can reconstruct the whole (holographic)")
    print()
    
    # Create test patterns
    size = 128
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Pattern 1: Holographic-like (interference fringes)
    holographic = np.sin(X * 20 + Y * 15) * np.cos(X * 10 - Y * 20)
    holographic = (holographic - holographic.min()) / (holographic.max() - holographic.min())
    
    # Pattern 2: Non-holographic (localized features)
    non_holographic = np.zeros_like(holographic)
    non_holographic[30:50, 30:50] = 1.0
    non_holographic[70:90, 70:90] = 0.7
    
    print("Testing with holographic (interference) vs localized patterns...")
    print()
    
    # Create node
    node = GaborFilterBankNode()
    
    # Test holographic
    node.input_data = {'image_in': [holographic.astype(np.float32)]}
    node.step()
    print(f"Holographic pattern:")
    print(f"  Phase coherence: {node._phase_coherence:.3f}")
    print(f"  Holographic score: {node._holographic_score:.3f}")
    print()
    
    # Test non-holographic
    node.input_data = {'image_in': [non_holographic.astype(np.float32)]}
    node.step()
    print(f"Localized pattern:")
    print(f"  Phase coherence: {node._phase_coherence:.3f}")
    print(f"  Holographic score: {node._holographic_score:.3f}")
    print()
    
    print("Higher phase coherence + holographic score = more Pribram-like")
    print("Feed this node your PhiHologram output to test real EEG interference!")