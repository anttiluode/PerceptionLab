"""
EphapticFieldNode v2 - Learning Ephaptic Substrate
===================================================

The main node for streaming brain data into a learning synthetic substrate.

Combines:
1. EPHAPTIC FIELD DYNAMICS (Pinotsis & Miller 2023)
   - Electric field as control parameter
   - Slower timescale than neural activity
   - Field enslaves/guides neural activity

2. HEBBIAN LEARNING
   - Weights evolve based on co-activation
   - Memory etched into connectivity structure
   - Small-world long-range connections

3. SPECTRAL INPUT from SourceLocalizationNode
   - mode_spectrum: Eigenmode activations (10-dim)
   - band_spectrum: Frequency bands δ,θ,α,β,γ (5-dim)  
   - full_spectrum: Combined (15-dim)
   - complex_modes: Phase-aware modes

The node learns from the brain's eigenmode stream and develops its own
internal structure that mirrors brain connectivity. The ephaptic field
provides stability and guides the emergent activity.

INPUT MODES (configurable):
- 'mode_spectrum': Use eigenmode activations only
- 'band_spectrum': Use frequency bands only
- 'full_spectrum': Use both combined
- 'complex_modes': Use complex modes with phase

INPUTS:
- spectrum_input: Main spectral input from SourceLocalization
- source_image: Optional brain image for spatial seeding
- complex_modes: Complex eigenmode data with phase
- learning_rate: How fast to adapt
- field_strength: Ephaptic coupling strength
- reset: Clear all learned state

OUTPUTS:
- ephaptic_field: The emergent electric field
- thought_field: Autonomous learned patterns
- weight_map: Learned connectivity
- spike_map: Current neural activity
- Multiple analysis signals

Created: December 2025
"""

import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import hilbert

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class EphapticFieldNode2(BaseNode):
    """
    Learning Ephaptic Substrate - streams brain data into a synthetic neural field
    that learns and develops its own dynamics.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Ephaptic Field (Learning)"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Electric blue
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            # From SourceLocalizationNode
            'mode_spectrum': 'spectrum',        # Eigenmode activations (10-dim)
            'band_spectrum': 'spectrum',        # Frequency bands (5-dim)
            'full_spectrum': 'spectrum',        # Combined (15-dim)
            'complex_modes': 'complex_spectrum', # Complex modes with phase
            'source_image': 'image',            # Brain activity image
            
            # Modulation inputs
            'learning_rate': 'signal',
            'field_strength': 'signal',
            'threshold_mod': 'signal',
            'coupling_mod': 'signal',
            
            # Control
            'reset': 'signal',
            'freeze': 'signal'
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Main visual outputs
            'ephaptic_field': 'image',          # The emergent field
            'thought_field': 'image',           # Autonomous activity
            'weight_map': 'image',              # Learned connectivity
            'spike_map': 'image',               # Current neural firing
            'potential_map': 'image',           # Membrane potentials
            'gradient_field': 'image',          # Field gradient vectors
            
            # Derived outputs
            'combined_view': 'image',           # Multi-panel visualization
            'field_fft': 'image',               # Spatial frequency content
            
            # Signals
            'firing_rate': 'signal',
            'synchrony': 'signal',
            'field_energy': 'signal',
            'field_stability': 'signal',
            'learning_delta': 'signal',
            'complexity': 'signal',
            'autonomy': 'signal',
            'dominant_mode': 'signal',
            
            # For downstream
            'eigenfrequencies': 'spectrum',
            'field_spectrum': 'spectrum'
        }
        
        # === GRID SIZE ===
        self.size = 128
        self.center = self.size // 2
        
        # === INPUT MODE ===
        self.input_mode = 'mode_spectrum'  # 'mode_spectrum', 'band_spectrum', 'full_spectrum', 'complex_modes'
        self.use_source_image = False       # Also use source_image for spatial seeding
        
        # === NEURAL STATE ===
        self.potential = np.zeros((self.size, self.size), dtype=np.float32)
        self.refractory = np.zeros((self.size, self.size), dtype=np.float32)
        self.current_spikes = np.zeros((self.size, self.size), dtype=np.float32)
        self.spike_history = np.zeros((self.size, self.size), dtype=np.float32)
        self.last_spike = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === DENDRITIC PLATEAU ===
        self.plateau = np.zeros((self.size, self.size), dtype=np.float32)
        self.plateau_duration = 10
        self.plateau_boost = 0.3
        
        # === EPHAPTIC FIELD ===
        self.field = np.zeros((self.size, self.size), dtype=np.float32)
        self.field_prev = np.zeros((self.size, self.size), dtype=np.float32)
        self.grad_x = np.zeros((self.size, self.size), dtype=np.float32)
        self.grad_y = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === HEBBIAN WEIGHTS ===
        self.weights = np.ones((self.size, self.size), dtype=np.float32)
        self.weight_delta = np.zeros((self.size, self.size), dtype=np.float32)
        self.prev_state = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === SMALL-WORLD CONNECTIONS ===
        self.n_long_range = 512
        self._init_small_world()
        
        # === PARAMETERS ===
        # Neural dynamics
        self.threshold = 0.7
        self.refractory_period = 5
        self.leak = 0.08
        self.coupling = 0.2
        self.input_gain = 0.5
        
        # Field dynamics
        self.field_tau = 0.95           # Slow evolution (control parameter)
        self.field_coupling = 0.15      # How much field affects neurons
        self.field_diffusion = 0.3
        
        # Learning
        self.base_learning_rate = 0.01
        self.weight_decay = 0.001
        self.long_range_strength = 0.3
        
        # Projection settings
        self.projection_mode = 'radial'  # 'radial', 'grid', 'random', 'eigenbasis'
        self.temporal_modulation = True
        self.phase_coupling = True       # Use phase from complex_modes
        
        # Build kernels
        self._build_local_kernel()
        self._build_greens_function()
        self._build_projection_basis()
        
        # Tracking
        self.field_history = []
        self.max_history = 50
        self.autonomous_activity = np.zeros((self.size, self.size), dtype=np.float32)
        
        self.t = 0
    
    def _init_small_world(self):
        """Initialize small-world long-range connections."""
        np.random.seed(42)
        
        self.lr_src_y = np.random.randint(0, self.size, self.n_long_range)
        self.lr_src_x = np.random.randint(0, self.size, self.n_long_range)
        self.lr_dst_y = np.random.randint(0, self.size, self.n_long_range)
        self.lr_dst_x = np.random.randint(0, self.size, self.n_long_range)
        
        # Ensure minimum distance
        for i in range(self.n_long_range):
            while True:
                dy = self.lr_dst_y[i] - self.lr_src_y[i]
                dx = self.lr_dst_x[i] - self.lr_src_x[i]
                if np.sqrt(dy**2 + dx**2) > 20:
                    break
                self.lr_dst_y[i] = np.random.randint(0, self.size)
                self.lr_dst_x[i] = np.random.randint(0, self.size)
        
        self.lr_weights = np.ones(self.n_long_range, dtype=np.float32) * 0.5
    
    def _build_local_kernel(self):
        """Build local coupling kernel."""
        k = np.zeros((7, 7), dtype=np.float32)
        center = 3
        for i in range(7):
            for j in range(7):
                d = np.sqrt((i - center)**2 + (j - center)**2)
                if 0.5 < d < 3.5:
                    k[i, j] = 1.0 / (d + 0.5)
        k[center, center] = 0
        k /= k.sum()
        self.local_kernel = k
    
    def _build_greens_function(self):
        """Build Green's function for Poisson equation."""
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(np.float32)
        r_smooth = np.maximum(r, 1.0)
        self.greens = -np.log(r_smooth) / (2 * np.pi)
        self.greens = self.greens / (np.abs(self.greens).max() + 1e-10)
        self.greens_fft = fft2(np.fft.ifftshift(self.greens))
    
    def _build_projection_basis(self):
        """Build basis functions for projecting spectra to 2D."""
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2).astype(np.float32)
        self.theta_grid = np.arctan2(y - self.center, x - self.center).astype(np.float32)
        
        # Eigenbasis patterns (approximate cortical eigenmodes)
        self.eigenbasis = []
        for n in range(20):
            if n == 0:
                # DC component
                pattern = np.ones((self.size, self.size), dtype=np.float32)
            elif n < 5:
                # Low modes: smooth gradients
                angle = n * np.pi / 4
                pattern = np.cos(angle) * (x - self.center) + np.sin(angle) * (y - self.center)
                pattern = pattern.astype(np.float32) / self.size
            else:
                # Higher modes: more complex patterns
                freq = (n - 4) * 0.5
                pattern = np.cos(freq * self.r_grid / 10 + n * self.theta_grid / 3)
                pattern = pattern.astype(np.float32)
            
            # Normalize
            pattern = pattern / (np.abs(pattern).max() + 1e-10)
            self.eigenbasis.append(pattern)
    
    def project_spectrum_to_2d(self, spectrum, phase_info=None):
        """
        Project 1D spectrum to 2D spatial pattern.
        
        Args:
            spectrum: 1D array of spectral values
            phase_info: Optional phase information for complex projection
        """
        if spectrum is None or len(spectrum) == 0:
            return np.zeros((self.size, self.size), dtype=np.float32)
        
        n_components = len(spectrum)
        drive = np.zeros((self.size, self.size), dtype=np.float32)
        
        if self.projection_mode == 'radial':
            # Project to concentric rings
            for i in range(n_components):
                inner = i * self.center / n_components
                outer = (i + 1) * self.center / n_components
                mask = (self.r_grid >= inner) & (self.r_grid < outer)
                
                value = float(spectrum[i])
                if self.temporal_modulation:
                    # Add temporal variation
                    phase = np.sin(self.t * 0.1 * (i + 1))
                    value *= (0.5 + 0.5 * phase)
                
                if phase_info is not None and i < len(phase_info):
                    # Modulate by phase
                    angle_mod = np.cos(self.theta_grid + phase_info[i])
                    drive[mask] = value * angle_mod[mask]
                else:
                    drive[mask] = value
        
        elif self.projection_mode == 'grid':
            # Project to grid cells
            grid_size = int(np.ceil(np.sqrt(n_components)))
            cell_size = self.size // grid_size
            for i in range(n_components):
                gy, gx = divmod(i, grid_size)
                y_start, y_end = gy * cell_size, (gy + 1) * cell_size
                x_start, x_end = gx * cell_size, (gx + 1) * cell_size
                drive[y_start:y_end, x_start:x_end] = float(spectrum[i])
        
        elif self.projection_mode == 'eigenbasis':
            # Project onto eigenmode-like patterns
            for i in range(min(n_components, len(self.eigenbasis))):
                drive += float(spectrum[i]) * self.eigenbasis[i]
        
        elif self.projection_mode == 'random':
            # Random projection (for comparison)
            np.random.seed(self.t % 1000)
            for i in range(n_components):
                mask = np.random.rand(self.size, self.size) > (1 - 1/n_components)
                drive[mask] += float(spectrum[i])
        
        return drive
    
    def _solve_poisson(self, source):
        """Solve Poisson equation for electric field."""
        source_fft = fft2(source)
        field_fft = source_fft * self.greens_fft
        return np.real(ifft2(field_fft)).astype(np.float32)
    
    def _compute_gradient(self):
        """Compute field gradient."""
        self.grad_x = cv2.Sobel(self.field, cv2.CV_32F, 1, 0, ksize=3)
        self.grad_y = cv2.Sobel(self.field, cv2.CV_32F, 0, 1, ksize=3)
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        # Spectral inputs from SourceLocalization
        mode_spec = self.get_blended_input('mode_spectrum', 'mean')
        band_spec = self.get_blended_input('band_spectrum', 'mean')
        full_spec = self.get_blended_input('full_spectrum', 'mean')
        complex_modes = self.get_blended_input('complex_modes', 'mean')
        source_img = self.get_blended_input('source_image', 'first')
        
        # Modulation
        learn_rate_mod = self.get_blended_input('learning_rate', 'sum')
        field_str_mod = self.get_blended_input('field_strength', 'sum')
        thresh_mod = self.get_blended_input('threshold_mod', 'sum')
        couple_mod = self.get_blended_input('coupling_mod', 'sum')
        reset_sig = self.get_blended_input('reset', 'sum')
        freeze_sig = self.get_blended_input('freeze', 'sum')
        
        # === RESET ===
        if reset_sig is not None and reset_sig > 0:
            self._reset_state()
            return
        
        # === SELECT INPUT BASED ON MODE ===
        spectrum = None
        phase_info = None
        
        if self.input_mode == 'mode_spectrum' and mode_spec is not None:
            spectrum = mode_spec
        elif self.input_mode == 'band_spectrum' and band_spec is not None:
            spectrum = band_spec
        elif self.input_mode == 'full_spectrum' and full_spec is not None:
            spectrum = full_spec
        elif self.input_mode == 'complex_modes' and complex_modes is not None:
            spectrum = np.abs(complex_modes)
            phase_info = np.angle(complex_modes)
        else:
            # Fallback: use whatever is available
            if mode_spec is not None:
                spectrum = mode_spec
            elif band_spec is not None:
                spectrum = band_spec
            elif full_spec is not None:
                spectrum = full_spec
        
        # === PARAMETER MODULATION ===
        threshold = self.threshold
        if thresh_mod is not None:
            threshold = np.clip(0.3 + thresh_mod * 0.7, 0.3, 1.0)
        
        coupling = self.coupling
        if couple_mod is not None:
            coupling = np.clip(self.coupling * (0.5 + couple_mod), 0.01, 0.5)
        
        learning_rate = self.base_learning_rate
        if learn_rate_mod is not None:
            learning_rate = self.base_learning_rate * np.clip(learn_rate_mod, 0, 10)
        
        field_coupling = self.field_coupling
        if field_str_mod is not None:
            field_coupling = np.clip(self.field_coupling * (0.5 + field_str_mod), 0, 0.5)
        
        is_frozen = freeze_sig is not None and freeze_sig > 0
        
        # === STORE PREVIOUS STATE ===
        self.prev_state = self.potential.copy()
        
        # === PROJECT SPECTRUM TO 2D DRIVE ===
        drive = self.project_spectrum_to_2d(spectrum, phase_info)
        
        # Normalize drive
        if np.max(np.abs(drive)) > 0:
            drive = drive / np.max(np.abs(drive))
        
        # === ADD SOURCE IMAGE IF ENABLED ===
        if self.use_source_image and source_img is not None:
            if source_img.dtype == np.uint8:
                src = source_img.astype(np.float32) / 255.0
            else:
                src = source_img.astype(np.float32)
            
            if src.shape[0] != self.size or src.shape[1] != self.size:
                src = cv2.resize(src, (self.size, self.size))
            
            if src.ndim == 3:
                src = np.mean(src, axis=2)
            
            # Blend with spectral drive
            drive = 0.7 * drive + 0.3 * src
        
        # === LOCAL NEIGHBOR COUPLING (weighted by learned weights) ===
        weighted_spikes = self.current_spikes * self.weights
        neighbor_input = convolve(weighted_spikes, self.local_kernel, mode='wrap')
        
        # === LONG-RANGE TELEPORTATION ===
        long_range_input = np.zeros_like(self.potential)
        src_activity = self.current_spikes[self.lr_src_y, self.lr_src_x]
        weighted_lr = src_activity * self.lr_weights
        np.add.at(long_range_input, (self.lr_dst_y, self.lr_dst_x), weighted_lr)
        
        # === EPHAPTIC FIELD COMPUTATION ===
        # Spikes generate field
        instant_field = self._solve_poisson(self.current_spikes * self.weights)
        
        # Field evolves slowly (control parameter behavior)
        self.field_prev = self.field.copy()
        self.field = self.field_tau * self.field + (1 - self.field_tau) * instant_field
        self.field = gaussian_filter(self.field, sigma=self.field_diffusion)
        
        # Compute gradient for coupling
        self._compute_gradient()
        grad_mag = np.sqrt(self.grad_x**2 + self.grad_y**2)
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-10)
        
        # === MEMBRANE DYNAMICS ===
        active_mask = self.refractory <= 0
        
        # Leak
        self.potential[active_mask] *= (1.0 - self.leak)
        
        # Plateau boost
        plateau_contribution = self.plateau * self.plateau_boost
        self.potential[active_mask] += plateau_contribution[active_mask]
        
        # Local coupling
        self.potential[active_mask] += coupling * neighbor_input[active_mask]
        
        # Long-range coupling
        self.potential[active_mask] += self.long_range_strength * long_range_input[active_mask]
        
        # External drive (from brain spectrum)
        self.potential[active_mask] += self.input_gain * drive[active_mask]
        
        # EPHAPTIC COUPLING: field gradient modulates potential
        self.potential[active_mask] += field_coupling * grad_mag_norm[active_mask]
        
        # Clamp
        self.potential = np.clip(self.potential, 0, 1.5)
        
        # === THRESHOLD & FIRE ===
        fire_mask = (self.potential >= threshold) & active_mask
        
        self.current_spikes = fire_mask.astype(np.float32)
        self.spike_history = self.spike_history * 0.95 + self.current_spikes * 0.05
        
        self.potential[fire_mask] = 0
        self.refractory[fire_mask] = self.refractory_period
        self.last_spike[fire_mask] = self.t
        self.plateau[fire_mask] = self.plateau_duration
        
        # === DECAY ===
        self.plateau = np.maximum(0, self.plateau - 1)
        self.refractory = np.maximum(0, self.refractory - 1)
        
        # === TRACK AUTONOMY ===
        input_present = spectrum is not None and np.max(np.abs(spectrum)) > 0.1
        if not input_present:
            self.autonomous_activity = self.autonomous_activity * 0.99 + self.current_spikes * 0.01
        
        # === HEBBIAN LEARNING ===
        if not is_frozen and learning_rate > 0:
            hebbian_update = learning_rate * self.prev_state * self.potential
            weight_decay_term = self.weight_decay * (self.weights - 1.0)
            
            self.weight_delta = hebbian_update - weight_decay_term
            self.weights += self.weight_delta
            self.weights = np.clip(self.weights, 0.1, 5.0)
            
            # Learn long-range weights
            src_prev = self.prev_state[self.lr_src_y, self.lr_src_x]
            dst_curr = self.potential[self.lr_dst_y, self.lr_dst_x]
            lr_hebbian = learning_rate * src_prev * dst_curr
            lr_decay = self.weight_decay * (self.lr_weights - 0.5)
            self.lr_weights += lr_hebbian - lr_decay
            self.lr_weights = np.clip(self.lr_weights, 0.05, 2.0)
        
        # === TRACK FIELD HISTORY ===
        field_energy = np.sum(self.grad_x**2 + self.grad_y**2)
        self.field_history.append(field_energy)
        if len(self.field_history) > self.max_history:
            self.field_history.pop(0)
    
    def _reset_state(self):
        """Reset all state to initial conditions."""
        self.potential.fill(0)
        self.refractory.fill(0)
        self.current_spikes.fill(0)
        self.spike_history.fill(0)
        self.plateau.fill(0)
        self.field.fill(0)
        self.field_prev.fill(0)
        self.weights.fill(1.0)
        self.weight_delta.fill(0)
        self.lr_weights.fill(0.5)
        self.autonomous_activity.fill(0)
        self.field_history.clear()
    
    def compute_synchrony(self):
        """Kuramoto order parameter."""
        period = 20.0
        phases = (self.t - self.last_spike) / period * 2 * np.pi
        return float(np.abs(np.mean(np.exp(1j * phases))))
    
    def compute_complexity(self):
        """Complexity of learned weights."""
        weight_var = np.var(self.weights)
        weight_fft = np.abs(fftshift(fft2(self.weights - self.weights.mean())))
        center_mask = self.r_grid < 20
        high_freq = weight_fft[~center_mask].mean() if (~center_mask).any() else 0
        low_freq = weight_fft[center_mask].mean() if center_mask.any() else 1e-10
        return float(np.clip(weight_var * (high_freq / (low_freq + 1e-10)) * 100, 0, 1))
    
    def compute_field_stability(self):
        """Field stability over time."""
        if len(self.field_history) < 5:
            return 0.5
        history = np.array(self.field_history)
        mean_e = np.mean(history)
        std_e = np.std(history)
        if mean_e < 1e-10:
            return 1.0
        return float(1.0 / (1.0 + std_e / mean_e))
    
    def get_output(self, port_name):
        if port_name == 'ephaptic_field':
            field_norm = self.field / (np.abs(self.field).max() + 1e-10)
            return ((field_norm + 1) / 2 * 255).astype(np.uint8)
        
        elif port_name == 'thought_field':
            thought = self.spike_history * self.weights
            return (thought / (thought.max() + 1e-10) * 255).astype(np.uint8)
        
        elif port_name == 'weight_map':
            w_norm = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min() + 1e-10)
            return (w_norm * 255).astype(np.uint8)
        
        elif port_name == 'spike_map':
            return (self.current_spikes * 255).astype(np.uint8)
        
        elif port_name == 'potential_map':
            return (np.clip(self.potential, 0, 1) * 255).astype(np.uint8)
        
        elif port_name == 'gradient_field':
            grad_mag = np.sqrt(self.grad_x**2 + self.grad_y**2)
            return (grad_mag / (grad_mag.max() + 1e-10) * 255).astype(np.uint8)
        
        elif port_name == 'combined_view':
            return self._render_combined_view()
        
        elif port_name == 'field_fft':
            spec = np.abs(fftshift(fft2(self.field)))
            spec_log = np.log(1 + spec * 10)
            return (spec_log / (spec_log.max() + 1e-10) * 255).astype(np.uint8)
        
        elif port_name == 'firing_rate':
            return float(np.mean(self.current_spikes))
        
        elif port_name == 'synchrony':
            return self.compute_synchrony()
        
        elif port_name == 'field_energy':
            return float(np.sum(self.grad_x**2 + self.grad_y**2))
        
        elif port_name == 'field_stability':
            return self.compute_field_stability()
        
        elif port_name == 'learning_delta':
            return float(np.mean(np.abs(self.weight_delta)))
        
        elif port_name == 'complexity':
            return self.compute_complexity()
        
        elif port_name == 'autonomy':
            auto_mean = np.mean(self.autonomous_activity)
            return float(np.clip(auto_mean * 100, 0, 1))
        
        elif port_name == 'dominant_mode':
            spec = np.abs(fftshift(fft2(self.spike_history)))
            radial = spec[self.center, self.center:]
            return float(np.argmax(radial) + 1)
        
        elif port_name == 'eigenfrequencies':
            spec = np.abs(fftshift(fft2(self.spike_history)))
            return spec[self.center, self.center:].astype(np.float32)
        
        elif port_name == 'field_spectrum':
            spec = np.abs(fftshift(fft2(self.field)))
            return spec[self.center, self.center:].astype(np.float32)
        
        return None
    
    def _render_combined_view(self):
        """Render 3x2 combined view."""
        h, w = self.size, self.size
        display = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # Row 1: Potential, Spikes, Field
        pot_img = (np.clip(self.potential, 0, 1) * 255).astype(np.uint8)
        display[:h, :w] = cv2.applyColorMap(pot_img, cv2.COLORMAP_VIRIDIS)
        
        spike_img = (self.current_spikes * 255).astype(np.uint8)
        display[:h, w:2*w] = cv2.applyColorMap(spike_img, cv2.COLORMAP_HOT)
        
        field_norm = self.field / (np.abs(self.field).max() + 1e-10)
        field_img = ((field_norm + 1) / 2 * 255).astype(np.uint8)
        display[:h, 2*w:] = cv2.applyColorMap(field_img, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        # Row 2: Weights, Thought, Gradient
        w_norm = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min() + 1e-10)
        weight_img = (w_norm * 255).astype(np.uint8)
        display[h:, :w] = cv2.applyColorMap(weight_img, cv2.COLORMAP_INFERNO)
        
        thought = self.spike_history * self.weights
        thought_img = (thought / (thought.max() + 1e-10) * 255).astype(np.uint8)
        display[h:, w:2*w] = cv2.applyColorMap(thought_img, cv2.COLORMAP_PLASMA)
        
        grad_mag = np.sqrt(self.grad_x**2 + self.grad_y**2)
        grad_img = (grad_mag / (grad_mag.max() + 1e-10) * 255).astype(np.uint8)
        display[h:, 2*w:] = cv2.applyColorMap(grad_img, cv2.COLORMAP_JET)
        
        return display
    
    def get_display_image(self):
        display = self._render_combined_view()
        h, w = self.size, self.size
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Potential", (5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Spikes", (w+5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Ephaptic Field", (2*w+5, 15), font, 0.35, (0,255,255), 1)
        cv2.putText(display, "Weights", (5, h+15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Thought", (w+5, h+15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Gradient", (2*w+5, h+15), font, 0.35, (255,255,255), 1)
        
        # Stats
        fr = np.mean(self.current_spikes) * 100
        sync = self.compute_synchrony()
        stab = self.compute_field_stability()
        cmplx = self.compute_complexity()
        
        stats = f"Fire:{fr:.1f}% Sync:{sync:.2f} Stab:{stab:.2f} Cmplx:{cmplx:.2f} Mode:{self.input_mode}"
        cv2.putText(display, stats, (5, h*2-10), font, 0.3, (255,255,255), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        input_modes = [
            ('mode_spectrum', 'mode_spectrum'),
            ('band_spectrum', 'band_spectrum'),
            ('full_spectrum', 'full_spectrum'),
            ('complex_modes', 'complex_modes'),
        ]
        projection_modes = [
            ('radial', 'radial'),
            ('grid', 'grid'),
            ('eigenbasis', 'eigenbasis'),
            ('random', 'random'),
        ]
        return [
            # Input settings
            ("Input Mode", "input_mode", self.input_mode, input_modes),
            ("Projection Mode", "projection_mode", self.projection_mode, projection_modes),
            ("Use Source Image", "use_source_image", self.use_source_image, [('True', True), ('False', False)]),
            ("Temporal Modulation", "temporal_modulation", self.temporal_modulation, [('True', True), ('False', False)]),
            ("Phase Coupling", "phase_coupling", self.phase_coupling, [('True', True), ('False', False)]),
            
            # Neural dynamics
            ("Threshold", "threshold", self.threshold, None),
            ("Leak Rate", "leak", self.leak, None),
            ("Coupling", "coupling", self.coupling, None),
            ("Input Gain", "input_gain", self.input_gain, None),
            ("Refractory Period", "refractory_period", self.refractory_period, None),
            
            # Field dynamics
            ("Field Tau", "field_tau", self.field_tau, None),
            ("Field Coupling", "field_coupling", self.field_coupling, None),
            ("Field Diffusion", "field_diffusion", self.field_diffusion, None),
            
            # Learning
            ("Learning Rate", "base_learning_rate", self.base_learning_rate, None),
            ("Weight Decay", "weight_decay", self.weight_decay, None),
            ("Long-Range Strength", "long_range_strength", self.long_range_strength, None),
            
            # Plateau
            ("Plateau Duration", "plateau_duration", self.plateau_duration, None),
            ("Plateau Boost", "plateau_boost", self.plateau_boost, None),
        ]
    
    def save_custom_state(self, folder_path, node_id):
        """Save learned state."""
        import os
        filename = f"ephaptic_state_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        np.savez(filepath,
                 weights=self.weights,
                 lr_weights=self.lr_weights,
                 field=self.field,
                 spike_history=self.spike_history,
                 autonomous_activity=self.autonomous_activity)
        print(f"[EphapticField] Saved state to {filename}")
        return filename
    
    def load_custom_state(self, filepath):
        """Load learned state."""
        try:
            data = np.load(filepath)
            self.weights = data['weights']
            self.lr_weights = data['lr_weights']
            if 'field' in data:
                self.field = data['field']
            if 'spike_history' in data:
                self.spike_history = data['spike_history']
            if 'autonomous_activity' in data:
                self.autonomous_activity = data['autonomous_activity']
            print(f"[EphapticField] Loaded state from {filepath}")
        except Exception as e:
            print(f"[EphapticField] Failed to load state: {e}")