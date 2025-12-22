"""
GradientFFTCompletionNode - Finding What's Hidden in the Gradient
=================================================================

"The gradient tells you where things change. 
 But what if change itself contains structure we're not seeing?"

The gradient field from EphapticFieldNode is a spatial derivative.
Mathematically: gradient ≈ multiplication by frequency in Fourier domain.

This means:
- Gradient magnitude ~ |ω| * |F(ω)|  (frequency-weighted spectrum)
- We're MISSING the low frequencies (they have small gradients)
- We're MISSING the phase (gradient gives magnitude of change only)

This node attempts to COMPLETE the spectrum:
1. Use gradient as constraint on high-frequency content
2. User injects hypotheses about low-frequency structure  
3. Iterative refinement (Gerchberg-Saxton-like) to find consistency
4. Reveal what COULD be hiding in the gradient

The philosophical point: The ephaptic field's gradient might contain
information about phase relationships that we're not extracting.
If consciousness operates in frequency domain, the gradient is
a PROJECTION of that domain - can we reconstruct more?

INPUTS:
- gradient_field: From EphapticFieldNode gradient output
- original_field: Optional - the field that made the gradient (for comparison)
- dc_injection: Manual DC (mean) level to inject
- low_freq_boost: How much to emphasize recovered low frequencies
- phase_seed: Seed for phase initialization (0=random, 1=from gradient angle)
- iterations: How many refinement cycles

OUTPUTS:
- completed_fft: The reconstructed full spectrum (magnitude)
- completed_phase: The reconstructed phase field
- reconstructed_field: Inverse FFT of completed spectrum
- hidden_structure: What was "added" by completion (not in gradient)
- residual: Difference from original (if provided)
- low_freq_content: Just the recovered low frequencies
- spectral_energy: Energy in different frequency bands

Created: December 2025
For Antti's quest to find what's hiding in the gradient
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, sobel
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class GradientFFTCompletionNode(BaseNode):
    """
    Attempts to complete a full FFT from gradient (partial frequency) information.
    Reveals hidden low-frequency structure and phase relationships.
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Gradient FFT Completion"
    NODE_COLOR = QtGui.QColor(200, 100, 255)  # Purple - hidden/mysterious
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'gradient_field': 'image',      # From ephaptic gradient output
            'gradient_x': 'image',          # Optional: separate x gradient
            'gradient_y': 'image',          # Optional: separate y gradient
            'original_field': 'image',      # Optional: for comparison
            
            # User controls for injection
            'dc_injection': 'signal',       # DC level (mean brightness)
            'low_freq_boost': 'signal',     # Boost factor for low frequencies
            'phase_seed_mode': 'signal',    # 0=random, 1=from gradient, 2=spiral
            'iterations': 'signal',         # Refinement iterations
            'regularization': 'signal',     # How much to smooth/constrain
            
            # Frequency band controls
            'ring_1_boost': 'signal',       # Innermost ring (lowest freq)
            'ring_2_boost': 'signal',       
            'ring_3_boost': 'signal',
            'ring_4_boost': 'signal',       # Outer ring (higher freq)
            
            'reset': 'signal'
        }
        
        self.outputs = {
            # Main outputs
            'completed_fft': 'image',       # Full spectrum magnitude
            'completed_phase': 'image',     # Phase field
            'reconstructed_field': 'image', # Inverse FFT result
            
            # Analysis outputs
            'hidden_structure': 'image',    # What completion added
            'residual': 'image',            # Difference from original
            'low_freq_content': 'image',    # Just the low frequencies
            'phase_coherence_map': 'image', # Where phase is stable
            
            # Combined view
            'combined_view': 'image',
            
            # Signals
            'total_energy': 'signal',
            'low_freq_energy': 'signal',
            'high_freq_energy': 'signal',
            'phase_coherence': 'signal',
            'completion_confidence': 'signal',
            
            # Spectrum for downstream
            'frequency_spectrum': 'spectrum'
        }
        
        self.size = 128
        self.center = self.size // 2
        
        # Build coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        self.theta_grid = np.arctan2(y - self.center, x - self.center)
        
        # Frequency ring masks
        self._build_ring_masks()
        
        # === STATE ===
        self.completed_spectrum = np.zeros((self.size, self.size), dtype=np.complex128)
        self.completed_magnitude = np.zeros((self.size, self.size))
        self.completed_phase = np.zeros((self.size, self.size))
        self.reconstructed = np.zeros((self.size, self.size))
        self.hidden_structure = np.zeros((self.size, self.size))
        
        # Phase history for coherence
        self.phase_history = deque(maxlen=20)
        
        # === PARAMETERS ===
        self.base_dc = 0.5
        self.base_low_boost = 2.0
        self.base_iterations = 10
        self.base_regularization = 0.1
        
        # Ring boosts (innermost to outer)
        self.ring_boosts = [2.0, 1.5, 1.0, 0.8]
        
        # Phase mode: 0=random, 1=gradient-derived, 2=spiral, 3=from_previous
        self.phase_mode = 1
        
        # Previous phase for continuity
        self.prev_phase = None
        
        self.t = 0
    
    def _build_ring_masks(self):
        """Build frequency ring masks for selective boosting."""
        self.ring_masks = []
        ring_edges = [0, 5, 15, 30, 64]  # Frequency boundaries
        
        for i in range(len(ring_edges) - 1):
            mask = (self.r_grid >= ring_edges[i]) & (self.r_grid < ring_edges[i+1])
            self.ring_masks.append(mask.astype(np.float32))
    
    def _gradient_to_frequency_constraint(self, grad_mag):
        """
        Convert gradient magnitude to frequency domain constraint.
        
        Gradient in spatial domain = multiplication by iω in frequency domain.
        So |gradient| ≈ |ω| * |F(ω)|
        
        To recover |F(ω)|, we divide by |ω| (with regularization for ω=0).
        """
        # FFT of gradient magnitude (gives us frequency structure of the gradient)
        grad_fft = fftshift(fft2(grad_mag))
        
        # The frequency weighting that gradient applies
        omega = self.r_grid + 1e-6  # Avoid division by zero
        
        # Inverse weighting to recover original spectrum
        # But we need to be careful - this amplifies low frequencies a LOT
        recovery_weight = 1.0 / (omega + self.base_regularization * self.size)
        
        # Apply recovery
        recovered_mag = np.abs(grad_fft) * recovery_weight
        
        return recovered_mag, np.angle(grad_fft)
    
    def _initialize_phase(self, grad_x, grad_y, mode):
        """
        Initialize phase field based on selected mode.
        """
        if mode == 0:
            # Random phase
            return np.random.uniform(-np.pi, np.pi, (self.size, self.size))
        
        elif mode == 1:
            # Derive from gradient direction
            # Phase ~ atan2(grad_y, grad_x) in spatial domain
            # This relates to local orientation
            grad_angle = np.arctan2(grad_y, grad_x)
            # Transform to frequency domain phase (this is approximate)
            return fftshift(np.angle(fft2(np.exp(1j * grad_angle))))
        
        elif mode == 2:
            # Spiral phase (creates vortex-like structure)
            return self.theta_grid * 2  # 2-fold spiral
        
        elif mode == 3 and self.prev_phase is not None:
            # Continue from previous (temporal coherence)
            return self.prev_phase + np.random.randn(self.size, self.size) * 0.1
        
        else:
            return np.zeros((self.size, self.size))
    
    def _gerchberg_saxton_iteration(self, mag_constraint, phase_estimate, spatial_constraint=None):
        """
        One iteration of Gerchberg-Saxton-like phase retrieval.
        
        We have:
        - Magnitude constraint from gradient
        - Phase estimate (being refined)
        - Optional spatial constraint (original field if available)
        """
        # Build spectrum from current estimates
        spectrum = mag_constraint * np.exp(1j * phase_estimate)
        
        # Transform to spatial domain
        spatial = np.real(ifft2(ifftshift(spectrum)))
        
        # Apply spatial constraints (if we have them)
        if spatial_constraint is not None:
            # Soft constraint: blend toward known spatial structure
            spatial = spatial * 0.7 + spatial_constraint * 0.3
        
        # Non-negativity (often a valid constraint for intensity fields)
        # spatial = np.maximum(spatial, 0)
        
        # Transform back to frequency domain
        new_spectrum = fftshift(fft2(spatial))
        
        # Keep the magnitude constraint, update phase
        new_phase = np.angle(new_spectrum)
        
        return new_phase
    
    def _apply_ring_boosts(self, magnitude, boosts):
        """Apply frequency-ring-specific boost factors."""
        boosted = magnitude.copy()
        for mask, boost in zip(self.ring_masks, boosts):
            boosted = boosted + mask * magnitude * (boost - 1)
        return boosted
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        gradient = self.get_blended_input('gradient_field', 'first')
        grad_x = self.get_blended_input('gradient_x', 'first')
        grad_y = self.get_blended_input('gradient_y', 'first')
        original = self.get_blended_input('original_field', 'first')
        
        # Control inputs
        dc_in = self.get_blended_input('dc_injection', 'sum')
        low_boost_in = self.get_blended_input('low_freq_boost', 'sum')
        phase_mode_in = self.get_blended_input('phase_seed_mode', 'sum')
        iter_in = self.get_blended_input('iterations', 'sum')
        reg_in = self.get_blended_input('regularization', 'sum')
        
        # Ring boosts
        ring_ins = [
            self.get_blended_input(f'ring_{i}_boost', 'sum')
            for i in range(1, 5)
        ]
        
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.prev_phase = None
            return
        
        if gradient is None:
            return
        
        # Apply input parameters
        dc_level = dc_in if dc_in is not None else self.base_dc
        low_boost = low_boost_in if low_boost_in is not None else self.base_low_boost
        iterations = int(iter_in) if iter_in is not None else self.base_iterations
        iterations = max(1, min(50, iterations))
        reg = reg_in if reg_in is not None else self.base_regularization
        phase_mode = int(phase_mode_in) if phase_mode_in is not None else self.phase_mode
        
        # Update ring boosts from inputs
        for i, r_in in enumerate(ring_ins):
            if r_in is not None:
                self.ring_boosts[i] = r_in
        
        # === PREPROCESS INPUTS ===
        # Normalize gradient
        if gradient.dtype == np.uint8:
            gradient = gradient.astype(np.float32) / 255.0
        if gradient.shape[0] != self.size:
            gradient = cv2.resize(gradient, (self.size, self.size))
        
        # Handle separate gradients or compute from combined
        if grad_x is not None and grad_y is not None:
            if grad_x.dtype == np.uint8:
                grad_x = grad_x.astype(np.float32) / 255.0
                grad_y = grad_y.astype(np.float32) / 255.0
            gx = cv2.resize(grad_x, (self.size, self.size)) if grad_x.shape[0] != self.size else grad_x
            gy = cv2.resize(grad_y, (self.size, self.size)) if grad_y.shape[0] != self.size else grad_y
        else:
            # Estimate gradients from magnitude using Sobel
            gx = sobel(gradient, axis=1)
            gy = sobel(gradient, axis=0)
        
        # Optional original for comparison
        orig_field = None
        if original is not None:
            if original.dtype == np.uint8:
                original = original.astype(np.float32) / 255.0
            orig_field = cv2.resize(original, (self.size, self.size)) if original.shape[0] != self.size else original
        
        # === GRADIENT TO FREQUENCY CONSTRAINT ===
        self.base_regularization = reg
        mag_constraint, grad_phase = self._gradient_to_frequency_constraint(gradient)
        
        # Apply ring boosts
        mag_constraint = self._apply_ring_boosts(mag_constraint, self.ring_boosts)
        
        # Add DC injection (the gradient kills DC)
        mag_constraint[self.center, self.center] += dc_level * self.size * self.size * low_boost
        
        # Boost low frequencies
        low_freq_boost_mask = np.exp(-self.r_grid**2 / (10**2)) * low_boost
        mag_constraint = mag_constraint * (1 + low_freq_boost_mask)
        
        # === PHASE RETRIEVAL ===
        # Initialize phase
        phase_estimate = self._initialize_phase(gx, gy, phase_mode)
        
        # Iterative refinement
        for _ in range(iterations):
            phase_estimate = self._gerchberg_saxton_iteration(
                mag_constraint, phase_estimate, orig_field
            )
        
        # === BUILD FINAL SPECTRUM ===
        self.completed_magnitude = mag_constraint
        self.completed_phase = phase_estimate
        self.completed_spectrum = mag_constraint * np.exp(1j * phase_estimate)
        
        # Reconstruct spatial field
        self.reconstructed = np.real(ifft2(ifftshift(self.completed_spectrum)))
        
        # === COMPUTE DERIVED QUANTITIES ===
        
        # Hidden structure: what's NOT in the original gradient
        # This is the low-frequency content we "invented"
        gradient_spectrum = fftshift(fft2(gradient))
        hidden_spectrum = self.completed_spectrum.copy()
        # Zero out high frequencies that were in gradient
        high_freq_mask = self.r_grid > 10
        hidden_spectrum[high_freq_mask] = 0
        self.hidden_structure = np.real(ifft2(ifftshift(hidden_spectrum)))
        
        # Store phase history
        self.phase_history.append(self.completed_phase.copy())
        self.prev_phase = self.completed_phase.copy()
        
    def get_output(self, port_name):
        if port_name == 'completed_fft':
            mag = np.log(1 + np.abs(self.completed_magnitude) * 10)
            return self._normalize_to_uint8(mag)
        
        elif port_name == 'completed_phase':
            phase_norm = (self.completed_phase + np.pi) / (2 * np.pi)
            return (phase_norm * 255).astype(np.uint8)
        
        elif port_name == 'reconstructed_field':
            return self._normalize_to_uint8(self.reconstructed)
        
        elif port_name == 'hidden_structure':
            return self._normalize_to_uint8(self.hidden_structure)
        
        elif port_name == 'residual':
            # Would need original input - placeholder
            return np.zeros((self.size, self.size), dtype=np.uint8)
        
        elif port_name == 'low_freq_content':
            # Extract just low frequencies
            low_mask = self.r_grid < 15
            low_spec = self.completed_spectrum * low_mask
            low_spatial = np.real(ifft2(ifftshift(low_spec)))
            return self._normalize_to_uint8(low_spatial)
        
        elif port_name == 'phase_coherence_map':
            if len(self.phase_history) > 5:
                phases = np.array(list(self.phase_history)[-10:])
                coherence = np.abs(np.mean(np.exp(1j * phases), axis=0))
                return (coherence * 255).astype(np.uint8)
            return np.zeros((self.size, self.size), dtype=np.uint8)
        
        elif port_name == 'combined_view':
            return self._render_combined()
        
        elif port_name == 'total_energy':
            return float(np.sum(np.abs(self.completed_spectrum)**2))
        
        elif port_name == 'low_freq_energy':
            low_mask = self.r_grid < 15
            return float(np.sum(np.abs(self.completed_spectrum[low_mask])**2))
        
        elif port_name == 'high_freq_energy':
            high_mask = self.r_grid >= 15
            return float(np.sum(np.abs(self.completed_spectrum[high_mask])**2))
        
        elif port_name == 'phase_coherence':
            if len(self.phase_history) > 5:
                phases = np.array(list(self.phase_history)[-10:])
                coherence = np.abs(np.mean(np.exp(1j * phases)))
                return float(coherence)
            return 0.0
        
        elif port_name == 'completion_confidence':
            # How much did we have to "invent"?
            # Lower confidence if we added lots of low frequency
            if np.sum(np.abs(self.completed_spectrum)**2) > 0:
                low_mask = self.r_grid < 15
                low_ratio = np.sum(np.abs(self.completed_spectrum[low_mask])**2) / np.sum(np.abs(self.completed_spectrum)**2)
                return float(1.0 - low_ratio)
            return 0.0
        
        elif port_name == 'frequency_spectrum':
            # Radial profile for downstream
            radial = np.zeros(64)
            for r in range(64):
                ring = (self.r_grid >= r) & (self.r_grid < r + 1)
                if np.sum(ring) > 0:
                    radial[r] = np.mean(np.abs(self.completed_spectrum[ring]))
            return radial.astype(np.float32)
        
        return None
    
    def _normalize_to_uint8(self, arr):
        arr = np.nan_to_num(arr)
        if arr.max() == arr.min():
            return np.zeros((self.size, self.size), dtype=np.uint8)
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        return (norm * 255).astype(np.uint8)
    
    def _render_combined(self):
        """Render 2x3 combined view."""
        h, w = self.size, self.size
        display = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # Row 1: Completed FFT, Phase, Reconstructed
        fft_img = self._normalize_to_uint8(np.log(1 + np.abs(self.completed_magnitude) * 10))
        display[:h, :w] = cv2.applyColorMap(fft_img, cv2.COLORMAP_VIRIDIS)
        
        phase_img = ((self.completed_phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        display[:h, w:2*w] = cv2.applyColorMap(phase_img, cv2.COLORMAP_HSV)
        
        recon_img = self._normalize_to_uint8(self.reconstructed)
        display[:h, 2*w:] = cv2.applyColorMap(recon_img, cv2.COLORMAP_PLASMA)
        
        # Row 2: Hidden Structure, Low Freq, Phase Coherence
        hidden_img = self._normalize_to_uint8(self.hidden_structure)
        display[h:, :w] = cv2.applyColorMap(hidden_img, cv2.COLORMAP_INFERNO)
        
        # Low frequency content
        low_mask = self.r_grid < 15
        low_spec = self.completed_spectrum * low_mask
        low_spatial = np.real(ifft2(ifftshift(low_spec)))
        low_img = self._normalize_to_uint8(low_spatial)
        display[h:, w:2*w] = cv2.applyColorMap(low_img, cv2.COLORMAP_TWILIGHT)
        
        # Phase coherence
        if len(self.phase_history) > 5:
            phases = np.array(list(self.phase_history)[-10:])
            coherence = np.abs(np.mean(np.exp(1j * phases), axis=0))
            coh_img = (coherence * 255).astype(np.uint8)
        else:
            coh_img = np.zeros((h, w), dtype=np.uint8)
        display[h:, 2*w:] = cv2.applyColorMap(coh_img, cv2.COLORMAP_JET)
        
        return display
    
    def get_display_image(self):
        display = self._render_combined()
        h, w = self.size, self.size
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Completed FFT", (5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Phase", (w+5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Reconstructed", (2*w+5, 15), font, 0.35, (0,255,255), 1)
        cv2.putText(display, "Hidden (Low-F)", (5, h+15), font, 0.35, (255,150,50), 1)
        cv2.putText(display, "Low Freq Only", (w+5, h+15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Phase Coherence", (2*w+5, h+15), font, 0.35, (255,255,255), 1)
        
        # Stats
        total_e = np.sum(np.abs(self.completed_spectrum)**2)
        low_mask = self.r_grid < 15
        low_e = np.sum(np.abs(self.completed_spectrum[low_mask])**2)
        low_ratio = low_e / (total_e + 1e-10)
        
        stats = f"LowF: {low_ratio*100:.1f}% Rings: [{self.ring_boosts[0]:.1f},{self.ring_boosts[1]:.1f},{self.ring_boosts[2]:.1f},{self.ring_boosts[3]:.1f}]"
        cv2.putText(display, stats, (5, h*2-5), font, 0.3, (200,200,200), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        phase_modes = [
            ('Random', '0'),
            ('From Gradient', '1'),
            ('Spiral', '2'),
            ('Temporal Continuity', '3'),
        ]
        return [
            # Main controls
            ("DC Injection", "base_dc", self.base_dc, None),
            ("Low Freq Boost", "base_low_boost", self.base_low_boost, None),
            ("Iterations", "base_iterations", self.base_iterations, None),
            ("Regularization", "base_regularization", self.base_regularization, None),
            ("Phase Mode", "phase_mode", str(self.phase_mode), phase_modes),
            
            # Ring boosts
            ("Ring 1 (DC area)", "ring_boost_0", self.ring_boosts[0], None),
            ("Ring 2 (Low freq)", "ring_boost_1", self.ring_boosts[1], None),
            ("Ring 3 (Mid freq)", "ring_boost_2", self.ring_boosts[2], None),
            ("Ring 4 (High freq)", "ring_boost_3", self.ring_boosts[3], None),
        ]
    
    def set_config_options(self, options):
        for key, value in options.items():
            if key == 'phase_mode':
                self.phase_mode = int(value)
            elif key.startswith('ring_boost_'):
                idx = int(key.split('_')[-1])
                self.ring_boosts[idx] = float(value)
            elif hasattr(self, key):
                setattr(self, key, type(getattr(self, key))(value))