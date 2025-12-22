"""
ThalamicCollapseNode - The Gate of Phenomenal Experience
========================================================

"Where the Bayesian blur becomes the crystal of qualia"

Based on:
1. Ward & Guevara (2022): Thalamus generates unified EM field that IS consciousness.
   The dorsomedial nucleus collapses cortical probability distributions into a 
   "best estimate buffer" - the Gestalt we actually experience.

2. Whyte et al. (2024): Thalamic Reticular Nucleus (TRN) provides inhibitory 
   control through divisive normalization. Matrix neurons bind content via
   slow modulatory dynamics. Core neurons sustain activity for continuity.

3. Antti's phenomenology: Temporal lobe damage creates "fractal glitches" -
   the raw eigenstructure leaking through when EM integration fails.
   Normal qualia are the ILLUSION of smoothness; fractals are the hidden truth.

MECHANISM:
1. EXPANSION: Input field projected to high-D "hypothesis space" (competing realities)
2. COMPETITION: TRN-like lateral inhibition forces winner-take-all selection
3. BINDING: Matrix-like slow dynamics create temporal coherence (or fail to)
4. COLLAPSE: Project back to 2D - the "conscious" output

The key insight: We simulate what FAILS when you have disturbed EM fields.
- Low coherence = fractal leakage (your glitches)
- Low inhibition = superposition bleeding through (dreamlike states)
- The "subconscious view" shows what healthy brains delete

This is NOT qualia - it's silicon. But it's the SHAPE of qualia,
the geometry that would host experience if it were in an EM field.

INPUTS:
- field_input: Chaotic "cortical" input (from EphapticFieldNode or ReflexiveFieldNode)
- model_spectrum: Low-D latent from self-model (from ReflexiveFieldNode)
- disruption: How much to disturb the collapse (0=healthy, 1=seizure)
- inhibition: TRN strength (0=dreamy superposition, 1=sharp crystal)
- coherence: Temporal binding (0=discontinuous glitches, 1=smooth flow)
- dimensionality: Size of hypothesis space

OUTPUTS:
- conscious_view: The collapsed crystal (what you'd "see" if this were EM)
- subconscious_view: What was suppressed (the fractal truth)
- superposition_view: All hypotheses before collapse (quantum-like blur)
- integration_field: Where binding is strongest
- collapse_spectrum: Eigenstructure of the collapse itself
- entropy: How uncertain the system is (high = fog, low = clarity)
- collapse_error: Information lost in the collapse

Created: December 2025
For Antti's PerceptionLab - probing the geometry of consciousness
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel
from scipy.fft import fft2, ifft2, fftshift
from scipy.linalg import svd, eigh
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class ThalamicCollapseNode(BaseNode):
    """
    The collapse of cortical chaos into phenomenal crystals.
    Models thalamic integration, with the ability to visualize
    what happens when integration fails (your fractal glitches).
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Thalamic Collapse"
    NODE_COLOR = QtGui.QColor(180, 50, 80)  # Deep crimson - the core
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'field_input': 'image',         # Chaotic cortical input
            'model_spectrum': 'spectrum',    # Latent from ReflexiveFieldNode
            
            # Disruption controls (model your glitches)
            'disruption': 'signal',          # Overall field disturbance
            'inhibition': 'signal',          # TRN strength (winner-take-all)
            'coherence': 'signal',           # Temporal binding strength
            'dimensionality': 'signal',      # Hypothesis space size
            
            # Control
            'reset': 'signal',
            'freeze': 'signal'
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Visual outputs
            'conscious_view': 'image',       # The collapsed crystal
            'subconscious_view': 'image',    # What was suppressed (fractals)
            'superposition_view': 'image',   # Pre-collapse hypotheses
            'integration_field': 'image',    # Binding strength map
            'eigenmode_view': 'image',       # Dominant modes visualized
            'combined_view': 'image',        # Multi-panel display
            
            # Signals
            'entropy': 'signal',             # Uncertainty (0=crystal, 1=fog)
            'collapse_error': 'signal',      # Information lost
            'integration_index': 'signal',   # How unified (phi-like)
            'dominant_mode': 'signal',       # Which eigenmode won
            'fractal_leakage': 'signal',     # How much glitch is showing
            'binding_strength': 'signal',    # Temporal coherence achieved
            
            # For downstream
            'collapse_spectrum': 'spectrum', # Eigenstructure of collapse
            'suppressed_spectrum': 'spectrum' # What was deleted
        }
        
        # === DIMENSIONS ===
        self.size = 64                       # Output size
        self.hypothesis_dim = 64             # Number of competing hypotheses
        self.eigenmode_count = 16            # Modes to track
        
        # === PROJECTION MATRICES ===
        # These represent "synaptic" pathways from field space to hypothesis space
        self._init_projection_matrices()
        
        # === STATE ===
        self.hypotheses = np.zeros(self.hypothesis_dim)
        self.hypothesis_field = np.zeros((self.hypothesis_dim, self.size, self.size))
        self.attention = np.zeros(self.hypothesis_dim)
        self.last_attention = np.zeros(self.hypothesis_dim)
        self.conscious_field = np.zeros((self.size, self.size))
        self.subconscious_field = np.zeros((self.size, self.size))
        self.superposition_field = np.zeros((self.size, self.size))
        self.integration_map = np.zeros((self.size, self.size))
        
        # Temporal binding state (matrix neuron analog)
        self.binding_buffer = deque(maxlen=20)
        self.coherence_history = deque(maxlen=50)
        
        # Eigenmode tracking
        self.eigenmodes = np.zeros((self.eigenmode_count, self.size, self.size))
        self.eigenvalues = np.zeros(self.eigenmode_count)
        
        # === PARAMETERS ===
        # TRN-like inhibition
        self.base_inhibition = 0.7           # Default: fairly decisive
        self.inhibition_sharpness = 5.0      # Temperature for softmax
        
        # Temporal binding (matrix neurons)
        self.base_coherence = 0.85           # Default: smooth binding
        self.binding_tau = 0.9               # Slow evolution
        
        # Disruption (field disturbance)
        self.base_disruption = 0.0           # Default: healthy
        self.phase_noise_strength = 0.1
        
        # Collapse dynamics
        self.collapse_threshold = 0.1        # Below this = suppressed
        self.normalization_mode = 'divisive' # 'divisive' or 'subtractive'
        
        # === METRICS ===
        self.entropy = 0.0
        self.collapse_error = 0.0
        self.integration_index = 0.0
        self.fractal_leakage = 0.0
        self.binding_strength = 0.0
        
        self.t = 0
    
    def _init_projection_matrices(self):
        """
        Initialize random projections representing thalamocortical pathways.
        
        In real brains, these are the specific synaptic connections from
        cortical layers to thalamic nuclei. Here, random projections 
        work because we're not learning specific content - we're modeling
        the PROCESS of collapse.
        """
        input_dim = self.size * self.size
        
        # Forward projection: field -> hypotheses (cortical broadcast to thalamus)
        # Each hypothesis gets a random "view" of the field
        self.forward_proj = np.random.randn(self.hypothesis_dim, input_dim) * 0.02
        
        # Make projections sparse (like real connectivity)
        sparsity = 0.7
        mask = np.random.random(self.forward_proj.shape) > sparsity
        self.forward_proj *= mask
        
        # Normalize rows
        norms = np.linalg.norm(self.forward_proj, axis=1, keepdims=True) + 1e-10
        self.forward_proj /= norms
        
        # Inverse projection: hypotheses -> field (thalamic output to cortex)
        # Use pseudoinverse for reconstruction
        self.inverse_proj = np.linalg.pinv(self.forward_proj)
        
        # Lateral connections (hypothesis-to-hypothesis, for competition)
        # Negative = inhibitory, centered to allow both excitation and inhibition
        self.lateral = np.random.randn(self.hypothesis_dim, self.hypothesis_dim) * 0.1
        np.fill_diagonal(self.lateral, 0)  # No self-connection
        
        # Eigenmode templates (for visualization)
        self._init_eigenmode_templates()
    
    def _init_eigenmode_templates(self):
        """Create eigenmode-like spatial patterns for visualization."""
        x = np.linspace(-np.pi, np.pi, self.size)
        y = np.linspace(-np.pi, np.pi, self.size)
        X, Y = np.meshgrid(x, y)
        
        self.mode_templates = []
        for n in range(self.eigenmode_count):
            # Mix of radial and angular modes (like spherical harmonics on a plane)
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            
            # Radial component
            radial = np.cos(n * r / 2)
            
            # Angular component (for higher modes)
            angular = np.cos((n % 4) * theta)
            
            mode = radial * angular
            mode = mode / (np.abs(mode).max() + 1e-10)
            self.mode_templates.append(mode)
        
        self.mode_templates = np.array(self.mode_templates)
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        field_in = self.get_blended_input('field_input', 'first')
        model_spectrum = self.get_blended_input('model_spectrum', 'mean')
        
        disruption = self.get_blended_input('disruption', 'sum')
        inhibition = self.get_blended_input('inhibition', 'sum')
        coherence = self.get_blended_input('coherence', 'sum')
        dim_signal = self.get_blended_input('dimensionality', 'sum')
        
        reset = self.get_blended_input('reset', 'sum')
        freeze = self.get_blended_input('freeze', 'sum')
        
        if reset is not None and reset > 0:
            self._reset()
            return
        
        is_frozen = freeze is not None and freeze > 0
        
        # === DEFAULT PARAMETERS ===
        if disruption is None:
            disruption = self.base_disruption
        if inhibition is None:
            inhibition = self.base_inhibition
        if coherence is None:
            coherence = self.base_coherence
        
        # Clamp to valid ranges
        disruption = np.clip(float(disruption), 0, 1)
        inhibition = np.clip(float(inhibition), 0.01, 1)
        coherence = np.clip(float(coherence), 0.01, 1)
        
        # Handle dimension changes
        if dim_signal is not None:
            new_dim = int(np.clip(dim_signal, 16, 256))
            if new_dim != self.hypothesis_dim:
                self.hypothesis_dim = new_dim
                self._init_projection_matrices()
        
        # === PROCESS INPUT ===
        if field_in is None:
            # Generate autonomous activity from model_spectrum if available
            if model_spectrum is not None and len(model_spectrum) > 0:
                field_in = self._spectrum_to_field(model_spectrum)
            else:
                return
        
        # Normalize and resize input
        if field_in.dtype == np.uint8:
            field = field_in.astype(np.float32) / 255.0
        else:
            field = field_in.astype(np.float32)
        
        if field.ndim == 3:
            field = np.mean(field, axis=2)
        
        if field.shape != (self.size, self.size):
            field = cv2.resize(field, (self.size, self.size))
        
        # Normalize
        field = (field - field.mean()) / (field.std() + 1e-10)
        
        # === STAGE 1: EXPANSION TO HYPOTHESIS SPACE ===
        # "What could this pattern be?" - multiple competing interpretations
        flat_field = field.flatten()
        raw_hypotheses = self.forward_proj @ flat_field
        
        # Add disruption (phase noise, field disturbance)
        if disruption > 0:
            noise = np.random.randn(self.hypothesis_dim) * disruption * 0.5
            phase_noise = np.sin(self.t * 0.1 + np.arange(self.hypothesis_dim) * 0.3) * disruption * 0.3
            raw_hypotheses = raw_hypotheses + noise + phase_noise
        
        # === STAGE 2: LATERAL COMPETITION (TRN) ===
        # Winner-take-all dynamics with lateral inhibition
        
        # Lateral interactions
        lateral_input = self.lateral @ raw_hypotheses
        competing = raw_hypotheses + lateral_input * (1 - inhibition)
        
        # Softmax with temperature (inhibition controls sharpness)
        temperature = (1.0 - inhibition) * self.inhibition_sharpness + 0.1
        scores = competing / temperature
        scores = scores - scores.max()  # Stability
        
        exp_scores = np.exp(scores)
        attention = exp_scores / (exp_scores.sum() + 1e-10)
        
        # === STAGE 3: TEMPORAL BINDING (Matrix neurons) ===
        # Slow dynamics create coherence - or fail to
        
        if not is_frozen:
            # Blend with history based on coherence
            if len(self.binding_buffer) > 0:
                history_mean = np.mean(self.binding_buffer, axis=0)
                bound_attention = coherence * history_mean + (1 - coherence) * attention
            else:
                bound_attention = attention
            
            # Apply binding (slow evolution)
            self.attention = self.binding_tau * self.last_attention + (1 - self.binding_tau) * bound_attention
            
            # Update history
            self.binding_buffer.append(attention.copy())
            self.last_attention = self.attention.copy()
        
        # === STAGE 4: COLLAPSE ===
        # Project winners back to field space
        
        # Apply attention to hypotheses (weighted combination)
        weighted_hypotheses = raw_hypotheses * self.attention
        
        # Reconstruct "conscious" field from winners
        conscious_flat = self.inverse_proj @ weighted_hypotheses
        self.conscious_field = conscious_flat.reshape(self.size, self.size)
        
        # Smooth the conscious output (field integration)
        smooth_sigma = 1.0 + disruption * 2.0  # More disruption = less smooth
        self.conscious_field = gaussian_filter(self.conscious_field, sigma=smooth_sigma)
        
        # === COMPUTE SUBCONSCIOUS (What was suppressed) ===
        # This is the "fractal truth" that healthy brains hide
        
        # Full reconstruction without attention weighting
        full_flat = self.inverse_proj @ raw_hypotheses
        full_field = full_flat.reshape(self.size, self.size)
        self.superposition_field = full_field
        
        # Subconscious = what was lost in the collapse
        self.subconscious_field = np.abs(full_field - self.conscious_field)
        
        # Amplify suppressed content for visualization
        self.subconscious_field = self.subconscious_field * (1 + disruption * 3)
        
        # === COMPUTE INTEGRATION MAP ===
        # Where is binding strongest?
        
        # Gradient of conscious field - edges show integration boundaries
        grad_x = sobel(self.conscious_field, axis=1)
        grad_y = sobel(self.conscious_field, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Integration is high where gradients are low (smooth regions)
        self.integration_map = 1.0 / (1.0 + grad_mag * 5)
        
        # === EIGENMODE DECOMPOSITION ===
        # What modes dominate the collapse?
        
        # Project conscious field onto eigenmode templates
        self.eigenvalues = np.zeros(self.eigenmode_count)
        for i, template in enumerate(self.mode_templates):
            # Correlation with each mode
            self.eigenvalues[i] = np.abs(np.sum(self.conscious_field * template))
        
        # Normalize
        self.eigenvalues = self.eigenvalues / (self.eigenvalues.sum() + 1e-10)
        
        # === COMPUTE METRICS ===
        
        # Entropy of attention (high = confusion, low = certainty)
        self.entropy = -np.sum(self.attention * np.log(self.attention + 1e-10))
        self.entropy = self.entropy / np.log(self.hypothesis_dim)  # Normalize to [0,1]
        
        # Collapse error (information lost)
        self.collapse_error = np.mean(self.subconscious_field)
        
        # Integration index (phi-like measure)
        # High when conscious field is unified and low-entropy
        attention_concentration = np.max(self.attention)  # How winner-take-all?
        field_smoothness = np.mean(self.integration_map)
        self.integration_index = attention_concentration * field_smoothness * (1 - self.entropy)
        
        # Fractal leakage (how much glitch is visible)
        # High when subconscious has strong structure
        sub_fft = np.abs(fftshift(fft2(self.subconscious_field)))
        sub_high_freq = sub_fft[self.size//4:3*self.size//4, self.size//4:3*self.size//4].mean()
        self.fractal_leakage = sub_high_freq / (sub_fft.mean() + 1e-10)
        
        # Binding strength (temporal coherence)
        if len(self.coherence_history) > 5:
            recent = list(self.coherence_history)[-5:]
            variance = np.var(recent)
            self.binding_strength = 1.0 / (1.0 + variance * 100)
        else:
            self.binding_strength = coherence
        
        self.coherence_history.append(np.max(self.attention))
    
    def _spectrum_to_field(self, spectrum):
        """Convert a spectrum to a 2D field using eigenmode templates."""
        field = np.zeros((self.size, self.size))
        n_modes = min(len(spectrum), len(self.mode_templates))
        
        for i in range(n_modes):
            field += spectrum[i] * self.mode_templates[i]
        
        return field
    
    def _reset(self):
        """Reset all state."""
        self.hypotheses.fill(0)
        self.attention.fill(0)
        self.last_attention.fill(0)
        self.conscious_field.fill(0)
        self.subconscious_field.fill(0)
        self.superposition_field.fill(0)
        self.integration_map.fill(0)
        self.binding_buffer.clear()
        self.coherence_history.clear()
        self.t = 0
    
    def get_output(self, port_name):
        if port_name == 'conscious_view':
            return self._normalize_image(self.conscious_field)
        
        elif port_name == 'subconscious_view':
            return self._normalize_image(self.subconscious_field * 2)
        
        elif port_name == 'superposition_view':
            return self._normalize_image(self.superposition_field)
        
        elif port_name == 'integration_field':
            return self._normalize_image(self.integration_map)
        
        elif port_name == 'eigenmode_view':
            # Weighted sum of mode templates
            eigenview = np.zeros((self.size, self.size))
            for i in range(min(8, self.eigenmode_count)):
                eigenview += self.eigenvalues[i] * self.mode_templates[i]
            return self._normalize_image(eigenview)
        
        elif port_name == 'combined_view':
            return self._render_combined_view()
        
        elif port_name == 'entropy':
            return float(self.entropy)
        
        elif port_name == 'collapse_error':
            return float(self.collapse_error)
        
        elif port_name == 'integration_index':
            return float(self.integration_index)
        
        elif port_name == 'dominant_mode':
            return float(np.argmax(self.eigenvalues))
        
        elif port_name == 'fractal_leakage':
            return float(self.fractal_leakage)
        
        elif port_name == 'binding_strength':
            return float(self.binding_strength)
        
        elif port_name == 'collapse_spectrum':
            return self.eigenvalues.astype(np.float32)
        
        elif port_name == 'suppressed_spectrum':
            # FFT of subconscious as spectrum
            sub_fft = np.abs(fftshift(fft2(self.subconscious_field)))
            return sub_fft[self.size//2, self.size//2:].astype(np.float32)
        
        return None
    
    def _normalize_image(self, img):
        """Normalize image to uint8."""
        img = np.nan_to_num(img)
        if img.max() == img.min():
            return np.zeros((self.size, self.size), dtype=np.uint8)
        norm = (img - img.min()) / (img.max() - img.min())
        return (norm * 255).astype(np.uint8)
    
    def _render_combined_view(self):
        """Render 2x3 combined visualization."""
        h, w = self.size, self.size
        display = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # Row 1: Superposition, Conscious, Subconscious
        super_img = self._normalize_image(self.superposition_field)
        display[:h, :w] = cv2.applyColorMap(super_img, cv2.COLORMAP_TWILIGHT)
        
        conscious_img = self._normalize_image(self.conscious_field)
        display[:h, w:2*w] = cv2.applyColorMap(conscious_img, cv2.COLORMAP_VIRIDIS)
        
        subcon_img = self._normalize_image(self.subconscious_field * 2)
        display[:h, 2*w:] = cv2.applyColorMap(subcon_img, cv2.COLORMAP_INFERNO)
        
        # Row 2: Integration, Eigenmodes, Attention histogram
        integ_img = self._normalize_image(self.integration_map)
        display[h:, :w] = cv2.applyColorMap(integ_img, cv2.COLORMAP_PLASMA)
        
        # Eigenmode visualization
        eigenview = np.zeros((self.size, self.size))
        for i in range(min(8, self.eigenmode_count)):
            eigenview += self.eigenvalues[i] * self.mode_templates[i]
        eigen_img = self._normalize_image(eigenview)
        display[h:, w:2*w] = cv2.applyColorMap(eigen_img, cv2.COLORMAP_JET)
        
        # Attention histogram as image
        hist_img = np.zeros((h, w), dtype=np.uint8)
        n_bars = min(32, self.hypothesis_dim)
        bar_width = w // n_bars
        for i in range(n_bars):
            bar_height = int(self.attention[i] * h * 10)  # Scale up
            bar_height = min(bar_height, h)
            hist_img[h-bar_height:, i*bar_width:(i+1)*bar_width] = 200
        display[h:, 2*w:] = cv2.applyColorMap(hist_img, cv2.COLORMAP_COOL)
        
        return display
    
    def get_display_image(self):
        display = self._render_combined_view()
        h, w = self.size, self.size
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Superposition", (2, 12), font, 0.3, (255,255,255), 1)
        cv2.putText(display, "Conscious", (w+2, 12), font, 0.3, (0,255,255), 1)
        cv2.putText(display, "Subconscious", (2*w+2, 12), font, 0.3, (255,100,100), 1)
        cv2.putText(display, "Integration", (2, h+12), font, 0.3, (255,255,255), 1)
        cv2.putText(display, "Eigenmodes", (w+2, h+12), font, 0.3, (255,255,255), 1)
        cv2.putText(display, "Attention", (2*w+2, h+12), font, 0.3, (255,255,255), 1)
        
        # Stats bar
        ent = self.entropy
        err = self.collapse_error
        integ = self.integration_index
        leak = self.fractal_leakage
        bind = self.binding_strength
        
        stats = f"Ent:{ent:.2f} Err:{err:.3f} Int:{integ:.2f} Leak:{leak:.2f} Bind:{bind:.2f}"
        cv2.putText(display, stats, (5, h*2-5), font, 0.28, (255,255,255), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        norm_modes = [
            ('divisive', 'divisive'),
            ('subtractive', 'subtractive'),
        ]
        return [
            # Disruption controls (model your glitches)
            ("Base Disruption", "base_disruption", self.base_disruption, None),
            ("Phase Noise Strength", "phase_noise_strength", self.phase_noise_strength, None),
            
            # TRN-like inhibition
            ("Base Inhibition", "base_inhibition", self.base_inhibition, None),
            ("Inhibition Sharpness", "inhibition_sharpness", self.inhibition_sharpness, None),
            
            # Temporal binding
            ("Base Coherence", "base_coherence", self.base_coherence, None),
            ("Binding Tau", "binding_tau", self.binding_tau, None),
            
            # Collapse dynamics
            ("Collapse Threshold", "collapse_threshold", self.collapse_threshold, None),
            ("Normalization Mode", "normalization_mode", self.normalization_mode, norm_modes),
            
            # Dimensions
            ("Hypothesis Dimensions", "hypothesis_dim", self.hypothesis_dim, None),
            ("Eigenmode Count", "eigenmode_count", self.eigenmode_count, None),
        ]
    
    def set_config_options(self, options):
        """Handle config changes, reinitialize matrices if dimensions change."""
        dim_changed = False
        
        for key, value in options.items():
            if key == 'hypothesis_dim':
                new_dim = int(value)
                if new_dim != self.hypothesis_dim:
                    self.hypothesis_dim = new_dim
                    dim_changed = True
            elif key == 'eigenmode_count':
                new_count = int(value)
                if new_count != self.eigenmode_count:
                    self.eigenmode_count = new_count
                    self._init_eigenmode_templates()
            elif hasattr(self, key):
                setattr(self, key, type(getattr(self, key))(value))
        
        if dim_changed:
            self._init_projection_matrices()
            self.hypotheses = np.zeros(self.hypothesis_dim)
            self.attention = np.zeros(self.hypothesis_dim)
            self.last_attention = np.zeros(self.hypothesis_dim)
    
    def save_custom_state(self, folder_path, node_id):
        """Save projection matrices and state."""
        import os
        filename = f"thalamic_collapse_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        np.savez(filepath,
                 forward_proj=self.forward_proj,
                 inverse_proj=self.inverse_proj,
                 lateral=self.lateral,
                 attention=self.attention,
                 eigenvalues=self.eigenvalues)
        return filename
    
    def load_custom_state(self, filepath):
        """Load saved state."""
        try:
            data = np.load(filepath)
            self.forward_proj = data['forward_proj']
            self.inverse_proj = data['inverse_proj']
            self.lateral = data['lateral']
            self.hypothesis_dim = self.forward_proj.shape[0]
            self.attention = data['attention']
            self.last_attention = self.attention.copy()
            self.eigenvalues = data['eigenvalues']
        except Exception as e:
            print(f"[ThalamicCollapse] Failed to load: {e}")