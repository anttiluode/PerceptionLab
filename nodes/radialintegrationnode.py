"""
RadialIntegrationNode - The Consciousness Cochlea
==================================================

"What stands at the center when all waves have summed?"

The biological cochlea unfolds frequency into space - a linear Fourier transform
in flesh, reading high-to-low along the basilar membrane spiral.

This node does something different: RADIAL integration from edge toward center.
Like a cochlea folded into a disc, or a thalamocortical loop collapsing 
probability distributions to a "best estimate."

The hypothesis:
- Chaos lives at the edges (high frequency, differentiated, features)
- Qualia emerges at the center (integrated, bound, unified)
- The transform between them is a standing wave that peaks at resonance

From Wright & Bourke: O(P,t) ↔ o(±p², t - |P-p|/ν)
The p² term compresses - points opposite each other map to the same location.
This is a 2-to-1 projection. Information collapses.

From Antti's insight: Energy radiates from center in concentric rings.
What if we reverse the flow? Integrate INWARD?

MECHANISM:
1. Sample the input field in concentric rings (like cochlear tonotopy)
2. Each ring integrates its content (sum, mean, or phase-locked average)
3. Rings propagate inward with time delays (like traveling waves)
4. The CENTER accumulates what survives integration
5. Resonant modes appear as stable patterns at specific radii

The "qualia point" is the center - where all frequencies have been
integrated, all phases have been summed, all chaos has collapsed.

INPUTS:
- chaos_field: The differentiated input (from gradient, ephaptic field, etc.)
- phase_field: Optional phase information for phase-locked integration
- integration_strength: How much each ring collapses information
- wave_speed: How fast integration propagates inward
- resonance_q: Quality factor - how sharply tuned the resonances are

OUTPUTS:
- qualia_point: The central integrated value (scalar or small region)
- radial_profile: Integration level at each radius (the "basilar membrane")
- standing_wave: The resonant pattern that emerges
- collapsed_field: The full field after radial integration
- resonance_spectrum: Which radial frequencies resonate
- binding_map: Where integration is strongest

Created: December 2025
For Antti's quest to find the transform from chaos to qualia
"""

import numpy as np
import cv2
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import hilbert
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class RadialIntegrationNode(BaseNode):
    """
    The Consciousness Cochlea - integrates from edge to center,
    collapsing chaos into qualia through radial wave propagation.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Radial Integration"
    NODE_COLOR = QtGui.QColor(255, 100, 150)  # Rose - the heart of it
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'chaos_field': 'image',          # The differentiated input
            'phase_field': 'image',          # Optional phase for coherent integration
            'integration_strength': 'signal', # How much to collapse per ring
            'wave_speed': 'signal',          # Propagation speed inward
            'resonance_q': 'signal',         # Quality factor of resonances
            'center_x': 'signal',            # Optional center offset
            'center_y': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            # Core outputs
            'qualia_point': 'signal',        # The central integrated value
            'qualia_region': 'image',        # Small region around center
            'radial_profile': 'spectrum',    # Integration vs radius
            'standing_wave': 'image',        # The resonant pattern
            'collapsed_field': 'image',      # Full integrated field
            
            # Analysis
            'resonance_spectrum': 'spectrum', # Radial frequency content
            'binding_map': 'image',          # Where integration is strongest
            'phase_coherence_radial': 'spectrum', # Phase lock by radius
            'traveling_wave': 'image',       # The inward wave visualization
            
            # Combined view
            'combined_view': 'image',
            
            # Signals
            'peak_radius': 'signal',         # Where resonance peaks
            'center_energy': 'signal',       # Energy at center
            'integration_completeness': 'signal', # How collapsed is it
            'dominant_ring': 'signal'        # Which ring dominates
        }
        
        self.size = 128
        self.center = self.size // 2
        self.n_rings = 64  # Number of concentric integration rings
        
        # Build coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        self.y_grid, self.x_grid = np.mgrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        self.theta_grid = np.arctan2(y - self.center, x - self.center)
        
        # Ring masks for integration
        self._build_ring_system()
        
        # === STATE ===
        # The "basilar membrane" - accumulated integration at each radius
        self.radial_accumulator = np.zeros(self.n_rings)
        self.radial_phase = np.zeros(self.n_rings)
        
        # Standing wave state
        self.standing_wave = np.zeros((self.size, self.size))
        self.standing_wave_complex = np.zeros((self.size, self.size), dtype=np.complex128)
        
        # Traveling wave buffer (for visualization)
        self.wave_history = deque(maxlen=32)
        
        # Output fields
        self.collapsed_field = np.zeros((self.size, self.size))
        self.binding_map = np.zeros((self.size, self.size))
        self.qualia_region = np.zeros((16, 16))
        
        # Metrics
        self.qualia_point = 0.0
        self.peak_radius = 0.0
        self.center_energy = 0.0
        
        # === PARAMETERS ===
        self.base_integration = 0.3      # How much to integrate per ring
        self.base_wave_speed = 2.0       # Rings per timestep inward
        self.base_resonance_q = 5.0      # Resonance sharpness
        self.damping = 0.95              # Wave damping
        self.phase_coupling = 0.5        # How much phase matters
        
        # Resonance frequencies (which ring distances resonate)
        self.resonance_radii = [8, 16, 24, 32]  # Will find these adaptively
        
        self.t = 0
    
    def _build_ring_system(self):
        """Build the concentric ring system for cochlea-like sampling."""
        self.ring_masks = []
        self.ring_indices = []
        
        max_r = self.center
        ring_width = max_r / self.n_rings
        
        for i in range(self.n_rings):
            r_inner = i * ring_width
            r_outer = (i + 1) * ring_width
            mask = (self.r_grid >= r_inner) & (self.r_grid < r_outer)
            self.ring_masks.append(mask)
            
            # Store indices for fast sampling
            indices = np.where(mask)
            self.ring_indices.append(indices)
        
        # Build radial coordinate for each pixel (which ring it belongs to)
        self.ring_assignment = np.clip(
            (self.r_grid / (self.center / self.n_rings)).astype(int),
            0, self.n_rings - 1
        )
    
    def _sample_ring(self, field, ring_idx, phase_field=None):
        """
        Sample a ring and compute its integrated value.
        This is like a hair cell reading the basilar membrane displacement.
        """
        mask = self.ring_masks[ring_idx]
        if np.sum(mask) == 0:
            return 0.0, 0.0
        
        values = field[mask]
        
        # Basic integration: mean value
        mean_val = np.mean(values)
        
        # If we have phase information, do phase-coherent integration
        if phase_field is not None:
            phases = phase_field[mask]
            # Phase-locked average: weight by phase coherence
            phasors = np.exp(1j * phases)
            coherence = np.abs(np.mean(phasors))
            mean_phasor = np.mean(phasors)
            phase = np.angle(mean_phasor)
            
            # Coherent integration amplifies aligned phases
            mean_val = mean_val * (1 + self.phase_coupling * coherence)
            return mean_val, phase
        
        return mean_val, 0.0
    
    def _propagate_inward(self, radial_values, radial_phases, speed):
        """
        Propagate integration wave inward, like traveling wave on basilar membrane.
        But reversed - from edge toward center.
        """
        n = len(radial_values)
        new_values = np.zeros(n)
        new_phases = np.zeros(n)
        
        # Integration propagates from outer to inner rings
        shift = int(speed)
        frac = speed - shift
        
        for i in range(n):
            # Source rings (outer)
            src_idx = min(i + shift, n - 1)
            src_idx_next = min(i + shift + 1, n - 1)
            
            # Interpolated value
            outer_val = radial_values[src_idx] * (1 - frac) + radial_values[src_idx_next] * frac
            outer_phase = radial_phases[src_idx] * (1 - frac) + radial_phases[src_idx_next] * frac
            
            # Integration: accumulate from outer rings with damping
            # Inner rings accumulate more (they receive from all outer rings)
            accumulation_factor = 1.0 + (n - i) / n * self.base_integration
            
            new_values[i] = (
                self.radial_accumulator[i] * self.damping + 
                outer_val * accumulation_factor * (1 - self.damping)
            )
            new_phases[i] = outer_phase
        
        return new_values, new_phases
    
    def _apply_resonance(self, radial_values, q_factor):
        """
        Apply resonance - certain radii resonate and amplify.
        Like the basilar membrane's frequency selectivity.
        """
        resonant_values = radial_values.copy()
        
        # Find peaks in the radial profile (natural resonances)
        # These are where the system wants to ring
        profile_smooth = gaussian_filter(radial_values, sigma=2)
        
        # Resonance amplification at specific radii
        for res_r in self.resonance_radii:
            if res_r < self.n_rings:
                # Gaussian resonance peak
                resonance = np.exp(-((np.arange(self.n_rings) - res_r)**2) / (2 * (q_factor**2)))
                resonant_values += radial_values * resonance * 0.5
        
        return resonant_values
    
    def _reconstruct_field(self, radial_values, radial_phases):
        """
        Reconstruct 2D field from radial profile.
        This creates the "collapsed" view - chaos integrated to structure.
        """
        field = np.zeros((self.size, self.size))
        
        for i in range(self.n_rings):
            mask = self.ring_masks[i]
            # Each ring gets its integrated value
            # Optionally modulated by phase
            if self.phase_coupling > 0:
                phase_mod = np.cos(self.theta_grid[mask] + radial_phases[i])
                field[mask] = radial_values[i] * (1 + 0.3 * phase_mod)
            else:
                field[mask] = radial_values[i]
        
        return field
    
    def _compute_standing_wave(self, field):
        """
        Compute standing wave pattern from the integration dynamics.
        Standing waves appear where inward and "reflected" waves interfere.
        """
        # Analytic signal for phase extraction
        rows_analytic = np.zeros_like(field, dtype=np.complex128)
        for i in range(self.size):
            rows_analytic[i, :] = hilbert(field[i, :])
        
        # Radial analytic signal (approximate)
        # This captures the standing wave pattern
        self.standing_wave_complex = self.standing_wave_complex * 0.9 + rows_analytic * 0.1
        self.standing_wave = np.abs(self.standing_wave_complex)
    
    def _compute_binding_map(self, field, phase_field):
        """
        Where is integration strongest? Where do features bind?
        """
        # Binding is strong where:
        # 1. Local variance is low (features have merged)
        # 2. Phase coherence is high (things are synchronized)
        
        # Local variance
        field_blur = gaussian_filter(field, sigma=3)
        field_var = gaussian_filter((field - field_blur)**2, sigma=3)
        low_variance = 1.0 / (1.0 + field_var * 10)
        
        # Phase coherence (if available)
        if phase_field is not None:
            phase_blur = gaussian_filter(np.cos(phase_field), sigma=3)
            coherence = np.abs(phase_blur)
        else:
            coherence = np.ones_like(field)
        
        self.binding_map = low_variance * coherence
    
    def _find_resonance_spectrum(self, radial_values):
        """
        FFT of radial profile gives resonance spectrum.
        Which radial frequencies are present?
        """
        spectrum = np.abs(fft(radial_values))[:self.n_rings // 2]
        return spectrum
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        chaos = self.get_blended_input('chaos_field', 'first')
        phase = self.get_blended_input('phase_field', 'first')
        
        int_strength = self.get_blended_input('integration_strength', 'sum')
        wave_spd = self.get_blended_input('wave_speed', 'sum')
        res_q = self.get_blended_input('resonance_q', 'sum')
        
        cx = self.get_blended_input('center_x', 'sum')
        cy = self.get_blended_input('center_y', 'sum')
        
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.radial_accumulator.fill(0)
            self.radial_phase.fill(0)
            self.standing_wave.fill(0)
            return
        
        if chaos is None:
            return
        
        # Apply parameters
        if int_strength is not None:
            self.base_integration = np.clip(int_strength, 0.01, 1.0)
        if wave_spd is not None:
            self.base_wave_speed = np.clip(wave_spd, 0.1, 10.0)
        if res_q is not None:
            self.base_resonance_q = np.clip(res_q, 1.0, 20.0)
        
        # Normalize input
        if chaos.dtype == np.uint8:
            chaos = chaos.astype(np.float32) / 255.0
        if len(chaos.shape) == 3:
            chaos = np.mean(chaos, axis=2)
        if chaos.shape[0] != self.size:
            chaos = cv2.resize(chaos, (self.size, self.size))
        
        # Handle optional phase field
        phase_field = None
        if phase is not None:
            if phase.dtype == np.uint8:
                phase = phase.astype(np.float32) / 255.0 * 2 * np.pi - np.pi
            if phase.shape[0] != self.size:
                phase = cv2.resize(phase, (self.size, self.size))
            phase_field = phase
        
        # === SAMPLE ALL RINGS (cochlea-like) ===
        ring_values = np.zeros(self.n_rings)
        ring_phases = np.zeros(self.n_rings)
        
        for i in range(self.n_rings):
            val, ph = self._sample_ring(chaos, i, phase_field)
            ring_values[i] = val
            ring_phases[i] = ph
        
        # Store current state for traveling wave visualization
        self.wave_history.append(ring_values.copy())
        
        # === PROPAGATE INWARD ===
        self.radial_accumulator, self.radial_phase = self._propagate_inward(
            ring_values, ring_phases, self.base_wave_speed
        )
        
        # === APPLY RESONANCE ===
        resonant_profile = self._apply_resonance(
            self.radial_accumulator, self.base_resonance_q
        )
        
        # === RECONSTRUCT COLLAPSED FIELD ===
        self.collapsed_field = self._reconstruct_field(resonant_profile, self.radial_phase)
        
        # === COMPUTE STANDING WAVE ===
        self._compute_standing_wave(self.collapsed_field)
        
        # === COMPUTE BINDING MAP ===
        self._compute_binding_map(self.collapsed_field, phase_field)
        
        # === EXTRACT QUALIA POINT (the center) ===
        center_region = self.collapsed_field[
            self.center-8:self.center+8,
            self.center-8:self.center+8
        ]
        self.qualia_region = center_region.copy()
        self.qualia_point = float(np.mean(center_region))
        self.center_energy = float(np.sum(center_region**2))
        
        # === FIND PEAK RESONANCE RADIUS ===
        self.peak_radius = float(np.argmax(resonant_profile))
        
        # === UPDATE ADAPTIVE RESONANCES ===
        # Find natural peaks in the profile
        profile_smooth = gaussian_filter(resonant_profile, sigma=2)
        gradient = np.gradient(profile_smooth)
        peaks = []
        for i in range(1, len(gradient) - 1):
            if gradient[i-1] > 0 and gradient[i+1] < 0:
                peaks.append(i)
        if len(peaks) >= 4:
            self.resonance_radii = peaks[:4]
    
    def get_output(self, port_name):
        if port_name == 'qualia_point':
            return float(self.qualia_point)
        
        elif port_name == 'qualia_region':
            region = self.qualia_region
            if region.max() > region.min():
                norm = (region - region.min()) / (region.max() - region.min())
            else:
                norm = region
            return (norm * 255).astype(np.uint8)
        
        elif port_name == 'radial_profile':
            return self.radial_accumulator.astype(np.float32)
        
        elif port_name == 'standing_wave':
            return self._normalize_to_uint8(self.standing_wave)
        
        elif port_name == 'collapsed_field':
            return self._normalize_to_uint8(self.collapsed_field)
        
        elif port_name == 'resonance_spectrum':
            return self._find_resonance_spectrum(self.radial_accumulator).astype(np.float32)
        
        elif port_name == 'binding_map':
            return self._normalize_to_uint8(self.binding_map)
        
        elif port_name == 'phase_coherence_radial':
            # Phase coherence at each radius
            coherence = np.abs(np.exp(1j * self.radial_phase))
            return coherence.astype(np.float32)
        
        elif port_name == 'traveling_wave':
            return self._render_traveling_wave()
        
        elif port_name == 'combined_view':
            return self._render_combined()
        
        elif port_name == 'peak_radius':
            return float(self.peak_radius)
        
        elif port_name == 'center_energy':
            return float(self.center_energy)
        
        elif port_name == 'integration_completeness':
            # How much has chaos collapsed?
            if len(self.wave_history) > 0:
                outer = np.mean([w[-10:].mean() for w in self.wave_history])
                inner = np.mean([w[:10].mean() for w in self.wave_history])
                if outer > 0:
                    return float(inner / outer)
            return 0.0
        
        elif port_name == 'dominant_ring':
            return float(np.argmax(self.radial_accumulator))
        
        return None
    
    def _normalize_to_uint8(self, arr):
        arr = np.nan_to_num(arr)
        if arr.max() == arr.min():
            return np.zeros((self.size, self.size), dtype=np.uint8)
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        return (norm * 255).astype(np.uint8)
    
    def _render_traveling_wave(self):
        """Visualize the traveling wave history."""
        h, w = 64, self.n_rings
        img = np.zeros((h, w), dtype=np.uint8)
        
        if len(self.wave_history) > 0:
            history = np.array(list(self.wave_history))
            # Normalize
            if history.max() > history.min():
                history = (history - history.min()) / (history.max() - history.min())
            # Resize to fit
            history_resized = cv2.resize(history, (w, h))
            img = (history_resized * 255).astype(np.uint8)
        
        return cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
    
    def _render_combined(self):
        """Render 2x3 combined view."""
        h, w = self.size, self.size
        display = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # Row 1: Collapsed Field, Standing Wave, Binding Map
        collapsed_img = self._normalize_to_uint8(self.collapsed_field)
        display[:h, :w] = cv2.applyColorMap(collapsed_img, cv2.COLORMAP_VIRIDIS)
        
        standing_img = self._normalize_to_uint8(self.standing_wave)
        display[:h, w:2*w] = cv2.applyColorMap(standing_img, cv2.COLORMAP_PLASMA)
        
        binding_img = self._normalize_to_uint8(self.binding_map)
        display[:h, 2*w:] = cv2.applyColorMap(binding_img, cv2.COLORMAP_HOT)
        
        # Row 2: Radial Profile, Traveling Wave, Qualia Region (enlarged)
        # Radial profile as bars
        profile_img = np.zeros((h, w, 3), dtype=np.uint8)
        profile = self.radial_accumulator / (self.radial_accumulator.max() + 1e-10)
        for i in range(min(self.n_rings, w)):
            bar_h = int(profile[i] * h * 0.9)
            x = int(i * w / self.n_rings)
            color = cv2.applyColorMap(np.array([[int(profile[i] * 255)]], dtype=np.uint8), 
                                       cv2.COLORMAP_TWILIGHT)[0, 0].tolist()
            cv2.rectangle(profile_img, (x, h - bar_h), (x + w//self.n_rings, h), color, -1)
        # Mark resonance radii
        for res_r in self.resonance_radii:
            x = int(res_r * w / self.n_rings)
            cv2.line(profile_img, (x, 0), (x, 20), (0, 255, 255), 1)
        display[h:, :w] = profile_img
        
        # Traveling wave
        travel_img = self._render_traveling_wave()
        travel_resized = cv2.resize(travel_img, (w, h))
        display[h:, w:2*w] = travel_resized
        
        # Qualia region (enlarged)
        if self.qualia_region.size > 0:
            qualia_norm = self.qualia_region.copy()
            if qualia_norm.max() > qualia_norm.min():
                qualia_norm = (qualia_norm - qualia_norm.min()) / (qualia_norm.max() - qualia_norm.min())
            qualia_img = (qualia_norm * 255).astype(np.uint8)
            qualia_colored = cv2.applyColorMap(qualia_img, cv2.COLORMAP_INFERNO)
            qualia_large = cv2.resize(qualia_colored, (w, h), interpolation=cv2.INTER_NEAREST)
            display[h:, 2*w:] = qualia_large
        
        return display
    
    def get_display_image(self):
        display = self._render_combined()
        h, w = self.size, self.size
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Collapsed Field", (5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Standing Wave", (w+5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Binding Map", (2*w+5, 15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Radial Profile", (5, h+15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Traveling Wave", (w+5, h+15), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "QUALIA POINT", (2*w+5, h+15), font, 0.35, (0,255,255), 1)
        
        # Stats
        q = self.qualia_point
        peak = self.peak_radius
        energy = self.center_energy
        
        stats = f"Qualia:{q:.3f} Peak@r={peak:.0f} CenterE:{energy:.2f}"
        cv2.putText(display, stats, (5, h*2-5), font, 0.3, (200,200,200), 1)
        
        # Draw center marker on collapsed field
        cv2.circle(display, (self.center, self.center), 3, (255, 255, 0), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Integration Strength", "base_integration", self.base_integration, None),
            ("Wave Speed (inward)", "base_wave_speed", self.base_wave_speed, None),
            ("Resonance Q", "base_resonance_q", self.base_resonance_q, None),
            ("Damping", "damping", self.damping, None),
            ("Phase Coupling", "phase_coupling", self.phase_coupling, None),
            ("Number of Rings", "n_rings", self.n_rings, None),
        ]
    
    def set_config_options(self, options):
        rebuild_rings = False
        for key, value in options.items():
            if key == 'n_rings':
                new_n = int(value)
                if new_n != self.n_rings:
                    self.n_rings = new_n
                    rebuild_rings = True
            elif hasattr(self, key):
                setattr(self, key, type(getattr(self, key))(value))
        
        if rebuild_rings:
            self._build_ring_system()
            self.radial_accumulator = np.zeros(self.n_rings)
            self.radial_phase = np.zeros(self.n_rings)