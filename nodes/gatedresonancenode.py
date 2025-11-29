"""
Gated Resonance Node - Excitable Medium
Each pixel is a neuron: accumulate, threshold, fire, refractory.
The question: does harmonic selectivity survive discretization?
"""

import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.fft import fft2, fftshift

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class GatedResonanceNode(BaseNode):
    """
    Excitable Medium Resonance.
    
    Each pixel is a simple neuron:
    - Resting potential (0)
    - Accumulates input from neighbors + external drive
    - Fires when crossing threshold
    - Goes refractory (cannot fire for N steps)
    - Decays back to rest
    
    The field should self-organize into:
    - Spiral waves
    - Traveling pulses  
    - Frequency-locked oscillations
    - Maybe... resonant geometries?
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Gated Resonance (Excitable)"
    NODE_COLOR = QtGui.QColor(200, 100, 50)  # Neural orange
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'frequency_input': 'spectrum',      # Harmonic drive
            'threshold_mod': 'signal',          # Adjust excitability
            'coupling_mod': 'signal',           # Neighbor influence
            'reset': 'signal'
        }
        
        self.outputs = {
            'potential_map': 'image',           # Membrane potentials
            'spike_map': 'image',               # Current firings
            'refractory_map': 'image',          # Recovery state
            'eigen_image': 'image',             # FFT of activity
            
            'firing_rate': 'signal',            # Population activity
            'synchrony': 'signal',              # Phase coherence
            'eigenfrequencies': 'spectrum'      # For analysis chain
        }
        
        self.size = 128
        self.center = self.size // 2
        
        # === NEURON STATE (per pixel) ===
        # Membrane potential: 0 = rest, 1 = threshold
        self.potential = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Refractory timer: >0 means cannot fire
        self.refractory = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Last spike times (for phase analysis)
        self.last_spike = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Accumulated spikes for rate estimation
        self.spike_history = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === PARAMETERS ===
        self.threshold = 0.8            # Firing threshold
        self.refractory_period = 5      # Steps before can fire again
        self.leak = 0.05                # Passive decay toward rest
        self.coupling = 0.15            # Neighbor influence strength
        self.input_gain = 0.3           # External drive strength
        
        # Coupling kernel selection
        self.kernel_type = "Square 8"
        self._last_kernel_type = self.kernel_type
        self._build_kernel()
        
    def _build_kernel(self):
        """Build coupling kernel based on selected type."""
        
        if self.kernel_type == "Square 8":
            # Standard 8-neighbor, 4-fold symmetry
            self.kernel = np.array([
                [0.05, 0.1, 0.05],
                [0.1,  0.0, 0.1],
                [0.05, 0.1, 0.05]
            ], dtype=np.float32)
            
        elif self.kernel_type == "Cross 4":
            # Only cardinal directions - strong 4-fold
            self.kernel = np.array([
                [0.0, 0.25, 0.0],
                [0.25, 0.0, 0.25],
                [0.0, 0.25, 0.0]
            ], dtype=np.float32)
            
        elif self.kernel_type == "Diagonal 4":
            # Only diagonals - 45° rotated 4-fold
            self.kernel = np.array([
                [0.25, 0.0, 0.25],
                [0.0,  0.0, 0.0],
                [0.25, 0.0, 0.25]
            ], dtype=np.float32)
            
        elif self.kernel_type == "Hexagonal":
            # Approximate hex on square grid (offset rows)
            # 6-fold symmetry approximation
            self.kernel = np.array([
                [0.0,  0.15, 0.15, 0.0],
                [0.15, 0.0,  0.0,  0.15],
                [0.15, 0.0,  0.0,  0.15],
                [0.0,  0.15, 0.15, 0.0]
            ], dtype=np.float32)
            
        elif self.kernel_type == "Radial 12":
            # 5x5 kernel with distance-weighted coupling
            # Approximates circular symmetry
            k = np.zeros((5, 5), dtype=np.float32)
            center = 2
            for i in range(5):
                for j in range(5):
                    d = np.sqrt((i - center)**2 + (j - center)**2)
                    if 0.5 < d < 2.5:
                        k[i, j] = 1.0 / (d + 0.5)
            k[center, center] = 0
            k /= k.sum()  # Normalize
            self.kernel = k
            
        elif self.kernel_type == "Radial 24":
            # 7x7 kernel - more neighbors, smoother
            k = np.zeros((7, 7), dtype=np.float32)
            center = 3
            for i in range(7):
                for j in range(7):
                    d = np.sqrt((i - center)**2 + (j - center)**2)
                    if 0.5 < d < 3.5:
                        k[i, j] = 1.0 / (d + 0.5)
            k[center, center] = 0
            k /= k.sum()
            self.kernel = k
            
        elif self.kernel_type == "Mexican Hat":
            # Center-surround: excitation near, inhibition far
            k = np.zeros((7, 7), dtype=np.float32)
            center = 3
            for i in range(7):
                for j in range(7):
                    d = np.sqrt((i - center)**2 + (j - center)**2)
                    if d > 0:
                        # Difference of Gaussians
                        k[i, j] = np.exp(-d**2 / 2) - 0.5 * np.exp(-d**2 / 8)
            k[center, center] = 0
            # Normalize positive and negative separately
            pos = k.copy(); pos[pos < 0] = 0
            neg = k.copy(); neg[neg > 0] = 0
            if pos.sum() > 0: pos /= pos.sum()
            if neg.sum() < 0: neg /= abs(neg.sum())
            self.kernel = pos + neg * 0.3  # Weaker inhibition
            
        elif self.kernel_type == "Star 6":
            # 6-pointed star pattern
            k = np.zeros((7, 7), dtype=np.float32)
            center = 3
            # 6 directions at 60° intervals
            angles = [0, 60, 120, 180, 240, 300]
            for angle in angles:
                rad = np.radians(angle)
                for r in [1, 2, 3]:
                    i = int(center + r * np.sin(rad) + 0.5)
                    j = int(center + r * np.cos(rad) + 0.5)
                    if 0 <= i < 7 and 0 <= j < 7:
                        k[i, j] = 1.0 / r
            k[center, center] = 0
            if k.sum() > 0: k /= k.sum()
            self.kernel = k.astype(np.float32)
            
        elif self.kernel_type == "Star 5":
            # 5-pointed star pattern (pentagon)
            k = np.zeros((7, 7), dtype=np.float32)
            center = 3
            angles = [0, 72, 144, 216, 288]
            for angle in angles:
                rad = np.radians(angle)
                for r in [1, 2, 3]:
                    i = int(center + r * np.sin(rad) + 0.5)
                    j = int(center + r * np.cos(rad) + 0.5)
                    if 0 <= i < 7 and 0 <= j < 7:
                        k[i, j] = 1.0 / r
            k[center, center] = 0
            if k.sum() > 0: k /= k.sum()
            self.kernel = k.astype(np.float32)
            
        else:
            # Fallback to square 8
            self.kernel = np.array([
                [0.05, 0.1, 0.05],
                [0.1,  0.0, 0.1],
                [0.05, 0.1, 0.05]
            ], dtype=np.float32)
        
        # Distance grid for projecting 1D spectra to 2D
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        
        # Time counter
        self.t = 0
        
        # Spike buffer for current frame
        self.current_spikes = np.zeros((self.size, self.size), dtype=np.float32)

    def project_to_2d(self, freq_1d):
        """Map 1D frequency spectrum to 2D radial pattern."""
        if freq_1d is None or len(freq_1d) == 0:
            return np.zeros((self.size, self.size), dtype=np.float32)
        
        freq_len = len(freq_1d)
        max_r = self.center
        
        # Map radius to frequency bin
        r_indices = np.clip(
            (self.r_grid / max_r * freq_len).astype(int), 
            0, freq_len - 1
        )
        
        return freq_1d[r_indices].astype(np.float32)

    def step(self):
        self.t += 1
        
        # Rebuild kernel if type changed
        if self.kernel_type != self._last_kernel_type:
            self._build_kernel()
            self._last_kernel_type = self.kernel_type
        
        # === GET INPUTS ===
        freq_input = self.get_blended_input('frequency_input', 'sum')
        thresh_mod = self.get_blended_input('threshold_mod', 'sum')
        couple_mod = self.get_blended_input('coupling_mod', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.potential[:] = 0
            self.refractory[:] = 0
            self.spike_history[:] = 0
            self.t = 0
            return
        
        # Modulate parameters
        threshold = self.threshold
        if thresh_mod is not None:
            threshold = np.clip(0.3 + thresh_mod * 0.7, 0.3, 1.0)
            
        coupling = self.coupling
        if couple_mod is not None:
            coupling = np.clip(self.coupling * (0.5 + couple_mod), 0.01, 0.5)
        
        # === EXTERNAL DRIVE ===
        # Project harmonic input to 2D
        if freq_input is not None:
            drive = self.project_to_2d(freq_input)
            # Normalize
            if np.max(drive) > 0:
                drive = drive / np.max(drive)
            # Add temporal modulation (makes it oscillate, not static)
            # Each frequency band oscillates at its natural rate
            freq_len = len(freq_input)
            for i in range(freq_len):
                # Frequency i oscillates at rate proportional to i
                phase = np.sin(self.t * 0.1 * (i + 1))
                mask = (self.r_grid >= i * self.center / freq_len) & \
                       (self.r_grid < (i + 1) * self.center / freq_len)
                drive[mask] *= (0.5 + 0.5 * phase)
        else:
            drive = np.zeros_like(self.potential)
        
        # === NEIGHBOR COUPLING ===
        # Spikes from neighbors propagate as excitation
        neighbor_input = convolve(self.current_spikes, self.kernel, mode='wrap')
        
        # === MEMBRANE DYNAMICS ===
        # Only update non-refractory neurons
        active_mask = self.refractory <= 0
        
        # Accumulate: leak toward rest + neighbor excitation + external drive
        self.potential[active_mask] *= (1.0 - self.leak)  # Leak
        self.potential[active_mask] += coupling * neighbor_input[active_mask]
        self.potential[active_mask] += self.input_gain * drive[active_mask]
        
        # Clamp potential
        self.potential = np.clip(self.potential, 0, 1.5)
        
        # === THRESHOLD & FIRE ===
        # Find who fires this step
        fire_mask = (self.potential >= threshold) & active_mask
        
        # Record spikes
        self.current_spikes = fire_mask.astype(np.float32)
        self.spike_history = self.spike_history * 0.95 + self.current_spikes * 0.05
        
        # Reset fired neurons
        self.potential[fire_mask] = 0
        self.refractory[fire_mask] = self.refractory_period
        self.last_spike[fire_mask] = self.t
        
        # === REFRACTORY DECAY ===
        self.refractory = np.maximum(0, self.refractory - 1)

    def compute_synchrony(self):
        """
        Measure phase coherence via Kuramoto order parameter.
        Uses last spike times to estimate phase.
        """
        # Convert spike times to phases (rough approximation)
        # Assume natural period ~ 20 steps
        period = 20.0
        phases = (self.t - self.last_spike) / period * 2 * np.pi
        
        # Kuramoto order parameter
        complex_phases = np.exp(1j * phases)
        mean_phase = np.mean(complex_phases)
        
        return np.abs(mean_phase)  # 0 = desynchronized, 1 = fully synchronized

    def get_output(self, port_name):
        if port_name == 'potential_map':
            # Normalize for display
            img = (np.clip(self.potential, 0, 1) * 255).astype(np.uint8)
            return img
            
        elif port_name == 'spike_map':
            # Current spikes (binary-ish)
            img = (self.current_spikes * 255).astype(np.uint8)
            return img
            
        elif port_name == 'refractory_map':
            # Refractory state
            ref_norm = self.refractory / max(self.refractory_period, 1)
            img = (np.clip(ref_norm, 0, 1) * 255).astype(np.uint8)
            return img
            
        elif port_name == 'eigen_image':
            # FFT of spike rate (the "standing wave" if it exists)
            spec = np.abs(fftshift(fft2(self.spike_history)))
            spec_log = np.log(1 + spec * 100)
            if spec_log.max() > 0:
                spec_log = spec_log / spec_log.max()
            return (spec_log * 255).astype(np.uint8)
            
        elif port_name == 'firing_rate':
            return float(np.mean(self.current_spikes))
            
        elif port_name == 'synchrony':
            return self.compute_synchrony()
            
        elif port_name == 'eigenfrequencies':
            # Radial average of FFT for spectrum output
            spec = np.abs(fftshift(fft2(self.spike_history)))
            # Take middle row from center outward
            return spec[self.center, self.center:]
            
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # 2x2 grid display
        display = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-Left: Membrane Potential (how charged each neuron is)
        pot_img = (np.clip(self.potential, 0, 1) * 255).astype(np.uint8)
        display[:h, :w] = cv2.applyColorMap(pot_img, cv2.COLORMAP_VIRIDIS)
        
        # Top-Right: Current Spikes (who's firing NOW)
        spike_img = (self.current_spikes * 255).astype(np.uint8)
        spike_color = cv2.applyColorMap(spike_img, cv2.COLORMAP_HOT)
        display[:h, w:] = spike_color
        
        # Bottom-Left: Firing Rate (accumulated activity)
        rate_img = (np.clip(self.spike_history * 10, 0, 1) * 255).astype(np.uint8)
        display[h:, :w] = cv2.applyColorMap(rate_img, cv2.COLORMAP_PLASMA)
        
        # Bottom-Right: FFT of activity (does geometry emerge?)
        spec = np.abs(fftshift(fft2(self.spike_history)))
        spec_log = np.log(1 + spec * 100)
        if spec_log.max() > 0:
            spec_log = spec_log / spec_log.max()
        spec_img = (spec_log * 255).astype(np.uint8)
        display[h:, w:] = cv2.applyColorMap(spec_img, cv2.COLORMAP_JET)
        
        # Kernel visualization (small inset in bottom-left corner)
        kh, kw = self.kernel.shape
        scale = 4  # Scale up for visibility
        k_vis = np.clip(self.kernel, 0, None)  # Only show positive
        if k_vis.max() > 0:
            k_vis = k_vis / k_vis.max()
        k_img = (k_vis * 255).astype(np.uint8)
        k_img = cv2.resize(k_img, (kw * scale, kh * scale), interpolation=cv2.INTER_NEAREST)
        k_color = cv2.applyColorMap(k_img, cv2.COLORMAP_INFERNO)
        
        # Place in bottom-left quadrant corner
        ky, kx = kh * scale, kw * scale
        display[h + 2:h + 2 + ky, 2:2 + kx] = k_color
        cv2.rectangle(display, (1, h + 1), (3 + kx, h + 3 + ky), (255, 255, 255), 1)
        
        # Labels
        cv2.putText(display, "Potential", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(display, "Spikes", (w+5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(display, "Rate", (kx + 10, h+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(display, "FFT", (w+5, h+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Stats
        firing_rate = np.mean(self.current_spikes) * 100
        sync = self.compute_synchrony()
        cv2.putText(display, f"Fire: {firing_rate:.1f}%  Sync: {sync:.2f}  K: {self.kernel_type}", 
                   (5, h*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        kernel_options = [
            ("Square 8", "Square 8"),
            ("Cross 4", "Cross 4"),
            ("Diagonal 4", "Diagonal 4"),
            ("Hexagonal", "Hexagonal"),
            ("Radial 12", "Radial 12"),
            ("Radial 24", "Radial 24"),
            ("Mexican Hat", "Mexican Hat"),
            ("Star 6", "Star 6"),
            ("Star 5", "Star 5"),
        ]
        return [
            ("Threshold", "threshold", self.threshold, None),
            ("Refractory Period", "refractory_period", self.refractory_period, None),
            ("Leak Rate", "leak", self.leak, None),
            ("Coupling", "coupling", self.coupling, None),
            ("Input Gain", "input_gain", self.input_gain, None),
            ("Kernel Type", "kernel_type", self.kernel_type, kernel_options),
        ]
    
    def on_config_changed(self):
        """Called when config changes - rebuild kernel if needed."""
        self._build_kernel()