"""
NestedOscillatorNode
--------------------
Reveals cross-frequency coupling through phase-amplitude analysis.

Two modes:
1. FREQUENCY MODE: Analyzes coupling between EEG-like frequency bands
2. IMAGE MODE: Creates radar-like visualization where frequency vectors 
   revolve and "fire" together when coupled
"""

import numpy as np
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class NestedOscillatorNode(BaseNode):
    NODE_CATEGORY = "Fractal Substrate"
    NODE_COLOR = QtGui.QColor(80, 40, 120)  # Deep purple for nested complexity

    def __init__(self, mode='image', resolution=256, n_bands=5, coupling_threshold=0.3):
        super().__init__()
        self.node_title = "Nested Oscillator"

        self.inputs = {
            'image': 'image',        # For image mode
            'delta': 'signal',       # For frequency mode
            'theta': 'signal',
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
        }

        self.outputs = {
            'coupling_map': 'image',      # Phase-amplitude coupling strength
            'radar_viz': 'image',         # Radar-like visualization
            'phase_structure': 'image',   # Where bands lock together
            'constraint_field': 'image',  # Hierarchical constraints
        }

        # Configuration
        self.mode = mode  # 'image' or 'frequency'
        self.resolution = int(resolution)
        self.n_bands = int(n_bands)
        self.coupling_threshold = float(coupling_threshold)
        
        # Frequency band definitions (Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
        }
        
        # State
        self.time = 0
        self.coupling_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.radar_viz = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self.phase_structure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.constraint_field = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Oscillator state for radar mode
        self.oscillator_phases = np.zeros(5)  # One phase per band
        self.oscillator_amplitudes = np.ones(5)
        
        # Phase history for coupling detection
        self.phase_history = []
        self.amp_history = []
        self.history_length = 100

    def _decompose_image_to_bands(self, image):
        """Extract frequency bands from image using wavelets/FFT"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if gray.shape != (self.resolution, self.resolution):
            gray = cv2.resize(gray, (self.resolution, self.resolution))
        
        gray = gray.astype(np.float32) / 255.0
        
        # FFT decomposition
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Create frequency masks for each band
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance to [0, 1]
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        
        bands_data = {}
        
        # Map normalized frequency to bands
        # Low dist = low frequency, high dist = high frequency
        for i, (name, (low, high)) in enumerate(self.bands.items()):
            # Create band-pass filter in frequency domain
            low_norm = low / 100.0  # Normalize to [0, 1]
            high_norm = high / 100.0
            
            mask = ((dist_norm >= low_norm) & (dist_norm < high_norm)).astype(np.float32)
            mask = gaussian_filter(mask, sigma=2)  # Smooth edges
            
            # Apply mask
            band_fft = fft_shift * mask
            
            # Inverse FFT to get band
            band_ifft = np.fft.ifftshift(band_fft)
            band = np.fft.ifft2(band_ifft)
            
            # Extract amplitude and phase
            amplitude = np.abs(band)
            phase = np.angle(band)
            
            bands_data[name] = {
                'amplitude': amplitude,
                'phase': phase,
                'mean_amp': np.mean(amplitude),
                'mean_phase': np.angle(np.sum(np.exp(1j * phase)))
            }
        
        return bands_data

    def _compute_pac(self, phase_slow, amp_fast):
        """Compute Phase-Amplitude Coupling"""
        # Modulation Index: how much fast amplitude depends on slow phase
        # Bin by phase
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        binned_amps = []
        for i in range(n_bins):
            mask = (phase_slow >= phase_bins[i]) & (phase_slow < phase_bins[i + 1])
            if np.any(mask):
                binned_amps.append(np.mean(amp_fast[mask]))
            else:
                binned_amps.append(0)
        
        binned_amps = np.array(binned_amps)
        
        # Normalize
        if binned_amps.max() > 0:
            binned_amps = binned_amps / binned_amps.max()
        
        # Compute modulation index (entropy-based)
        p = binned_amps / (binned_amps.sum() + 1e-10)
        p = p + 1e-10  # Avoid log(0)
        
        H = -np.sum(p * np.log(p))
        H_max = np.log(n_bins)
        
        # Modulation index: 1 - normalized entropy
        MI = 1 - (H / H_max)
        
        return MI

    def _create_radar_visualization(self, bands_data):
        """Create radar-like visualization where vectors fire together"""
        h, w = self.resolution, self.resolution
        radar = np.zeros((h, w, 3), dtype=np.float32)
        
        # Center point
        cy, cx = h // 2, w // 2
        
        # Update oscillator phases based on frequency
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        frequencies = [2, 6, 10.5, 21.5, 65]  # Representative frequencies
        
        for i, (name, freq) in enumerate(zip(band_names, frequencies)):
            # Update phase
            self.oscillator_phases[i] += freq * 0.01  # Time step
            self.oscillator_phases[i] %= (2 * np.pi)
            
            # Update amplitude from image data
            if name in bands_data:
                self.oscillator_amplitudes[i] = bands_data[name]['mean_amp']
        
        # Draw concentric rings for each band
        max_radius = min(cx, cy) - 10
        
        for i, (name, freq) in enumerate(zip(band_names, frequencies)):
            # Radius for this band
            radius = max_radius * (i + 1) / len(band_names)
            
            # Current angle
            angle = self.oscillator_phases[i]
            
            # Amplitude modulates brightness
            amp = self.oscillator_amplitudes[i]
            
            # Color for this band
            colors = [
                [0.5, 0, 0],    # Delta - red
                [0, 0.5, 0.5],  # Theta - cyan
                [0, 0.5, 0],    # Alpha - green
                [0.5, 0.5, 0],  # Beta - yellow
                [0.5, 0, 0.5],  # Gamma - magenta
            ]
            color = np.array(colors[i]) * amp
            
            # Draw rotating vector
            end_x = int(cx + radius * np.cos(angle))
            end_y = int(cy + radius * np.sin(angle))
            
            cv2.line(radar, (cx, cy), (end_x, end_y), color.tolist(), 2)
            
            # Draw circle
            cv2.circle(radar, (cx, cy), int(radius), color.tolist(), 1)
            
            # Where vectors align, create bright spots
            y, x = np.ogrid[:h, :w]
            dist_from_ray = np.abs(
                (y - cy) * np.cos(angle) - (x - cx) * np.sin(angle)
            )
            
            # Create glow along ray
            glow = np.exp(-dist_from_ray**2 / (radius * 0.1)**2) * amp
            
            for c in range(3):
                radar[:, :, c] += glow * color[c]
        
        # Check for coupling (when phases align)
        coupling_score = 0
        for i in range(len(band_names) - 1):
            phase_diff = np.abs(self.oscillator_phases[i] - self.oscillator_phases[i + 1])
            phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # Wrap
            
            if phase_diff < 0.5:  # Aligned
                coupling_score += 1
        
        # When coupled, create central flash
        if coupling_score > 0:
            flash = np.zeros((h, w), dtype=np.float32)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            flash = np.exp(-dist**2 / (max_radius * 0.3)**2) * coupling_score / len(band_names)
            
            for c in range(3):
                radar[:, :, c] += flash
        
        # Normalize and convert
        radar = np.clip(radar, 0, 1)
        radar = (radar * 255).astype(np.uint8)
        
        return radar

    def _compute_coupling_map(self, bands_data):
        """Compute phase-amplitude coupling between all band pairs"""
        h, w = self.resolution, self.resolution
        coupling_map = np.zeros((h, w), dtype=np.float32)
        
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        # For each slow-fast pair
        for i in range(len(band_names) - 1):
            slow_name = band_names[i]
            fast_name = band_names[i + 1]
            
            if slow_name in bands_data and fast_name in bands_data:
                slow_phase = bands_data[slow_name]['phase']
                fast_amp = bands_data[fast_name]['amplitude']
                
                # Compute local PAC
                pac = self._compute_pac(slow_phase.flatten(), fast_amp.flatten())
                
                # Add to coupling map
                coupling_map += pac * fast_amp
        
        # Normalize
        if coupling_map.max() > 0:
            coupling_map = coupling_map / coupling_map.max()
        
        return coupling_map

    def _compute_phase_structure(self, bands_data):
        """Find where phases are locked across bands"""
        h, w = self.resolution, self.resolution
        phase_lock = np.zeros((h, w), dtype=np.float32)
        
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        phases = []
        
        for name in band_names:
            if name in bands_data:
                phases.append(bands_data[name]['phase'])
        
        if len(phases) > 1:
            # Compute phase coherence
            # When all phases similar, high coherence
            phases = np.array(phases)
            
            # Circular variance
            mean_phase = np.angle(np.sum(np.exp(1j * phases), axis=0))
            
            # Phase lock value
            for p in phases:
                phase_diff = np.abs(p - mean_phase)
                phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
                phase_lock += np.exp(-phase_diff)
            
            phase_lock = phase_lock / len(phases)
        
        return phase_lock

    def _compute_constraint_field(self, bands_data):
        """Compute hierarchical constraints (slow modulating fast)"""
        h, w = self.resolution, self.resolution
        constraint = np.zeros((h, w), dtype=np.float32)
        
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        # Each slow band constrains all faster bands
        for i in range(len(band_names) - 1):
            slow_name = band_names[i]
            
            if slow_name in bands_data:
                slow_amp = bands_data[slow_name]['amplitude']
                
                # Accumulated constraint from this level
                for j in range(i + 1, len(band_names)):
                    fast_name = band_names[j]
                    if fast_name in bands_data:
                        fast_amp = bands_data[fast_name]['amplitude']
                        
                        # Constraint = how much slow amp modulates fast amp
                        constraint += slow_amp * fast_amp
        
        # Normalize
        if constraint.max() > 0:
            constraint = constraint / constraint.max()
        
        return constraint

    def step(self):
        if self.mode == 'image':
            # IMAGE MODE: Decompose image and create radar viz
            image = self.get_blended_input('image', 'first')
            
            if image is not None:
                # Decompose to frequency bands
                bands_data = self._decompose_image_to_bands(image)
                
                # Create outputs
                self.coupling_map = self._compute_coupling_map(bands_data)
                self.radar_viz = self._create_radar_visualization(bands_data)
                self.phase_structure = self._compute_phase_structure(bands_data)
                self.constraint_field = self._compute_constraint_field(bands_data)
        
        else:  # frequency mode
            # FREQUENCY MODE: Analyze EEG-like signals
            # Get all band signals
            delta = self.get_blended_input('delta', 'mean')
            theta = self.get_blended_input('theta', 'mean')
            alpha = self.get_blended_input('alpha', 'mean')
            beta = self.get_blended_input('beta', 'mean')
            gamma = self.get_blended_input('gamma', 'mean')
            
            # Update oscillator phases from signals
            signals = [delta, theta, alpha, beta, gamma]
            for i, sig in enumerate(signals):
                if sig is not None:
                    # Use signal to drive amplitude
                    self.oscillator_amplitudes[i] = np.abs(sig)
            
            # Create synthetic frequency data for visualization
            # (In real use, would analyze signal phase/amplitude over time)
            bands_data = {}
            band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            
            for i, (name, sig) in enumerate(zip(band_names, signals)):
                if sig is not None:
                    # Create synthetic spatial patterns based on signal
                    h, w = self.resolution, self.resolution
                    cy, cx = h // 2, w // 2
                    
                    y, x = np.ogrid[:h, :w]
                    angle = np.arctan2(y - cy, x - cx)
                    
                    amplitude = np.ones((h, w)) * np.abs(sig)
                    phase = angle + self.oscillator_phases[i]
                    
                    bands_data[name] = {
                        'amplitude': amplitude,
                        'phase': phase,
                        'mean_amp': np.abs(sig),
                        'mean_phase': self.oscillator_phases[i]
                    }
            
            if bands_data:
                self.coupling_map = self._compute_coupling_map(bands_data)
                self.radar_viz = self._create_radar_visualization(bands_data)
                self.phase_structure = self._compute_phase_structure(bands_data)
                self.constraint_field = self._compute_constraint_field(bands_data)
        
        self.time += 1

    def get_output(self, port_name):
        if port_name == 'coupling_map':
            return self.coupling_map
        elif port_name == 'radar_viz':
            return self.radar_viz.astype(np.float32) / 255.0
        elif port_name == 'phase_structure':
            return self.phase_structure
        elif port_name == 'constraint_field':
            return self.constraint_field
        return None

    def get_display_image(self):
        display_w = 512
        display_h = 512
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        quad_size = display_w // 2
        
        # Top left: Radar visualization
        radar_resized = cv2.resize(self.radar_viz, (quad_size, quad_size))
        display[:quad_size, :quad_size] = radar_resized
        
        # Top right: Coupling map
        coupling_u8 = (self.coupling_map * 255).astype(np.uint8)
        coupling_color = cv2.applyColorMap(coupling_u8, cv2.COLORMAP_HOT)
        coupling_resized = cv2.resize(coupling_color, (quad_size, quad_size))
        display[:quad_size, quad_size:] = coupling_resized
        
        # Bottom left: Phase structure
        phase_u8 = (self.phase_structure * 255).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_u8, cv2.COLORMAP_TWILIGHT)
        phase_resized = cv2.resize(phase_color, (quad_size, quad_size))
        display[quad_size:, :quad_size] = phase_resized
        
        # Bottom right: Constraint field
        constraint_u8 = (self.constraint_field * 255).astype(np.uint8)
        constraint_color = cv2.applyColorMap(constraint_u8, cv2.COLORMAP_VIRIDIS)
        constraint_resized = cv2.resize(constraint_color, (quad_size, quad_size))
        display[quad_size:, quad_size:] = constraint_resized
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'RADAR', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'COUPLING', (quad_size + 10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'PHASE LOCK', (10, quad_size + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'CONSTRAINTS', (quad_size + 10, quad_size + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Mode indicator
        mode_text = f'Mode: {self.mode.upper()}'
        cv2.putText(display, mode_text, (10, display_h - 10), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Mode", "mode", self.mode, ['image', 'frequency']),
            ("Resolution", "resolution", self.resolution, None),
            ("N Bands", "n_bands", self.n_bands, None),
            ("Coupling Threshold", "coupling_threshold", self.coupling_threshold, None),
        ]