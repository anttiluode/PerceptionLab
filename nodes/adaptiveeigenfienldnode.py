"""
Adaptive Eigenfield Node
========================
"The field becomes the limiting factor."

This node synthesizes several key insights:
1. From selfconsistentresonantloopnode: harmonics naturally produce structure 
   (1→block, 2→complex, 6→star, higher→breakdown). The system should derive
   harmonics from the signal, not hardcode them.

2. From best.py: stable patterns can be detected and tracked. When coherent
   regions persist, they become "cells" - exactly like morphogenetic fields.

3. From the Raj paper: brain eigenmodes are conserved low-frequency patterns
   that govern diffusion. Low eigenmodes = coarse structure, high = fine detail.

4. From the DNA/THz papers: resonant frequencies emerge from geometry and coupling.
   The system has natural frequencies determined by its structure.

The node:
- Derives num_waves from spectral peaks in input (adaptive harmonics)
- Computes graph Laplacian eigenmodes for field topology
- Projects eigenmodes onto the field with amplitude/phase from signal
- Detects stable coherent regions (cells)
- Zoom selects which eigenmodes dominate (low zoom = slow modes, high = fast)
- Field limits harmony at high complexity (biological reality)

CREATED: December 2025
AUTHORS: Antti + Claude
"""

import numpy as np
import cv2
from collections import deque
from scipy.fft import fft2, ifft2, fftshift, fft, fftfreq
from scipy.ndimage import gaussian_filter, label, binary_erosion, binary_dilation
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode):
            return None


class StablePattern:
    """A detected stable structure - like a cell or coherent domain"""
    def __init__(self, id, mask, position, volume, phase_coherence):
        self.id = id
        self.mask = mask.copy()
        self.position = position  # Center of mass
        self.volume = volume
        self.phase_coherence = phase_coherence
        self.age = 0
        self.color = np.random.rand(3)  # For visualization
        
    def update(self, new_mask=None, new_position=None, new_coherence=None):
        if new_mask is not None:
            self.mask = new_mask.copy()
        if new_position is not None:
            self.position = new_position
        if new_coherence is not None:
            self.phase_coherence = new_coherence
        self.age += 1


class AdaptiveEigenfieldNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Adaptive Eigenfield"
    NODE_COLOR = QtGui.QColor(180, 100, 255)  # Purple for eigenmodes
    
    def __init__(self):
        super().__init__()
        self.node_title = "Adaptive Eigenfield (Signal-Derived Eigenmodes)"
        
        self.inputs = {
            'eeg_signal': 'signal',           # Raw signal to buffer
            'eeg_spectrum': 'spectrum',       # Direct spectrum input (6-band)
            'frequency_input': 'spectrum',    # Alternative spectrum
            'zoom': 'signal',                 # Eigenmode selection (0=slow only, 1=all)
            'coupling': 'signal',             # Field coupling strength
            'damping': 'signal',              # Energy dissipation
            'tension': 'signal',              # Wave propagation speed
            'topology': 'signal',             # 0=box, 1=torus
            'reset': 'signal'
        }
        
        self.outputs = {
            'display': 'image',
            'field': 'complex_spectrum',      # The main eigenfield
            'eigenspectrum': 'spectrum',      # Current eigenvalues
            'num_modes': 'signal',            # Number of active eigenmodes
            'num_patterns': 'signal',         # Detected stable patterns
            'criticality': 'signal',          # Edge of chaos metric
            'total_energy': 'signal',
            'pattern_field': 'image',         # Visualization of stable patterns
        }
        
        # Field parameters
        self.field_size = 128
        self.dt = 0.1
        self.damping = 0.001
        self.tension = 5.0
        self.coupling = 0.5
        self.zoom = 0.5  # 0 = show only slowest modes, 1 = all modes
        self.topology = 'box'  # 'box' or 'torus'
        
        # Signal processing
        self.buffer_size = 512
        self.sample_rate = 160.0
        self.signal_buffer = deque(maxlen=self.buffer_size)
        
        # Eigenmode system
        self.max_modes = 32  # Maximum number of eigenmodes to compute
        self._eigenvectors = None
        self._eigenvalues = None
        self._mode_amplitudes = np.zeros(self.max_modes)
        self._mode_phases = np.zeros(self.max_modes)
        self._num_active_modes = 6  # Derived from signal peaks
        
        # Initialize eigenmodes
        self._compute_laplacian_eigenmodes()
        
        # Field state (like best.py)
        self.field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.field_prev = np.zeros_like(self.field)
        self._init_field()
        
        # Stable pattern tracking
        self.patterns = {}
        self.next_pattern_id = 1
        self.pattern_mask = np.zeros((self.field_size, self.field_size), dtype=bool)
        self.last_detection_time = 0
        
        # Metrics
        self.total_energy = 0.0
        self.energy_history = deque(maxlen=200)
        self.criticality = 0.0
        self.criticality_history = deque(maxlen=200)
        
        # Display
        self._display = np.zeros((600, 900, 3), dtype=np.uint8)
        self.epoch = 0
    
    def _compute_laplacian_eigenmodes(self):
        """Compute eigenmodes of the 2D grid Laplacian"""
        n = self.field_size
        n_sq = n * n
        
        # Build sparse Laplacian matrix for 2D grid
        # Each point connected to 4 neighbors (or wrapped for torus)
        main_diag = np.ones(n_sq) * 4
        off_diag = np.ones(n_sq - 1) * -1
        
        # Handle row boundaries (no connection across rows)
        for i in range(n - 1, n_sq - 1, n):
            off_diag[i] = 0
        
        row_diag = np.ones(n_sq - n) * -1
        
        L = diags([main_diag, off_diag, off_diag, row_diag, row_diag], 
                  [0, -1, 1, -n, n], format='csr')
        
        # Compute smallest eigenvalues (slowest modes)
        try:
            eigenvalues, eigenvectors = eigsh(L.astype(np.float64), 
                                               k=min(self.max_modes, n_sq - 2), 
                                               which='SM')
            self._eigenvalues = eigenvalues
            self._eigenvectors = eigenvectors
        except Exception as e:
            print(f"Eigenmode computation failed: {e}")
            # Fallback to simple sine modes
            self._eigenvalues = np.arange(1, self.max_modes + 1).astype(float)
            self._eigenvectors = np.zeros((n_sq, self.max_modes))
            for m in range(self.max_modes):
                # Simple standing wave approximation
                kx = (m % 8) + 1
                ky = (m // 8) + 1
                x = np.arange(n)
                X, Y = np.meshgrid(x, x)
                mode = np.sin(np.pi * kx * X / n) * np.sin(np.pi * ky * Y / n)
                self._eigenvectors[:, m] = mode.flatten()
    
    def _init_field(self):
        """Initialize field with small random perturbation"""
        n = self.field_size
        c = n // 2
        r = n // 6
        
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        
        # Gaussian seed + small noise
        self.field = 0.5 * np.exp(-((X - c)**2 + (Y - c)**2) / (2 * r**2))
        self.field = self.field.astype(np.complex128)
        self.field += (np.random.randn(n, n) + 1j * np.random.randn(n, n)) * 0.05
        
        self.field_prev = self.field.copy()
    
    def _derive_modes_from_signal(self):
        """Derive number of active modes from spectral peaks in input"""
        if len(self.signal_buffer) < self.buffer_size // 4:
            return
        
        try:
            sig = np.array(list(self.signal_buffer))
            sig = sig - np.mean(sig)
            
            if np.std(sig) < 1e-10:
                return
            
            # FFT
            spectrum = np.abs(fft(sig * np.hanning(len(sig))))
            freqs = fftfreq(len(sig), 1.0 / self.sample_rate)
            
            # Only positive frequencies
            pos_mask = freqs > 0
            spectrum_pos = spectrum[pos_mask]
            
            if len(spectrum_pos) == 0:
                return
            
            # Find peaks
            threshold = np.mean(spectrum_pos) * 1.5
            peaks, properties = find_peaks(spectrum_pos, height=threshold, distance=5)
            
            # Number of significant peaks determines mode count
            num_peaks = len(peaks)
            
            if num_peaks == 0:
                self._num_active_modes = 2  # Minimum
            elif num_peaks == 1:
                self._num_active_modes = 4
            elif num_peaks <= 3:
                self._num_active_modes = 6
            elif num_peaks <= 6:
                self._num_active_modes = 12
            else:
                self._num_active_modes = min(num_peaks * 2, self.max_modes)
            
            # Set mode amplitudes from peak heights
            self._mode_amplitudes[:] = 0
            if num_peaks > 0:
                heights = properties['peak_heights']
                max_height = np.max(heights) if len(heights) > 0 else 1.0
                
                for i, (peak_idx, height) in enumerate(zip(peaks, heights)):
                    if i < self.max_modes:
                        self._mode_amplitudes[i] = height / max_height
                        # Phase from signal phase at that frequency
                        self._mode_phases[i] = np.angle(fft(sig)[pos_mask][peak_idx]) if peak_idx < len(spectrum_pos) else 0
                        
        except Exception as e:
            pass  # Keep previous mode count
    
    def _project_eigenmodes_to_field(self):
        """Project active eigenmodes onto the 2D field with zoom-based selection"""
        if self._eigenvectors is None:
            return np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        
        n = self.field_size
        result = np.zeros((n, n), dtype=np.complex128)
        
        # Zoom determines which modes are active
        # zoom=0: only mode 0 (slowest)
        # zoom=1: all modes up to num_active
        max_mode_idx = max(1, int(self.zoom * self._num_active_modes))
        max_mode_idx = min(max_mode_idx, self._eigenvectors.shape[1])
        
        for m in range(max_mode_idx):
            if m >= len(self._mode_amplitudes):
                break
                
            amp = self._mode_amplitudes[m]
            phase = self._mode_phases[m]
            
            if amp < 1e-6:
                amp = 0.1  # Default amplitude for unset modes
            
            # Get eigenmode and reshape to 2D
            if m < self._eigenvectors.shape[1]:
                mode_1d = self._eigenvectors[:, m]
                mode_2d = mode_1d.reshape(n, n)
                
                # Add with amplitude and phase
                result += amp * mode_2d * np.exp(1j * phase)
        
        # Normalize
        max_val = np.max(np.abs(result))
        if max_val > 1e-10:
            result /= max_val
        
        return result
    
    def _step_field_physics(self):
        """Evolve field with wave equation physics (inspired by best.py)"""
        n = self.field_size
        
        # Get current eigenmode projection
        eigenmode_contribution = self._project_eigenmodes_to_field()
        
        # Boundary mode based on topology
        mode = 'wrap' if self.topology == 'torus' else 'reflect'
        
        # Laplacian via convolution
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64)
        
        # Apply to real and imaginary parts
        lap_real = cv2.filter2D(np.real(self.field).astype(np.float64), -1, kernel, 
                                borderType=cv2.BORDER_WRAP if mode == 'wrap' else cv2.BORDER_REFLECT)
        lap_imag = cv2.filter2D(np.imag(self.field).astype(np.float64), -1, kernel,
                                borderType=cv2.BORDER_WRAP if mode == 'wrap' else cv2.BORDER_REFLECT)
        lap = lap_real + 1j * lap_imag
        
        # Wave equation with damping
        # φ_new = 2φ - φ_old + dt²(c²∇²φ - V'(φ)) - damping*(φ - φ_old)
        
        # Non-linear potential (encourages phase coherence)
        mag = np.abs(self.field)
        V_prime = -self.field + 0.2 * self.field * mag**2
        
        # Wave speed modulated by tension
        c2 = self.tension / (1.0 + mag**2 + 1e-6)
        
        acc = c2 * lap - V_prime
        
        # Velocity
        vel = self.field - self.field_prev
        
        # Update
        field_new = (self.field + (1 - self.damping * self.dt) * vel + 
                    self.dt**2 * acc)
        
        # Couple in eigenmode structure
        field_new = (1 - self.coupling * 0.01) * field_new + self.coupling * 0.01 * eigenmode_contribution
        
        # Store history
        self.field_prev = self.field.copy()
        self.field = field_new
        
        # Normalize to prevent blowup
        max_mag = np.max(np.abs(self.field))
        if max_mag > 5.0:
            self.field /= (max_mag / 5.0)
    
    def _detect_stable_patterns(self):
        """Detect coherent regions in the field (cells)"""
        # Only run periodically
        import time
        current_time = time.time()
        if current_time - self.last_detection_time < 0.3:
            return
        self.last_detection_time = current_time
        
        n = self.field_size
        
        # Coherence = local phase consistency
        phase = np.angle(self.field)
        
        # Compute local phase variance (low = coherent)
        phase_blurred = gaussian_filter(phase, sigma=3)
        phase_diff = np.abs(phase - phase_blurred)
        coherence = 1.0 - np.clip(phase_diff / np.pi, 0, 1)
        
        # Also consider magnitude
        mag = np.abs(self.field)
        mag_norm = mag / (np.max(mag) + 1e-10)
        
        # Pattern mask: high coherence AND significant magnitude
        pattern_criterion = coherence * mag_norm
        binary_mask = pattern_criterion > 0.5
        
        # Clean up
        binary_mask = binary_erosion(binary_mask, iterations=1)
        binary_mask = binary_dilation(binary_mask, iterations=2)
        
        # Label connected components
        labeled, num_features = label(binary_mask)
        
        # Track patterns
        active_ids = set()
        min_volume = 20
        
        for i in range(1, num_features + 1):
            component_mask = (labeled == i)
            volume = np.sum(component_mask)
            
            if volume < min_volume:
                continue
            
            # Get centroid
            coords = np.where(component_mask)
            position = (np.mean(coords[0]), np.mean(coords[1]))
            
            # Mean phase coherence in this region
            region_coherence = np.mean(coherence[component_mask])
            
            # Try to match with existing pattern
            matched = False
            closest_id = None
            min_dist = float('inf')
            
            for pid, pattern in self.patterns.items():
                dist = np.sqrt((pattern.position[0] - position[0])**2 + 
                              (pattern.position[1] - position[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_id = pid
            
            if closest_id is not None and min_dist < 15:
                self.patterns[closest_id].update(component_mask, position, region_coherence)
                active_ids.add(closest_id)
                matched = True
            
            if not matched:
                new_id = self.next_pattern_id
                self.next_pattern_id += 1
                self.patterns[new_id] = StablePattern(new_id, component_mask, position, 
                                                       volume, region_coherence)
                active_ids.add(new_id)
        
        # Age out patterns not detected
        to_remove = []
        for pid in self.patterns:
            if pid not in active_ids:
                self.patterns[pid].age -= 2
                if self.patterns[pid].age <= 0:
                    to_remove.append(pid)
        
        for pid in to_remove:
            del self.patterns[pid]
        
        # Update global mask
        self.pattern_mask = np.zeros((n, n), dtype=bool)
        for pattern in self.patterns.values():
            self.pattern_mask |= pattern.mask
    
    def _compute_metrics(self):
        """Compute energy and criticality metrics"""
        # Energy
        mag = np.abs(self.field)
        grad_x = np.gradient(np.real(self.field), axis=0)
        grad_y = np.gradient(np.real(self.field), axis=1)
        
        kinetic = 0.5 * np.sum(np.abs(self.field - self.field_prev)**2)
        potential = 0.5 * np.sum(grad_x**2 + grad_y**2)
        
        self.total_energy = kinetic + potential
        self.energy_history.append(self.total_energy)
        
        # Criticality: variance of energy history (high variance = critical)
        if len(self.energy_history) > 10:
            energy_arr = np.array(list(self.energy_history))
            mean_e = np.mean(energy_arr)
            if mean_e > 1e-10:
                self.criticality = np.std(energy_arr) / mean_e
            else:
                self.criticality = 0.0
            self.criticality = np.clip(self.criticality, 0, 1)
        
        self.criticality_history.append(self.criticality)
    
    def step(self):
        self.epoch += 1
        
        # Get inputs
        reset = self.get_blended_input('reset', 'sum')
        if reset is not None and reset > 0.5:
            self._init_field()
            self.patterns = {}
            self.next_pattern_id = 1
            self.energy_history.clear()
            self.criticality_history.clear()
            return
        
        # Update parameters from inputs
        zoom_in = self.get_blended_input('zoom', 'sum')
        if zoom_in is not None:
            self.zoom = np.clip(float(zoom_in), 0, 1)
        
        coupling_in = self.get_blended_input('coupling', 'sum')
        if coupling_in is not None:
            self.coupling = np.clip(float(coupling_in), 0, 1)
        
        damping_in = self.get_blended_input('damping', 'sum')
        if damping_in is not None:
            self.damping = np.clip(float(damping_in), 0, 0.1)
        
        tension_in = self.get_blended_input('tension', 'sum')
        if tension_in is not None:
            self.tension = np.clip(float(tension_in), 0.1, 20)
        
        topology_in = self.get_blended_input('topology', 'sum')
        if topology_in is not None:
            self.topology = 'torus' if float(topology_in) > 0.5 else 'box'
        
        # Buffer signal
        sig_in = self.get_blended_input('eeg_signal', 'sum')
        if sig_in is not None:
            if isinstance(sig_in, np.ndarray):
                for s in sig_in.flatten()[:10]:
                    self.signal_buffer.append(float(s))
            else:
                self.signal_buffer.append(float(sig_in))
        
        # Process spectrum input
        spectrum_in = self.get_blended_input('eeg_spectrum', 'sum')
        if spectrum_in is None:
            spectrum_in = self.get_blended_input('frequency_input', 'sum')
        
        if spectrum_in is not None and isinstance(spectrum_in, np.ndarray):
            # Use spectrum peaks to set mode amplitudes directly
            spec = np.abs(spectrum_in)
            if len(spec) > 0:
                max_spec = np.max(spec)
                if max_spec > 1e-10:
                    spec = spec / max_spec
                    for i, val in enumerate(spec[:self.max_modes]):
                        self._mode_amplitudes[i] = val
                    self._num_active_modes = max(2, min(len(spec), self.max_modes))
        else:
            # Derive modes from buffered signal
            self._derive_modes_from_signal()
        
        # Physics step
        self._step_field_physics()
        
        # Pattern detection
        self._detect_stable_patterns()
        
        # Metrics
        self._compute_metrics()
        
        # Update display
        self._update_display()
    
    def _update_display(self):
        """Generate visualization"""
        img = np.zeros((600, 900, 3), dtype=np.uint8)
        
        # Main field visualization (left side)
        field_size_display = 256
        
        # Field magnitude with phase as hue
        mag = np.abs(self.field)
        phase = np.angle(self.field)
        
        mag_norm = mag / (np.max(mag) + 1e-10)
        
        hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = (mag_norm * 255).astype(np.uint8)
        
        field_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        field_resized = cv2.resize(field_color, (field_size_display, field_size_display))
        
        img[20:20 + field_size_display, 20:20 + field_size_display] = field_resized
        
        cv2.putText(img, "EIGENFIELD (Phase=Hue, Mag=Bright)", (20, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Pattern overlay
        if np.any(self.pattern_mask):
            pattern_overlay = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
            for pattern in self.patterns.values():
                c = (pattern.color * 255).astype(np.uint8)
                pattern_overlay[pattern.mask] = c
            
            pattern_resized = cv2.resize(pattern_overlay, (field_size_display, field_size_display))
            # Blend
            alpha = 0.3
            img[20:20 + field_size_display, 20:20 + field_size_display] = \
                cv2.addWeighted(field_resized, 1 - alpha, pattern_resized, alpha, 0)
        
        # Eigenspectrum visualization (right top)
        spec_x, spec_y = 300, 30
        spec_w, spec_h = 250, 100
        
        cv2.putText(img, f"ACTIVE EIGENMODES (n={self._num_active_modes})", (spec_x, spec_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        if self._eigenvalues is not None:
            num_show = min(self._num_active_modes, len(self._eigenvalues))
            max_amp = max(np.max(self._mode_amplitudes[:num_show]), 1e-10)
            
            bar_width = max(3, spec_w // num_show - 2)
            
            for i in range(num_show):
                x = spec_x + i * (bar_width + 2)
                amp = self._mode_amplitudes[i] / max_amp if i < len(self._mode_amplitudes) else 0
                height = int(amp * spec_h)
                
                # Color by eigenvalue (slow=red, fast=blue)
                hue = int(120 * (i / max(num_show - 1, 1)))  # Green to cyan
                color = cv2.cvtColor(np.array([[[hue, 200, 200]]], dtype=np.uint8), 
                                    cv2.COLOR_HSV2BGR)[0, 0]
                
                cv2.rectangle(img, (x, spec_y + spec_h - height), 
                             (x + bar_width, spec_y + spec_h),
                             tuple(int(c) for c in color), -1)
        
        # Zoom indicator
        zoom_y = spec_y + spec_h + 30
        cv2.putText(img, f"ZOOM: {self.zoom:.2f}", (spec_x, zoom_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.rectangle(img, (spec_x + 80, zoom_y - 10), 
                     (spec_x + 80 + int(self.zoom * 100), zoom_y),
                     (100, 200, 255), -1)
        
        cv2.putText(img, "low=coarse structure | high=fine detail", (spec_x, zoom_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Metrics panel
        metrics_x, metrics_y = 300, 200
        
        cv2.putText(img, "METRICS", (metrics_x, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        
        cv2.putText(img, f"Energy: {self.total_energy:.2f}", (metrics_x, metrics_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(img, f"Criticality: {self.criticality:.3f}", (metrics_x, metrics_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        bar_w = int(self.criticality * 150)
        cv2.rectangle(img, (metrics_x, metrics_y + 55), 
                     (metrics_x + bar_w, metrics_y + 65),
                     (100, 200, 255), -1)
        
        cv2.putText(img, f"Stable Patterns: {len(self.patterns)}", (metrics_x, metrics_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        cv2.putText(img, f"Topology: {self.topology.upper()}", (metrics_x, metrics_y + 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Pattern list
        pattern_x, pattern_y = 300, 340
        cv2.putText(img, "DETECTED CELLS (age/coherence):", (pattern_x, pattern_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        for i, (pid, pattern) in enumerate(list(self.patterns.items())[:8]):
            c = (pattern.color * 255).astype(np.uint8)
            y = pattern_y + 20 + i * 20
            cv2.rectangle(img, (pattern_x, y - 10), (pattern_x + 15, y + 5),
                         tuple(int(x) for x in c), -1)
            cv2.putText(img, f"#{pid}: age={pattern.age}, coh={pattern.phase_coherence:.2f}", 
                       (pattern_x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Criticality history
        hist_x, hist_y = 20, 320
        hist_w, hist_h = 250, 80
        
        cv2.putText(img, "CRITICALITY HISTORY", (hist_x, hist_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        if len(self.criticality_history) > 2:
            hist = np.array(list(self.criticality_history))
            hist = hist / (np.max(hist) + 1e-10)
            
            for i in range(1, len(hist)):
                x1 = hist_x + int((i - 1) / len(hist) * hist_w)
                x2 = hist_x + int(i / len(hist) * hist_w)
                y1 = hist_y + 10 + hist_h - int(hist[i - 1] * hist_h)
                y2 = hist_y + 10 + hist_h - int(hist[i] * hist_h)
                cv2.line(img, (x1, y1), (x2, y2), (100, 200, 255), 1)
        
        # Theory notes
        theory_y = 450
        cv2.putText(img, "ADAPTIVE EIGENFIELD HYPOTHESIS:", (20, theory_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        cv2.putText(img, "Eigenmodes derived from input signal spectrum, not hardcoded.", 
                   (20, theory_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1)
        cv2.putText(img, f"Simple signal -> few modes -> blocks. Complex -> many -> stars -> breakdown.", 
                   (20, theory_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1)
        cv2.putText(img, "ZOOM selects eigenmode range: 0=slow(coarse), 1=fast(fine)", 
                   (20, theory_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1)
        cv2.putText(img, "Stable regions = 'cells' with coherent phase. Field limits harmony at complexity.", 
                   (20, theory_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1)
        
        # Parameters
        cv2.putText(img, f"epoch={self.epoch} | coupling={self.coupling:.2f} | damping={self.damping:.4f} | tension={self.tension:.1f}",
                   (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        self._display = img
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'field':
            return self.field
        elif name == 'eigenspectrum':
            if self._eigenvalues is not None:
                return self._eigenvalues[:self._num_active_modes]
            return np.zeros(6)
        elif name == 'num_modes':
            return float(self._num_active_modes)
        elif name == 'num_patterns':
            return float(len(self.patterns))
        elif name == 'criticality':
            return float(self.criticality)
        elif name == 'total_energy':
            return float(self.total_energy)
        elif name == 'pattern_field':
            # Return visualization of just the patterns
            img = np.zeros((self.field_size, self.field_size), dtype=np.uint8)
            for pattern in self.patterns.values():
                brightness = int(pattern.age * 10)
                img[pattern.mask] = min(255, brightness)
            return img
        return None
    
    def get_display_image(self):
        h, w = self._display.shape[:2]
        return QtGui.QImage(self._display.data, w, h, w * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Zoom (Eigenmode Selection)", "zoom", self.zoom, None),
            ("Coupling", "coupling", self.coupling, None),
            ("Damping", "damping", self.damping, None),
            ("Tension", "tension", self.tension, None),
            ("Topology", "topology", self.topology, [("Box", "box"), ("Torus", "torus")]),
        ]