"""
Eigenmode Phase Analyzer - Complex Field Representation of Brain Dynamics
==========================================================================

This node extracts PHASE from eigenmode time series, enabling:

1. PHASE LAG DETECTION: Which modes lead/lag others (causality)
2. PHASE LOCKING: Are modes synchronized or independent?
3. COMPLEX FIELD: Amplitude + Phase = full holographic representation
4. INTERFERENCE PATTERNS: How modes constructively/destructively combine

THEORY:
Each eigenmode is a spatial basis function. With amplitude only, you know
"how much" of each pattern. With phase, you know "when" each pattern peaks.

Phase relationships reveal:
- Information flow direction (leader-follower)
- Binding (synchronized modes = unified percept)
- State transitions (phase slips = mode switching)

OUTPUTS:
- phase_image: Visualization of mode phases (polar plot)
- phase_lag_matrix: NxN matrix of phase lags between modes
- complex_modes: Complex-valued mode activations (magnitude + phase)
- phase_coherence: Overall phase synchronization (0-1)
- lead_mode: Which mode is currently leading
- lag_mode: Which mode is currently lagging
- phase_velocity: Rate of phase change
- interference_field: 2D interference pattern from mode superposition

Created: December 2025
"""

import numpy as np
import cv2
from collections import deque
from scipy.signal import hilbert

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
            self.input_data = {}
        def get_blended_input(self, name, mode): 
            return None
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}


class ModePhaseAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Mode Phase Analyzer"
    NODE_COLOR = QtGui.QColor(200, 100, 255)  # Purple for phase/complex
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'mode_spectrum': 'spectrum',
        }
        
        self.outputs = {
            'phase_image': 'image',
            'interference_field': 'image',
            'phase_lag_matrix': 'image',
            'complex_modes': 'complex_spectrum',  # The key output!
            'phase_spectrum': 'spectrum',          # Just the phases
            'phase_coherence': 'signal',
            'lead_mode': 'signal',
            'lag_mode': 'signal', 
            'phase_velocity': 'signal',
            'mean_phase': 'signal',
            'low_high_phase_diff': 'signal',       # Phase diff between low/high modes
        }
        
        # Config
        self.n_modes = 10
        self.history_length = 64  # Need enough for Hilbert transform
        self.field_size = 128     # Interference field resolution
        
        # State
        self.mode_history = deque(maxlen=self.history_length)
        
        # Computed values
        self.amplitudes = np.zeros(self.n_modes)
        self.phases = np.zeros(self.n_modes)
        self.complex_modes = np.zeros(self.n_modes, dtype=np.complex64)
        self.phase_lag_matrix = np.zeros((self.n_modes, self.n_modes))
        self.phase_velocities = np.zeros(self.n_modes)
        self.prev_phases = np.zeros(self.n_modes)
        
        # Images
        self.phase_image = None
        self.interference_image = None
        self.lag_matrix_image = None
        
        self.frame_count = 0
        
    def step(self):
        self.frame_count += 1
        
        # Get mode spectrum
        modes = self.get_blended_input('mode_spectrum', 'mean')
        if modes is None:
            modes = np.zeros(self.n_modes)
        else:
            modes = np.array(modes).flatten()
            if len(modes) < self.n_modes:
                modes = np.pad(modes, (0, self.n_modes - len(modes)))
            elif len(modes) > self.n_modes:
                modes = modes[:self.n_modes]
        
        self.mode_history.append(modes.copy())
        
        # Need enough history for Hilbert transform
        if len(self.mode_history) >= self.history_length // 2:
            self._compute_phases()
            
            if self.frame_count % 3 == 0:
                self._compute_phase_lags()
                self._render_phase_plot()
                self._render_interference_field()
                self._render_lag_matrix()
    
    def _compute_phases(self):
        """Extract instantaneous phase using Hilbert transform"""
        history = np.array(list(self.mode_history))
        n_samples = len(history)
        
        if n_samples < 8:
            return
        
        self.prev_phases = self.phases.copy()
        
        for i in range(self.n_modes):
            signal = history[:, i]
            
            # Remove DC offset
            signal = signal - np.mean(signal)
            
            # Hilbert transform for analytic signal
            try:
                analytic = hilbert(signal)
                
                # Current (most recent) values
                self.amplitudes[i] = np.abs(analytic[-1])
                self.phases[i] = np.angle(analytic[-1])
                
                # Complex representation
                self.complex_modes[i] = analytic[-1]
                
            except Exception:
                self.amplitudes[i] = np.abs(signal[-1])
                self.phases[i] = 0.0
                self.complex_modes[i] = signal[-1] + 0j
        
        # Phase velocity (how fast phase is changing)
        phase_diff = self.phases - self.prev_phases
        # Unwrap phase jumps
        phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
        self.phase_velocities = phase_diff
    
    def _compute_phase_lags(self):
        """Compute phase lag between all mode pairs"""
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                if i != j:
                    # Phase difference (i relative to j)
                    diff = self.phases[i] - self.phases[j]
                    # Wrap to [-pi, pi]
                    diff = np.mod(diff + np.pi, 2*np.pi) - np.pi
                    self.phase_lag_matrix[i, j] = diff
    
    def _render_phase_plot(self):
        """Render polar plot of mode phases"""
        h, w = 200, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        cx, cy = w // 2, h // 2
        max_r = min(cx, cy) - 20
        
        # Draw reference circles
        for r_frac in [0.33, 0.66, 1.0]:
            r = int(max_r * r_frac)
            cv2.circle(img, (cx, cy), r, (40, 40, 40), 1)
        
        # Draw reference lines (0, 90, 180, 270 degrees)
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x_end = int(cx + max_r * np.cos(angle))
            y_end = int(cy - max_r * np.sin(angle))  # Negative because y is inverted
            cv2.line(img, (cx, cy), (x_end, y_end), (40, 40, 40), 1)
        
        # Color gradient for modes
        colors = []
        for i in range(self.n_modes):
            # Blue to red gradient
            b = int(255 * (1 - i / self.n_modes))
            r = int(255 * i / self.n_modes)
            colors.append((b, 100, r))
        
        # Normalize amplitudes for display
        amp_max = np.max(self.amplitudes) + 1e-6
        
        # Draw each mode as a vector from center
        for i in range(self.n_modes):
            amp_norm = self.amplitudes[i] / amp_max
            r = int(max_r * amp_norm)
            
            # Phase determines angle (0 = right, increases counter-clockwise)
            x = int(cx + r * np.cos(self.phases[i]))
            y = int(cy - r * np.sin(self.phases[i]))  # Negative for screen coords
            
            # Draw line from center
            cv2.line(img, (cx, cy), (x, y), colors[i], 2)
            
            # Draw dot at end
            cv2.circle(img, (x, y), 5, colors[i], -1)
            
            # Label
            label_x = int(cx + (max_r + 10) * np.cos(self.phases[i]))
            label_y = int(cy - (max_r + 10) * np.sin(self.phases[i]))
            cv2.putText(img, str(i+1), (label_x-5, label_y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
        
        # Title
        cv2.putText(img, "Mode Phases", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Phase coherence indicator
        coherence = self._compute_coherence()
        cv2.putText(img, f"Coh: {coherence:.2f}", (5, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 150), 1)
        
        self.phase_image = img
    
    def _render_interference_field(self):
        """
        Create 2D interference pattern from mode superposition.
        This is where spatial + temporal + phase combine into a field.
        """
        size = self.field_size
        field = np.zeros((size, size), dtype=np.complex64)
        
        # Create spatial basis patterns for each mode
        # Using simple 2D wave patterns as proxy for eigenmodes
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        for i in range(self.n_modes):
            # Each mode gets a different spatial frequency
            # Low modes = low spatial freq, high modes = high spatial freq
            freq = (i + 1) * 0.5
            
            # Create 2D wave pattern (simplified eigenmode proxy)
            # Alternate between horizontal, vertical, diagonal patterns
            if i % 4 == 0:
                spatial_pattern = np.cos(freq * X)
            elif i % 4 == 1:
                spatial_pattern = np.cos(freq * Y)
            elif i % 4 == 2:
                spatial_pattern = np.cos(freq * (X + Y) / np.sqrt(2))
            else:
                spatial_pattern = np.cos(freq * (X - Y) / np.sqrt(2))
            
            # Modulate by complex amplitude (magnitude AND phase!)
            field += self.complex_modes[i] * spatial_pattern
        
        # Convert complex field to displayable image
        # Option 1: Magnitude
        magnitude = np.abs(field)
        
        # Option 2: Real part (shows interference fringes)
        real_part = np.real(field)
        
        # Option 3: Phase (shows wavefronts)
        phase = np.angle(field)
        
        # Combine into RGB: R=magnitude, G=real, B=phase
        mag_norm = magnitude / (np.max(magnitude) + 1e-6)
        real_norm = (real_part - real_part.min()) / (real_part.max() - real_part.min() + 1e-6)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :, 2] = (mag_norm * 255).astype(np.uint8)      # Red = magnitude
        img[:, :, 1] = (real_norm * 200).astype(np.uint8)     # Green = real part
        img[:, :, 0] = (phase_norm * 150).astype(np.uint8)    # Blue = phase
        
        # Add title
        cv2.putText(img, "Interference", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.interference_image = img
    
    def _render_lag_matrix(self):
        """Render phase lag matrix"""
        h, w = 120, 120
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        cell = 10
        ox, oy = 15, 15
        
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                x = ox + j * cell
                y = oy + i * cell
                
                # Phase lag: positive = i leads j (red), negative = i lags j (blue)
                lag = self.phase_lag_matrix[i, j]
                
                if lag > 0:
                    intensity = min(lag / np.pi, 1.0)
                    color = (0, int(100 * intensity), int(255 * intensity))  # Red
                else:
                    intensity = min(-lag / np.pi, 1.0)
                    color = (int(255 * intensity), int(100 * intensity), 0)  # Blue
                
                cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), color, -1)
        
        cv2.putText(img, "Phase Lag", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(img, "Red=lead", (w-50, h-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 200), 1)
        
        self.lag_matrix_image = img
    
    def _compute_coherence(self):
        """Compute phase coherence (how synchronized are all modes?)"""
        # Mean resultant length of phase vectors
        # 1 = all modes in phase, 0 = uniformly distributed
        if np.sum(self.amplitudes) < 1e-6:
            return 0.0
        
        # Weight by amplitude
        weights = self.amplitudes / (np.sum(self.amplitudes) + 1e-6)
        
        # Complex mean
        mean_vector = np.sum(weights * np.exp(1j * self.phases))
        
        return float(np.abs(mean_vector))
    
    def get_output(self, port_name):
        if port_name == 'phase_image':
            return self.phase_image
        elif port_name == 'interference_field':
            return self.interference_image
        elif port_name == 'phase_lag_matrix':
            return self.lag_matrix_image
        elif port_name == 'complex_modes':
            return self.complex_modes.astype(np.complex64)
        elif port_name == 'phase_spectrum':
            return self.phases.astype(np.float32)
        elif port_name == 'phase_coherence':
            return self._compute_coherence()
        elif port_name == 'lead_mode':
            # Find mode that leads most others (most positive phase)
            mean_lags = np.mean(self.phase_lag_matrix, axis=1)
            return float(np.argmax(mean_lags) + 1)
        elif port_name == 'lag_mode':
            # Find mode that lags most others
            mean_lags = np.mean(self.phase_lag_matrix, axis=1)
            return float(np.argmin(mean_lags) + 1)
        elif port_name == 'phase_velocity':
            return float(np.mean(np.abs(self.phase_velocities)))
        elif port_name == 'mean_phase':
            # Circular mean of phases
            mean_vec = np.mean(np.exp(1j * self.phases))
            return float(np.angle(mean_vec))
        elif port_name == 'low_high_phase_diff':
            # Phase difference between low modes (1-3) and high modes (8-10)
            low_phase = np.mean(self.phases[:3])
            high_phase = np.mean(self.phases[7:])
            diff = low_phase - high_phase
            return float(np.mod(diff + np.pi, 2*np.pi) - np.pi)
        return None
    
    def get_display_image(self):
        if self.phase_image is not None:
            img = np.ascontiguousarray(self.phase_image)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        w, h = 100, 50
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "Collecting...", (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None),
            ("Field Size", "field_size", self.field_size, None),
        ]