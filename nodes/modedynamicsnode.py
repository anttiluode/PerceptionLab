"""
Eigenmode Dynamics Analyzer - Temporal Structure of Spatial Modes
==================================================================

This node captures what you observed: modes have different temporal signatures
than raw band power. Specifically:

1. ATTACK/DECAY ASYMMETRY: Bands spike fast, modes taper slowly
2. MODE COUPLING: Which modes transition together?
3. PERSISTENCE: How long does each mode "ring" after activation?
4. HIERARCHY: Do low modes predict high mode activity?

THEORY:
The Graph Laplacian eigenmodes form a basis for activity diffusion.
Low modes = slow, global spread (long decay)
High modes = fast, local activity (quick decay)

If low modes show longer persistence, that validates Raj's diffusion model.
If modes show predictable sequences, that reveals the brain's "trajectory"
through state space.

INPUTS (from EigenmodeEEGNode):
  - mode_1 through mode_10
  - alpha_power (for comparison)

OUTPUTS:
  - dynamics_image: Visualization of mode dynamics
  - persistence_spectrum: Decay constants per mode (latent)
  - mode_velocity: Rate of change in mode space (signal)
  - transition_matrix: Which modes follow which (image)
  - low_high_lag: Time lag between low and high mode activation (signal)
  - attack_ratio: Band attack speed / Mode attack speed (signal)
  - state_stability: How stable is current mode configuration (signal)
  - dominant_trajectory: Current movement direction in mode space (signal)

Created: December 2025
For: PerceptionLab v11
"""

import numpy as np
import cv2
from collections import deque

# === PERCEPTION LAB COMPATIBILITY ===
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


class ModeDynamicsNode(BaseNode):
    """
    Analyzes the temporal dynamics of eigenmode activations
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Mode Dynamics"
    NODE_COLOR = QtGui.QColor(100, 200, 150)  # Teal-green for dynamics
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'mode_1': 'signal',
            'mode_2': 'signal',
            'mode_3': 'signal',
            'mode_4': 'signal',
            'mode_5': 'signal',
            'mode_6': 'signal',
            'mode_7': 'signal',
            'mode_8': 'signal',
            'mode_9': 'signal',
            'mode_10': 'signal',
            'alpha_power': 'signal',  # For attack comparison
            'theta_power': 'signal',
        }
        
        # === OUTPUTS ===
        self.outputs = {
            'dynamics_image': 'image',
            'transition_matrix': 'image',
            'persistence_spectrum': 'spectrum',  # 10-dim decay constants
            'mode_velocity': 'signal',           # Speed through mode space
            'low_high_lag': 'signal',            # Lag between low/high modes
            'attack_ratio': 'signal',            # Band vs mode attack speed
            'state_stability': 'signal',         # Mode configuration stability
            'dominant_trajectory': 'signal',     # Direction of movement
            'mode_entropy': 'signal',            # Distribution entropy
            'coupling_strength': 'signal',       # Inter-mode coupling
        }
        
        # === CONFIGURATION ===
        self.history_length = 200
        self.decay_window = 30      # Frames to measure decay
        self.transition_threshold = 0.3
        
        # === STATE ===
        self.n_modes = 10
        
        # Mode histories
        self.mode_history = [deque(maxlen=self.history_length) for _ in range(self.n_modes)]
        self.alpha_history = deque(maxlen=self.history_length)
        self.theta_history = deque(maxlen=self.history_length)
        
        # Computed dynamics
        self.persistence = np.ones(self.n_modes)      # Decay time constants
        self.attack_times = np.ones(self.n_modes)     # Rise times
        self.transition_matrix = np.zeros((self.n_modes, self.n_modes))  # Mode i -> Mode j
        
        # Derivative histories for velocity
        self.mode_derivatives = [deque(maxlen=50) for _ in range(self.n_modes)]
        
        # Peak detection state
        self.last_peaks = np.zeros(self.n_modes)
        self.peak_times = np.zeros(self.n_modes)
        self.frame_count = 0
        
        # Images
        self.dynamics_image = None
        self.transition_image = None
        
    def step(self):
        """Collect mode values and analyze dynamics"""
        self.frame_count += 1
        
        # Collect current mode values
        current_modes = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            val = self.get_blended_input(f'mode_{i+1}', 'sum')
            current_modes[i] = float(val) if val is not None else 0.0
            self.mode_history[i].append(current_modes[i])
        
        # Collect band values
        alpha = self.get_blended_input('alpha_power', 'sum')
        theta = self.get_blended_input('theta_power', 'sum')
        self.alpha_history.append(float(alpha) if alpha is not None else 0.0)
        self.theta_history.append(float(theta) if theta is not None else 0.0)
        
        # Compute derivatives
        for i in range(self.n_modes):
            if len(self.mode_history[i]) >= 2:
                deriv = self.mode_history[i][-1] - self.mode_history[i][-2]
                self.mode_derivatives[i].append(deriv)
        
        # Update dynamics analysis every few frames
        if self.frame_count % 5 == 0:
            self._analyze_persistence()
            self._analyze_transitions(current_modes)
            self._render_dynamics()
            self._render_transitions()
    
    def _analyze_persistence(self):
        """
        Measure how long each mode "rings" after activation.
        This is the key observation: low modes should persist longer.
        """
        for i in range(self.n_modes):
            history = np.array(list(self.mode_history[i]))
            if len(history) < self.decay_window:
                continue
            
            # Find peaks in absolute value
            abs_history = np.abs(history)
            
            # Simple peak detection: find local maxima
            peaks = []
            for j in range(1, len(abs_history) - 1):
                if abs_history[j] > abs_history[j-1] and abs_history[j] > abs_history[j+1]:
                    if abs_history[j] > np.mean(abs_history) + np.std(abs_history):
                        peaks.append(j)
            
            if len(peaks) > 0:
                # Measure decay after most recent peak
                last_peak = peaks[-1]
                if last_peak < len(history) - 5:
                    # Fit exponential decay: A * exp(-t/tau)
                    decay_segment = abs_history[last_peak:min(last_peak + self.decay_window, len(history))]
                    if len(decay_segment) > 3 and decay_segment[0] > 0:
                        # Estimate tau from half-life
                        peak_val = decay_segment[0]
                        half_val = peak_val / 2
                        
                        # Find when it crosses half
                        half_idx = np.where(decay_segment < half_val)[0]
                        if len(half_idx) > 0:
                            tau = half_idx[0] / 0.693  # t_half = tau * ln(2)
                            self.persistence[i] = 0.9 * self.persistence[i] + 0.1 * tau
                        else:
                            # Hasn't decayed to half yet - long persistence
                            self.persistence[i] = 0.9 * self.persistence[i] + 0.1 * self.decay_window
    
    def _analyze_transitions(self, current_modes):
        """
        Track which modes activate after which.
        Builds a transition probability matrix.
        """
        # Find currently dominant mode
        abs_modes = np.abs(current_modes)
        if np.max(abs_modes) > self.transition_threshold:
            dominant = np.argmax(abs_modes)
            
            # Check if this is a new dominant mode
            prev_dominant = np.argmax(self.last_peaks)
            if dominant != prev_dominant and self.last_peaks[prev_dominant] > self.transition_threshold:
                # Record transition: prev -> current
                self.transition_matrix[prev_dominant, dominant] += 1
            
            self.last_peaks = abs_modes.copy()
        
        # Normalize transition matrix for display
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
    
    def _render_dynamics(self):
        """Visualize the dynamics: persistence bars + mode traces"""
        h, w = 200, 350
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # === LEFT SIDE: Persistence bars ===
        bar_width = 15
        max_persist = max(np.max(self.persistence), 1)
        
        cv2.putText(img, "Persistence (tau)", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        for i in range(self.n_modes):
            x = 10 + i * bar_width
            bar_h = int((self.persistence[i] / max_persist) * 80)
            
            # Color gradient: low modes = blue (slow), high modes = red (fast)
            r = int(255 * i / self.n_modes)
            b = int(255 * (1 - i / self.n_modes))
            color = (b, 100, r)
            
            cv2.rectangle(img, (x, 100 - bar_h), (x + bar_width - 2, 100), color, -1)
            
            # Mode number
            cv2.putText(img, str(i+1), (x + 2, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # === RIGHT SIDE: Mode traces (recent history) ===
        trace_x = 180
        trace_w = 160
        trace_h = 180
        
        cv2.putText(img, "Mode Traces", (trace_x, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw traces for modes 1, 3, 5, 7, 9 (subset for clarity)
        trace_modes = [0, 2, 4, 6, 8]
        trace_colors = [
            (255, 100, 100),  # Mode 1 - blue
            (100, 255, 100),  # Mode 3 - green
            (100, 100, 255),  # Mode 5 - red
            (255, 255, 100),  # Mode 7 - cyan
            (255, 100, 255),  # Mode 9 - magenta
        ]
        
        for mi, mode_idx in enumerate(trace_modes):
            history = list(self.mode_history[mode_idx])
            if len(history) < 2:
                continue
            
            # Normalize
            hist_arr = np.array(history[-100:])  # Last 100 samples
            if len(hist_arr) > 1:
                h_min, h_max = hist_arr.min(), hist_arr.max()
                if h_max > h_min:
                    hist_norm = (hist_arr - h_min) / (h_max - h_min)
                else:
                    hist_norm = np.zeros_like(hist_arr)
                
                # Draw trace
                n_points = len(hist_norm)
                for j in range(1, n_points):
                    x1 = trace_x + int((j-1) / n_points * trace_w)
                    x2 = trace_x + int(j / n_points * trace_w)
                    y1 = 25 + mi * 35 + int((1 - hist_norm[j-1]) * 30)
                    y2 = 25 + mi * 35 + int((1 - hist_norm[j]) * 30)
                    cv2.line(img, (x1, y1), (x2, y2), trace_colors[mi], 1)
                
                # Label
                cv2.putText(img, f"M{mode_idx+1}", (trace_x + trace_w + 5, 35 + mi * 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, trace_colors[mi], 1)
        
        # === BOTTOM: Velocity indicator ===
        velocity = self._compute_velocity()
        vel_bar_w = int(min(abs(velocity) * 50, 100))
        vel_color = (100, 255, 100) if velocity > 0 else (100, 100, 255)
        
        cv2.putText(img, f"Velocity: {velocity:.2f}", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.rectangle(img, (100, h - 25), (100 + vel_bar_w, h - 15), vel_color, -1)
        
        self.dynamics_image = img
    
    def _render_transitions(self):
        """Render transition matrix as heatmap"""
        h, w = 150, 150
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Normalize matrix
        mat = self.transition_matrix.copy()
        mat_max = mat.max()
        if mat_max > 0:
            mat = mat / mat_max
        
        cell_size = 12
        offset_x, offset_y = 25, 20
        
        cv2.putText(img, "Transitions", (5, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                x = offset_x + j * cell_size
                y = offset_y + i * cell_size
                
                val = mat[i, j]
                color = (int(val * 50), int(val * 200), int(val * 255))
                
                cv2.rectangle(img, (x, y), (x + cell_size - 1, y + cell_size - 1), color, -1)
        
        # Axis labels
        cv2.putText(img, "From", (2, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (120, 120, 120), 1)
        cv2.putText(img, "To", (w - 20, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (120, 120, 120), 1)
        
        self.transition_image = img
    
    def _compute_velocity(self):
        """Compute velocity through mode space (magnitude of mode derivatives)"""
        derivs = []
        for i in range(self.n_modes):
            if len(self.mode_derivatives[i]) > 0:
                derivs.append(self.mode_derivatives[i][-1])
            else:
                derivs.append(0.0)
        return np.linalg.norm(derivs)
    
    def _compute_stability(self):
        """How stable is the current mode configuration?"""
        # Measure variance of modes over recent history
        variances = []
        for i in range(self.n_modes):
            if len(self.mode_history[i]) > 10:
                recent = list(self.mode_history[i])[-20:]
                variances.append(np.var(recent))
        
        if len(variances) > 0:
            # High variance = low stability
            return 1.0 / (1.0 + np.mean(variances))
        return 0.5
    
    def _compute_entropy(self):
        """Entropy of mode distribution (how spread is activity?)"""
        # Get current absolute mode values
        current = []
        for i in range(self.n_modes):
            if len(self.mode_history[i]) > 0:
                current.append(abs(self.mode_history[i][-1]))
            else:
                current.append(0.0)
        
        current = np.array(current)
        total = current.sum()
        
        if total > 0:
            probs = current / total
            probs = probs[probs > 0]  # Remove zeros for log
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy / np.log(self.n_modes)  # Normalize to 0-1
        return 0.0
    
    def _compute_low_high_lag(self):
        """
        Measure if low modes lead or lag high modes.
        Positive = low modes lead (global precedes local)
        Negative = high modes lead (local precedes global)
        """
        # Compare mode 1-3 (low) vs mode 8-10 (high)
        if len(self.mode_history[0]) < 30:
            return 0.0
        
        low_signal = np.array([
            list(self.mode_history[i])[-30:] for i in range(3)
        ]).mean(axis=0)
        
        high_signal = np.array([
            list(self.mode_history[i])[-30:] for i in range(7, 10)
        ]).mean(axis=0)
        
        # Cross-correlation to find lag
        low_signal = (low_signal - np.mean(low_signal)) / (np.std(low_signal) + 1e-6)
        high_signal = (high_signal - np.mean(high_signal)) / (np.std(high_signal) + 1e-6)
        
        corr = np.correlate(low_signal, high_signal, mode='full')
        lag = np.argmax(corr) - len(low_signal) + 1
        
        return float(lag)
    
    def get_output(self, port_name):
        """Return outputs"""
        if port_name == 'dynamics_image':
            return self.dynamics_image
            
        elif port_name == 'transition_matrix':
            return self.transition_image
            
        elif port_name == 'persistence_spectrum':
            return self.persistence.astype(np.float32)
            
        elif port_name == 'mode_velocity':
            return self._compute_velocity()
            
        elif port_name == 'low_high_lag':
            return self._compute_low_high_lag()
            
        elif port_name == 'attack_ratio':
            # Compare alpha attack to mode attack
            if len(self.alpha_history) > 5 and len(self.mode_history[0]) > 5:
                alpha_deriv = abs(self.alpha_history[-1] - self.alpha_history[-2])
                mode_deriv = abs(self.mode_history[0][-1] - self.mode_history[0][-2])
                return float(alpha_deriv / (mode_deriv + 1e-6))
            return 1.0
            
        elif port_name == 'state_stability':
            return self._compute_stability()
            
        elif port_name == 'dominant_trajectory':
            # Which mode is increasing most?
            max_deriv = 0
            max_mode = 0
            for i in range(self.n_modes):
                if len(self.mode_derivatives[i]) > 0:
                    d = self.mode_derivatives[i][-1]
                    if abs(d) > abs(max_deriv):
                        max_deriv = d
                        max_mode = i + 1
            return float(max_mode * np.sign(max_deriv))
            
        elif port_name == 'mode_entropy':
            return self._compute_entropy()
            
        elif port_name == 'coupling_strength':
            # Mean off-diagonal transition probability
            mat = self.transition_matrix.copy()
            np.fill_diagonal(mat, 0)
            total = mat.sum()
            if total > 0:
                return float(total / (self.n_modes * (self.n_modes - 1)))
            return 0.0
            
        return None
    
    def get_display_image(self):
        """Return display for node preview"""
        if self.dynamics_image is not None:
            img = np.ascontiguousarray(self.dynamics_image)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        else:
            w, h = 100, 50
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(img, "Waiting...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            img = np.ascontiguousarray(img)
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None),
            ("Decay Window", "decay_window", self.decay_window, None),
            ("Transition Threshold", "transition_threshold", self.transition_threshold, None),
        ]