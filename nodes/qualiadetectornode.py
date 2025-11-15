"""
Qualia Detector Node
--------------------
Implements the consciousness equation:

Q(t) = FD[ P(t+1 | S(t-∞:t)) - S(t) ]

Where:
- Q(t) = qualia intensity at time t
- FD[] = fractal dimension operator
- P(t+1) = predicted next state (from past trajectory)
- S(t-∞:t) = sensory history (slow_latent)
- S(t) = current sensation (fast_latent)

Qualia emerges from the fractal structure of prediction error
between what you expected to sense and what you actually sense.
"""

import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class QualiaDetectorNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 100, 255)  # Bright magenta
    
    def __init__(self, history_length=50):
        super().__init__()
        self.node_title = "Qualia Detector"
        
        self.inputs = {
            'fast_latent': 'spectrum',    # Present sensation S(t)
            'slow_latent': 'spectrum',    # Past state / prediction basis
        }
        
        self.outputs = {
            'qualia_intensity': 'signal',      # Q(t) - the consciousness level
            'prediction_error': 'signal',      # ||P(t+1) - S(t)||
            'error_fd': 'signal',              # FD of error history
            'predicted_sensation': 'spectrum', # P(t+1) for visualization
        }
        
        self.history_length = int(history_length)
        
        # State
        self.slow_history = deque(maxlen=self.history_length)
        self.error_history = deque(maxlen=self.history_length)
        
        self.qualia_intensity = 0.0
        self.prediction_error = 0.0
        self.error_fd = 1.0
        self.predicted_sensation = None
        
        # Initialize histories
        for _ in range(self.history_length):
            self.error_history.append(0.0)
    
    def _predict_next_state(self, slow_history):
        """
        Predict next state P(t+1) from trajectory of past states.
        Uses simple linear extrapolation from recent history.
        """
        if len(slow_history) < 2:
            return slow_history[-1] if len(slow_history) > 0 else None
        
        # Get last two states
        recent = np.array(list(slow_history)[-5:])  # Last 5 frames
        
        # Fit linear trend and extrapolate
        if len(recent) >= 2:
            # Simple momentum-based prediction
            velocity = recent[-1] - recent[-2]
            prediction = recent[-1] + velocity
            return prediction
        
        return recent[-1]
    
    def _calculate_fd_1d(self, series):
        """Calculate fractal dimension using Higuchi method"""
        series = np.array(series)
        N = len(series)
        
        if N < 10:
            return 1.0
        
        k_max = min(8, N // 4)
        L_k = []
        k_vals = []
        
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                subseries = series[idx]
                
                L_m = np.sum(np.abs(np.diff(subseries))) * (N - 1) / ((len(idx) - 1) * k)
                Lk += L_m
            
            if Lk > 0:
                L_k.append(np.log(Lk / k))
                k_vals.append(np.log(1.0 / k))
        
        if len(k_vals) < 2:
            return 1.0
        
        coeffs = np.polyfit(k_vals, L_k, 1)
        fd = coeffs[0]
        
        return np.clip(fd, 1.0, 2.0)
    
    def step(self):
        fast_latent = self.get_blended_input('fast_latent', 'first')
        slow_latent = self.get_blended_input('slow_latent', 'first')
        
        if fast_latent is None or slow_latent is None:
            self.qualia_intensity *= 0.95
            return
        
        # Store slow latent history (represents S(t-∞:t))
        self.slow_history.append(slow_latent.copy())
        
        if len(self.slow_history) < 2:
            return
        
        # 1. PREDICT next sensation P(t+1) from past trajectory
        self.predicted_sensation = self._predict_next_state(self.slow_history)
        
        if self.predicted_sensation is None:
            return
        
        # 2. PROJECT to same dimensionality as fast_latent for comparison
        # Use dimensionality of fast (the actual sensation)
        min_dim = min(len(fast_latent), len(self.predicted_sensation))
        predicted_proj = self.predicted_sensation[:min_dim]
        sensation_proj = fast_latent[:min_dim]
        
        # 3. COMPUTE prediction error: ||P(t+1) - S(t)||
        error_vector = predicted_proj - sensation_proj
        self.prediction_error = np.linalg.norm(error_vector)
        
        # Store error history
        self.error_history.append(self.prediction_error)
        
        # 4. MEASURE fractal dimension of error time series
        self.error_fd = self._calculate_fd_1d(list(self.error_history))
        
        # 5. COMPUTE qualia intensity
        # Q(t) = FD[error] weighted by error magnitude
        # High FD + high error = vivid consciousness
        # Low FD or low error = dim consciousness
        
        # Normalize error to 0-1 range (assuming max ~2.0 for normalized latents)
        normalized_error = np.clip(self.prediction_error / 2.0, 0.0, 1.0)
        
        # Normalize FD to 0-1 range (1.0 to 2.0 → 0.0 to 1.0)
        normalized_fd = (self.error_fd - 1.0)
        
        # Qualia = error magnitude × error complexity
        # Both contribute: need surprise (error) AND rich structure (FD)
        self.qualia_intensity = normalized_error * normalized_fd * 0.5 + normalized_fd * 0.5
        
        # Alternative formulation (can experiment):
        # self.qualia_intensity = normalized_fd  # Pure complexity
        # self.qualia_intensity = normalized_error * normalized_fd  # Error × complexity
    
    def get_output(self, port_name):
        if port_name == 'qualia_intensity':
            return self.qualia_intensity
        elif port_name == 'prediction_error':
            return self.prediction_error
        elif port_name == 'error_fd':
            return self.error_fd
        elif port_name == 'predicted_sensation':
            return self.predicted_sensation
        return None
    
    def get_display_image(self):
        w, h = 256, 256
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Top: Error history plot
        if len(self.error_history) > 1:
            errors = np.array(list(self.error_history))
            
            # Normalize for display
            if errors.max() > errors.min():
                norm_errors = (errors - errors.min()) / (errors.max() - errors.min())
            else:
                norm_errors = errors * 0
            
            # Draw as line
            y_coords = h//2 - 10 - (norm_errors * (h//2 - 40)).astype(int)
            x_coords = np.linspace(0, w - 1, len(errors)).astype(int)
            
            pts = np.vstack((x_coords, y_coords)).T
            cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        
        # Middle: Qualia intensity bar
        y_bar_start = h//2 + 10
        qualia_w = int(np.clip(self.qualia_intensity, 0, 1) * w)
        
        # Color code: dim (blue) → vivid (magenta)
        color_r = int(255 * self.qualia_intensity)
        color_b = int(255 * (1.0 - self.qualia_intensity * 0.5))
        cv2.rectangle(display, (0, y_bar_start), (qualia_w, y_bar_start + 40), 
                     (color_r, 0, color_b), -1)
        
        # Bottom: FD bar
        y_fd_start = y_bar_start + 50
        fd_normalized = (self.error_fd - 1.0)  # 0-1
        fd_w = int(np.clip(fd_normalized, 0, 1) * w)
        cv2.rectangle(display, (0, y_fd_start), (fd_w, y_fd_start + 20), (0, 255, 0), -1)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(display, "PREDICTION ERROR", (10, 20), font, 0.4, (255, 255, 255), 1)
        
        # Qualia intensity with descriptor
        qualia_state = "VIVID" if self.qualia_intensity > 0.7 else \
                      "DIM" if self.qualia_intensity < 0.3 else "MODERATE"
        cv2.putText(display, f"QUALIA: {qualia_state}", (10, y_bar_start + 25), 
                   font, 0.5, (255, 255, 255), 2)
        cv2.putText(display, f"{self.qualia_intensity:.3f}", (w - 70, y_bar_start + 25), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Metrics
        cv2.putText(display, f"Error: {self.prediction_error:.3f}", (10, h - 50),
                   font, 0.4, (0, 255, 255), 1)
        cv2.putText(display, f"FD: {self.error_fd:.3f}", (10, h - 30),
                   font, 0.4, (0, 255, 0), 1)
        
        # The equation
        cv2.putText(display, "Q(t) = FD[P(t+1) - S(t)]", (10, h - 10),
                   font, 0.35, (150, 150, 150), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None),
        ]