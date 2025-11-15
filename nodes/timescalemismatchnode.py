"""
Timescale Mismatch Analyzer
----------------------------
Analyzes the disagreement between fast and slow latent spaces.

This is where consciousness emerges: when fast predictions diverge from slow predictions,
the fractal dimension of that divergence measures the "texture" of awareness.
"""

import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class TimescaleMismatchNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(200, 120, 180)
    
    def __init__(self, history_length=100):
        super().__init__()
        self.node_title = "Timescale Mismatch Analyzer"
        
        self.inputs = {
            'fast_latent': 'spectrum',
            'slow_latent': 'spectrum',
        }
        
        self.outputs = {
            'disagreement': 'signal',           # Instant mismatch
            'disagreement_fd': 'signal',        # Fractal dimension of disagreement over time
            'phase_alignment': 'signal',        # How in-sync are they
            'surprise_event': 'signal',         # Spike when major mismatch
        }
        
        self.history_length = int(history_length)
        
        # State
        self.disagreement_history = deque(maxlen=self.history_length)
        self.disagreement_value = 0.0
        self.disagreement_fd = 1.0
        self.phase_alignment = 1.0
        self.surprise_event = 0.0
        
        # For fractal dimension calculation
        for _ in range(self.history_length):
            self.disagreement_history.append(0.0)
    
    def _calculate_fd_1d(self, series):
        """Calculate fractal dimension of 1D time series using Higuchi method"""
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
                # Subseries
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                subseries = series[idx]
                
                # Length of curve
                L_m = np.sum(np.abs(np.diff(subseries))) * (N - 1) / ((len(idx) - 1) * k)
                Lk += L_m
            
            if Lk > 0:
                L_k.append(np.log(Lk / k))
                k_vals.append(np.log(1.0 / k))
        
        if len(k_vals) < 2:
            return 1.0
        
        # Fit line
        coeffs = np.polyfit(k_vals, L_k, 1)
        fd = coeffs[0]
        
        return np.clip(fd, 1.0, 2.0)
    
    def step(self):
        fast_latent = self.get_blended_input('fast_latent', 'first')
        slow_latent = self.get_blended_input('slow_latent', 'first')
        
        if fast_latent is None or slow_latent is None:
            self.disagreement_value *= 0.95
            return
        
        # Project both to common dimensionality for comparison
        # Use the smaller dimension
        min_dim = min(len(fast_latent), len(slow_latent))
        fast_proj = fast_latent[:min_dim]
        slow_proj = slow_latent[:min_dim]
        
        # Normalize
        fast_norm = fast_proj / (np.linalg.norm(fast_proj) + 1e-8)
        slow_norm = slow_proj / (np.linalg.norm(slow_proj) + 1e-8)
        
        # Disagreement = Euclidean distance between normalized latents
        self.disagreement_value = np.linalg.norm(fast_norm - slow_norm)
        
        # Phase alignment = cosine similarity
        self.phase_alignment = np.dot(fast_norm, slow_norm)
        self.phase_alignment = (self.phase_alignment + 1.0) / 2.0  # Map to 0-1
        
        # Store in history
        self.disagreement_history.append(self.disagreement_value)
        
        # Calculate fractal dimension of disagreement time series
        self.disagreement_fd = self._calculate_fd_1d(list(self.disagreement_history))
        
        # Surprise event: sudden spike in disagreement
        recent_mean = np.mean(list(self.disagreement_history)[-20:]) if len(self.disagreement_history) > 20 else 0
        recent_std = np.std(list(self.disagreement_history)[-20:]) if len(self.disagreement_history) > 20 else 1
        
        if self.disagreement_value > recent_mean + 2 * recent_std:
            self.surprise_event = 1.0
        else:
            self.surprise_event *= 0.8  # Decay
    
    def get_output(self, port_name):
        if port_name == 'disagreement':
            return self.disagreement_value
        elif port_name == 'disagreement_fd':
            return self.disagreement_fd
        elif port_name == 'phase_alignment':
            return self.phase_alignment
        elif port_name == 'surprise_event':
            return self.surprise_event
        return None
    
    def get_display_image(self):
        w, h = 256, 192
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Plot disagreement history
        if len(self.disagreement_history) > 1:
            points = np.array(list(self.disagreement_history))
            
            # Normalize
            if points.max() > points.min():
                norm_points = (points - points.min()) / (points.max() - points.min())
            else:
                norm_points = points * 0
            
            # Draw as line
            y_coords = (h * 2 // 3) - (norm_points * (h * 2 // 3 - 20)).astype(int)
            x_coords = np.linspace(0, w - 1, len(points)).astype(int)
            
            pts = np.vstack((x_coords, y_coords)).T
            cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
        
        # Draw surprise events as red spikes
        if self.surprise_event > 0.5:
            spike_h = int(self.surprise_event * h * 2 // 3)
            cv2.line(display, (w - 1, h * 2 // 3), (w - 1, h * 2 // 3 - spike_h), (0, 0, 255), 3)
        
        # Bottom third: metrics
        y_start = h * 2 // 3
        
        # Phase alignment bar
        align_w = int(self.phase_alignment * w)
        cv2.rectangle(display, (0, y_start), (align_w, y_start + 20), (255, 255, 0), -1)
        
        # FD bar
        fd_normalized = (self.disagreement_fd - 1.0) / 1.0  # Map 1-2 to 0-1
        fd_w = int(fd_normalized * w)
        cv2.rectangle(display, (0, y_start + 25), (fd_w, y_start + 45), (0, 255, 255), -1)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'DISAGREEMENT HISTORY', (10, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, f'Current: {self.disagreement_value:.4f}', (10, 40), font, 0.3, (200, 200, 200), 1)
        
        cv2.putText(display, f'Alignment: {self.phase_alignment:.3f}', 
                   (10, y_start + 15), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display, f'FD: {self.disagreement_fd:.3f}', 
                   (10, y_start + 40), font, 0.3, (255, 255, 255), 1)
        
        if self.surprise_event > 0.1:
            cv2.putText(display, 'SURPRISE!', (w - 80, h - 10), 
                       font, 0.5, (0, 0, 255), 2)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None),
        ]