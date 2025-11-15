"""
Phase Coupling Node - Cross-Frequency Synchronization
------------------------------------------------------
Measures phase-locking between fast and slow latent streams.

When oscillations synchronize = binding = unified consciousness
When desynchronized = fragmented = parallel processing

Uses Phase-Locking Value (PLV) and coherence metrics.
"""

import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class PhaseCouplingNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Cyan
    
    def __init__(self, history_length=100):
        super().__init__()
        self.node_title = "Phase Coupling"
        
        self.inputs = {
            'fast_stream': 'spectrum',  # High-frequency (dendritic)
            'slow_stream': 'spectrum',  # Low-frequency (somatic)
        }
        
        self.outputs = {
            'phase_coherence': 'signal',      # 0-1 (locked vs drifting)
            'coupling_strength': 'signal',    # How strongly bound
            'sync_event': 'signal',           # Spike when locking occurs
            'desync_event': 'signal',         # Spike when unlocking occurs
            'dominant_coupling': 'signal',    # Which dim couples strongest
        }
        
        self.history_length = int(history_length)
        
        # State
        self.fast_history = deque(maxlen=self.history_length)
        self.slow_history = deque(maxlen=self.history_length)
        self.coherence_history = deque(maxlen=50)
        
        self.phase_coherence = 0.0
        self.coupling_strength = 0.0
        self.sync_event = 0.0
        self.desync_event = 0.0
        self.dominant_coupling = 0.0
        
        self.prev_coherence = 0.0
        
    def _compute_phase_from_signal(self, signal_history):
        """
        Extract phase from time series using Hilbert-like approach.
        For discrete time series, use differentiation + arctan.
        """
        if len(signal_history) < 3:
            return None
            
        # Convert to array
        signals = np.array(signal_history)  # Shape: (time, dims)
        
        # Compute velocity (derivative)
        velocity = np.diff(signals, axis=0)
        
        # Compute phase as angle in phase space
        # For each dimension, phase = arctan(velocity / position)
        phases = np.arctan2(velocity[:-1], signals[:-2])
        
        return phases
    
    def _phase_locking_value(self, phase1, phase2):
        """
        Compute Phase-Locking Value between two phase signals.
        PLV = |mean(exp(i * phase_difference))|
        Returns value between 0 (no locking) and 1 (perfect locking)
        """
        if phase1 is None or phase2 is None:
            return 0.0
            
        # Get minimum common dimensions
        min_dim = min(phase1.shape[1], phase2.shape[1])
        phase1 = phase1[:, :min_dim]
        phase2 = phase2[:, :min_dim]
        
        # Align time dimension
        min_time = min(phase1.shape[0], phase2.shape[0])
        phase1 = phase1[:min_time]
        phase2 = phase2[:min_time]
        
        # Phase difference
        phase_diff = phase1 - phase2
        
        # PLV per dimension
        plv_per_dim = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
        
        # Average across dimensions
        plv = np.mean(plv_per_dim)
        
        return float(plv), plv_per_dim
    
    def step(self):
        fast_stream = self.get_blended_input('fast_stream', 'first')
        slow_stream = self.get_blended_input('slow_stream', 'first')
        
        if fast_stream is None or slow_stream is None:
            self.phase_coherence *= 0.95
            self.coupling_strength *= 0.95
            self.sync_event *= 0.8
            self.desync_event *= 0.8
            return
        
        # Store history
        self.fast_history.append(fast_stream.copy())
        self.slow_history.append(slow_stream.copy())
        
        if len(self.fast_history) < 10 or len(self.slow_history) < 10:
            return
        
        # Extract phases
        fast_phases = self._compute_phase_from_signal(list(self.fast_history))
        slow_phases = self._compute_phase_from_signal(list(self.slow_history))
        
        if fast_phases is None or slow_phases is None:
            return
        
        # Compute phase-locking value
        plv, plv_per_dim = self._phase_locking_value(fast_phases, slow_phases)
        
        self.phase_coherence = plv
        
        # Store coherence history
        self.coherence_history.append(plv)
        
        # Coupling strength = variance of coherence (stable = strong coupling)
        if len(self.coherence_history) > 5:
            coherence_variance = np.var(list(self.coherence_history)[-20:])
            # Invert: low variance = stable = strong coupling
            self.coupling_strength = np.clip(1.0 - coherence_variance * 10, 0.0, 1.0)
        
        # Dominant coupling dimension
        if plv_per_dim is not None and len(plv_per_dim) > 0:
            self.dominant_coupling = float(np.argmax(plv_per_dim))
        
        # Detect sync/desync events
        coherence_change = self.phase_coherence - self.prev_coherence
        
        # Sync event: sudden increase in coherence
        if coherence_change > 0.2:
            self.sync_event = 1.0
        else:
            self.sync_event *= 0.7
        
        # Desync event: sudden decrease in coherence
        if coherence_change < -0.2:
            self.desync_event = 1.0
        else:
            self.desync_event *= 0.7
        
        self.prev_coherence = self.phase_coherence
    
    def get_output(self, port_name):
        if port_name == 'phase_coherence':
            return self.phase_coherence
        elif port_name == 'coupling_strength':
            return self.coupling_strength
        elif port_name == 'sync_event':
            return self.sync_event
        elif port_name == 'desync_event':
            return self.desync_event
        elif port_name == 'dominant_coupling':
            return self.dominant_coupling
        return None
    
    def get_display_image(self):
        w, h = 256, 256
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Top: Coherence history
        if len(self.coherence_history) > 1:
            coherence_arr = np.array(list(self.coherence_history))
            
            y_coords = h//3 - 10 - (coherence_arr * (h//3 - 40)).astype(int)
            x_coords = np.linspace(0, w - 1, len(coherence_arr)).astype(int)
            
            pts = np.vstack((x_coords, y_coords)).T
            cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        
        # Middle: Phase coherence bar
        y_mid = h//3 + 10
        coherence_w = int(np.clip(self.phase_coherence, 0, 1) * w)
        
        # Color: desynchronized (red) → synchronized (cyan)
        color_r = int(255 * (1.0 - self.phase_coherence))
        color_g = int(255 * self.phase_coherence)
        color_b = int(255 * self.phase_coherence)
        cv2.rectangle(display, (0, y_mid), (coherence_w, y_mid + 40), 
                     (color_r, color_g, color_b), -1)
        
        # Coupling strength bar
        y_coupling = y_mid + 50
        coupling_w = int(np.clip(self.coupling_strength, 0, 1) * w)
        cv2.rectangle(display, (0, y_coupling), (coupling_w, y_coupling + 20), (0, 255, 0), -1)
        
        # Event indicators
        if self.sync_event > 0.5:
            cv2.circle(display, (w - 40, h//3 + 30), 15, (0, 255, 255), -1)
            cv2.putText(display, "SYNC", (w - 60, h//3 + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if self.desync_event > 0.5:
            cv2.circle(display, (w - 40, h//3 + 60), 15, (0, 0, 255), -1)
            cv2.putText(display, "DESYNC", (w - 75, h//3 + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(display, "PHASE COHERENCE", (10, 20), font, 0.4, (255, 255, 255), 1)
        
        # State
        if self.phase_coherence > 0.7:
            state = "SYNCHRONIZED"
            color = (0, 255, 255)
        elif self.phase_coherence < 0.3:
            state = "FRAGMENTED"
            color = (0, 0, 255)
        else:
            state = "TRANSITIONAL"
            color = (255, 255, 0)
        
        cv2.putText(display, state, (10, y_mid + 25), font, 0.5, color, 2)
        cv2.putText(display, f"{self.phase_coherence:.3f}", (w - 70, y_mid + 25), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Metrics
        cv2.putText(display, f"Coherence: {self.phase_coherence:.3f}", (10, h - 60),
                   font, 0.4, (0, 255, 255), 1)
        cv2.putText(display, f"Coupling:  {self.coupling_strength:.3f}", (10, h - 40),
                   font, 0.4, (0, 255, 0), 1)
        cv2.putText(display, f"Dom Dim:   {int(self.dominant_coupling)}", (10, h - 20),
                   font, 0.4, (255, 255, 255), 1)
        
        # Theory note
        cv2.putText(display, "Fast-Slow Phase Lock = Binding", (10, h - 5),
                   font, 0.3, (150, 150, 150), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None),
        ]