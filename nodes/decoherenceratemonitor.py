"""
Decoherence Rate Monitor - Measures how fast quantum-like states decay
Tracks the rate at which coherence is lost over time
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class DecoherenceMonitorNode(BaseNode):
    """
    Monitors decoherence rate by tracking coherence decay over time.
    Fits exponential decay model to coherence measurements.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(150, 150, 200)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Decoherence Monitor"
        
        self.inputs = {
            'coherence_in': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'decoherence_rate': 'signal',  # Rate constant (1/frames)
            'half_life': 'signal',  # Frames until coherence halves
            'projected_lifetime': 'signal',  # Frames until coherence ~0
            'decay_fit_quality': 'signal'  # R² of exponential fit
        }
        
        self.coherence_history = []
        self.time_stamps = []
        self.max_history = 500
        
        self.decoherence_rate = 0.0
        self.half_life = 0.0
        self.lifetime = 0.0
        self.fit_quality = 0.0
        
        self.frame_count = 0
        
    def step(self):
        coherence = self.get_blended_input('coherence_in', 'sum')
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        if reset > 0.5:
            self.coherence_history = []
            self.time_stamps = []
            self.frame_count = 0
            
        if coherence is not None:
            self.coherence_history.append(coherence)
            self.time_stamps.append(self.frame_count)
            self.frame_count += 1
            
            if len(self.coherence_history) > self.max_history:
                self.coherence_history.pop(0)
                self.time_stamps.pop(0)
                
        # Fit exponential decay if enough data
        if len(self.coherence_history) > 20:
            self._fit_decay()
            
    def _fit_decay(self):
        """Fit exponential decay: C(t) = C₀ * exp(-λt)"""
        times = np.array(self.time_stamps)
        coherences = np.array(self.coherence_history)
        
        # Remove zeros and negative values for log fit
        valid = coherences > 1e-6
        if valid.sum() < 10:
            return
            
        times = times[valid]
        coherences = coherences[valid]
        
        # Linear fit in log space: log(C) = log(C₀) - λt
        log_coherences = np.log(coherences)
        
        # Fit line
        coeffs = np.polyfit(times - times[0], log_coherences, 1)
        self.decoherence_rate = -coeffs[0]  # λ = -slope
        
        # Half-life: t₁/₂ = ln(2) / λ
        if self.decoherence_rate > 1e-6:
            self.half_life = np.log(2) / self.decoherence_rate
            self.lifetime = 4.6 / self.decoherence_rate  # ~99% decay
        else:
            self.half_life = float('inf')
            self.lifetime = float('inf')
            
        # Fit quality (R²)
        predicted = np.exp(coeffs[1] + coeffs[0] * (times - times[0]))
        ss_res = np.sum((coherences - predicted) ** 2)
        ss_tot = np.sum((coherences - coherences.mean()) ** 2)
        
        if ss_tot > 1e-9:
            self.fit_quality = 1.0 - (ss_res / ss_tot)
        else:
            self.fit_quality = 0.0
            
    def get_output(self, port_name):
        if port_name == 'decoherence_rate':
            return float(self.decoherence_rate)
        elif port_name == 'half_life':
            return float(min(self.half_life, 1000.0))  # Cap at 1000 frames
        elif port_name == 'projected_lifetime':
            return float(min(self.lifetime, 5000.0))  # Cap at 5000 frames
        elif port_name == 'decay_fit_quality':
            return float(self.fit_quality)
        return None
        
    def get_display_image(self):
        """Visualize coherence decay"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.coherence_history) < 2:
            cv2.putText(img, "Collecting data...", (10, 128),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
            
        # Plot coherence over time
        times = np.array(self.time_stamps)
        coherences = np.array(self.coherence_history)
        
        # Normalize time to plot width
        time_range = times.max() - times.min() if times.max() > times.min() else 1
        
        for i in range(1, len(times)):
            x1 = int((times[i-1] - times[0]) / time_range * w)
            y1 = int((1.0 - coherences[i-1]) * (h - 50))
            x2 = int((times[i] - times[0]) / time_range * w)
            y2 = int((1.0 - coherences[i]) * (h - 50))
            
            x1 = np.clip(x1, 0, w-1)
            y1 = np.clip(y1, 0, h-50)
            x2 = np.clip(x2, 0, w-1)
            y2 = np.clip(y2, 0, h-50)
            
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
        # Draw exponential fit if available
        if self.decoherence_rate > 1e-6 and len(times) > 20:
            for x in range(0, w, 2):
                t = (x / w) * time_range
                c = np.exp(-self.decoherence_rate * t)
                y = int((1.0 - c) * (h - 50))
                y = np.clip(y, 0, h-50)
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
                
        # Info text
        cv2.putText(img, f"Rate: {self.decoherence_rate:.5f} /frame", (5, h-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(img, f"Half-life: {self.half_life:.1f} frames", (5, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(img, f"R²: {self.fit_quality:.3f}", (5, h-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        
        # Lifetime indicator
        if self.half_life < 100:
            color = (0, 0, 255)  # Red = fast decay
            status = "RAPID DECAY"
        elif self.half_life < 500:
            color = (0, 255, 255)  # Yellow = moderate
            status = "MODERATE"
        else:
            color = (0, 255, 0)  # Green = slow decay
            status = "STABLE"
            
        cv2.putText(img, status, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
