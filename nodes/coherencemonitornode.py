"""
Coherence Monitor Node - Measures quantum-like properties of latent states
Tracks entropy, purity, stability - the hallmarks of coherent states
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class CoherenceMonitorNode(BaseNode):
    """
    Monitors how "quantum-like" a state is by tracking multiple metrics.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(180, 180, 100)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Coherence Monitor"
        
        self.inputs = {
            'state': 'spectrum'
        }
        self.outputs = {
            'coherence': 'signal',  # Overall coherence (0-1)
            'entropy': 'signal',  # Shannon entropy (normalized)
            'purity': 'signal',  # State purity (0-1)
            'stability': 'signal',  # Temporal stability (0-1)
            'energy': 'signal'  # State energy
        }
        
        self.history = []
        self.max_history = 100
        
        # Metrics
        self.coherence_value = 0.0
        self.entropy_value = 0.0
        self.purity_value = 0.0
        self.stability_value = 0.0
        self.energy_value = 0.0
        
    def step(self):
        state = self.get_blended_input('state', 'first')
        
        if state is None:
            return
            
        # Store history
        self.history.append(state.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # 1. Entropy (low = coherent, pure state)
        # Convert to probability distribution
        state_abs = np.abs(state)
        state_sum = state_abs.sum()
        if state_sum > 1e-9:
            probs = state_abs / state_sum
            # Shannon entropy
            self.entropy_value = -np.sum(probs * np.log(probs + 1e-9))
            # Normalize by max possible entropy
            max_entropy = np.log(len(state))
            self.entropy_value = self.entropy_value / max_entropy
        else:
            self.entropy_value = 0.0
            
        # 2. Purity (high = pure state, low = mixed state)
        # For density matrix ρ, purity = Tr(ρ²)
        # For state vector, purity ≈ sum of squared probabilities
        if state_sum > 1e-9:
            probs = state_abs / state_sum
            self.purity_value = np.sum(probs ** 2)
        else:
            self.purity_value = 0.0
            
        # 3. Temporal stability (low variance over time = coherent)
        if len(self.history) > 10:
            recent = np.array(self.history[-10:])
            # Compute variance across time for each dimension
            variance = np.var(recent, axis=0).mean()
            # Convert to stability metric (high = stable)
            self.stability_value = 1.0 / (1.0 + variance * 10.0)
        else:
            self.stability_value = 0.5
            
        # 4. Energy (magnitude of state vector)
        self.energy_value = np.sum(state ** 2)
        
        # 5. Overall coherence (combination of metrics)
        # High purity + low entropy + high stability = high coherence
        self.coherence_value = (
            self.purity_value * 0.4 +
            (1.0 - self.entropy_value) * 0.3 +
            self.stability_value * 0.3
        )
        
    def get_output(self, port_name):
        if port_name == 'coherence':
            return float(self.coherence_value)
        elif port_name == 'entropy':
            return float(self.entropy_value)
        elif port_name == 'purity':
            return float(self.purity_value)
        elif port_name == 'stability':
            return float(self.stability_value)
        elif port_name == 'energy':
            return float(self.energy_value)
        return None
        
    def get_display_image(self):
        """Visualize all coherence metrics"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw metrics as bars
        metrics = [
            ("Coherence", self.coherence_value, (0, 255, 255)),
            ("Purity", self.purity_value, (0, 255, 0)),
            ("Stability", self.stability_value, (255, 255, 0)),
            ("Entropy (inv)", 1.0 - self.entropy_value, (255, 0, 255))
        ]
        
        bar_height = h // len(metrics)
        
        for i, (name, value, color) in enumerate(metrics):
            y_start = i * bar_height
            bar_width_px = int(value * (w - 60))
            
            # Draw bar
            cv2.rectangle(img, (50, y_start + 10), (50 + bar_width_px, y_start + bar_height - 10),
                         color, -1)
            
            # Draw label
            cv2.putText(img, name, (5, y_start + bar_height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                       
            # Draw value
            cv2.putText(img, f"{value:.3f}", (w - 50, y_start + bar_height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Overall status
        status = "COHERENT" if self.coherence_value > 0.7 else "DECOHERENT" if self.coherence_value < 0.3 else "MIXED"
        status_color = (0, 255, 0) if self.coherence_value > 0.7 else (0, 0, 255) if self.coherence_value < 0.3 else (255, 255, 0)
        
        cv2.putText(img, status, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)