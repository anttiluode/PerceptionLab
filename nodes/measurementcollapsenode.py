"""
Measurement Collapse Node - Forces probabilistic state to definite outcome
Based on quantum measurement postulate: measurement destroys superposition
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class MeasurementCollapseNode(BaseNode):
    """
    Collapses a superposition state to a definite eigenstate.
    Implements probabilistic measurement with Born rule.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(220, 100, 100)
    
    def __init__(self, collapse_strength=10.0):
        super().__init__()
        self.node_title = "Measurement"
        
        self.inputs = {
            'state_in': 'spectrum',
            'trigger': 'signal',
            'basis': 'spectrum'
        }
        self.outputs = {
            'state_out': 'spectrum',
            'collapsed_state': 'spectrum',
            'measurement_result': 'signal',
            'probability': 'signal',
            'measured': 'signal'
        }
        
        self.collapse_strength = float(collapse_strength)
        
        # INITIALIZE properly
        self.collapsed = np.zeros(16, dtype=np.float32)
        self.result = 0.0
        self.prob = 0.0
        self.was_measured = 0.0
        
    def step(self):
        state = self.get_blended_input('state_in', 'first')
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        basis = self.get_blended_input('basis', 'first')
        
        if state is None:
            if self.collapsed is None:
                self.collapsed = np.zeros(16, dtype=np.float32)
            return
            
        self.was_measured = 0.0
        
        if trigger > 0.5:
            # MEASUREMENT EVENT
            self.was_measured = 1.0
            
            # If custom basis provided, project onto it first
            if basis is not None and len(basis) == len(state):
                projection = np.dot(state, basis) / (np.dot(basis, basis) + 1e-9)
                state_to_measure = state * projection
            else:
                state_to_measure = state
                
            # Born rule: probabilities from squared amplitudes
            amplitudes = np.abs(state_to_measure)
            probabilities = amplitudes ** 2
            prob_sum = probabilities.sum()
            
            if prob_sum > 1e-9:
                probabilities = probabilities / prob_sum
                
                # Stochastic collapse
                outcome_idx = np.random.choice(len(state), p=probabilities)
                
                # Collapse
                self.collapsed = np.zeros_like(state, dtype=np.float32)
                self.collapsed[outcome_idx] = np.sign(state[outcome_idx]) if state[outcome_idx] != 0 else 1.0
                
                # Apply collapse strength
                self.collapsed = np.tanh(self.collapsed * self.collapse_strength).astype(np.float32)
                
                # Record measurement outcome
                self.result = outcome_idx / len(state)
                self.prob = probabilities[outcome_idx]
            else:
                # State is zero
                self.collapsed = np.zeros_like(state, dtype=np.float32)
                self.collapsed[0] = 1.0
                self.result = 0.0
                self.prob = 1.0
        else:
            # No measurement
            self.collapsed = state.copy().astype(np.float32)
            
            # Compute most likely outcome
            amplitudes = np.abs(state)
            if amplitudes.sum() > 1e-9:
                dominant_idx = np.argmax(amplitudes)
                self.result = dominant_idx / len(state)
                probabilities = amplitudes ** 2
                probabilities = probabilities / probabilities.sum()
                self.prob = probabilities[dominant_idx]
            else:
                self.result = 0.0
                self.prob = 0.0
                
    def get_output(self, port_name):
        if port_name == 'state_out':
            if self.collapsed is not None:
                return self.collapsed.astype(np.float32)
            return np.zeros(16, dtype=np.float32)
            
        elif port_name == 'collapsed_state':
            if self.collapsed is not None:
                return np.tanh(self.collapsed * self.collapse_strength).astype(np.float32)
            return np.zeros(16, dtype=np.float32)
            
        elif port_name == 'measurement_result':
            return float(self.result)
        elif port_name == 'probability':
            return float(self.prob)
        elif port_name == 'measured':
            return float(self.was_measured)
        return None
        
    def get_display_image(self):
        """Visualize measurement process"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.collapsed is None:
            cv2.putText(img, "Waiting for state...", (10, 128),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
            
        dimensions = len(self.collapsed)
        bar_width = max(1, w // dimensions)
        
        # Normalize for display
        state_norm = self.collapsed.copy()
        state_max = np.abs(state_norm).max()
        if state_max > 1e-6:
            state_norm = state_norm / state_max
            
        # Draw state
        for i, val in enumerate(state_norm):
            x = i * bar_width
            h_bar = int(abs(val) * 100)
            y_base = 150
            
            # Highlight measured eigenstate
            if abs(val) > 0.8:
                color = (255, 255, 0)
            elif val >= 0:
                color = (0, int(255 * abs(val)), 255)
            else:
                color = (255, int(255 * abs(val)), 0)
                
            if val >= 0:
                cv2.rectangle(img, (x, y_base-h_bar), (x+bar_width-1, y_base), color, -1)
            else:
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+h_bar), color, -1)
                
        # Baseline
        cv2.line(img, (0, 150), (w, 150), (100,100,100), 1)
        
        # Measurement info
        if self.was_measured > 0.5:
            cv2.putText(img, "MEASURED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(img, "Ready to measure", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                       
        cv2.putText(img, f"Result: {self.result:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"P(outcome): {self.prob:.3f}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Collapse Strength", "collapse_strength", self.collapse_strength, None)
        ]