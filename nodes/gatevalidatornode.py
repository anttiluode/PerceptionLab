"""
Gate Validator Node - Tests if Whisper Gates actually work
Validates quantum gate operations like Hadamard, Pauli-X, etc.
"""

import numpy as np
import cv2
from scipy import stats

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class GateValidatorNode(BaseNode):
    """
    Validates quantum gate operations by statistical testing.
    Runs repeated trials and checks if outcomes match expected distributions.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 220, 100)
    
    def __init__(self, num_trials=50):
        super().__init__()
        self.node_title = "Gate Validator"
        
        self.inputs = {
            'initial_state': 'spectrum',
            'final_state': 'spectrum',
            'gate_type': 'signal',  # 0=Hadamard, 1=Pauli-X, 2=Pauli-Z, etc.
            'trigger': 'signal'
        }
        self.outputs = {
            'is_valid': 'signal',  # 1.0 if gate worked, 0.0 if failed
            'deviation': 'signal',  # How far from expected
            'confidence': 'signal',  # Statistical confidence (0-1)
            'p_value': 'signal'  # Statistical p-value
        }
        
        self.num_trials = int(num_trials)
        
        self.trials = []
        self.is_testing = False
        self.trial_count = 0
        
        self.is_valid = 0.0
        self.deviation = 0.0
        self.confidence = 0.0
        self.p_value = 1.0
        
    def step(self):
        initial = self.get_blended_input('initial_state', 'first')
        final = self.get_blended_input('final_state', 'first')
        gate_type_signal = self.get_blended_input('gate_type', 'sum') or 0.0
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        
        if initial is None or final is None:
            return
            
        gate_type = int(gate_type_signal)
        
        # Start test
        if trigger > 0.5 and not self.is_testing:
            self.is_testing = True
            self.trials = []
            self.trial_count = 0
            
        # Collect trials
        if self.is_testing and self.trial_count < self.num_trials:
            self.trials.append({
                'initial': initial.copy(),
                'final': final.copy()
            })
            self.trial_count += 1
            
            if self.trial_count >= self.num_trials:
                self.is_testing = False
                self._validate_gate(gate_type)
                
    def _validate_gate(self, gate_type):
        """Validate gate operation against expected distribution"""
        if len(self.trials) == 0:
            return
            
        # Extract final states
        finals = np.array([t['final'] for t in self.trials])
        
        # Compute mean and std
        mean_final = finals.mean(axis=0)
        std_final = finals.std(axis=0)
        
        if gate_type == 0:  # Hadamard
            # Expected: all dimensions near 0 (equal superposition)
            expected = np.zeros_like(mean_final)
            self.deviation = np.abs(mean_final - expected).mean()
            
            # Should have high variance (superposition)
            expected_std = 0.5
            std_deviation = np.abs(std_final.mean() - expected_std)
            
            # Valid if mean near 0 and std near 0.5
            self.is_valid = 1.0 if (self.deviation < 0.2 and std_deviation < 0.3) else 0.0
            
        elif gate_type == 1:  # Pauli-X (bit flip)
            # Expected: negative of initial (or pushed toward +1)
            initials = np.array([t['initial'] for t in self.trials])
            mean_initial = initials.mean(axis=0)
            
            expected = -mean_initial
            self.deviation = np.abs(mean_final - expected).mean()
            
            # Valid if final ≈ -initial
            self.is_valid = 1.0 if self.deviation < 0.3 else 0.0
            
        elif gate_type == 2:  # Pauli-Z (phase flip)
            # Expected: alternate dimensions flipped
            expected = mean_final.copy()
            expected[1::2] *= -1
            
            self.deviation = np.abs(mean_final - expected).mean()
            self.is_valid = 1.0 if self.deviation < 0.3 else 0.0
            
        else:  # Identity or unknown
            # Expected: final ≈ initial
            initials = np.array([t['initial'] for t in self.trials])
            mean_initial = initials.mean(axis=0)
            
            self.deviation = np.abs(mean_final - mean_initial).mean()
            self.is_valid = 1.0 if self.deviation < 0.1 else 0.0
            
        # Statistical test (t-test against expected)
        # Simplified: check if deviation is significant
        if len(self.trials) > 10:
            # One-sample t-test
            deviations = [np.abs(t['final'] - t['initial']).mean() for t in self.trials]
            t_stat, self.p_value = stats.ttest_1samp(deviations, 0.0)
            self.confidence = 1.0 - self.p_value
        else:
            self.p_value = 1.0
            self.confidence = 0.0
            
    def get_output(self, port_name):
        if port_name == 'is_valid':
            return float(self.is_valid)
        elif port_name == 'deviation':
            return float(self.deviation)
        elif port_name == 'confidence':
            return float(self.confidence)
        elif port_name == 'p_value':
            return float(self.p_value)
        return None
        
    def get_display_image(self):
        """Visualize validation results"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Progress
        progress = self.trial_count / self.num_trials
        progress_width = int(progress * w)
        cv2.rectangle(img, (0, 0), (progress_width, 30), (0, 255, 0), -1)
        
        cv2.putText(img, f"Trials: {self.trial_count}/{self.num_trials}",
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0) if progress > 0.5 else (255,255,255), 1)
        
        # Results
        if self.trial_count >= self.num_trials:
            # Validation status
            if self.is_valid > 0.5:
                status = "PASS ✓"
                color = (0, 255, 0)
            else:
                status = "FAIL ✗"
                color = (0, 0, 255)
                
            cv2.putText(img, status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Metrics
            cv2.putText(img, f"Deviation: {self.deviation:.3f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(img, f"Confidence: {self.confidence:.3f}", (10, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(img, f"p-value: {self.p_value:.4f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Deviation bar
            dev_width = int(min(self.deviation, 1.0) * w)
            dev_color = (0, 255, 0) if self.deviation < 0.2 else (255, 255, 0) if self.deviation < 0.5 else (255, 0, 0)
            cv2.rectangle(img, (0, 200), (dev_width, 220), dev_color, -1)
            
        else:
            cv2.putText(img, "Testing...", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Num Trials", "num_trials", self.num_trials, None)
        ]