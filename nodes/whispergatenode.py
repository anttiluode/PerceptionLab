"""
Whisper Gate Node - Applies infinitesimal bias to guide evolution
Based on Whisper Quantum Computer's "Ultra-Light Gates"

Instead of forcing a state change, whispers a suggestion through statistical bias.
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class WhisperGateNode(BaseNode):
    """
    Generates gentle bias vectors to guide chaotic field evolution.
    Multiple gate types implement different transformations.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(150, 100, 150)
    
    def __init__(self, gate_type='hadamard', strength=0.01):
        super().__init__()
        self.node_title = f"Whisper Gate"
        
        self.inputs = {
            'state_in': 'spectrum',
            'strength': 'signal',  # How loud the whisper (0.001 - 1.0)
            'target': 'spectrum'  # Optional target state to whisper toward
        }
        self.outputs = {
            'bias_out': 'spectrum',
            'gate_active': 'signal'
        }
        
        self.gate_type = gate_type  # 'hadamard', 'pauli_x', 'pauli_z', 'phase', 'identity', 'custom'
        self.strength = float(strength)
        self.bias = None
        
    def step(self):
        state = self.get_blended_input('state_in', 'first')
        strength_signal = self.get_blended_input('strength', 'sum')
        target = self.get_blended_input('target', 'first')
        
        if strength_signal is not None:
            strength = strength_signal * 0.1  # Scale down for ultra-light
        else:
            strength = self.strength
            
        if state is None:
            self.bias = np.zeros(16)  # Default dimension
            return
            
        dimensions = len(state)
        
        # Generate bias based on gate type
        if self.gate_type == 'hadamard':
            # Create 50/50 superposition bias (push toward zero)
            target_state = np.zeros_like(state)
            self.bias = (target_state - state) * strength
            
        elif self.gate_type == 'pauli_x':
            # Flip bias (push toward opposite sign)
            target_state = -state
            self.bias = (target_state - state) * strength * 0.5
            
        elif self.gate_type == 'pauli_z':
            # Phase flip (invert alternate dimensions)
            target_state = state.copy()
            target_state[1::2] *= -1  # Flip every other dimension
            self.bias = (target_state - state) * strength
            
        elif self.gate_type == 'phase':
            # Rotate in phase space (shift dimensions)
            target_state = np.roll(state, 1)
            self.bias = (target_state - state) * strength
            
        elif self.gate_type == 'identity':
            # No bias (useful for testing)
            self.bias = np.zeros_like(state)
            
        elif self.gate_type == 'custom':
            # Use provided target state
            if target is not None and len(target) == dimensions:
                self.bias = (target - state) * strength
            else:
                self.bias = np.zeros_like(state)
                
        elif self.gate_type == 'amplify':
            # Push away from zero (increase magnitude)
            self.bias = state * strength
            
        elif self.gate_type == 'dampen':
            # Push toward zero (decrease magnitude)
            self.bias = -state * strength
            
        else:
            self.bias = np.zeros_like(state)
            
    def get_output(self, port_name):
        if port_name == 'bias_out':
            return self.bias.astype(np.float32) if self.bias is not None else None
        elif port_name == 'gate_active':
            return 1.0 if np.abs(self.bias).max() > 1e-6 else 0.0
        return None
        
    def get_display_image(self):
        """Visualize the bias vector"""
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.bias is None:
            cv2.putText(img, "Waiting for input...", (10, 64),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
            
        dimensions = len(self.bias)
        bar_width = max(1, w // dimensions)
        
        # Normalize for display
        bias_norm = self.bias.copy()
        bias_max = np.abs(bias_norm).max()
        if bias_max > 1e-6:
            bias_norm = bias_norm / bias_max
            
        for i, val in enumerate(bias_norm):
            x = i * bar_width
            h_bar = int(abs(val) * (h//2 - 10))
            y_base = h // 2
            
            if val >= 0:
                color = (0, int(255 * abs(val)), 0)  # Green = positive bias
                cv2.rectangle(img, (x, y_base-h_bar), (x+bar_width-1, y_base), color, -1)
            else:
                color = (int(255 * abs(val)), 0, 0)  # Red = negative bias
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+h_bar), color, -1)
                
        # Baseline
        cv2.line(img, (0, h//2), (w, h//2), (100,100,100), 1)
        
        # Gate type label
        cv2.putText(img, f"Gate: {self.gate_type.upper()}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.putText(img, f"Strength: {self.strength:.4f}", (5, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Gate Type", "gate_type", self.gate_type, 
             ['hadamard', 'pauli_x', 'pauli_z', 'phase', 'identity', 'custom', 'amplify', 'dampen']),
            ("Strength", "strength", self.strength, None)
        ]