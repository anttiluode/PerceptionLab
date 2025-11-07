"""
Meta-Dynamic Coupler Node - A simplified model of the Meta-Dynamic Ephaptic
Intelligence System. The agent learns to adjust its own internal coupling
parameter (alpha) based on prediction success.

Outputs the current learned physics parameter (alpha).
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class MetaDynamicCouplerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(255, 100, 255) # Meta-Dynamic Magenta
    
    def __init__(self, initial_coupling=0.5, learning_rate=0.01):
        super().__init__()
        self.node_title = "Meta-Dynamic Coupler"
        
        self.inputs = {
            'input_a': 'signal',          # Primary input field
            'success_target': 'signal',   # Target value (The goal state)
        }
        self.outputs = {
            'agent_output': 'signal',
            'current_coupling': 'signal', # The learned physics parameter
        }
        
        # --- Meta-Dynamic State Variables ---
        self.current_coupling = float(initial_coupling) # Alpha (the "rule")
        self.learning_rate = float(learning_rate)
        self.stabilizer = 0.5 # Keeps coupling adjustment smooth
        
        # Internal processing state
        self.internal_state = 0.0
        self.agent_output = 0.0

    def step(self):
        # 1. Get Inputs
        input_A = self.get_blended_input('input_a', 'sum') or 0.0
        target = self.get_blended_input('success_target', 'sum') or 0.0
        
        # 2. Agent's Forward Pass (The Decision/Output)
        # Decision = Internal State * Coupling + Input
        self.internal_state = self.internal_state * self.stabilizer + input_A
        self.agent_output = np.tanh(self.internal_state * self.current_coupling)
        
        # 3. Calculate Error (Success/Failure)
        # Goal: Make the output match the target using minimal change.
        error = target - self.agent_output
        
        # 4. Meta-Dynamic Learning (Rewriting the Rule/Physics)
        # The coupling (alpha) is adjusted based on the error and the input state.
        # This is a simplified form of gradient descent on the coupling equation itself.
        
        # Derivative of output w.r.t. coupling: d(tanh(I*a))/da = I * sech²(I*a)
        # We approximate the gradient as: Error * Input * (1 - Output^2)
        
        approx_grad_coupling = input_A * (1.0 - self.agent_output**2)
        
        # Update Coupling: Adjust the rule to reduce the error.
        coupling_change = self.learning_rate * error * approx_grad_coupling
        
        self.current_coupling += coupling_change
        
        # Clamp coupling to a sensible range
        self.current_coupling = np.clip(self.current_coupling, 0.01, 5.0)

    def get_output(self, port_name):
        if port_name == 'agent_output':
            return self.agent_output
        elif port_name == 'current_coupling':
            return self.current_coupling
        return None
        
    def get_display_image(self):
        w, h = 96, 96
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Visualize Learning Progress (Color represents Coupling Value)
        norm_coupling = (self.current_coupling - 0.01) / 4.99 # Normalize 0.01 to 5.0
        
        # Map coupling to green/red (success/over-coupling)
        r = int(np.clip(norm_coupling * 255, 0, 255))
        g = int(np.clip((1 - norm_coupling) * 255, 0, 255))
        
        cv2.rectangle(img, (0, 0), (w, h), (g, 0, r), -1)
        
        # Draw current output value
        output_norm = (self.agent_output + 1) / 2.0 # Map [-1, 1] to [0, 1]
        out_bar_h = int(output_norm * h)
        cv2.rectangle(img, (w//4, h - out_bar_h), (w//2, h), (255, 255, 255), -1)

        # Draw Target value
        target = self.get_blended_input('success_target', 'sum') or 0.0
        target_norm = (target + 1) / 2.0 
        target_y = h - int(target_norm * h)
        cv2.line(img, (w//2 + 5, target_y), (w - 5, target_y), (255, 255, 0), 2)
        
        cv2.putText(img, f"a={self.current_coupling:.2f}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Initial Coupling (α)", "current_coupling", self.current_coupling, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
        ]