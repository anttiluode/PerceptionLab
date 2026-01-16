"""
Attention Latch Node - "The Gamma Lock"
=======================================
Gives the system AGENCY.
It decides whether to 'Look' (Phase 0) or 'Dream' (Phase Oscillation)
based on the interestingness (Fractality) of the input.

- Input: Fractal Beta (from Analyzer)
- Input: Theta Wave (from Oscillator)
- Output: Final Phase Shift (To Stack)
"""

import numpy as np
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class AttentionLatchNode(BaseNode):
    NODE_CATEGORY = "Control"
    NODE_TITLE = "Attention Latch (Agency)"
    NODE_COLOR = QtGui.QColor(255, 50, 50) # Red (Urgency)
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'fractal_beta': 'signal', # How interesting is the world?
            'theta_wave': 'signal',   # The default scanning rhythm
            'threshold': 'signal'     # Sensitivity (-2.0 is typical pink noise)
        }
        self.outputs = {
            'final_phase': 'signal',  # The command to the Brain
            'attention_state': 'signal' # 1.0 = LOCKED, 0.0 = SCANNING
        }
        
    def step(self):
        # 1. READ INPUTS
        beta = self.get_blended_input('fractal_beta', 'mean')
        if beta is None: beta = -3.0 # Boring/Smooth
        
        theta = self.get_blended_input('theta_wave', 'mean')
        if theta is None: theta = 0.0
        
        thresh = self.get_blended_input('threshold', 'mean')
        if thresh is None: thresh = -2.5 # Typical "Complex Image" threshold
        
        # 2. DECIDE
        # Beta is usually negative (-1.0 is Noise, -3.0 is Blur)
        # Interesting things (Faces) are usually around -2.0 to -2.5
        
        # If Beta is "Rougher" (Higher) than threshold, we pay attention
        # (Remember: -2.0 is > -2.5)
        is_interesting = (beta > thresh)
        
        if is_interesting:
            # GAMMA LOCK: Force eyes open
            final_phase = 0.0 
            state = 1.0 # Locked
        else:
            # THETA SCAN: Let the mind wander
            final_phase = theta
            state = 0.0 # Scanning
            
        # 3. OUTPUT
        self.outputs['final_phase'] = final_phase
        self.outputs['attention_state'] = state

    def get_output(self, name):
        return self.outputs.get(name)