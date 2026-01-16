"""
Neuro-Oscillator Node - "The Heartbeat"
=======================================
Generates biological brain rhythms to drive the Phase Gate.
Simulates the 'Scanning' mechanism of attention.

- Theta (4-8Hz): Memory/Context scanning.
- Alpha (8-12Hz): Idling/Gating.
- Gamma (40Hz): Binding/Fusion.
"""

import numpy as np
import time
import math
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class NeuroOscillatorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_TITLE = "Neuro-Oscillator"
    NODE_COLOR = QtGui.QColor(0, 200, 200) # Cyan
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'frequency': 'signal', # Hz (Speed of the strobe)
            'offset': 'signal'     # Base Phase (Starting point)
        }
        self.outputs = {
            'wave_out': 'signal',  # The Sine Wave (-1 to 1)
            'phase_out': 'signal', # The Ramp (0 to 1)
            'gate_pulse': 'signal' # The Trigger (Spike at 0)
        }
        
        self.phase_accumulator = 0.0
        self.last_time = time.time()
        
    def step(self):
        # 1. TIME KEEPING
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 2. READ INPUTS
        freq = self.get_blended_input('frequency', 'mean')
        if freq is None: freq = 1.0 # Default 1 Hz
        
        offset = self.get_blended_input('offset', 'mean')
        if offset is None: offset = 0.0
        
        # 3. OSCILLATE
        # Increment phase: dPhase = Frequency * dt
        self.phase_accumulator += freq * dt
        
        # Wrap around 0.0 -> 1.0
        self.phase_accumulator %= 1.0
        
        # 4. GENERATE WAVES
        # Linear Phase (Sawtooth) 0 -> 1
        current_phase = (self.phase_accumulator + offset) % 1.0
        
        # Sine Wave (Smooth oscillation) -1 -> 1
        # Maps 0..1 phase to 0..2pi radians
        sine_wave = math.sin(current_phase * 2 * math.pi)
        
        # Gate Pulse (Sharp spike at Phase 0)
        # Useful for resetting things or triggering updates
        pulse = math.exp(-100 * (current_phase - 0.5)**2) # Gaussian bump
        
        # 5. OUTPUTS
        self.outputs['phase_out'] = current_phase
        self.outputs['wave_out'] = sine_wave
        self.outputs['gate_pulse'] = pulse

    def get_output(self, name):
        return self.outputs.get(name)