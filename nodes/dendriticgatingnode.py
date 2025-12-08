"""
Dendritic Gate Node (v2 - Stroboscopic Trigger)
-----------------------------------------------
Now includes a 'strobe' output that fires a single impulse 
at the exact center of the 11ms window.
Use this to trigger the Phase Space Plotter to capture "The Moment".
"""

import numpy as np
import cv2

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None

class DendriticGateNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Dendritic Gate (Choice)"
    NODE_COLOR = QtGui.QColor(255, 100, 50) 

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'input_stream': 'signal',  
            'gamma_phase': 'signal',   
            'window_width': 'signal'   
        }
        
        self.outputs = {
            'gated_signal': 'signal',
            'rejected_signal': 'signal', 
            'gate_status': 'signal',
            'strobe': 'signal'        # NEW: Fires once per window
        }
        
        self.phase = 0.0
        self.prev_open = False
        self.display = np.zeros((150, 300, 3), dtype=np.uint8)

    def step(self):
        # 1. READ INPUTS
        signal_in = self.get_blended_input('input_stream', 'mean') or 0.0
        
        ext_phase = self.get_blended_input('gamma_phase', 'mean')
        if ext_phase is not None:
            self.phase = ext_phase % (2*np.pi)
        else:
            self.phase = (self.phase + 0.2) % (2*np.pi)
            
        width = self.get_blended_input('window_width', 'mean') or 0.2
        width = np.clip(width, 0.05, 1.0)
        
        # 2. DREBITZ GATING
        gate_openness = np.cos(self.phase) 
        threshold = 1.0 - width
        is_open = gate_openness > threshold
        
        # 3. STROBE GENERATION
        # Fire a trigger at the exact peak (phase ~ 0)
        # We detect the transition into the open state to trigger once
        strobe = 0.0
        if is_open and not self.prev_open:
            strobe = 1.0
        self.prev_open = is_open

        # 4. SIGNAL PROCESSING
        if is_open:
            perceived = signal_in
            rejected = 0.0
            status = 1.0
        else:
            perceived = 0.0
            rejected = signal_in
            status = 0.0
            
        # 5. VISUALIZATION
        self._draw_gate(signal_in, is_open, strobe)
        
        self.set_output('gated_signal', perceived)
        self.set_output('rejected_signal', rejected)
        self.set_output('gate_status', status)
        self.set_output('strobe', strobe)

    def _draw_gate(self, signal, is_open, strobe):
        self.display.fill(20)
        
        # Visual Flash on Strobe
        if strobe > 0.5:
            self.display.fill(60)

        # Gate Bars
        gate_color = (0, 255, 0) if is_open else (0, 0, 255)
        cv2.rectangle(self.display, (140, 20), (160, 130), gate_color, 2)
        
        # Signal Trace
        sig_y = int(75 - signal * 50)
        sig_y = np.clip(sig_y, 0, 149)
        cv2.line(self.display, (0, 75), (140, sig_y), (150, 150, 150), 1)
        
        if is_open:
            cv2.line(self.display, (160, sig_y), (300, sig_y), (0, 255, 0), 2)
            
        cv2.putText(self.display, f"Phase: {self.phase:.2f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name): return getattr(self, '_outs', {}).get(name)
    def set_output(self, name, val): 
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val