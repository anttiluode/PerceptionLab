"""
The Free Will Node (Causal Re-Entry)
------------------------------------
Implements the 'Impossible Freedom' by exploiting the 11ms Delay.
It takes a noisy input (The World) and mixes it with a delayed prediction (The Self).

The 'Will' is the gain on the feedback loop.
- Low Will: The system is driven by the environment (Deterministic).
- High Will: The system is driven by its own history (Agency).
"""

import numpy as np
import cv2
from collections import deque

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None

class FreeWillNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Causal Re-Entry (Free Will)"
    NODE_COLOR = QtGui.QColor(255, 215, 0) # Gold

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'sensory_input': 'signal',   # The World (Chaos/Gamma)
            'intent_bias': 'signal'      # The Top-Down Control (Alpha)
        }
        
        self.outputs = {
            'conscious_output': 'signal', # The "Chosen" Reality
            'prediction_error': 'signal', # Surprise (Dopamine)
            'monitor_view': 'image'       # Visualizer
        }
        
        # The "11ms" Time Loop
        # At 60fps, 11ms is roughly 1 frame, but let's give it depth.
        self.delay_len = 10
        self.memory = deque(maxlen=self.delay_len)
        for _ in range(self.delay_len): self.memory.append(0.0)
        
        # Internal State
        self.current_state = 0.0
        self.prediction = 0.0
        self.display_buffer = deque(maxlen=100)
        self.display = np.zeros((150, 300, 3), dtype=np.uint8)

    def step(self):
        # 1. READ THE WORLD (Gamma)
        sensory = self.get_blended_input('sensory_input', 'mean')
        
        # 2. READ THE WILL (Alpha)
        # 0.0 = Passenger, 1.0 = Driver
        will = self.get_blended_input('intent_bias', 'mean')
        
        if sensory is None: sensory = 0.0
        if will is None: will = 0.5 
        will = np.clip(will, 0.0, 1.0)
        
        # 3. THE TIME TRAVEL (Reading the Past)
        # We look at what we were doing 11ms ago to predict now.
        past_self = self.memory[0] 
        
        # Simple Linear Extrapolation (The "Expectation")
        # In a real brain, this is the whole cortex's job.
        # Here, we assume "I will continue to be what I was."
        self.prediction = past_self
        
        # 4. THE CHOICE (Re-Entry)
        # We blend the Raw Input with the Self-Prediction.
        # Output = (1 - Will) * World + (Will) * Self
        
        # If Will is high, we SUPPRESS the sensory noise.
        # We force the output to match our prediction.
        
        # But we also must ADAPT. We can't ignore the world forever.
        # So we add the sensory signal, but dampened by our Will.
        
        self.current_state = (sensory * (1.0 - will)) + (self.prediction * will)
        
        # Update Memory (The Loop)
        self.memory.append(self.current_state)
        
        # 5. ERROR MONITORING
        # How wrong were we? This is the "Surprise" signal.
        error = abs(sensory - self.prediction)
        
        # 6. VISUALIZATION
        self.display_buffer.append((sensory, self.current_state, will))
        self._draw_monitor()
        
        # 7. OUTPUTS
        self.set_output('conscious_output', self.current_state)
        self.set_output('prediction_error', error)
        self.set_output('monitor_view', self.display)

    def _draw_monitor(self):
        self.display.fill(20)
        h, w = 150, 300
        mid = h // 2
        
        if len(self.display_buffer) < 2: return
        
        # Draw traces
        pts_sensory = []
        pts_output = []
        
        for i, (s, o, w_val) in enumerate(self.display_buffer):
            x = int(i * (w / 100))
            
            # Sensory = Red (Chaos)
            y_s = int(mid - s * 40)
            pts_sensory.append((x, y_s))
            
            # Output = Green (Order)
            y_o = int(mid - o * 40)
            pts_output.append((x, y_o))
            
        # Draw Lines
        for i in range(1, len(pts_sensory)):
            # Draw Sensory (Red)
            cv2.line(self.display, pts_sensory[i-1], pts_sensory[i], (50, 50, 200), 1)
            
            # Draw Output (Green) - Thickness based on Will
            will_strength = self.display_buffer[i][2]
            thick = 1 + int(will_strength * 2)
            cv2.line(self.display, pts_output[i-1], pts_output[i], (50, 255, 100), thick)
            
        # Stats
        curr_will = self.display_buffer[-1][2]
        cv2.putText(self.display, f"WILL: {curr_will:.2f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if curr_will > 0.8:
            status = "Stubborn"
            col = (0, 255, 255)
        elif curr_will < 0.2:
            status = "Passive"
            col = (100, 100, 100)
        else:
            status = "Adaptive"
            col = (0, 255, 0)
            
        cv2.putText(self.display, status, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    def get_output(self, name):
        if name == 'monitor_view': return self.display
        return getattr(self, '_outs', {}).get(name)
        
    def set_output(self, name, val):
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val