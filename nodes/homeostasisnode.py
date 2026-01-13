import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name): return None

class HomeostasisNode(BaseNode):
    """
    The Regulator (Adaptive Scheduler)
    ----------------------------------
    Implements Biological Homeostasis.
    - Monitors 'Stress' (Error/Entropy).
    - If Stress is HIGH -> Lowers Control (Panic Mode / Neuroplasticity).
    - If Stress is LOW  -> Raises Control (Refinement / Crystallization).
    
    Connect 'Stress Map' from Surface Tension -> into 'visual_stress'.
    Connect 'control_signal' -> into 'Tension' on Surface Tension node.
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(0, 255, 128) # Life Green
    
    def __init__(self):
        super().__init__()
        self._output_values = {}
        self.node_title = "Homeostasis (Regulator)"
        
        self.inputs = {
            'visual_stress': 'image', # Input from Stress Map
        }
        
        self.outputs = {
            'control_signal': 'signal', # Adaptive Lambda
            'state_viz': 'image'        # Visualizing the "Mood"
        }
        
        self.history = []
        self.current_state = 0.5
        
    def step(self):
        # 1. Measure the Chaos
        stress_img = self.get_input('visual_stress')
        
        current_stress = 0.0
        if stress_img is not None:
            # Calculate mean energy of the stress map
            current_stress = np.mean(stress_img) / 255.0
        
        # 2. The Homeostatic Loop (PID-like Controller)
        target_stress = 0.1  # The system "wants" a little bit of novelty
        
        # Error = What we have vs What we want
        error = target_stress - current_stress
        
        # Adjust internal state (The "Hormone Level")
        # If stress is too high (error negative), drop the state (relax).
        # If stress is too low (error positive), raise the state (focus).
        self.current_state += error * 0.05 
        
        # Clamp to 0-1
        self.current_state = max(0.01, min(0.99, self.current_state))
        
        # 3. Output Control Signal
        # High State = High Tension (Crystallization)
        self.set_output('control_signal', self.current_state)
        
        # 4. Visualization (The "Mood Ring")
        # Green = Balanced, Red = Panic, Blue = Bored
        viz = np.zeros((64, 64, 3), dtype=np.uint8)
        
        if current_stress > target_stress * 1.5:
            viz[:, :, 2] = 255 # Red (Panic)
        elif current_stress < target_stress * 0.5:
            viz[:, :, 0] = 255 # Blue (Bored)
        else:
            viz[:, :, 1] = 255 # Green (Flow State)
            
        # Draw a bar showing the current control level
        height = int(self.current_state * 64)
        cv2.rectangle(viz, (0, 64-height), (20, 64), (255, 255, 255), -1)
        
        self.set_output('state_viz', viz)

    # Boilerplate
    def get_input(self, n): 
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(n)
        return self.input_data.get(n, [None])[0]
    def set_output(self, n, v): self._output_values[n] = v
    def get_output(self, n): return self._output_values.get(n)