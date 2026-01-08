import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): self.inputs={}; self.outputs={}; self.input_data={}
        def get_blended_input(self, name): return None

class HomeostaticRegulatorNode(BaseNode):
    """
    Homeostatic Regulator (The Thalamus)
    ------------------------------------
    Manages the Energy Budget of the artificial brain.
    - High Focus (Gamma) drains energy.
    - Low Focus (Delta) recharges energy.
    - Forces 'Sleep' when depleted.
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(50, 50, 50) # Dark Sleep
    
    def __init__(self):
        super().__init__()
        self.node_title = "Homeostatic Regulator"
        
        self.inputs = {
            'novelty': 'signal',   # From Manifold (Do we WANT to stay awake?)
        }
        
        self.outputs = {
            'focus_command': 'signal', # To Ephaptic Lens (0.0=Delta, 1.0=Gamma)
            'energy_level': 'signal',  # Virtual Battery
            'state_view': 'image'      # Visual of the battery
        }
        
        self.config = {
            'max_energy': 100.0,
            'drain_rate': 0.5,     # Cost of thinking
            'recharge_rate': 0.8,  # Speed of sleep
            'wake_threshold': 0.6  # How surprising input must be to wake us up
        }
        
        self.energy = 80.0
        self.is_asleep = False
        self.current_focus = 1.0
        
        self._output_values = {}

    # --- Compatibility ---
    def get_input(self, name):
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value): self._output_values[name] = value
    def get_output(self, name): return self._output_values.get(name, None)
    # ---------------------

    def step(self):
        novelty = self.get_input('novelty')
        if novelty is None: novelty = 0.0
        novelty = float(novelty)
        
        # 1. Determine State (Sleep/Wake Logic)
        
        if self.is_asleep:
            # We are sleeping. Can we wake up?
            # Rule: Must have recovered enough energy AND see something surprising
            if self.energy > 90.0:
                self.is_asleep = False # Natural wake up
            elif self.energy > 20.0 and novelty > self.config['wake_threshold']:
                self.is_asleep = False # Startled awake
            else:
                self.is_asleep = True # Keep sleeping
        else:
            # We are awake. Do we crash?
            if self.energy <= 0.0:
                self.is_asleep = True # Forced sleep (Exhaustion)
            elif novelty < 0.1 and self.energy < 50.0:
                self.is_asleep = True # Bored nap
        
        # 2. Adjust Focus (The Output)
        target_focus = 0.1 if self.is_asleep else 1.0
        
        # Smooth transition (Drowsiness)
        self.current_focus = (self.current_focus * 0.9) + (target_focus * 0.1)
        
        # 3. Metabolism (The Battery)
        if self.current_focus > 0.5:
            # High Res = Drain
            drain = self.config['drain_rate'] * self.current_focus
            self.energy -= drain
        else:
            # Low Res = Recharge
            self.energy += self.config['recharge_rate']
            
        self.energy = max(0.0, min(self.config['max_energy'], self.energy))
        
        # 4. Outputs
        self.set_output('focus_command', [self.current_focus])
        self.set_output('energy_level', [self.energy])
        self._render_state()

    def _render_state(self):
        img = np.zeros((100, 200, 3), dtype=np.float32)
        
        # Draw Battery Bar
        energy_pct = self.energy / self.config['max_energy']
        
        # Color: Green (Full) -> Red (Empty)
        col = (0.0, energy_pct, 1.0 - energy_pct)
        if self.is_asleep: col = (0.8, 0.2, 0.8) # Purple for Sleep
        
        width = int(energy_pct * 190)
        cv2.rectangle(img, (5, 50), (5 + width, 90), col, -1)
        cv2.rectangle(img, (5, 50), (195, 90), (1,1,1), 1)
        
        # Text
        state_txt = "DELTA (SLEEP)" if self.is_asleep else "GAMMA (AWAKE)"
        cv2.putText(img, state_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1,1,1), 1)
        
        self.set_output('state_view', img)