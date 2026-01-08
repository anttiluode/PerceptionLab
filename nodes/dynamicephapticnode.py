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

class DynamicEphapticNode(BaseNode):
    """
    Dynamic Ephaptic Node (The Lens)
    --------------------------------
    Converts Image -> Vector, but allows real-time control over the
    'Bandwidth' of reality.
    
    Input:
        - image_in: The Reality
        - focus: 0.0 (Blurry/Coarse) to 1.0 (Sharp/Fine)
    Output:
        - field_vector: The compressed summary (ephaptic field)
    """
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(100, 255, 200) # Field Green
    
    def __init__(self):
        super().__init__()
        self.node_title = "Ephaptic Lens"
        
        self.inputs = {
            'image_in': 'image',
            'focus': 'signal' # The Control Knob
        }
        
        self.outputs = {
            'field_vector': 'spectrum'
        }
        
        # Max resolution (The "Ultraviolet Cutoff")
        self.max_dim = 256
        self.vector = np.zeros(self.max_dim, dtype=np.float32)
        
        # Buffer for output
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
        img = self.get_input('image_in')
        focus = self.get_input('focus')
        
        if img is None: return
        if focus is None: focus = 1.0 # Default to full res
        
        # Clamp focus 0..1
        focus = max(0.01, min(1.0, float(focus)))
        
        # 1. Downsample Image to Vector
        # We crush it to 16x16 (256) max
        if img.ndim == 3: img = np.mean(img, axis=2)
        
        target_res = 16
        small = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_AREA)
        vec_full = small.flatten()
        
        # 2. Apply The Ephaptic Cutoff (The "Lens")
        # We only keep the first N components, zero the rest.
        # This simulates a "Low Pass Filter" on reality.
        
        effective_dim = int(self.max_dim * focus)
        
        # We don't just crop; we mask.
        # This keeps the vector size constant (for stable wiring) 
        # but kills the high-frequency information.
        
        self.vector[:] = 0
        if effective_dim > 0:
            self.vector[:effective_dim] = vec_full[:effective_dim]
            
        # Optional: Normalize energy so getting focused doesn't just make it "louder"
        # but actually "sharper"
        # (Preserve total energy of the active part)
        current_energy = np.sum(np.abs(self.vector)) + 1e-6
        full_energy = np.sum(np.abs(vec_full)) + 1e-6
        
        # Gain compensation (optional, keep it raw for now to see the drop)
        
        self.set_output('field_vector', self.vector)