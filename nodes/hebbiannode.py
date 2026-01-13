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

class HebbianMemoryNode(BaseNode):
    """
    The Cortex (Long-Term Memory)
    -----------------------------
    Implements Hebbian Learning: 'Integration over Time'.
    - Inputs: A noisy/shifting stream.
    - Process: Exponential Moving Average (EMA).
    - Result: Extracts the 'Platonic Ideal' (Stable Structure) from the noise.
    
    If you feed it the Scrambled output, it will try to find the 
    average pixel values, effectively 'learning' the background.
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(255, 0, 128) # Brain Pink
    
    def __init__(self):
        super().__init__()
        self._output_values = {}
        self.node_title = "Hebbian Memory"
        
        self.inputs = {
            'sensory_input': 'image',
            'plasticity': 'signal' # Learning Rate (0.0 = Frozen, 1.0 = Amnesia)
        }
        
        self.outputs = {
            'memory_trace': 'image', # What the brain 'remembers'
            'novelty_map': 'image'   # Difference between Memory and Reality
        }
        
        self.memory = None # Latent State
        self.res = 64
        
    def step(self):
        # 1. Get Inputs
        sensory = self.get_input('sensory_input')
        rate = self.get_input('plasticity')
        
        if rate is None: rate = 0.05 # Default slow learning
        
        if sensory is None: return
        
        # Resize/Normalize
        if sensory.shape[:2] != (self.res, self.res):
            sensory = cv2.resize(sensory, (self.res, self.res))
            
        if sensory.ndim == 3: sensory = np.mean(sensory, axis=2)
        
        if sensory.max() > 1.05: sensory = sensory / 255.0
        
        # 2. Initialize Memory (First birth)
        if self.memory is None:
            self.memory = np.zeros_like(sensory)
            
        # 3. Hebbian Update (EMA)
        # Memory_new = Memory_old * (1-rate) + Input * rate
        self.memory = (self.memory * (1.0 - rate)) + (sensory * rate)
        
        # 4. Novelty Detection (Dopamine Signal)
        # Novelty = | Reality - Memory |
        novelty = np.abs(sensory - self.memory)
        
        # 5. Output
        # Memory Trace (The Learned Structure)
        mem_viz = (np.clip(self.memory, 0, 1) * 255).astype(np.uint8)
        mem_viz = cv2.cvtColor(mem_viz, cv2.COLOR_GRAY2RGB)
        self.set_output('memory_trace', mem_viz)
        
        # Novelty Map (What is surprising?)
        nov_viz = (np.clip(novelty * 5.0, 0, 1) * 255).astype(np.uint8)
        nov_viz = cv2.applyColorMap(nov_viz, cv2.COLORMAP_MAGMA)
        self.set_output('novelty_map', nov_viz)

    # Boilerplate
    def get_input(self, n): 
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(n)
        return self.input_data.get(n, [None])[0]
    def set_output(self, n, v): self._output_values[n] = v
    def get_output(self, n): return self._output_values.get(n)