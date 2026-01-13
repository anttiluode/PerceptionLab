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

class ScramblerNode(BaseNode):
    """
    The Reality Breaker
    -------------------
    Deterministically scrambles image pixels.
    This creates 'High Surface Tension' (Artificially High Entropy).
    """
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(150, 0, 0) # Villain Red
    
    def __init__(self):
        super().__init__()
        self._output_values = {}
        self.node_title = "Reality Scrambler"
        self.inputs = {'image_in': 'image'}
        self.outputs = {'scrambled_out': 'image'}
        
        self.res = 64 
        self.perm = np.random.permutation(self.res * self.res)

    def step(self):
        # 1. Get Input (Handle missing)
        img = self.get_input('image_in')
        if img is None: 
            # If no input, output meaningful static instead of black
            img = np.random.rand(self.res, self.res) 

        # 2. Resize
        small = cv2.resize(img, (self.res, self.res))
        
        # 3. Auto-Normalize (The Fix for the "Black Screen")
        # If max is 1.0 (Float), scale to 255. If 255 (Int), leave it.
        if small.max() <= 1.05:
            small = small * 255.0
            
        # Ensure Grayscale & Integer
        if small.ndim == 3: small = np.mean(small, axis=2)
        small = small.astype(np.uint8)

        # 4. Scramble
        flat = small.flatten()
        scrambled = flat[self.perm]
        
        # 5. Output
        out_img = scrambled.reshape((self.res, self.res))
        
        # Convert to RGB so Display accepts it
        disp = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)
        
        self.set_output('scrambled_out', disp)

    # Boilerplate
    def get_input(self, n): 
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(n)
        return self.input_data.get(n, [None])[0]
    def set_output(self, n, v): self._output_values[n] = v
    def get_output(self, n): return self._output_values.get(n)