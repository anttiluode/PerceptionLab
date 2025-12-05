"""
Bio-Tools: Utilities for the Artificial Life Ecosystem
------------------------------------------------------
1. Genomic Noise: Generates organic 1/f noise vectors for DNA seeding.
2. Vector Blender: Manually mix two DNA strands.
"""

import numpy as np

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None

class GenomicNoiseNode(BaseNode):
    """
    Generates 'Pink Noise' (1/f) vectors.
    Biological systems are rarely random; they are correlated.
    This creates DNA that looks more 'organic' and less like static.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(100, 150, 100) # Sage Green

    def __init__(self):
        super().__init__()
        self.node_title = "Genomic Noise"
        
        self.inputs = {
            'volatility': 'signal', # How fast the noise changes
            'roughness': 'signal'   # High frequency content
        }
        
        self.outputs = {
            'dna_spectrum': 'spectrum', # Vector output
            'value': 'signal'           # Single value
        }
        
        self.length = 128
        self.state = np.zeros(self.length)
        self.smooth_state = np.zeros(self.length)
        
        # Buffers
        self.out_spectrum = np.zeros(self.length)
        self.out_value = 0.0

    def step(self):
        vol = self.get_blended_input('volatility', 'mean')
        rough = self.get_blended_input('roughness', 'mean')
        
        if vol is None: vol = 0.1
        if rough is None: rough = 0.5
        
        # Generate new target noise
        target = np.random.randn(self.length) * rough
        
        # Smoothly interpolate (Brownian motion-ish)
        self.state = self.state * (1.0 - vol) + target * vol
        
        # Apply smoothing for "structure"
        # Simple moving average to simulate correlations
        kernel_size = 3
        self.smooth_state = np.convolve(self.state, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Normalize to 0..1 range typically expected by DNA, 
        # but centered around 0 is also fine for phases.
        # Let's keep it raw but bounded slightly
        self.out_spectrum = np.clip(self.smooth_state, -2.0, 2.0)
        self.out_value = float(np.mean(np.abs(self.out_spectrum)))

    def get_output(self, name):
        if name == 'dna_spectrum': return self.out_spectrum
        if name == 'value': return self.out_value
        return None


class VectorMathNode(BaseNode):
    """
    Simple math for DNA vectors.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(100, 100, 150)

    def __init__(self):
        super().__init__()
        self.node_title = "Vector Math"
        
        self.inputs = {
            'vec_a': 'spectrum',
            'vec_b': 'spectrum',
            'op_mode': 'signal' # 0=Add, 1=Sub, 2=Mult
        }
        self.outputs = {
            'result': 'spectrum'
        }
        self.out_result = np.zeros(128)

    def step(self):
        a = self.get_blended_input('vec_a', 'mean')
        b = self.get_blended_input('vec_b', 'mean')
        mode = self.get_blended_input('op_mode', 'mean')
        
        if a is None: a = np.zeros(128)
        if b is None: b = np.zeros(128)
        
        # Resize to match
        target_len = max(len(a), len(b))
        if len(a) < target_len: a = np.resize(a, target_len)
        if len(b) < target_len: b = np.resize(b, target_len)
        
        if mode is None: mode = 0
        
        if mode < 0.5: # ADD
            res = a + b
        elif mode < 1.5: # SUB
            res = a - b
        else: # MULT
            res = a * b
            
        self.out_result = res

    def get_output(self, name):
        if name == 'result': return self.out_result
        return None