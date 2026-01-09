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

class MERARenormNode2(BaseNode):
    """
    MERA Renormalization Node
    -------------------------
    Implements Multi-scale Entanglement Renormalization Ansatz.
    
    Takes a signal and builds a coarse-graining pyramid where:
    - Each level removes short-range entanglement (detail)
    - Passes long-range structure upward (bulk)
    - The "deep tensor" at top encodes the IR (macroscopic) state
    
    This is the computational analog of RG flow in physics:
    UV (fine) -> IR (coarse), with entanglement structure preserved.
    
    Outputs show how complexity collapses at coarse scales.
    """
    NODE_CATEGORY = "Physics"
    NODE_COLOR = QtGui.QColor(100, 50, 200)  # Purple for RG flow
    
    def __init__(self):
        super().__init__()
        self.node_title = "MERA Renorm"
        
        self.inputs = {
            'signal_in': 'spectrum',
            'image_in': 'image',
        }
        
        self.outputs = {
            'bulk_view': 'image',      # Pyramid visualization
            'deep_tensor': 'spectrum',  # Top-level coarse state
            'entropy_flow': 'spectrum', # Entropy at each scale
            'scale_metric': 'signal',   # Complexity ratio UV/IR
        }
        
        self.config = {
            'levels': 5,           # Depth of coarse-graining
            'disentangle': 0.3,    # Strength of local decorrelation
        }
        
        self._output_values = {}
        self.pyramid = []
        self.entropies = []

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'):
            return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value):
        self._output_values[name] = value
    
    def get_output(self, name):
        return self._output_values.get(name, None)

    def _disentangle(self, layer):
        """
        Local disentangling unitary - removes short-range correlations.
        Simplified: decorrelate adjacent pairs.
        """
        d = self.config['disentangle']
        n = len(layer)
        result = layer.copy()
        
        for i in range(0, n - 1, 2):
            # Mix adjacent values to remove local correlation
            a, b = layer[i], layer[i + 1]
            # Rotation-like mixing
            result[i] = a * (1 - d) + b * d
            result[i + 1] = b * (1 - d) - a * d
        
        return result

    def _isometry(self, layer):
        """
        Isometry: coarse-grain by factor of 2.
        Maps 2 sites -> 1 site, preserving relevant info.
        """
        n = len(layer)
        if n < 2:
            return layer
        
        coarse = np.zeros(n // 2)
        for i in range(n // 2):
            # Isometry combines pairs (like block-spin RG)
            coarse[i] = np.sqrt(layer[2*i]**2 + layer[2*i + 1]**2)
        
        return coarse

    def _compute_entropy(self, layer):
        """Estimate entropy of layer (proxy: normalized variance)"""
        if len(layer) < 2:
            return 0.0
        probs = np.abs(layer) / (np.sum(np.abs(layer)) + 1e-10)
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log(probs + 1e-10))

    def step(self):
        # Get input - prefer spectrum, fall back to image
        signal = self.get_input('signal_in')
        image = self.get_input('image_in')
        
        if signal is not None:
            data = np.array(signal, dtype=np.float32).flatten()
        elif image is not None:
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            # Flatten image to 1D for MERA
            data = image.flatten().astype(np.float32)
        else:
            return
        
        # Normalize
        data = data / (np.max(np.abs(data)) + 1e-10)
        
        # Pad to power of 2
        n = len(data)
        target = 2 ** int(np.ceil(np.log2(max(n, 4))))
        if n < target:
            data = np.pad(data, (0, target - n), mode='constant')
        
        # === BUILD MERA PYRAMID ===
        self.pyramid = [data.copy()]
        self.entropies = [self._compute_entropy(data)]
        
        current = data
        for level in range(self.config['levels']):
            if len(current) < 4:
                break
            
            # 1. Disentangle (remove short-range)
            disentangled = self._disentangle(current)
            
            # 2. Isometry (coarse-grain)
            coarse = self._isometry(disentangled)
            
            self.pyramid.append(coarse.copy())
            self.entropies.append(self._compute_entropy(coarse))
            current = coarse
        
        # === OUTPUTS ===
        
        # Deep tensor (top of pyramid)
        self.set_output('deep_tensor', self.pyramid[-1])
        
        # Entropy flow
        self.set_output('entropy_flow', np.array(self.entropies))
        
        # Scale metric (complexity ratio)
        if len(self.entropies) >= 2 and self.entropies[-1] > 0:
            ratio = self.entropies[0] / (self.entropies[-1] + 1e-10)
        else:
            ratio = 1.0
        self.set_output('scale_metric', float(np.clip(ratio / 10.0, 0, 1)))
        
        # Render pyramid view
        self._render_pyramid()

    def _render_pyramid(self):
        """Visualize the MERA pyramid"""
        if not self.pyramid:
            return
        
        # Create canvas
        max_width = len(self.pyramid[0])
        height = len(self.pyramid) * 40
        canvas = np.zeros((height, max_width, 3), dtype=np.uint8)
        
        for level, layer in enumerate(self.pyramid):
            y = level * 40 + 5
            n = len(layer)
            
            # Normalize layer for display
            layer_norm = np.abs(layer) / (np.max(np.abs(layer)) + 1e-10)
            
            # Center the layer
            offset = (max_width - n) // 2
            
            for i, val in enumerate(layer_norm):
                x = offset + i
                if 0 <= x < max_width:
                    # Color by value (blue->yellow)
                    intensity = int(val * 255)
                    canvas[y:y+30, x:x+1] = [intensity, intensity, 50 + intensity // 2]
        
        # Add entropy annotations
        for level, entropy in enumerate(self.entropies):
            y = level * 40 + 20
            cv2.putText(canvas, f"S={entropy:.2f}", (5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Resize for visibility
        canvas = cv2.resize(canvas, (512, 256), interpolation=cv2.INTER_NEAREST)
        
        self.set_output('bulk_view', canvas)
