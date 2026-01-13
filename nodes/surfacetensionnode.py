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

class SurfaceTensionNode(BaseNode):
    """
    Surface Optimization Engine (The Healer)
    ----------------------------------------
    Implements the Barabási 'Minimal Surface' logic.
    - Input: Noisy/Scrambled Image.
    - Dynamics: A 'Membrane' that tries to match input BUT resists bending.
    - Output: The 'Relaxed' (Denoised) state.
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(0, 200, 255) # Water Blue
    
    def __init__(self):
        super().__init__()
        self._output_values = {}   # <--- CRITICAL FIX 1: Lifecycle State
        self.node_title = "Surface Tension Crystal"
        
        self.inputs = {
            'noisy_input': 'image',
            'tension': 'signal' # Strength of the soap bubble effect
        }
        
        self.outputs = {
            'relaxed_view': 'image', # The Denoised Result
            'stress_map': 'image'    # Visualizing the Pain
        }
        
        self.res = 64
        # The "Internal State" (The Membrane)
        self.membrane = np.zeros((self.res, self.res), dtype=np.float32)
        
    def step(self):
        # 1. Get Input
        target = self.get_input('noisy_input')
        tension_strength = self.get_input('tension')
        
        # Default if unconnected
        if tension_strength is None: tension_strength = 0.5 
        
        # <--- CRITICAL FIX 2: Handle Empty Input (Keep Pipeline Alive)
        if target is None: 
            blank = np.zeros((256, 256, 3), dtype=np.uint8)
            self.set_output('relaxed_view', blank)
            self.set_output('stress_map', blank)
            return

        # Ensure target matches resolution
        if target.shape[:2] != (self.res, self.res):
            target = cv2.resize(target, (self.res, self.res))
        
        # Ensure Grayscale for Physics
        if target.ndim == 3: 
            target = np.mean(target, axis=2) 
        
        # <--- CRITICAL FIX 3: Robust Normalization (Float vs Int)
        # Handle both 0-255 (Integer) and 0.0-1.0 (Float) inputs
        if target.max() > 1.05: 
            target = target / 255.0
        # (If it's already 0-1, we leave it alone)
        
        # --- THE PHYSICS ENGINE ---
        
        # 2. Calculate Laplacian (Curvature/Stress)
        # "How much is this pixel different from its average neighbor?"
        up = np.roll(self.membrane, 1, axis=0)
        down = np.roll(self.membrane, -1, axis=0)
        left = np.roll(self.membrane, 1, axis=1)
        right = np.roll(self.membrane, -1, axis=1)
        
        average_neighbor = (up + down + left + right) * 0.25
        curvature = average_neighbor - self.membrane
        
        # 3. Update Membrane (The Optimization)
        # Update = (Pull towards Data) + (Pull towards Smoothness)
        
        alpha = 0.1 * (1.0 - tension_strength) # Data Term
        beta = 0.5 * tension_strength          # Tension Term
        
        # Euler Integration
        self.membrane += (target - self.membrane) * alpha + curvature * beta
        
        # 4. Visualization (Output as correct uint8 images)
        
        # View A: The Denoised World
        relaxed_img = (np.clip(self.membrane, 0, 1) * 255).astype(np.uint8)
        relaxed_img = cv2.applyColorMap(relaxed_img, cv2.COLORMAP_VIRIDIS)
        relaxed_img = cv2.resize(relaxed_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.set_output('relaxed_view', relaxed_img)
        
        # View B: The Stress Map (Highlights Edges & Noise)
        stress = np.abs(target - self.membrane)
        stress_img = (np.clip(stress * 5.0, 0, 1) * 255).astype(np.uint8) # Boost contrast
        stress_img = cv2.applyColorMap(stress_img, cv2.COLORMAP_INFERNO)
        stress_img = cv2.resize(stress_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.set_output('stress_map', stress_img)

    # Boilerplate
    def get_input(self, n): 
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(n)
        return self.input_data.get(n, [None])[0]
    def set_output(self, n, v): self._output_values[n] = v
    def get_output(self, n): return self._output_values.get(n)