"""
Adaptive Cortical Stack - "Matter Fits Computation"
===================================================
A 3-layer bidirectional network where the "Physics" (Connectivity) 
evolves based on the signal flow.

MECHANISM:
1. Signal flows UP (Sensory -> Deep) and DOWN (Deep -> Sensory).
2. "Riverbed Plasticity": If signal flows through a connection, 
   that connection gets wider (Lower Energy).
3. The system literally "carves" the geometry to fit the input.

This turns the graph into a physical memory of the signal.
"""

import numpy as np
import cv2
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class CorticalStack2Node(BaseNode):
    NODE_CATEGORY = "Experimental"
    NODE_TITLE = "Adaptive Cortical Stack"
    NODE_COLOR = QtGui.QColor(220, 180, 40) # Gold for "Precious Structure"
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'sensory_in': 'image',   # Bottom-up input
            'plasticity': 'signal'   # How soft is the matter? (0=Stone, 1=Clay)
        }
        self.outputs = {
            'layer_1_view': 'image', # Sensory
            'layer_2_view': 'image', # Processing
            'layer_3_view': 'image', # Deep State
            'structure_map': 'image' # Visualizes the "Carved Paths"
        }
        
        # Stack Dimensions
        self.W, self.H = 64, 64 # Keep small for physics speed
        self.n_layers = 3
        
        # STATE: The "Water" (Activity)
        # 3 Layers of 64x64 grids
        self.activity = np.zeros((self.n_layers, self.H, self.W), dtype=np.float32)
        
        # MATTER: The "Riverbeds" (Conductivity Horizontal)
        # We store horizontal conductivity. 1.0 = Open, 0.1 = Blocked
        # Initialized to "Rough Stone" (low conductivity)
        self.conductivity = np.ones((self.n_layers, self.H, self.W), dtype=np.float32) * 0.1
        
        # Vertical Connectivity (Fixed for now, just simple flow)
        self.feedforward = 0.2
        self.feedback = 0.1

    def step(self):
        # 1. INPUT (Inject Water)
        inp = self.get_blended_input('sensory_in', 'mean')
        rate = self.get_blended_input('plasticity', 'mean')
        if rate is None: rate = 0.05
        
        if inp is not None:
            # Resize and normalize
            if inp.ndim > 2: inp = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
            inp = cv2.resize(inp.astype(np.float32), (self.W, self.H))
            if inp.max() > 1.0: inp /= 255.0
            
            # Inject into Layer 0 (Sensory)
            # We mix it with existing activity (Momentum)
            self.activity[0] = self.activity[0] * 0.8 + inp * 0.2

        # 2. SIGNAL FLOW (The Computation)
        new_activity = self.activity.copy()
        
        for l in range(self.n_layers):
            # A. Horizontal Diffusion (Flow within layer)
            # This uses the Variable Conductivity (The "Matter")
            # Flow = Conductivity * (Neighbor - Self)
            curr = self.activity[l]
            cond = self.conductivity[l]
            
            # Simple 4-neighbor Laplacian with variable weights
            # (Using OpenCV blur as a cheap approximation of diffusion on the carved manifold)
            # We modulate the blur amount by the conductivity map
            
            # Fast approximation: Blur the activity, then mix based on conductivity
            diffused = cv2.GaussianBlur(curr, (3,3), 0)
            
            # High conductivity = Takes neighbor values (Smooth)
            # Low conductivity = Keeps own value (Rough)
            # This simulates the geometry constraining the flow
            flow = (diffused - curr) * cond
            new_activity[l] += flow * 0.5
            
            # B. Vertical Flow (Between Layers)
            if l < self.n_layers - 1: # Feedforward (Up)
                new_activity[l+1] += (self.activity[l] - self.activity[l+1]) * self.feedforward
            
            if l > 0: # Feedback (Down)
                new_activity[l-1] += (self.activity[l] - self.activity[l-1]) * self.feedback

        # 3. PLASTICITY (Changing the Matter)
        # "Riverbed Rule": Where flux is high, erode the rock (increase conductivity).
        # Flux ~ Magnitude of activity (simplified)
        
        for l in range(self.n_layers):
            # Erosion: Activity erodes resistance
            # If a pixel is active, it wants to connect to neighbors
            erosion = self.activity[l] * rate * 0.1
            
            # Recovery: Matter slowly hardens back to stone if unused
            hardening = 0.005
            
            self.conductivity[l] = np.clip(
                self.conductivity[l] + erosion - hardening, 
                0.0, 1.0
            )

        # Update State
        self.activity = np.clip(new_activity, 0, 1)

        # 4. OUTPUTS
        self.outputs['layer_1_view'] = self.activity[0]
        self.outputs['layer_2_view'] = self.activity[1]
        self.outputs['layer_3_view'] = self.activity[2]
        self.outputs['structure_map'] = self.conductivity[0] # Visualize the physical change

    def get_output(self, name):
        val = self.outputs.get(name)
        if val is None: return None
        # Convert to BGR for display pipeline
        return cv2.cvtColor((val * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)