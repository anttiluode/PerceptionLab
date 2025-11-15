"""
Instanton Reservoir Node (The "Bucket" System)
-----------------------------------------------
This is the "Slow Layer" (Cortex) from your theory.

It takes in the "Fast Layer" (LatentEncoder) signal and accumulates
it in a grid of "buckets" (instantons).

The buckets "leak" into each other (ephaptic coupling/diffusion)
and "evaporate" (strategic forgetting).

The output is the "living memory" of the system.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class InstantonReservoirNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(50, 100, 180)  # Deep Blue
    
    def __init__(self, diffusion_factor=0.1, decay_factor=0.01, accumulation=0.05):
        super().__init__()
        self.node_title = "Instanton Reservoir (Memory)"
        
        self.inputs = {
            'latents_in': 'image', # From LatentEncoder (Fast Layer)
        }
        self.outputs = {
            'image_out': 'image',  # The "Slow Layer" memory state
        }
        
        # Configurable physics
        self.diffusion_factor = float(diffusion_factor) # How much buckets leak (ephaptic)
        self.decay_factor = float(decay_factor)         # How fast memory fades (forgetting)
        self.accumulation = float(accumulation)       # How fast buckets fill (learning)
        
        self.reservoir_state = None
        
        # Kernel for diffusion (the "global wave")
        self.diffusion_kernel = np.array([
            [0.5, 1.0, 0.5],
            [1.0, -6.0, 1.0],
            [0.5, 1.0, 0.5]
        ]) * self.diffusion_factor

    def step(self):
        latents_in = self.get_blended_input('latents_in', 'first')
        
        if latents_in is None:
            return
            
        if self.reservoir_state is None:
            # Initialize the bucket grid
            self.reservoir_state = np.zeros_like(latents_in, dtype=np.float32)

        # 1. Diffusion (Ephaptic Coupling / Global Wave)
        # The "leaking" between buckets
        diffusion = cv2.filter2D(self.reservoir_state, -1, self.diffusion_kernel)
        
        # 2. Decay (Strategic Forgetting / Evaporation)
        decay = self.reservoir_state * self.decay_factor
        
        # 3. Accumulation (Learning / "Rain")
        # Add the "fast" signal from the encoder
        accumulation = latents_in * self.accumulation
        
        # Update the state:
        self.reservoir_state += diffusion - decay + accumulation
        
        # Clamp values
        self.reservoir_state = np.clip(self.reservoir_state, -5.0, 5.0)

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.reservoir_state
        return None

    def get_display_image(self):
        if self.reservoir_state is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
            
        # Normalize for display
        img = self.reservoir_state
        norm_img = img - img.min()
        if norm_img.max() > 0:
            norm_img /= norm_img.max()
            
        img_u8 = (norm_img * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_OCEAN)
        
        display_size = 256
        img_resized = cv2.resize(img_color, (display_size, display_size), 
                                 interpolation=cv2.INTER_NEAREST)
                                 
        cv2.putText(img_resized, "SLOW LAYER (CORTEX)", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        img_resized = np.ascontiguousarray(img_resized)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Diffusion (Leak)", "diffusion_factor", self.diffusion_factor, None),
            ("Decay (Forget)", "decay_factor", self.decay_factor, None),
            ("Accumulation (Learn)", "accumulation", self.accumulation, None),
        ]