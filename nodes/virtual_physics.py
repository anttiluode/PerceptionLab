"""
Virtual Physics Node - The "Software Webcam"
============================================
Simulates the physics of the real world (Diffusion/Blur and Noise) 
to create a closed-loop solver without a camera.

USAGE:
Connect [Analog Solver] -> [Virtual Physics] -> [Analog Solver]
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

class VirtualPhysicsNode(BaseNode):
    NODE_CATEGORY = "Experimental"
    NODE_TITLE = "Virtual Physics (Diffusion)"
    NODE_COLOR = QtGui.QColor(100, 100, 100)
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'image_in': 'image',
            'diffusion': 'signal',  # How fast heat flows (Blur amount)
            'entropy': 'signal'     # Simulates real-world noise
        }
        self.outputs = {
            'image_out': 'image'    # Fed back to the Solver
        }
        
    def step(self):
        # 1. Get the "Problem State" from the Solver
        img = self.get_blended_input('image_in', 'mean')
        if img is None: return

        # Handle formatting (ensure float 0-1)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # 2. SIMULATE PHYSICS (The "Trick")
        
        # A. Diffusion (The Solver Mechanism)
        # This replaces the optical blur of the webcam.
        # It spreads values to neighbors, which solves the Laplace equation.
        diff_amount = self.get_blended_input('diffusion', 'mean')
        if diff_amount is None: diff_amount = 1.5 # Default "Physics" constant
        
        # We use a Gaussian Blur to simulate heat dissipation
        k_size = int(diff_amount * 2) * 2 + 1 # Force odd number
        blurred = cv2.GaussianBlur(img, (k_size, k_size), 0)
        
        # B. Entropy (Optional Realism)
        # Adds noise to test if the solver is robust (like a dirty lens)
        noise_level = self.get_blended_input('entropy', 'mean')
        if noise_level and noise_level > 0.01:
            noise = np.random.normal(0, noise_level * 0.1, img.shape).astype(np.float32)
            blurred = blurred + noise
            blurred = np.clip(blurred, 0, 1)

        # 3. OUTPUT
        self.outputs['image_out'] = blurred

    def get_output(self, name):
        if name == 'image_out':
            return self.outputs.get('image_out')
        return None

    def get_config_options(self):
        # Allow manual tuning of the physics
        return [
            ("Diffusion Rate", "diffusion_rate", 1.5, "float"),
            ("Noise Level", "noise_level", 0.0, "float")
        ]