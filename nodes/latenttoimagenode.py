"""
Latent to Image Node (Holographic Decoder)
==========================================
Projects low-dimensional latent variables into high-dimensional image space.
This is the "Generator" of the Perception Lab.

COLOR: Orange (255, 140, 0)
"""

import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class LatentToImageNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Latent > Image"
    NODE_COLOR = QtGui.QColor(255, 140, 0)  # Bright Orange
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'latent_x': 'signal',       # Controls geometry/frequency
            'latent_y': 'signal',       # Controls phase/rotation
            'latent_z': 'signal',       # Controls complexity/detail
            'seed_vector': 'signal'     # Random seed for the "Style"
        }
        
        self.outputs = {
            'generated_image': 'image',        # The spatial result
            'complex_field': 'complex_spectrum' # The quantum state
        }
        
        self.size = 128
        self.latent_dim = 3
        
        # Internal "Style" Matrix (The Generator's Memory)
        # Randomly initialized, acts as the basis functions
        np.random.seed(42)
        self.basis_functions = np.random.randn(self.size, self.size, 3).astype(np.float32)
        
        self.current_image = np.zeros((self.size, self.size), dtype=np.float32)

    def step(self):
        # 1. Get Latent Variables
        x = self.get_blended_input('latent_x', 'sum')
        y = self.get_blended_input('latent_y', 'sum')
        z = self.get_blended_input('latent_z', 'sum')
        seed = self.get_blended_input('seed_vector', 'sum')
        
        # Defaults if disconnected
        if x is None: x = 0.5
        if y is None: y = 0.5
        if z is None: z = 0.5
        
        # 2. Reseed "Style" if seed changes significantly
        if seed is not None and abs(seed - 0.0) > 0.01:
             np.random.seed(int(seed * 100))
             self.basis_functions = np.random.randn(self.size, self.size, 3).astype(np.float32)

        # 3. Holographic Projection Logic
        # We treat the latent variables as weights for the basis functions
        # But we do it in Frequency Space (K-Space) for "Holographic" feel
        
        # Create coordinate grid
        Y, X = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        R = np.sqrt((X-center)**2 + (Y-center)**2) / center
        Angle = np.arctan2(Y-center, X-center)
        
        # Latent X: Controls Frequency / Scale
        freq = 5.0 + x * 20.0
        
        # Latent Y: Controls Phase / Rotation
        phase = y * np.pi * 2.0
        
        # Latent Z: Controls Complexity (Harmonics)
        harmonics = 1.0 + z * 5.0
        
        # 4. Generate Field (The "Thought")
        # Base wave
        field = np.sin(R * freq + phase + self.basis_functions[:,:,0])
        
        # Add Harmonics (Complexity)
        field += 0.5 * np.sin(Angle * harmonics + self.basis_functions[:,:,1])
        
        # Add "Style" interference
        field *= np.cos(self.basis_functions[:,:,2] * z)
        
        # 5. Output
        self.current_image = np.clip((field + 1.0) * 0.5, 0, 1) # Normalize 0-1
        
        # Generate complex spectrum for other nodes
        spectrum = np.fft.fftshift(np.fft.fft2(self.current_image))
        
        self.set_output('generated_image', self.current_image)
        self.set_output('complex_field', spectrum)

    def get_output(self, name):
        if name == 'generated_image':
            return (self.current_image * 255).astype(np.uint8)
        if name == 'complex_field':
            # Recompute spectrum if needed, or cache it
            return np.fft.fftshift(np.fft.fft2(self.current_image))
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # Display the Generated Latent Image
        img_u8 = (self.current_image * 255).astype(np.uint8)
        
        # Apply a "Latent" colormap (Plasma is good for energy/latent)
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_PLASMA)
        
        # Overlay Latent Values
        cv2.putText(colored, "LATENT SPACE", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(colored.data, w, h, w*3, QtGui.QImage.Format.Format_BGR888)