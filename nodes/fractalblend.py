"""
FractalBlendNode

Uses a Julia set calculation as a dynamic mask
to blend between two input images.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class FractalBlendNode(BaseNode):
    """
    Blends two images using a fractal (Julia set) mask.
    """
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(100, 220, 180) # Teal

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Fractal Blender"
        
        self.inputs = {
            'image_in1': 'image', # Background image
            'image_in2': 'image', # Foreground image
            'c_real': 'signal',   # Julia set 'c' real part
            'c_imag': 'signal',   # Julia set 'c' imaginary part
            'max_iter': 'signal'  # Fractal detail (0-1)
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.blended_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Pre-calculate the 'Z' grid
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.z_real = (x / (self.size - 1) - 0.5) * 4.0
        self.z_imag = (y / (self.size - 1) - 0.5) * 4.0
        
    def _prepare_image(self, img):
        """Helper to resize and format an input image."""
        if img is None:
            return None
        
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        if img_resized.ndim == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
        
        if img_resized.max() > 1.0:
            img_resized = img_resized.astype(np.float32) / 255.0
            
        return np.clip(img_resized, 0, 1)

    def step(self):
        # --- 1. Get Control Signals ---
        c_real = self.get_blended_input('c_real', 'sum') or -0.7
        c_imag = self.get_blended_input('c_imag', 'sum') or 0.27
        
        # Max iterations: 10 to 80
        iter_in = self.get_blended_input('max_iter', 'sum') or 0.2
        max_iter = int(10 + iter_in * 70)
        
        # --- 2. Get and Prepare Input Images ---
        img1 = self._prepare_image(self.get_blended_input('image_in1', 'first'))
        img2 = self._prepare_image(self.get_blended_input('image_in2', 'first'))
        
        # Handle missing images
        if img1 is None and img2 is None:
            self.blended_image *= 0.9 # Fade to black
            return
        elif img1 is None:
            img1 = np.zeros((self.size, self.size, 3), dtype=np.float32)
        elif img2 is None:
            img2 = np.zeros((self.size, self.size, 3), dtype=np.float32)

        # --- 3. Perform Fractal Calculation (Julia Set) ---
        
        # Initialize Z and C grids
        Zr = self.z_real.copy()
        Zi = self.z_imag.copy()
        Cr = c_real
        Ci = c_imag
        
        # Output mask (stores escape time)
        fractal_mask = np.full(Zr.shape, max_iter, dtype=np.float32)
        
        # Create a boolean mask for pixels still iterating
        active = np.ones(Zr.shape, dtype=bool)

        for i in range(max_iter):
            if not active.any(): # Stop if all pixels escaped
                break
            
            # Check for escape
            mag_sq = Zr[active]**2 + Zi[active]**2
            escaped = mag_sq > 4.0
            
            # Store iteration count for newly escaped pixels
            fractal_mask[active][escaped] = i
            
            # Update active mask (remove escaped pixels)
            active[active] = ~escaped
            
            if not active.any():
                break

            # Z = Z^2 + C
            # Z.real = Z.real^2 - Z.imag^2 + C.real
            # Z.imag = 2 * Z.real * Z.imag + C.imag
            Zr_temp = Zr[active]**2 - Zi[active]**2 + Cr
            Zi[active] = 2 * Zr[active] * Zi[active] + Ci
            Zr[active] = Zr_temp

        # --- 4. Normalize mask and blend ---
        
        # Normalize the mask from 0 to 1
        mask_norm = (fractal_mask / (max_iter - 1.0))
        # Use sine for a smoother, pulsing blend
        mask_smooth = (np.sin(mask_norm * np.pi * 2.0 - np.pi/2.0) + 1.0) * 0.5
        
        # Expand mask to 3 channels (H, W, 1) for broadcasting
        mask_3d = mask_smooth[..., np.newaxis]
        
        # Blend: img1 is background, img2 is foreground
        self.blended_image = (img1 * (1.0 - mask_3d)) + (img2 * mask_3d)

    def get_output(self, port_name):
        if port_name == 'image':
            return self.blended_image
        return None