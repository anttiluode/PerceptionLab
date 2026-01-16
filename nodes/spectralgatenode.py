"""
Spectral Gate Node - "The Radio Tuner"
======================================
Filters the Holographic signal in the Frequency Domain.
Allows you to separate "The Ghost" (Low Freq) from "The Noise" (High Freq).

- Low Cut: Removes large structures (DC offset).
- High Cut: Removes fine details (Noise).
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SpectralGateNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Spectral Radio Tuner"
    NODE_COLOR = QtGui.QColor(100, 100, 255) # Phase Blue
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'image_in': 'image',
            'high_cut': 'signal', # Removes Noise (Blur)
            'low_cut': 'signal'   # Removes Context (Edge Detect)
        }
        self.outputs = {
            'filtered_img': 'image',
            'mask_view': 'image'  # Visualize the filter ring
        }
        
        self.last_img = None

    def step(self):
        img = self.get_blended_input('image_in', 'mean')
        if img is None: return
        
        # Ensure Grayscale for FFT
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img.astype(np.float32), (64, 64)) / 255.0
        
        # 1. TO FREQUENCY DOMAIN
        f = fft2(img)
        fshift = fftshift(f) # Center the frequencies
        
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        
        # 2. CREATE THE MASK (The Radio Tuner)
        high_cut = self.get_blended_input('high_cut', 'mean')
        if high_cut is None: high_cut = 1.0 # All Pass
        
        low_cut = self.get_blended_input('low_cut', 'mean')
        if low_cut is None: low_cut = 0.0 # All Pass
        
        # Map 0..1 inputs to pixel radii
        max_radius = min(crow, ccol)
        r_out = int(high_cut * max_radius)
        r_in = int(low_cut * max_radius)
        
        # Create Ring Mask
        mask = np.zeros((rows, cols), np.float32)
        
        # Draw White Circle (Outer Limit)
        cv2.circle(mask, (ccol, crow), r_out, 1.0, -1)
        
        # Draw Black Circle (Inner Limit)
        if r_in > 0:
            cv2.circle(mask, (ccol, crow), r_in, 0.0, -1)
            
        # 3. APPLY FILTER
        fshift_filtered = fshift * mask
        
        # 4. BACK TO SPACE DOMAIN (The Dream)
        f_ishift = ifftshift(fshift_filtered)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        self.last_img = np.clip(img_back, 0, 1)
        self.outputs['filtered_img'] = self.last_img
        self.outputs['mask_view'] = mask

    def get_output(self, name):
        val = self.outputs.get(name)
        if val is None: return None
        return cv2.cvtColor((val * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)