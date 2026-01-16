"""
Spectral Amplifier Node - "The Dark Energy Detector"
====================================================
Splits the image into Low (Ghost) and High (Detail) frequencies.
Allows you to boost the invisible 'Black' signals into visibility.

- Low Freq Gain: Cranks up the 'Ghost' (The underlying concept).
- High Freq Gain: Cranks up the 'Texture' (The noise/edges).
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

class SpectralAmplifierNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Spectral Amplifier"
    NODE_COLOR = QtGui.QColor(160, 100, 200) # Violet
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'image_in': 'image',
            'low_gain': 'signal',   # Boost the Ghost
            'high_gain': 'signal',  # Boost the Noise
            'crossover': 'signal'   # Where to split (Frequency)
        }
        self.outputs = {
            'amplified_out': 'image',
            'ghost_view': 'image',  # Just the Low Freq
            'detail_view': 'image'  # Just the High Freq
        }
        
    def step(self):
        img = self.get_blended_input('image_in', 'mean')
        if img is None: return
        
        # 1. READ PARAMS
        l_gain = self.get_blended_input('low_gain', 'mean')
        if l_gain is None: l_gain = 5.0 # Default: Massive boost to see ghosts
        
        h_gain = self.get_blended_input('high_gain', 'mean')
        if h_gain is None: h_gain = 1.0
        
        crossover = self.get_blended_input('crossover', 'mean')
        if crossover is None: crossover = 0.1 # 10% of spectrum is "Low"
        
        # 2. PREPARE FFT
        # Handle Color or Gray
        is_color = (img.ndim == 3)
        if is_color:
            # For simplicity, process luminance, or split channels? 
            # Let's simple-convert to float and process channels together if possible,
            # but FFT needs 2D. Let's do Channel-wise for full holography.
            planes = cv2.split(img)
        else:
            planes = [img]
            
        out_planes = []
        ghost_planes = []
        detail_planes = []
        
        for p in planes:
            # Normalize
            p_float = p.astype(np.float32) / 255.0 if p.max() > 1.0 else p.astype(np.float32)
            p_float = cv2.resize(p_float, (64, 64))
            
            # FFT
            f = np.fft.fft2(p_float)
            fshift = np.fft.fftshift(f)
            
            rows, cols = p_float.shape
            crow, ccol = rows//2, cols//2
            
            # 3. CREATE MASKS
            # Distance from center
            y, x = np.ogrid[:rows, :cols]
            dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
            max_r = min(crow, ccol)
            
            # Soft Crossover (Sigmoid or Linear ramp)
            radius = crossover * max_r
            # Low Pass Mask (1 at center, 0 at edge)
            # Using simple boolean circle for now, or Gaussian
            # Gaussian is better for "Ghost" look (no ringing)
            sigma = radius if radius > 0.1 else 0.1
            low_mask = np.exp(-(dist**2)/(2*(sigma**2)))
            
            high_mask = 1.0 - low_mask
            
            # 4. AMPLIFY
            # Apply Gains
            f_low = fshift * low_mask * l_gain
            f_high = fshift * high_mask * h_gain
            
            f_recombined = f_low + f_high
            
            # 5. INVERSE FFT
            img_low = np.fft.ifft2(np.fft.ifftshift(f_low))
            img_high = np.fft.ifft2(np.fft.ifftshift(f_high))
            img_rec = np.fft.ifft2(np.fft.ifftshift(f_recombined))
            
            # Magnitude
            ghost_planes.append(np.abs(img_low))
            detail_planes.append(np.abs(img_high))
            out_planes.append(np.abs(img_rec))
            
        # 6. MERGE CHANNELS
        if is_color:
            final_ghost = cv2.merge([np.clip(p,0,1) for p in ghost_planes])
            final_detail = cv2.merge([np.clip(p,0,1) for p in detail_planes])
            final_out = cv2.merge([np.clip(p,0,1) for p in out_planes])
        else:
            final_ghost = np.clip(ghost_planes[0], 0, 1)
            final_detail = np.clip(detail_planes[0], 0, 1)
            final_out = np.clip(out_planes[0], 0, 1)

        # 7. OUTPUTS
        # Auto-normalize the ghost view so you can SEE it even if it's tiny
        if final_ghost.max() > 0:
            norm_ghost = final_ghost / final_ghost.max()
        else:
            norm_ghost = final_ghost
            
        self.outputs['amplified_out'] = final_out
        self.outputs['ghost_view'] = norm_ghost # This is the "Night Vision"
        self.outputs['detail_view'] = final_detail

    def get_output(self, name):
        val = self.outputs.get(name)
        if val is None: return None
        # Convert to display format
        disp = (np.clip(val, 0, 1) * 255).astype(np.uint8)
        if disp.ndim == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        return disp