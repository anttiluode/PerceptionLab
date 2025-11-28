import numpy as np
import cv2
from scipy.fft import fft2, fftshift

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class HolographicLatentFFTNode(BaseNode):
    """
    Holographic Latent Encoder.
    
    Standard 2D FFT, but the Spectrum is MODULATED by a Latent Vector (EEG).
    This allows the Brain to "Sculpt" the image in the Frequency Domain.
    
    Mechanism:
    1. Image -> FFT -> Raw Spectrum
    2. EEG Vector -> Projected to Rings (The "Filter")
    3. Raw Spectrum * Filter = Modulated Spectrum
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Holographic FFT (Latent)"
    NODE_COLOR = QtGui.QColor(120, 100, 255) # Deep Phase Blue
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',
            'latent_vector': 'spectrum',    # Wire EEG/VectorSplitter here
            'mod_strength': 'signal'        # 0.0 = Bypass, 1.0 = Full Filter
        }
        
        self.outputs = {
            'complex_spectrum': 'complex_spectrum', # To iFFT Node
            'filter_view': 'image',                 # Visual of the EEG Filter
            'magnitude_view': 'image'               # Resulting Spectrum
        }
        
        self.spectrum = None
        self.filter_mask = None
        self.cached_mag = None
        
        # Grid state
        self.size = 128
        self.center = self.size // 2
        self._build_grid()

    def _build_grid(self):
        y, x = np.ogrid[:self.size, :self.size]
        # Distance from center (0 to ~64)
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)

    def project_latent(self, vector):
        """Map 1D EEG vector to 2D Spectral Rings"""
        if vector is None or len(vector) == 0:
            return np.ones((self.size, self.size), dtype=np.float32)
            
        # Resize grid if vector implies higher resolution? 
        # For now we assume standard 128 visualization size or match image
        
        # Create the ring profile
        # We stretch the vector to cover the radius
        max_r = self.center
        vec_len = len(vector)
        
        # Map radius to vector index
        r_indices = np.clip(self.r_grid * (vec_len / max_r), 0, vec_len - 1).astype(int)
        
        # Project
        rings = vector[r_indices]
        
        # FFT puts low freqs in corners (unshifted), so we need to inverse-shift 
        # this ring pattern to match the raw FFT layout
        return np.fft.ifftshift(rings)

    def step(self):
        # 1. Get Inputs
        img = self.get_blended_input('image_in', 'mean')
        latent = self.get_blended_input('latent_vector', 'sum')
        strength = self.get_blended_input('mod_strength', 'sum')
        
        # Default strength 1.0 if unconnected, but 0.0 if explicitly set low
        if strength is None: strength = 1.0
        
        if img is None:
            return
            
        # 2. Prepare Image
        h, w = img.shape[:2]
        if h != self.size or w != self.size:
            # Resize internal grid to match image
            self.size = h
            self.center = h // 2
            self._build_grid()
            
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            if img.max() > 1.0: img /= 255.0
            
        # 3. FFT (The Hologram)
        raw_spectrum = fft2(img)
        
        # 4. Latent Modulation (The Filter)
        if latent is not None:
            # Project EEG to Rings
            mask = self.project_latent(latent)
            
            # Normalize mask to 0-1
            if np.max(mask) > 0: mask /= np.max(mask)
            
            # Blend based on strength
            # Result = (1-Strength)*1.0 + Strength*Mask
            # If Strength=0, Filter is all 1s (Pass-through)
            self.filter_mask = (1.0 - strength) + (strength * mask)
            
            # Apply Filter
            self.spectrum = raw_spectrum * self.filter_mask
        else:
            self.filter_mask = np.ones_like(raw_spectrum, dtype=np.float32)
            self.spectrum = raw_spectrum
            
        # 5. Visualization
        fshift = fftshift(self.spectrum)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-9)
        
        m_min, m_max = magnitude.min(), magnitude.max()
        if m_max > m_min:
            self.cached_mag = (magnitude - m_min) / (m_max - m_min)
        else:
            self.cached_mag = np.zeros_like(magnitude)

    def get_output(self, port_name):
        if port_name == 'complex_spectrum':
            return self.spectrum
        elif port_name == 'filter_view':
            # Shift back to center for viewing
            if self.filter_mask is not None:
                return fftshift(self.filter_mask)
            return None
        elif port_name == 'magnitude_view':
            return self.cached_mag
        return None

    def get_display_image(self):
        if self.cached_mag is None: return None
        
        h, w = self.cached_mag.shape
        display = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Left: The Modulated Spectrum
        mag_u8 = (np.clip(self.cached_mag, 0, 1) * 255).astype(np.uint8)
        display[:, :w] = cv2.applyColorMap(mag_u8, cv2.COLORMAP_INFERNO)
        
        # Right: The EEG Filter (The "Lens")
        if self.filter_mask is not None:
            # Shift so low freq is in center
            mask_view = fftshift(self.filter_mask)
            mask_u8 = (np.clip(mask_view, 0, 1) * 255).astype(np.uint8)
            display[:, w:] = cv2.applyColorMap(mask_u8, cv2.COLORMAP_OCEAN)
            
        cv2.putText(display, "Spectrum", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(display, "Latent Filter", (w+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0], 
                           display.shape[1]*3, QtGui.QImage.Format.Format_RGB888)