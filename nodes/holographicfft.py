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

class HolographicFFTNode(BaseNode):
    """
    Holographic Encoder (2D FFT).
    Transforms a spatial image into a 2D Complex Frequency Domain.
    Preserves ALL spatial information in the Phase.
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Holographic FFT (2D)"
    NODE_COLOR = QtGui.QColor(100, 100, 255) # Phase Blue
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {'image_in': 'image'}
        self.outputs = {
            'complex_spectrum': 'complex_spectrum', # The 2D Hologram
            'magnitude_view': 'image',              # Visualizable Power Spectrum
            'phase_view': 'image'                   # Visualizable Phase
        }
        
        self.spectrum = None
        self.cached_mag = None

    def step(self):
        # 1. Get Input
        img = self.get_blended_input('image_in', 'mean')
        
        if img is None:
            return
            
        # 2. Prepare Image (Grayscale Float)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            if img.max() > 1.0: img /= 255.0
            
        # 3. Perform 2D FFT
        # We do NOT shift here for the data output, only for display
        self.spectrum = fft2(img)
        
        # 4. Visualization (Magnitude)
        # Shift zero frequency to center for viewing
        fshift = fftshift(self.spectrum)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-9)
        
        # Normalize magnitude for display
        m_min, m_max = magnitude.min(), magnitude.max()
        if m_max > m_min:
            self.cached_mag = (magnitude - m_min) / (m_max - m_min)
        else:
            self.cached_mag = np.zeros_like(magnitude)

    def get_output(self, port_name):
        if port_name == 'complex_spectrum':
            return self.spectrum
        elif port_name == 'magnitude_view':
            return self.cached_mag
        return None

    def get_display_image(self):
        if self.cached_mag is None: return None
        
        # Display Magnitude Spectrum
        img_u8 = (np.clip(self.cached_mag, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)