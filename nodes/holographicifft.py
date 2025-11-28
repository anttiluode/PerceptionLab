import numpy as np
import cv2
from scipy.fft import ifft2, ifftshift

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class HolographicIFFTNode(BaseNode):
    """
    Holographic Decoder (2D Inverse FFT).
    Reconstructs a spatial image from a 2D Complex Spectrum.
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Holographic iFFT (Reconstruct)"
    NODE_COLOR = QtGui.QColor(100, 200, 255) # Light Blue
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {'complex_spectrum': 'complex_spectrum'}
        self.outputs = {'reconstructed_image': 'image'}
        
        self.reconstruction = None

    def step(self):
        # 1. Get Spectrum
        spec = self.get_blended_input('complex_spectrum', 'mean')
        
        if spec is None or spec.ndim != 2:
            return

        # 2. Perform Inverse 2D FFT
        # We assume the input is standard unshifted FFT data
        complex_img = ifft2(spec)
        
        # 3. Extract Magnitude (The Image)
        # Real images correspond to the magnitude of the complex result
        self.reconstruction = np.abs(complex_img)
        
        # Normalize 0-1
        r_min, r_max = self.reconstruction.min(), self.reconstruction.max()
        if r_max > r_min:
            self.reconstruction = (self.reconstruction - r_min) / (r_max - r_min)

    def get_output(self, port_name):
        if port_name == 'reconstructed_image':
            return self.reconstruction
        return None

    def get_display_image(self):
        if self.reconstruction is None: return None
        
        # Display Reconstruction
        img_u8 = (np.clip(self.reconstruction, 0, 1) * 255).astype(np.uint8)
        
        h, w = img_u8.shape
        return QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)