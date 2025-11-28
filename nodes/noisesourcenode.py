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

class NoiseSourceNode(BaseNode):
    """
    The Chaos Generator.
    Outputs pure random energy to drive the Holographic Inverse.
    
    Modes:
    - Structure (2D): For direct image reconstruction.
    - Spectrum (1D): For driving resonance nodes.
    """
    NODE_CATEGORY = "Source"
    NODE_TITLE = "Noise Source (The Beam)"
    NODE_COLOR = QtGui.QColor(150, 150, 150) # Grey
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'amplitude': 'signal'
        }
        
        self.outputs = {
            'noise_image': 'image',      # 2D White Noise
            'noise_spectrum': 'spectrum' # 1D White Noise
        }
        
        self.size = 128
        self.amp = 1.0
        self.last_noise = None

    def step(self):
        # 1. Get Amplitude
        mod = self.get_blended_input('amplitude', 'sum')
        if mod is not None:
            self.amp = float(mod)
        
        # 2. Generate 2D Noise (The Beam)
        # We use Uniform noise 0-1 for visualization, 
        # or Gaussian for physics. Let's use Gaussian centered at 0.5.
        self.last_noise = np.random.randn(self.size, self.size).astype(np.float32)
        
        # Scale
        self.last_noise *= self.amp
        
        # 3. Generate 1D Noise (The Signal)
        self.last_spec = np.random.rand(16).astype(np.float32) * self.amp

    def get_output(self, port_name):
        if port_name == 'noise_image':
            # Shift to 0-1 range for image pipeline compatibility if needed, 
            # but usually physics nodes want raw +/- values.
            # Let's return raw.
            return self.last_noise
        elif port_name == 'noise_spectrum':
            return self.last_spec
        return None

    def get_display_image(self):
        if self.last_noise is None: return None
        
        # Visualize Static
        # Normalize -3 to +3 sigma -> 0 to 1
        disp = (self.last_noise / 3.0) + 0.5
        img = (np.clip(disp, 0, 1) * 255).astype(np.uint8)
        
        # Color Map (TV Static)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return QtGui.QImage(color_img.data, self.size, self.size, 
                           self.size*3, QtGui.QImage.Format.Format_RGB888)