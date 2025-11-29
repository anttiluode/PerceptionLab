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

class ComplexInterferenceNode(BaseNode):
    """
    Complex Field Interferometer.
    Takes two complex spectra (A and B) and computes their interference.
    
    Output = A + B (Linear Superposition)
          or A * B (Convolution / Filtering)
          or Cross-Correlation
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Complex Interference"
    NODE_COLOR = QtGui.QColor(160, 100, 255) # Wave Violet
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_a': 'complex_spectrum',
            'complex_b': 'complex_spectrum',
            'mode_select': 'signal', # 0=Add, 1=Mult, 2=Subtract
            'mix_ratio': 'signal'    # 0.0=A only, 1.0=B only (for Add mode)
        }
        
        self.outputs = {
            'interference_out': 'complex_spectrum',
            'magnitude_view': 'image'
        }
        
        self.result = None
        self.cached_mag = None
        self.size = 128

    def step(self):
        # 1. Get Inputs
        spec_a = self.get_blended_input('complex_a', 'first')
        spec_b = self.get_blended_input('complex_b', 'first')
        mode = int(self.get_blended_input('mode_select', 'sum') or 0)
        mix = self.get_blended_input('mix_ratio', 'sum')
        if mix is None: mix = 0.5
        
        # 2. Validation & Resizing
        if spec_a is None and spec_b is None: return
        
        # Handle single inputs
        if spec_a is None: spec_a = np.zeros_like(spec_b)
        if spec_b is None: spec_b = np.zeros_like(spec_a)
        
        # Ensure sizes match (crop/pad to largest?)
        # For simplicity, we assume standard grid size or resize to A
        if spec_a.shape != spec_b.shape:
            # Resize B to match A
            # Complex resize is tricky, let's just crop/pad or require match
            # Returning None prevents crash
            if spec_a.shape != spec_b.shape:
                return 

        self.size = spec_a.shape[0]

        # 3. INTERFERENCE PHYSICS
        if mode == 0: # Superposition (Add)
            # Weighted mix
            # Result = (1-mix)*A + (mix)*B
            # This simulates two light beams shining on the same spot
            self.result = (spec_a * (1.0 - mix)) + (spec_b * mix)
            
        elif mode == 1: # Convolution (Multiply)
            # Multiplication in Frequency Domain = Convolution in Spatial Domain
            # This effectively filters Image A with Image B
            self.result = spec_a * spec_b
            
        elif mode == 2: # Subtraction (Phase Cancellation)
            # Useful for "removing" a known signal from a mix
            self.result = spec_a - spec_b

        elif mode == 3: # Phase Conjugation (Time Reversal)
            # A * conjugate(B)
            # This is Cross-Correlation
            self.result = spec_a * np.conj(spec_b)

        # 4. Visualization
        mag = np.log(np.abs(np.fft.fftshift(self.result)) + 1)
        if mag.max() > 0: mag /= mag.max()
        self.cached_mag = mag

    def get_output(self, port_name):
        if port_name == 'interference_out':
            return self.result
        elif port_name == 'magnitude_view':
            return self.cached_mag
        return None

    def get_display_image(self):
        if self.cached_mag is None: return None
        
        h, w = self.size, self.size
        img_u8 = (np.clip(self.cached_mag, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
        
        cv2.putText(img_color, "INTERFERENCE", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(img_color.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)