import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class SelfConsistentResonanceNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Strange Loop (Criticality)"
    NODE_COLOR = QtGui.QColor(255, 50, 100) # Red for Instability
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'frequency_input': 'spectrum',
            'feedback_modulation': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'structure': 'image',           # Top-Left: Geometry
            'tension_map': 'image',         # Top-Right: Stress
            'scars_insulation': 'image',    # Bot-Left: Memory
            'eigen_image': 'image',         # Bot-Right: The Star (Log-Scaled for Analyzer)
            
            'eigenfrequencies': 'spectrum', # 1D Spectrum
            'criticality_metric': 'signal'  # 0-1
        }
        
        self.size = 128
        self.center = self.size // 2
        
        # 1. The Substrate
        self.structure = np.ones((self.size, self.size), dtype=np.complex128)
        self.structure += (np.random.randn(self.size, self.size) + 
                           1j * np.random.randn(self.size, self.size)) * 0.1
        
        # 2. Tension Map
        self.tension = np.zeros((self.size, self.size), dtype=np.float32)
        
        # 3. Transfer Function
        self.transfer_function = np.ones((self.size, self.size), dtype=np.float32)
        
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        
        self.avalanche_count = 0.0

    def compute_eigenfrequencies(self):
        return np.abs(fftshift(fft2(self.structure)))

    def project_to_2d(self, freq_1d):
        if freq_1d is None or len(freq_1d) == 0:
            return np.zeros((self.size, self.size))
        
        max_r = self.center
        freq_len = len(freq_1d)
        r_flat = np.clip(self.r_grid.ravel(), 0, freq_len - 1).astype(int)
        return freq_1d[r_flat].reshape(self.size, self.size)

    def step(self):
        # 1. INPUTS
        freq_input = self.get_blended_input('frequency_input', 'sum')
        feedback_mod = self.get_blended_input('feedback_modulation', 'sum') or 0.0
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.structure = np.ones((self.size, self.size), dtype=np.complex128)
            self.structure += (np.random.randn(self.size, self.size) + 
                               1j * np.random.randn(self.size, self.size)) * 0.1
            self.tension[:] = 0
            self.transfer_function[:] = 1.0
            return

        if freq_input is None:
            self.tension *= 0.9
            return

        # 2. PHYSICS
        eigen = self.compute_eigenfrequencies()
        eigen_norm = eigen / (np.max(eigen) + 1e-9)
        
        input_2d = self.project_to_2d(freq_input)
        input_2d /= (np.max(input_2d) + 1e-9)
        
        resistance = input_2d * (1.0 - eigen_norm)
        threshold = 0.6 + (feedback_mod * 0.3) 
        self.tension += resistance * 0.1
        
        # 3. AVALANCHE
        critical_mask = self.tension > threshold
        self.avalanche_count = np.sum(critical_mask)
        
        if self.avalanche_count > 0:
            self.structure[critical_mask] *= -1 
            self.transfer_function[critical_mask] *= 0.8 
            self.tension[critical_mask] = 0 
            self.structure = gaussian_filter(np.real(self.structure), sigma=0.5) + \
                             1j * gaussian_filter(np.imag(self.structure), sigma=0.5)

        # Passive Evolution
        self.structure *= np.exp(1j * (0.05 * self.transfer_function))
        
        # Normalize internal state
        mag = np.abs(self.structure)
        self.structure[mag > 1.0] /= mag[mag > 1.0]

    def get_output(self, port_name):
        # Helper for image normalization
        def normalize_img(arr):
            arr_abs = np.abs(arr)
            m = np.max(arr_abs)
            if m > 1e-9: arr_abs /= m
            return (arr_abs * 255).astype(np.uint8)

        if port_name == 'structure':
            return normalize_img(self.structure)
            
        elif port_name == 'tension_map':
            return normalize_img(self.tension)
            
        elif port_name == 'scars_insulation':
            return normalize_img(self.transfer_function)
            
        elif port_name == 'eigen_image':
            # --- THE FIX: LOG SCALING ---
            spec = self.compute_eigenfrequencies()
            # Log scale lifts the star structure out of the darkness
            spec_log = np.log(1 + spec)
            return normalize_img(spec_log)

        elif port_name == 'eigenfrequencies':
            spec = self.compute_eigenfrequencies()
            center = self.size // 2
            return spec[center, center:] 
            
        elif port_name == 'criticality_metric':
            return float(np.clip(self.avalanche_count / 100.0, 0, 1))
            
        return None

    def get_display_image(self):
        # 1. Structure
        img_struc = np.abs(self.structure)
        if img_struc.max() > 0: img_struc /= img_struc.max()
        c_struc = cv2.applyColorMap((img_struc * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
        
        # 2. Tension
        img_tension = np.clip(self.tension, 0, 1)
        c_tension = cv2.applyColorMap((img_tension * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # 3. Scars
        img_trans = self.transfer_function
        c_trans = cv2.applyColorMap((img_trans * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        
        # 4. Star (Raw for Display)
        # We keep this RAW/DIRTY for the display so you can see the detailed noise wrapping
        raw_eigen = self.compute_eigenfrequencies()
        img_eig = (raw_eigen * 255).astype(np.uint8) 
        c_eig = cv2.applyColorMap(img_eig, cv2.COLORMAP_JET)

        # Assemble
        top = np.hstack((c_struc, c_tension))
        bot = np.hstack((c_trans, c_eig))
        full = np.vstack((top, bot))
        
        status = "CRITICAL" if self.avalanche_count > 10 else "Charging"
        cv2.putText(full, f"Mode: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return QtGui.QImage(full.data, full.shape[1], full.shape[0], 
                           full.shape[1]*3, QtGui.QImage.Format.Format_RGB888)