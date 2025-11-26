import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter, laplace

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
            'structure': 'image',
            'eigenfrequencies': 'spectrum',
            'tension_map': 'image',         # NEW: Visualizing the stress
            'criticality_metric': 'signal'   # How close to chaos?
        }
        
        self.size = 128
        self.center = self.size // 2
        
        # 1. The Substrate (Complex Phase Field)
        self.structure = np.ones((self.size, self.size), dtype=np.complex128)
        self.structure += (np.random.randn(self.size, self.size) + 
                           1j * np.random.randn(self.size, self.size)) * 0.1
        
        # 2. Tension Map (Potential Energy)
        self.tension = np.zeros((self.size, self.size), dtype=np.float32)
        
        # 3. Transfer Function (The "Scars" / Insulation)
        # Starts open (1.0), closes down as it burns out
        self.transfer_function = np.ones((self.size, self.size), dtype=np.float32)
        
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        
        self.avalanche_count = 0.0

    def compute_eigenfrequencies(self):
        # The song of the shape
        return np.abs(fftshift(fft2(self.structure)))

    def project_to_2d(self, freq_1d):
        if freq_1d is None or len(freq_1d) == 0:
            return np.zeros((self.size, self.size))
        
        # Map spectrum to rings
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
            self.tension[:] = 0
            self.transfer_function[:] = 1.0
            return

        if freq_input is None:
            self.tension *= 0.9 # Leak energy
            return

        # 2. THE PHYSICS OF CONFLICT
        
        # A. The System's Current State
        eigen = self.compute_eigenfrequencies()
        eigen /= (np.max(eigen) + 1e-9)
        
        # B. The External Forcing
        input_2d = self.project_to_2d(freq_input)
        input_2d /= (np.max(input_2d) + 1e-9)
        
        # C. Mismatch Calculation (Where does the input fight the structure?)
        # If input matches eigen, resonance is high (Easy flow).
        # If input hits where eigen is low, RESISTANCE generates TENSION.
        resistance = input_2d * (1.0 - eigen)
        
        # Modulate by Qubit/Feedback (The "Right Temple" Error)
        # Low feedback allows tension to build faster
        threshold = 0.6 + (feedback_mod * 0.3) 
        
        # D. Accumulate Tension (Integrate over time)
        # This is the "Charging" phase
        self.tension += resistance * 0.1
        
        # E. THE SNAP (Phase Transition)
        # Where tension exceeds threshold, the structure BREAKS.
        critical_mask = self.tension > threshold
        self.avalanche_count = np.sum(critical_mask)
        
        if self.avalanche_count > 0:
            # 1. Structural Inversion (The Pop)
            # We flip the phase at critical points, creating a sudden topological defect
            self.structure[critical_mask] *= -1 
            
            # 2. Loop Extrusion (The Scar)
            # The system learns to IGNORE this frequency area to prevent future pain.
            # It creates an "insulator" (TAD boundary)
            self.transfer_function[critical_mask] *= 0.8 
            
            # 3. Energy Release
            self.tension[critical_mask] = 0 # Discharge
            
            # 4. Shockwave (Coupling to neighbors)
            # The collapse pulls neighbors in, potentially causing a chain reaction next frame
            self.structure = gaussian_filter(np.real(self.structure), sigma=0.5) + \
                             1j * gaussian_filter(np.imag(self.structure), sigma=0.5)

        # F. Passive Evolution (Drift)
        # Rotate phase slowly based on transfer function (insulated areas move slower)
        self.structure *= np.exp(1j * (0.05 * self.transfer_function))
        
        # Normalize to prevent explosion
        mag = np.abs(self.structure)
        self.structure[mag > 1.0] /= mag[mag > 1.0]

    def get_output(self, port_name):
        if port_name == 'structure':
            return np.abs(self.structure)
        elif port_name == 'eigenfrequencies':
            # Radial mean for spectrum graph
            spec = self.compute_eigenfrequencies()
            center = self.size // 2
            return spec[center, center:] 
        elif port_name == 'criticality_metric':
            # Normalize count to 0-1 range
            return float(np.clip(self.avalanche_count / 100.0, 0, 1))
        return None

    def get_display_image(self):
        # 4-Panel Diagnosis
        
        # 1. Tension (The buildup)
        img_tension = (np.clip(self.tension, 0, 1) * 255).astype(np.uint8)
        c_tension = cv2.applyColorMap(img_tension, cv2.COLORMAP_HOT)
        
        # 2. Structure Phase (The Geometry)
        img_phase = ((np.angle(self.structure) / np.pi + 1) * 127).astype(np.uint8)
        c_phase = cv2.applyColorMap(img_phase, cv2.COLORMAP_TWILIGHT)
        
        # 3. Transfer Function (The Scars/Memory)
        img_trans = (self.transfer_function * 255).astype(np.uint8)
        c_trans = cv2.applyColorMap(img_trans, cv2.COLORMAP_BONE)
        
        # 4. Eigenfrequencies (The Current Mode)
        img_eig = (self.compute_eigenfrequencies() * 255).astype(np.uint8)
        c_eig = cv2.applyColorMap(img_eig, cv2.COLORMAP_JET)

        # Layout
        top = np.hstack((c_phase, c_tension))
        bot = np.hstack((c_trans, c_eig))
        full = np.vstack((top, bot))
        
        # Overlay Text
        status = "CRITICAL" if self.avalanche_count > 10 else "Charging..."
        cv2.putText(full, f"Mode: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return QtGui.QImage(full.data, full.shape[1], full.shape[0], 
                           full.shape[1]*3, QtGui.QImage.Format.Format_RGB888)