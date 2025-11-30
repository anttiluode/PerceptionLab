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
    # Fallback for testing without host
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class SelfConsistentResonanceNode(BaseNode):
    """
    The Wednesday Loop:
    1. Structure (Space) determines Eigenfrequencies (Time).
    2. Resonance drives Growth (Loop Extrusion).
    3. Growth changes Structure.
    
    This node implements the "Strange Loop" where the mind listens 
    to its own geometry.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Self-Consistent Loop"
    NODE_COLOR = QtGui.QColor(255, 100, 255) # Magenta for emergence
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'frequency_input': 'spectrum',      # From EEG Source
            'feedback_modulation': 'signal',    # From Qubit (The "Error")
            'reset': 'signal'
        }
        
        self.outputs = {
            'structure': 'image',               # The Folded Grid
            'eigenfrequencies': 'spectrum',     # The "Song" of the Shape
            'resonance_field': 'image',         # Where energy lands
            'consciousness_metric': 'signal'    # Mutual Information (0.0 - 1.0)
        }
        
        # --- SIMULATION STATE ---
        self.size = 128
        self.center = self.size // 2
        
        # 1. The Substrate (Complex Quantum Foam)
        self.structure = np.ones((self.size, self.size), dtype=np.complex128)
        self.structure += np.random.randn(self.size, self.size) * 0.01
        
        # 2. The Transfer Function (TAD Insulation)
        # Defines where resonance is ALLOWED to happen.
        self.transfer_function = np.ones((self.size, self.size), dtype=np.float32)
        
        # 3. Memory (Hysteresis)
        self.frequency_memory = np.zeros((self.size, self.size), dtype=np.float32)
        
        # 4. Precomputed Radial Map for fast projection
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        
        self.self_consistency = 0.0
        
    def compute_eigenfrequencies(self):
        """
        What song does the current geometry sing?
        FFT of the Structure.
        """
        # Spatial Structure -> Frequency Domain
        structure_fft = np.abs(fftshift(fft2(self.structure)))
        
        # Normalize
        structure_fft = structure_fft / (np.max(structure_fft) + 1e-9)
        return structure_fft

    def project_to_2d(self, freq_1d):
        """
        Fast projection of 1D EEG spectrum onto 2D radial grid.
        Replaces the slow loop.
        """
        if freq_1d is None or len(freq_1d) == 0:
            return np.zeros((self.size, self.size))
            
        # Interpolate 1D spectrum onto the precomputed 2D radial grid
        # We limit frequency range to the size of the grid radius
        max_r = self.center
        freq_len = len(freq_1d)
        
        # Scale indices to match spectrum length
        r_flat = self.r_grid.ravel()
        # Clip radius to avoid index errors
        r_flat = np.clip(r_flat, 0, freq_len - 1)
        
        # Map values
        projected = freq_1d[r_flat.astype(int)].reshape(self.size, self.size)
        return projected

    def step(self):
        # 1. GET INPUTS
        freq_input = self.get_blended_input('frequency_input', 'sum')
        feedback_mod = self.get_blended_input('feedback_modulation', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        # Handle Reset
        if reset is not None and reset > 0.5:
            self.structure = np.ones((self.size, self.size), dtype=np.complex128) + \
                             (np.random.randn(self.size, self.size) * 0.01)
            self.transfer_function[:] = 1.0
            return

        # Decay if no input
        if freq_input is None:
            self.structure *= 0.95
            return

        # Ensure feedback modulation is sane (default 1.0 if not connected)
        mod = 1.0
        if feedback_mod is not None:
            # Qubit usually outputs 0 or 1, or a probability. 
            # We want it to modulate slightly around 1.0
            mod = 0.8 + (float(feedback_mod) * 0.4) 

        # 2. COMPUTE EIGENFREQUENCIES (The Brain's "Expectation")
        eigen_2d = self.compute_eigenfrequencies()
        
        # 3. PROJECT INPUT (The Sensory Data)
        input_2d = self.project_to_2d(freq_input)
        
        # 4. RESONANCE (Interaction)
        # Resonance happens where Input matches Eigenfrequencies, 
        # modulated by the Feedback (Qubit error)
        resonance_field = input_2d * eigen_2d * mod
        
        # 5. LOOP EXTRUSION (Growth)
        # "TADs form loops where insulation is weak"
        # We grow the structure based on resonance intensity
        growth_force = np.tanh(resonance_field * 5.0) # Non-linear saturation
        
        # Apply growth to complex structure
        # Real part = Density, Imaginary part = Phase/Flow
        self.structure += growth_force * 0.05
        
        # Phase rotation (Time evolution)
        self.structure *= np.exp(1j * 0.1) 
        
        # Diffusion (Topology smoothing)
        # Without this, it becomes white noise. This acts as the "Insulation"
        real_smooth = gaussian_filter(np.real(self.structure), sigma=1.0)
        imag_smooth = gaussian_filter(np.imag(self.structure), sigma=1.0)
        self.structure = real_smooth + 1j * imag_smooth
        
        # Normalize to keep within bounds
        mag = np.abs(self.structure)
        mask = mag > 1.0
        self.structure[mask] /= mag[mask] # normalize only peaks

        # 6. COMPUTE CONSCIOUSNESS METRIC (Closing the Loop)
        # How similar is the structure's song to the input song?
        # High metric = The structure "understands" the input.
        # We simply compare the total energy of resonance vs total input energy
        total_input = np.sum(input_2d) + 1e-9
        total_resonance = np.sum(resonance_field)
        
        # Ratio of captured energy
        self.self_consistency = np.clip(total_resonance / total_input, 0.0, 1.0)
        
        # 7. UPDATE TRANSFER FUNCTION (Plasticity)
        # The structure remembers where it resonated
        self.transfer_function = (self.transfer_function * 0.95) + (resonance_field * 0.05)

    def get_output(self, port_name):
        if port_name == 'structure':
            return np.abs(self.structure)
            
        elif port_name == 'eigenfrequencies':
            # Collapse 2D eigenfrequencies back to 1D for the graph output
            # We take a radial mean
            eigen_2d = self.compute_eigenfrequencies()
            # Simple approximation: take the diagonal or center slice
            mid = self.size // 2
            return eigen_2d[mid, mid:] 
            
        elif port_name == 'resonance_field':
            return np.abs(self.transfer_function)
            
        elif port_name == 'consciousness_metric':
            return float(self.self_consistency)
            
        return None

    def get_display_image(self):
        """
        4-Panel Visualization of the Process
        Top-Left: Structure (Real)
        Top-Right: Phase (Imaginary)
        Bot-Left: Resonance (Where Input matches Structure)
        Bot-Right: Transfer Function (History)
        """
        # Helper for normalization
        def norm(arr):
            arr = np.abs(arr)
            m = np.max(arr)
            if m > 1e-9: arr /= m
            return (arr * 255).astype(np.uint8)

        # 1. Structure
        img_struct = norm(self.structure)
        color_struct = cv2.applyColorMap(img_struct, cv2.COLORMAP_VIRIDIS)
        
        # 2. Phase
        img_phase = norm(np.angle(self.structure))
        color_phase = cv2.applyColorMap(img_phase, cv2.COLORMAP_TWILIGHT)
        
        # 3. Resonance (The "Now")
        # We need to recalculate or store this, let's just visualize the transfer function difference
        img_res = norm(self.compute_eigenfrequencies())
        color_res = cv2.applyColorMap(img_res, cv2.COLORMAP_MAGMA)
        
        # 4. Transfer Function (The "Memory")
        img_mem = norm(self.transfer_function)
        color_mem = cv2.applyColorMap(img_mem, cv2.COLORMAP_INFERNO)
        
        # Stitch
        top = np.hstack((color_struct, color_phase))
        bot = np.hstack((color_res, color_mem))
        full = np.vstack((top, bot))
        
        # Overlay Metric
        cv2.putText(full, f"Loop Integrity: {self.self_consistency:.3f}", (10, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return QtGui.QImage(full.data, full.shape[1], full.shape[0], 
                           full.shape[1]*3, QtGui.QImage.Format.Format_RGB888)