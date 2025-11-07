"""
Topological XOR Node - Simulates a logic gate (A XOR B) realized by the 
physical annihilation of wave-like particles (solitons) within a structured
potential scaffold.

Outputs the computation state as an image and the logical result as a signal.
Ported from topological_xor.py.
Requires: pip install numpy scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: TopologicalXORNode requires 'scipy'.")


# --- Simulation Constants (optimized for node) ---
GRID_SIZE = 96
DT = 0.1  
DAMPING = 0.99 
A_LAUNCH_POS = 0.25 # Y position for Input A (normalized)
B_LAUNCH_POS = 0.75 # Y position for Input B (normalized)
OUTPUT_Y_POS = 0.5  # Y position for the output channel (normalized)


class TopologicalGate:
    def __init__(self, size):
        self.size = size
        self.psi = np.zeros((size, size), dtype=np.complex64)
        self.psi_prev = np.zeros((size, size), dtype=np.complex64)
        self.information_density = np.zeros((size, size), dtype=np.float32)
        self.environmental_potential = self._create_xor_gate_scaffold()
        self.output_y_px = int(OUTPUT_Y_POS * self.size)
        
        # State tracking for XOR result
        self.result = 0.0
        self.last_state_check_time = 0

    def _create_xor_gate_scaffold(self):
        """Creates a hard-coded potential to act as an XOR gate."""
        potential = np.ones((self.size, self.size), dtype=np.float32) * 0.1

        channel_width = 8
        junction_x = self.size // 2

        # --- Input Wire A (from top-left) ---
        yA = int(A_LAUNCH_POS * self.size)
        potential[yA - channel_width//2 : yA + channel_width//2, :junction_x] = -0.1
        
        # --- Input Wire B (from bottom-left) ---
        yB = int(B_LAUNCH_POS * self.size)
        potential[yB - channel_width//2 : yB + channel_width//2, :junction_x] = -0.1
            
        # --- Output Wire C (to the right) ---
        output_y = int(OUTPUT_Y_POS * self.size)
        potential[output_y - channel_width//2 : output_y + channel_width//2, junction_x:] = -0.1
        
        return gaussian_filter(potential, sigma=2.0)

    def evolve(self):
        """Evolve the field using non-linear dynamics for particle interaction."""
        laplacian = (np.roll(self.psi, 1, axis=0) + np.roll(self.psi, -1, axis=0) +
                     np.roll(self.psi, 1, axis=1) + np.roll(self.psi, -1, axis=1) - 4 * self.psi)

        # Non-linear potential for annihilation/stability
        psi_sq = np.abs(self.psi)**2
        # Non-linear term (simplified Mexican Hat potential derivative)
        nonlinear_term = self.psi * (psi_sq - 1.0) 

        # The evolution equation (Non-linear wave evolution)
        psi_next = (2 * self.psi - self.psi_prev * DAMPING +
                    DT**2 * (laplacian - nonlinear_term) - 
                    self.environmental_potential * self.psi)

        self.psi_prev, self.psi = self.psi, psi_next
        
        # Calculate information density (gradient squared) for visualization
        grad_x, grad_y = np.gradient(np.abs(self.psi))
        self.information_density = grad_x**2 + grad_y**2

    def launch_soliton(self, start_y, amplitude=2.5):
        """Launches a soliton (a '1' bit) down the wire at a specific Y-position."""
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        start_x = 5
        
        dist_sq = (xx - start_x)**2 + (yy - start_y)**2
        
        pulse = amplitude * np.exp(-dist_sq / 10.0)
        self.psi += pulse.astype(np.complex64)

    def measure_output(self, measure_window=5, measure_time_step=20):
        """Measures the field amplitude in the output channel to determine the XOR result."""
        
        # Only check once every X steps to give the field time to settle
        if self.last_state_check_time < measure_time_step:
            self.last_state_check_time += 1
            return self.result
            
        self.last_state_check_time = 0 # Reset timer
        
        # Define the measurement area (far right of the grid)
        measurement_area = self.psi[self.output_y_px - measure_window:self.output_y_px + measure_window, 
                                   self.size - measure_window*2:self.size - measure_window]
        
        # The result is 1 if the field amplitude is significant (a soliton survived)
        max_amplitude = np.max(np.abs(measurement_area))
        
        # If the amplitude is above a threshold, the result is 1
        if max_amplitude > 0.5:
            self.result = 1.0
            # Annihilate the soliton to prepare for the next computation
            self.psi[self.output_y_px - measure_window:self.output_y_px + measure_window, 
                     self.size - measure_window*2:self.size - measure_window] = 0j
        else:
            self.result = 0.0
            
        return self.result
        
    def reset_field(self):
        """Clear the field for a new computation."""
        self.psi.fill(0j)
        self.psi_prev.fill(0j)
        self.information_density.fill(0.0)
        self.result = 0.0
        self.last_state_check_time = 0


class TopologicalXORNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(150, 50, 50) # Chaotic Red for Computation
    
    def __init__(self, size=96):
        super().__init__()
        self.node_title = "Topological XOR"
        
        self.inputs = {
            'input_A': 'signal', # 0 or 1 bit
            'input_B': 'signal', # 0 or 1 bit
            'compute_trigger': 'signal', # Rising edge triggers launch
            'reset': 'signal'
        }
        self.outputs = {
            'output_C': 'signal',          # XOR result (0 or 1)
            'computation_image': 'image',  # Information Density
            'xor_state': 'signal'          # 0=Idle, 1=Computing, 2=Done
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "XOR (No SciPy!)"
            return
            
        self.size = int(size)
        self.sim = TopologicalGate(self.size)
        self.last_trigger_val = 0.0
        self.current_result = 0.0
        self.computation_state = 0.0 # 0=Idle, 1=Computing, 2=Done

    def _launch_if_one(self, signal, y_norm_pos):
        """Launches a soliton if the signal is high (>= 0.5)."""
        if signal >= 0.5:
            self.sim.launch_soliton(int(y_norm_pos * self.size))

    def randomize(self):
        """The 'randomize' button acts as a full reset here."""
        self.sim.reset_field()
        self.computation_state = 0.0

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # 1. Handle Inputs
        trigger_val = self.get_blended_input('compute_trigger', 'sum') or 0.0
        reset_sig = self.get_blended_input('reset', 'sum')
        input_A_sig = self.get_blended_input('input_A', 'sum') or 0.0
        input_B_sig = self.get_blended_input('input_B', 'sum') or 0.0

        if reset_sig is not None and reset_sig > 0.5:
            self.randomize()
            return
            
        # 2. Computation Logic (Rising edge triggers launch)
        if trigger_val > 0.5 and self.last_trigger_val <= 0.5:
            self.sim.reset_field() # Ensure clean start
            self.computation_state = 1.0 # State: Computing
            
            # Launch solitons based on input bits (0 or 1)
            self._launch_if_one(round(input_A_sig), A_LAUNCH_POS)
            self._launch_if_one(round(input_B_sig), B_LAUNCH_POS)
            
        self.last_trigger_val = trigger_val

        # 3. Always Evolve the Physics
        self.sim.evolve()
        
        # 4. Measure Output (if computing)
        if self.computation_state == 1.0:
            self.current_result = self.sim.measure_output()
            # If the output has been measured and the field is quiet, computation is done
            if self.current_result in [0.0, 1.0] and self.sim.last_state_check_time == 0:
                self.computation_state = 2.0 # State: Done

    def get_output(self, port_name):
        if port_name == 'output_C':
            return self.current_result
        elif port_name == 'computation_image':
            # Output Information Density
            max_val = np.max(self.sim.information_density)
            if max_val > 1e-9:
                return self.sim.information_density.T / max_val
            return self.sim.information_density.T
        elif port_name == 'xor_state':
            return self.computation_state
            
        return None
        
    def get_display_image(self):
        # 1. Base Visualization: Information Density
        img_data = self.get_output('computation_image')
        if img_data is None: 
            return None
            
        img_u8 = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        # 2. Output Indicator
        bar_color = (0, 0, 0)
        if self.computation_state == 2.0:
             bar_color = (0, 255, 0) if self.current_result == 1.0 else (0, 0, 255)
        elif self.computation_state == 1.0:
             bar_color = (255, 255, 0)
             
        h, w = img_color.shape[:2]
        cv2.rectangle(img_color, (w-15, 0), (w, 15), bar_color, -1) # Top right status light
        
        # 3. Resize to thumbnail size
        img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
        ]