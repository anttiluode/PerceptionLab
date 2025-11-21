"""
Ephaptic Field Resonator Node
-----------------------------
Simulates the "Slaving Principle" of the cortical field.
It treats the brain not as a computer (discrete bits) but as a conductive
medium (continuous field).

Mechanism:
1. Input signals act as "current injections" into a 2D grid.
2. The grid simulates "Volume Conduction" (Diffusion + Decay).
3. The resulting "Field" forces the inputs to resonate or die out.

Visualizes:
- The "Slow Wave" (The Ephaptic Field) as Color.
- The "Fast Spikes" (Neural Activity) as Brightness.
"""

import numpy as np
from PyQt6 import QtGui
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------

class EphapticFieldNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(100, 60, 120)  # Deep Purple (Tissue)
    
    def __init__(self, diffusion=0.1, decay=0.95, resolution=128):
        super().__init__()
        self.node_title = "Ephaptic Field (The Substrate)"
        
        self.inputs = {
            'input_vector': 'spectrum', # The EEG signals (spatial vector)
            'coupling_strength': 'signal' # Modulate the field conductivity
        }
        
        self.outputs = {
            'field_state': 'image',    # The visual field
            'order_parameter': 'signal' # The global coherence (0-1)
        }
        
        self.res = int(resolution)
        self.diffusion = float(diffusion)
        self.decay = float(decay)
        
        # The "Cortical Sheet"
        # Two layers: Current State (Field) and Derivative (Change)
        self.field = np.zeros((self.res, self.res), dtype=np.float32)
        
        # Map inputs to spatial locations (Circular layout like a head)
        self.input_map = self._generate_input_map(16) # Assume 16 channels max
        
        self.cached_image = np.zeros((self.res, self.res, 3), dtype=np.uint8)
        self.order_param = 0.0

    def _generate_input_map(self, n_channels):
        """Maps vector indices to X,Y coordinates on the grid"""
        coords = []
        center = self.res / 2.0
        radius = self.res * 0.35
        
        for i in range(n_channels):
            angle = (i / n_channels) * 2.0 * np.pi
            # Fp1/Fp2 are usually at top, Occipital at bottom. 
            # We map 0 to Top (Frontal).
            x = int(center + radius * np.sin(angle))
            y = int(center - radius * np.cos(angle))
            coords.append((x, y))
        return coords

    def step(self):
        # 1. Get Inputs
        signals = self.get_blended_input('input_vector', 'mean')
        coupling_mod = self.get_blended_input('coupling_strength', 'sum')
        
        # Effective diffusion (Ephaptic Strength)
        eff_diffusion = self.diffusion
        if coupling_mod is not None:
            eff_diffusion *= (1.0 + coupling_mod)
            
        # 2. Inject Signals (The Neurons firing into the Field)
        if signals is not None and isinstance(signals, (list, np.ndarray, tuple)):
            # Handle scalar or vector
            sig_arr = np.array(signals).flatten()
            
            for i, val in enumerate(sig_arr):
                if i < len(self.input_map):
                    x, y = self.input_map[i]
                    # Inject voltage (add to field)
                    # We clamp magnitude to avoid explosion
                    self.field[y, x] += np.clip(val * 0.5, -10, 10)
        
        # 3. Physics Simulation (The "Cortical Matter")
        # Diffusion: Energy spreads to neighbors (Volume Conduction)
        # We use Gaussian Blur as a fast approximation of the Heat Equation
        
        k_size = max(3, int(eff_diffusion * 20) | 1) # Ensure odd kernel
        blurred = cv2.GaussianBlur(self.field, (k_size, k_size), 0)
        
        # Decay: Energy dissipates (Resistance)
        self.field = blurred * self.decay
        
        # 4. Compute Order Parameter (The "Slave" Metric)
        # High variance = Chaotic/Desynchronized
        # High magnitude + Low Variance = Synchronized/Slaved
        total_energy = np.sum(np.abs(self.field))
        if total_energy > 0:
            # Calculate spatial coherence (simplistic)
            self.order_param = np.max(self.field) / (total_energy / (self.res**2) + 1e-9)
            self.order_param = np.clip(self.order_param / 100.0, 0, 1)
        
        # 5. Visualization
        self._update_vis()

    def _update_vis(self):
        # Normalize field for display (-1 to 1 -> 0 to 255)
        # We use a colormap to show Potential
        
        disp_field = np.clip(self.field, -1.0, 1.0)
        norm_field = ((disp_field + 1.0) / 2.0 * 255).astype(np.uint8)
        
        # Apply "Plasma" colormap (Energy field look)
        colored = cv2.applyColorMap(norm_field, cv2.COLORMAP_PLASMA)
        
        # Overlay the Input Points (The "Neurons")
        # This shows the contrast between the Source (Neuron) and the Medium (Field)
        for x, y in self.input_map:
            val = self.field[y, x]
            color = (255, 255, 255) if val > 0 else (0, 0, 0)
            cv2.circle(colored, (x, y), 2, color, -1)
            
        self.cached_image = colored

    def get_output(self, port_name):
        if port_name == 'field_state':
            return self.field
        elif port_name == 'order_parameter':
            return self.order_param
        return None

    def get_display_image(self):
        return QtGui.QImage(self.cached_image.data, self.res, self.res, 
                           self.res * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Diffusion (Connectivity)", "diffusion", self.diffusion, None),
            ("Decay (Memory)", "decay", self.decay, None)
        ]