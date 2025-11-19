# spikingeigenmodenode.py
"""
Spiking Eigenmode Node (The Neural Drum)
----------------------------------------
Treats the 55 DNA coefficients as input currents into 55 
Resonant Integrate-and-Fire Neurons.

Instead of a static map, this node 'rings' like a drumhead 
when specific shapes are detected, adding TIME and RHYTHM 
to the morphological process.
"""

import numpy as np
import cv2
from scipy.special import jn, jn_zeros
from scipy.ndimage import gaussian_filter
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SpikingEigenmodeNode(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(200, 60, 60) # "Spiking" Red

    def __init__(self, resolution=256, decay=0.1, threshold=0.5):
        super().__init__()
        self.node_title = "Spiking Eigenmodes (The Drum)"
        
        self.inputs = {
            'dna_current': 'spectrum', # Input Current (from Scanner)
            'inhibition': 'signal'     # Global inhibition (calms the drum)
        }
        
        self.outputs = {
            'drum_surface': 'image',   # The visual wave pattern
            'spike_activity': 'spectrum', # Which modes just fired (55-dim)
            'total_energy': 'signal'   # Total volume of the drum
        }
        
        self.resolution = int(resolution)
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.num_modes = 55 

        # Physics: 55 Integrate-and-Fire Neurons
        self.voltages = np.zeros(self.num_modes, dtype=np.float32)
        self.ringing_amplitudes = np.zeros(self.num_modes, dtype=np.float32)
        
        # Precompute the "Bell Shapes" (Bessel Modes)
        self.basis_functions = []
        self._precompute_basis()
        
        self.output_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)

    def _precompute_basis(self):
        h, w = self.resolution, self.resolution
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        x_norm = (x - cx) / (w / 2)
        y_norm = (y - cy) / (h / 2)
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-9
        theta = np.arctan2(y_norm, x_norm)
        mask = (r <= 1.0).astype(np.float32)

        # Generate 55 modes (n=1..5, m=0..5)
        # We treat 'n' as the "Pitch" (frequency) of the bell
        for n in range(1, 6):
            for m in range(0, 6):
                try:
                    zeros = jn_zeros(m, n)
                    k = zeros[-1]
                    
                    radial = jn(m, k * r)
                    
                    if m == 0:
                        mode = radial * mask
                        # Normalize so they all "ring" at same volume
                        mode /= (np.linalg.norm(mode) + 1e-9)
                        self.basis_functions.append(mode)
                    else:
                        # Cosine Mode
                        mode_c = radial * np.cos(m * theta) * mask
                        mode_c /= (np.linalg.norm(mode_c) + 1e-9)
                        self.basis_functions.append(mode_c)
                        
                        # Sine Mode
                        mode_s = radial * np.sin(m * theta) * mask
                        mode_s /= (np.linalg.norm(mode_s) + 1e-9)
                        self.basis_functions.append(mode_s)
                except:
                    continue
        
        # Trim to 55 if we went over
        self.basis_functions = self.basis_functions[:self.num_modes]

    def step(self):
        # 1. Get Input Current (DNA)
        currents = self.get_blended_input('dna_current', 'first')
        inhibition = self.get_blended_input('inhibition', 'sum') or 0.0
        
        if currents is None:
            currents = np.zeros(self.num_modes)
        
        if len(currents) > self.num_modes:
            currents = currents[:self.num_modes]
        elif len(currents) < self.num_modes:
            currents = np.pad(currents, (0, self.num_modes - len(currents)))

        # 2. Neuron Dynamics (Integrate and Fire)
        # Charge up the neurons based on input matching
        # Abs() because we care about magnitude of match, not sign
        self.voltages += np.abs(currents) * 0.5 
        
        # Apply Decay (Leak)
        self.voltages *= (0.9 - inhibition * 0.1)
        
        # Check for Spikes
        spikes = (self.voltages > self.threshold).astype(np.float32)
        
        # Reset fired neurons
        self.voltages[spikes > 0] = 0.0
        
        # 3. The "Ringing" Physics
        # When a neuron spikes, it "strikes" the bell (adds energy to amplitude)
        self.ringing_amplitudes += spikes * 1.0 
        
        # The ringing decays over time (Damping)
        self.ringing_amplitudes *= (1.0 - self.decay)
        
        # 4. Synthesize the Sound (Visual Pattern)
        self.output_map.fill(0.0)
        
        for i in range(min(len(self.ringing_amplitudes), len(self.basis_functions))):
            amp = self.ringing_amplitudes[i]
            if amp > 0.01: # Optimization: don't draw silent modes
                # Add the mode to the map, weighted by its ringing volume
                # We use alternating signs for visual interference patterns
                sign = 1 if i % 2 == 0 else -1 
                self.output_map += self.basis_functions[i] * amp * sign
        
        # Normalize for display
        # Sigmoid to squish extreme resonances
        self.output_map = np.tanh(self.output_map * 2.0)
        
    def get_output(self, port_name):
        if port_name == 'drum_surface':
            return self.output_map
        elif port_name == 'spike_activity':
            return self.ringing_amplitudes
        elif port_name == 'total_energy':
            return float(np.sum(self.ringing_amplitudes))
        return None

    def get_display_image(self):
        # Map -1..1 to 0..255
        img_norm = (self.output_map + 1.0) / 2.0
        img_u8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
        
        # Overlay spike raster
        for i in range(self.num_modes):
            if self.ringing_amplitudes[i] > 0.1:
                x = int((i / self.num_modes) * self.resolution)
                h = int(self.ringing_amplitudes[i] * 20)
                cv2.rectangle(img_color, (x, 0), (x+2, h), (255, 255, 255), -1)

        return QtGui.QImage(img_color.data, self.resolution, self.resolution, self.resolution * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Decay (Damping)", "decay", self.decay, None),
            ("Fire Threshold", "threshold", self.threshold, None),
        ]