"""
World Substrate Node - The "External World" for the Human Attractor.
A complex, self-evolving field that generates perception, reward, and pain signals.

- Perception (psi_external) = Average field energy
- Reward (dopamine) = Field stability/coherence
- Pain (pain_stimulus) = Sudden, chaotic instanton/decay events

Based on the physics of ResonantInstantonModel from instantonassim x.py
Requires: pip install numpy scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from scipy.ndimage import gaussian_filter
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: WorldSubstrateNode requires 'scipy'.")

# --- Core Physics Engine (from instantonassim x.py) ---
class WorldField:
    def __init__(self, grid_size=96, dt=0.05, c=1.0, a=0.1, b=0.1, gamma=0.02, substrate_noise=0.0005):
        self.grid_size = grid_size
        self.dt = dt
        self.c = c
        self.a = a
        self.b = b
        self.gamma = gamma
        self.substrate_noise = substrate_noise
        
        self.phi = np.zeros((grid_size, grid_size))
        self.phi_prev = np.zeros((grid_size, grid_size))
        self.stability_metric = 1.0
        self.time = 0.0
        
        # Output signals
        self.psi_external_out = 0.0
        self.dopamine_out = 0.0
        self.pain_out = 0.0
        
        self.initialize_field()

    def initialize_field(self):
        """Initialize with a complex, multi-modal field."""
        position = (self.grid_size // 2, self.grid_size // 2)
        self.phi = np.zeros((self.grid_size, self.grid_size))
        
        # Create a complex initial state
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        r = np.sqrt((x - position[0])**2 + (y - position[1])**2)
        
        # Add a few "lumps" (pseudo-atoms)
        self.phi += 1.5 * np.exp(-r**2 / (2 * (self.grid_size/8)**2))
        self.phi += 1.0 * np.exp(-((x - 20)**2 + (y - 30)**2) / (2 * (self.grid_size/12)**2))
        self.phi -= 1.0 * np.exp(-((x - 70)**2 + (y - 60)**2) / (2 * (self.grid_size/10)**2))
        
        self.phi_prev = self.phi.copy()
        self.time = 0.0
        self.stability_metric = 1.0

    def _laplacian(self, field):
        field_padded = np.pad(field, 1, mode='wrap')
        laplacian = (field_padded[:-2, 1:-1] + field_padded[2:, 1:-1] + 
                     field_padded[1:-1, :-2] + field_padded[1:-1, 2:] - 
                     4 * field_padded[1:-1, 1:-1])
        return laplacian
    
    def _biharmonic(self, field):
        return self._laplacian(self._laplacian(field))

    def step(self):
        phi_old = self.phi.copy()
        
        laplacian_phi = self._laplacian(self.phi)
        biharmonic_phi = self._biharmonic(self.phi) if self.gamma != 0 else 0
        noise = self.substrate_noise * np.random.normal(size=self.phi.shape)
        
        accel = (self.c**2 * laplacian_phi + 
                 self.a * self.phi - 
                 self.b * self.phi**3 - 
                 self.gamma * biharmonic_phi + 
                 noise)
        
        phi_new = 2 * self.phi - self.phi_prev + self.dt**2 * accel
        
        self.phi_prev = self.phi
        self.phi = phi_new
        self.time += self.dt
        
        # --- Compute Outputs for the Human ---
        
        # 1. Pain Signal (Instanton Event)
        # A sudden, chaotic change in the field = pain
        delta_phi_mag = np.mean(np.abs(phi_new - phi_old))
        # If change is large (> 0.01), register as a pain event
        self.pain_out = np.clip(delta_phi_mag * 100.0, 0.0, 1.0)
        
        # 2. Dopamine Signal (Stability)
        # Stability is high if the field is coherent (low variance)
        field_variance = np.std(self.phi)
        self.stability_metric = np.clip(1.0 - field_variance, 0.0, 1.0)
        # Dopamine is high when stability is high
        self.dopamine_out = self.stability_metric
        
        # 3. Perception Signal (psi_external)
        # What the human "sees" is the total energy/activity of the field
        self.psi_external_out = np.mean(np.abs(self.phi))


class WorldSubstrateNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(20, 150, 150) # Biological Teal
    
    def __init__(self, grid_size=96, substrate_noise=0.0005):
        super().__init__()
        self.node_title = "World Substrate"
        
        self.inputs = {
            'reset': 'signal'
        }
        self.outputs = {
            'field_image': 'image',        # The "World"
            'psi_external': 'signal',      # World perception signal
            'dopamine': 'signal',          # World stability (reward)
            'pain_stimulus': 'signal'      # World instability (pain)
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "World (No SciPy!)"
            return
            
        self.grid_size = int(grid_size)
        self.substrate_noise = float(substrate_noise)
        
        self.sim = WorldField(grid_size=self.grid_size, substrate_noise=self.substrate_noise)
        self.last_reset_sig = 0.0

    def randomize(self):
        if SCIPY_AVAILABLE:
            self.sim.initialize_field()

    def step(self):
        if not SCIPY_AVAILABLE: return

        reset_in = self.get_blended_input('reset', 'sum')
        if reset_in is not None and reset_in > 0.5 and self.last_reset_sig <= 0.5:
            self.randomize()
        self.last_reset_sig = reset_in or 0.0
        
        self.sim.step()
    
    def get_output(self, port_name):
        if port_name == 'field_image':
            phi_norm = (self.sim.phi - np.min(self.sim.phi)) / (np.max(self.sim.phi) - np.min(self.sim.phi) + 1e-9)
            return phi_norm.astype(np.float32)
            
        elif port_name == 'psi_external':
            return self.sim.psi_external_out
            
        elif port_name == 'dopamine':
            return self.sim.dopamine_out
            
        elif port_name == 'pain_stimulus':
            return self.sim.pain_out
            
        return None
    
    def get_display_image(self):
        field_data = self.get_output('field_image')
        if field_data is None: return None

        img_u8 = (np.clip(field_data, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        # Draw stability metric
        s = self.sim.stability_metric
        color = (0, 255 * s, 255 * (1-s)) # Green for stable, Red for unstable (BGR)
        cv2.rectangle(img_color, (5, 5), (self.sim.grid_size - 5, 15), color, -1)
        cv2.putText(img_color, f"Stab: {s:.2f}", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        # Draw pain metric
        p = self.sim.pain_out
        if p > 0.3:
             cv2.putText(img_color, f"PAIN!", (self.sim.grid_size//2 - 10, self.sim.grid_size//2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        img_thumb = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_thumb = np.ascontiguousarray(img_thumb)

        h, w = img_thumb.shape[:2]
        return QtGui.QImage(img_thumb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Grid Size (NxN)", "grid_size", self.grid_size, None),
            ("Substrate Noise", "substrate_noise", self.substrate_noise, None),
        ]