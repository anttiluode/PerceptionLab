"""
Emergent Reality Node - Simulates "Reality as a Living Computation"
Ported from live.py. Models emergent physics (mass, energy, spacetime speed)
from iterative non-linear wave computations.

Outputs key fields (Intensity, Processing Speed) as images and global
metrics (Energy, Curvature) as signals.
Requires: pip install numpy scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import sys
import os
import random
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import fft2, ifft2, fftfreq
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: EmergentRealityNode requires 'scipy'.")


# --- Core Simulation Classes (from live.py) ---

class RealitySimulator:
    def __init__(self, size=64, dt=0.005, c0=1.0, domain_size=10.0):
        self.size = size
        self.dt = dt
        self.c0 = c0  # Base processing speed
        self.domain_size = domain_size
        
        self.x = np.linspace(-domain_size, domain_size, size)
        self.y = np.linspace(-domain_size, domain_size, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        kx = fftfreq(size, d=(self.x[1] - self.x[0])) * 2 * np.pi
        ky = fftfreq(size, d=(self.y[1] - self.y[0])) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K_squared = self.KX**2 + self.KY**2
        
        self.phi = np.zeros((size, size), dtype=complex)
        self.phi_prev = self.phi.copy() # For better stability
        
        # Physics parameters (simplified from live.py)
        self.alpha_quantum = 0.01
        self.alpha_gravity = 2.0
        self.current_alpha = self.alpha_gravity # Start in a stable regime
        
        self.a = 0.8   # Linear coefficient
        self.b = 0.05  # Nonlinear coefficient
        self.damping = 0.001
        
        self.time = 0
        self.step_count = 0
        
        # Initial seeding
        self.create_initial_state()
        
    def create_initial_state(self):
        """Seed the field with a couple of stable structures"""
        self.phi.fill(0)
        self.create_particle_cluster(center_x=-2, center_y=0, num_particles=3)
        self.create_massive_object(x_pos=2, y_pos=0, mass=5.0)
        self.add_quantum_foam(strength=0.1)

    def effective_speed_squared(self):
        """c²_eff = c₀² / (1 + α|Φ|²). Emergent spacetime metric."""
        phi_intensity = np.abs(self.phi)**2
        return self.c0**2 / (1 + self.current_alpha * phi_intensity)
    
    def create_particle_cluster(self, center_x=0, center_y=0, num_particles=3, spread=1.0, amplitude=1.5):
        """Create particle-like solitons (simplified)"""
        for i in range(num_particles):
            angle = 2 * np.pi * i / num_particles + random.random() * 0.5
            r = spread * random.random()
            x_pos = center_x + r * np.cos(angle)
            y_pos = center_y + r * np.sin(angle)
            
            r_from_center = np.sqrt((self.X - x_pos)**2 + (self.Y - y_pos)**2)
            envelope = amplitude * np.exp(-r_from_center**2 / 1.0)
            
            particle = envelope * np.exp(1j * 0.5 * (self.X - x_pos))
            self.phi += particle
            
    def create_massive_object(self, x_pos=0, y_pos=0, mass=5.0, width=3.0):
        """Create a massive object that warps spacetime significantly"""
        r_from_center = np.sqrt((self.X - x_pos)**2 + (self.Y - y_pos)**2)
        envelope = mass * np.exp(-r_from_center**2 / (2 * width**2))
        
        theta = np.arctan2(self.Y - y_pos, self.X - x_pos)
        spiral_phase = 0.2 * theta
        
        massive_object = envelope * np.exp(1j * spiral_phase)
        self.phi += massive_object

    def add_quantum_foam(self, strength=0.05):
        """Add continuous random fluctuations (simplified noise)"""
        if strength > 0.0:
            noise_real = np.random.randn(self.size, self.size) * strength
            self.phi += noise_real
    
    def wave_equation_step(self):
        """The core processing step (modified Klein-Gordon/Non-linear Schrödinger)"""
        
        # 1. Compute Derivatives
        phi_fft = fft2(self.phi)
        laplacian_fft = -self.K_squared * phi_fft
        laplacian = ifft2(laplacian_fft)
        
        # 2. Get Effective Speed and Nonlinear Terms
        c_eff_squared = self.effective_speed_squared()
        nonlinear_term = self.a * self.phi - self.b * np.abs(self.phi)**2 * self.phi
        damping_term = -self.damping * self.phi
        
        # 3. Time Evolution (Implicit in the formula, based on live.py)
        phi_new = (self.phi + 
                  self.dt * c_eff_squared * laplacian + 
                  self.dt * nonlinear_term +
                  self.dt * damping_term)
        
        # Update field and step count
        self.phi_prev = self.phi.copy()
        self.phi = phi_new
        self.time += self.dt
        self.step_count += 1
        
        # Simple re-normalization to prevent full collapse/blow-up
        self.phi *= 0.999 # Slight decay helps stability

    def measure_energy(self):
        """Measure total field energy (Approximation)"""
        return np.sum(np.abs(self.phi)**2)
    
    def measure_spacetime_curvature(self):
        """Measure the variation in processing speed (Spacetime Curvature)"""
        c_eff = np.sqrt(self.effective_speed_squared())
        mean_c = np.mean(c_eff)
        if mean_c < 1e-9: return 0.0
        return np.std(c_eff) / mean_c # Curvature is fractional change


class EmergentRealityNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(255, 150, 50) # Orange for Emergent Physics
    
    def __init__(self, resolution=64, alpha_resistence=2.0, steps_per_frame=5):
        super().__init__()
        self.node_title = "Emergent Reality"
        
        self.inputs = {
            'alpha_control': 'signal', # Controls the key Alpha parameter
            'reset': 'signal'
        }
        self.outputs = {
            'intensity': 'image',        # Matter/Energy Density |Φ|²
            'speed_of_light': 'image',   # Processing Speed c_eff
            'total_energy': 'signal',    # Global Energy Metric
            'curvature': 'signal',       # Global Curvature Metric
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Reality (No SciPy!)"
            return
            
        self.resolution = int(resolution)
        self.current_alpha = float(alpha_resistence)
        self.steps_per_frame = int(steps_per_frame)
        
        # Initialize simulation
        self.sim = RealitySimulator(size=self.resolution, dt=0.005, c0=1.0)
        self.sim.current_alpha = self.current_alpha
        
        # Outputs
        self.intensity_data = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.speed_data = self.intensity_data.copy()
        self.energy_value = 0.0
        self.curvature_value = 0.0

    def randomize(self):
        """Called by 'R' button - reset/reseed the universe"""
        if SCIPY_AVAILABLE:
            self.sim.create_initial_state()

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # 1. Update control parameter
        alpha_in = self.get_blended_input('alpha_control', 'sum')
        if alpha_in is not None:
            # Map signal [-1, 1] to alpha resistance [0.01, 5.0]
            self.current_alpha = np.clip((alpha_in + 1.0) / 2.0 * 5.0, 0.01, 5.0)
            self.sim.current_alpha = self.current_alpha
            
        # 2. Check for reset
        reset_sig = self.get_blended_input('reset', 'sum')
        if reset_sig is not None and reset_sig > 0.5:
            self.randomize()

        # 3. Run simulation steps
        for _ in range(self.steps_per_frame):
            self.sim.wave_equation_step()
            
        # 4. Generate outputs
        self.energy_value = self.sim.measure_energy()
        self.curvature_value = self.sim.measure_spacetime_curvature()
        
        intensity_raw = np.abs(self.sim.phi)**2
        speed_raw = np.sqrt(self.sim.effective_speed_squared())
        
        # Normalize intensity for image output [0, 1]
        max_i = np.max(intensity_raw)
        self.intensity_data = intensity_raw / (max_i + 1e-9)
        
        # Normalize speed (c_eff) for image output [0, 1]
        min_c, max_c = np.min(speed_raw), np.max(speed_raw)
        range_c = max_c - min_c
        self.speed_data = (speed_raw - min_c) / (range_c + 1e-9)
        

    def get_output(self, port_name):
        if port_name == 'intensity':
            return self.intensity_data
        elif port_name == 'speed_of_light':
            return self.speed_data
        elif port_name == 'total_energy':
            # Scale energy to a manageable signal range (e.g., 0-10)
            return np.clip(self.energy_value / 5000.0, 0.0, 10.0) 
        elif port_name == 'curvature':
            # Curvature is already fractional (0-1)
            return np.clip(self.curvature_value * 10.0, 0.0, 1.0) # Scale up to 0-1
        return None
        
    def get_display_image(self):
        # Visualize Intensity data (Matter Density)
        img_u8 = (np.clip(self.intensity_data, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap (Hot for intensity)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_HOT)
        
        # Add Curvature bar at bottom
        bar_h = 5
        curvature_color = int(np.clip(self.curvature_value * 255 * 10, 0, 255))
        img_color[-bar_h:, :] = [curvature_color, curvature_color, 0] # Yellowish bar
        
        # Resize to thumbnail size
        img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "resolution", self.resolution, None),
            ("Initial Alpha (α)", "alpha_resistence", self.current_alpha, None),
            ("Steps per Frame", "steps_per_frame", self.steps_per_frame, None),
        ]