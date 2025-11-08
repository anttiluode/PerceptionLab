"""
Topological Atom Node - Simulates a field configuration (atom) with resonant shell
structure and allows for rotational manipulation (phase twist) to test topological 
protection against substrate noise.

Ported from instantonassim x.py
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
    print("Warning: TopologicalAtomNode requires 'scipy'.")


# --- Core Physics Engine (from instantonassim x.py) ---
class ResonantInstantonModel:
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
        self.instanton_events = []
        self.stability_metric = 1.0
        self.topological_charge = 0.0 # New metric
        self.current_rotation = 0.0   # Current angle of the structure
        
        self.time = 0
        self.frame_count = 0
        
        self.initialize_atom(atomic_number=6, stable_isotope=True) # Default to stable Carbon

    def initialize_atom(self, atomic_number, position=None, stable_isotope=True):
        if position is None: position = (self.grid_size // 2, self.grid_size // 2)
        self.phi = np.zeros((self.grid_size, self.grid_size))
        self.phi_prev = np.zeros((self.grid_size, self.grid_size))
        self.instanton_events = []
        self.stability_metric = 1.0
        
        core_radius = 4 + np.log(1 + atomic_number)
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        r = np.sqrt((x - position[0])**2 + (y - position[1])**2)
        core_amplitude = 1.0 + 0.2 * atomic_number
        
        # Create nuclear core
        self.phi = core_amplitude * np.exp(-r**2 / (2 * core_radius**2))
        
        # Add shells based on simplified quantum numbers (standing waves)
        shell_config = self._calculate_shell_configuration(atomic_number)
        for shell, electrons in enumerate(shell_config):
            if electrons > 0:
                shell_radius = self._shell_radius(shell + 1)
                shell_amplitude = 0.3 * (electrons / (2*(2*shell+1)**2))
                shell_wave = shell_amplitude * np.cos(np.pi * r / shell_radius)**2 * (r < 2*shell_radius)
                self.phi += shell_wave
        
        if not stable_isotope:
            asymmetry = 0.1 * np.sin(3 * np.arctan2(y - position[1], x - position[0]))
            self.phi += asymmetry * np.exp(-r**2 / (2 * core_radius**2))
            self.stability_metric = 0.7 + 0.3 * np.random.random()
        
        self.phi_prev = self.phi.copy()
        self.time = 0
        self.frame_count = 0

    def _calculate_shell_configuration(self, atomic_number):
        shell_capacity = [2, 8, 18, 32]
        shells = []
        electrons_left = atomic_number
        for capacity in shell_capacity:
            if electrons_left >= capacity:
                shells.append(capacity); electrons_left -= capacity
            else:
                shells.append(electrons_left); electrons_left = 0; break
        while electrons_left > 0:
            next_capacity = 2 * (len(shells) + 1)**2
            if electrons_left >= next_capacity:
                shells.append(next_capacity); electrons_left -= next_capacity
            else:
                shells.append(electrons_left); electrons_left = 0
        return shells
    
    def _shell_radius(self, n):
        base_radius = 8
        return base_radius * n**2
    
    def _laplacian(self, field):
        field_padded = np.pad(field, 1, mode='wrap')
        laplacian = (field_padded[:-2, 1:-1] + field_padded[2:, 1:-1] + 
                     field_padded[1:-1, :-2] + field_padded[1:-1, 2:] - 
                     4 * field_padded[1:-1, 1:-1])
        return laplacian
    
    def _detect_instanton_event(self, phi_old, phi_new):
        delta_phi = phi_new - phi_old
        delta_phi_smoothed = gaussian_filter(delta_phi, sigma=1.0)
        threshold = 0.1 * np.max(np.abs(self.phi))
        significant_changes = np.abs(delta_phi_smoothed) > threshold
        
        if np.any(significant_changes):
            self.instanton_events.append({'time': self.time, 'magnitude': np.max(np.abs(delta_phi_smoothed))})
            return True
        return False
    
    def _update_stability(self):
        recent_count = sum(1 for event in self.instanton_events 
                           if event['time'] > self.time - 100 * self.dt)
        if recent_count > 5: self.stability_metric -= 0.01
        else: self.stability_metric = min(1.0, self.stability_metric + 0.001)
        self.stability_metric = max(0.0, min(1.0, self.stability_metric))

    def rotate_field(self, angle_rad):
        """
        Applies a rotation (phase twist) to the current field configuration.
        This is the test for topological protection.
        """
        if abs(angle_rad) < 1e-6: return

        center = self.grid_size // 2
        
        # 1. Define the rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 2. Get coordinates relative to center
        y, x = np.mgrid[:self.grid_size, :self.grid_size]
        x_c = x - center
        y_c = y - center

        # 3. Apply rotation to coordinates
        x_rot = x_c * cos_a - y_c * sin_a
        y_rot = x_c * sin_a + y_c * cos_a
        
        # 4. Map rotated coordinates back to grid indices
        x_rot_idx = np.clip(np.round(x_rot + center).astype(int), 0, self.grid_size - 1)
        y_rot_idx = np.clip(np.round(y_rot + center).astype(int), 0, self.grid_size - 1)

        # 5. Create a new field by sampling the old one at rotated positions
        phi_rotated = self.phi[y_rot_idx, x_rot_idx]
        
        # Update current field and record rotation
        self.phi = phi_rotated
        self.phi_prev = phi_rotated # Ensure stability after rotation
        self.current_rotation = (self.current_rotation + angle_rad) % (2 * np.pi)

    def compute_topological_charge(self):
        """
        Computes the topological charge (winding number) of the structure.
        For a purely radial field, this is near zero. For a vortex, it's non-zero.
        We simplify: Charge = Mean gradient magnitude divided by stability.
        """
        grad_mag = np.mean(np.abs(np.gradient(self.phi)))
        # Scale and use stability as a denominator (more stable = lower perceived charge)
        self.topological_charge = (grad_mag * 100) / (self.stability_metric + 0.1)

    def step(self):
        # Save current field
        phi_old = self.phi.copy()
        
        # Compute field evolution terms
        laplacian_phi = self._laplacian(self.phi)
        
        # Add substrate noise (decoherence force)
        noise = self.substrate_noise * np.random.normal(size=self.phi.shape)
        
        # Field equation (Simplified wave equation)
        accel = (self.c**2 * laplacian_phi + 
                 self.a * self.phi - 
                 self.b * self.phi**3 + 
                 noise)
        
        # Update field using velocity Verlet integration
        phi_new = 2 * self.phi - self.phi_prev + self.dt**2 * accel
        
        # Update field state
        self.phi_prev = self.phi
        self.phi = phi_new
        
        # Detect instanton events
        self._detect_instanton_event(phi_old, self.phi)
        
        # Update stability metric and charge
        self._update_stability()
        self.compute_topological_charge()
        
        self.time += self.dt
        self.frame_count += 1
        return self.stability_metric


class TopologicalAtomNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 50, 200) # Deep Quantum Purple
    
    def __init__(self, atomic_number=6, stable=True, rotation_speed=0.0):
        super().__init__()
        self.node_title = "Topological Atom"
        
        self.inputs = {
            'noise_strength': 'signal',   # Substrate noise (decoherence)
            'rotation_rate': 'signal',    # External rotation force
            'reset': 'signal'
        }
        self.outputs = {
            'field_image': 'image',
            'stability': 'signal',        # Stability Metric [0, 1]
            'charge': 'signal',           # Topological Charge
            'rotation_angle': 'signal'    # Current rotation angle [-1, 1]
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Atom (No SciPy!)"
            return
            
        self.atomic_number = int(atomic_number)
        self.stable = bool(stable)
        self.rotation_speed_base = float(rotation_speed)
        
        self.sim = ResonantInstantonModel(grid_size=96, substrate_noise=0.0005)
        self.sim.initialize_atom(self.atomic_number, stable_isotope=self.stable)
        self.last_reset_sig = 0.0

    def randomize(self):
        self.sim.initialize_atom(self.atomic_number, stable_isotope=self.stable)

    def step(self):
        if not SCIPY_AVAILABLE: return

        # 1. Handle Inputs
        noise_in = self.get_blended_input('noise_strength', 'sum')
        rotation_in = self.get_blended_input('rotation_rate', 'sum')
        reset_in = self.get_blended_input('reset', 'sum')

        # Update noise (decoherence)
        if noise_in is not None:
            self.sim.substrate_noise = np.clip(noise_in * 0.01, 0.0001, 0.01)

        # Update rotation
        rotation_rate = self.rotation_speed_base + (rotation_in * 0.1) if rotation_in is not None else self.rotation_speed_base
        self.sim.rotate_field(rotation_rate * self.sim.dt)

        # Handle reset
        if reset_in is not None and reset_in > 0.5 and self.last_reset_sig <= 0.5:
            self.randomize()
        self.last_reset_sig = reset_in or 0.0
        
        # 2. Evolve simulation
        self.sim.step()
    
    def get_output(self, port_name):
        if port_name == 'field_image':
            # Normalize field output for display
            phi = self.sim.phi
            v_abs = np.max(np.abs(phi))
            return np.clip(phi / (v_abs + 1e-9), -1.0, 1.0)
            
        elif port_name == 'stability':
            return self.sim.stability_metric
            
        elif port_name == 'charge':
            return self.sim.topological_charge
            
        elif port_name == 'rotation_angle':
            # Normalize angle [0, 2pi] to signal [-1, 1]
            return (self.sim.current_rotation / (2 * np.pi)) * 2.0 - 1.0
            
        return None
    
    def get_display_image(self):
        # Render the field configuration (Field amplitude)
        field_data = self.get_output('field_image')
        if field_data is None: return None

        # Map [-1, 1] data to Red/Blue color map
        img_rgb = np.zeros((*field_data.shape, 3), dtype=np.uint8)
        
        # Red: Positive field (Vacuum 1); Blue: Negative field (Vacuum 0)
        img_rgb[:, :, 0] = np.clip(field_data * 255, 0, 255) # Red channel (positive part)
        img_rgb[:, :, 2] = np.clip(-field_data * 255, 0, 255) # Blue channel (negative part)
        
        # Draw stability metric on top
        s = self.sim.stability_metric
        color = (255 * (1-s), 255 * s, 0) # Green for stable, Red for unstable (BGR)
        cv2.rectangle(img_rgb, (5, 5), (self.sim.grid_size - 5, 15), color, -1)
        
        # Resize to thumbnail
        img_thumb = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_thumb = np.ascontiguousarray(img_thumb)

        h, w = img_thumb.shape[:2]
        return QtGui.QImage(img_thumb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Atomic Number (Z)", "atomic_number", self.atomic_number, None),
            ("Stable Isotope?", "stable", self.stable, [(True, True), (False, False)]),
            ("Base Rot. Speed", "rotation_speed", self.rotation_speed_base, None),
        ]