"""
Fractal Quantum Gate Node - A Schrödinger-like wave simulator with fractal potential
and animated quantum gate operations (Hadamard, NOT, Entanglement).
Ported from nphard2.py (Schrödinger equation) and bmonsphere.py (Gates/Potential).
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
    print("Warning: FractalQuantumGateNode requires 'scipy'.")


class FractalQuantumGateNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(150, 100, 255)  # Purple/Violet for Quantum Gates
    
    def __init__(self, size=64, dt=0.05, potential_strength=1.5):
        super().__init__()
        self.node_title = "Fractal Quantum Gate"
        
        self.inputs = {
            'potential_strength': 'signal', # Control V_eff strength
            'damping': 'signal',          # Control wave decay
            'operation_trigger': 'signal' # Trigger a quantum operation
        }
        self.outputs = {
            'prob_density': 'image',      # |ψ|² (Probability)
            'phase_field': 'image',       # Phase (Angle)
            'current_operation': 'signal' # Shows if gate is active
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "FOG (No SciPy!)"
            return
            
        self.size = int(size)
        self.dt = float(dt)
        self.time = 0
        
        # Physics parameters
        self.hbar_eff = 1.0
        self.mass_eff = 1.0
        self.potential_strength = float(potential_strength)
        self.damping = 0.005
        
        # State grids
        self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
        self.potential = self._generate_fractal_potential()
        
        # Operation tracking
        self.operation = None # "hadamard", "x_gate", "entanglement"
        self.operation_step = 0
        self.total_steps = 30
        self.last_trigger_val = 0.0

        self._initialize_wave_packet()
    
    def _generate_fractal_potential(self):
        """Generate a static potential field (simplified version of source code)."""
        if not SCIPY_AVAILABLE:
            return np.zeros((self.size, self.size))

        potential = np.zeros((self.size, self.size))
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0
        
        yy, xx = np.mgrid[:self.size, :self.size]
        
        for i in range(octaves):
            freq = lacunarity ** i
            amp = persistence ** i
            
            # Use simple sin/cos modulation on position for pseudo-fractal structure
            noise_x = np.sin(xx / self.size * freq * 2 * np.pi)
            noise_y = np.cos(yy / self.size * freq * 2 * np.pi)
            noise_val = noise_x * noise_y

            potential += amp * noise_val
        
        # Normalize and smooth
        potential = (potential - np.min(potential)) / (np.max(potential) - np.min(potential) + 1e-9)
        return gaussian_filter(potential, sigma=1.0)
    
    def _initialize_wave_packet(self):
        """Initialize a Gaussian wave packet."""
        center = (self.size // 4, self.size // 4)
        sigma = self.size * 0.06
        kx, ky = 1.5, 1.0 # Base momentum
        
        y0, x0 = center
        yy, xx = np.mgrid[:self.size, :self.size]
        
        envelope = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (4 * sigma**2))
        phase = kx * (xx - x0) + ky * (yy - y0)
        self.psi = (envelope * np.exp(1j * phase)).astype(np.complex64)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 1e-9:
            self.psi /= norm

    def randomize(self):
        """Called by 'R' button - Re-initializes the wave packet and potential."""
        self.potential = self._generate_fractal_potential()
        self._initialize_wave_packet()
        self.operation = None
        self.operation_step = 0
        
    def _apply_gate(self, progress):
        """Simplified gate application (animation/interpolation)."""
        current_psi = self.psi.copy()
        
        if self.operation == "hadamard":
            # H: superposition, represented as splitting/reflection
            reflected_psi = np.roll(current_psi, self.size//2, axis=0) # Shift half way
            target_psi = (current_psi + reflected_psi)
            
        elif self.operation == "x_gate":
            # X: NOT gate, represented as vertical flip
            target_psi = np.flip(current_psi, axis=0)
            
        elif self.operation == "entanglement":
            # Entanglement: create correlation/diagonal structure
            correlated_psi = np.diag(np.ones(self.size)) + np.diag(np.ones(self.size-1), 1)
            correlated_psi = np.pad(correlated_psi, (0, self.size-correlated_psi.shape[0]), 'constant')[:self.size, :self.size] # Handle padding/truncation
            phase_pattern = np.exp(1j * np.pi * self.potential)
            target_psi = correlated_psi.astype(np.complex64) * phase_pattern
        else:
            return
            
        # Normalize target state
        norm_target = np.sqrt(np.sum(np.abs(target_psi)**2))
        if norm_target > 1e-9:
            target_psi /= norm_target
            
        # Interpolate
        self.psi = (1 - progress) * current_psi + progress * target_psi
        
        # Ensure final normalization
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 1e-9:
            self.psi /= norm
    
    def _update_dynamics(self):
        """Evolve the wave function using Schrödinger-like dynamics."""
        # Calculate Laplacian (Periodic boundaries are implicit with roll)
        lap_psi = (np.roll(self.psi, 1, axis=0) + np.roll(self.psi, -1, axis=0) +
                   np.roll(self.psi, 1, axis=1) + np.roll(self.psi, -1, axis=1) - 4 * self.psi)
        
        # Potential term (only using the static fractal potential V)
        V_eff = self.potential_strength * self.potential
        
        # Schrödinger-like evolution: i*dpsi/dt = H*psi -> dpsi = -i * H * dt
        H_psi = (-self.hbar_eff**2 / (2 * self.mass_eff) * lap_psi + V_eff * self.psi)
        
        # Euler update
        self.psi += (-1j / self.hbar_eff) * H_psi * self.dt
        
        # Apply damping
        self.psi *= (1 - self.damping * self.dt)
        
        # Re-normalize periodically
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 1e-9:
            self.psi /= norm

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # Get inputs
        pot_in = self.get_blended_input('potential_strength', 'sum')
        damp_in = self.get_blended_input('damping', 'sum')
        trigger_val = self.get_blended_input('operation_trigger', 'sum') or 0.0

        if pot_in is not None:
            self.potential_strength = np.clip(pot_in, 0.0, 5.0)
            
        if damp_in is not None:
            self.damping = np.clip(damp_in * 0.1, 0.001, 0.1) # Map to small range

        # --- Handle Gate Trigger ---
        if trigger_val > 0.5 and self.last_trigger_val <= 0.5:
            # Trigger detected (rising edge)
            if self.operation is None:
                # Cycle through gates
                gates = ["hadamard", "x_gate", "entanglement"]
                
                # Simple cycling logic based on current operation
                try:
                    current_idx = (gates.index(self.operation) + 1) if self.operation in gates else 0
                except ValueError:
                    current_idx = 0
                    
                self.operation = gates[current_idx]
                self.operation_step = 0
            
        self.last_trigger_val = trigger_val
        # --- End Gate Trigger ---

        if self.operation and self.operation_step < self.total_steps:
            # Operation in progress
            progress = self.operation_step / self.total_steps
            self._apply_gate(progress)
            self.operation_step += 1
            if self.operation_step >= self.total_steps:
                self.operation = None
        else:
            # Regular evolution
            self._update_dynamics()
        
        self.time += self.dt

    def get_output(self, port_name):
        if port_name == 'prob_density':
            # Output probability density: |ψ|²
            prob_density = np.abs(self.psi)**2
            max_val = np.max(prob_density)
            if max_val > 1e-9:
                return prob_density / max_val
            return prob_density
            
        elif port_name == 'phase_field':
            # Output normalized phase: [0, 1]
            phase = np.angle(self.psi)
            return (phase + np.pi) / (2 * np.pi)
            
        elif port_name == 'current_operation':
            # Output 1.0 if any gate is active
            return 1.0 if self.operation else 0.0
            
        return None
        
    def get_display_image(self):
        # Visualize probability density with phase color
        prob_density = np.abs(self.psi)**2
        phase = np.angle(self.psi)

        # Normalize amplitude and map phase to hue
        amp_norm = prob_density / (np.max(prob_density) + 1e-9)
        hue = ((np.angle(self.psi) + np.pi) / (2*np.pi) * 180).astype(np.uint8)
        sat = (amp_norm * 255).astype(np.uint8)
        val = (amp_norm * 255).astype(np.uint8)
        
        hsv = np.stack([hue, sat, val], axis=-1)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add operation indicator
        if self.operation:
            bar_color = (0, 0, 255) # Blue for Quantum
            if self.operation == 'hadamard': bar_color = (255, 165, 0) # Orange
            elif self.operation == 'x_gate': bar_color = (255, 0, 0) # Red
            elif self.operation == 'entanglement': bar_color = (0, 255, 0) # Green
            
            h, w = rgb.shape[:2]
            rgb[:3, :] = bar_color # Top status bar
            
        # Resize for display thumbnail (96x96)
        img_resized = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
            ("Timestep (dt)", "dt", self.dt, None),
            ("Potential Strength", "potential_strength", self.potential_strength, None),
            ("Gate Duration (steps)", "total_steps", self.total_steps, None),
        ]