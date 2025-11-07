"""
Quantum Wave Node - A PyTorch-based simulator for a 2D quantum wave function.
Implements the time-dependent Schrödinger equation (free particle).
Place this file in the 'nodes' folder
Requires: pip install torch numpy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import torch

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

# Use GPU if available
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")

# Simulation parameters (natural units: ℏ = 1, mass = 1)
LX, LY = 10.0, 10.0
DT = 1e-3  # Time step

class QuantumWaveNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 150, 255) # Complex Blue
    
    def __init__(self, resolution=128, k_momentum=5.0, steps_per_frame=10):
        super().__init__()
        self.node_title = "Quantum Wave"
        
        # Inputs allow external control over the simulation speed or initial state
        self.inputs = {
            'momentum_x': 'signal', # Control k0x
            'reset': 'signal'
        }
        self.outputs = {
            'image': 'image',        # Probability density |ψ|²
            'total_prob': 'signal'   # Should always be 1.0 (Normalization check)
        }
        
        self.Nx = self.Ny = int(resolution)
        self.k0x = float(k_momentum)
        self.k0y = 0.0
        self.steps_per_frame = int(steps_per_frame)
        
        self.dx = LX / self.Nx
        self.dy = LY / self.Ny
        
        # Internal state
        self.psi = None
        self.initialize_wavefunction()
        
    def normalize(self, psi):
        """Normalize the wavefunction (PyTorch version)"""
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) * self.dx * self.dy)
        if norm.item() > 1e-9:
            return psi / norm
        return psi # Return original if norm is zero/near-zero

    def laplacian(self, psi):
        """Precompute the Laplacian operator with periodic boundaries"""
        dx, dy = self.dx, self.dy
        
        psi_roll_x_forward = torch.roll(psi, shifts=-1, dims=0)
        psi_roll_x_backward = torch.roll(psi, shifts=1, dims=0)
        psi_roll_y_forward = torch.roll(psi, shifts=-1, dims=1)
        psi_roll_y_backward = torch.roll(psi, shifts=1, dims=1)
        
        lap = (psi_roll_x_forward + psi_roll_x_backward - 2*psi) / (dx**2) \
              + (psi_roll_y_forward + psi_roll_y_backward - 2*psi) / (dy**2)
        return lap

    def evolve(self, psi, dt):
        """Time evolution using the Euler method: ∂ψ/∂t = -i/2 ∇²ψ"""
        dpsi_dt = -1j * 0.5 * self.laplacian(psi)
        psi_new = psi + dpsi_dt * dt
        psi_new = self.normalize(psi_new)
        return psi_new

    def initialize_wavefunction(self):
        """Define the initial state (Gaussian wave packet with momentum)"""
        x = torch.linspace(-LX/2, LX/2, self.Nx, device=DEVICE)
        y = torch.linspace(-LY/2, LY/2, self.Ny, device=DEVICE)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        x0, y0 = 0.0, 0.0         # Center of the packet
        sigma = 1.0               # Width of the packet
        
        # Create a real-valued Gaussian envelope
        envelope = torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        
        # Add a complex phase for momentum
        phase = torch.exp(1j * (self.k0x * X + self.k0y * Y))
        psi0 = envelope * phase

        self.psi = self.normalize(psi0).type(torch.complex64)
        
    def randomize(self):
        """Called by 'R' button - restart simulation"""
        self.initialize_wavefunction()

    def step(self):
        # Update parameters from inputs
        mom_in = self.get_blended_input('momentum_x', 'sum')
        if mom_in is not None:
            # Map input signal [-1, 1] to a wide momentum range [-10, 10]
            new_k0x = mom_in * 10.0
            # If momentum changes significantly, reinitialize the wave
            if abs(new_k0x - self.k0x) > 1.0:
                 self.k0x = new_k0x
                 self.initialize_wavefunction()
            
        # Check for reset signal
        reset_sig = self.get_blended_input('reset', 'sum')
        if reset_sig is not None and reset_sig > 0.5:
            self.initialize_wavefunction()

        # Perform time steps
        for _ in range(self.steps_per_frame):
            self.psi = self.evolve(self.psi, DT)
            
        # Calculate output metric
        self.total_probability = torch.sum(torch.abs(self.psi)**2 * self.dx * self.dy).item()

    def get_output(self, port_name):
        if port_name == 'image':
            # Output probability density: |ψ|²
            prob_density_np = torch.abs(self.psi).pow(2).cpu().numpy()
            
            # Normalize to [0, 1]
            max_val = np.max(prob_density_np)
            if max_val > 1e-9:
                return prob_density_np / max_val
            return prob_density_np
            
        elif port_name == 'total_prob':
            # Should be ~1.0
            return self.total_probability
        return None
        
    def get_display_image(self):
        # Get the density image
        prob_density = self.get_output('image')
        if prob_density is None:
            return None
            
        # Resize for display thumbnail (64x64) and convert to RGB (viridis-like)
        img_u8 = (prob_density * 255).astype(np.uint8)
        
        # Apply colormap (viridis)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        # Resize to thumbnail size
        img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "resolution", self.Nx, None),
            ("Initial Momentum (k0x)", "k0x", self.k0x, None),
            ("Steps per Frame", "steps_per_frame", self.steps_per_frame, None),
        ]