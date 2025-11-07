"""
Topological Sieve Node - Simulates a Quantum Cellular Automaton performing a
Prime Number Sieve via topological annihilation (interference).

Outputs:
- Information Density (The computation state image).
- Prime Index (The current prime being tested).
- Final Result (A signal that goes high when computation is complete).
Ported from topological_prime_sieve.py.
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
    print("Warning: TopologicalSieveNode requires 'scipy'.")


# --- Simulation Constants (from source) ---
GRID_SIZE_X, GRID_SIZE_Y = 128, 128
SIEVE_ROWS, SIEVE_COLS = 10, 10
MAX_NUMBER = SIEVE_ROWS * SIEVE_COLS
PRIMES_TO_SIEVE = [2, 3, 5, 7] # Sieve for primes up to sqrt(100)
DT = 0.1  
DAMPING = 0.998


class TopologicalSieve:
    """Manages the dynamics of the ψ field within a lattice scaffold."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.psi = np.zeros((width, height), dtype=np.complex64)
        self.psi_prev = np.zeros((width, height), dtype=np.complex64)
        self.information_density = np.zeros((width, height), dtype=np.float32)
        
        self.lattice_locations, self.environmental_potential = self._create_lattice_scaffold(SIEVE_ROWS, SIEVE_COLS)
        self._initialize_atoms()
        
        self.frame = 0
        self.state = "SETTLING"
        self.prime_index_to_sieve = 0
        self.frames_since_last_action = 0
        self.last_trigger_val = 0.0

    def _create_lattice_scaffold(self, rows, cols):
        """Creates a grid of potential wells to represent numbers."""
        potential = np.zeros((self.width, self.height), dtype=np.float32)
        locations = {}
        
        spacing_x = self.width / (cols + 1)
        spacing_y = self.height / (rows + 1)

        for r in range(rows):
            for c in range(cols):
                number = r * cols + c + 1
                if number > MAX_NUMBER: continue
                
                cx = int((c + 1) * spacing_x)
                cy = int((r + 1) * spacing_y)
                locations[number] = (cx, cy)
                
                yy, xx = np.mgrid[0:self.height, 0:self.width]
                dist_sq = (xx - cx)**2 + (yy - cy)**2
                potential -= np.exp(-dist_sq / (spacing_x / 3)**2)

        return locations, gaussian_filter(potential, sigma=2.0)

    def _initialize_atoms(self):
        """Places a stable vortex-antivortex pair (an 'atom') in each well."""
        for number, (cx, cy) in self.lattice_locations.items():
            if number == 1: continue
            
            offset = 1 
            amplitude = 1.0 
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            
            dist_sq_p = (xx - (cx + offset))**2 + (yy - cy)**2
            self.psi += (amplitude * np.exp(-dist_sq_p / 5.0)).T.astype(np.complex64)
            
            dist_sq_n = (xx - (cx - offset))**2 + (yy - cy)**2
            self.psi -= (amplitude * np.exp(-dist_sq_n / 5.0)).T.astype(np.complex64)
        self.psi_prev = self.psi.copy()
            
    def launch_sieve_wave(self, prime):
        """Launches a destructive wave tuned to annihilate multiples of the prime."""
        amplitude = 0.25 
        
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        
        grid_spacing = self.width / (SIEVE_COLS + 1)
        k = 2 * np.pi / (prime * grid_spacing)
        
        wave = amplitude * (np.sin(k * xx) * np.sin(k * yy))
        
        self.psi += wave.T.astype(np.complex64)

    def evolve(self):
        """Evolve the field using non-linear dynamics."""
        self.frame += 1
        self.frames_since_last_action += 1
        
        # --- Field evolution physics ---
        laplacian = (np.roll(self.psi, 1, axis=0) + np.roll(self.psi, -1, axis=0) +
                     np.roll(self.psi, 1, axis=1) + np.roll(self.psi, -1, axis=1) - 4 * self.psi)
        
        psi_sq = np.abs(self.psi)**2
        nonlinear_term = self.psi * (psi_sq - 0.5)

        # Update equation (non-linear wave evolution)
        psi_next = (2 * self.psi - self.psi_prev * DAMPING +
                    DT**2 * (0.5 * laplacian - nonlinear_term) - 
                    0.1 * self.environmental_potential * self.psi)

        self.psi_prev, self.psi = self.psi, psi_next
        
        # Measure information density (gradient magnitude)
        grad_x, grad_y = np.gradient(np.abs(self.psi))
        self.information_density = grad_x**2 + grad_y**2


class TopologicalSieveNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(150, 100, 255) # Quantum Computing Purple
    
    def __init__(self, size=96):
        super().__init__()
        self.node_title = "Topological Sieve"
        
        self.inputs = {
            'prime_trigger': 'signal', # Trigger the next sieving step
            'reset': 'signal'
        }
        self.outputs = {
            'image': 'image',              # Information Density
            'prime_index': 'signal',       # Current prime being processed (2, 3, 5, 7, 0)
            'computation_done': 'signal'   # 1.0 when complete, 0.0 otherwise
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Sieve (No SciPy!)"
            return
            
        self.size = int(size)
        self.sim = TopologicalSieve(self.size, self.size)
        
        self.last_trigger_val = 0.0
        self.time_of_last_launch = 0.0
        self.is_done = False
        self.current_prime = 0.0

    def _execute_sieve_step(self):
        """Advances the state machine: settles, sieves, or finishes."""
        
        if self.sim.state == "SETTLING" and self.sim.frames_since_last_action > 50:
            self.sim.state = "SIEVING"
            self.sim.frames_since_last_action = 0
            
        if self.sim.state == "SIEVING":
            # Check if current launch has settled (gives the wave time to annihilate)
            if self.sim.frames_since_last_action > 100: 
                if self.sim.prime_index_to_sieve < len(PRIMES_TO_SIEVE):
                    prime = PRIMES_TO_SIEVE[self.sim.prime_index_to_sieve]
                    self.sim.launch_sieve_wave(prime)
                    self.sim.prime_index_to_sieve += 1
                    self.sim.frames_since_last_action = 0
                    self.current_prime = float(prime)
                else:
                    self.sim.state = "DONE"
                    self.is_done = True
                    self.current_prime = 0.0

    def randomize(self):
        """Resets the simulation to the initial atomic state."""
        if SCIPY_AVAILABLE:
            self.sim = TopologicalSieve(self.size, self.size)
            self.is_done = False
            self.current_prime = 0.0

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # 1. Handle Inputs
        trigger_val = self.get_blended_input('prime_trigger', 'sum') or 0.0
        reset_sig = self.get_blended_input('reset', 'sum')

        if reset_sig is not None and reset_sig > 0.5:
            self.randomize()
            return
            
        # 2. Manual Step Control (Rising edge)
        if trigger_val > 0.5 and self.last_trigger_val <= 0.5 and not self.is_done:
            self._execute_sieve_step()
            
        self.last_trigger_val = trigger_val

        # 3. Always Evolve the Physics
        self.sim.evolve()


    def get_output(self, port_name):
        if port_name == 'image':
            # Output Information Density
            max_val = np.max(self.sim.information_density)
            if max_val > 1e-9:
                return self.sim.information_density.T / max_val
            return self.sim.information_density.T
            
        elif port_name == 'prime_index':
            # Output the current prime being processed
            return self.current_prime 
            
        elif port_name == 'computation_done':
            return 1.0 if self.is_done else 0.0
            
        return None
        
    def get_display_image(self):
        # Visualize Information Density
        img_data = self.get_output('image')
        if img_data is None: return None
        
        img_u8 = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap (Hot for Information Density)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_HOT)
        
        # Add State overlay (Green for Settling, Red for Done)
        if self.sim.state == "DONE":
             cv2.rectangle(img_color, (0, 0), (self.size, 10), (0, 255, 0), -1) # Green bar
        elif self.sim.state == "SIEVING":
             cv2.rectangle(img_color, (0, 0), (self.size, 10), (255, 165, 0), -1) # Orange bar
             
        # Draw the labels for the surviving/annihilated atoms (complex, so skipping for now)
        
        # Resize to thumbnail size
        img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
        ]