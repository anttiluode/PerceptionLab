"""
Fractal Surfer Node - Simulates a consciousness "surfer" on a quantum field.
Logic ported from the user-provided fractal_surfer.html file.
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

# --- Internal classes based on fractal_surfer.html ---

class QuantumField:
    """Numpy implementation of the QuantumField class."""
    def __init__(self, size):
        self.size = size
        self.mu = np.zeros((size, size), dtype=np.float32)
        self.sigma = np.zeros((size, size), dtype=np.float32)
        self.collapsed = np.zeros((size, size), dtype=np.float32)
        self.reset()

    def reset(self):
        self.mu = (np.random.rand(self.size, self.size) - 0.5) * 0.2
        self.sigma = 0.8 + np.random.rand(self.size, self.size) * 0.4
        self.collapsed.fill(0.0)

    def _laplacian(self, field):
        """Compute the laplacian using np.roll for periodic boundaries."""
        return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4 * field)

    def evolve(self, rate):
        """Evolve the mu and sigma fields."""
        mu_lap = self._laplacian(self.mu)
        sigma_lap = self._laplacian(self.sigma)
        
        self.mu = self.mu + rate * mu_lap * 0.1
        self.mu *= 0.995 # Damping
        
        self.sigma = self.sigma + rate * sigma_lap * 0.02
        self.sigma *= 1.0002 # Entropy increase
        self.sigma = np.clip(self.sigma, 0.1, 2.0)
    
    def injectChaos(self):
        self.mu += (np.random.rand(self.size, self.size) - 0.5) * 0.5
        self.sigma += np.random.rand(self.size, self.size) * 0.3
        self.sigma = np.clip(self.sigma, 0.1, 2.0)

class FractalSurfer:
    """Numpy implementation of the FractalSurfer class."""
    def __init__(self, quantumField, search_radius):
        self.field = quantumField
        self.size = quantumField.size
        self.x = self.size / 2.0
        self.y = self.size / 2.0
        self.memory = 0.0
        self.sensation = 0.0
        self.collapseCount = 0
        self.search_radius = int(search_radius)

    def _gaussian_random(self, mu, sigma):
        """Box-Muller transform for Gaussian random numbers."""
        u, v = np.random.rand(2)
        z0 = np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)
        return z0 * sigma + mu

    def update(self, exploration, plasticity, feedback):
        x, y = int(self.x), int(self.y)
        
        # 1. Wave function collapse
        local_mu = self.field.mu[y, x]
        local_sigma = self.field.sigma[y, x]
        self.sensation = self._gaussian_random(local_mu, local_sigma)
        
        self.field.collapsed[y, x] = self.sensation
        self.collapseCount += 1
        
        # 2. Learning from experience
        learning_signal = np.abs(self.sensation)
        if learning_signal > 0.3:
            self.memory = (1 - plasticity) * self.memory + plasticity * learning_signal
        self.memory *= 0.999 # Memory decay
        
        # 3. Consciousness feedback (reduce uncertainty)
        uncertainty_reduction = self.memory * feedback
        self.field.sigma[y, x] = np.maximum(0.1, self.field.sigma[y, x] - uncertainty_reduction)
        
        # 4. Navigate
        self.navigate(exploration)

    def navigate(self, exploration_bias):
        """Find the best nearby location and move towards it."""
        cx, cy = int(self.x), int(self.y)
        r = self.search_radius
        
        # Create coordinates for the search area
        x_coords = np.arange(cx - r, cx + r + 1) % self.size
        y_coords = np.arange(cy - r, cy + r + 1) % self.size
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Get field values in the search area
        potential = self.field.mu[yy, xx]
        uncertainty = self.field.sigma[yy, xx]
        
        # Calculate distance penalty
        dx = (xx - cx + self.size/2) % self.size - self.size/2
        dy = (yy - cy + self.size/2) % self.size - self.size/2
        distance = np.sqrt(dx**2 + dy**2)
        
        # Score = weighted combo of potential, uncertainty, and distance
        score = ( (1 - exploration_bias) * potential + 
                  exploration_bias * uncertainty -
                  distance * 0.01 )
        
        # Find the best location
        best_idx = np.unravel_index(np.argmax(score), score.shape)
        bestX, bestY = x_coords[best_idx[1]], y_coords[best_idx[0]]
        
        # Move towards best location
        smoothing = 0.15
        self.x = (1 - smoothing) * self.x + smoothing * bestX
        self.y = (1 - smoothing) * self.y + smoothing * bestY
        
    def getCoherence(self):
        avg_uncertainty = np.mean(self.field.sigma)
        return np.maximum(0, 1 - avg_uncertainty / 2.0)

# --- The Node ---

class FractalSurferNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(60, 180, 160) # A generative teal
    
    def __init__(self, grid_size=64, search_radius=8):
        super().__init__()
        self.node_title = "Fractal Surfer"
        
        self.inputs = {
            'energy_in': 'signal',
            'exploration_in': 'signal',
            'plasticity_in': 'signal'
        }
        self.outputs = {
            'quantum_sea': 'image',
            'reality': 'image',
            'coherence': 'signal',
            'surfer_x': 'signal',
            'surfer_y': 'signal'
        }
        
        self.size = int(grid_size)
        self.search_radius = int(search_radius)
        
        # Initialize simulation state
        self.field = QuantumField(self.size)
        self.surfer = FractalSurfer(self.field, self.search_radius)
        
        self.feedback_strength = 0.1 # From original script
        
        self.display_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def step(self):
        # Get control signals
        evolution_rate = self.get_blended_input('energy_in', 'sum') or 0.0
        exploration = (self.get_blended_input('exploration_in', 'sum') or 0.0 + 1.0) / 2.0 # Map [-1,1] to [0,1]
        plasticity = (self.get_blended_input('plasticity_in', 'sum') or 0.0 + 1.0) / 2.0 # Map [-1,1] to [0,1]
        
        # Clamp plasticity to valid range
        plasticity = np.clip(plasticity * 0.1, 0.001, 0.1) 
        
        # Only evolve if energy is positive
        if evolution_rate > 0.0:
            self.field.evolve(evolution_rate)
        
        self.surfer.update(exploration, plasticity, self.feedback_strength)
        
        # Update the display image
        self._render_quantum_field()

    def _render_quantum_field(self):
        """Internal render function for quantum sea."""
        # Map mu (potential) to red
        potential = np.clip((self.field.mu + 1.0) / 2.0, 0, 1)
        # Map sigma (uncertainty) to green
        uncertainty = np.clip(self.field.sigma / 2.0, 0, 1)
        # Blue channel
        blue = np.clip((1 - uncertainty) * 0.5 + potential * 0.5, 0, 1)
        
        self.display_img[:,:,0] = (potential * 255).astype(np.uint8) # Red
        self.display_img[:,:,1] = (uncertainty * 255).astype(np.uint8) # Green
        self.display_img[:,:,2] = (blue * 255).astype(np.uint8) # Blue
        
        # Draw the surfer
        sx, sy = int(self.surfer.x), int(self.surfer.y)
        cv2.circle(self.display_img, (sx, sy), 2, (255, 255, 255), -1)

    def get_output(self, port_name):
        if port_name == 'quantum_sea':
            return self.display_img.astype(np.float32) / 255.0
        elif port_name == 'reality':
            return self.field.collapsed # Already [0,1]
        elif port_name == 'coherence':
            return self.surfer.getCoherence()
        elif port_name == 'surfer_x':
            return (self.surfer.x / self.size) * 2.0 - 1.0 # Map to [-1, 1]
        elif port_name == 'surfer_y':
            return (self.surfer.y / self.size) * 2.0 - 1.0 # Map to [-1, 1]
        return None
        
    def get_display_image(self):
        rgb = np.ascontiguousarray(self.display_img)
        h, w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def randomize(self):
        """Called by 'R' button, injects chaos"""
        self.field.injectChaos()

    def get_config_options(self):
        return [
            ("Grid Size", "size", self.size, None),
            ("Search Radius", "search_radius", self.search_radius, None),
        ]