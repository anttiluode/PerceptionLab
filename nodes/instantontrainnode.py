"""
Instanton Train Node - Simulates topological solitons and quantum tunneling events
Models instantons as localized spacetime events that mediate vacuum transitions.

Based on instanton theory from QFT:
- Instantons are classical solutions to equations of motion in imaginary time
- They represent tunneling events between different vacuum states
- Have finite action and create a "train" of events in spacetime

Place this file in the 'nodes' folder
Requires: pip install scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import time

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: InstantonTrainNode requires 'scipy'.")


class Instanton:
    """
    Represents a single instanton event - a localized spacetime bubble.
    """
    def __init__(self, position, size, strength, vacuum_state):
        self.position = np.array(position, dtype=np.float32)  # (x, y, t)
        self.size = float(size)  # Instanton radius
        self.strength = float(strength)  # Action/coupling strength
        self.vacuum_state = int(vacuum_state)  # Which vacuum (0 or 1)
        self.age = 0.0
        self.lifetime = np.random.uniform(10, 30)  # How long it persists
        self.velocity = np.random.randn(2) * 0.1  # Drift velocity
        
    def profile(self, x, y):
        """
        Calculate instanton profile at position (x, y).
        Uses the standard instanton solution profile.
        """
        dx = x - self.position[0]
        dy = y - self.position[1]
        r_squared = dx**2 + dy**2
        
        # Standard instanton profile: ρ² / (r² + ρ²)
        # where ρ is the instanton size
        rho_squared = self.size**2
        profile = rho_squared / (r_squared + rho_squared)
        
        # Modulate by age (fade in/out)
        age_factor = 1.0
        if self.age < 5:
            age_factor = self.age / 5.0  # Fade in
        elif self.age > self.lifetime - 5:
            age_factor = (self.lifetime - self.age) / 5.0  # Fade out
        
        return profile * self.strength * age_factor
    
    def update(self, dt, grid_size):
        """Update instanton position and age."""
        self.age += dt
        
        # Drift in spacetime
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        
        # Wrap around boundaries
        self.position[0] %= grid_size[0]
        self.position[1] %= grid_size[1]
        
        return self.age < self.lifetime  # Return True if still alive


class InstantonTrainNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(150, 50, 150)  # Deep purple for quantum
    
    def __init__(self, grid_size=96, max_instantons=20):
        super().__init__()
        self.node_title = "Instanton Train"
        
        self.inputs = {
            'tunneling_rate': 'signal',      # Controls spawn rate
            'coupling_strength': 'signal',    # Controls instanton strength
            'vacuum_bias': 'signal',          # Bias toward vacuum 0 or 1
            'perturbation': 'image',          # External field perturbation
            'reset': 'signal'
        }
        
        self.outputs = {
            'vacuum_field': 'image',          # Current vacuum state field
            'action_density': 'image',        # Topological action density
            'tunneling_events': 'signal',     # Number of active instantons
            'winding_number': 'signal',       # Topological charge
            'vacuum_0_density': 'signal',     # Density in vacuum 0
            'vacuum_1_density': 'signal',     # Density in vacuum 1
            'average_action': 'signal'        # Average instanton action
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Instanton (No SciPy!)"
            return
        
        # Grid parameters
        self.grid_size = (int(grid_size), int(grid_size))
        self.max_instantons = int(max_instantons)
        
        # Physical parameters
        self.tunneling_rate = 0.1  # Base rate of instanton creation
        self.coupling_strength = 1.0
        self.vacuum_bias = 0.0  # -1 to 1, bias toward vacuum 0 or 1
        
        # State fields
        self.vacuum_field = np.zeros(self.grid_size, dtype=np.float32)  # -1 to 1
        self.action_density = np.zeros(self.grid_size, dtype=np.float32)
        
        # Instanton collection
        self.instantons = []
        
        # Metrics
        self.winding_number = 0.0
        self.vacuum_0_density = 0.5
        self.vacuum_1_density = 0.5
        self.average_action = 0.0
        
        # Time tracking
        self.time = 0.0
        self.last_spawn_time = 0.0
        self.dt = 0.1
        
        # Initialize with random vacuum configuration
        self._initialize_vacuum()
    
    def _initialize_vacuum(self):
        """Initialize the vacuum field with a smooth random configuration."""
        # Start with random noise
        noise = np.random.randn(*self.grid_size)
        # Smooth it to create domain structure
        self.vacuum_field = gaussian_filter(noise, sigma=5.0)
        # Normalize to [-1, 1]
        vmin, vmax = self.vacuum_field.min(), self.vacuum_field.max()
        if vmax > vmin:
            self.vacuum_field = 2.0 * (self.vacuum_field - vmin) / (vmax - vmin) - 1.0
    
    def _spawn_instanton(self):
        """Create a new instanton event."""
        if len(self.instantons) >= self.max_instantons:
            return
        
        # Random position
        position = np.array([
            np.random.uniform(0, self.grid_size[0]),
            np.random.uniform(0, self.grid_size[1]),
            self.time
        ])
        
        # Size varies (smaller = more localized, higher action)
        size = np.random.uniform(3.0, 8.0)
        
        # Strength proportional to coupling
        strength = self.coupling_strength * np.random.uniform(0.8, 1.2)
        
        # Vacuum state based on bias
        if np.random.random() < (self.vacuum_bias + 1.0) / 2.0:
            vacuum_state = 1
        else:
            vacuum_state = 0
        
        instanton = Instanton(position, size, strength, vacuum_state)
        self.instantons.append(instanton)
    
    def _update_instantons(self):
        """Update all instantons and remove dead ones."""
        alive_instantons = []
        
        for inst in self.instantons:
            if inst.update(self.dt, self.grid_size):
                alive_instantons.append(inst)
        
        self.instantons = alive_instantons
    
    def _compute_vacuum_field(self):
        """Compute the vacuum field from all active instantons."""
        # Start with the base field (slowly decays toward zero)
        self.vacuum_field *= 0.99
        
        # Add bias drift
        self.vacuum_field += self.vacuum_bias * 0.01
        
        # Create coordinate grids
        x = np.arange(self.grid_size[0])
        y = np.arange(self.grid_size[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Add contribution from each instanton
        for inst in self.instantons:
            profile = inst.profile(X, Y)
            
            # Instantons flip the vacuum locally
            if inst.vacuum_state == 1:
                self.vacuum_field += profile
            else:
                self.vacuum_field -= profile
        
        # Clamp to valid range
        self.vacuum_field = np.clip(self.vacuum_field, -1.0, 1.0)
    
    def _compute_action_density(self):
        """
        Compute the action density (topological charge density).
        This measures local field gradients - where tunneling is occurring.
        """
        # Calculate gradient magnitude
        grad_x = np.roll(self.vacuum_field, -1, axis=0) - np.roll(self.vacuum_field, 1, axis=0)
        grad_y = np.roll(self.vacuum_field, -1, axis=1) - np.roll(self.vacuum_field, 1, axis=1)
        
        # Action density ~ gradient squared (kinetic term)
        self.action_density = grad_x**2 + grad_y**2
        
        # Add potential term (double-well potential)
        # V(φ) = (φ² - 1)² has minima at φ = ±1 (two vacua)
        potential = (self.vacuum_field**2 - 1.0)**2
        self.action_density += potential * 0.5
        
        # Smooth for visualization
        self.action_density = gaussian_filter(self.action_density, sigma=1.0)
    
    def _compute_winding_number(self):
        """
        Compute topological winding number (topological charge).
        This counts the net number of vacuum transitions.
        """
        # Simple approximation: count domain walls
        # A domain wall is where the field crosses zero
        zero_crossings_x = np.sum(self.vacuum_field[:-1, :] * self.vacuum_field[1:, :] < 0)
        zero_crossings_y = np.sum(self.vacuum_field[:, :-1] * self.vacuum_field[:, 1:] < 0)
        
        # Winding number is proportional to number of crossings
        self.winding_number = (zero_crossings_x + zero_crossings_y) / 100.0
    
    def _compute_vacuum_densities(self):
        """Calculate the fraction of space in each vacuum."""
        # Vacuum 0 is where field < 0, Vacuum 1 is where field > 0
        self.vacuum_0_density = np.sum(self.vacuum_field < 0) / self.vacuum_field.size
        self.vacuum_1_density = np.sum(self.vacuum_field > 0) / self.vacuum_field.size
    
    def _compute_average_action(self):
        """Calculate average instanton action."""
        if len(self.instantons) > 0:
            total_action = sum(inst.strength * (inst.size**2) for inst in self.instantons)
            self.average_action = total_action / len(self.instantons)
        else:
            self.average_action = 0.0
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        # Get control inputs
        tunneling_in = self.get_blended_input('tunneling_rate', 'sum')
        coupling_in = self.get_blended_input('coupling_strength', 'sum')
        bias_in = self.get_blended_input('vacuum_bias', 'sum')
        perturbation = self.get_blended_input('perturbation', 'mean')
        reset_sig = self.get_blended_input('reset', 'sum')
        
        # Handle reset
        if reset_sig is not None and reset_sig > 0.5:
            self._reset()
            return
        
        # Update parameters from inputs
        if tunneling_in is not None:
            # Map [-1, 1] to [0, 0.5]
            self.tunneling_rate = (tunneling_in + 1.0) / 2.0 * 0.5
        
        if coupling_in is not None:
            # Map [-1, 1] to [0.5, 2.0]
            self.coupling_strength = 0.5 + (coupling_in + 1.0) / 2.0 * 1.5
        
        if bias_in is not None:
            # Direct mapping [-1, 1]
            self.vacuum_bias = np.clip(bias_in, -1.0, 1.0)
        
        # Apply external perturbation
        if perturbation is not None:
            perturb_resized = cv2.resize(perturbation, 
                                        (self.grid_size[1], self.grid_size[0]),
                                        interpolation=cv2.INTER_AREA)
            # Perturbation nudges the vacuum field
            self.vacuum_field += (perturb_resized - 0.5) * 0.1
            self.vacuum_field = np.clip(self.vacuum_field, -1.0, 1.0)
        
        # Decide whether to spawn a new instanton
        spawn_probability = self.tunneling_rate * self.dt
        if np.random.random() < spawn_probability:
            self._spawn_instanton()
        
        # Update all instantons
        self._update_instantons()
        
        # Compute the vacuum field
        self._compute_vacuum_field()
        
        # Compute action density
        self._compute_action_density()
        
        # Compute metrics
        self._compute_winding_number()
        self._compute_vacuum_densities()
        self._compute_average_action()
        
        # Advance time
        self.time += self.dt
    
    def _reset(self):
        """Reset the simulation."""
        self.instantons = []
        self._initialize_vacuum()
        self.action_density = np.zeros(self.grid_size, dtype=np.float32)
        self.time = 0.0
        self.winding_number = 0.0
    
    def get_output(self, port_name):
        if port_name == 'vacuum_field':
            # Normalize to [0, 1] for output
            return (self.vacuum_field + 1.0) / 2.0
        
        elif port_name == 'action_density':
            # Normalize action density
            if self.action_density.max() > 1e-9:
                return self.action_density / self.action_density.max()
            return self.action_density
        
        elif port_name == 'tunneling_events':
            return float(len(self.instantons))
        
        elif port_name == 'winding_number':
            return self.winding_number
        
        elif port_name == 'vacuum_0_density':
            return self.vacuum_0_density
        
        elif port_name == 'vacuum_1_density':
            return self.vacuum_1_density
        
        elif port_name == 'average_action':
            return self.average_action
        
        return None
    
    def get_display_image(self):
        # Create RGB visualization
        img = np.zeros((*self.grid_size, 3), dtype=np.float32)
        
        # Red channel: Vacuum 1 regions (positive field)
        img[:, :, 0] = np.clip(self.vacuum_field, 0, 1)
        
        # Blue channel: Vacuum 0 regions (negative field)
        img[:, :, 2] = np.clip(-self.vacuum_field, 0, 1)
        
        # Green channel: Action density (tunneling events)
        action_norm = self.action_density / (self.action_density.max() + 1e-9)
        img[:, :, 1] = action_norm * 0.8
        
        # Draw instanton centers
        for inst in self.instantons:
            x, y = int(inst.position[0]), int(inst.position[1])
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                # Bright spot at instanton center
                size = max(1, int(inst.size / 2))
                x_min, x_max = max(0, x-size), min(self.grid_size[0], x+size)
                y_min, y_max = max(0, y-size), min(self.grid_size[1], y+size)
                
                if inst.vacuum_state == 1:
                    img[x_min:x_max, y_min:y_max, 0] = 1.0  # Red for vacuum 1
                else:
                    img[x_min:x_max, y_min:y_max, 2] = 1.0  # Blue for vacuum 0
        
        # Convert to uint8
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Resize to thumbnail
        img_resized = cv2.resize(img_u8, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", self.grid_size[0], None),
            ("Max Instantons", "max_instantons", self.max_instantons, None),
        ]