"""
CirculatorSwarmNode

Simulates "bits" (Circulators) moving through the
Circulation Field. Implements particle advection and
collision/interaction.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class CirculatorSwarmNode(BaseNode):
    """
    Moves particles (Circulators) along an input vector field.
    """
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(220, 180, 100) # Gold

    def __init__(self, size=256, particle_count=300):
        super().__init__()
        self.node_title = "Circulator Swarm"
        
        self.inputs = {
            'vector_field_in': 'image', # From CirculationFieldNode
            'repulsion': 'signal',      # 0-1, strength of collisions
            'damping': 'signal'         # 0-1, how much to follow field
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.particle_count = int(particle_count)
        
        # Initialize particles
        self.positions = np.random.rand(self.particle_count, 2) * self.size
        self.velocities = (np.random.rand(self.particle_count, 2) - 0.5) * 2.0
        
        # Fading trail buffer
        self.trail_buffer = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def _prepare_field(self, img):
        """Helper to resize and format the vector field."""
        if img is None:
            return np.zeros((self.size, self.size, 2), dtype=np.float32)
        
        # Ensure float32
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0: # (Assumes 0-255 if not 0-1)
            img = img / 255.0
            
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Convert from [0, 1] (R,G) to [-1, 1] (vx, vy)
        vx = (img_resized[..., 0] * 2.0) - 1.0
        vy = (img_resized[..., 1] * 2.0) - 1.0
        
        return np.dstack([vx, vy])

    def step(self):
        # --- 1. Get Inputs ---
        vector_field = self._prepare_field(self.get_blended_input('vector_field_in', 'first'))
        repulsion = (self.get_blended_input('repulsion', 'sum') or 0.1) * 20.0
        damping = 1.0 - (self.get_blended_input('damping', 'sum') or 0.1) # 0.9 to 1.0
        
        # --- 2. Update Particle Velocities ---
        
        # a) Get field velocity at each particle's position
        int_pos = self.positions.astype(int)
        px = np.clip(int_pos[:, 0], 0, self.size - 1)
        py = np.clip(int_pos[:, 1], 0, self.size - 1)
        
        field_velocities = vector_field[py, px] # (N, 2) array
        
        # b) Apply damping (follow the field)
        self.velocities = self.velocities * damping + field_velocities * (1.0 - damping)
        
        # c) Apply collisions ("Interactions")
        if repulsion > 0:
            for i in range(self.particle_count):
                # Vectorized repulsion (broadcasting)
                diffs = self.positions[i] - self.positions
                dists_sq = np.sum(diffs**2, axis=1)
                
                # Avoid self-repulsion and divide-by-zero
                dists_sq[i] = np.inf 
                dists_sq[dists_sq < 1] = 1 # Min distance
                
                # Force = 1/r^2
                repel_force = repulsion * diffs / dists_sq[:, np.newaxis]
                
                # Sum forces from all other particles
                self.velocities[i] += np.sum(repel_force, axis=0)
        
        # Clamp velocity
        self.velocities = np.clip(self.velocities, -5.0, 5.0)
        
        # --- 3. Update Positions ---
        self.positions += self.velocities
        
        # Wrap around edges
        self.positions = self.positions % self.size
        
        # --- 4. Draw ---
        self.trail_buffer *= 0.85 # Fade trails
        
        int_pos = self.positions.astype(int)
        px = int_pos[:, 0]
        py = int_pos[:, 1]
        
        # Draw all particles
        self.trail_buffer[py, px] = 1.0

    def get_output(self, port_name):
        if port_name == 'image':
            return self.trail_buffer
        return None