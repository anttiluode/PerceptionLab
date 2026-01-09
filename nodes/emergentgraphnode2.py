import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name): return None

class EmergentGraphNode2(BaseNode):
    """
    Emergent Graph Node
    -------------------
    No fixed grid - particles float in continuous space.
    
    Physics:
    - Gravity of meaning: co-active particles pull together
    - Dark energy: inactive particles drift apart
    - Geometry emerges from correlation structure
    
    Watch galaxies of related concepts form,
    connected by filaments of co-activation.
    
    This is geometry learning itself - spacetime as emergent phenomenon.
    """
    NODE_CATEGORY = "Physics"
    NODE_COLOR = QtGui.QColor(20, 20, 80)  # Dark blue (space)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Emergent Graph"
        
        self.inputs = {
            'activation': 'spectrum',
            'reset': 'signal',
        }
        
        self.outputs = {
            'universe_view': 'image',
            'density': 'signal',
            'cluster_count': 'signal',
            'positions': 'spectrum',  # Flattened positions for downstream
        }
        
        self.config = {
            'n_particles': 128,
            'gravity': 0.015,
            'dark_energy': 0.002,
            'interaction_radius': 0.3,
            'damping': 0.95,
            'canvas_size': 400,
        }
        
        self._output_values = {}
        self._init_universe()

    def _init_universe(self):
        n = self.config['n_particles']
        
        # Positions (0 to 1 in each dimension)
        self.positions = np.random.rand(n, 2).astype(np.float32)
        
        # Velocities
        self.velocities = np.zeros((n, 2), dtype=np.float32)
        
        # Activations (0 to 1)
        self.activations = np.zeros(n, dtype=np.float32)
        
        # Connection weights (Hebbian)
        self.weights = np.zeros((n, n), dtype=np.float32)

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'):
            return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value):
        self._output_values[name] = value
    
    def get_output(self, name):
        return self._output_values.get(name, None)

    def step(self):
        reset = self.get_input('reset')
        if reset is not None and reset > 0.5:
            self._init_universe()
            return
        
        n = self.config['n_particles']
        
        # Get external activation
        act_in = self.get_input('activation')
        if act_in is not None:
            act_vec = np.array(act_in, dtype=np.float32).flatten()
            # Map input to particles
            if len(act_vec) > 0:
                # Interpolate to particle count
                indices = np.linspace(0, len(act_vec) - 1, n).astype(int)
                external_act = np.abs(act_vec[indices])
                external_act /= np.max(external_act) + 1e-10
                
                # Blend with existing activation
                self.activations = self.activations * 0.7 + external_act * 0.3
        
        # Decay activations
        self.activations *= 0.95
        
        # === PHYSICS ===
        gravity = self.config['gravity']
        dark_e = self.config['dark_energy']
        radius = self.config['interaction_radius']
        
        # Compute pairwise distances
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-10)
        
        # Direction vectors (normalized)
        direction = diff / (dist[:, :, np.newaxis] + 1e-10)
        
        # Interaction mask
        interact_mask = (dist < radius) & (dist > 0.01)
        
        # Co-activation (product of activations)
        coact = self.activations[:, np.newaxis] * self.activations[np.newaxis, :]
        
        # === GRAVITY OF MEANING ===
        # Pull together particles that are co-active
        attraction = gravity * coact * interact_mask
        
        # === DARK ENERGY ===
        # Push apart particles that are not co-active
        repulsion = dark_e * (1 - coact) * interact_mask / (dist + 0.1)
        
        # Net force
        force_magnitude = attraction - repulsion
        forces = np.sum(force_magnitude[:, :, np.newaxis] * (-direction), axis=1)
        
        # === HEBBIAN LEARNING ===
        # Strengthen connections between co-active, nearby particles
        hebbian_update = coact * interact_mask * 0.01
        self.weights += hebbian_update
        self.weights *= 0.99  # Decay
        np.clip(self.weights, 0, 1, out=self.weights)
        
        # Connection-based attraction (learned structure)
        learned_attraction = self.weights * interact_mask * gravity * 0.5
        learned_forces = np.sum(learned_attraction[:, :, np.newaxis] * (-direction), axis=1)
        
        # Total force
        total_force = forces + learned_forces
        
        # Update velocities and positions
        damping = self.config['damping']
        self.velocities = self.velocities * damping + total_force
        self.positions += self.velocities
        
        # Boundary conditions (wrap around)
        self.positions = np.mod(self.positions, 1.0)
        
        # === OUTPUTS ===
        
        # Density (average distance to nearest neighbors)
        sorted_dist = np.sort(dist, axis=1)
        avg_nn_dist = np.mean(sorted_dist[:, 1:4])  # 3 nearest neighbors
        density = float(1.0 / (avg_nn_dist + 0.01))
        self.set_output('density', np.clip(density / 20, 0, 1))
        
        # Cluster count (simple: count groups where dist < threshold)
        cluster_threshold = 0.1
        visited = np.zeros(n, dtype=bool)
        clusters = 0
        for i in range(n):
            if not visited[i] and self.activations[i] > 0.2:
                clusters += 1
                # BFS to mark cluster
                stack = [i]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    neighbors = np.where((dist[node] < cluster_threshold) & 
                                        (self.activations > 0.2) & 
                                        ~visited)[0]
                    stack.extend(neighbors)
        
        self.set_output('cluster_count', float(clusters))
        
        # Positions spectrum
        self.set_output('positions', self.positions.flatten())
        
        # Render
        self._render_universe()

    def _render_universe(self):
        size = self.config['canvas_size']
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        
        n = self.config['n_particles']
        radius = self.config['interaction_radius']
        
        # Draw connections first (faint)
        for i in range(n):
            if self.activations[i] < 0.1:
                continue
            
            p1 = (int(self.positions[i, 0] * size), 
                  int(self.positions[i, 1] * size))
            
            for j in range(i + 1, n):
                if self.weights[i, j] < 0.1:
                    continue
                
                p2 = (int(self.positions[j, 0] * size),
                      int(self.positions[j, 1] * size))
                
                # Distance check (avoid wrap-around lines)
                dx = abs(self.positions[i, 0] - self.positions[j, 0])
                dy = abs(self.positions[i, 1] - self.positions[j, 1])
                if dx > 0.5 or dy > 0.5:
                    continue
                
                weight = self.weights[i, j]
                color = (int(50 * weight), int(100 * weight), int(150 * weight))
                cv2.line(canvas, p1, p2, color, 1)
        
        # Draw particles
        for i in range(n):
            x = int(self.positions[i, 0] * size)
            y = int(self.positions[i, 1] * size)
            act = self.activations[i]
            
            # Size based on activation
            particle_size = max(2, int(act * 8))
            
            # Color: blue (inactive) to yellow (active)
            r = int(255 * act)
            g = int(200 * act)
            b = int(255 * (1 - act) * 0.5)
            
            cv2.circle(canvas, (x, y), particle_size, (b, g, r), -1)
            
            # Glow for very active
            if act > 0.5:
                cv2.circle(canvas, (x, y), particle_size + 3, 
                          (b // 2, g // 2, r // 2), 1)
        
        # Info
        density = self.get_output('density') or 0
        clusters = self.get_output('cluster_count') or 0
        cv2.putText(canvas, f"Density: {density:.2f}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(canvas, f"Clusters: {int(clusters)}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        self.set_output('universe_view', canvas)
