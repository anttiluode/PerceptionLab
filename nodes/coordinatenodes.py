"""
Particle Attractor Field Node - ULTRA-SAFE EDITION

NO ANTIALIASING - just simple pixel drawing
Absolute bounds protection - cannot possibly go out of range
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ParticleAttractorNode(BaseNode):
    """Particle swarm attracted to x/y coordinate position - ULTRA SAFE VERSION"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(200, 100, 180)
    
    def __init__(self, particle_count=300, size=256, attraction_strength=0.5):
        super().__init__()
        self.node_title = "Particle Attractor (Safe)"
        
        self.inputs = {
            'x_coord': 'signal',
            'y_coord': 'signal',
            'strength': 'signal',
            'chaos': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'density': 'signal'
        }
        
        self.particle_count = int(particle_count)
        self.size = int(size)
        self.attraction_strength = float(attraction_strength)
        
        # Initialize particles in center region only
        margin = self.size * 0.1
        self.positions = np.random.rand(self.particle_count, 2) * (self.size - 2*margin) + margin
        self.velocities = np.zeros((self.particle_count, 2), dtype=np.float32)
        
        # Trail buffer
        self.trail_buffer = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Output
        self.density = 0.0
        
    def step(self):
        # Get inputs
        x_coord = self.get_blended_input('x_coord', 'sum') or 0.0
        y_coord = self.get_blended_input('y_coord', 'sum') or 0.0
        strength = self.get_blended_input('strength', 'sum')
        if strength is None:
            strength = self.attraction_strength
        chaos = self.get_blended_input('chaos', 'sum') or 0.0
        
        # Attractor position
        attractor_x = np.clip((x_coord + 1.0) * 0.5 * self.size, 0, self.size - 1)
        attractor_y = np.clip((y_coord + 1.0) * 0.5 * self.size, 0, self.size - 1)
        attractor = np.array([attractor_x, attractor_y])
        
        # Forces
        to_attractor = attractor - self.positions
        distances = np.linalg.norm(to_attractor, axis=1, keepdims=True)
        distances = np.maximum(distances, 10.0)  # Prevent extreme forces
        
        # Attraction (clamped)
        forces = to_attractor / (distances ** 2) * strength * 50
        forces = np.clip(forces, -20, 20)
        
        # Chaos
        if chaos > 0.01:
            forces += (np.random.rand(self.particle_count, 2) - 0.5) * chaos * 5
        
        # Update
        self.velocities += forces * 0.1
        self.velocities = np.clip(self.velocities, -5, 5)
        self.velocities *= 0.9
        self.positions += self.velocities
        
        # ABSOLUTE HARD CLAMP - cannot escape
        self.positions = np.clip(self.positions, 0, self.size - 1.01)
        
        # Fade
        self.trail_buffer *= 0.92
        
        # Draw - NO ANTIALIASING, just simple pixels
        for i in range(len(self.positions)):
            x = int(self.positions[i, 0])
            y = int(self.positions[i, 1])
            
            # Paranoid bounds check
            if 0 <= x < self.size and 0 <= y < self.size:
                self.trail_buffer[y, x] += 1.0
        
        # Density
        attractor_distances = np.linalg.norm(self.positions - attractor, axis=1)
        close_particles = np.sum(attractor_distances < self.size * 0.15)
        self.density = close_particles / self.particle_count
        
    def get_output(self, port_name):
        if port_name == 'image':
            normalized = np.clip(self.trail_buffer / (np.max(self.trail_buffer) + 1e-9), 0, 1)
            colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            return colored.astype(np.float32) / 255.0
        elif port_name == 'density':
            return self.density
        return None


class StrangeAttractorNode(BaseNode):
    """Strange attractor - ULTRA SAFE VERSION"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(180, 100, 200)
    
    def __init__(self, size=256, attractor_type='lorenz'):
        super().__init__()
        self.node_title = f"Strange Attractor ({attractor_type})"
        
        self.inputs = {
            'param_a': 'signal',
            'param_b': 'signal',
            'speed': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'chaos': 'signal'
        }
        
        self.size = int(size)
        self.attractor_type = attractor_type
        self.state = np.array([0.1, 0.0, 0.0])
        self.trail_buffer = np.zeros((self.size, self.size), dtype=np.float32)
        self.history = []
        self.chaos_measure = 0.0
        
    def step(self):
        param_a = self.get_blended_input('param_a', 'sum') or 0.0
        param_b = self.get_blended_input('param_b', 'sum') or 0.0
        speed = self.get_blended_input('speed', 'sum') or 1.0
        
        if self.attractor_type == 'lorenz':
            sigma = 10.0 + param_a * 5.0
            rho = 28.0 + param_b * 10.0
            beta = 8.0 / 3.0
            
            x, y, z = self.state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            dt = 0.01 * speed
            self.state += np.array([dx, dy, dz]) * dt
            
            proj_x = (x / 30.0 + 1.0) * 0.5 * self.size
            proj_y = (z / 50.0) * 0.5 * self.size + self.size * 0.5
            
        else:  # rossler or aizawa
            a = 0.2 + param_a * 0.1
            b = 0.2 + param_b * 0.1
            c = 5.7
            
            x, y, z = self.state
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            dt = 0.05 * speed
            self.state += np.array([dx, dy, dz]) * dt
            
            proj_x = (x / 15.0 + 1.0) * 0.5 * self.size
            proj_y = (y / 15.0 + 1.0) * 0.5 * self.size
        
        self.trail_buffer *= 0.98
        
        # ULTRA SAFE drawing
        x_px = int(np.clip(proj_x, 0, self.size - 1))
        y_px = int(np.clip(proj_y, 0, self.size - 1))
        
        if 0 <= x_px < self.size and 0 <= y_px < self.size:
            self.trail_buffer[y_px, x_px] += 1.0
        
        self.history.append(np.copy(self.state))
        if len(self.history) > 100:
            self.history.pop(0)
        
        if len(self.history) > 10:
            recent = np.array(self.history[-10:])
            variance = np.var(recent, axis=0)
            self.chaos_measure = np.mean(variance) / 100.0
        else:
            self.chaos_measure = 0.0
            
    def get_output(self, port_name):
        if port_name == 'image':
            normalized = np.clip(self.trail_buffer / (np.max(self.trail_buffer) + 1e-9), 0, 1)
            colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            return colored.astype(np.float32) / 255.0
        elif port_name == 'chaos':
            return self.chaos_measure
        return None


class ReactionDiffusionNode(BaseNode):
    """Reaction-diffusion - ULTRA SAFE VERSION"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(150, 200, 100)
    
    def __init__(self, size=128, pattern='spots'):
        super().__init__()
        self.node_title = "Reaction-Diffusion"
        
        self.inputs = {
            'feed_rate': 'signal',
            'kill_rate': 'signal',
            'seed': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'pattern_energy': 'signal'
        }
        
        self.size = int(size)
        self.pattern = pattern
        
        self.A = np.ones((self.size, self.size), dtype=np.float32)
        self.B = np.zeros((self.size, self.size), dtype=np.float32)
        
        center = self.size // 2
        radius = self.size // 10
        y, x = np.ogrid[-center:self.size-center, -center:self.size-center]
        mask = x*x + y*y <= radius*radius
        self.B[mask] = 1.0
        
        self.Da = 1.0
        self.Db = 0.5
        self.last_seed = 0.0
        self.pattern_energy = 0.0
        
    def step(self):
        feed_rate = self.get_blended_input('feed_rate', 'sum')
        kill_rate = self.get_blended_input('kill_rate', 'sum')
        seed = self.get_blended_input('seed', 'sum') or 0.0
        
        feed = 0.055 if feed_rate is None else 0.01 + (feed_rate + 1.0) * 0.05
        kill = 0.062 if kill_rate is None else 0.03 + (kill_rate + 1.0) * 0.04
        
        if seed > 0.5 and self.last_seed <= 0.5:
            x = self.size // 2
            y = self.size // 2
            radius = max(2, self.size // 20)
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i*i + j*j <= radius*radius:
                        xi = (x + i) % self.size
                        yi = (y + j) % self.size
                        if 0 <= xi < self.size and 0 <= yi < self.size:
                            self.B[yi, xi] = 1.0
        self.last_seed = seed
        
        kernel = np.array([[0.05, 0.2, 0.05],
                          [0.2, -1.0, 0.2],
                          [0.05, 0.2, 0.05]])
        
        laplaceA = cv2.filter2D(self.A, -1, kernel, borderType=cv2.BORDER_WRAP)
        laplaceB = cv2.filter2D(self.B, -1, kernel, borderType=cv2.BORDER_WRAP)
        
        reaction = self.A * self.B * self.B
        
        dA = self.Da * laplaceA - reaction + feed * (1.0 - self.A)
        dB = self.Db * laplaceB + reaction - (kill + feed) * self.B
        
        dt = 1.0
        self.A += dA * dt
        self.B += dB * dt
        
        self.A = np.clip(self.A, 0, 1)
        self.B = np.clip(self.B, 0, 1)
        
        self.pattern_energy = float(np.var(self.B))
        
    def get_output(self, port_name):
        if port_name == 'image':
            colored = cv2.applyColorMap((self.B * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            return colored.astype(np.float32) / 255.0
        elif port_name == 'pattern_energy':
            return self.pattern_energy
        return None