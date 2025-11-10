"""
More Coordinate-Driven Nodes - ULTRA SAFE VERSION

Wave interference, Voronoi fields, Lissajous curves, Flow field
All with bulletproof bounds checking
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class WaveInterferenceNode(BaseNode):
    """Wave interference - SAFE"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 180, 220)
    
    def __init__(self, size=256, num_sources=3):
        super().__init__()
        self.node_title = "Wave Interference"
        
        self.inputs = {
            'source1_x': 'signal',
            'source1_y': 'signal',
            'source2_x': 'signal',
            'source2_y': 'signal',
            'frequency': 'signal',
            'phase_speed': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'intensity': 'signal'
        }
        
        self.size = int(size)
        self.num_sources = int(num_sources)
        self.sources = np.random.rand(self.num_sources, 2) * self.size
        self.phase = 0.0
        
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.coords = np.stack([x, y], axis=-1)
        
        self.field = np.zeros((self.size, self.size), dtype=np.float32)
        self.intensity = 0.0
        
    def step(self):
        s1x = self.get_blended_input('source1_x', 'sum') or 0.0
        s1y = self.get_blended_input('source1_y', 'sum') or 0.0
        s2x = self.get_blended_input('source2_x', 'sum') or 0.0
        s2y = self.get_blended_input('source2_y', 'sum') or 0.0
        freq = self.get_blended_input('frequency', 'sum') or 0.0
        phase_speed = self.get_blended_input('phase_speed', 'sum') or 1.0
        
        self.sources[0] = [(s1x + 1) * 0.5 * self.size, (s1y + 1) * 0.5 * self.size]
        if len(self.sources) > 1:
            self.sources[1] = [(s2x + 1) * 0.5 * self.size, (s2y + 1) * 0.5 * self.size]
        
        for i in range(2, len(self.sources)):
            angle = (i / len(self.sources)) * 2 * np.pi + self.phase * 0.1
            self.sources[i] = [
                self.size * 0.5 + np.cos(angle) * self.size * 0.3,
                self.size * 0.5 + np.sin(angle) * self.size * 0.3
            ]
        
        wave_frequency = 0.05 + freq * 0.05
        self.phase += 0.1 * phase_speed
        
        field = np.zeros((self.size, self.size), dtype=np.float32)
        
        for source in self.sources:
            dx = self.coords[:, :, 0] - source[0]
            dy = self.coords[:, :, 1] - source[1]
            dist = np.sqrt(dx**2 + dy**2)
            wave = np.sin(dist * wave_frequency - self.phase)
            amplitude = 1.0 / (1.0 + dist / 100.0)
            field += wave * amplitude
        
        self.field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        center = self.size // 2
        self.intensity = float(self.field[center, center])
        
    def get_output(self, port_name):
        if port_name == 'image':
            colored = cv2.applyColorMap((self.field * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
            return colored.astype(np.float32) / 255.0
        elif port_name == 'intensity':
            return self.intensity
        return None


class VoronoiFieldNode(BaseNode):
    """Voronoi field - SAFE"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(220, 150, 100)
    
    def __init__(self, size=256, num_seeds=8):
        super().__init__()
        self.node_title = "Voronoi Field"
        
        self.inputs = {
            'seed1_x': 'signal',
            'seed1_y': 'signal',
            'seed2_x': 'signal',
            'seed2_y': 'signal',
            'rotation': 'signal',
            'scale': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'edge_density': 'signal'
        }
        
        self.size = int(size)
        self.num_seeds = int(num_seeds)
        self.seeds = np.random.rand(self.num_seeds, 2) * self.size
        self.colors = np.random.rand(self.num_seeds, 3)
        self.image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.edge_density = 0.0
        
    def step(self):
        s1x = self.get_blended_input('seed1_x', 'sum') or 0.0
        s1y = self.get_blended_input('seed1_y', 'sum') or 0.0
        s2x = self.get_blended_input('seed2_x', 'sum') or 0.0
        s2y = self.get_blended_input('seed2_y', 'sum') or 0.0
        rotation = self.get_blended_input('rotation', 'sum') or 0.0
        scale = self.get_blended_input('scale', 'sum') or 0.0
        
        self.seeds[0] = [(s1x + 1) * 0.5 * self.size, (s1y + 1) * 0.5 * self.size]
        if self.num_seeds > 1:
            self.seeds[1] = [(s2x + 1) * 0.5 * self.size, (s2y + 1) * 0.5 * self.size]
        
        angle_offset = rotation * np.pi
        scale_factor = 0.3 + scale * 0.2
        
        for i in range(2, self.num_seeds):
            angle = (i / self.num_seeds) * 2 * np.pi + angle_offset
            self.seeds[i] = [
                self.size * 0.5 + np.cos(angle) * self.size * scale_factor,
                self.size * 0.5 + np.sin(angle) * self.size * scale_factor
            ]
        
        image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        y, x = np.mgrid[0:self.size, 0:self.size]
        
        min_dist = np.full((self.size, self.size), np.inf)
        closest_seed = np.zeros((self.size, self.size), dtype=int)
        
        for i, seed in enumerate(self.seeds):
            dx = x - seed[0]
            dy = y - seed[1]
            dist = np.sqrt(dx**2 + dy**2)
            mask = dist < min_dist
            min_dist[mask] = dist[mask]
            closest_seed[mask] = i
        
        for i in range(self.num_seeds):
            mask = closest_seed == i
            image[mask] = self.colors[i]
        
        edges = np.zeros((self.size, self.size), dtype=np.float32)
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                if closest_seed[i, j] != closest_seed[i-1, j] or \
                   closest_seed[i, j] != closest_seed[i, j-1]:
                    edges[i, j] = 1.0
        
        edges_colored = np.stack([edges, edges, edges], axis=-1)
        image = image * (1 - edges_colored * 0.5) + edges_colored * 0.5
        
        self.image = image
        self.edge_density = float(np.mean(edges))
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.image
        elif port_name == 'edge_density':
            return self.edge_density
        return None


class LissajousNode(BaseNode):
    """Lissajous curves - ULTRA SAFE VERSION"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(180, 120, 220)
    
    def __init__(self, size=256, trail_length=100):
        super().__init__()
        self.node_title = "Lissajous Curves"
        
        self.inputs = {
            'freq_x': 'signal',
            'freq_y': 'signal',
            'phase': 'signal',
            'speed': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'symmetry': 'signal'
        }
        
        self.size = int(size)
        self.trail_length = max(10, int(trail_length))
        
        # Trail buffer - use list for safety
        self.trail = [[self.size // 2, self.size // 2] for _ in range(self.trail_length)]
        self.trail_idx = 0
        
        self.t = 0.0
        self.symmetry = 0.0
        
    def step(self):
        fx = self.get_blended_input('freq_x', 'sum') or 0.0
        fy = self.get_blended_input('freq_y', 'sum') or 0.0
        phase = self.get_blended_input('phase', 'sum') or 0.0
        speed = self.get_blended_input('speed', 'sum') or 1.0
        
        freq_x = 1.0 + fx * 2.0
        freq_y = 1.0 + fy * 2.0
        phase_shift = phase * np.pi
        
        x = np.sin(freq_x * self.t + phase_shift)
        y = np.sin(freq_y * self.t)
        
        px = int(np.clip((x + 1) * 0.5 * self.size, 0, self.size - 1))
        py = int(np.clip((y + 1) * 0.5 * self.size, 0, self.size - 1))
        
        # SAFE: Update current trail position
        self.trail[self.trail_idx] = [px, py]
        
        # SAFE: Increment with wrap
        self.trail_idx = (self.trail_idx + 1) % self.trail_length
        
        self.t += 0.05 * speed
        
        # Symmetry calculation
        if self.trail_length > 20:
            recent = np.array(self.trail[-20:])
            variance = np.var(recent, axis=0)
            self.symmetry = 1.0 / (1.0 + np.mean(variance) / 100.0)
        else:
            self.symmetry = 0.0
        
    def get_output(self, port_name):
        if port_name == 'image':
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            
            # Convert trail to numpy array and clamp
            points = np.array(self.trail, dtype=np.int32)
            points = np.clip(points, 0, self.size - 1)
            
            # Draw lines
            for i in range(len(points) - 1):
                p1 = tuple(points[i])
                p2 = tuple(points[i + 1])
                color_intensity = int((i / len(points)) * 255)
                color = (color_intensity, 100, 255 - color_intensity)
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
            
            # Draw current point
            current_idx = (self.trail_idx - 1 + self.trail_length) % self.trail_length
            current = tuple(points[current_idx])
            cv2.circle(image, current, 5, (255, 255, 255), -1)
            
            return image.astype(np.float32) / 255.0
        elif port_name == 'symmetry':
            return self.symmetry
        return None


class FlowFieldNode(BaseNode):
    """Flow field - ULTRA SAFE VERSION"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(120, 200, 150)
    
    def __init__(self, size=256, particle_count=200):
        super().__init__()
        self.node_title = "Flow Field"
        
        self.inputs = {
            'offset_x': 'signal',
            'offset_y': 'signal',
            'scale': 'signal',
            'strength': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'turbulence': 'signal'
        }
        
        self.size = int(size)
        self.particle_count = int(particle_count)
        
        # Initialize particles in safe zone
        self.particles = np.random.rand(self.particle_count, 2) * (self.size - 2) + 1
        self.trail_buffer = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.turbulence = 0.0
        
    def step(self):
        ox = self.get_blended_input('offset_x', 'sum') or 0.0
        oy = self.get_blended_input('offset_y', 'sum') or 0.0
        scale = self.get_blended_input('scale', 'sum') or 0.0
        strength = self.get_blended_input('strength', 'sum') or 1.0
        
        noise_scale = 0.02 + scale * 0.03
        offset = np.array([ox * 100, oy * 100])
        
        for i in range(len(self.particles)):
            pos = self.particles[i]
            noise_pos = (pos + offset) * noise_scale
            
            angle = np.sin(noise_pos[0]) * np.cos(noise_pos[1]) * 2 * np.pi
            vx = np.cos(angle) * strength
            vy = np.sin(angle) * strength
            
            # Limit velocity
            vx = np.clip(vx, -5, 5)
            vy = np.clip(vy, -5, 5)
            
            self.particles[i] += [vx, vy]
            
            # HARD clamp
            self.particles[i] = np.clip(self.particles[i], 0, self.size - 1)
            
            # Draw - SAFE
            x = int(self.particles[i][0])
            y = int(self.particles[i][1])
            
            if 0 <= x < self.size and 0 <= y < self.size:
                color = np.clip(np.array([vx, vy, 0.5]) * 0.5 + 0.5, 0, 1)
                self.trail_buffer[y, x] = color
        
        self.trail_buffer *= 0.95
        
        # Turbulence
        velocities = []
        for pos in self.particles:
            noise_pos = (pos + offset) * noise_scale
            angle = np.sin(noise_pos[0]) * np.cos(noise_pos[1]) * 2 * np.pi
            velocities.append([np.cos(angle), np.sin(angle)])
        
        self.turbulence = float(np.var(velocities))
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.trail_buffer
        elif port_name == 'turbulence':
            return self.turbulence
        return None