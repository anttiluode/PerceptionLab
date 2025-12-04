"""
Enhanced Flow Field Node - Controllable Lightning Generator

Adjustable initialization patterns, live parameter control,
and the ability to capture/restore particle states.
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class FlowFieldEnhancedNode(BaseNode):
    """Flow field with full control - chase the lightning"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 220, 180)
    
    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Flow Field Enhanced"
        
        self.inputs = {
            # Field control
            'offset_x': 'signal',
            'offset_y': 'signal',
            'scale': 'signal',
            'strength': 'signal',
            # Enhanced control
            'particle_count': 'signal',    # 10-1000
            'init_pattern': 'signal',      # 0=random, 1=line, 2=circle, 3=grid, 4=center, 5=spiral
            'trail_decay': 'signal',       # 0.8-0.99
            'seed': 'signal',              # random seed (integer part used)
            'reset': 'signal',             # >0.5 triggers reset
            'line_angle': 'signal',        # for line init pattern
            'curl': 'signal',              # adds curl to the field
        }
        self.outputs = {
            'image': 'image',
            'turbulence': 'signal',
            'coherence': 'signal',         # how aligned are particle velocities
            'particle_image': 'image',     # just the particles, no trail
        }
        
        self.size = int(size)
        
        # State
        self.particles = None
        self.velocities = None
        self.trail_buffer = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.particle_buffer = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Metrics
        self.turbulence = 0.0
        self.coherence = 0.0
        
        # Track last settings for change detection
        self.last_count = 200
        self.last_pattern = 0
        self.last_seed = -1
        self.last_reset = 0.0
        
        # Initialize
        self._init_particles(200, 0, None, 0.0)
        
    def _init_particles(self, count, pattern, seed, line_angle):
        """Initialize particles with given pattern"""
        count = int(np.clip(count, 10, 2000))
        
        if seed is not None and seed >= 0:
            np.random.seed(int(seed))
        
        if pattern == 0:  # Random
            self.particles = np.random.rand(count, 2) * (self.size - 2) + 1
            
        elif pattern == 1:  # Line (adjustable angle)
            t = np.linspace(0.1, 0.9, count)
            angle = line_angle * np.pi  # -1 to 1 maps to -pi to pi
            cx, cy = self.size / 2, self.size / 2
            length = self.size * 0.4
            self.particles = np.stack([
                cx + (t - 0.5) * length * 2 * np.cos(angle),
                cy + (t - 0.5) * length * 2 * np.sin(angle)
            ], axis=1)
            
        elif pattern == 2:  # Circle
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            radius = self.size * 0.35
            self.particles = np.stack([
                self.size/2 + np.cos(angles) * radius,
                self.size/2 + np.sin(angles) * radius
            ], axis=1)
            
        elif pattern == 3:  # Grid
            side = int(np.sqrt(count))
            xs = np.linspace(self.size * 0.1, self.size * 0.9, side)
            ys = np.linspace(self.size * 0.1, self.size * 0.9, side)
            xx, yy = np.meshgrid(xs, ys)
            self.particles = np.stack([xx.flatten(), yy.flatten()], axis=1)[:count]
            
        elif pattern == 4:  # Center burst
            angles = np.random.rand(count) * 2 * np.pi
            radii = np.random.rand(count) * self.size * 0.1
            self.particles = np.stack([
                self.size/2 + np.cos(angles) * radii,
                self.size/2 + np.sin(angles) * radii
            ], axis=1)
            
        elif pattern == 5:  # Spiral
            t = np.linspace(0, 4 * np.pi, count)
            r = np.linspace(10, self.size * 0.4, count)
            self.particles = np.stack([
                self.size/2 + np.cos(t) * r,
                self.size/2 + np.sin(t) * r
            ], axis=1)
            
        elif pattern == 6:  # Diagonal cross
            half = count // 2
            t1 = np.linspace(0.1, 0.9, half) * self.size
            t2 = np.linspace(0.1, 0.9, count - half) * self.size
            p1 = np.stack([t1, t1], axis=1)  # diagonal
            p2 = np.stack([t2, self.size - t2], axis=1)  # anti-diagonal
            self.particles = np.vstack([p1, p2])
            
        elif pattern == 7:  # Few particles (sparse - for lightning)
            count = min(count, 20)  # Force sparse
            self.particles = np.random.rand(count, 2) * (self.size - 2) + 1
            
        else:  # Default random
            self.particles = np.random.rand(count, 2) * (self.size - 2) + 1
        
        # Initialize velocities
        self.velocities = np.zeros_like(self.particles)
        
        # Clear trail on reset
        self.trail_buffer *= 0.0
        
    def step(self):
        # Get inputs
        ox = self.get_blended_input('offset_x', 'sum') or 0.0
        oy = self.get_blended_input('offset_y', 'sum') or 0.0
        scale = self.get_blended_input('scale', 'sum') or 0.0
        strength = self.get_blended_input('strength', 'sum') or 1.0
        
        particle_count = self.get_blended_input('particle_count', 'sum')
        particle_count = int(particle_count * 100 + 100) if particle_count else 200
        
        init_pattern = self.get_blended_input('init_pattern', 'sum')
        init_pattern = int((init_pattern + 1) * 4) if init_pattern else 0
        init_pattern = np.clip(init_pattern, 0, 7)
        
        trail_decay = self.get_blended_input('trail_decay', 'sum')
        trail_decay = 0.9 + (trail_decay or 0) * 0.09  # 0.81 to 0.99
        trail_decay = np.clip(trail_decay, 0.8, 0.995)
        
        seed_in = self.get_blended_input('seed', 'sum')
        seed = int(seed_in * 1000) if seed_in else -1
        
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        line_angle = self.get_blended_input('line_angle', 'sum') or 0.0
        
        curl = self.get_blended_input('curl', 'sum') or 0.0
        
        # Check for reinit triggers
        need_reinit = False
        if reset > 0.5 and self.last_reset <= 0.5:
            need_reinit = True
        if seed >= 0 and seed != self.last_seed:
            need_reinit = True
        if init_pattern != self.last_pattern:
            need_reinit = True
        if abs(particle_count - self.last_count) > 10:
            need_reinit = True
            
        if need_reinit:
            self._init_particles(particle_count, init_pattern, seed if seed >= 0 else None, line_angle)
            
        self.last_count = particle_count
        self.last_pattern = init_pattern
        self.last_seed = seed
        self.last_reset = reset
        
        # Field parameters
        noise_scale = 0.02 + scale * 0.03
        offset = np.array([ox * 100, oy * 100])
        
        # Clear particle buffer
        self.particle_buffer *= 0
        
        # Move particles
        new_velocities = []
        for i in range(len(self.particles)):
            pos = self.particles[i]
            noise_pos = (pos + offset) * noise_scale
            
            # Base angle from noise
            angle = np.sin(noise_pos[0]) * np.cos(noise_pos[1]) * 2 * np.pi
            
            # Add curl (rotation component)
            if curl != 0:
                dx = pos[0] - self.size/2
                dy = pos[1] - self.size/2
                r = np.sqrt(dx*dx + dy*dy) + 1
                curl_angle = np.arctan2(dy, dx) + np.pi/2  # perpendicular
                angle += curl * curl_angle * (self.size / r) * 0.1
            
            vx = np.cos(angle) * strength
            vy = np.sin(angle) * strength
            
            # Momentum (smooths the lightning)
            vx = self.velocities[i, 0] * 0.3 + vx * 0.7
            vy = self.velocities[i, 1] * 0.3 + vy * 0.7
            
            # Limit velocity
            speed = np.sqrt(vx*vx + vy*vy)
            max_speed = 5.0
            if speed > max_speed:
                vx *= max_speed / speed
                vy *= max_speed / speed
            
            self.velocities[i] = [vx, vy]
            new_velocities.append([vx, vy])
            
            self.particles[i] += [vx, vy]
            
            # Wrap or clamp
            self.particles[i] = np.clip(self.particles[i], 0, self.size - 1)
            
            # Draw
            x = int(self.particles[i][0])
            y = int(self.particles[i][1])
            
            if 0 <= x < self.size and 0 <= y < self.size:
                # Color by velocity direction
                color = np.array([
                    0.5 + vx * 0.3,
                    0.5 + vy * 0.3,
                    0.8
                ])
                color = np.clip(color, 0, 1)
                self.trail_buffer[y, x] = color
                self.particle_buffer[y, x] = [1.0, 1.0, 1.0]  # white dots
        
        # Trail decay
        self.trail_buffer *= trail_decay
        
        # Compute metrics
        vels = np.array(new_velocities)
        self.turbulence = float(np.var(vels))
        
        # Coherence: how aligned are velocities?
        if len(vels) > 1:
            mean_vel = np.mean(vels, axis=0)
            mean_speed = np.linalg.norm(mean_vel)
            avg_speed = np.mean(np.linalg.norm(vels, axis=1))
            self.coherence = mean_speed / (avg_speed + 1e-6)
        else:
            self.coherence = 0.0
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.trail_buffer
        elif port_name == 'particle_image':
            return self.particle_buffer
        elif port_name == 'turbulence':
            return self.turbulence
        elif port_name == 'coherence':
            return self.coherence
        return None
    
    def draw_custom(self, painter):
        """Show current settings"""
        painter.setPen(QtGui.QColor(200, 255, 200))
        painter.setFont(QtGui.QFont("Consolas", 8))
        
        info = f"P:{len(self.particles) if self.particles is not None else 0}"
        info += f" Pat:{self.last_pattern}"
        info += f" Coh:{self.coherence:.2f}"
        
        painter.drawText(5, self.height - 25, info)


class FlowFieldEEGNode(BaseNode):
    """Flow field specifically tuned for EEG lightning effects"""
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(80, 200, 220)
    
    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Flow Field EEG"
        
        self.inputs = {
            # EEG inputs directly
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            # Optional field input
            'field_image': 'image',  # can drive from holographic
            # Control
            'sensitivity': 'signal',
            'reset': 'signal',
        }
        self.outputs = {
            'image': 'image',
            'turbulence': 'signal',
            'coherence': 'signal',
            'arc_intensity': 'signal',  # how "lightning-like" is current frame
        }
        
        self.size = int(size)
        
        # Sparse particles for lightning effect
        self.particle_count = 50
        self.particles = None
        self.velocities = None
        self.trail_buffer = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Metrics
        self.turbulence = 0.0
        self.coherence = 0.0
        self.arc_intensity = 0.0
        
        # Field cache
        self.field_angle = np.zeros((self.size, self.size), dtype=np.float32)
        
        self._init_particles()
        
    def _init_particles(self):
        """Initialize sparse particles in curved line - good for arcs"""
        t = np.linspace(0, 1, self.particle_count)
        # Slight curve
        self.particles = np.stack([
            self.size * 0.2 + t * self.size * 0.6,
            self.size * 0.5 + np.sin(t * np.pi) * self.size * 0.2
        ], axis=1)
        self.velocities = np.zeros_like(self.particles)
        
    def step(self):
        # Get EEG bands
        delta = self.get_blended_input('delta', 'sum') or 0.0
        theta = self.get_blended_input('theta', 'sum') or 0.0
        alpha = self.get_blended_input('alpha', 'sum') or 0.0
        beta = self.get_blended_input('beta', 'sum') or 0.0
        gamma = self.get_blended_input('gamma', 'sum') or 0.0
        
        sensitivity = self.get_blended_input('sensitivity', 'sum') or 1.0
        sensitivity = 0.5 + sensitivity * 2.0
        
        reset = self.get_blended_input('reset', 'sum') or 0.0
        if reset > 0.5:
            self._init_particles()
            self.trail_buffer *= 0
        
        # Optional field image
        field_img = self.get_blended_input('field_image', 'image')
        
        # Build angle field from EEG or image
        if field_img is not None and isinstance(field_img, np.ndarray):
            # Use image luminance as angle
            if len(field_img.shape) == 3:
                lum = np.mean(field_img, axis=2)
            else:
                lum = field_img
            # Resize if needed
            if lum.shape[0] != self.size:
                lum = cv2.resize(lum, (self.size, self.size))
            self.field_angle = lum * 2 * np.pi
        else:
            # Generate field from EEG
            y, x = np.mgrid[0:self.size, 0:self.size]
            cx, cy = self.size / 2, self.size / 2
            
            # Each band creates different spatial pattern
            angle = np.zeros((self.size, self.size), dtype=np.float32)
            
            # Delta: large slow swirls
            angle += delta * np.sin((x - cx) * 0.02) * np.cos((y - cy) * 0.02) * np.pi
            
            # Theta: medium waves
            angle += theta * np.sin((x + y) * 0.05) * np.pi
            
            # Alpha: circular pattern
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            angle += alpha * np.sin(r * 0.1) * np.pi
            
            # Beta: diagonal stripes
            angle += beta * np.sin((x - y) * 0.08) * np.pi
            
            # Gamma: fine noise
            angle += gamma * (np.random.rand(self.size, self.size) - 0.5) * np.pi
            
            self.field_angle = angle
        
        # Strength from total power
        total_power = abs(delta) + abs(theta) + abs(alpha) + abs(beta) + abs(gamma)
        strength = (0.5 + total_power * 0.5) * sensitivity
        
        # Move particles
        new_velocities = []
        arc_sum = 0.0
        
        for i in range(len(self.particles)):
            pos = self.particles[i]
            
            # Get angle from field
            px = int(np.clip(pos[0], 0, self.size - 1))
            py = int(np.clip(pos[1], 0, self.size - 1))
            angle = self.field_angle[py, px]
            
            vx = np.cos(angle) * strength
            vy = np.sin(angle) * strength
            
            # Momentum
            vx = self.velocities[i, 0] * 0.4 + vx * 0.6
            vy = self.velocities[i, 1] * 0.4 + vy * 0.6
            
            # Limit
            speed = np.sqrt(vx*vx + vy*vy)
            if speed > 8:
                vx *= 8 / speed
                vy *= 8 / speed
                arc_sum += 1  # Fast particle = arc-like
            
            self.velocities[i] = [vx, vy]
            new_velocities.append([vx, vy])
            
            self.particles[i] += [vx, vy]
            self.particles[i] = np.clip(self.particles[i], 0, self.size - 1)
            
            # Draw with intensity based on speed
            x = int(self.particles[i][0])
            y = int(self.particles[i][1])
            
            if 0 <= x < self.size and 0 <= y < self.size:
                intensity = min(1.0, speed / 4.0)
                # Cyan-white for lightning
                color = np.array([0.3 + intensity * 0.7, 0.8 + intensity * 0.2, 1.0])
                self.trail_buffer[y, x] = np.maximum(self.trail_buffer[y, x], color)
        
        # Slow decay for persistent trails
        self.trail_buffer *= 0.92
        
        # Metrics
        vels = np.array(new_velocities)
        self.turbulence = float(np.var(vels))
        
        if len(vels) > 1:
            mean_vel = np.mean(vels, axis=0)
            mean_speed = np.linalg.norm(mean_vel)
            avg_speed = np.mean(np.linalg.norm(vels, axis=1))
            self.coherence = mean_speed / (avg_speed + 1e-6)
        
        self.arc_intensity = arc_sum / len(self.particles)
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.trail_buffer
        elif port_name == 'turbulence':
            return self.turbulence
        elif port_name == 'coherence':
            return self.coherence
        elif port_name == 'arc_intensity':
            return self.arc_intensity
        return None