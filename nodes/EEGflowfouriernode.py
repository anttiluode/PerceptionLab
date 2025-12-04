"""
EEG Flow Fourier Node

A carefully designed node for exploring how EEG signals
create structure in flow fields and what eigenmodes emerge.

The pipeline:
  EEG → Vector Field → Particle Trajectories → Density → FFT → Eigenmodes

Key insight: Different mappings from EEG to vector field
produce radically different eigenmode structures.
"""

import numpy as np
import cv2
from scipy import ndimage

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class EEGFlowFourierNode(BaseNode):
    """
    EEG → Flow Field → FFT eigenmode explorer
    
    This node lets you experiment with different ways of mapping
    brain signals to spatial dynamics, then see what Fourier
    structure emerges.
    """
    NODE_CATEGORY = "IHT_Core"
    NODE_COLOR = QtGui.QColor(60, 180, 200)
    
    def __init__(self, size=256):
        super().__init__()
        self.node_title = "EEG Flow Fourier"
        
        self.inputs = {
            # EEG band inputs
            'delta': 'signal',      # 1-4 Hz
            'theta': 'signal',      # 4-8 Hz  
            'alpha': 'signal',      # 8-13 Hz
            'beta': 'signal',       # 13-30 Hz
            'gamma': 'signal',      # 30-45 Hz
            'raw': 'signal',        # raw EEG signal
            
            # Control inputs
            'field_mode': 'signal',      # 0-5: how EEG maps to vector field
            'init_mode': 'signal',       # 0-7: particle initialization
            'particle_count': 'signal',  # number of particles (scaled)
            'speed': 'signal',           # particle speed multiplier
            'decay': 'signal',           # trail decay rate
            'reset': 'signal',           # >0.5 resets particles
            
            # Advanced
            'field_scale': 'signal',     # spatial frequency of field
            'momentum': 'signal',        # particle momentum (smoothing)
            'inject_x': 'signal',        # manual field injection
            'inject_y': 'signal',
        }
        
        self.outputs = {
            # Visual outputs
            'flow_image': 'image',           # the flow field trails
            'fft_magnitude': 'image',        # FFT magnitude (log scaled)
            'fft_phase': 'image',            # FFT phase
            'eigenmode_image': 'image',      # colorized eigenmode view
            
            # Data outputs  
            'complex_spectrum': 'complex_spectrum',  # for holographic nodes
            'dominant_frequency': 'signal',          # strongest spatial freq
            'spectral_entropy': 'signal',            # complexity measure
            'flow_coherence': 'signal',              # how organized is flow
            'eigenmode_centroid': 'signal',          # where is spectral mass
        }
        
        self.size = int(size)
        self.half = self.size // 2
        
        # Particle system
        self.particles = None
        self.velocities = None
        self.particle_count = 500
        
        # Buffers
        self.trail_buffer = np.zeros((self.size, self.size), dtype=np.float32)
        self.field_x = np.zeros((self.size, self.size), dtype=np.float32)
        self.field_y = np.zeros((self.size, self.size), dtype=np.float32)
        
        # FFT results
        self.fft_result = None
        self.magnitude = None
        self.phase = None
        
        # Metrics
        self.dominant_freq = 0.0
        self.spectral_entropy = 0.0
        self.flow_coherence = 0.0
        self.eigenmode_centroid = 0.0
        
        # Coordinate grids (precomputed)
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.x_grid = x.astype(np.float32)
        self.y_grid = y.astype(np.float32)
        self.cx, self.cy = self.size / 2, self.size / 2
        self.r_grid = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        self.theta_grid = np.arctan2(y - self.cy, x - self.cx)
        
        # Frequency grid for FFT analysis
        fx = np.fft.fftfreq(self.size)
        fy = np.fft.fftfreq(self.size)
        self.freq_x, self.freq_y = np.meshgrid(fx, fy)
        self.freq_r = np.sqrt(self.freq_x**2 + self.freq_y**2)
        
        # State tracking
        self.last_init_mode = -1
        self.last_reset = 0.0
        self.frame_count = 0
        
        # Initialize
        self._init_particles(0)
        
    def _init_particles(self, mode):
        """Initialize particles with various patterns"""
        n = self.particle_count
        
        if mode == 0:  # Random uniform
            self.particles = np.random.rand(n, 2) * self.size
            
        elif mode == 1:  # Horizontal line
            t = np.linspace(0.05, 0.95, n)
            self.particles = np.stack([
                t * self.size,
                np.ones(n) * self.cy
            ], axis=1)
            
        elif mode == 2:  # Vertical line
            t = np.linspace(0.05, 0.95, n)
            self.particles = np.stack([
                np.ones(n) * self.cx,
                t * self.size
            ], axis=1)
            
        elif mode == 3:  # Circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = self.size * 0.4
            self.particles = np.stack([
                self.cx + np.cos(angles) * r,
                self.cy + np.sin(angles) * r
            ], axis=1)
            
        elif mode == 4:  # Grid
            side = int(np.sqrt(n))
            xs = np.linspace(0.1, 0.9, side) * self.size
            ys = np.linspace(0.1, 0.9, side) * self.size
            xx, yy = np.meshgrid(xs, ys)
            self.particles = np.stack([xx.flatten(), yy.flatten()], axis=1)[:n]
            
        elif mode == 5:  # Center point
            angles = np.random.rand(n) * 2 * np.pi
            radii = np.random.rand(n) * 5  # tight cluster
            self.particles = np.stack([
                self.cx + np.cos(angles) * radii,
                self.cy + np.sin(angles) * radii
            ], axis=1)
            
        elif mode == 6:  # Diagonal
            t = np.linspace(0.05, 0.95, n)
            self.particles = np.stack([
                t * self.size,
                t * self.size
            ], axis=1)
            
        elif mode == 7:  # Cross
            half = n // 2
            t1 = np.linspace(0.05, 0.95, half)
            t2 = np.linspace(0.05, 0.95, n - half)
            p1 = np.stack([t1 * self.size, np.ones(half) * self.cy], axis=1)
            p2 = np.stack([np.ones(n-half) * self.cx, t2 * self.size], axis=1)
            self.particles = np.vstack([p1, p2])
            
        elif mode == 8:  # Spiral
            t = np.linspace(0, 6*np.pi, n)
            r = np.linspace(5, self.size * 0.45, n)
            self.particles = np.stack([
                self.cx + np.cos(t) * r,
                self.cy + np.sin(t) * r
            ], axis=1)
            
        else:  # Sparse random (good for lightning)
            n = min(n, 50)
            self.particles = np.random.rand(n, 2) * self.size
            
        self.velocities = np.zeros((len(self.particles), 2), dtype=np.float32)
        self.trail_buffer *= 0  # Clear trails on reinit
        
    def _build_field_mode0(self, bands):
        """Mode 0: Radial - bands control ring frequencies"""
        delta, theta, alpha, beta, gamma = bands
        
        field = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Each band creates concentric ripples at different scales
        field += delta * np.sin(self.r_grid * 0.02) * 2
        field += theta * np.sin(self.r_grid * 0.05) * 2
        field += alpha * np.sin(self.r_grid * 0.10) * 2
        field += beta * np.sin(self.r_grid * 0.20) * 2
        field += gamma * np.sin(self.r_grid * 0.40) * 2
        
        # Convert to vector field (perpendicular to radius = circular flow)
        self.field_x = -np.sin(self.theta_grid) * field
        self.field_y = np.cos(self.theta_grid) * field
        
    def _build_field_mode1(self, bands):
        """Mode 1: Cartesian - bands control x/y wave frequencies"""
        delta, theta, alpha, beta, gamma = bands
        
        # X component from odd bands
        self.field_x = (
            delta * np.sin(self.y_grid * 0.03) +
            alpha * np.sin(self.y_grid * 0.08) +
            gamma * np.sin(self.y_grid * 0.20)
        )
        
        # Y component from even bands  
        self.field_y = (
            theta * np.sin(self.x_grid * 0.05) +
            beta * np.sin(self.x_grid * 0.15)
        )
        
    def _build_field_mode2(self, bands):
        """Mode 2: Interference - bands are point sources"""
        delta, theta, alpha, beta, gamma = bands
        
        # Five sources at different positions
        sources = [
            (self.cx, self.cy * 0.3, delta),           # top
            (self.cx * 0.3, self.cy, theta),           # left
            (self.cx * 1.7, self.cy, alpha),           # right
            (self.cx, self.cy * 1.7, beta),            # bottom
            (self.cx, self.cy, gamma),                 # center
        ]
        
        potential = np.zeros((self.size, self.size), dtype=np.float32)
        for sx, sy, amp in sources:
            r = np.sqrt((self.x_grid - sx)**2 + (self.y_grid - sy)**2) + 1
            potential += amp * np.sin(r * 0.1) / (1 + r * 0.01)
        
        # Gradient of potential = force field
        self.field_y, self.field_x = np.gradient(potential)
        
    def _build_field_mode3(self, bands):
        """Mode 3: Vortex - bands control rotation strength at different radii"""
        delta, theta, alpha, beta, gamma = bands
        
        # Rotation strength varies with radius
        rotation = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Inner to outer rings controlled by bands
        rotation += delta * np.exp(-self.r_grid**2 / (self.size * 0.1)**2)
        rotation += theta * np.exp(-(self.r_grid - self.size*0.15)**2 / (self.size * 0.1)**2)
        rotation += alpha * np.exp(-(self.r_grid - self.size*0.25)**2 / (self.size * 0.1)**2)
        rotation += beta * np.exp(-(self.r_grid - self.size*0.35)**2 / (self.size * 0.1)**2)
        rotation += gamma * np.exp(-(self.r_grid - self.size*0.45)**2 / (self.size * 0.1)**2)
        
        # Perpendicular to radius (tangential flow)
        self.field_x = -np.sin(self.theta_grid) * rotation
        self.field_y = np.cos(self.theta_grid) * rotation
        
    def _build_field_mode4(self, bands):
        """Mode 4: Diagonal waves - creates X patterns in FFT"""
        delta, theta, alpha, beta, gamma = bands
        
        diag1 = self.x_grid + self.y_grid  # diagonal
        diag2 = self.x_grid - self.y_grid  # anti-diagonal
        
        wave1 = (
            delta * np.sin(diag1 * 0.02) +
            alpha * np.sin(diag1 * 0.06) +
            gamma * np.sin(diag1 * 0.15)
        )
        
        wave2 = (
            theta * np.sin(diag2 * 0.03) +
            beta * np.sin(diag2 * 0.10)
        )
        
        # Field follows diagonal gradients
        self.field_x = wave1 + wave2
        self.field_y = wave1 - wave2
        
    def _build_field_mode5(self, bands):
        """Mode 5: Fractal/turbulent - bands at octave frequencies"""
        delta, theta, alpha, beta, gamma = bands
        
        self.field_x = np.zeros((self.size, self.size), dtype=np.float32)
        self.field_y = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Octave frequencies (doubling)
        freqs = [0.01, 0.02, 0.04, 0.08, 0.16]
        amps = [delta, theta, alpha, beta, gamma]
        
        for freq, amp in zip(freqs, amps):
            phase_x = np.random.rand() * 2 * np.pi
            phase_y = np.random.rand() * 2 * np.pi
            self.field_x += amp * np.sin(self.x_grid * freq * 2 * np.pi + phase_x) * np.cos(self.y_grid * freq * np.pi)
            self.field_y += amp * np.cos(self.x_grid * freq * np.pi) * np.sin(self.y_grid * freq * 2 * np.pi + phase_y)
    
    def step(self):
        self.frame_count += 1
        
        # Get EEG bands
        delta = self.get_blended_input('delta', 'sum') or 0.0
        theta = self.get_blended_input('theta', 'sum') or 0.0
        alpha = self.get_blended_input('alpha', 'sum') or 0.0
        beta = self.get_blended_input('beta', 'sum') or 0.0
        gamma = self.get_blended_input('gamma', 'sum') or 0.0
        raw = self.get_blended_input('raw', 'sum') or 0.0
        
        # Normalize bands
        bands = np.array([delta, theta, alpha, beta, gamma])
        band_sum = np.sum(np.abs(bands)) + 1e-6
        bands_norm = bands / band_sum  # relative power
        
        # Get control inputs
        field_mode = self.get_blended_input('field_mode', 'sum') or 0.0
        field_mode = int(np.clip((field_mode + 1) * 3, 0, 5))  # 0-5
        
        init_mode = self.get_blended_input('init_mode', 'sum') or 0.0
        init_mode = int(np.clip((init_mode + 1) * 4, 0, 9))  # 0-9
        
        particle_count_in = self.get_blended_input('particle_count', 'sum') or 0.0
        self.particle_count = int(np.clip(200 + particle_count_in * 400, 50, 2000))
        
        speed = self.get_blended_input('speed', 'sum') or 0.0
        speed = 1.0 + speed * 2.0
        
        decay = self.get_blended_input('decay', 'sum') or 0.0
        decay = np.clip(0.92 + decay * 0.07, 0.85, 0.995)
        
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        field_scale = self.get_blended_input('field_scale', 'sum') or 0.0
        field_scale = 1.0 + field_scale
        
        momentum = self.get_blended_input('momentum', 'sum') or 0.0
        momentum = np.clip(0.3 + momentum * 0.5, 0.0, 0.9)
        
        inject_x = self.get_blended_input('inject_x', 'sum') or 0.0
        inject_y = self.get_blended_input('inject_y', 'sum') or 0.0
        
        # Check for reinit
        need_reinit = False
        if reset > 0.5 and self.last_reset <= 0.5:
            need_reinit = True
        if init_mode != self.last_init_mode:
            need_reinit = True
        if self.particles is None or len(self.particles) != self.particle_count:
            need_reinit = True
            
        if need_reinit:
            self._init_particles(init_mode)
            
        self.last_init_mode = init_mode
        self.last_reset = reset
        
        # Build vector field based on mode
        if field_mode == 0:
            self._build_field_mode0(bands)
        elif field_mode == 1:
            self._build_field_mode1(bands)
        elif field_mode == 2:
            self._build_field_mode2(bands)
        elif field_mode == 3:
            self._build_field_mode3(bands)
        elif field_mode == 4:
            self._build_field_mode4(bands)
        else:
            self._build_field_mode5(bands)
        
        # Apply field scale
        self.field_x *= field_scale
        self.field_y *= field_scale
        
        # Add injection
        self.field_x += inject_x
        self.field_y += inject_y
        
        # Add raw EEG as global perturbation
        self.field_x += raw * 0.5
        self.field_y += raw * 0.5
        
        # Move particles
        velocities_list = []
        for i in range(len(self.particles)):
            px = int(np.clip(self.particles[i, 0], 0, self.size - 1))
            py = int(np.clip(self.particles[i, 1], 0, self.size - 1))
            
            # Get field at particle position
            vx = self.field_x[py, px] * speed
            vy = self.field_y[py, px] * speed
            
            # Apply momentum
            vx = self.velocities[i, 0] * momentum + vx * (1 - momentum)
            vy = self.velocities[i, 1] * momentum + vy * (1 - momentum)
            
            # Limit speed
            spd = np.sqrt(vx*vx + vy*vy)
            if spd > 10:
                vx *= 10 / spd
                vy *= 10 / spd
            
            self.velocities[i] = [vx, vy]
            velocities_list.append([vx, vy])
            
            # Update position
            self.particles[i, 0] += vx
            self.particles[i, 1] += vy
            
            # Wrap at boundaries (periodic)
            self.particles[i, 0] = self.particles[i, 0] % self.size
            self.particles[i, 1] = self.particles[i, 1] % self.size
            
            # Draw to trail buffer
            px = int(self.particles[i, 0])
            py = int(self.particles[i, 1])
            if 0 <= px < self.size and 0 <= py < self.size:
                self.trail_buffer[py, px] = 1.0
        
        # Decay trail
        self.trail_buffer *= decay
        
        # Compute FFT of trail buffer
        self.fft_result = np.fft.fft2(self.trail_buffer)
        self.fft_result = np.fft.fftshift(self.fft_result)
        
        self.magnitude = np.abs(self.fft_result)
        self.phase = np.angle(self.fft_result)
        
        # Compute metrics
        self._compute_metrics(velocities_list)
        
    def _compute_metrics(self, velocities_list):
        """Compute spectral and flow metrics"""
        
        # Dominant frequency (peak in magnitude, excluding DC)
        mag_copy = self.magnitude.copy()
        mag_copy[self.half-2:self.half+3, self.half-2:self.half+3] = 0  # zero DC region
        peak_idx = np.unravel_index(np.argmax(mag_copy), mag_copy.shape)
        self.dominant_freq = self.freq_r[peak_idx]
        
        # Spectral entropy
        mag_norm = self.magnitude / (np.sum(self.magnitude) + 1e-10)
        mag_flat = mag_norm.flatten()
        mag_flat = mag_flat[mag_flat > 1e-10]
        self.spectral_entropy = -np.sum(mag_flat * np.log(mag_flat))
        self.spectral_entropy = self.spectral_entropy / np.log(len(mag_flat))  # normalize to 0-1
        
        # Eigenmode centroid (average frequency weighted by magnitude)
        total_mag = np.sum(self.magnitude) + 1e-10
        self.eigenmode_centroid = np.sum(self.freq_r * self.magnitude) / total_mag
        
        # Flow coherence
        if len(velocities_list) > 1:
            vels = np.array(velocities_list)
            mean_vel = np.mean(vels, axis=0)
            mean_speed = np.linalg.norm(mean_vel)
            avg_speed = np.mean(np.linalg.norm(vels, axis=1)) + 1e-6
            self.flow_coherence = mean_speed / avg_speed
        else:
            self.flow_coherence = 0.0
            
    def get_output(self, port_name):
        if port_name == 'flow_image':
            # Colorize trail buffer
            img = np.stack([
                self.trail_buffer * 0.3,
                self.trail_buffer * 0.8,
                self.trail_buffer * 1.0
            ], axis=-1)
            return np.clip(img, 0, 1).astype(np.float32)
            
        elif port_name == 'fft_magnitude':
            if self.magnitude is None:
                return np.zeros((self.size, self.size, 3), dtype=np.float32)
            # Log scale for visibility
            mag_log = np.log(self.magnitude + 1)
            mag_norm = mag_log / (np.max(mag_log) + 1e-6)
            # Colormap
            colored = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            return colored.astype(np.float32) / 255.0
            
        elif port_name == 'fft_phase':
            if self.phase is None:
                return np.zeros((self.size, self.size, 3), dtype=np.float32)
            # Phase to 0-1
            phase_norm = (self.phase + np.pi) / (2 * np.pi)
            colored = cv2.applyColorMap((phase_norm * 255).astype(np.uint8), cv2.COLORMAP_HSV)
            return colored.astype(np.float32) / 255.0
            
        elif port_name == 'eigenmode_image':
            if self.magnitude is None or self.phase is None:
                return np.zeros((self.size, self.size, 3), dtype=np.float32)
            # Magnitude as brightness, phase as hue
            mag_log = np.log(self.magnitude + 1)
            mag_norm = mag_log / (np.max(mag_log) + 1e-6)
            phase_norm = (self.phase + np.pi) / (2 * np.pi)
            
            # HSV: phase=hue, 1=sat, magnitude=value
            hsv = np.stack([
                (phase_norm * 180).astype(np.uint8),
                np.ones_like(mag_norm, dtype=np.uint8) * 255,
                (mag_norm * 255).astype(np.uint8)
            ], axis=-1)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return rgb.astype(np.float32) / 255.0
            
        elif port_name == 'complex_spectrum':
            return self.fft_result
            
        elif port_name == 'dominant_frequency':
            return float(self.dominant_freq)
            
        elif port_name == 'spectral_entropy':
            return float(self.spectral_entropy)
            
        elif port_name == 'flow_coherence':
            return float(self.flow_coherence)
            
        elif port_name == 'eigenmode_centroid':
            return float(self.eigenmode_centroid)
            
        return None
    
    def draw_custom(self, painter):
        """Show current state"""
        painter.setPen(QtGui.QColor(200, 255, 255))
        painter.setFont(QtGui.QFont("Consolas", 8))
        
        info = f"P:{len(self.particles) if self.particles is not None else 0}"
        info += f" Coh:{self.flow_coherence:.2f}"
        info += f" Ent:{self.spectral_entropy:.2f}"
        
        painter.drawText(5, self.height - 25, info)


class EEGFlowFourierCompactNode(BaseNode):
    """
    Simplified version - fewer inputs, good defaults
    Just wire EEG and explore
    """
    NODE_CATEGORY = "IHT_Core"
    NODE_COLOR = QtGui.QColor(80, 160, 220)
    
    def __init__(self, size=256):
        super().__init__()
        self.node_title = "EEG→Flow→FFT"
        
        self.inputs = {
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal', 
            'beta': 'signal',
            'gamma': 'signal',
            'mode': 'signal',      # 0-5 field modes
            'init': 'signal',      # 0-9 init patterns  
            'reset': 'signal',
        }
        
        self.outputs = {
            'flow': 'image',
            'fft': 'image',
            'spectrum': 'complex_spectrum',
            'entropy': 'signal',
            'coherence': 'signal',
        }
        
        self.size = int(size)
        self.half = self.size // 2
        
        # Particle system - moderate count for good patterns
        self.particle_count = 400
        self.particles = None
        self.velocities = None
        
        # Buffers
        self.trail = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Precomputed grids
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cx, self.cy = self.size/2, self.size/2
        self.r = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        self.theta = np.arctan2(y - self.cy, x - self.cx)
        
        # FFT frequency grid
        fx = np.fft.fftfreq(self.size)
        fy = np.fft.fftfreq(self.size)
        self.freq_x, self.freq_y = np.meshgrid(fx, fy)
        self.freq_r = np.sqrt(self.freq_x**2 + self.freq_y**2)
        
        # Outputs
        self.fft_result = None
        self.entropy = 0.0
        self.coherence = 0.0
        
        # State
        self.last_init = -1
        self.last_reset = 0.0
        
        self._init_particles(0)
        
    def _init_particles(self, mode):
        n = self.particle_count
        mode = int(mode) % 10
        
        if mode == 0:
            self.particles = np.random.rand(n, 2) * self.size
        elif mode == 1:
            t = np.linspace(0.05, 0.95, n)
            self.particles = np.stack([t * self.size, np.ones(n) * self.cy], axis=1)
        elif mode == 2:
            t = np.linspace(0.05, 0.95, n)
            self.particles = np.stack([np.ones(n) * self.cx, t * self.size], axis=1)
        elif mode == 3:
            a = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = self.size * 0.4
            self.particles = np.stack([self.cx + np.cos(a)*r, self.cy + np.sin(a)*r], axis=1)
        elif mode == 4:
            side = int(np.sqrt(n))
            xs = np.linspace(0.1, 0.9, side) * self.size
            ys = np.linspace(0.1, 0.9, side) * self.size
            xx, yy = np.meshgrid(xs, ys)
            self.particles = np.stack([xx.flatten(), yy.flatten()], axis=1)[:n]
        elif mode == 5:
            a = np.random.rand(n) * 2 * np.pi
            r = np.random.rand(n) * 5
            self.particles = np.stack([self.cx + np.cos(a)*r, self.cy + np.sin(a)*r], axis=1)
        elif mode == 6:
            t = np.linspace(0.05, 0.95, n)
            self.particles = np.stack([t * self.size, t * self.size], axis=1)
        elif mode == 7:
            half = n // 2
            t1 = np.linspace(0.05, 0.95, half)
            t2 = np.linspace(0.05, 0.95, n - half)
            p1 = np.stack([t1 * self.size, np.ones(half) * self.cy], axis=1)
            p2 = np.stack([np.ones(n-half) * self.cx, t2 * self.size], axis=1)
            self.particles = np.vstack([p1, p2])
        elif mode == 8:
            t = np.linspace(0, 6*np.pi, n)
            r = np.linspace(5, self.size * 0.45, n)
            self.particles = np.stack([self.cx + np.cos(t)*r, self.cy + np.sin(t)*r], axis=1)
        else:
            self.particles = np.random.rand(min(n, 30), 2) * self.size
            
        self.velocities = np.zeros((len(self.particles), 2), dtype=np.float32)
        self.trail *= 0
        
    def step(self):
        # Get bands
        d = self.get_blended_input('delta', 'sum') or 0.0
        t = self.get_blended_input('theta', 'sum') or 0.0
        a = self.get_blended_input('alpha', 'sum') or 0.0
        b = self.get_blended_input('beta', 'sum') or 0.0
        g = self.get_blended_input('gamma', 'sum') or 0.0
        
        mode = self.get_blended_input('mode', 'sum') or 0.0
        mode = int(np.clip((mode + 1) * 3, 0, 5))
        
        init = self.get_blended_input('init', 'sum') or 0.0
        init = int(np.clip((init + 1) * 5, 0, 9))
        
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        # Reinit check
        if (reset > 0.5 and self.last_reset <= 0.5) or init != self.last_init:
            self._init_particles(init)
        self.last_init = init
        self.last_reset = reset
        
        # Build field based on mode (simplified versions)
        if mode == 0:  # Radial
            field = d * np.sin(self.r * 0.02) + t * np.sin(self.r * 0.05) + a * np.sin(self.r * 0.1) + b * np.sin(self.r * 0.2) + g * np.sin(self.r * 0.4)
            fx = -np.sin(self.theta) * field
            fy = np.cos(self.theta) * field
        elif mode == 1:  # Cartesian
            fx = d * np.sin(self.y * 0.03) + a * np.sin(self.y * 0.08) + g * np.sin(self.y * 0.2)
            fy = t * np.sin(self.x * 0.05) + b * np.sin(self.x * 0.15)
        elif mode == 2:  # Vortex
            rot = d * np.exp(-self.r**2/(self.size*0.2)**2) + a * np.exp(-(self.r-self.size*0.3)**2/(self.size*0.15)**2)
            fx = -np.sin(self.theta) * rot
            fy = np.cos(self.theta) * rot
        elif mode == 3:  # Diagonal
            diag1, diag2 = self.x + self.y, self.x - self.y
            w1 = d * np.sin(diag1 * 0.02) + a * np.sin(diag1 * 0.06)
            w2 = t * np.sin(diag2 * 0.03) + b * np.sin(diag2 * 0.1)
            fx, fy = w1 + w2, w1 - w2
        else:  # Turbulent
            fx = d * np.sin(self.x * 0.02) * np.cos(self.y * 0.01) + g * np.sin(self.x * 0.16)
            fy = t * np.cos(self.x * 0.01) * np.sin(self.y * 0.04) + b * np.sin(self.y * 0.08)
        
        # Move particles
        vels = []
        for i in range(len(self.particles)):
            px = int(np.clip(self.particles[i, 0], 0, self.size-1))
            py = int(np.clip(self.particles[i, 1], 0, self.size-1))
            
            vx = self.velocities[i, 0] * 0.3 + fx[py, px] * 0.7
            vy = self.velocities[i, 1] * 0.3 + fy[py, px] * 0.7
            
            spd = np.sqrt(vx*vx + vy*vy)
            if spd > 8:
                vx, vy = vx * 8/spd, vy * 8/spd
                
            self.velocities[i] = [vx, vy]
            vels.append([vx, vy])
            
            self.particles[i] += [vx, vy]
            self.particles[i] = self.particles[i] % self.size
            
            px = int(self.particles[i, 0])
            py = int(self.particles[i, 1])
            if 0 <= px < self.size and 0 <= py < self.size:
                self.trail[py, px] = 1.0
        
        self.trail *= 0.93
        
        # FFT
        self.fft_result = np.fft.fftshift(np.fft.fft2(self.trail))
        mag = np.abs(self.fft_result)
        
        # Entropy
        mag_norm = mag / (np.sum(mag) + 1e-10)
        mag_flat = mag_norm.flatten()
        mag_flat = mag_flat[mag_flat > 1e-10]
        self.entropy = -np.sum(mag_flat * np.log(mag_flat)) / np.log(len(mag_flat))
        
        # Coherence
        if len(vels) > 1:
            v = np.array(vels)
            self.coherence = np.linalg.norm(np.mean(v, axis=0)) / (np.mean(np.linalg.norm(v, axis=1)) + 1e-6)
        
    def get_output(self, port_name):
        if port_name == 'flow':
            return np.stack([self.trail*0.3, self.trail*0.8, self.trail], axis=-1).astype(np.float32)
        elif port_name == 'fft':
            if self.fft_result is None:
                return np.zeros((self.size, self.size, 3), dtype=np.float32)
            mag = np.log(np.abs(self.fft_result) + 1)
            mag = mag / (np.max(mag) + 1e-6)
            return cv2.applyColorMap((mag * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS).astype(np.float32) / 255.0
        elif port_name == 'spectrum':
            return self.fft_result
        elif port_name == 'entropy':
            return float(self.entropy)
        elif port_name == 'coherence':
            return float(self.coherence)
        return None