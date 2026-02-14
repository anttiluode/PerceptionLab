"""
Morphogenesis World Node (The Garden)
--------------------------------------
Takes 2D mandala/eigenstructure patterns and grows them into a living world.

The mandala is not displayed AS a mandala. Instead, it is treated as a
"seed instruction" — a frequency-space blueprint that drives a reaction-diffusion
system. The moiré interference patterns from the input act as morphogen 
gradients, causing structures to branch, fold, and differentiate.

Think of it as: the mandala is the DNA, this node is the embryo.

Input mandala → FFT → extract radial modes as morphogen concentrations
→ reaction-diffusion on a 2D grid → structures emerge, complexify, branch

Over time, the world accumulates structure. New inputs perturb it.
The world never resets — it only grows.

Pipeline: Noise → ImageToVector → EigenToImage (mandala) → THIS NODE → world
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, uniform_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class MorphogenesisWorldNode(BaseNode):
    NODE_CATEGORY = "World"
    NODE_COLOR = QtGui.QColor(60, 140, 80)  # Green — life
    
    def __init__(self):
        super().__init__()
        self.node_title = "Morphogenesis"
        
        self.inputs = {
            'pattern_in': 'image',       # Mandala / eigenstructure image
            'energy': 'signal',           # Optional: external energy/drive signal
            'perturbation': 'image'       # Optional: spatial perturbation field
        }
        
        self.outputs = {
            'world': 'image',            # The living world — main output
            'activator': 'image',        # Raw activator field (U)
            'inhibitor': 'image',        # Raw inhibitor field (V)  
            'complexity': 'signal',      # Scalar: how complex is the world now
            'morphogen_map': 'image'     # Where the input pattern is driving growth
        }
        
        # World grid
        self.size = 256
        
        # Reaction-diffusion state (Gray-Scott model)
        # U = activator, V = inhibitor
        self.U = np.ones((self.size, self.size), dtype=np.float64)
        self.V = np.zeros((self.size, self.size), dtype=np.float64)
        
        # Seed: small square of V in the center to bootstrap
        c = self.size // 2
        r = 8
        self.U[c-r:c+r, c-r:c+r] = 0.50
        self.V[c-r:c+r, c-r:c+r] = 0.25
        
        # Gray-Scott parameters (these get modulated by the input pattern)
        self.feed_base = 0.037      # F: feed rate — how fast U is replenished
        self.kill_base = 0.060      # k: kill rate — how fast V decays
        self.Du = 0.16              # Diffusion rate of activator
        self.Dv = 0.08              # Diffusion rate of inhibitor
        self.dt = 1.0               # Time step
        self.steps_per_frame = 8    # Sub-steps per visual frame
        
        # Morphogen field — derived from input pattern
        # This spatially modulates F and k across the grid
        self.morphogen = np.zeros((self.size, self.size), dtype=np.float64)
        self.feed_field = np.full((self.size, self.size), self.feed_base)
        self.kill_field = np.full((self.size, self.size), self.kill_base)
        
        # Modulation strength: how much does the input pattern affect F and k
        self.pattern_strength = 0.012
        
        # Complexity tracking
        self.complexity_value = 0.0
        self.age = 0
        
        # Display
        self.world_display = np.zeros((self.size, self.size, 3), dtype=np.uint8)
    
    def _laplacian(self, field):
        """Discrete Laplacian via convolution (5-point stencil)"""
        # Using uniform_filter is faster than explicit convolution
        # Laplacian ≈ neighbor_average - center
        avg = uniform_filter(field, size=3, mode='wrap')
        return avg - field
    
    def _extract_morphogen(self, pattern):
        """
        Convert input mandala/pattern into a morphogen gradient field.
        
        The key insight: we don't use the pattern as pixels.
        We decompose it into radial frequency modes and use those
        as spatially-varying chemical concentrations that drive
        the reaction-diffusion system.
        
        Low-frequency modes → large-scale feed rate gradients (where things grow)
        High-frequency modes → local kill rate perturbations (where things branch)
        """
        if pattern is None:
            return
        
        # Handle dimensions
        if pattern.ndim == 3:
            pattern = np.mean(pattern, axis=2)
        
        # Normalize to 0-1
        p = pattern.astype(np.float64)
        pmax = np.max(np.abs(p))
        if pmax > 0:
            p = p / pmax
        
        # Resize to world grid
        p_resized = cv2.resize(p, (self.size, self.size), 
                               interpolation=cv2.INTER_LINEAR)
        
        # Decompose into frequency bands via FFT
        fft = np.fft.fftshift(np.fft.fft2(p_resized))
        magnitude = np.abs(fft)
        
        # Create radial distance map in frequency space
        cy, cx = self.size // 2, self.size // 2
        yy, xx = np.ogrid[:self.size, :self.size]
        r_freq = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        # Low-frequency band (large-scale structure) → modulates feed rate
        low_mask = (r_freq < self.size * 0.1).astype(np.float64)
        low_band = np.abs(np.fft.ifft2(np.fft.ifftshift(fft * low_mask)))
        
        # High-frequency band (fine detail) → modulates kill rate  
        high_mask = (r_freq > self.size * 0.1).astype(np.float64)
        high_band = np.abs(np.fft.ifft2(np.fft.ifftshift(fft * high_mask)))
        
        # Normalize bands
        if low_band.max() > 0:
            low_band = low_band / low_band.max()
        if high_band.max() > 0:
            high_band = high_band / high_band.max()
        
        # Blend into morphogen field with momentum (don't overwrite — accumulate)
        self.morphogen = 0.95 * self.morphogen + 0.05 * (low_band + high_band * 0.5)
        
        # Clamp morphogen
        self.morphogen = np.clip(self.morphogen, 0, 1)
        
        # Update spatially-varying parameters
        # Feed rate: higher where low-frequency morphogen is strong
        # (more growth in areas the mandala's broad structure indicates)
        self.feed_field = self.feed_base + self.pattern_strength * low_band
        
        # Kill rate: modulated by high-frequency content
        # (more branching/complexity where the mandala has fine detail)
        self.kill_field = self.kill_base + self.pattern_strength * 0.5 * high_band
    
    def _reaction_diffusion_step(self):
        """
        One step of Gray-Scott reaction-diffusion.
        
        dU/dt = Du * ∇²U - U*V² + F*(1-U)
        dV/dt = Dv * ∇²V + U*V² - (F+k)*V
        
        F and k are spatially varying (driven by the morphogen field).
        """
        lap_U = self._laplacian(self.U)
        lap_V = self._laplacian(self.V)
        
        uvv = self.U * self.V * self.V
        
        dU = self.Du * lap_U - uvv + self.feed_field * (1.0 - self.U)
        dV = self.Dv * lap_V + uvv - (self.feed_field + self.kill_field) * self.V
        
        self.U += dU * self.dt
        self.V += dV * self.dt
        
        # Clamp to valid range
        self.U = np.clip(self.U, 0.0, 1.0)
        self.V = np.clip(self.V, 0.0, 1.0)
    
    def _seed_from_pattern(self, pattern):
        """
        Occasionally inject new V seeds at locations where the 
        morphogen field is strong but V is low.
        This is how the mandala's structure propagates into the world —
        not as a direct image, but as locations where new growth begins.
        """
        if self.age % 30 != 0:  # Only seed every 30 frames
            return
        
        # Find high-morphogen, low-V regions (fertile ground)
        fertile = (self.morphogen > 0.3) & (self.V < 0.1)
        
        # Randomly seed a small fraction of fertile pixels
        seed_mask = fertile & (np.random.random((self.size, self.size)) < 0.005)
        
        if np.any(seed_mask):
            # Plant seeds: small V deposits
            self.V[seed_mask] = 0.25
            self.U[seed_mask] = 0.50
    
    def _compute_complexity(self):
        """
        Estimate structural complexity of the world.
        Uses gradient energy — more edges/structure = higher complexity.
        """
        grad_x = np.diff(self.V, axis=1)
        grad_y = np.diff(self.V, axis=0)
        
        # Gradient energy
        energy = np.mean(grad_x**2) + np.mean(grad_y**2)
        
        # Spectral complexity: number of significant frequency components
        fft_v = np.abs(np.fft.fft2(self.V))
        threshold = fft_v.max() * 0.01
        n_modes = np.sum(fft_v > threshold)
        spectral = n_modes / fft_v.size
        
        self.complexity_value = float(energy * 1000 + spectral)
    
    def _render_world(self):
        """
        Render U and V fields into a color image.
        
        Color mapping inspired by biological tissue:
        - U dominant (empty space): dark blue-black
        - V dominant (structure): warm colors
        - Transition zones: green/cyan (active growth fronts)
        """
        # Normalize for display
        u_disp = np.clip(self.U, 0, 1)
        v_disp = np.clip(self.V, 0, 1)
        
        # Color channels
        # Red: V concentration (structure)
        # Green: growth fronts (where U and V are both moderate)
        # Blue: empty space (high U, low V)
        
        growth_front = np.exp(-((u_disp - 0.5)**2 + (v_disp - 0.15)**2) / 0.02)
        
        r = (v_disp * 255 * 1.5).astype(np.float64)
        g = (growth_front * 180 + v_disp * 60).astype(np.float64)
        b = ((1.0 - v_disp) * u_disp * 80 + v_disp * 40).astype(np.float64)
        
        # Add morphogen glow (subtle indication of where input drives growth)
        g += self.morphogen * 30
        
        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        self.world_display = np.stack([r, g, b], axis=-1)
    
    def step(self):
        # Get inputs
        pattern = self.get_blended_input('pattern_in', 'first')
        energy = self.get_blended_input('energy', 'sum')
        perturbation = self.get_blended_input('perturbation', 'first')
        
        # Extract morphogen field from input pattern
        if pattern is not None:
            self._extract_morphogen(pattern)
        
        # Optional energy modulation — scales pattern_strength
        if energy is not None and isinstance(energy, (int, float)):
            # Energy drives how aggressively the pattern reshapes the world
            self.pattern_strength = 0.004 + abs(float(energy)) * 0.02
        
        # Optional perturbation — direct kick to V field
        if perturbation is not None:
            if perturbation.ndim == 3:
                perturbation = np.mean(perturbation, axis=2)
            p = cv2.resize(perturbation.astype(np.float64), 
                          (self.size, self.size))
            pmax = np.max(np.abs(p))
            if pmax > 0:
                p = p / pmax
            # Add as small perturbation to V
            self.V += p * 0.01
            self.V = np.clip(self.V, 0, 1)
        
        # Seed new growth from morphogen hotspots
        self._seed_from_pattern(pattern)
        
        # Run reaction-diffusion sub-steps
        for _ in range(self.steps_per_frame):
            self._reaction_diffusion_step()
        
        # Compute complexity metric
        self._compute_complexity()
        
        # Render
        self._render_world()
        
        self.age += 1
    
    def get_output(self, port_name):
        if port_name == 'world':
            # Return grayscale V field as the "world image"
            return (self.V * 255).astype(np.uint8)
        elif port_name == 'activator':
            return (self.U * 255).astype(np.uint8)
        elif port_name == 'inhibitor':
            return (self.V * 255).astype(np.uint8)
        elif port_name == 'complexity':
            return self.complexity_value
        elif port_name == 'morphogen_map':
            return (np.clip(self.morphogen, 0, 1) * 255).astype(np.uint8)
        return None
    
    def get_display_image(self):
        """Show the living world"""
        # Resize for display
        display = cv2.resize(self.world_display, (192, 192), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Add age counter
        cv2.putText(display, f"t={self.age}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add complexity readout
        cv2.putText(display, f"C={self.complexity_value:.1f}", (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
        
        h, w = display.shape[:2]
        return QtGui.QImage(display.data, w, h, w * 3, 
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Grid Size", "size", self.size, None),
            ("Feed Rate (F)", "feed_base", self.feed_base, "float"),
            ("Kill Rate (k)", "kill_base", self.kill_base, "float"),
            ("Diffusion U", "Du", self.Du, "float"),
            ("Diffusion V", "Dv", self.Dv, "float"),
            ("Steps/Frame", "steps_per_frame", self.steps_per_frame, None),
            ("Pattern Strength", "pattern_strength", self.pattern_strength, "float"),
        ]
    
    def set_config_options(self, options):
        rebuild = False
        if "size" in options:
            new_size = int(options["size"])
            if new_size != self.size:
                self.size = new_size
                rebuild = True
        if "feed_base" in options:
            self.feed_base = float(options["feed_base"])
        if "kill_base" in options:
            self.kill_base = float(options["kill_base"])
        if "Du" in options:
            self.Du = float(options["Du"])
        if "Dv" in options:
            self.Dv = float(options["Dv"])
        if "steps_per_frame" in options:
            self.steps_per_frame = int(options["steps_per_frame"])
        if "pattern_strength" in options:
            self.pattern_strength = float(options["pattern_strength"])
        
        if rebuild:
            self.U = np.ones((self.size, self.size), dtype=np.float64)
            self.V = np.zeros((self.size, self.size), dtype=np.float64)
            c = self.size // 2
            r = 8
            self.U[c-r:c+r, c-r:c+r] = 0.50
            self.V[c-r:c+r, c-r:c+r] = 0.25
            self.morphogen = np.zeros((self.size, self.size), dtype=np.float64)
            self.feed_field = np.full((self.size, self.size), self.feed_base)
            self.kill_field = np.full((self.size, self.size), self.kill_base)
            self.age = 0