
"""
Hyper-Vector Scout Node
=======================
Surfing the 1D DNA Ribbon to find 2D Worlds.

Concept:
Imagine an infinite 1D terrain (fractal noise). This node acts as a "sliding window"
over that terrain. The data in the window becomes the Radial Profile for the
EigenToImage projection.

- 'Position' slides the window along the infinite ribbon.
- 'Aperture' (Vector Size) determines how much DNA we fold into the mandala.
- 'Complexity' adds fractal detail to the ribbon itself.

This is a procedural world generator where 1D Time maps to 2D Space.
"""

import numpy as np
import cv2
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, method): return None

class HyperVectorScoutNode(BaseNode):
    NODE_CATEGORY = "Generator"
    NODE_TITLE = "Hyper-Vector Scout"
    NODE_COLOR = QtGui.QColor(255, 100, 150) # Hot Pink / Neon

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'nav_speed': 'signal',   # Speed of sliding along the ribbon
            'zoom_mod': 'signal',    # Modulate aperture size
            'complexity': 'signal',  # Fractal detail of the ribbon
        }
        
        self.outputs = {
            'mandala': 'image',          # The visual world
            'dna_segment': 'spectrum',   # The 1D vector being visualized
            'position': 'signal',        # Where we are on the ribbon
        }
        
        # Internal State
        self.ribbon_length = 100000 # The "Infinite" buffer
        self.ribbon = None
        self.current_pos = 0.0
        self.seed = 42
        
        # DNA Parameters
        self.base_vector_size = 64
        self.octaves = 4
        self.persistence = 0.5
        self.smoothing = 1.0
        self.iterations = 5 # Phase recovery steps
        
        # Pre-generate the world
        self._generate_ribbon()
        
        # Radial mapping cache
        self.output_size = 256
        self._build_radial_map()

    def _generate_ribbon(self):
        """Generates the 1D Fractal Noise Terrain (The DNA)."""
        np.random.seed(self.seed)
        
        # Base layer
        self.ribbon = np.zeros(self.ribbon_length, dtype=np.float32)
        
        # Fractal summation (1D Perlin-ish)
        amp = 1.0
        freq = 0.005
        
        for i in range(self.octaves):
            phase = np.random.uniform(0, 1000)
            t = np.arange(self.ribbon_length) * freq + phase
            self.ribbon += np.sin(t) * amp
            # Add some sharp "events"
            if i == 1:
                self.ribbon += (np.random.rand(self.ribbon_length) - 0.5) * amp * 0.5
                
            amp *= self.persistence
            freq *= 2.0
            
        # Normalize 0..1
        self.ribbon = (self.ribbon - self.ribbon.min()) / (self.ribbon.max() - self.ribbon.min())

    def _build_radial_map(self):
        """Precompute polar map for fast projection."""
        s = self.output_size
        c = s // 2
        y, x = np.ogrid[:s, :s]
        self.r_grid = np.sqrt((x - c)**2 + (y - c)**2)
        self.max_r = c
        
    def _project_dna_to_world(self, dna_segment):
        """The Core Physics: 1D DNA -> 2D Radial Projection."""
        # 1. Expand DNA to match radius
        target_len = int(self.max_r)
        
        # Simple interpolation to fit DNA to radius
        x_src = np.linspace(0, len(dna_segment)-1, len(dna_segment))
        x_dst = np.linspace(0, len(dna_segment)-1, target_len)
        radial_profile = np.interp(x_dst, x_src, dna_segment)
        
        # 2. Project to 2D (Angle-blind)
        # Map radius to profile index
        r_idx = np.clip(self.r_grid.astype(int), 0, target_len-1)
        mandala = radial_profile[r_idx]
        
        return mandala

    def _phase_recovery(self, magnitude, iters):
        """Gerchberg-Saxton to create the hyper-dimensional structure."""
        # Random phase seed
        phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        for _ in range(iters):
            # Freq domain constraint: impose radial magnitude
            spectrum = magnitude * np.exp(1j * phase)
            spatial = np.fft.ifft2(np.fft.ifftshift(spectrum))
            
            # Spatial domain constraint: Real & Positive
            spatial = np.abs(spatial)
            
            # Back to Freq
            new_spec = np.fft.fftshift(np.fft.fft2(spatial))
            phase = np.angle(new_spec)
            
        # Final render
        final = magnitude * np.exp(1j * phase)
        img = np.abs(np.fft.ifft2(np.fft.ifftshift(final)))
        return img

    def step(self):
        # 1. Inputs
        speed = self.get_blended_input('nav_speed', 'sum')
        if speed is None: speed = 0.5
        
        zoom = self.get_blended_input('zoom_mod', 'sum')
        if zoom is None: zoom = 0.0
        
        # 2. Move along the Ribbon
        self.current_pos += speed
        if self.current_pos >= self.ribbon_length - 1000:
            self.current_pos = 0 # Loop the universe
            
        # 3. Determine Aperture (Vector Size)
        # This modulates how much "history" wraps into the circle
        aperture = int(self.base_vector_size + (zoom * 100))
        aperture = max(4, min(aperture, 1024))
        
        # 4. Extract DNA Segment
        start_idx = int(self.current_pos)
        end_idx = start_idx + aperture
        
        # Wrap handling
        if end_idx < self.ribbon_length:
            segment = self.ribbon[start_idx:end_idx]
        else:
            segment = self.ribbon[start_idx:] # Just clamp for edge case
            
        # 5. The Genesis Projection
        # A. Radial Projection (The "WTF" step)
        mandala_mag = self._project_dna_to_world(segment)
        
        # B. Phase Recovery (The "Life" step)
        # This creates the complexity/moiré/structure
        world_img = self._phase_recovery(mandala_mag, self.iterations)
        
        # 6. Post-Process
        # Normalize
        world_img = (world_img - world_img.min()) / (world_img.max() - world_img.min() + 1e-9)
        
        # Store output
        self.output_image = world_img
        self.output_dna = segment

    def get_output(self, port_name):
        if port_name == 'mandala':
            return (self.output_image * 255).astype(np.uint8)
        elif port_name == 'dna_segment':
            return self.output_dna
        elif port_name == 'position':
            return self.current_pos
        return None
        
    def get_display_image(self):
        if not hasattr(self, 'output_image'): return None
        img = (self.output_image * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        return QtGui.QImage(img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Seed", "seed", self.seed, "int"),
            ("Base Vector Size", "base_vector_size", self.base_vector_size, "int"),
            ("Octaves (Roughness)", "octaves", self.octaves, "int"),
            ("Iterations (Clarity)", "iterations", self.iterations, "int"),
        ]
        
    def set_config_options(self, options):
        if "seed" in options:
            self.seed = int(options["seed"])
            self._generate_ribbon()
        if "base_vector_size" in options:
            self.base_vector_size = int(options["base_vector_size"])
        if "octaves" in options:
            self.octaves = int(options["octaves"])
            self._generate_ribbon()
        if "iterations" in options:
            self.iterations = int(options["iterations"])
