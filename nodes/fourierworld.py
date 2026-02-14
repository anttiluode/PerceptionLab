"""
Fourier World Node (The Scribe)
--------------------------------
The mandala is not displayed. It is read as a musical score.

Each radial frequency mode from the input becomes a rotating arm
in a Fourier epicycle machine. The arms trace paths on a persistent
canvas. New inputs change the arm configuration — lengths, speeds,
phases — so the drawing style evolves continuously.

Unlike reaction-diffusion (which converges), this NEVER settles.
The arms keep spinning, the paths keep accumulating, the world
keeps growing in complexity forever.

Multiple drawing heads run in parallel, each seeded from a different
frequency band of the input. They interfere on the shared canvas.

Pipeline: EEG/Noise → EigenToImage (mandala) → THIS NODE → ever-growing world
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class FourierWorldNode(BaseNode):
    NODE_CATEGORY = "World"
    NODE_COLOR = QtGui.QColor(140, 60, 120)  # Purple — the scribe
    
    def __init__(self):
        super().__init__()
        self.node_title = "Fourier World"
        
        self.inputs = {
            'pattern_in': 'image',       # Mandala / eigenstructure
            'energy': 'signal',          # Drive intensity
            'color_signal': 'signal'     # Modulates hue
        }
        
        self.outputs = {
            'world': 'image',            # The accumulated canvas
            'trace': 'image',            # Current frame's trace only
            'complexity': 'signal',      # How much has been drawn
            'arm_image': 'image'         # Visualization of the arm state
        }
        
        # Canvas
        self.size = 512
        # 3 layers: R, G, B accumulation (float for precision)
        self.canvas = np.zeros((self.size, self.size, 3), dtype=np.float64)
        self.trace_frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Drawing heads — each is an array of Fourier arms
        self.n_heads = 4
        self.max_arms = 24  # Per head
        
        # Arm state: [amplitude, frequency, phase] per arm per head
        self.arms = np.zeros((self.n_heads, self.max_arms, 3), dtype=np.float64)
        # Position history for each head (for trail drawing)
        self.head_pos = np.zeros((self.n_heads, 2), dtype=np.float64)  # current x,y
        self.head_prev = np.zeros((self.n_heads, 2), dtype=np.float64)
        
        # Time
        self.t = 0.0
        self.dt = 0.02
        self.age = 0
        
        # Drawing params
        self.line_alpha = 0.6       # How bright each stroke is
        self.decay = 0.9995          # Canvas slowly fades (almost not at all)
        self.n_substeps = 12        # Drawing substeps per frame for smooth lines
        self.trail_thickness = 1
        
        # Head colors (hue offsets)
        self.head_hues = [0.0, 0.25, 0.5, 0.75]  # Spread across hue wheel
        self.hue_shift = 0.0
        
        # Complexity
        self.complexity_value = 0.0
        self.total_ink = 0.0
        
        # Initialize arms with some defaults so it draws immediately
        self._init_default_arms()
    
    def _init_default_arms(self):
        """Give each head a different spiral pattern to start"""
        for h in range(self.n_heads):
            n_active = 8 + h * 2
            for i in range(min(n_active, self.max_arms)):
                # Amplitude decreases with arm index (harmonic series)
                self.arms[h, i, 0] = self.size * 0.12 / (i + 1)
                # Frequency: each arm spins at a different rate
                self.arms[h, i, 1] = (i + 1) * (1.0 + h * 0.3)
                # Phase: offset per head for variety
                self.arms[h, i, 2] = h * np.pi / 2 + i * 0.5
    
    def _extract_arms_from_pattern(self, pattern):
        """
        Decompose input mandala into Fourier arm parameters.
        
        The mandala's radial frequency content becomes arm amplitudes.
        The angular content becomes phase offsets.
        Different quadrants of the FFT feed different heads.
        """
        if pattern is None:
            return
            
        if pattern.ndim == 3:
            pattern = np.mean(pattern, axis=2)
        
        p = pattern.astype(np.float64)
        pmax = np.max(np.abs(p))
        if pmax > 1e-10:
            p = p / pmax
        else:
            return
        
        # Resize to manageable size for FFT
        p_small = cv2.resize(p, (64, 64), interpolation=cv2.INTER_AREA)
        
        # FFT
        fft = np.fft.fftshift(np.fft.fft2(p_small))
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Extract radial profile (amplitudes for arms)
        cy, cx = 32, 32
        
        for h in range(min(self.n_heads, self.arms.shape[0])):
            # Each head reads from a different angular sector
            angle_start = h * np.pi / 2
            angle_end = angle_start + np.pi / 2
            
            # Sample along a radial line in this sector
            for i in range(min(self.max_arms, 20)):
                r = i + 1
                angle = angle_start + (angle_end - angle_start) * (i / 20.0)
                
                # Sample FFT at this frequency
                fx = int(cx + r * np.cos(angle))
                fy = int(cy + r * np.sin(angle))
                
                if 0 <= fx < 64 and 0 <= fy < 64:
                    mag = magnitude[fy, fx]
                    ph = phase[fy, fx]
                    
                    # Smooth blend into current arms (don't snap — morph)
                    target_amp = mag * self.size * 0.003 / (i + 1)
                    target_freq = (i + 1) * (0.5 + mag * 2.0)
                    target_phase = ph
                    
                    # Interpolate (the arms "learn" the new pattern gradually)
                    blend = 0.03  # Slow morphing
                    self.arms[h, i, 0] += blend * (target_amp - self.arms[h, i, 0])
                    self.arms[h, i, 1] += blend * (target_freq - self.arms[h, i, 1])
                    # Phase wrapping
                    dp = target_phase - self.arms[h, i, 2]
                    dp = (dp + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                    self.arms[h, i, 2] += blend * dp
    
    def _compute_head_position(self, head_idx):
        """
        Sum all arms for a given head to get the drawing position.
        This is the classic Fourier epicycle: each arm tip is the 
        center of the next arm's rotation.
        """
        if head_idx >= self.arms.shape[0]:
            return self.size / 2.0, self.size / 2.0
            
        x = self.size / 2.0
        y = self.size / 2.0
        
        n_arms = min(self.max_arms, self.arms.shape[1])
        for i in range(n_arms):
            amp = self.arms[head_idx, i, 0]
            freq = self.arms[head_idx, i, 1]
            phase = self.arms[head_idx, i, 2]
            
            if amp < 0.1:  # Skip negligible arms
                continue
            
            angle = freq * self.t + phase
            x += amp * np.cos(angle)
            y += amp * np.sin(angle)
        
        # Wrap positions to stay on canvas (toroidal)
        x = x % self.size
        y = y % self.size
        
        return x, y
    
    def _hue_to_rgb(self, hue, saturation=0.8, value=0.9):
        """Convert HSV to RGB (0-1 range)"""
        hue = hue % 1.0
        c = value * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = value - c
        
        if hue < 1/6:
            r, g, b = c, x, 0
        elif hue < 2/6:
            r, g, b = x, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x
        elif hue < 4/6:
            r, g, b = 0, x, c
        elif hue < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return r + m, g + m, b + m
    
    def _draw_line_on_canvas(self, x0, y0, x1, y1, color_rgb, alpha):
        """Draw an antialiased line onto the float canvas"""
        # Clamp to canvas bounds
        ix0 = int(np.clip(x0, 0, self.size - 1))
        iy0 = int(np.clip(y0, 0, self.size - 1))
        ix1 = int(np.clip(x1, 0, self.size - 1))
        iy1 = int(np.clip(y1, 0, self.size - 1))
        
        # Use OpenCV to draw on a temp layer
        temp = np.zeros((self.size, self.size, 3), dtype=np.float64)
        cv2.line(temp, (ix0, iy0), (ix1, iy1), color_rgb, 
                 self.trail_thickness, cv2.LINE_AA)
        
        # Additive blend onto canvas
        self.canvas += temp * alpha
    
    def step(self):
        # Get inputs
        pattern = self.get_blended_input('pattern_in', 'first')
        energy = self.get_blended_input('energy', 'sum')
        color_sig = self.get_blended_input('color_signal', 'sum')
        
        # Safety: ensure n_heads matches actual array sizes
        actual_heads = min(self.n_heads, self.arms.shape[0], 
                          self.head_pos.shape[0], len(self.head_hues))
        
        # Update arms from pattern
        if pattern is not None:
            self._extract_arms_from_pattern(pattern)
        
        # Energy modulates drawing speed and intensity
        if energy is not None and isinstance(energy, (int, float)):
            energy_val = abs(float(energy))
            self.dt = 0.01 + energy_val * 0.05
            self.line_alpha = 0.3 + energy_val * 0.5
        
        # Color signal shifts hue
        if color_sig is not None and isinstance(color_sig, (int, float)):
            self.hue_shift = float(color_sig) * 0.1
        
        # Slow decay of canvas (prevents total whiteout)
        self.canvas *= self.decay
        
        # Clear trace frame
        self.trace_frame[:] = 0
        
        # Sub-step drawing for smooth lines
        dt_sub = self.dt / self.n_substeps
        
        for step in range(self.n_substeps):
            for h in range(actual_heads):
                # Save previous position
                prev_x, prev_y = self.head_pos[h]
                
                # Compute new position
                new_x, new_y = self._compute_head_position(h)
                self.head_pos[h] = [new_x, new_y]
                
                # Skip first frame (no previous position)
                if self.age == 0 and step == 0:
                    continue
                
                # Skip if both points are the same (no movement)
                if abs(new_x - prev_x) < 0.01 and abs(new_y - prev_y) < 0.01:
                    continue
                
                # Color for this head
                hue = self.head_hues[h] + self.hue_shift + self.t * 0.001
                r, g, b = self._hue_to_rgb(hue)
                color = (r, g, b)
                
                # Draw line from previous to current
                self._draw_line_on_canvas(
                    prev_x, prev_y, new_x, new_y,
                    color, self.line_alpha / self.n_substeps
                )
                
                # Also draw on trace frame (current frame only)
                ix0 = int(np.clip(prev_x, 0, self.size-1))
                iy0 = int(np.clip(prev_y, 0, self.size-1))
                ix1 = int(np.clip(new_x, 0, self.size-1))
                iy1 = int(np.clip(new_y, 0, self.size-1))
                trace_color = (int(r*255), int(g*255), int(b*255))
                cv2.line(self.trace_frame, (ix0, iy0), (ix1, iy1),
                        trace_color, 1, cv2.LINE_AA)
            
            self.t += dt_sub
        
        # Track complexity (total ink on canvas)
        self.total_ink = np.sum(self.canvas)
        self.complexity_value = float(np.log1p(self.total_ink))
        
        self.age += 1
    
    def _get_canvas_uint8(self):
        """Convert float canvas to display image"""
        # Soft clamp with tanh to prevent blowout while preserving detail
        display = np.tanh(self.canvas * 0.5) 
        display = (display * 255).astype(np.uint8)
        return display
    
    def get_output(self, port_name):
        if port_name == 'world':
            img = self._get_canvas_uint8()
            # Return grayscale version
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return gray
        elif port_name == 'trace':
            return cv2.cvtColor(self.trace_frame, cv2.COLOR_RGB2GRAY)
        elif port_name == 'complexity':
            return self.complexity_value
        elif port_name == 'arm_image':
            return self._render_arms()
        return None
    
    def _render_arms(self):
        """Small visualization of the current arm configuration"""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cx, cy = 64, 64
        scale = 0.4
        
        for h in range(min(self.n_heads, self.arms.shape[0], len(self.head_hues))):
            x, y = float(cx), float(cy)
            hue = self.head_hues[h]
            r, g, b = self._hue_to_rgb(hue)
            color = (int(r*200), int(g*200), int(b*200))
            
            for i in range(self.max_arms):
                amp = self.arms[h, i, 0] * scale
                freq = self.arms[h, i, 1]
                phase = self.arms[h, i, 2]
                
                if amp < 0.3:
                    continue
                
                angle = freq * self.t + phase
                nx = x + amp * np.cos(angle)
                ny = y + amp * np.sin(angle)
                
                cv2.line(img, (int(x), int(y)), (int(nx), int(ny)), 
                        color, 1, cv2.LINE_AA)
                x, y = nx, ny
            
            # Draw head position dot
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
        
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def get_display_image(self):
        """Show the accumulated canvas"""
        display = self._get_canvas_uint8()
        
        # Resize for display
        display = cv2.resize(display, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Add info overlay
        cv2.putText(display, f"t={self.age}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"C={self.complexity_value:.1f}", (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
        
        h, w = display.shape[:2]
        return QtGui.QImage(display.data, w, h, w * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Canvas Size", "size", self.size, None),
            ("Num Heads", "n_heads", self.n_heads, None),
            ("Max Arms/Head", "max_arms", self.max_arms, None),
            ("Line Alpha", "line_alpha", self.line_alpha, "float"),
            ("Canvas Decay", "decay", self.decay, "float"),
            ("Sub-steps", "n_substeps", self.n_substeps, None),
            ("Trail Width", "trail_thickness", self.trail_thickness, None),
            ("Time Step", "dt", self.dt, "float"),
        ]
    
    def set_config_options(self, options):
        rebuild = False
        if "size" in options:
            new_size = int(options["size"])
            if new_size != self.size:
                self.size = new_size
                rebuild = True
        if "n_heads" in options:
            new_heads = int(options["n_heads"])
            if new_heads != self.n_heads:
                self.n_heads = min(new_heads, 8)
                self.arms = np.zeros((self.n_heads, self.max_arms, 3))
                self.head_pos = np.zeros((self.n_heads, 2))
                self.head_prev = np.zeros((self.n_heads, 2))
                self.head_hues = [i / self.n_heads for i in range(self.n_heads)]
                self._init_default_arms()
        if "max_arms" in options:
            self.max_arms = int(options["max_arms"])
        if "line_alpha" in options:
            self.line_alpha = float(options["line_alpha"])
        if "decay" in options:
            self.decay = float(options["decay"])
        if "n_substeps" in options:
            self.n_substeps = max(1, int(options["n_substeps"]))
        if "trail_thickness" in options:
            self.trail_thickness = max(1, int(options["trail_thickness"]))
        if "dt" in options:
            self.dt = float(options["dt"])
        
        if rebuild:
            self.canvas = np.zeros((self.size, self.size, 3), dtype=np.float64)
            self.trace_frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            self.head_pos = np.zeros((self.n_heads, 2))
            self.age = 0
            self.total_ink = 0.0