"""
Attractor Vision Node
======================
"The attractor doesn't see the field - it sees what's NOT possible."

This node computes what an attractor EXCLUDES - the negative space,
the shadow, the states that are impossible given current dynamics.

The key insight: An attractor perceives by FILTERING. What it filters
out becomes the signal for the next layer. The shadow becomes the prompt.

Three-layer stack:
1. FIELD - The substrate (EEG, waves, dynamics)
2. ATTRACTOR - Emerges from field (stable manifold, constraint satisfaction)
3. PROJECTION - What attractor excludes = vision/thought for next layer

This is how LLMs work too:
- Weights = frozen field
- Attention = attractor dynamics  
- Output = what's LEFT after filtering (exclusion → generation)

The brain does this dynamically - weights aren't frozen, field is alive.

INPUTS:
- field_state: The current field (complex_spectrum or spectrum)
- attractor_state: The current attractor basin (from manifold nodes)
- temperature: How sharp the exclusion boundary is

OUTPUTS:
- excluded_states: The negative space (what's impossible)
- vision_field: The projection (exclusion as signal for next layer)
- exclusion_boundary: The edge between possible/impossible
- attractor_prompt: The shadow formatted as "prompt" for next layer
"""

import numpy as np
import cv2
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode):
            return None


class AttractorVisionNode(BaseNode):
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Attractor Vision"
    NODE_COLOR = QtGui.QColor(200, 100, 180)  # Purple-pink - perception color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Attractor Vision (Exclusion Field)"
        
        self.inputs = {
            'field_state': 'complex_spectrum',    # The living field
            'attractor_basin': 'spectrum',         # Current attractor state
            'constraint_field': 'spectrum',        # What constrains the attractor
            'temperature': 'signal',               # Sharpness of exclusion
            'theta_phase': 'signal',               # Temporal gating
        }
        
        self.outputs = {
            'display': 'image',
            'excluded_field': 'complex_spectrum',  # The negative space
            'vision_field': 'complex_spectrum',    # Exclusion as signal
            'exclusion_boundary': 'image',         # Edge visualization
            'attractor_prompt': 'spectrum',        # Shadow as tokens
            'exclusion_entropy': 'signal',         # How much is excluded
            'vision_clarity': 'signal',            # How clear is the vision
        }
        
        # State
        self.epoch = 0
        self.field_size = 64
        self.embed_dim = 32
        
        # The field and its attractor
        self.current_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.attractor_basin = np.zeros(self.embed_dim)
        
        # Exclusion computation
        self.possible_states = np.ones((self.field_size, self.field_size))
        self.excluded_states = np.zeros((self.field_size, self.field_size))
        self.exclusion_boundary = np.zeros((self.field_size, self.field_size))
        
        # The vision - what emerges from exclusion
        self.vision_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.attractor_prompt = np.zeros(self.embed_dim)
        
        # Metrics
        self.exclusion_entropy = 0.0
        self.vision_clarity = 0.0
        
        # History for temporal dynamics
        self.field_history = deque(maxlen=30)
        self.exclusion_history = deque(maxlen=30)
        
        # Display
        self._display = np.zeros((550, 900, 3), dtype=np.uint8)
    
    def _process_field_input(self, raw_input):
        """Convert various input types to complex field"""
        if raw_input is None:
            return np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        
        if isinstance(raw_input, np.ndarray):
            if raw_input.dtype == np.complex128 or raw_input.dtype == np.complex64:
                if raw_input.ndim == 2:
                    if raw_input.shape != (self.field_size, self.field_size):
                        # Resize
                        mag = np.abs(raw_input)
                        phase = np.angle(raw_input)
                        mag_resized = cv2.resize(mag.astype(np.float32), 
                                                  (self.field_size, self.field_size))
                        phase_resized = cv2.resize(phase.astype(np.float32),
                                                    (self.field_size, self.field_size))
                        return mag_resized * np.exp(1j * phase_resized)
                    return raw_input.astype(np.complex128)
                elif raw_input.ndim == 1:
                    # 1D spectrum - tile into 2D
                    n = len(raw_input)
                    field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
                    for i in range(min(n, self.field_size)):
                        field[i, :] = raw_input[i] if i < n else 0
                    return field
            else:
                # Real array - create complex with zero phase
                if raw_input.ndim == 2:
                    resized = cv2.resize(raw_input.astype(np.float32),
                                         (self.field_size, self.field_size))
                    return resized.astype(np.complex128)
                elif raw_input.ndim == 1:
                    field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
                    for i in range(min(len(raw_input), self.field_size)):
                        field[i, :] = raw_input[i]
                    return field
        
        return np.zeros((self.field_size, self.field_size), dtype=np.complex128)
    
    def _process_attractor_input(self, raw_input):
        """Convert attractor state to embedding"""
        if raw_input is None:
            return np.zeros(self.embed_dim)
        
        if isinstance(raw_input, np.ndarray):
            if raw_input.ndim == 1:
                if len(raw_input) >= self.embed_dim:
                    return raw_input[:self.embed_dim].astype(np.float64)
                else:
                    result = np.zeros(self.embed_dim)
                    result[:len(raw_input)] = raw_input
                    return result
            elif raw_input.ndim == 2:
                # Take mean or first row
                if raw_input.shape[0] > 0:
                    row = raw_input[0] if raw_input.shape[1] >= self.embed_dim else raw_input.flatten()
                    return self._process_attractor_input(row)
        
        return np.zeros(self.embed_dim)
    
    def step(self):
        self.epoch += 1
        
        # === GET INPUTS ===
        field_raw = self.get_blended_input('field_state', 'first')
        attractor_raw = self.get_blended_input('attractor_basin', 'mean')
        constraint_raw = self.get_blended_input('constraint_field', 'mean')
        temperature = self.get_blended_input('temperature', 'sum')
        theta = self.get_blended_input('theta_phase', 'sum') or 0.0
        
        temperature = float(temperature) if temperature is not None else 1.0
        temperature = max(0.01, min(10.0, temperature))
        
        # === PROCESS INPUTS ===
        self.current_field = self._process_field_input(field_raw)
        self.attractor_basin = self._process_attractor_input(attractor_raw)
        constraint = self._process_attractor_input(constraint_raw)
        
        has_field = np.abs(self.current_field).max() > 1e-10
        has_attractor = np.linalg.norm(self.attractor_basin) > 1e-10
        
        # Store history
        if has_field:
            self.field_history.append(self.current_field.copy())
        
        # === COMPUTE EXCLUSION ===
        # The attractor "sees" by computing what's impossible
        
        if has_field and has_attractor:
            # Method: The attractor basin defines a "compatibility function"
            # States incompatible with the attractor are EXCLUDED
            
            # Project field onto attractor basis
            field_magnitude = np.abs(self.current_field)
            field_phase = np.angle(self.current_field)
            
            # Create attractor influence map
            # Each point in field space has a "compatibility" with attractor
            x = np.linspace(-1, 1, self.field_size)
            y = np.linspace(-1, 1, self.field_size)
            X, Y = np.meshgrid(x, y)
            
            # Attractor creates a basin in field space
            # Use first few attractor components as basin center
            center_x = np.tanh(self.attractor_basin[0]) if len(self.attractor_basin) > 0 else 0
            center_y = np.tanh(self.attractor_basin[1]) if len(self.attractor_basin) > 1 else 0
            basin_width = 0.3 + 0.7 * np.exp(-np.linalg.norm(self.attractor_basin) * 0.1)
            
            # Distance from attractor center
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # Compatibility: Gaussian basin
            compatibility = np.exp(-distance**2 / (2 * basin_width**2 * temperature))
            
            # EXCLUSION = 1 - compatibility
            # States far from attractor are excluded
            self.excluded_states = 1.0 - compatibility
            self.possible_states = compatibility
            
            # The boundary is where compatibility transitions
            # Gradient magnitude of compatibility field
            grad_x = np.gradient(compatibility, axis=1)
            grad_y = np.gradient(compatibility, axis=0)
            self.exclusion_boundary = np.sqrt(grad_x**2 + grad_y**2)
            
            # === THE KEY INSIGHT ===
            # VISION = what the exclusion PROJECTS
            # The excluded states, viewed from the attractor, become the "prompt"
            
            # Vision field: The field MASKED by exclusion
            # What remains after exclusion is what the attractor "sees"
            self.vision_field = self.current_field * self.excluded_states
            
            # But also: the exclusion pattern itself carries information
            # The SHAPE of what's excluded tells the next layer what to do
            
            # Attractor prompt: Encode exclusion pattern as tokens
            # Sample exclusion at different radii from attractor center
            n_samples = self.embed_dim
            angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            radii = np.linspace(0.1, 1.0, n_samples)
            
            for i in range(n_samples):
                # Sample at this angle and radius
                sample_x = center_x + radii[i] * np.cos(angles[i] + theta)
                sample_y = center_y + radii[i] * np.sin(angles[i] + theta)
                
                # Map to grid indices
                ix = int((sample_x + 1) / 2 * (self.field_size - 1))
                iy = int((sample_y + 1) / 2 * (self.field_size - 1))
                
                ix = np.clip(ix, 0, self.field_size - 1)
                iy = np.clip(iy, 0, self.field_size - 1)
                
                # The prompt encodes: what's excluded at this direction/distance
                self.attractor_prompt[i] = self.excluded_states[iy, ix]
            
            # === METRICS ===
            # Exclusion entropy: How much of state space is excluded?
            excl_flat = self.excluded_states.flatten()
            excl_flat = excl_flat / (excl_flat.sum() + 1e-10)
            excl_flat = np.clip(excl_flat, 1e-10, 1.0)
            self.exclusion_entropy = -np.sum(excl_flat * np.log(excl_flat))
            
            # Vision clarity: How sharp is the exclusion boundary?
            self.vision_clarity = self.exclusion_boundary.max() / (self.exclusion_boundary.mean() + 1e-10)
            
        else:
            # No input - everything is possible, nothing is excluded
            self.possible_states = np.ones((self.field_size, self.field_size))
            self.excluded_states = np.zeros((self.field_size, self.field_size))
            self.exclusion_boundary = np.zeros((self.field_size, self.field_size))
            self.vision_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
            self.attractor_prompt = np.zeros(self.embed_dim)
            self.exclusion_entropy = 0.0
            self.vision_clarity = 0.0
        
        # Store exclusion history
        self.exclusion_history.append(self.exclusion_entropy)
        
        # === SET OUTPUTS ===
        self.outputs['excluded_field'] = self.vision_field  # The shadow
        self.outputs['vision_field'] = self.current_field * self.possible_states  # What remains
        self.outputs['exclusion_boundary'] = self.exclusion_boundary.astype(np.float32)
        self.outputs['attractor_prompt'] = self.attractor_prompt.astype(np.float32)
        self.outputs['exclusion_entropy'] = float(self.exclusion_entropy)
        self.outputs['vision_clarity'] = float(self.vision_clarity)
        
        # === RENDER ===
        self._render_display(has_field, has_attractor, temperature)
    
    def _render_display(self, has_field, has_attractor, temperature):
        """Render visualization"""
        img = self._display
        img[:] = (15, 12, 20)  # Dark purple background
        h, w = img.shape[:2]
        
        # === TITLE ===
        cv2.putText(img, "ATTRACTOR VISION - The Exclusion Field", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 220), 2)
        cv2.putText(img, "\"The attractor sees by what it excludes\"", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 120, 170), 1)
        
        panel_size = 160
        panel_y = 70
        
        # === PANEL 1: Current Field ===
        p1_x = 20
        cv2.putText(img, "FIELD (Substrate)", (p1_x, panel_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 200), 1)
        
        field_mag = np.abs(self.current_field)
        if field_mag.max() > 0:
            field_norm = field_mag / field_mag.max()
        else:
            field_norm = field_mag
        field_u8 = (field_norm * 255).astype(np.uint8)
        field_color = cv2.applyColorMap(field_u8, cv2.COLORMAP_VIRIDIS)
        field_resized = cv2.resize(field_color, (panel_size, panel_size))
        img[panel_y:panel_y+panel_size, p1_x:p1_x+panel_size] = field_resized
        
        # === PANEL 2: Possible States ===
        p2_x = 200
        cv2.putText(img, "POSSIBLE (Attractor Basin)", (p2_x, panel_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 150), 1)
        
        poss_u8 = (self.possible_states * 255).astype(np.uint8)
        poss_color = cv2.applyColorMap(poss_u8, cv2.COLORMAP_SUMMER)
        poss_resized = cv2.resize(poss_color, (panel_size, panel_size))
        img[panel_y:panel_y+panel_size, p2_x:p2_x+panel_size] = poss_resized
        
        # === PANEL 3: Excluded States (THE SHADOW) ===
        p3_x = 380
        cv2.putText(img, "EXCLUDED (The Shadow)", (p3_x, panel_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 150), 1)
        
        excl_u8 = (self.excluded_states * 255).astype(np.uint8)
        excl_color = cv2.applyColorMap(excl_u8, cv2.COLORMAP_HOT)
        excl_resized = cv2.resize(excl_color, (panel_size, panel_size))
        img[panel_y:panel_y+panel_size, p3_x:p3_x+panel_size] = excl_resized
        
        # === PANEL 4: Exclusion Boundary ===
        p4_x = 560
        cv2.putText(img, "BOUNDARY (Edge of Seeing)", (p4_x, panel_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        
        bound_norm = self.exclusion_boundary / (self.exclusion_boundary.max() + 1e-10)
        bound_u8 = (bound_norm * 255).astype(np.uint8)
        bound_color = cv2.applyColorMap(bound_u8, cv2.COLORMAP_MAGMA)
        bound_resized = cv2.resize(bound_color, (panel_size, panel_size))
        img[panel_y:panel_y+panel_size, p4_x:p4_x+panel_size] = bound_resized
        
        # === PANEL 5: Vision Field ===
        p5_x = 740
        cv2.putText(img, "VISION (Exclusion as Signal)", (p5_x, panel_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 255), 1)
        
        vision_mag = np.abs(self.vision_field)
        if vision_mag.max() > 0:
            vision_norm = vision_mag / vision_mag.max()
        else:
            vision_norm = vision_mag
        vision_u8 = (vision_norm * 255).astype(np.uint8)
        vision_color = cv2.applyColorMap(vision_u8, cv2.COLORMAP_TWILIGHT)
        vision_resized = cv2.resize(vision_color, (panel_size, panel_size))
        img[panel_y:panel_y+panel_size, p5_x:p5_x+panel_size] = vision_resized
        
        # === ATTRACTOR PROMPT (The Shadow as Tokens) ===
        prompt_y = 260
        prompt_x = 20
        prompt_w = 700
        prompt_h = 80
        
        cv2.rectangle(img, (prompt_x, prompt_y), (prompt_x+prompt_w, prompt_y+prompt_h),
                     (30, 25, 40), -1)
        cv2.putText(img, "ATTRACTOR PROMPT (Shadow as Tokens for Next Layer)", 
                   (prompt_x + 10, prompt_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 180, 220), 1)
        
        # Draw prompt as bar chart
        bar_w = prompt_w // self.embed_dim
        max_val = self.attractor_prompt.max() + 1e-10
        
        for i in range(self.embed_dim):
            val = self.attractor_prompt[i] / max_val
            bar_h = int(val * 50)
            bx = prompt_x + 10 + i * bar_w
            by = prompt_y + prompt_h - 10
            
            # Color by value
            intensity = int(val * 255)
            color = (intensity, 50, 255 - intensity)
            
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 1, by), color, -1)
        
        # === METRICS ===
        met_y = 360
        met_x = 20
        
        cv2.rectangle(img, (met_x, met_y), (met_x + 400, met_y + 100), (25, 20, 35), -1)
        cv2.putText(img, "METRICS", (met_x + 10, met_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.putText(img, f"Exclusion Entropy: {self.exclusion_entropy:.4f}", 
                   (met_x + 10, met_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(img, f"Vision Clarity: {self.vision_clarity:.4f}", 
                   (met_x + 10, met_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(img, f"Temperature: {temperature:.3f}", 
                   (met_x + 200, met_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(img, f"Epoch: {self.epoch}", 
                   (met_x + 200, met_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        # Input status
        if has_field and has_attractor:
            status = "Field + Attractor: Computing exclusion"
            color = (100, 255, 150)
        elif has_field:
            status = "Field only: Need attractor input"
            color = (255, 200, 100)
        elif has_attractor:
            status = "Attractor only: Need field input"
            color = (255, 200, 100)
        else:
            status = "No input: All states possible"
            color = (150, 150, 150)
        
        cv2.putText(img, status, (met_x + 10, met_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # === INTERPRETATION ===
        interp_x = 450
        interp_y = 360
        
        cv2.rectangle(img, (interp_x, interp_y), (interp_x + 430, interp_y + 100), 
                     (25, 20, 35), -1)
        cv2.putText(img, "INTERPRETATION", (interp_x + 10, interp_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        lines = [
            "The attractor doesn't see the field directly.",
            "It sees by EXCLUSION - what's impossible.",
            "The shadow becomes the prompt for the next layer.",
            "Vision = Exclusion projected forward.",
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(img, line, (interp_x + 10, interp_y + 40 + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 140, 180), 1)
        
        # === EXCLUSION HISTORY ===
        hist_x = 20
        hist_y = 480
        hist_w = 860
        hist_h = 60
        
        cv2.rectangle(img, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h),
                     (25, 20, 35), -1)
        cv2.putText(img, "EXCLUSION HISTORY", (hist_x + 10, hist_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        if len(self.exclusion_history) > 1:
            history = list(self.exclusion_history)
            max_h = max(history) + 1e-10
            points = []
            for i, val in enumerate(history):
                x = hist_x + 10 + int(i / len(history) * (hist_w - 20))
                y = hist_y + hist_h - 10 - int((val / max_h) * (hist_h - 20))
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], (200, 150, 255), 2)
        
        self._display = img
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display