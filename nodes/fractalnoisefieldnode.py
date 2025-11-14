"""
FractalNoiseFieldNode (Simplified but Functional)
--------------------------------------------------
Generates multi-scale fractal noise where complexity matches across scales.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class FractalNoiseFieldNode(BaseNode):
    NODE_CATEGORY = "Fractal Substrate"
    NODE_COLOR = QtGui.QColor(20, 20, 80)

    def __init__(self, field_size=256, octaves=4, persistence=0.5):
        super().__init__()
        self.node_title = "Fractal Noise Field"

        self.inputs = {
            'perturbation': 'image',
        }

        self.outputs = {
            'noise_field': 'image',
            'complexity_map': 'image',
            'alignment_field': 'image',
            'phase_structure': 'image',  # FIXED: matches what StructureDegradation expects
        }

        self.field_size = int(field_size)
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        
        self.noise_field = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        self.complexity_map = np.zeros_like(self.noise_field)
        self.alignment_field = np.zeros_like(self.noise_field)
        self.phase_structure = np.zeros_like(self.noise_field)
        
        self.time = 0

    def _generate_octave_noise(self):
        """Simple multi-scale noise"""
        result = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0
        
        for octave in range(self.octaves):
            # Generate noise at this scale
            scale = int(self.field_size / frequency)
            if scale < 2:
                scale = 2
            
            small_noise = np.random.randn(scale, scale)
            large_noise = cv2.resize(small_noise, (self.field_size, self.field_size), 
                                    interpolation=cv2.INTER_LINEAR)
            
            result += large_noise * amplitude
            amplitude *= self.persistence
            frequency *= 2.0
        
        # Normalize
        if result.std() > 0:
            result = (result - result.mean()) / result.std()
        
        return result

    def _compute_local_complexity(self, field):
        """Estimate complexity using edge density"""
        # Simple but effective: edge strength correlates with fractal dimension
        edges = cv2.Sobel(field, cv2.CV_32F, 1, 1, ksize=3)
        edges = np.abs(edges)
        
        # Local complexity = smoothed edge density
        complexity = cv2.GaussianBlur(edges, (15, 15), 0)
        
        # Normalize
        if complexity.max() > 0:
            complexity = complexity / complexity.max()
        
        return complexity

    def _compute_alignment(self, noise_field):
        """Where complexity is consistent across scales = information channels"""
        # Compute complexity at multiple scales
        complexities = []
        for blur_size in [5, 11, 21]:
            blurred = cv2.GaussianBlur(noise_field, (blur_size, blur_size), 0)
            comp = self._compute_local_complexity(blurred)
            complexities.append(comp)
        
        # Where complexity variance is LOW = good alignment
        complexity_stack = np.stack(complexities, axis=0)
        variance = np.var(complexity_stack, axis=0)
        
        # Invert: low variance = high alignment
        alignment = 1.0 - np.clip(variance * 5, 0, 1)
        
        return alignment

    def step(self):
        # Generate base noise
        self.noise_field = self._generate_octave_noise()
        
        # Apply perturbation if provided
        perturbation = self.get_blended_input('perturbation', 'mean')
        if perturbation is not None:
            if perturbation.shape[:2] != (self.field_size, self.field_size):
                perturbation = cv2.resize(perturbation, (self.field_size, self.field_size))
            if perturbation.ndim == 3:
                perturbation = np.mean(perturbation, axis=2)
            
            # Gentle deformation
            perturbation_norm = (perturbation - perturbation.mean())
            if perturbation_norm.std() > 0:
                perturbation_norm = perturbation_norm / perturbation_norm.std()
            self.noise_field += perturbation_norm * 0.2
        
        # Compute complexity map
        self.complexity_map = self._compute_local_complexity(self.noise_field)
        
        # Compute alignment field (where information exists)
        self.alignment_field = self._compute_alignment(self.noise_field)
        
        # Phase structure (FFT phase)
        fft = np.fft.fft2(self.noise_field)
        phase = np.angle(fft)
        self.phase_structure = (phase + np.pi) / (2 * np.pi)  # Normalize to 0-1
        
        self.time += 1

    def get_output(self, port_name):
        if port_name == 'noise_field':
            return self.noise_field
        elif port_name == 'complexity_map':
            return self.complexity_map
        elif port_name == 'alignment_field':
            return self.alignment_field
        elif port_name == 'phase_structure':  # FIXED
            return self.phase_structure
        return None

    def get_display_image(self):
        display_w = 512
        display_h = 512
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Quadrants: noise, complexity, alignment, phase
        quad_size = display_w // 2
        
        # Top left: Noise
        noise_u8 = ((self.noise_field + 2) * 63).astype(np.uint8)
        noise_color = cv2.applyColorMap(noise_u8, cv2.COLORMAP_VIRIDIS)
        noise_resized = cv2.resize(noise_color, (quad_size, quad_size))
        display[:quad_size, :quad_size] = noise_resized
        
        # Top right: Complexity
        comp_u8 = (self.complexity_map * 255).astype(np.uint8)
        comp_color = cv2.applyColorMap(comp_u8, cv2.COLORMAP_HOT)
        comp_resized = cv2.resize(comp_color, (quad_size, quad_size))
        display[:quad_size, quad_size:] = comp_resized
        
        # Bottom left: Alignment (WHERE INFO EXISTS)
        align_u8 = (self.alignment_field * 255).astype(np.uint8)
        align_color = cv2.applyColorMap(align_u8, cv2.COLORMAP_RAINBOW)
        align_resized = cv2.resize(align_color, (quad_size, quad_size))
        display[quad_size:, :quad_size] = align_resized
        
        # Bottom right: Phase
        phase_u8 = (self.phase_structure * 255).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_u8, cv2.COLORMAP_TWILIGHT)
        phase_resized = cv2.resize(phase_color, (quad_size, quad_size))
        display[quad_size:, quad_size:] = phase_resized
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'NOISE', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'COMPLEXITY', (quad_size + 10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'ALIGNMENT', (10, quad_size + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'PHASE', (quad_size + 10, quad_size + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Field Size", "field_size", self.field_size, None),
            ("Octaves", "octaves", self.octaves, None),
            ("Persistence", "persistence", self.persistence, None),
        ]