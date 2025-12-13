"""
Token Interference Engine - Holographic Token Relationships
============================================================
Takes token streams from WaveletTokenEngine or NeuralTransformer
and creates interference patterns showing how tokens relate.

THIS REVEALS THE HIDDEN STRUCTURE.

INPUTS:
- tokens_a: Primary token stream (e.g., frontal)
- tokens_b: Secondary token stream (e.g., temporal)  
- phase_reference: Phase signal for alignment
- interference_mode: 0=multiply, 1=add, 2=phase_conjugate, 3=cross_correlation

OUTPUTS:
- display: Full visualization
- interference_field: Complex field (purple port)
- coherence_map: Where tokens align
- relationship_vector: 64-dim encoding of token relationship

MODES:
0. MULTIPLY: Direct wave interference (constructive/destructive)
1. ADD: Superposition (linear combination)
2. PHASE_CONJUGATE: Time-reversal (memory retrieval pattern)
3. CROSS_CORRELATION: Sliding similarity

The interference pattern shows:
- Bright regions = tokens are aligned (in-phase)
- Dark regions = tokens cancel (out-of-phase)
- Patterns = frequency/spatial relationships between token sets
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

class TokenInterferenceEngine2(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Token Interference"
    NODE_COLOR = QtGui.QColor(150, 50, 255)  # Purple - interference color
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'tokens_a': 'spectrum',      # Primary tokens
            'tokens_b': 'spectrum',      # Secondary tokens
            'phase_reference': 'signal', # For alignment
            'interference_mode': 'signal', # 0-3
            'zoom': 'signal',            # Pattern zoom
        }
        
        self.outputs = {
            'display': 'image',
            'interference_field': 'complex_spectrum',
            'coherence_map': 'image',
            'relationship_vector': 'spectrum',
        }
        
        # State
        self.field_size = 512
        self.field_a = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.field_b = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.interference = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.coherence = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        
        # History for temporal analysis
        self.history_a = deque(maxlen=100)
        self.history_b = deque(maxlen=100)
        
        # Embedding for relationship vector
        self.embed_dim = 64
        self.relationship_vector = np.zeros(self.embed_dim)
        
        # Display
        self._display = np.zeros((800, 1200, 3), dtype=np.uint8)
        self._coherence_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    def _sanitize_tokens(self, data):
        """Convert input to valid token array"""
        if data is None:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, str):
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, (list, tuple)):
            try:
                data = np.array(data)
            except:
                return np.zeros((0, 3), dtype=np.float32)
        if not hasattr(data, 'ndim'):
            return np.zeros((0, 3), dtype=np.float32)
        if data.ndim == 1:
            if len(data) == 3:
                return data.reshape(1, 3)
            return np.zeros((0, 3), dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return data.astype(np.float32)
    
    def _tokens_to_field(self, tokens, phase_offset=0.0):
        """Convert token array to complex wave field"""
        field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        
        if len(tokens) == 0:
            return field
        
        x = np.linspace(-np.pi, np.pi, self.field_size)
        y = np.linspace(-np.pi, np.pi, self.field_size)
        X, Y = np.meshgrid(x, y)
        
        for tok in tokens:
            token_id = int(tok[0])
            amplitude = tok[1]
            phase = tok[2]
            
            # Token ID determines wave vector direction
            # Use golden angle for nice spread
            angle = token_id * 2.39996323  # Golden angle in radians
            
            # Token ID also affects frequency
            k = 1 + (token_id % 8)  # 1-8 spatial frequency
            
            kx = k * np.cos(angle)
            ky = k * np.sin(angle)
            
            # Create plane wave
            wave = amplitude * np.exp(1j * (kx * X + ky * Y + phase + phase_offset))
            field += wave
        
        return field
    
    def _compute_interference(self, field_a, field_b, mode=0):
        """Compute interference between two fields"""
        if mode == 0:  # MULTIPLY
            result = field_a * np.conj(field_b)
        
        elif mode == 1:  # ADD (superposition)
            result = field_a + field_b
        
        elif mode == 2:  # PHASE CONJUGATE
            result = field_a * field_b  # Double the phase
        
        elif mode == 3:  # CROSS CORRELATION (via FFT)
            fft_a = np.fft.fft2(field_a)
            fft_b = np.fft.fft2(field_b)
            result = np.fft.ifft2(fft_a * np.conj(fft_b))
        
        else:
            result = field_a * np.conj(field_b)
        
        return result
    
    def _compute_coherence(self, field_a, field_b):
        """Compute local coherence map"""
        # Phase difference
        phase_a = np.angle(field_a)
        phase_b = np.angle(field_b)
        phase_diff = phase_a - phase_b
        
        # Coherence = cos(phase_diff)^2
        coherence = np.cos(phase_diff) ** 2
        
        # Smooth
        from scipy.ndimage import gaussian_filter
        coherence = gaussian_filter(coherence, sigma=5)
        
        return coherence.astype(np.float32)
    
    def _compute_relationship(self, tokens_a, tokens_b):
        """Compute relationship vector between token sets"""
        vec = np.zeros(self.embed_dim)
        
        if len(tokens_a) == 0 or len(tokens_b) == 0:
            return vec
        
        # Cross-product features
        for ta in tokens_a:
            for tb in tokens_b:
                # Phase relationship
                phase_diff = ta[2] - tb[2]
                
                # Amplitude product
                amp_prod = ta[1] * tb[1]
                
                # Token ID interaction
                id_sum = (int(ta[0]) + int(tb[0])) % self.embed_dim
                id_diff = abs(int(ta[0]) - int(tb[0])) % self.embed_dim
                
                # Accumulate features
                vec[id_sum] += amp_prod * np.cos(phase_diff)
                vec[id_diff] += amp_prod * np.sin(phase_diff)
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def step(self):
        # Get inputs
        raw_a = self.get_blended_input('tokens_a', 'mean')
        raw_b = self.get_blended_input('tokens_b', 'mean')
        
        phase_ref = self.get_blended_input('phase_reference', 'sum')
        if phase_ref is None:
            phase_ref = 0.0
        
        mode_val = self.get_blended_input('interference_mode', 'sum')
        mode = int(mode_val) if mode_val else 0
        mode = max(0, min(3, mode))
        
        zoom_val = self.get_blended_input('zoom', 'sum')
        zoom = float(zoom_val) if zoom_val and zoom_val > 0 else 1.0
        
        # Sanitize tokens
        tokens_a = self._sanitize_tokens(raw_a)
        tokens_b = self._sanitize_tokens(raw_b)
        
        # Store history
        self.history_a.append(tokens_a.copy() if len(tokens_a) > 0 else None)
        self.history_b.append(tokens_b.copy() if len(tokens_b) > 0 else None)
        
        # Convert to fields
        self.field_a = self._tokens_to_field(tokens_a, phase_ref)
        self.field_b = self._tokens_to_field(tokens_b, 0.0)
        
        # Compute interference
        self.interference = self._compute_interference(self.field_a, self.field_b, mode)
        
        # Compute coherence
        self.coherence = self._compute_coherence(self.field_a, self.field_b)
        
        # Compute relationship vector
        self.relationship_vector = self._compute_relationship(tokens_a, tokens_b)
        
        # Update outputs
        self.outputs['interference_field'] = self.interference
        self.outputs['relationship_vector'] = self.relationship_vector.astype(np.float32)
        
        # Render coherence image
        coh_u8 = (self.coherence * 255).clip(0, 255).astype(np.uint8)
        coh_resized = cv2.resize(coh_u8, (256, 256))
        self._coherence_img = cv2.applyColorMap(coh_resized, cv2.COLORMAP_VIRIDIS)
        self.outputs['coherence_map'] = self._coherence_img
        
        # Render main display
        self._render_display(tokens_a, tokens_b, mode, zoom)
    
    def _render_display(self, tokens_a, tokens_b, mode, zoom):
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === LEFT: Field A ===
        self._render_field(img, self.field_a, 20, 20, 350, 350, "FIELD A (Primary)")
        
        # === CENTER LEFT: Field B ===
        self._render_field(img, self.field_b, 390, 20, 350, 350, "FIELD B (Secondary)")
        
        # === CENTER RIGHT: Interference ===
        self._render_field(img, self.interference, 760, 20, 400, 350, 
                          f"INTERFERENCE (Mode {mode})")
        
        # === BOTTOM LEFT: Token bars A ===
        self._render_token_bars(img, tokens_a, 20, 400, 350, 150, "PRIMARY TOKENS", (255, 100, 100))
        
        # === BOTTOM CENTER: Token bars B ===
        self._render_token_bars(img, tokens_b, 390, 400, 350, 150, "SECONDARY TOKENS", (100, 255, 100))
        
        # === BOTTOM RIGHT: Coherence + Relationship ===
        self._render_analysis(img, 760, 400, 400, 350)
        
        self._display = img
    
    def _render_field(self, img, field, x0, y0, width, height, title):
        """Render complex field as HSV image"""
        # Get magnitude and phase
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # Normalize magnitude
        mag_norm = magnitude / (magnitude.max() + 1e-9)
        
        # Create HSV
        hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        hsv[:,:,0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:,:,1] = 255
        hsv[:,:,2] = (mag_norm * 255).clip(0, 255).astype(np.uint8)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Resize and place
        resized = cv2.resize(rgb, (width, height))
        img[y0:y0+height, x0:x0+width] = resized
        
        # Border
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (80, 80, 100), 2)
        
        # Title
        cv2.putText(img, title, (x0 + 10, y0 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _render_token_bars(self, img, tokens, x0, y0, width, height, title, color):
        """Render token amplitude bars"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        if len(tokens) == 0:
            cv2.putText(img, "No tokens", (x0 + width//2 - 40, y0 + height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            bar_width = max(5, (width - 20) // len(tokens))
            
            for i, tok in enumerate(tokens):
                token_id = int(tok[0])
                amplitude = tok[1]
                phase = tok[2]
                
                bar_height = int(min(amplitude * 30, height - 40))
                bx = x0 + 10 + i * bar_width
                by = y0 + height - 20
                
                # Color varies with phase
                hue = int((phase + np.pi) / (2 * np.pi) * 180)
                hsv_color = np.array([[[hue, 255, 200]]], dtype=np.uint8)
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                
                cv2.rectangle(img, (bx, by - bar_height), (bx + bar_width - 2, by),
                             rgb_color, -1)
                
                # Token ID label
                cv2.putText(img, str(token_id), (bx, by + 15),
                           cv2.FONT_HERSHEY_PLAIN, 0.6, (150, 150, 150), 1)
        
        # Title
        cv2.putText(img, title, (x0 + 10, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _render_analysis(self, img, x0, y0, width, height):
        """Render coherence and relationship analysis"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        # Coherence map (top half)
        coh_size = min(150, height // 2 - 20)
        coh_x = x0 + 10
        coh_y = y0 + 20
        
        coh_resized = cv2.resize(self._coherence_img, (coh_size, coh_size))
        img[coh_y:coh_y+coh_size, coh_x:coh_x+coh_size] = coh_resized
        
        cv2.putText(img, "COHERENCE", (coh_x, coh_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Mean coherence value
        mean_coh = np.mean(self.coherence)
        cv2.putText(img, f"Mean: {mean_coh:.3f}", (coh_x + coh_size + 20, coh_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        # Relationship vector (bottom half)
        vec_y = y0 + height // 2 + 20
        vec_x = x0 + 10
        vec_width = width - 20
        vec_height = 80
        
        # Draw as bar chart
        bar_w = vec_width // len(self.relationship_vector)
        for i, v in enumerate(self.relationship_vector):
            bar_h = int(abs(v) * 50)
            bx = vec_x + i * bar_w
            
            if v >= 0:
                color = (100, 200, 100)
                cv2.rectangle(img, (bx, vec_y + 40 - bar_h), (bx + bar_w - 1, vec_y + 40),
                             color, -1)
            else:
                color = (100, 100, 200)
                cv2.rectangle(img, (bx, vec_y + 40), (bx + bar_w - 1, vec_y + 40 + bar_h),
                             color, -1)
        
        cv2.putText(img, "RELATIONSHIP VECTOR", (vec_x, vec_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Vector norm
        vec_norm = np.linalg.norm(self.relationship_vector)
        cv2.putText(img, f"Strength: {vec_norm:.3f}", (vec_x + vec_width - 100, vec_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # Mode explanations
        modes = ["MULTIPLY", "ADD", "PHASE_CONJ", "CROSS_CORR"]
        cv2.putText(img, "MODES:", (x0 + width - 120, y0 + height - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        for i, m in enumerate(modes):
            cv2.putText(img, f"{i}: {m}", (x0 + width - 120, y0 + height - 60 + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'coherence_map':
            return self._coherence_img
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display