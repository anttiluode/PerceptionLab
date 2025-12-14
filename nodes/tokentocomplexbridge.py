"""
Token-to-Spectrum Bridge - Converts Tokens to Holographic Field
================================================================
Bridges the gap between token-based representations and 
complex spectrum visualizations.

The GenerativeDecoder outputs tokens (N x 3 arrays: id, amplitude, phase)
but visualization nodes like ConsciousnessSpectrumNode expect 
complex spectra (1D or 2D complex arrays).

This node converts tokens → interference field → complex spectrum

ARCHITECTURE:
1. Takes token array from decoder
2. Generates holographic interference pattern
3. Computes FFT to get complex spectrum
4. Outputs in format compatible with visualization nodes

OUTPUTS:
- display: Token visualization
- complex_spectrum: 1D complex spectrum (for consciousness nodes)
- interference_field: 2D complex field
- power_spectrum: Real-valued power
- phase_spectrum: Phase angles

CREATED: December 2025
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

class TokenToSpectrumBridge(BaseNode):
    """
    Converts token representations to complex spectra for visualization.
    """
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Token → Spectrum"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Light blue - bridge
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'tokens': 'spectrum',           # From GenerativeDecoder
            'phase_reference': 'signal',    # Optional phase alignment
            'field_size': 'signal',         # Output resolution
        }
        
        self.outputs = {
            'display': 'image',
            'complex_spectrum': 'complex_spectrum',  # 1D for consciousness nodes
            'interference_field': 'complex_spectrum', # 2D field
            'power_spectrum': 'spectrum',    # |FFT|²
            'phase_spectrum': 'spectrum',    # angle(FFT)
            'spectrum_1d': 'spectrum',       # Real 1D for simpler nodes
        }
        
        # === PARAMETERS ===
        self.field_size = 128  # Default field resolution
        self.spectrum_size = 64  # 1D spectrum length
        
        # === STATE ===
        self.current_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        self.current_spectrum = np.zeros(self.spectrum_size, dtype=np.complex128)
        
        # === DISPLAY ===
        self._display = np.zeros((400, 600, 3), dtype=np.uint8)
    
    def _sanitize_tokens(self, data):
        """Ensure we have a valid token array"""
        if data is None:
            return np.zeros((0, 3), dtype=np.float32)
        
        if isinstance(data, str):
            return np.zeros((0, 3), dtype=np.float32)
        
        if isinstance(data, (list, tuple)):
            try:
                data = np.array(data, dtype=np.float32)
            except:
                return np.zeros((0, 3), dtype=np.float32)
        
        if not hasattr(data, 'shape'):
            return np.zeros((0, 3), dtype=np.float32)
        
        # Handle 1D array
        if data.ndim == 1:
            if len(data) % 3 == 0 and len(data) > 0:
                return data.reshape(-1, 3).astype(np.float32)
            elif len(data) == 3:
                return data.reshape(1, 3).astype(np.float32)
            else:
                # It might be a 1D spectrum already - return empty tokens
                return np.zeros((0, 3), dtype=np.float32)
        
        # Handle 2D array
        if data.ndim == 2:
            if data.shape[1] >= 3:
                return data[:, :3].astype(np.float32)
            else:
                return np.zeros((0, 3), dtype=np.float32)
        
        return np.zeros((0, 3), dtype=np.float32)
    
    def _tokens_to_field(self, tokens, phase_ref=0.0):
        """Convert tokens to 2D complex interference field"""
        size = self.field_size
        field = np.zeros((size, size), dtype=np.complex128)
        
        if len(tokens) == 0:
            return field
        
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        for tok in tokens:
            token_id = int(tok[0]) % 20
            amplitude = float(tok[1])
            phase = float(tok[2])
            
            # Skip low-amplitude tokens
            if amplitude < 0.01:
                continue
            
            # Wave parameters from token ID
            # Golden angle spread for nice patterns
            angle = token_id * 2.39996323  # Golden angle
            k = 1 + (token_id % 8)  # Spatial frequency 1-8
            
            kx = k * np.cos(angle + phase_ref)
            ky = k * np.sin(angle + phase_ref)
            
            # Create plane wave
            wave = amplitude * np.exp(1j * (kx * X + ky * Y + phase))
            field += wave
        
        # Normalize
        max_mag = np.abs(field).max()
        if max_mag > 1e-9:
            field = field / max_mag
        
        return field
    
    def _field_to_spectrum(self, field):
        """Convert 2D field to 1D complex spectrum via radial FFT"""
        # 2D FFT
        fft_2d = np.fft.fftshift(np.fft.fft2(field))
        
        # Radial profile (1D spectrum)
        size = field.shape[0]
        center = size // 2
        
        spectrum = np.zeros(self.spectrum_size, dtype=np.complex128)
        
        for r in range(self.spectrum_size):
            # Sample at this radius
            n_samples = max(8, int(2 * np.pi * r))
            angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            
            values = []
            for theta in angles:
                x = int(center + r * np.cos(theta))
                y = int(center + r * np.sin(theta))
                
                if 0 <= x < size and 0 <= y < size:
                    values.append(fft_2d[y, x])
            
            if values:
                spectrum[r] = np.mean(values)
        
        return spectrum
    
    def step(self):
        # Get inputs
        raw_tokens = self.get_blended_input('tokens', 'mean')
        phase_ref = self.get_blended_input('phase_reference', 'sum')
        size_val = self.get_blended_input('field_size', 'sum')
        
        phase_ref = float(phase_ref) if phase_ref else 0.0
        
        if size_val and size_val > 16:
            self.field_size = int(min(size_val, 256))
        
        # Sanitize tokens
        tokens = self._sanitize_tokens(raw_tokens)
        
        # Generate field from tokens
        self.current_field = self._tokens_to_field(tokens, phase_ref)
        
        # Extract 1D spectrum
        self.current_spectrum = self._field_to_spectrum(self.current_field)
        
        # Compute power and phase spectra
        power = np.abs(self.current_spectrum) ** 2
        phase = np.angle(self.current_spectrum)
        
        # === UPDATE OUTPUTS ===
        self.outputs['complex_spectrum'] = self.current_spectrum.astype(np.complex64)
        self.outputs['interference_field'] = self.current_field.astype(np.complex64)
        self.outputs['power_spectrum'] = power.astype(np.float32)
        self.outputs['phase_spectrum'] = phase.astype(np.float32)
        
        # Real 1D spectrum (magnitude) for simpler nodes
        self.outputs['spectrum_1d'] = np.abs(self.current_spectrum).astype(np.float32)
        
        # Render display
        self._render_display(tokens)
    
    def _render_display(self, tokens):
        """Visualize the conversion"""
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === LEFT: Token bars ===
        self._render_tokens(img, 10, 30, 180, h-60, tokens)
        
        # === CENTER: Interference field ===
        self._render_field(img, 200, 30, 180, 180)
        
        # === RIGHT: Spectrum ===
        self._render_spectrum(img, 400, 30, 180, 180)
        
        # === BOTTOM: Flow diagram ===
        cv2.putText(img, "TOKENS", (70, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.arrowedLine(img, (140, h-35), (200, h-35), (150, 150, 150), 2)
        cv2.putText(img, "FIELD", (250, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.arrowedLine(img, (320, h-35), (380, h-35), (150, 150, 150), 2)
        cv2.putText(img, "SPECTRUM", (420, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Title
        cv2.putText(img, "TOKEN -> SPECTRUM BRIDGE", (w//2 - 100, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        self._display = img
    
    def _render_tokens(self, img, x0, y0, width, height, tokens):
        """Render token bars"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        if len(tokens) == 0:
            cv2.putText(img, "No tokens", (x0+40, y0+height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return
        
        bar_h = min(20, (height - 20) // len(tokens))
        
        for i, tok in enumerate(tokens[:15]):  # Max 15 tokens shown
            token_id = int(tok[0])
            amplitude = float(tok[1])
            phase = float(tok[2])
            
            y = y0 + 10 + i * bar_h
            bar_w = int(min(amplitude * 50, width - 30))
            
            # Color by phase
            hue = int((phase + np.pi) / (2 * np.pi) * 180)
            hsv = np.array([[[hue, 255, 200]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            
            cv2.rectangle(img, (x0+25, y), (x0+25+bar_w, y+bar_h-2), rgb, -1)
            cv2.putText(img, f"{token_id}", (x0+5, y+bar_h-4),
                       cv2.FONT_HERSHEY_PLAIN, 0.6, (150, 150, 150), 1)
    
    def _render_field(self, img, x0, y0, width, height):
        """Render interference field"""
        # Convert complex field to color image
        magnitude = np.abs(self.current_field)
        phase = np.angle(self.current_field)
        
        # HSV: hue=phase, value=magnitude
        mag_norm = magnitude / (magnitude.max() + 1e-9)
        
        hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        hsv[:,:,0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:,:,1] = 255
        hsv[:,:,2] = (mag_norm * 255).astype(np.uint8)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Resize and place
        resized = cv2.resize(rgb, (width, height))
        img[y0:y0+height, x0:x0+width] = resized
        
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (100, 100, 100), 1)
    
    def _render_spectrum(self, img, x0, y0, width, height):
        """Render 1D spectrum"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        spectrum = np.abs(self.current_spectrum)
        if spectrum.max() < 1e-9:
            return
        
        spectrum_norm = spectrum / spectrum.max()
        
        bar_w = max(1, width // len(spectrum))
        for i, val in enumerate(spectrum_norm):
            bx = x0 + i * bar_w
            bar_h = int(val * (height - 20))
            
            # Color gradient
            color = (int(100 + val * 155), int(200 * val), int(255 * (1-val)))
            
            cv2.rectangle(img, (bx, y0+height-10-bar_h), (bx+bar_w-1, y0+height-10),
                         color, -1)
        
        cv2.putText(img, "Power Spectrum", (x0+10, y0+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display