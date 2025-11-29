"""
Image to Complex Spectrum Adapter
Converts image data to complex number format for wiring flexibility.
Multiple encoding modes to explore different representations.
"""

import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class ImageToComplexNode(BaseNode):
    """
    Adapter: Image → Complex Spectrum (Purple Port)
    
    Encodes image data as complex numbers without FFT.
    This allows direct image→resonance wiring.
    
    Encoding modes:
    - Brightness→Magnitude: pixel brightness = amplitude, phase = 0
    - Brightness→Phase: amplitude = 1, pixel brightness = phase angle
    - Gradient→Complex: dx = real, dy = imaginary (edge encoding)
    - Polar: radius from center = magnitude, angle = phase
    - Dual Channel: R = real, G = imaginary (if color input)
    """
    NODE_CATEGORY = "Adapter"
    NODE_TITLE = "Image → Complex"
    NODE_COLOR = QtGui.QColor(180, 100, 220)  # Purple to match complex ports
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',
            'phase_offset': 'signal',      # Rotate all phases
            'amplitude_scale': 'signal'    # Scale magnitudes
        }
        
        self.outputs = {
            'complex_spectrum': 'complex_spectrum',  # The purple port
            'magnitude_view': 'image',
            'phase_view': 'image'
        }
        
        self.encoding_mode = "Brightness→Magnitude"
        self.complex_field = None
        self.size = 128
        
    def step(self):
        img = self.get_blended_input('image_in', 'mean')
        phase_offset = self.get_blended_input('phase_offset', 'sum') or 0.0
        amp_scale = self.get_blended_input('amplitude_scale', 'sum')
        if amp_scale is None:
            amp_scale = 1.0
        
        if img is None:
            return
            
        # CRITICAL: The host's get_blended_input converts to float64
        # OpenCV ONLY accepts uint8 or float32 for cvtColor
        # Convert to uint8 FIRST, then work from there
        
        # Step 1: Get to uint8 no matter what input type
        if img.dtype in [np.float64, np.float32]:
            # Normalize to 0-255 and convert to uint8
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img_normalized = (img - img_min) / (img_max - img_min)
            else:
                img_normalized = np.zeros_like(img)
            img_u8 = (img_normalized * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            img_u8 = img
        else:
            img_u8 = img.astype(np.uint8)
            
        # Step 2: Convert to grayscale (now safe - img_u8 is uint8)
        if img_u8.ndim == 3:
            img_gray_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
            img_color = img_u8
        else:
            img_gray_u8 = img_u8
            img_color = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            
        # Step 3: Convert to float32 0-1 for math operations
        img_gray = img_gray_u8.astype(np.float32) / 255.0
            
        h, w = img_gray.shape
        self.size = max(h, w)
        
        # Convert parameters to float32
        phase_offset = np.float32(phase_offset)
        amp_scale = np.float32(amp_scale)
        
        # === ENCODING MODES ===
        
        if self.encoding_mode == "Brightness→Magnitude":
            # Pixel value = amplitude, phase = 0 (or offset)
            magnitude = img_gray * amp_scale
            phase = np.ones_like(img_gray) * phase_offset * 2 * np.pi
            self.complex_field = (magnitude * np.exp(1j * phase)).astype(np.complex64)
            
        elif self.encoding_mode == "Brightness→Phase":
            # Amplitude = 1, pixel value = phase angle
            magnitude = np.ones_like(img_gray) * amp_scale
            phase = img_gray * 2 * np.pi + phase_offset * 2 * np.pi
            self.complex_field = (magnitude * np.exp(1j * phase)).astype(np.complex64)
            
        elif self.encoding_mode == "Gradient→Complex":
            # Sobel gradients: dx = real, dy = imaginary
            dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
            self.complex_field = ((dx + 1j * dy) * amp_scale).astype(np.complex64)
            # Apply phase rotation
            self.complex_field *= np.exp(1j * phase_offset * 2 * np.pi).astype(np.complex64)
            
        elif self.encoding_mode == "Polar":
            # Distance from center = magnitude, angle from center = phase
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.float32)
            theta = np.arctan2(y - cy, x - cx).astype(np.float32)
            
            # Normalize radius
            r_max = np.sqrt(cx**2 + cy**2)
            magnitude = (r / r_max) * amp_scale
            phase = theta + phase_offset * 2 * np.pi
            self.complex_field = (magnitude * np.exp(1j * phase)).astype(np.complex64)
            
        elif self.encoding_mode == "Dual Channel":
            # R channel = real, G channel = imaginary
            if img_color.ndim == 3:
                # img_color is already uint8
                r_chan = img_color[:, :, 2].astype(np.float32) / 255.0
                g_chan = img_color[:, :, 1].astype(np.float32) / 255.0
                # Center around zero: 0.5 → 0
                real_part = (r_chan - 0.5) * 2 * amp_scale
                imag_part = (g_chan - 0.5) * 2 * amp_scale
                self.complex_field = (real_part + 1j * imag_part).astype(np.complex64)
                # Apply phase rotation
                self.complex_field *= np.exp(1j * phase_offset * 2 * np.pi).astype(np.complex64)
            else:
                # Fallback to brightness mode
                self.complex_field = (img_gray * amp_scale * np.exp(1j * phase_offset * 2 * np.pi)).astype(np.complex64)
                
        elif self.encoding_mode == "Laplacian":
            # Laplacian = real, original = imaginary (edge + content)
            lap = cv2.Laplacian(img_gray, cv2.CV_32F)
            lap_norm = lap / (np.abs(lap).max() + 1e-9)
            self.complex_field = ((lap_norm + 1j * img_gray) * amp_scale).astype(np.complex64)
            self.complex_field *= np.exp(1j * phase_offset * 2 * np.pi).astype(np.complex64)
            
        elif self.encoding_mode == "FFT (Standard)":
            # Standard FFT for comparison
            from scipy.fft import fft2
            self.complex_field = (fft2(img_gray) * amp_scale).astype(np.complex64)
            self.complex_field *= np.exp(1j * phase_offset * 2 * np.pi).astype(np.complex64)
            
        else:
            # Default: brightness to magnitude
            self.complex_field = (img_gray * amp_scale * np.exp(1j * phase_offset * 2 * np.pi)).astype(np.complex64)

    def get_output(self, port_name):
        if self.complex_field is None:
            return None
            
        if port_name == 'complex_spectrum':
            return self.complex_field
            
        elif port_name == 'magnitude_view':
            mag = np.abs(self.complex_field).astype(np.float32)
            if mag.max() > 0:
                mag = mag / mag.max()
            return (mag * 255).astype(np.uint8)
            
        elif port_name == 'phase_view':
            phase = np.angle(self.complex_field).astype(np.float32)
            # Map -pi..pi to 0..1
            phase_norm = (phase + np.pi) / (2 * np.pi)
            return (phase_norm * 255).astype(np.uint8)
            
        return None

    def get_display_image(self):
        if self.complex_field is None:
            return None
            
        h, w = self.complex_field.shape
        
        # Side by side: Magnitude | Phase
        display = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Magnitude (left)
        mag = np.abs(self.complex_field).astype(np.float32)
        if mag.max() > 0:
            mag = mag / mag.max()
        mag_u8 = (mag * 255).astype(np.uint8)
        display[:, :w] = cv2.applyColorMap(mag_u8, cv2.COLORMAP_INFERNO)
        
        # Phase (right)
        phase = np.angle(self.complex_field).astype(np.float32)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        phase_u8 = (phase_norm * 255).astype(np.uint8)
        display[:, w:] = cv2.applyColorMap(phase_u8, cv2.COLORMAP_HSV)
        
        # Labels
        cv2.putText(display, "Magnitude", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Phase", (w + 5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, self.encoding_mode, (5, h - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        mode_options = [
            ("Brightness→Magnitude", "Brightness→Magnitude"),
            ("Brightness→Phase", "Brightness→Phase"),
            ("Gradient→Complex", "Gradient→Complex"),
            ("Polar", "Polar"),
            ("Dual Channel", "Dual Channel"),
            ("Laplacian", "Laplacian"),
            ("FFT (Standard)", "FFT (Standard)"),
        ]
        return [
            ("Encoding Mode", "encoding_mode", self.encoding_mode, mode_options),
        ]