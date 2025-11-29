"""
Complex Spectrum to Image Adapter
Extracts viewable image data from complex number fields.
Multiple decoding modes for different visualizations.
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


class ComplexToImageNode(BaseNode):
    """
    Adapter: Complex Spectrum (Purple Port) → Image
    
    Extracts viewable images from complex number fields.
    
    Decoding modes:
    - Magnitude: |z| - amplitude/power
    - Phase: arg(z) - angle
    - Real: Re(z) - real component
    - Imaginary: Im(z) - imaginary component  
    - Interference: Re(z * e^(i*t)) - animated phase scan
    - Color Encode: Magnitude→Brightness, Phase→Hue
    """
    NODE_CATEGORY = "Adapter"
    NODE_TITLE = "Complex → Image"
    NODE_COLOR = QtGui.QColor(180, 100, 220)  # Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_spectrum': 'complex_spectrum',
            'phase_scan': 'signal',        # For interference mode
            'contrast': 'signal'           # Gamma adjustment
        }
        
        self.outputs = {
            'image_out': 'image',
            'magnitude': 'image',
            'phase': 'image'
        }
        
        self.decoding_mode = "Magnitude"
        self.output_image = None
        self.t = 0
        
    def step(self):
        self.t += 1
        
        spectrum = self.get_blended_input('complex_spectrum', 'mean')
        phase_scan = self.get_blended_input('phase_scan', 'sum')
        contrast = self.get_blended_input('contrast', 'sum')
        
        if spectrum is None:
            return
            
        if not np.iscomplexobj(spectrum):
            # If real array passed, treat as magnitude with zero phase
            spectrum = spectrum.astype(np.complex64)
        else:
            # Ensure complex64 not complex128 (OpenCV hates float64)
            spectrum = spectrum.astype(np.complex64)
            
        # Default contrast
        if contrast is None:
            contrast = 1.0
        gamma = 0.5 + contrast * 1.5  # Range 0.5 to 2.0
        
        # Phase scan: use input or auto-animate
        if phase_scan is None:
            scan_phase = self.t * 0.05  # Auto rotate
        else:
            scan_phase = phase_scan * 2 * np.pi
            
        # === DECODING MODES ===
        
        if self.decoding_mode == "Magnitude":
            result = np.abs(spectrum).astype(np.float32)
            
        elif self.decoding_mode == "Phase":
            phase = np.angle(spectrum)
            result = ((phase + np.pi) / (2 * np.pi)).astype(np.float32)  # 0 to 1
            
        elif self.decoding_mode == "Real":
            result = np.real(spectrum).astype(np.float32)
            # Shift to positive
            result = result - result.min()
            
        elif self.decoding_mode == "Imaginary":
            result = np.imag(spectrum).astype(np.float32)
            result = result - result.min()
            
        elif self.decoding_mode == "Interference":
            # Multiply by rotating phasor and take real part
            # This "scans" through the hologram
            scanned = spectrum * np.exp(1j * scan_phase).astype(np.complex64)
            result = np.real(scanned).astype(np.float32)
            result = result - result.min()
            
        elif self.decoding_mode == "Log Magnitude":
            mag = np.abs(spectrum).astype(np.float32)
            result = np.log(1 + mag * 10)
            
        elif self.decoding_mode == "Color Encode":
            # HSV: Hue = phase, Value = magnitude
            mag = np.abs(spectrum).astype(np.float32)
            phase = np.angle(spectrum).astype(np.float32)
            
            if mag.max() > 0:
                mag_norm = mag / mag.max()
            else:
                mag_norm = mag
                
            hue = ((phase + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            sat = np.ones_like(hue) * 255
            val = (np.power(mag_norm, gamma) * 255).astype(np.uint8)
            
            hsv = np.stack([hue, sat, val], axis=-1)
            self.output_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return  # Skip normalization below
            
        elif self.decoding_mode == "Gradient Decode":
            # Inverse of gradient encoding: real=dx, imag=dy
            # Integrate to recover image (Poisson reconstruction approx)
            dx = np.real(spectrum).astype(np.float32)
            dy = np.imag(spectrum).astype(np.float32)
            # Simple integration (cumsum approximation)
            result = np.cumsum(dx, axis=1) + np.cumsum(dy, axis=0)
            result = result - result.min()
            
        else:
            result = np.abs(spectrum).astype(np.float32)
            
        # Normalize to 0-1
        result = result.astype(np.float32)
        if result.max() > result.min():
            result = (result - result.min()) / (result.max() - result.min())
        else:
            result = np.zeros_like(result, dtype=np.float32)
            
        # Apply gamma/contrast
        result = np.power(result, gamma)
        
        # Convert to uint8
        self.output_image = (result * 255).astype(np.uint8)

    def get_output(self, port_name):
        if self.output_image is None:
            return None
            
        if port_name == 'image_out':
            return self.output_image
            
        elif port_name == 'magnitude':
            spectrum = self.get_blended_input('complex_spectrum', 'mean')
            if spectrum is None:
                return None
            mag = np.abs(spectrum).astype(np.float32)
            if mag.max() > 0:
                mag = mag / mag.max()
            return (mag * 255).astype(np.uint8)
            
        elif port_name == 'phase':
            spectrum = self.get_blended_input('complex_spectrum', 'mean')
            if spectrum is None:
                return None
            phase = np.angle(spectrum).astype(np.float32)
            phase_norm = (phase + np.pi) / (2 * np.pi)
            return (phase_norm * 255).astype(np.uint8)
            
        return None

    def get_display_image(self):
        if self.output_image is None:
            return None
            
        img = self.output_image
        
        # Handle color vs grayscale
        if img.ndim == 2:
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        else:
            img_color = img.copy()
            
        h, w = img_color.shape[:2]
        
        cv2.putText(img_color, self.decoding_mode, (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, w, h, w * 3, 
                           QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        mode_options = [
            ("Magnitude", "Magnitude"),
            ("Phase", "Phase"),
            ("Real", "Real"),
            ("Imaginary", "Imaginary"),
            ("Interference", "Interference"),
            ("Log Magnitude", "Log Magnitude"),
            ("Color Encode", "Color Encode"),
            ("Gradient Decode", "Gradient Decode"),
        ]
        return [
            ("Decoding Mode", "decoding_mode", self.decoding_mode, mode_options),
        ]