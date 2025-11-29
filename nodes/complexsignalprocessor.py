"""
Complex Signal Processor
Manipulates complex spectra via signal inputs.
All parameters controllable by other nodes in real-time.
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


class ComplexSignalProcessorNode(BaseNode):
    """
    Complex Spectrum Processor with Signal Control
    
    Takes a complex spectrum and manipulates it based on signal inputs.
    All operations preserve the complex nature and stay within working range.
    
    Signal inputs (all 0-1 range, centered at 0.5 for bidirectional):
    - phase_rotate: Global phase rotation (0.5 = no change)
    - magnitude: Amplitude scaling (0.5 = unity, 0 = zero, 1 = 2x)
    - freq_shift_x/y: Translate in frequency space (0.5 = no shift)
    - phase_noise: Add phase noise (0 = none, 1 = full scramble)
    - band_center: Center frequency for bandpass (0-1)
    - band_width: Width of frequency band (0 = DC only, 1 = all)
    - spatial_rotate: Rotate image via phase gradient (0.5 = no rotation)
    - contrast: Magnitude contrast/gamma (0.5 = linear)
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Complex Signal Processor"
    NODE_COLOR = QtGui.QColor(180, 100, 220)  # Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_in': 'complex_spectrum',
            'phase_rotate': 'signal',      # 0-1, 0.5 = no change
            'magnitude': 'signal',         # 0-1, 0.5 = unity gain
            'freq_shift_x': 'signal',      # 0-1, 0.5 = no shift
            'freq_shift_y': 'signal',      # 0-1, 0.5 = no shift
            'phase_noise': 'signal',       # 0-1, amount of phase scramble
            'band_center': 'signal',       # 0-1, center of bandpass
            'band_width': 'signal',        # 0-1, width of bandpass
            'spatial_rotate': 'signal',    # 0-1, 0.5 = no rotation
            'contrast': 'signal',          # 0-1, magnitude gamma
            'mix': 'signal',               # 0-1, dry/wet mix
        }
        
        self.outputs = {
            'complex_out': 'complex_spectrum',
            'magnitude_view': 'image',
            'phase_view': 'image',
            'diff_view': 'image',          # Difference from input
        }
        
        self.complex_field = None
        self.input_field = None
        self.size = 128
        
        # For smooth noise (not jarring per-frame)
        self.noise_phase = None
        self.noise_seed = 0
        
        # Debug: track actual signal values
        self.debug_signals = {}
        
    def step(self):
        # Get complex input
        spectrum = self.get_blended_input('complex_in', 'mean')
        
        if spectrum is None:
            return
            
        # Ensure complex64
        if not np.iscomplexobj(spectrum):
            spectrum = spectrum.astype(np.complex64)
        else:
            spectrum = spectrum.astype(np.complex64)
            
        self.input_field = spectrum.copy()
        h, w = spectrum.shape
        self.size = max(h, w)
        
        # Get all signal inputs (default to neutral values)
        phase_rot = self.get_blended_input('phase_rotate', 'sum')
        magnitude = self.get_blended_input('magnitude', 'sum')
        freq_x = self.get_blended_input('freq_shift_x', 'sum')
        freq_y = self.get_blended_input('freq_shift_y', 'sum')
        phase_noise = self.get_blended_input('phase_noise', 'sum')
        band_center = self.get_blended_input('band_center', 'sum')
        band_width = self.get_blended_input('band_width', 'sum')
        spatial_rot = self.get_blended_input('spatial_rotate', 'sum')
        contrast = self.get_blended_input('contrast', 'sum')
        mix = self.get_blended_input('mix', 'sum')
        
        # Default neutral values
        if phase_rot is None: phase_rot = 0.5
        if magnitude is None: magnitude = 0.5
        if freq_x is None: freq_x = 0.5
        if freq_y is None: freq_y = 0.5
        if phase_noise is None: phase_noise = 0.0
        if band_center is None: band_center = 0.5
        if band_width is None: band_width = 1.0
        if spatial_rot is None: spatial_rot = 0.5
        if contrast is None: contrast = 0.5
        if mix is None: mix = 1.0
        
        # Clamp to working range
        phase_rot = np.clip(phase_rot, 0, 1)
        magnitude = np.clip(magnitude, 0, 1)
        freq_x = np.clip(freq_x, 0, 1)
        freq_y = np.clip(freq_y, 0, 1)
        phase_noise = np.clip(phase_noise, 0, 1)
        band_center = np.clip(band_center, 0, 1)
        band_width = np.clip(band_width, 0.01, 1)  # Avoid zero width
        spatial_rot = np.clip(spatial_rot, 0, 1)
        contrast = np.clip(contrast, 0, 1)
        mix = np.clip(mix, 0, 1)
        
        # DEBUG: Store values for display
        self.debug_signals = {
            'phase': phase_rot,
            'mag': magnitude,
            'freqX': freq_x,
            'freqY': freq_y,
            'noise': phase_noise,
            'bandC': band_center,
            'bandW': band_width,
            'rot': spatial_rot,
            'contr': contrast,
            'mix': mix,
        }
        
        # === PROCESSING CHAIN ===
        
        result = spectrum.copy()
        
        # 1. Global Phase Rotation
        # 0.5 = no change, 0 = -π, 1 = +π
        phase_offset = (phase_rot - 0.5) * 2 * np.pi
        result = result * np.exp(1j * phase_offset).astype(np.complex64)
        
        # 2. Magnitude Scaling with Contrast
        # Extract magnitude and phase
        mag = np.abs(result).astype(np.float32)
        phase = np.angle(result).astype(np.float32)
        
        # Normalize magnitude for processing
        mag_max = mag.max()
        if mag_max > 0:
            mag_norm = mag / mag_max
        else:
            mag_norm = mag
            
        # Apply contrast (gamma)
        # contrast 0.5 = gamma 1.0 (linear)
        # contrast 0 = gamma 2.0 (compress), contrast 1 = gamma 0.5 (expand)
        gamma = 2.0 - contrast * 1.5  # Range 2.0 to 0.5
        gamma = max(0.1, gamma)  # Safety
        mag_norm = np.power(mag_norm, gamma)
        
        # Scale magnitude
        # magnitude 0.5 = 1.0x, 0 = 0x, 1 = 2x
        mag_scale = magnitude * 2.0
        mag_norm = mag_norm * mag_scale
        
        # Reconstruct (keep original scale reference)
        if mag_max > 0:
            result = (mag_norm * mag_max * np.exp(1j * phase)).astype(np.complex64)
        
        # 3. Frequency Shift (translation in frequency domain)
        # This is equivalent to modulation in spatial domain
        if abs(freq_x - 0.5) > 0.01 or abs(freq_y - 0.5) > 0.01:
            shift_x = int((freq_x - 0.5) * w)
            shift_y = int((freq_y - 0.5) * h)
            result = np.roll(result, shift_x, axis=1)
            result = np.roll(result, shift_y, axis=0)
        
        # 4. Phase Noise (controlled scrambling)
        if phase_noise > 0.01:
            # Generate or update noise field
            if self.noise_phase is None or self.noise_phase.shape != (h, w):
                self.noise_phase = np.random.uniform(-np.pi, np.pi, (h, w)).astype(np.float32)
                
            # Slowly evolve noise for organic feel
            self.noise_seed += 0.02
            noise_evolution = np.sin(self.noise_phase + self.noise_seed)
            
            # Mix noise into phase
            current_phase = np.angle(result)
            noise_amount = phase_noise * np.pi * noise_evolution
            new_phase = current_phase + noise_amount
            
            # Reconstruct with noisy phase
            result = (np.abs(result) * np.exp(1j * new_phase)).astype(np.complex64)
        
        # 5. Bandpass Filter
        if band_width < 0.99:
            # Create frequency coordinate grid
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            
            # Distance from center (normalized 0-1)
            r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2).astype(np.float32)
            r = r / r.max()  # Normalize
            
            # Bandpass parameters
            center = band_center
            width = band_width * 0.5  # Half-width
            
            # Gaussian bandpass
            band_response = np.exp(-((r - center) ** 2) / (2 * width ** 2 + 1e-6))
            band_response = band_response.astype(np.float32)
            
            # Apply to shifted spectrum
            from scipy.fft import fftshift, ifftshift
            result_shifted = fftshift(result)
            result_shifted = result_shifted * band_response
            result = ifftshift(result_shifted)
        
        # 6. Spatial Rotation (via linear phase gradient)
        # Adding a linear phase ramp rotates the image when inverse transformed
        if abs(spatial_rot - 0.5) > 0.01:
            rotation_amount = (spatial_rot - 0.5) * 2 * np.pi
            
            # Create rotation via phase gradient
            # This creates a "tilt" in phase space
            y_grid, x_grid = np.meshgrid(
                np.linspace(-1, 1, w),
                np.linspace(-1, 1, h)
            )
            
            # Circular phase ramp for rotation effect
            angle_grid = np.arctan2(x_grid, y_grid)
            phase_ramp = rotation_amount * 0.5 * (
                np.cos(angle_grid) * x_grid + np.sin(angle_grid) * y_grid
            )
            
            result = result * np.exp(1j * phase_ramp).astype(np.complex64)
        
        # 7. Dry/Wet Mix
        if mix < 0.99:
            result = (spectrum * (1 - mix) + result * mix).astype(np.complex64)
        
        # Normalize to prevent explosion
        result_max = np.abs(result).max()
        input_max = np.abs(spectrum).max()
        if result_max > 0 and input_max > 0:
            # Keep similar energy level to input
            scale_factor = input_max / result_max
            # Soft limiting - don't scale up too much
            scale_factor = min(scale_factor, 2.0)
            result = result * scale_factor
        
        self.complex_field = result.astype(np.complex64)

    def get_output(self, port_name):
        if self.complex_field is None:
            return None
            
        if port_name == 'complex_out':
            return self.complex_field
            
        elif port_name == 'magnitude_view':
            mag = np.abs(self.complex_field).astype(np.float32)
            if mag.max() > 0:
                mag = mag / mag.max()
            return (mag * 255).astype(np.uint8)
            
        elif port_name == 'phase_view':
            phase = np.angle(self.complex_field).astype(np.float32)
            phase_norm = (phase + np.pi) / (2 * np.pi)
            return (phase_norm * 255).astype(np.uint8)
            
        elif port_name == 'diff_view':
            if self.input_field is None:
                return None
            # Magnitude of difference
            diff = np.abs(self.complex_field - self.input_field).astype(np.float32)
            if diff.max() > 0:
                diff = diff / diff.max()
            return (diff * 255).astype(np.uint8)
            
        return None

    def get_display_image(self):
        if self.complex_field is None:
            return None
            
        h, w = self.complex_field.shape
        
        # Three panels: Input Mag | Output Mag | Phase
        panel_w = w
        display = np.zeros((h, panel_w * 3, 3), dtype=np.uint8)
        
        # Input magnitude (left)
        if self.input_field is not None:
            in_mag = np.abs(self.input_field).astype(np.float32)
            if in_mag.max() > 0:
                in_mag = in_mag / in_mag.max()
            in_u8 = (in_mag * 255).astype(np.uint8)
            display[:, :panel_w] = cv2.applyColorMap(in_u8, cv2.COLORMAP_VIRIDIS)
        
        # Output magnitude (center)
        out_mag = np.abs(self.complex_field).astype(np.float32)
        if out_mag.max() > 0:
            out_mag = out_mag / out_mag.max()
        out_u8 = (out_mag * 255).astype(np.uint8)
        display[:, panel_w:panel_w*2] = cv2.applyColorMap(out_u8, cv2.COLORMAP_INFERNO)
        
        # Output phase (right)
        phase = np.angle(self.complex_field).astype(np.float32)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        phase_u8 = (phase_norm * 255).astype(np.uint8)
        display[:, panel_w*2:] = cv2.applyColorMap(phase_u8, cv2.COLORMAP_HSV)
        
        # Labels
        cv2.putText(display, "IN", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "OUT", (panel_w + 5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "PHASE", (panel_w * 2 + 5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # DEBUG: Show actual signal values received
        y_pos = h - 8
        for name, val in self.debug_signals.items():
            if val is not None and abs(val - 0.5) > 0.01:  # Only show non-neutral
                txt = f"{name[:6]}:{val:.2f}"
                cv2.putText(display, txt, (5, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                y_pos -= 12
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return []  # All control via signals - no config needed!