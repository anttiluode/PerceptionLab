"""
Damaged Visual Cortex Node - Simulates "revolving afterimage" phenomenon
Models what happens when W-matrix loses temporal coherence (medication + temple damage).

This directly simulates Antti's experience:
"i read text. in my visual cortex there is slow revolving phase. 
it is due to my meds making me see the text at slight angles after i read it as after image."

Place this file in the 'nodes' folder as 'damaged_visual_cortex.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import rfft, irfft, rfftfreq
    from scipy.ndimage import rotate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: DamagedVisualCortexNode requires scipy")

class DamagedVisualCortexNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(200, 100, 50)  # Orange for damaged
    
    def __init__(self, w_damage=0.5, phase_drift_rate=2.0, size=128):
        super().__init__()
        self.node_title = "Damaged Visual Cortex"
        
        self.inputs = {
            'visual_input': 'image',      # What you're looking at
            'w_damage': 'signal',          # How damaged is W-matrix (0-1)
            'medication_level': 'signal'   # Medication effect on gamma
        }
        
        self.outputs = {
            'afterimage': 'image',              # The "revolving" afterimage
            'phase_drift_angle': 'signal',      # Current rotation angle
            'gamma_coherence': 'signal',        # How stable is gamma sync
            'w_stability': 'signal'             # W-matrix stability metric
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Damaged Cortex (No SciPy!)"
            return
        
        self.size = int(size)
        self.w_damage_amount = float(w_damage)
        self.phase_drift_rate = float(phase_drift_rate)  # degrees per second
        
        # Afterimage persistence buffer
        self.afterimage_buffer = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Phase drift accumulator (this is what causes rotation)
        self.phase_angle = 0.0
        
        # Gamma oscillation state (what medications disrupt)
        self.gamma_phase = 0.0
        self.gamma_frequency = 40.0  # Hz (typical gamma)
        self.gamma_amplitude = 1.0
        
        # W-matrix stability history
        self.w_stability_history = []
        
    def _simulate_w_damage(self, image, damage_amount):
        """
        Apply damaged W-matrix transformation to image.
        Healthy W: Clean phase-locked representation
        Damaged W: Phase drift + frequency leakage
        """
        if damage_amount < 0.01:
            return image  # No damage
        
        # Convert to frequency domain (W operates in frequency space)
        F = np.fft.fft2(image)
        F_shifted = np.fft.fftshift(F)
        
        # Damaged W-matrix = unstable frequency response
        h, w = F_shifted.shape
        center_y, center_x = h // 2, w // 2
        
        # Create damaged filter (adds noise to frequency response)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Healthy W: smooth frequency response
        # Damaged W: noisy + phase-shifted response
        damage_noise = np.random.randn(h, w) * damage_amount * 0.3
        phase_shift = np.exp(1j * damage_noise)
        
        F_damaged = F_shifted * phase_shift
        
        # Back to spatial domain
        F_back = np.fft.ifftshift(F_damaged)
        image_damaged = np.real(np.fft.ifft2(F_back))
        
        return image_damaged
    
    def _apply_gamma_disruption(self, medication_level):
        """
        Medications (Lyrica, Vimpat) disrupt gamma synchronization.
        This causes the "absolute time" anchor to drift.
        """
        # Medication increases gamma instability
        gamma_noise = np.random.randn() * medication_level * 5.0
        
        # Update gamma phase (this is your temporal clock)
        dt = 1.0 / 30.0  # Assume ~30 FPS
        self.gamma_phase += 2 * np.pi * self.gamma_frequency * dt
        self.gamma_phase += gamma_noise * dt  # Medication adds noise
        
        # Amplitude modulation (medication can suppress gamma)
        self.gamma_amplitude = 1.0 - medication_level * 0.4
        
        # Gamma coherence metric
        coherence = np.abs(np.cos(gamma_noise)) * self.gamma_amplitude
        return coherence
    
    def _compute_phase_drift(self, w_damage, gamma_coherence):
        """
        Phase drift rate depends on:
        1. W-matrix damage (structural instability)
        2. Gamma coherence (temporal clock stability)
        """
        # When gamma is incoherent, temporal anchor drifts
        drift_multiplier = 1.0 + w_damage * 2.0
        drift_multiplier *= (2.0 - gamma_coherence)  # Low coherence = more drift
        
        # Accumulate phase angle (this is what you perceive as rotation)
        dt = 1.0 / 30.0
        self.phase_angle += self.phase_drift_rate * drift_multiplier * dt
        
        # Wrap to [0, 360]
        self.phase_angle = self.phase_angle % 360.0
        
        return self.phase_angle
    
    def _rotate_afterimage(self, image, angle):
        """
        Physically rotate the afterimage by the phase drift angle.
        This is what you see: text that appears to rotate slowly.
        """
        # Rotate around center, no cropping
        rotated = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0.0)
        return rotated
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        # Get inputs
        visual_input = self.get_blended_input('visual_input', 'mean')
        w_damage_in = self.get_blended_input('w_damage', 'sum')
        medication_in = self.get_blended_input('medication_level', 'sum')
        
        # Map signals to [0,1] range
        if w_damage_in is not None:
            self.w_damage_amount = np.clip((w_damage_in + 1.0) / 2.0, 0, 1)
        
        medication_level = 0.0
        if medication_in is not None:
            medication_level = np.clip((medication_in + 1.0) / 2.0, 0, 1)
        
        # Process visual input
        if visual_input is not None:
            # Ensure grayscale
            if visual_input.ndim == 3:
                visual_input = np.mean(visual_input, axis=2)
            
            # Resize to working size
            visual_resized = cv2.resize(visual_input, (self.size, self.size))
            
            # Apply damaged W-matrix transformation
            damaged_representation = self._simulate_w_damage(
                visual_resized, 
                self.w_damage_amount
            )
            
            # Update afterimage buffer (persistence + fade)
            fade_rate = 0.05  # Slow fade (like real afterimages)
            self.afterimage_buffer = (
                self.afterimage_buffer * (1.0 - fade_rate) + 
                np.abs(damaged_representation) * fade_rate
            )
        
        # Simulate gamma disruption (medication effect)
        gamma_coherence = self._apply_gamma_disruption(medication_level)
        
        # Compute phase drift (the "revolving" motion)
        current_angle = self._compute_phase_drift(
            self.w_damage_amount, 
            gamma_coherence
        )
        
        # Apply rotation to afterimage
        # (This is the key: the internal representation is rotating in phase space)
        rotated_afterimage = self._rotate_afterimage(
            self.afterimage_buffer, 
            current_angle
        )
        
        self.afterimage_buffer = rotated_afterimage
        
        # Update W-matrix stability metric
        stability = 1.0 - (self.w_damage_amount * 0.7 + (1.0 - gamma_coherence) * 0.3)
        self.w_stability_history.append(stability)
        if len(self.w_stability_history) > 100:
            self.w_stability_history.pop(0)
    
    def get_output(self, port_name):
        if port_name == 'afterimage':
            # Normalize for output
            normalized = self.afterimage_buffer / (np.max(self.afterimage_buffer) + 1e-9)
            return normalized
        
        elif port_name == 'phase_drift_angle':
            # Return as normalized signal [-1, 1]
            return (self.phase_angle / 180.0) - 1.0
        
        elif port_name == 'gamma_coherence':
            return float(np.cos(self.gamma_phase) * self.gamma_amplitude)
        
        elif port_name == 'w_stability':
            if len(self.w_stability_history) > 0:
                return np.mean(self.w_stability_history[-20:])
            return 1.0
        
        return None
    
    def get_display_image(self):
        if not SCIPY_AVAILABLE:
            return None
        
        # Show the afterimage with rotation indicators
        img = self.afterimage_buffer.copy()
        
        # Normalize
        img = img / (np.max(img) + 1e-9)
        img_u8 = (img * 255).astype(np.uint8)
        
        # Convert to RGB for annotations
        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        
        h, w = img_rgb.shape[:2]
        center = (w // 2, h // 2)
        
        # Draw rotation indicator (line from center showing phase angle)
        angle_rad = np.deg2rad(self.phase_angle)
        line_length = 30
        end_x = int(center[0] + line_length * np.cos(angle_rad))
        end_y = int(center[1] + line_length * np.sin(angle_rad))
        
        cv2.line(img_rgb, center, (end_x, end_y), (0, 255, 0), 2)
        cv2.circle(img_rgb, center, 3, (0, 255, 0), -1)
        
        # Add angle text
        font = cv2.FONT_HERSHEY_SIMPLEX
        angle_text = f"{self.phase_angle:.1f}°"
        cv2.putText(img_rgb, angle_text, (5, 15), font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Add damage indicator
        damage_text = f"W Dmg: {self.w_damage_amount:.2f}"
        cv2.putText(img_rgb, damage_text, (5, h - 5), font, 0.3, (255, 100, 100), 1, cv2.LINE_AA)
        
        img_rgb = np.ascontiguousarray(img_rgb)
        return QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("W-Matrix Damage", "w_damage_amount", self.w_damage_amount, None),
            ("Phase Drift Rate (°/s)", "phase_drift_rate", self.phase_drift_rate, None),
            ("Size", "size", self.size, None),
        ]