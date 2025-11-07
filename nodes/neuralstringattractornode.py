"""
Neural String Attractor Node - Converts phase space coordinates into a strange attractor
Inspired by the Neural String Attractor HTML system.
Uses multiple "neural strings" that resonate with input frequencies and generate attractor dynamics.

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class NeuralString:
    """A single vibrating neural string with frequency resonance"""
    def __init__(self, string_id, length=64):
        self.id = string_id
        self.length = length
        self.values = np.random.randn(length).astype(np.float32) * 0.1
        self.previous_values = self.values.copy()
        
        # String properties
        self.frequency = 100 + np.random.rand() * 900  # 100-1000 Hz
        self.phase = np.random.rand() * 2 * np.pi
        self.energy = 0.0
        self.coherence = 0.0
        self.is_active = False
        
    def apply_frequency(self, input_freq, amplitude=0.1):
        """Apply frequency modulation with resonance"""
        # Calculate resonance (peaks when input_freq matches string frequency)
        resonance = np.exp(-np.abs(self.frequency - input_freq) / 200.0)
        
        # Update phase
        self.phase += self.frequency * 0.01 * resonance
        self.phase %= (2 * np.pi)
        
        # Apply wave to string
        for i in range(self.length):
            spatial_phase = (i / self.length) * 2 * np.pi
            wave = np.sin(self.phase + spatial_phase) * amplitude * resonance
            self.values[i] += wave
            
        return resonance
    
    def update(self):
        """Update string physics (diffusion and damping)"""
        self.previous_values = self.values.copy()
        
        # Diffusion (neighbor averaging)
        for i in range(1, self.length - 1):
            diffusion = (self.values[i-1] + self.values[i+1] - 2 * self.values[i]) * 0.1
            self.values[i] += diffusion
            
        # Damping
        self.values *= 0.99
        
        # Calculate metrics
        self.energy = np.sqrt(np.mean(self.values ** 2))
        
        # Coherence (lower variance = higher coherence)
        mean_val = np.mean(self.values)
        variance = np.mean((self.values - mean_val) ** 2)
        self.coherence = np.exp(-variance)
        
        self.is_active = self.energy > 0.01
        
    def get_output(self):
        """Get scalar output representing string state"""
        return self.energy * self.coherence * np.sin(self.phase)


class NeuralStringAttractorNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 60, 180)  # Neural purple
    
    def __init__(self, num_strings=8, string_length=64):
        super().__init__()
        self.node_title = "Neural String Attractor"
        
        self.inputs = {
            'phase_x': 'signal',      # From WebcamPhaseNode
            'phase_y': 'signal',
            'phase_z': 'signal',
            'energy': 'signal',       # Used to modulate frequency
            'frequency': 'signal'     # Direct frequency control
        }
        
        self.outputs = {
            'attractor_x': 'signal',  # 3D attractor coordinates
            'attractor_y': 'signal',
            'attractor_z': 'signal',
            'coherence': 'signal',    # Average string coherence
            'attractor_image': 'image',  # Visual trajectory
            'string_viz': 'image'     # Neural strings visualization
        }
        
        self.num_strings = int(num_strings)
        self.string_length = int(string_length)
        
        # Create neural strings
        self.strings = [NeuralString(i, self.string_length) for i in range(self.num_strings)]
        
        # Attractor trajectory history
        self.trajectory = np.zeros((500, 3), dtype=np.float32)
        self.trajectory_idx = 0
        
        # Current attractor position
        self.attractor_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Visualization buffers
        self.attractor_img = np.zeros((128, 128), dtype=np.uint8)
        self.strings_img = np.zeros((64, 128), dtype=np.uint8)
        
        # Base frequency (modulated by inputs)
        self.base_frequency = 1000.0
        
    def step(self):
        # Get inputs
        phase_x = self.get_blended_input('phase_x', 'sum') or 0.0
        phase_y = self.get_blended_input('phase_y', 'sum') or 0.0
        phase_z = self.get_blended_input('phase_z', 'sum') or 0.0
        energy = self.get_blended_input('energy', 'sum') or 0.0
        freq_control = self.get_blended_input('frequency', 'sum')
        
        # Calculate input frequency (base + modulation from energy)
        if freq_control is not None:
            # Direct frequency control (map [-1,1] to [500, 2000] Hz)
            input_frequency = 500 + (freq_control + 1.0) * 750.0
        else:
            # Frequency from energy (500-2000 Hz range)
            input_frequency = 500 + energy * 1500.0
            
        self.base_frequency = input_frequency
        
        # Update each neural string
        active_count = 0
        total_coherence = 0.0
        
        for string in self.strings:
            resonance = string.apply_frequency(input_frequency, energy)
            string.update()
            
            if string.is_active:
                active_count += 1
            total_coherence += string.coherence
            
        avg_coherence = total_coherence / self.num_strings
        
        # Generate attractor point from neural string outputs
        outputs = np.array([s.get_output() for s in self.strings])
        
        # Combine string outputs into 3D attractor coordinates
        # Mix phase space inputs with neural string dynamics
        self.attractor_pos[0] = (outputs[0] + outputs[1] * 0.5 + outputs[2] * 0.25) + phase_x * 0.3
        self.attractor_pos[1] = (outputs[3] + outputs[4] * 0.5 + outputs[5] * 0.25) + phase_y * 0.3
        self.attractor_pos[2] = (outputs[6] + outputs[7] * 0.5 + avg_coherence) + phase_z * 0.3
        
        # Clamp to reasonable range
        self.attractor_pos = np.clip(self.attractor_pos, -2.0, 2.0)
        
        # Store in trajectory
        self.trajectory[self.trajectory_idx] = self.attractor_pos
        self.trajectory_idx = (self.trajectory_idx + 1) % len(self.trajectory)
        
        # Update visualizations
        self._update_attractor_viz()
        self._update_strings_viz()
        
    def _update_attractor_viz(self):
        """Render 2D projection of 3D attractor trajectory"""
        self.attractor_img *= 0  # Clear
        
        # Project 3D to 2D (X-Y plane with Z affecting brightness)
        for i in range(len(self.trajectory)):
            x, y, z = self.trajectory[i]
            
            # Map to image coordinates
            px = int((x + 2.0) / 4.0 * 127)
            py = int((y + 2.0) / 4.0 * 127)
            
            px = np.clip(px, 0, 127)
            py = np.clip(py, 0, 127)
            
            # Brightness from Z and age
            age_factor = 1.0 - (abs(i - self.trajectory_idx) / len(self.trajectory))
            z_factor = (z + 2.0) / 4.0
            brightness = int(age_factor * z_factor * 255)
            
            # Draw point
            self.attractor_img[py, px] = max(self.attractor_img[py, px], brightness)
            
        # Blur for smooth trails
        self.attractor_img = cv2.GaussianBlur(self.attractor_img, (3, 3), 0)
        
    def _update_strings_viz(self):
        """Render neural strings as waveforms"""
        self.strings_img *= 0  # Clear
        
        h, w = self.strings_img.shape
        
        for i, string in enumerate(self.strings):
            if not string.is_active:
                continue
                
            # Y position for this string
            y_base = int((i + 0.5) / self.num_strings * h)
            
            # Draw waveform
            for j in range(self.string_length):
                x = int(j / self.string_length * w)
                
                # Wave amplitude
                amp = string.values[j]
                y_offset = int(amp * 10)
                y = np.clip(y_base + y_offset, 0, h - 1)
                
                # Brightness from energy
                brightness = int(string.energy * 255)
                
                self.strings_img[y, x] = max(self.strings_img[y, x], brightness)
                
    def get_output(self, port_name):
        if port_name == 'attractor_x':
            return float(self.attractor_pos[0])
        elif port_name == 'attractor_y':
            return float(self.attractor_pos[1])
        elif port_name == 'attractor_z':
            return float(self.attractor_pos[2])
        elif port_name == 'coherence':
            return float(np.mean([s.coherence for s in self.strings]))
        elif port_name == 'attractor_image':
            return self.attractor_img.astype(np.float32) / 255.0
        elif port_name == 'string_viz':
            return self.strings_img.astype(np.float32) / 255.0
        return None
        
    def get_display_image(self):
        # Show attractor visualization
        img_u8 = np.ascontiguousarray(self.attractor_img)
        
        # Apply colormap for better visibility
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
        
        # Draw current position marker
        x = int((self.attractor_pos[0] + 2.0) / 4.0 * 127)
        y = int((self.attractor_pos[1] + 2.0) / 4.0 * 127)
        x = np.clip(x, 0, 127)
        y = np.clip(y, 0, 127)
        cv2.circle(img_color, (x, y), 3, (255, 255, 255), -1)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Number of Strings", "num_strings", self.num_strings, None),
            ("String Length", "string_length", self.string_length, None),
        ]