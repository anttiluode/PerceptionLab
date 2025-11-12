"""
Chaotic Field Node - Ultra-sensitive nonlinear dynamical system
Based on Whisper Quantum Computer principles but integrated for Perception Lab

Acts as a computational substrate for probabilistic operations on latent vectors.
Uses Lorenz attractor dynamics extended to N dimensions.
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ChaoticFieldNode(BaseNode):
    """
    Simulates a chaotic attractor field that can be gently biased.
    Replaces simple Gaussian noise with structured chaotic dynamics.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 100, 220)
    
    def __init__(self, dimensions=16, chaos_strength=1.0):
        super().__init__()
        self.node_title = "Chaotic Field"
        
        self.inputs = {
            'state_in': 'spectrum',  # Input latent vector
            'bias_vector': 'spectrum',  # Gentle statistical bias (Whisper Gate)
            'measurement_trigger': 'signal',  # Collapse to definite state
            'chaos_strength': 'signal',  # Modulate chaos intensity
            'reset': 'signal'
        }
        self.outputs = {
            'field_state': 'spectrum',  # Current chaotic state
            'collapsed_state': 'spectrum',  # After measurement
            'coherence': 'signal',  # How stable the field is
            'energy': 'signal'  # Field energy level
        }
        
        self.dimensions = int(dimensions)
        self.chaos_strength = float(chaos_strength)
        
        # Internal chaotic state
        self.field = np.random.randn(self.dimensions) * 0.01
        self.velocity = np.zeros(self.dimensions)
        self.coherence_level = 1.0
        self.energy_level = 0.0
        
        # Lorenz-like parameters for chaos
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        
        # History for coherence tracking
        self.history = []
        self.max_history = 50
        
    def step(self):
        state_in = self.get_blended_input('state_in', 'first')
        bias = self.get_blended_input('bias_vector', 'first')
        measure = self.get_blended_input('measurement_trigger', 'sum') or 0.0
        chaos_mod = self.get_blended_input('chaos_strength', 'sum')
        reset_signal = self.get_blended_input('reset', 'sum') or 0.0
        
        if chaos_mod is not None:
            chaos_strength = chaos_mod
        else:
            chaos_strength = self.chaos_strength
            
        # Reset field
        if reset_signal > 0.5:
            if state_in is not None:
                self.field = state_in.copy() * 0.1  # Seed from input
            else:
                self.field = np.random.randn(self.dimensions) * 0.01
            self.coherence_level = 1.0
            self.history = []
            
        # Inject input state as gentle attraction
        if state_in is not None and len(state_in) >= self.dimensions:
            attraction = (state_in[:self.dimensions] - self.field) * 0.01
            self.field += attraction
            
        # Chaotic evolution (Lorenz attractor per triplet of dimensions)
        dt = 0.01 * chaos_strength
        
        # Process dimensions in groups of 3 (Lorenz triplets)
        for i in range(0, self.dimensions - 2, 3):
            x, y, z = self.field[i:i+3]
            
            # Lorenz equations
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            
            # Apply gentle bias (Whisper Gate influence)
            if bias is not None and i + 2 < len(bias):
                dx += bias[i] * 0.001  # Ultra-light influence
                dy += bias[i+1] * 0.001
                dz += bias[i+2] * 0.001
                
            self.velocity[i:i+3] = [dx, dy, dz]
            
        # Handle remaining dimensions (if not divisible by 3)
        remainder = self.dimensions % 3
        if remainder > 0:
            idx = self.dimensions - remainder
            # Simple damped oscillator for remaining dims
            self.velocity[idx:] = -self.field[idx:] * 0.5
            
        # Update field
        self.field += self.velocity * dt
        
        # Add ultra-light noise (like audio hardware noise in Whisper)
        self.field += np.random.randn(self.dimensions) * 0.0001 * chaos_strength
        
        # Calculate energy
        self.energy_level = np.sum(self.field ** 2)
        
        # Store history
        self.history.append(self.field.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Calculate coherence (low variance over time = high coherence)
        if len(self.history) > 10:
            recent = np.array(self.history[-10:])
            variance = np.var(recent, axis=0).mean()
            self.coherence_level = 1.0 / (1.0 + variance * 10.0)
        
        # Coherence degrades naturally over time (decoherence)
        self.coherence_level *= 0.998
        
        # Measurement collapses the field
        if measure > 0.5:
            # "Measure" by amplifying dominant modes and suppressing others
            self.collapsed = np.tanh(self.field * 5.0)
            self.coherence_level = 0.0  # Measurement destroys coherence
        else:
            self.collapsed = self.field.copy()
            
    def get_output(self, port_name):
        if port_name == 'field_state':
            return self.field.astype(np.float32)
        elif port_name == 'collapsed_state':
            return self.collapsed.astype(np.float32)
        elif port_name == 'coherence':
            return float(self.coherence_level)
        elif port_name == 'energy':
            return float(self.energy_level)
        return None
        
    def get_display_image(self):
        """Visualize field state and coherence"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Top half: Field state as waveform
        bar_width = max(1, w // self.dimensions)
        
        # Normalize field for display
        field_norm = self.field.copy()
        field_max = np.abs(field_norm).max()
        if field_max > 1e-6:
            field_norm = field_norm / field_max
            
        for i, val in enumerate(field_norm):
            x = i * bar_width
            h_bar = int(abs(val) * 80)
            y_base = 100
            
            # Color by value sign
            if val >= 0:
                color = (0, int(255 * abs(val)), 255)
                cv2.rectangle(img, (x, y_base-h_bar), (x+bar_width-1, y_base), color, -1)
            else:
                color = (255, int(255 * abs(val)), 0)
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+h_bar), color, -1)
                
        # Baseline
        cv2.line(img, (0, 100), (w, 100), (100,100,100), 1)
        
        # Bottom half: Coherence indicator
        coherence_width = int(self.coherence_level * w)
        coherence_color = (0, int(255 * self.coherence_level), 0)
        cv2.rectangle(img, (0, 180), (coherence_width, 200), coherence_color, -1)
        
        # Text info
        cv2.putText(img, f"Coherence: {self.coherence_level:.3f}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"Energy: {self.energy_level:.3f}", (5, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Chaos indicator
        chaos_text = "CHAOTIC" if self.coherence_level < 0.3 else "COHERENT"
        chaos_color = (0, 0, 255) if self.coherence_level < 0.3 else (0, 255, 0)
        cv2.putText(img, chaos_text, (5, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, chaos_color, 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Dimensions", "dimensions", self.dimensions, None),
            ("Chaos Strength", "chaos_strength", self.chaos_strength, None)
        ]