"""
Holographic Cortical Stack V2 - "The Ghost Port"
================================================
Adds a dedicated 'holographic_in' port to allow phase-preserved feedback.
Prevents the "Broadcast Error" by separating 2D Sensory inputs from 3D Holographic inputs.

- sensory_in: Magnitude only (Webcam/Noise). Phase is forced to 0.
- holographic_in: Complex/Color (Feedback). Phase is PRESERVED.
"""

import numpy as np
import cv2
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class ComplexCorticalStackNode(BaseNode):
    NODE_CATEGORY = "Experimental"
    NODE_TITLE = "Holographic Stack (Ghost Port)"
    NODE_COLOR = QtGui.QColor(100, 0, 255) # Deep Purple
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'sensory_in': 'image',       # 2D Magnitude (Webcam/Noise)
            'holographic_in': 'image',   # 3D Color/Complex (Feedback Loop)
            'plasticity': 'signal',
            'deep_phase_shift': 'signal'
        }
        self.outputs = {
            'layer_1_holo': 'image', 
            'layer_2_holo': 'image', 
            'layer_3_holo': 'image', 
            'structure_mag': 'image' 
        }
        
        self.W, self.H = 64, 64
        self.n_layers = 3
        
        self.activity = np.zeros((self.n_layers, self.H, self.W), dtype=np.complex64)
        self.conductivity = np.ones((self.n_layers, self.H, self.W), dtype=np.float32) * 0.1
        self.feedforward = 0.2 + 0j
        self.feedback = 0.2 + 0j

    def _complex_to_hsv(self, z):
        """Converts complex field to HSV image."""
        mag = np.abs(z)
        phase = np.angle(z)
        v = np.tanh(mag * 2.0) * 255.0
        h = ((phase + np.pi) / (2 * np.pi)) * 179.0
        s = np.ones_like(h) * 200.0
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _hsv_to_complex(self, bgr_img):
        """Reconstructs complex field from HSV image (Preserving Phase)."""
        if bgr_img is None: return None
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Invert the mapping
        # H (0-179) -> Phase (-pi to pi)
        phase = (h.astype(np.float32) / 179.0) * (2 * np.pi) - np.pi
        
        # V (0-255) -> Mag (approx inverse tanh)
        # We just scale linearly for stability, tanh inverse is unstable at 1.0
        mag = v.astype(np.float32) / 255.0
        
        return mag * np.exp(1j * phase)

    def step(self):
        # 1. INPUTS
        rate = self.get_blended_input('plasticity', 'mean')
        if rate is None: rate = 0.05
        
        phase_shift = self.get_blended_input('deep_phase_shift', 'mean')
        if phase_shift is None: phase_shift = 0.0
        phase_shift_rad = phase_shift * 2.0 * np.pi

        # A. SENSORY (Magnitude, Phase 0)
        inp_mag = self.get_blended_input('sensory_in', 'mean')
        
        # B. HOLOGRAPHIC (Complex, Preserved Phase) - The Ghost
        inp_holo = self.get_blended_input('holographic_in', 'mean')
        
        # Mix them into Layer 0
        current_l0 = self.activity[0] * 0.8 # Decay
        
        # Inject Sensory
        if inp_mag is not None:
            if inp_mag.ndim > 2: inp_mag = cv2.cvtColor(inp_mag, cv2.COLOR_BGR2GRAY)
            inp_mag = cv2.resize(inp_mag.astype(np.float32), (self.W, self.H)) / 255.0
            current_l0 += inp_mag * np.exp(1j * 0.0) * 0.2
            
        # Inject Holographic Feedback (The Ghost)
        if inp_holo is not None:
            # Resize if needed
            if inp_holo.shape[:2] != (self.H, self.W):
                inp_holo = cv2.resize(inp_holo, (self.W, self.H))
            
            # Recover complex signal from the color image
            z_holo = self._hsv_to_complex(inp_holo)
            # Inject!
            current_l0 += z_holo * 0.2
            
        self.activity[0] = current_l0

        # 2. SIGNAL FLOW
        new_activity = self.activity.copy()
        
        for l in range(self.n_layers):
            # Diffusion
            curr = self.activity[l]
            cond = self.conductivity[l]
            blur_r = cv2.GaussianBlur(curr.real, (3,3), 0)
            blur_i = cv2.GaussianBlur(curr.imag, (3,3), 0)
            diffused = blur_r + 1j * blur_i
            new_activity[l] += (diffused - curr) * cond * 0.5
            
            # Feedforward
            if l < self.n_layers - 1:
                new_activity[l+1] += (self.activity[l] - self.activity[l+1]) * self.feedforward
            
            # Feedback
            if l > 0:
                source = self.activity[l]
                if l == self.n_layers - 1:
                    source = source * np.exp(1j * phase_shift_rad)
                new_activity[l-1] += (source - self.activity[l-1]) * self.feedback

        # 3. PLASTICITY
        for l in range(self.n_layers):
            mag = np.abs(self.activity[l])
            erosion = mag * rate * 0.1
            hardening = 0.005
            self.conductivity[l] = np.clip(self.conductivity[l] + erosion - hardening, 0.0, 1.0)

        self.activity = new_activity 

        # 4. OUTPUTS
        self.outputs['layer_1_holo'] = self._complex_to_hsv(self.activity[0])
        self.outputs['layer_2_holo'] = self._complex_to_hsv(self.activity[1])
        self.outputs['layer_3_holo'] = self._complex_to_hsv(self.activity[2])
        self.outputs['structure_mag'] = (self.conductivity[0] * 255).astype(np.uint8)

    def get_output(self, name):
        val = self.outputs.get(name)
        if val is None: return None
        if name == 'structure_mag':
             return cv2.cvtColor(val, cv2.COLOR_GRAY2BGR)
        return val