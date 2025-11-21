"""
Holographic Interference Node
-----------------------------
Visualizes the interference pattern between two signals, treating one as a 
reference beam and the other as an object beam. This is fundamental to 
holographic reconstruction.

Inputs:
- reference_signal: The "reference beam" (e.g., Frontal EEG channel)
- object_signal: The "object beam" (e.g., Visual EEG channel)

Outputs:
- interference_pattern: Image visualizing the interference
- phase_difference: Signal representing the phase difference
- coherence: Signal representing the coherence (stability of phase difference)
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------

class HolographicInterferenceNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(100, 200, 255)  # Holographic Cyan
    
    def __init__(self, window_size=128):
        super().__init__()
        self.node_title = "Holographic Interference"
        
        self.inputs = {
            'reference_signal': 'signal',
            'object_signal': 'signal'
        }
        
        self.outputs = {
            'interference_pattern': 'image',
            'phase_difference': 'signal',
            'coherence': 'signal'
        }
        
        self.window_size = int(window_size)
        self.ref_buffer = deque(maxlen=self.window_size)
        self.obj_buffer = deque(maxlen=self.window_size)
        
        self.interference_img = np.zeros((128, 128, 3), dtype=np.uint8)
        self.current_phase_diff = 0.0
        self.current_coherence = 0.0
        
    def step(self):
        # 1. Get Inputs
        ref_sig = self.get_blended_input('reference_signal', 'sum')
        obj_sig = self.get_blended_input('object_signal', 'sum')
        
        if ref_sig is None or obj_sig is None:
            return
            
        self.ref_buffer.append(ref_sig)
        self.obj_buffer.append(obj_sig)
        
        if len(self.ref_buffer) < self.window_size:
            return
            
        # 2. Compute Analytic Signals (Hilbert Transform approximation)
        # For real-time, we can use a simple quadrature filter or just recent history
        # Here we use the recent buffer as a short time window
        
        ref_arr = np.array(self.ref_buffer)
        obj_arr = np.array(self.obj_buffer)
        
        # Simple FFT-based analytic signal for the window
        ref_fft = np.fft.fft(ref_arr)
        obj_fft = np.fft.fft(obj_arr)
        
        # Compute Cross-Spectrum
        cross_spec = ref_fft * np.conj(obj_fft)
        
        # 3. Extract Phase Difference and Coherence
        # Phase difference at the dominant frequency
        dom_freq_idx = np.argmax(np.abs(cross_spec))
        phase_diff = np.angle(cross_spec[dom_freq_idx])
        
        self.current_phase_diff = phase_diff / np.pi # Normalize to [-1, 1]
        
        # Coherence: Magnitude of mean cross-spectrum / mean of magnitudes
        # (Simplified time-domain coherence for this window)
        coherence = np.abs(np.mean(cross_spec)) / (np.std(ref_arr) * np.std(obj_arr) + 1e-9)
        self.current_coherence = np.clip(coherence, 0.0, 1.0)
        
        # 4. Visualize Interference Pattern
        # We create a 2D pattern where X represents time/phase and Y represents amplitude interaction
        
        h, w = 128, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map phase difference to Hue
        hue = int(((self.current_phase_diff + 1.0) / 2.0) * 179)
        
        # Map coherence to Saturation
        sat = int(self.current_coherence * 255)
        
        # Map instantaneous amplitude product to Value pattern
        # We'll draw interference fringes
        x = np.arange(w)
        freq = 5.0 # Fringe frequency
        
        # The "Hologram": Intensity = |R + O|^2 = |R|^2 + |O|^2 + 2|R||O|cos(phase_diff)
        # We visualize the cosine term (the interference)
        fringes = np.cos(x * freq * 0.1 + phase_diff)
        
        val_pattern = ((fringes + 1.0) / 2.0 * 255).astype(np.uint8)
        val_grid = np.tile(val_pattern, (h, 1))
        
        # Create HSV image
        hsv_img = np.zeros((h, w, 3), dtype=np.uint8)
        hsv_img[:, :, 0] = hue
        hsv_img[:, :, 1] = sat
        hsv_img[:, :, 2] = val_grid
        
        self.interference_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    def get_output(self, port_name):
        if port_name == 'interference_pattern':
            return self.interference_img.astype(np.float32) / 255.0
        elif port_name == 'phase_difference':
            return self.current_phase_diff
        elif port_name == 'coherence':
            return self.current_coherence
        return None

    def get_display_image(self):
        img = self.interference_img.copy()
        
        # Overlay stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Phase: {self.current_phase_diff:.2f}pi", (5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Coherence: {self.current_coherence:.2f}", (5, 30), font, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Window Size", "window_size", self.window_size, None)
        ]