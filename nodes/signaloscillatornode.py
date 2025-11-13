"""
Signal Oscillator Node
Generates a stable, rhythmic sine wave, acting as a
"Theta Wave Proxy" or "Gamma Clock" for temporal gating.
"""

import numpy as np
from PyQt6 import QtGui
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# --------------------------

class SignalOscillatorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80) # Source Green
    
    def __init__(self, frequency=8.0, amplitude=1.0, wave_type='sine'):
        super().__init__()
        self.node_title = "Signal Oscillator"
        
        self.inputs = {
            'freq_mod': 'signal',   # Modulate frequency
            'amp_mod': 'signal'    # Modulate amplitude
        }
        self.outputs = {
            'signal': 'signal'
        }
        
        # Configurable
        self.base_frequency = float(frequency)
        self.base_amplitude = float(amplitude)
        self.wave_type = str(wave_type)
        
        # Internal state
        self.current_frequency = self.base_frequency
        self.current_amplitude = self.base_amplitude
        self.phase = 0.0 # in radians
        self.output_value = 0.0
        
        # For display
        self.history = np.zeros(128, dtype=np.float32)

    def step(self):
        # 1. Get Inputs
        freq_mod = self.get_blended_input('freq_mod', 'sum') or 0.0
        amp_mod = self.get_blended_input('amp_mod', 'sum')
        
        # 2. Update Parameters
        # Freq mod is additive
        self.current_frequency = self.base_frequency * (1.0 + freq_mod)
        
        # Amp mod is multiplicative
        if amp_mod is not None:
            self.current_amplitude = self.base_amplitude * np.clip(amp_mod, 0.0, 1.0)
        else:
            self.current_amplitude = self.base_amplitude

        # 3. Calculate Phase Increment
        # Assuming a 30 FPS step rate for the host
        fps = 30.0
        phase_increment = (2 * np.pi * self.current_frequency) / fps
        self.phase = (self.phase + phase_increment) % (2 * np.pi)
        
        # 4. Generate Waveform
        if self.wave_type == 'sine':
            self.output_value = np.sin(self.phase) * self.current_amplitude
        elif self.wave_type == 'square':
            self.output_value = np.sign(np.sin(self.phase)) * self.current_amplitude
        elif self.wave_type == 'saw':
            self.output_value = ((self.phase / (2 * np.pi)) * 2.0 - 1.0) * self.current_amplitude
        
        # 5. Update display history
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.output_value

    def get_output(self, port_name):
        if port_name == 'signal':
            return self.output_value
        return None
        
    def get_display_image(self):
        w, h = 128, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Normalize history from [-A, +A] to [0, h-1]
        vis_data = (self.history / (2.0 * self.base_amplitude + 1e-9)) + 0.5
        vis_data = vis_data * (h - 1)
        
        for i in range(w - 1):
            if i >= len(vis_data): break
            y1 = int(np.clip(vis_data[i], 0, h - 1))
            y2 = int(np.clip(vis_data[i+1], 0, h - 1))
            cv2.line(img, (i, y1), (i+1, y2), (255, 255, 255), 1)

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Frequency (Hz)", "base_frequency", self.base_frequency, None),
            ("Amplitude", "base_amplitude", self.base_amplitude, None),
            ("Wave Type", "wave_type", self.wave_type, [
                ("Sine", "sine"),
                ("Square", "square"),
                ("Sawtooth", "saw")
            ])
        ]