"""
EEG Simulator Node - Generates a simulated multi-channel EEG signal
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

class EEGSimulatorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80) # Source Green
    
    def __init__(self, sample_rate=250.0):
        super().__init__()
        self.node_title = "EEG Simulator"
        self.outputs = {'signal': 'signal'}
        
        self.sample_rate = sample_rate
        self.time = 0.0
        
        # Inspired by MNE channel names
        self.channels = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "T7", "T8"]
        self.selected_channel = self.channels[0]
        
        # Internal state for each channel's oscillators
        self.channel_state = {}
        for ch in self.channels:
            self.channel_state[ch] = {
                'phase': np.random.rand(4) * 2 * np.pi,
                'freqs': np.array([
                    np.random.uniform(2, 4),    # Delta
                    np.random.uniform(5, 8),    # Theta
                    np.random.uniform(9, 12),   # Alpha
                    np.random.uniform(15, 25)   # Beta
                ]),
                'amps': np.array([
                    np.random.uniform(0.5, 1.0),
                    np.random.uniform(0.2, 0.5),
                    np.random.uniform(0.1, 0.8), # Alpha can be strong
                    np.random.uniform(0.05, 0.2)
                ]) * 0.2 # Scale down
            }
        
        self.output_value = 0.0
        self.history = np.zeros(64) # For display

    def step(self):
        dt = 1.0 / self.sample_rate
        self.time += dt
        
        # Get the state for the selected channel
        state = self.channel_state[self.selected_channel]
        
        # Update phases
        state['phase'] += state['freqs'] * dt * 2 * np.pi
        
        # Compute sines
        sines = np.sin(state['phase'])
        
        # Modulate alpha rhythm (make it bursty)
        alpha_mod = (np.sin(self.time * 0.2 * 2 * np.pi) + 1.0) / 2.0 # Slow modulation
        sines[2] *= alpha_mod
        
        # Sum oscillators
        signal = np.dot(sines, state['amps'])
        
        # Add noise
        noise = (np.random.rand() - 0.5) * 0.1
        self.output_value = signal + noise
        
        # Update display history
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.output_value

    def get_output(self, port_name):
        if port_name == 'signal':
            return self.output_value
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Normalize history from [-1, 1] to [0, h-1]
        vis_data = (self.history + 1.0) / 2.0 * (h - 1)
        
        for i in range(w - 1):
            y1 = int(np.clip(vis_data[i], 0, h - 1))
            y2 = int(np.clip(vis_data[i+1], 0, h - 1))
            # Draw line segment
            img = cv2.line(img, (i, y1), (i+1, y2), (255, 255, 255), 1)

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        # Create channel options for the dropdown menu
        channel_options = [(ch, ch) for ch in self.channels]
        
        return [
            ("Channel", "selected_channel", self.selected_channel, channel_options)
        ]
