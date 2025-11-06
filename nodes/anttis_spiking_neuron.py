"""
Antti's Spiking Neuron - A Leaky Integrate-and-Fire (LIF) neuron
Transforms input signals into spikes. Can be chained.
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

class SpikingNeuronNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Neural orange
    
    def __init__(self, threshold=1.0, tau_m=0.1, resistance=5.0, refractory_ms=0.05):
        super().__init__()
        self.node_title = "Spiking Neuron (LIF)"
        
        self.inputs = {'signal_in': 'signal'}
        self.outputs = {'spike_out': 'signal'}
        
        # --- Neuron Parameters ---
        # These are configurable (see get_config_options)
        self.V_rest = 0.0
        self.V_threshold = float(threshold)
        self.V_reset = 0.0
        self.tau_m = float(tau_m)             # Membrane time constant (sec)
        self.R_m = float(resistance)          # Membrane resistance (scales input)
        self.refractory_period = float(refractory_ms) # Refractory period (sec)
        
        # --- Neuron State ---
        self.V_m = self.V_rest                # Current membrane potential
        self.refractory_timer = 0.0           # Countdown timer for refractory period
        self.output_signal = 0.0              # Output spike
        self.dt = 1.0 / 30.0                  # Assume ~30 FPS step rate

    def step(self):
        # 1. Reset output
        self.output_signal = 0.0
        
        # 2. Check refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= self.dt
            self.V_m = self.V_reset # Keep potential at reset
            return

        # 3. Get total input current (crucially, using 'sum' blend mode)
        # This allows multiple neurons to connect and sum their inputs
        I_in = self.get_blended_input('signal_in', 'sum') or 0.0
        
        # 4. Leaky Integrate-and-Fire (LIF) equation
        # tau_m * dV/dt = (V_rest - V) + R_m * I_in
        # dV = [ (V_rest - V_m) + (R_m * I_in) ] / tau_m * dt
        dV = (((self.V_rest - self.V_m) + self.R_m * I_in) / self.tau_m) * self.dt
        
        self.V_m += dV
        
        # 5. Check for spike
        if self.V_m >= self.V_threshold:
            self.output_signal = 1.0          # Fire!
            self.V_m = self.V_reset           # Reset potential
            self.refractory_timer = self.refractory_period # Start refractory timer

    def get_output(self, port_name):
        if port_name == 'spike_out':
            return self.output_signal
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Max voltage to display (to see threshold)
        max_viz_v = self.V_threshold * 1.2
        
        # Draw threshold line (Red)
        thresh_y = h - int(np.clip(self.V_threshold / max_viz_v, 0, 1) * h)
        cv2.line(img, (0, thresh_y), (w, thresh_y), (0, 0, 255), 1)

        # Draw resting line (Gray)
        rest_y = h - int(np.clip(self.V_rest / max_viz_v, 0, 1) * h)
        cv2.line(img, (0, rest_y), (w, rest_y), (100, 100, 100), 1)

        # Draw membrane potential bar
        vm_y = h - int(np.clip(self.V_m / max_viz_v, 0, 1) * h)
        
        if self.output_signal == 1.0:
            bar_color = (0, 255, 255) # Yellow
        elif self.refractory_timer > 0:
            bar_color = (255, 100, 0) # Blue
        else:
            bar_color = (0, 255, 0) # Green
            
        cv2.rectangle(img, (w//2 - 5, vm_y), (w//2 + 5, h), bar_color, -1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Threshold", "V_threshold", self.V_threshold, None),
            ("Leak (tau_m)", "tau_m", self.tau_m, None),
            ("Input (R_m)", "R_m", self.R_m, None),
            ("Refractory (sec)", "refractory_period", self.refractory_period, None),
        ]