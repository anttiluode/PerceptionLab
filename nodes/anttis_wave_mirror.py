"""
Antti's Wave Mirror - Learns an image, then evolves it.
Inspired by mirror.py
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import time

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

class WaveNeuron:
    """Simplified WaveNeuron class from mirror.py"""
    def __init__(self, w, h):
        # WaveNeuron is designed for grayscale/single-channel data (w, h)
        self.frequency = np.random.uniform(0.1, 1.0, (h, w)).astype(np.float32)
        self.amplitude = np.random.uniform(0.5, 1.0, (h, w)).astype(np.float32)
        self.phase = np.random.uniform(0, 2 * np.pi, (h, w)).astype(np.float32)
        
    def activate(self, input_signal, t):
        # Vectorized activation
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) + input_signal
        
    def train(self, target, t, learning_rate):
        # Target must be (h, w) shape to match output
        output = self.activate(0, t) # Get internal activation
        error = target - output
        
        sin_term = np.sin(2 * np.pi * self.frequency * t + self.phase)
        cos_term = np.cos(2 * np.pi * self.frequency * t + self.phase)
        
        self.amplitude += learning_rate * error * sin_term
        self.phase += learning_rate * error * self.amplitude * cos_term
        self.frequency += learning_rate * error * self.amplitude * (2 * np.pi * t) * cos_term
        
        # Clamp values to reasonable ranges
        self.amplitude = np.clip(self.amplitude, 0.1, 2.0)
        self.frequency = np.clip(self.frequency, 0.01, 2.0)

class WaveMirrorNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(60, 180, 160) # A teal/aqua color
    
    def __init__(self, width=80, height=60, training_duration=300):
        super().__init__()
        self.node_title = "Antti's Mirror"
        
        self.inputs = {'image_in': 'image'}
        self.outputs = {'image_out': 'image'}
        
        self.w, self.h = int(width), int(height)
        self.training_duration = int(training_duration)
        self.learning_rate = 0.01
        
        # Internal state
        self.wnn = WaveNeuron(self.w, self.h)
        self.output_image = np.zeros((self.h, self.w), dtype=np.float32)
        self.training_counter = 0
        self.start_time = time.time()
        self.is_trained = False

    def step(self):
        t = time.time() - self.start_time
        input_image = self.get_blended_input('image_in', 'mean')
        
        if input_image is None:
            input_image = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            # 1. Resize the input
            input_image = cv2.resize(input_image, (self.w, self.h), interpolation=cv2.INTER_AREA)

            # 2. FIX: Convert to Grayscale if the input is color (ndim == 3)
            if input_image.ndim == 3:
                # Convert BGR/RGB to Grayscale (assuming input is float 0-1)
                input_image = cv2.cvtColor(input_image.astype(np.float32), cv2.COLOR_BGR2GRAY)
            
        if self.training_counter < self.training_duration:
            # --- Training Phase ---
            self.wnn.train(input_image, t, self.learning_rate)
            self.training_counter += 1
            # Show the input image while training
            self.output_image = input_image
            self.is_trained = False
        else:
            # --- Evolution Phase ---
            if not self.is_trained:
                self.is_trained = True
                print("WaveMirror: Training complete. Entering evolution phase.")
                
            # "Lives its own life" by using 0 as input
            input_signal = np.zeros((self.h, self.w), dtype=np.float32)
            self.output_image = self.wnn.activate(input_signal, t)

    def get_output(self, port_name):
        if port_name == 'image_out':
            # Normalize for output
            out = self.output_image - np.min(self.output_image)
            out_max = np.max(out)
            if out_max > 1e-6:
                out = out / out_max
            return out
        return None
        
    def get_display_image(self):
        # Display internal state
        out_img = self.get_output('image_out')
        if out_img is None:
            out_img = np.zeros((self.h, self.w), dtype=np.float32)
            
        img_u8 = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)
        
        # Add status bar
        if not self.is_trained:
            status_color = (0, 255, 0) # Green for training
            progress = int((self.training_counter / self.training_duration) * self.w)
            cv2.rectangle(img_u8, (0, self.h - 5), (progress, self.h - 1), status_color, -1)
        else:
            status_color = (0, 0, 255) # Red for evolving
            cv2.rectangle(img_u8, (0, self.h - 5), (self.w - 1, self.h - 1), status_color, -1)

        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Training Frames", "training_duration", self.training_duration, None)
        ]
