"""
FFT Cochlea Node - Performs frequency analysis on signals and images
Place this file in the 'nodes' folder
"""

import numpy as np
import math
import cv2
from scipy.fft import rfft
from PyQt6 import QtGui

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

class FFTCochleaNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40)
    
    def __init__(self, freq_bins=64):
        super().__init__()
        self.node_title = "FFT Cochlea"
        self.inputs = {'image': 'image', 'signal': 'signal'}
        self.outputs = {
            'spectrum': 'spectrum', 
            'signal': 'signal', 
            'image': 'image', 
            'complex_spectrum': 'complex_spectrum'
        }
        
        self.freq_bins = freq_bins
        self.buffer = np.zeros(128, dtype=np.float32)
        self.x = 0.0
        self.internal_freq = np.random.uniform(2.0, 15.0)
        self.cochlea_img = np.zeros((64, 64), dtype=np.uint8) 
        self.spectrum_data = None
        self.complex_spectrum_data = None
        
    def step(self):
        u = self.get_blended_input('signal', 'sum') or 0.0
        
        alpha = 0.45
        decay = 0.92
        gain = 0.9
        
        newx = decay * self.x + gain * math.tanh(u + alpha * self.x)
        self.x = newx
        
        self.buffer *= 0.998
        if abs(self.x) > 0.09:
            amp = np.tanh(self.x) * 0.25
            t = np.linspace(0, 1, 10)
            sig = amp * np.sin(2*np.pi*(self.internal_freq + amp*10) * t)
            self.buffer[:-len(sig)] = self.buffer[len(sig):]
            self.buffer[-len(sig):] = sig
            
        img = self.get_blended_input('image', 'mean')
        if img is not None:
            self.compute_image_spectrum(img)
        else:
            self.compute_buffer_spectrum()
            
    def compute_buffer_spectrum(self):
        f = np.fft.fft(self.buffer)
        fsh = np.fft.fftshift(f)
        mag = np.abs(fsh)
        center = len(mag)//2
        half = min(self.freq_bins//2, center-1)
        spec = mag[center-half:center+half]
        self.spectrum_data = spec
        self.complex_spectrum_data = None
        self.update_display_from_spectrum(spec)
        
    def compute_image_spectrum(self, img):
        if img.ndim != 2:
            return
        
        spec = rfft(img.astype(np.float64), axis=1)
        self.complex_spectrum_data = spec.copy()
        mag = np.abs(spec)
        
        if mag.shape[1] > self.freq_bins:
            indices = np.linspace(0, mag.shape[1]-1, self.freq_bins).astype(int)
            mag = mag[:, indices]
        
        self.spectrum_data = np.mean(mag, axis=0)
        
        display = np.log1p(mag)
        display = (display - display.min()) / (display.max() - display.min() + 1e-9)
        
        h_target, w_target = self.cochlea_img.shape
        display_u8 = (display * 255).astype(np.uint8)
        self.cochlea_img = cv2.resize(display_u8, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        
    def update_display_from_spectrum(self, spec):
        arr = np.log1p(spec)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        
        w, h = self.cochlea_img.shape
        self.cochlea_img = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(min(len(arr), w)):
            v = int(255 * arr[i])
            self.cochlea_img[h - v:, i] = 255
        self.cochlea_img = np.flipud(self.cochlea_img)
        
    def get_output(self, port_name):
        if port_name == 'spectrum':
            return self.spectrum_data
        elif port_name == 'signal':
            return self.x
        elif port_name == 'image':
            return self.cochlea_img.astype(np.float32) / 255.0
        elif port_name == 'complex_spectrum':
            return self.complex_spectrum_data
        return None
        
    def get_display_image(self):
        img = np.ascontiguousarray(self.cochlea_img)
        h, w = img.shape
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        
    def randomize(self):
        self.internal_freq = np.random.uniform(2.0, 15.0)
        self.x = np.random.uniform(-0.5, 0.5)