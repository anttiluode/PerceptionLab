"""
Large Moire Field Node - The "Eye" and "V1" of the Attentional Field Computer.
Encodes a webcam image into a single-channel "fast field" using a 
convolutional network and holographic (wave) evolution.

This is a simplified version of SensoryEncoderNode, focused only on 
generating the visual field and motion signal, without X/Y tracking.

Ported from afc6.py
Requires: pip install torch numpy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import sys
import os
import time 

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

# --- Dependency Check ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: LargeMoireFieldNode requires 'torch'.")
    print("Please run: pip install torch")

# Use GPU if available
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
except Exception:
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32

# --- Core Architectural Components (from afc6.py) ---

class HolographicField(nn.Module):
    """A field that evolves based on wave dynamics. (afc6.py)"""
    def __init__(self, dimensions=(64, 64), num_channels=1):
        super().__init__()
        self.dimensions = dimensions
        self.damping_map = nn.Parameter(torch.full((1, num_channels, *dimensions), 0.02, dtype=torch.float32))
        
        k_freq = [torch.fft.fftfreq(n, d=1 / n) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2 = sum(k ** 2 for k in k_grid)
        self.register_buffer('k2', k2)

    def evolve(self, field_state, steps=1):
        """Evolve the field state using spectral methods."""
        field_fft = torch.fft.fft2(field_state)
        decay = torch.exp(-self.k2.unsqueeze(0).unsqueeze(0) * F.softplus(self.damping_map))
        for _ in range(steps):
            field_fft *= decay
        return torch.fft.ifft2(field_fft).real

class SensoryEncoder(nn.Module):
    """The 'Eye' and 'V1'. Encodes images to a single-channel fast field. (afc6.py)"""
    def __init__(self, field_dims=(64, 64)):
        super().__init__()
        self.field = HolographicField(field_dims, num_channels=1)
        self.image_to_drive = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d(field_dims)
        )
        self.gamma_freq = 7.5
        self.receptive_threshold = 0.0

    def get_gamma_phase(self):
        return (time.time() * self.gamma_freq * 2 * np.pi) % (2 * np.pi)

    def is_receptive_phase(self, phase):
        return np.cos(phase) > self.receptive_threshold

    def forward(self, image_tensor):
        drive_pattern = self.image_to_drive(image_tensor)
        fast_pattern = self.field.evolve(drive_pattern, steps=5)
        phase = self.get_gamma_phase()
        receptive = self.is_receptive_phase(phase)
        return fast_pattern, phase, receptive

# --- The Main Node Class ---

class LargeMoireFieldNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 50, 200) # Deep purple
    
    def __init__(self, size=64):
        super().__init__()
        self.node_title = "Large Moire Field"
        
        self.inputs = {'image_in': 'image'}
        self.outputs = {
            'fast_field': 'image',    # The 64x64 evolved pattern
            'motion_signal': 'signal',# A signal representing change/motion
            'gamma_phase': 'signal',  # The internal clock signal
            'is_receptive': 'signal'  # The 1.0/0.0 gate signal
        }
        
        if not TORCH_AVAILABLE:
            self.node_title = "Moire Field (No Torch!)"
            return
            
        self.size = int(size)
        
        # 1. Initialize the PyTorch model
        self.model = SensoryEncoder(field_dims=(self.size, self.size)).to(DEVICE)
        self.model.eval() # Set to evaluation mode
        
        # 2. Internal state
        self.fast_field_data = np.zeros((self.size, self.size), dtype=np.float32)
        self.last_fast_field = torch.zeros(1, 1, self.size, self.size, device=DEVICE)
        self.motion_value = 0.0
        self.gamma_phase = 0.0
        self.is_receptive = 0.0

    @torch.no_grad() # Disable gradient calculations for speed
    def step(self):
        if not TORCH_AVAILABLE:
            return
            
        # 1. Get input image
        img_in = self.get_blended_input('image_in', 'mean')
        
        if img_in is None:
            # Evolve the last known field if no new input
            self.model.field.evolve(self.last_fast_field, steps=1)
            self.fast_field_data *= 0.95 # Fade out
            return
            
        # 2. Pre-process image for the model
        if img_in.ndim == 2: # Grayscale
            img_in = cv2.cvtColor(img_in.astype(np.float32), cv2.COLOR_GRAY2RGB)
        
        img_tensor = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor * 2.0 - 1.0).to(DEVICE)

        # 3. Run the model (forward pass)
        fast_pattern_tensor, phase, receptive = self.model(img_tensor)
        
        # 4. Calculate Motion
        motion_diff = torch.abs(fast_pattern_tensor - self.last_fast_field).mean()
        self.motion_value = motion_diff.item() * 100.0 
        
        # 5. Store outputs
        self.fast_field_data = fast_pattern_tensor.cpu().squeeze().numpy()
        self.last_fast_field = fast_pattern_tensor.detach()
        self.gamma_phase = (phase / (2 * np.pi)) * 2.0 - 1.0 
        self.is_receptive = 1.0 if receptive else 0.0

    def get_output(self, port_name):
        if port_name == 'fast_field':
            # Normalize for visualization
            max_val = np.max(self.fast_field_data)
            min_val = np.min(self.fast_field_data)
            range_val = max_val - min_val
            if range_val > 1e-9:
                return (self.fast_field_data - min_val) / range_val
            return self.fast_field_data
            
        elif port_name == 'motion_signal':
            return self.motion_value
        elif port_name == 'gamma_phase':
            return self.gamma_phase
        elif port_name == 'is_receptive':
            return self.is_receptive
        return None
        
    def get_display_image(self):
        # Display the fast field
        img_data = self.get_output('fast_field')
        if img_data is None: 
            return None
            
        img_u8 = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap (Inferno, as in afc6.py)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        
        # Add gate status bar
        if self.is_receptive:
            cv2.rectangle(img_color, (0, 0), (self.size, 5), (0, 255, 0), -1) # Green
        else:
            cv2.rectangle(img_color, (0, 0), (self.size, 5), (0, 0, 255), -1) # Red
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
        ]