"""
Spacetime Volume Node (The Minkowski Slicer)
--------------------------------------------
Treats the history of images as a 3D solid object (X, Y, Time).
Allows you to slice through Time to see the 'shape' of events.

Inputs:
    image_slice: The current 2D frame (a slice of 'Now').
    slice_axis: 0=XY (Normal), 1=XT (Slitscan), 2=YT (Waterfall).
    slice_index: Where to cut the crystal.
"""

import numpy as np
import cv2
from collections import deque
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SpacetimeVolumeNode(BaseNode):
    NODE_CATEGORY = "Deep Math"
    NODE_TITLE = "Spacetime Crystal"
    NODE_COLOR = QtGui.QColor(40, 80, 180) # Deep Time Blue
    
    def __init__(self, depth=128):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',
            'slice_axis': 'signal',  # 0=XY, 1=XT, 2=YT
            'slice_index': 'signal'  # Normalized 0-1
        }
        
        self.outputs = {
            'spacetime_slice': 'image',
            'temporal_complexity': 'signal' # Entropy of the time axis
        }
        
        self.depth = int(depth)
        # The Crystal: A circular buffer of frames
        # Shape: (Depth, Height, Width, Channels)
        self.volume = None 
        self.frame_idx = 0
        self.viz_cache = None

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        axis_sig = self.get_blended_input('slice_axis', 'sum')
        idx_sig = self.get_blended_input('slice_index', 'sum')
        
        if img_in is None:
            return

        # 1. Initialize Volume if needed
        if self.volume is None or self.volume.shape[1:3] != img_in.shape[:2]:
            h, w = img_in.shape[:2]
            # Ensure RGB
            if img_in.ndim == 2:
                c = 1
            else:
                c = img_in.shape[2]
            
            self.volume = np.zeros((self.depth, h, w, c), dtype=np.float32)
            
        # 2. Push 'Now' into the Crystal (Rolling buffer)
        # We roll the array so index 0 is always 'Now' or 'Oldest'
        self.volume = np.roll(self.volume, 1, axis=0)
        
        # Ensure dims match
        if img_in.ndim == 2:
            img_in = img_in[..., np.newaxis]
            
        self.volume[0] = img_in
        
        # 3. Slice the Crystal
        axis = int(axis_sig) if axis_sig is not None else 1 # Default to XT (Slitscan)
        axis = np.clip(axis, 0, 2)
        
        idx_norm = idx_sig if idx_sig is not None else 0.5
        
        if axis == 0: # XY Plane (Standard Video)
            # Slicing through Z (Time) gives a past frame
            t_idx = int(idx_norm * (self.depth - 1))
            sliced = self.volume[t_idx]
            
        elif axis == 1: # XT Plane (Slitscan)
            # Y is fixed, X and T vary
            # We slice at a specific Y height
            h_idx = int(idx_norm * (self.volume.shape[1] - 1))
            # Result shape: (Depth, Width, C) -> (Time, X, C)
            sliced = self.volume[:, h_idx, :, :]
            
        elif axis == 2: # YT Plane (Waterfall)
            # X is fixed, Y and T vary
            w_idx = int(idx_norm * (self.volume.shape[2] - 1))
            # Result shape: (Depth, Height, C) -> (Time, Y, C)
            sliced = self.volume[:, :, w_idx, :]

        # 4. Measure Temporal Complexity
        # Calculate variance along the time axis (How much did it change?)
        if self.volume is not None:
            temporal_variance = np.var(self.volume, axis=0).mean()
            self.set_output('temporal_complexity', float(temporal_variance))

        self.viz_cache = sliced
        
    def get_output(self, port_name):
        if port_name == 'spacetime_slice':
            return self.viz_cache
        return super().get_output(port_name) # Handle signal outputs via set_output

    def set_output(self, name, val):
        if not hasattr(self, 'outputs_data'): self.outputs_data = {}
        self.outputs_data[name] = val

    def get_display_image(self):
        if self.viz_cache is None:
            return None
            
        img = self.viz_cache
        
        # Normalize
        if img.max() > 0:
            img = img / img.max()
            
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Handle grayscale
        if img_u8.shape[-1] == 1:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
            
        # Resize for display
        h, w = img_u8.shape[:2]
        return QtGui.QImage(img_u8.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)