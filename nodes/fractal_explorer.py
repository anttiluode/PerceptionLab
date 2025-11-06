"""
Fractal Explorer Nodes - Real-time Mandelbrot and Julia set generators
Requires: pip install numba
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

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: FractalExplorer nodes require 'numba'.")
    print("Please run: pip install numba")

# ======================================================================
# HIGH-SPEED JIT-COMPILED FRACTAL FUNCTIONS
# ======================================================================

@jit(nopython=True, fastmath=True)
def compute_mandelbrot(width, height, center_x, center_y, zoom, max_iter):
    """
    Fast Numba-compiled Mandelbrot set calculator.
    """
    result = np.zeros((height, width), dtype=np.int32)
    
    # Calculate scale
    scale = 2.0 / (width * zoom)
    
    for y in range(height):
        for x in range(width):
            # Map pixel to complex plane
            c_real = center_x + (x - width / 2) * scale
            c_imag = center_y + (y - height / 2) * scale
            
            z_real = 0.0
            z_imag = 0.0
            
            n = 0
            while n < max_iter:
                if z_real * z_real + z_imag * z_imag > 4.0:
                    break
                
                # z = z*z + c
                new_z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = new_z_real
                
                n += 1
                
            result[y, x] = n
            
    return result

@jit(nopython=True, fastmath=True)
def compute_julia(width, height, c_real, c_imag, max_iter):
    """
    Fast Numba-compiled Julia set calculator.
    """
    result = np.zeros((height, width), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            # Map pixel to z in complex plane
            z_real = (x - width / 2) * 2.0 / width
            z_imag = (y - height / 2) * 2.0 / height
            
            n = 0
            while n < max_iter:
                if z_real * z_real + z_imag * z_imag > 4.0:
                    break
                
                # z = z*z + c
                new_z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = new_z_real
                
                n += 1
                
            result[y, x] = n
            
    return result

# ======================================================================
# MANDELBROT NODE
# ======================================================================

class MandelbrotNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(50, 80, 160) # Deep blue
    
    def __init__(self, resolution=128, max_iterations=30):
        super().__init__()
        self.node_title = "Mandelbrot Explorer"
        
        self.inputs = {'zoom': 'signal', 'x_pos': 'signal', 'y_pos': 'signal'}
        self.outputs = {'image': 'image'}
        
        self.resolution = int(resolution)
        self.max_iterations = int(max_iterations)
        
        # Internal navigation state
        self.center_x = -0.7
        self.center_y = 0.0
        self.zoom = 0.5
        
        self.fractal_data = np.zeros((self.resolution, self.resolution), dtype=np.int32)
        
        if not NUMBA_AVAILABLE:
            self.node_title = "Mandelbrot (No Numba!)"

    def step(self):
        if not NUMBA_AVAILABLE:
            return
            
        # Get signals
        zoom_in = self.get_blended_input('zoom', 'sum') or 0.0
        move_x = self.get_blended_input('x_pos', 'sum') or 0.0
        move_y = self.get_blended_input('y_pos', 'sum') or 0.0
        
        # Update navigation state
        # A positive zoom signal (0 to 1) increases zoom
        self.zoom *= (1.0 + (zoom_in * 0.1))
        # Move signals ( -1 to 1) pan the view
        self.center_x += (move_x * 0.1) / self.zoom
        self.center_y += (move_y * 0.1) / self.zoom
        
        # Compute the fractal
        self.fractal_data = compute_mandelbrot(
            self.resolution, self.resolution,
            self.center_x, self.center_y,
            self.zoom, self.max_iterations
        )

    def get_output(self, port_name):
        if port_name == 'image':
            # Normalize iteration data to [0, 1]
            if self.max_iterations > 0:
                return self.fractal_data.astype(np.float32) / self.max_iterations
        return None
        
    def get_display_image(self):
        # Normalize and apply a color map
        img_norm = self.fractal_data.astype(np.float32) / self.max_iterations
        img_u8 = (img_norm * 255).astype(np.uint8)
        
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Max Iterations", "max_iterations", self.max_iterations, None),
        ]

# ======================================================================
# JULIA NODE
# ======================================================================

class JuliaNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 80, 180) # Generative Purple
    
    def __init__(self, resolution=128, max_iterations=40):
        super().__init__()
        self.node_title = "Julia Set Explorer"
        
        self.inputs = {'c_real': 'signal', 'c_imag': 'signal'}
        self.outputs = {'image': 'image'}
        
        self.resolution = int(resolution)
        self.max_iterations = int(max_iterations)
        
        # Internal state
        self.c_real = -0.7
        self.c_imag = 0.27015
        
        self.fractal_data = np.zeros((self.resolution, self.resolution), dtype=np.int32)
        
        if not NUMBA_AVAILABLE:
            self.node_title = "Julia (No Numba!)"

    def step(self):
        if not NUMBA_AVAILABLE:
            return
            
        # Get signals
        # Map input signals [-1, 1] to a good range for c, e.g., [-1, 1]
        self.c_real = self.get_blended_input('c_real', 'sum') or self.c_real
        self.c_imag = self.get_blended_input('c_imag', 'sum') or self.c_imag
        
        # Compute the fractal
        self.fractal_data = compute_julia(
            self.resolution, self.resolution,
            self.c_real, self.c_imag,
            self.max_iterations
        )

    def get_output(self, port_name):
        if port_name == 'image':
            # Normalize iteration data to [0, 1]
            if self.max_iterations > 0:
                return self.fractal_data.astype(np.float32) / self.max_iterations
        return None
        
    def get_display_image(self):
        # Normalize and apply a color map
        img_norm = self.fractal_data.astype(np.float32) / self.max_iterations
        img_u8 = (img_norm * 255).astype(np.uint8)
        
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Max Iterations", "max_iterations", self.max_iterations, None),
        ]