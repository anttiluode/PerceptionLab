"""
Conway's Game of Life Node - A classic cellular automaton
Place this file in the 'nodes' folder
Requires: pip install scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: GameOfLifeNode requires 'scipy'.")
    print("Please run: pip install scipy")

class GameOfLifeNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 140, 100) # A generative green
    
    def __init__(self, width=128, height=96, update_frames=5):
        super().__init__()
        self.node_title = "Game of Life"
        
        self.inputs = {
            'seed_image': 'image',
            'reset': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'population': 'signal'
        }
        
        self.w = int(width)
        self.h = int(height)
        self.update_interval = int(update_frames) # Run logic every N frames
        self.frame_count = 0
        
        self.grid = np.zeros((self.h, self.w), dtype=np.uint8)
        self.population = 0.0
        self.randomize() # Start with a random grid
        
        # Kernel for counting neighbors
        self.neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
        self.neighbor_kernel[1, 1] = 0
        
        if not SCIPY_AVAILABLE:
            self.node_title = "GoL (No SciPy!)"

    def randomize(self):
        """Re-seed the grid with random noise."""
        self.grid = np.random.randint(0, 2, (self.h, self.w), dtype=np.uint8)
        
    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # Check for reset signal
        reset_sig = self.get_blended_input('reset', 'sum')
        if reset_sig is not None and reset_sig > 0.5:
            self.randomize()
            
        # Check for seed image
        seed_img = self.get_blended_input('seed_image', 'mean')
        if seed_img is not None:
            seed_resized = cv2.resize(seed_img, (self.w, self.h))
            # "Paint" the seed onto the grid
            self.grid[seed_resized > 0.5] = 1
            
        # Control simulation speed
        self.frame_count += 1
        if self.frame_count < self.update_interval:
            return # Skip logic update this frame
        self.frame_count = 0

        # --- Conway's Game of Life Logic ---
        
        # 1. Count neighbors using 2D convolution
        neighbors = convolve2d(self.grid, self.neighbor_kernel, mode='same', boundary='wrap')
        
        # 2. Apply rules
        # Rule 1: A live cell with 2 or 3 neighbors stays alive
        stay_alive_mask = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        
        # Rule 2: A dead cell with exactly 3 neighbors becomes alive
        come_alive_mask = (self.grid == 0) & (neighbors == 3)
        
        # 3. Update grid
        self.grid = np.zeros_like(self.grid, dtype=np.uint8)
        self.grid[stay_alive_mask | come_alive_mask] = 1
        
        # 4. Update metrics
        self.population = np.mean(self.grid)

    def get_output(self, port_name):
        if port_name == 'image':
            return self.grid.astype(np.float32)
        elif port_name == 'population':
            return self.population
        return None
        
    def get_display_image(self):
        img_u8 = (self.grid * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Width", "w", self.w, None),
            ("Height", "h", self.h, None),
            ("Update (Frames)", "update_interval", self.update_interval, None),
        ]