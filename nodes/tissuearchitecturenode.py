# tissuearchitectnode.py
"""
Tissue Architect Node (The Leggett Assembler)
---------------------------------------------
Implements the Physics of the Leggett et al. (2019) paper.
Combines 'Raw Matter' (Noise) with 'Anatomy' (Eigenmodes)
using Diffusion-Limited Aggregation (DLA) and Jamming physics.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class TissueArchitectNode(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(40, 150, 100) # Biological Green

    def __init__(self):
        super().__init__()
        self.node_title = "Tissue Architect (DLA)"
        
        self.inputs = {
            'anatomy_mask': 'image',   # The Eigenmode (The Blueprint)
            'bio_matter': 'image',     # Pink Noise (The Raw Material)
            'jamming_limit': 'signal'  # Density limit (Stop growing)
        }
        
        self.outputs = {
            'tissue_structure': 'image', # The Resulting Growth
            'density_map': 'image',      # Where is it jammed?
            'active_growth': 'image'     # Where is it growing right now?
        }
        
        self.resolution = 256
        # The living tissue state (Persistent)
        self.tissue_grid = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Parameters
        self.growth_rate = 0.1
        self.decay_rate = 0.01 # Tissue naturally dies off slowly
        self.jamming_threshold = 0.8 # Confluency limit

    def step(self):
        # 1. Get Inputs
        mask = self.get_blended_input('anatomy_mask', 'first')
        matter = self.get_blended_input('bio_matter', 'first')
        jam_sig = self.get_blended_input('jamming_limit', 'sum')
        
        if jam_sig is not None:
            self.jamming_threshold = np.clip(jam_sig, 0.1, 1.0)

        # Handle missing inputs (safety)
        if mask is None: mask = np.zeros((self.resolution, self.resolution))
        if matter is None: matter = np.random.rand(self.resolution, self.resolution)

        # Resize inputs if necessary
        if mask.shape != self.tissue_grid.shape:
            mask = cv2.resize(mask, (self.resolution, self.resolution))
        if matter.shape != self.tissue_grid.shape:
            matter = cv2.resize(matter, (self.resolution, self.resolution))

        # 2. The Leggett Physics Engine
        
        # A. Availability: Where is there matter to grow? (From Pink Noise)
        available_matter = matter
        
        # B. Guidance: Where does the DNA want to grow? (From Eigenmode)
        # We treat the Eigenmode as a probability field.
        guidance = mask
        
        # C. Jamming: Where is it already full?
        # If tissue > threshold, growth is inhibited.
        jamming_factor = 1.0 - np.clip(self.tissue_grid, 0, self.jamming_threshold) / self.jamming_threshold
        jamming_factor = np.clip(jamming_factor, 0, 1)
        
        # D. The Growth Step
        # New Growth = Matter * Guidance * Space_Available
        new_growth = available_matter * guidance * jamming_factor * self.growth_rate
        
        # Apply Growth
        self.tissue_grid += new_growth
        
        # E. Apply Metabolism (Decay)
        # Tissue needs energy to stay alive. If the 'Anatomy' moves (Eye moves), 
        # the old tissue behind it should die off (or stay as scar tissue).
        # We decay slightly everywhere.
        self.tissue_grid *= (1.0 - self.decay_rate)
        
        # Clip to stable range
        self.tissue_grid = np.clip(self.tissue_grid, 0, 1.0)

    def get_output(self, port_name):
        if port_name == 'tissue_structure':
            return self.tissue_grid
        elif port_name == 'density_map':
            return self.tissue_grid # In this simple model, structure = density
        elif port_name == 'active_growth':
            return self.tissue_grid # Placeholder
        return None

    def get_display_image(self):
        # Render: Green tissue on black background
        img_u8 = (self.tissue_grid * 255).astype(np.uint8)
        
        # Use a "Tissue" colormap (Pink/Red/White)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_PINK)
        
        return QtGui.QImage(img_color.data, self.resolution, self.resolution, self.resolution * 3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Growth Speed", "growth_rate", self.growth_rate, None),
            ("Metabolic Decay", "decay_rate", self.decay_rate, None),
            ("Jamming Limit", "jamming_threshold", self.jamming_threshold, None)
        ]