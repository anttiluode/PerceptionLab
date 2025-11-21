"""
Eigen-Spatial Projector Node
----------------------------
Maps 5 EEG frequency bands (Delta, Theta, Alpha, Beta, Gamma) to 
3D Spherical Harmonics to visualize the "Global Workspace" shape.

Inputs:
- delta, theta, alpha, beta, gamma: Signal inputs (power)
- delta_phase, etc.: Signal inputs (phase, optional)

Outputs:
- projection_image: 2D rendering of the 3D eigen-shape
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from scipy.special import sph_harm

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------

class EigenSpatialProjectorNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(150, 100, 255) # Violet
    
    def __init__(self, resolution=128):
        super().__init__()
        self.node_title = "Eigen-Spatial Projector"
        
        self.inputs = {
            'delta': 'signal', 'theta': 'signal', 
            'alpha': 'signal', 'beta': 'signal', 'gamma': 'signal'
        }
        
        self.outputs = {
            'projection_image': 'image'
        }
        
        self.resolution = int(resolution)
        self.display_img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        
        # Precompute sphere grid
        self.theta, self.phi = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
        
        # Harmonic definitions (l, m) for each band
        self.harmonics = {
            'delta': (1, 0), # Dipole
            'theta': (2, 0), # Quadrupole
            'alpha': (2, 1),
            'beta': (3, 0),
            'gamma': (3, 2)
        }

    def step(self):
        # 1. Get Band Powers
        powers = {}
        for band in self.harmonics:
            val = self.get_blended_input(band, 'sum')
            powers[band] = val if val is not None else 0.0
            
        # 2. Construct Shape (Linear combination of spherical harmonics)
        # Radius r(theta, phi) = 1 + sum( power * Y_lm(theta, phi) )
        
        r = np.ones_like(self.theta) * 2.0 # Base radius
        
        for band, (l, m) in self.harmonics.items():
            weight = powers[band]
            if weight > 0.01:
                Y_lm = sph_harm(m, l, self.phi, self.theta)
                # Take real part for geometry
                r += weight * np.real(Y_lm) * 2.0
                
        # 3. Render (Simple 3D to 2D projection)
        # Convert spherical to cartesian
        x = r * np.sin(self.theta) * np.cos(self.phi)
        y = r * np.sin(self.theta) * np.sin(self.phi)
        z = r * np.cos(self.theta)
        
        # Project to 2D image plane (Orthographic)
        # Rotate slightly to see structure
        rot_x = x + z * 0.5
        rot_y = y + z * 0.2
        
        # Normalize to image bounds
        scale = self.resolution / 8.0
        center = self.resolution / 2.0
        
        px = (rot_x * scale + center).astype(int)
        py = (rot_y * scale + center).astype(int)
        
        # Draw
        self.display_img.fill(0)
        
        # Mask for valid pixels
        mask = (px >= 0) & (px < self.resolution) & (py >= 0) & (py < self.resolution)
        
        # Color map based on radius (depth)
        colors = ((r - r.min()) / (r.max() - r.min() + 1e-9) * 255).astype(np.uint8)
        
        # Draw points (simple cloud)
        for i in range(px.shape[0]):
            for j in range(px.shape[1]):
                if mask[i, j]:
                    c = int(colors[i, j])
                    # Pseudo-depth shading
                    cv2.circle(self.display_img, (px[i, j], py[i, j]), 1, (c, c, 255), -1)
                    
        # Apply glow
        self.display_img = cv2.GaussianBlur(self.display_img, (3, 3), 0)

    def get_output(self, port_name):
        if port_name == 'projection_image':
            return self.display_img.astype(np.float32) / 255.0
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, self.resolution, self.resolution, 
                           self.resolution * 3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None)
        ]