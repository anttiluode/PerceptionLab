"""
Ricci Flow Manifold Node (FIXED v2)
------------------------------------
Implements Hamilton's Ricci Flow equation: ∂g/∂t = -2Ric(g)

FIXED v2: 
- Proper dimension handling (accepts 32x32 directly)
- Removed 'default' parameter from get_blended_input
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode

class RicciFlowNode(BaseNode):
    NODE_CATEGORY = "Deep Math"
    NODE_TITLE = "Ricci Flow Manifold"
    NODE_COLOR = QtGui.QColor(100, 0, 150)
    
    def __init__(self):
        super().__init__()
        self.inputs = {'metric_tensor': 'signal'} 
        self.outputs = {
            'curvature_scalar': 'signal', 
            'manifold_vis': 'image',
            'winding_number': 'signal'
        }
        
        # Use dim=34 so that interior (dim-2) = 32
        self.dim = 34
        self.manifold = np.random.rand(self.dim, self.dim).astype(np.float64) * 0.5 + 0.25
        self.dt = 0.01
        
        # Track accumulated curvature for winding number
        self.total_curvature = 0.0

    def _compute_ricci_curvature(self, g):
        """
        Approximates Ricci Curvature on interior points.
        Input: (N, N) array
        Output: (N-2, N-2) array (interior only)
        """
        g_center = g[1:-1, 1:-1]
        g_up = g[0:-2, 1:-1]
        g_down = g[2:, 1:-1]
        g_left = g[1:-1, 0:-2]
        g_right = g[1:-1, 2:]
        
        # Discrete Laplacian
        laplacian = (g_up + g_down + g_left + g_right) - 4 * g_center
        
        return -0.5 * laplacian

    def step(self):
        # 1. Get Input - NO default parameter
        input_energy = self.get_blended_input('metric_tensor')
        
        if input_energy is not None:
            # Add perturbation at center
            center = self.dim // 2
            perturbation = np.clip(input_energy * 0.0001, -0.1, 0.1)
            self.manifold[center, center] += perturbation
        
        # 2. Compute Ricci Curvature
        # Pad manifold (34x34 → 36x36)
        g_padded = np.pad(self.manifold, 1, mode='edge')
        
        # Compute curvature on interior (36x36 → 34x34)
        ricci_tensor = self._compute_ricci_curvature(g_padded)
        
        # 3. Apply Ricci Flow
        # ricci_tensor is now (34x34), but we want to update interior (32x32)
        # Solution: Only update the interior of manifold
        interior_slice = slice(1, -1)
        
        # ricci_tensor shape should be (self.dim-2, self.dim-2) = (32, 32)
        # manifold[1:-1, 1:-1] shape is also (32, 32)
        
        if ricci_tensor.shape[0] == self.dim:
            # ricci_tensor is full size (34x34), take interior
            self.manifold[interior_slice, interior_slice] -= 2 * ricci_tensor[interior_slice, interior_slice] * self.dt
        elif ricci_tensor.shape[0] == self.dim - 2:
            # ricci_tensor is already interior size (32x32) - this is what we expect
            self.manifold[interior_slice, interior_slice] -= 2 * ricci_tensor * self.dt
        else:
            # Unexpected shape, resize as fallback
            target_size = self.dim - 2
            ricci_resized = cv2.resize(ricci_tensor, (target_size, target_size))
            self.manifold[interior_slice, interior_slice] -= 2 * ricci_resized * self.dt
        
        # 4. Normalize
        self.manifold = np.clip(self.manifold, 0.0, 1.0)
        
        # 5. Accumulate curvature for winding number
        current_curvature = np.sum(np.abs(self.manifold - 0.5))
        self.total_curvature += current_curvature * self.dt
        
    def get_output(self, port_name):
        if port_name == 'curvature_scalar':
            return np.sum(np.abs(self.manifold - 0.5))
        
        elif port_name == 'winding_number':
            # Topological invariant (Gauss-Bonnet: ∫R = 2πχ)
            return int(self.total_curvature / (2 * np.pi))
        
        elif port_name == 'manifold_vis':
            img = (self.manifold * 255).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
            h, w, c = img.shape
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
        return None