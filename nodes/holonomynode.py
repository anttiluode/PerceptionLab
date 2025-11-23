"""
Holonomy Node - Measures Geometric Phase (Berry Phase)
------------------------------------------------------
Connects VAE Latent (Manifold Position) and Phase Space (Momentum).
Calculates the 'Curvature' of the thought process.

If the system loops back to the start but is 'changed' (Holonomy != 0),
it indicates non-integrable memory or topological learning.
"""

import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class HolonomyNode(BaseNode):
    NODE_CATEGORY = "Deep Math"
    NODE_TITLE = "Holonomy (Geometric Phase)"
    NODE_COLOR = QtGui.QColor(100, 0, 200) # Indigo

    def __init__(self, history_len=100):
        super().__init__()
        
        # --- THE FIX: Explicitly defining ports ---
        self.inputs = {
            'vae_latent': 'spectrum',    # The "Position" on the manifold
            'phase_vector': 'spectrum',  # The "Momentum" or Phase Velocity
            'reset': 'signal'
        }
        
        self.outputs = {
            'holonomy_scalar': 'signal',   # The accumulated geometric phase
            'curvature_vis': 'image',      # Visualization of the fiber bundle
            'berry_curvature': 'signal'    # Instantaneous curvature
        }
        
        self.history_len = int(history_len)
        
        # State Memory
        self.path_history = deque(maxlen=self.history_len)
        self.accumulated_phase = 0.0
        self.last_vector = None
        
        # Visualization buffer
        self.display_img = np.zeros((256, 256, 3), dtype=np.uint8)

    def _project_to_2d(self, vec):
        """
        Projects high-dimensional vectors to 2D complex plane 
        to measure angle changes.
        """
        if len(vec) < 2:
            return 1.0 + 0.0j # Default unit vector
        # Take first two principal components (simplified)
        return vec[0] + 1j * vec[1]

    def step(self):
        # 1. Gather Inputs
        z = self.get_blended_input('vae_latent', 'first') # The Manifold point
        p = self.get_blended_input('phase_vector', 'first') # The Tangent vector
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.accumulated_phase = 0.0
            self.path_history.clear()
            self.last_vector = None

        if z is None or p is None:
            return

        # 2. The Math: Parallel Transport
        # We treat the VAE latent (z) and Phase (p) as defining a Fiber Bundle.
        # We want to see if transporting 'z' along path 'p' induces a rotation.
        
        # Project high-dim vectors to complex plane to measure angle
        z_complex = self._project_to_2d(z)
        p_complex = self._project_to_2d(p)
        
        # Current state vector in the total space
        # Combining them tells us the total state of the system
        current_vector = z_complex * np.conj(p_complex) 
        
        curvature = 0.0
        
        if self.last_vector is not None:
            # Calculate the angular difference (The Connection Form)
            # angle_diff = arg(v_t * conj(v_t-1))
            relative_rotation = current_vector * np.conj(self.last_vector)
            angle_change = np.angle(relative_rotation)
            
            # The Berry Curvature is the rate of this angular change
            curvature = angle_change
            
            # Holonomy is the path integral of the curvature
            self.accumulated_phase += angle_change

        self.last_vector = current_vector
        self.path_history.append(self.accumulated_phase)
        
        # 3. Visualization (The Holonomy Loop)
        self.display_img.fill(0)
        h, w, _ = self.display_img.shape
        center_x, center_y = w // 2, h // 2
        
        # Draw the "Fiber" (The rotating phase)
        radius = 100
        # The needle points to the current accumulated phase
        dx = int(np.cos(self.accumulated_phase) * radius)
        dy = int(np.sin(self.accumulated_phase) * radius)
        
        # Color shifts based on Curvature intensity (Stress)
        c_val = int(np.clip(abs(curvature) * 1000, 0, 255))
        color = (c_val, 255 - c_val, 255)
        
        cv2.line(self.display_img, (center_x, center_y), (center_x + dx, center_y + dy), color, 2)
        cv2.circle(self.display_img, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Draw History Trail (The Winding Number)
        pts = []
        for i, phase_val in enumerate(self.path_history):
            # Map time to radius (spiraling out)
            r = (i / self.history_len) * radius
            px = int(center_x + np.cos(phase_val) * r)
            py = int(center_y + np.sin(phase_val) * r)
            pts.append([px, py])
            
        if len(pts) > 1:
            cv2.polylines(self.display_img, [np.array(pts)], False, (100, 100, 100), 1)

        # Text Info
        cv2.putText(self.display_img, f"Holonomy: {self.accumulated_phase/np.pi:.2f}pi", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def get_output(self, port_name):
        if port_name == 'holonomy_scalar':
            return float(self.accumulated_phase)
        elif port_name == 'curvature_vis':
            return self.display_img
        elif port_name == 'berry_curvature':
            # Return the derivative of the phase
            if len(self.path_history) >= 2:
                return float(self.path_history[-1] - self.path_history[-2])
            return 0.0
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)