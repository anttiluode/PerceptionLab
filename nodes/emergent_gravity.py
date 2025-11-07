"""
Emergent Gravity Node - Simulates a 2D potential field from constraint density
Implements the $\rho_C$ -> $T_{\mu\nu}^{(C)}$ -> $G_{\mu\nu}$ link from the IHT-AI paper
in a simplified, real-time 2D model.

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class EmergentGravityNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(60, 60, 100)  # Dark, "heavy" blue
    
    def __init__(self, g_coupling=1.0, blur_strength=21):
        super().__init__()
        self.node_title = "Emergent Gravity"
        
        self.inputs = {
            'constraint_density': 'image', # $\rho_C$ from IHTPhaseFieldNode
            'g_coupling': 'signal'         # Gravitational constant G
        }
        
        self.outputs = {
            'gravity_potential': 'image',   # The $\Phi$ field (potential well)
            'curvature_field': 'image',   # Approx. $\nabla^2\Phi$ (spacetime bending)
            'total_mass': 'signal'        # Total integrated constraint $\int \rho_C$
        }
        
        self.g_coupling = float(g_coupling)
        self.blur_strength = int(blur_strength)
        
        # Internal state
        self.potential_field = None
        self.curvature_field = None
        self.total_mass = 0.0

    def _normalize_for_vis(self, field):
        """Safely normalize a 2D field to [0, 1] for image output."""
        if field is None:
            return None # Return None, not a default array
        
        min_v, max_v = field.min(), field.max()
        range_v = max_v - min_v
        
        if range_v < 1e-9:
            return np.zeros_like(field, dtype=np.float32)
            
        return (field - min_v) / range_v
        
    def step(self):
        # Update parameters from inputs
        g_signal = self.get_blended_input('g_coupling', 'sum')
        if g_signal is not None:
            # Map signal [-1, 1] to a positive range [0, 2]
            self.g_coupling = (g_signal + 1.0)
            
        rho_c = self.get_blended_input('constraint_density', 'mean')
        
        if rho_c is None:
            if self.potential_field is not None:
                self.potential_field *= 0.95
            if self.curvature_field is not None: # Check before multiplying
                self.curvature_field *= 0.95
            self.total_mass *= 0.95
            return
            
        # Ensure blur strength is odd
        if self.blur_strength % 2 == 0:
            self.blur_strength += 1
            
        # 1. Calculate Total "Mass" (Total Constraint)
        self.total_mass = np.sum(rho_c)
        
        # 2. Calculate Gravitational Potential $\Phi$
        # A Gaussian blur is a fast, real-time approximation of the
        # gravitational potential well created by the mass density $\rho_C$.
        self.potential_field = cv2.GaussianBlur(
            rho_c, 
            (self.blur_strength, self.blur_strength), 
            0
        )
        
        # 3. Calculate Curvature (Approx. $\nabla^2\Phi$)
        # The Laplacian of the potential field shows where the potential
        # is "bending" the most, i.e., the curvature.
        self.curvature_field = cv2.Laplacian(self.potential_field, cv2.CV_32F, ksize=3)
        
        # Apply coupling constant
        self.potential_field *= self.g_coupling
        self.curvature_field *= self.g_coupling
        
    def get_output(self, port_name):
        if port_name == 'gravity_potential':
            return self._normalize_for_vis(self.potential_field)
            
        elif port_name == 'curvature_field':
            # Curvature can be positive or negative, so we take abs()
            # Check for None before np.abs()
            if self.curvature_field is None:
                return None
            return self._normalize_for_vis(np.abs(self.curvature_field))
            
        elif port_name == 'total_mass':
            return self.total_mass
            
        return None
        
    def get_display_image(self):
        # --- START FIX ---
        # We visualize the curvature field, as it's more dynamic
        
        # Check if self.curvature_field is None before calling np.abs
        if self.curvature_field is None:
            vis_field = None
        else:
            vis_field = np.abs(self.curvature_field)
            
        vis_field_normalized = self._normalize_for_vis(vis_field)
        
        if vis_field_normalized is None:
             vis_field_normalized = np.zeros((64, 64), dtype=np.float32)
        # --- END FIX ---

        img_u8 = (vis_field_normalized * 255).astype(np.uint8)
        
        # Apply a colormap to make it look "gravitational"
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_BONE)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("G Coupling (Strength)", "g_coupling", self.g_coupling, None),
            ("Blur (Range)", "blur_strength", self.blur_strength, None),
        ]