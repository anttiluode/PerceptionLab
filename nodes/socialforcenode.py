"""
Social Force Node (The Peace Treaty) - FIXED
============================================
Implements repulsive forces between agents to solve Zero-Sum conflicts.
Fixed dtype casting error in image normalization.

PHYSICS:
- Modifies the Decoherence Landscape γ(k) for each agent.
- Effective_Gamma_A = Base_Gamma + (Address_B * Repulsion)
- Effective_Gamma_B = Base_Gamma + (Address_A * Repulsion)
"""

import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class SocialForceNode(BaseNode):
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Social Force (Repulsion)"
    NODE_COLOR = QtGui.QColor(200, 80, 80)  # Aggressive Red
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'address_a': 'image',          # Where Agent A is
            'address_b': 'image',          # Where Agent B is
            'base_landscape': 'image',     # The natural environment (Decoherence Field)
            'repulsion_strength': 'signal' # How much they hate each other
        }
        
        self.outputs = {
            'landscape_a': 'image',        # Modified map for Agent A
            'landscape_b': 'image',        # Modified map for Agent B
            'stress_field': 'image'        # Visual of the tension
        }
        
        self.size = 128
        self.repulsion = 0.5
        
        # State
        self.map_a = np.zeros((self.size, self.size), dtype=np.float32)
        self.map_b = np.zeros((self.size, self.size), dtype=np.float32)
        self.stress = np.zeros((self.size, self.size), dtype=np.float32)

    def step(self):
        # 1. Get Inputs
        addr_a = self.get_input_img('address_a')
        addr_b = self.get_input_img('address_b')
        base = self.get_input_img('base_landscape')
        rep_sig = self.get_blended_input('repulsion_strength', 'sum')
        
        if rep_sig is not None:
            self.repulsion = np.clip(float(rep_sig), 0.0, 5.0)
            
        # Default base if missing
        if base is None:
            # Generate radial gradient (standard physics)
            y, x = np.ogrid[:self.size, :self.size]
            center = self.size // 2
            r = np.sqrt((x - center)**2 + (y - center)**2) / center
            base = np.clip(r, 0, 1).astype(np.float32)

        # Default addresses if missing
        if addr_a is None: addr_a = np.zeros_like(base)
        if addr_b is None: addr_b = np.zeros_like(base)

        # 2. Compute Exclusion Forces
        # The presence of B increases decoherence for A, and vice versa.
        
        # Force = Address * Strength
        force_on_a = addr_b * self.repulsion
        force_on_b = addr_a * self.repulsion
        
        # 3. Apply to Landscape
        # New Gamma = Base Gamma + Force
        # We clip at 0.99 because 1.0 means instant death (singularity)
        self.map_a = np.clip(base + force_on_a, 0.0, 0.99)
        self.map_b = np.clip(base + force_on_b, 0.0, 0.99)
        
        # 4. Compute Stress Field (Where both are trying to exist)
        self.stress = addr_a * addr_b

    def get_input_img(self, name):
        img = self.get_blended_input(name, 'first')
        if img is not None:
            # FIX: Explicitly cast to float32 BEFORE any operations
            # This prevents the uint8 division error
            img = img.astype(np.float32)
            
            if img.ndim == 3: img = np.mean(img, axis=2)
            if img.shape != (self.size, self.size):
                img = cv2.resize(img, (self.size, self.size))
            
            # Normalize
            mx = np.max(img)
            if mx > 1e-9: img /= mx
            return img
        return None

    def get_output(self, name):
        if name == 'landscape_a': return (self.map_a * 255).astype(np.uint8)
        if name == 'landscape_b': return (self.map_b * 255).astype(np.uint8)
        if name == 'stress_field': return (self.stress * 255).astype(np.uint8)
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # Visualizing the "Treaty"
        # Red areas = High Repulsion (Forbidden)
        # Blue areas = Base Landscape
        
        # Composite A's view (Left) and B's view (Right)
        view_a = cv2.applyColorMap((self.map_a * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        view_b = cv2.applyColorMap((self.map_b * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        full = np.hstack((view_a, view_b))
        
        cv2.putText(full, f"Landscape A (Rep={self.repulsion})", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(full, "Landscape B", (w + 5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Use the safe helper from main if available
        if hasattr(__main__, 'numpy_to_qimage'):
            return __main__.numpy_to_qimage(full)
        
        # Fallback
        return QtGui.QImage(full.data, w*2, h, w*2*3, QtGui.QImage.Format.Format_BGR888)
        
    def get_config_options(self):
        return [("Repulsion Strength", "repulsion", self.repulsion, 'float')]