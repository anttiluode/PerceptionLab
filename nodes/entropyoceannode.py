"""
Entropy Ocean Node (Dynamic Decoherence)
========================================
Generates a shifting, time-varying decoherence landscape.

PURPOSE:
To force the 'Diamond' attractor to surf. 
If the environment is static, the attractor crystallizes and 'dies' (stops processing).
If the environment moves, the attractor must constantly update its W-matrix to survive.

OUTPUTS:
- decoherence_map: The changing landscape γ(k,t).
- drift_vector: The average direction of the current (for visualization).
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

class EntropyOceanNode(BaseNode):
    NODE_CATEGORY = "IHT_Core"
    NODE_TITLE = "Entropy Ocean"
    NODE_COLOR = QtGui.QColor(0, 80, 160)  # Deep Ocean Blue
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'drift_speed': 'signal',     # How fast time moves (0.0 - 2.0)
            'turbulence': 'signal',      # Wave height (0.0 - 1.0)
            'complexity': 'signal',      # Number of wave layers
            'center_bias': 'signal'      # Strength of the central "Bowl"
        }
        
        self.outputs = {
            'decoherence_map': 'image',  # The γ(k, t) field
            'protection_map': 'image',   # 1 - γ (The safe zones)
            'storm_level': 'signal'      # Current aggregate chaos
        }
        
        self.size = 128
        self.time = 0.0
        
        # Internal coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        # Normalized coordinates (-1 to 1)
        self.nx = (x - center) / center
        self.ny = (y - center) / center
        self.radius = np.sqrt(self.nx**2 + self.ny**2)
        
        # State
        self.gamma_field = np.zeros((self.size, self.size), dtype=np.float32)
        self.protection = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Default Params
        self.speed = 0.1
        self.turb = 0.3
        self.bias = 0.5

    def step(self):
        # 1. Get Inputs
        s = self.get_blended_input('drift_speed', 'sum')
        t = self.get_blended_input('turbulence', 'sum')
        b = self.get_blended_input('center_bias', 'sum')
        
        if s is not None: self.speed = np.clip(float(s), 0.0, 5.0)
        if t is not None: self.turb = np.clip(float(t), 0.0, 2.0)
        if b is not None: self.bias = np.clip(float(b), 0.0, 1.0)
        
        # Increment internal time
        self.time += self.speed * 0.1
        
        # 2. Construct the Field
        
        # A. The Bowl (Static gravity)
        # Keeps things generally centered so we don't drift off the grid
        bowl = np.clip(self.radius * self.bias * 2.0, 0, 1)
        
        # B. The Waves (Dynamic Noise)
        # We use superposition of sine waves to simulate fluid surface
        # Wave 1: Slow, large
        w1 = np.sin(self.nx * 3.0 + self.time * 0.5) * np.cos(self.ny * 2.5 + self.time * 0.2)
        
        # Wave 2: Medium, diagonal
        w2 = np.sin((self.nx + self.ny) * 5.0 - self.time * 1.2)
        
        # Wave 3: Fast ripples
        w3 = np.cos(self.nx * 10.0 + self.time) * np.sin(self.ny * 10.0 + self.time)
        
        # Combine
        waves = (w1 * 0.5 + w2 * 0.3 + w3 * 0.2) * self.turb
        
        # 3. Final Gamma Calculation
        # γ = Bowl + Waves
        # Clip to ensure valid physics (0 = safe, 1 = instant decoherence)
        raw_gamma = bowl + waves
        
        # Offset to keep mean sensible
        raw_gamma += 0.1 
        
        self.gamma_field = np.clip(raw_gamma, 0.0, 0.98).astype(np.float32)
        self.protection = 1.0 - self.gamma_field
        
    def get_output(self, name):
        if name == 'decoherence_map':
            return (self.gamma_field * 255).astype(np.uint8)
        elif name == 'protection_map':
            return (self.protection * 255).astype(np.uint8)
        elif name == 'storm_level':
            return float(self.turb + np.sin(self.time)*0.1)
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # Visualize the Protection Map (The Safe Zones)
        # Low Gamma = High Protection = Brighter
        
        # Map to nice ocean colors
        # Deep blue = Dangerous (High Gamma)
        # Cyan/Green = Safe (Low Gamma)
        
        vis = (self.protection * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(vis, cv2.COLORMAP_OCEAN)
        
        # Add Vector Field Overlay (Visual flair to show drift)
        center = w // 2
        # Calculate a fake drift vector based on time
        dx = int(np.cos(self.time * 0.5) * 20)
        dy = int(np.sin(self.time * 0.3) * 20)
        
        # Draw Arrow from center
        cv2.arrowedLine(color_map, (center, center), (center + dx, center + dy), (255, 255, 255), 2)
        
        # Text Info
        cv2.putText(color_map, "ENTROPY OCEAN", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        status = "CALM" if self.turb < 0.3 else "CHOPPY" if self.turb < 0.8 else "STORM"
        cv2.putText(color_map, f"{status} (T={self.time:.1f})", (5, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 255), 1)
        
        return QtGui.QImage(color_map.data, w, h, w*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Drift Speed", "speed", self.speed, "float"),
            ("Turbulence", "turb", self.turb, "float"),
            ("Bowl Bias", "bias", self.bias, "float"),
        ]