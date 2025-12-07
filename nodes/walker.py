import numpy as np
import cv2
from collections import deque

# --- STRICT COMPATIBILITY BOILERPLATE ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0
        def step(self): pass
        def get_output(self, name): return None
        def get_display_image(self): return None

class LatentWalkerNode(BaseNode):
    """
    Latent Walker v2 (The Fractal Surfer)
    -------------------------------------
    NOW WITH:
    - Trajectory Trail (Visualizing the history of thought)
    - Enhanced Moire Contrast (Sharper interference)
    - Orbital Gravity (Keeps the surfer from drifting to infinity)
    """
    NODE_CATEGORY = "AI_Experiment"
    NODE_TITLE = "Latent Surfer v2"
    NODE_COLOR = QtGui.QColor(0, 160, 170) # Slightly brighter Teal

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'drift_x_in': 'signal',   
            'drift_y_in': 'signal',   
            'turbulence_in': 'signal' 
        }
        
        self.outputs = {
            'latent_view': 'image',    
            'pos_x': 'signal',         
            'pos_y': 'signal',         
            'velocity_mag': 'signal'   
        }
        
        # PHYSICS STATE
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # HISTORY TRAIL (The Wake)
        self.trail = deque(maxlen=50) # Remember last 50 steps
        
        # UNIVERSE SETTINGS
        self.view_size = 256
        self.universe_seed = np.random.rand(4) * 100
        
        self._outputs = {}

    def get_interference_pattern(self, px, py, size):
        """
        Generates the 'Bulk' reality.
        v2 Update: Sharpened math for better Moire effects.
        """
        x = np.linspace(px - 3, px + 3, size) # Increased field of view slightly
        y = np.linspace(py - 3, py + 3, size)
        xv, yv = np.meshgrid(x, y)
        
        s = self.universe_seed
        
        # 1. Base Carrier Wave (The "Time" dimension)
        z = np.sin(xv * 4.0 + s[0]) + np.cos(yv * 4.0 + s[1])
        
        # 2. The Moire Generator (High frequency interference)
        # This creates the "Black Balls" (Destructive Interference)
        z += 0.8 * np.sin(xv * 15.0 + yv * 10.0) * np.cos(xv * 5.0 - yv * 5.0)
        
        # 3. Deep Structure (Low frequency gravity wells)
        dist = np.sqrt(xv**2 + yv**2)
        z += 0.4 * np.cos(dist * 3.0 + s[2])
        
        # Normalize and Contrast Stretch (make black blacker)
        z_norm = (z - z.min()) / ((z.max() - z.min()) + 1e-6)
        z_norm = np.power(z_norm, 1.5) # Gamma correction for contrast
        return z_norm

    def step(self):
        # 1. INPUTS
        dx_val = self.get_blended_input('drift_x_in', 'sum')
        dy_val = self.get_blended_input('drift_y_in', 'sum')
        turb_val = self.get_blended_input('turbulence_in', 'sum')

        dx_sig = float(dx_val) if dx_val is not None else 0.0
        dy_sig = float(dy_val) if dy_val is not None else 0.0
        turb = float(turb_val) if turb_val is not None else 0.0
        
        # 2. PHYSICS
        # v2: Added "Orbital Gravity" - a weak pull to center (0,0)
        # This creates a "Solor System" effect where thoughts orbit the self.
        gravity_x = -self.pos_x * 0.01 
        gravity_y = -self.pos_y * 0.01
        
        # Apply Forces
        self.vel_x += (dx_sig * 1.5) + gravity_x
        self.vel_y += (dy_sig * 1.5) + gravity_y
        
        # Turbulence
        if abs(turb) > 0:
            jitter = abs(turb) * 0.2
            self.vel_x += np.random.normal(0, jitter)
            self.vel_y += np.random.normal(0, jitter)
        
        # Friction
        self.vel_x *= 0.92
        self.vel_y *= 0.92
        
        # Move
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y
        
        # 3. TRAIL UPDATE
        # We store relative coordinates (0.5 + offset) for drawing on the 256x256 image
        # Center of image is always the *current* position, so trail moves relative to us.
        self.trail.append((self.pos_x, self.pos_y))
        
        # 4. RENDER
        # Get the mathematical landscape
        view_gray = self.get_interference_pattern(self.pos_x, self.pos_y, self.view_size)
        
        # Convert to RGB to draw the red trail
        view_rgb = (view_gray * 255).astype(np.uint8)
        view_rgb = cv2.cvtColor(view_rgb, cv2.COLOR_GRAY2RGB)
        
        # Draw the "Wake" (The path of the surfer)
        # We map the trail points relative to the current center (128, 128)
        center_w = self.view_size // 2
        scale = self.view_size / 6.0 # Matches the linspace(px-3, px+3)
        
        for i in range(1, len(self.trail)):
            # Previous point relative to current pos
            prev_x = (self.trail[i-1][0] - self.pos_x) * scale + center_w
            prev_y = (self.trail[i-1][1] - self.pos_y) * scale + center_w
            
            # Current point relative to current pos
            curr_x = (self.trail[i][0] - self.pos_x) * scale + center_w
            curr_y = (self.trail[i][1] - self.pos_y) * scale + center_w
            
            # Fade out the tail
            alpha = int(255 * (i / len(self.trail)))
            color = (0, 0, 255) # Red trail
            
            cv2.line(view_rgb, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), color, 2)
            
        # Draw the "Self" (The Observer) at the center
        cv2.circle(view_rgb, (center_w, center_w), 3, (0, 255, 255), -1)

        # 5. OUTPUTS
        self._outputs['latent_view'] = view_rgb
        self._outputs['pos_x'] = self.pos_x
        self._outputs['pos_y'] = self.pos_y
        self._outputs['velocity_mag'] = np.sqrt(self.vel_x**2 + self.vel_y**2)

    def get_output(self, name):
        return self._outputs.get(name)

    def get_display_image(self):
        return self._outputs.get('latent_view')