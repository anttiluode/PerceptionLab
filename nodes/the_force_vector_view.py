"""
Phi Instanton Scout Node
========================
"The Force Vector View"

This node calculates the forces acting on the brain's trajectory to distinguish
between PASSIVE DWELLING (falling into an attractor) and ACTIVE SCOUTING 
(climbing out of one).

Mechanism:
1.  Builds a dynamic "Potential Energy Surface" based on dwell history.
2.  Calculates the Gradient (Gravity) at the current position.
3.  Compares Actual Velocity vector vs. Gravity vector.
    - Aligned = Falling (Relaxation)
    - Opposed = Climbing (Tunneling/Computation)

Outputs a visualization of these forces in real-time.
"""

import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
import math

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, method): return None

class PhiInstantonScoutNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(200, 50, 50) # Red/Active

    def __init__(self):
        super().__init__()
        self.node_title = "Instanton Scout"
        
        self.inputs = {
            'delta_field': 'complex_spectrum',
            'theta_field': 'complex_spectrum',
            'alpha_field': 'complex_spectrum',
            'beta_field':  'complex_spectrum',
            'gamma_field': 'complex_spectrum',
            'reset': 'signal'
        }
        
        self.outputs = {
            'render': 'image',
            'scout_energy': 'signal', # Positive when climbing, negative when falling
            'is_tunneling': 'signal'  # 1.0 if actively escaping an attractor
        }

        # Config
        self.grid_size = 128
        self.decay_rate = 0.995
        self.dig_rate = 0.5
        
        # State
        self.potential_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.ipca = IncrementalPCA(n_components=2, batch_size=10)
        self.data_buffer = []
        self.auto_scale = 5.0
        
        # Kinematics
        self.pos = np.array([self.grid_size/2, self.grid_size/2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_raw_pos = None
        
        self.display_image = None

    def get_gradient_at(self, x, y):
        """Calculates the 'Gravity' vector (slope of the potential map) at x,y"""
        ix, iy = int(x), int(y)
        ix = np.clip(ix, 1, self.grid_size-2)
        iy = np.clip(iy, 1, self.grid_size-2)
        
        # Sobel-ish gradient
        dz_dx = (self.potential_map[iy, ix+1] - self.potential_map[iy, ix-1]) * 0.5
        dz_dy = (self.potential_map[iy+1, ix] - self.potential_map[iy-1, ix]) * 0.5
        
        # Gravity points DOWNHILL (negative gradient)
        # But since we dig HOLES (negative values), deep is low.
        # The gradient points towards higher values (ridges). 
        # We want the vector that points into the hole.
        # If hole is negative: Gradient points OUT of hole.
        # So Gravity = -Gradient.
        
        # Wait, self.potential_map stores DEPTH as POSITIVE in the dig step usually?
        # Let's check dig logic below. Ah, usually we add to ridges or subtract for holes.
        # Let's standardize: Map = Height. Valleys are low.
        
        return np.array([dz_dx, dz_dy], dtype=np.float32)

    def step(self):
        # 1. Reset
        val = self.get_blended_input('reset', 'max')
        if val is not None and val > 0.5:
            self.potential_map[:] = 0
            self.data_buffer = []
            self.ipca = IncrementalPCA(n_components=2, batch_size=10)
            return

        # 2. Collect Vector
        bands = ['delta_field', 'theta_field', 'alpha_field', 'beta_field', 'gamma_field']
        state_vector = []
        has_signal = False
        for b in bands:
            data = self.get_blended_input(b, 'first')
            if data is not None:
                mag = float(np.mean(np.abs(data))) if isinstance(data, np.ndarray) else float(np.abs(data))
                state_vector.extend([mag, 0.0]) # Pad for PCA
                if mag > 1e-5: has_signal = True
            else:
                state_vector.extend([0.0, 0.0])

        if not has_signal: return

        # 3. PCA & Positioning
        target_x, target_y = 0.0, 0.0
        try:
            vec = np.array([state_vector])
            if not hasattr(self.ipca, 'components_'):
                self.data_buffer.append(state_vector)
                if len(self.data_buffer) >= 10:
                    self.ipca.partial_fit(np.array(self.data_buffer))
                    self.data_buffer = []
            else:
                self.ipca.partial_fit(vec)
                coords = self.ipca.transform(vec)
                
                # Auto-scale
                mag = np.max(np.abs(coords))
                if mag > 1e-9:
                    self.auto_scale = 0.99 * self.auto_scale + 0.01 * (20.0 / mag) # Keep within ~+/-20 range
                
                target_x = coords[0,0] * self.auto_scale
                target_y = coords[0,1] * self.auto_scale if coords.shape[1] > 1 else 0.0
        except: pass

        # Map to Grid Coordinates
        center = self.grid_size / 2
        scale = self.grid_size / 60.0
        
        curr_x = center + target_x * scale
        curr_y = center + target_y * scale
        
        # Calculate Velocity (Frame-to-Frame movement)
        new_pos = np.array([curr_x, curr_y], dtype=np.float32)
        
        # Smooth position slightly for physics stability
        self.pos = self.pos * 0.7 + new_pos * 0.3
        
        if self.prev_raw_pos is not None:
            self.vel = self.pos - self.prev_raw_pos
        self.prev_raw_pos = self.pos.copy()

        # 4. Dig the Terrain (The Memory)
        # Valleys are negative values (Depth)
        ix, iy = int(self.pos[0]), int(self.pos[1])
        if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
            # Dig a hole
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        # We subtract to make a hole (Attractor)
                        force = np.exp(-(dx**2 + dy**2)/4.0) * self.dig_rate
                        self.potential_map[ny, nx] -= force 

        # Decay (Erosion) - tends back to 0
        self.potential_map *= self.decay_rate

        # 5. PHYSICS ANALYSIS: Gravity vs Velocity
        # Map is negative in holes. Gradient points UPHILL (out of hole).
        # So "Gravity" (force pulling in) is -Gradient.
        gradient = self.get_gradient_at(self.pos[0], self.pos[1])
        gravity = -gradient 
        
        # Work: Dot product of Force (Gravity) and Displacement (Velocity)
        # Work < 0: Moving against gravity (Climbing/Scouting)
        # Work > 0: Moving with gravity (Falling/Dwelling)
        
        # Normalize for angle calc
        vel_mag = np.linalg.norm(self.vel)
        grav_mag = np.linalg.norm(gravity)
        
        scout_energy = 0.0
        tunneling = 0.0
        
        if vel_mag > 0.1 and grav_mag > 0.1:
            # Dot product
            dot = np.dot(gravity, self.vel)
            # If dot is negative, we are moving AGAINST the pull -> SCOUTING
            scout_energy = -dot 
            
            if scout_energy > 2.0: # Threshold for active tunneling
                tunneling = 1.0

        self.outputs['scout_energy'] = float(scout_energy)
        self.outputs['is_tunneling'] = float(tunneling)

        # 6. RENDER
        H, W = 300, 300
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw Heatmap (Potential)
        # Map is negative (holes). Normalize for display.
        # Valleys (Deep negative) -> Bright Blue/Purple
        # Plains (Zero) -> Black
        
        # Normalize: find deepest hole
        min_val = np.min(self.potential_map)
        if min_val < -0.1:
            norm_map = np.clip(self.potential_map / min_val, 0, 1) # 1.0 at deepest
            heatmap = (norm_map * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_OCEAN)
            canvas = cv2.resize(heatmap_color, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            canvas[:] = (20, 20, 20)

        # Draw Particle
        px = int(self.pos[0] / self.grid_size * W)
        py = int(self.pos[1] / self.grid_size * H)
        
        # Draw Vectors
        # Velocity (Yellow)
        vx_end = int(px + self.vel[0] * 10)
        vy_end = int(py + self.vel[1] * 10)
        cv2.arrowedLine(canvas, (px, py), (vx_end, vy_end), (0, 255, 255), 1, tipLength=0.3)
        
        # Gravity (Blue) - where it WANTS to go
        gx_end = int(px + gravity[0] * 20)
        gy_end = int(py + gravity[1] * 20)
        cv2.arrowedLine(canvas, (px, py), (gx_end, gy_end), (255, 100, 0), 1, tipLength=0.3)
        
        # Draw Scout Indicator
        color = (0, 255, 0) # Green (Passive)
        status = "DWELLING"
        
        if tunneling > 0.5:
            color = (0, 0, 255) # Red (Active Tunneling)
            status = "SCOUTING / TUNNELING"
            cv2.circle(canvas, (px, py), 15, color, 2)
        elif vel_mag < 0.2 and grav_mag > 0.5:
            color = (255, 0, 255) # Purple (Trapped/Stuck)
            status = "TRAPPED"
            
        cv2.circle(canvas, (px, py), 5, color, -1)
        
        # HUD
        cv2.putText(canvas, f"MODE: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(canvas, f"SCOUT FORCE: {scout_energy:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # Vectors Legend
        cv2.putText(canvas, "VELOCITY (ACTUAL)", (10, H-25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cv2.putText(canvas, "GRAVITY (PULL)", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1)

        out_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.display_image = out_rgb

    def get_display_image(self):
        return self.display_image

    def get_config_options(self):
        return [
            ("Grid Res", "grid_size", self.grid_size, "int"),
            ("Memory Fade", "decay_rate", self.decay_rate, "float"),
            ("Dig Depth", "dig_rate", self.dig_rate, "float"),
        ]

    def set_config_options(self, options):
        if 'grid_size' in options: self.grid_size = int(options['grid_size'])
        if 'decay_rate' in options: self.decay_rate = float(options['decay_rate'])
        if 'dig_rate' in options: self.dig_rate = float(options['dig_rate'])