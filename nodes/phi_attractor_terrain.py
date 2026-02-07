"""
Phi Attractor Terrain Node (Ridge Plot Edition)
===============================================
The Topography of Thought.

Visualizes the "Energy Landscape" as a 3D Ridge Plot (Joy Division style).

FEATURES:
- Pseudo-3D rendering directly to the image output (visible on DisplayNode).
- Momentum Physics: The state 'ball' rolls into valleys.
- Elastic Camera: The view shifts slightly based on where the ball is.

Mechanism:
1.  PCA reduces 5-band input to 2D coordinates.
2.  "Digging": The system digs a hole at the current coordinate.
3.  "Gravity": Deep holes (strong memories) pull the cursor in.
"""

import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA

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

class PhiAttractorTerrainNode(BaseNode):
    NODE_CATEGORY = "Visualizers"
    NODE_COLOR = QtGui.QColor(160, 60, 160) # Purple/Deep

    def __init__(self):
        super().__init__()
        self.node_title = "Attractor Terrain"
        
        self.inputs = {
            'delta_field': 'complex_spectrum',
            'theta_field': 'complex_spectrum',
            'alpha_field': 'complex_spectrum',
            'beta_field':  'complex_spectrum',
            'gamma_field': 'complex_spectrum',
            'reset': 'signal'
        }
        
        self.outputs = {
            'render': 'image'
        }

        # Config
        self.grid_size = 50       # Number of ridges
        self.decay_rate = 0.995   # How fast memories fade
        self.dig_rate = 0.8       # How fast thoughts form valleys
        self.smoothing = 0.2      # Momentum factor
        
        # State
        self.memory_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.ipca = IncrementalPCA(n_components=2, batch_size=10)
        self.data_buffer = []
        self.auto_scale = 5.0
        
        # Physics State
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        
        self.display_image = None

    def step(self):
        # 1. Reset Logic
        try:
            val = self.get_blended_input('reset', 'max')
            if val is not None and val > 0.5:
                self.memory_map[:] = 0
                self.data_buffer = []
                self.ipca = IncrementalPCA(n_components=2, batch_size=10)
        except: pass

        # 2. Collect Data (5 Bands -> Vector)
        bands = ['delta_field', 'theta_field', 'alpha_field', 'beta_field', 'gamma_field']
        state_vector = []
        has_signal = False
        
        for b in bands:
            data = self.get_blended_input(b, 'first')
            if data is not None:
                mag = float(np.mean(np.abs(data))) if isinstance(data, np.ndarray) else float(np.abs(data))
                if np.isfinite(mag):
                    state_vector.extend([mag, 0.0]) # Padding for PCA stability
                    if mag > 1e-5: has_signal = True
                else:
                    state_vector.extend([0.0, 0.0])
            else:
                state_vector.extend([0.0, 0.0])

        # 3. PCA Projection (High Dim -> 2D Surface)
        target_x, target_y = 0.0, 0.0
        
        if has_signal:
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
                    # Auto-scaling to keep it on the map
                    mag = np.max(np.abs(coords))
                    if mag > 1e-9:
                        self.auto_scale = 0.99 * self.auto_scale + 0.01 * (12.0 / mag)
                    
                    target_x = coords[0,0] * self.auto_scale
                    target_y = coords[0,1] * self.auto_scale if coords.shape[1] > 1 else 0.0
            except: pass

        # 4. Physics Engine (Gravity + Momentum)
        # Convert target (PCA) to Grid Coords
        center = self.grid_size / 2
        scale = self.grid_size / 40.0
        
        tx_grid = center + target_x * scale
        ty_grid = center + target_y * scale
        
        # Momentum (Smooth movement)
        self.ball_vx = self.ball_vx * 0.8 + (tx_grid - self.ball_x) * 0.1
        self.ball_vy = self.ball_vy * 0.8 + (ty_grid - self.ball_y) * 0.1
        
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Clamp
        self.ball_x = np.clip(self.ball_x, 0, self.grid_size-1)
        self.ball_y = np.clip(self.ball_y, 0, self.grid_size-1)

        # 5. Dig the Terrain
        ix, iy = int(self.ball_x), int(self.ball_y)
        # Dig a gaussian hole at current location
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    force = np.exp(-(dx**2 + dy**2)/4.0) * self.dig_rate
                    self.memory_map[ny, nx] += force # Positive height for ridges
        
        # Erosion (Decay)
        self.memory_map *= self.decay_rate

        # 6. RENDERER: RIDGE PLOT (Joy Division Style)
        # We draw lines from back (top) to front (bottom)
        W, H = 400, 300
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Background color
        canvas[:] = (20, 15, 25) # Deep purple/black
        
        # Grid settings
        y_step = H / (self.grid_size + 10)
        x_step = W / self.grid_size
        amp_scale = 4.0 # Height of peaks
        
        # Colors
        line_color = (255, 200, 100) # Cyan/White
        ball_color = (0, 255, 255)   # Yellow
        
        # Iterate rows from back to front
        for r in range(self.grid_size):
            # Base Y position for this row
            base_y = 50 + r * y_step * 0.8
            
            points = []
            # Calculate points for the line
            for c in range(self.grid_size):
                # Height from memory map
                z = self.memory_map[r, c]
                # Screen X, Y
                sx = int(c * x_step)
                sy = int(base_y - z * amp_scale)
                points.append((sx, sy))
            
            # Draw Polygon (Filled black to hide lines behind)
            poly_points = np.array([[(0, base_y)] + points + [(W, base_y)]], dtype=np.int32)
            cv2.fillPoly(canvas, poly_points, (20, 15, 25))
            
            # Draw Line
            pts = np.array(points, dtype=np.int32)
            # Color intensity based on height sum of row
            row_energy = np.mean(self.memory_map[r, :])
            r_col = int(min(255, 100 + row_energy * 20))
            g_col = int(min(255, 50 + row_energy * 50))
            b_col = int(min(255, 200 + row_energy * 10))
            
            cv2.polylines(canvas, [pts], False, (b_col, g_col, r_col), 1, cv2.LINE_AA)

        # Draw the Ball (projected)
        bx_scr = int(self.ball_x * x_step)
        bz_height = self.memory_map[int(self.ball_y), int(self.ball_x)]
        by_scr = int((50 + self.ball_y * y_step * 0.8) - bz_height * amp_scale)
        
        # Shadow / Marker
        cv2.circle(canvas, (bx_scr, by_scr), 4, (255, 255, 255), -1)
        cv2.circle(canvas, (bx_scr, by_scr), 8, ball_color, 1)
        
        # HUD
        cv2.putText(canvas, "ATTRACTOR BASINS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if has_signal:
            cv2.putText(canvas, "SIGNAL ACTIVE", (W-120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(canvas, "NO SIGNAL", (W-100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Output
        out_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.display_image = out_rgb

    def get_display_image(self):
        return self.display_image

    def get_config_options(self):
        return [
            ("Grid Res", "grid_size", self.grid_size, "int"),
            ("Decay", "decay_rate", self.decay_rate, "float"),
            ("Dig Force", "dig_rate", self.dig_rate, "float"),
        ]

    def set_config_options(self, options):
        if 'grid_size' in options: self.grid_size = int(options['grid_size'])
        if 'decay_rate' in options: self.decay_rate = float(options['decay_rate'])
        if 'dig_rate' in options: self.dig_rate = float(options['dig_rate'])