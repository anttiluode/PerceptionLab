"""
Autonomous FractalSurferNode (v3)
--------------------------------
A self-contained "consciousness" model that generates its own
fractal "world" and uses an internal "logic" to surf the
edge of chaos, seeking maximum complexity.

This version implements a "two-state logic" inspired by the
"thin sheet of logic" and "fractal surfer" concepts.

LOGIC:
- STATE 1 (SURFING): If complexity is high, steer gently along the edge.
- STATE 2 (LOST/BORED): If complexity is low (stuck in a solid
  color void), stop zooming and "kick" back towards the "home"
  fractal region to find the "shallow water" (the edge) again.
"""

import numpy as np
import cv2
import time

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

# --- Numba JIT for high-speed fractal math ---
# [cite_start]This function is borrowed from your fractal_explorer.py [cite: 6464-6468]
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: AutonomousFractalSurferNode requires 'numba' for speed.")

@jit(nopython=True, fastmath=True)
def compute_mandelbrot_core(width, height, center_x, center_y, zoom, max_iter):
    """
    Fast Numba-compiled Mandelbrot set calculator.
    """
    result = np.zeros((height, width), dtype=np.float32)
    
    # Calculate scale
    scale_x = 3.0 / (width * zoom)
    scale_y = 2.0 / (height * zoom)
    
    for y in range(height):
        for x in range(width):
            # Map pixel to complex plane
            c_real = center_x + (x - width / 2) * scale_x
            c_imag = center_y + (y - height / 2) * scale_y
            
            z_real = 0.0
            z_imag = 0.0
            
            n = 0
            while n < max_iter:
                if z_real * z_real + z_imag * z_imag > 4.0:
                    break
                
                # z = z*z + c
                new_z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = new_z_real
                
                n += 1
                
            result[y, x] = n / max_iter # Normalized iteration count
            
    return result

class AutonomousFractalSurferNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 200, 250) # Crystalline Blue
    
    def __init__(self, resolution=128, max_iterations=50):
        super().__init__()
        self.node_title = "Fractal Surfer"
        
        self.inputs = {
            'zoom_speed': 'signal',   # How fast to zoom (0-1)
            'steer_damp': 'signal',   # How much to resist steering (0-1)
            'reset': 'signal'
        }
        self.outputs = {
            'image': 'image',         # The "Surfer's View"
            'complexity': 'signal',     # The "Logic" signal (what it feels)
            'x_pos': 'signal',
            'y_pos': 'signal',
            'zoom': 'signal'
        }
        
        if not NUMBA_AVAILABLE:
            self.node_title = "Surfer (No Numba!)"
        
        self.resolution = int(resolution)
        self.max_iterations = int(max_iterations)
        
        # --- Internal Surfer State ---
        self.home_x = -0.7 # The "safe harbor" of the main fractal
        self.home_y = 0.0
        self.center_x = self.home_x
        self.center_y = self.home_y
        self.zoom = 1.0
        
        # --- Internal Logic State ---
        self.fractal_data = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.complexity = 0.0
        self.is_panicked = False # Our "logic" state flag
        
        # Store nudge for smooth steering
        self.nudge_x = 0.0
        self.nudge_y = 0.0
        
        # --- Tunable Parameters ---
        self.boredom_threshold = 0.05 # How simple the view must be to "panic"
        self.kick_strength = 0.2      # How hard to "kick" when bored (as a fraction of zoom)

    def randomize(self):
        """Reset to the 'home' position"""
        self.center_x = self.home_x
        self.center_y = self.home_y
        self.zoom = 1.0
        
    # -----------------------------------------------------------------
    # --- THIS IS THE FIXED "THIN LOGIC" (v3) ---
    # -----------------------------------------------------------------
    def _find_steering_vector(self):
        """
        The "Thin Logic" of the surfer.
        Decides what to do based on the current view.
        """
        if self.fractal_data.size == 0:
            return 0, 0
            
        # 1. Measure Overall "Boredom" (The Meta-Logic)
        # We use Standard Deviation as a robust measure of complexity.
        self.complexity = np.std(self.fractal_data)
        
        # --- THE "THIN LOGIC" RULE ---
        if self.complexity < self.boredom_threshold:
            # STATE 1: "PANIC" / SEEK "SHALLOW WATER"
            # The view is too simple.
            self.is_panicked = True
            
            # Calculate a vector pointing from our current "lost"
            # position back to the "home" fractal at (-0.7, 0.0).
            # This is a targeted, intelligent escape, not a random kick.
            target_nudge_x = self.home_x - self.center_x
            target_nudge_y = self.home_y - self.center_y
            
            # Normalize the vector
            norm = np.sqrt(target_nudge_x**2 + target_nudge_y**2) + 1e-9
            target_nudge_x /= norm
            target_nudge_y /= norm
            
            return target_nudge_x, target_nudge_y

        else:
            # STATE 2: "SURF" (The Original Logic)
            # The view is complex. Find the most "interesting" edge.
            self.is_panicked = False
            
            # Score is high (max 1.0) only when fractal_data is 0.5
            score_map = self.fractal_data * (1.0 - self.fractal_data) * 4.0
            
            max_idx = np.argmax(score_map)
            target_y, target_x = np.unravel_index(max_idx, score_map.shape)
            
            # Calculate Nudge Vector (steer towards most complex point)
            center = self.resolution // 2
            nudge_x = (target_x - center) / center # -1 to 1
            nudge_y = (target_y - center) / center # -1 to 1
            
            return nudge_x, nudge_y
    # -----------------------------------------------------------------

    def step(self):
        if not NUMBA_AVAILABLE:
            return
            
        # 1. Get Inputs
        zoom_speed = self.get_blended_input('zoom_speed', 'sum') or 0.01
        steer_damp = self.get_blended_input('steer_damp', 'sum') or 0.1
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        if reset > 0.5:
            self.randomize()

        # 2. Render the "World" from the current position
        self.fractal_data = compute_mandelbrot_core(
            self.resolution, self.resolution,
            self.center_x, self.center_y,
            self.zoom, self.max_iterations
        )

        # 3. "Thin Logic" decides where to go next
        target_nudge_x, target_nudge_y = self._find_steering_vector()
        
        # 4. Apply Steering (with Damping)
        smoothing_factor = 1.0 - np.clip(steer_damp, 0.0, 0.95)
        self.nudge_x = (self.nudge_x * (1.0 - smoothing_factor)) + (target_nudge_x * smoothing_factor)
        self.nudge_y = (self.nudge_y * (1.0 - smoothing_factor)) + (target_nudge_y * smoothing_factor)

        # 5. Act on the "World" (Update State)
        
        # --- MODIFIED ACTION BASED ON LOGIC STATE ---
        if self.is_panicked:
            # "LOST/BORED" STATE: Stop zooming, kick hard
            self.zoom *= 0.98 # Back up!
            
            # The "kick" is now relative to the *view*, not the zoom
            # We move 20% of the current view's width/height
            kick_distance = self.kick_strength / self.zoom 
            self.center_x += self.nudge_x * kick_distance
            self.center_y += self.nudge_y * kick_distance
            
        else:
            # "SURFING" STATE: Gentle steering and keep zooming
            
            # The nudge is scaled by zoom, so we move precisely
            self.center_x += self.nudge_x / (self.zoom * 2.0)
            self.center_y += self.nudge_y / (self.zoom * 2.0)
            
            # Apply the endless zoom
            self.zoom *= (1.0 + (zoom_speed * 0.05))

    def get_output(self, port_name):
        if port_name == 'image':
            return self.fractal_data
        elif port_name == 'complexity':
            return self.complexity * 5.0 # Boost signal
        elif port_name == 'x_pos':
            return self.center_x
        elif port_name == 'y_pos':
            return self.center_y
        elif port_name == 'zoom':
            return self.zoom
        return None
        
    def get_display_image(self):
        # Apply a colormap for a nice visual
        img_u8 = (np.clip(self.fractal_data, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
        
        # Draw steering vector
        h, w, _ = img_color.shape
        center = (w // 2, h // 2)
        
        # Scale nudge vector for display
        if self.is_panicked:
            # Red, strong arrow pointing "home"
            display_x = self.nudge_x * (w / 2)
            display_y = self.nudge_y * (h / 2)
            arrow_color = (0, 0, 255) # Red for "panic"
        else:
            # Green, gentle arrow pointing to edge
            display_x = self.nudge_x * (w / 2)
            display_y = self.nudge_y * (h / 2)
            arrow_color = (0, 255, 0) # Green for "surfing"

        target_x = int(center[0] + display_x)
        target_y = int(center[1] + display_y)
        cv2.arrowedLine(img_color, center, (target_x, target_y), arrow_color, 1)
        
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Max Iterations", "max_iterations", self.max_iterations, None),
            ("Boredom Threshold", "boredom_threshold", self.boredom_threshold, None),
            ("Panic Kick Strength", "kick_strength", self.kick_strength, None)
        ]