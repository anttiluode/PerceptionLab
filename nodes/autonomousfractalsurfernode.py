"""
TrueFractalSurferNode (v11 - Asynchronous)
--------------------------------
This node implements the "two-brain" P-KAS model.
It fixes the "massive slowth" by moving the "Soma"
(the deep fractal calculation) onto a separate thread.

The "Dendrite" (the main step() function) runs at full
speed, making steering decisions based on the last
available "thought" from the Soma.

This enables true, infinite, real-time surfing.
"""

import numpy as np
import cv2
import time
import threading # We need this for the "Soma"

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

# --- Numba JIT for high-speed fractal math ---
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: TrueFractalSurferNode requires 'numba' for speed.")

@jit(nopython=True, fastmath=True)
def compute_mandelbrot_core(width, height, center_x, center_y, zoom, max_iter):
    """
    Fast Numba-compiled Mandelbrot set calculator.
    This is the "Soma" - it's allowed to be slow.
    """
    result = np.zeros((height, width), dtype=np.float32)
    scale_x = 3.0 / (width * zoom)
    scale_y = 2.0 / (height * zoom)
    
    for y in range(height):
        for x in range(width):
            c_real = center_x + (x - width / 2) * scale_x
            c_imag = center_y + (y - height / 2) * scale_y
            
            z_real = 0.0
            z_imag = 0.0
            
            n = 0
            while n < int(max_iter):
                if z_real * z_real + z_imag * z_imag > 4.0:
                    break
                new_z_real = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = new_z_real
                n += 1
                
            result[y, x] = n / max_iter
            
    return result

class TrueFractalSurferNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 200, 250) # Crystalline Blue
    
    def __init__(self, resolution=128, base_iterations=50, home_strength=0.05, boredom_threshold=0.1, iteration_scale=10.0):
        super().__init__()
        self.node_title = "True Surfer (Async)"
        
        self.inputs = {
            'zoom_speed': 'signal',
            'steer_damp': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'image': 'image',
            'complexity': 'signal',
            'x_pos': 'signal',
            'y_pos': 'signal',
            'zoom': 'signal',
            'depth': 'signal'
        }
        
        if not NUMBA_AVAILABLE:
            self.node_title = "Surfer (No Numba!)"
        
        self.resolution = int(resolution)
        self.base_iterations = int(base_iterations)
        self.iteration_scale = float(iteration_scale)
        self.home_strength = float(home_strength) 
        self.boredom_threshold = float(boredom_threshold)
        
        # --- Internal Surfer State ---
        self.home_x, self.home_y = -0.7, 0.0
        self.center_x, self.center_y = self.home_x, self.home_y
        self.zoom = 1.0
        self.current_max_iter = self.base_iterations
        
        # --- Internal Logic State ---
        self.complexity = 0.0
        self.nudge_x, self.nudge_y = 0.0, 0.0
        
        # --- Asynchronous "Soma" (The Slow Brain) ---
        self.soma_thread = None
        self.soma_is_working = False
        self.soma_lock = threading.Lock() # To safely pass data
        
        # Data to pass to the thread
        self.job_x = self.center_x
        self.job_y = self.center_y
        self.job_zoom = self.zoom
        self.job_max_iter = self.current_max_iter
        
        # Data to get back from the thread
        self.completed_fractal_data = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Start the "Soma"
        self.is_running = True
        self.start_soma_thread()

    def randomize(self):
        """Reset to the 'home' position"""
        with self.soma_lock:
            self.center_x, self.center_y = self.home_x, self.home_y
            self.zoom = 1.0
            self.current_max_iter = self.base_iterations

    # -----------------------------------------------------------------
    # --- "THIN LOGIC" (The Fast Brain / Dendrite) ---
    # -----------------------------------------------------------------
    def _find_steering_vector(self, fractal_data):
        """
        The "Thin Logic" of the surfer.
        Calculates a steering vector as a blend of two forces.
        """
        if fractal_data.size == 0:
            return 0, 0
            
        self.complexity = np.std(fractal_data)
        
        # "Surf Force" (Steer to complex edge)
        score_map = fractal_data * (1.0 - fractal_data) * 4.0
        max_idx = np.argmax(score_map)
        target_y, target_x = np.unravel_index(max_idx, score_map.shape)
        
        center = self.resolution // 2
        surf_nudge_x = (target_x - center) / center
        surf_nudge_y = (target_y - center) / center

        # "Home Force" (Steer to "shallows")
        home_nudge_x = self.home_x - self.center_x
        home_nudge_y = self.home_y - self.center_y
        
        norm = np.sqrt(home_nudge_x**2 + home_nudge_y**2) + 1e-9
        home_nudge_x = (home_nudge_x / norm) * self.home_strength
        home_nudge_y = (home_nudge_y / norm) * self.home_strength
        
        # Logic Blend Weight
        surf_weight = np.clip(self.complexity / self.boredom_threshold, 0.0, 1.0)
        home_weight = 1.0 - surf_weight
        
        # Combine Forces
        target_nudge_x = (surf_nudge_x * surf_weight) + (home_nudge_x * home_weight)
        target_nudge_y = (surf_nudge_y * surf_weight) + (home_nudge_y * home_weight)
        
        return target_nudge_x, target_nudge_y
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # --- "SOMA" THREAD (The Slow Brain) ---
    # -----------------------------------------------------------------
    def start_soma_thread(self):
        """Starts the background calculation thread."""
        if self.soma_is_working or not self.is_running:
            return
            
        self.soma_is_working = True
        self.soma_thread = threading.Thread(target=self.soma_worker, daemon=True)
        self.soma_thread.start()

    def soma_worker(self):
        """
        This is the "Soma." It runs in the background.
        It just does one job: calculate the fractal.
        """
        # Get the job parameters
        with self.soma_lock:
            x, y, z, i = self.job_x, self.job_y, self.job_zoom, self.job_max_iter
        
        # --- THE SLOW, DEEP CALCULATION ---
        fractal_data = compute_mandelbrot_core(
            self.resolution, self.resolution,
            x, y, z, i
        )
        # ---------------------------------
        
        # Safely pass the result back to the main thread
        with self.soma_lock:
            self.completed_fractal_data = fractal_data
            self.soma_is_working = False

    # -----------------------------------------------------------------
    # --- "DENDRITE" (The Fast Brain, runs every frame) ---
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

        # 2. Check on the "Soma" (the thread)
        if not self.soma_is_working:
            # --- The "Soma" is done! Time to "think" ---
            
            # A. Get the "perception" (the finished fractal)
            with self.soma_lock:
                fractal_data_to_process = self.completed_fractal_data.copy()
            
            # B. Run the "Thin Logic" (Dendrite)
            target_nudge_x, target_nudge_y = self._find_steering_vector(fractal_data_to_process)
            
            # C. Apply Steering (with Damping)
            smoothing_factor = 1.0 - np.clip(steer_damp, 0.0, 0.95)
            self.nudge_x = (self.nudge_x * (1.0 - smoothing_factor)) + (target_nudge_x * smoothing_factor)
            self.nudge_y = (self.nudge_y * (1.0 - smoothing_factor)) + (target_nudge_y * smoothing_factor)

            # D. Act on the "World" (Update next job's parameters)
            self.center_x += self.nudge_x / (self.zoom * 2.0)
            self.center_y += self.nudge_y / (self.zoom * 2.0)
            self.zoom *= (1.0 + (zoom_speed * 0.05))
            
            # E. Calculate "Depth of Vision" for the *next* frame
            self.current_max_iter = int(self.base_iterations + np.sqrt(max(1.0, self.zoom)) * self.iteration_scale)

            # F. Give the "Soma" its *new* job
            with self.soma_lock:
                self.job_x = self.center_x
                self.job_y = self.center_y
                self.job_zoom = self.zoom
                self.job_max_iter = self.current_max_iter
                
            self.start_soma_thread() # Wake up the "Soma"

        # (If the Soma is still working, the Dendrite does nothing
        #  but wait. It continues to output the *last* frame).

    def get_output(self, port_name):
        # We *always* output the last *completed* data
        if port_name == 'image':
            return self.completed_fractal_data
        elif port_name == 'complexity':
            return self.complexity * 5.0 # Boost signal
        elif port_name == 'x_pos':
            return self.center_x
        elif port_name == 'y_pos':
            return self.center_y
        elif port_name == 'zoom':
            return self.zoom
        elif port_name == 'depth':
            return float(self.current_max_iter)
        return None
        
    def get_display_image(self):
        # We *always* display the last *completed* data
        img_u8 = (np.clip(self.completed_fractal_data, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
        
        # Draw steering vector
        h, w, _ = img_color.shape
        center = (w // 2, h // 2)
        
        is_surfing = self.complexity > self.boredom_threshold
        arrow_color = (0, 255, 0) if is_surfing else (0, 0, 255)

        target_x = int(center[0] + self.nudge_x * w)
        target_y = int(center[1] + self.nudge_y * h)
        
        cv2.arrowedLine(img_color, center, (target_x, target_y), arrow_color, 1)
        
        # Display the current iteration depth
        cv2.putText(img_color, f"Depth: {self.current_max_iter}", (5, h - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # --- NEW: Show when the "Soma" (thread) is busy ---
        if self.soma_is_working:
            cv2.putText(img_color, "CALCULATING...", (5, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Base Iterations", "base_iterations", self.base_iterations, None),
            ("Iteration Scale", "iteration_scale", self.iteration_scale, None),
            ("Home Strength", "home_strength", self.home_strength, None),
            ("Complexity Sensitivity", "boredom_threshold", self.boredom_threshold, None)
        ]
        
    def close(self):
        # Clean up the thread
        self.is_running = False
        if self.soma_thread is not None:
            self.soma_thread.join(timeout=0.5)
        super().close()
