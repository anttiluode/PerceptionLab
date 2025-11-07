"""
Fractal Steering Pilot Node - Implements a feedback mechanism that analyzes the
complexity (contrast) of a fractal image and outputs a subtle steering vector
(X and Y nudges) designed to maximize the visible complexity.

Simulates the 'Fractal Surfer' honing in on a maximum information boundary.
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class FractalSteeringPilotNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(150, 100, 200) # Deep Steering Purple
    
    def __init__(self, nudge_factor=0.005, complexity_smoothing=0.9):
        super().__init__()
        self.node_title = "Fractal Steering Pilot"
        
        self.inputs = {
            'image_in': 'image',          # Current fractal image to analyze
            'steering_factor': 'signal'   # External control for nudge strength
        }
        self.outputs = {
            'x_nudge': 'signal',          # Nudge for X position
            'y_nudge': 'signal',          # Nudge for Y position
            'complexity': 'signal',       # Measured complexity (StDev)
        }
        
        self.nudge_factor = float(nudge_factor)
        self.complexity_smoothing = float(complexity_smoothing)
        
        # State tracking
        self.measured_complexity = 0.0
        self.last_nudge_x = 0.0
        self.last_nudge_y = 0.0

    def _measure_complexity(self, img):
        """Measures complexity using standard deviation (contrast)."""
        # Contrast (Standard Deviation) is an excellent, fast proxy for complexity.
        if img.size < 100: 
            return 0.0
        
        return np.std(img)

    def _calculate_steering_vector(self, complexity):
        """
        Calculates the steering vector based on complexity.
        Goal: Drift away from low-complexity areas, and drift randomly but slowly
        within high-complexity areas to explore boundaries.
        """
        
        # 1. Normalize complexity: Assume 0.3 is high complexity for a normalized image.
        target_complexity = 0.3 
        
        # 2. Steering based on perceived need:
        if complexity < target_complexity:
            # Low complexity (flat color): aggressively drift away from center
            # Direction vector: Random normalized direction
            angle = np.random.uniform(0, 2 * np.pi)
            base_nudge = self.nudge_factor * 2.0 # Higher speed to escape
        else:
            # High complexity (boundary): small, local exploration
            # Direction vector: Small random nudge
            angle = np.random.uniform(0, 2 * np.pi)
            base_nudge = self.nudge_factor * 0.5 # Slower speed to stick to boundary

        # 3. Apply steering factor and randomness
        nudge_x = base_nudge * np.cos(angle)
        nudge_y = base_nudge * np.sin(angle)
        
        return nudge_x, nudge_y

    def step(self):
        # 1. Get Inputs
        img_in = self.get_blended_input('image_in', 'mean')
        steering_factor_in = self.get_blended_input('steering_factor', 'sum') or 1.0
        
        if img_in is None or img_in.size == 0:
            return
        
        # Ensure image is grayscale (0-1)
        if img_in.ndim == 3:
             img_in = cv2.cvtColor(img_in.astype(np.float32), cv2.COLOR_BGR2GRAY)

        # 2. Measure Complexity
        new_complexity = self._measure_complexity(img_in)
        
        # Smooth the complexity metric to prevent chaotic jumps
        self.measured_complexity = (self.measured_complexity * self.complexity_smoothing +
                                    new_complexity * (1.0 - self.complexity_smoothing))

        # 3. Calculate Steering
        nudge_x, nudge_y = self._calculate_steering_vector(self.measured_complexity)
        
        # Apply external scaling factor
        self.last_nudge_x = nudge_x * steering_factor_in
        self.last_nudge_y = nudge_y * steering_factor_in


    def get_output(self, port_name):
        if port_name == 'x_nudge':
            return self.last_nudge_x
        elif port_name == 'y_nudge':
            return self.last_nudge_y
        elif port_name == 'complexity':
            # Normalize complexity to the 0-1 signal range
            return np.clip(self.measured_complexity * 4.0, 0.0, 1.0)
        return None
        
# In nodes/fractalsteeringpilotnode.py (Update get_display_image method, around line 124)

    def get_display_image(self):
        w, h = 96, 96
        # --- FIX: Change img initialization to 3 channels (RGB) ---
        img = np.zeros((h, w, 3), dtype=np.uint8) 
        # --- END FIX ---
        
        # 1. Visualize Learning Progress (Color represents Coupling Value)
        norm_coupling = self.measured_complexity * 255.0 * 2.0 
        comp_u8 = np.clip(norm_coupling, 0, 255).astype(np.uint8)
        
        # Green channel indicates high complexity, Red channel indicates low/escape
        color = (int(255 - comp_u8), int(comp_u8), 0) # BGR tuple with standard ints
        
        # This line was crashing:
        color = (int(255 - comp_u8), int(comp_u8), 0)
        
        # Draw arrow showing current nudge direction
        nudge_scale = 30
        end_x = int(w/2 + self.last_nudge_x * nudge_scale)
        end_y = int(h/2 + self.last_nudge_y * nudge_scale)
        
        # Draw the arrow in white
        cv2.arrowedLine(img, (w//2, h//2), (end_x, end_y), (255, 255, 255), 1)
        
        # Draw text in white
        cv2.putText(img, f"C: {self.measured_complexity:.2f}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        img = np.ascontiguousarray(img)
        # We must return a QImage with 3 channels (Format_BGR888)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Base Nudge Factor", "nudge_factor", self.nudge_factor, None),
            ("Complexity Smoothing", "complexity_smoothing", self.complexity_smoothing, None),
        ]