"""
Phase Explorer Node - An automated probe to find the "Goldilocks Zone"
by mapping the phase space of a toy universe's fundamental constants.

Ported from goldilocks_explorer.py
Requires: pip install numpy scipy
"""

import numpy as np
from PyQt6 import QtGui, QtCore
import cv2
import sys
import os
import threading
import time

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: PhaseExplorerNode requires 'scipy'.")


# --- Core Physics Engine (from explorer.py) ---
class UniverseSimulator:
    def __init__(self, grid_size=64, params=None):
        self.grid_size = grid_size
        self.params = params
        self.phi = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.phi_old = np.zeros_like(self.phi)
        self.lambda_coupling = self.params['lambda_coupling']
        self.vev_sq = self.params['vev']**2
        self.spin_force = self.params['spin_force']
        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1]], dtype=np.float32)
        self.singularity_threshold = 50.0
        self.heat_death_threshold = 0.1
        self._initialize_field()

    def _initialize_field(self):
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        cx1, cx2 = self.grid_size // 2 - 8, self.grid_size // 2 + 8
        cy = self.grid_size // 2
        radius = self.grid_size / 8.0
        self.phi = np.full_like(self.phi, self.params['vev'])
        self.phi += 2.0 * np.exp(-((x - cx1)**2 + (y - cy)**2) / (2 * radius**2))
        self.phi += -2.0 * np.exp(-((x - cx2)**2 + (y - cy)**2) / (2 * radius**2))
        self.phi_old = np.copy(self.phi)

    def _apply_spin_forces(self):
        if self.spin_force == 0: return 0
        grad_y, grad_x = np.gradient(self.phi)
        return (grad_y - grad_x) * self.spin_force

    def run(self, max_steps=400):
        for step in range(max_steps):
            potential_accel = self.lambda_coupling * self.phi * (self.phi**2 - self.vev_sq)
            lap_phi = convolve2d(self.phi, self.laplacian_kernel, 'same', 'wrap')
            spin_accel = self._apply_spin_forces()
            total_accel = -potential_accel + lap_phi + spin_accel
            
            velocity = self.phi - self.phi_old
            dt = 0.05
            phi_new = self.phi + (1.0 - 0.01*dt)*velocity + (dt**2)*total_accel
            self.phi_old, self.phi = self.phi, phi_new
            
            if np.max(np.abs(self.phi)) > self.singularity_threshold:
                return "SINGULARITY"
        
        if np.max(np.abs(self.phi - self.params['vev'])) < self.heat_death_threshold:
             return "HEAT DEATH"
        return "STABLE & COMPLEX"

# --- The Main Node Class ---

class PhaseExplorerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(220, 180, 40) # Golden
    
    def __init__(self, num_trials=500, grid_size=64):
        super().__init__()
        self.node_title = "Phase Explorer"
        
        self.inputs = {'trigger': 'signal'}
        self.outputs = {
            'phase_diagram': 'image',
            'status': 'signal'
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Explorer (No SciPy!)"
            return
            
        self.num_trials = int(num_trials)
        self.grid_size = int(grid_size)
        self.results = []
        
        self.param_space = {
            'lambda_coupling': (0.1, 2.0),
            'spin_force': (0.0, 0.8),
            'vev': (1.0, 1.0)
        }
        
        self.last_trigger = 0.0
        self.is_running = False
        self.progress = 0.0 # 0.0 to 1.0
        self.output_image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.thread = None

    def _exploration_thread(self):
        """Runs the heavy simulation in a separate thread."""
        self.is_running = True
        self.progress = 0.0
        self.results = []
        
        for i in range(self.num_trials):
            if not self.is_running: # Allow early exit
                break
                
            # 1. Randomly select laws
            trial_params = {
                'lambda_coupling': np.random.uniform(*self.param_space['lambda_coupling']),
                'spin_force': np.random.uniform(*self.param_space['spin_force']),
                'vev': np.random.uniform(*self.param_space['vev']),
            }
            
            # 2. Create and run universe
            simulator = UniverseSimulator(grid_size=self.grid_size, params=trial_params)
            outcome = simulator.run()

            # 3. Log the laws and outcome
            self.results.append({
                'params': trial_params,
                'outcome': outcome
            })
            
            # 4. Update progress
            self.progress = (i + 1) / self.num_trials
            
        # 5. When done, generate the plot
        if self.is_running: # Check if finished, not cancelled
            self.output_image = self._plot_phase_diagram()
            self.is_running = False

    def _plot_phase_diagram(self):
        """Draws the phase diagram onto a numpy array (OpenCV)."""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if not self.results:
            return img
            
        lambda_vals = [r['params']['lambda_coupling'] for r in self.results]
        spin_vals = [r['params']['spin_force'] for r in self.results]
        
        color_map = {
            "STABLE & COMPLEX": (0, 215, 255), # Gold/Yellow (BGR)
            "SINGULARITY": (0, 0, 255),      # Red
            "HEAT DEATH": (10, 10, 10)       # Dark Gray
        }
        
        # Normalize coordinates to image size
        spin_min, spin_max = self.param_space['spin_force']
        lambda_min, lambda_max = self.param_space['lambda_coupling']
        
        for i, outcome in enumerate([r['outcome'] for r in self.results]):
            x = int( (spin_vals[i] - spin_min) / (spin_max - spin_min) * (w - 1) )
            y = int( (1.0 - (lambda_vals[i] - lambda_min) / (lambda_max - lambda_min)) * (h - 1) )
            
            color = color_map.get(outcome, (255, 255, 255))
            cv2.circle(img, (x, y), 2, color, -1)
            
        # Draw Goldilocks Zone box (approximate)
        x1 = int( (0.2 - spin_min) / (spin_max - spin_min) * (w - 1) )
        x2 = int( (0.5 - spin_min) / (spin_max - spin_min) * (w - 1) )
        y1 = int( (1.0 - (0.7 - lambda_min) / (lambda_max - lambda_min)) * (h - 1) )
        y2 = int( (1.0 - (0.2 - lambda_min) / (lambda_max - lambda_min)) * (h - 1) )
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), 1)
        
        return img

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        trigger_val = self.get_blended_input('trigger', 'sum') or 0.0
        
        # On rising edge, start the simulation thread
        if trigger_val > 0.5 and self.last_trigger <= 0.5:
            if not self.is_running:
                print("Starting Phase Exploration...")
                self.thread = threading.Thread(target=self._exploration_thread, daemon=True)
                self.thread.start()
            
        self.last_trigger = trigger_val

    def get_output(self, port_name):
        if port_name == 'phase_diagram':
            return self.output_image.astype(np.float32) / 255.0
        elif port_name == 'status':
            return self.progress
        return None
        
    def get_display_image(self):
        if self.is_running:
            # Show a progress bar
            w, h = 96, 96
            img = np.zeros((h, w, 3), dtype=np.uint8)
            progress_w = int(self.progress * w)
            cv2.rectangle(img, (0, h//2 - 10), (progress_w, h//2 + 10), (0, 255, 0), -1)
            cv2.putText(img, f"{(self.progress * 100):.0f}%", (w//2 - 15, h//2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
        else:
            # Show the final plot
            img_rgb = np.ascontiguousarray(self.output_image)
            h, w = img_rgb.shape[:2]
            if w == 0 or h == 0:
                 img_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
                 h, w = 96, 96
                 cv2.putText(img_rgb, "Ready", (20, 45), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Num Trials", "num_trials", self.num_trials, None),
            ("Grid Size (NxN)", "grid_size", self.grid_size, None),
        ]
        
    def close(self):
        self.is_running = False # Signal thread to stop
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        super().close()