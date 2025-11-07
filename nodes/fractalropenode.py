"""
Fractal Rope Node - Implements Tim Palmer's geometric model of quantum reality.
Simulates a fractal helix bundle and strand selection during a measurement event.
Ported from palmers_rope.py.
Requires: pip install numpy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui


# --- Core Geometric Classes (from palmers_rope.py) ---

class FractalStrand:
    """A single strand in the fractal rope"""
    
    def __init__(self, base_trajectory, fractal_level=0, amplitude=1.0):
        self.base_trajectory = base_trajectory.astype(np.float32)
        self.fractal_level = fractal_level
        self.amplitude = amplitude
        self.sub_strands = []
        self.selected = False
        self.coherence = 1.0 # Stability metric

    # --- FIX: ADD MISSING METHOD ---
    def add_fractal_detail(self, n_sub_strands=3, detail_level=0.3):
        """Add fractal sub-structure to this strand (Helixes within helixes)"""
        if self.fractal_level < 2:  # Limit recursion depth for performance
            for i in range(n_sub_strands):
                # Create sub-trajectory wound around base trajectory
                sub_trajectory = self.create_sub_helix(i, n_sub_strands, detail_level)
                sub_strand = FractalStrand(sub_trajectory, 
                                         self.fractal_level + 1, 
                                         self.amplitude * detail_level)
                self.sub_strands.append(sub_strand)
                # Recursively add detail (Palmers' concept: Uncertainty = Geometric bundling)
                sub_strand.add_fractal_detail(n_sub_strands=2, detail_level=0.2) 
    # --- END FIX ---
    
    def create_sub_helix(self, index, total_strands, detail_level):
        """Create a helical sub-trajectory wound around the base"""
        t = np.linspace(0, 1, len(self.base_trajectory))
        
        phase = 2 * np.pi * index / total_strands
        frequency = 8 + 4 * self.fractal_level
        
        helix_x = detail_level * np.cos(frequency * 2 * np.pi * t + phase)
        helix_y = detail_level * np.sin(frequency * 2 * np.pi * t + phase)
        helix_z = detail_level * 0.5 * np.sin(frequency * 4 * np.pi * t + phase)
        
        sub_trajectory = self.base_trajectory.copy()
        sub_trajectory[:, 0] += helix_x
        sub_trajectory[:, 1] += helix_y
        sub_trajectory[:, 2] += helix_z
        
        return sub_trajectory.astype(np.float32)

class FractalRope:
    """The complete fractal rope structure"""
    
    def __init__(self, n_main_strands=6, length=40):
        self.n_main_strands = n_main_strands
        self.length = length
        self.main_strands = []
        self.selected_strand = None
        self.time = 0.0
        
        self.create_main_rope()
        
        # This loop now calls the fixed method
        for strand in self.main_strands:
            strand.add_fractal_detail()
    
    def create_main_rope(self):
        """Create the main helical rope structure"""
        t = np.linspace(0, 4*np.pi, self.length)
        
        centerline = np.array([
            t,
            2 * np.sin(t),
            2 * np.cos(t)
        ]).T
        
        for i in range(self.n_main_strands):
            phase = 2 * np.pi * i / self.n_main_strands
            radius = 1.5
            helix_freq = 3
            
            helix_x = radius * np.cos(helix_freq * t + phase)
            helix_y = radius * np.sin(helix_freq * t + phase)
            helix_z = 0.5 * np.sin(helix_freq * 2 * t + phase)
            
            main_trajectory = centerline.copy()
            main_trajectory[:, 0] += helix_x
            main_trajectory[:, 1] += helix_y
            main_trajectory[:, 2] += helix_z
            
            strand = FractalStrand(main_trajectory, fractal_level=0)
            self.main_strands.append(strand)
    
    def apply_measurement(self, selection_radius=2.0):
        """Apply measurement - select coherent strand cluster"""
        
        mp = np.array([
            np.random.uniform(5, 7), 
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1)
        ])
        
        selected_strands = []
        self.selected_strand = None

        for strand in self.main_strands:
            distances = np.linalg.norm(strand.base_trajectory - mp, axis=1)
            min_distance = np.min(distances)
            
            strand.selected = False
            
            if min_distance < selection_radius:
                strand.selected = True
                strand.coherence = 1.0 / (1.0 + min_distance)
                selected_strands.append(strand)
            else:
                strand.coherence = 0.05 # Decohered state
        
        if selected_strands:
            self.selected_strand = max(selected_strands, key=lambda s: s.coherence)
            
        return len(selected_strands) 

    def evolve(self, dt=0.1):
        """Evolve the rope structure"""
        self.time += dt
        
        for strand in self.main_strands:
            noise_amplitude = 0.01
            noise = np.random.normal(0, noise_amplitude, strand.base_trajectory.shape).astype(np.float32)
            strand.base_trajectory += noise
            
            if self.selected_strand is not strand:
                strand.coherence = max(0.01, strand.coherence * 0.95)

# --- The Main Node Class ---

class FractalRopeNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 100, 100) # Geometric Gray
    
    def __init__(self, n_strands=6, resolution=96, selection_radius=2.0):
        super().__init__()
        self.node_title = "Fractal Rope (Palmer)"
        
        self.inputs = {
            'measurement_trigger': 'signal'
        }
        self.outputs = {
            'measured_image': 'image',
            'coherence_out': 'signal',
            'uncertainty': 'signal' # Number of strands in the bundle
        }
        
        self.n_strands = int(n_strands)
        self.resolution = int(resolution)
        self.selection_radius = float(selection_radius)
        
        self.rope = FractalRope(n_main_strands=self.n_strands, length=40)
        self.last_trigger_val = 0.0
        self.last_uncertainty = float(self.n_strands)

    def step(self):
        # 1. Get inputs
        trigger_val = self.get_blended_input('measurement_trigger', 'sum') or 0.0
        
        # 2. Check for measurement trigger (rising edge)
        if trigger_val > 0.5 and self.last_trigger_val <= 0.5:
            num_selected = self.rope.apply_measurement(self.selection_radius)
            self.last_uncertainty = np.clip(num_selected / self.n_strands, 0.0, 1.0)
        else:
            self.rope.evolve()

        self.last_trigger_val = trigger_val

    def get_output(self, port_name):
        if port_name == 'coherence_out':
            if self.rope.selected_strand:
                return self.rope.selected_strand.coherence
            return 0.0
            
        elif port_name == 'uncertainty':
            return self.last_uncertainty
            
        elif port_name == 'measured_image':
            img = self._draw_cross_section()
            return img / 255.0 
            
        return None
        
    def _draw_cross_section(self):
        """Draws the cross-section visualization for the node's output port."""
        w, h = self.resolution, self.resolution
        img = np.zeros((h, w, 3), dtype=np.uint8)
        center = w // 2
        
        cross_section_x = 5.0 
        
        # Draw background uncertainty circle (faded)
        uncertainty_radius = int(self.last_uncertainty * center * 0.8)
        cv2.circle(img, (center, center), uncertainty_radius, (50, 50, 50), -1)

        for strand in self.rope.main_strands:
            x_coords = strand.base_trajectory[:, 0]
            closest_idx = np.argmin(np.abs(x_coords - cross_section_x))
            
            y = strand.base_trajectory[closest_idx, 1]
            z = strand.base_trajectory[closest_idx, 2]
            
            # Map YZ coordinates (range approx. [-4, 4]) to screen (0, w)
            y_screen = int(np.clip((y / 8.0 + 0.5) * w, 0, w-1))
            z_screen = int(np.clip((z / 8.0 + 0.5) * h, 0, h-1))
            
            # Draw strand (color based on coherence/selection)
            if strand.selected:
                color_val = int(strand.coherence * 255)
                color = (0, color_val, 255) # Cyan/Red for selected
                radius = 3
            else:
                color_val = int(strand.coherence * 255)
                color = (color_val, color_val, color_val) # Gray for decohered
                radius = 1
                
            cv2.circle(img, (y_screen, z_screen), radius, color, -1)

        if self.rope.selected_strand:
            y = self.rope.selected_strand.base_trajectory[closest_idx, 1]
            z = self.rope.selected_strand.base_trajectory[closest_idx, 2]
            y_screen = int(np.clip((y / 8.0 + 0.5) * w, 0, w-1))
            z_screen = int(np.clip((z / 8.0 + 0.5) * h, 0, h-1))
            cv2.circle(img, (y_screen, z_screen), 5, (255, 255, 255), 1) 

        return img

    def get_display_image(self):
        img_rgb = self._draw_cross_section()
        img_rgb = np.ascontiguousarray(img_rgb)
        
        h, w = img_rgb.shape[:2]
        return QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Num Main Strands", "n_strands", self.n_strands, None),
            ("Selection Radius", "selection_radius", self.selection_radius, None),
        ]