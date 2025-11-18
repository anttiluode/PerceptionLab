"""
Cortical 3D Growth Node (Turbulence-Enhanced)
----------------------------------------------
Enhanced version that accepts turbulence signal to amplify growth.

Tests hypothesis: High constraint violation (turbulence) → faster growth → more folds

When turbulence signal is connected:
- Growth rate amplifies in proportion to turbulence
- High-turbulence regions develop faster
- Folds form preferentially where turbulence was highest

This models learning: brain grows structure to reduce constraint violation.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class TurbulenceEnhancedGrowthNode(BaseNode):
    """
    Grows 3D cortical structure driven by eigenmode activation,
    with optional turbulence-based growth amplification.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(180, 80, 180)  # Purple for morphogenesis
    
    def __init__(self):
        super().__init__()
        self.node_title = "Turbulence Growth"
        
        self.inputs = {
            'lobe_activation': 'image',       # From eigenmode node
            'growth_rate': 'signal',          # Modulate growth speed
            'turbulence_signal': 'signal',    # NEW: Turbulence amplification
            'turbulence_field': 'image',      # NEW: Spatial turbulence map
            'reset': 'signal'                 # Reset simulation
        }
        
        self.outputs = {
            'thickness_map': 'image',
            'fold_density': 'signal',
            'surface_area': 'signal',
            'fractal_estimate': 'signal',
            'structure_3d': 'image',
            'turbulence_response': 'signal',  # NEW: How much turbulence affected growth
        }
        
        # Simulation parameters
        self.resolution = 128
        self.dt = 0.01
        self.base_growth = 0.001
        self.fold_threshold = 2.5
        self.compression_strength = 0.3
        self.diffusion = 0.1
        
        # NEW: Turbulence parameters
        self.turbulence_amplification = 2.0  # How much turbulence boosts growth
        self.turbulence_mode = 'signal'       # 'signal', 'field', or 'both'
        
        # State variables
        self.thickness = np.ones((self.resolution, self.resolution), dtype=np.float32)
        self.height_field = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.pressure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.time_step = 0
        
        # NEW: Track turbulence response
        self.turbulence_response_value = 0.0
        
        # For fractal measurement
        self.area_history = []
        
        # Initialize measurement values
        self.fold_density_value = 0.0
        self.surface_area_value = 0.0
        self.fractal_dim_value = 2.0
        
    def step(self):
        # Get inputs
        activation = self.get_blended_input('lobe_activation', 'replace')
        growth_mod = self.get_blended_input('growth_rate', 'sum')
        reset_signal = self.get_blended_input('reset', 'sum')
        
        # NEW: Get turbulence inputs
        turbulence_signal = self.get_blended_input('turbulence_signal', 'sum')
        turbulence_field = self.get_blended_input('turbulence_field', 'replace')
        
        # Reset if triggered
        if reset_signal is not None and reset_signal > 0.5:
            self.reset_simulation()
            return
        
        if activation is None:
            return
            
        # Convert activation to grayscale if needed
        if len(activation.shape) == 3:
            activation_gray = cv2.cvtColor(activation, cv2.COLOR_BGR2GRAY)
        else:
            activation_gray = activation
            
        # Resize to match resolution
        activation_resized = cv2.resize(activation_gray, (self.resolution, self.resolution))
        activation_normalized = activation_resized.astype(np.float32) / 255.0
        
        # Modulate growth rate
        if growth_mod is not None:
            total_growth_rate = self.base_growth * (1.0 + growth_mod)
        else:
            total_growth_rate = self.base_growth
        
        # NEW: Calculate turbulence amplification
        turbulence_amp = self.calculate_turbulence_amplification(
            turbulence_signal, 
            turbulence_field
        )
        
        # Apply turbulence boost
        amplified_growth_rate = total_growth_rate * turbulence_amp
        
        # Track how much turbulence affected growth
        self.turbulence_response_value = float(np.mean(turbulence_amp) - 1.0)
        
        # === GROWTH PHASE ===
        growth_field = activation_normalized * amplified_growth_rate * self.dt
        self.thickness += growth_field
        
        # === CONSTRAINT PHASE ===
        excess = np.clip(self.thickness - self.fold_threshold, 0, None)
        self.pressure = excess ** 2
        
        # === FOLDING PHASE ===
        grad_y, grad_x = np.gradient(self.thickness)
        laplacian = cv2.Laplacian(self.thickness, cv2.CV_32F)
        
        fold_force_x = -grad_x * self.pressure * self.compression_strength
        fold_force_y = -grad_y * self.pressure * self.compression_strength
        fold_force_z = -laplacian * self.pressure * self.compression_strength * 0.5
        
        self.height_field += fold_force_z * self.dt
        
        fold_magnitude = np.sqrt(fold_force_x**2 + fold_force_y**2 + fold_force_z**2)
        thickness_redistribution = fold_magnitude * 0.1
        self.thickness -= thickness_redistribution
        self.thickness = np.clip(self.thickness, 0.1, 10.0)
        
        # === DIFFUSION PHASE ===
        self.thickness = gaussian_filter(self.thickness, sigma=self.diffusion)
        self.height_field = gaussian_filter(self.height_field, sigma=self.diffusion)
        
        # === MEASUREMENT ===
        self.measure_properties()
        
        self.time_step += 1
        
    def calculate_turbulence_amplification(self, turb_signal, turb_field):
        """
        Calculate spatially-varying growth amplification from turbulence.
        
        Returns: amplification map (1.0 = no boost, 2.0 = double growth, etc.)
        """
        amp_map = np.ones((self.resolution, self.resolution), dtype=np.float32)
        
        # Mode 1: Global turbulence signal
        if self.turbulence_mode in ['signal', 'both']:
            if turb_signal is not None:
                # Scale turbulence to amplification
                # turb_signal typically in [0, 0.1] range
                global_amp = 1.0 + (turb_signal * self.turbulence_amplification * 10.0)
                amp_map *= global_amp
        
        # Mode 2: Spatial turbulence field
        if self.turbulence_mode in ['field', 'both']:
            if turb_field is not None:
                # Convert field to grayscale if needed
                if len(turb_field.shape) == 3:
                    turb_gray = cv2.cvtColor(turb_field, cv2.COLOR_BGR2GRAY)
                else:
                    turb_gray = turb_field
                
                # Resize to match resolution
                turb_resized = cv2.resize(turb_gray, (self.resolution, self.resolution))
                turb_normalized = turb_resized.astype(np.float32) / 255.0
                
                # Map to amplification
                spatial_amp = 1.0 + (turb_normalized * self.turbulence_amplification)
                amp_map *= spatial_amp
        
        return amp_map
        
    def measure_properties(self):
        """Measure fold density and estimate fractal dimension"""
        self.fold_density_value = np.std(self.height_field)
        
        grad_y, grad_x = np.gradient(self.height_field)
        surface_element = np.sqrt(1 + grad_x**2 + grad_y**2)
        self.surface_area_value = np.sum(surface_element)
        
        # Quick fractal estimate
        binary = (self.height_field > np.mean(self.height_field)).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            
            if area > 100 and perimeter > 10:
                self.fractal_dim_value = 2.0 * np.log(perimeter) / np.log(area)
                self.fractal_dim_value = np.clip(self.fractal_dim_value, 1.0, 3.0)
            else:
                self.fractal_dim_value = 2.0
        else:
            self.fractal_dim_value = 2.0
            
    def reset_simulation(self):
        """Reset to initial state"""
        self.thickness = np.ones((self.resolution, self.resolution), dtype=np.float32)
        self.height_field = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.pressure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.time_step = 0
        self.area_history = []
        self.turbulence_response_value = 0.0
        
    def get_output(self, port_name):
        if port_name == 'fold_density':
            return float(self.fold_density_value)
        elif port_name == 'surface_area':
            return float(self.surface_area_value)
        elif port_name == 'fractal_estimate':
            return float(self.fractal_dim_value)
        elif port_name == 'thickness_map':
            return self.thickness
        elif port_name == 'structure_3d':
            return self.height_field
        elif port_name == 'turbulence_response':
            return self.turbulence_response_value
        return None
        
    def get_display_image(self):
        """5-panel visualization with turbulence response"""
        w, h = 640, 512
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Panel sizes
        panel_w = 320
        panel_h = 256
        
        # Panel 1: Thickness map (top-left)
        thick_vis = cv2.normalize(self.thickness, None, 0, 255, cv2.NORM_MINMAX)
        thick_vis = thick_vis.astype(np.uint8)
        thick_color = cv2.applyColorMap(thick_vis, cv2.COLORMAP_HOT)
        thick_resized = cv2.resize(thick_color, (panel_w, panel_h))
        img[0:panel_h, 0:panel_w] = thick_resized
        cv2.putText(img, "THICKNESS", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Panel 2: Height field / folding (top-right)
        height_vis = cv2.normalize(self.height_field, None, 0, 255, cv2.NORM_MINMAX)
        height_vis = height_vis.astype(np.uint8)
        height_color = cv2.applyColorMap(height_vis, cv2.COLORMAP_VIRIDIS)
        height_resized = cv2.resize(height_color, (panel_w, panel_h))
        img[0:panel_h, panel_w:] = height_resized
        cv2.putText(img, "HEIGHT (FOLDS)", (panel_w+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Panel 3: Pressure map (bottom-left)
        pressure_vis = cv2.normalize(self.pressure, None, 0, 255, cv2.NORM_MINMAX)
        pressure_vis = pressure_vis.astype(np.uint8)
        pressure_color = cv2.applyColorMap(pressure_vis, cv2.COLORMAP_JET)
        pressure_resized = cv2.resize(pressure_color, (panel_w, panel_h))
        img[panel_h:, 0:panel_w] = pressure_resized
        cv2.putText(img, "PRESSURE", (5, panel_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Panel 4: 3D structure (bottom-right)
        grad_y, grad_x = np.gradient(self.height_field)
        light_dir = np.array([-1, -1, 2])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        normals_x = -grad_x
        normals_y = -grad_y
        normals_z = np.ones_like(grad_x)
        
        norm_length = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2)
        normals_x /= (norm_length + 1e-8)
        normals_y /= (norm_length + 1e-8)
        normals_z /= (norm_length + 1e-8)
        
        shading = normals_x * light_dir[0] + normals_y * light_dir[1] + normals_z * light_dir[2]
        shading = np.clip(shading, 0, 1)
        
        shading_vis = (shading * 255).astype(np.uint8)
        shading_color = cv2.applyColorMap(shading_vis, cv2.COLORMAP_BONE)
        shading_resized = cv2.resize(shading_color, (panel_w, panel_h))
        img[panel_h:, panel_w:] = shading_resized
        cv2.putText(img, "3D STRUCTURE", (panel_w+5, panel_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Add metrics at bottom
        metrics_y = h - 50
        cv2.putText(img, f"Step: {self.time_step}", (5, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(img, f"Fold: {self.fold_density_value:.3f}", (100, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(img, f"df≈{self.fractal_dim_value:.2f}", (220, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        # NEW: Show turbulence response
        cv2.putText(img, f"Turb Response: {self.turbulence_response_value:.3f}", (320, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        mode_options = [
            ("Signal Only", "signal"),
            ("Field Only", "field"),
            ("Both", "both")
        ]
        
        return [
            ("Growth Rate", "base_growth", self.base_growth, None),
            ("Fold Threshold", "fold_threshold", self.fold_threshold, None),
            ("Compression", "compression_strength", self.compression_strength, None),
            ("Diffusion", "diffusion", self.diffusion, None),
            ("Resolution", "resolution", self.resolution, None),
            ("Turbulence Amp", "turbulence_amplification", self.turbulence_amplification, None),
            ("Turbulence Mode", "turbulence_mode", self.turbulence_mode, mode_options),
        ]