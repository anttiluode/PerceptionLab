"""
Cortical 3D Growth Node
Simulates eigenmode-driven cortical morphogenesis in 3D.

Takes eigenmode activation map and grows a 3D cortical structure,
implementing buckling/folding when thickness exceeds constraints.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from scipy.interpolate import interp2d

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class Cortical3DGrowthNode(BaseNode):
    """
    Grows 3D cortical structure driven by eigenmode activation.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(180, 80, 180)  # Purple for morphogenesis
    
    def __init__(self):
        super().__init__()
        self.node_title = "3D Cortical Growth"
        
        self.inputs = {
            'lobe_activation': 'image',      # From eigenmode node
            'growth_rate': 'signal',         # Modulate growth speed
            'reset': 'signal'                # Reset simulation
        }
        
        self.outputs = {
            'thickness_map': 'image',        # 2D thickness distribution
            'fold_density': 'signal',        # How much folding
            'surface_area': 'signal',        # Total surface area
            'fractal_estimate': 'signal',    # Quick df estimate
            'structure_3d': 'image'          # 3D visualization slice
        }
        
        # Simulation parameters
        self.resolution = 128           # Grid resolution
        self.dt = 0.01                  # Time step
        self.base_growth = 0.001        # Base growth rate
        self.fold_threshold = 2.5       # When to start folding
        self.compression_strength = 0.3 # How strong buckling is
        self.diffusion = 0.1           # Spatial smoothing
        
        # State variables
        self.thickness = np.ones((self.resolution, self.resolution), dtype=np.float32)
        self.height_field = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.pressure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.time_step = 0
        
        # For fractal measurement
        self.area_history = []
        
        # Initialize measurement values (needed for display before first step)
        self.fold_density_value = 0.0
        self.surface_area_value = 0.0
        self.fractal_dim_value = 2.0
        
    def step(self):
        # Get inputs
        activation = self.get_blended_input('lobe_activation', 'replace')
        growth_mod = self.get_blended_input('growth_rate', 'sum')
        reset_signal = self.get_blended_input('reset', 'sum')
        
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
        
        # === GROWTH PHASE ===
        # Where eigenmodes are active → cortex grows thicker
        growth_field = activation_normalized * total_growth_rate * self.dt
        self.thickness += growth_field
        
        # === CONSTRAINT PHASE ===
        # Compute "pressure" where thickness exceeds threshold
        excess = np.clip(self.thickness - self.fold_threshold, 0, None)
        self.pressure = excess ** 2  # Quadratic pressure
        
        # === FOLDING PHASE ===
        # Pressure causes height deformation (buckling)
        # Compute curvature from thickness gradient
        grad_y, grad_x = np.gradient(self.thickness)
        laplacian = cv2.Laplacian(self.thickness, cv2.CV_32F)
        
        # Fold direction opposes pressure gradient
        fold_force_x = -grad_x * self.pressure * self.compression_strength
        fold_force_y = -grad_y * self.pressure * self.compression_strength
        
        # Also influenced by local curvature (buckles inward)
        fold_force_z = -laplacian * self.pressure * self.compression_strength * 0.5
        
        # Apply folding to height field
        self.height_field += fold_force_z * self.dt
        
        # Redistribute thickness where folding occurs
        # Folded regions compress laterally
        fold_magnitude = np.sqrt(fold_force_x**2 + fold_force_y**2 + fold_force_z**2)
        thickness_redistribution = fold_magnitude * 0.1
        self.thickness -= thickness_redistribution
        self.thickness = np.clip(self.thickness, 0.1, 10.0)  # Bounds
        
        # === DIFFUSION PHASE ===
        # Smooth to prevent instabilities
        self.thickness = gaussian_filter(self.thickness, sigma=self.diffusion)
        self.height_field = gaussian_filter(self.height_field, sigma=self.diffusion)
        
        # === MEASUREMENT ===
        self.measure_properties()
        
        self.time_step += 1
        
    def measure_properties(self):
        """Measure fold density and estimate fractal dimension"""
        # Fold density: variance in height field
        self.fold_density_value = np.std(self.height_field)
        
        # Surface area estimate using gradient
        grad_y, grad_x = np.gradient(self.height_field)
        surface_element = np.sqrt(1 + grad_x**2 + grad_y**2)
        self.surface_area_value = np.sum(surface_element)
        
        # Quick fractal estimate using perimeter-area relationship
        # For 2D projection: perimeter ~ area^(df/2)
        # So df ≈ 2 * log(perimeter) / log(area)
        
        # Threshold height field to get "cortex vs background"
        binary = (self.height_field > np.mean(self.height_field)).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            
            if area > 100 and perimeter > 10:
                # Estimate fractal dimension
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
        
    def get_output(self, port_name):
        if port_name == 'fold_density':
            return float(self.fold_density_value)
        elif port_name == 'surface_area':
            return float(self.surface_area_value)
        elif port_name == 'fractal_estimate':
            return float(self.fractal_dim_value)
        elif port_name == 'thickness_map':
            # Already an image
            return self.thickness
        elif port_name == 'structure_3d':
            # Return height field as output
            return self.height_field
        return None
        
    def get_display_image(self):
        """4-panel visualization"""
        w, h = 512, 512
        panel_size = 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Panel 1: Thickness map (top-left)
        thick_vis = cv2.normalize(self.thickness, None, 0, 255, cv2.NORM_MINMAX)
        thick_vis = thick_vis.astype(np.uint8)
        thick_color = cv2.applyColorMap(thick_vis, cv2.COLORMAP_HOT)
        thick_resized = cv2.resize(thick_color, (panel_size, panel_size))
        img[0:panel_size, 0:panel_size] = thick_resized
        cv2.putText(img, "THICKNESS", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Panel 2: Height field / folding (top-right)
        height_vis = cv2.normalize(self.height_field, None, 0, 255, cv2.NORM_MINMAX)
        height_vis = height_vis.astype(np.uint8)
        height_color = cv2.applyColorMap(height_vis, cv2.COLORMAP_VIRIDIS)
        height_resized = cv2.resize(height_color, (panel_size, panel_size))
        img[0:panel_size, panel_size:] = height_resized
        cv2.putText(img, "HEIGHT (FOLDS)", (panel_size+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Panel 3: Pressure map (bottom-left)
        pressure_vis = cv2.normalize(self.pressure, None, 0, 255, cv2.NORM_MINMAX)
        pressure_vis = pressure_vis.astype(np.uint8)
        pressure_color = cv2.applyColorMap(pressure_vis, cv2.COLORMAP_JET)
        pressure_resized = cv2.resize(pressure_color, (panel_size, panel_size))
        img[panel_size:, 0:panel_size] = pressure_resized
        cv2.putText(img, "PRESSURE", (5, panel_size+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Panel 4: 3D-like rendering (bottom-right)
        # Create pseudo-3D by computing shaded relief
        grad_y, grad_x = np.gradient(self.height_field)
        # Fake lighting from top-left
        light_dir = np.array([-1, -1, 2])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Normal vectors
        normals_x = -grad_x
        normals_y = -grad_y
        normals_z = np.ones_like(grad_x)
        
        # Normalize
        norm_length = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2)
        normals_x /= (norm_length + 1e-8)
        normals_y /= (norm_length + 1e-8)
        normals_z /= (norm_length + 1e-8)
        
        # Dot product with light
        shading = normals_x * light_dir[0] + normals_y * light_dir[1] + normals_z * light_dir[2]
        shading = np.clip(shading, 0, 1)
        
        # Colorize
        shading_vis = (shading * 255).astype(np.uint8)
        shading_color = cv2.applyColorMap(shading_vis, cv2.COLORMAP_BONE)
        shading_resized = cv2.resize(shading_color, (panel_size, panel_size))
        img[panel_size:, panel_size:] = shading_resized
        cv2.putText(img, "3D STRUCTURE", (panel_size+5, panel_size+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Add metrics at bottom
        metrics_y = h - 30
        cv2.putText(img, f"Step: {self.time_step}", (5, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(img, f"Fold Density: {self.fold_density_value:.3f}", (120, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(img, f"Area: {self.surface_area_value:.1f}", (280, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(img, f"df≈{self.fractal_dim_value:.2f}", (400, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Growth Rate", "base_growth", self.base_growth, None),
            ("Fold Threshold", "fold_threshold", self.fold_threshold, None),
            ("Compression", "compression_strength", self.compression_strength, None),
            ("Diffusion", "diffusion", self.diffusion, None),
            ("Resolution", "resolution", self.resolution, None),
        ]