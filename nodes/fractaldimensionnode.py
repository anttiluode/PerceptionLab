"""
Fractal Dimension Node
Implements the coarse-graining method from the primate brain paper
to measure fractal dimension across multiple spatial scales.

Measures At (total area), Ae (exposed area), T (thickness) at each scale
and computes the scaling exponent to determine fractal dimension.
"""

import numpy as np
import cv2
from scipy.spatial import ConvexHull

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class FractalDimensionNode(BaseNode):
    """
    Measures fractal dimension using multi-scale analysis.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(100, 180, 100)  # Green for measurement
    
    def __init__(self):
        super().__init__()
        self.node_title = "Fractal Dimension"
        
        self.inputs = {
            'structure_3d': 'image',     # Height field from growth node
            'thickness_map': 'image',    # Thickness distribution
            'trigger': 'signal'          # When to measure
        }
        
        self.outputs = {
            'fractal_dimension': 'signal',
            'slope_alpha': 'signal',       # The 1.25 slope from paper
            'offset_k': 'signal',          # The k offset
            'scaling_plot': 'image',       # Visualization
            'measurement_ready': 'signal'  # 1.0 when measurement complete
        }
        
        # Measurement settings
        self.num_scales = 10
        self.min_voxel = 2      # Minimum voxel size (pixels)
        self.max_voxel = 64     # Maximum voxel size (pixels)
        
        # Results storage
        self.scales = []
        self.At_values = []  # Total area
        self.Ae_values = []  # Exposed area
        self.T_values = []   # Average thickness
        
        self.fractal_dim = 2.0
        self.slope = 1.0
        self.offset = 0.0
        self.measurement_complete = False
        
    def step(self):
        structure = self.get_blended_input('structure_3d', 'replace')
        thickness = self.get_blended_input('thickness_map', 'replace')
        trigger = self.get_blended_input('trigger', 'sum')
        
        if structure is None:
            return
            
        # Only measure when triggered or continuously
        if trigger is not None and trigger < 0.5:
            self.measurement_complete = False
            return
            
        # Convert to grayscale if needed
        if len(structure.shape) == 3:
            structure_gray = cv2.cvtColor(structure, cv2.COLOR_BGR2GRAY)
        else:
            structure_gray = structure
            
        if thickness is not None:
            if len(thickness.shape) == 3:
                thickness_gray = cv2.cvtColor(thickness, cv2.COLOR_BGR2GRAY)
            else:
                thickness_gray = thickness
        else:
            # Use structure as proxy
            thickness_gray = structure_gray
        
        # Normalize to 0-1
        structure_norm = structure_gray.astype(np.float32) / 255.0
        thickness_norm = thickness_gray.astype(np.float32) / 255.0
        
        # Perform multi-scale measurement
        self.measure_across_scales(structure_norm, thickness_norm)
        
        # Compute fractal dimension from scaling
        self.compute_fractal_dimension()
        
        self.measurement_complete = True
        
    def measure_across_scales(self, height_field, thickness_field):
        """
        Measure At, Ae, T at multiple scales using voxelization.
        This implements the paper's coarse-graining method.
        """
        self.scales = []
        self.At_values = []
        self.Ae_values = []
        self.T_values = []
        
        # Generate logarithmically spaced scales
        voxel_sizes = np.logspace(
            np.log10(self.min_voxel), 
            np.log10(self.max_voxel), 
            self.num_scales
        )
        
        for voxel_size in voxel_sizes:
            voxel_size = int(voxel_size)
            if voxel_size < 1:
                continue
                
            # === COARSE-GRAIN ===
            At, Ae, T = self.coarse_grain_at_scale(height_field, thickness_field, voxel_size)
            
            if At > 0 and Ae > 0 and T > 0:
                self.scales.append(voxel_size)
                self.At_values.append(At)
                self.Ae_values.append(Ae)
                self.T_values.append(T)
                
    def coarse_grain_at_scale(self, height_field, thickness_field, voxel_size):
        """
        Voxelize the surface at given scale and measure properties.
        
        Returns:
            At: Total surface area (accounting for height variations)
            Ae: Exposed (convex hull) area
            T: Average thickness
        """
        h, w = height_field.shape
        
        # Downsample to voxel_size grid
        new_h = max(1, h // voxel_size)
        new_w = max(1, w // voxel_size)
        
        # Resize using max pooling to preserve peaks
        height_coarse = cv2.resize(height_field, (new_w, new_h), interpolation=cv2.INTER_AREA)
        thickness_coarse = cv2.resize(thickness_field, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # === MEASURE At (Total surface area) ===
        # Compute surface area including height variations
        grad_y, grad_x = np.gradient(height_coarse)
        # Surface element: sqrt(1 + |∇h|²)
        surface_element = np.sqrt(1 + grad_x**2 + grad_y**2)
        At = np.sum(surface_element) * (voxel_size ** 2)  # Scale by voxel area
        
        # === MEASURE Ae (Exposed area) ===
        # Convex hull of projected surface
        # For 2D: just the bounding rectangle area
        # (In 3D this would be the convex hull)
        Ae = new_h * new_w * (voxel_size ** 2)
        
        # Alternative: actual convex hull
        # Get points where height > threshold
        threshold = np.mean(height_coarse)
        points = np.argwhere(height_coarse > threshold)
        
        if len(points) > 3:
            try:
                hull = ConvexHull(points)
                Ae = hull.volume * (voxel_size ** 2)  # volume is area in 2D
            except:
                # Fall back to bounding box
                pass
        
        # === MEASURE T (Average thickness) ===
        T = np.mean(thickness_coarse)
        
        return At, Ae, T
        
    def compute_fractal_dimension(self):
        """
        Fit the scaling law: At * T^0.5 = k * Ae^α
        
        Taking log: log(At * √T) = log(k) + α * log(Ae)
        
        Slope α should be 1.25 for df=2.5 (since α = df/2)
        """
        if len(self.scales) < 3:
            return
            
        # Convert to numpy arrays
        At_arr = np.array(self.At_values)
        Ae_arr = np.array(self.Ae_values)
        T_arr = np.array(self.T_values)
        
        # Compute LHS and RHS of scaling law
        y = np.log10(At_arr * np.sqrt(T_arr + 1e-6))
        x = np.log10(Ae_arr + 1e-6)
        
        # Linear regression: y = offset + slope * x
        # Using numpy polyfit
        coeffs = np.polyfit(x, y, deg=1)
        self.slope = coeffs[0]
        self.offset = coeffs[1]
        
        # Fractal dimension: df = 2 * slope
        self.fractal_dim = 2.0 * self.slope
        self.fractal_dim = np.clip(self.fractal_dim, 1.0, 3.0)
        
    def get_output(self, port_name):
        if port_name == 'fractal_dimension':
            return float(self.fractal_dim)
        elif port_name == 'slope_alpha':
            return float(self.slope)
        elif port_name == 'offset_k':
            return float(10 ** self.offset)  # Convert from log
        elif port_name == 'measurement_ready':
            return 1.0 if self.measurement_complete else 0.0
        return None
        
    def get_display_image(self):
        """Visualize the scaling relationship"""
        w, h = 512, 512
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.scales) < 2:
            cv2.putText(img, "Waiting for measurement...", (20, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
        # === PLOT: log(At * √T) vs log(Ae) ===
        At_arr = np.array(self.At_values)
        Ae_arr = np.array(self.Ae_values)
        T_arr = np.array(self.T_values)
        
        y_data = np.log10(At_arr * np.sqrt(T_arr + 1e-6))
        x_data = np.log10(Ae_arr + 1e-6)
        
        # Normalize to plot space
        margin = 50
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        def to_plot_coords(x, y):
            px = int(margin + (x - x_min) / (x_max - x_min) * plot_w)
            py = int(h - margin - (y - y_min) / (y_max - y_min) * plot_h)
            return px, py
        
        # Draw axes
        cv2.line(img, (margin, h - margin), (w - margin, h - margin), (100, 100, 100), 2)
        cv2.line(img, (margin, h - margin), (margin, margin), (100, 100, 100), 2)
        
        # Draw data points
        for i in range(len(x_data)):
            px, py = to_plot_coords(x_data[i], y_data[i])
            cv2.circle(img, (px, py), 5, (0, 255, 255), -1)
            
        # Draw regression line
        x_fit = np.array([x_min, x_max])
        y_fit = self.offset + self.slope * x_fit
        
        px1, py1 = to_plot_coords(x_fit[0], y_fit[0])
        px2, py2 = to_plot_coords(x_fit[1], y_fit[1])
        cv2.line(img, (px1, py1), (px2, py2), (255, 0, 255), 2)
        
        # Draw reference line (slope = 1.25 from paper)
        y_ref = y_data.mean() + 1.25 * (x_fit - x_data.mean())
        px1, py1 = to_plot_coords(x_fit[0], y_ref[0])
        px2, py2 = to_plot_coords(x_fit[1], y_ref[1])
        cv2.line(img, (px1, py1), (px2, py2), (0, 255, 0), 1)
        
        # Labels
        cv2.putText(img, "log(Ae)", (w - margin - 60, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "log(At*√T)", (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Results
        results_y = margin - 10
        cv2.putText(img, f"Slope α = {self.slope:.3f} (theory: 1.25)", 
                   (margin + 10, results_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        results_y += 20
        cv2.putText(img, f"Fractal dim df = {self.fractal_dim:.3f} (theory: 2.5)", 
                   (margin + 10, results_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        results_y += 20
        offset_k = 10 ** self.offset
        cv2.putText(img, f"Offset k = {offset_k:.4f} (theory: 0.228)", 
                   (margin + 10, results_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Legend
        legend_y = h - margin + 30
        cv2.circle(img, (margin + 10, legend_y), 5, (0, 255, 255), -1)
        cv2.putText(img, "Measured", (margin + 20, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.line(img, (margin + 80, legend_y), (margin + 100, legend_y), (255, 0, 255), 2)
        cv2.putText(img, "Fit", (margin + 105, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.line(img, (margin + 140, legend_y), (margin + 160, legend_y), (0, 255, 0), 1)
        cv2.putText(img, "Theory", (margin + 165, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Num Scales", "num_scales", self.num_scales, None),
            ("Min Voxel Size", "min_voxel", self.min_voxel, None),
            ("Max Voxel Size", "max_voxel", self.max_voxel, None),
        ]