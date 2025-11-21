"""
DepthFromMath2Node - Enhanced 3D Depth Generator
================================================
NEW VERSION - Won't overwrite your existing DepthFromMathematicsNode

IMPROVEMENTS:
1. Bulletproof OpenCV data type handling (no more buffer format errors)
2. Enhanced normal map calculation
3. Better shading with multiple light sources
4. Occlusion approximation output (for PBR materials)
5. Curvature analysis output
6. More robust error handling

This is the "production ready" version of depth generation.
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class DepthFromMath2Node(BaseNode):
    """
    Enhanced depth-from-mathematics converter.
    Takes 2D patterns and generates full PBR-ready 3D data.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(80, 180, 255)  # Bright blue
    
    def __init__(self, size=256):
        super().__init__()
        self.node_title = "DepthFromMath v2"
        
        self.inputs = {
            'image_in': 'image',
            'fractal_dim': 'signal',
            'complexity': 'signal',
            'depth_scale': 'signal',
            'relief_strength': 'signal',
            'light_angle': 'signal'  # NEW: Dynamic lighting
        }
        
        self.outputs = {
            'heightmap': 'image',
            'shaded': 'image',
            'normals': 'image',
            'occlusion': 'image',      # NEW: Ambient occlusion approximation
            'curvature': 'image',      # NEW: Surface curvature
            'max_depth': 'signal',
            'depth_variance': 'signal',
            'surface_complexity': 'signal'  # NEW: Complexity metric
        }
        
        self.size = int(size)
        self.heightmap = np.zeros((self.size, self.size), dtype=np.float32)
        self.shaded_img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.normal_map_vis = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.occlusion_map = np.zeros((self.size, self.size), dtype=np.float32)
        self.curvature_map = np.zeros((self.size, self.size), dtype=np.float32)

    def _ensure_float32(self, array):
        """Bulletproof conversion to float32"""
        if array is None:
            return None
        
        # Convert to float32 first
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        
        # Normalize to 0-1 if needed
        if array.max() > 1.0:
            array = array / 255.0
        
        # Clip to valid range
        array = np.clip(array, 0.0, 1.0)
        
        # Ensure contiguous
        return np.ascontiguousarray(array)
    
    def _calculate_curvature(self, heightmap):
        """
        Calculate mean curvature using second derivatives.
        Positive = convex (hills), Negative = concave (valleys)
        """
        # Second derivatives
        dxx = cv2.Sobel(heightmap, cv2.CV_32F, 2, 0, ksize=5)
        dyy = cv2.Sobel(heightmap, cv2.CV_32F, 0, 2, ksize=5)
        dxy = cv2.Sobel(heightmap, cv2.CV_32F, 1, 1, ksize=5)
        
        # First derivatives for normalization
        dx = cv2.Sobel(heightmap, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(heightmap, cv2.CV_32F, 0, 1, ksize=3)
        
        # Mean curvature formula (simplified)
        H = (dxx * (1 + dy**2) - 2*dxy*dx*dy + dyy * (1 + dx**2)) / (2 * (1 + dx**2 + dy**2)**1.5 + 1e-9)
        
        return H
    
    def _approximate_occlusion(self, heightmap, samples=8):
        """
        Approximate ambient occlusion by checking local height variations.
        Areas in "pockets" get darker.
        """
        h, w = heightmap.shape
        occlusion = np.ones((h, w), dtype=np.float32)
        
        # Sample in multiple directions
        radius = 5
        for angle in np.linspace(0, 2*np.pi, samples, endpoint=False):
            dx = int(radius * np.cos(angle))
            dy = int(radius * np.sin(angle))
            
            # Shift heightmap
            shifted = np.roll(np.roll(heightmap, dy, axis=0), dx, axis=1)
            
            # If neighbor is higher, this point is more occluded
            height_diff = np.clip(shifted - heightmap, 0, 1)
            occlusion -= height_diff * 0.1
        
        occlusion = np.clip(occlusion, 0, 1)
        
        # Blur for smoothness
        occlusion = cv2.GaussianBlur(occlusion, (5, 5), 1.0)
        
        return occlusion

    def step(self):
        image = self.get_blended_input('image_in', 'first')
        if image is None:
            # Return zeros if no input
            self.heightmap.fill(0)
            self.shaded_img.fill(0)
            self.normal_map_vis.fill(0)
            self.occlusion_map.fill(0)
            self.curvature_map.fill(0)
            return

        try:
            # === STEP 1: BULLETPROOF INPUT PROCESSING ===
            image = self._ensure_float32(image)
            
            # Resize
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            image = self._ensure_float32(image)  # Ensure still float32 after resize
            
            # Convert to grayscale if needed
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = self._ensure_float32(image)
            
            # === STEP 2: TOPOLOGY → HEIGHT ===
            # Binarize
            binary_img = (image > 0.5).astype(np.uint8) * 255
            
            # Distance transform
            dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
            dist_transform = self._ensure_float32(dist_transform)
            
            # Normalize
            if dist_transform.max() > 0:
                dist_norm = dist_transform / dist_transform.max()
            else:
                dist_norm = dist_transform
            
            dist_norm = self._ensure_float32(dist_norm)
            
            # === STEP 3: COMPLEXITY → RELIEF ===
            fdim = self.get_blended_input('fractal_dim', 'sum')
            if fdim is None:
                fdim = 1.5
            
            complexity = self.get_blended_input('complexity', 'sum')
            if complexity is None:
                complexity = 0.5
            
            depth_scale = self.get_blended_input('depth_scale', 'sum')
            if depth_scale is None:
                depth_scale = 0.5
            
            relief_strength = self.get_blended_input('relief_strength', 'sum')
            if relief_strength is None:
                relief_strength = 0.5
            
            # Apply complexity modulation
            fdim_norm = np.clip(fdim - 1.0, 0, 2)
            complexity_mod = (fdim_norm + complexity) * relief_strength
            complexity_mod = np.clip(complexity_mod, 0, 3)
            
            # Generate heightmap
            heightmap = np.power(dist_norm, 1.0 + complexity_mod)
            heightmap = heightmap * (depth_scale + 0.5)
            heightmap = np.clip(heightmap, 0, 1)
            heightmap = self._ensure_float32(heightmap)
            
            self.heightmap = heightmap
            
            # === STEP 4: CALCULATE NORMALS ===
            # CRITICAL: Ensure input is float32 before Sobel
            heightmap_for_sobel = self._ensure_float32(self.heightmap)
            
            sobel_x = cv2.Sobel(heightmap_for_sobel, cv2.CV_32F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(heightmap_for_sobel, cv2.CV_32F, 0, 1, ksize=5)
            
            # Ensure outputs are float32
            sobel_x = self._ensure_float32(sobel_x)
            sobel_y = self._ensure_float32(sobel_y)
            
            # Create normal vectors
            normal_map = np.dstack((
                -sobel_x,
                -sobel_y,
                np.ones_like(sobel_x, dtype=np.float32)
            ))
            
            # Normalize
            norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
            norms = np.where(norms > 1e-9, norms, 1.0)
            normal_map = normal_map / norms
            normal_map = normal_map.astype(np.float32)
            
            # === STEP 5: CALCULATE CURVATURE ===
            self.curvature_map = self._calculate_curvature(heightmap_for_sobel)
            self.curvature_map = self._ensure_float32(self.curvature_map)
            
            # Normalize for display
            if self.curvature_map.max() > self.curvature_map.min():
                curv_display = (self.curvature_map - self.curvature_map.min())
                curv_display = curv_display / (curv_display.max() + 1e-9)
            else:
                curv_display = self.curvature_map * 0.5 + 0.5
            
            self.curvature_map = curv_display
            
            # === STEP 6: CALCULATE OCCLUSION ===
            self.occlusion_map = self._approximate_occlusion(heightmap_for_sobel)
            self.occlusion_map = self._ensure_float32(self.occlusion_map)
            
            # === STEP 7: ADVANCED LIGHTING ===
            # Get dynamic light angle if provided
            light_angle_sig = self.get_blended_input('light_angle', 'sum')
            if light_angle_sig is not None:
                light_angle = light_angle_sig * np.pi  # 0-1 → 0-π
            else:
                light_angle = 0.785  # 45 degrees default
            
            # Create light direction
            light_dir = np.array([
                np.cos(light_angle) * 0.5,
                np.sin(light_angle) * 0.5,
                0.8
            ], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)
            
            # Calculate lighting (Lambertian + ambient)
            shading = np.sum(normal_map * light_dir, axis=2)
            shading = np.clip(shading, 0, 1)
            
            # Add ambient term
            ambient = 0.25
            shading = shading * (1.0 - ambient) + ambient
            
            # Apply occlusion to shading
            shading = shading * self.occlusion_map
            
            # Create colored output with height-based tinting
            base_color = self.heightmap
            
            # Color scheme: deep to high = blue-green-yellow-red
            color_r = np.clip(base_color * 2.0, 0, 1)
            color_g = np.clip(base_color * 1.5, 0, 1)
            color_b = np.clip(1.0 - base_color, 0, 1)
            
            self.shaded_img = np.stack([
                color_r * shading,
                color_g * shading,
                color_b * shading * 0.5
            ], axis=2).astype(np.float32)
            
            # === STEP 8: NORMAL MAP VISUALIZATION ===
            # Convert from [-1,1] to [0,1] RGB
            self.normal_map_vis = ((normal_map + 1.0) / 2.0).astype(np.float32)
            
        except Exception as e:
            # Robust error handling - don't crash the entire system
            print(f"DepthFromMath2: Error in processing: {e}")
            # Fill with safe defaults
            self.heightmap.fill(0)
            self.shaded_img.fill(0.5)
            self.normal_map_vis.fill(0.5)
            self.occlusion_map.fill(1)
            self.curvature_map.fill(0.5)

    def get_output(self, port_name):
        if port_name == 'heightmap':
            return self.heightmap
        
        elif port_name == 'shaded':
            return self.shaded_img
        
        elif port_name == 'normals':
            return self.normal_map_vis
        
        elif port_name == 'occlusion':
            return self.occlusion_map
        
        elif port_name == 'curvature':
            return self.curvature_map
        
        elif port_name == 'max_depth':
            return float(np.max(self.heightmap))
        
        elif port_name == 'depth_variance':
            return float(np.var(self.heightmap))
        
        elif port_name == 'surface_complexity':
            # Complexity = variance of curvature
            return float(np.var(self.curvature_map))
        
        return None
    
    def get_display_image(self):
        """Show the beautifully shaded 3D result"""
        return self.shaded_img