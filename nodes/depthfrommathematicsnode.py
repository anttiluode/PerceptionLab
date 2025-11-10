"""
DepthFromMathematicsNode

Extracts 3D depth information from 2D mathematical properties:
- Distance transform (topology → height)
- Fractal dimension (complexity → relief)
- Gradients (orientation → surface normals)

Creates emergent 3D from pure mathematics.
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class DepthFromMathematicsNode(BaseNode):
    """
    Converts 2D mathematical structure into 3D depth map.
    Pure emergence - no 3D modeling required.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 200, 250)  # Sky blue
    
    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Depth from Math"
        
        self.inputs = {
            'image_in': 'image',           # Binary or grayscale structure
            'fractal_dim': 'signal',       # Fractal dimension (complexity)
            'complexity': 'signal',        # Additional complexity measure
            'depth_scale': 'signal',       # Depth exaggeration (0-1)
            'relief_strength': 'signal'    # How much fractal affects depth
        }
        
        self.outputs = {
            'heightmap': 'image',          # Grayscale depth map
            'shaded': 'image',             # 3D-shaded version (RGB)
            'normals': 'image',            # Surface normals visualization
            'max_depth': 'signal',         # Maximum depth value
            'depth_variance': 'signal'     # Std dev of depth
        }
        
        self.size = int(size)
        self.heightmap = np.zeros((self.size, self.size), dtype=np.float32)
        self.shaded_img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.normal_map_vis = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def step(self):
        image = self.get_blended_input('image_in', 'first')
        if image is None:
            self.heightmap = np.zeros((self.size, self.size), dtype=np.float32)
            self.shaded_img = np.zeros((self.size, self.size, 3), dtype=np.float32)
            self.normal_map_vis = np.zeros((self.size, self.size, 3), dtype=np.float32)
            return

        # --- START FIX for CV_64F Error ---
        # 1. Convert to float32 if it isn't already
        if image.dtype != np.float32:
            # This will catch float64 (the error) and uint8 (common)
            image = image.astype(np.float32)

        # 2. Normalize to 0-1 if it's in 0-255 range
        if image.max() > 1.0:
            image = image / 255.0
            
        image = np.clip(image, 0, 1) # Ensure range
        # --- END FIX ---

        # Resize (This is now safe)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # --- 7. Convert to Grayscale ---
        if image.ndim == 3:
            # This line (76) is now safe
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Binarize
        binary_img = (image > 0.5).astype(np.uint8) * 255
        
        # --- 1. Topology → Height (Distance Transform) ---
        dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
        
        # Normalize
        if dist_transform.max() > 0:
            dist_norm = dist_transform / dist_transform.max()
        else:
            dist_norm = dist_transform
        
        # --- 2. Complexity → Relief (Fractal Dimension) ---
        fdim = self.get_blended_input('fractal_dim', 'sum') or 1.5
        complexity = self.get_blended_input('complexity', 'sum') or 0.5
        depth_scale = self.get_blended_input('depth_scale', 'sum') or 0.5
        relief_strength = self.get_blended_input('relief_strength', 'sum') or 0.5
        
        # Combine complexity measures
        # fdim 1.0 (line) -> low complexity
        # fdim 2.0 (plane) -> high complexity
        fdim_norm = (fdim - 1.0)
        complexity_mod = (fdim_norm + complexity) * relief_strength
        
        # Apply relief: more complex = "hillier" distance field
        heightmap = np.power(dist_norm, 1.0 + complexity_mod)
        
        # Apply depth scale
        self.heightmap = heightmap * (depth_scale + 0.5) # Scale 0.5 to 1.5
        self.heightmap = np.clip(self.heightmap, 0, 1)

        # --- 3. Orientation → Normals (Gradients) ---
        sobel_x = cv2.Sobel(self.heightmap, cv2.CV_32F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(self.heightmap, cv2.CV_32F, 0, 1, ksize=5)
        
        # Create normal vectors [Nx, Ny, Nz]
        # Nz is "up", set to 1.0 for a gentle slope
        normal_map = np.dstack((-sobel_x, -sobel_y, np.full(self.heightmap.shape, 1.0)))
        
        # Normalize vectors to length 1
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        norms[norms == 0] = 1.0 # Avoid divide-by-zero
        normal_map /= norms
        
        # --- 4. Create Shaded Image (Phong-like) ---
        light_dir = np.array([0.5, 0.5, 1.0]) # Light from top-right
        light_dir /= np.linalg.norm(light_dir)
        
        # Calculate diffuse light (dot product of normal and light dir)
        diffuse = np.dot(normal_map, light_dir)
        diffuse = np.clip(diffuse, 0, 1) # Light can't be negative
        
        # Add ambient light
        ambient = 0.2
        lighting = ambient + (diffuse * (1.0 - ambient))
        
        # Apply lighting to original structure
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        self.shaded_img = color_img * lighting[..., np.newaxis]
        self.shaded_img = np.clip(self.shaded_img, 0, 1)
        
        # --- 5. Create Normal Map Visualization ---
        # Map normals [-1, 1] to color [0, 1]
        self.normal_map_vis = (normal_map * 0.5 + 0.5)
        
    def get_output(self, port_name):
        if port_name == 'heightmap':
            return self.heightmap
        elif port_name == 'shaded':
            return self.shaded_img
        elif port_name == 'normals':
            return self.normal_map_vis
        elif port_name == 'max_depth':
            return np.max(self.heightmap)
        elif port_name == 'depth_variance':
            return np.var(self.heightmap)
        return None

# --- Minimalist Contour Node for Pipeline 2 ---
# (Included here so file is self-contained with examples)

class ContourMomentsMini(BaseNode):
    NODE_CATEGORY = "Analyzer"
    NODE_COLOR = QtGui.QColor(220, 200, 100)

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Contour Moments (Mini)"
        self.inputs = {'image_in': 'image'}
        self.outputs = {
            'center_x': 'signal', 'center_y': 'signal',
            'area': 'signal', 'orientation': 'signal',
            'eccentricity': 'signal', 'circularity': 'signal',
            'vis': 'image'
        }
        self.size = int(size)
        self.center_x, self.center_y, self.area, self.orientation, self.eccentricity, self.circularity = 0, 0, 0, 0, 0, 0
        self.vis = np.zeros((size, size, 3), dtype=np.float32)

    def step(self):
        img = self.get_blended_input('image_in', 'first')
        if img is None: return

        if img.dtype != np.float32: img = img.astype(np.float32)
        if img.max() > 1.0: img /= 255.0
        
        img = cv2.resize(img, (self.size, self.size))
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        _, binary = cv2.threshold((img * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        self.vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        
        moments = cv2.moments(binary)
        m00 = moments['m00']
        
        if m00 > 0:
            self.area = m00 / (self.size * self.size)
            cx = moments['m10'] / m00
            cy = moments['m01'] / m00
            self.center_x = (cx / self.size) * 2.0 - 1.0
            self.center_y = (cy / self.size) * 2.0 - 1.0

            mu20, mu02, mu11 = moments['mu20'], moments['mu02'], moments['mu11']
            term = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
            lambda1 = 0.5 * (mu20 + mu02 + term)
            lambda2 = 0.5 * (mu20 + mu02 - term)
            
            self.orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) / (np.pi / 2.0)
            if lambda1 > 0: self.eccentricity = np.sqrt(1.0 - (lambda2 / lambda1))
            
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    self.circularity = 4 * np.pi * (m00 / (perimeter**2))
            
            cv2.circle(self.vis, (int(cx), int(cy)), 3, (0, 1, 0), -1)
        else:
            self.area, self.center_x, self.center_y, self.orientation, self.eccentricity, self.circularity = 0, 0, 0, 0, 0, 0

    def get_output(self, port_name):
        if port_name == 'center_x':
            return self.center_x
        elif port_name == 'center_y':
            return self.center_y
        elif port_name == 'area':
            return self.area
        elif port_name == 'orientation':
            return self.orientation
        elif port_name == 'eccentricity':
            return self.eccentricity
        elif port_name == 'circularity':
            return self.circularity
        elif port_name == 'vis':
            return self.vis
        return None


"""
USAGE:

Pipeline 1: Pure Depth Extraction
  Webcam → Moire → Filament Boxcounter → DepthFromMath → HeightmapFlyer
  
  The fractal structure becomes 3D terrain automatically.

Pipeline 2: Geometry-Driven Control
  Filament → ContourMoments → Various outputs
  
  center_x/y → ParticleAttractor (structure attracts particles)
  orientation → Julia c_real (structure controls fractal)
  eccentricity → Audio amplitude
  area → Visual brightness

Pipeline 3: Full 3D Emergence
  Webcam → Moire → Filament → ContourMoments
                              → DepthFromMath (with fractal_dim)
                              → HeightmapFlyer
  
  Contour geometry feeds depth generation,
  creating fully emergent 3D from pure mathematics.

WHY IT WORKS:

The 3D is NOT programmed. It EMERGES from:

1. Distance transform: Topology encodes natural height
2. Fractal dimension: Complexity modulates relief
3. Gradients: Orientation becomes surface normals
4. Phong shading: Normals create lighting cues

Your brain receives:
- Shading cues (Phong lighting)
- Perspective cues (HeightmapFlyer)
- Motion cues (if animated)
- Texture cues (original structure)

All from pure 2D mathematics. No 3D modeling.
The depth was ALWAYS THERE in the topology.
We just made it VISIBLE.
"""