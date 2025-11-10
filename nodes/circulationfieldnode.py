"""
CirculationFieldNode

Generates the "Circulation medium" (spacetime) as a
2D vector field based on Perlin noise.

[FIXED-v2] Replaced buggy .repeat() logic with cv2.resize()
to fix broadcasting error when size is not divisible by res.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class CirculationFieldNode(BaseNode):
    """
    Generates a 2D vector field representing the "Circulation medium"
    """
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 150, 220) # Blue

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Circulation Field"
        
        self.inputs = {
            'speed': 'signal',   # How fast the field evolves
            'scale': 'signal',   # Zoom level of the field
            'strength': 'signal' # Magnitude of the vectors
        }
        self.outputs = {
            'vector_field': 'image',  # Raw [vx, vy, 0] data
            'field_viz': 'image'      # Human-readable visualization
        }
        
        self.size = int(size)
        self.z_offset = 0.0 # Time dimension for 3D noise
        
        # We need two noise fields, one for X and one for Y
        self.noise_res = (8, 8)
        self.noise_seed_x = np.random.rand(self.noise_res[0]+1, self.noise_res[1]+1)
        self.noise_seed_y = np.random.rand(self.noise_res[0]+1, self.noise_res[1]+1)
        
        # Initialize output arrays to prevent AttributeError on first frame
        self.vx = np.zeros((self.size, self.size), dtype=np.float32)
        self.vy = np.zeros((self.size, self.size), dtype=np.float32)
        self.viz = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def _generate_noise_slice(self, seed):
        """
        Generates a 2D slice of Perlin-like noise.
        [FIXED] This version uses cv2.resize for robust interpolation.
        """
        # --- Smooth interpolation function ---
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        # --- 1. Get base parameters ---
        res = self.noise_res
        shape = (self.size, self.size)
        
        # --- 2. Create gradient angles ---
        # (Using z_offset for 3D time-varying noise)
        angles = 2*np.pi * (seed + self.z_offset)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        
        # --- 3. Create coordinate grid ---
        # This grid is (size, size, 2) and goes from [0, res]
        delta = (res[0] / shape[0], res[1] / shape[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        
        # --- 4. Get corner gradients ---
        # [FIX] Use cv2.resize(..., interpolation=cv2.INTER_NEAREST)
        # This replaces the buggy .repeat(d[0], 0).repeat(d[1], 1) logic
        # dsize is (w, h), which corresponds to (shape[1], shape[0])
        dsize = (shape[1], shape[0]) 
        
        g00 = cv2.resize(gradients[0:-1, 0:-1], dsize, interpolation=cv2.INTER_NEAREST)
        g10 = cv2.resize(gradients[1:  , 0:-1], dsize, interpolation=cv2.INTER_NEAREST)
        g01 = cv2.resize(gradients[0:-1, 1:  ], dsize, interpolation=cv2.INTER_NEAREST)
        g11 = cv2.resize(gradients[1:  , 1:  ], dsize, interpolation=cv2.INTER_NEAREST)

        # --- 5. Calculate dot products (ramps) ---
        # All arrays (grid, g00, g10, g01, g11) are now guaranteed
        # to be (size, size, 2), so this math is safe.
        n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        
        # --- 6. Interpolate ---
        t = f(grid) # Apply smoothstep to the grid
        
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        
        # Final result is (size, size)
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

    def step(self):
        # --- 1. Get Controls ---
        speed = self.get_blended_input('speed', 'sum') or 0.1
        scale = self.get_blended_input('scale', 'sum') or 1.0
        strength = self.get_blended_input('strength', 'sum') or 1.0
        
        self.z_offset += speed * 0.05
        
        # --- 2. Generate Vector Field ---
        # Map scale to noise resolution
        res_val = int(4 + scale * 12)
        self.noise_res = (res_val, res_val)
        
        # Ensure seeds match new resolution
        if self.noise_seed_x.shape[0] != self.noise_res[0] + 1:
            self.noise_seed_x = np.random.rand(self.noise_res[0]+1, self.noise_res[1]+1)
            self.noise_seed_y = np.random.rand(self.noise_res[0]+1, self.noise_res[1]+1)

        # Generate noise maps for X and Y velocities
        # Result is in [-1, 1] range
        self.vx = self._generate_noise_slice(self.noise_seed_x) * strength
        self.vy = self._generate_noise_slice(self.noise_seed_y) * strength
        
        # --- 3. Create Visualization ---
        self.viz = np.zeros((self.size, self.size, 3), dtype=np.float32)
        step = 10
        for y in range(0, self.size, step):
            for x in range(0, self.size, step):
                vx = self.vx[y, x] * 5 # Scale for viz
                vy = self.vy[y, x] * 5
                
                pt1 = (x, y)
                pt2 = (int(x + vx), int(y + vy))
                
                # Clip points to be inside the image
                pt1 = (np.clip(pt1[0], 0, self.size-1), np.clip(pt1[1], 0, self.size-1))
                pt2 = (np.clip(pt2[0], 0, self.size-1), np.clip(pt2[1], 0, self.size-1))
                
                cv2.arrowedLine(self.viz, pt1, pt2, (1,1,1), 1, cv2.LINE_AA)

    def get_output(self, port_name):
        if port_name == 'vector_field':
            # Output as [vx, vy, 0] image in [-1, 1] range
            # We map this to [0, 1] for image compatibility
            # R = (vx+1)/2, G = (vy+1)/2, B = 0
            field_img = np.dstack([
                (self.vx + 1.0) / 2.0, 
                (self.vy + 1.0) / 2.0, 
                np.zeros((self.size, self.size))
            ])
            return field_img.astype(np.float32)
            
        elif port_name == 'field_viz':
            return self.viz
            
        return None

    def get_display_image(self):
        # We need to return a QImage, but numpy_to_qimage is in the host
        # A simple fix is to just return the float array and let the host handle it
        return self.viz