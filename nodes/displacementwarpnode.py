"""
DisplacementWarpNode

Uses a heightmap to "pop out" or distort a texture,
creating a powerful, liquid-like 3D effect.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class DisplacementWarpNode(BaseNode):
    """
    Distorts an image based on a heightmap.
    """
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(150, 100, 220) # Purple

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Displacement Warp"
        
        self.inputs = {
            'image_in': 'image',      # The texture (e.g., checkerboard)
            'heightmap_in': 'image',  # The displacement map (e.g., your pyramid)
            'strength': 'signal'      # 0-1, how much to distort
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        
        # Pre-calculate grids
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.grid_x = x.astype(np.float32)
        self.grid_y = y.astype(np.float32)
        
        # --- START FIX ---
        # Initialize the output variable so it exists before step() runs
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        # --- END FIX ---

    def _prepare_image(self, img):
        """Helper to resize and format an input image."""
        if img is None:
            return None
        
        # Ensure float32 in 0-1 range
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
            
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        return np.clip(img_resized, 0, 1)

    def step(self):
        # --- 1. Get Images and Controls ---
        img_texture = self._prepare_image(self.get_blended_input('image_in', 'first'))
        img_heightmap = self._prepare_image(self.get_blended_input('heightmap_in', 'first'))
        
        strength = (self.get_blended_input('strength', 'sum') or 0.2) * 100.0 # Scale to pixels
        
        # --- 2. Handle Missing Inputs ---
        if img_texture is None:
            # If no texture, just show the heightmap
            self.display_image = img_heightmap if img_heightmap is not None else \
                                 np.zeros((self.size, self.size, 3), dtype=np.float32)
            return
            
        if img_heightmap is None:
            # If no heightmap, just pass the texture through
            self.display_image = img_texture
            return
            
        # Ensure heightmap is grayscale
        if img_heightmap.ndim == 3:
            img_heightmap_gray = cv2.cvtColor(img_heightmap, cv2.COLOR_RGB2GRAY)
        else:
            img_heightmap_gray = img_heightmap
            
        # --- 3. Apply Displacement ---
        # Where heightmap is "high" (1.0), this will be a large offset
        # Where it's "low" (0.0), this will be 0 offset
        displacement = img_heightmap_gray * strength
        
        # Create the remap "flow"
        # We "push" pixels outwards from the center of the height
        map_x = (self.grid_x + displacement).astype(np.float32)
        map_y = (self.grid_y + displacement).astype(np.float32)
        
        # --- 4. Apply Warp ---
        self.display_image = cv2.remap(
            img_texture, map_x, map_y, 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101 # Reflects for cool psychedelic tiling
        )

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_image
        return None