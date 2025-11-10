"""
FilamentBoxcountNode

Extracts bright "filaments" from an image via thresholding,
displays them, and calculates their fractal dimension using
a box-counting algorithm.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class FilamentBoxcountNode(BaseNode):
    """
    Analyzes the fractal dimension of filaments in an image.
    """
    NODE_CATEGORY = "Analyzer"
    NODE_COLOR = QtGui.QColor(220, 180, 100) # Gold

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Filament Boxcounter"
        
        self.inputs = {
            'image_in': 'image',
            'threshold': 'signal' # 0-1, controls filament detection
        }
        self.outputs = {
            'image': 'image',         # The binary filament image
            'fractal_dim': 'signal',  # The calculated fractal dimension (1.0 - 2.0)
            'density': 'signal'       # How many pixels are "on" (0-1)
        }
        
        # Box counting is SLOW on large images.
        # We process a downscaled version for speed.
        self.size = int(size) 
        
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.fractal_dim = 1.0
        self.density = 0.0

    def _box_count(self, binary_img):
        """
        Performs a box-counting algorithm on a binary image.
        Uses a fast method optimized for sparse pixels.
        """
        # Find the coordinates of all "on" pixels
        pixels = np.argwhere(binary_img > 0)
        
        if len(pixels) == 0:
            return 1.0 # No dimension if no pixels

        # Use 8 scales, from 2 up to size/2
        max_log = np.log2(self.size // 2)
        scales = np.logspace(1.0, max_log, num=8, base=2)
        scales = np.unique(np.round(scales).astype(int))
        
        counts = []
        valid_scales = []
        
        for scale in scales:
            if scale < 2: continue
            
            # Use a set to store unique box indices
            # This is much faster than iterating over a full grid
            box_indices = set()
            for y, x in pixels:
                box_indices.add( (y // scale, x // scale) )
            
            # We must have at least one box to count
            if len(box_indices) > 0:
                counts.append(len(box_indices))
                valid_scales.append(scale)
        
        if len(counts) < 2:
            return 1.0 # Not enough data to fit a line

        # Fit a line to log(counts) vs log(scales)
        # The fractal dimension D is the *negative* slope.
        # N(s) ∝ s^(-D)  =>  log(N) = -D * log(s) + C
        try:
            coeffs = np.polyfit(np.log(valid_scales), np.log(counts), 1)
            dimension = -coeffs[0]
        except np.linalg.LinAlgError:
            dimension = 1.0 # Fitting failed
        
        # A 2D fractal dimension must be between 1 (a line) and 2 (a filled plane)
        return np.clip(dimension, 1.0, 2.0)

    def step(self):
        # --- 1. Get and Prepare Image ---
        img = self.get_blended_input('image_in', 'first')
        if img is None:
            return # Do nothing if no image

        # Resize for performance
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Convert to grayscale
        if img_resized.ndim == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_resized
            
        # Ensure 0-1 float
        if img_gray.max() > 1.0:
            img_gray = img_gray.astype(np.float32) / 255.0
        
        # --- 2. Extract Filaments ---
        threshold = self.get_blended_input('threshold', 'sum') or 0.5
        
        # Apply threshold to get the binary image
        _ , binary_img = cv2.threshold(
            (img_gray * 255).astype(np.uint8), 
            int(threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # --- 3. Analyze ---
        self.fractal_dim = self._box_count(binary_img)
        self.density = np.sum(binary_img > 0) / binary_img.size
        
        # --- 4. Prepare Display ---
        # Convert the B/W filament image to color for display
        self.display_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        self.display_image = self.display_image.astype(np.float32) / 255.0

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_image
        elif port_name == 'fractal_dim':
            return self.fractal_dim
        elif port_name == 'density':
            return self.density
        return None