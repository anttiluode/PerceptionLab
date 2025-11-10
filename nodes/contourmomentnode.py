"""
ContourMomentNode

Calculates geometric moments from a binary (B&W) image
to extract actionable control signals:
- Center of Mass (x, y)
- Area (how much white)
- Orientation (angle of the main shape)
- Eccentricity (how "stretched" the shape is)
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class ContourMomentNode(BaseNode):
    """
    Extracts geometric features from a binary image using moments.
    """
    NODE_CATEGORY = "Analyzer"
    NODE_COLOR = QtGui.QColor(220, 200, 100) # Gold

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Contour Moments"
        
        self.inputs = {
            'image_in': 'image',
            'threshold': 'signal' # To convert grayscale to B&W
        }
        self.outputs = {
            'image': 'image',      # The B&W image + overlay
            'center_x': 'signal',  # Normalized -1 to 1
            'center_y': 'signal',  # Normalized -1 to 1
            'area': 'signal',      # Normalized 0 to 1
            'orientation': 'signal', # Normalized -1 to 1 (-90 to +90 deg)
            'eccentricity': 'signal' # Normalized 0 to 1
        }
        
        # We downscale for performance
        self.size = int(size) 
        
        # Internal state
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        self.center_x = 0.0
        self.center_y = 0.0
        self.area = 0.0
        self.orientation = 0.0
        self.eccentricity = 0.0

    def step(self):
        # --- 1. Get and Prepare Image ---
        img = self.get_blended_input('image_in', 'first')
        
        if img is None:
            # Decay signals if no image
            self.area *= 0.95
            self.eccentricity *= 0.95
            return

        # --- START FIX (for float64 error) ---
        # We must ensure the image is float32 *before* any OpenCV operations
        
        # 1. Convert to float32 if it isn't already
        if img.dtype != np.float32:
             # This will catch float64 (the error) and uint8 (common)
            img = img.astype(np.float32)

        # 2. Normalize to 0-1 if it's in 0-255 range
        if img.max() > 1.0:
            img = img / 255.0
            
        img = np.clip(img, 0, 1) # Ensure range
        # --- END FIX ---
        
        # Resize for performance and consistency
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Convert to grayscale
        if img_resized.ndim == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_resized
        
        # --- 2. Get Binary Image ---
        threshold = self.get_blended_input('threshold', 'sum') or 0.5
        
        _ , binary = cv2.threshold(
            (img_gray * 255).astype(np.uint8), 
            int(threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # --- 3. Calculate Moments ---
        moments = cv2.moments(binary)
        m00 = moments['m00'] # This is the total area (in pixels)

        if m00 > 0:
            # --- Area ---
            self.area = m00 / (self.size * self.size) # Normalized 0-1
            
            # --- Center of Mass ---
            cx = moments['m10'] / m00
            cy = moments['m01'] / m00
            
            # Normalize -1 to 1
            self.center_x = (cx / self.size) * 2.0 - 1.0
            self.center_y = (cy / self.size) * 2.0 - 1.0
            
            # --- Orientation & Eccentricity ---
            mu20 = moments['mu20']
            mu02 = moments['mu02']
            mu11 = moments['mu11']
            
            term = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
            lambda1 = 0.5 * (mu20 + mu02 + term) # Major axis
            lambda2 = 0.5 * (mu20 + mu02 - term) # Minor axis

            angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            self.orientation = angle_rad / (np.pi / 2.0) # Normalize -1 to 1

            if lambda1 > 0 and lambda2 >= 0:
                self.eccentricity = np.sqrt(1.0 - (lambda2 / lambda1))
            else:
                self.eccentricity = 0.0
            
        else:
            # No contours, set all to 0
            self.area = 0.0
            self.center_x = 0.0
            self.center_y = 0.0
            self.orientation = 0.0
            self.eccentricity = 0.0
            
        # --- 4. Prepare Display Image ---
        self.display_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        self.display_image = self.display_image.astype(np.float32) / 255.0
        
        if m00 > 0:
            # Convert normalized coords back to pixel space
            cx_px = int((self.center_x + 1.0) * 0.5 * self.size)
            cy_px = int((self.center_y + 1.0) * 0.5 * self.size)
            
            # Draw Center of Mass (Green Circle)
            cv2.circle(self.display_image, (cx_px, cy_px), 5, (0, 1, 0), -1) 

            # Draw Orientation Line (Magenta)
            angle_rad = self.orientation * (np.pi / 2.0)
            length = self.eccentricity * (self.size / 4.0) + 10 
            
            dx = np.cos(angle_rad) * length
            dy = np.sin(angle_rad) * length
            
            p1 = (int(cx_px - dx), int(cy_px - dy))
            p2 = (int(cx_px + dx), int(cy_px + dy))
            cv2.line(self.display_image, p1, p2, (1, 0, 1), 2)
            
        self.display_image = np.clip(self.display_image, 0, 1)


    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_image
        elif port_name == 'center_x':
            return self.center_x
        elif port_name == 'center_y':
            return self.center_y
        elif port_name == 'area':
            return self.area
        elif port_name == 'orientation':
            return self.orientation
        elif port_name == 'eccentricity':
            return self.eccentricity
        return None