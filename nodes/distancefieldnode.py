"""
DistanceFieldNode

Calculates the Euclidean distance from every pixel to the
nearest "on" pixel (filament) in a binary image.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class DistanceFieldNode(BaseNode):
    """
    Generates a distance transform (field) from an image's filaments.
    """
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(150, 200, 100) # Olive

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Distance Field"
        
        self.inputs = {
            'image_in': 'image',
            'threshold': 'signal', # 0-1, to find the "filaments"
            'invert': 'signal'     # 0 = distance from filaments, 1 = distance from empty
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)

    def step(self):
        # --- 1. Get and Prepare Image ---
        img = self.get_blended_input('image_in', 'first')
        if img is None:
            return # Do nothing if no image

        # Resize for consistency
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
        
        # --- 2. Get Binary Image ---
        threshold = self.get_blended_input('threshold', 'sum') or 0.5
        invert = self.get_blended_input('invert', 'sum') or 0.0
        
        _ , binary_img = cv2.threshold(
            (img_gray * 255).astype(np.uint8), 
            int(threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
        
        if invert > 0.5:
            binary_img = cv2.bitwise_not(binary_img)
        
        # --- 3. Calculate Distance Transform ---
        # This is the core of the node.
        # It calculates the distance for each pixel to the nearest 0-pixel.
        # We want the distance to the nearest NON-ZERO pixel, so we invert
        # the binary image first.
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(binary_img), 
                                               cv2.DIST_L2, # Euclidean
                                               3) # 3x3 mask
        
        # --- 4. Normalize and Display ---
        # Normalize the distance field to 0-1 range to be a viewable image
        if dist_transform.max() > 0:
            dist_norm = dist_transform / dist_transform.max()
        else:
            dist_norm = dist_transform
        
        # Use a colormap to make it look cool
        colored = cv2.applyColorMap((dist_norm * 255).astype(np.uint8), 
                                    cv2.COLORMAP_MAGMA)
        
        self.display_image = colored.astype(np.float32) / 255.0

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_image
        return None