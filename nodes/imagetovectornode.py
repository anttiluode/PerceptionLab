"""
Image To Vector Node (The Bridge)
---------------------------------
Downsamples a 2D image into a 1D latent vector.
Crucial for connecting Visual/Physics nodes (Images) to Cognitive nodes (Vectors).
Fixes the 'broadcast input array' crash in the Observer.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ImageToVectorNode(BaseNode):
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(120, 120, 120)
    
    def __init__(self, output_dim=16):
        super().__init__()
        self.node_title = "Image -> Vector"
        
        self.inputs = {
            'image_in': 'image'
        }
        
        self.outputs = {
            'vector_out': 'spectrum'
        }
        
        self.output_dim = int(output_dim)
        self.vector = np.zeros(self.output_dim, dtype=np.float32)

    def step(self):
        img = self.get_blended_input('image_in', 'first')
        
        if img is None:
            return
            
        # 1. Handle dimensions
        if img.ndim == 3:
            # Flatten RGB to Grayscale
            img = np.mean(img, axis=2)
            
        # 2. Calculate grid size for downsampling
        # We want 'output_dim' pixels total. Sqrt(16) = 4x4 grid.
        side = int(np.ceil(np.sqrt(self.output_dim)))
        
        # 3. Resize (Downsample)
        # This averages the pixels, effectively integrating the field information
        tiny_img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        
        # 4. Flatten
        flat = tiny_img.flatten()
        
        # 5. Trim or Pad to exact dimension
        if len(flat) >= self.output_dim:
            self.vector = flat[:self.output_dim]
        else:
            self.vector[:len(flat)] = flat
            
        # Normalize
        if np.max(np.abs(self.vector)) > 0:
            self.vector /= np.max(np.abs(self.vector))

    def get_output(self, port_name):
        if port_name == 'vector_out':
            return self.vector
        return None
        
    def get_display_image(self):
        # Visualizer: Bar graph of the vector
        w, h = 128, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.output_dim > 0:
            bar_w = w // self.output_dim
            for i, val in enumerate(self.vector):
                height = int(val * h)
                cv2.rectangle(img, (i*bar_w, h-height), ((i+1)*bar_w-1, h), (0, 255, 0), -1)
            
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [("Output Dim", "output_dim", self.output_dim, None)]