"""
Box Counting Node
------------------
Measures the Fractal Dimension (FD) of an input image using
a simplified box-counting (Higuchi) method.

High FD = High complexity, rough texture (e.g., static)
Low FD  = Low complexity, smooth, simple (e.g., flat color)
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class BoxCountingNode(BaseNode):
    NODE_CATEGORY = "Analyzers"
    NODE_COLOR = QtGui.QColor(0, 150, 130)  # Teal
    
    def __init__(self, k_max=8):
        super().__init__()
        self.node_title = "Fractal Dimension (Box Count)"
        
        self.inputs = {
            'image_in': 'image',
        }
        self.outputs = {
            'fractal_dimension': 'signal',
            'debug_image': 'image',
        }
        
        self.k_max = int(k_max)
        self.fractal_dimension = 0.0
        self.debug_image = np.zeros((256, 256, 3), dtype=np.uint8)

    def higuchi_fd(self, img):
        """A simplified 2D Higuchi/box-counting estimator"""
        if img.ndim == 3:
            img = np.mean(img, axis=2)
            
        N, M = img.shape
        L = []
        x = []
        
        for k in range(1, self.k_max + 1):
            Lk = 0
            for m in range(k):
                for n in range(k):
                    # Create the sub-series
                    sub_img = img[m::k, n::k]
                    if sub_img.size == 0:
                        continue
                    
                    # Sum of absolute differences
                    diff = np.abs(np.diff(sub_img.ravel()))
                    Lk += np.sum(diff)
                    
            if Lk == 0:
                continue
                
            # Average length
            norm_factor = (N * M - 1) / k**2
            L.append(np.log(Lk / (k**2 * norm_factor)))
            x.append(np.log(1.0 / k))
            
        if len(x) < 2:
            return 0.0  # Not enough data to fit
        
        # Fit line to log-log plot
        coeffs = np.polyfit(x, L, 1)
        return coeffs[0]  # The slope is the fractal dimension

    def step(self):
        image_in = self.get_blended_input('image_in', 'first')
        if image_in is None:
            return

        # Downscale for performance
        img_small = cv2.resize(image_in, (64, 64), interpolation=cv2.INTER_AREA)

        # Calculate Fractal Dimension
        fd = self.higuchi_fd(img_small)
        
        # Smooth the output
        self.fractal_dimension = (0.9 * self.fractal_dimension) + (0.1 * fd)
        
        # Update debug image
        self.debug_image.fill(0)
        cv2.putText(self.debug_image, "Fractal Dimension", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.debug_image, f"{self.fractal_dimension:.4f}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3)
        
        # Draw a simple bar graph
        bar_h = int(np.clip(self.fractal_dimension, 0, 3) / 3.0 * 200)
        cv2.rectangle(self.debug_image, (200, 230 - bar_h), (230, 230), (0, 255, 128), -1)

    def get_output(self, port_name):
        if port_name == 'fractal_dimension':
            return self.fractal_dimension
        elif port_name == 'debug_image':
            return self.debug_image
        return None

    def get_display_image(self):
        img = self.debug_image.copy()
        img_resized = np.ascontiguousarray(img)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("K Max (Detail)", "k_max", self.k_max, None),
        ]