import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import cv2
import numpy as np

class ZoomNode(BaseNode):
    """
    ZOOM NODE: Recursive Fractal Driver
    -----------------------------------
    Zooms into an image by a specified factor around a center point.
    When used in a loop with Eigen->Image, it creates a 'Phase-Inertial Dive'.
    
    Logic:
    1. Crop a central (or offset) region.
    2. Interpolate back to full resolution.
    3. The interpolation 'error' becomes the next frame's seeds.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 200, 100)  # Green - Growth/Zoom

    def __init__(self, zoom_factor=1.02, center_x=0.5, center_y=0.5):
        super().__init__()
        self.node_title = "Zoom Dive"
        
        self.inputs = {
            'image_in': 'image'
        }
        
        self.outputs = {
            'image_out': 'image'
        }
        
        self.zoom_factor = float(zoom_factor)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.output_img = None

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is None:
            return

        # Handle grayscale vs color
        h, w = img_in.shape[:2]
        
        # Calculate crop dimensions
        # If zoom_factor > 1, we crop a smaller area
        # If zoom_factor < 1, we pad (zoom out)
        new_w = w / self.zoom_factor
        new_h = h / self.zoom_factor
        
        # Calculate top-left based on center
        x1 = int(w * self.center_x - new_w / 2)
        y1 = int(h * self.center_y - new_h / 2)
        x2 = int(x1 + new_w)
        y2 = int(y1 + new_h)
        
        # Clamp and handle out-of-bounds (Pad with edge color if zooming out)
        if self.zoom_factor >= 1.0:
            # Traditional Zoom In
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)
            
            crop = img_in[y1_c:y2_c, x1_c:x2_c]
            
            # Use INTER_CUBIC to create the 'LSD' effect - smooth gradients
            # that the EigenNode can latch onto.
            self.output_img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            # Zoom Out (Padding)
            # Create a larger canvas and put the image in it, then resize down
            # Or simpler: resize original and place on background
            rescaled = cv2.resize(img_in, (int(w * self.zoom_factor), int(h * self.zoom_factor)), 
                                  interpolation=cv2.INTER_AREA)
            rh, rw = rescaled.shape[:2]
            
            # Start with a black/gray canvas
            canvas = np.zeros_like(img_in)
            
            # Paste in center
            # (Simplifying for now to just center)
            dy = (h - rh) // 2
            dx = (w - rw) // 2
            canvas[dy:dy+rh, dx:dx+rw] = rescaled
            self.output_img = canvas

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.output_img
        return None
        
    def get_display_image(self):
        if self.output_img is None:
            return None
        
        img = self.output_img
        # Protect against NaN/Inf
        img = np.nan_to_num(img)
        
        # Convert to BGR/RGB for display
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # BGR to RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        h, w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Zoom Factor", "zoom_factor", self.zoom_factor, "float"),
            ("Center X (0.1-0.9)", "center_x", self.center_x, "float"),
            ("Center Y (0.1-0.9)", "center_y", self.center_y, "float"),
        ]
    
    def set_config_options(self, options):
        if "zoom_factor" in options:
            self.zoom_factor = float(options["zoom_factor"])
        if "center_x" in options:
            self.center_x = float(options["center_x"])
        if "center_y" in options:
            self.center_y = float(options["center_y"])