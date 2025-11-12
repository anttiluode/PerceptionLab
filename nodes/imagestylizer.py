import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class ImageStylizerNode(BaseNode):
    """
    Applies an artistic filter (like painting or pencil sketch) to an image.
    """
    NODE_CATEGORY = "Image"
    NODE_COLOR = QtGui.QColor(100, 180, 180) # Cyan-ish

    def __init__(self, mode='Oil Painting'):
        super().__init__()
        self.node_title = "Image Stylizer"
        
        # --- Inputs and Outputs ---
        self.inputs = {'image_in': 'image'}
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable Modes ---
        self.modes = ['Oil Painting', 'Pencil Sketch (Color)', 'Pencil Sketch (Gray)']
        self.mode = mode if mode in self.modes else self.modes[0]
        
        # --- Internal State ---
        self.stylized_image = np.zeros((64, 64, 3), dtype=np.float32)

    def get_config_options(self):
        """
        Returns options for the right-click config dialog.
        Format: (display_name, key, current_value, options_list)
        """
        # options_list is a list of (display_name, value) tuples
        options_list = [(mode, mode) for mode in self.modes]
        
        return [
            ("Style Mode", "mode", self.mode, options_list)
        ]

    def set_config_options(self, options):
        """
        Receives a dictionary from the config dialog: {'mode': 'New Mode'}
        """
        if "mode" in options:
            self.mode = options["mode"]

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is None:
            return

        # 1. Convert from (0-1 float) to (0-255 uint8) for OpenCV
        try:
            img_u8 = (np.clip(img_in, 0, 1) * 255).astype(np.uint8)
            
            # Ensure 3-channel BGR
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            elif img_u8.shape[2] == 4:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGR)
            
            img_u8 = np.ascontiguousarray(img_u8)
        except Exception as e:
            print(f"Stylizer input conversion error: {e}")
            self.stylized_image = img_in # Pass through on error
            return

        # 2. Apply selected style
        try:
            if self.mode == 'Oil Painting':
                # Uses cv2.stylization for a painting-like effect
                stylized = cv2.stylization(img_u8, sigma_s=60, sigma_r=0.45)
            
            elif self.mode == 'Pencil Sketch (Color)':
                # cv2.pencilSketch returns two images: grayscale and color
                _gray_sketch, stylized = cv2.pencilSketch(img_u8, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            
            elif self.mode == 'Pencil Sketch (Gray)':
                # We take the grayscale output here
                stylized, _color_sketch = cv2.pencilSketch(img_u8, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            
            else:
                stylized = img_u8 # Default case, just pass through

            # 3. Convert back to (0-1 float) for the node pipeline
            
            # If we got a 2D grayscale image, convert it back to 3D
            if stylized.ndim == 2:
                stylized = cv2.cvtColor(stylized, cv2.COLOR_GRAY2BGR)
            
            self.stylized_image = (stylized.astype(np.float32) / 255.0)

        except Exception as e:
            print(f"ImageStylizerNode CV error: {e}")
            self.stylized_image = img_in # Fallback to original

    def get_output(self, port_name):
        """
        This is the "pull" method called by the host.
        """
        if port_name == 'image_out':
            return self.stylized_image
        return None

    def get_display_image(self):
        """
        Returns a QImage for the node's internal display.
        """
        if self.stylized_image is None or self.stylized_image.size == 0:
            return None
        
        # Convert 0-1 float to 0-255 uint8
        img_u8 = (np.clip(self.stylized_image, 0, 1) * 255).astype(np.uint8)
        
        # Resize for a standard 96x96 preview
        img_resized = cv2.resize(img_u8, (96, 96), interpolation=cv2.INTER_NEAREST)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        channels = img_resized.shape[2] if img_resized.ndim == 3 else 1
        
        if channels == 3:
            # Create QImage from 24-bit RGB data
            return QtGui.QImage(img_resized.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        else:
            # Create QImage from 8-bit Grayscale data
            return QtGui.QImage(img_resized.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)