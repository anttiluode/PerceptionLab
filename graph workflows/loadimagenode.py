import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
QtWidgets = __main__.QtWidgets # Need this for the file dialog

import numpy as np
import cv2
import os

class LoadImageNode(BaseNode):
    """
    Loads a static image from a file and outputs it as an image signal.
    Includes a "Browse..." button in its config.
    """
    NODE_CATEGORY = "Input"
    NODE_COLOR = QtGui.QColor(180, 150, 80) # Brown-ish

    def __init__(self, file_path=""):
        super().__init__()
        self.node_title = "Load Image"
        
        # --- Inputs and Outputs ---
        self.inputs = {}
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable ---
        self.file_path = file_path
        
        # --- Internal State ---
        self.image_buffer = None
        self._load_image() # Load image on creation

    def get_config_options(self):
        """
        Returns options for the right-click config dialog.
        "file_open" is the special key our new host dialog looks for.
        """
        return [
            ("File Path", "file_path", self.file_path, "file_open"),
        ]

    def set_config_options(self, options):
        """Receives a dictionary from the config dialog."""
        if "file_path" in options:
            self.file_path = options["file_path"]
            self._load_image() # Reload the image when path is set

    def _load_image(self):
        """Internal helper to load and process the image."""
        if not self.file_path or not os.path.exists(self.file_path):
            # Create a placeholder error image
            self.image_buffer = np.zeros((64, 64, 3), dtype=np.float32)
            cv2.putText(self.image_buffer, "NO FILE", (5, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 0, 0), 1)
            return

        try:
            # Load image using OpenCV
            img = cv2.imread(self.file_path)
            
            if img is None:
                raise Exception(f"Failed to read image file: {self.file_path}")
                
            # Convert from BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize from 0-255 (uint8) to 0-1 (float32)
            self.image_buffer = (img.astype(np.float32) / 255.0)
            
        except Exception as e:
            print(f"LoadImageNode Error: {e}")
            self.image_buffer = np.zeros((64, 64, 3), dtype=np.float32)
            cv2.putText(self.image_buffer, "ERROR", (5, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 0, 0), 1)

    def step(self):
        # This node is static, so step() does nothing.
        pass

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.image_buffer
        return None

    def get_display_image(self):
        # Return the loaded buffer for display
        return self.image_buffer