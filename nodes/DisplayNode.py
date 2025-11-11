import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import cv2
import numpy as np

class DisplayNode(BaseNode):
    """
    Displays an image input directly.
    """
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(100, 100, 150)

    def __init__(self):
        super().__init__()
        self.node_title = "Display"
        self.inputs = {'image_in': 'image'}
        self.outputs = {}
        self.image = np.zeros((256, 256, 3), dtype=np.uint8)

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is not None:
            # Convert to 0-255 uint8 BGR
            if img_in.dtype == np.float32 or img_in.dtype == np.float64:
                img = (np.clip(img_in, 0, 1) * 255).astype(np.uint8)
            else:
                img = img_in.astype(np.uint8)

            # Handle grayscale
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Handle RGB
            elif img.shape[2] == 3:
                # Assuming input is RGB, convert to BGR for display
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
                # ^ Let's assume the host handles RGB, if not, uncomment this
                pass
            
            self.image = cv2.resize(img, (256, 256))
        else:
            self.image = (self.image * 0.9).astype(np.uint8) # Fade out

    def get_display_image(self):
        return QtGui.QImage(self.image.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)