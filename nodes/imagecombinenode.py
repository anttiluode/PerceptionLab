import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class ImageCombineNode(BaseNode):
    """
    Combines two images using a selected operation. (v3 - Fixed set_output error)
    """
    NODE_CATEGORY = "Image"
    NODE_COLOR = QtGui.QColor(100, 180, 100) # Image-ops green

    def __init__(self, mode='Average'):
        super().__init__()
        self.node_title = "Image Combiner"
        
        # --- Inputs and Outputs ---
        self.inputs = {
            'image_in_a': 'image',
            'image_in_b': 'image'
        }
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable Mode ---
        self.modes = ['Average', 'Add', 'Subtract', 'Multiply', 'Screen', 'HStack', 'VStack']
        self.mode = mode if mode in self.modes else self.modes[0]
        self.config = {'mode': self.modes.index(self.mode)}
        
        # --- Internal State ---
        self.combined_image = np.zeros((64, 64, 3), dtype=np.float32)

    def get_config_options(self):
        # This creates the dropdown menu
        return {
            "mode": (self.modes, self.mode)
        }

    def set_config_options(self, options):
        if "mode" in options:
            self.mode = options["mode"]
            self.config["mode"] = self.modes.index(self.mode)

    def step(self):
        # --- Get inputs ---
        img_a = self.get_blended_input('image_in_a', 'first')
        img_b = self.get_blended_input('image_in_b', 'first')

        # --- Handle missing inputs ---
        if img_a is None and img_b is None:
            # Nothing to do, internal image remains as is
            return
        if img_a is None:
            self.combined_image = img_b # Pass through B
            return # We are done for this step
        if img_b is None:
            self.combined_image = img_a # Pass through A
            return # We are done for this step

        # --- Pre-processing: Ensure images are compatible ---
        try:
            # 1. Ensure same shape (resize B to match A)
            if img_a.shape != img_b.shape:
                target_h, target_w = img_a.shape[:2]
                img_b = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 2. Ensure same channel count
            if img_a.ndim == 2 and img_b.ndim == 3:
                img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
            if img_b.ndim == 2 and img_a.ndim == 3:
                img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
            if img_a.ndim == 3 and img_b.ndim == 2:
                img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
            if img_b.ndim == 3 and img_a.ndim == 2:
                img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)

        except Exception as e:
            print(f"ImageCombineNode resize/channel error: {e}")
            self.combined_image = img_a # Fallback
            return

        # --- Apply selected combination mode ---
        try:
            if self.mode == 'Average':
                self.combined_image = (img_a * 0.5) + (img_b * 0.5)
            elif self.mode == 'Add':
                self.combined_image = img_a + img_b
            elif self.mode == 'Subtract':
                self.combined_image = img_a - img_b
            elif self.mode == 'Multiply':
                self.combined_image = img_a * img_b
            elif self.mode == 'Screen':
                # Screen blend mode: 1 - (1 - a) * (1 - b)
                self.combined_image = 1.0 - (1.0 - img_a) * (1.0 - img_b)
            elif self.mode == 'HStack':
                self.combined_image = np.hstack((img_a, img_b))
            elif self.mode == 'VStack':
                self.combined_image = np.vstack((img_a, img_b))
                
            # Ensure output is valid
            self.combined_image = np.clip(self.combined_image, 0, 1)

        except Exception as e:
            print(f"ImageCombineNode error: {e}")
            self.combined_image = img_a # Fallback to image A

        # --- NOTE: NO set_output() call here. The step is done. ---

    def get_output(self, port_name):
        """
        This is the "pull" method called by the host.
        """
        if port_name == 'image_out':
            return self.combined_image
        return None

    def get_display_image(self):
        if self.combined_image is None or self.combined_image.size == 0:
            return None
        
        # Create a display-friendly version
        img_u8 = (np.clip(self.combined_image, 0, 1) * 255).astype(np.uint8)
        
        # Handle potentially large stacked images by resizing to a standard display size
        if self.mode in ['HStack', 'VStack']:
            max_dim = 96
            h, w = img_u8.shape[:2]
            if h == 0 or w == 0: return None
            
            if h > w:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            else:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            
            new_w = max(1, new_w) # ensure non-zero
            new_h = max(1, new_h) # ensure non-zero
            
            img_resized = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = cv2.resize(img_u8, (96, 96), interpolation=cv2.INTER_NEAREST)
        
        img_resized = np.ascontiguousarray(img_resized) # Ensure contiguous
        h, w = img_resized.shape[:2]
        channels = img_resized.shape[2] if img_resized.ndim == 3 else 1
        
        if channels == 3:
            return QtGui.QImage(img_resized.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        else: # Grayscale
            return QtGui.QImage(img_resized.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)