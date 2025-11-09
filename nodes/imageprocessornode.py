"""
Image Processor Node
--------------------
A simple utility node to adjust the brightness and contrast of an
incoming image stream.

- 'Brightness' adds or subtracts from all pixel values.
- 'Contrast' multiplies the pixel values relative to the midpoint (0.5).

Place this file in the 'nodes' folder
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

if QtGui is None:
    print("CRITICAL: ImageProcessorNode could not import QtGui from host.")

class ImageProcessorNode(BaseNode):
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(150, 150, 150)  # Neutral Gray
    
    def __init__(self, brightness=0.0, contrast=1.0):
        super().__init__()
        self.node_title = "Image Processor"
        
        self.inputs = {
            'image_in': 'image',
        }
        
        self.outputs = {
            'image_out': 'image',
            'brightness_signal': 'signal',
            'contrast_signal': 'signal'
        }
        
        if QtGui is None:
            self.node_title = "Image Processor (ERROR)"
            self._error = True
            return
        self._error = False
            
        # --- Configurable Parameters ---
        self.brightness = float(brightness)
        self.contrast = float(contrast)

        # --- Internal State ---
        self.processed_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.input_image_buffer = None

    def step(self):
        if self._error: return
            
        # --- 1. Get Input Image ---
        img_in = self.get_blended_input('image_in', 'mean')
        
        if img_in is None:
            # If no input, just hold the last processed image
            return
            
        # --- 2. Convert to Float (0.0 - 1.0) for processing ---
        # We need to handle both grayscale and color images
        if img_in.ndim == 2: # Grayscale
            img_float = img_in.astype(np.float32)
            if img_in.dtype == np.uint8:
                img_float /= 255.0
        elif img_in.ndim == 3 and img_in.shape[2] == 3: # Color (RGB)
            img_float = img_in.astype(np.float32)
            if img_in.dtype == np.uint8:
                img_float /= 255.0
        else: # Unrecognized format
            return
            
        self.input_image_buffer = img_float # Store for display
            
        # --- 3. Apply Brightness & Contrast ---
        # Formula: new_val = (old_val - 0.5) * contrast + 0.5 + brightness
        
        # Apply contrast
        processed_float = (img_float - 0.5) * self.contrast + 0.5
        
        # Apply brightness
        processed_float = processed_float + (self.brightness / 100.0) # Brightness as -100 to 100
        
        # Clip values to valid 0.0 - 1.0 range
        np.clip(processed_float, 0.0, 1.0, out=processed_float)
        
        # --- 4. Convert back to uint8 for output ---
        processed_u8 = (processed_float * 255).astype(np.uint8)
        
        # Ensure output is 3-channel RGB for compatibility
        if processed_u8.ndim == 2:
            self.processed_image = cv2.cvtColor(processed_u8, cv2.COLOR_GRAY2RGB)
        else:
            self.processed_image = processed_u8
        
    def get_output(self, port_name):
        if self._error: return None
        if port_name == 'image_out':
            return self.processed_image
        elif port_name == 'brightness_signal':
            return self.brightness
        elif port_name == 'contrast_signal':
            return self.contrast
        return None

    def get_display_image(self):
        if self._error: return None
        if self.input_image_buffer is None: return None

        # Create a side-by-side "Before" and "After"
        display_h = 128
        display_w = 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # --- Left side: "Before" (Input) ---
        if self.input_image_buffer.ndim == 2:
            before_u8 = (np.clip(self.input_image_buffer, 0, 1) * 255).astype(np.uint8)
            before_img = cv2.cvtColor(before_u8, cv2.COLOR_GRAY2RGB)
        else:
            before_u8 = (np.clip(self.input_image_buffer, 0, 1) * 255).astype(np.uint8)
            before_img = before_u8 # Already RGB
            
        before_resized = cv2.resize(before_img, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        display[:, :display_h] = before_resized
        
        # --- Right side: "After" (Processed Output) ---
        after_resized = cv2.resize(self.processed_image, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        display[:, display_w-display_h:] = after_resized
        
        # Add dividing line
        display[:, display_h-1:display_h+1] = [255, 255, 255]
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'IN', (10, 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'OUT', (display_h + 10, 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Add current values
        b_text = f"B: {self.brightness:.1f}"
        c_text = f"C: {self.contrast:.2f}"
        cv2.putText(display, b_text, (10, display_h - 10), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display, c_text, (display_h + 10, display_h - 10), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        # Config options: [("Display Name", "variable_name", current_value, options_list)]
        # For sliders, options_list is None
        return [
            ("Brightness", "brightness", self.brightness, None),
            ("Contrast", "contrast", self.contrast, None),
        ]