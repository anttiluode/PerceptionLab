"""
Image Processor Node (FIXED)
--------------------
A simple utility node to adjust the brightness and contrast of an
incoming image stream.

- 'Brightness' adds or subtracts from all pixel values.
- 'Contrast' multiplies the pixel values relative to the midpoint (0.5).

FIX v2: This version preserves the input data type and dimensions 
(e.g., 2D float) for its 'image_out' port, which fixes compatibility
with nodes that expect a specific format (like Scalogram).

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
        self.processed_image = None # This will hold the format-preserved image
        self.display_in_rgb = np.zeros((64, 64, 3), dtype=np.uint8) # For "Before" display
        self.display_out_rgb = np.zeros((64, 64, 3), dtype=np.uint8) # For "After" display


    def step(self):
        if self._error: return
            
        # --- 1. Get Input Image ---
        img_in = self.get_blended_input('image_in', 'mean')
        
        if img_in is None:
            return
            
        # --- 2. Store original properties ---
        original_dtype = img_in.dtype
        
        # --- 3. Convert to Float (0.0 - 1.0) for processing ---
        if original_dtype == np.uint8:
            img_float = img_in.astype(np.float32) / 255.0
        else:
            # Assumes it's a float array (e.g., from CorticalReconstruction)
            img_float = img_in.astype(np.float32) 
            
        # --- 4. Apply Brightness & Contrast ---
        # Formula: new_val = (old_val - 0.5) * contrast + 0.5 + brightness
        
        # Apply contrast
        processed_float = (img_float - 0.5) * self.contrast + 0.5
        
        # Apply brightness
        processed_float = processed_float + (self.brightness / 100.0) # Brightness as -100 to 100
        
        # Clip values to valid 0.0 - 1.0 range
        np.clip(processed_float, 0.0, 1.0, out=processed_float)
        
        # --- 5. Convert back to original format for OUTPUT port ---
        if original_dtype == np.uint8:
            self.processed_image = (processed_float * 255).astype(np.uint8)
        else:
            # IMPORTANT: Keep it as float if it came in as float
            self.processed_image = processed_float.astype(original_dtype)
            
        # --- 6. Create separate uint8 RGB versions for DISPLAY ---
        
        # Create "Before" display
        if img_float.ndim == 2:
            before_u8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
            self.display_in_rgb = cv2.cvtColor(before_u8, cv2.COLOR_GRAY2RGB)
        elif img_float.shape[2] == 3:
            before_u8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
            self.display_in_rgb = before_u8
        
        # Create "After" display
        if processed_float.ndim == 2:
            after_u8 = (np.clip(processed_float, 0, 1) * 255).astype(np.uint8)
            self.display_out_rgb = cv2.cvtColor(after_u8, cv2.COLOR_GRAY2RGB)
        elif processed_float.shape[2] == 3:
            after_u8 = (np.clip(processed_float, 0, 1) * 255).astype(np.uint8)
            self.display_out_rgb = after_u8
        
        
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
        if self.processed_image is None: return None

        # Create a side-by-side "Before" and "After"
        display_h = 128
        display_w = 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # --- Left side: "Before" (Input) ---
        before_resized = cv2.resize(self.display_in_rgb, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        display[:, :display_h] = before_resized
        
        # --- Right side: "After" (Processed Output) ---
        after_resized = cv2.resize(self.display_out_rgb, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
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
