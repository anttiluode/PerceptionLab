import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class TemporalBlurNode(BaseNode):
    """
    Blurs images over time by blending the current frame
    with a memory of the previous frame. Creates "ghosting"
    or "motion blur" trails.
    """
    NODE_CATEGORY = "Image"
    NODE_COLOR = QtGui.QColor(180, 100, 180) # Magenta

    def __init__(self, feedback=0.90):
        super().__init__()
        self.node_title = "Temporal Blur (Ghosting)"
        
        # --- Inputs and Outputs ---
        self.inputs = {'image_in': 'image', 'reset': 'signal'}
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable ---
        # feedback: How much of the PREVIOUS frame to keep (0.0 to 1.0)
        # High value (0.9) = long, blurry trails
        # Low value (0.1) = short, choppy trails
        self.feedback = float(feedback)
        
        # --- Internal State ---
        self.memory_buffer = None

    def get_config_options(self):
        """Returns options for the right-click config dialog."""
        return [
            ("Feedback (0.1-0.99)", "feedback", self.feedback, None),
        ]

    def set_config_options(self, options):
        """Receives a dictionary from the config dialog."""
        if "feedback" in options:
            # Clamp value to be safe
            self.feedback = np.clip(float(options["feedback"]), 0.1, 0.99)
        
    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        reset_signal = self.get_blended_input('reset', 'sum')

        if reset_signal is not None and reset_signal > 0:
            self.memory_buffer = None # Clear the memory
            return

        if img_in is None:
            return # Nothing to process

        # --- Initialize buffer on first run or after reset ---
        if self.memory_buffer is None:
            self.memory_buffer = img_in.copy()
            return
            
        # --- Ensure buffer and input shapes match ---
        try:
            if self.memory_buffer.shape != img_in.shape:
                # Resize input to match memory (e.g., if resolution changed)
                h, w = self.memory_buffer.shape[:2]
                img_in = cv2.resize(img_in, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Ensure 3-channel if one is 3-channel
            if self.memory_buffer.ndim == 2 and img_in.ndim == 3:
                self.memory_buffer = cv2.cvtColor(self.memory_buffer, cv2.COLOR_GRAY2BGR)
            if img_in.ndim == 2 and self.memory_buffer.ndim == 3:
                img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)

        except Exception as e:
            print(f"TemporalBlurNode resize error: {e}")
            self.memory_buffer = img_in.copy() # Fallback
            return

        # --- The Blur Logic (Feedback) ---
        # output = (old_frame * feedback) + (new_frame * (1.0 - feedback))
        
        self.memory_buffer = (self.memory_buffer * self.feedback) + (img_in * (1.0 - self.feedback))
        
        # We don't need to clip, as feedback+(1-feedback) = 1.0, 
        # so values should stay in 0-1 range.
        # self.memory_buffer = np.clip(self.memory_buffer, 0, 1)

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.memory_buffer
        return None

    def get_display_image(self):
        return self.memory_buffer