"""
PsychedelicWarpNode

Applies a "liquid" sinusoidal warp, color-cycling,
and video feedback to an image. Perfect for that
'melting checkerboard' effect.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class PsychedelicWarpNode(BaseNode):
    """
    Applies a "liquid" psychedelic distortion filter.
    """
    NODE_CATEGORY = "Filter"
    NODE_COLOR = QtGui.QColor(220, 100, 220) # Psychedelic Magenta

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Psychedelic Warp"
        
        self.inputs = {
            'image_in': 'image',
            'warp_speed': 'signal',   # How fast the "liquid" moves
            'warp_strength': 'signal',# How much the image distorts
            'feedback': 'signal',     # 0 (no trails) to 1 (infinite trails)
            'hue_shift': 'signal'     # -1 to 1, speed of color cycling
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        
        # Internal buffer for feedback
        self.buffer = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Internal "time" for warp animation
        self.t = 0.0
        
        # Pre-calculate grids
        y, x = np.mgrid[0:self.size, 0:self.size]
        self.grid_x = x.astype(np.float32)
        self.grid_y = y.astype(np.float32)

    def step(self):
        # --- 1. Get Control Signals ---
        warp_speed = self.get_blended_input('warp_speed', 'sum') or 0.2
        warp_strength = (self.get_blended_input('warp_strength', 'sum') or 0.3) * 50.0
        feedback = self.get_blended_input('feedback', 'sum') or 0.9
        hue_shift = (self.get_blended_input('hue_shift', 'sum') or 0.05) * 10.0
        
        # Clamp feedback to prevent 1.0 (which would block new images)
        feedback_amount = np.clip(feedback, 0.0, 0.98)

        # --- 2. Get and Prepare Input Image ---
        img = self.get_blended_input('image_in', 'first')
        if img is None:
            # If no input, just fade the buffer
            self.buffer *= feedback_amount
            return

        # Resize and format
        img_resized = cv2.resize(img, (self.size, self.size), 
                                 interpolation=cv2.INTER_LINEAR)
        if img_resized.ndim == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        if img_resized.dtype != np.float32:
            img_resized = img_resized.astype(np.float32)
        if img_resized.max() > 1.0:
            img_resized /= 255.0
            
        img_resized = np.clip(img_resized, 0, 1)
        
        # --- 3. Apply Psychedelic Color Shift ---
        # Convert to HSV, shift Hue, convert back
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        
        # Add hue shift (and wrap around 0-180)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180.0
        
        processed_input = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        # --- 4. Create Liquid Warp ---
        self.t += warp_speed * 0.1
        
        # Create a moving, sinusoidal displacement map
        dx = np.sin((self.grid_y / 20.0) + self.t) * warp_strength
        dy = np.cos((self.grid_x / 20.0) + self.t) * warp_strength
        
        map_x = (self.grid_x + dx).astype(np.float32)
        map_y = (self.grid_y + dy).astype(np.float32)
        
        # --- 5. Apply Warp and Feedback ---
        # Warp the *last* frame (the buffer)
        warped_buffer = cv2.remap(
            self.buffer, map_x, map_y, 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # --- 6. Blend ---
        # Blend the warped old frame with the new color-shifted frame
        self.buffer = (warped_buffer * feedback_amount) + \
                     (processed_input * (1.0 - feedback_amount))
        
        self.buffer = np.clip(self.buffer, 0, 1)

    def get_output(self, port_name):
        if port_name == 'image':
            return self.buffer
        return None