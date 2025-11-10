"""
HeightmapFlyerNode (Pseudo-3D "Mode 7" Renderer)

Takes a 2D image as a ground/heightmap and renders it
with a 3D perspective "fly-over" camera.
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class HeightmapFlyerNode(BaseNode):
    """
    Simulates a 3D fly-over of a 2D heightmap image.
    """
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 150, 220) # Blue/Purple

    def __init__(self, size=256):
        super().__init__()
        self.node_title = "Heightmap Flyer"
        
        self.inputs = {
            'image_in': 'image',    # The ground texture
            'pitch': 'signal',      # 0 (top-down) to 1 (max perspective)
            'yaw': 'signal',        # -1 to 1 (rotation)
            'speed_y': 'signal',    # -1 to 1 (forward/back)
            'speed_x': 'signal',    # -1 to 1 (strafe left/right)
            'zoom': 'signal'        # 0 to 1 (altitude/scale)
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.display_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Keep track of our "position" in the world
        self.scroll_x = 0.0
        self.scroll_y = 0.0

    def step(self):
        # --- 1. Get Control Signals ---
        pitch_in = self.get_blended_input('pitch', 'sum') or 0.2
        yaw_in = self.get_blended_input('yaw', 'sum') or 0.0
        speed_y_in = self.get_blended_input('speed_y', 'sum') or 0.0
        speed_x_in = self.get_blended_input('speed_x', 'sum') or 0.0
        zoom_in = self.get_blended_input('zoom', 'sum') or 0.5

        # --- 2. Get and Prepare Image ---
        img = self.get_blended_input('image_in', 'first')
        if img is None:
            # Use a simple checkerboard if no image is connected
            y, x = np.mgrid[0:self.size, 0:self.size]
            check = ((x // 32) + (y // 32)) % 2
            img = np.stack([check] * 3, axis=-1).astype(np.float32)
        
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), 
                             interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Ensure float32 in 0-1 range (fixes potential cvtColor errors)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        img = np.clip(img, 0, 1)
        h, w = self.size, self.size

        # --- 3. Update Camera Position ---
        self.scroll_x += speed_x_in * 5.0
        self.scroll_y += speed_y_in * 5.0
        self.scroll_x %= w
        self.scroll_y %= h

        # --- 4. Build Transformation Matrices ---
        
        # a) Zoom (Altitude) and Translation (X/Y position)
        zoom_val = 1.0 + zoom_in * 2.0 # Scale from 1x to 3x

        # --- START FIX ---
        # We must use 3x3 matrices (homogeneous coords) to combine affine transforms.
        
        # M_scroll_zoom is (3, 3)
        M_scroll_zoom_3x3 = np.float32([
            [zoom_val, 0, self.scroll_x],
            [0, zoom_val, self.scroll_y],
            [0, 0, 1]
        ])
        
        # b) Yaw (Rotation)
        center = (w // 2, h // 2)
        angle_deg = yaw_in * 90.0
        
        # M_yaw_2x3 is (2, 3)
        M_yaw_2x3 = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        # M_yaw_3x3 is (3, 3)
        M_yaw_3x3 = np.vstack([M_yaw_2x3, [0, 0, 1]])
        
        # Combine affine transforms (scroll, zoom, yaw)
        # This is now a (3, 3) @ (3, 3) multiplication
        # The order matters: apply zoom/scroll first, THEN yaw
        M_affine_3x3 = M_yaw_3x3 @ M_scroll_zoom_3x3
        
        # Get the final (2, 3) matrix for warpAffine
        M_affine = M_affine_3x3[0:2, :]
        # --- END FIX ---
        
        # Apply affine transforms
        # BORDER_WRAP makes the world tile infinitely
        pre_transformed = cv2.warpAffine(img, M_affine, (w, h), 
                                         borderMode=cv2.BORDER_WRAP)
        
        # c) Pitch (Perspective)
        pitch_amount = np.clip(pitch_in, 0, 0.9) * (w / 2.2)
        
        src_pts = np.float32([
            [0, 0], [w - 1, 0],
            [w - 1, h - 1], [0, h - 1]
        ])
        
        dst_pts = np.float32([
            [pitch_amount, 0], [w - 1 - pitch_amount, 0],
            [w - 1, h - 1], [0, h - 1]
        ])
        
        M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # --- 5. Apply Final Transform ---
        self.display_image = cv2.warpPerspective(
            pre_transformed, M_perspective, (w, h), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0) # Fill horizon with black
        )

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_image
        return None