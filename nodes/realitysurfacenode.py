"""
Reality Surface Node (The Living Landscape)
-------------------------------------------
Visualizes the trajectory of consciousness as a 3D terrain.
X/Y = Phase Space (The Path)
Z   = Entropy (The Complexity/Energy)

This node builds a rolling mesh history of your mental state.
"""

import numpy as np
import cv2

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None
        def set_output(self, name, val): pass

class RealitySurfaceNode(BaseNode):
    NODE_CATEGORY = "Visualizers"
    NODE_COLOR = QtGui.QColor(255, 100, 200) # Neon Pink

    def __init__(self):
        super().__init__()
        self.node_title = "Reality Surface"
        
        self.inputs = {
            'phase_x': 'signal',    # Trajectory X
            'phase_y': 'signal',    # Trajectory Y
            'entropy_z': 'signal'   # Height (Complexity)
        }
        
        self.outputs = {
            'surface_view': 'image',
            'holonomy': 'signal'    # How "twisted" the path is
        }
        
        # State: A rolling buffer of 3D points
        self.history_len = 100
        self.grid_width = 20 # Points per row
        
        # We store points as (x, y, z)
        # Initial flat sheet
        self.points = []
        for i in range(self.history_len):
            self.points.append([0.0, 0.0, 0.0])
            
        self.display = np.zeros((300, 400, 3), dtype=np.uint8)
        self.cam_angle = 0.0

    def step(self):
        # 1. Get Data
        px = self.get_blended_input('phase_x', 'mean') or 0.0
        py = self.get_blended_input('phase_y', 'mean') or 0.0
        ez = self.get_blended_input('entropy_z', 'mean') or 0.0
        
        # 2. Update History (The "Moving Tape")
        # We push a new point to the front
        # Scale inputs to be visible
        new_pt = [px * 50.0, ez * 100.0, py * 50.0] # Map Y to Depth-ish
        
        self.points.pop(0)
        self.points.append(new_pt)
        
        # 3. Calculate Holonomy (Curvature)
        # Simple metric: sum of angles between last 3 vectors
        if len(self.points) > 5:
            p1 = np.array(self.points[-1])
            p2 = np.array(self.points[-3])
            p3 = np.array(self.points[-5])
            v1 = p1 - p2
            v2 = p2 - p3
            # Angle
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm > 0:
                cos_angle = np.dot(v1, v2) / norm
                holonomy = 1.0 - np.clip(cos_angle, -1, 1)
            else:
                holonomy = 0.0
        else:
            holonomy = 0.0
            
        self.set_output('holonomy', float(holonomy))

    def get_display_image(self):
        # 4. Render the Landscape (Wireframe Projection)
        self.display.fill(20) # Dark bg
        
        h, w = self.display.shape[:2]
        cx, cy = w//2, h//2
        
        # Projection Constants
        f = 300.0 # Focal length
        
        # Camera Transform (Slow rotation)
        self.cam_angle += 0.01
        ca = np.cos(0.5) # Fixed view angle for stability
        sa = np.sin(0.5)
        
        # Process points
        proj_points = []
        
        # We draw the history as a "Ribbon" or "Grid"
        # Let's draw it as a flowing ribbon
        for i, pt in enumerate(self.points):
            x, y, z = pt
            
            # Artificial "Depth" progression for history
            # Newer points are closer (z=0), older are deeper (z=200)
            depth_offset = (self.history_len - i) * 5.0
            
            # Rotate world
            # x' = x
            # y' = y*ca - z*sa
            # z' = y*sa + z*ca
            
            # Apply depth offset to Z
            r_x = x 
            r_y = y - 50 # Move down a bit
            r_z = z + depth_offset + 100 # Move away
            
            # Project
            if r_z > 1:
                u = int(cx + (r_x * f) / r_z)
                v = int(cy + (r_y * f) / r_z)
                proj_points.append((u, v))
            else:
                proj_points.append(None)
                
        # Draw Ribbon
        for i in range(len(proj_points) - 1):
            p1 = proj_points[i]
            p2 = proj_points[i+1]
            
            if p1 and p2:
                # Color based on height (Entropy) of the original point
                # Original Y is index 1 in the list [x, y, z] above (mapped from entropy)
                height_val = self.points[i][1] 
                hue = int(120 + height_val) % 180
                
                # Convert HSV to BGR for OpenCV
                # Brightness fades with distance (i is index, low i = old = far)
                fade = int((i / self.history_len) * 255)
                
                color_hsv = np.uint8([[[hue, 255, fade]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0,0]
                color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
                
                # Draw cross-lines to make it look like a grid/ladder
                cv2.line(self.display, p1, p2, color, 2)
                
                # Draw vertical "Stalks" to ground plane to show height
                ground_y = self.points[i][1] * 0 + 50 # Zero height relative
                # Project ground point
                gr_z = self.points[i][2] + (self.history_len - i) * 5.0 + 100
                if gr_z > 1:
                    gu = int(cx + (self.points[i][0] * f) / gr_z)
                    gv = int(cy + ((-50) * f) / gr_z) # Fixed ground y
                    cv2.line(self.display, p1, (gu, gv), (50, 50, 50), 1)

        self.set_output('surface_view', self.display)
        return self.get_output('surface_view') # Return image for display node

    def get_output(self, name):
        if name == 'surface_view': return self.display
        if name == 'holonomy': return 0.0 # Placeholder
        return None
        
    def set_output(self, name, val): pass