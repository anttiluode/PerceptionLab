"""
Phase Drift Node
----------------
Adds rotation and radial drift to transform static eigenmodes
into the flowing tunnel/spiral hallucination patterns.

Static hexagon → vertical stripes in cortical view
Rotating hexagon → diagonal stripes (SPIRAL)
Radially drifting → horizontal flow (TUNNEL)
Both → the full psychedelic experience
"""

import numpy as np
import cv2

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class PhaseDriftNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_TITLE = "Phase Drift (Tunnel/Spiral)"
    NODE_COLOR = QtGui.QColor(255, 150, 50)  # Orange
    
    def __init__(self, rotation_speed=1.0, radial_speed=0.0, spiral_twist=0.0):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',           # The eigenmode (star)
            'rotation_mod': 'signal',       # Modulate rotation speed
            'radial_mod': 'signal',         # Modulate radial drift
            'reset': 'signal'
        }
        
        self.outputs = {
            'image_out': 'image',           # Drifting pattern
            'cortical_view': 'image',       # Built-in log-polar transform
            'phase_angle': 'signal'         # Current rotation phase
        }
        
        # Drift parameters
        self.rotation_speed = float(rotation_speed)  # degrees per frame
        self.radial_speed = float(radial_speed)      # pixels per frame (log scale)
        self.spiral_twist = float(spiral_twist)      # couples rotation to radius
        
        # State
        self.current_angle = 0.0
        self.current_radial_offset = 0.0
        self.frame_count = 0
        
        # Cached outputs
        self.last_output = None
        self.last_cortical = None
        
    def apply_rotation(self, img, angle):
        """Rotate image around center"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_WRAP)
    
    def apply_radial_drift(self, img, offset):
        """Shift pattern radially (zoom in/out effect)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Convert to polar
        dx = x - center[0]
        dy = y - center[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Apply radial offset (in log space for proper scaling)
        r_new = r * np.exp(offset * 0.01)
        
        # Convert back to Cartesian
        x_new = (center[0] + r_new * np.cos(theta)).astype(np.float32)
        y_new = (center[1] + r_new * np.sin(theta)).astype(np.float32)
        
        # Remap
        return cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    def apply_spiral_twist(self, img, twist_amount):
        """Apply radius-dependent rotation (creates spiral)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        y, x = np.ogrid[:h, :w]
        dx = x - center[0]
        dy = y - center[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Twist angle depends on radius
        theta_new = theta + twist_amount * r * 0.001
        
        x_new = (center[0] + r * np.cos(theta_new)).astype(np.float32)
        y_new = (center[1] + r * np.sin(theta_new)).astype(np.float32)
        
        return cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def log_polar_transform(self, img):
        """Convert to cortical coordinates"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        max_radius = min(center[0], center[1])
        
        # Ensure uint8
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
        cortical = cv2.warpPolar(img, (w, h), center, max_radius, flags)
        cortical = cv2.rotate(cortical, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return cortical

    def step(self):
        img = self.get_blended_input('image_in', 'first')
        rot_mod = self.get_blended_input('rotation_mod', 'sum') or 0.0
        rad_mod = self.get_blended_input('radial_mod', 'sum') or 0.0
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.current_angle = 0.0
            self.current_radial_offset = 0.0
            self.frame_count = 0
            return
            
        if img is None:
            return
        
        # Ensure float 0-1
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Update drift state
        effective_rotation = self.rotation_speed * (1.0 + rot_mod)
        effective_radial = self.radial_speed * (1.0 + rad_mod)
        
        self.current_angle += effective_rotation
        self.current_radial_offset += effective_radial
        self.frame_count += 1
        
        # Keep angle in bounds
        self.current_angle = self.current_angle % 360.0
        
        # Apply transforms
        result = img.copy()
        
        # 1. Apply spiral twist (radius-dependent rotation)
        if abs(self.spiral_twist) > 0.001:
            result = self.apply_spiral_twist(result, self.spiral_twist * self.frame_count)
        
        # 2. Apply rotation
        if abs(self.current_angle) > 0.001:
            result = self.apply_rotation(result, self.current_angle)
        
        # 3. Apply radial drift
        if abs(self.current_radial_offset) > 0.001:
            result = self.apply_radial_drift(result, self.current_radial_offset)
        
        self.last_output = result
        
        # Generate cortical view
        self.last_cortical = self.log_polar_transform(result)

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.last_output
        elif port_name == 'cortical_view':
            if self.last_cortical is not None:
                return self.last_cortical.astype(np.float32) / 255.0
            return None
        elif port_name == 'phase_angle':
            return self.current_angle
        return None

    def get_display_image(self):
        """Side-by-side: drifting pattern and cortical view"""
        if self.last_output is None or self.last_cortical is None:
            return None
        
        # Prepare left panel (drifting eigenmode)
        left = self.last_output
        if left.dtype != np.uint8:
            left = (np.clip(left, 0, 1) * 255).astype(np.uint8)
        left = cv2.resize(left, (128, 128))
        left_color = cv2.applyColorMap(left, cv2.COLORMAP_JET)
        
        # Prepare right panel (cortical view)
        right = self.last_cortical
        if right.dtype != np.uint8:
            right = (np.clip(right, 0, 1) * 255).astype(np.uint8)
        right = cv2.resize(right, (128, 128))
        right_color = cv2.applyColorMap(right, cv2.COLORMAP_INFERNO)
        
        # Combine
        combined = np.hstack((left_color, right_color))
        
        # Labels
        cv2.putText(combined, "Retinal", (10, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, "Cortical", (138, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, f"Rot: {self.current_angle:.1f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        return QtGui.QImage(combined.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Rotation Speed (deg/frame)", "rotation_speed", self.rotation_speed, None),
            ("Radial Drift Speed", "radial_speed", self.radial_speed, None),
            ("Spiral Twist", "spiral_twist", self.spiral_twist, None),
        ]
