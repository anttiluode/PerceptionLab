"""
Rotating Moiré Interference Node
---------------------------------
Generates 2D moiré patterns with ROTATING coordinate systems.

Each wave pattern rotates independently, creating spinning interference.
Rotation can be driven by:
1. Signal inputs (real-time control from EEG, etc.)
2. Base rotation speeds (auto-rotation)

When frequencies beat AND coordinate systems spin = dynamic topology.
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class RotatingMoireNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 180, 220)  # Spinning Teal
    
    def __init__(self, 
                 size=128, 
                 base_speed_1=0.0,
                 base_speed_2=0.0,
                 freq_scale_1=20.0,
                 freq_scale_2=20.0):
        super().__init__()
        self.node_title = "Rotating Moiré"
        self.size = int(size)
        
        # Rotation speeds (radians per frame)
        self.base_speed_1 = float(base_speed_1)
        self.base_speed_2 = float(base_speed_2)
        
        # Frequency scales for the wave patterns
        self.freq_scale_1 = float(freq_scale_1)
        self.freq_scale_2 = float(freq_scale_2)
        
        # Current rotation angles (accumulated)
        self.rotation_angle_1 = 0.0
        self.rotation_angle_2 = 0.0
        
        self.inputs = {
            'freq_1': 'signal',         # Frequency of pattern 1
            'freq_2': 'signal',         # Frequency of pattern 2
            'rotation_1': 'signal',     # Rotation control for pattern 1
            'rotation_2': 'signal',     # Rotation control for pattern 2
            'speed_override_1': 'signal',  # Override base speed
            'speed_override_2': 'signal',  # Override base speed
        }
        self.outputs = {
            'image': 'image',
            'rotation_1_out': 'signal',  # Current rotation angle 1
            'rotation_2_out': 'signal',  # Current rotation angle 2
        }
        
        # Pre-calculate coordinate grids
        self._init_grids()
        self.output_image = np.zeros((self.size, self.size), dtype=np.float32)

    def _init_grids(self):
        """Creates normalized coordinate grids [-1, 1] centered at origin"""
        if self.size == 0: 
            self.size = 1
        
        # Create grids from -1 to 1
        u_vec = np.linspace(-1, 1, self.size, dtype=np.float32)
        v_vec = np.linspace(-1, 1, self.size, dtype=np.float32)
        
        # V (rows, vertical), U (cols, horizontal)
        self.U_base, self.V_base = np.meshgrid(u_vec, v_vec)
        self.output_image = np.zeros((self.size, self.size), dtype=np.float32)

    def _rotate_coords(self, U, V, angle):
        """Rotate coordinate system by angle (radians)"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        U_rot = U * cos_a - V * sin_a
        V_rot = U * sin_a + V * cos_a
        
        return U_rot, V_rot

    def step(self):
        # Check if size changed
        if self.U_base.shape[0] != self.size:
            self._init_grids()
        
        # 1. Get frequency inputs (map to frequency range)
        freq_1 = ((self.get_blended_input('freq_1', 'sum') or 0.0) + 1.0) * 0.5 * self.freq_scale_1
        freq_2 = ((self.get_blended_input('freq_2', 'sum') or 0.0) + 1.0) * 0.5 * self.freq_scale_2
        
        # 2. Get rotation control inputs
        rot_control_1 = self.get_blended_input('rotation_1', 'sum')
        rot_control_2 = self.get_blended_input('rotation_2', 'sum')
        
        # 3. Get speed overrides
        speed_override_1 = self.get_blended_input('speed_override_1', 'sum')
        speed_override_2 = self.get_blended_input('speed_override_2', 'sum')
        
        # 4. Calculate rotation increments
        # If rotation control is provided, use it directly
        # Otherwise, use base speed (optionally overridden)
        if rot_control_1 is not None:
            # Direct angle control (signal controls absolute angle)
            self.rotation_angle_1 = rot_control_1 * np.pi  # Map [-1,1] to [-pi,pi]
        else:
            # Auto-rotation at base speed (or override speed)
            if speed_override_1 is not None:
                speed = speed_override_1 * 0.1  # Scale the override
            else:
                speed = self.base_speed_1
            self.rotation_angle_1 += speed
        
        if rot_control_2 is not None:
            self.rotation_angle_2 = rot_control_2 * np.pi
        else:
            if speed_override_2 is not None:
                speed = speed_override_2 * 0.1
            else:
                speed = self.base_speed_2
            self.rotation_angle_2 += speed
        
        # Keep angles in reasonable range
        self.rotation_angle_1 = self.rotation_angle_1 % (2 * np.pi)
        self.rotation_angle_2 = self.rotation_angle_2 % (2 * np.pi)
        
        # 5. Rotate coordinate systems
        U1, V1 = self._rotate_coords(self.U_base, self.V_base, self.rotation_angle_1)
        U2, V2 = self._rotate_coords(self.U_base, self.V_base, self.rotation_angle_2)
        
        # 6. Generate wave patterns in rotated coordinates
        # Use radial distance for more interesting patterns
        field1 = np.sin(U1 * freq_1 * np.pi)
        field2 = np.cos(V2 * freq_2 * np.pi)
        
        # 7. Interference pattern
        moire_value = np.cos(field1 * np.pi - field2 * np.pi)
        
        # 8. Normalize to [0, 1]
        self.output_image = (moire_value + 1.0) / 2.0

    def get_output(self, port_name):
        if port_name == 'image':
            return self.output_image
        elif port_name == 'rotation_1_out':
            return float(self.rotation_angle_1 / np.pi)  # Normalize to [-1, 1] range
        elif port_name == 'rotation_2_out':
            return float(self.rotation_angle_2 / np.pi)
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.output_image, 0, 1) * 255).astype(np.uint8)
        
        # Add rotation angle indicators
        img_color = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        
        # Draw rotation indicators as small arrows
        center = self.size // 2
        radius = min(20, self.size // 10)
        
        # Arrow for rotation 1 (red)
        x1 = int(center + radius * np.cos(self.rotation_angle_1))
        y1 = int(center + radius * np.sin(self.rotation_angle_1))
        cv2.arrowedLine(img_color, (center, center), (x1, y1), (0, 0, 255), 1, tipLength=0.3)
        
        # Arrow for rotation 2 (cyan)
        x2 = int(center + radius * np.cos(self.rotation_angle_2))
        y2 = int(center + radius * np.sin(self.rotation_angle_2))
        cv2.arrowedLine(img_color, (center, center), (x2, y2), (255, 255, 0), 1, tipLength=0.3)
        
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, self.size, self.size, 3 * self.size, 
                           QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution", "size", self.size, None),
            ("Base Speed 1 (rad/frame)", "base_speed_1", self.base_speed_1, None),
            ("Base Speed 2 (rad/frame)", "base_speed_2", self.base_speed_2, None),
            ("Freq Scale 1", "freq_scale_1", self.freq_scale_1, None),
            ("Freq Scale 2", "freq_scale_2", self.freq_scale_2, None),
        ]