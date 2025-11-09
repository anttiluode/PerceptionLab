"""
Optical Flow Motion Tracker Node

This node ACTUALLY extracts coordinate data from webcam movement.
Uses Lucas-Kanade optical flow to track motion vectors.

Real use cases:
- Gesture control interfaces
- Motion-reactive installations
- Game input via webcam
- Accessibility tools (head tracking for mouse control)
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class OpticalFlowNode(BaseNode):
    """Tracks motion in video and outputs motion vectors as coordinates"""
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(50, 150, 200)
    
    def __init__(self, points_to_track=20, quality_level=0.3, min_distance=7):
        super().__init__()
        self.node_title = "Optical Flow Tracker"
        
        self.inputs = {'image': 'image'}
        self.outputs = {
            'motion_x': 'signal',      # Average horizontal motion
            'motion_y': 'signal',      # Average vertical motion
            'motion_magnitude': 'signal',  # Speed of motion
            'motion_angle': 'signal',  # Direction (-1 to 1, maps to -180 to 180 degrees)
            'flow_vis': 'image',       # Visualization of motion vectors
            'has_motion': 'signal'     # 1.0 if significant motion detected
        }
        
        # Parameters
        self.points_to_track = int(points_to_track)
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        
        # State
        self.prev_gray = None
        self.prev_points = None
        
        # Outputs
        self.motion_x = 0.0
        self.motion_y = 0.0
        self.motion_magnitude = 0.0
        self.motion_angle = 0.0
        self.has_motion = 0.0
        self.flow_vis = None
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=self.points_to_track,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7
        )
        
    def step(self):
        image = self.get_blended_input('image', 'mean')
        
        if image is None:
            return
            
        # Convert to grayscale uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                gray = (image * 255).astype(np.uint8)
            else:
                gray = np.clip(image, 0, 255).astype(np.uint8)
        else:
            gray = image
            
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, 
                mask=None, 
                **self.feature_params
            )
            self.flow_vis = np.zeros((*gray.shape, 3), dtype=np.uint8)
            return
            
        # Calculate optical flow
        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_points,
                None,
                **self.lk_params
            )
            
            # Select good points
            if next_points is not None:
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                if len(good_new) > 0:
                    # Calculate motion vectors
                    motion_vectors = good_new - good_old
                    
                    # Average motion
                    avg_motion = np.mean(motion_vectors, axis=0)
                    self.motion_x = float(avg_motion[0]) / gray.shape[1]  # Normalize by width
                    self.motion_y = float(avg_motion[1]) / gray.shape[0]  # Normalize by height
                    
                    # Motion magnitude (speed)
                    magnitudes = np.linalg.norm(motion_vectors, axis=1)
                    self.motion_magnitude = float(np.mean(magnitudes)) / gray.shape[1]
                    
                    # Motion angle
                    if self.motion_magnitude > 0.001:
                        angle_rad = np.arctan2(self.motion_y, self.motion_x)
                        self.motion_angle = float(angle_rad / np.pi)  # Normalize to -1 to 1
                        self.has_motion = 1.0
                    else:
                        self.motion_angle = 0.0
                        self.has_motion = 0.0
                    
                    # Create visualization
                    self.flow_vis = np.zeros((*gray.shape, 3), dtype=np.uint8)
                    
                    # Draw tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        
                        # Draw line
                        cv2.line(self.flow_vis, (a, b), (c, d), (0, 255, 0), 2)
                        # Draw point
                        cv2.circle(self.flow_vis, (a, b), 3, (0, 0, 255), -1)
                    
                    # Draw average motion vector
                    h, w = gray.shape
                    center = (w // 2, h // 2)
                    end = (
                        int(center[0] + self.motion_x * w * 10),
                        int(center[1] + self.motion_y * h * 10)
                    )
                    cv2.arrowedLine(self.flow_vis, center, end, (255, 0, 0), 3, tipLength=0.3)
                    
                    # Update points for next frame
                    self.prev_points = good_new.reshape(-1, 1, 2)
                else:
                    # No good points, reset
                    self.prev_points = None
                    self.has_motion = 0.0
            else:
                self.prev_points = None
                self.has_motion = 0.0
        
        # Redetect features if we lost tracking
        if self.prev_points is None or len(self.prev_points) < self.points_to_track // 2:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                mask=None,
                **self.feature_params
            )
        
        # Update previous frame
        self.prev_gray = gray
        
    def get_output(self, port_name):
        if port_name == 'motion_x':
            return self.motion_x
        elif port_name == 'motion_y':
            return self.motion_y
        elif port_name == 'motion_magnitude':
            return self.motion_magnitude
        elif port_name == 'motion_angle':
            return self.motion_angle
        elif port_name == 'has_motion':
            return self.has_motion
        elif port_name == 'flow_vis':
            if self.flow_vis is not None:
                return self.flow_vis.astype(np.float32) / 255.0
        return None


class MotionToCoordinatesNode(BaseNode):
    """Converts motion signals to accumulated position coordinates"""
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 150, 50)
    
    def __init__(self, sensitivity=0.5, decay=0.95, bounds=1.0):
        super().__init__()
        self.node_title = "Motion → Coordinates"
        
        self.inputs = {
            'motion_x': 'signal',
            'motion_y': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'x_coord': 'signal',  # Accumulated X position (-1 to 1)
            'y_coord': 'signal',  # Accumulated Y position (-1 to 1)
            'distance_from_center': 'signal',  # 0 to 1
            'normalized_angle': 'signal'  # 0 to 1 (for circular mapping)
        }
        
        self.sensitivity = float(sensitivity)
        self.decay = float(decay)
        self.bounds = float(bounds)
        
        # State
        self.x = 0.0
        self.y = 0.0
        self.last_reset = 0.0
        
    def step(self):
        motion_x = self.get_blended_input('motion_x', 'sum') or 0.0
        motion_y = self.get_blended_input('motion_y', 'sum') or 0.0
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        # Reset on trigger
        if reset > 0.5 and self.last_reset <= 0.5:
            self.x = 0.0
            self.y = 0.0
        self.last_reset = reset
        
        # Accumulate motion with decay
        self.x = self.x * self.decay + motion_x * self.sensitivity
        self.y = self.y * self.decay + motion_y * self.sensitivity
        
        # Clamp to bounds
        self.x = np.clip(self.x, -self.bounds, self.bounds)
        self.y = np.clip(self.y, -self.bounds, self.bounds)
        
    def get_output(self, port_name):
        if port_name == 'x_coord':
            return self.x
        elif port_name == 'y_coord':
            return self.y
        elif port_name == 'distance_from_center':
            return np.sqrt(self.x**2 + self.y**2) / self.bounds
        elif port_name == 'normalized_angle':
            angle = np.arctan2(self.y, self.x)
            return (angle + np.pi) / (2 * np.pi)  # 0 to 1
        return None


"""
COMMERCIAL APPLICATIONS:

1. GESTURE CONTROL:
   Webcam → OpticalFlow → MotionToCoordinates → Control any parameter
   Use case: Hands-free control for music production, VJ software, accessibility

2. HEAD TRACKING MOUSE:
   Webcam → OpticalFlow → Scale motion_x/y → Mouse control
   Use case: Accessibility tool for people with limited hand mobility
   Market: Assistive technology (high willingness to pay)

3. MOTION-REACTIVE ART:
   Webcam → OpticalFlow → Drive fractal params, colors, effects
   Use case: Interactive installations, museums, retail displays
   Market: B2B (museums, experiential marketing)

4. WEBCAM GAME CONTROLLER:
   OpticalFlow → Map to game inputs
   Use case: Alternative controller for rhythm games, casual games
   Market: Gaming accessories

TO USE:
1. Save as OpticalFlowNode.py in your nodes folder
2. Restart Perception Lab
3. Connect webcam → OpticalFlowNode
4. Use motion_x/y to control ANYTHING
5. MotionToCoordinates accumulates motion into position for cursor-like control
"""