"""
Webcam Phase Node - Extracts motion dynamics into 3D phase space coordinates
This is different from FFT - it tracks MOTION VECTORS and converts them to attractor-ready signals.

Inspired by the Neural String Attractor system.
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

class WebcamPhaseNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 140, 180)  # Webcam blue-cyan

    def __init__(self, device_id=0, motion_sensitivity=1.0):
        super().__init__()
        self.node_title = "Webcam Phase"

        self.outputs = {
            'phase_x': 'signal',      # X-axis motion (horizontal)
            'phase_y': 'signal',      # Y-axis motion (vertical)
            'phase_z': 'signal',      # Z-axis (temporal change/energy)
            'motion_image': 'image',  # Visual feedback
            'energy': 'signal'        # Total motion energy
        }

        self.device_id = int(device_id)
        self.motion_sensitivity = float(motion_sensitivity)

        # OpenCV capture
        self.cap = None
        self.previous_frame = None
        self.previous_gray = None

        # Motion history buffer (for temporal phase Z)
        self.motion_history = np.zeros(30, dtype=np.float32)
        self.history_idx = 0

        # Phase space coordinates
        self.phase_x = 0.0
        self.phase_y = 0.0
        self.phase_z = 0.0
        self.energy = 0.0

        # Motion visualization
        self.motion_vis = np.zeros((120, 160), dtype=np.uint8)

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=50,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.tracked_points = None

        self.setup_source()

    def setup_source(self):
        """Initialize webcam capture"""
        if self.cap and self.cap.isOpened():
            self.cap.release()

        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.node_title = f"Webcam Phase ({self.device_id})"
            else:
                self.node_title = "Webcam Phase (NO CAM)"
        except Exception as e:
            print(f"Webcam Phase Error: {e}")
            self.cap = None
            self.node_title = "Webcam Phase (ERROR)"

    def step(self):
        if not self.cap or not self.cap.isOpened():
            # Decay outputs if no camera
            self.phase_x *= 0.95
            self.phase_y *= 0.95
            self.phase_z *= 0.95
            self.energy *= 0.95
            return

        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            self.tracked_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return

        # --- OPTICAL FLOW TRACKING ---
        if self.tracked_points is not None and len(self.tracked_points) > 0:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray, self.tracked_points, None, **self.lk_params
            )

            if new_points is not None:
                # Select good points
                good_new = new_points[status == 1]
                good_old = self.tracked_points[status == 1]

                if len(good_new) > 0:
                    # Calculate motion vectors
                    motion_vectors = good_new - good_old

                    # Extract phase coordinates from motion
                    # X: Horizontal motion (average X displacement)
                    self.phase_x = np.mean(motion_vectors[:, 0]) * self.motion_sensitivity

                    # Y: Vertical motion (average Y displacement)
                    self.phase_y = np.mean(motion_vectors[:, 1]) * self.motion_sensitivity

                    # Energy: Magnitude of motion
                    motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
                    self.energy = np.mean(motion_magnitudes) * self.motion_sensitivity * 0.1

                    # Store in history for Z calculation
                    self.motion_history[self.history_idx] = self.energy
                    self.history_idx = (self.history_idx + 1) % len(self.motion_history)

                    # Z: Temporal dynamics (change in energy over time)
                    energy_gradient = np.gradient(self.motion_history)
                    self.phase_z = np.mean(energy_gradient) * 10.0

                    # Clamp to reasonable ranges
                    self.phase_x = np.clip(self.phase_x, -1.0, 1.0)
                    self.phase_y = np.clip(self.phase_y, -1.0, 1.0)
                    self.phase_z = np.clip(self.phase_z, -1.0, 1.0)
                    self.energy = np.clip(self.energy, 0.0, 1.0)

                    # Update tracked points
                    self.tracked_points = good_new.reshape(-1, 1, 2)

                    # Create motion visualization
                    self.create_motion_visualization(gray, good_old, good_new)
                else:
                    # No good points, re-detect
                    self.tracked_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            else:
                # Flow calculation failed, re-detect
                self.tracked_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        else:
            # No points tracked, detect new ones
            self.tracked_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        # Refresh points periodically
        if np.random.rand() < 0.05:  # 5% chance each frame
            self.tracked_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        self.previous_gray = gray

    def create_motion_visualization(self, gray, old_points, new_points):
        """Create a visual representation of motion vectors"""
        # Resize for output
        vis_gray = cv2.resize(gray, (160, 120))

        # Normalize to 0-255
        vis = cv2.normalize(vis_gray, None, 0, 255, cv2.NORM_MINMAX)

        # Draw motion vectors
        scale = 160 / gray.shape[1]  # Scaling factor for coordinates

        for old_pt, new_pt in zip(old_points, new_points):
            old_scaled = (int(old_pt[0] * scale), int(old_pt[1] * scale))
            new_scaled = (int(new_pt[0] * scale), int(new_pt[1] * scale))

            # Draw line
            cv2.arrowedLine(vis, old_scaled, new_scaled, 255, 1, tipLength=0.3)
            # Draw points
            cv2.circle(vis, new_scaled, 2, 255, -1)

        self.motion_vis = vis

    def get_output(self, port_name):
        if port_name == 'phase_x':
            return self.phase_x
        elif port_name == 'phase_y':
            return self.phase_y
        elif port_name == 'phase_z':
            return self.phase_z
        elif port_name == 'energy':
            return self.energy
        elif port_name == 'motion_image':
            return self.motion_vis.astype(np.float32) / 255.0
        return None

    def get_display_image(self):
        # Show motion visualization
        img = np.ascontiguousarray(self.motion_vis)
        h, w = img.shape
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        # Get available cameras
        camera_options = [("Default Camera (0)", 0), ("Secondary (1)", 1), ("Third (2)", 2)]

        return [
            ("Camera Device", "device_id", self.device_id, camera_options),
            ("Motion Sensitivity", "motion_sensitivity", self.motion_sensitivity, None),
        ]