"""
EphapticPerturbationNode (v1.3 - Fixed Remap Crash)
-----------------------------------------------------------------
Ephaptic fields don't transmit information. They gently DEFORM the
fractal structure of the noise field, like wind on water.

v1.3: Added explicit float32 casting to 'map_x' and 'map_y' to prevent
      OpenCV assertion failures when inputs are float64.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class EphapticPerturbationNode(BaseNode):
    NODE_CATEGORY = "Fractal Substrate"
    NODE_COLOR = QtGui.QColor(50, 150, 150)  # Teal wave

    def __init__(self, perturbation_strength=0.3, spatial_scale=32.0, temporal_smoothing=0.8, motion_sensitivity=1.0, flow_blend=0.6):
        super().__init__()
        self.node_title = "Ephaptic Perturbation"

        self.inputs = {
            'source_image': 'image',      # Webcam or other real-world input
            'noise_field': 'image',       # Base fractal field to perturb
            'modulation': 'signal',       # Optional scalar modulation
        }

        self.outputs = {
            'perturbed_field': 'image',        # The "steered" field
            'flow_visualization': 'image',     # Webcam + flow overlay (church glass window!)
        }

        # Configurable parameters
        self.perturbation_strength = float(perturbation_strength)
        self.spatial_scale = float(spatial_scale)
        self.temporal_smoothing = float(temporal_smoothing)
        self.motion_sensitivity = float(motion_sensitivity)
        self.flow_blend = float(flow_blend)  # How much flow vs webcam in visualization

        # Internal state
        self.prev_gray = None
        self.flow_field = None
        self.deformation_strength_value = 0.0
        self.perturbed_field_output = None 
        self.flow_viz_output = None  # The beautiful window
        self.grid_size = 256 # Default safety

    def _calculate_optical_flow(self, frame):
        """Calculates dense optical flow"""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Resize to grid size
        if gray.shape[0] != self.grid_size or gray.shape[1] != self.grid_size:
            gray = cv2.resize(gray, (self.grid_size, self.grid_size), 
                             interpolation=cv2.INTER_AREA)
        
        if self.prev_gray is None:
            self.prev_gray = gray 
            self.flow_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
            return
             
        # Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        
        # Smooth the flow field
        self.flow_field = (self.flow_field * self.temporal_smoothing) + (flow * (1.0 - self.temporal_smoothing))

    def _warp_field(self, field, flow, strength):
        """Warps the noise field based on the optical flow"""
        h, w = field.shape
        
        # Create a mapping grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Force float32 for grid
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # Apply the flow field as a perturbation
        # [FIX] Force result to float32. 
        # Python math might promote this to float64 if strength is a double, which crashes cv2.remap
        map_x = (grid_x + flow[:, :, 0] * strength).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1] * strength).astype(np.float32)
        
        # Remap the field
        # cv2.remap REQUIRES map1 and map2 to be CV_32FC1 (float32)
        perturbed = cv2.remap(field, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return perturbed

    def _generate_flow_visualization(self, source_image):
        """Generate the beautiful church glass window effect"""
        if source_image is None or self.flow_field is None:
            return None
        
        # Ensure source is uint8 BGR
        if source_image.dtype != np.uint8:
            source_u8 = (np.clip(source_image, 0, 1) * 255).astype(np.uint8)
        else:
            source_u8 = source_image
        
        if source_u8.ndim == 2:
            source_u8 = cv2.cvtColor(source_u8, cv2.COLOR_GRAY2BGR)
        
        # Resize to match flow field
        if source_u8.shape[0] != self.grid_size or source_u8.shape[1] != self.grid_size:
            source_u8 = cv2.resize(source_u8, (self.grid_size, self.grid_size))
        
        # Convert flow to HSV colors
        mag, ang = cv2.cartToPolar(self.flow_field[:, :, 0], self.flow_field[:, :, 1])
        hsv = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        hsv[:, :, 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Blend source + flow (THE CHURCH GLASS EFFECT)
        blended = cv2.addWeighted(source_u8, 1.0 - self.flow_blend, flow_color, self.flow_blend, 0)
        
        return blended

    def step(self):
        # 1. Get inputs
        source_image = self.get_blended_input('source_image', 'first')
        noise_field = self.get_blended_input('noise_field', 'first')
        modulation = self.get_blended_input('modulation', 'sum')
        
        if noise_field is None:
            if self.perturbed_field_output is not None:
                self.perturbed_field_output *= 0.95 # Fade out
            return
            
        self.grid_size = noise_field.shape[0]

        # 2. Calculate perturbation (e.g., from webcam motion)
        if source_image is not None:
            # Convert to 0-255 uint8 if it's not
            if source_image.dtype != np.uint8:
                source_image = (np.clip(source_image, 0, 1) * 255).astype(np.uint8)
                
            self._calculate_optical_flow(source_image)
            
            # Use flow magnitude as deformation strength
            self.deformation_strength_value = np.mean(np.linalg.norm(self.flow_field, axis=2)) * self.motion_sensitivity
            
            # Generate the beautiful visualization
            self.flow_viz_output = self._generate_flow_visualization(source_image)
        else:
            # If no source, just have a gentle random drift
            if self.flow_field is None:
                self.flow_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
            self.flow_field += (np.random.randn(self.grid_size, self.grid_size, 2) * 0.1).astype(np.float32)
            self.flow_field *= self.temporal_smoothing
            self.deformation_strength_value = 0.0
            self.flow_viz_output = None

        # 3. Apply perturbation
        # Use modulation signal if present, otherwise use internal value
        strength = modulation if modulation is not None else self.deformation_strength_value
        strength = float(strength) * self.perturbation_strength # Ensure float
        
        perturbed_field = self._warp_field(noise_field, self.flow_field, strength)
        self.perturbed_field_output = perturbed_field

    def get_output(self, port_name):
        if port_name == 'perturbed_field':
            return self.perturbed_field_output
        elif port_name == 'flow_visualization':
            # Return as 0-1 float for other nodes
            if self.flow_viz_output is not None:
                return self.flow_viz_output.astype(np.float32) / 255.0
            return None
        return None

    def get_display_image(self):
        display_w, display_h = 256, 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Top-left: Source Image (if available)
        source_image = self.get_blended_input('source_image', 'first')
        if source_image is not None:
            if source_image.dtype != np.uint8:
                source_image_u8 = (np.clip(source_image, 0, 1) * 255).astype(np.uint8)
            else:
                source_image_u8 = source_image
            
            if source_image_u8.ndim == 2:
                source_image_u8 = cv2.cvtColor(source_image_u8, cv2.COLOR_GRAY2BGR)
                
            source_resized = cv2.resize(source_image_u8, (display_w // 2, display_h // 2))
            display[:display_h//2, :display_w//2] = source_resized
        
        # Top-right: Flow Visualization (THE CHURCH GLASS WINDOW)
        if self.flow_viz_output is not None:
            flow_viz_resized = cv2.resize(self.flow_viz_output, (display_w // 2, display_h // 2))
            display[:display_h//2, display_w//2:] = flow_viz_resized
        
        # Bottom: Perturbed Field Output
        if hasattr(self, 'perturbed_field_output') and self.perturbed_field_output is not None:
            perturbed_u8 = (np.clip(self.perturbed_field_output, 0, 1) * 255).astype(np.uint8)
            perturbed_color = cv2.applyColorMap(perturbed_u8, cv2.COLORMAP_VIRIDIS)
            perturbed_resized = cv2.resize(perturbed_color, (display_w, display_h // 2))
            display[display_h//2:, :] = perturbed_resized
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'SOURCE', (10, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'FLOW VIZ', (display_w//2 + 10, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'PERTURBED FIELD', (10, display_h//2 + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f'Deformation: {self.deformation_strength_value:.4f}', 
                   (10, display_h - 10), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, display_w * 3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Perturbation Strength", "perturbation_strength", self.perturbation_strength, None),
            ("Spatial Scale", "spatial_scale", self.spatial_scale, None),
            ("Temporal Smoothing", "temporal_smoothing", self.temporal_smoothing, None),
            ("Motion Sensitivity", "motion_sensitivity", self.motion_sensitivity, None),
            ("Flow Blend (Viz)", "flow_blend", self.flow_blend, None),
        ]