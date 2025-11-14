"""
EphapticPerturbationNode (v1.1 - Crash Fixed)
-----------------------------------------------------------------
Ephaptic fields don't transmit information. They gently DEFORM the
fractal structure of the noise field, like wind on water.

v1.1: Fixed AttributeError by correcting the internal
      'perturbed_field_output' variable name.
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

    def __init__(self, perturbation_strength=0.3, spatial_scale=32.0, temporal_smoothing=0.8, motion_sensitivity=1.0):
        super().__init__()
        self.node_title = "Ephaptic Perturbation"

        self.inputs = {
            'source_image': 'image',      # Webcam or other real-world input
            'noise_field': 'image',       # Base fractal field to perturb
            'modulation': 'signal',       # Optional scalar modulation
        }

        self.outputs = {
            'perturbed_field': 'image',   # The "steered" field
        }

        # Configurable parameters
        self.perturbation_strength = float(perturbation_strength)
        self.spatial_scale = float(spatial_scale)
        self.temporal_smoothing = float(temporal_smoothing)
        self.motion_sensitivity = float(motion_sensitivity)

        # Internal state
        self.prev_gray = None
        self.flow_field = None
        self.deformation_strength_value = 0.0
        
        # --- THIS IS THE FIX ---
        # Initialize the output variable that was missing
        self.perturbed_field_output = None 
        # ---------------------

    def _calculate_optical_flow(self, frame):
        """Calculates dense optical flow"""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # --- FIX: ALWAYS RESIZE THE NEW FRAME ---
        # The new frame ('gray') must be resized to the target grid size
        # *before* it's compared to the previous frame.
        if gray.shape[0] != self.grid_size or gray.shape[1] != self.grid_size:
            gray = cv2.resize(gray, (self.grid_size, self.grid_size), 
                             interpolation=cv2.INTER_AREA)
        # --- END FIX ---
        
        if self.prev_gray is None:
            # 'gray' is already resized, so this is safe
            self.prev_gray = gray 
            self.flow_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
            return

        # --- FIX: THIS CHECK IS NO LONGER NEEDED AND WAS THE BUG ---
        # The old, problematic 'if' block is removed from here.
        # --- END FIX ---
             
        # Farneback Optical Flow
        # Now, self.prev_gray and gray are GUARANTEED to be the same size.
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Update prev_gray for the *next* frame
        self.prev_gray = gray
        
        # Smooth the flow field
        self.flow_field = (self.flow_field * self.temporal_smoothing) + (flow * (1.0 - self.temporal_smoothing))

    def _warp_field(self, field, flow, strength):
        """Warps the noise field based on the optical flow"""
        h, w = field.shape
        
        # Create a mapping grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # Apply the flow field as a perturbation
        map_x = grid_x + flow[:, :, 0] * strength
        map_y = grid_y + flow[:, :, 1] * strength
        
        # Remap the field
        perturbed = cv2.remap(field, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return perturbed

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
        else:
            # If no source, just have a gentle random drift
            if self.flow_field is None:
                self.flow_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
            self.flow_field += (np.random.randn(self.grid_size, self.grid_size, 2) * 0.1)
            self.flow_field *= self.temporal_smoothing
            self.deformation_strength_value = 0.0

        # 3. Apply perturbation
        # Use modulation signal if present, otherwise use internal value
        strength = modulation if modulation is not None else self.deformation_strength_value
        strength *= self.perturbation_strength # Scale by main knob
        
        perturbed_field = self._warp_field(noise_field, self.flow_field, strength)
        
        # --- THIS IS THE FIX ---
        # Save the result to the *correct* class variable
        self.perturbed_field_output = perturbed_field
        # ---------------------

    def get_output(self, port_name):
        if port_name == 'perturbed_field':
            # This line will no longer crash
            return self.perturbed_field_output
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
        
        # Top-right: Flow Field
        if self.flow_field is not None:
            mag, ang = cv2.cartToPolar(self.flow_field[..., 0], self.flow_field[..., 1])
            hsv = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_resized = cv2.resize(flow_rgb, (display_w // 2, display_h // 2))
            display[:display_h//2, display_w//2:] = flow_resized
        
        # Bottom: Perturbed Field Output
        # This line will no longer crash
        if hasattr(self, 'perturbed_field_output') and self.perturbed_field_output is not None:
            perturbed_u8 = (np.clip(self.perturbed_field_output, 0, 1) * 255).astype(np.uint8)
            perturbed_color = cv2.applyColorMap(perturbed_u8, cv2.COLORMAP_VIRIDIS)
            perturbed_resized = cv2.resize(perturbed_color, (display_w, display_h // 2))
            display[display_h//2:, :] = perturbed_resized
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'SOURCE + FLOW', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'PERTURBATION', (display_w//2 + 10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
            ("Motion Sensitivity", "motion_sensitivity", self.motion_sensitivity, None)
        ]