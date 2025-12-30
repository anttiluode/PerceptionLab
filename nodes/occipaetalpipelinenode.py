"""
Occipital Pipeline Node
========================

Implements the visual cortex processing hierarchy:
V1 → V2 → V4 → MT

Takes activity from a neural source (like CrystalChip's activity_view)
and processes it through biologically-inspired stages:

V1: Gabor filters (orientation, spatial frequency)
V2: Contour integration, border ownership
V4: Curvature, shape primitives, color
MT: Motion energy, optical flow

This recreates how the occipital lobe processes visual information,
but applied to the crystal's internal activity patterns.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode): 
            return None
    from PyQt6 import QtGui


class OccipitalPipelineNode(BaseNode):
    """
    Visual cortex hierarchy applied to neural activity patterns.
    """
    
    NODE_NAME = "Occipital Pipeline"
    NODE_TITLE = "Occipital Pipeline"
    NODE_CATEGORY = "Processing"
    NODE_COLOR = QtGui.QColor(255, 100, 150) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "activity_in": "image",      # Neural activity pattern (from crystal)
            "image_in": "image",         # Or raw image input
            "motion_sensitivity": "signal",  # MT sensitivity
            "detail_level": "signal",    # V1 spatial frequency preference
        }
        
        self.outputs = {
            "v1_output": "image",        # Orientation energy
            "v2_output": "image",        # Contour/boundary
            "v4_output": "image",        # Shape primitives
            "mt_output": "image",        # Motion energy
            "full_pipeline": "image",    # Combined visualization
            "dominant_orientation": "signal",
            "motion_energy": "signal",
            "complexity": "signal",      # Visual complexity measure
        }
        
        # Processing size
        self.size = 64
        
        # V1: Gabor filter bank
        self.n_orientations = 8
        self.n_scales = 3
        self.gabor_filters = []
        self._build_gabor_bank()
        
        # MT: Motion detection (need previous frame)
        self.prev_frame = None
        self.motion_history = None
        
        # Output storage
        self.v1_response = None
        self.v2_response = None
        self.v4_response = None
        self.mt_response = None
        
        # Statistics
        self.dominant_orientation = 0.0
        self.motion_energy_value = 0.0
        self.complexity_value = 0.0
        
        self.step_count = 0
        self.display_image = None
        
        self._output_values = {
            "dominant_orientation": 0.0,
            "motion_energy": 0.0,
            "complexity": 0.0,
        }
    
    def _build_gabor_bank(self):
        """Build V1-like Gabor filter bank."""
        self.gabor_filters = []
        
        # Multiple orientations and scales
        for scale in range(self.n_scales):
            sigma = 2.0 + scale * 2.0  # Increasing receptive field size
            wavelength = 4.0 + scale * 4.0
            
            for i in range(self.n_orientations):
                theta = i * np.pi / self.n_orientations
                
                # Gabor kernel
                kernel_size = int(sigma * 6) | 1  # Ensure odd
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    sigma=sigma,
                    theta=theta,
                    lambd=wavelength,
                    gamma=0.5,  # Aspect ratio
                    psi=0,      # Phase
                    ktype=cv2.CV_32F
                )
                
                self.gabor_filters.append({
                    'kernel': kernel,
                    'orientation': theta,
                    'scale': scale,
                    'sigma': sigma
                })
    
    def _read_input(self, name, default=None):
        """Read an input value."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first" if default is None else "mean")
                return val if val is not None else default
            except:
                return default
        return default
    
    def step(self):
        self.step_count += 1
        
        # Get input - prefer activity_in, fall back to image_in
        activity = self._read_input("activity_in")
        if activity is None:
            activity = self._read_input("image_in")
        
        if activity is None:
            self._update_display()
            return
        
        # Handle QImage conversion if needed
        if hasattr(activity, 'width') and hasattr(activity, 'bits'):
            activity = self._qimage_to_numpy(activity)
            if activity is None:
                return
        
        # Ensure proper format
        if activity.dtype != np.float32:
            activity = activity.astype(np.float32)
            if activity.max() > 1.0:
                activity /= 255.0
        
        # Convert to grayscale if needed
        if len(activity.shape) == 3:
            gray = np.mean(activity, axis=2)
        else:
            gray = activity
        
        # Resize to processing size
        if gray.shape[0] != self.size:
            gray = cv2.resize(gray, (self.size, self.size))
        
        # === V1: GABOR FILTERING (Orientation/Spatial Frequency) ===
        self.v1_response = self._process_v1(gray)
        
        # === V2: CONTOUR INTEGRATION ===
        self.v2_response = self._process_v2(self.v1_response)
        
        # === V4: SHAPE/CURVATURE ===
        self.v4_response = self._process_v4(self.v2_response)
        
        # === MT: MOTION ENERGY ===
        self.mt_response = self._process_mt(gray)
        
        # Store previous frame for motion
        self.prev_frame = gray.copy()
        
        # Compute statistics
        self._compute_statistics()
        
        self._update_display()
    
    def _qimage_to_numpy(self, qimg):
        """Convert QImage to numpy array."""
        try:
            width = qimg.width()
            height = qimg.height()
            bytes_per_line = qimg.bytesPerLine()
            ptr = qimg.bits()
            if ptr is None:
                return None
            ptr.setsize(height * bytes_per_line)
            arr = np.array(ptr).reshape(height, bytes_per_line)
            
            # Assume RGB888 or similar
            if bytes_per_line >= width * 3:
                arr = arr[:, :width*3].reshape(height, width, 3)
            else:
                arr = arr[:, :width]
            
            return arr.astype(np.float32) / 255.0
        except:
            return None
    
    def _process_v1(self, gray):
        """
        V1: Simple and complex cell responses via Gabor filtering.
        Returns orientation energy map.
        """
        h, w = gray.shape
        
        # Accumulate responses across orientations and scales
        orientation_energy = np.zeros((self.n_orientations, h, w), dtype=np.float32)
        
        for filt in self.gabor_filters:
            # Convolve
            response = cv2.filter2D(gray, -1, filt['kernel'])
            
            # Rectify (simple cell) and square (complex cell energy)
            energy = response ** 2
            
            # Accumulate by orientation
            ori_idx = int(filt['orientation'] / np.pi * self.n_orientations) % self.n_orientations
            orientation_energy[ori_idx] += energy
        
        # Combined orientation energy (max across orientations)
        combined = np.max(orientation_energy, axis=0)
        
        # Store dominant orientation at each location
        self.orientation_map = np.argmax(orientation_energy, axis=0)
        
        # Normalize
        combined = combined / (combined.max() + 1e-6)
        
        return combined
    
    def _process_v2(self, v1_output):
        """
        V2: Contour integration and border ownership.
        Uses non-maximum suppression and hysteresis.
        """
        # Edge detection on V1 output
        # Compute gradients
        gx = cv2.Sobel(v1_output, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(v1_output, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        # Non-maximum suppression (thin edges)
        v2_output = self._non_max_suppression(magnitude, direction)
        
        # Normalize
        v2_output = v2_output / (v2_output.max() + 1e-6)
        
        return v2_output
    
    def _non_max_suppression(self, magnitude, direction):
        """Thin edges by suppressing non-maximum gradient values."""
        h, w = magnitude.shape
        result = np.zeros_like(magnitude)
        
        # Quantize direction to 4 orientations
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Determine neighbors based on gradient direction
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
                elif 22.5 <= angle[i,j] < 67.5:
                    n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
                elif 67.5 <= angle[i,j] < 112.5:
                    n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
                else:
                    n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]
                
                # Keep only if local maximum
                if magnitude[i,j] >= n1 and magnitude[i,j] >= n2:
                    result[i,j] = magnitude[i,j]
        
        return result
    
    def _process_v4(self, v2_output):
        """
        V4: Curvature and shape primitives.
        Detects corners, curves, and texture patterns.
        """
        # Harris corner detection (curvature-sensitive)
        v2_uint8 = (v2_output * 255).astype(np.uint8)
        corners = cv2.cornerHarris(v2_uint8, blockSize=3, ksize=3, k=0.04)
        
        # Curvature via Laplacian
        laplacian = cv2.Laplacian(v2_output, cv2.CV_32F)
        curvature = np.abs(laplacian)
        
        # Combine corners and curvature
        v4_output = corners / (corners.max() + 1e-6) + curvature / (curvature.max() + 1e-6)
        v4_output = v4_output / (v4_output.max() + 1e-6)
        
        return v4_output
    
    def _process_mt(self, gray):
        """
        MT/V5: Motion energy.
        Computes optical flow and motion magnitude.
        """
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return np.zeros_like(gray)
        
        # Simple frame difference (motion energy)
        motion = np.abs(gray - self.prev_frame)
        
        # Temporal smoothing
        if self.motion_history is None:
            self.motion_history = motion.copy()
        else:
            self.motion_history = 0.7 * self.motion_history + 0.3 * motion
        
        # Normalize
        mt_output = self.motion_history / (self.motion_history.max() + 1e-6)
        
        return mt_output
    
    def _compute_statistics(self):
        """Compute summary statistics of visual processing."""
        
        # Dominant orientation (from V1)
        if hasattr(self, 'orientation_map'):
            # Histogram of orientations weighted by energy
            hist = np.zeros(self.n_orientations)
            for i in range(self.n_orientations):
                mask = self.orientation_map == i
                hist[i] = np.sum(self.v1_response[mask])
            
            self.dominant_orientation = float(np.argmax(hist) * 180 / self.n_orientations)
        
        # Motion energy (from MT)
        if self.mt_response is not None:
            self.motion_energy_value = float(np.mean(self.mt_response))
        
        # Complexity (from V4)
        if self.v4_response is not None:
            self.complexity_value = float(np.std(self.v4_response))
        
        # Update outputs
        self._output_values["dominant_orientation"] = self.dominant_orientation
        self._output_values["motion_energy"] = self.motion_energy_value
        self._output_values["complexity"] = self.complexity_value
    
    def get_output(self, port_name):
        if port_name == "v1_output":
            return self.v1_response
        elif port_name == "v2_output":
            return self.v2_response
        elif port_name == "v4_output":
            return self.v4_response
        elif port_name == "mt_output":
            return self.mt_response
        elif port_name == "full_pipeline":
            return self._render_full_pipeline()
        elif port_name in self._output_values:
            return self._output_values[port_name]
        return None
    
    def _render_full_pipeline(self):
        """Render all stages in a grid."""
        if self.v1_response is None:
            return np.zeros((self.size * 2, self.size * 2, 3), dtype=np.uint8)
        
        # 2x2 grid of processing stages
        h, w = self.size, self.size
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # V1 (top-left) - Orientation energy
        v1_vis = (self.v1_response * 255).astype(np.uint8)
        v1_color = cv2.applyColorMap(v1_vis, cv2.COLORMAP_HOT)
        grid[:h, :w] = v1_color
        
        # V2 (top-right) - Contours
        v2_vis = (self.v2_response * 255).astype(np.uint8)
        v2_color = cv2.applyColorMap(v2_vis, cv2.COLORMAP_BONE)
        grid[:h, w:] = v2_color
        
        # V4 (bottom-left) - Shape/curvature
        v4_vis = (self.v4_response * 255).astype(np.uint8)
        v4_color = cv2.applyColorMap(v4_vis, cv2.COLORMAP_VIRIDIS)
        grid[h:, :w] = v4_color
        
        # MT (bottom-right) - Motion
        mt_vis = (self.mt_response * 255).astype(np.uint8)
        mt_color = cv2.applyColorMap(mt_vis, cv2.COLORMAP_MAGMA)
        grid[h:, w:] = mt_color
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, "V1:Orient", (5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(grid, "V2:Contour", (w + 5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(grid, "V4:Shape", (5, h + 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(grid, "MT:Motion", (w + 5, h + 15), font, 0.4, (255, 255, 255), 1)
        
        return grid
    
    def _update_display(self):
        """Create main display image."""
        w, h = 400, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "OCCIPITAL PIPELINE", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 150), 2)
        cv2.putText(img, "V1 -> V2 -> V4 -> MT", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Pipeline visualization
        if self.v1_response is not None:
            pipeline = self._render_full_pipeline()
            pipeline_small = cv2.resize(pipeline, (256, 256))
            img[50:306, 10:266] = pipeline_small[:250, :]  # Fit in display
        
        # Stats
        stats_x = 280
        cv2.putText(img, f"Step: {self.step_count}", (stats_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(img, f"Orient: {self.dominant_orientation:.0f} deg", (stats_x, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        cv2.putText(img, f"Motion: {self.motion_energy_value:.3f}", (stats_x, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)
        cv2.putText(img, f"Complex: {self.complexity_value:.3f}", (stats_x, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 200), 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image
    
    def get_config_options(self):
        return [
            ("Processing Size", "size", self.size, None),
            ("Num Orientations", "n_orientations", self.n_orientations, None),
            ("Num Scales", "n_scales", self.n_scales, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            rebuild = False
            for key, value in options.items():
                if hasattr(self, key):
                    if key in ['n_orientations', 'n_scales'] and getattr(self, key) != value:
                        rebuild = True
                    setattr(self, key, value)
            
            if rebuild:
                self._build_gabor_bank()