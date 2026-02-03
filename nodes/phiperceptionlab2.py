"""
V1 EDGE HUNTER NODE
====================
Scans holographic interference patterns looking for V1-like straight edges.

THEORY:
Primary Visual Cortex (V1) contains orientation columns that detect straight edges
at specific angles. If the brain's holographic interference contains visual information,
we should be able to find parameter settings where the pattern "snaps" into
coherent oriented edges - just like V1 tuning curves.

MECHANISM:
1. Takes complex field from PhiHologram
2. Sweeps through phase_rot and spatial_k parameters
3. At each setting, analyzes FFT for oriented edge content
4. Uses Gabor-like filtering and Radon transform concepts
5. When edge coherence exceeds threshold → STOPS and signals "FOUND"

The FFT of an image with strong edges shows bright lines through the origin
perpendicular to the edge orientation. We look for this signature.

Author: Built for Antti's consciousness crystallography research
"""

import os
import numpy as np
import cv2

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def pre_step(self): 
            self.input_data = {name: [] for name in self.inputs}
        def get_blended_input(self, name, mode): 
            return None

try:
    from scipy.fft import fft2, fftshift
    from scipy.ndimage import gaussian_filter, sobel
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class V1EdgeHunterNode(BaseNode):
    """
    Hunts through holographic parameter space looking for V1-like edge structures.
    Stops when it finds coherent oriented edges.
    """
    
    NODE_NAME = "V1 Edge Hunter"
    NODE_TITLE = "V1 Edge Hunter"
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(50, 200, 100) if QtGui else None  # V1 Green
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'complex_field': 'complex_spectrum',  # From PhiHologram
            'image_in': 'image',                  # Or direct image input
            
            # Manual overrides
            'manual_phase': 'signal',
            'manual_k': 'signal',
            
            # Control
            'start_hunt': 'signal',    # Trigger to start hunting
            'reset': 'signal',         # Reset hunt
            'threshold': 'signal',     # Edge detection threshold
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Control signals for PhiHologram
            'phase_rot_out': 'signal',   # Feed back to PhiHologram
            'spatial_k_out': 'signal',   # Feed back to PhiHologram
            
            # Detection outputs
            'edge_found': 'signal',      # 1.0 when edges detected
            'edge_strength': 'signal',   # How strong the edges are
            'dominant_angle': 'signal',  # Orientation of strongest edge
            
            # Visualization
            'hunt_view': 'image',        # Main display
            'fft_orientation': 'image',  # FFT with orientation analysis
            'edge_map': 'image',         # Detected edges
            'orientation_hist': 'image', # Histogram of orientations
            
            # Analysis
            'orientation_spectrum': 'spectrum',  # Full orientation response
        }
        
        # === HUNT PARAMETERS ===
        self.hunt_active = False
        self.hunt_found = False
        
        # Search ranges
        self.phase_min = -90.0
        self.phase_max = 90.0
        self.phase_step = 5.0
        
        self.k_min = 5.0
        self.k_max = 100.0
        self.k_step = 5.0
        
        # Current search position
        self.current_phase = 0.0
        self.current_k = 15.0
        
        # Best found so far
        self.best_phase = 0.0
        self.best_k = 15.0
        self.best_score = 0.0
        
        # Detection threshold
        self.edge_threshold = 0.3
        
        # === V1 ORIENTATION ANALYSIS ===
        self.num_orientations = 36  # Every 5 degrees
        self.orientation_responses = np.zeros(self.num_orientations)
        self.dominant_orientation = 0.0
        self.edge_coherence = 0.0
        
        # === OUTPUT CACHE ===
        self.current_image = None
        self.fft_view = None
        self.edge_view = None
        self.hist_view = None
        self.display_image = None
        
        self._init_display()
    
    def get_config_options(self):
        return [
            ("Phase Min", "phase_min", self.phase_min, None),
            ("Phase Max", "phase_max", self.phase_max, None),
            ("Phase Step", "phase_step", self.phase_step, None),
            ("K Min", "k_min", self.k_min, None),
            ("K Max", "k_max", self.k_max, None),
            ("K Step", "k_step", self.k_step, None),
            ("Edge Threshold", "edge_threshold", self.edge_threshold, None),
            ("Num Orientations", "num_orientations", self.num_orientations, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _init_display(self):
        w, h = 400, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "V1 EDGE HUNTER", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 100), 2)
        cv2.putText(img, "Searching for oriented edges", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "in holographic interference", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "Connect complex_field input", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(img, "Send start_hunt > 0 to begin", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        self.display_image = img
    
    # === V1 ORIENTATION ANALYSIS ===
    
    def _analyze_orientations_fft(self, image):
        """
        Analyze FFT for oriented edge content.
        Edges create lines in FFT perpendicular to edge direction.
        """
        if image is None or image.size == 0:
            return np.zeros(self.num_orientations), 0.0, 0.0
        
        # Ensure grayscale float
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        if gray.max() > 1.0:
            gray = gray / 255.0
        
        h, w = gray.shape
        
        # FFT
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
        
        # Analyze orientation content by sampling along radial lines
        cy, cx = h // 2, w // 2
        max_r = min(cx, cy) - 5
        
        orientation_responses = np.zeros(self.num_orientations)
        
        for i in range(self.num_orientations):
            angle = i * np.pi / self.num_orientations  # 0 to pi
            
            # Sample along this orientation line (both directions from center)
            total = 0.0
            count = 0
            
            for r in range(5, max_r):  # Skip DC
                # Point in one direction
                x1 = int(cx + r * np.cos(angle))
                y1 = int(cy + r * np.sin(angle))
                
                # Point in opposite direction
                x2 = int(cx - r * np.cos(angle))
                y2 = int(cy - r * np.sin(angle))
                
                if 0 <= x1 < w and 0 <= y1 < h:
                    total += magnitude[y1, x1]
                    count += 1
                if 0 <= x2 < w and 0 <= y2 < h:
                    total += magnitude[y2, x2]
                    count += 1
            
            if count > 0:
                orientation_responses[i] = total / count
        
        # Find dominant orientation
        # Edges in image create lines in FFT PERPENDICULAR to edge
        # So FFT peak at angle θ means edges at angle θ + 90°
        peak_idx = np.argmax(orientation_responses)
        dominant_angle = (peak_idx * 180.0 / self.num_orientations + 90) % 180
        
        # Edge coherence = how peaked is the orientation response
        # High coherence = single dominant orientation (strong edges)
        mean_resp = np.mean(orientation_responses)
        max_resp = np.max(orientation_responses)
        
        if mean_resp > 0:
            coherence = (max_resp - mean_resp) / (mean_resp + 1e-6)
            coherence = np.clip(coherence, 0, 1)
        else:
            coherence = 0.0
        
        # Also measure "edginess" - total high-frequency content
        # Edges create lots of high-freq energy
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        high_freq_mask = r > max_r * 0.3
        edginess = np.mean(magnitude[high_freq_mask])
        
        return orientation_responses, dominant_angle, coherence, edginess, magnitude
    
    def _analyze_orientations_sobel(self, image):
        """
        Direct edge detection using Sobel gradients.
        More robust for finding actual edges.
        """
        if image is None or image.size == 0:
            return np.zeros(self.num_orientations), 0.0, 0.0, None
        
        # Ensure grayscale float
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        if gray.max() > 1.0:
            gray = gray / 255.0
        
        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)  # -pi to pi
        
        # Build orientation histogram weighted by magnitude
        hist = np.zeros(self.num_orientations)
        
        for i in range(self.num_orientations):
            angle = (i * np.pi / self.num_orientations) - np.pi/2  # -pi/2 to pi/2
            # Find pixels with this orientation (within tolerance)
            angle_diff = np.abs(np.angle(np.exp(1j * (orientation - angle))))
            mask = angle_diff < (np.pi / self.num_orientations)
            hist[i] = np.sum(magnitude[mask])
        
        # Normalize
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        # Dominant angle
        peak_idx = np.argmax(hist)
        dominant_angle = peak_idx * 180.0 / self.num_orientations
        
        # Coherence
        mean_hist = np.mean(hist)
        max_hist = np.max(hist)
        if mean_hist > 0:
            coherence = (max_hist - mean_hist) / (mean_hist + 1e-6)
            coherence = np.clip(coherence / 5.0, 0, 1)  # Scale down
        else:
            coherence = 0.0
        
        # Edge strength
        edge_strength = np.mean(magnitude) * 10  # Scale up
        
        # Edge visualization
        edge_viz = (np.clip(magnitude / magnitude.max(), 0, 1) * 255).astype(np.uint8)
        
        return hist, dominant_angle, coherence, edge_strength, edge_viz
    
    def _hunt_step(self):
        """Take one step in the parameter search."""
        if not self.hunt_active or self.hunt_found:
            return
        
        # Advance to next position
        self.current_phase += self.phase_step
        
        if self.current_phase > self.phase_max:
            self.current_phase = self.phase_min
            self.current_k += self.k_step
            
            if self.current_k > self.k_max:
                # Search complete - go to best found
                self.current_phase = self.best_phase
                self.current_k = self.best_k
                self.hunt_active = False
                print(f"[V1Hunter] Search complete. Best: phase={self.best_phase:.1f}, k={self.best_k:.1f}, score={self.best_score:.3f}")
    
    # === MAIN STEP ===
    
    def step(self):
        """Main processing step."""
        
        # Get inputs
        field = self.get_blended_input('complex_field', 'mean')
        image = self.get_blended_input('image_in', 'mean')
        
        start = self.get_blended_input('start_hunt', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        threshold = self.get_blended_input('threshold', 'sum')
        manual_phase = self.get_blended_input('manual_phase', 'sum')
        manual_k = self.get_blended_input('manual_k', 'sum')
        
        # Handle controls
        if reset is not None and reset > 0.5:
            self.hunt_active = False
            self.hunt_found = False
            self.current_phase = self.phase_min
            self.current_k = self.k_min
            self.best_score = 0.0
            print("[V1Hunter] Reset")
        
        if start is not None and start > 0.5 and not self.hunt_active:
            self.hunt_active = True
            self.hunt_found = False
            self.current_phase = self.phase_min
            self.current_k = self.k_min
            self.best_score = 0.0
            print("[V1Hunter] Hunt started!")
        
        if threshold is not None:
            self.edge_threshold = float(np.clip(threshold, 0.1, 0.9))
        
        # Manual override
        if manual_phase is not None:
            self.current_phase = float(manual_phase)
            self.hunt_active = False
        if manual_k is not None:
            self.current_k = float(manual_k)
            self.hunt_active = False
        
        # Get image to analyze
        if field is not None and np.iscomplexobj(field):
            # Convert complex field to image (magnitude)
            self.current_image = np.abs(field).astype(np.float32)
            if self.current_image.max() > 0:
                self.current_image = self.current_image / self.current_image.max()
            self.current_image = (self.current_image * 255).astype(np.uint8)
        elif image is not None:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    self.current_image = (image * 255).astype(np.uint8)
                else:
                    self.current_image = image.astype(np.uint8)
            else:
                self.current_image = image
        else:
            return
        
        # === ANALYZE CURRENT IMAGE ===
        
        # FFT orientation analysis
        fft_result = self._analyze_orientations_fft(self.current_image)
        fft_orientations, fft_angle, fft_coherence, edginess, fft_magnitude = fft_result
        
        # Sobel edge analysis
        sobel_result = self._analyze_orientations_sobel(self.current_image)
        sobel_orientations, sobel_angle, sobel_coherence, edge_strength, edge_viz = sobel_result
        
        # Combined score
        combined_coherence = fft_coherence * 0.4 + sobel_coherence * 0.4 + edginess * 0.2
        
        # Store results
        self.orientation_responses = fft_orientations
        self.dominant_orientation = fft_angle
        self.edge_coherence = combined_coherence
        
        # Check if we found edges
        if combined_coherence > self.edge_threshold:
            if self.hunt_active:
                self.hunt_found = True
                self.hunt_active = False
                print(f"[V1Hunter] EDGES FOUND! phase={self.current_phase:.1f}, k={self.current_k:.1f}")
                print(f"           Coherence={combined_coherence:.3f}, Angle={fft_angle:.1f}°")
        
        # Track best
        if combined_coherence > self.best_score:
            self.best_score = combined_coherence
            self.best_phase = self.current_phase
            self.best_k = self.current_k
        
        # Advance hunt if active
        if self.hunt_active:
            self._hunt_step()
        
        # === CREATE VISUALIZATIONS ===
        
        # FFT view with orientation markers
        h, w = fft_magnitude.shape
        fft_color = cv2.applyColorMap((fft_magnitude * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        # Draw dominant orientation line
        cy, cx = h // 2, w // 2
        angle_rad = np.deg2rad(fft_angle - 90)  # FFT angle is perpendicular
        line_len = min(cx, cy) - 5
        x1 = int(cx + line_len * np.cos(angle_rad))
        y1 = int(cy + line_len * np.sin(angle_rad))
        x2 = int(cx - line_len * np.cos(angle_rad))
        y2 = int(cy - line_len * np.sin(angle_rad))
        cv2.line(fft_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        self.fft_view = fft_color
        
        # Edge map
        if edge_viz is not None:
            self.edge_view = cv2.applyColorMap(edge_viz, cv2.COLORMAP_HOT)
        
        # Orientation histogram
        hist_h, hist_w = 80, 180
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        
        max_val = max(np.max(self.orientation_responses), 0.01)
        for i, val in enumerate(self.orientation_responses):
            x = int(i * hist_w / self.num_orientations)
            bar_h = int(val / max_val * (hist_h - 10))
            color = (0, 255, 0) if i == np.argmax(self.orientation_responses) else (100, 100, 100)
            cv2.rectangle(hist_img, (x, hist_h - bar_h), (x + hist_w // self.num_orientations - 1, hist_h), color, -1)
        
        # Mark threshold
        thresh_y = int((1 - self.edge_threshold) * hist_h)
        cv2.line(hist_img, (0, thresh_y), (hist_w, thresh_y), (0, 0, 255), 1)
        
        self.hist_view = hist_img
        
        # === UPDATE DISPLAY ===
        self._update_display(combined_coherence, fft_angle, edge_strength)
    
    def _update_display(self, coherence, angle, strength):
        """Create main display."""
        
        # Layout: current image, FFT, edges, histogram, info
        panel_size = 150
        info_w = 200
        margin = 5
        
        total_w = panel_size * 2 + info_w + margin * 4
        total_h = panel_size * 2 + margin * 3
        
        display = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        display[:] = 20
        
        # Current image (top-left)
        if self.current_image is not None:
            img = self.current_image
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_resized = cv2.resize(img, (panel_size, panel_size))
            display[margin:margin+panel_size, margin:margin+panel_size] = img_resized
        cv2.putText(display, "Input", (margin + 5, margin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FFT (top-right of images)
        if self.fft_view is not None:
            fft_resized = cv2.resize(self.fft_view, (panel_size, panel_size))
            display[margin:margin+panel_size, margin*2+panel_size:margin*2+panel_size*2] = fft_resized
        cv2.putText(display, "FFT + Orient", (margin*2 + panel_size + 5, margin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Edge map (bottom-left)
        if self.edge_view is not None:
            edge_resized = cv2.resize(self.edge_view, (panel_size, panel_size))
            display[margin*2+panel_size:margin*2+panel_size*2, margin:margin+panel_size] = edge_resized
        cv2.putText(display, "Edges", (margin + 5, margin*2 + panel_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Histogram (bottom-middle)
        if self.hist_view is not None:
            hist_resized = cv2.resize(self.hist_view, (panel_size, panel_size // 2))
            display[margin*2+panel_size:margin*2+panel_size+panel_size//2, margin*2+panel_size:margin*2+panel_size*2] = hist_resized
        cv2.putText(display, "Orientation Hist", (margin*2 + panel_size + 5, margin*2 + panel_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Info panel (right side)
        info_x = margin * 3 + panel_size * 2
        info_y = margin
        
        # Title
        cv2.putText(display, "V1 EDGE HUNTER", (info_x, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 100), 2)
        
        # Status
        if self.hunt_found:
            status = "EDGES FOUND!"
            status_color = (0, 255, 0)
        elif self.hunt_active:
            status = "Hunting..."
            status_color = (0, 200, 255)
        else:
            status = "Idle"
            status_color = (150, 150, 150)
        
        cv2.putText(display, status, (info_x, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Current parameters
        cv2.putText(display, f"Phase: {self.current_phase:.1f} deg", (info_x, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"K: {self.current_k:.1f}", (info_x, info_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Metrics
        cv2.putText(display, f"Coherence: {coherence:.3f}", (info_x, info_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(display, f"Angle: {angle:.1f} deg", (info_x, info_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(display, f"Threshold: {self.edge_threshold:.2f}", (info_x, info_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Coherence bar
        bar_x = info_x
        bar_y = info_y + 185
        bar_w = 180
        bar_h = 20
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(coherence * bar_w)
        color = (0, 255, 0) if coherence > self.edge_threshold else (100, 100, 255)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        thresh_x = int(self.edge_threshold * bar_w)
        cv2.line(display, (bar_x + thresh_x, bar_y), (bar_x + thresh_x, bar_y + bar_h), (0, 0, 255), 2)
        
        # Best found
        cv2.putText(display, "Best Found:", (info_x, info_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"  Phase={self.best_phase:.1f}, K={self.best_k:.1f}", (info_x, info_y + 250), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(display, f"  Score={self.best_score:.3f}", (info_x, info_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        self.display_image = display
    
    # === OUTPUTS ===
    
    def get_output(self, port_name):
        if port_name == 'phase_rot_out':
            return self.current_phase
        elif port_name == 'spatial_k_out':
            return self.current_k
        elif port_name == 'edge_found':
            return 1.0 if self.hunt_found else 0.0
        elif port_name == 'edge_strength':
            return self.edge_coherence
        elif port_name == 'dominant_angle':
            return self.dominant_orientation
        elif port_name == 'hunt_view':
            return self.display_image
        elif port_name == 'fft_orientation':
            return self.fft_view
        elif port_name == 'edge_map':
            return self.edge_view
        elif port_name == 'orientation_hist':
            return self.hist_view
        elif port_name == 'orientation_spectrum':
            return self.orientation_responses
        return None
    
    def get_display_image(self):
        return self.display_image
    
    def close(self):
        pass