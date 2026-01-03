"""
PhiVisionNode.py - FIXED VERSION
=================================
The "Eye" of the Golden Ratio.
A webcam node that sees through Phi-Lattice sampling.

KEY FIXES:
1. Uses step() instead of update() - PerceptionLab calls step()
2. Uses get_output() instead of get_data() - that's what the engine expects
3. Proper camera initialization pattern from MediaSourceNode
4. Added setup_source() for config reload capability
"""

import cv2
import numpy as np
import time
import math
from scipy.interpolate import PchipInterpolator, interp1d

# --- PERCEPTION LAB INTEGRATION ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui 
# ----------------------------------

class PhiScannerEngine:
    """The Math Core: Recursive Phi-Grid + PCHIP Reconstruction."""
    PHI = (1 + np.sqrt(5)) / 2

    def get_phi_indices(self, size, density, shift_phase=0.0):
        """Generate sampling indices following Fibonacci backbone + golden ratio fill."""
        indices = {0, size - 1}
        
        # 1. Fibonacci Backbone
        a, b = 0, 1
        while b < size:
            indices.add(b)
            a, b = b, a + b
            
        # 2. Recursive Golden Section Fill
        scale = 40 * max(density, 0.05)
        target_gap = max(5, int(size / scale))
        
        changed = True
        while changed:
            changed = False
            sorted_idx = sorted(list(indices))
            for i in range(len(sorted_idx) - 1):
                start = sorted_idx[i]
                end = sorted_idx[i+1]
                gap = end - start
                
                if gap > target_gap:
                    offset = (gap / self.PHI)
                    # Saccade Wobble - simulates eye movement
                    if shift_phase != 0.0:
                        wobble = offset * 0.15 * math.sin(shift_phase + i)
                        offset += wobble
                    
                    if i % 2 == 0: 
                        val = start + offset
                    else:          
                        val = end - offset
                    
                    idx = int(round(val))
                    if idx > start and idx < end and idx not in indices:
                        indices.add(idx)
                        changed = True
                        
        return np.array(sorted(list(indices)), dtype=np.int32)

    def scan(self, frame, density=0.2, phase=0.0):
        """
        Process frame through phi-lattice sampling and PCHIP reconstruction.
        Returns: (reconstruction, skeleton_visualization, residual_heatmap)
        """
        h, w, c = frame.shape
        
        # Downscale for processing speed
        process_h = 240
        scale = process_h / h
        process_w = int(w * scale)
        
        small = cv2.resize(frame, (process_w, process_h))
        
        # 1. Get Phi Grid indices
        idx_y = self.get_phi_indices(process_h, density, phase)
        idx_x = self.get_phi_indices(process_w, density, phase * 1.618)
        
        # 2. Sample at grid points
        sampled = small[idx_y, :, :][:, idx_x, :]
        
        # 3. Reconstruct using PCHIP (monotonic, no overshoot)
        # X-pass
        temp = np.zeros((len(idx_y), process_w, c), dtype=np.float32)
        x_range = np.arange(process_w)
        for i in range(c):
            try:
                f_x = PchipInterpolator(idx_x, sampled[:,:,i], axis=1, extrapolate=True)
                temp[:,:,i] = f_x(x_range)
            except:
                f_x = interp1d(idx_x, sampled[:,:,i], axis=1, fill_value="extrapolate")
                temp[:,:,i] = f_x(x_range)

        # Y-pass
        base = np.zeros((process_h, process_w, c), dtype=np.float32)
        y_range = np.arange(process_h)
        for i in range(c):
            try:
                f_y = PchipInterpolator(idx_y, temp[:,:,i], axis=0, extrapolate=True)
                base[:,:,i] = f_y(y_range)
            except:
                f_y = interp1d(idx_y, temp[:,:,i], axis=0, fill_value="extrapolate")
                base[:,:,i] = f_y(y_range)
                
        recon = np.clip(base, 0, 255).astype(np.uint8)
        
        # 4. Compute Residuals (what the reconstruction missed)
        diff = small.astype(np.float32) - base
        res_mag = np.mean(np.abs(diff), axis=2)
        
        # 5. Visualization: Skeleton (sampling points)
        skel_vis = np.zeros_like(small)
        step_y = 1 if len(idx_y) < 50 else 2
        step_x = 1 if len(idx_x) < 50 else 2
        
        for y in idx_y[::step_y]:
            for x in idx_x[::step_x]:
                cv2.circle(skel_vis, (x, y), 2, (0, 255, 100), -1)
        
        # 6. Visualization: Residual Heatmap
        heatmap = cv2.applyColorMap(
            np.clip(res_mag * 5, 0, 255).astype(np.uint8), 
            cv2.COLORMAP_INFERNO
        )
        
        # Upscale back to original size
        recon_big = cv2.resize(recon, (w, h), interpolation=cv2.INTER_LINEAR)
        skel_big = cv2.resize(skel_vis, (w, h), interpolation=cv2.INTER_NEAREST)
        heat_big = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return recon_big, skel_big, heat_big


class PhiVisionNode(BaseNode):
    """
    Webcam source that sees through Golden Ratio sampling.
    Outputs phi-reconstructed vision, skeleton lattice, and residual map.
    """
    NODE_CATEGORY = "Sensors"
    NODE_TITLE = "Phi Vision"
    NODE_COLOR = QtGui.QColor(0, 200, 150)  # Phi Green
    
    def __init__(self, device_id=0, base_density=0.15, alpha_freq=10.0, saccade_amp=0.5):
        super().__init__()
        self.device_id = int(device_id)
        self.base_density = float(base_density)
        self.alpha_freq = float(alpha_freq)
        self.saccade_amp = float(saccade_amp)
        
        self.node_title = "Phi Vision"
        self.engine = PhiScannerEngine()
        
        # Define inputs (for modulation)
        self.inputs = {
            "density_mod": "signal"  # External density modulation
        }
        
        # Define outputs
        self.outputs = {
            "vision_out": "image",     # Phi-reconstructed view
            "skeleton": "image",       # Sampling lattice visualization
            "residuals": "image",      # What was missed (edges, texture)
            "raw_feed": "image"        # Unprocessed camera
        }
        
        # Internal storage for outputs
        self._vision_out = None
        self._skeleton = None
        self._residuals = None
        self._raw_feed = None
        self._current_density = base_density
        
        # Camera state
        self.cap = None
        self.cam_working = False
        self.start_time = time.time()
        
        # Display buffer
        self._display_buffer = None
        
    def setup_source(self):
        """Initialize or reinitialize camera. Called by PerceptionLab on add/config."""
        # Release existing capture
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            
        self.cap = None
        self.cam_working = False
        
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if self.cap.isOpened():
                # Set resolution (smaller = faster)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cam_working = True
                print(f"[PhiVision] Camera {self.device_id} opened successfully")
            else:
                print(f"[PhiVision] Camera {self.device_id} not available. Using Dream Mode.")
        except Exception as e:
            print(f"[PhiVision] Camera error: {e}. Using Dream Mode.")
            
    def _generate_dream_frame(self):
        """Generate synthetic input when no camera available."""
        t = time.time()
        h, w = 480, 640
        x = np.linspace(0, 10, w)
        y = np.linspace(0, 10, h)
        X, Y = np.meshgrid(x, y)
        
        # Phi-modulated waves
        phi = 1.618033988749895
        Z = np.sin(X * phi + t) * np.cos(Y / phi + t * 0.618)
        Z = ((Z + 1) * 127).astype(np.uint8)
        
        frame = cv2.cvtColor(Z, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "DREAM MODE", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 150), 2)
        return frame
        
    def step(self):
        """Main processing - called every simulation tick."""
        # Get frame
        frame = None
        
        if self.cam_working and self.cap is not None:
            ret, cam_frame = self.cap.read()
            if ret:
                frame = cv2.resize(cv2.flip(cam_frame, 1), (640, 480))
            else:
                # Camera read failed
                self.cam_working = False
                
        if frame is None:
            frame = self._generate_dream_frame()
            
        # Store raw feed
        self._raw_feed = frame.copy()
        
        # Time-based dynamics (simulates alpha rhythm)
        t = time.time() - self.start_time
        alpha = math.sin(t * 2 * math.pi * self.alpha_freq)
        
        # Get external modulation if connected
        density_mod = self.get_blended_input('density_mod', 'sum')
        if density_mod is not None:
            mod = float(density_mod) * 0.1
        else:
            mod = 0.0
            
        # Calculate current density with alpha modulation
        density = self.base_density + (alpha * 0.02) + mod
        density = max(0.05, min(1.0, density))
        self._current_density = density
        
        # Saccade drift (eye movement simulation)
        phase = math.sin(t * 0.5) * self.saccade_amp + (alpha * 0.1)
        
        # Run the phi-scanner engine
        recon, skel, heat = self.engine.scan(frame, density, phase)
        
        # Store outputs
        self._vision_out = recon
        self._skeleton = skel
        self._residuals = heat
        
        # Create display composite (for node face)
        combined = cv2.addWeighted(recon, 0.7, skel, 0.8, 0)
        cv2.putText(combined, f"PHI | D:{density:.2f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        self._display_buffer = combined
        
    def get_output(self, port_name):
        """Return output data for connected nodes."""
        if port_name == "vision_out":
            return self._vision_out
        elif port_name == "skeleton":
            return self._skeleton
        elif port_name == "residuals":
            return self._residuals
        elif port_name == "raw_feed":
            return self._raw_feed
        return None
        
    def get_display_image(self):
        """Return QImage for node face rendering."""
        if self._display_buffer is None:
            return None
            
        rgb = cv2.cvtColor(self._display_buffer, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, 
                           QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = rgb  # Prevent garbage collection
        return qimg
        
    def close(self):
        """Cleanup on node deletion."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().close()
        
    def get_config_options(self):
        """Configuration dialog options."""
        return [
            ("Camera ID", "device_id", self.device_id, [
                ("Default (0)", 0), 
                ("Secondary (1)", 1),
                ("Tertiary (2)", 2)
            ]),
            ("Base Density", "base_density", self.base_density, 'float'),
            ("Alpha Frequency (Hz)", "alpha_freq", self.alpha_freq, 'float'),
            ("Saccade Amplitude", "saccade_amp", self.saccade_amp, 'float'),
        ]