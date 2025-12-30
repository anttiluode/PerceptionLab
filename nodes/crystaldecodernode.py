"""
Crystal Decoder Node
====================

Systematically probes a crystal to decode what it learned from EEG.

Methods:
1. FREQUENCY SWEEP - Which temporal frequencies resonate?
2. SPATIAL SWEEP - Which spatial patterns activate it?
3. ATTRACTOR MAPPING - Where does it naturally settle?
4. RESPONSE CLUSTERING - Group similar output states

The goal: Understand the crystal's "vocabulary" so we can
decode its outputs back to meaningful EEG states.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque

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


class CrystalDecoderNode(BaseNode):
    """
    Probes and decodes crystal responses to understand learned EEG patterns.
    """
    
    NODE_NAME = "Crystal Decoder"
    NODE_TITLE = "Crystal Decoder"
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(255, 200, 50) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            # From crystal
            "crystal_activity": "image",    # activity_view from crystal
            "crystal_delta": "signal",
            "crystal_theta": "signal", 
            "crystal_alpha": "signal",
            "crystal_beta": "signal",
            "crystal_gamma": "signal",
            "crystal_energy": "signal",
            
            # Control
            "probe_trigger": "signal",      # Trigger a probe sequence
        }
        
        self.outputs = {
            # Probe outputs (feed back to crystal)
            "probe_signal": "signal",       # Current probe frequency
            "probe_image": "image",         # Current probe pattern
            
            # Analysis outputs
            "frequency_response": "image",   # Which frequencies resonate
            "spatial_response": "image",     # Which patterns activate
            "attractor_map": "image",        # Natural settling states
            "state_clusters": "image",       # Clustered output states
            "decoder_display": "image",      # Main visualization
            
            # Decoded signals
            "decoded_state": "signal",       # Current decoded state ID
            "arousal": "signal",             # Decoded arousal level
            "valence": "signal",             # Decoded valence (if possible)
        }
        
        # Probe parameters
        self.probe_mode = "idle"  # idle, freq_sweep, spatial_sweep, attractor
        self.probe_step = 0
        self.probe_duration = 50  # Steps per probe
        
        # Frequency sweep (0.5 - 50 Hz range, mapped to signal values)
        self.freq_bins = 20
        self.freq_responses = np.zeros(self.freq_bins)
        self.current_freq_idx = 0
        
        # Spatial sweep (different patterns)
        self.spatial_patterns = []
        self.spatial_responses = []
        self._generate_spatial_patterns()
        self.current_spatial_idx = 0
        
        # Attractor mapping
        self.attractor_history = deque(maxlen=500)
        self.attractor_clusters = None
        self.n_attractors = 5
        
        # State clustering
        self.state_history = deque(maxlen=1000)
        self.cluster_centers = None
        self.n_clusters = 8
        
        # Current probe output
        self.current_probe_signal = 0.0
        self.current_probe_image = None
        
        # Response tracking
        self.response_accumulator = []
        
        # Decoded values
        self.decoded_state_id = 0
        self.decoded_arousal = 0.5
        self.decoded_valence = 0.5
        
        # Display
        self.step_count = 0
        self.display_image = None
        
        self._update_display()
    
    def _generate_spatial_patterns(self):
        """Generate probe patterns for spatial sweep."""
        size = 64
        self.spatial_patterns = []
        
        # 1. Uniform (baseline)
        self.spatial_patterns.append(('uniform', np.ones((size, size)) * 0.5))
        
        # 2. Center spot (foveal)
        center = np.zeros((size, size))
        cv2.circle(center, (size//2, size//2), size//4, 1.0, -1)
        center = cv2.GaussianBlur(center, (15, 15), 5)
        self.spatial_patterns.append(('center', center))
        
        # 3. Horizontal stripes (different frequencies)
        for freq in [2, 4, 8, 16]:
            pattern = np.sin(np.linspace(0, freq * np.pi, size)).reshape(-1, 1)
            pattern = np.tile(pattern, (1, size))
            pattern = (pattern + 1) / 2
            self.spatial_patterns.append((f'h_stripe_{freq}', pattern))
        
        # 4. Vertical stripes
        for freq in [2, 4, 8, 16]:
            pattern = np.sin(np.linspace(0, freq * np.pi, size)).reshape(1, -1)
            pattern = np.tile(pattern, (size, 1))
            pattern = (pattern + 1) / 2
            self.spatial_patterns.append((f'v_stripe_{freq}', pattern))
        
        # 5. Diagonal stripes
        y, x = np.mgrid[:size, :size]
        for freq in [4, 8]:
            pattern = np.sin((x + y) / size * freq * np.pi)
            pattern = (pattern + 1) / 2
            self.spatial_patterns.append((f'd_stripe_{freq}', pattern))
        
        # 6. Concentric circles
        y, x = np.mgrid[:size, :size]
        r = np.sqrt((x - size//2)**2 + (y - size//2)**2)
        for freq in [2, 4, 8]:
            pattern = np.sin(r / size * freq * np.pi)
            pattern = (pattern + 1) / 2
            self.spatial_patterns.append((f'circle_{freq}', pattern))
        
        # 7. Checkerboard
        for check_size in [4, 8, 16]:
            pattern = np.indices((size, size)).sum(axis=0) // check_size % 2
            self.spatial_patterns.append((f'checker_{check_size}', pattern.astype(float)))
        
        # 8. Noise patterns
        np.random.seed(42)
        noise = np.random.rand(size, size)
        self.spatial_patterns.append(('noise', noise))
        
        # 9. Gabor-like patterns (oriented)
        for angle in [0, 45, 90, 135]:
            theta = np.radians(angle)
            x_rot = x * np.cos(theta) + y * np.sin(theta)
            pattern = np.sin(x_rot / size * 8 * np.pi)
            pattern = pattern * np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2 * (size//3)**2))
            pattern = (pattern + 1) / 2
            self.spatial_patterns.append((f'gabor_{angle}', pattern))
        
        # Initialize response array
        self.spatial_responses = [0.0] * len(self.spatial_patterns)
    
    def _read_signal(self, name, default=0.0):
        """Read a signal input."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                return float(val)
            except Exception:
                return default
        return default
    
    def _read_image(self, name):
        """Read an image input, handling QImage and numpy arrays."""
        fn = getattr(self, "get_blended_input", None)
        if not callable(fn):
            return None
        
        try:
            val = fn(name, "first")
            if val is None:
                return None
            
            # If it's already a numpy array, return it
            if isinstance(val, np.ndarray):
                return val
            
            # If it's a QImage, convert it
            if hasattr(val, 'width') and hasattr(val, 'height'):
                return self._qimage_to_numpy(val)
            
            return None
        except Exception as e:
            return None
    
    def _qimage_to_numpy(self, qimg):
        """Convert QImage to numpy array safely."""
        try:
            if qimg is None:
                return None
            
            w = qimg.width()
            h = qimg.height()
            
            if w <= 0 or h <= 0:
                return None
            
            # Get the bits pointer
            ptr = qimg.bits()
            if ptr is None:
                return None
            
            bytes_per_line = qimg.bytesPerLine()
            ptr.setsize(h * bytes_per_line)
            
            # Create array
            arr = np.array(ptr, dtype=np.uint8).reshape(h, bytes_per_line)
            
            # Handle different formats
            if bytes_per_line >= w * 4:
                # RGBA or ARGB format
                arr = arr[:, :w*4].reshape(h, w, 4)
                arr = arr[:, :, :3]  # Take RGB only
            elif bytes_per_line >= w * 3:
                # RGB format
                arr = arr[:, :w*3].reshape(h, w, 3)
            else:
                # Grayscale or unknown
                arr = arr[:, :w].reshape(h, w)
                arr = np.stack([arr, arr, arr], axis=2)
            
            return arr.astype(np.float32) / 255.0
        except Exception as e:
            return None
    
    def step(self):
        self.step_count += 1
        
        # Read crystal state
        activity = self._read_image("crystal_activity")
        delta = self._read_signal("crystal_delta", 0.0)
        theta = self._read_signal("crystal_theta", 0.0)
        alpha = self._read_signal("crystal_alpha", 0.0)
        beta = self._read_signal("crystal_beta", 0.0)
        gamma = self._read_signal("crystal_gamma", 0.0)
        energy = self._read_signal("crystal_energy", 0.0)
        
        # Build state vector
        state_vec = np.array([delta, theta, alpha, beta, gamma], dtype=np.float32)
        
        # Store in history for clustering (only if we have valid data)
        if np.any(state_vec != 0):
            self.state_history.append(state_vec.copy())
        
        # Check for probe trigger
        trigger = self._read_signal("probe_trigger", 0.0)
        if trigger > 0.5 and self.probe_mode == "idle":
            self._start_probe_sequence()
        
        # Run probe if active
        if self.probe_mode != "idle":
            self._run_probe_step(energy, state_vec)
        
        # Decode current state
        self._decode_state(state_vec)
        
        # Update attractor tracking (only if we have valid activity)
        if activity is not None:
            self._update_attractor_map(activity)
        
        # Periodic clustering
        if self.step_count % 100 == 0 and len(self.state_history) > 50:
            self._cluster_states()
        
        self._update_display()
    
    def _start_probe_sequence(self):
        """Start a full probe sequence."""
        self.probe_mode = "freq_sweep"
        self.probe_step = 0
        self.current_freq_idx = 0
        self.freq_responses = np.zeros(self.freq_bins)
        self.response_accumulator = []
        print("[Decoder] Starting frequency sweep...")
    
    def _run_probe_step(self, energy, state_vec):
        """Execute one step of the current probe."""
        
        if self.probe_mode == "freq_sweep":
            # Generate frequency probe signal
            freq = 0.5 + (self.current_freq_idx / self.freq_bins) * 49.5  # 0.5 to 50 Hz
            t = self.probe_step / 100.0  # Assume 100 Hz update rate
            self.current_probe_signal = np.sin(2 * np.pi * freq * t)
            
            # Accumulate response
            self.response_accumulator.append(energy)
            
            self.probe_step += 1
            
            # Move to next frequency
            if self.probe_step >= self.probe_duration:
                # Store mean response for this frequency
                if len(self.response_accumulator) > 0:
                    self.freq_responses[self.current_freq_idx] = np.mean(self.response_accumulator)
                self.response_accumulator = []
                self.probe_step = 0
                self.current_freq_idx += 1
                
                if self.current_freq_idx >= self.freq_bins:
                    # Done with freq sweep, move to spatial
                    self.probe_mode = "spatial_sweep"
                    self.current_spatial_idx = 0
                    print("[Decoder] Starting spatial sweep...")
        
        elif self.probe_mode == "spatial_sweep":
            # Use current spatial pattern
            if self.current_spatial_idx < len(self.spatial_patterns):
                name, pattern = self.spatial_patterns[self.current_spatial_idx]
                
                # Convert to uint8 image for output
                self.current_probe_image = (pattern * 255).astype(np.uint8)
                
                # Accumulate response
                self.response_accumulator.append(energy)
                
                self.probe_step += 1
                
                if self.probe_step >= self.probe_duration:
                    # Store response
                    if len(self.response_accumulator) > 0:
                        self.spatial_responses[self.current_spatial_idx] = np.mean(self.response_accumulator)
                    self.response_accumulator = []
                    self.probe_step = 0
                    self.current_spatial_idx += 1
                    
                    if self.current_spatial_idx >= len(self.spatial_patterns):
                        self.probe_mode = "idle"
                        self.current_probe_image = None
                        self.current_probe_signal = 0.0
                        print("[Decoder] Probe sequence complete!")
                        self._analyze_responses()
    
    def _analyze_responses(self):
        """Analyze probe responses to characterize crystal."""
        # Find peak frequency
        if np.max(self.freq_responses) > 0:
            peak_freq_idx = np.argmax(self.freq_responses)
            peak_freq = 0.5 + (peak_freq_idx / self.freq_bins) * 49.5
            print(f"[Decoder] Peak frequency response: {peak_freq:.1f} Hz")
        
        # Find best spatial patterns
        if np.max(self.spatial_responses) > 0:
            sorted_spatial = sorted(enumerate(self.spatial_responses), 
                                   key=lambda x: x[1], reverse=True)
            print("[Decoder] Top spatial patterns:")
            for i, (idx, resp) in enumerate(sorted_spatial[:5]):
                name = self.spatial_patterns[idx][0]
                print(f"  {i+1}. {name}: {resp:.2f}")
    
    def _decode_state(self, state_vec):
        """Decode current state into meaningful values."""
        state_sum = np.sum(state_vec)
        if state_sum == 0:
            return
        
        # Normalize
        state_norm = state_vec / (state_sum + 1e-6)
        
        # Arousal: High beta/gamma = high arousal, high delta/theta = low
        self.decoded_arousal = (state_norm[3] + state_norm[4]) - (state_norm[0] + state_norm[1]) * 0.5
        self.decoded_arousal = np.clip((self.decoded_arousal + 1) / 2, 0, 1)
        
        # If we have cluster centers, find nearest cluster
        if self.cluster_centers is not None and len(self.cluster_centers) > 0:
            try:
                distances = np.linalg.norm(self.cluster_centers - state_vec, axis=1)
                self.decoded_state_id = int(np.argmin(distances))
            except Exception:
                pass
    
    def _update_attractor_map(self, activity):
        """Track where activity patterns settle."""
        if activity is None:
            return
        
        try:
            # Ensure it's a numpy array
            if not isinstance(activity, np.ndarray):
                return
            
            # Check for valid shape
            if activity.size == 0:
                return
            
            # Convert to 2D if needed
            if len(activity.shape) == 3:
                activity = np.mean(activity, axis=2)
            
            if activity.shape[0] < 2 or activity.shape[1] < 2:
                return
            
            # Downsample activity to manageable size
            small = cv2.resize(activity.astype(np.float32), (16, 16))
            self.attractor_history.append(small.flatten().copy())
            
        except Exception as e:
            pass  # Silently skip bad frames
    
    def _cluster_states(self):
        """Cluster accumulated states."""
        if len(self.state_history) < 20:
            return
        
        try:
            # Simple k-means-like clustering
            states = np.array(list(self.state_history))
            
            # Initialize centers randomly
            n = min(self.n_clusters, len(states))
            indices = np.random.choice(len(states), n, replace=False)
            centers = states[indices].copy()
            
            # Iterate
            for _ in range(10):
                # Assign to nearest center
                distances = np.array([[np.linalg.norm(s - c) for c in centers] for s in states])
                assignments = np.argmin(distances, axis=1)
                
                # Update centers
                for i in range(n):
                    mask = assignments == i
                    if np.any(mask):
                        centers[i] = states[mask].mean(axis=0)
            
            self.cluster_centers = centers
        except Exception as e:
            pass
    
    def get_output(self, port_name):
        if port_name == "probe_signal":
            return self.current_probe_signal
        elif port_name == "probe_image":
            return self.current_probe_image
        elif port_name == "frequency_response":
            return self._render_freq_response()
        elif port_name == "spatial_response":
            return self._render_spatial_response()
        elif port_name == "decoder_display":
            return self._render_decoder_display()
        elif port_name == "decoded_state":
            return float(self.decoded_state_id)
        elif port_name == "arousal":
            return float(self.decoded_arousal)
        elif port_name == "valence":
            return float(self.decoded_valence)
        return None
    
    def _render_freq_response(self):
        """Render frequency response curve."""
        h, w = 100, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        max_resp = np.max(self.freq_responses)
        if max_resp > 0:
            # Normalize
            resp_norm = self.freq_responses / (max_resp + 1e-6)
            
            # Draw curve
            for i in range(self.freq_bins - 1):
                x1 = int(i / self.freq_bins * w)
                x2 = int((i + 1) / self.freq_bins * w)
                y1 = int((1 - resp_norm[i]) * (h - 20)) + 10
                y2 = int((1 - resp_norm[i + 1]) * (h - 20)) + 10
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Labels
        cv2.putText(img, "0.5Hz", (5, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)
        cv2.putText(img, "50Hz", (w-35, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)
        
        return img
    
    def _render_spatial_response(self):
        """Render spatial pattern responses as bar chart."""
        n_patterns = len(self.spatial_patterns)
        h, w = 100, max(200, n_patterns * 8)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        max_resp = np.max(self.spatial_responses) if len(self.spatial_responses) > 0 else 0
        if max_resp > 0:
            resp_norm = np.array(self.spatial_responses) / (max_resp + 1e-6)
            
            bar_w = w // n_patterns
            for i, resp in enumerate(resp_norm):
                x = i * bar_w
                bar_h = int(resp * (h - 20))
                color = (int(255 * (1-resp)), int(255 * resp), 100)
                cv2.rectangle(img, (x, h - 10 - bar_h), (x + bar_w - 1, h - 10), color, -1)
        
        return img
    
    def _render_decoder_display(self):
        """Main decoder visualization."""
        w, h = 400, 350
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "CRYSTAL DECODER", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 50), 2)
        
        # Mode indicator
        mode_colors = {"idle": (100, 100, 100), "freq_sweep": (0, 255, 0), "spatial_sweep": (255, 100, 0)}
        color = mode_colors.get(self.probe_mode, (100, 100, 100))
        cv2.putText(img, f"Mode: {self.probe_mode}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Frequency response
        cv2.putText(img, "Frequency Response", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        freq_img = self._render_freq_response()
        freq_img = cv2.resize(freq_img, (180, 80))
        img[80:160, 10:190] = freq_img
        
        # Spatial response
        cv2.putText(img, "Spatial Response", (210, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        spatial_img = self._render_spatial_response()
        spatial_img = cv2.resize(spatial_img, (180, 80))
        img[80:160, 210:390] = spatial_img
        
        # Decoded state
        cv2.putText(img, "DECODED STATE", (10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        
        # Arousal bar
        cv2.putText(img, "Arousal:", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        bar_w = int(self.decoded_arousal * 150)
        cv2.rectangle(img, (80, 200), (80 + bar_w, 215), (0, 100 + int(155 * self.decoded_arousal), 255), -1)
        cv2.rectangle(img, (80, 200), (230, 215), (100, 100, 100), 1)
        
        # State ID
        cv2.putText(img, f"State Cluster: {self.decoded_state_id}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1)
        
        # Cluster visualization
        if self.cluster_centers is not None and len(self.cluster_centers) > 0:
            cv2.putText(img, "State Clusters (D,T,A,B,G):", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            for i, center in enumerate(self.cluster_centers[:5]):
                y = 285 + i * 12
                # Normalize for display
                max_c = np.max(center)
                if max_c > 0:
                    c_norm = center / (max_c + 1e-6)
                else:
                    c_norm = center
                cv2.putText(img, f"{i}:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
                for j, val in enumerate(c_norm[:5]):
                    x = 30 + j * 30
                    bar_h = int(val * 10)
                    colors = [(100,100,255), (100,255,100), (255,255,100), (255,150,100), (255,100,255)]
                    cv2.rectangle(img, (x, y - bar_h), (x + 20, y), colors[j], -1)
        
        # Instructions
        cv2.putText(img, "Send probe_trigger > 0.5 to start", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        return img
    
    def _update_display(self):
        """Update display image."""
        img = self._render_decoder_display()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            h, w = img_rgb.shape[:2]
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, 
                               QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image
    
    def get_config_options(self):
        return [
            ("Freq Bins", "freq_bins", self.freq_bins, None),
            ("Probe Duration", "probe_duration", self.probe_duration, None),
            ("Num Clusters", "n_clusters", self.n_clusters, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)