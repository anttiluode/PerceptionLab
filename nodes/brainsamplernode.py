"""
Brain Sampler Node - The Unified Solution
=========================================
Based on the insight: "The box sides are the key."

ARCHITECTURE:
1. Temporal Detection: Find theta windows (boxes) in frontal channel
2. Key Extraction: The box transition (corner) = sampling moment
   - Key = velocity vector (dx/dt, dy/dt) in phase space
3. Content Sampling: When key "turns" (high velocity), sample ALL channels
4. Multi-Scale Analysis: Analyze sampled content (bands, attractors, features)
5. Output: Stream of "percepts" captured at box corners

The box doesn't contain qualia. The box IS the sampling clock.
The corners are when sampling happens. The sides are what's held.
"""

import numpy as np
import cv2
from scipy.signal import butter, lfilter, hilbert
from scipy.ndimage import gaussian_filter
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None
    from PyQt6 import QtGui

class BrainSamplerNode(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Brain Sampler (Box Key)"
    NODE_COLOR = QtGui.QColor(0, 200, 150)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'frontal_theta': 'signal',    # The clock (drives sampling)
            'full_eeg': 'spectrum',       # The content (what to sample)
            'sample_threshold': 'signal'  # Modulate sensitivity
        }
        
        self.outputs = {
            'sample_snapshot': 'image',   # Current sample visualization
            'sample_trigger': 'signal',   # 1.0 when sampling, 0.0 when holding
            'attractor_state': 'signal',  # Box/Star classification
            'sample_stream': 'spectrum',  # History of samples
            'spatial_field': 'image',     # The filtered spatial pattern (the "sun")
            'complex_field': 'complex_spectrum'  # Complex FFT for interference
        }
        
        # Config
        self.fs = 256.0
        self.buffer_len = 256
        self.min_corner_velocity = 0.1  # Minimum |dψ/dt| to trigger sample
        
        # State
        self.theta_buffer = deque(maxlen=self.buffer_len)
        self.eeg_history = deque(maxlen=self.buffer_len)
        self.last_phase_point = (0.0, 0.0)
        self.sample_buffer = []  # List of captured samples
        
        # Current sample outputs
        self.current_spatial_field = None  # The raw spatial pattern
        self.current_complex_field = None  # FFT of spatial for interference
        
        # Visualization
        self.current_sample_viz = np.zeros((200, 300, 3), dtype=np.uint8)
        self.is_sampling = False
        
        # Filters
        self.theta_filter = butter(3, [4, 8], btype='band', fs=self.fs, output='ba')
        self.band_filters = self._make_band_filters()
        
    def _make_band_filters(self):
        bands = {
            'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
            'Beta': (13, 30), 'Gamma': (30, 45)
        }
        filters = {}
        for name, (low, high) in bands.items():
            filters[name] = butter(3, [low, high], btype='band', fs=self.fs, output='ba')
        return filters
    
    def step(self):
        # 1. Get inputs
        theta_sig = self.get_blended_input('frontal_theta', 'sum')
        eeg_array = self.get_blended_input('full_eeg', 'sum')
        
        if theta_sig is None: theta_sig = 0.0
        if eeg_array is None: eeg_array = np.zeros(16)
        
        # Ensure array
        if not isinstance(eeg_array, np.ndarray):
            eeg_array = np.array([eeg_array])
        
        # Store in buffers
        self.theta_buffer.append(theta_sig)
        self.eeg_history.append(eeg_array.copy())
        
        if len(self.theta_buffer) < 64:
            return  # Not enough data yet
        
        # 2. DETECT BOX CORNER (The Key Turning)
        
        # Filter theta
        theta_arr = np.array(self.theta_buffer)
        b, a = self.theta_filter
        theta_filt = lfilter(b, a, theta_arr)
        
        # Get phase space coordinates (Takens embedding)
        delay = 15  # ~150ms at 100Hz, adjust for your fs
        if len(theta_filt) > delay:
            x = theta_filt[-1]
            y = theta_filt[-delay-1]
            
            # Calculate velocity (the KEY)
            dx = x - self.last_phase_point[0]
            dy = y - self.last_phase_point[1]
            self.last_phase_point = (x, y)
            
            # Key magnitude = corner sharpness
            key_velocity = np.sqrt(dx**2 + dy**2)
            
            # Threshold
            thresh = self.min_corner_velocity
            thresh_mod = self.get_blended_input('sample_threshold', 'sum')
            if thresh_mod is not None:
                thresh += thresh_mod * 0.1
            
            # SAMPLE TRIGGER: High velocity = corner = sample now!
            self.is_sampling = key_velocity > thresh
            
            if self.is_sampling:
                # 3. CAPTURE SAMPLE (When key turns)
                self._capture_sample(eeg_array, key_velocity)
        else:
            self.is_sampling = False
    
    def _capture_sample(self, eeg_array, key_strength):
        """
        Called when we hit a box corner.
        Analyzes the current EEG state across multiple scales.
        """
        
        # Get recent EEG history for analysis
        if len(self.eeg_history) < 128:
            return
        
        # Stack into array (time x channels)
        eeg_matrix = np.array(list(self.eeg_history)[-128:])
        
        # A. FREQUENCY ANALYSIS (Which bands are active?)
        band_powers = {}
        for name, (b, a) in self.band_filters.items():
            # Average across channels
            avg_signal = np.mean(eeg_matrix, axis=1)
            filtered = lfilter(b, a, avg_signal)
            power = np.std(filtered[-32:])  # RMS of recent window
            band_powers[name] = power
        
        # B. ATTRACTOR CLASSIFICATION
        total_power = sum(band_powers.values()) + 1e-9
        theta_ratio = band_powers['Theta'] / total_power
        high_ratio = (band_powers['Beta'] + band_powers['Gamma']) / total_power
        
        if total_power < 0.01:
            attractor_type = 0  # Silence
        elif theta_ratio > 0.45:
            attractor_type = 1  # BOX (theta dominant)
        elif 0.3 < high_ratio < 0.6:
            attractor_type = 2  # STAR (balanced)
        else:
            attractor_type = 3  # CHAOS
        
        # C. SPATIAL PATTERN (Simple 2D projection)
        # Map channels to 2D grid for visualization
        n_ch = len(eeg_array)
        grid_size = int(np.ceil(np.sqrt(n_ch)))
        spatial_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        for i, val in enumerate(eeg_array):
            r = i // grid_size
            c = i % grid_size
            if r < grid_size and c < grid_size:
                spatial_map[r, c] = val
        
        # Smooth and normalize
        spatial_map = gaussian_filter(spatial_map, sigma=0.8)
        if np.max(np.abs(spatial_map)) > 1e-9:
            spatial_map = spatial_map / np.max(np.abs(spatial_map))
        
        # STORE spatial field for output
        self.current_spatial_field = spatial_map
        
        # COMPUTE complex FFT for interference
        # Pad to power of 2 for efficiency
        target_size = 64
        if spatial_map.shape[0] < target_size:
            padded = np.zeros((target_size, target_size), dtype=np.float32)
            padded[:spatial_map.shape[0], :spatial_map.shape[1]] = spatial_map
        else:
            padded = cv2.resize(spatial_map, (target_size, target_size))
        
        # FFT (shift zero-frequency to center)
        complex_fft = np.fft.fftshift(np.fft.fft2(padded))
        self.current_complex_field = complex_fft
        
        # D. CREATE SAMPLE RECORD
        sample = {
            'timestamp': len(self.sample_buffer),
            'key_strength': key_strength,
            'band_powers': band_powers,
            'attractor': attractor_type,
            'spatial': spatial_map,
            'channels': eeg_array.copy()
        }
        
        self.sample_buffer.append(sample)
        if len(self.sample_buffer) > 100:
            self.sample_buffer.pop(0)
        
        # E. VISUALIZE
        self._render_sample(sample)
    
    def _render_sample(self, sample):
        """Create visualization of the captured sample."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # LEFT: Band powers as bars
        powers = sample['band_powers']
        band_names = list(powers.keys())
        n_bands = len(band_names)
        
        bar_width = 40
        spacing = 10
        
        for i, name in enumerate(band_names):
            power = powers[name]
            height = int(power * 150)
            x = 10 + i * (bar_width + spacing)
            y = 180
            
            # Color by band
            colors = {
                'Delta': (100, 100, 200),
                'Theta': (100, 200, 100),
                'Alpha': (200, 200, 100),
                'Beta': (200, 100, 100),
                'Gamma': (200, 100, 200)
            }
            color = colors.get(name, (150, 150, 150))
            
            cv2.rectangle(img, (x, y), (x+bar_width, y-height), color, -1)
            cv2.putText(img, name[0], (x+5, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # RIGHT: Spatial pattern
        spatial = sample['spatial']
        spatial_viz = ((spatial + 1) * 127.5).astype(np.uint8)
        spatial_viz = cv2.resize(spatial_viz, (100, 100))
        spatial_color = cv2.applyColorMap(spatial_viz, cv2.COLORMAP_VIRIDIS)
        img[20:120, 190:290] = spatial_color
        
        # TOP: Attractor state
        att_type = sample['attractor']
        att_names = ['VOID', 'BOX', 'STAR', 'CHAOS']
        att_colors = [(50,50,50), (0,255,100), (255,215,0), (255,50,50)]
        
        cv2.putText(img, att_names[att_type], (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, att_colors[att_type], 2)
        
        # Key strength indicator
        key_bar = int(sample['key_strength'] * 280)
        cv2.rectangle(img, (10, 195), (10+key_bar, 200), (0,255,255), -1)
        
        self.current_sample_viz = img
    
    def get_output(self, port_name):
        if port_name == 'sample_snapshot':
            return self.current_sample_viz
        
        elif port_name == 'sample_trigger':
            return 1.0 if self.is_sampling else 0.0
        
        elif port_name == 'attractor_state':
            if len(self.sample_buffer) > 0:
                return float(self.sample_buffer[-1]['attractor'])
            return 0.0
        
        elif port_name == 'sample_stream':
            # Return array of recent attractor states
            if len(self.sample_buffer) > 0:
                return np.array([s['attractor'] for s in self.sample_buffer])
            return np.zeros(1)
        
        elif port_name == 'spatial_field':
            # The raw spatial pattern (the "sun")
            if self.current_spatial_field is not None:
                # Return as image (0-255)
                viz = ((self.current_spatial_field + 1) * 127.5).astype(np.uint8)
                return cv2.applyColorMap(viz, cv2.COLORMAP_VIRIDIS)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        elif port_name == 'complex_field':
            # The FFT for interference experiments
            if self.current_complex_field is not None:
                return self.current_complex_field
            return np.zeros((64, 64), dtype=np.complex128)
        
        return None
    
    def get_display_image(self):
        # Create dashboard
        display = np.zeros((200, 600, 3), dtype=np.uint8)
        
        # Left: Current sample
        display[:, :300] = self.current_sample_viz
        
        # Right: Sample history
        if len(self.sample_buffer) > 0:
            # Plot attractor trajectory
            history = [s['attractor'] for s in self.sample_buffer[-50:]]
            
            for i in range(len(history)-1):
                x1 = 320 + i * 5
                y1 = 100 - int(history[i] * 30)
                x2 = 320 + (i+1) * 5
                y2 = 100 - int(history[i+1] * 30)
                
                color_map = {0: (50,50,50), 1: (0,255,100), 
                           2: (255,215,0), 3: (255,50,50)}
                color = color_map.get(history[i], (150,150,150))
                
                cv2.line(display, (x1, y1), (x2, y2), color, 2)
        
        # Status
        status = "SAMPLING" if self.is_sampling else "HOLDING"
        status_col = (0,255,255) if self.is_sampling else (100,100,100)
        cv2.putText(display, status, (320, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 2)
        
        cv2.putText(display, f"Samples: {len(self.sample_buffer)}", (320, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        return display
    
    def get_config_options(self):
        return [
            ("Sample Rate (Hz)", "fs", 256.0, "float"),
            ("Corner Threshold", "min_corner_velocity", 0.1, "float"),
            ("Phase Delay (samples)", "delay", 15, "int")
        ]