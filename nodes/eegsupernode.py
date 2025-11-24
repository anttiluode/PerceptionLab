"""
EEG Super-Loader (Standardized)
-------------------------------
1. Loads .EDF files via standard Host File Picker.
2. If no file is loaded, generates SYNTHETIC NOISE (Mock Mode).
3. Amplifies and Projects to any Latent Size.
"""

import numpy as np
import os
import sys
from scipy import signal

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

# Try to import MNE for reading EDF files
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: 'mne' not found. Using Mock Mode.")

class EEGSUPERFileSourceNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 140, 160) # Clinical Blue

    def __init__(self):
        super().__init__()
        self.node_title = "EEG Super-Loader"
        
        self.inputs = {
            'amplification': 'signal'
        }
        
        self.outputs = {
            'latent_vector': 'spectrum',
            'raw_alpha': 'signal',
            'raw_beta': 'signal',
            'status': 'signal'
        }
        
        # Config
        self.edf_file = ""
        self.output_dim = 16
        self.sampling_rate = 256
        self.chunk_size = 32
        
        # Internal State
        self.raw_data = None
        self.num_channels = 0
        self.current_index = 0
        self.total_samples = 0
        self.playback_speed = 1.0
        
        # Filters & Projection
        self.filters = {}
        self.band_ranges = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
            'beta': (13, 30), 'gamma': (30, 100)
        }
        self.projection_matrix = None
        self.output_vector = np.zeros(self.output_dim)
        self.current_bands = np.zeros(5)
        
        # Init
        self.init_projection()
        self.init_filters()

        if not MNE_AVAILABLE:
            self.node_title = "EEG (MNE Missing)"
            self.NODE_COLOR = QtGui.QColor(200, 50, 50) # Red warning

    def load_edf(self, filepath):
        if not MNE_AVAILABLE:
            print("Error: Install 'mne' to load real files (pip install mne)")
            return

        if not os.path.exists(filepath):
            return
            
        try:
            # Load Data
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            
            # Pick channels
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            if len(picks) == 0: picks = range(len(raw.ch_names))
                
            self.raw_data = raw.get_data(picks=picks)
            self.sampling_rate = int(raw.info['sfreq'])
            self.num_channels, self.total_samples = self.raw_data.shape
            self.edf_file = filepath  # Store full path
            
            # Reset
            self.current_index = 0
            self.init_filters()
            filename = os.path.basename(filepath)
            self.node_title = f"EEG: {filename}"
            self.NODE_COLOR = QtGui.QColor(50, 200, 100) # Green Success
            print(f"Loaded: {filename} ({self.total_samples} samples)")
            
        except Exception as e:
            print(f"Error loading EDF: {e}")
            self.node_title = "EEG Load Error"

    def init_filters(self):
        nyq = 0.5 * self.sampling_rate
        for band, (low, high) in self.band_ranges.items():
            if high >= nyq: high = nyq - 0.1
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            self.filters[band] = (b, a)

    def init_projection(self):
        # 5 Bands -> N Outputs
        self.projection_matrix = np.random.randn(self.output_dim, 5)
        self.projection_matrix /= np.sqrt(5)
        self.output_vector = np.zeros(self.output_dim)

    def step(self):
        gain = self.get_blended_input('amplification', 'sum')
        if gain is None: gain = 1.0

        # --- MODE 1: REAL FILE ---
        if self.raw_data is not None:
            start = int(self.current_index)
            end = start + self.chunk_size
            
            if end >= self.total_samples:
                self.current_index = 0
                start = 0
                end = self.chunk_size
                
            chunk = self.raw_data[:, start:end]
            self.current_index += self.chunk_size * self.playback_speed
            
            # Average channels
            avg_signal = np.mean(chunk, axis=0)
            
            # Filter Bands
            band_powers = []
            for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                if len(avg_signal) > 10:
                    try:
                        b, a = self.filters[band_name]
                        filtered = signal.filtfilt(b, a, avg_signal)
                        power = np.sqrt(np.mean(filtered**2))
                    except: power = 0.0
                else: power = 0.0
                band_powers.append(power)
            
            self.current_bands = np.array(band_powers)

        # --- MODE 2: MOCK DATA (If no file loaded) ---
        else:
            # Generate random "Brain-like" noise
            noise = np.random.rand(5) * 0.1
            noise[2] += 0.5 # Boost Alpha
            self.current_bands = noise * gain

        # --- PROJECTION ---
        if self.projection_matrix.shape[0] != self.output_dim:
            self.init_projection()
            
        projected = np.dot(self.projection_matrix, self.current_bands)
        self.output_vector = np.tanh(projected * gain * 5.0)

    def get_output(self, port_name):
        if port_name == 'latent_vector':
            return self.output_vector
        elif port_name == 'raw_alpha':
            return self.current_bands[2] * 10.0
        elif port_name == 'raw_beta':
            return self.current_bands[3] * 10.0
        elif port_name == 'status':
            return 1.0 if self.raw_data is not None else 0.0
        return None

    def get_config_options(self):
        # Uses "file_open" type to trigger Host OS file picker
        return [
            ("EDF File", "edf_file", self.edf_file, "file_open"),
            ("Output Size", "output_dim", int(self.output_dim), None),
            ("Speed", "playback_speed", float(self.playback_speed), None)
        ]
        
    def set_config_options(self, options):
        # Handle File Loading
        if "edf_file" in options:
            new_path = options["edf_file"]
            # Only trigger load if path changed or is not empty
            if new_path and new_path != self.edf_file:
                self.load_edf(new_path)
            
        if "output_dim" in options:
            self.output_dim = int(options["output_dim"])
            self.init_projection()
            
        if "playback_speed" in options:
            self.playback_speed = float(options["playback_speed"])