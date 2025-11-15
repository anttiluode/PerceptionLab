"""
EEG Experiment Node (All-in-One)
Loads a single .edf file and performs the full Sensation vs. Prediction experiment.

Combines the logic of:
1. DualStreamEEGNode (to get all band powers)
2. Two LatentAssemblerNodes (to package signals into vectors)

It outputs the two final, synchronized 'spectrum' vectors (orange ports)
ready to be plugged into the analyzer nodes.
"""
import cv2

import numpy as np
from PyQt6 import QtGui
import os
import sys

# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

try:
    import mne
    from scipy import signal
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# Define brain regions
EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}

class EEGExperimentNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 140, 160) # A clinical blue
    
    # Define the 6 components
    BANDS_LIST = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'raw_signal']
    # Add the full latents as options
    SOURCE_OPTIONS = BANDS_LIST + ['fast_latent_full', 'slow_latent_full']
    
    def __init__(self, 
                 edf_file_path="", 
                 selected_region="Occipital", 
                 slow_momentum=0.9,
                 fast_stream_source='raw_signal',
                 slow_stream_source='alpha',
                 latent_dim=6,
                 signal_gain=1.0,
                 raw_power_scale=10.0,
                 band_power_scale=20.0):
        super().__init__()
        self.node_title = "EEG Experiment"
     
        self.outputs = {
            'fast_stream_out': 'spectrum',  # Sensation
            'slow_stream_out': 'spectrum'   # Prediction
        }
        
        self.edf_file_path = edf_file_path
        self.selected_region = selected_region
        self.slow_momentum = float(slow_momentum)
        self.fast_stream_source = fast_stream_source
        self.slow_stream_source = slow_stream_source
        self.latent_dim = int(latent_dim)
        self.signal_gain = float(signal_gain)
        self.raw_power_scale = float(raw_power_scale)
        self.band_power_scale = float(band_power_scale)
        
        self._last_path = ""
        self._last_region = ""
        
        self.raw = None
        self.fs = 100.0
        self.current_time = 0.0
        self.window_size = 1.0
      
        # Internal state dictionaries
        self.fast_latent_powers = {band: 0.0 for band in self.BANDS_LIST}
        self.slow_latent_powers = {band: 0.0 for band in self.BANDS_LIST}
        
        # Output vectors
        self.fast_stream_vector = np.zeros(self.latent_dim, dtype=np.float32)
        self.slow_stream_vector = np.zeros(self.latent_dim, dtype=np.float32)

        if not MNE_AVAILABLE:
            self.node_title = "EEG (MNE Required!)"

    def load_edf(self):
        """Loads or re-loads the EDF file based on config."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_file_path):
            self.raw = None; self.node_title = f"EEG (File Not Found)"; return
        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            if self.selected_region != "All":
                region_channels = EEG_REGIONS[self.selected_region]
                available_channels = [ch for ch in region_channels if ch in raw.ch_names]
                if not available_channels:
                    print(f"Warning: No channels found for region {self.selected_region}"); self.raw = None; return
                raw.pick_channels(available_channels)
            raw.resample(self.fs, verbose=False)
            self.raw = raw; self.current_time = 0.0
            self._last_path = self.edf_file_path; self._last_region = self.selected_region
            self.node_title = f"EEG ({self.selected_region})"
            print(f"Successfully loaded EEG: {self.edf_file_path}")
        except Exception as e:
            self.raw = None; self.node_title = f"EEG (Load Error)"; print(f"Error loading EEG file {self.edf_file_path}: {e}")

    def step(self):
        # Check if config changed
        if (self.edf_file_path != self._last_path or 
            self.selected_region != self._last_region or 
            self.raw is None):
            self.load_edf()

        if self.raw is None:
            self.fast_stream_vector *= 0.95
            self.slow_stream_vector *= 0.95
            return

        # Get data for the current time window
        start_sample = int(self.current_time * self.fs); end_sample = start_sample + int(self.window_size * self.fs)
        if end_sample >= self.raw.n_times:
            self.current_time = 0.0; start_sample = 0; end_sample = int(self.window_size * self.fs)
        data, _ = self.raw[:, start_sample:end_sample]
        if data.ndim > 1: data = np.mean(data, axis=0)
        if data.size == 0: return
            
        # --- 1. Calculate ALL band powers (Fast Latent) ---
        raw_power = np.log1p(np.mean(data**2))
        self.fast_latent_powers['raw_signal'] = self.fast_latent_powers['raw_signal'] * 0.8 + (raw_power * self.raw_power_scale) * 0.2
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
        nyq = self.fs / 2.0
        for band, (low, high) in bands.items():
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            filtered = signal.filtfilt(b, a, data)
            power = np.log1p(np.mean(filtered**2)) * self.band_power_scale
            self.fast_latent_powers[band] = self.fast_latent_powers[band] * 0.8 + power * 0.2
        
        # --- 2. Calculate Slow Latent (Prediction) ---
        for band in self.BANDS_LIST:
            fast_val = self.fast_latent_powers.get(band, 0.0)
            slow_val = self.slow_latent_powers.get(band, 0.0)
            self.slow_latent_powers[band] = (slow_val * self.slow_momentum + fast_val * (1.0 - self.slow_momentum))
        
        # --- 3. Assemble Output Vectors ---
        self.fast_stream_vector = self._assemble_vector(self.fast_stream_source) * self.signal_gain
        self.slow_stream_vector = self._assemble_vector(self.slow_stream_source) * self.signal_gain
        
        self.current_time += (1.0 / 30.0)

    def _assemble_vector(self, source_name):
        """Helper to create an output vector based on the selected source."""
        output_vec = np.zeros(self.latent_dim, dtype=np.float32)
        
        if source_name in self.BANDS_LIST:
            # Single signal mode (like LatentAssembler)
            val = self.fast_latent_powers.get(source_name, 0.0)
            if self.latent_dim > 0:
                output_vec[0] = val # Put the signal in the first slot
        
        elif source_name == 'fast_latent_full':
            # Full 6-band vector mode
            full_vec = np.array([self.fast_latent_powers[band] for band in self.BANDS_LIST], dtype=np.float32)
            self._resize_vector(full_vec, output_vec) # Resize to fit output_dim
            
        elif source_name == 'slow_latent_full':
            # Full 6-band SLOW vector mode
            full_vec = np.array([self.slow_latent_powers[band] for band in self.BANDS_LIST], dtype=np.float32)
            self._resize_vector(full_vec, output_vec) # Resize to fit output_dim
            
        return output_vec

    def _resize_vector(self, vec, target_vec):
        """Pads or truncates a vector to fit in the target vector."""
        current_dim = len(vec)
        target_dim = len(target_vec)
        if current_dim == target_dim:
            target_vec[:] = vec
        elif current_dim > target_dim:
            target_vec[:] = vec[:target_dim] # Truncate
        else:
            target_vec[:current_dim] = vec # Pad

    def get_output(self, port_name):
        if port_name == 'fast_stream_out':
            return self.fast_stream_vector
        elif port_name == 'slow_stream_out':
            return self.slow_stream_vector
        return None
        
    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw Fast Vector (Top)
        self._draw_vector(img, self.fast_stream_vector, "Fast Stream", (0, 200, 200), 0)
        # Draw Slow Vector (Bottom)
        self._draw_vector(img, self.slow_stream_vector, "Slow Stream", (200, 200, 0), h // 2)

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def _draw_vector(self, img, vector, label, color, y_offset):
        w, h = img.shape[1], img.shape[0] // 2
        
        if vector is None or len(vector) == 0:
            return

        bar_width = max(1, w // len(vector))
        val_max = np.abs(vector).max()
        if val_max < 1e-6: val_max = 1.0
        
        for i, val in enumerate(vector):
            x = i * bar_width
            norm_val = val / val_max
            bar_h = int(abs(norm_val) * (h - 20))
            y_base = y_offset + h // 2 + 5
            
            if val >= 0:
                cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color, -1)
            else:
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color, -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, (5, y_offset + 15), font, 0.4, color, 1)

    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        # Create dropdown options for source selection
        source_dropdown_options = []
        for name in self.SOURCE_OPTIONS:
            source_dropdown_options.append((name.replace("_", " ").title(), name))
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, "file_open"),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("Slow Momentum", "slow_momentum", self.slow_momentum, None),
            ("Output Latent Dim", "latent_dim", self.latent_dim, None),
            ("Fast Stream Source", "fast_stream_source", self.fast_stream_source, source_dropdown_options),
            ("Slow Stream Source", "slow_stream_source", self.slow_stream_source, source_dropdown_options),
            ("Signal Gain", "signal_gain", self.signal_gain, None),
            ("Raw Power Scale", "raw_power_scale", self.raw_power_scale, None),
            ("Band Power Scale", "band_power_scale", self.band_power_scale, None),
        ]