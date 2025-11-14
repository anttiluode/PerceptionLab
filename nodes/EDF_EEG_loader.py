"""
EEG File Source Node - Loads a real .edf file and streams band power
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import os
import sys

# Add parent directory to path to import BaseNode
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

# Define brain regions from brain_set_system.py
EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}

class EEGFileSourceNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 140, 160) # A clinical blue
    
    def __init__(self, edf_file_path=""):
        super().__init__()
        self.node_title = "EEG File Source"
     
        self.outputs = {
            'delta': 'signal', 
            'theta': 'signal', 
            'alpha': 'signal', 
            'beta': 'signal', 
            'gamma': 'signal',
            # --- FIX: ADD NEW RAW SIGNAL OUTPUT ---
            'raw_signal': 'signal' 
        }
        
        self.edf_file_path = edf_file_path
        self.selected_region = "Occipital"
        self._last_path = ""
        self._last_region = ""
        
        self.raw = None
        self.fs = 100.0 # Resample to this frequency
        self.current_time = 0.0
        self.window_size = 1.0 # 1-second window
      
        self.output_powers = {band: 0.0 for band in self.outputs}
        self.output_powers['raw_signal'] = 0.0 # Initialize new output
        self.history = np.zeros(64) # For display

        if not MNE_AVAILABLE:
            self.node_title = "EEG (MNE Required!)"
            print("Error: EEGFileSourceNode requires 'mne' and 'scipy'.")
            print("Please run: pip install mne")

    def load_edf(self):
        """Loads or re-loads the EDF file based on config."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_file_path):
            self.raw = None
            self.node_title = f"EEG (File Not Found)"
            return

        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            if self.selected_region != "All":
                region_channels = EEG_REGIONS[self.selected_region]
                available_channels = [ch for ch in region_channels if ch in raw.ch_names]
                if not available_channels:
                    print(f"Warning: No channels found for region {self.selected_region}")
                    self.raw = None
                    return
                raw.pick_channels(available_channels)
            
            raw.resample(self.fs, verbose=False)
            self.raw = raw
            self.current_time = 0.0
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            self.node_title = f"EEG ({self.selected_region})"
            print(f"Successfully loaded EEG: {self.edf_file_path}")
           
        except Exception as e:
            self.raw = None
            self.node_title = f"EEG (Load Error)"
            print(f"Error loading EEG file {self.edf_file_path}: {e}")

    def step(self):
        # Check if config changed
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()

        if self.raw is None:
            # If no file is loaded, all outputs will decay to 0
            for band in self.output_powers:
                self.output_powers[band] *= 0.95
            self.history *= 0.95
            return # Do nothing if no data

        # Get data for the current time window
        start_sample = int(self.current_time * self.fs)
        end_sample = start_sample + int(self.window_size * self.fs)
        
        if end_sample >= self.raw.n_times:
            self.current_time = 0.0 # Loop
            start_sample = 0
            end_sample = int(self.window_size * self.fs)
            
        data, _ = self.raw[:, start_sample:end_sample]
        
        # Average across all selected channels
        if data.ndim > 1:
            data = np.mean(data, axis=0)

        if data.size == 0:
            return
            
        # --- FIX 1: Calculate raw_signal as total log power (not DC offset) ---
        raw_power = np.log1p(np.mean(data**2))
        # We give it a boost to make it visible
        self.output_powers['raw_signal'] = self.output_powers['raw_signal'] * 0.8 + (raw_power * 10.0) * 0.2

        # Calculate band powers 
        bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
            'beta': (13, 30), 'gamma': (30, 45)
        }
        
        nyq = self.fs / 2.0
        
        for band, (low, high) in bands.items():
            if band in self.outputs:
                b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
                filtered = signal.filtfilt(b, a, data)
                
                # --- FIX 2: Boost the band power *after* log1p ---
                # This makes it comparable to the raw signal's boost.
                # You can change 20.0 to a higher/lower number to adjust sensitivity.
                power = np.log1p(np.mean(filtered**2)) * 20.0 
                
                # Smooth the output
                self.output_powers[band] = self.output_powers[band] * 0.8 + power * 0.2
        
        # Update display history with alpha power
        self.history[:-1] = self.history[1:]
        # Use the newly scaled power for the display
        self.history[-1] = self.output_powers['alpha'] 
        
        # Increment time
        self.current_time += (1.0 / 30.0) # Assume ~30fps step rate

    def get_output(self, port_name):
        return self.output_powers.get(port_name, 0.0)
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Draw waveform (alpha history)
        vis_data = self.history
        vis_data = (vis_data - np.min(vis_data)) / (np.max(vis_data) - np.min(vis_data) + 1e-9)
        vis_data = vis_data * (h - 1)
        
        for i in range(w - 1):
            y1 = int(np.clip(vis_data[i], 0, h - 1))
            img[h - 1 - y1, i] = 255
            
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
        ]
