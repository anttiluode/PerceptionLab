"""
Turbulence Field Node (Fixed - Direct EEG Loading)
--------------------------------------------------
Loads EEG data directly and computes the 64×64 interaction matrix.

Measures turbulence as the product of:
- Activity level (signal strength)
- Phase desynchrony (how out-of-phase channels are)
- Coherence (correlation strength)

High turbulence = channels fighting (high activity, poor coordination)
Low turbulence = channels synchronized (stable attractor)

This node reveals the interaction field that drives morphogenesis.
"""

import numpy as np
import cv2
import os
from collections import deque
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

try:
    import mne
    from scipy.signal import hilbert
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


class TurbulenceFieldNode(BaseNode):
    """
    Loads EEG and measures neural turbulence as channel interaction matrix.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(200, 100, 150)  # Pink-purple for turbulence
    
    def __init__(self, edf_file_path=""):
        super().__init__()
        self.node_title = "Turbulence Field"
        
        # No inputs - this node loads EEG directly
        self.inputs = {}
        
        self.outputs = {
            'turbulence_matrix': 'image',    # NxN heat map
            'turbulence_scalar': 'signal',   # Average turbulence
            'max_turbulence': 'signal',      # Peak turbulence
            'phase_field': 'image',          # Phase relationship map
            'dominant_mode': 'signal',       # Which interaction dominates
            # Also output band powers like the original loader
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
        }
        
        # Configuration
        self.edf_file_path = edf_file_path
        self.selected_region = "All"  # Use all channels by default
        self.window_size = 1.0        # 1-second window
        self.history_length = 100     # Samples for phase estimation
        self.smoothing_sigma = 1.0    # Gaussian smoothing
        self.fs = 100.0               # Resample to this frequency
        
        # Weights for turbulence calculation
        self.phase_weight = 0.4
        self.coherence_weight = 0.3
        self.activity_weight = 0.3
        
        # EEG loading
        self.raw = None
        self.current_time = 0.0
        self._last_path = ""
        self._last_region = ""
        self.num_channels = 0
        self.channel_names = []
        
        # State
        self.turbulence_matrix = None
        self.phase_matrix = None
        self.channel_history = deque(maxlen=self.history_length)
        
        # For visualization
        self.turbulence_scalar = 0.0
        self.max_turb = 0.0
        
        # Band powers for output
        self.band_powers = {
            'delta': 0.0, 'theta': 0.0, 'alpha': 0.0, 
            'beta': 0.0, 'gamma': 0.0
        }
        
        if not MNE_AVAILABLE:
            self.node_title = "Turbulence (MNE Required!)"
            print("Error: TurbulenceFieldNode requires 'mne' and 'scipy'.")
        
    def load_edf(self):
        """Loads or re-loads the EDF file based on config."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_file_path):
            self.raw = None
            self.num_channels = 0
            self.node_title = f"Turbulence (No File)"
            return

        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            # Select region if specified
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
            self.num_channels = len(raw.ch_names)
            self.channel_names = raw.ch_names
            self.current_time = 0.0
            
            # Initialize matrices
            self.turbulence_matrix = np.zeros((self.num_channels, self.num_channels), dtype=np.float32)
            self.phase_matrix = np.zeros((self.num_channels, self.num_channels), dtype=np.float32)
            
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            self.node_title = f"Turbulence ({self.num_channels}ch)"
            print(f"Successfully loaded EEG: {self.edf_file_path}")
            print(f"Channels: {self.num_channels}")
           
        except Exception as e:
            self.raw = None
            self.num_channels = 0
            self.node_title = f"Turbulence (Error)"
            print(f"Error loading EEG file {self.edf_file_path}: {e}")

    def step(self):
        # Check if config changed
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()

        if self.raw is None or self.num_channels == 0:
            # Decay outputs if no file
            self.turbulence_scalar *= 0.95
            self.max_turb *= 0.95
            for band in self.band_powers:
                self.band_powers[band] *= 0.95
            return

        # Get data for the current time window
        start_sample = int(self.current_time * self.fs)
        end_sample = start_sample + int(self.window_size * self.fs)
        
        if end_sample >= self.raw.n_times:
            self.current_time = 0.0  # Loop
            start_sample = 0
            end_sample = int(self.window_size * self.fs)
            
        data, _ = self.raw[:, start_sample:end_sample]  # Shape: (num_channels, samples)
        
        if data.size == 0:
            return
        
        # Store current channel values in history
        # Take the mean across the time window for each channel
        channel_snapshot = np.mean(data, axis=1)  # Shape: (num_channels,)
        self.channel_history.append(channel_snapshot)
        
        # Calculate band powers (average across all channels)
        self.calculate_band_powers(data)
        
        # Need enough history for turbulence calculation
        if len(self.channel_history) >= 10:
            self.compute_turbulence()
        
        # Increment time
        self.current_time += (1.0 / 30.0)  # Assume ~30fps step rate
        
    def calculate_band_powers(self, data):
        """Calculate band powers from multi-channel data"""
        from scipy import signal as scipy_signal
        
        # Average across channels for band power output
        if data.ndim > 1:
            data_avg = np.mean(data, axis=0)
        else:
            data_avg = data
            
        bands = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
            'beta': (13, 30), 'gamma': (30, 45)
        }
        
        nyq = self.fs / 2.0
        
        for band, (low, high) in bands.items():
            try:
                b, a = scipy_signal.butter(4, [low/nyq, high/nyq], btype='band')
                filtered = scipy_signal.filtfilt(b, a, data_avg)
                power = np.log1p(np.mean(filtered**2)) * 20.0
                
                # Smooth the output
                self.band_powers[band] = self.band_powers[band] * 0.8 + power * 0.2
            except:
                pass
    
    def compute_turbulence(self):
        """
        Compute the NxN turbulence interaction matrix.
        
        Turbulence[i,j] = Activity[i,j] × Desync[i,j] × Coherence[i,j]
        """
        history_array = np.array(self.channel_history)  # Shape: (history_length, num_channels)
        
        # Compute for each channel pair
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                if i == j:
                    # No self-interaction turbulence
                    self.turbulence_matrix[i, j] = 0
                    self.phase_matrix[i, j] = 0
                    continue
                
                signal_i = history_array[:, i]
                signal_j = history_array[:, j]
                
                # 1. ACTIVITY: Average signal strength
                activity_i = np.std(signal_i) + 1e-8
                activity_j = np.std(signal_j) + 1e-8
                activity = (activity_i + activity_j) / 2.0
                
                # 2. PHASE DESYNCHRONY: Using Hilbert transform
                try:
                    # Analytic signal for phase extraction
                    analytic_i = hilbert(signal_i)
                    analytic_j = hilbert(signal_j)
                    
                    phase_i = np.angle(analytic_i[-1])  # Most recent phase
                    phase_j = np.angle(analytic_j[-1])
                    
                    phase_diff = np.abs(phase_i - phase_j)
                    # Wrap to [0, π]
                    if phase_diff > np.pi:
                        phase_diff = 2 * np.pi - phase_diff
                    
                    # Convert to desynchrony: 0 if in-phase, 1 if anti-phase
                    desync = phase_diff / np.pi
                    
                    self.phase_matrix[i, j] = phase_diff
                    
                except:
                    # If Hilbert fails, use simple correlation phase
                    desync = 0.5
                    self.phase_matrix[i, j] = np.pi / 2
                
                # 3. COHERENCE: Correlation strength
                correlation = np.corrcoef(signal_i, signal_j)[0, 1]
                coherence = np.abs(correlation)
                
                # TURBULENCE: Weighted combination
                turb = (self.activity_weight * activity + 
                       self.phase_weight * desync + 
                       self.coherence_weight * coherence)
                
                self.turbulence_matrix[i, j] = turb
        
        # Smooth the matrix
        self.turbulence_matrix = gaussian_filter(self.turbulence_matrix, sigma=self.smoothing_sigma)
        
        # Compute summary statistics
        self.turbulence_scalar = float(np.mean(self.turbulence_matrix))
        self.max_turb = float(np.max(self.turbulence_matrix))
        
    def get_output(self, port_name):
        if port_name == 'turbulence_matrix':
            return self.turbulence_matrix if self.turbulence_matrix is not None else np.zeros((8, 8))
        elif port_name == 'turbulence_scalar':
            return self.turbulence_scalar
        elif port_name == 'max_turbulence':
            return self.max_turb
        elif port_name == 'phase_field':
            return self.phase_matrix if self.phase_matrix is not None else np.zeros((8, 8))
        elif port_name == 'dominant_mode':
            if self.turbulence_matrix is not None:
                channel_turb = np.sum(self.turbulence_matrix, axis=1)
                return float(np.argmax(channel_turb))
            return 0.0
        elif port_name in self.band_powers:
            return self.band_powers[port_name]
        return None
    
    def get_display_image(self):
        """
        4-panel visualization:
        Top-left: Turbulence matrix
        Top-right: Phase matrix  
        Bottom-left: Row sums (per-channel turbulence)
        Bottom-right: Statistics
        """
        w, h = 512, 512
        panel_size = 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # If no data yet
        if self.turbulence_matrix is None:
            cv2.putText(img, "Loading EEG...", (w//2 - 80, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
        # === PANEL 1: Turbulence Matrix ===
        turb_norm = cv2.normalize(self.turbulence_matrix, None, 0, 255, cv2.NORM_MINMAX)
        turb_u8 = turb_norm.astype(np.uint8)
        turb_color = cv2.applyColorMap(turb_u8, cv2.COLORMAP_HOT)
        turb_resized = cv2.resize(turb_color, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
        img[0:panel_size, 0:panel_size] = turb_resized
        
        # Label
        cv2.putText(img, f"TURBULENCE ({self.num_channels}x{self.num_channels})", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === PANEL 2: Phase Matrix ===
        phase_norm = cv2.normalize(self.phase_matrix, None, 0, 255, cv2.NORM_MINMAX)
        phase_u8 = phase_norm.astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_u8, cv2.COLORMAP_TWILIGHT)
        phase_resized = cv2.resize(phase_color, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
        img[0:panel_size, panel_size:] = phase_resized
        
        cv2.putText(img, "PHASE FIELD", (panel_size + 5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === PANEL 3: Per-Channel Turbulence (Bar Graph) ===
        channel_turb = np.sum(self.turbulence_matrix, axis=1)  # Sum across rows
        
        # Create bar graph
        bar_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        
        if np.max(channel_turb) > 0:
            channel_turb_norm = channel_turb / np.max(channel_turb)
            
            bar_width = max(1, panel_size // self.num_channels)
            for i in range(self.num_channels):
                height = int(channel_turb_norm[i] * (panel_size - 20))
                x = i * bar_width
                
                # Color based on intensity
                intensity = int(channel_turb_norm[i] * 255)
                color = (0, intensity, 255 - intensity)
                
                cv2.rectangle(bar_panel, 
                            (x, panel_size - height), 
                            (x + bar_width - 1, panel_size), 
                            color, -1)
        
        img[panel_size:, 0:panel_size] = bar_panel
        cv2.putText(img, "CHANNEL TURBULENCE", (5, panel_size + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === PANEL 4: Statistics ===
        stats_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        
        # Draw statistics text
        y_pos = 30
        line_height = 22
        
        stats = [
            f"Mean Turb: {self.turbulence_scalar:.4f}",
            f"Max Turb: {self.max_turb:.4f}",
            f"Channels: {self.num_channels}",
            f"Region: {self.selected_region}",
            f"History: {len(self.channel_history)}/{self.history_length}",
            "",
            "Band Powers:",
            f"  Delta: {self.band_powers['delta']:.2f}",
            f"  Theta: {self.band_powers['theta']:.2f}",
            f"  Alpha: {self.band_powers['alpha']:.2f}",
            f"  Beta: {self.band_powers['beta']:.2f}",
            f"  Gamma: {self.band_powers['gamma']:.2f}",
        ]
        
        for line in stats:
            cv2.putText(stats_panel, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_pos += line_height
        
        img[panel_size:, panel_size:] = stats_panel
        cv2.putText(img, "STATISTICS", (panel_size + 5, panel_size + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("History Length", "history_length", self.history_length, None),
            ("Smoothing", "smoothing_sigma", self.smoothing_sigma, None),
            ("Activity Weight", "activity_weight", self.activity_weight, None),
            ("Phase Weight", "phase_weight", self.phase_weight, None),
            ("Coherence Weight", "coherence_weight", self.coherence_weight, None),
        ]