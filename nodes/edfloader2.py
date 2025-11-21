"""
EDF EEG Loader Node - Holographic Analysis (Fixed for v6 Host)
--------------------------------------------------------------
Loads .edf files and computes channel-to-channel interference (coherence).
Compatible with perception_lab_hostv6.py architecture.

Outputs:
- signal: Vector of all channel values at current time (spectrum).
- interference: 2D Correlation matrix image (The Hologram).
- gamma_phase: Instantaneous phase of global Gamma (30-90Hz).
"""

import numpy as np
import cv2
import os
import sys

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui
# -----------------------------

try:
    import mne
    from scipy.signal import butter, filtfilt, hilbert
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: EDFLoaderNode requires 'mne' and 'scipy'.")

class EDFLoaderNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 140, 160) # Clinical Blue
    
    def __init__(self, file_path="", window_ms=100, speed=1.0):
        super().__init__()
        self.node_title = "EDF Holographic Loader"
        
        # --- v6 Architecture: Define ports directly ---
        self.inputs = {
            'trigger': 'signal',      # 1.0 to restart/sync
            'speed_mod': 'signal'     # Modulate playback speed
        }
        
        self.outputs = {
            'signal': 'spectrum',       # All channels at t (Vector)
            'interference': 'image',    # Correlation Matrix (Hologram)
            'gamma_phase': 'signal'     # Global Gamma Phase (0-1)
        }
        
        # --- Configuration ---
        self.file_path = file_path
        self.window_ms = float(window_ms)
        self.speed = float(speed)
        
        # --- Internal State ---
        self.raw = None
        self.data = None
        self.times = None
        self.sfreq = 0
        self.current_sample = 0
        
        self._last_path = ""
        self.cached_matrix = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Output buffers
        self.out_signal = np.zeros(16, dtype=np.float32)
        self.out_interference = np.zeros((64, 64), dtype=np.float32)
        self.out_gamma = 0.0
        
        if not MNE_AVAILABLE:
            self.node_title = "EDF Loader (Libs Missing!)"

    def get_config_options(self):
        """Defines the Right-Click -> Configure menu"""
        return [
            ("EDF File", "file_path", self.file_path, "file_open"),
            ("Window (ms)", "window_ms", self.window_ms, None),
            ("Speed", "speed", self.speed, None),
        ]

    def load_edf(self):
        """Loads the EDF file using MNE"""
        if not MNE_AVAILABLE or not os.path.exists(self.file_path):
            self.raw = None
            self.node_title = "EDF (No File)"
            return

        try:
            # Load data
            self.raw = mne.io.read_raw_edf(self.file_path, preload=True, verbose=False)
            
            # Basic clean up: Pick EEG channels if possible, or just first 64
            picks = mne.pick_types(self.raw.info, eeg=True, meg=False, stim=False, exclude='bads')
            if len(picks) == 0:
                picks = range(min(64, len(self.raw.ch_names)))
                
            self.raw.pick(picks)
            
            # Convert to uV and get data array
            self.data = self.raw.get_data() * 1e6 
            self.times = self.raw.times
            self.sfreq = self.raw.info['sfreq']
            self.current_sample = 0
            
            self.node_title = f"EDF: {os.path.basename(self.file_path)}"
            self._last_path = self.file_path
            print(f"Loaded EDF: {self.data.shape[0]} channels, {self.data.shape[1]} samples")
            
        except Exception as e:
            print(f"EDF Load Error: {e}")
            self.node_title = "EDF (Error)"
            self.raw = None

    def _compute_interference(self, chunk):
        """Calculates correlation matrix (The Hologram)"""
        if chunk.size == 0: return np.zeros((1,1))
        
        # Center data
        chunk_centered = chunk - np.mean(chunk, axis=1, keepdims=True)
        
        # Correlation: (N, T) @ (T, N) -> (N, N)
        # This shows how every channel resonates with every other channel
        try:
            cov = np.corrcoef(chunk_centered)
            cov = np.nan_to_num(cov, nan=0.0)
            return cov
        except Exception:
            return np.zeros((chunk.shape[0], chunk.shape[0]))

    def _extract_gamma_phase(self, chunk):
        """Extracts phase of 30-90Hz band from first channel"""
        if chunk.shape[1] < 10: return 0.0
        
        try:
            nyq = 0.5 * self.sfreq
            low, high = 30.0 / nyq, 90.0 / nyq
            b, a = butter(4, [low, high], btype='band')
            
            # Use first channel
            filtered = filtfilt(b, a, chunk[0, :])
            analytic = hilbert(filtered)
            phase = np.angle(analytic[-1]) # Phase at most recent sample
            
            # Normalize -pi..pi to 0..1
            return (phase + np.pi) / (2 * np.pi)
        except Exception:
            return 0.0

    def step(self):
        if not MNE_AVAILABLE: return

        # 1. Check Config / Load File
        if self.file_path != self._last_path:
            self.load_edf()
            
        if self.raw is None: return

        # 2. Handle Inputs
        reset = self.get_blended_input('trigger', 'sum')
        speed_mod = self.get_blended_input('speed_mod', 'sum') or 0.0
        
        if reset is not None and reset > 0.5:
            self.current_sample = 0
            
        # 3. Advance Time
        step_size = int(self.sfreq * 0.033 * self.speed * (1.0 + speed_mod)) # ~30fps
        self.current_sample += step_size
        
        window_samples = int((self.window_ms / 1000.0) * self.sfreq)
        
        # Loop if end reached
        if self.current_sample + window_samples >= self.data.shape[1]:
            self.current_sample = 0
            
        # 4. Extract Chunk
        start = self.current_sample
        end = start + window_samples
        chunk = self.data[:, start:end]
        
        # 5. Compute Holographic Data
        self.out_interference = self._compute_interference(chunk)
        self.out_gamma = self._extract_gamma_phase(chunk)
        
        # 6. Output Signal (Current State Vector)
        # Return the last sample of the chunk as the instantaneous vector
        current_vec = chunk[:, -1]
        # Normalize for the system (uV can be large, map to approx -1..1)
        self.out_signal = np.clip(current_vec / 50.0, -1.0, 1.0).astype(np.float32)
        
        # 7. Update Visualization Cache
        self._update_vis(chunk)

    def _update_vis(self, chunk):
        """Render the interference matrix and raw waves"""
        w, h = 128, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Top Half: Interference Matrix (The Hologram)
        matrix_sz = 64
        if self.out_interference.shape[0] > 0:
            # Map -1..1 to 0..255
            norm_mat = (self.out_interference + 1.0) / 2.0
            norm_mat = np.clip(norm_mat, 0, 1)
            
            mat_u8 = (norm_mat * 255).astype(np.uint8)
            mat_color = cv2.applyColorMap(mat_u8, cv2.COLORMAP_JET)
            mat_resized = cv2.resize(mat_color, (w, 64), interpolation=cv2.INTER_NEAREST)
            img[0:64, :] = mat_resized
            
        # Bottom Half: Raw Waves (First 8 channels)
        if chunk.shape[1] > 1:
            n_ch = min(8, chunk.shape[0])
            chunk_len = chunk.shape[1]
            
            for i in range(n_ch):
                sig = chunk[i, :]
                # Simple normalization
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
                sig = np.clip(sig, -2, 2)
                
                # Map to pixel coordinates
                y_offset = 64 + (i * (64 // n_ch)) + (32 // n_ch)
                pts = []
                for t in range(0, w, 2): # Subsample width
                    idx = int((t / w) * chunk_len)
                    val = sig[idx]
                    y = int(y_offset - val * 3)
                    pts.append((t, y))
                
                # Draw line
                for j in range(1, len(pts)):
                    cv2.line(img, pts[j-1], pts[j], (200, 255, 200), 1)

        self.cached_matrix = img

    def get_output(self, port_name):
        if port_name == 'signal':
            return self.out_signal
        elif port_name == 'interference':
            # Return float matrix 0..1
            return (self.out_interference + 1.0) / 2.0
        elif port_name == 'gamma_phase':
            return self.out_gamma
        return None
        
    def get_display_image(self):
        # Return cached visualization
        if self.cached_matrix is None: return None
        
        img = self.cached_matrix
        
        # Add Gamma Indicator
        gamma_col = int(self.out_gamma * 255)
        cv2.rectangle(img, (0, 124), (int(self.out_gamma*128), 128), (gamma_col, 255-gamma_col, 255), -1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)