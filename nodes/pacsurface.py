"""
Phase-Amplitude Coupling (PAC) Surface Node
-------------------------------------------
Visualizes Cross-Frequency Coupling by plotting High-Frequency (Gamma) power
against Low-Frequency (Theta) phase. This reveals the "Neural Syntax".

Inputs:
- raw_eeg: The raw EEG signal
- theta_phase: Pre-calculated theta phase (optional, can self-calculate)

Outputs:
- pac_surface: Image showing the coupling pattern
- modulation_index: Signal representing strength of coupling
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque
from scipy.signal import hilbert, butter, filtfilt

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------

class PACSurfaceNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(200, 100, 150) # Magenta
    
    def __init__(self, phase_bins=36, history_len=500):
        super().__init__()
        self.node_title = "PAC Surface (Syntax)"
        
        self.inputs = {
            'raw_eeg': 'signal',
            'theta_phase': 'signal' # Optional external phase
        }
        
        self.outputs = {
            'pac_surface': 'image',
            'modulation_index': 'signal'
        }
        
        self.n_bins = int(phase_bins)
        self.history_len = int(history_len)
        
        # Buffers
        self.signal_buffer = deque(maxlen=self.history_len)
        
        # PAC State
        self.amplitude_bins = np.zeros(self.n_bins)
        self.bin_counts = np.zeros(self.n_bins)
        self.modulation_index = 0.0
        
        self.surface_img = np.zeros((128, 256, 3), dtype=np.uint8)
        
    def _extract_phase_amp(self, signal_arr):
        """Extract Theta Phase and Gamma Amplitude from signal"""
        # Theta (4-8 Hz)
        b_theta, a_theta = butter(3, [4/50, 8/50], btype='band') # Assumes 100Hz fs
        theta_filt = filtfilt(b_theta, a_theta, signal_arr)
        theta_analytic = hilbert(theta_filt)
        theta_phase = np.angle(theta_analytic)
        
        # Gamma (30-45 Hz)
        b_gamma, a_gamma = butter(3, [30/50, 45/50], btype='band')
        gamma_filt = filtfilt(b_gamma, a_gamma, signal_arr)
        gamma_analytic = hilbert(gamma_filt)
        gamma_amp = np.abs(gamma_analytic)
        
        return theta_phase, gamma_amp

    def step(self):
        sig_in = self.get_blended_input('raw_eeg', 'sum')
        
        if sig_in is None:
            return
            
        self.signal_buffer.append(sig_in)
        
        if len(self.signal_buffer) < 100:
            return
            
        # Convert buffer to array
        sig_arr = np.array(self.signal_buffer)
        
        # Calculate Phase/Amp
        # (In a real real-time system, we'd optimize filters, 
        # but for the node step size this batch processing of history is okay)
        theta_phase, gamma_amp = self._extract_phase_amp(sig_arr)
        
        # We only care about the most recent points for the update
        # But for stability, we re-bin the whole history window
        
        self.amplitude_bins.fill(0)
        self.bin_counts.fill(0)
        
        # Map phases (-pi to pi) to bins (0 to n_bins-1)
        # phase + pi -> 0..2pi
        bin_indices = ((theta_phase + np.pi) / (2 * np.pi) * self.n_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        # Accumulate
        np.add.at(self.amplitude_bins, bin_indices, gamma_amp)
        np.add.at(self.bin_counts, bin_indices, 1)
        
        # Average
        mean_amps = np.zeros_like(self.amplitude_bins)
        mask = self.bin_counts > 0
        mean_amps[mask] = self.amplitude_bins[mask] / self.bin_counts[mask]
        
        # Calculate Modulation Index (KL Divergence from uniform)
        # Normalize distribution
        if np.sum(mean_amps) > 0:
            p = mean_amps / np.sum(mean_amps)
            h = -np.sum(p[p>0] * np.log(p[p>0]))
            h_max = np.log(self.n_bins)
            self.modulation_index = (h_max - h) / h_max
        
        # Visualization
        self._draw_surface(mean_amps)

    def _draw_surface(self, mean_amps):
        self.surface_img.fill(0)
        h, w, _ = self.surface_img.shape
        
        # Draw the phase-amplitude curve
        # x = phase bin, y = mean amplitude
        
        max_amp = np.max(mean_amps) + 1e-9
        
        pts = []
        for i in range(self.n_bins):
            x = int(i / self.n_bins * w)
            y = int(h - (mean_amps[i] / max_amp * (h - 20)) - 10)
            pts.append([x, y])
            
            # Draw bars
            color_val = int(mean_amps[i] / max_amp * 255)
            cv2.rectangle(self.surface_img, (x, y), (x + w//self.n_bins, h), (color_val, 100, 255-color_val), -1)
            
        # Draw smooth curve
        if len(pts) > 1:
            cv2.polylines(self.surface_img, [np.array(pts)], False, (255, 255, 255), 2)
            
        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.surface_img, f"MI: {self.modulation_index:.4f}", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(self.surface_img, "-PI", (5, h-5), font, 0.4, (200, 200, 200), 1)
        cv2.putText(self.surface_img, "+PI", (w-30, h-5), font, 0.4, (200, 200, 200), 1)

    def get_output(self, port_name):
        if port_name == 'pac_surface':
            return self.surface_img.astype(np.float32) / 255.0
        elif port_name == 'modulation_index':
            return self.modulation_index
        return None

    def get_display_image(self):
        return QtGui.QImage(self.surface_img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Phase Bins", "n_bins", self.n_bins, None),
            ("History Length", "history_len", self.history_len, None)
        ]