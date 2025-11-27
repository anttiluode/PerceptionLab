"""
EEG File Source Node – 16-Band Frequency Split + Raw
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import os
import sys

# Correct way to import BaseNode from the PA way
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

try:
    import mne
    from scipy import signal
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# Same brain regions as before
EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}

# 16 evenly-spaced 1-Hz-wide bands from 1–45 Hz (plus a final broad gamma if you want)
BANDS_16 = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 30), (30, 45)
]
BAND_NAMES_16 = [f"band_{low}_{high}Hz" for low, high in BANDS_16]


class EEGFileSourceNode16(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(60, 140, 160)  # Clinical blue

    def __init__(self, edf_file_path=""):
        super().__init__()
        self.node_title = "EEG 16-Band Source"

        # 16 band outputs + raw
        self.outputs = {name: 'signal' for name in BAND_NAMES_16}
        self.outputs['raw_signal'] = 'signal'

        self.edf_file_path = edf_file_path
        self.selected_region = "Occipital"
        self._last_path = ""
        self._last_region = ""

        self.raw = None
        self.fs = 100.0                     # Resample target
        self.current_time = 0.0
        self.window_size = 1.0              # 1-second analysis window

        # One value per output port
        self.output_powers = {key: 0.0 for key in self.outputs}
        self.history = np.zeros(64)         # For mini waveform display (still uses one band)

        if not MNE_AVAILABLE is False:
            self.node_title = "EEG (MNE Required!)"
            print("Error: This node requires 'mne' and 'scipy'. Run: pip install mne scipy")

    def load_edf(self):
        if not MNE_AVAILABLE or not os.path.exists(self.edf_file_path):
            self.raw = None
            self.node_title = "EEG (File Not Found)"
            return

        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            raw.rename_channels(lambda x: x.strip().replace('.', '').upper())

            if self.selected_region != "All":
                region_ch = EEG_REGIONS[self.selected_region]
                available = [ch for ch in region_ch if ch in raw.ch_names]
                if not available:
                    print(f"No channels for region {self.selected_region}")
                    self.raw = None
                    return
                raw.pick_channels(available)

            raw.resample(self.fs, verbose=False)
            self.raw = raw
            self.current_time = 0.0
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            self.node_title = f"EEG 16-Band ({self.selected_region})"
            print(f"Loaded: {self.edf_file_path} → {len(raw.ch_names)} ch @ {self.fs}Hz")

        except Exception as e:
            self.raw = None
            self.node_title = "EEG (Load Error)"
            print(f"EDF load error: {e}")

    def step(self):
        # Reload if config changed
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()

        if self.raw is None:
            return

        start_s = int(self.current_time * self.fs)
        end_s = start_s + int(self.window_size * self.fs)

        if end_s >= self.raw.n_times:
            self.current_time = 0.0
            start_s = 0
            end_s = int(self.window_size * self.fs)

        data, _ = self.raw[:, start_s:end_s]

        # Average across channels
        if data.ndim > 1:
            data = np.mean(data, axis=0)

        if data.size == 0:
            return

        # Raw signal output (scaled for visibility in PA)
        self.output_powers['raw_signal'] = float(np.mean(data)) * 5.0

        nyq = self.fs / 2.0

        # Compute power for each of the 16 bands
        for band_name, (low, high) in zip(BAND_NAMES_16, BANDS_16):
            b, a = signal.butter(4, [low / nyq, high / nyq], btype='band')
            filtered = signal.filtfilt(b, a, data)
            power = np.log1p(np.mean(filtered ** 2))          # log(1+x) for smoother scale

            # Exponential smoothing (same feel as original node)
            self.output_powers[band_name] = (self.output_powers[band_name] * 0.8 +
                                             power * 0.2)

        # Update mini-display with, say, 8–9 Hz band (classic alpha) or any you prefer
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.output_powers['band_8_9Hz'] * 0.5

        # Advance time (~30 fps)
        self.current_time += 1.0 / 30.0

    def get_output(self, port_name):
        return self.output_powers.get(port_name, 0.0)

    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)

        vis = self.history.copy()
        if vis.ptp() > 0:
            vis = (vis - vis.min()) / vis.ptp()
        vis = vis * (h - 1)

        for i in range(w - 1):
            y = int(np.clip(vis[i], 0, h - 1))
            img[h - 1 - y, i] = 255

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS]
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
        ]