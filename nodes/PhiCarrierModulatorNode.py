"""
Phi Carrier-Modulator Node (Final Load Fix)
==========================================
"The Architecture of Thought" - Updated for Perception Lab V9+

FIXED:
- Removed QTimer race condition.
- Implemented Lazy-Loading in step() to sync with JSON restoration.
- Added render_placeholder for empty states.
"""

import numpy as np
import cv2
import os
from PyQt6 import QtWidgets, QtGui, QtCore

# Try to import MNE
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
except Exception:
    class BaseNode:
        def __init__(self): self.inputs={}; self.outputs={}

class PhiCarrierModulatorNode(BaseNode):
    NODE_CATEGORY = "Visualizers"
    NODE_COLOR = QtGui.QColor(0, 140, 220) # Electric Blue

    def __init__(self):
        super().__init__()
        self.node_title = "Carrier / Modulator"
        
        self.inputs = {'speed': 'signal'}
        self.outputs = {'render': 'image'}

        # Physics Constants
        self.k_carrier = 0.05
        self.k_modulator = 5.0
        
        # Render Config
        self.res = 256
        self.file_path = ""
        self._last_loaded_path = None # TRACKER FOR LAZY LOAD
        self.is_synthetic = False
        
        # Internal State
        self.data = None
        self.times = None
        self.sampling_rate = 160.0
        self.n_ch = 0
        self.dists = None
        self.current_idx = 0
        self.display_image = None
        
        # UI Elements
        self.lbl_status = QtWidgets.QLabel("No Source Loaded")
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px; margin-top: 4px;")
        self.btn_load = QtWidgets.QPushButton("LOAD EEG SOURCE")
        self.btn_load.setStyleSheet("""
            QPushButton { 
                background-color: #223344; color: #88ccff; border: 1px solid #446688;
                padding: 6px; font-weight: bold; border-radius: 4px;
            }
            QPushButton:hover { background-color: #334455; }
        """)
        self.btn_load.clicked.connect(self.load_file_dialog)
        
        # Precompute Grid
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(-1, 1, self.res), 
            np.linspace(-1, 1, self.res)
        )

    def get_custom_widget(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        l.setContentsMargins(4,4,4,4)
        l.addWidget(self.btn_load)
        l.addWidget(self.lbl_status)
        w.setLayout(l)
        return w

    def load_file_dialog(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.btn_load, "Select EEG Source", "", "EDF Files (*.edf);;All Files (*)"
        )
        if fname:
            self.file_path = fname # Setting this triggers the lazy load in step()

    def load_data(self, path):
        if not MNE_AVAILABLE:
            self.lbl_status.setText("MNE Missing -> Synthetic")
            self.generate_synthetic_data()
            return

        try:
            self.lbl_status.setText("Parsing Physics...")
            QtWidgets.QApplication.processEvents()
            
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.pick('eeg', exclude='bads') if 'eeg' in raw else raw.pick(range(min(60, len(raw.ch_names))))
            raw.filter(1, 60, verbose=False) 
            
            self.data = raw.get_data() * 1e6 
            self.times = raw.times
            self.sampling_rate = raw.info['sfreq']
            self.n_ch = len(raw.ch_names)
            self.is_synthetic = False
            self.current_idx = 0
            
            self.rebuild_geometry(self.n_ch)
            self.lbl_status.setText(f"REAL: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"File Load Error: {e}")
            self.lbl_status.setText("Load Error -> Synthetic")
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        self.is_synthetic = True
        self.n_ch = 64
        self.sampling_rate = 160.0
        duration = 10 
        t = np.linspace(0, duration, int(duration*self.sampling_rate))
        self.data = np.zeros((self.n_ch, len(t)))
        for i in range(self.n_ch):
            self.data[i] = np.sin(2 * np.pi * 10 * t) * 20 + np.random.normal(0, 2, len(t))
        self.rebuild_geometry(self.n_ch)
        self.lbl_status.setText("MODE: SYNTHETIC")

    def rebuild_geometry(self, n_ch):
        if n_ch <= 0: return
        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        el_x, el_y = 0.8 * np.cos(theta), 0.8 * np.sin(theta)
        self.dists = np.zeros((n_ch, self.res, self.res), dtype=np.float32)
        for i in range(n_ch):
            self.dists[i] = np.sqrt((self.x_grid - el_x[i])**2 + (self.y_grid - el_y[i])**2)

    def step(self):
        # LAZY LOAD TRIGGER
        if self.file_path and self.file_path != self._last_loaded_path:
            self._last_loaded_path = self.file_path
            if os.path.exists(self.file_path):
                self.load_data(self.file_path)

        if self.data is None or self.dists is None: 
            self.render_placeholder()
            return

        # Playback Logic
        speed = self.get_blended_input('speed', 'max') or 1.0
        self.current_idx += int(speed * 2)
        if self.current_idx >= self.data.shape[1] - 128: self.current_idx = 0
        
        segment = self.data[:, self.current_idx:self.current_idx + 128]
        fft_res = np.fft.rfft(segment, axis=1)
        freqs = np.fft.rfftfreq(128, d=1/self.sampling_rate)
        
        c_mask = (freqs >= 8) & (freqs <= 12)
        m_mask = (freqs >= 30) & (freqs <= 50)
        
        c_coeffs = np.sum(fft_res[:, c_mask], axis=1)
        m_coeffs = np.sum(fft_res[:, m_mask], axis=1)

        # Holographic Render
        mag_c = np.abs(np.sum(np.exp(1j * (np.angle(c_coeffs)[:, None, None] - self.dists * self.k_carrier)), axis=0))
        mag_m = np.abs(np.sum(np.exp(1j * (np.angle(m_coeffs)[:, None, None] - self.dists * self.k_modulator)), axis=0))**2
        
        layer_c = cv2.applyColorMap((mag_c / (np.max(mag_c)+1e-9) * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        layer_m = cv2.applyColorMap((mag_m / (np.max(mag_m)+1e-9) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        final = cv2.addWeighted(layer_c.astype(np.float32), 0.7, layer_m.astype(np.float32), 0.9, 0)
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        cv2.putText(final, f"MODE: {'REAL' if not self.is_synthetic else 'SYNTH'}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        out_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.display_image = out_rgb

    def render_placeholder(self):
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(canvas, "LOAD EEG", (85, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
        self.display_image = canvas 
        self.outputs['render'] = canvas

    def get_display_image(self): return self.display_image

    def get_config_options(self):
        return [
            ("Carrier K", "k_carrier", self.k_carrier, "float"),
            ("Modulator K", "k_modulator", self.k_modulator, "float"),
            ("File Path", "file_path", self.file_path, "str")
        ]

    def set_config_options(self, options):
        if 'k_carrier' in options: self.k_carrier = float(options['k_carrier'])
        if 'k_modulator' in options: self.k_modulator = float(options['k_modulator'])
        if 'file_path' in options: self.file_path = str(options['file_path'])