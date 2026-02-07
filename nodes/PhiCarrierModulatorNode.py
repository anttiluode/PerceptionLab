"""
Phi Carrier-Modulator Node (Full Production)
============================================
"The Architecture of Thought"

Visualizes the brain as a Dual-Layer Holographic System.

LAYER 1: THE CARRIER (The Ocean) - k=0.05
- Represents Global Coherence / Alpha Waves.
- Rendered in Deep Blue/Teal (Ocean Colormap).

LAYER 2: THE MODULATOR (The Lightning) - k=5.0
- Represents Local Computation / Gamma Waves.
- Rendered in Electric Gold (Jet Colormap) with additive blending.

FEATURES:
- FULL MNE LOADER: Parses .EDF files, strips junk channels automatically.
- SYNTHETIC CORE: Generates phantom data if file load fails.
- OUTPUT FIX: Fully integrated with PerceptionLab wiring and display.

Usage:
1. Click "LOAD EEG SOURCE".
2. If successful, HUD shows "MODE: REAL EEG".
3. Connect [render] output to a Display Node or Screen Node.
"""

import numpy as np
import cv2
import os
from PyQt6 import QtWidgets, QtGui, QtCore

# Try to import MNE for medical file support
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
        
        self.inputs = {
            'speed': 'signal'
        }
        
        self.outputs = {
            'render': 'image'
        }

        # Physics Constants
        self.k_carrier = 0.05
        self.k_modulator = 5.0
        
        # Render Config
        self.res = 256
        self.file_path = ""
        self.is_synthetic = False
        
        # Internal State
        self.data = None
        self.times = None
        self.sampling_rate = 160.0
        self.n_ch = 0
        self.dists = None
        self.current_idx = 0
        
        # Display Buffer (Critical for Host UI)
        self.display_image = None
        
        # UI Elements
        self.lbl_status = QtWidgets.QLabel("No Source Loaded")
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px; margin-top: 4px;")
        self.btn_load = QtWidgets.QPushButton("LOAD EEG SOURCE")
        self.btn_load.setStyleSheet("""
            QPushButton { 
                background-color: #223344; 
                color: #88ccff; 
                border: 1px solid #446688;
                padding: 6px;
                font-weight: bold;
                border-radius: 4px;
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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select EEG Source", "", "EDF Files (*.edf);;All Files (*)")
        if fname:
            self.load_data(fname)

    def load_data(self, path):
        if not MNE_AVAILABLE:
            self.lbl_status.setText("MNE Missing -> Synthetic")
            self.generate_synthetic_data()
            return

        try:
            self.lbl_status.setText("Parsing Physics...")
            QtWidgets.QApplication.processEvents()
            
            # 1. Load Data
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            
            # 2. CRITICAL FIX: Robust Channel Picking
            # This logic prevents crashes by ignoring 'Stimulus' or 'Annotation' tracks
            if 'eeg' in raw:
                raw.pick_types(eeg=True, exclude='bads')
            else:
                # Fallback: Just grab the first 60 channels if types aren't defined
                raw.pick(range(min(60, len(raw.ch_names))))

            # 3. Filter (1-60Hz) to remove drift and mains noise
            raw.filter(1, 60, verbose=False) 
            
            self.data = raw.get_data() * 1e6 # Convert to microvolts
            self.times = raw.times
            self.sampling_rate = raw.info['sfreq']
            self.n_ch = len(raw.ch_names)
            self.file_path = path
            self.is_synthetic = False
            
            self.rebuild_geometry(self.n_ch)
            self.lbl_status.setText(f"REAL: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"File Load Error: {e}")
            self.lbl_status.setText("Load Error -> Synthetic")
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """Generates a mathematically perfect phantom brain for testing."""
        print("Generating Synthetic Data...")
        self.is_synthetic = True
        self.n_ch = 64
        self.sampling_rate = 160.0
        duration = 10 # seconds
        t = np.linspace(0, duration, int(duration*self.sampling_rate))
        
        self.data = np.zeros((self.n_ch, len(t)))
        for i in range(self.n_ch):
            # Carrier: 10Hz Alpha Wave (Global Sync)
            alpha = np.sin(2 * np.pi * 10 * t) * 20
            
            # Modulator: 40Hz Gamma Wave (Traveling)
            phase_lag = i * 0.2
            gamma = np.sin(2 * np.pi * 40 * t + phase_lag) * 10
            
            # Add Pulsing Envelope to Gamma
            pulse = (np.sin(2 * np.pi * 0.5 * t) + 1)
            
            self.data[i] = alpha + (gamma * pulse) + np.random.normal(0, 2, len(t))

        self.rebuild_geometry(self.n_ch)
        self.lbl_status.setText("MODE: SYNTHETIC")

    def rebuild_geometry(self, n_ch):
        """Builds the holographic emitter array."""
        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        el_x = 0.8 * np.cos(theta)
        el_y = 0.8 * np.sin(theta)
        
        self.dists = np.zeros((n_ch, self.res, self.res), dtype=np.float32)
        for i in range(n_ch):
            self.dists[i] = np.sqrt((self.x_grid - el_x[i])**2 + (self.y_grid - el_y[i])**2)

    def step(self):
        # 1. Handle Missing Data
        if self.data is None: 
            canvas = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(canvas, "LOAD EEG", (70, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
            self.display_image = canvas 
            self.outputs['render'] = canvas
            return

        # 2. Playback Control
        speed = self.get_blended_input('speed', 'max')
        if speed is None: speed = 1.0
        self.current_idx += int(speed * 2)

        # Loop Logic
        if self.current_idx >= self.data.shape[1]: self.current_idx = 0
        
        # 3. Physics Window
        window = 128
        start = self.current_idx
        end = start + window
        if end > self.data.shape[1]: 
            self.current_idx = 0
            return
        
        segment = self.data[:, start:end]
        
        # 4. Spectral Decomposition (FFT)
        fft_res = np.fft.rfft(segment, axis=1)
        freqs = np.fft.rfftfreq(window, d=1/self.sampling_rate)
        
        # 5. Band Extraction
        # Carrier (Alpha 8-12Hz)
        carrier_mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(carrier_mask): carrier_mask[len(freqs)//10] = True
        
        # Modulator (Gamma 30-50Hz)
        modulator_mask = (freqs >= 30) & (freqs <= 50)
        if not np.any(modulator_mask): modulator_mask[len(freqs)//3] = True
        
        carrier_coeffs = np.sum(fft_res[:, carrier_mask], axis=1)
        modulator_coeffs = np.sum(fft_res[:, modulator_mask], axis=1)

        # 6. Render Carrier (The Ocean)
        phases_c = np.angle(carrier_coeffs)
        holo_c = np.sum(np.exp(1j * (phases_c[:, None, None] - self.dists * self.k_carrier)), axis=0)
        mag_c = np.abs(holo_c)
        c_max = np.max(mag_c) + 1e-9
        norm_c = (mag_c / c_max * 255).astype(np.uint8)
        layer_carrier = cv2.applyColorMap(norm_c, cv2.COLORMAP_OCEAN)
        
        # 7. Render Modulator (The Lightning)
        phases_m = np.angle(modulator_coeffs)
        holo_m = np.sum(np.exp(1j * (phases_m[:, None, None] - self.dists * self.k_modulator)), axis=0)
        mag_m = np.abs(holo_m)
        
        # Sharpening Filter for Gamma
        mag_m = mag_m ** 2 
        
        m_max = np.max(mag_m) + 1e-9
        norm_m = (mag_m / m_max * 255).astype(np.uint8)
        layer_modulator = cv2.applyColorMap(norm_m, cv2.COLORMAP_JET)
        
        # 8. Composite (Bicameral View)
        final = layer_carrier.astype(np.float32)
        mod_float = layer_modulator.astype(np.float32)
        
        # Additive Blend: 70% Ocean + 90% Lightning
        final = cv2.addWeighted(final, 0.7, mod_float, 0.9, 0)
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # 9. Heads-Up Display (HUD)
        status_color = (0, 255, 0) if not self.is_synthetic else (0, 0, 255)
        status_text = "MODE: REAL EEG" if not self.is_synthetic else "MODE: SYNTHETIC"
        cv2.putText(final, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        cv2.putText(final, f"CARRIER (k={self.k_carrier})", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
        cv2.putText(final, f"MODULATOR (k={self.k_modulator})", (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)

        # 10. Output to Wire & Host
        out_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.display_image = out_rgb

    # --- CRITICAL CONNECTIVITY METHODS ---
    def get_output(self, port_name):
        return self.outputs.get(port_name, None)

    def get_display_image(self):
        return self.display_image

    def get_config_options(self):
        return [
            ("Carrier K", "k_carrier", self.k_carrier, "float"),
            ("Modulator K", "k_modulator", self.k_modulator, "float"),
            ("Last File", "file_path", self.file_path, "str")
        ]

    def set_config_options(self, options):
        if 'k_carrier' in options: self.k_carrier = float(options['k_carrier'])
        if 'k_modulator' in options: self.k_modulator = float(options['k_modulator'])
        if 'file_path' in options: 
            fp = str(options['file_path'])
            if fp and os.path.exists(fp):
                self.load_data(fp)