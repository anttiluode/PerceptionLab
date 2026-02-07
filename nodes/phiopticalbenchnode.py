"""
Phi Fractal Bench (Complete Implementation)
===========================================
"The Eye of the Beholder" - High-Performance Holographic Lens.

FIXED:
- Implemented Lazy Loading inside step() to sync with JSON restoration.
- Explicitly defined all grid attributes in __init__ to prevent AttributeErrors.
- Removed unstable Lifecycle Locks and QTimers.
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

class PhiOpticalBenchNode(BaseNode):
    NODE_CATEGORY = "Visualizers"
    NODE_COLOR = QtGui.QColor(220, 180, 50) # Optical Gold

    def __init__(self):
        super().__init__()
        self.node_title = "Phi Fractal Bench"
        
        self.inputs = {
            'speed': 'signal',
            'focus_z': 'signal',
            'lens_k': 'signal'
        }
        
        self.outputs = {
            'render': 'image'
        }

        # Parameters
        self.focus_z = 0.0   
        self.lens_k = 5.0    
        self.curvature = 0.5 
        self.res = 128 
        
        # State
        self.data = None
        self.sampling_rate = 160.0
        self.current_idx = 0
        self.n_ch = 0
        self.file_path = ""
        self._last_loaded_path = None # TRACKER FOR LAZY LOAD
        self.is_synthetic = False
        self.display_image = None
        
        # Geometry Initialization
        self.emitters = None
        self.pixel_x = None
        self.pixel_y = None
        self.grid_x = None
        self.grid_y = None
        
        # Build grid immediately so attributes exist
        self.rebuild_grid()
        
        # UI Elements
        self.lbl_status = QtWidgets.QLabel("No Source")
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px; margin-top: 4px;")
        self.btn_load = QtWidgets.QPushButton("LOAD EEG SOURCE")
        self.btn_load.setStyleSheet("""
            QPushButton { 
                background-color: #223344; color: #88ccff; 
                border: 1px solid #446688; padding: 6px; font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #334455; }
        """)
        self.btn_load.clicked.connect(self.load_file_dialog)

    def rebuild_grid(self):
        """Pre-calculate the coordinate matrices for ray tracing."""
        x = np.linspace(-1, 1, self.res)
        y = np.linspace(-1, 1, self.res)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.pixel_x = self.grid_x.reshape(1, self.res, self.res)
        self.pixel_y = self.grid_y.reshape(1, self.res, self.res)

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
            self.file_path = fname # Step() will detect the change and load

    def load_data(self, path):
        """Handles heavy EDF ingestion and filtering."""
        if not MNE_AVAILABLE:
            self.lbl_status.setText("MNE Missing")
            self.generate_synthetic_data()
            return

        try:
            self.lbl_status.setText("Parsing...")
            QtWidgets.QApplication.processEvents()
            
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            
            if 'eeg' in raw:
                raw.pick('eeg', exclude='bads')
            else:
                raw.pick(range(min(60, len(raw.ch_names))))

            raw.filter(8, 12, verbose=False) 
            
            self.data = raw.get_data() * 1e6 
            self.sampling_rate = raw.info['sfreq']
            self.n_ch = len(raw.ch_names)
            self.is_synthetic = False
            self.current_idx = 0
            
            self.rebuild_optics(self.n_ch)
            self.lbl_status.setText(f"READY: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"Load Error: {e}")
            self.lbl_status.setText("Error -> Synthetic")
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        self.is_synthetic = True
        self.n_ch = 64
        self.sampling_rate = 160.0
        duration = 10 
        t = np.linspace(0, duration, int(duration*self.sampling_rate))
        self.data = np.zeros((self.n_ch, len(t)))
        for i in range(self.n_ch):
            self.data[i] = np.sin(2*np.pi*10*t + i*0.1) * 20 
        self.rebuild_optics(self.n_ch)
        self.lbl_status.setText("MODE: SYNTHETIC")

    def rebuild_optics(self, n_ch):
        if n_ch <= 0: return
        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        r = 0.8
        ex = r * np.cos(theta)
        ey = r * np.sin(theta)
        ez = np.full_like(ex, -self.curvature)
        self.emitters = np.stack((ex, ey, ez), axis=1)

    def step(self):
        # LAZY LOAD TRIGGER: Handles both Fresh and JSON load
        if self.file_path and self.file_path != self._last_loaded_path:
            self._last_loaded_path = self.file_path
            if os.path.exists(self.file_path):
                self.load_data(self.file_path)

        # Defensive Guard: Ensure data exists
        if self.data is None:
            self.render_placeholder()
            return
        
        if self.grid_x is None: self.rebuild_grid()

        # Inputs
        speed = self.get_blended_input('speed', 'max') or 1.0
        self.current_idx += int(speed * 2)
        if self.current_idx >= self.data.shape[1] - 64: self.current_idx = 0

        z_sig = self.get_blended_input('focus_z', 'max')
        k_sig = self.get_blended_input('lens_k', 'max')
        if z_sig is not None: self.focus_z = z_sig * 5.0
        if k_sig is not None: self.lens_k = k_sig * 50.0

        # Physics
        window = 64 
        start = self.current_idx
        end = start + window
        segment = self.data[:, start:end]
        fft_res = np.fft.rfft(segment, axis=1)
        freqs = np.fft.rfftfreq(window, d=1/self.sampling_rate)
        
        mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(mask): mask[len(freqs)//10] = True
        complex_field = np.sum(fft_res[:, mask], axis=1) 

        # Ray Tracing Interference Field
        total_field = np.zeros((self.res, self.res), dtype=np.complex64)
        if self.emitters is None or len(self.emitters) == 0: return

        for i in range(self.n_ch):
            ex, ey, ez = self.emitters[i]
            dx = self.pixel_x - ex
            dy = self.pixel_y - ey
            dz = self.focus_z - ez
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            
            waves = complex_field[i] * np.exp(-1j * self.lens_k * dists)
            waves *= (1.0 / (dists + 0.1))
            total_field += np.sum(waves, axis=0)

        # Rendering
        intensity = np.abs(total_field)
        i_max = np.max(intensity) + 1e-9
        norm = (intensity / i_max * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        
        cv2.putText(heatmap, f"Z: {self.focus_z:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(heatmap, f"K: {self.lens_k:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        status_c = (0, 255, 0) if not self.is_synthetic else (0, 0, 255)
        cv2.circle(heatmap, (10, 240), 4, status_c, -1)

        out_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.display_image = out_rgb

    def render_placeholder(self):
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(canvas, "LOAD EEG", (85, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
        self.display_image = canvas 
        self.outputs['render'] = canvas

    def get_display_image(self):
        return self.display_image

    def get_config_options(self):
        return [
            ("Focal Depth", "focus_z", self.focus_z, "float"),
            ("Lens K", "lens_k", self.lens_k, "float"),
            ("File Path", "file_path", self.file_path, "str")
        ]

    def set_config_options(self, options):
        if 'focus_z' in options: self.focus_z = float(options['focus_z'])
        if 'lens_k' in options: self.lens_k = float(options['lens_k'])
        if 'file_path' in options: 
            self.file_path = str(options['file_path'])

    def close(self):
        self.data = None