"""
Phi Fractal Bench (Standalone Source)
=====================================
"The Eye of the Beholder"

High-Performance Holographic Lens with DIRECT EDF LOADING.
Allows you to sweep the Holographic Lens from the Macro (Universe) to the Micro (Pixel)
using raw EEG data directly.

CONTROLS:
- Focus Z: Moves the observation plane through the depth of the brain.
- Lens K (Zoom): Changes the spatial frequency.
    - Low K (0.05) = Global Binding.
    - High K (50+) = Fractal Texture / Cortical Columns.

FEATURES:
- FULL MNE LOADER: Parses .EDF files directly.
- SYNTHETIC FALLBACK: Generates phantom data if load fails.
- BATCHED MATH: Prevents system freeze.
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
            'speed': 'signal',      # Playback Speed
            'focus_z': 'signal',    # Dynamic Depth
            'lens_k': 'signal'      # Dynamic Scale / Zoom
        }
        
        self.outputs = {
            'render': 'image'
        }

        # Optical Parameters
        self.focus_z = 1.0   
        self.lens_k = 10.0   
        self.curvature = 0.5 
        self.res = 128 # Default resolution
        
        # State
        self.n_ch = 0
        self.emitters = None    # (N, 3)
        self.display_image = None # Buffer
        
        # Data State
        self.data = None
        self.times = None
        self.sampling_rate = 160.0
        self.current_idx = 0
        self.file_path = ""
        self.is_synthetic = False
        
        # UI Elements
        self.lbl_status = QtWidgets.QLabel("No Source Loaded")
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px; margin-top: 4px;")
        self.btn_load = QtWidgets.QPushButton("LOAD EEG SOURCE")
        self.btn_load.setStyleSheet("""
            QPushButton { 
                background-color: #443322; 
                color: #ffcc88; 
                border: 1px solid #886644;
                padding: 6px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #554433; }
        """)
        self.btn_load.clicked.connect(self.load_file_dialog)
        
        # Grid Setup
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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select EEG Source", "", "EDF Files (*.edf);;All Files (*)")
        if fname:
            self.load_data(fname)

    def load_data(self, path):
        if not MNE_AVAILABLE:
            self.lbl_status.setText("MNE Missing -> Synthetic")
            self.generate_synthetic_data()
            return

        try:
            self.lbl_status.setText("Parsing Geometry...")
            QtWidgets.QApplication.processEvents()
            
            # 1. Load Data
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            
            # 2. Pick Channels
            if 'eeg' in raw:
                raw.pick_types(eeg=True, exclude='bads')
            else:
                raw.pick(range(min(60, len(raw.ch_names))))

            # 3. Filter
            raw.filter(1, 60, verbose=False) 
            
            self.data = raw.get_data() * 1e6 
            self.times = raw.times
            self.sampling_rate = raw.info['sfreq']
            self.n_ch = len(raw.ch_names)
            self.file_path = path
            self.is_synthetic = False
            
            self.rebuild_optics(self.n_ch)
            self.lbl_status.setText(f"REAL: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"File Load Error: {e}")
            self.lbl_status.setText("Load Error -> Synthetic")
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """Generates a phantom brain signal."""
        print("Generating Synthetic Data...")
        self.is_synthetic = True
        self.n_ch = 64
        self.sampling_rate = 160.0
        duration = 10 
        t = np.linspace(0, duration, int(duration*self.sampling_rate))
        
        self.data = np.zeros((self.n_ch, len(t)))
        for i in range(self.n_ch):
            # Complex interference pattern
            freq = 8 + (i % 4) * 2 # Varying frequencies
            self.data[i] = np.sin(2 * np.pi * freq * t + i*0.5) * 20
            
        self.rebuild_optics(self.n_ch)
        self.lbl_status.setText("MODE: SYNTHETIC")

    def rebuild_optics(self, n_ch):
        if n_ch == 0: return

        # 1. Build the Concave Lens (Scalp Geometry)
        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        r = 0.8
        ex = r * np.cos(theta)
        ey = r * np.sin(theta)
        ez = np.full_like(ex, -self.curvature) # Negative Z = Concave
        
        self.emitters = np.stack((ex, ey, ez), axis=1) # (N, 3)
        print(f"[Optics] Geometry rebuilt for {n_ch} channels.")

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

        if self.current_idx >= self.data.shape[1]: self.current_idx = 0
        
        # 3. Physics Window
        window = 128
        start = self.current_idx
        end = start + window
        if end > self.data.shape[1]: 
            self.current_idx = 0
            return
        
        segment = self.data[:, start:end]
        
        # 4. Get Field (Alpha Band Focus)
        fft_res = np.fft.rfft(segment, axis=1)
        freqs = np.fft.rfftfreq(window, d=1/self.sampling_rate)
        
        # Focus on Alpha/Beta for structure (8-20Hz)
        mask = (freqs >= 8) & (freqs <= 20)
        if not np.any(mask): mask[len(freqs)//10] = True
        
        field = np.sum(fft_res[:, mask], axis=1) # Complex weights

        # 5. Update Controls
        z_sig = self.get_blended_input('focus_z', 'max')
        k_sig = self.get_blended_input('lens_k', 'max')
        
        if z_sig is not None: self.focus_z = z_sig * 5.0
        if k_sig is not None: self.lens_k = k_sig * 50.0

        # 6. Batched Ray Tracing
        total_field = np.zeros((self.res, self.res), dtype=np.complex64)
        batch_size = 16 
        
        for start_i in range(0, self.n_ch, batch_size):
            end_i = min(start_i + batch_size, self.n_ch)
            
            # Slice Batch
            batch_emitters = self.emitters[start_i:end_i] # (B, 3)
            batch_field = field[start_i:end_i]            # (B,)
            
            # Geometry
            ex = batch_emitters[:, 0][:, None, None]
            ey = batch_emitters[:, 1][:, None, None]
            ez = batch_emitters[:, 2][:, None, None]
            
            dx = self.pixel_x - ex
            dy = self.pixel_y - ey
            dz = self.focus_z - ez
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Holographic Summation
            complex_weights = batch_field[:, None, None]
            waves = complex_weights * np.exp(-1j * self.lens_k * dists)
            
            # Attenuation
            attenuation = 1.0 / (dists + 0.1)
            waves *= attenuation
            
            total_field += np.sum(waves, axis=0)

        # 7. Render
        intensity = np.abs(total_field)
        intensity = intensity ** 1.5 
        i_max = np.max(intensity) + 1e-9
        norm = (intensity / i_max * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        
        # HUD
        cv2.putText(heatmap, f"Z: {self.focus_z:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(heatmap, f"K: {self.lens_k:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Output
        out_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.display_image = out_rgb

    def get_output(self, port_name):
        return self.outputs.get(port_name, None)

    def get_display_image(self):
        return self.display_image

    def get_config_options(self):
        return [
            ("Focal Depth", "focus_z", self.focus_z, "float"),
            ("Lens K", "lens_k", self.lens_k, "float"),
            ("Resolution", "res", self.res, "int"),
            ("Scalp Curvature", "curvature", self.curvature, "float"),
            ("Last File", "file_path", self.file_path, "str")
        ]

    def set_config_options(self, options):
        if 'focus_z' in options: self.focus_z = float(options['focus_z'])
        if 'lens_k' in options: self.lens_k = float(options['lens_k'])
        if 'curvature' in options: self.curvature = float(options['curvature'])
        if 'file_path' in options: 
            fp = str(options['file_path'])
            if fp and os.path.exists(fp):
                self.load_data(fp)
        if 'res' in options: 
            new_res = int(options['res'])
            if new_res != self.res:
                self.res = new_res
                # Rebuild Grid
                x = np.linspace(-1, 1, self.res)
                y = np.linspace(-1, 1, self.res)
                self.grid_x, self.grid_y = np.meshgrid(x, y)
                self.pixel_x = self.grid_x.reshape(1, self.res, self.res)
                self.pixel_y = self.grid_y.reshape(1, self.res, self.res)