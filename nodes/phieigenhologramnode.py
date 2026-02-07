"""
Phi Eigen-Hologram Node (Final Production Fix)
==============================================
"The Geometry of the Ghost"

FIXED:
- Implemented Lazy Loading in step() for JSON/EDF stability.
- Fixed config/attribute race conditions.
- Optimized dual-render pipeline.
"""

import numpy as np
import cv2
import os
import scipy.linalg
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

class PhiEigenHologramNode(BaseNode):
    NODE_CATEGORY = "Visualizers"
    NODE_COLOR = QtGui.QColor(180, 100, 255) # Eigen Purple

    def __init__(self):
        super().__init__()
        self.node_title = "Phi Eigen-Hologram"
        
        self.inputs = {
            'speed': 'signal',
            'lens_k': 'signal'
        }
        
        self.outputs = {
            'render': 'image',
            'metastability': 'signal'
        }

        # Parameters
        self.res = 128
        self.lens_k = 10.0
        self.focus_z = 1.0
        self.curvature = 0.5
        self.eigen_update_rate = 5 
        
        # Data State
        self.frame_count = 0
        self.data = None
        self.n_ch = 0
        self.current_idx = 0
        self.sampling_rate = 160.0
        self.file_path = ""
        self._last_loaded_path = None # TRACKER FOR LAZY LOAD
        self.is_synthetic = False
        
        # Geometry State
        self.emitters = None 
        self.grid_x = None
        self.grid_y = None
        self.pixel_x = None
        self.pixel_y = None
        
        # Eigen State
        self.current_eigenvector = None
        self.metastability_index = 0.0
        self.display_image = None
        
        # UI
        self.lbl_status = QtWidgets.QLabel("No Source")
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px;")
        self.btn_load = QtWidgets.QPushButton("LOAD EEG")
        self.btn_load.setStyleSheet("""
            QPushButton { 
                background: #332244; color: #cc88ff; border: 1px solid #664488; 
                padding: 6px; font-weight: bold; border-radius: 4px;
            }
            QPushButton:hover { background: #443355; }
        """)
        self.btn_load.clicked.connect(self.load_file_dialog)
        
        # Initial Grid Build
        self.rebuild_grid()

    def rebuild_grid(self):
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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select EEG", "", "EDF Files (*.edf);;All Files (*)")
        if fname: self.file_path = fname

    def load_data(self, path):
        if not MNE_AVAILABLE:
            self.generate_synthetic_data()
            return

        try:
            self.lbl_status.setText("Physics Ingestion...")
            QtWidgets.QApplication.processEvents()
            
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            
            if 'eeg' in raw:
                raw.pick_types(eeg=True, exclude='bads')
            else:
                raw.pick(range(min(60, len(raw.ch_names))))

            raw.filter(1, 60, verbose=False) 
            
            self.data = raw.get_data() * 1e6 
            self.sampling_rate = raw.info['sfreq']
            self.n_ch = len(raw.ch_names)
            self.is_synthetic = False
            self.current_idx = 0
            
            self.rebuild_emitters(self.n_ch)
            self.lbl_status.setText(f"REAL: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"Eigen Load Error: {e}")
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        self.is_synthetic = True
        self.n_ch = 64
        self.sampling_rate = 160.0
        t = np.linspace(0, 10, 1600)
        self.data = np.zeros((64, 1600))
        for i in range(64):
            self.data[i] = np.sin(2*np.pi*10*t) * 10 + np.sin(2*np.pi*10*t + i*0.1) * 10
        self.rebuild_emitters(64)
        self.lbl_status.setText("SYNTHETIC")

    def rebuild_emitters(self, n_ch):
        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        r = 0.8
        ex = r * np.cos(theta)
        ey = r * np.sin(theta)
        ez = np.full_like(ex, -self.curvature)
        self.emitters = np.stack((ex, ey, ez), axis=1)

    def compute_eigenmodes(self, segment):
        """Calculates Graph Laplacian Eigenmodes based on PLV coherence."""
        analytic = np.fft.fft(segment, axis=1)
        freq_bin = 10 # Alpha focus
        phases = np.angle(analytic[:, freq_bin]) 
        
        phase_diffs = phases[:, None] - phases[None, :]
        plv_matrix = np.abs(np.exp(1j * phase_diffs)) 
        
        adjacency = plv_matrix
        np.fill_diagonal(adjacency, 0)
        degrees = np.sum(adjacency, axis=1)
        laplacian = np.diag(degrees) - adjacency
        
        try:
            vals, vecs = scipy.linalg.eigh(laplacian)
            activity = np.abs(analytic[:, freq_bin])
            projections = [np.abs(np.dot(activity, vecs[:, i])) for i in range(min(5, self.n_ch))]
            best_idx = np.argmax(projections)
            
            self.current_eigenvector = vecs[:, best_idx]
            self.metastability_index = np.std(projections)
        except:
            pass

    def render_hologram(self, values, title):
        if self.emitters is None or self.pixel_x is None: 
            return np.zeros((self.res, self.res, 3), dtype=np.uint8)
        
        total_field = np.zeros((self.res, self.res), dtype=np.complex64)
        batch = 16
        for i in range(0, self.n_ch, batch):
            end = min(i+batch, self.n_ch)
            b_emitters = self.emitters[i:end]
            b_vals = values[i:end]
            
            dx = self.pixel_x - b_emitters[:,0][:,None,None]
            dy = self.pixel_y - b_emitters[:,1][:,None,None]
            dz = self.focus_z - b_emitters[:,2][:,None,None]
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            
            waves = b_vals[:,None,None] * np.exp(-1j * self.lens_k * dists)
            total_field += np.sum(waves / (dists+0.1), axis=0)

        intensity = np.abs(total_field)**1.5
        norm = (intensity / (np.max(intensity)+1e-9) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        cv2.putText(heatmap, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        return heatmap

    def step(self):
        # LAZY LOAD TRIGGER
        if self.file_path and self.file_path != self._last_loaded_path:
            self._last_loaded_path = self.file_path
            if os.path.exists(self.file_path):
                self.load_data(self.file_path)

        if self.data is None: 
            self.render_placeholder()
            return

        # Double check grid
        if self.pixel_x is None: self.rebuild_grid()

        # Input signals
        k_sig = self.get_blended_input('lens_k', 'max')
        if k_sig is not None: self.lens_k = k_sig * 50.0

        speed = self.get_blended_input('speed', 'max') or 1.0
        self.current_idx = int(self.current_idx + speed*2) % (self.data.shape[1] - 128)
            
        segment = self.data[:, self.current_idx:self.current_idx+128]
        
        # Periodic Eigen Update
        self.frame_count += 1
        if self.frame_count % self.eigen_update_rate == 0:
            self.compute_eigenmodes(segment)
            
        # Raw Field (Alpha focus)
        fft_res = np.fft.rfft(segment, axis=1)
        field = np.sum(fft_res[:, 8:12], axis=1) 
        
        # Dual-Screen Render
        img_left = self.render_hologram(field, "REALITY (Alpha)")
        
        if self.current_eigenvector is not None:
            img_right = self.render_hologram(self.current_eigenvector, "IDEAL (Eigen)")
        else:
            img_right = np.zeros_like(img_left)
            
        combined = np.hstack((img_left, img_right))
        
        # Metastability Meter
        meta_h = int(np.clip(self.metastability_index * 10, 0, 100))
        cv2.rectangle(combined, (combined.shape[1]-5, combined.shape[0]), 
                      (combined.shape[1], combined.shape[0]-meta_h), (0, 255, 255), -1)

        out_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        self.outputs['render'] = out_rgb
        self.outputs['metastability'] = float(self.metastability_index)
        self.display_image = out_rgb

    def render_placeholder(self):
        img = np.zeros((256, 512, 3), dtype=np.uint8)
        cv2.putText(img, "WAITING FOR EEG", (170, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
        self.outputs['render'] = img
        self.display_image = img

    def get_output(self, port_name):
        return self.outputs.get(port_name)

    def get_display_image(self):
        return self.display_image

    def get_config_options(self):
        return [("Lens K", "lens_k", self.lens_k, "float"),
                ("File Path", "file_path", self.file_path, "str")]

    def set_config_options(self, options):
        if 'lens_k' in options: self.lens_k = float(options['lens_k'])
        if 'file_path' in options: self.file_path = str(options['file_path'])

    def close(self):
        self.data = None