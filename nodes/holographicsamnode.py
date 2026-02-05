import numpy as np
import mne
from mne.beamformer import make_lcmv, apply_lcmv_cov
import cv2
import os
from PyQt6 import QtGui, QtWidgets
import __main__

# --- ROBUST BASE NODE FALLBACK ---
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    # If the system BaseNode isn't found, define a robust mock
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self._outputs = {}
        
        def get_input(self, name):
            # Fallback: return 0.0 for signals
            return 0.0
            
        def get_blended_input(self, name, method='first'):
            # Fallback: return None for images
            return None
            
        def set_output(self, name, value):
            self._outputs[name] = value

# Standard 10-20 positions for validation
ELECTRODE_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 
    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'
]

class HolographicSAMNode(BaseNode):
    """
    Holographic SAM Beamformer (Radar Imaging).
    
    FEATURES:
    - Crash-Proof BaseNode inheritance.
    - 'Load File' Trigger (Pulse 'load_trigger' > 0.5 to pick file).
    - Radar Power Map (Covariance Beamforming).
    """
    NODE_CATEGORY = "Radar Imaging"
    NODE_TITLE = "SAM Beamformer"
    NODE_COLOR = QtGui.QColor(255, 140, 0) # Radar Orange

    def __init__(self):
        super().__init__()
        self.inputs = {
            'load_trigger': 'signal',      # Pulse > 0.5 to open file dialog
        }
        self.outputs = {
            'radar_image': 'image',        # The Radar Heatmap
            'max_power': 'signal'          
        }
        
        self.subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
        self.raw = None
        self.cov = None
        self.filters = None
        self.src = None
        self.fwd = None
        
        self._img_cache = None
        self.initialized = False
        self.edf_path = "e:\\docshouse\\450\\2.edf" # Default Path

    def pick_file(self):
        """Internal function to manually choose EDF file."""
        # Check if QApplication exists (required for QFileDialog)
        app = QtWidgets.QApplication.instance()
        if app is None:
            # If running standalone without a GUI app instance
            app = QtWidgets.QApplication([])
            
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select EEG Data", os.path.dirname(self.edf_path), "EDF Files (*.edf)"
        )
        if fname:
            self.edf_path = fname
            print(f"[SAM] Selected: {self.edf_path}")
            self.initialized = False # Force re-init with new file
            self._init_radar()

    def _init_radar(self):
        """Initializes the Beamformer using Covariance (Crash-Proof)."""
        try:
            print(f"[SAM] Initializing Radar on: {self.edf_path}")
            
            # 1. Setup Template
            if not os.path.isdir(os.path.join(self.subjects_dir, 'fsaverage')):
                print("[SAM] Downloading fsaverage template...")
                mne.datasets.fetch_fsaverage(subjects_dir=self.subjects_dir, verbose=False)
            
            self.src = mne.setup_source_space(
                'fsaverage', spacing='ico4', subjects_dir=self.subjects_dir, add_dist=False, verbose=False
            )
            
            # 2. Load Data
            if not os.path.exists(self.edf_path):
                print(f"[SAM] File not found: {self.edf_path}")
                # Don't crash, just wait for user to pick a valid file
                return

            self.raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            
            # 3. Clean Channels (Aggressive Mapping)
            mapping = {}
            for ch in self.raw.ch_names:
                clean = ch.replace('EEG', '').replace('eeg', '').replace('-Ref', '').replace('-REF', '').strip()
                clean = clean.replace('.', '').replace('Z', 'z').replace('FP', 'Fp')
                for std in ELECTRODE_NAMES:
                    if clean.upper() == std.upper():
                        mapping[ch] = std
                        break
            
            if not mapping:
                print("[SAM] Error: No valid 10-20 channels found. Check electrode names.")
                return

            self.raw.rename_channels(mapping)
            self.raw.pick_channels(list(mapping.values()))
            self.raw.set_montage('standard_1020', on_missing='ignore')
            self.raw.set_eeg_reference('average', projection=True, verbose=False)
            
            # 4. Compute Physics (Forward + Covariance)
            bem_model = mne.make_bem_model(subject='fsaverage', conductivity=(0.3, 0.006, 0.3),
                                         subjects_dir=self.subjects_dir, verbose=False)
            bem_sol = mne.make_bem_solution(bem_model, verbose=False)
            
            self.fwd = mne.make_forward_solution(
                self.raw.info, trans='fsaverage', src=self.src, bem=bem_sol, eeg=True, verbose=False
            )
            
            # Covariance Matrix (The Radar Signature)
            self.cov = mne.compute_raw_covariance(self.raw, tmin=0, tmax=None, verbose=False)
            
            # 5. Compute Filters (The Radar Lens)
            self.filters = make_lcmv(
                self.raw.info, self.fwd, self.cov, reg=0.05,
                pick_ori='max-power', weight_norm='unit-noise-gain',
                reduce_rank=True, verbose=False
            )
            
            print("[SAM] Radar Online. Beamformer Ready.")
            self.initialized = True
            
        except Exception as e:
            print(f"[SAM] Init Error: {e}")

    def step(self):
        # --- SAFE INPUT HANDLING ---
        trigger = 0.0
        if hasattr(self, 'get_input'):
            trigger = self.get_input('load_trigger')
            
        # Check for Load Trigger
        if trigger > 0.5:
            # Simple debounce: Only trigger if we haven't just re-initialized
            # In a real node graph, you might want a dedicated button or edge trigger
            self.pick_file()
        
        if not self.initialized:
            # Try to init if we have a default path, otherwise wait
            if os.path.exists(self.edf_path):
                self._init_radar()
            return

        try:
            # 1. Apply Beamformer to COVARIANCE (Static Power Map)
            stc = apply_lcmv_cov(self.cov, self.filters, verbose=False)
            
            # 2. Extract Power (Magnitude)
            power_map = stc.data[:, 0]
            
            # 3. Project to Image
            img_size = 128
            radar_img = np.zeros((img_size, img_size), dtype=np.float32)
            
            lh_verts = self.src[0]['rr']
            rh_verts = self.src[1]['rr']
            verts = np.concatenate([lh_verts, rh_verts])
            
            norm_x = ((verts[:, 0] + 0.07) / 0.14 * (img_size-10) + 5).astype(int)
            norm_y = ((verts[:, 1] + 0.07) / 0.14 * (img_size-10) + 5).astype(int)
            norm_x = np.clip(norm_x, 0, img_size-1)
            norm_y = np.clip(norm_y, 0, img_size-1)
            
            # Thresholding (Show top 25% hotspots)
            threshold = np.percentile(power_map, 75)
            indices = np.where(power_map > threshold)[0]
            
            for idx in indices:
                x, y = norm_x[idx], norm_y[idx]
                val = power_map[idx]
                radar_img[img_size-1-y, x] = max(radar_img[img_size-1-y, x], val)
            
            radar_img = cv2.GaussianBlur(radar_img, (3, 3), 0)
            radar_img = radar_img / (radar_img.max() + 1e-9)
            
            # 4. Render
            img_u8 = (np.clip(radar_img, 0, 1) * 255).astype(np.uint8)
            self._img_cache = cv2.applyColorMap(img_u8, cv2.COLORMAP_HOT)
            
            cv2.putText(self._img_cache, "SAM RADAR", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Safe output setting
            if hasattr(self, '_outputs'):
                self._outputs['max_power'] = float(power_map.max())
            
        except Exception as e:
            print(f"[SAM] Step Error: {e}")

    def get_display_image(self):
        if self._img_cache is None: return None
        img = self._img_cache
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888).copy()