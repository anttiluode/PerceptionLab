import numpy as np
import mne
import cv2
import os
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode: pass

# Standard 10-20 positions (Normalized 0-1 for image sampling)
ELECTRODE_COORDS = {
    'Fp1': (0.35, 0.05), 'Fp2': (0.65, 0.05),
    'F7': (0.15, 0.20), 'F3': (0.33, 0.20), 'Fz': (0.50, 0.20), 'F4': (0.67, 0.20), 'F8': (0.85, 0.20),
    'T7': (0.05, 0.50), 'C3': (0.30, 0.50), 'Cz': (0.50, 0.50), 'C4': (0.70, 0.50), 'T8': (0.95, 0.50),
    'P7': (0.15, 0.75), 'P3': (0.33, 0.75), 'Pz': (0.50, 0.75), 'P4': (0.67, 0.75), 'P8': (0.85, 0.75),
    'O1': (0.35, 0.95), 'Oz': (0.50, 0.95), 'O2': (0.65, 0.95)
}

class HolographicTomographyNode(BaseNode):
    """
    Holographic Tomography (Anatomical Projection).
    Projects Binding Map onto 3D Brain Template (sLORETA).
    
    FIXED: 
    - Added self._outputs initialization to prevent crash.
    - Corrected 3D coordinate mapping.
    - Added Hemisphere Balance metric.
    """
    NODE_CATEGORY = "Tomography"
    NODE_TITLE = "Holographic Tomography"
    NODE_COLOR = QtGui.QColor(255, 80, 20)

    def __init__(self):
        super().__init__()
        self.inputs = {
            'binding_map': 'image',        
            'complex_field': 'complex_spectrum' 
        }
        self.outputs = {
            'source_volume': 'image',      
            'hemisphere_balance': 'signal', # Metric (-1 to 1)
            'lobe_activation': 'signal'    
        }
        
        # --- FIX: Initialize the storage dictionary ---
        self._outputs = {} 
        
        # MNE State
        self.subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
        self.inverse_operator = None
        self.src = None
        self.fwd = None
        self._brain_cache = None     
        self._output_cache = None    
        self._init_mne()

    def _init_mne(self):
        """Initialize generic brain template for projection."""
        try:
            if not os.path.isdir(os.path.join(self.subjects_dir, 'fsaverage')):
                print("Downloading fsaverage template...")
                mne.datasets.fetch_fsaverage(subjects_dir=self.subjects_dir, verbose=False)

            # Setup Source Space
            self.src = mne.setup_source_space(
                'fsaverage', spacing='ico4', subjects_dir=self.subjects_dir, add_dist=False, verbose=False
            )

            # Setup Montage & Dummy Info
            montage = mne.channels.make_standard_montage('standard_1020')
            temp_info = mne.create_info(ch_names=list(ELECTRODE_COORDS.keys()), sfreq=256, ch_types='eeg')
            temp_info.set_montage(montage)
            
            # Create Dummy Raw for Reference
            dummy_data = np.zeros((len(ELECTRODE_COORDS), 1))
            dummy_raw = mne.io.RawArray(dummy_data, temp_info, verbose=False)
            dummy_raw.set_eeg_reference('average', projection=True, verbose=False)
            self.info = dummy_raw.info

            # Forward Solution
            model = mne.make_bem_model(
                subject='fsaverage', ico=4, conductivity=(0.3, 0.006, 0.3),
                subjects_dir=self.subjects_dir, verbose=False
            )
            bem = mne.make_bem_solution(model, verbose=False)
            self.fwd = mne.make_forward_solution(
                self.info, trans='fsaverage', src=self.src, bem=bem, eeg=True, verbose=False
            )
            
            # Inverse Operator
            cov = mne.make_ad_hoc_cov(self.info)
            self.inverse_operator = mne.minimum_norm.make_inverse_operator(
                self.info, self.fwd, cov, loose=0.2, depth=0.8, verbose=False
            )
            print("MNE Tomography Initialized.")

        except Exception as e:
            print(f"MNE Init Failed: {e}")

    def step(self):
        binding = self.get_blended_input('binding_map', 'first')
        if binding is None or self.inverse_operator is None: return

        # 1. Sample Binding Map
        h, w = binding.shape[:2]
        if binding.ndim == 3: binding = np.mean(binding, axis=2)
        
        sensor_data = []
        for name, (rx, ry) in ELECTRODE_COORDS.items():
            px, py = int(rx * w), int(ry * h)
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            val = binding[py, px] / 255.0
            sensor_data.append(val)
        sensor_data = np.array(sensor_data)

        try:
            # 2. Apply Inverse
            evoked = mne.EvokedArray(sensor_data[:, np.newaxis], self.info)
            stc = mne.minimum_norm.apply_inverse(
                evoked, self.inverse_operator, lambda2=1.0/9.0, method='sLORETA', verbose=False
            )
            
            # 3. Project to Image (CORRECTED MAPPING)
            src_data = stc.data[:, 0]
            
            # Get correct vertices for LH and RH
            lh_indices = stc.vertices[0]
            rh_indices = stc.vertices[1]
            
            # Extract coordinates for ACTIVE sources only
            lh_coords = self.src[0]['rr'][lh_indices]
            rh_coords = self.src[1]['rr'][rh_indices]
            coords = np.concatenate([lh_coords, rh_coords])
            
            # --- HEMISPHERE BALANCE CALCULATION ---
            n_lh = len(lh_indices)
            lh_power = np.mean(src_data[:n_lh])
            rh_power = np.mean(src_data[n_lh:])
            
            # Balance: (L - R) / (L + R) -> -1 (Right) to +1 (Left)
            total_p = lh_power + rh_power + 1e-9
            balance = (lh_power - rh_power) / total_p
            
            # Store metric safely
            self._outputs['hemisphere_balance'] = float(balance)
            
            # --- VISUALIZATION ---
            brain_img = np.zeros((128, 128), dtype=np.float32)
            
            # Normalize physical coords to image
            norm_x = ((coords[:, 0] + 0.07) / 0.14 * 100 + 14).astype(int)
            norm_y = ((coords[:, 1] + 0.07) / 0.14 * 100 + 14).astype(int)
            norm_x = np.clip(norm_x, 0, 127)
            norm_y = np.clip(norm_y, 0, 127)
            
            # Lower Threshold to 45%
            threshold = np.percentile(src_data, 45) 
            strong_indices = np.where(src_data > threshold)[0]
            
            for idx in strong_indices:
                x, y = norm_x[idx], norm_y[idx]
                val = src_data[idx]
                brain_img[127-y, x] = max(brain_img[127-y, x], val)
                
            brain_img = cv2.GaussianBlur(brain_img, (3, 3), 0)
            img_norm = brain_img / (brain_img.max() + 1e-9)
            
            # Create Display
            img_u8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
            
            # Draw Balance Indicator
            self._draw_balance_ui(colored, balance)
            
            self._output_cache = colored
            
        except Exception as e:
            print(f"Tomography Step Error: {e}")

    def _draw_balance_ui(self, img, balance):
        h, w = img.shape[:2]
        center = w // 2
        bar_w = 60
        
        cv2.rectangle(img, (center - bar_w, h - 20), (center + bar_w, h - 10), (50, 50, 50), -1)
        
        offset = int(balance * bar_w)
        offset = max(-bar_w, min(bar_w, offset))
        
        color = (0, 0, 255) if balance > 0 else (255, 0, 0)
        cv2.circle(img, (center - offset, h - 15), 4, color, -1)
        
        text = f"L {balance:.2f} R"
        cv2.putText(img, text, (center - 40, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'source_volume': return self._output_cache 
        # Safely access the dict
        if name == 'hemisphere_balance': return self._outputs.get('hemisphere_balance', 0.0)
        return None

    def get_display_image(self):
        if self._output_cache is None: return None
        img = self._output_cache
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888).copy()