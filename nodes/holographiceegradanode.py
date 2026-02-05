"""
SAM BEAMFORMER NODE: RADAR-STYLE EEG SOURCE IMAGING
==================================================
Implements Synthetic Aperture Magnetometry (SAM) beamformer.
Pure physics — no AI. Treats electrodes as phased array radar.

Input: Raw EEG (or your holographic phases as "virtual sensors").
Output: 3D source power map (volumetric "brain scan").

Key Physics (from Robinson & Vrba 1999 / Van Veen LCMV):
- Scan grid points in head volume.
- For each point: Compute leadfield (forward model — how source at point reaches electrodes).
- Adaptive weights: Suppress interference, maximize power from target point.
- Virtual sensor: Reconstructed time-series/power at each voxel.

Uses MNE for forward/leadfield (realistic head model).
No template needed beyond standard — outputs dense 3D activity.

Drop as holographicsamnode.py in your nodes folder.
"""

import numpy as np
import mne
from mne.beamformer import make_lcmv, apply_lcmv
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode: pass

class SAMBeamformerNode(BaseNode):
    """
    Synthetic Aperture Magnetometry (SAM) Beamformer Node.
    Radar-style source reconstruction from EEG phases/voltages.
    """
    NODE_CATEGORY = "Radar Imaging"
    NODE_TITLE = "SAM Beamformer"
    NODE_COLOR = QtGui.QColor(255, 140, 0)  # Orange

    def __init__(self):
        super().__init__()
        self.inputs = {
            'eeg_raw': 'raw_eeg',          # MNE Raw object or your phase data
            'focus_k': 'signal'            # Optional k-tuning (your spatial frequency)
        }
        self.outputs = {
            'source_map': 'image',         # 2D slice of 3D power
            'power_max': 'signal'          # Peak source power
        }

        # MNE Setup (fsaverage template — auto-download)
        self.subject = 'fsaverage'
        self.subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
        self.src = mne.setup_source_space(self.subject, spacing='oct6', add_dist=False)
        self.bem = mne.make_bem_model(subject=self.subject, conductivity=(0.3,))
        self.bem_sol = mne.make_bem_solution(self.bem)
        
        # Dummy info for forward
        self.info = None
        self.fwd = None
        self.filters = None
        self._power_map = None

    def _setup_forward(self, raw):
        """Build forward model from Raw EEG"""
        if self.info is None or self.fwd is None:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            self.info = raw.info
            
            self.fwd = mne.make_forward_solution(
                self.info, trans='fsaverage', src=self.src,
                bem=self.bem_sol, eeg=True, mindist=5.0, verbose=False
            )
            
            # Noise covariance (ad hoc for demo)
            noise_cov = mne.make_ad_hoc_cov(self.info)
            
            # LCMV filters (SAM variant)
            self.filters = make_lcmv(
                self.info, self.fwd, noise_cov, reg=0.05,
                pick_ori='max-power', weight_norm='unit-noise-gain',
                verbose=False
            )

    def step(self):
        raw = self.get_blended_input('eeg_raw', 'first')
        if raw is None: return
        
        self._setup_forward(raw)
        
        # Apply beamformer
        stc = apply_lcmv_raw(raw, self.filters, verbose=False)
        
        # Power map (mean over time)
        data = stc.data**2
        power = data.mean(axis=1)  # [n_sources]
        
        # Project to volume/grid (simple: max over depth)
        # For full 3D: use stc.in_volume() or custom grid
        brain = stc.as_volume(self.src, mri_resolution=True)
        vol_data = brain.data
        
        # Extract central slice for display
        slice_idx = vol_data.shape[0] // 2
        slice_2d = vol_data[slice_idx, :, :]
        
        # Normalize & colorize
        norm = slice_2d / (slice_2d.max() + 1e-9)
        vis = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)
        
        # Peak power metric
        max_power = power.max()
        self._outputs['power_max'] = float(max_power)
        
        self._power_map = colored

    def get_display_image(self):
        if self._power_map is None: return None
        img = cv2.resize(self._power_map, (256, 256))
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888).copy()

# Standalone test (if run directly)
if __name__ == "__main__":
    print("SAM Beamformer Node")
    print("Connect Raw EEG → eeg_raw")
    print("Outputs high-res source power map (radar-style virtual sensor grid)")