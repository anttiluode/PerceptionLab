"""
Phi Hologram Node (Sub-1.0 K Fix)
=================================
Projects EEG phase data as a complex interference field.

FIXED:
- Allowed spatial_k to go below 1.0 (down to 0.001) to see the "Global One".
- Updated clipping range in step() and config.
"""

import os
import numpy as np
import cv2

# --- HOST IMPORT BLOCK (Standard Pattern) ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def pre_step(self): 
            self.input_data = {name: [] for name in self.inputs}
        def get_blended_input(self, name, mode): 
            return None

try:
    import mne
    import scipy.signal
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("[PhiHologram] Warning: mne/scipy not available. Install with: pip install mne scipy")


# Standard 10-20 electrode positions (normalized -1 to 1)
ELECTRODE_POS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6), 'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'T3': (-0.9, 0.0), 'T7': (-0.9, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0, 0.0), 'C4': (0.4, 0.0), 'T4': (0.9, 0.0), 'T8': (0.9, 0.0),
    'T5': (-0.7, -0.5), 'P7': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5), 'P4': (0.35, -0.5), 'T6': (0.7, -0.5), 'P8': (0.7, -0.5),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85)
}

# Frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


class PhiHologramNode(BaseNode):
    """
    Neural interferometer - projects EEG phase into 2D holographic interference patterns.
    """
    
    NODE_NAME = "Phi Hologram"
    NODE_TITLE = "Phi Hologram"
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(180, 40, 220) if QtGui else None  # Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'phase_rot': 'signal',      
            'spatial_k': 'signal',      
            'speed': 'signal',          
            'reset': 'signal',          
        }
        
        self.outputs = {
            'delta_hologram': 'image',
            'theta_hologram': 'image',
            'alpha_hologram': 'image',
            'beta_hologram': 'image',
            'gamma_hologram': 'image',
            'combined_hologram': 'image',
            'delta_field': 'complex_spectrum',
            'theta_field': 'complex_spectrum',
            'alpha_field': 'complex_spectrum',
            'beta_field': 'complex_spectrum',
            'gamma_field': 'complex_spectrum',
            'delta_power': 'signal',
            'theta_power': 'signal',
            'alpha_power': 'signal',
            'beta_power': 'signal',
            'gamma_power': 'signal',
            'time_seconds': 'signal',
        }
        
        self.edf_file_path = ""
        self.resolution = 128
        self.default_k = 1.0  # LOWERED DEFAULT
        self.default_rot = 0.0
        self.display_band = "alpha"  
        
        self._last_path = ""
        self.raw = None
        self.channel_data = {}
        self.active_elecs = {}
        self.times = None
        self.sfreq = 256.0
        self.current_idx = 0
        self.loaded_path = ""
        self.status_msg = "No EDF loaded"
        
        self.X = None
        self.Y = None
        self.dist_maps = {}
        self.band_phases = {band: {} for band in BANDS}  
        self.field_cache = {band: None for band in BANDS}
        self.image_cache = {band: None for band in BANDS}
        self.power_cache = {band: 0.0 for band in BANDS}
        self.combined_image = None
        self.display_image = None
        self._init_display()
    
    def get_config_options(self):
        band_options = [(b, b) for b in BANDS.keys()]
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, "str"),
            ("Resolution", "resolution", self.resolution, "int"),
            ("Default Spatial K", "default_k", self.default_k, "float"),
            ("Default Phase Rot", "default_rot", self.default_rot, "float"),
            ("Display Band", "display_band", self.display_band, band_options),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            if 'edf_file_path' in options:
                self._last_path = ""  
    
    def _init_grid(self, res):
        x = np.linspace(-1.5, 1.5, res).astype(np.float32)
        y = np.linspace(-1.5, 1.5, res).astype(np.float32)
        self.X, self.Y = np.meshgrid(x, y)
        self.dist_maps = {}
        for name, (ex, ey) in self.active_elecs.items():
            self.dist_maps[name] = np.sqrt((self.X - ex)**2 + (self.Y - ey)**2).astype(np.float32)
    
    def _init_display(self):
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "PHI HOLOGRAM", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 40, 220), 2)
        self.display_image = img
    
    def _load_edf(self):
        if not self.edf_file_path or not os.path.exists(self.edf_file_path):
            self.status_msg = "File not found"
            return False
        
        try:
            self.raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose='error')
            self.sfreq = self.raw.info['sfreq']
            self.times = self.raw.times
            self.current_idx = 0
            self.channel_data = {}
            self.active_elecs = {}
            
            ch_names = self.raw.ch_names
            for std_name, pos in ELECTRODE_POS.items():
                match = None
                for ch in ch_names:
                    ch_clean = ch.strip().replace('.', '').upper()
                    if std_name.upper() in ch_clean or ch_clean in std_name.upper():
                        match = ch
                        break
                if match:
                    self.channel_data[std_name] = self.raw.get_data(picks=[match])[0]
                    self.active_elecs[std_name] = pos
            
            self._init_grid(self.resolution)
            self._compute_all_phases()
            self.status_msg = f"Loaded: {len(self.active_elecs)} channels"
            return True
        except Exception as e:
            self.status_msg = f"Load error: {str(e)[:30]}"
            return False
    
    def _compute_all_phases(self):
        for band_name, (lo, hi) in BANDS.items():
            self.band_phases[band_name] = {}
            for elec_name, data in self.channel_data.items():
                try:
                    nyq = self.sfreq / 2.0
                    lo_norm = max(0.5, lo) / nyq
                    hi_norm = min(hi, nyq - 1) / nyq
                    b, a = scipy.signal.butter(3, [lo_norm, hi_norm], btype='band')
                    filtered = scipy.signal.filtfilt(b, a, data)
                    analytic = scipy.signal.hilbert(filtered)
                    self.band_phases[band_name][elec_name] = np.angle(analytic).astype(np.float32)
                except:
                    self.band_phases[band_name][elec_name] = np.zeros_like(data)
    
    def _generate_hologram(self, band_name, time_idx, k, rot_deg):
        if not self.active_elecs or self.X is None:
            return None, None, 0
        
        res = self.X.shape[0]
        field = np.zeros((res, res), dtype=np.complex64)
        rot_rad = np.deg2rad(rot_deg)
        total_amplitude = 0.0
        
        for elec_name in self.active_elecs:
            if elec_name not in self.band_phases[band_name] or elec_name not in self.dist_maps:
                continue
            phi = self.band_phases[band_name][elec_name][time_idx]
            dist = self.dist_maps[elec_name]
            theta = phi - (k * dist) + rot_rad
            field += np.exp(1j * theta)
            total_amplitude += 1.0
        
        if total_amplitude == 0: return None, None, 0
        field /= total_amplitude
        intensity = np.abs(field) ** 2
        i_max = np.max(intensity)
        if i_max > 0: intensity /= i_max
        
        img_u8 = (np.clip(intensity, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        return field, colored, float(np.mean(intensity))
    
    def step(self):
        if self.edf_file_path != self._last_path:
            self._last_path = self.edf_file_path
            if self.edf_file_path: self._load_edf()
        
        if self.raw is None or not self.active_elecs: return
        
        # --- INPUT PARSING ---
        rot = self.get_blended_input('phase_rot', 'sum')
        rot = float(rot) if rot is not None else self.default_rot
        
        k = self.get_blended_input('spatial_k', 'sum')
        k = float(k) if k is not None else self.default_k
        # FIX: UNLOCKED K-RANGE (Allows 0.05)
        k = float(np.clip(k, 0.001, 100.0))
        
        speed = self.get_blended_input('speed', 'sum')
        speed = float(np.clip(speed if speed is not None else 1.0, 0.1, 10.0))
        
        if (self.get_blended_input('reset', 'sum') or 0) > 0.5: self.current_idx = 0
        
        self.current_idx = (self.current_idx + int((self.sfreq / 30.0) * speed)) % len(self.times)
        
        for band_name in BANDS:
            f, img, p = self._generate_hologram(band_name, self.current_idx, k, rot)
            if f is not None:
                self.field_cache[band_name], self.image_cache[band_name], self.power_cache[band_name] = f, img, p
        
        self._create_combined_view()
        self._update_display(k, rot)
    
    def _create_combined_view(self):
        alpha = self.image_cache.get('alpha')
        if alpha is None: return
        h, w = alpha.shape[:2]
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        for i, band in enumerate(['beta', 'alpha', 'gamma']): # BGR stack
            img = self.image_cache.get(band)
            if img is not None: combined[:, :, i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.combined_image = combined
    
    def _update_display(self, k, rot):
        band_img = self.image_cache.get(self.display_band)
        if band_img is None: return
        
        h, w = band_img.shape[:2]
        panel_h = 60
        display = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        display[panel_h:, :] = band_img
        
        t_sec = self.times[self.current_idx]
        cv2.putText(display, f"PHI HOLOGRAM [{self.display_band.upper()}]", (5, 15), 0, 0.4, (180, 40, 220), 1)
        cv2.putText(display, f"T={t_sec:.2f}s  K={k:.2f}  Rot={rot:.1f}", (5, 32), 0, 0.35, (200, 200, 200), 1)
        
        for i, (band, power) in enumerate(self.power_cache.items()):
            x = 5 + i * 25
            cv2.rectangle(display, (x, 45), (x + 20, 45 - int(power * 15)), (100, 200, 255), -1)
            cv2.putText(display, band[0].upper(), (x + 3, 55), 0, 0.25, (150, 150, 150), 1)
        
        self.display_image = display
    
    def get_output(self, port_name):
        if 'hologram' in port_name: return self.image_cache.get(port_name.split('_')[0])
        if 'combined' in port_name: return self.combined_image
        if 'field' in port_name: return self.field_cache.get(port_name.split('_')[0])
        if 'power' in port_name: return self.power_cache.get(port_name.split('_')[0], 0.0)
        if port_name == 'time_seconds': return float(self.times[self.current_idx])
        return None
    
    def get_display_image(self): return self.display_image
    def close(self): self.raw = None