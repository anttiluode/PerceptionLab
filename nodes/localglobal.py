"""
Local-Global Moiré Interferometer Node
======================================
Antti Luode (PerceptionLab) | 2026

Inspired by the Deerskin Architecture:
This node computes the instantaneous spatial phase of the cortex, creates a 
"Global" reference field via spatial diffusion (the 't' zoom-out setting), 
and interferes it with the "Local" field. 

If local circuits lock to the global pacemaker, the field is coherent (flat).
If local circuits decouple (fragmented field), macroscopic Moiré fringes appear.
"""

import os
import numpy as np
import cv2
import traceback

try:
    import mne
    from scipy.signal import butter, filtfilt, hilbert
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}

# ── 19-Channel 2D Coordinates ──
ELEC_COORDS_2D = {
    'FP1': (-0.3, 0.7),  'FP2': (0.3, 0.7),
    'F7':  (-0.7, 0.4),  'F3':  (-0.4, 0.4), 'FZ': (0.0, 0.4), 'F4': (0.4, 0.4), 'F8': (0.7, 0.4),
    'T3':  (-0.8, 0.0),  'C3':  (-0.4, 0.0), 'CZ': (0.0, 0.0), 'C4': (0.4, 0.0), 'T4': (0.8, 0.0),
    'T5':  (-0.7, -0.4), 'P3':  (-0.4, -0.4), 'PZ': (0.0, -0.4), 'P4': (0.4, -0.4), 'T6': (0.7, -0.4),
    'O1':  (-0.3, -0.7), 'O2':  (0.3, -0.7)
}

BANDS = {
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta (13-30 Hz)": [13.0, 30.0],
    "Gamma (30-45 Hz)": [30.0, 45.0]
}

class LocalGlobalMoireNode(BaseNode):
    NODE_CATEGORY = "Deerskin Architecture"
    NODE_TITLE = "Local-Global Moiré Interferometer"
    NODE_COLOR = QtGui.QColor(0, 200, 255) # Cyan

    def __init__(self):
        super().__init__()
        
        self.inputs = {}
        self.outputs = {'interferogram': 'image'}
        
        # Configuration
        self.edf_path = ""
        self._last_path = ""
        
        self.selected_band = "Theta (4-8 Hz)"
        self._last_band = "Theta (4-8 Hz)"
        
        # The 't' settings: Diffusion sigmas
        self.local_diffusion = 1.0   # Sharp local field
        self.global_diffusion = 40.0 # 't' zoom-out (Macroscopic field)
        
        self.fringe_density = 1.0    # K-multiplier for interference sensitivity
        self.aperture_shape = 1      # 1=Circle, 0=Square
        self.colormap = 1            # 0=Gray, 1=Twilight, 2=Ocean
        
        # State
        self.res = 200
        self.n_ch = 19
        self.sfreq = 250
        
        self.raw_data = None
        self.complex_analytic = None # Stores A * e^(i * phase)
        self.current_frame = 0
        self.cached_image = np.zeros((self.res, self.res, 3), dtype=np.uint8)
        
        self.status_msg = "No EDF. Using Synthetic."
        
        self._init_geometry()

    def _init_geometry(self):
        x = np.linspace(-1.1, 1.1, self.res)
        y = np.linspace(-1.1, 1.1, self.res)
        xx, yy = np.meshgrid(x, y, indexing='ij') 
        self.grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
        self.mask = (self.grid_coords[:, 0]**2 + self.grid_coords[:, 1]**2) <= 1.0

        # Precompute spatial weights for mapping 19 channels to the grid
        self.spatial_weights = np.zeros((len(self.grid_coords), self.n_ch))
        for i, (name, coord) in enumerate(ELEC_COORDS_2D.items()):
            dists = np.linalg.norm(self.grid_coords - np.array(coord), axis=1)
            self.spatial_weights[:, i] = 1.0 / (dists**3 + 0.05)
            
        self.spatial_weights /= self.spatial_weights.sum(axis=1, keepdims=True)

    def get_config_options(self):
        band_opts = [(k, k) for k in BANDS.keys()]
        return[
            ("EDF File Path", "edf_path", self.edf_path, 'string'),
            ("Oscillatory Band", "selected_band", self.selected_band, band_opts),
            ("Local Diffusion (px)", "local_diffusion", self.local_diffusion, 'float'),
            ("Global Zoom 't' (px)", "global_diffusion", self.global_diffusion, 'float'),
            ("Fringe Density (k)", "fringe_density", self.fringe_density, 'float'),
            ("Aperture (1=Circ, 0=Sq)", "aperture_shape", self.aperture_shape, 'int'),
            ("Color (0=Gr,1=Tw,2=Oc)", "colormap", self.colormap, 'int')
        ]
        
    def set_config_options(self, options):
        if isinstance(options, dict):
            for k, v in options.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def _load_edf(self, path, band_name):
        if not MNE_AVAILABLE:
            self.status_msg = "Error: MNE/SciPy not installed."
            return

        if not os.path.exists(path):
            self.status_msg = "File not found."
            self.raw_data = None
            return

        try:
            self.status_msg = f"Loading {os.path.basename(path)}..."
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if raw.info['sfreq'] != self.sfreq:
                raw.resample(self.sfreq, verbose=False)
                
            ch_names_upper =[c.upper().replace(' ', '').replace('-', '').replace('.', '') for c in raw.ch_names]
            full_raw_data = np.zeros((self.n_ch, raw.n_times))
            
            found_count = 0
            for i, target_name in enumerate(ELEC_COORDS_2D.keys()):
                for file_ch_idx, file_ch_name in enumerate(ch_names_upper):
                    if target_name == file_ch_name:
                        full_raw_data[i, :] = raw.get_data()[found_index:=file_ch_idx, :]
                        found_count += 1
                        break
            
            if found_count == 0:
                self.status_msg = "Error: Channels missing!"
                self.raw_data = None
                return

            self.raw_data = full_raw_data
            
            # Apply Bandpass filter
            low_f, high_f = BANDS[band_name]
            b, a = butter(4,[low_f/(self.sfreq/2), high_f/(self.sfreq/2)], btype='band')
            filtered_data = np.array([filtfilt(b, a, ch) for ch in self.raw_data])
            
            # Apply Hilbert Transform to get Complex Analytic Signal
            self.complex_analytic = hilbert(filtered_data, axis=1)
            
            self.current_frame = 0
            self.status_msg = f"LIVE ({band_name[:5]}): {os.path.basename(path)}"
            
        except Exception as e:
            self.status_msg = "Load Error!"
            self.raw_data = None
            traceback.print_exc()

    def _apply_spatial_diffusion(self, complex_grid, sigma):
        """Blurs the complex signal to create macroscopic fields."""
        if sigma <= 0.1:
            return complex_grid
            
        real_part = np.real(complex_grid)
        imag_part = np.imag(complex_grid)
        
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        if ksize % 2 == 0: ksize += 1
        
        blurred_real = cv2.GaussianBlur(real_part, (ksize, ksize), sigma)
        blurred_imag = cv2.GaussianBlur(imag_part, (ksize, ksize), sigma)
        
        return blurred_real + 1j * blurred_imag

    def step(self):
        current_path = str(self.edf_path).strip().strip('\"').strip('\'')
        if current_path != self._last_path or self.selected_band != self._last_band:
            self._last_path = current_path
            self._last_band = self.selected_band
            if current_path != "":
                self._load_edf(current_path, self.selected_band)
            else:
                self.raw_data = None
                self.status_msg = "No EDF. Using Synthetic."

        time_sec = 0
        if self.complex_analytic is not None:
            if self.current_frame >= self.complex_analytic.shape[1]:
                self.current_frame = 0 
                
            # Get current 19-channel complex state
            current_Z = self.complex_analytic[:, self.current_frame]
            
            # Advance frame (~50ms jump for visualization)
            self.current_frame += int(self.sfreq * 0.05) 
            time_sec = self.current_frame / self.sfreq
        else:
            # Synthetic Travelling Wave for demo
            time_sec = self.current_frame * 0.05
            x_c = np.array([c[0] for c in ELEC_COORDS_2D.values()])
            y_c = np.array([c[1] for c in ELEC_COORDS_2D.values()])
            
            # A spiral/diagonal wave
            synth_phase = 2 * np.pi * 6.0 * time_sec + 4.0 * x_c + 2.0 * y_c
            current_Z = np.exp(1j * synth_phase)
            self.current_frame += 1

        # 1. Project discrete 19 electrodes onto 200x200 spatial grid
        Z_grid_flat = np.dot(self.spatial_weights, current_Z)
        Z_grid = Z_grid_flat.reshape((self.res, self.res))
        
        # 2. Extract Local vs Global Fields via Diffusion ('t' scaling)
        Z_local = self._apply_spatial_diffusion(Z_grid, self.local_diffusion)
        Z_global = self._apply_spatial_diffusion(Z_grid, self.global_diffusion)
        
        # Normalize amplitudes to pure phase (e^i*theta)
        Z_local /= (np.abs(Z_local) + 1e-9)
        Z_global /= (np.abs(Z_global) + 1e-9)
        
        # 3. INTERFEROMETRY: Calculate Phase Difference
        # Z1 * conj(Z2) gives the phase difference between the two fields
        Z_diff = Z_local * np.conj(Z_global)
        phase_diff = np.angle(Z_diff)
        
        # Moiré interference equation: cos(k * (Phase_local - Phase_global))
        moire_pattern = np.cos(self.fringe_density * phase_diff)
        
        # 4. Visualization & Masking
        # Map from [-1, 1] to [0, 255]
        display_img = ((moire_pattern + 1.0) * 127.5).astype(np.uint8)
        
        if self.colormap == 1:
            display_color = cv2.applyColorMap(display_img, cv2.COLORMAP_TWILIGHT_SHIFTED)
        elif self.colormap == 2:
            display_color = cv2.applyColorMap(display_img, cv2.COLORMAP_OCEAN)
        else:
            display_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
        # Apply Aperture
        if self.aperture_shape == 1:
            mask_2d = self.mask.reshape((self.res, self.res))  # Reshape mask to 2D
            display_color[~mask_2d] = 0
            cv2.circle(display_color, (self.res//2, self.res//2), int(self.res//2 * 0.95), (82, 61, 52), 2)
            
        # UI Overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_color, f"Local 't': {self.local_diffusion:.1f}px", (10, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display_color, f"Global 't': {self.global_diffusion:.1f}px", (10, 35), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display_color, f"k-Fringe: {self.fringe_density:.1f}x", (10, 50), font, 0.4, (255, 255, 255), 1)
        
        cv2.putText(display_color, self.status_msg, (10, self.res - 10), font, 0.4, (0, 255, 255), 1)
        
        self.cached_image = display_color

    def get_output(self, name):
        if name == 'interferogram':
            return self.cached_image
        return None

    def get_display_image(self):
        return self.cached_image