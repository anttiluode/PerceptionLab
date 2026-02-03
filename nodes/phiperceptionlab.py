"""
Phi Hologram Node
=================
Projects EEG phase data as a complex interference field (neural hologram).

Theory:
If the brain encodes information via phase interference (ephaptic fields),
we can mathematically reverse-project those phase differences onto a 2D plane
to reveal the "Interference Geometry" of the current state.

Mechanism:
1. Treat electrodes as "emitters" on a 2D plane (top-down view).
2. For every pixel on the canvas, calculate the phase sum from all electrodes.
3. WAVE INTERFERENCE: 
   Intensity(x,y) = | Sum( Amplitude_i * exp(j * (Phase_i - Distance_i * k)) ) |

The PHI TWIST:
Apply a rotation to the phase based on the "Golden Angle" or user-selected 
offset to see if the image "snaps" into focus.

Outputs holograms for ALL frequency bands simultaneously.

Author: Built for Antti's consciousness crystallography research
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
    Outputs images for all frequency bands simultaneously.
    """
    
    NODE_NAME = "Phi Hologram"
    NODE_TITLE = "Phi Hologram"
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(180, 40, 220) if QtGui else None  # Purple
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS (signal controls) ===
        self.inputs = {
            'phase_rot': 'signal',      # Phase rotation in degrees (-90 to 90)
            'spatial_k': 'signal',      # Spatial frequency / wave number (1-50)
            'speed': 'signal',          # Playback speed multiplier
            'reset': 'signal',          # Reset to beginning
        }
        
        # === OUTPUTS ===
        # Image outputs for each band
        self.outputs = {
            # Per-band hologram images
            'delta_hologram': 'image',
            'theta_hologram': 'image',
            'alpha_hologram': 'image',
            'beta_hologram': 'image',
            'gamma_hologram': 'image',
            
            # Combined/main view
            'combined_hologram': 'image',
            
            # Complex field outputs (for FFT pipeline)
            'delta_field': 'complex_spectrum',
            'theta_field': 'complex_spectrum',
            'alpha_field': 'complex_spectrum',
            'beta_field': 'complex_spectrum',
            'gamma_field': 'complex_spectrum',
            
            # Band power signals (for other nodes)
            'delta_power': 'signal',
            'theta_power': 'signal',
            'alpha_power': 'signal',
            'beta_power': 'signal',
            'gamma_power': 'signal',
            
            # Time info
            'time_seconds': 'signal',
        }
        
        # === CONFIG (saved/loaded) ===
        self.edf_file_path = ""
        self.resolution = 128
        self.default_k = 15.0
        self.default_rot = 0.0
        self.display_band = "alpha"  # Which band to show in node display
        
        # === INTERNAL STATE ===
        self._last_path = ""
        self.raw = None
        self.channel_data = {}
        self.active_elecs = {}
        self.times = None
        self.sfreq = 256.0
        self.current_idx = 0
        self.loaded_path = ""
        self.status_msg = "No EDF loaded"
        
        # Pre-computed grids
        self.X = None
        self.Y = None
        self.dist_maps = {}
        
        # Phase cache per band
        self.band_phases = {band: {} for band in BANDS}  # band -> {elec: phase_array}
        
        # Output caches
        self.field_cache = {band: None for band in BANDS}
        self.image_cache = {band: None for band in BANDS}
        self.power_cache = {band: 0.0 for band in BANDS}
        self.combined_image = None
        
        # Display
        self.display_image = None
        self._init_display()
    
    # === CONFIG SYSTEM (Required for PerceptionLab) ===
    
    def get_config_options(self):
        """Return config options for the sidebar."""
        band_options = [(b, b) for b in BANDS.keys()]
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Resolution", "resolution", self.resolution, None),
            ("Default Spatial K", "default_k", self.default_k, None),
            ("Default Phase Rot", "default_rot", self.default_rot, None),
            ("Display Band", "display_band", self.display_band, band_options),
        ]
    
    def set_config_options(self, options):
        """Apply config options."""
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            # Trigger reload if path changed
            if 'edf_file_path' in options:
                self._last_path = ""  # Force reload check
    
    # === INITIALIZATION ===
    
    def _init_grid(self, res):
        """Pre-calculate the spatial grid and distance maps."""
        x = np.linspace(-1.5, 1.5, res).astype(np.float32)
        y = np.linspace(-1.5, 1.5, res).astype(np.float32)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Rebuild distance maps for all electrodes
        self.dist_maps = {}
        for name, (ex, ey) in self.active_elecs.items():
            self.dist_maps[name] = np.sqrt((self.X - ex)**2 + (self.Y - ey)**2).astype(np.float32)
    
    def _init_display(self):
        """Create initial display image."""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "PHI HOLOGRAM", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 40, 220), 2)
        cv2.putText(img, "Load EDF file", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "in config panel", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        self.display_image = img
    
    # === EDF LOADING ===
    
    def _load_edf(self):
        """Load EDF file and extract channel data."""
        if not self.edf_file_path or not os.path.exists(self.edf_file_path):
            self.status_msg = "File not found"
            return False
        
        if not MNE_AVAILABLE:
            self.status_msg = "MNE not installed"
            return False
        
        try:
            print(f"[PhiHologram] Loading: {self.edf_file_path}")
            self.raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose='error')
            self.sfreq = self.raw.info['sfreq']
            self.times = self.raw.times
            self.current_idx = 0
            self.loaded_path = self.edf_file_path
            
            # Map channels to standard positions
            self.channel_data = {}
            self.active_elecs = {}
            
            ch_names = self.raw.ch_names
            for std_name, pos in ELECTRODE_POS.items():
                # Try to find matching channel
                match = None
                for ch in ch_names:
                    ch_clean = ch.strip().replace('.', '').upper()
                    if std_name.upper() in ch_clean or ch_clean in std_name.upper():
                        match = ch
                        break
                
                if match:
                    self.channel_data[std_name] = self.raw.get_data(picks=[match])[0]
                    self.active_elecs[std_name] = pos
            
            # Initialize grid
            self._init_grid(self.resolution)
            
            # Pre-compute phases for all bands
            self._compute_all_phases()
            
            self.status_msg = f"Loaded: {len(self.active_elecs)} channels"
            print(f"[PhiHologram] {self.status_msg}")
            return True
            
        except Exception as e:
            self.status_msg = f"Load error: {str(e)[:30]}"
            print(f"[PhiHologram] {self.status_msg}")
            return False
    
    def _compute_all_phases(self):
        """Pre-compute Hilbert phases for all bands and all electrodes."""
        if not self.channel_data:
            return
        
        for band_name, (lo, hi) in BANDS.items():
            self.band_phases[band_name] = {}
            
            for elec_name, data in self.channel_data.items():
                try:
                    # Bandpass filter
                    nyq = self.sfreq / 2.0
                    lo_norm = max(0.5, lo) / nyq
                    hi_norm = min(hi, nyq - 1) / nyq
                    
                    if lo_norm >= hi_norm:
                        self.band_phases[band_name][elec_name] = np.zeros_like(data)
                        continue
                    
                    b, a = scipy.signal.butter(3, [lo_norm, hi_norm], btype='band')
                    filtered = scipy.signal.filtfilt(b, a, data)
                    
                    # Hilbert transform for instantaneous phase
                    analytic = scipy.signal.hilbert(filtered)
                    self.band_phases[band_name][elec_name] = np.angle(analytic).astype(np.float32)
                    
                except Exception as e:
                    print(f"[PhiHologram] Phase compute error for {elec_name}/{band_name}: {e}")
                    self.band_phases[band_name][elec_name] = np.zeros_like(data)
    
    # === HOLOGRAM GENERATION ===
    
    def _generate_hologram(self, band_name, time_idx, k, rot_deg):
        """Generate hologram for a specific band at a specific time."""
        if not self.active_elecs or self.X is None:
            return None, None
        
        res = self.X.shape[0]
        field = np.zeros((res, res), dtype=np.complex64)
        rot_rad = np.deg2rad(rot_deg)
        
        total_amplitude = 0.0
        
        for elec_name in self.active_elecs:
            if elec_name not in self.band_phases[band_name]:
                continue
            if elec_name not in self.dist_maps:
                continue
            
            # Get phase at this time
            phases = self.band_phases[band_name][elec_name]
            if time_idx >= len(phases):
                continue
            
            phi = phases[time_idx]
            dist = self.dist_maps[elec_name]
            
            # Wave equation: exp(j * (phase - k*distance + rotation))
            theta = phi - (k * dist) + rot_rad
            wave = np.cos(theta) + 1j * np.sin(theta)
            field += wave
            total_amplitude += 1.0
        
        if total_amplitude == 0:
            return None, None
        
        # Normalize by number of electrodes
        field = field / total_amplitude
        
        # Intensity = |field|^2
        intensity = np.abs(field) ** 2
        
        # Normalize for display
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)
        
        # Create colored image
        img_u8 = (np.clip(intensity, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        
        # Compute band power (mean intensity)
        power = float(np.mean(intensity))
        
        return field, colored, power
    
    # === MAIN STEP ===
    
    def step(self):
        """Main processing step - called each frame."""
        
        # Check if we need to load/reload
        if self.edf_file_path != self._last_path:
            self._last_path = self.edf_file_path
            if self.edf_file_path:
                self._load_edf()
        
        if self.raw is None or not self.active_elecs:
            return
        
        # Get input parameters
        rot = self.get_blended_input('phase_rot', 'sum')
        if rot is None:
            rot = self.default_rot
        rot = float(np.clip(rot, -90, 90))
        
        k = self.get_blended_input('spatial_k', 'sum')
        if k is None:
            k = self.default_k
        k = float(np.clip(k, 1.0, 50.0))
        
        speed = self.get_blended_input('speed', 'sum')
        if speed is None:
            speed = 1.0
        speed = float(np.clip(speed, 0.1, 10.0))
        
        reset = self.get_blended_input('reset', 'sum')
        if reset is not None and reset > 0.5:
            self.current_idx = 0
        
        # Advance time
        step_size = int((self.sfreq / 30.0) * speed)  # ~30fps target
        self.current_idx = (self.current_idx + step_size) % len(self.times)
        
        # Generate holograms for ALL bands
        for band_name in BANDS:
            result = self._generate_hologram(band_name, self.current_idx, k, rot)
            if result[0] is not None:
                self.field_cache[band_name] = result[0]
                self.image_cache[band_name] = result[1]
                self.power_cache[band_name] = result[2]
        
        # Create combined view (stack alpha, beta, gamma as RGB)
        self._create_combined_view()
        
        # Update display
        self._update_display(k, rot)
    
    def _create_combined_view(self):
        """Create combined RGB hologram from multiple bands."""
        alpha = self.image_cache.get('alpha')
        beta = self.image_cache.get('beta')
        gamma = self.image_cache.get('gamma')
        
        if alpha is None:
            return
        
        h, w = alpha.shape[:2]
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Alpha -> Green, Beta -> Blue, Gamma -> Red
        if alpha is not None:
            combined[:, :, 1] = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        if beta is not None:
            combined[:, :, 0] = cv2.cvtColor(beta, cv2.COLOR_BGR2GRAY)
        if gamma is not None:
            combined[:, :, 2] = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)
        
        self.combined_image = combined
    
    def _update_display(self, k, rot):
        """Update the node's display image."""
        # Get the selected band's hologram
        band_img = self.image_cache.get(self.display_band)
        
        if band_img is None:
            self._init_display()
            return
        
        # Create display with info overlay
        h, w = band_img.shape[:2]
        
        # Scale up if small
        if h < 200:
            scale = 200 // h + 1
            band_img = cv2.resize(band_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            h, w = band_img.shape[:2]
        
        # Create display with info panel
        panel_h = 60
        display = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        display[panel_h:, :] = band_img
        
        # Info panel
        t_sec = self.times[self.current_idx] if self.times is not None else 0
        
        cv2.putText(display, f"PHI HOLOGRAM [{self.display_band.upper()}]", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 40, 220), 1)
        cv2.putText(display, f"T={t_sec:.2f}s  K={k:.1f}  Rot={rot:.1f}", (5, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Band powers mini bar
        bar_y = 45
        bar_w = 20
        for i, (band, power) in enumerate(self.power_cache.items()):
            x = 5 + i * (bar_w + 5)
            bar_h = int(power * 12)
            cv2.rectangle(display, (x, bar_y), (x + bar_w, bar_y - bar_h), (100, 200, 255), -1)
            cv2.putText(display, band[0].upper(), (x + 3, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
        
        # Draw electrode positions on image
        scale_x = w / 3.0
        scale_y = (h - panel_h) / 3.0
        offset_x = w / 2
        offset_y = panel_h + (h - panel_h) / 2
        
        for elec, (ex, ey) in self.active_elecs.items():
            px = int(offset_x + ex * scale_x)
            py = int(offset_y - ey * scale_y)  # Flip Y
            cv2.circle(display, (px, py), 2, (0, 255, 255), -1)
        
        self.display_image = display
    
    # === OUTPUT METHODS ===
    
    def get_output(self, port_name):
        """Return output for the specified port."""
        
        # Hologram images
        if port_name == 'delta_hologram':
            return self.image_cache.get('delta')
        elif port_name == 'theta_hologram':
            return self.image_cache.get('theta')
        elif port_name == 'alpha_hologram':
            return self.image_cache.get('alpha')
        elif port_name == 'beta_hologram':
            return self.image_cache.get('beta')
        elif port_name == 'gamma_hologram':
            return self.image_cache.get('gamma')
        elif port_name == 'combined_hologram':
            return self.combined_image
        
        # Complex fields
        elif port_name == 'delta_field':
            return self.field_cache.get('delta')
        elif port_name == 'theta_field':
            return self.field_cache.get('theta')
        elif port_name == 'alpha_field':
            return self.field_cache.get('alpha')
        elif port_name == 'beta_field':
            return self.field_cache.get('beta')
        elif port_name == 'gamma_field':
            return self.field_cache.get('gamma')
        
        # Power signals
        elif port_name == 'delta_power':
            return self.power_cache.get('delta', 0.0)
        elif port_name == 'theta_power':
            return self.power_cache.get('theta', 0.0)
        elif port_name == 'alpha_power':
            return self.power_cache.get('alpha', 0.0)
        elif port_name == 'beta_power':
            return self.power_cache.get('beta', 0.0)
        elif port_name == 'gamma_power':
            return self.power_cache.get('gamma', 0.0)
        
        # Time
        elif port_name == 'time_seconds':
            if self.times is not None and self.current_idx < len(self.times):
                return float(self.times[self.current_idx])
            return 0.0
        
        return None
    
    def get_display_image(self):
        """Return the node's display image."""
        return self.display_image
    
    # === CLEANUP ===
    
    def close(self):
        """Cleanup when node is removed."""
        self.raw = None
        self.channel_data = {}
        self.band_phases = {}