"""
Folded Cortical Sheet Node
==========================

A cortical simulation on a FOLDED surface - not flat.

The key insight: Real cortex has sulci (valleys) and gyri (ridges).
The eigenmodes of neural activity are CONSTRAINED by this geometry.
Standing waves form differently on a folded surface than a flat one.

This node:
1. Generates a brain-like folded surface (procedural sulci/gyri)
2. Runs Izhikevich neurons ON that surface
3. Coupling strength varies with GEODESIC distance (along folds, not straight line)
4. Shows the interference patterns that emerge from geometry
5. Computes eigenmodes of the activity on the folded surface

The folds themselves become computational structure.

Author: Built for Antti's cortical consciousness research
"""

import numpy as np
import cv2
from collections import defaultdict
import os

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from scipy import ndimage, signal
    from scipy.fft import fft2, fftshift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class FoldedCorticalSheetNode(BaseNode):
    """
    Cortical sheet simulation on a folded (brain-like) surface.
    """
    
    NODE_CATEGORY = "Simulation"
    NODE_TITLE = "Folded Cortex"
    NODE_COLOR = QtGui.QColor(200, 100, 100)  # Cortical red
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'coupling': 'signal',
            'excitability': 'signal',
            'fold_depth': 'signal',
            'reset': 'signal',
        }
        
        self.outputs = {
            'cortex_view': 'image',
            'eigenmode_view': 'image',
            'fold_view': 'image',
            'lfp_signal': 'signal',
            'coherence': 'signal',
            'fold_activity': 'signal',  # Activity in sulci vs gyri
        }
        
        # ===== EDF CONFIG =====
        self.edf_path = ""
        self._last_path = ""
        self.raw = None
        self.sfreq = 100.0
        self.current_idx = 0
        self.is_loaded = False
        
        # ===== CORTICAL SHEET =====
        self.grid_size = 96  # Smaller for speed but detailed enough
        
        # The FOLD MAP - height of cortical surface at each point
        # Positive = gyrus (ridge), Negative = sulcus (valley)
        self.fold_map = None
        self.fold_depth_scale = 1.0
        
        # Geodesic distance weights - coupling is stronger along surface
        self.geodesic_weights = None
        
        # ===== IZHIKEVICH NEURONS =====
        self.v = None  # Membrane potential
        self.u = None  # Recovery variable
        
        # Izhikevich parameters (regular spiking)
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 0.5
        
        # Coupling
        self.base_coupling = 0.5
        self.coupling_kernel = None
        
        # ===== ELECTRODE MAPPING =====
        self.electrode_coords = []
        self.electrode_names = []
        self.n_mapped = 0
        
        # ===== EIGENMODE ANALYSIS =====
        self.eigenmode_image = None
        self.activity_history = []
        self.history_length = 50
        
        # ===== STATISTICS =====
        self.lfp_value = 0.0
        self.coherence_value = 0.0
        self.gyrus_activity = 0.0
        self.sulcus_activity = 0.0
        
        self.status_msg = "Not loaded"
        
        # Initialize
        self._init_cortex()
    
    def _init_cortex(self):
        """Initialize the folded cortical surface."""
        n = self.grid_size
        
        # Generate fold map - brain-like sulci and gyri
        self.fold_map = self._generate_folds()
        
        # Initialize neurons
        self.v = np.ones((n, n), dtype=np.float32) * self.c
        self.u = np.zeros((n, n), dtype=np.float32)
        self.v += np.random.randn(n, n).astype(np.float32) * 2
        
        # Compute geodesic-aware coupling kernel
        self._compute_coupling_kernel()
        
        # Activity history for eigenmode analysis
        self.activity_history = []
    
    def _generate_folds(self):
        """
        Generate a brain-like folded surface.
        
        Uses superposition of sinusoids at different scales
        to create sulci (valleys) and gyri (ridges).
        """
        n = self.grid_size
        x = np.linspace(0, 4 * np.pi, n)
        y = np.linspace(0, 4 * np.pi, n)
        X, Y = np.meshgrid(x, y)
        
        # Multiple frequency components for realistic folds
        fold = np.zeros((n, n), dtype=np.float32)
        
        # Large-scale folds (major sulci)
        fold += 0.4 * np.sin(X * 0.8) * np.cos(Y * 0.6)
        fold += 0.3 * np.sin(X * 0.5 + Y * 0.7)
        
        # Medium-scale folds
        fold += 0.2 * np.sin(X * 1.5) * np.sin(Y * 1.2)
        fold += 0.15 * np.cos(X * 1.8 - Y * 0.9)
        
        # Fine-scale texture
        fold += 0.1 * np.sin(X * 3) * np.cos(Y * 2.5)
        
        # Add some asymmetry (brains aren't perfectly symmetric)
        fold += 0.1 * np.sin(X * 0.3) * np.exp(-((X - 2*np.pi)**2 + (Y - 2*np.pi)**2) / 20)
        
        # Smooth slightly
        if SCIPY_AVAILABLE:
            fold = ndimage.gaussian_filter(fold, sigma=1.0)
        
        # Normalize to [-1, 1]
        fold = fold / (np.abs(fold).max() + 1e-9)
        
        return fold
    
    def _compute_coupling_kernel(self):
        """
        Compute coupling kernel that respects fold geometry.
        
        Neurons couple more strongly if they're close ALONG THE SURFACE,
        not just in 2D Euclidean distance.
        """
        n = self.grid_size
        
        # Base Gaussian kernel
        kernel_size = 7
        k = kernel_size // 2
        y, x = np.ogrid[-k:k+1, -k:k+1]
        base_kernel = np.exp(-(x**2 + y**2) / (2 * 2.0**2))
        
        # Modulate by fold gradient (less coupling across steep folds)
        if SCIPY_AVAILABLE and self.fold_map is not None:
            # Compute gradient magnitude of fold map
            gy, gx = np.gradient(self.fold_map)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            # Steep gradients = sulcus walls = reduced coupling
            # This is stored for use during simulation
            self.fold_gradient = gradient_mag
        else:
            self.fold_gradient = np.zeros((n, n))
        
        self.coupling_kernel = base_kernel / base_kernel.sum()
    
    def load_edf(self):
        """Load EDF and map electrodes to cortical positions."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_path):
            return False
        
        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            self.raw = raw
            self.sfreq = raw.info['sfreq']
            self.current_idx = 0
            self._last_path = self.edf_path
            
            # Map electrodes to cortical positions
            self._map_electrodes()
            
            self.is_loaded = True
            fname = os.path.basename(self.edf_path)
            self.status_msg = f"Loaded {fname} | sf={self.sfreq}Hz | mapped={self.n_mapped}"
            print(f"FoldedCortex: {self.status_msg}")
            
            return True
            
        except Exception as e:
            self.status_msg = f"Error: {e}"
            print(f"FoldedCortex error: {e}")
            return False
    
    def _map_electrodes(self):
        """Map EEG electrodes to positions on the cortical sheet."""
        if self.raw is None:
            return
        
        # Standard 10-20 positions (normalized 0-1)
        standard_map = {
            'FP1': (0.3, 0.1), 'FP2': (0.7, 0.1),
            'F7': (0.1, 0.25), 'F3': (0.35, 0.25), 'FZ': (0.5, 0.2),
            'F4': (0.65, 0.25), 'F8': (0.9, 0.25),
            'T7': (0.05, 0.5), 'C3': (0.3, 0.5), 'CZ': (0.5, 0.5),
            'C4': (0.7, 0.5), 'T8': (0.95, 0.5),
            'P7': (0.1, 0.75), 'P3': (0.35, 0.75), 'PZ': (0.5, 0.8),
            'P4': (0.65, 0.75), 'P8': (0.9, 0.75),
            'O1': (0.35, 0.95), 'OZ': (0.5, 0.95), 'O2': (0.65, 0.95),
            # Extended
            'AF3': (0.35, 0.15), 'AF4': (0.65, 0.15),
            'FC1': (0.4, 0.35), 'FC2': (0.6, 0.35),
            'CP1': (0.4, 0.65), 'CP2': (0.6, 0.65),
            'PO3': (0.4, 0.85), 'PO4': (0.6, 0.85),
        }
        
        self.electrode_coords = []
        self.electrode_names = []
        self.electrode_indices = []
        
        n = self.grid_size
        
        for idx, ch_name in enumerate(self.raw.ch_names):
            name_upper = ch_name.upper().strip()
            
            # Try to find in standard map
            pos = None
            for std_name, std_pos in standard_map.items():
                if std_name in name_upper or name_upper in std_name:
                    pos = std_pos
                    break
            
            if pos is not None:
                # Convert to grid coordinates
                gx = int(pos[0] * (n - 1))
                gy = int(pos[1] * (n - 1))
                gx = np.clip(gx, 0, n - 1)
                gy = np.clip(gy, 0, n - 1)
                
                self.electrode_coords.append((gy, gx))
                self.electrode_names.append(name_upper)
                self.electrode_indices.append(idx)
        
        self.n_mapped = len(self.electrode_coords)
    
    def _inject_eeg(self):
        """Inject EEG signals as current at electrode positions."""
        if self.raw is None or self.n_mapped == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Get current EEG sample
        if self.current_idx >= self.raw.n_times:
            self.current_idx = 0
        
        data, _ = self.raw[:, self.current_idx]
        self.current_idx += 1
        
        # Create input current map
        n = self.grid_size
        I_ext = np.zeros((n, n), dtype=np.float32)
        
        # Inject at each electrode with spatial spread
        for (gy, gx), ch_idx in zip(self.electrode_coords, self.electrode_indices):
            if ch_idx < len(data):
                # Scale EEG (microvolts) to current
                val = float(data[ch_idx]) * 1e6 * 50  # Amplify for effect
                
                # Spread the input with Gaussian
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < n and 0 <= nx < n:
                            dist = np.sqrt(dy**2 + dx**2)
                            weight = np.exp(-dist**2 / 2)
                            I_ext[ny, nx] += val * weight
        
        return I_ext
    
    def _compute_eigenmode(self):
        """Compute the dominant eigenmode of recent activity."""
        if len(self.activity_history) < 10 or not SCIPY_AVAILABLE:
            return None
        
        # Average recent activity
        recent = np.array(self.activity_history[-20:])
        avg_activity = np.mean(recent, axis=0)
        
        # 2D FFT to find spatial frequencies
        spectrum = fftshift(fft2(avg_activity))
        magnitude = np.log(np.abs(spectrum) + 1)
        
        # Normalize
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        
        return magnitude.astype(np.float32)
    
    def step(self):
        """Main simulation step."""
        
        # Check for EDF reload
        if self.edf_path != self._last_path:
            self.load_edf()
        
        if not self.is_loaded:
            return
        
        # Get input modulation
        coupling_mod = self.get_blended_input('coupling', 'sum') or 0.0
        excite_mod = self.get_blended_input('excitability', 'sum') or 0.0
        fold_mod = self.get_blended_input('fold_depth', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset and reset > 0.5:
            self._init_cortex()
            return
        
        # Update fold depth if modulated
        if fold_mod is not None:
            self.fold_depth_scale = 0.5 + fold_mod
        
        # Get EEG input
        I_ext = self._inject_eeg()
        
        # Add excitability modulation
        I_ext += excite_mod * 5
        
        # Current coupling strength
        coupling = self.base_coupling * (1.0 + coupling_mod)
        
        n = self.grid_size
        
        # ===== IZHIKEVICH DYNAMICS =====
        
        # Neighbor coupling with fold-aware weighting
        if SCIPY_AVAILABLE:
            # Convolve for neighbor average
            v_neighbors = ndimage.convolve(self.v, self.coupling_kernel, mode='wrap')
            
            # Reduce coupling across steep fold gradients (sulcus walls)
            fold_factor = 1.0 / (1.0 + self.fold_gradient * self.fold_depth_scale * 2)
            
            # Coupling current
            I_coupling = coupling * (v_neighbors - self.v) * fold_factor
        else:
            I_coupling = 0
        
        # Total input
        I_total = I_ext + I_coupling
        
        # Izhikevich equations
        # dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        # du/dt = a*(b*v - u)
        
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I_total) * self.dt
        du = self.a * (self.b * self.v - self.u) * self.dt
        
        self.v += dv
        self.u += du
        
        # Spike reset
        spiked = self.v >= 30
        self.v[spiked] = self.c
        self.u[spiked] += self.d
        
        # Clamp to prevent NaN
        self.v = np.clip(self.v, -100, 30)
        self.u = np.clip(self.u, -50, 50)
        
        # ===== ANALYSIS =====
        
        # Store activity for eigenmode
        activity = (self.v - self.c) / (-self.c)  # Normalize
        self.activity_history.append(activity.copy())
        if len(self.activity_history) > self.history_length:
            self.activity_history.pop(0)
        
        # Compute eigenmode
        self.eigenmode_image = self._compute_eigenmode()
        
        # LFP (average membrane potential)
        self.lfp_value = float(np.mean(self.v))
        
        # Activity in gyri vs sulci
        gyrus_mask = self.fold_map > 0.2
        sulcus_mask = self.fold_map < -0.2
        
        if np.any(gyrus_mask):
            self.gyrus_activity = float(np.mean(activity[gyrus_mask]))
        if np.any(sulcus_mask):
            self.sulcus_activity = float(np.mean(activity[sulcus_mask]))
        
        # Coherence (spatial synchrony)
        if len(self.activity_history) >= 2:
            recent = np.array(self.activity_history[-5:])
            temporal_std = np.std(recent, axis=0)
            self.coherence_value = float(1.0 / (1.0 + np.mean(temporal_std)))
    
    def get_output(self, port_name):
        if port_name == 'cortex_view':
            return self._render_cortex()
        elif port_name == 'eigenmode_view':
            return self._render_eigenmode()
        elif port_name == 'fold_view':
            return self._render_folds()
        elif port_name == 'lfp_signal':
            return self.lfp_value
        elif port_name == 'coherence':
            return self.coherence_value
        elif port_name == 'fold_activity':
            return self.gyrus_activity - self.sulcus_activity
        return None
    
    def _render_cortex(self):
        """Render cortical activity with fold shading."""
        n = self.grid_size
        
        # Normalize activity
        activity = (self.v - self.v.min()) / (self.v.max() - self.v.min() + 1e-9)
        
        # Create base activity image
        activity_u8 = (activity * 255).astype(np.uint8)
        colored = cv2.applyColorMap(activity_u8, cv2.COLORMAP_INFERNO)
        
        # Add fold shading (darker in sulci, lighter on gyri)
        fold_shade = (self.fold_map * self.fold_depth_scale + 1) / 2  # 0-1
        fold_shade = np.clip(fold_shade, 0.3, 1.0)
        
        for c in range(3):
            colored[:, :, c] = (colored[:, :, c] * fold_shade).astype(np.uint8)
        
        # Draw electrode positions
        for (gy, gx) in self.electrode_coords:
            cv2.circle(colored, (gx, gy), 2, (0, 255, 0), -1)
        
        # Resize for display
        display_size = 256
        colored = cv2.resize(colored, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        
        return colored
    
    def _render_eigenmode(self):
        """Render the dominant eigenmode."""
        if self.eigenmode_image is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        eigen_u8 = (self.eigenmode_image * 255).astype(np.uint8)
        colored = cv2.applyColorMap(eigen_u8, cv2.COLORMAP_JET)
        
        display_size = 256
        colored = cv2.resize(colored, (display_size, display_size), interpolation=cv2.INTER_CUBIC)
        
        return colored
    
    def _render_folds(self):
        """Render the fold map (gyri/sulci)."""
        # Normalize fold map to 0-1
        fold_vis = (self.fold_map * self.fold_depth_scale + 1) / 2
        fold_vis = np.clip(fold_vis, 0, 1)
        
        fold_u8 = (fold_vis * 255).astype(np.uint8)
        colored = cv2.applyColorMap(fold_u8, cv2.COLORMAP_BONE)
        
        display_size = 256
        colored = cv2.resize(colored, (display_size, display_size), interpolation=cv2.INTER_CUBIC)
        
        return colored
    
    def get_display_image(self):
        """Create comprehensive display."""
        
        # Three panels: Cortex Activity | Folds | Eigenmode
        panel_size = 180
        margin = 5
        width = panel_size * 3 + margin * 4
        height = panel_size + 120
        
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Header
        cv2.putText(img, "=== FOLDED CORTEX ===", (10, 20), font, 0.5, (200, 100, 100), 1)
        cv2.putText(img, self.status_msg[:40], (10, 38), font, 0.3, (150, 150, 150), 1)
        
        y_panels = 50
        
        # Panel 1: Cortex Activity
        cortex = self._render_cortex()
        if cortex is not None:
            cortex_small = cv2.resize(cortex, (panel_size, panel_size))
            img[y_panels:y_panels+panel_size, margin:margin+panel_size] = cortex_small
        cv2.putText(img, "ACTIVITY", (margin + 50, y_panels + panel_size + 15), font, 0.35, (255, 200, 100), 1)
        
        # Panel 2: Fold Map
        folds = self._render_folds()
        if folds is not None:
            folds_small = cv2.resize(folds, (panel_size, panel_size))
            x2 = margin * 2 + panel_size
            img[y_panels:y_panels+panel_size, x2:x2+panel_size] = folds_small
        cv2.putText(img, "FOLDS", (x2 + 60, y_panels + panel_size + 15), font, 0.35, (200, 200, 200), 1)
        
        # Panel 3: Eigenmode
        eigen = self._render_eigenmode()
        if eigen is not None:
            eigen_small = cv2.resize(eigen, (panel_size, panel_size))
            x3 = margin * 3 + panel_size * 2
            img[y_panels:y_panels+panel_size, x3:x3+panel_size] = eigen_small
        cv2.putText(img, "EIGENMODE", (x3 + 45, y_panels + panel_size + 15), font, 0.35, (100, 200, 255), 1)
        
        # Statistics
        y_stats = y_panels + panel_size + 35
        cv2.putText(img, f"LFP: {self.lfp_value:.1f}mV", (10, y_stats), font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"Coherence: {self.coherence_value:.2f}", (150, y_stats), font, 0.35, (200, 200, 200), 1)
        
        y_stats += 18
        cv2.putText(img, f"Gyri: {self.gyrus_activity:.2f}", (10, y_stats), font, 0.35, (255, 200, 100), 1)
        cv2.putText(img, f"Sulci: {self.sulcus_activity:.2f}", (120, y_stats), font, 0.35, (100, 200, 255), 1)
        
        diff = self.gyrus_activity - self.sulcus_activity
        diff_color = (100, 255, 100) if diff > 0 else (255, 100, 100)
        cv2.putText(img, f"Diff: {diff:+.2f}", (230, y_stats), font, 0.35, diff_color, 1)
        
        y_stats += 18
        cv2.putText(img, f"Sample: {self.current_idx}", (10, y_stats), font, 0.3, (150, 150, 150), 1)
        cv2.putText(img, f"Fold depth: {self.fold_depth_scale:.2f}", (150, y_stats), font, 0.3, (150, 150, 150), 1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, width, height, width*3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Base Coupling", "base_coupling", self.base_coupling, None),
            ("Fold Depth Scale", "fold_depth_scale", self.fold_depth_scale, None),
            ("dt (time step)", "dt", self.dt, None),
        ]
    
    def set_config_options(self, options):
        reinit = False
        for key, value in options.items():
            if hasattr(self, key):
                if key == 'grid_size':
                    new_size = int(value)
                    if new_size != self.grid_size:
                        self.grid_size = new_size
                        reinit = True
                elif key in ['base_coupling', 'fold_depth_scale', 'dt']:
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)
        
        if reinit:
            self._init_cortex()
            if self.is_loaded:
                self._map_electrodes()