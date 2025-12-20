"""
Source Localization Node - MNE Inverse Solution for PerceptionLab
==================================================================

This node bridges the gap between electrode-space EEG and cortical source space.
It performs proper source localization using MNE's inverse solutions, then
outputs source-space data that can feed into anatomically-valid eigenmode analysis.

PIPELINE:
1. Load EEG (EDF/FIF/etc)
2. Set up fsaverage source space + BEM model
3. Create forward solution (electrode -> source)
4. Create inverse operator
5. Apply inverse to get source estimates
6. Compute source-space eigenmodes
7. Project source activity onto eigenmodes
8. Output eigenmode activations (can feed into phase analyzer, etc.)

This makes the full pipeline anatomically valid:
EEG -> Source Localization -> Source Eigenmodes -> Phase/Dynamics Analysis

INPUTS:
- gain_mod: Amplification control
- speed_mod: Playback speed control

OUTPUTS:
- source_image: 2D flatmap of source activity
- mode_spectrum: Eigenmode activations (source-space)
- complex_modes: Complex eigenmode activations with phase
- band_spectrum: Frequency band powers at source level
- raw_source: Raw source activity signal

Created: December 2025
"""

import numpy as np
import cv2
from pathlib import Path
from collections import deque
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import gaussian_filter

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
        def get_blended_input(self, name, mode):
            return None

# MNE imports
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("[SourceLocalizationNode] MNE not available - install with: pip install mne")


class SourceLocalizationNode(BaseNode):
    """
    Performs MNE source localization and eigenmode decomposition in source space.
    """
    NODE_CATEGORY = "EEG"
    NODE_TITLE = "Source Localization"
    NODE_COLOR = QtGui.QColor(50, 150, 200)  # Blue for EEG/source
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'gain_mod': 'signal',
            'speed_mod': 'signal',
        }
        
        self.outputs = {
            # Images
            'source_image': 'image',          # 2D flatmap
            'eigenmode_image': 'image',       # Mode activations over time
            'lh_image': 'image',              # Left hemisphere
            'rh_image': 'image',              # Right hemisphere
            
            # Spectra (for downstream nodes)
            'mode_spectrum': 'spectrum',       # Real eigenmode activations (10-dim)
            'complex_modes': 'complex_spectrum', # Complex with phase
            'band_spectrum': 'spectrum',       # Frequency bands (5-dim)
            'full_spectrum': 'spectrum',       # Modes + bands combined (15-dim)
            
            # Signals
            'mode_1': 'signal',
            'mode_2': 'signal',
            'mode_3': 'signal',
            'mode_4': 'signal',
            'mode_5': 'signal',
            'mode_6': 'signal',
            'mode_7': 'signal',
            'mode_8': 'signal',
            'mode_9': 'signal',
            'mode_10': 'signal',
            'delta_power': 'signal',
            'theta_power': 'signal',
            'alpha_power': 'signal',
            'beta_power': 'signal',
            'gamma_power': 'signal',
            'raw_source': 'signal',
            'dominant_mode': 'signal',
        }
        
        # === CONFIG ===
        self.edf_path = ""
        self.source_spacing = 'oct5'  # oct5 is good balance of speed/resolution
        self.inverse_method = 'sLORETA'  # sLORETA, dSPM, MNE, eLORETA
        self.n_eigenmodes = 20  # Eigenmodes to compute
        self.n_output_modes = 10  # Modes to output
        self.time_window = 0.5  # Seconds per frame
        self.amplification = 1e6
        
        # Frequency bands
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
        
        # === STATE ===
        self.is_loaded = False
        self.is_initialized = False
        self.load_error = ""
        
        # MNE objects
        self.raw = None
        self.src = None
        self.fwd = None
        self.inv = None
        self.stc = None
        self.subjects_dir = None
        
        # Eigenmode data
        self.eigenmodes = {'lh': None, 'rh': None}
        self.eigenvalues = {'lh': None, 'rh': None}
        self.n_lh_vertices = 0
        self.n_rh_vertices = 0
        
        # Flatmap projection data
        self.flatmap_coords = {'lh': None, 'rh': None}
        self.flatmap_size = (200, 400)  # H x W (both hemispheres)
        
        # Playback state
        self.source_data = None  # (n_vertices, n_times)
        self.times = None
        self.fs = 100.0
        self.playback_idx = 0
        self.frame_count = 0
        
        # Current outputs
        self._mode_activations = np.zeros(self.n_output_modes)
        self._complex_modes = np.zeros(self.n_output_modes, dtype=np.complex64)
        self._band_powers = np.zeros(5)
        self._source_image = None
        self._eigenmode_image = None
        
        # Mode history for visualization
        self.mode_history = deque(maxlen=100)
        
        # Initialize MNE
        if MNE_AVAILABLE:
            self._init_mne()
        
        # Threading for heavy computation
        self._loading_thread = None
        self._loading = False
    
    def _init_mne(self):
        """Initialize MNE fsaverage data"""
        try:
            # Set up subjects directory
            self.subjects_dir = Path.home() / 'mne_data'
            self.subjects_dir.mkdir(exist_ok=True)
            
            # Check/download fsaverage
            fsaverage_path = self.subjects_dir / 'fsaverage'
            if not fsaverage_path.exists():
                print("[SourceLocalization] Downloading fsaverage (first time only)...")
                mne.datasets.fetch_fsaverage(subjects_dir=str(self.subjects_dir), verbose=False)
            
            print("[SourceLocalization] fsaverage ready")
            self.is_initialized = True
            
        except Exception as e:
            self.load_error = f"MNE init error: {e}"
            print(f"[SourceLocalization] {self.load_error}")
    
    def _load_eeg_threaded(self):
        """Background thread for loading EEG"""
        self._loading = True
        try:
            self._load_eeg_impl()
        finally:
            self._loading = False
    
    def _load_eeg(self):
        """Start loading EEG in background thread"""
        if self._loading:
            return False
        
        import threading
        self._loading_thread = threading.Thread(target=self._load_eeg_threaded, daemon=True)
        self._loading_thread.start()
        return True
    
    def _load_eeg_impl(self):
        """Load EEG file and set up source localization (runs in background)"""
        if not self.edf_path or not Path(self.edf_path).exists():
            self.load_error = "No valid EEG file path"
            return False
        
        try:
            print(f"[SourceLocalization] Loading {self.edf_path}...")
            
            # Load EEG
            if self.edf_path.endswith('.fif'):
                self.raw = mne.io.read_raw_fif(self.edf_path, preload=True, verbose=False)
            elif self.edf_path.endswith('.edf') or self.edf_path.endswith('.bdf'):
                self.raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            else:
                self.raw = mne.io.read_raw(self.edf_path, preload=True, verbose=False)
            
            # Pick EEG channels
            self.raw.pick(['eeg'], exclude='bads')
            
            # Normalize channel names - handle common issues
            mapping = {}
            for ch in self.raw.ch_names:
                new_name = ch.strip()
                # Remove trailing dots (common in some EDF files)
                new_name = new_name.rstrip('.')
                # Remove trailing numbers after dash (e.g., "Fp1-0" -> "Fp1")
                if '-' in new_name:
                    new_name = new_name.split('-')[0]
                # Standardize case: first letter upper, rest as needed
                # Standard 10-20 uses: Fp1, Fz, Cz, Pz, Oz, F3, C3, P3, O1, etc.
                if len(new_name) >= 2:
                    # Handle special prefixes
                    if new_name.upper().startswith(('FP', 'AF', 'FC', 'FT', 'CP', 'TP', 'PO')):
                        new_name = new_name[:2].capitalize() + new_name[2:]
                    else:
                        new_name = new_name[0].upper() + new_name[1:]
                    # Ensure 'z' is lowercase for midline electrodes
                    new_name = new_name.replace('Z', 'z')
                
                if new_name != ch:
                    mapping[ch] = new_name
            
            if mapping:
                print(f"[SourceLocalization] Renaming channels: {list(mapping.items())[:5]}...")
                self.raw.rename_channels(mapping)
            
            # Apply standard montage
            montage = mne.channels.make_standard_montage('standard_1020')
            montage_ch_names_upper = [ch.upper() for ch in montage.ch_names]
            montage_ch_names_set = set(montage.ch_names)
            
            # Find channels that exist in montage
            valid_channels = []
            missing_channels = []
            for ch in self.raw.ch_names:
                # Direct match
                if ch in montage_ch_names_set:
                    valid_channels.append(ch)
                # Case-insensitive match
                elif ch.upper() in montage_ch_names_upper:
                    valid_channels.append(ch)
                else:
                    missing_channels.append(ch)
            
            if missing_channels:
                print(f"[SourceLocalization] Dropping {len(missing_channels)} channels without montage positions: {missing_channels[:10]}...")
                if len(valid_channels) < 10:
                    print(f"[SourceLocalization] WARNING: Only {len(valid_channels)} valid channels found!")
                    print(f"[SourceLocalization] Valid: {valid_channels}")
                    print(f"[SourceLocalization] Montage expects names like: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2...")
                self.raw.drop_channels(missing_channels)
            
            if len(self.raw.ch_names) < 10:
                raise ValueError(f"Only {len(self.raw.ch_names)} channels remain after dropping unmapped channels. Check channel names.")
            
            # Now apply montage
            try:
                self.raw.set_montage(montage, on_missing='ignore', match_case=False)
            except Exception as e:
                print(f"[SourceLocalization] Montage warning: {e}")
            
            # Verify all channels have locations
            chs_without_loc = []
            for ch in self.raw.info['chs']:
                if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH:
                    loc = ch['loc'][:3]
                    if np.allclose(loc, 0):
                        chs_without_loc.append(ch['ch_name'])
            
            if chs_without_loc:
                print(f"[SourceLocalization] Dropping {len(chs_without_loc)} channels without locations: {chs_without_loc}")
                self.raw.drop_channels(chs_without_loc)
            
            print(f"[SourceLocalization] {len(self.raw.ch_names)} channels with valid positions: {self.raw.ch_names[:10]}...")
            
            # Set average reference
            self.raw.set_eeg_reference('average', projection=True)
            self.raw.apply_proj()
            
            # Filter
            self.raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
            
            # Resample if needed
            if self.raw.info['sfreq'] > 150:
                self.raw.resample(100, verbose=False)
            
            self.fs = self.raw.info['sfreq']
            
            print(f"[SourceLocalization] Loaded: {len(self.raw.ch_names)} channels, {self.raw.times[-1]:.1f}s, {self.fs}Hz")
            
            # Set up source space
            self._setup_source_space()
            
            # Compute forward and inverse
            self._compute_inverse()
            
            # Apply inverse to get source estimates
            self._apply_inverse()
            
            # Compute source-space eigenmodes
            self._compute_source_eigenmodes()
            
            # Set up flatmap projection
            self._setup_flatmap()
            
            self.is_loaded = True
            self.load_error = ""
            return True
            
        except Exception as e:
            self.load_error = str(e)
            print(f"[SourceLocalization] Load error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_source_space(self):
        """Set up cortical source space"""
        print("[SourceLocalization] Setting up source space...")
        
        self.src = mne.setup_source_space(
            'fsaverage',
            spacing=self.source_spacing,
            subjects_dir=str(self.subjects_dir),
            add_dist=False,
            verbose=False
        )
        
        self.n_lh_vertices = len(self.src[0]['vertno'])
        self.n_rh_vertices = len(self.src[1]['vertno'])
        
        print(f"[SourceLocalization] Source space: LH={self.n_lh_vertices}, RH={self.n_rh_vertices} vertices")
    
    def _compute_inverse(self):
        """Compute forward solution and inverse operator"""
        print("[SourceLocalization] Computing BEM model...")
        
        # BEM model (3-layer for EEG)
        bem_model = mne.make_bem_model(
            'fsaverage',
            ico=4,
            conductivity=(0.3, 0.006, 0.3),  # brain, skull, scalp
            subjects_dir=str(self.subjects_dir),
            verbose=False
        )
        bem_sol = mne.make_bem_solution(bem_model, verbose=False)
        
        print("[SourceLocalization] Computing forward solution...")
        
        # Forward solution
        self.fwd = mne.make_forward_solution(
            self.raw.info,
            trans='fsaverage',
            src=self.src,
            bem=bem_sol,
            eeg=True,
            mindist=5.0,
            verbose=False
        )
        
        print("[SourceLocalization] Computing inverse operator...")
        
        # Noise covariance (from data)
        noise_cov = mne.compute_raw_covariance(self.raw, method='empirical', verbose=False)
        
        # Inverse operator
        self.inv = mne.minimum_norm.make_inverse_operator(
            self.raw.info,
            self.fwd,
            noise_cov,
            loose=0.2,
            depth=0.8,
            verbose=False
        )
        
        print("[SourceLocalization] Inverse operator ready")
    
    def _apply_inverse(self):
        """Apply inverse solution to get source time series"""
        print("[SourceLocalization] Applying inverse solution...")
        
        # Method parameters
        method_params = {
            'sLORETA': {'method': 'sLORETA', 'lambda2': 1.0 / 9.0},
            'dSPM': {'method': 'dSPM', 'lambda2': 1.0 / 9.0},
            'MNE': {'method': 'MNE', 'lambda2': 1.0 / 9.0},
            'eLORETA': {'method': 'eLORETA', 'lambda2': 1.0 / 9.0},
        }
        
        params = method_params.get(self.inverse_method, method_params['sLORETA'])
        
        # Apply inverse
        self.stc = mne.minimum_norm.apply_inverse_raw(
            self.raw,
            self.inv,
            lambda2=params['lambda2'],
            method=params['method'],
            verbose=False
        )
        
        self.source_data = self.stc.data
        self.times = self.stc.times
        
        print(f"[SourceLocalization] Source data: {self.source_data.shape} (vertices x time)")
    
    def _compute_source_eigenmodes(self):
        """Compute eigenmodes on source-space mesh"""
        print("[SourceLocalization] Computing source-space eigenmodes...")
        
        for hemi_idx, hemi in enumerate(['lh', 'rh']):
            surf = self.src[hemi_idx]
            vertno = surf['vertno']  # Indices of vertices we're using
            n_vertices = len(vertno)
            
            # Create mapping from full surface indices to our subset indices
            # vertno contains the original vertex indices we're using
            # We need to map those to 0, 1, 2, ... n_vertices-1
            full_to_subset = {v: i for i, v in enumerate(vertno)}
            
            # Get triangles that use only our vertices
            # use_tris contains triangles in terms of SUBSET indices already
            # But let's verify and build adjacency properly
            
            # Actually, let's build adjacency directly from vertex positions
            # This is more robust than relying on triangle indexing
            rr = surf['rr'][vertno]  # Positions of our vertices
            
            # Build adjacency based on distance (k-nearest neighbors)
            from scipy.spatial import cKDTree
            
            tree = cKDTree(rr)
            k_neighbors = 7  # Each vertex connected to ~6 neighbors on cortex
            
            # Find k nearest neighbors for each vertex
            distances, indices = tree.query(rr, k=k_neighbors)
            
            # Build adjacency matrix
            rows = []
            cols = []
            data = []
            
            for i in range(n_vertices):
                for j_idx in range(1, k_neighbors):  # Skip self (index 0)
                    j = indices[i, j_idx]
                    if j < n_vertices:  # Valid neighbor
                        rows.append(i)
                        cols.append(j)
                        # Weight by inverse distance (optional, can use 1.0)
                        w = 1.0 / (distances[i, j_idx] + 1e-10)
                        data.append(w)
            
            A = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
            A = A.tocsr()
            
            # Make symmetric
            A = A.maximum(A.T)
            
            # Graph Laplacian
            d = np.array(A.sum(axis=1)).ravel()
            D = diags(d, dtype=np.float32)
            L = (D - A).astype(np.float32)
            
            # Normalize
            d_inv_sqrt = 1.0 / np.sqrt(d + 1e-10)
            D_inv_sqrt = diags(d_inv_sqrt, dtype=np.float32)
            L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
            
            # Regularize
            L_reg = L_normalized + 1e-8 * diags(np.ones(n_vertices), dtype=np.float32)
            
            # Compute eigenmodes
            n_modes = min(self.n_eigenmodes + 1, n_vertices - 2)
            
            try:
                evals, evecs = eigsh(L_reg.tocsr(), k=n_modes, which='SM', tol=1e-4, maxiter=5000)
            except Exception as e:
                print(f"[SourceLocalization] Eigenmode computation warning: {e}")
                # Fallback: use non-normalized Laplacian
                L_reg = L + 1e-6 * diags(np.ones(n_vertices), dtype=np.float32)
                evals, evecs = eigsh(L_reg.tocsr(), k=n_modes, which='SM', tol=1e-3, maxiter=3000)
            
            # Sort by eigenvalue
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            # Skip first (constant) mode
            self.eigenmodes[hemi] = evecs[:, 1:].astype(np.float32)
            self.eigenvalues[hemi] = evals[1:]
            
            print(f"[SourceLocalization] {hemi.upper()}: {n_modes-1} eigenmodes, λ range: {evals[1]:.4f} - {evals[-1]:.4f}")
    
    def _setup_flatmap(self):
        """Set up 2D flatmap projection coordinates"""
        H, W = self.flatmap_size
        half_W = W // 2
        
        for hemi_idx, hemi in enumerate(['lh', 'rh']):
            surf = self.src[hemi_idx]
            vertno = surf['vertno']
            rr = surf['rr'][vertno]
            
            # Use Y-Z projection (lateral view) instead of spherical
            # This avoids wraparound artifacts
            y = rr[:, 1]  # anterior-posterior
            z = rr[:, 2]  # superior-inferior
            
            # Normalize to [0, 1]
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
            
            y_norm = (y - y_min) / (y_max - y_min + 1e-10)
            z_norm = (z - z_min) / (z_max - z_min + 1e-10)
            
            # Map to pixel coordinates with padding
            pad = 0.05
            
            if hemi == 'lh':
                # Left hemisphere goes on left side of image
                px = (pad * half_W + (1 - 2*pad) * y_norm * (half_W - 1)).astype(np.int32)
            else:
                # Right hemisphere goes on right side
                px = (half_W + pad * half_W + (1 - 2*pad) * y_norm * (half_W - 1)).astype(np.int32)
            
            # Z maps to vertical (flip so superior is at top)
            py = ((1 - z_norm) * (1 - 2*pad) * (H - 1) + pad * H).astype(np.int32)
            
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)
            
            self.flatmap_coords[hemi] = {'px': px, 'py': py, 'n': len(vertno)}
    
    def step(self):
        """Process one frame of source-localized data"""
        self.frame_count += 1
        
        # Check if we need to load (and not already loading)
        if not self.is_loaded and self.edf_path and not self._loading:
            if hasattr(self, '_last_path') and self._last_path != self.edf_path:
                self._load_eeg()
            elif not hasattr(self, '_last_path'):
                self._load_eeg()
            self._last_path = self.edf_path
        
        # If still loading or not loaded, just return
        if self._loading or not self.is_loaded or self.source_data is None:
            return
        
        # Get modulation
        speed_mod = self.get_blended_input('speed_mod', 'sum')
        if speed_mod is None:
            speed_mod = 1.0
        
        gain_mod = self.get_blended_input('gain_mod', 'sum')
        if gain_mod is None:
            gain_mod = 1.0
        
        # Advance playback
        samples_per_frame = int(self.time_window * self.fs * speed_mod)
        self.playback_idx += samples_per_frame
        
        n_samples = self.source_data.shape[1]
        if self.playback_idx >= n_samples - samples_per_frame:
            self.playback_idx = 0
        
        # Get current window of source data
        start_idx = int(self.playback_idx)
        end_idx = min(start_idx + samples_per_frame, n_samples)
        
        source_window = self.source_data[:, start_idx:end_idx]
        
        # Compute outputs
        self._compute_mode_activations(source_window, gain_mod)
        self._compute_band_powers(source_window)
        self._render_source_image(source_window)
        self._render_eigenmode_image()
    
    def _compute_mode_activations(self, source_window, gain):
        """Project source activity onto eigenmodes"""
        # Average over time window
        source_mean = source_window.mean(axis=1)
        
        # Split into hemispheres
        lh_data = source_mean[:self.n_lh_vertices]
        rh_data = source_mean[self.n_lh_vertices:]
        
        # Project onto eigenmodes
        mode_acts = np.zeros(self.n_output_modes)
        
        for i in range(min(self.n_output_modes, self.eigenmodes['lh'].shape[1])):
            # Combine both hemispheres
            lh_proj = np.dot(self.eigenmodes['lh'][:, i], lh_data)
            rh_proj = np.dot(self.eigenmodes['rh'][:, i], rh_data)
            mode_acts[i] = (lh_proj + rh_proj) * gain * self.amplification
        
        self._mode_activations = mode_acts
        
        # Compute complex modes using Hilbert on recent history
        self.mode_history.append(mode_acts.copy())
        
        if len(self.mode_history) > 20:
            history = np.array(list(self.mode_history))
            for i in range(self.n_output_modes):
                analytic = hilbert(history[:, i])
                self._complex_modes[i] = analytic[-1]
        else:
            self._complex_modes = mode_acts.astype(np.complex64)
    
    def _compute_band_powers(self, source_window):
        """Compute frequency band powers from source data"""
        # Use mean source activity
        mean_source = source_window.mean(axis=0)
        
        if len(mean_source) < 10:
            return
        
        for i, (band_name, (fmin, fmax)) in enumerate(self.bands.items()):
            try:
                # Bandpass filter
                nyq = self.fs / 2
                if fmax >= nyq:
                    fmax = nyq - 1
                
                b, a = butter(4, [fmin/nyq, fmax/nyq], btype='band')
                filtered = filtfilt(b, a, mean_source, padlen=min(len(mean_source)-1, 10))
                
                # Power
                self._band_powers[i] = np.mean(filtered ** 2) * self.amplification
            except:
                self._band_powers[i] = 0.0
    
    def _render_source_image(self, source_window):
        """Render 2D flatmap of source activity"""
        H, W = self.flatmap_size
        
        # Average source activity
        source_mean = source_window.mean(axis=1)
        
        # Create accumulator image
        acc_val = np.zeros((H, W), dtype=np.float32)
        acc_cnt = np.zeros((H, W), dtype=np.float32)
        
        # Splat LH
        lh_data = source_mean[:self.n_lh_vertices]
        px_lh = self.flatmap_coords['lh']['px']
        py_lh = self.flatmap_coords['lh']['py']
        
        np.add.at(acc_val, (py_lh, px_lh), lh_data)
        np.add.at(acc_cnt, (py_lh, px_lh), 1.0)
        
        # Splat RH
        rh_data = source_mean[self.n_lh_vertices:]
        px_rh = self.flatmap_coords['rh']['px']
        py_rh = self.flatmap_coords['rh']['py']
        
        np.add.at(acc_val, (py_rh, px_rh), rh_data)
        np.add.at(acc_cnt, (py_rh, px_rh), 1.0)
        
        # Average
        acc_cnt[acc_cnt == 0] = 1
        img_data = acc_val / acc_cnt
        
        # Smooth
        img_data = gaussian_filter(img_data, sigma=2)
        
        # Normalize
        vmax = np.percentile(np.abs(img_data), 99) + 1e-10
        img_norm = np.clip(img_data / vmax, -1, 1)
        
        # Map to colormap (RdBu)
        img_u8 = ((img_norm + 1) / 2 * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        # Mask where no data
        mask = (acc_cnt > 0.5).astype(np.float32)
        mask = gaussian_filter(mask, sigma=2)
        img_color = (img_color.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
        
        # Add labels
        cv2.putText(img_color, "Source Activity", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_color, "LH", (W//4, H-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img_color, "RH", (3*W//4, H-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        self._source_image = img_color
    
    def _render_eigenmode_image(self):
        """Render eigenmode activations over time"""
        H, W = 150, 300
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        if len(self.mode_history) < 5:
            cv2.putText(img, "Collecting...", (10, H//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            self._eigenmode_image = img
            return
        
        # Draw mode traces
        history = np.array(list(self.mode_history))
        n_samples, n_modes = history.shape
        
        # Normalize
        h_max = np.abs(history).max() + 1e-10
        history_norm = history / h_max
        
        # Colors for modes
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (150, 200, 100), (100, 150, 200),
            (200, 100, 150)
        ]
        
        trace_h = H - 30
        for i in range(min(n_modes, 10)):
            trace = history_norm[:, i]
            for j in range(1, len(trace)):
                x1 = int((j-1) / len(trace) * (W - 20)) + 10
                x2 = int(j / len(trace) * (W - 20)) + 10
                y1 = int(trace_h/2 + trace[j-1] * trace_h/2 * 0.8)
                y2 = int(trace_h/2 + trace[j] * trace_h/2 * 0.8)
                cv2.line(img, (x1, y1), (x2, y2), colors[i % len(colors)], 1)
        
        # Labels
        cv2.putText(img, "Source Eigenmodes", (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"t={self.playback_idx/self.fs:.1f}s", (W-80, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Mode bars at bottom
        bar_y = H - 20
        bar_w = (W - 20) // 10
        for i in range(10):
            x = 10 + i * bar_w
            val = abs(self._mode_activations[i]) / (np.abs(self._mode_activations).max() + 1e-10)
            bar_h = int(val * 15)
            cv2.rectangle(img, (x, bar_y - bar_h), (x + bar_w - 2, bar_y), colors[i], -1)
        
        self._eigenmode_image = img
    
    def get_output(self, port_name):
        # Images
        if port_name == 'source_image':
            return self._source_image
        elif port_name == 'eigenmode_image':
            return self._eigenmode_image
        elif port_name == 'lh_image':
            if self._source_image is not None:
                return self._source_image[:, :self.flatmap_size[1]//2]
            return None
        elif port_name == 'rh_image':
            if self._source_image is not None:
                return self._source_image[:, self.flatmap_size[1]//2:]
            return None
        
        # Spectra
        elif port_name == 'mode_spectrum':
            return self._mode_activations.astype(np.float32)
        elif port_name == 'complex_modes':
            return self._complex_modes
        elif port_name == 'band_spectrum':
            return self._band_powers.astype(np.float32)
        elif port_name == 'full_spectrum':
            return np.concatenate([self._band_powers, self._mode_activations]).astype(np.float32)
        
        # Individual mode signals
        elif port_name.startswith('mode_'):
            try:
                idx = int(port_name.split('_')[1]) - 1
                return float(self._mode_activations[idx])
            except:
                return 0.0
        
        # Band signals
        elif port_name == 'delta_power':
            return float(self._band_powers[0])
        elif port_name == 'theta_power':
            return float(self._band_powers[1])
        elif port_name == 'alpha_power':
            return float(self._band_powers[2])
        elif port_name == 'beta_power':
            return float(self._band_powers[3])
        elif port_name == 'gamma_power':
            return float(self._band_powers[4])
        
        # Other
        elif port_name == 'raw_source':
            if self.source_data is not None:
                idx = int(self.playback_idx)
                return float(self.source_data[:, idx].mean() * self.amplification)
            return 0.0
        elif port_name == 'dominant_mode':
            return float(np.argmax(np.abs(self._mode_activations)) + 1)
        
        return None
    
    def get_display_image(self):
        if self._source_image is not None:
            img = np.ascontiguousarray(self._source_image)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
        # Status display
        w, h = 300, 150
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self._loading:
            # Show loading animation
            cv2.putText(img, "Loading EEG...", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            cv2.putText(img, "This takes 30-60 seconds", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(img, "(BEM + Forward + Inverse)", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            # Animated dots
            dots = "." * ((self.frame_count // 10) % 4)
            cv2.putText(img, dots, (180, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        elif self.load_error:
            cv2.putText(img, "Error:", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            # Word wrap error message
            error_lines = [self.load_error[i:i+35] for i in range(0, len(self.load_error), 35)]
            for i, line in enumerate(error_lines[:3]):
                cv2.putText(img, line, (10, 55 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        elif not self.is_loaded:
            cv2.putText(img, "Set edf_path in config", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(img, "then restart playback", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            cv2.putText(img, "Ready - press Start", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("EDF Path", "edf_path", self.edf_path, None),
            ("Source Spacing", "source_spacing", self.source_spacing,
             [('oct5', 'oct5'), ('oct6', 'oct6'), ('ico4', 'ico4'), ('ico5', 'ico5')]),
            ("Inverse Method", "inverse_method", self.inverse_method,
             [('sLORETA', 'sLORETA'), ('dSPM', 'dSPM'), ('MNE', 'MNE'), ('eLORETA', 'eLORETA')]),
            ("Time Window (s)", "time_window", self.time_window, None),
            ("Amplification", "amplification", self.amplification, None),
        ]