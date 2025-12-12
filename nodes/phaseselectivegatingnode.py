"""
Phase-Selective Gating Analyzer (Self-Contained)
=================================================
Complete pipeline with internal EEG processing.

FEATURES:
1. Loads EEG file internally
2. Source-localizes to brain regions using MNE
3. Extracts synchronized theta/beta/gamma from multiple regions
4. Computes phase-selective gating
5. Generates network topology + interference patterns

INPUTS:
- speed: Playback speed
- gain: Signal amplification
- smoothing: Visual decay rate

OUTPUTS:
- display: Combined visualization (network + histogram + interference)
- interference_field: Holographic pattern
- control_matrix: Complex spectrum for interference experiments
- gating_strength: Array of coupling values
"""

import numpy as np
import cv2
import os
from scipy.signal import butter, lfilter, hilbert, find_peaks
from scipy.ndimage import gaussian_filter
from collections import deque

# MNE import safety
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0

class PhaseSelectiveGatingNode(BaseNode):
    NODE_CATEGORY = "Perception Lab"
    NODE_TITLE = "Phase-Selective Gating (Complete)"
    NODE_COLOR = QtGui.QColor(138, 43, 226)
    
    def __init__(self):
        super().__init__()
        
        # NO INPUTS - fully self-contained
        self.inputs = {}
        
        self.outputs = {
            'display': 'image',
            'interference_field': 'image',
            'control_matrix': 'complex_spectrum',
            'gating_strength': 'spectrum',
            'topology_state': 'signal'
        }
        
        # === CONFIGURATION (User editable) ===
        self.edf_path = r"E:\DocsHouse\450\2.edf"
        self.base_gain = 20.0
        self.base_speed = 1.0
        
        # === INTERNAL STATE ===
        self.fs = 160.0
        self.is_loaded = False
        self.load_error = ""
        self.needs_load = True
        
        # Source-localized time series (full length)
        self.frontal_theta_series = None
        self.frontal_beta_series = None
        self.temporal_gamma_series = None
        self.parietal_beta_series = None
        self.occipital_gamma_series = None
        
        # Phase series for theta
        self.theta_phase_series = None
        
        # Playback
        self.playback_idx = 0.0
        
        # === PHASE-LOCKING STATE ===
        self.phase_bins = 36
        self.buffer_len = 512
        
        # Buffers for recent data
        self.theta_buffer = deque(maxlen=self.buffer_len)
        self.signal_buffers = {
            'frontal_beta': deque(maxlen=self.buffer_len),
            'temporal_gamma': deque(maxlen=self.buffer_len),
            'parietal_beta': deque(maxlen=self.buffer_len),
            'occipital_gamma': deque(maxlen=self.buffer_len),
        }
        
        # Accumulated phase histograms
        self.phase_hists = {
            'frontal_beta': np.zeros(self.phase_bins),
            'temporal_gamma': np.zeros(self.phase_bins),
            'parietal_beta': np.zeros(self.phase_bins),
            'occipital_gamma': np.zeros(self.phase_bins),
        }
        
        # Current gating strengths
        self.gating_strengths = {
            'frontal_beta': 0.0,
            'temporal_gamma': 0.0,
            'parietal_beta': 0.0,
            'occipital_gamma': 0.0,
        }
        
        # Visualization
        self.network_image = None
        self.hist_image = None
        self.interference_image = None
        self.interference_matrix = None
        self.control_fft = None
        
        self._output_display = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def get_config_options(self):
        """Right-click configuration menu"""
        return [
            ("EEG File Path", "edf_path", self.edf_path, "file_open"),
            ("Playback Speed", "base_speed", self.base_speed, "float"),
            ("Signal Gain", "base_gain", self.base_gain, "float"),
            ("Reload File", "needs_load", True, "button")
        ]
    
    # === MNE PROCESSING (Like ThetaBetaGatingNode) ===
    
    def _clean_names(self, raw):
        rename = {}
        for ch in raw.ch_names:
            clean = ch.replace('.', '').strip().upper()
            if clean == "FZ": clean = "Fz"
            if clean == "CZ": clean = "Cz"
            if clean == "PZ": clean = "Pz"
            if clean == "OZ": clean = "Oz"
            if clean == "FP1": clean = "Fp1"
            if clean == "FP2": clean = "Fp2"
            rename[ch] = clean
        raw.rename_channels(rename)
        return raw
    
    def _get_region_mask(self, coords, region_name):
        if region_name == "frontal":
            return coords[:, 1] > 0.05
        elif region_name == "occipital":
            return coords[:, 1] < -0.05
        elif region_name == "parietal":
            return (coords[:, 1] < 0.0) & (coords[:, 1] > -0.06) & (coords[:, 2] > 0.04)
        elif region_name == "temporal":
            return (coords[:, 1] < 0.0) & (coords[:, 2] < 0.0) & (np.abs(coords[:, 0]) > 0.03)
        else:
            return np.ones(len(coords), dtype=bool)
    
    def setup_source(self):
        """Complete MNE pipeline - loads and processes EEG"""
        if not MNE_AVAILABLE:
            self.load_error = "MNE not installed"
            return
        
        if not os.path.exists(self.edf_path):
            self.load_error = "File not found"
            return
        
        try:
            print(f"[Gating] Loading: {self.edf_path}")
            
            # Load raw
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            self.fs = raw.info['sfreq']
            
            # Clean and set montage
            raw = self._clean_names(raw)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            raw.set_eeg_reference('average', projection=True, verbose=False)
            
            # Sphere model
            sphere = mne.make_sphere_model(
                r0=(0., 0., 0.), head_radius=0.095,
                info=raw.info, 
                relative_radii=(0.90, 0.92, 0.97, 1.0),
                sigmas=(0.33, 1.0, 0.004, 0.33),
                verbose=False
            )
            
            # Volume source space
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
            src = mne.setup_volume_source_space(
                subject='fsaverage', pos=30.0,
                sphere=sphere, bem=None,
                subjects_dir=subjects_dir, verbose=False
            )
            
            # Forward solution
            fwd = mne.make_forward_solution(
                raw.info, trans=None, src=src, bem=sphere,
                eeg=True, meg=False, verbose=False
            )
            
            # Sanitize forward matrix
            G = fwd['sol']['data']
            if not np.all(np.isfinite(G)):
                np.nan_to_num(G, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                fwd['sol']['data'] = G
            
            # Covariance and inverse
            cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
            inv = mne.minimum_norm.make_inverse_operator(
                raw.info, fwd, cov, depth=None, loose='auto', verbose=False
            )
            
            print("[Gating] Extracting sources...")
            
            # === EXTRACT ALL SIGNALS ===
            
            # 1. Frontal Theta (clock)
            raw_theta = raw.copy().filter(4, 8, verbose=False)
            stc_theta = mne.minimum_norm.apply_inverse_raw(
                raw_theta, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            coords = src[0]['rr'][stc_theta.vertices[0]]
            mask_frontal = self._get_region_mask(coords, "frontal")
            frontal_theta = np.mean(stc_theta.data[mask_frontal], axis=0)
            self.frontal_theta_series = (frontal_theta - np.mean(frontal_theta)) / (np.std(frontal_theta) + 1e-9)
            
            # Compute phase once
            self.theta_phase_series = np.angle(hilbert(self.frontal_theta_series))
            
            # 2. Frontal Beta
            raw_fb = raw.copy().filter(12, 30, verbose=False)
            stc_fb = mne.minimum_norm.apply_inverse_raw(
                raw_fb, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            fb_data = np.mean(stc_fb.data[mask_frontal], axis=0)
            self.frontal_beta_series = (fb_data - np.mean(fb_data)) / (np.std(fb_data) + 1e-9)
            
            # 3. Temporal Gamma (fix: use 30-70 to avoid Nyquist)
            mask_temporal = self._get_region_mask(coords, "temporal")
            nyquist = self.fs / 2.0
            gamma_high = min(70, nyquist - 1)
            raw_tg = raw.copy().filter(30, gamma_high, verbose=False)
            stc_tg = mne.minimum_norm.apply_inverse_raw(
                raw_tg, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            tg_data = np.mean(stc_tg.data[mask_temporal], axis=0)
            self.temporal_gamma_series = (tg_data - np.mean(tg_data)) / (np.std(tg_data) + 1e-9)
            
            # 4. Parietal Beta
            mask_parietal = self._get_region_mask(coords, "parietal")
            stc_pb = mne.minimum_norm.apply_inverse_raw(
                raw_fb, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            pb_data = np.mean(stc_pb.data[mask_parietal], axis=0)
            self.parietal_beta_series = (pb_data - np.mean(pb_data)) / (np.std(pb_data) + 1e-9)
            
            # 5. Occipital Gamma (use same safe range)
            mask_occipital = self._get_region_mask(coords, "occipital")
            stc_og = mne.minimum_norm.apply_inverse_raw(
                raw_tg, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            og_data = np.mean(stc_og.data[mask_occipital], axis=0)
            self.occipital_gamma_series = (og_data - np.mean(og_data)) / (np.std(og_data) + 1e-9)
            
            self.is_loaded = True
            self.load_error = ""
            print(f"[Gating] Ready. {len(self.frontal_theta_series)} samples at {self.fs}Hz")
            
        except Exception as e:
            self.load_error = str(e)
            print(f"[Gating] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def step(self):
        # Auto-load on first step
        if self.needs_load:
            self.setup_source()
            self.needs_load = False
        
        if not self.is_loaded:
            self._render_error()
            return
        
        # Use internal speed/gain (no inputs)
        speed = self.base_speed
        gain = self.base_gain
        
        # Get current index
        idx = int(self.playback_idx)
        if idx >= len(self.frontal_theta_series) - 1:
            self.playback_idx = 0
            idx = 0
        
        # Get current values
        theta_val = self.frontal_theta_series[idx] * gain
        theta_phase = self.theta_phase_series[idx]
        
        fb_val = self.frontal_beta_series[idx] * gain
        tg_val = self.temporal_gamma_series[idx] * gain
        pb_val = self.parietal_beta_series[idx] * gain
        og_val = self.occipital_gamma_series[idx] * gain
        
        # Update buffers
        self.theta_buffer.append(theta_val)
        self.signal_buffers['frontal_beta'].append(fb_val)
        self.signal_buffers['temporal_gamma'].append(tg_val)
        self.signal_buffers['parietal_beta'].append(pb_val)
        self.signal_buffers['occipital_gamma'].append(og_val)
        
        # Need enough data
        if len(self.theta_buffer) < 128:
            self.playback_idx += speed
            return
        
        # === PHASE-LOCKING ANALYSIS ===
        
        # Get recent phase data
        phase_window = 256
        start_idx = max(0, idx - phase_window)
        end_idx = idx
        
        theta_phase_window = self.theta_phase_series[start_idx:end_idx]
        
        # Analyze each signal
        for name, series in [
            ('frontal_beta', self.frontal_beta_series),
            ('temporal_gamma', self.temporal_gamma_series),
            ('parietal_beta', self.parietal_beta_series),
            ('occipital_gamma', self.occipital_gamma_series)
        ]:
            signal_window = series[start_idx:end_idx]
            
            # Envelope
            envelope = np.abs(hilbert(signal_window))
            
            # Detect bursts
            threshold = np.mean(envelope) + 1.5 * np.std(envelope)
            peaks, _ = find_peaks(envelope, height=threshold, distance=int(self.fs/40))
            
            if len(peaks) > 0:
                # Get phases at burst times
                burst_phases = theta_phase_window[peaks]
                
                # Accumulate into histogram
                for bp in burst_phases:
                    normalized_phase = (bp + np.pi) / (2 * np.pi)
                    bin_idx = int(normalized_phase * self.phase_bins) % self.phase_bins
                    self.phase_hists[name][bin_idx] += 1.0
                
                # Decay
                self.phase_hists[name] *= 0.99
                
                # Compute MVL
                x = np.sum(np.cos(burst_phases))
                y = np.sum(np.sin(burst_phases))
                mvl = np.sqrt(x**2 + y**2) / len(burst_phases)
                self.gating_strengths[name] = mvl
            else:
                self.gating_strengths[name] *= 0.95
        
        # === RENDER ===
        self._compute_interference(theta_phase)
        self._render_network_graph(theta_phase)
        self._render_phase_histograms()
        self._render_interference_field()
        self._composite_display()
        
        self.playback_idx += speed
    
    def _compute_interference(self, current_phase):
        size = 128
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        theta_wave = np.sin(3 * X + current_phase)
        composite = theta_wave.copy()
        
        region_positions = {
            'frontal_beta': (0, np.pi/2),
            'temporal_gamma': (-np.pi/2, 0),
            'parietal_beta': (np.pi/2, 0),
            'occipital_gamma': (0, -np.pi/2),
        }
        
        for name, (px, py) in region_positions.items():
            strength = self.gating_strengths[name]
            if strength > 0.01:
                distance = np.sqrt((X - px)**2 + (Y - py)**2)
                wave = strength * np.sin(5 * distance)
                composite += wave
        
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-9)
        self.interference_matrix = composite
        self.control_fft = np.fft.fftshift(np.fft.fft2(composite))
    
    def _render_network_graph(self, current_phase):
        w, h = 400, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        center = (w//2, h//2)
        radius = 120
        
        nodes = {
            'frontal_beta': (center[0], center[1] - radius),
            'temporal_gamma': (center[0] - radius, center[1]),
            'parietal_beta': (center[0] + radius, center[1]),
            'occipital_gamma': (center[0], center[1] + radius),
        }
        
        cv2.circle(img, center, 15, (255, 255, 255), -1)
        cv2.putText(img, "θ", (center[0]-8, center[1]+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        hand_len = 40
        hand_x = int(center[0] + hand_len * np.cos(current_phase - np.pi/2))
        hand_y = int(center[1] + hand_len * np.sin(current_phase - np.pi/2))
        cv2.line(img, center, (hand_x, hand_y), (100, 255, 255), 2)
        
        colors = {
            'frontal_beta': (255, 100, 100),
            'temporal_gamma': (100, 255, 100),
            'parietal_beta': (255, 255, 100),
            'occipital_gamma': (100, 100, 255),
        }
        
        for name, pos in nodes.items():
            strength = self.gating_strengths[name]
            thickness = int(1 + strength * 10)
            color = colors[name]
            
            cv2.line(img, center, pos, color, thickness)
            node_size = int(8 + strength * 15)
            cv2.circle(img, pos, node_size, color, -1)
            cv2.circle(img, pos, node_size+2, (255, 255, 255), 1)
            
            label = name.split('_')[0][:4].upper()
            offset = 25
            label_pos = (pos[0] - 15, pos[1] - offset if pos[1] < center[1] else pos[1] + offset + 10)
            cv2.putText(img, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(img, f"{strength:.2f}", (pos[0] - 15, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        cv2.putText(img, "GATING TOPOLOGY", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        overall = np.mean(list(self.gating_strengths.values()))
        cv2.putText(img, f"Coherence: {overall:.3f}", (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show playback position
        if hasattr(self, 'frontal_theta_series') and self.frontal_theta_series is not None:
            total_len = len(self.frontal_theta_series)
            progress = self.playback_idx / total_len if total_len > 0 else 0
            cv2.putText(img, f"Time: {self.playback_idx/self.fs:.1f}s / {total_len/self.fs:.1f}s", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        self.network_image = img
    
    def _render_phase_histograms(self):
        w, h = 400, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        colors = {
            'frontal_beta': (255, 100, 100),
            'temporal_gamma': (100, 255, 100),
            'parietal_beta': (255, 255, 100),
            'occipital_gamma': (100, 100, 255),
        }
        
        center = (w//2, h//2)
        max_radius = 150
        
        cv2.circle(img, center, max_radius, (50, 50, 50), 1)
        cv2.putText(img, "0°", (center[0] + max_radius + 10, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        for i, (name, hist) in enumerate(self.phase_hists.items()):
            if np.max(hist) < 0.1:
                continue
            
            hist_norm = hist / (np.max(hist) + 1e-9)
            color = colors[name]
            
            for bin_idx, value in enumerate(hist_norm):
                if value < 0.05:
                    continue
                
                angle = (bin_idx / self.phase_bins) * 2 * np.pi - np.pi
                inner_r = max_radius * (0.3 + i * 0.15)
                outer_r = inner_r + value * 30
                
                x1 = int(center[0] + inner_r * np.cos(angle))
                y1 = int(center[1] + inner_r * np.sin(angle))
                x2 = int(center[0] + outer_r * np.cos(angle))
                y2 = int(center[1] + outer_r * np.sin(angle))
                
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
        
        legend_y = 20
        for name, color in colors.items():
            label = name.split('_')[0][:4].upper()
            cv2.rectangle(img, (10, legend_y), (25, legend_y+10), color, -1)
            cv2.putText(img, label, (30, legend_y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            legend_y += 15
        
        cv2.putText(img, "PHASE HISTOGRAM", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        self.hist_image = img
    
    def _render_interference_field(self):
        if self.interference_matrix is None:
            self.interference_image = np.zeros((256, 256, 3), dtype=np.uint8)
            return
        
        pattern = self.interference_matrix
        pattern_u8 = (pattern * 255).astype(np.uint8)
        pattern_color = cv2.applyColorMap(pattern_u8, cv2.COLORMAP_TWILIGHT)
        
        h, w = pattern_color.shape[:2]
        cv2.putText(pattern_color, "F", (w//2-5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(pattern_color, "T", (10, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(pattern_color, "P", (w-20, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(pattern_color, "O", (w//2-5, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        self.interference_image = pattern_color
    
    def _composite_display(self):
        """Create final display output"""
        if self.network_image is None:
            return
        
        # Left panel: network + histogram stacked
        left = np.vstack([
            self.network_image,
            self.hist_image if self.hist_image is not None else np.zeros((400, 400, 3), dtype=np.uint8)
        ])
        
        # Right panel: large interference
        if self.interference_image is not None:
            right = cv2.resize(self.interference_image, (400, 800))
        else:
            right = np.zeros((800, 400, 3), dtype=np.uint8)
        
        self._output_display = np.hstack([left, right])
    
    def _render_error(self):
        img = np.zeros((800, 1200, 3), dtype=np.uint8)
        
        if not self.load_error:
            # Show loading progress
            txt = f"LOADING EEG..."
            cv2.putText(img, txt, (400, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
            cv2.putText(img, "Processing source localization...", (350, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        else:
            txt = f"ERROR: {self.load_error}"
            cv2.putText(img, txt, (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            cv2.putText(img, "Check console for details", (350, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        self._output_display = img
    
    def get_output(self, port_name):
        if port_name == 'display':
            return self._output_display
        
        elif port_name == 'interference_field':
            return self.interference_image if self.interference_image is not None else np.zeros((256, 256, 3), dtype=np.uint8)
        
        elif port_name == 'control_matrix':
            if self.control_fft is not None:
                return self.control_fft
            return np.zeros((128, 128), dtype=np.complex128)
        
        elif port_name == 'gating_strength':
            return np.array([
                self.gating_strengths['frontal_beta'],
                self.gating_strengths['temporal_gamma'],
                self.gating_strengths['parietal_beta'],
                self.gating_strengths['occipital_gamma'],
            ])
        
        elif port_name == 'topology_state':
            return float(np.mean(list(self.gating_strengths.values())))
        
        return None
    
    def get_display_image(self):
        return self._output_display