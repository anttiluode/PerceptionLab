"""
Neural Transformer Node - The Biological Attention Engine
==========================================================
The synthesis: Theta timing + Wavelet tokens + QKV attention + Interference

THIS IS THE NODE YOU WANTED.

ARCHITECTURE:
1. Loads EEG internally (MNE source localization)
2. Extracts theta phase as the MASTER CLOCK (box corners = sample moments)
3. At each theta sample, extracts wavelet tokens from all regions/bands
4. Tokens become Query (frontal), Key (temporal), Value (sensory)
5. Computes biological attention: "What is frontal lobe attending to?"
6. Outputs attention matrix, context vector, interference pattern

OUTPUTS:
- display: Full dashboard (FL-studio tokens + attention matrix + interference)
- attention_matrix: Image showing Q->K alignment
- context_vector: The "thought" - weighted sum of Values
- token_stream: All active tokens (for external analysis)
- interference: Complex field for further processing
- theta_phase: Current phase (for synchronization)

WHAT YOU SEE:
- TOP: Token piano roll (FL Studio style) showing brain activity over time
- MIDDLE LEFT: Attention matrix (which tokens attend to which)
- MIDDLE RIGHT: Context sphere (the resulting "focus")
- BOTTOM: Interference pattern (holographic brain state)

CREATED: December 2025
AUTHOR: Claude + Antti
"""

import numpy as np
import cv2
import os
from collections import deque
from scipy.signal import hilbert, butter, lfilter, find_peaks
from scipy.ndimage import gaussian_filter

# === MNE IMPORT ===
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# === PERCEPTION LAB COMPATIBILITY ===
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode): 
            return 0.0

# === CONSTANTS ===
REGIONS = ['frontal', 'temporal', 'parietal', 'occipital']
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}
BAND_COLORS = {
    'delta': (139, 69, 19),    # Brown
    'theta': (255, 100, 100),  # Red
    'alpha': (100, 255, 100),  # Green  
    'beta': (255, 255, 100),   # Yellow
    'gamma': (100, 100, 255),  # Blue
}
REGION_COLORS = {
    'frontal': (255, 100, 100),
    'temporal': (100, 255, 100),
    'parietal': (255, 255, 100),
    'occipital': (100, 100, 255),
}

class NeuralTransformerNode(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Neural Transformer"
    NODE_COLOR = QtGui.QColor(255, 50, 150)  # Hot pink - the synthesis color
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS (minimal - mostly self-contained) ===
        self.inputs = {
            'temperature': 'float',     # Attention sharpness
            'gate_threshold': 'float',  # Theta corner sensitivity
        }
        
        # === OUTPUTS ===
        self.outputs = {
            'display': 'image',
            'attention_matrix': 'image',
            'context_vector': 'spectrum',
            'token_stream': 'spectrum',
            # Regional token outputs for interference experiments
            'frontal_tokens': 'spectrum',
            'temporal_tokens': 'spectrum',
            'parietal_tokens': 'spectrum',
            'occipital_tokens': 'spectrum',
            'interference': 'complex_spectrum',
            'theta_phase': 'signal',
            'sample_trigger': 'signal',  # 1.0 when sampling, 0.0 otherwise
        }
        
        # === CONFIG ===
        self.edf_path = r"E:\DocsHouse\450\2.edf"
        self.base_gain = 20.0
        self.base_speed = 1.0
        self.embed_dim = 64  # Token embedding dimension
        self.n_tokens_vocab = len(REGIONS) * len(BANDS)  # 20 possible tokens
        
        # === STATE ===
        self.fs = 160.0
        self.is_loaded = False
        self.load_error = ""
        self.needs_load = True
        self.playback_idx = 0.0
        
        # Source series (full recordings)
        self.source_series = {r: None for r in REGIONS}
        self.theta_phase_series = None
        self.theta_velocity_series = None
        
        # Token vocabulary - fixed random embeddings
        np.random.seed(42)
        self.embedding_matrix = np.random.randn(self.n_tokens_vocab, self.embed_dim)
        self.embedding_matrix /= np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        
        # Build token ID map
        self.token_vocab = {}
        idx = 0
        for region in REGIONS:
            for band in BANDS.keys():
                self.token_vocab[(region, band)] = idx
                idx += 1
        
        # === RUNTIME STATE ===
        self.current_tokens = []  # Active tokens this frame
        self.token_history = deque(maxlen=300)  # FL Studio roll
        self.is_sampling = False  # Are we at a box corner?
        
        # Attention state
        self.attention_weights = np.zeros((10, 10))  # Q x K
        self.context_vector = np.zeros(self.embed_dim)
        
        # Interference
        self.interference_field = np.zeros((256, 256), dtype=np.complex128)
        
        # Display
        self._display = np.zeros((900, 1400, 3), dtype=np.uint8)
        self._attention_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
    def get_config_options(self):
        return [
            ("EEG File", "edf_path", self.edf_path, "file_open"),
            ("Base Gain", "base_gain", self.base_gain, "float"),
            ("Playback Speed", "base_speed", self.base_speed, "float"),
            ("Reload", "needs_load", True, "button"),
        ]
    
    # ========== MNE PROCESSING ==========
    
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
    
    def _get_region_mask(self, coords, region):
        if region == "frontal":
            return coords[:, 1] > 0.05
        elif region == "occipital":
            return coords[:, 1] < -0.05
        elif region == "parietal":
            return (coords[:, 1] < 0.0) & (coords[:, 1] > -0.06) & (coords[:, 2] > 0.04)
        elif region == "temporal":
            return (coords[:, 1] < 0.0) & (coords[:, 2] < 0.0) & (np.abs(coords[:, 0]) > 0.03)
        return np.ones(len(coords), dtype=bool)
    
    def setup_source(self):
        """Full MNE pipeline - loads EEG and extracts all regions"""
        if not MNE_AVAILABLE:
            self.load_error = "MNE not installed"
            return
        
        if not os.path.exists(self.edf_path):
            self.load_error = f"File not found: {self.edf_path}"
            return
        
        try:
            print(f"[NeuralTransformer] Loading: {self.edf_path}")
            
            # Load raw
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            self.fs = raw.info['sfreq']
            
            # Clean and montage
            raw = self._clean_names(raw)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            raw.set_eeg_reference('average', projection=True, verbose=False)
            
            # Broadband filter
            nyq = self.fs / 2.0
            high_freq = min(70, nyq - 1)
            raw.filter(0.5, high_freq, verbose=False)
            
            # Sphere model
            sphere = mne.make_sphere_model(
                r0=(0., 0., 0.), head_radius=0.095,
                info=raw.info,
                relative_radii=(0.90, 0.92, 0.97, 1.0),
                sigmas=(0.33, 1.0, 0.004, 0.33),
                verbose=False
            )
            
            # Source space
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
            if not os.path.exists(os.path.join(subjects_dir, 'fsaverage')):
                print("[NeuralTransformer] Downloading fsaverage...")
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
            
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
            
            # Fix NaN in forward
            G = fwd['sol']['data']
            if not np.all(np.isfinite(G)):
                G[~np.isfinite(G)] = 0.0
                fwd['sol']['data'] = G
            
            # Inverse
            cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
            inv = mne.minimum_norm.make_inverse_operator(
                raw.info, fwd, cov, depth=None, loose='auto', verbose=False
            )
            
            # Apply inverse
            stc = mne.minimum_norm.apply_inverse_raw(
                raw, inv, lambda2=1.0/9.0, method='dSPM', verbose=False
            )
            
            # Extract regions
            coords = src[0]['rr'][stc.vertices[0]]
            
            for region in REGIONS:
                mask = self._get_region_mask(coords, region)
                if np.sum(mask) == 0:
                    print(f"[NeuralTransformer] Warning: {region} empty, using global")
                    mask[:] = True
                
                region_data = np.mean(stc.data[mask], axis=0)
                region_data = (region_data - np.mean(region_data)) / (np.std(region_data) + 1e-9)
                self.source_series[region] = region_data
            
            # Extract theta phase and velocity for gating
            print("[NeuralTransformer] Computing theta phase...")
            frontal = self.source_series['frontal']
            
            # Bandpass for theta
            nyq = self.fs / 2.0
            b, a = butter(3, [4/nyq, 8/nyq], btype='band')
            theta_filt = lfilter(b, a, frontal)
            
            # Hilbert transform
            analytic = hilbert(theta_filt)
            self.theta_phase_series = np.angle(analytic)
            
            # Compute phase velocity (for box corner detection)
            phase_unwrap = np.unwrap(self.theta_phase_series)
            self.theta_velocity_series = np.gradient(phase_unwrap)
            
            self.is_loaded = True
            self.load_error = ""
            print(f"[NeuralTransformer] Ready. {len(frontal)} samples at {self.fs}Hz")
            
        except Exception as e:
            self.load_error = str(e)
            print(f"[NeuralTransformer] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== TOKEN EXTRACTION ==========
    
    def _extract_tokens(self, idx, threshold_mult=1.5):
        """Extract wavelet tokens at current position"""
        tokens = []
        window_len = 256
        
        if idx + window_len >= len(self.source_series['frontal']):
            return tokens
        
        nyq = self.fs / 2.0
        
        for region in REGIONS:
            series = self.source_series[region]
            if series is None:
                continue
            
            window = series[idx:idx + window_len] * self.base_gain
            
            for band_name, (low, high) in BANDS.items():
                # Nyquist check
                if high >= nyq:
                    high = nyq - 0.1
                if low >= high:
                    continue
                
                # Bandpass
                b, a = butter(3, [low/nyq, high/nyq], btype='band')
                band_signal = lfilter(b, a, window)
                
                # Hilbert
                analytic = hilbert(band_signal)
                envelope = np.abs(analytic)
                phase = np.angle(analytic)
                
                # Center values
                mid = window_len // 2
                amp = envelope[mid]
                phi = phase[mid]
                
                # Threshold
                local_mean = np.mean(envelope)
                local_std = np.std(envelope)
                thresh = local_mean + threshold_mult * local_std
                
                # Create token if burst detected
                if amp > thresh and amp > 0.1:
                    token_id = self.token_vocab.get((region, band_name), 0)
                    tokens.append({
                        'id': token_id,
                        'region': region,
                        'band': band_name,
                        'amplitude': float(amp),
                        'phase': float(phi),
                        'frequency': (low + high) / 2,
                    })
        
        return tokens
    
    # ========== ATTENTION MECHANISM ==========
    
    def _compute_attention(self, tokens, temperature=1.0):
        """
        Biological QKV attention:
        - Q = frontal tokens (the "seeker")
        - K = temporal tokens (the "map")
        - V = sensory tokens (parietal + occipital, the "payload")
        """
        # Separate by role
        q_tokens = [t for t in tokens if t['region'] == 'frontal']
        k_tokens = [t for t in tokens if t['region'] == 'temporal']
        v_tokens = [t for t in tokens if t['region'] in ['parietal', 'occipital']]
        
        if not q_tokens or not k_tokens or not v_tokens:
            # Not enough diversity for attention
            self.attention_weights = np.eye(5) * 0.2
            self.context_vector = np.zeros(self.embed_dim)
            return
        
        # Embed tokens
        def embed_tokens(token_list):
            vectors = []
            weights = []
            for t in token_list:
                vec = self.embedding_matrix[t['id'] % self.n_tokens_vocab]
                vectors.append(vec * t['amplitude'])
                weights.append(t['amplitude'])
            return np.array(vectors), np.array(weights)
        
        Q, q_amp = embed_tokens(q_tokens)  # (n_q, 64)
        K, k_amp = embed_tokens(k_tokens)  # (n_k, 64)
        V, v_amp = embed_tokens(v_tokens)  # (n_v, 64)
        
        # Scaled dot-product attention: softmax(Q @ K.T / sqrt(d))
        scale = np.sqrt(self.embed_dim)
        scores = np.matmul(Q, K.T) / scale  # (n_q, n_k)
        
        # Temperature scaling
        scores = scores / max(temperature, 0.1)
        
        # Softmax (row-wise)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-9)
        
        # Store for visualization
        self.attention_weights = attn_weights
        
        # Context: aggregate V weighted by attention
        # Use max attention per query to weight values
        max_attn_per_q = np.max(attn_weights, axis=1)  # (n_q,)
        global_attn = np.mean(max_attn_per_q)
        
        self.context_vector = np.sum(V * v_amp[:, None], axis=0) * global_attn
    
    # ========== INTERFERENCE GENERATION ==========
    
    def _generate_interference(self, tokens, current_phase):
        """Generate holographic interference pattern from tokens"""
        size = 256
        field = np.zeros((size, size), dtype=np.complex128)
        
        if not tokens:
            self.interference_field = field
            return
        
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Each token creates a wave component
        for t in tokens:
            # Position based on region
            region_angles = {
                'frontal': 0,
                'temporal': np.pi/2,
                'parietal': np.pi,
                'occipital': 3*np.pi/2
            }
            region_angle = region_angles.get(t['region'], 0)
            
            # Frequency based on band
            band_freqs = {
                'delta': 1, 'theta': 2, 'alpha': 3, 'beta': 5, 'gamma': 8
            }
            k = band_freqs.get(t['band'], 2)
            
            # Wave vector direction
            kx = k * np.cos(region_angle + current_phase)
            ky = k * np.sin(region_angle + current_phase)
            
            # Add plane wave
            wave = t['amplitude'] * np.exp(1j * (kx * X + ky * Y + t['phase']))
            field += wave
        
        # Normalize
        field = field / (np.abs(field).max() + 1e-9)
        self.interference_field = field
    
    # ========== MAIN STEP ==========
    
    def step(self):
        if self.needs_load:
            self.setup_source()
            self.needs_load = False
        
        if not self.is_loaded:
            self._render_error()
            return
        
        # Get input parameters
        temp_val = self.get_blended_input("temperature", "sum")
        temperature = float(temp_val) if temp_val and temp_val > 0 else 1.0
        
        gate_val = self.get_blended_input("gate_threshold", "sum")
        gate_threshold = float(gate_val) if gate_val and gate_val > 0 else 0.5
        
        # Current position
        idx = int(self.playback_idx)
        total_len = len(self.source_series['frontal'])
        
        if idx >= total_len - 300:
            self.playback_idx = 0
            idx = 0
        
        # Get theta state
        current_phase = self.theta_phase_series[idx]
        current_velocity = abs(self.theta_velocity_series[idx])
        
        # BOX CORNER DETECTION
        # High velocity = at a corner = SAMPLE NOW
        velocity_threshold = np.percentile(np.abs(self.theta_velocity_series), 80) * gate_threshold
        self.is_sampling = current_velocity > velocity_threshold
        
        # Extract tokens (always, but mark if sampling)
        self.current_tokens = self._extract_tokens(idx)
        
        # Add to history
        self.token_history.append({
            'tokens': list(self.current_tokens),
            'is_sample': self.is_sampling,
            'phase': current_phase,
            'time': idx / self.fs
        })
        
        # Compute attention if we have tokens
        if self.current_tokens:
            self._compute_attention(self.current_tokens, temperature)
        
        # Generate interference
        self._generate_interference(self.current_tokens, current_phase)
        
        # Render
        self._render_dashboard(current_phase, current_velocity, velocity_threshold)
        
        # Advance
        self.playback_idx += self.base_speed
        
        # Update outputs
        self._update_outputs(current_phase)
    
    def _update_outputs(self, current_phase):
        # Token stream (all tokens)
        if self.current_tokens:
            arr = np.array([[t['id'], t['amplitude'], t['phase']] 
                           for t in self.current_tokens], dtype=np.float32)
            self.outputs['token_stream'] = arr
        else:
            self.outputs['token_stream'] = np.zeros((1, 3), dtype=np.float32)
        
        # Regional token outputs
        for region in REGIONS:
            region_tokens = [t for t in self.current_tokens if t['region'] == region]
            output_key = f'{region}_tokens'
            if region_tokens:
                arr = np.array([[t['id'], t['amplitude'], t['phase']] 
                               for t in region_tokens], dtype=np.float32)
                self.outputs[output_key] = arr
            else:
                self.outputs[output_key] = np.zeros((1, 3), dtype=np.float32)
        
        # Context vector
        self.outputs['context_vector'] = self.context_vector.astype(np.float32)
        
        # Interference (complex)
        self.outputs['interference'] = self.interference_field
        
        # Theta phase
        self.outputs['theta_phase'] = float(current_phase)
        
        # Sample trigger
        self.outputs['sample_trigger'] = 1.0 if self.is_sampling else 0.0
        
        # Attention matrix image
        attn_vis = cv2.resize(self.attention_weights.astype(np.float32), (256, 256), 
                             interpolation=cv2.INTER_NEAREST)
        attn_u8 = (attn_vis * 255).clip(0, 255).astype(np.uint8)
        self._attention_img = cv2.applyColorMap(attn_u8, cv2.COLORMAP_VIRIDIS)
        self.outputs['attention_matrix'] = self._attention_img
    
    # ========== RENDERING ==========
    
    def _render_dashboard(self, phase, velocity, vel_thresh):
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === TOP: FL STUDIO TOKEN ROLL ===
        roll_height = 350
        self._render_token_roll(img, 0, 0, w, roll_height)
        
        # === MIDDLE LEFT: ATTENTION MATRIX ===
        attn_x, attn_y = 20, roll_height + 20
        attn_size = 300
        self._render_attention_matrix(img, attn_x, attn_y, attn_size)
        
        # === MIDDLE CENTER: CONTEXT SPHERE ===
        sphere_x = attn_x + attn_size + 40
        sphere_y = roll_height + 20
        self._render_context_sphere(img, sphere_x, sphere_y, 300)
        
        # === MIDDLE RIGHT: THETA CLOCK ===
        clock_x = sphere_x + 340
        clock_y = roll_height + 20
        self._render_theta_clock(img, clock_x, clock_y, phase, velocity, vel_thresh)
        
        # === BOTTOM: INTERFERENCE FIELD ===
        interf_y = roll_height + 340
        self._render_interference(img, 20, interf_y, w - 40, h - interf_y - 20)
        
        self._display = img
    
    def _render_token_roll(self, img, x0, y0, width, height):
        """FL Studio style piano roll for tokens"""
        # Background
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 35), -1)
        
        # Grid
        n_rows = self.n_tokens_vocab
        row_height = height // n_rows
        
        # Draw horizontal lines and labels
        for i, ((region, band), token_id) in enumerate(self.token_vocab.items()):
            y = y0 + i * row_height
            
            # Alternating row backgrounds
            bg_col = (35, 35, 40) if i % 2 == 0 else (30, 30, 35)
            cv2.rectangle(img, (x0 + 80, y), (x0 + width, y + row_height), bg_col, -1)
            
            # Label
            label = f"{region[:3].upper()}-{band[:3]}"
            cv2.putText(img, label, (x0 + 5, y + row_height - 5),
                       cv2.FONT_HERSHEY_PLAIN, 0.7, REGION_COLORS[region], 1)
        
        # Draw tokens from history
        hist_len = min(len(self.token_history), width - 100)
        time_step = max(1, (width - 100) // max(hist_len, 1))
        
        for t_idx in range(hist_len):
            frame = self.token_history[-(hist_len - t_idx)]
            tokens = frame['tokens']
            is_sample = frame['is_sample']
            
            x = x0 + 90 + t_idx * time_step
            
            # Vertical sample marker
            if is_sample:
                cv2.line(img, (x, y0), (x, y0 + height), (100, 255, 255), 1)
            
            # Draw token bars
            for tok in tokens:
                token_id = tok['id']
                y = y0 + token_id * row_height
                
                # Color by region, brightness by amplitude
                color = REGION_COLORS[tok['region']]
                brightness = min(1.0, tok['amplitude'] / 3.0)
                color = tuple(int(c * brightness) for c in color)
                
                cv2.rectangle(img, (x, y + 2), (x + time_step - 1, y + row_height - 2),
                             color, -1)
        
        # Title and stats
        cv2.putText(img, "NEURAL TOKEN STREAM", (x0 + width//2 - 100, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(img, f"Active: {len(self.current_tokens)}", (x0 + width - 150, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
    
    def _render_attention_matrix(self, img, x0, y0, size):
        """Render QKV attention matrix"""
        # Background
        cv2.rectangle(img, (x0, y0), (x0+size, y0+size), (40, 40, 50), -1)
        
        # Resize attention weights to fit
        attn = self.attention_weights
        if attn.shape[0] > 0 and attn.shape[1] > 0:
            attn_resized = cv2.resize(attn.astype(np.float32), (size-20, size-40),
                                     interpolation=cv2.INTER_NEAREST)
            attn_u8 = (attn_resized * 255).clip(0, 255).astype(np.uint8)
            attn_color = cv2.applyColorMap(attn_u8, cv2.COLORMAP_INFERNO)
            
            img[y0+30:y0+size-10, x0+10:x0+size-10] = attn_color
        
        # Labels
        cv2.putText(img, "Q (Frontal)", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 100, 100), 1)
        cv2.putText(img, "K (Temporal)", (x0 + size - 80, y0 + size - 5),
                   cv2.FONT_HERSHEY_PLAIN, 0.9, (100, 255, 100), 1)
        cv2.putText(img, "ATTENTION", (x0 + size//2 - 40, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _render_context_sphere(self, img, x0, y0, size):
        """Render the context vector as a glowing sphere"""
        cv2.rectangle(img, (x0, y0), (x0+size, y0+size), (30, 30, 40), -1)
        
        center = (x0 + size//2, y0 + size//2)
        
        # Context magnitude
        ctx_mag = np.linalg.norm(self.context_vector)
        radius = int(20 + min(ctx_mag * 30, 100))
        
        # Draw glowing sphere
        for r in range(radius, 10, -5):
            alpha = (radius - r) / radius
            color = (int(50 + 100 * alpha), int(150 * alpha), int(255 * alpha))
            cv2.circle(img, center, r, color, -1)
        
        # Core
        cv2.circle(img, center, 10, (255, 255, 255), -1)
        
        # Labels
        cv2.putText(img, "CONTEXT", (x0 + size//2 - 35, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Focus: {ctx_mag:.2f}", (x0 + size//2 - 40, y0 + size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    
    def _render_theta_clock(self, img, x0, y0, phase, velocity, thresh):
        """Render theta phase clock with sampling indicator"""
        size = 150
        center = (x0 + size//2, y0 + size//2)
        radius = 60
        
        # Clock face
        cv2.circle(img, center, radius, (60, 60, 70), 2)
        
        # Phase hand
        hand_len = radius - 10
        hand_x = int(center[0] + hand_len * np.cos(phase - np.pi/2))
        hand_y = int(center[1] + hand_len * np.sin(phase - np.pi/2))
        
        # Color based on sampling state
        if self.is_sampling:
            hand_color = (100, 255, 255)  # Cyan = SAMPLING
            cv2.circle(img, center, radius + 5, (100, 255, 255), 2)
        else:
            hand_color = (200, 200, 200)  # Gray = holding
        
        cv2.line(img, center, (hand_x, hand_y), hand_color, 2)
        cv2.circle(img, (hand_x, hand_y), 5, hand_color, -1)
        
        # Velocity bar
        bar_x = x0 + size + 10
        bar_height = int(min(velocity / thresh, 2.0) * 50)
        cv2.rectangle(img, (bar_x, y0 + 100), (bar_x + 15, y0 + 100 - bar_height),
                     (100, 200, 100) if self.is_sampling else (100, 100, 100), -1)
        cv2.line(img, (bar_x - 5, y0 + 50), (bar_x + 20, y0 + 50), (255, 100, 100), 1)
        
        # Labels
        cv2.putText(img, "THETA", (x0 + size//2 - 25, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        state_txt = "SAMPLE!" if self.is_sampling else "hold"
        cv2.putText(img, state_txt, (x0 + size//2 - 25, y0 + size - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, hand_color, 1)
    
    def _render_interference(self, img, x0, y0, width, height):
        """Render interference pattern at bottom"""
        if self.interference_field is None:
            return
        
        # Get magnitude and phase
        magnitude = np.abs(self.interference_field)
        phase = np.angle(self.interference_field)
        
        # Create HSV image (hue=phase, value=magnitude)
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[:,:,0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:,:,1] = 255
        hsv[:,:,2] = (magnitude * 255).clip(0, 255).astype(np.uint8)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Resize to fit
        resized = cv2.resize(rgb, (width, height))
        img[y0:y0+height, x0:x0+width] = resized
        
        # Label
        cv2.putText(img, "INTERFERENCE FIELD", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _render_error(self):
        img = self._display
        img[:] = (20, 20, 25)
        
        if not self.load_error:
            cv2.putText(img, "LOADING NEURAL TRANSFORMER...", (400, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
            cv2.putText(img, "Processing MNE source localization...", (350, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        else:
            cv2.putText(img, f"ERROR: {self.load_error}", (50, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        self._display = img
    
    # ========== OUTPUT METHODS ==========
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'attention_matrix':
            return self._attention_img
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display