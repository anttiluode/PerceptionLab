"""
Φ-Dwell Macroscope Node
========================
Real-time eigenmode phase portrait of brain dynamics.

Takes all 5 frequency band fields from PhiHologramNode and produces:
1. Live 2D phase portrait — brain's trajectory through eigenmode space with fading trails
2. Attractor detection — colored regions where the brain dwells
3. Regime meter — is the brain currently critical, bursty, or clocklike?
4. Band dominance ring — which frequency band controls the current state
5. Dwell timer — real-time measurement of how long current state has persisted
6. Eigenmode bar chart — live coefficient magnitudes across all 8 modes

The core idea: at each frame, decompose the EEG holographic field into
graph Laplacian eigenmodes. This produces a point in N-dimensional eigenmode 
space. Over time, the brain traces an orbit. Different cognitive states trace 
different orbits — different shapes, different attractor basins.

This is the "macroscope" — you can't see what the brain is thinking,
but you can see HOW it's thinking: which spatial scales are active,
how stable the current state is, and when it transitions.

Place this file in the 'nodes' folder of PerceptionLab.

Theory: Bistrom & Claude (2025). Φ-Dwell: Eigenmode Phase-Field Metastability Analyzer.
        Wang et al. (2017). Brain network eigenmodes.
        Baker & Cariani (2025). A time-domain account of brain function.
"""

import numpy as np
import cv2
from collections import deque
import time

# --- HOST IMPORT ---
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
            self.input_data = {}
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}
        def get_blended_input(self, name, mode):
            return None

# ═══════════════════════════════════════════════════════════════
# ELECTRODE POSITIONS (10-20 system for PhiHologramNode compatibility)
# ═══════════════════════════════════════════════════════════════

ELECTRODE_POS_20 = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6),
    'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'T7': (-0.9, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0, 0.0),
    'C4': (0.4, 0.0), 'T8': (0.9, 0.0),
    'P7': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5),
    'P4': (0.35, -0.5), 'P8': (0.7, -0.5),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85)
}

BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_COLORS = {
    'delta': (255, 80, 80),    # Red
    'theta': (255, 180, 40),   # Orange  
    'alpha': (80, 255, 80),    # Green
    'beta':  (80, 160, 255),   # Blue
    'gamma': (200, 80, 255),   # Purple
}

# Regime colors
REGIME_COLORS = {
    'critical':  (80, 255, 180),   # Teal-green
    'bursty':    (255, 180, 40),   # Orange
    'clocklike': (80, 160, 255),   # Blue
    'random':    (120, 120, 120),  # Grey
}

MODE_NAMES = ['A-P', 'L-R', 'C-P', 'Diag', 'M5', 'M6', 'M7', 'M8']


def build_graph_laplacian(positions, sigma=0.4):
    """Build graph Laplacian and return eigenmodes."""
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])
    
    # Gaussian adjacency
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
            A[i,j] = np.exp(-d**2 / (2*sigma**2))
            A[j,i] = A[i,j]
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    eigvals, eigvecs = np.linalg.eigh(L)
    return names, coords, eigvals, eigvecs


class PhiDwellMacroscopeNode(BaseNode):
    """
    Φ-Dwell Macroscope: Real-time eigenmode phase portrait of brain dynamics.
    
    Connects to PhiHologramNode's 5 band field outputs.
    Produces a live visualization of the brain's orbit through eigenmode space.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(180, 40, 220)  # Consciousness purple
    
    def __init__(self):
        super().__init__()
        self.node_title = "Φ-Dwell Macroscope"
        
        # Inputs: 5 complex fields from PhiHologramNode
        self.inputs = {
            'delta_field': 'complex_spectrum',
            'theta_field': 'complex_spectrum',
            'alpha_field': 'complex_spectrum',
            'beta_field': 'complex_spectrum',
            'gamma_field': 'complex_spectrum',
        }
        
        # Outputs: signals for downstream analysis
        self.outputs = {
            'portrait':        'image',      # The phase portrait visualization
            'regime_map':      'image',      # Heatmap of attractor basins
            'dominant_mode':   'signal',     # Which eigenmode is strongest (1-8)
            'dominant_band':   'signal',     # Which band is strongest (0-4)
            'dwell_time':      'signal',     # Current dwell duration in ms
            'regime_index':    'signal',     # 0=random, 1=critical, 2=bursty, 3=clocklike
            'cv_running':      'signal',     # Running CV of dwell times
            'metastability':   'signal',     # Composite metastability score 0-1
        }
        
        # --- Eigenmode setup ---
        self.n_modes = 8
        self.e_names, self.e_coords, self.eigvals, self.eigvecs = \
            build_graph_laplacian(ELECTRODE_POS_20)
        # Skip mode 0 (constant), use modes 1..8
        self.V = self.eigvecs[:, 1:self.n_modes+1].copy()
        # Normalize each eigenvector
        for m in range(self.n_modes):
            norm = np.linalg.norm(self.V[:, m])
            if norm > 1e-10:
                self.V[:, m] /= norm
        
        # --- Trajectory history ---
        self.max_trail = 600       # ~20 seconds at 30fps
        self.trail_x = deque(maxlen=self.max_trail)
        self.trail_y = deque(maxlen=self.max_trail)
        self.trail_band = deque(maxlen=self.max_trail)  # dominant band at each point
        self.trail_mode = deque(maxlen=self.max_trail)  # dominant mode at each point
        
        # --- Eigenmode coefficients history (for regime detection) ---
        self.coeff_history = deque(maxlen=300)  # ~10s of coefficient vectors
        self.phase_history = deque(maxlen=300)   # phase angles per mode
        
        # --- Dwell tracking ---
        self.current_dwell = 0      # frames in current dwell
        self.last_phase = None      # last eigenmode phase vector
        self.dwell_times = deque(maxlen=200)  # recent completed dwells
        self.dwell_threshold = np.pi / 4  # 45 degrees = transition
        
        # --- Attractor map (persistent heatmap) ---
        self.portrait_size = 400
        self.attractor_map = np.zeros((self.portrait_size, self.portrait_size), dtype=np.float32)
        self.attractor_decay = 0.998  # slow decay
        
        # --- Running statistics ---
        self.regime = 'random'
        self.regime_confidence = 0.0
        self.running_cv = 0.0
        self.metastability_score = 0.0
        self.dominant_mode_idx = 0
        self.dominant_band_idx = 0
        self.current_dwell_ms = 0
        
        # --- Display cache ---
        self.display_img = np.zeros((self.portrait_size, self.portrait_size, 3), dtype=np.uint8)
        self.regime_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # --- Frame counter for timing ---
        self.frame_count = 0
        self.fps_estimate = 30.0
        self.last_time = time.time()
        
        # --- PCA axes for 2D projection (will be computed from data) ---
        self.pca_axes = None  # (2, n_modes*n_bands) projection matrix
        self.pca_center = None
        self.pca_scale = 1.0
        
    def _extract_eigenmode_coefficients(self, field, band_idx):
        """
        Extract eigenmode coefficients from a holographic complex field.
        
        The field is (res, res) complex from PhiHologramNode.
        We sample it at electrode positions and project onto eigenmodes.
        """
        if field is None or not np.iscomplexobj(field):
            return np.zeros(self.n_modes), np.zeros(self.n_modes)
        
        res = field.shape[0]
        n_elec = len(self.e_names)
        
        # Sample field at electrode positions
        z = np.zeros(n_elec, dtype=np.complex64)
        for i, name in enumerate(self.e_names):
            ex, ey = self.e_coords[i]
            # Map electrode position (-1,1) to pixel coords
            px = int(np.clip((ex + 1.5) / 3.0 * res, 0, res-1))
            py = int(np.clip((1.5 - ey) / 3.0 * res, 0, res-1))
            z[i] = field[py, px]
        
        # Extract phase per electrode
        phases = np.angle(z)
        
        # Form unit complex field
        z_real = np.cos(phases)
        z_imag = np.sin(phases)
        
        # Project onto each eigenmode
        coeffs = np.zeros(self.n_modes)
        mode_phases = np.zeros(self.n_modes)
        
        for m in range(self.n_modes):
            pr = np.dot(z_real, self.V[:, m])
            pi = np.dot(z_imag, self.V[:, m])
            coeffs[m] = np.sqrt(pr**2 + pi**2)
            mode_phases[m] = np.arctan2(pi, pr)
        
        return coeffs, mode_phases
    
    def _detect_regime(self):
        """
        Classify current dynamical regime from recent dwell statistics.
        
        Returns: regime name, confidence
        
        Based on Φ-Dwell findings:
        - Critical: CV > 1.2, moderate kurtosis (30-70). Power-law-like.
        - Bursty: CV > 1.0, extreme kurtosis (>100). Beta-burst-like.
        - Clocklike: CV < 0.8. Regular, gamma-like.
        - Random: CV ~ 1.0, low kurtosis (<10). Exponential.
        """
        if len(self.dwell_times) < 10:
            return 'random', 0.0
        
        dwells = np.array(self.dwell_times)
        mean_d = np.mean(dwells)
        if mean_d < 1e-6:
            return 'random', 0.0
        
        cv = np.std(dwells) / mean_d
        self.running_cv = cv
        
        # Excess kurtosis
        if np.std(dwells) > 0:
            kurt = float(np.mean(((dwells - mean_d) / np.std(dwells))**4) - 3)
        else:
            kurt = 0.0
        
        # Classification
        if cv < 0.7:
            regime = 'clocklike'
            confidence = min(1.0, (0.7 - cv) / 0.3)
        elif cv > 1.3 and kurt > 80:
            regime = 'bursty'
            confidence = min(1.0, (kurt - 80) / 200)
        elif cv > 1.1 and 15 < kurt < 100:
            regime = 'critical'
            confidence = min(1.0, (cv - 1.1) / 0.5)
        else:
            regime = 'random'
            confidence = 0.5
        
        return regime, confidence
    
    def _compute_metastability(self):
        """
        Composite metastability score 0-1.
        High = brain is at criticality (power-law dwells, moderate CV, heavy tails).
        Low = noise-like or rigid.
        """
        if len(self.dwell_times) < 5:
            return 0.0
        
        dwells = np.array(self.dwell_times)
        mean_d = np.mean(dwells)
        if mean_d < 1e-6:
            return 0.0
        
        cv = np.std(dwells) / mean_d
        
        # Metastability peaks at CV ≈ 1.3 (critical regime)
        cv_score = 1.0 - abs(cv - 1.3) / 1.3
        cv_score = max(0.0, min(1.0, cv_score))
        
        # Long dwells relative to mean (heavy tail presence)
        max_ratio = np.max(dwells) / mean_d if mean_d > 0 else 0
        tail_score = min(1.0, max_ratio / 10.0)
        
        # Variance of log-dwells (log-normal = high, exponential = low)
        log_d = np.log(dwells + 1)
        log_var = np.var(log_d) / (np.mean(log_d) + 1e-6)
        log_score = min(1.0, log_var / 2.0)
        
        return 0.4 * cv_score + 0.3 * tail_score + 0.3 * log_score
    
    def _update_pca(self, full_vector):
        """Update PCA projection axes from accumulated data."""
        if len(self.coeff_history) < 30:
            # Not enough data yet — use first two eigenmode bands
            return
        
        data = np.array(self.coeff_history)  # (T, n_modes*5)
        center = np.mean(data, axis=0)
        centered = data - center
        
        # Covariance
        cov = np.dot(centered.T, centered) / len(centered)
        
        # Top 2 eigenvectors of covariance = principal components
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        
        self.pca_axes = eigvecs[:, idx[:2]].T  # (2, D)
        self.pca_center = center
        
        # Compute scale from projected data
        projected = np.dot(centered, self.pca_axes.T)
        self.pca_scale = max(np.std(projected) * 3, 0.01)
    
    def _project_to_2d(self, full_vector):
        """Project the full eigenmode vector to 2D for the phase portrait."""
        if self.pca_axes is None or self.pca_center is None:
            # Fallback: use mode 1 (A-P) vs mode 2 (L-R) of dominant band
            return full_vector[0], full_vector[1]
        
        centered = full_vector - self.pca_center
        p = np.dot(self.pca_axes, centered) / self.pca_scale
        return float(p[0]), float(p[1])
    
    def step(self):
        self.frame_count += 1
        
        # Estimate FPS
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps_estimate = 0.95 * self.fps_estimate + 0.05 * (1.0 / max(dt, 0.001))
        self.last_time = now
        
        # --- Collect eigenmode coefficients from all 5 bands ---
        all_coeffs = []
        all_phases = []
        band_powers = np.zeros(5)
        
        for bi, band in enumerate(BAND_NAMES):
            field = self.get_blended_input(f'{band}_field', 'first')
            if field is not None and np.iscomplexobj(field):
                coeffs, phases = self._extract_eigenmode_coefficients(field, bi)
            else:
                coeffs = np.zeros(self.n_modes)
                phases = np.zeros(self.n_modes)
            
            all_coeffs.append(coeffs)
            all_phases.append(phases)
            band_powers[bi] = np.sum(coeffs)
        
        # Full state vector: (5 bands × 8 modes) = 40 dimensions
        full_coeffs = np.concatenate(all_coeffs)
        full_phases = np.concatenate(all_phases)
        
        self.coeff_history.append(full_coeffs.copy())
        self.phase_history.append(full_phases.copy())
        
        # --- Dominant band and mode ---
        self.dominant_band_idx = int(np.argmax(band_powers))
        dom_band_coeffs = all_coeffs[self.dominant_band_idx]
        self.dominant_mode_idx = int(np.argmax(dom_band_coeffs))
        
        # --- Dwell detection (on dominant band's eigenmode phases) ---
        current_phase_vec = all_phases[self.dominant_band_idx]
        
        if self.last_phase is not None:
            phase_diff = np.abs(current_phase_vec - self.last_phase)
            phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
            max_jump = np.max(phase_diff)
            
            if max_jump < self.dwell_threshold:
                self.current_dwell += 1
            else:
                if self.current_dwell > 0:
                    self.dwell_times.append(self.current_dwell)
                self.current_dwell = 1
        else:
            self.current_dwell = 1
        
        self.last_phase = current_phase_vec.copy()
        self.current_dwell_ms = int(self.current_dwell * (1000.0 / max(self.fps_estimate, 1)))
        
        # --- Regime detection (every 10 frames) ---
        if self.frame_count % 10 == 0:
            self.regime, self.regime_confidence = self._detect_regime()
            self.metastability_score = self._compute_metastability()
        
        # --- Update PCA projection (every 30 frames) ---
        if self.frame_count % 30 == 0 and len(self.coeff_history) >= 30:
            self._update_pca(full_coeffs)
        
        # --- Project to 2D ---
        px, py = self._project_to_2d(full_coeffs)
        
        self.trail_x.append(px)
        self.trail_y.append(py)
        self.trail_band.append(self.dominant_band_idx)
        self.trail_mode.append(self.dominant_mode_idx)
        
        # --- Update attractor heatmap ---
        hx = int(np.clip((px + 1.5) / 3.0 * self.portrait_size, 0, self.portrait_size - 1))
        hy = int(np.clip((1.5 - py) / 3.0 * self.portrait_size, 0, self.portrait_size - 1))
        
        self.attractor_map *= self.attractor_decay
        # Gaussian splat
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < self.portrait_size and 0 <= ny < self.portrait_size:
                    w = np.exp(-(dx*dx + dy*dy) / 4.0) * 0.5
                    self.attractor_map[ny, nx] += w
        
        # --- Render display ---
        self._render_portrait(all_coeffs, all_phases, band_powers)
    
    def _render_portrait(self, all_coeffs, all_phases, band_powers):
        """Render the full macroscope visualization."""
        S = self.portrait_size  # 400
        img = np.zeros((S, S, 3), dtype=np.uint8)
        
        # === BACKGROUND: Attractor heatmap ===
        heat = self.attractor_map.copy()
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_u8 = (np.clip(heat, 0, 1) * 180).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
        # Darken it significantly so trails are visible
        img = (heat_color * 0.4).astype(np.uint8)
        
        # === TRAJECTORY TRAIL ===
        n = len(self.trail_x)
        if n > 1:
            for i in range(1, n):
                # Convert trail coords to pixel
                x1 = int(np.clip((self.trail_x[i-1] + 1.5) / 3.0 * S, 0, S-1))
                y1 = int(np.clip((1.5 - self.trail_y[i-1]) / 3.0 * S, 0, S-1))
                x2 = int(np.clip((self.trail_x[i] + 1.5) / 3.0 * S, 0, S-1))
                y2 = int(np.clip((1.5 - self.trail_y[i]) / 3.0 * S, 0, S-1))
                
                # Fade with age
                age = (n - i) / max(n, 1)
                alpha = max(0.05, 1.0 - age * 0.9)
                
                # Color by dominant band
                band_idx = self.trail_band[i]
                band_name = BAND_NAMES[band_idx]
                color = BAND_COLORS[band_name]
                c = tuple(int(v * alpha) for v in color)
                
                thickness = 2 if i > n - 5 else 1
                cv2.line(img, (x1, y1), (x2, y2), c, thickness, cv2.LINE_AA)
            
            # Current position: bright dot
            cx = int(np.clip((self.trail_x[-1] + 1.5) / 3.0 * S, 0, S-1))
            cy = int(np.clip((1.5 - self.trail_y[-1]) / 3.0 * S, 0, S-1))
            band_color = BAND_COLORS[BAND_NAMES[self.dominant_band_idx]]
            cv2.circle(img, (cx, cy), 6, band_color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # === TOP PANEL: Regime indicator ===
        regime_color = REGIME_COLORS.get(self.regime, (128, 128, 128))
        
        # Regime bar at top
        bar_width = int(self.regime_confidence * (S - 20))
        cv2.rectangle(img, (10, 8), (10 + bar_width, 18), regime_color, -1)
        cv2.rectangle(img, (10, 8), (S - 10, 18), (80, 80, 80), 1)
        
        # Regime text
        cv2.putText(img, f"{self.regime.upper()}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, regime_color, 1, cv2.LINE_AA)
        
        # Metastability score
        ms_text = f"M={self.metastability_score:.2f}"
        cv2.putText(img, ms_text, (S - 80, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # === RIGHT PANEL: Eigenmode bar chart ===
        bar_x = S - 60
        bar_max_h = 80
        dom_coeffs = all_coeffs[self.dominant_band_idx]
        max_coeff = max(np.max(dom_coeffs), 0.01)
        
        for m in range(min(self.n_modes, 8)):
            bar_h = int(dom_coeffs[m] / max_coeff * bar_max_h)
            y_bottom = S - 40
            y_top = y_bottom - bar_h
            x_left = bar_x + m * 7
            
            # Color: brighter for dominant mode
            if m == self.dominant_mode_idx:
                c = (255, 255, 255)
            else:
                c = (100, 140, 180)
            
            cv2.rectangle(img, (x_left, y_top), (x_left + 5, y_bottom), c, -1)
        
        # Mode labels
        cv2.putText(img, "MODES", (bar_x, S - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
        
        # === BOTTOM PANEL: Band powers + dwell timer ===
        # Band power ring (5 colored bars)
        bp_max = max(np.max(band_powers), 0.01)
        for bi in range(5):
            bw = int(band_powers[bi] / bp_max * 50)
            y = S - 18
            x = 10 + bi * 55
            color = BAND_COLORS[BAND_NAMES[bi]]
            cv2.rectangle(img, (x, y - 6), (x + bw, y), color, -1)
            cv2.putText(img, BAND_NAMES[bi][0].upper(), (x, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        # Dwell timer
        dwell_text = f"DWELL: {self.current_dwell_ms}ms"
        dwell_color = (80, 255, 80) if self.current_dwell_ms > 100 else (200, 200, 200)
        cv2.putText(img, dwell_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dwell_color, 1, cv2.LINE_AA)
        
        # CV indicator
        cv_text = f"CV={self.running_cv:.2f}"
        cv_color = (255, 180, 40) if self.running_cv > 1.0 else (150, 150, 150)
        cv2.putText(img, cv_text, (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cv_color, 1, cv2.LINE_AA)
        
        # Dominant state label
        dom_text = f"{BAND_NAMES[self.dominant_band_idx].upper()} / {MODE_NAMES[min(self.dominant_mode_idx, 7)]}"
        cv2.putText(img, dom_text, (10, S - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                    BAND_COLORS[BAND_NAMES[self.dominant_band_idx]], 1, cv2.LINE_AA)
        
        # === PHASE PORTRAIT LABEL ===
        cv2.putText(img, "PHI-DWELL", (S//2 - 40, S - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 60, 140), 1, cv2.LINE_AA)
        
        self.display_img = img
        
        # --- Also render regime map (secondary output) ---
        self._render_regime_map(all_coeffs, band_powers)
    
    def _render_regime_map(self, all_coeffs, band_powers):
        """Render the regime/attractor map as secondary output."""
        S = 256
        img = np.zeros((S, S, 3), dtype=np.uint8)
        
        # Rescale attractor map to 256x256
        heat = cv2.resize(self.attractor_map, (S, S))
        if heat.max() > 0:
            heat = heat / heat.max()
        
        # Color by regime
        regime_c = REGIME_COLORS.get(self.regime, (128, 128, 128))
        for c in range(3):
            img[:, :, c] = (heat * regime_c[c]).astype(np.uint8)
        
        # Mode coefficient time series (last 100 frames)
        if len(self.coeff_history) > 2:
            hist = list(self.coeff_history)[-min(100, len(self.coeff_history)):]
            data = np.array(hist)  # (T, 40)
            
            # Plot the dominant band's 8 mode coefficients
            dom_bi = self.dominant_band_idx
            mode_data = data[:, dom_bi*self.n_modes:(dom_bi+1)*self.n_modes]
            
            T = mode_data.shape[0]
            for m in range(min(8, mode_data.shape[1])):
                vals = mode_data[:, m]
                v_max = max(np.max(np.abs(vals)), 0.01)
                
                for t in range(1, T):
                    x1 = int((t-1) / T * S)
                    x2 = int(t / T * S)
                    y1 = int(S//2 - vals[t-1] / v_max * 40)
                    y2 = int(S//2 - vals[t] / v_max * 40)
                    
                    brightness = 80 + m * 20
                    c = (brightness, brightness, brightness) if m != self.dominant_mode_idx else (255, 255, 100)
                    cv2.line(img, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
        
        self.regime_img = img
    
    # === OUTPUT METHODS ===
    
    def get_output(self, port_name):
        if port_name == 'portrait':
            return self.display_img
        elif port_name == 'regime_map':
            return self.regime_img
        elif port_name == 'dominant_mode':
            return float(self.dominant_mode_idx + 1)
        elif port_name == 'dominant_band':
            return float(self.dominant_band_idx)
        elif port_name == 'dwell_time':
            return float(self.current_dwell_ms)
        elif port_name == 'regime_index':
            regime_map = {'random': 0, 'critical': 1, 'bursty': 2, 'clocklike': 3}
            return float(regime_map.get(self.regime, 0))
        elif port_name == 'cv_running':
            return float(self.running_cv)
        elif port_name == 'metastability':
            return float(self.metastability_score)
        return None
    
    def get_display_image(self):
        if self.display_img is None:
            return None
        img = self.display_img
        h, w = img.shape[:2]
        img_c = np.ascontiguousarray(img)
        return QtGui.QImage(img_c.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
    
    def get_config_options(self):
        return [
            ("Trail Length (frames)", "max_trail", self.max_trail, None),
            ("Dwell Threshold (deg)", "_dwell_deg", int(np.degrees(self.dwell_threshold)), None),
            ("Attractor Decay", "attractor_decay", self.attractor_decay, "float"),
        ]
    
    def set_config_options(self, options):
        if 'max_trail' in options:
            self.max_trail = int(options['max_trail'])
            # Resize deques
            new_x = deque(self.trail_x, maxlen=self.max_trail)
            new_y = deque(self.trail_y, maxlen=self.max_trail)
            new_b = deque(self.trail_band, maxlen=self.max_trail)
            new_m = deque(self.trail_mode, maxlen=self.max_trail)
            self.trail_x = new_x
            self.trail_y = new_y
            self.trail_band = new_b
            self.trail_mode = new_m
        if '_dwell_deg' in options:
            self.dwell_threshold = np.radians(float(options['_dwell_deg']))
        if 'attractor_decay' in options:
            self.attractor_decay = float(options['attractor_decay'])