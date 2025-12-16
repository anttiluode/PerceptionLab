"""
Attractor Swarm Node
=====================
"Not one eye looking at the field - many eyes looking at each other AND the field."

This implements the full theory developed by Antti, Claude, ChatGPT, and Gemini:

1. SINGLE ATTRACTOR + OPTIC:
   - An attractor observes the field through a coupling kernel K_κ
   - Coupling = integration window width = spectral bandwidth
   - F_eff(t) = (S * K_κ)(t)

2. MULTIPLE ATTRACTORS + INTER-OPTICS:
   - Each attractor i has its own field coupling κ_i
   - Each attractor i observes other attractors j through inter-optics κ_ij
   - m_ij(t) = O_κij[x_j(t)] - "how sharply does i sample j?"
   
3. THE THREE REGIMES:
   - Low κ: over-integration → soup (structure dies)
   - Critical κ: balanced → lattices/stars (maximal structure)
   - High κ: bandwidth > Nyquist → stripes (aliasing collapse)

4. COALITION FORMATION:
   - When κ_ij rises between a subset, they share high-detail info
   - They lock phases → form "super-attractor" coalitions
   - This is attention implemented as signal theory

5. HOMEOSTATIC CONTROL:
   - κ_ij adapts based on whether j helps i reduce error/increase structure
   - The network learns who to couple to, moment by moment

CREATED: December 2025
AUTHORS: Antti + Claude + ChatGPT + Gemini
"""

import numpy as np
import cv2
from collections import deque
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter

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
            return None


class AttractorSwarmNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Attractor Swarm"
    NODE_COLOR = QtGui.QColor(255, 100, 50)  # Orange - swarm intelligence
    
    def __init__(self):
        super().__init__()
        self.node_title = "Attractor Swarm (Multi-Observer Optics)"
        
        self.inputs = {
            'theta_signal': 'signal',
            'alpha_signal': 'signal',
            'beta_signal': 'signal',
            'gamma_signal': 'signal',
            'token_stream': 'spectrum',
            'global_coupling': 'signal',    # Base coupling for all
            'adaptation_rate': 'signal',    # How fast optics adapt
            'lattice_zoom': 'signal',
            'lattice_freq': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'display': 'image',
            'swarm_field': 'complex_spectrum',      # Combined lattice
            'coupling_matrix': 'spectrum',          # Who couples to whom
            'coalition_labels': 'spectrum',         # Which attractors form groups
            'dominant_attractor': 'signal',         # Which one is "winning"
            'global_symmetry': 'signal',            # 6-fold symmetry score
            'anisotropy': 'signal',                 # Stripe detection
            'criticality_score': 'signal',          # How close to critical
        }
        
        # === SWARM PARAMETERS ===
        self.N = 4  # Number of attractors (one per band initially)
        self.field_size = 64
        self.epoch = 0
        
        # === ATTRACTOR STATES ===
        # Each attractor has a complex state (amplitude + phase)
        self.states = np.ones(self.N, dtype=np.complex128)
        self.state_phases = np.zeros(self.N)
        
        # === FIELD OBSERVATIONS ===
        # What each attractor sees from the raw field
        self.field_obs = np.zeros(self.N, dtype=np.complex128)
        
        # === OPTICS MATRICES ===
        # κ_i: how each attractor couples to the field
        self.field_kappa = np.ones(self.N) * 0.5  # Start at critical
        
        # κ_ij: how attractor i couples to attractor j (inter-optics)
        # This is the key new structure
        self.inter_kappa = np.ones((self.N, self.N)) * 0.3
        np.fill_diagonal(self.inter_kappa, 0)  # Don't self-couple
        
        # === INTEGRATION WINDOWS (derived from kappa) ===
        # Higher kappa = sharper window = more high-freq detail
        self.integration_windows = np.ones(self.N) * 10  # samples
        
        # === HISTORIES FOR WINDOWED INTEGRATION ===
        self.history_len = 100
        self.band_histories = [deque(maxlen=self.history_len) for _ in range(4)]
        self.state_histories = [deque(maxlen=self.history_len) for _ in range(self.N)]
        
        # === ADAPTATION ===
        self.adaptation_rate = 0.01
        self.symmetry_target = 0.5  # Try to stay near critical
        
        # === METRICS ===
        self.symmetry_scores = np.zeros(self.N)
        self.anisotropy_scores = np.zeros(self.N)
        self.coalition_matrix = np.zeros((self.N, self.N))
        
        # === LATTICE PARAMETERS ===
        self.lattice_zoom = 1.0
        self.lattice_freq = 4.0
        
        # === FIELDS ===
        self.individual_fields = [np.zeros((self.field_size, self.field_size), dtype=np.complex128) 
                                  for _ in range(self.N)]
        self.combined_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        
        # === DISPLAY ===
        self._display = np.zeros((700, 1000, 3), dtype=np.uint8)
        
        # === LABELS ===
        self.attractor_names = ['θ-Slow', 'α-Mid', 'β-Fast', 'γ-Ultra']
        self.attractor_colors = [(100, 150, 255), (100, 255, 150), (255, 200, 100), (255, 100, 150)]
    
    def _parse_input(self, val):
        """Parse various input formats to float"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        if isinstance(val, np.ndarray):
            return float(np.mean(np.abs(val))) if val.size > 0 else 0.0
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return float(val[0]) if not hasattr(val[0], '__len__') else 0.0
        return 0.0
    
    def _apply_optic(self, signal_history, kappa):
        """
        Apply the optic kernel to a signal history.
        
        This is the core operation: convolution with Gaussian kernel
        whose width is controlled by kappa.
        
        High kappa = narrow window = high-freq passes
        Low kappa = wide window = averaging/smoothing
        """
        if len(signal_history) < 5:
            return 0.0 + 0j
        
        sig = np.array(list(signal_history))
        n = len(sig)
        
        # Kappa controls window sharpness
        # Higher kappa = sharper (smaller sigma)
        sigma = max(1.0, 10.0 / (kappa + 0.1))
        
        # Create Gaussian kernel
        t = np.arange(n)
        center = n - 1  # Weight toward recent
        kernel = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        kernel = kernel / (kernel.sum() + 1e-10)
        
        # Apply kernel (weighted integration)
        integrated = np.sum(sig * kernel)
        
        # Estimate phase from recent samples
        if len(sig) > 10:
            try:
                analytic = scipy_signal.hilbert(sig - np.mean(sig))
                phase = np.angle(analytic[-1])
                amp = np.abs(integrated)
                return amp * np.exp(1j * phase)
            except:
                return integrated + 0j
        
        return integrated + 0j
    
    def _compute_inter_observation(self, i, j):
        """
        Compute how attractor i observes attractor j through their inter-optic.
        
        m_ij(t) = O_κij[x_j(t)]
        """
        if len(self.state_histories[j]) < 5:
            return 0.0 + 0j
        
        kappa_ij = self.inter_kappa[i, j]
        
        # Get j's state history as real values for integration
        j_history = [np.abs(s) for s in self.state_histories[j]]
        
        return self._apply_optic(j_history, kappa_ij)
    
    def _create_attractor_field(self, attractor_idx):
        """
        Create the lattice field for one attractor based on its state
        and its coupling to others.
        """
        size = self.field_size
        span = np.pi * self.lattice_zoom
        
        x = np.linspace(-span, span, size)
        y = np.linspace(-span, span, size)
        X, Y = np.meshgrid(x, y)
        
        field = np.zeros((size, size), dtype=np.complex128)
        
        # This attractor's state
        state = self.states[attractor_idx]
        amp = np.abs(state)
        phase = np.angle(state)
        
        # Base frequency modulated by attractor index
        base_freq = self.lattice_freq * (1 + attractor_idx * 0.2)
        
        # 6 waves at 60° for hexagonal lattice
        for i in range(6):
            angle = i * np.pi / 3 + phase
            
            # Modulate amplitude by inter-coupling
            # Attractors that are strongly coupled contribute more
            coupling_boost = 1.0
            for j in range(self.N):
                if j != attractor_idx:
                    coupling_boost += 0.2 * self.inter_kappa[attractor_idx, j] * np.abs(self.states[j])
            
            wave_amp = amp * coupling_boost / self.N
            
            kx = base_freq * np.cos(angle)
            ky = base_freq * np.sin(angle)
            
            wave = wave_amp * np.exp(1j * (kx * X + ky * Y))
            field += wave
        
        return field
    
    def _compute_symmetry_score(self, field):
        """
        Compute 6-fold symmetry score from field.
        High score = hexagonal/star pattern (critical regime)
        Low score = stripes or soup
        """
        # FFT of magnitude
        mag = np.abs(field)
        fft = np.fft.fftshift(np.fft.fft2(mag))
        power = np.abs(fft) ** 2
        
        # Sample at 60° intervals around center
        center = self.field_size // 2
        radius = self.field_size // 4
        
        angles = np.arange(0, 360, 60) * np.pi / 180
        samples = []
        
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            if 0 <= x < self.field_size and 0 <= y < self.field_size:
                samples.append(power[y, x])
        
        if len(samples) < 6:
            return 0.0
        
        samples = np.array(samples)
        
        # High symmetry = all samples similar
        mean_power = np.mean(samples)
        if mean_power < 1e-10:
            return 0.0
        
        std_power = np.std(samples)
        symmetry = 1.0 - (std_power / (mean_power + 1e-10))
        
        return max(0, min(1, symmetry))
    
    def _compute_anisotropy(self, field):
        """
        Compute anisotropy (stripe-ness) of field.
        High anisotropy = collapsed into stripes (over-coupled)
        Low anisotropy = isotropic (soup or lattice)
        """
        mag = np.abs(field)
        
        # Compute directional gradients
        gx = np.abs(np.diff(mag, axis=1)).mean()
        gy = np.abs(np.diff(mag, axis=0)).mean()
        
        # Anisotropy = difference between directional gradients
        total = gx + gy + 1e-10
        anisotropy = abs(gx - gy) / total
        
        return anisotropy
    
    def _detect_coalitions(self):
        """
        Detect which attractors form coalitions based on:
        - High inter-coupling
        - Phase synchrony
        - Similar symmetry scores
        """
        coalition = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Coupling strength
                coupling = (self.inter_kappa[i, j] + self.inter_kappa[j, i]) / 2
                
                # Phase coherence
                phase_diff = np.abs(np.angle(self.states[i]) - np.angle(self.states[j]))
                phase_coherence = np.cos(phase_diff)
                
                # Symmetry similarity
                sym_diff = np.abs(self.symmetry_scores[i] - self.symmetry_scores[j])
                sym_similarity = 1.0 - sym_diff
                
                # Coalition score
                score = coupling * (0.5 + 0.3 * phase_coherence + 0.2 * sym_similarity)
                
                coalition[i, j] = score
                coalition[j, i] = score
        
        self.coalition_matrix = coalition
        return coalition
    
    def _adapt_optics(self):
        """
        Homeostatic adaptation of optics.
        
        Rule: if attractor j helps attractor i maintain critical regime,
        increase κ_ij. Otherwise decrease.
        """
        for i in range(self.N):
            # Target: maximize symmetry, minimize anisotropy
            reward_i = self.symmetry_scores[i] - self.anisotropy_scores[i]
            
            # Field coupling: try to stay near critical
            error = self.symmetry_target - self.symmetry_scores[i]
            self.field_kappa[i] += self.adaptation_rate * error
            self.field_kappa[i] = np.clip(self.field_kappa[i], 0.1, 2.0)
            
            # Inter-coupling: strengthen connections that help
            for j in range(self.N):
                if i == j:
                    continue
                
                # How much does j's influence correlate with i's reward?
                j_influence = np.abs(self.states[j]) * self.inter_kappa[i, j]
                
                # Simple rule: if both have good symmetry, strengthen
                j_reward = self.symmetry_scores[j] - self.anisotropy_scores[j]
                combined = reward_i * j_reward
                
                self.inter_kappa[i, j] += self.adaptation_rate * combined
                self.inter_kappa[i, j] = np.clip(self.inter_kappa[i, j], 0.05, 1.0)
    
    def step(self):
        self.epoch += 1
        
        # === GET INPUTS ===
        theta = self._parse_input(self.get_blended_input('theta_signal', 'sum'))
        alpha = self._parse_input(self.get_blended_input('alpha_signal', 'sum'))
        beta = self._parse_input(self.get_blended_input('beta_signal', 'sum'))
        gamma = self._parse_input(self.get_blended_input('gamma_signal', 'sum'))
        
        global_coupling = self._parse_input(self.get_blended_input('global_coupling', 'sum'))
        adaptation = self._parse_input(self.get_blended_input('adaptation_rate', 'sum'))
        zoom = self._parse_input(self.get_blended_input('lattice_zoom', 'sum'))
        freq = self._parse_input(self.get_blended_input('lattice_freq', 'sum'))
        reset = self._parse_input(self.get_blended_input('reset', 'sum'))
        
        token_stream = self.get_blended_input('token_stream', 'sum')
        
        # Handle reset
        if reset > 0.5:
            self.states = np.ones(self.N, dtype=np.complex128)
            self.inter_kappa = np.ones((self.N, self.N)) * 0.3
            np.fill_diagonal(self.inter_kappa, 0)
            self.field_kappa = np.ones(self.N) * 0.5
            return
        
        # Update parameters
        if global_coupling > 0:
            self.field_kappa[:] = global_coupling
        if adaptation > 0:
            self.adaptation_rate = adaptation
        if zoom > 0:
            self.lattice_zoom = np.clip(zoom, 0.25, 8.0)
        if freq > 0:
            self.lattice_freq = np.clip(freq, 1.0, 16.0)
        
        # Extract from token stream
        if token_stream is not None:
            try:
                if isinstance(token_stream, np.ndarray) and len(token_stream) >= 4:
                    theta = float(token_stream[0]) if theta == 0 else theta
                    alpha = float(token_stream[1]) if alpha == 0 else alpha
                    beta = float(token_stream[2]) if beta == 0 else beta
                    gamma = float(token_stream[3]) if gamma == 0 else gamma
            except:
                pass
        
        # === UPDATE HISTORIES ===
        bands = [theta, alpha, beta, gamma]
        for i, b in enumerate(bands):
            self.band_histories[i].append(b)
        
        # === FIELD OBSERVATIONS ===
        # Each attractor observes the field through its optic
        for i in range(self.N):
            self.field_obs[i] = self._apply_optic(self.band_histories[i], self.field_kappa[i])
        
        # === INTER-ATTRACTOR OBSERVATIONS ===
        inter_obs = np.zeros((self.N, self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    inter_obs[i, j] = self._compute_inter_observation(i, j)
        
        # === STATE UPDATE ===
        # Each attractor's state is pulled by:
        # 1. Its field observation
        # 2. Its observations of other attractors
        
        eta_field = 0.3  # Field coupling strength
        eta_inter = 0.2  # Inter-attractor coupling strength
        
        for i in range(self.N):
            # Field pull
            field_pull = eta_field * self.field_kappa[i] * self.field_obs[i]
            
            # Inter-attractor pull
            inter_pull = 0j
            for j in range(self.N):
                if i != j:
                    inter_pull += eta_inter * self.inter_kappa[i, j] * inter_obs[i, j]
            
            # Update state
            self.states[i] = (1 - eta_field - eta_inter) * self.states[i] + field_pull + inter_pull
            
            # Normalize to prevent blowup
            if np.abs(self.states[i]) > 10:
                self.states[i] = self.states[i] / np.abs(self.states[i]) * 10
        
        # === RECORD STATE HISTORIES ===
        for i in range(self.N):
            self.state_histories[i].append(self.states[i])
        
        # === CREATE INDIVIDUAL FIELDS ===
        for i in range(self.N):
            self.individual_fields[i] = self._create_attractor_field(i)
        
        # === COMPUTE METRICS ===
        for i in range(self.N):
            self.symmetry_scores[i] = self._compute_symmetry_score(self.individual_fields[i])
            self.anisotropy_scores[i] = self._compute_anisotropy(self.individual_fields[i])
        
        # === COMBINE FIELDS ===
        # Weighted by symmetry score - better observers contribute more
        self.combined_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        total_weight = 0
        for i in range(self.N):
            weight = self.symmetry_scores[i] + 0.1  # Avoid zero weight
            self.combined_field += weight * self.individual_fields[i]
            total_weight += weight
        self.combined_field /= (total_weight + 1e-10)
        
        # === DETECT COALITIONS ===
        self._detect_coalitions()
        
        # === ADAPT OPTICS ===
        if self.adaptation_rate > 0:
            self._adapt_optics()
        
        # === UPDATE DISPLAY ===
        self._update_display()
    
    def _update_display(self):
        """Create comprehensive visualization"""
        img = np.zeros((700, 1000, 3), dtype=np.uint8)
        img[:] = (20, 25, 30)
        
        # === TITLE ===
        cv2.putText(img, "ATTRACTOR SWARM - Multi-Observer Optics", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 50), 2)
        cv2.putText(img, f"Epoch: {self.epoch} | N={self.N} attractors", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # === INDIVIDUAL ATTRACTOR PANELS ===
        panel_size = 100
        panel_y = 80
        
        for i in range(self.N):
            panel_x = 20 + i * (panel_size + 30)
            
            # Label
            cv2.putText(img, self.attractor_names[i], (panel_x, panel_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.attractor_colors[i], 1)
            
            # Field visualization
            field = self.individual_fields[i]
            mag = np.abs(field)
            phase = np.angle(field)
            
            hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
            hsv[:, :, 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
            hsv[:, :, 1] = 200
            hsv[:, :, 2] = (mag / (mag.max() + 1e-10) * 255).astype(np.uint8)
            
            field_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            field_resized = cv2.resize(field_color, (panel_size, panel_size))
            
            img[panel_y:panel_y + panel_size, panel_x:panel_x + panel_size] = field_resized
            
            # Metrics
            cv2.putText(img, f"Sym: {self.symmetry_scores[i]:.2f}", 
                       (panel_x, panel_y + panel_size + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            cv2.putText(img, f"Ani: {self.anisotropy_scores[i]:.2f}", 
                       (panel_x, panel_y + panel_size + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            cv2.putText(img, f"κ: {self.field_kappa[i]:.2f}", 
                       (panel_x, panel_y + panel_size + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # === COMBINED FIELD ===
        combined_x, combined_y = 550, 80
        combined_size = 180
        
        cv2.putText(img, "COMBINED FIELD", (combined_x, combined_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
        
        mag = np.abs(self.combined_field)
        phase = np.angle(self.combined_field)
        
        hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = (mag / (mag.max() + 1e-10) * 255).astype(np.uint8)
        
        combined_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined_resized = cv2.resize(combined_color, (combined_size, combined_size))
        img[combined_y:combined_y + combined_size, combined_x:combined_x + combined_size] = combined_resized
        
        # Global metrics
        global_sym = self._compute_symmetry_score(self.combined_field)
        global_ani = self._compute_anisotropy(self.combined_field)
        
        cv2.putText(img, f"Global Sym: {global_sym:.3f}", (combined_x, combined_y + combined_size + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
        cv2.putText(img, f"Global Ani: {global_ani:.3f}", (combined_x, combined_y + combined_size + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
        
        # Criticality score
        criticality = global_sym * (1 - global_ani)
        cv2.putText(img, f"CRITICALITY: {criticality:.3f}", (combined_x, combined_y + combined_size + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100) if criticality > 0.3 else (255, 100, 100), 1)
        
        # === INTER-COUPLING MATRIX ===
        matrix_x, matrix_y = 780, 80
        cell_size = 40
        
        cv2.putText(img, "INTER-OPTICS κ_ij", (matrix_x, matrix_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        for i in range(self.N):
            for j in range(self.N):
                x = matrix_x + j * cell_size
                y = matrix_y + i * cell_size
                
                val = self.inter_kappa[i, j]
                intensity = int(val * 255)
                
                if i == j:
                    color = (40, 40, 40)
                else:
                    color = (intensity, intensity // 2, 50)
                
                cv2.rectangle(img, (x, y), (x + cell_size - 2, y + cell_size - 2), color, -1)
                cv2.putText(img, f"{val:.1f}", (x + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # === COALITION MATRIX ===
        coal_x, coal_y = 780, 280
        
        cv2.putText(img, "COALITIONS", (coal_x, coal_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        for i in range(self.N):
            for j in range(self.N):
                x = coal_x + j * cell_size
                y = coal_y + i * cell_size
                
                val = self.coalition_matrix[i, j]
                intensity = int(val * 255)
                
                color = (50, intensity, intensity // 2)
                
                cv2.rectangle(img, (x, y), (x + cell_size - 2, y + cell_size - 2), color, -1)
        
        # === STATE DISPLAY ===
        state_x, state_y = 20, 280
        cv2.putText(img, "ATTRACTOR STATES", (state_x, state_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        
        for i in range(self.N):
            y = state_y + 25 + i * 30
            
            amp = np.abs(self.states[i])
            phase = np.angle(self.states[i])
            
            # Amplitude bar
            bar_width = int(min(amp * 50, 150))
            cv2.rectangle(img, (state_x, y), (state_x + bar_width, y + 15), self.attractor_colors[i], -1)
            
            # Phase indicator
            phase_x = state_x + 160 + int(20 * np.cos(phase))
            phase_y = y + 7 + int(7 * np.sin(phase))
            cv2.circle(img, (phase_x, phase_y), 4, (255, 255, 255), -1)
            
            cv2.putText(img, f"{self.attractor_names[i]}: |{amp:.1f}| ∠{np.degrees(phase):.0f}°",
                       (state_x + 200, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.attractor_colors[i], 1)
        
        # === FIELD KAPPA DISPLAY ===
        kappa_x, kappa_y = 20, 420
        cv2.putText(img, "FIELD COUPLING κ_i (integration windows)", (kappa_x, kappa_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 150), 1)
        
        for i in range(self.N):
            y = kappa_y + 20 + i * 25
            kappa = self.field_kappa[i]
            
            bar_width = int(kappa * 100)
            
            # Color based on regime
            if kappa < 0.3:
                color = (150, 150, 100)  # Low - soup
                regime = "SOUP"
            elif kappa > 0.8:
                color = (100, 100, 200)  # High - stripes
                regime = "STRIPES"
            else:
                color = (100, 200, 100)  # Critical
                regime = "CRITICAL"
            
            cv2.rectangle(img, (kappa_x, y), (kappa_x + bar_width, y + 15), color, -1)
            cv2.putText(img, f"{self.attractor_names[i]}: κ={kappa:.2f} [{regime}]",
                       (kappa_x + 120, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # === THEORY BOX ===
        theory_y = 560
        cv2.putText(img, "OPTICS OF INFORMATION:", (20, theory_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        cv2.putText(img, "F_eff(t) = (S * K_kappa)(t)  |  kappa = integration window = spectral bandwidth",
                   (20, theory_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 150, 120), 1)
        cv2.putText(img, "Low kappa -> averaging -> soup  |  Critical kappa -> interference -> lattice  |  High kappa -> aliasing -> stripes",
                   (20, theory_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 150, 120), 1)
        cv2.putText(img, "Multiple attractors can tune optics BETWEEN each other -> coalition formation -> attention",
                   (20, theory_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 150, 120), 1)
        
        # === DOMINANT ATTRACTOR ===
        dominant = np.argmax(self.symmetry_scores)
        cv2.putText(img, f"Dominant: {self.attractor_names[dominant]}", (750, theory_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.attractor_colors[dominant], 1)
        
        # === PARAMETERS ===
        cv2.putText(img, f"zoom={self.lattice_zoom:.1f} | freq={self.lattice_freq:.1f} | adapt={self.adaptation_rate:.3f}",
                   (20, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        self._display = img
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'swarm_field':
            return self.combined_field
        elif name == 'coupling_matrix':
            return self.inter_kappa.flatten()
        elif name == 'coalition_labels':
            # Simple coalition detection: threshold the matrix
            labels = np.zeros(self.N)
            for i in range(self.N):
                labels[i] = np.argmax(self.coalition_matrix[i, :])
            return labels
        elif name == 'dominant_attractor':
            return float(np.argmax(self.symmetry_scores))
        elif name == 'global_symmetry':
            return float(self._compute_symmetry_score(self.combined_field))
        elif name == 'anisotropy':
            return float(self._compute_anisotropy(self.combined_field))
        elif name == 'criticality_score':
            sym = self._compute_symmetry_score(self.combined_field)
            ani = self._compute_anisotropy(self.combined_field)
            return float(sym * (1 - ani))
        return None
    
    def get_display_image(self):
        h, w = self._display.shape[:2]
        return QtGui.QImage(self._display.data, w, h, w * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Adaptation Rate", "adaptation_rate", self.adaptation_rate, None),
            ("Lattice Zoom", "lattice_zoom", self.lattice_zoom, None),
            ("Lattice Freq", "lattice_freq", self.lattice_freq, None),
        ]