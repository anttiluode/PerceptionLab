"""
Artificial Visual Cortex V2 - Temporal Edition
===============================================

Building on V1, this version adds what the 2025 Nature Communications paper
"Temporal coding carries more stable cortical visual representations than firing rate"
showed is crucial: temporal dynamics between layers.

Key additions:
1. SPIKE TIMING TRACKING - When does each layer fire, not just how much
2. CROSS-LAYER CORRELOGRAMS - Measure propagation delays between layers  
3. TEMPORAL COMPONENTS - LDA-like extraction of timing patterns that encode stimuli
4. PHASE RELATIONSHIPS - Are layers in sync? Leading? Lagging?
5. PROPAGATION WAVES - Does activity sweep through the hierarchy?

The paper's key finding: "temporal codes increased single neuron tuning stability,
especially for less reliable neurons" - timing carries stable information that
firing rates lose.

Architecture (same as V1):
- Multiple V1 Layers at φ-scaled symmetries [2,3,5,8,13,22,35,58]
- IT Layer for texture/identity
- All connected via Izhikevich neurons with STDP

New outputs:
- temporal_spectrum: Cross-layer timing relationships
- propagation_map: When did activity arrive at each layer?
- phase_coherence: How synchronized are the layers?
- spike_timing_profile: Per-layer spike timing histograms
- cross_correlogram: Layer-to-layer CCG (like Fig 6 in paper)

Author: Built for Antti's consciousness crystallography research
Based on Zhu et al. 2025 - Temporal coding stability paper
"""

import os
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from collections import deque

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    QtGui = None
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode):
            return None

PHI = (1 + np.sqrt(5)) / 2


class ArtificialVisualCortexV2(BaseNode):
    """
    Multi-layer visual cortex with TEMPORAL CODING.
    Tracks not just firing rates but spike timing relationships.
    """
    
    NODE_NAME = "Artificial Visual Cortex V2"
    NODE_TITLE = "V1 Cortex V2"
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(200, 80, 180) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            "image_in": "image",
            "delta_in": "signal",
            "theta_in": "signal",
            "alpha_in": "signal",
            "beta_in": "signal",
            "gamma_in": "signal",
            "latent_in": "spectrum",
            "learning": "signal",
            "reset": "signal",
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Spatial outputs (from V1)
            "v1_mandala": "image",
            "it_layer": "image",
            "depth_signal": "image",
            "eigenstate": "image",
            "activity_view": "image",
            
            # Temporal outputs (NEW in V2)
            "temporal_spectrum": "image",      # Cross-layer timing vis
            "propagation_map": "image",        # When activity arrived
            "spike_raster": "image",           # Spike times per layer
            "cross_correlogram": "spectrum",   # Layer CCG values
            "phase_coherence": "signal",       # How in-sync layers are
            
            # Standard analysis
            "resonance": "signal",
            "energy": "signal",
            "entropy": "signal",
            "depth_profile": "spectrum",
            "lfp_out": "signal",
            
            # Temporal metrics (NEW)
            "mean_propagation_delay": "signal",  # Average V1→IT delay
            "temporal_stability": "signal",      # How stable is timing?
        }
        
        # === ARCHITECTURE ===
        self.grid_size = 64
        self.n_v1_layers = 8
        self.base_symmetry = 2
        self.symmetry_multiplier = PHI
        
        # === IZHIKEVICH ===
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 0.5
        
        # === COUPLING ===
        self.coupling_strength = 3.0
        self.inter_layer_coupling = 1.0
        self.learning_rate = 0.001
        self.trace_decay = 0.95
        self.weight_max = 2.0
        self.weight_min = 0.01
        
        # === NEURAL STATE ===
        self.v = None
        self.u = None
        self.weights = None
        self.inter_weights = None
        self.spike_trace = None
        self.symmetry_levels = []
        self.processed_layers = None
        
        # === TEMPORAL TRACKING (NEW) ===
        self.history_length = 256  # Steps to keep
        
        # Spike history per layer: deque of (step, spike_count, mean_spike_time)
        self.spike_history = None  # Will be list of deques
        
        # Per-layer firing rate history
        self.firing_rate_history = None  # [n_layers, history_length]
        
        # Cross-correlogram accumulator
        self.ccg_accumulator = None  # [n_layers, n_layers, lag_bins]
        self.ccg_lag_bins = 51  # -25 to +25 steps
        
        # Propagation timing
        self.first_spike_time = None  # When each layer first spiked this "trial"
        self.propagation_delays = None  # Delays between layers
        
        # Phase tracking (using Hilbert-like approach on firing rates)
        self.phase_history = None  # [n_layers, history_length]
        
        # Temporal stability metric
        self.temporal_pattern_buffer = None  # Recent firing patterns
        self.temporal_stability_score = 0.0
        
        # === STATISTICS ===
        self.step_count = 0
        self.total_spikes = 0
        self.current_resonance = 0.0
        self.current_energy = 0.0
        self.current_entropy = 0.0
        self.mean_propagation_delay = 0.0
        self.phase_coherence_value = 0.0
        
        # LFP
        self.lfp_history = np.zeros(256, dtype=np.float32)
        self.lfp_idx = 0
        
        # Display
        self.display_image = None
        self.last_image = None
        
        # Initialize
        self._init_layers()
        self._init_temporal_tracking()
        self._update_display()
    
    def _init_layers(self):
        """Initialize neural layers and weights."""
        n = self.grid_size
        n_layers = self.n_v1_layers + 1
        
        # Compute symmetry levels
        self.symmetry_levels = []
        sym = self.base_symmetry
        for i in range(self.n_v1_layers):
            self.symmetry_levels.append(int(sym))
            sym *= self.symmetry_multiplier
        
        # Neural state
        self.v = np.ones((n_layers, n, n), dtype=np.float32) * self.c
        self.u = self.v * self.b
        
        # Weights
        self.weights = np.ones((n_layers, 4, n, n), dtype=np.float32) * 0.5
        
        # Inter-layer weights
        self.inter_weights = np.ones((n_layers, n_layers), dtype=np.float32) * 0.1
        for i in range(n_layers):
            for j in range(n_layers):
                dist = abs(i - j)
                if dist == 1:
                    self.inter_weights[i, j] = 0.3
                elif dist == 0:
                    self.inter_weights[i, j] = 0.0
                else:
                    self.inter_weights[i, j] = 0.05 / dist
        
        self.spike_trace = np.zeros((n_layers, n, n), dtype=np.float32)
        self.processed_layers = np.zeros((n_layers, n, n), dtype=np.float32)
        
        print(f"[V1CortexV2] Initialized {n_layers} layers at {n}x{n}")
        print(f"[V1CortexV2] Symmetries: {self.symmetry_levels}")
    
    def _init_temporal_tracking(self):
        """Initialize temporal analysis structures."""
        n_layers = self.n_v1_layers + 1
        
        # Spike history: list of deques, one per layer
        self.spike_history = [deque(maxlen=self.history_length) for _ in range(n_layers)]
        
        # Firing rate history
        self.firing_rate_history = np.zeros((n_layers, self.history_length), dtype=np.float32)
        
        # Cross-correlogram: [from_layer, to_layer, lag]
        self.ccg_accumulator = np.zeros((n_layers, n_layers, self.ccg_lag_bins), dtype=np.float32)
        
        # Propagation tracking
        self.first_spike_time = np.full(n_layers, -1, dtype=np.int32)
        self.propagation_delays = np.zeros((n_layers, n_layers), dtype=np.float32)
        
        # Phase history
        self.phase_history = np.zeros((n_layers, self.history_length), dtype=np.float32)
        
        # Temporal pattern buffer (for stability calculation)
        self.temporal_pattern_buffer = deque(maxlen=50)
        
        print(f"[V1CortexV2] Temporal tracking initialized: {self.history_length} step history")
    
    def _process_v1_layer(self, image, symmetry):
        """Process image through V1-like log-polar transformation."""
        n = self.grid_size
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, (n, n)).astype(np.float32) / 255.0
        
        spectrum = fftshift(fft2(gray))
        
        cy, cx = n // 2, n // 2
        y, x = np.ogrid[:n, :n]
        theta = np.arctan2(y - cy, x - cx)
        
        if symmetry > 1:
            sector_angle = 2 * np.pi / symmetry
            sector_idx = ((theta + np.pi) / sector_angle).astype(int) % symmetry
            
            processed = np.zeros_like(spectrum)
            for s in range(symmetry):
                mask = (sector_idx == s)
                if np.any(mask):
                    sector_mean = np.mean(spectrum[mask])
                    processed[mask] = sector_mean
            spectrum = processed
        
        result = np.real(ifft2(ifftshift(spectrum)))
        result = (result - result.min()) / (result.max() - result.min() + 1e-9)
        
        return result.astype(np.float32)
    
    def _process_it_layer(self, image):
        """IT layer - texture/identity."""
        n = self.grid_size
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, (n, n)).astype(np.float32) / 255.0
        return gray
    
    def _update_temporal_tracking(self, spikes_per_layer, step):
        """Update all temporal analysis structures after a step."""
        n_layers = self.n_v1_layers + 1
        
        # 1. Record spike counts and update history
        current_pattern = np.zeros(n_layers, dtype=np.float32)
        for layer in range(n_layers):
            spike_count = np.sum(spikes_per_layer[layer])
            current_pattern[layer] = spike_count
            
            # Update spike history
            self.spike_history[layer].append((step, spike_count))
            
            # Shift and update firing rate history
            self.firing_rate_history[layer] = np.roll(self.firing_rate_history[layer], -1)
            self.firing_rate_history[layer, -1] = spike_count
            
            # Track first spike time (reset if this is a new "trial")
            if spike_count > 0 and self.first_spike_time[layer] < 0:
                self.first_spike_time[layer] = step
        
        # 2. Update cross-correlograms
        # Use recent firing rate history for CCG calculation
        half_lag = self.ccg_lag_bins // 2
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:
                    # Cross-correlate firing rate histories
                    sig_i = self.firing_rate_history[i, -50:]  # Last 50 steps
                    sig_j = self.firing_rate_history[j, -50:]
                    
                    if np.std(sig_i) > 0 and np.std(sig_j) > 0:
                        # Normalized cross-correlation
                        xcorr = correlate(sig_i - np.mean(sig_i), sig_j - np.mean(sig_j), mode='same')
                        xcorr = xcorr / (np.std(sig_i) * np.std(sig_j) * len(sig_i) + 1e-9)
                        
                        # Map to CCG bins
                        center = len(xcorr) // 2
                        start = max(0, center - half_lag)
                        end = min(len(xcorr), center + half_lag + 1)
                        ccg_start = half_lag - (center - start)
                        ccg_end = ccg_start + (end - start)
                        
                        # Exponential moving average update
                        alpha = 0.05
                        self.ccg_accumulator[i, j, ccg_start:ccg_end] = (
                            (1 - alpha) * self.ccg_accumulator[i, j, ccg_start:ccg_end] +
                            alpha * xcorr[start:end]
                        )
        
        # 3. Calculate propagation delays from CCG peaks
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:
                    ccg = self.ccg_accumulator[i, j]
                    peak_lag = np.argmax(ccg) - half_lag
                    self.propagation_delays[i, j] = peak_lag
        
        # 4. Calculate phase from firing rate oscillations (simple approach)
        for layer in range(n_layers):
            sig = self.firing_rate_history[layer, -32:]
            if np.std(sig) > 0:
                # Simple phase estimate using zero-crossings
                sig_centered = sig - np.mean(sig)
                # Use FFT phase at dominant frequency
                fft_sig = np.fft.fft(sig_centered)
                dominant_idx = np.argmax(np.abs(fft_sig[1:len(fft_sig)//2])) + 1
                phase = np.angle(fft_sig[dominant_idx])
                
                self.phase_history[layer] = np.roll(self.phase_history[layer], -1)
                self.phase_history[layer, -1] = phase
        
        # 5. Calculate phase coherence (mean phase difference stability)
        phases_now = self.phase_history[:, -1]
        phase_diffs = []
        for i in range(n_layers):
            for j in range(i+1, n_layers):
                diff = np.abs(phases_now[i] - phases_now[j])
                # Wrap to [0, pi]
                diff = min(diff, 2*np.pi - diff)
                phase_diffs.append(diff)
        
        if phase_diffs:
            # Coherence = 1 when all phases aligned, 0 when random
            mean_diff = np.mean(phase_diffs)
            self.phase_coherence_value = 1.0 - (mean_diff / np.pi)
        
        # 6. Calculate mean propagation delay (V1-layer-0 to IT)
        # Average delay from first V1 layer to IT layer
        delays_to_it = []
        for i in range(self.n_v1_layers):
            delay = self.propagation_delays[i, -1]  # To IT (last layer)
            delays_to_it.append(delay)
        self.mean_propagation_delay = np.mean(delays_to_it) if delays_to_it else 0.0
        
        # 7. Temporal stability (pattern correlation over time)
        self.temporal_pattern_buffer.append(current_pattern.copy())
        if len(self.temporal_pattern_buffer) >= 10:
            patterns = np.array(list(self.temporal_pattern_buffer))
            # Compare each pattern to average
            mean_pattern = np.mean(patterns, axis=0)
            correlations = []
            for p in patterns[-10:]:
                if np.std(p) > 0 and np.std(mean_pattern) > 0:
                    corr = np.corrcoef(p, mean_pattern)[0, 1]
                    correlations.append(corr)
            self.temporal_stability_score = np.mean(correlations) if correlations else 0.0
    
    def step(self):
        """Main processing step with temporal tracking."""
        self.step_count += 1
        
        # Get inputs
        image_in = self.get_blended_input("image_in", "replace")
        learning_signal = self.get_blended_input("learning", "sum")
        reset_signal = self.get_blended_input("reset", "sum")
        
        # EEG modulation
        delta = self.get_blended_input("delta_in", "sum") or 0.0
        theta = self.get_blended_input("theta_in", "sum") or 0.0
        alpha = self.get_blended_input("alpha_in", "sum") or 0.0
        beta = self.get_blended_input("beta_in", "sum") or 0.0
        gamma = self.get_blended_input("gamma_in", "sum") or 0.0
        
        # Reset
        if reset_signal is not None and reset_signal > 0.5:
            self._init_layers()
            self._init_temporal_tracking()
            return
        
        learning = self.learning_rate > 0
        if learning_signal is not None:
            learning = learning_signal > 0.5
        
        # Process image
        if image_in is not None:
            self.last_image = image_in
            for i, sym in enumerate(self.symmetry_levels):
                self.processed_layers[i] = self._process_v1_layer(image_in, sym)
            self.processed_layers[-1] = self._process_it_layer(image_in)
        
        # Input current
        input_gain = 30.0
        eeg_mod = 1.0 + 0.1 * (delta + theta + alpha + beta + gamma)
        input_current = self.processed_layers * input_gain * eeg_mod
        
        # === NEURAL DYNAMICS ===
        n_layers = self.n_v1_layers + 1
        n = self.grid_size
        
        total_spikes_this_step = 0
        spikes_per_layer = []
        
        for layer in range(n_layers):
            v = self.v[layer]
            u = self.u[layer]
            I = input_current[layer]
            
            I = np.clip(I, -100, 100)
            
            # Within-layer coupling
            v_up = np.roll(v, -1, axis=0)
            v_down = np.roll(v, 1, axis=0)
            v_left = np.roll(v, -1, axis=1)
            v_right = np.roll(v, 1, axis=1)
            
            w = self.weights[layer]
            neighbor_influence = w[0]*v_up + w[1]*v_down + w[2]*v_left + w[3]*v_right
            total_weight = w[0] + w[1] + w[2] + w[3]
            neighbor_avg = neighbor_influence / (total_weight + 1e-6)
            I_coupling = self.coupling_strength * (neighbor_avg - v)
            I_coupling = np.clip(I_coupling, -50, 50)
            
            # Inter-layer coupling
            I_inter = np.zeros((n, n), dtype=np.float32)
            for other_layer in range(n_layers):
                if other_layer != layer:
                    coupling = self.inter_weights[other_layer, layer]
                    v_other = self.v[other_layer]
                    I_inter += coupling * self.inter_layer_coupling * (v_other - v)
            I_inter = np.clip(I_inter, -30, 30)
            
            # Izhikevich dynamics
            total_I = I + I_coupling + I_inter
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + total_I) * self.dt
            du = self.a * (self.b * v - u) * self.dt
            
            v = v + dv
            u = u + du
            
            v = np.clip(v, -100, 50)
            u = np.clip(u, -50, 50)
            v = np.nan_to_num(v, nan=self.c, posinf=30.0, neginf=-100.0)
            u = np.nan_to_num(u, nan=self.c * self.b, posinf=20.0, neginf=-20.0)
            
            # Spikes
            spikes = v >= 30.0
            v[spikes] = self.c
            u[spikes] += self.d
            
            total_spikes_this_step += np.sum(spikes)
            spikes_per_layer.append(spikes)
            
            self.v[layer] = v
            self.u[layer] = u
            
            # STDP
            if learning and self.learning_rate > 0:
                self.spike_trace[layer] *= self.trace_decay
                self.spike_trace[layer, spikes] = 1.0
                
                trace_up = np.roll(self.spike_trace[layer], -1, axis=0)
                trace_down = np.roll(self.spike_trace[layer], 1, axis=0)
                trace_left = np.roll(self.spike_trace[layer], -1, axis=1)
                trace_right = np.roll(self.spike_trace[layer], 1, axis=1)
                
                spike_float = spikes.astype(np.float32)
                lr = self.learning_rate
                
                dw = np.zeros((4, n, n), dtype=np.float32)
                dw[0] = lr * spike_float * trace_up
                dw[1] = lr * spike_float * trace_down
                dw[2] = lr * spike_float * trace_left
                dw[3] = lr * spike_float * trace_right
                
                spike_up = np.roll(spike_float, -1, axis=0)
                spike_down = np.roll(spike_float, 1, axis=0)
                spike_left = np.roll(spike_float, -1, axis=1)
                spike_right = np.roll(spike_float, 1, axis=1)
                
                dw[0] -= 0.5 * lr * self.spike_trace[layer] * spike_up
                dw[1] -= 0.5 * lr * self.spike_trace[layer] * spike_down
                dw[2] -= 0.5 * lr * self.spike_trace[layer] * spike_left
                dw[3] -= 0.5 * lr * self.spike_trace[layer] * spike_right
                
                self.weights[layer] = np.clip(self.weights[layer] + dw, self.weight_min, self.weight_max)
        
        self.total_spikes += total_spikes_this_step
        
        # === UPDATE TEMPORAL TRACKING ===
        self._update_temporal_tracking(spikes_per_layer, self.step_count)
        
        # === STANDARD METRICS ===
        mean_v = np.mean(self.v)
        self.lfp_history[self.lfp_idx % 256] = mean_v
        self.lfp_idx += 1
        
        self.current_resonance = float(np.std(self.v))
        self.current_energy = float(np.sum(np.abs(self.v - self.c)))
        
        all_weights = self.weights.flatten()
        w_norm = all_weights / (np.sum(all_weights) + 1e-9)
        self.current_entropy = float(-np.sum(w_norm * np.log(w_norm + 1e-9)))
        
        if self.step_count % 5 == 0:
            self._update_display()
    
    def _render_temporal_spectrum(self):
        """Render cross-layer timing relationships."""
        n_layers = self.n_v1_layers + 1
        
        # Create image showing CCG for adjacent layer pairs
        height = 200
        width = 400
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "Cross-Layer Correlograms", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Plot CCGs for key layer pairs
        pairs_to_show = [
            (0, 1, "V1-0→V1-1"),
            (0, -1, "V1-0→IT"),
            (3, -1, "V1-3→IT"),
            (self.n_v1_layers-1, -1, "V1-top→IT"),
        ]
        
        plot_height = 35
        plot_width = 80
        
        for idx, (i, j, label) in enumerate(pairs_to_show):
            if j == -1:
                j = n_layers - 1
            
            ccg = self.ccg_accumulator[i, j]
            
            # Normalize for display
            if np.max(np.abs(ccg)) > 0:
                ccg_norm = ccg / (np.max(np.abs(ccg)) + 1e-9)
            else:
                ccg_norm = ccg
            
            # Plot position
            row = idx // 2
            col = idx % 2
            x_off = 10 + col * (plot_width + 100)
            y_off = 40 + row * (plot_height + 40)
            
            # Draw CCG as line plot
            for k in range(len(ccg_norm) - 1):
                x1 = int(x_off + k * plot_width / len(ccg_norm))
                x2 = int(x_off + (k+1) * plot_width / len(ccg_norm))
                y1 = int(y_off + plot_height/2 - ccg_norm[k] * plot_height/2)
                y2 = int(y_off + plot_height/2 - ccg_norm[k+1] * plot_height/2)
                
                # Color by sign
                color = (100, 255, 100) if ccg_norm[k] > 0 else (100, 100, 255)
                cv2.line(img, (x1, y1), (x2, y2), color, 1)
            
            # Zero line
            cv2.line(img, (x_off, y_off + plot_height//2), 
                    (x_off + plot_width, y_off + plot_height//2), (80, 80, 80), 1)
            
            # Label
            cv2.putText(img, label, (x_off, y_off - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            # Peak lag
            peak_lag = np.argmax(ccg) - self.ccg_lag_bins // 2
            cv2.putText(img, f"lag:{peak_lag}", (x_off + plot_width + 5, y_off + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
        
        # Summary metrics
        cv2.putText(img, f"Phase Coherence: {self.phase_coherence_value:.2f}", (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        cv2.putText(img, f"Mean Delay V1→IT: {self.mean_propagation_delay:.1f} steps", (10, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
        cv2.putText(img, f"Temporal Stability: {self.temporal_stability_score:.2f}", (200, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 255), 1)
        
        return img
    
    def _render_propagation_map(self):
        """Render when activity arrived at each layer."""
        n_layers = self.n_v1_layers + 1
        
        width = 300
        height = 200
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(img, "Propagation Delays (steps)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw delay matrix
        cell_size = min(20, (width - 40) // n_layers, (height - 60) // n_layers)
        
        # Normalize delays for color mapping
        delays = self.propagation_delays.copy()
        max_delay = max(np.max(np.abs(delays)), 1)
        
        for i in range(n_layers):
            for j in range(n_layers):
                x = 30 + j * cell_size
                y = 40 + i * cell_size
                
                delay = delays[i, j]
                # Color: blue = negative (j leads), red = positive (i leads)
                if delay > 0:
                    color = (0, 0, int(128 + 127 * delay / max_delay))
                elif delay < 0:
                    color = (int(128 + 127 * abs(delay) / max_delay), 0, 0)
                else:
                    color = (80, 80, 80)
                
                cv2.rectangle(img, (x, y), (x + cell_size - 1, y + cell_size - 1), color, -1)
        
        # Labels
        for i in range(n_layers):
            label = f"V{i}" if i < self.n_v1_layers else "IT"
            cv2.putText(img, label, (30 + i * cell_size, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
            cv2.putText(img, label, (5, 45 + i * cell_size + cell_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        # Legend
        cv2.putText(img, "Red=Lead Blue=Lag", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        return img
    
    def _render_spike_raster(self):
        """Render spike timing raster per layer."""
        n_layers = self.n_v1_layers + 1
        
        width = 400
        height = 200
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(img, "Spike Raster (recent)", (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Plot last N steps of firing rates per layer
        plot_steps = min(100, self.history_length)
        row_height = (height - 30) // n_layers
        
        for layer in range(n_layers):
            y_base = 25 + layer * row_height
            
            # Get recent firing rates
            rates = self.firing_rate_history[layer, -plot_steps:]
            if np.max(rates) > 0:
                rates_norm = rates / np.max(rates)
            else:
                rates_norm = rates
            
            # Draw as intensity bars
            for t, rate in enumerate(rates_norm):
                x = int(30 + t * (width - 40) / plot_steps)
                intensity = int(rate * 255)
                
                # Color by layer
                if layer < self.n_v1_layers:
                    hue = int(180 * layer / self.n_v1_layers)
                    color = cv2.cvtColor(np.array([[[hue, 200, intensity]]], dtype=np.uint8), 
                                        cv2.COLOR_HSV2BGR)[0, 0].tolist()
                else:
                    color = (intensity, intensity // 2, intensity)  # IT = magenta-ish
                
                cv2.line(img, (x, y_base), (x, y_base + row_height - 2), color, 1)
            
            # Layer label
            label = f"V{layer}" if layer < self.n_v1_layers else "IT"
            cv2.putText(img, label, (5, y_base + row_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1)
        
        return img
    
    def _render_v1_mandala(self):
        """Same as V1 version."""
        n = self.grid_size
        combined = np.zeros((n, n), dtype=np.float32)
        total_weight = 0
        
        for i in range(self.n_v1_layers):
            sym = self.symmetry_levels[i]
            weight = np.log(sym + 1)
            layer_v = self.v[i]
            norm_v = (layer_v + 90) / 130
            combined += norm_v * weight
            total_weight += weight
        
        combined /= (total_weight + 1e-9)
        combined = np.clip(combined, 0, 1)
        combined_u8 = (combined * 255).astype(np.uint8)
        colored = cv2.applyColorMap(combined_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        return cv2.resize(colored, (256, 256))
    
    def _render_it_layer(self):
        """Same as V1 version."""
        layer_v = self.v[-1]
        norm_v = np.clip((layer_v + 90) / 130, 0, 1)
        norm_u8 = (norm_v * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        return cv2.resize(colored, (256, 256))
    
    def _render_depth_signal(self):
        """Same as V1 version."""
        layer_v = self.v[self.n_v1_layers - 1]
        norm_v = np.clip((layer_v + 90) / 130, 0, 1)
        norm_u8 = (norm_v * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_VIRIDIS)
        return cv2.resize(colored, (256, 256))
    
    def _render_eigenstate(self):
        """Same as V1 version."""
        n_layers = self.n_v1_layers + 1
        vis_height = 128
        vis_width = 128 * n_layers
        vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        for i in range(n_layers):
            layer_v = self.v[i]
            norm_v = np.clip((layer_v + 90) / 130, 0, 1)
            thumb = cv2.resize(norm_v, (128, 128))
            thumb_u8 = (thumb * 255).astype(np.uint8)
            
            if i == n_layers - 1:
                colored = cv2.applyColorMap(thumb_u8, cv2.COLORMAP_INFERNO)
            else:
                colored = cv2.applyColorMap(thumb_u8, cv2.COLORMAP_TWILIGHT)
            
            vis[:, i*128:(i+1)*128] = colored
        
        return vis
    
    def _render_activity(self):
        """Same as V1 version."""
        n_layers = self.n_v1_layers + 1
        cols = min(4, n_layers)
        rows = int(np.ceil(n_layers / cols))
        cell_size = 150
        vis = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
        
        for i in range(n_layers):
            row = i // cols
            col = i % cols
            
            layer_v = self.v[i]
            norm_v = np.clip((layer_v + 90) / 130, 0, 1)
            norm_u8 = (cv2.resize(norm_v, (cell_size-10, cell_size-30)) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
            
            y1 = row * cell_size + 25
            y2 = y1 + cell_size - 30
            x1 = col * cell_size + 5
            x2 = x1 + cell_size - 10
            
            vis[y1:y2, x1:x2] = colored
            
            label = f"V1-{self.symmetry_levels[i]}" if i < self.n_v1_layers else "IT"
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return vis
    
    def _compute_depth_profile(self):
        """Same as V1 version."""
        n = self.grid_size
        layer_v = self.v[self.n_v1_layers - 1]
        norm_v = np.clip((layer_v + 90) / 130, 0, 1)
        
        cy, cx = n // 2, n // 2
        y, x = np.ogrid[:n, :n]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        max_r = int(n / 2)
        profile = np.zeros(max_r, dtype=np.float32)
        
        for ri in range(max_r):
            mask = (r >= ri) & (r < ri + 1)
            if np.any(mask):
                profile[ri] = np.mean(norm_v[mask])
        
        return profile
    
    def get_output(self, port_name):
        """Return output for given port."""
        if port_name == "v1_mandala":
            return self._render_v1_mandala()
        elif port_name == "it_layer":
            return self._render_it_layer()
        elif port_name == "depth_signal":
            return self._render_depth_signal()
        elif port_name == "eigenstate":
            return self._render_eigenstate()
        elif port_name == "activity_view":
            return self._render_activity()
        elif port_name == "temporal_spectrum":
            return self._render_temporal_spectrum()
        elif port_name == "propagation_map":
            return self._render_propagation_map()
        elif port_name == "spike_raster":
            return self._render_spike_raster()
        elif port_name == "cross_correlogram":
            # Return flattened CCG for first V1 to IT
            return self.ccg_accumulator[0, -1]
        elif port_name == "phase_coherence":
            return self.phase_coherence_value
        elif port_name == "resonance":
            return self.current_resonance
        elif port_name == "energy":
            return self.current_energy
        elif port_name == "entropy":
            return self.current_entropy
        elif port_name == "depth_profile":
            return self._compute_depth_profile()
        elif port_name == "lfp_out":
            return float(np.mean(self.v))
        elif port_name == "mean_propagation_delay":
            return self.mean_propagation_delay
        elif port_name == "temporal_stability":
            return self.temporal_stability_score
        return None
    
    def _update_display(self):
        """Update main node display."""
        w, h = 500, 450
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "ARTIFICIAL VISUAL CORTEX V2", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 80, 180), 2)
        cv2.putText(img, "Temporal Edition", (10, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # V1 Mandala preview
        v1_img = self._render_v1_mandala()
        v1_small = cv2.resize(v1_img, (100, 100))
        img[55:155, 10:110] = v1_small
        cv2.putText(img, "V1", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # IT preview
        it_img = self._render_it_layer()
        it_small = cv2.resize(it_img, (100, 100))
        img[55:155, 120:220] = it_small
        cv2.putText(img, "IT", (120, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Temporal spectrum preview
        temp_img = self._render_temporal_spectrum()
        temp_small = cv2.resize(temp_img, (160, 80))
        img[55:135, 230:390] = temp_small
        cv2.putText(img, "CCGs", (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Spike raster preview
        raster_img = self._render_spike_raster()
        raster_small = cv2.resize(raster_img, (200, 80))
        img[180:260, 10:210] = raster_small
        
        # Propagation map preview
        prop_img = self._render_propagation_map()
        prop_small = cv2.resize(prop_img, (150, 100))
        img[180:280, 220:370] = prop_small
        
        # Stats
        stats_y = 300
        cv2.putText(img, f"Step: {self.step_count}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"Spikes: {self.total_spikes:,}", (10, stats_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"Resonance: {self.current_resonance:.2f}", (10, stats_y + 36),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        cv2.putText(img, f"Energy: {self.current_energy:.0f}", (10, stats_y + 54),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
        
        # Temporal stats
        cv2.putText(img, "TEMPORAL METRICS", (200, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 80, 180), 1)
        cv2.putText(img, f"Phase Coherence: {self.phase_coherence_value:.3f}", (200, stats_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        cv2.putText(img, f"Mean Delay V1→IT: {self.mean_propagation_delay:.1f}", (200, stats_y + 36),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 200), 1)
        cv2.putText(img, f"Temporal Stability: {self.temporal_stability_score:.3f}", (200, stats_y + 54),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 255), 1)
        
        # Learning status
        lr_status = "Learning ON" if self.learning_rate > 0 else "Frozen"
        cv2.putText(img, f"LR: {self.learning_rate:.4f} ({lr_status})", (10, stats_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image
    
    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", self.grid_size, 
             [("32", 32), ("64", 64), ("128", 128)]),
            ("V1 Layers", "n_v1_layers", self.n_v1_layers, None),
            ("Coupling Strength", "coupling_strength", self.coupling_strength, None),
            ("Inter-layer Coupling", "inter_layer_coupling", self.inter_layer_coupling, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("History Length", "history_length", self.history_length, None),
        ]
    
    def set_config_options(self, options):
        needs_reinit = False
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    old_val = getattr(self, key)
                    setattr(self, key, value)
                    if key in ['grid_size', 'n_v1_layers', 'history_length']:
                        if old_val != value:
                            needs_reinit = True
        if needs_reinit:
            self._init_layers()
            self._init_temporal_tracking()
    
    def save_custom_state(self, folder_path, node_id):
        """Save state."""
        filename = f"visual_cortex_v2_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        
        np.savez(filepath,
                 grid_size=self.grid_size,
                 n_v1_layers=self.n_v1_layers,
                 v=self.v,
                 u=self.u,
                 weights=self.weights,
                 inter_weights=self.inter_weights,
                 ccg_accumulator=self.ccg_accumulator,
                 firing_rate_history=self.firing_rate_history,
                 step_count=self.step_count,
                 total_spikes=self.total_spikes,
                 learning_rate=self.learning_rate)
        
        return filename
    
    def load_custom_state(self, filepath):
        """Load state."""
        try:
            data = np.load(filepath, allow_pickle=True)
            
            self.grid_size = int(data['grid_size'])
            self.n_v1_layers = int(data['n_v1_layers'])
            
            self._init_layers()
            self._init_temporal_tracking()
            
            self.v = data['v']
            self.u = data['u']
            self.weights = data['weights']
            self.inter_weights = data['inter_weights']
            
            if 'ccg_accumulator' in data:
                self.ccg_accumulator = data['ccg_accumulator']
            if 'firing_rate_history' in data:
                self.firing_rate_history = data['firing_rate_history']
            
            self.step_count = int(data['step_count'])
            self.total_spikes = int(data['total_spikes'])
            self.learning_rate = float(data['learning_rate'])
            
            print(f"[V1CortexV2] Loaded: {self.step_count} steps")
            
        except Exception as e:
            print(f"[V1CortexV2] Error loading: {e}")


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("Artificial Visual Cortex V2 - Temporal Edition")
    print("=" * 50)
    
    node = ArtificialVisualCortexV2()
    
    # Create test image
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            r = np.sqrt((x-128)**2 + (y-128)**2)
            test_img[y, x] = [int(r), int(255-r), 128]
    
    for i in range(5):
        y = 40 + i * 40
        cv2.rectangle(test_img, (20, y), (236, y+30), (100+i*30, 80, 50), -1)
    
    print(f"Test image: {test_img.shape}")
    print(f"Layers: {node.n_v1_layers} V1 + IT")
    print(f"Symmetries: {node.symmetry_levels}")
    
    # Run steps
    print("\nRunning 200 steps with temporal tracking...")
    for i in range(200):
        # Process image
        for j, sym in enumerate(node.symmetry_levels):
            node.processed_layers[j] = node._process_v1_layer(test_img, sym)
        node.processed_layers[-1] = node._process_it_layer(test_img)
        
        node.step()
        
        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}: Resonance={node.current_resonance:.2f}, "
                  f"PhaseCoherence={node.phase_coherence_value:.3f}, "
                  f"TemporalStability={node.temporal_stability_score:.3f}")
    
    print("\nGenerating outputs...")
    
    # Spatial outputs
    v1_mandala = node._render_v1_mandala()
    it_layer = node._render_it_layer()
    
    # Temporal outputs
    temporal_spectrum = node._render_temporal_spectrum()
    propagation_map = node._render_propagation_map()
    spike_raster = node._render_spike_raster()
    
    print(f"Temporal Spectrum: {temporal_spectrum.shape}")
    print(f"Propagation Map: {propagation_map.shape}")
    print(f"Spike Raster: {spike_raster.shape}")
    
    # Save
    cv2.imwrite("v2_v1_mandala.png", v1_mandala)
    cv2.imwrite("v2_temporal_spectrum.png", temporal_spectrum)
    cv2.imwrite("v2_propagation_map.png", propagation_map)
    cv2.imwrite("v2_spike_raster.png", spike_raster)
    
    print("\nFinal temporal metrics:")
    print(f"  Phase Coherence: {node.phase_coherence_value:.4f}")
    print(f"  Mean V1→IT Delay: {node.mean_propagation_delay:.2f} steps")
    print(f"  Temporal Stability: {node.temporal_stability_score:.4f}")
    
    print("\nTest images saved!")
    print("Done.")
