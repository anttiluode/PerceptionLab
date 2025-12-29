"""
Grammar Geometry Node V2 - Fixed Crystal Collapse Issue
=========================================================

Key fixes:
1. Prevents crystal from collapsing to uniform state
2. Adds structural noise to maintain complexity
3. Boosts higher frequencies more aggressively  
4. Uses entropy-based chord mixing
5. Adds time-varying perturbation to prevent static states

The crystal was dying because delta-dominated input (~81%) creates
nearly uniform chord → uniform crystal input → collapse to uniform state.

Author: Built for Antti's consciousness research
"""

import numpy as np
import cv2
from collections import defaultdict, Counter
from pathlib import Path
import os

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

try:
    import mne
    from scipy import signal
    from scipy.fft import fft2, ifft2, fftshift
    from scipy.ndimage import gaussian_filter
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}


class GrammarGeometryNodeV2(BaseNode):
    """
    The unified Grammar → Geometry pipeline.
    V2: Fixed crystal collapse, better dynamics.
    """
    
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Grammar Geometry V2"
    NODE_COLOR = QtGui.QColor(255, 180, 100)  # Slightly different orange
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'external_trigger': 'signal',
        }
        
        self.outputs = {
            # Band powers
            'delta': 'signal',
            'theta': 'signal', 
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            
            # Grammar states
            'fast_state': 'signal',
            'medium_state': 'signal',
            'slow_state': 'signal',
            'markov_order': 'signal',
            
            # Cross-scale metrics
            'nesting': 'signal',
            'constraint': 'signal',
            'coherence': 'signal',
            
            # Holographic outputs
            'interference_image': 'image',
            'crystal_image': 'image',
            'geometry_image': 'image',
            'spectrum_out': 'spectrum',
            
            # Complex spectrum for chaining
            'complex_spectrum': 'complex_spectrum',
        }
        
        # ===== EDF CONFIG =====
        self.edf_file_path = ""
        self.selected_region = "All"
        self._last_path = ""
        self._last_region = ""
        
        # ===== PROCESSING =====
        self.sfreq = 100.0
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
        
        # EEG state
        self.raw = None
        self.band_powers = {band: 0.0 for band in self.bands}
        self.band_powers_log = {band: 0.0 for band in self.bands}
        self.total_power = 1e-12
        
        # ===== THREE TIMESCALES =====
        self.fast_window = 0.1    # 100ms
        self.medium_window = 0.5  # 500ms  
        self.slow_window = 2.0    # 2000ms
        
        self.fast_time = 0.0
        self.medium_time = 0.0
        self.slow_time = 0.0
        
        # State sequences
        self.fast_seq = []
        self.medium_seq = []
        self.slow_seq = []
        
        # Transitions
        self.fast_trans = defaultdict(lambda: defaultdict(int))
        self.medium_trans = defaultdict(lambda: defaultdict(int))
        self.slow_trans = defaultdict(lambda: defaultdict(int))
        
        # Current states
        self.fast_state = 0
        self.medium_state = 0
        self.slow_state = 0
        
        # Clustering
        self.n_states = 8
        self.fast_features = []
        self.medium_features = []
        self.slow_features = []
        self.fast_clusterer = None
        self.medium_clusterer = None
        self.slow_clusterer = None
        self.fast_scaler = None
        self.medium_scaler = None
        self.slow_scaler = None
        self.fast_fitted = False
        self.medium_fitted = False
        self.slow_fitted = False
        
        # ===== CROSS-SCALE METRICS =====
        self.nesting_score = 0.0
        self.constraint_score = 0.0
        self.markov_order = 1
        
        # ===== HOLOGRAPHIC SYSTEM =====
        self.holo_size = 128
        self.interference_field = np.zeros((self.holo_size, self.holo_size), dtype=np.complex128)
        self.complex_spectrum = None
        
        # ===== EIGEN CRYSTAL V2 - More robust =====
        self.crystal_size = 64
        self.crystal_structure = self._init_crystal()
        self.crystal_tension = np.zeros((self.crystal_size, self.crystal_size), dtype=np.float32)
        self.crystal_r_grid = self._make_r_grid(self.crystal_size)
        self.settle_steps = 20
        self.diffusion = 0.25    # Less diffusion = sharper
        self.phase_rate = 0.12   # Faster phase = more dynamics
        self.tension_rate = 0.2  # Higher tension = more responsive
        self.threshold = 0.3     # Lower threshold = more flips
        self.current_coherence = 0.0
        
        # V2: Time counter for perturbation
        self.time_step = 0
        
        # ===== OUTPUT IMAGES =====
        self.interference_image = None
        self.crystal_image = None
        self.geometry_image = None
        self.output_spectrum = np.zeros(64, dtype=np.float32)
        
        # ===== TRACKING =====
        self.samples_processed = 0
        self.analysis_count = 0
        
        if not MNE_AVAILABLE:
            self.node_title = "Grammar Geometry V2 (MNE Required!)"
    
    def _init_crystal(self):
        """Initialize crystal with more structure to prevent collapse."""
        size = self.crystal_size
        structure = np.ones((size, size), dtype=np.complex128)
        
        # Add initial spatial structure (prevents collapse to uniform)
        y, x = np.ogrid[:size, :size]
        center = size // 2
        r = np.sqrt((x - center)**2 + (y - center)**2)
        theta = np.arctan2(y - center, x - center)
        
        # Initial spiral pattern - this seeds structure
        initial_pattern = np.cos(r * 0.3) * np.cos(theta * 3) * 0.3
        structure = structure * np.exp(1j * initial_pattern)
        
        # Add small random perturbation
        noise = np.random.randn(size, size) * 0.1
        structure = structure * np.exp(1j * noise)
        
        return structure
    
    def _make_r_grid(self, size):
        """Create radial grid."""
        y, x = np.ogrid[:size, :size]
        center = size // 2
        return np.sqrt((x - center)**2 + (y - center)**2)
    
    def load_edf(self):
        """Load EDF file."""
        if not MNE_AVAILABLE:
            print("Warning: MNE not available")
            return
        
        if not self.edf_file_path or not os.path.exists(self.edf_file_path):
            return
        
        try:
            self.raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            
            # Filter to region
            if self.selected_region != "All":
                region_chs = EEG_REGIONS.get(self.selected_region, [])
                available = [ch for ch in region_chs if ch in self.raw.ch_names]
                if available:
                    self.raw.pick_channels(available)
            
            # Reset time pointers
            self.fast_time = 0.0
            self.medium_time = 0.0
            self.slow_time = 0.0
            
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            
            duration = self.raw.n_times / self.sfreq
            print(f"GrammarGeometry V2: {len(self.raw.ch_names)} ch, {duration:.1f}s")
            
        except Exception as e:
            print(f"Error loading EDF: {e}")
            self.raw = None
    
    def _get_window_data(self, start_time, window_size):
        """Get EEG data for a time window."""
        if self.raw is None:
            return None, start_time
        
        max_time = self.raw.n_times / self.sfreq
        
        if start_time >= max_time - window_size:
            start_time = 0.0  # Loop
        
        start_samp = int(start_time * self.sfreq)
        end_samp = int((start_time + window_size) * self.sfreq)
        end_samp = min(end_samp, self.raw.n_times)
        
        if end_samp <= start_samp:
            return None, start_time
        
        data = self.raw.get_data(start=start_samp, stop=end_samp)
        new_time = start_time + window_size
        
        return data, new_time
    
    def _extract_features(self, data):
        """Extract band power features."""
        if data is None or data.size == 0:
            return None
        
        # Average across channels
        avg_signal = np.mean(data, axis=0)
        
        if len(avg_signal) < 10:
            return None
        
        # Compute PSD
        nperseg = min(len(avg_signal), int(self.sfreq))
        try:
            freqs, psd = signal.welch(avg_signal, fs=self.sfreq, nperseg=nperseg)
        except:
            return None
        
        # Extract band powers
        features = []
        total = 0.0
        band_vals = {}
        
        for band_name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs < high)
            power = np.mean(psd[mask]) if np.any(mask) else 1e-12
            band_vals[band_name] = power
            total += power
        
        # Store relative powers
        self.total_power = max(total, 1e-12)
        for band_name, power in band_vals.items():
            self.band_powers[band_name] = power / self.total_power
            self.band_powers_log[band_name] = np.log10(power + 1e-12)
            features.append(self.band_powers_log[band_name])
        
        return features
    
    def _fit_clusterer(self, features, name):
        """Fit KMeans clusterer with variance check."""
        if not SKLEARN_AVAILABLE or len(features) < 50:
            return None, None, False
        
        X = np.array(features)
        
        # Check variance
        var = np.var(X, axis=0).mean()
        if var < 1e-6:
            return None, None, False
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_clusters = min(self.n_states, len(X) // 5)
            n_clusters = max(2, n_clusters)
            
            clusterer = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
            clusterer.fit(X_scaled)
            
            print(f"GramGeo V2 {name}: {n_clusters} clusters on {len(X)} samples")
            return clusterer, scaler, True
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return None, None, False
    
    def _get_state(self, features, clusterer, scaler, fitted):
        """Get state from features."""
        if not fitted or clusterer is None:
            # Hash-based fallback
            h = hash(tuple([int(f * 1000) for f in features])) % self.n_states
            return h
        
        try:
            X = np.array([features])
            X_scaled = scaler.transform(X)
            return int(clusterer.predict(X_scaled)[0])
        except:
            return 0
    
    def _update_transitions(self, old_state, new_state, trans_dict):
        """Update transition counts."""
        trans_dict[old_state][new_state] += 1
    
    def _compute_markov_order(self):
        """Detect Markov order from medium sequence."""
        seq = self.medium_seq
        if len(seq) < 100:
            return 1
        
        errors = [0, 0, 0]
        
        for i in range(3, len(seq)):
            order1_pred = self._most_likely_next(seq[i-1:i])
            order2_pred = self._most_likely_next(seq[i-2:i])
            order3_pred = self._most_likely_next(seq[i-3:i])
            
            actual = seq[i]
            if order1_pred != actual: errors[0] += 1
            if order2_pred != actual: errors[1] += 1
            if order3_pred != actual: errors[2] += 1
        
        min_err = min(errors)
        if errors[2] == min_err:
            return 3
        elif errors[1] == min_err:
            return 2
        return 1
    
    def _most_likely_next(self, context):
        """Predict most likely next state."""
        if len(context) == 0:
            return 0
        
        last = context[-1]
        trans = self.medium_trans[last]
        if not trans:
            return last
        return max(trans.keys(), key=lambda k: trans[k])
    
    def _compute_nesting(self):
        """Measure if fast patterns predict slow changes."""
        if len(self.medium_seq) < 20:
            return 0.0
        
        medium_changes = sum(1 for i in range(1, len(self.medium_seq)) 
                            if self.medium_seq[i] != self.medium_seq[i-1])
        
        fast_bigrams = set()
        for i in range(len(self.fast_seq) - 1):
            if self.fast_seq[i] != self.fast_seq[i+1]:
                fast_bigrams.add((self.fast_seq[i], self.fast_seq[i+1]))
        
        if medium_changes > 0:
            return min(1.0, len(fast_bigrams) / (medium_changes * 2 + 1))
        return 0.0
    
    def _compute_constraint(self):
        """Measure if slow state constrains fast transitions."""
        if len(self.slow_seq) < 3 or len(self.fast_seq) < 50:
            return 0.0
        
        slow_to_fast = defaultdict(set)
        ratio = len(self.fast_seq) / max(len(self.slow_seq), 1)
        
        for i, slow in enumerate(self.slow_seq[:-1]):
            fast_start = int(i * ratio)
            fast_end = int((i + 1) * ratio)
            
            for j in range(fast_start, min(fast_end - 1, len(self.fast_seq) - 1)):
                slow_to_fast[slow].add((self.fast_seq[j], self.fast_seq[j+1]))
        
        if len(slow_to_fast) < 2:
            return 0.0
        
        states = list(slow_to_fast.keys())
        distances = []
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                set_i = slow_to_fast[states[i]]
                set_j = slow_to_fast[states[j]]
                if len(set_i | set_j) > 0:
                    distances.append(1 - len(set_i & set_j) / len(set_i | set_j))
        
        return np.mean(distances) if distances else 0.0
    
    # ===== V2: IMPROVED CHORD CREATION =====
    
    def _grammar_to_chord(self):
        """Convert grammar states to holographic chord with BETTER balance."""
        n_harmonics = 7
        chord = np.zeros(n_harmonics, dtype=np.float32)
        
        # Base from grammar states
        fast_norm = self.fast_state / max(self.n_states - 1, 1)
        medium_norm = self.medium_state / max(self.n_states - 1, 1)
        slow_norm = self.slow_state / max(self.n_states - 1, 1)
        
        # V2: More balanced base structure (all harmonics active)
        chord[0] = 0.5 + 0.5 * slow_norm     # delta
        chord[1] = 0.4 + 0.6 * slow_norm     # theta
        chord[2] = 0.4 + 0.6 * medium_norm   # alpha
        chord[3] = 0.3 + 0.5 * medium_norm   # beta low
        chord[4] = 0.3 + 0.4 * medium_norm   # beta high
        chord[5] = 0.3 + 0.5 * fast_norm     # gamma
        chord[6] = 0.3 + 0.4 * fast_norm     # high gamma
        
        # V2: AGGRESSIVE higher frequency boost
        # The problem was delta dominating - we need to counteract this
        bp = self.band_powers
        
        # Use log-transformed relative powers for better balance
        eps = 0.01
        delta_rel = bp.get('delta', 0) + eps
        theta_rel = bp.get('theta', 0) + eps
        alpha_rel = bp.get('alpha', 0) + eps
        beta_rel = bp.get('beta', 0) + eps
        gamma_rel = bp.get('gamma', 0) + eps
        
        # Compute spectral entropy - high entropy = more balanced spectrum
        powers = np.array([delta_rel, theta_rel, alpha_rel, beta_rel, gamma_rel])
        powers_norm = powers / powers.sum()
        spectral_entropy = -np.sum(powers_norm * np.log(powers_norm + 1e-9))
        max_entropy = np.log(5)  # Maximum possible entropy for 5 bands
        entropy_ratio = spectral_entropy / max_entropy  # 0-1
        
        # V2: Inverse weighting - boost WEAK bands more
        # This prevents delta from overwhelming everything
        inverse_weights = 1.0 / (powers_norm + 0.1)
        inverse_weights = inverse_weights / inverse_weights.max()
        
        # Apply with moderate strength
        chord[0] *= 0.5 + inverse_weights[0] * 0.5 * delta_rel * 10
        chord[1] *= 0.5 + inverse_weights[1] * 0.5 * theta_rel * 10
        chord[2] *= 0.5 + inverse_weights[2] * 0.5 * alpha_rel * 10
        chord[3] *= 0.5 + inverse_weights[3] * 0.5 * beta_rel * 8
        chord[4] *= 0.5 + inverse_weights[3] * 0.4 * beta_rel * 8
        chord[5] *= 0.5 + inverse_weights[4] * 0.6 * gamma_rel * 15  # Extra gamma boost
        chord[6] *= 0.5 + inverse_weights[4] * 0.5 * gamma_rel * 12
        
        # Add entropy modulation - high entropy = more balanced chord
        chord *= (0.7 + 0.3 * entropy_ratio)
        
        # Markov order boost
        if self.markov_order >= 2:
            chord[2:5] *= 1.1
        if self.markov_order >= 3:
            chord[0:2] *= 1.15
        
        # V2: Add time-varying component to prevent static states
        t = self.time_step * 0.05
        time_mod = 0.9 + 0.1 * np.sin(t + np.arange(7) * 0.7)
        chord *= time_mod
        
        # Normalize but preserve ratios
        max_val = chord.max()
        if max_val > 0:
            chord = chord / max_val
        
        # V2: Ensure minimum values - NO harmonic should be zero
        chord = np.maximum(chord, 0.15)
        
        # Re-normalize
        chord = chord / chord.max()
        
        return chord
    
    def _chord_to_interference(self, chord):
        """Generate 2D interference pattern from chord."""
        size = self.holo_size
        center = size // 2
        
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        theta = np.arctan2(y - center, x - center)
        
        field = np.zeros((size, size), dtype=np.complex128)
        
        # V2: Time-varying phase for more dynamics
        t = self.time_step * 0.03
        
        for i, intensity in enumerate(chord):
            if intensity < 0.01:
                continue
            
            freq = (i + 1) * 2.0
            phase_offset = t * (i + 1) * 0.5
            
            ring = np.exp(1j * (freq * r / center * np.pi + i * theta + phase_offset))
            ring *= intensity
            
            field += ring
        
        # Interference between harmonics
        for i in range(len(chord) - 1):
            for j in range(i + 1, len(chord)):
                if chord[i] > 0.1 and chord[j] > 0.1:
                    beat_freq = abs(j - i) * 1.5
                    beat = np.cos(beat_freq * r / center * np.pi + t)
                    field += beat * chord[i] * chord[j] * 0.5
        
        self.interference_field = field
        self.complex_spectrum = fft2(field)
        
        return field
    
    def _project_chord_to_rings(self, chord):
        """Project chord to radial ring pattern for crystal."""
        size = self.crystal_size
        center = size // 2
        r_grid = self.crystal_r_grid
        
        ring_width = center / len(chord)
        pattern = np.zeros((size, size), dtype=np.float32)
        
        for i, intensity in enumerate(chord):
            inner = i * ring_width
            outer = (i + 1) * ring_width
            mask = (r_grid >= inner) & (r_grid < outer)
            pattern[mask] = intensity
        
        # V2: Add angular modulation to prevent radial collapse
        y, x = np.ogrid[:size, :size]
        theta = np.arctan2(y - center, x - center)
        t = self.time_step * 0.02
        angular_mod = 0.8 + 0.2 * np.cos(theta * 4 + t)
        pattern = pattern * angular_mod
        
        return pattern
    
    def _settle_crystal(self, chord):
        """Run crystal dynamics with improved stability."""
        for step in range(self.settle_steps):
            input_2d = self._project_chord_to_rings(chord)
            
            if input_2d.max() > 1e-9:
                input_2d = input_2d / input_2d.max()
            
            # V2: Add noise injection to prevent collapse
            noise = np.random.randn(self.crystal_size, self.crystal_size) * 0.02
            input_2d = input_2d + np.abs(noise)
            
            # Compute eigenmode
            eigen = np.abs(fftshift(fft2(self.crystal_structure)))
            eigen_norm = eigen / (eigen.max() + 1e-9)
            
            # Resistance
            resistance = input_2d * (1.0 - eigen_norm)
            self.crystal_tension += resistance * self.tension_rate
            
            # V2: Also add tension from uniformity (penalize collapse)
            uniformity = 1.0 - np.std(np.abs(self.crystal_structure))
            self.crystal_tension += uniformity * 0.1
            
            # Critical points flip
            critical = self.crystal_tension > self.threshold
            if np.sum(critical) > 0:
                self.crystal_structure[critical] *= -1
                self.crystal_tension[critical] = 0
                self.crystal_structure = (
                    gaussian_filter(np.real(self.crystal_structure), self.diffusion) +
                    1j * gaussian_filter(np.imag(self.crystal_structure), self.diffusion)
                )
            
            # Phase rotation
            self.crystal_structure *= np.exp(1j * self.phase_rate)
            
            # Normalize magnitude
            mag = np.abs(self.crystal_structure)
            self.crystal_structure[mag > 1.0] /= mag[mag > 1.0]
            
            # V2: Prevent collapse to uniform - inject structure if too uniform
            if np.std(np.abs(self.crystal_structure)) < 0.05:
                # Reset with structure
                self.crystal_structure = self._init_crystal()
        
        # Compute coherence
        phase = np.angle(self.crystal_structure)
        self.current_coherence = float(np.abs(np.mean(np.exp(1j * phase))))
        
        # Get eigenmode image
        eigen = np.abs(fftshift(fft2(self.crystal_structure)))
        eigen_log = np.log(1 + eigen)
        eigen_norm = eigen_log / (eigen_log.max() + 1e-9)
        
        return eigen_norm
    
    def _create_geometry_image(self, interference, crystal):
        """Create combined geometry visualization."""
        size = 256
        
        interf_resized = cv2.resize(np.abs(interference).astype(np.float32), (size, size))
        crystal_resized = cv2.resize(crystal.astype(np.float32), (size, size))
        
        if interf_resized.max() > 0:
            interf_resized = interf_resized / interf_resized.max()
        if crystal_resized.max() > 0:
            crystal_resized = crystal_resized / crystal_resized.max()
        
        combined = interf_resized * 0.4 + crystal_resized * 0.6
        combined = combined / (combined.max() + 1e-9)
        
        combined_u8 = (combined * 255).astype(np.uint8)
        colored = cv2.applyColorMap(combined_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        return colored
    
    def _eigenmode_to_spectrum(self, eigenmode):
        """Convert eigenmode to radial spectrum."""
        size = eigenmode.shape[0]
        center = size // 2
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
        
        r_max = min(center, 64)
        spectrum = np.zeros(r_max, dtype=np.float32)
        
        for i in range(r_max):
            mask = (r == i)
            if np.any(mask):
                spectrum[i] = np.mean(eigenmode[mask])
        
        return spectrum
    
    def _update_holographics(self):
        """Update holographic system with current chord."""
        chord = self._grammar_to_chord()
        
        # Create interference
        interference = self._chord_to_interference(chord)
        
        # Settle crystal
        crystal_eigen = self._settle_crystal(chord)
        
        # Create images
        # Interference
        interf_mag = np.abs(interference)
        if interf_mag.max() > 0:
            interf_mag = interf_mag / interf_mag.max()
        interf_u8 = (interf_mag * 255).astype(np.uint8)
        self.interference_image = cv2.applyColorMap(
            cv2.resize(interf_u8, (256, 256), interpolation=cv2.INTER_CUBIC),
            cv2.COLORMAP_TWILIGHT_SHIFTED
        )
        
        # Crystal
        crystal_u8 = (crystal_eigen * 255).astype(np.uint8)
        self.crystal_image = cv2.applyColorMap(
            cv2.resize(crystal_u8, (256, 256), interpolation=cv2.INTER_CUBIC),
            cv2.COLORMAP_JET
        )
        
        # Combined
        self.geometry_image = self._create_geometry_image(interference, crystal_eigen)
        
        # Output spectrum
        self.output_spectrum = self._eigenmode_to_spectrum(crystal_eigen)
    
    def step(self):
        """Main processing step."""
        
        # Increment time
        self.time_step += 1
        
        # Check for config changes
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()
        
        if self.raw is None:
            return
        
        # ===== PROCESS FAST SCALE (100ms) =====
        data_fast, new_fast_time = self._get_window_data(self.fast_time, self.fast_window)
        if data_fast is not None:
            features = self._extract_features(data_fast)
            if features:
                self.fast_features.append(features)
                
                if not self.fast_fitted and len(self.fast_features) >= 80:
                    self.fast_clusterer, self.fast_scaler, self.fast_fitted = \
                        self._fit_clusterer(self.fast_features, "FAST")
                
                old_state = self.fast_state
                self.fast_state = self._get_state(features, self.fast_clusterer, 
                                                   self.fast_scaler, self.fast_fitted)
                self._update_transitions(old_state, self.fast_state, self.fast_trans)
                self.fast_seq.append(self.fast_state)
                
            self.fast_time = new_fast_time
        
        # ===== PROCESS MEDIUM SCALE (500ms) =====
        data_medium, new_medium_time = self._get_window_data(self.medium_time, self.medium_window)
        if data_medium is not None:
            features = self._extract_features(data_medium)
            if features:
                self.medium_features.append(features)
                
                if not self.medium_fitted and len(self.medium_features) >= 80:
                    self.medium_clusterer, self.medium_scaler, self.medium_fitted = \
                        self._fit_clusterer(self.medium_features, "MEDIUM")
                
                old_state = self.medium_state
                self.medium_state = self._get_state(features, self.medium_clusterer,
                                                     self.medium_scaler, self.medium_fitted)
                self._update_transitions(old_state, self.medium_state, self.medium_trans)
                self.medium_seq.append(self.medium_state)
                
            self.medium_time = new_medium_time
        
        # ===== PROCESS SLOW SCALE (2000ms) =====
        data_slow, new_slow_time = self._get_window_data(self.slow_time, self.slow_window)
        if data_slow is not None:
            features = self._extract_features(data_slow)
            if features:
                self.slow_features.append(features)
                
                if not self.slow_fitted and len(self.slow_features) >= 80:
                    self.slow_clusterer, self.slow_scaler, self.slow_fitted = \
                        self._fit_clusterer(self.slow_features, "SLOW")
                
                old_state = self.slow_state
                self.slow_state = self._get_state(features, self.slow_clusterer,
                                                   self.slow_scaler, self.slow_fitted)
                self._update_transitions(old_state, self.slow_state, self.slow_trans)
                self.slow_seq.append(self.slow_state)
                
            self.slow_time = new_slow_time
        
        self.samples_processed += 1
        
        # Periodic analysis
        if self.samples_processed % 5 == 0:
            self.analysis_count += 1
            self.markov_order = self._compute_markov_order()
            self.nesting_score = self._compute_nesting()
            self.constraint_score = self._compute_constraint()
            self._update_holographics()
    
    def get_output(self, port_name):
        if port_name == 'delta':
            return float(self.band_powers.get('delta', 0))
        elif port_name == 'theta':
            return float(self.band_powers.get('theta', 0))
        elif port_name == 'alpha':
            return float(self.band_powers.get('alpha', 0))
        elif port_name == 'beta':
            return float(self.band_powers.get('beta', 0))
        elif port_name == 'gamma':
            return float(self.band_powers.get('gamma', 0))
        elif port_name == 'fast_state':
            return float(self.fast_state)
        elif port_name == 'medium_state':
            return float(self.medium_state)
        elif port_name == 'slow_state':
            return float(self.slow_state)
        elif port_name == 'markov_order':
            return float(self.markov_order)
        elif port_name == 'nesting':
            return float(self.nesting_score)
        elif port_name == 'constraint':
            return float(self.constraint_score)
        elif port_name == 'coherence':
            return float(self.current_coherence)
        elif port_name == 'interference_image':
            return self.interference_image
        elif port_name == 'crystal_image':
            return self.crystal_image
        elif port_name == 'geometry_image':
            return self.geometry_image
        elif port_name == 'spectrum_out':
            return self.output_spectrum
        elif port_name == 'complex_spectrum':
            return self.complex_spectrum
        return None
    
    def get_display_image(self):
        """Create comprehensive display."""
        
        width, height = 800, 700
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Header
        cv2.putText(img, "=== GRAMMAR GEOMETRY V2 ===", (10, 28), font, 0.7, (255, 180, 100), 2)
        
        if self.edf_file_path:
            fname = os.path.basename(self.edf_file_path)[:20]
            cv2.putText(img, fname, (10, 50), font, 0.35, (150, 150, 150), 1)
        
        cv2.putText(img, f"Samples: {self.samples_processed} | Analysis: #{self.analysis_count}", 
                   (10, 68), font, 0.35, (150, 150, 150), 1)
        
        y = 90
        
        # Grammar section
        cv2.putText(img, "GRAMMAR STATES:", (10, y), font, 0.5, (100, 200, 255), 1)
        y += 25
        
        cv2.putText(img, f"FAST (100ms):   S{self.fast_state}", (20, y), font, 0.4, (255, 150, 150), 1)
        cv2.putText(img, f"Fitted: {self.fast_fitted}", (200, y), font, 0.3, 
                   (100, 255, 100) if self.fast_fitted else (255, 100, 100), 1)
        y += 18
        
        cv2.putText(img, f"MEDIUM (500ms): S{self.medium_state}", (20, y), font, 0.4, (150, 255, 150), 1)
        cv2.putText(img, f"Fitted: {self.medium_fitted}", (200, y), font, 0.3,
                   (100, 255, 100) if self.medium_fitted else (255, 100, 100), 1)
        y += 18
        
        cv2.putText(img, f"SLOW (2s):      S{self.slow_state}", (20, y), font, 0.4, (150, 150, 255), 1)
        cv2.putText(img, f"Fitted: {self.slow_fitted}", (200, y), font, 0.3,
                   (100, 255, 100) if self.slow_fitted else (255, 100, 100), 1)
        y += 25
        
        # Markov order
        order_colors = [(200, 200, 200), (100, 255, 100), (255, 255, 100), (255, 150, 100)]
        cv2.putText(img, f"Markov Order: {self.markov_order}", (20, y), font, 0.45, 
                   order_colors[min(self.markov_order, 3)], 1)
        y += 25
        
        cv2.putText(img, f"Nesting: {self.nesting_score:.1%}", (20, y), font, 0.4, (255, 200, 100), 1)
        cv2.putText(img, f"Constraint: {self.constraint_score:.1%}", (180, y), font, 0.4, (100, 200, 255), 1)
        y += 20
        
        cv2.putText(img, f"Crystal Coherence: {self.current_coherence:.2f}", (20, y), font, 0.4, (200, 100, 255), 1)
        y += 30
        
        cv2.line(img, (0, y), (width, y), (80, 80, 80), 1)
        y += 10
        
        # Images
        img_size = 180
        
        if self.interference_image is not None:
            interf_small = cv2.resize(self.interference_image, (img_size, img_size))
            img[y:y+img_size, 10:10+img_size] = interf_small
            cv2.putText(img, "INTERFERENCE", (15, y+img_size+15), font, 0.35, (255, 100, 255), 1)
        
        if self.crystal_image is not None:
            crystal_small = cv2.resize(self.crystal_image, (img_size, img_size))
            img[y:y+img_size, 200:200+img_size] = crystal_small
            cv2.putText(img, "EIGEN CRYSTAL", (205, y+img_size+15), font, 0.35, (100, 255, 255), 1)
        
        if self.geometry_image is not None:
            geo_small = cv2.resize(self.geometry_image, (img_size, img_size))
            img[y:y+img_size, 390:390+img_size] = geo_small
            cv2.putText(img, "GEOMETRY", (395, y+img_size+15), font, 0.35, (255, 255, 100), 1)
        
        y += img_size + 35
        
        # Chord
        cv2.putText(img, "GRAMMAR CHORD:", (10, y), font, 0.45, (200, 200, 200), 1)
        y += 20
        
        chord = self._grammar_to_chord()
        chord_labels = ['d', 't', 'a', 'bL', 'bH', 'g', 'gH']
        bar_width = 40
        bar_max_h = 60
        
        for i, (val, label) in enumerate(zip(chord, chord_labels)):
            x = 20 + i * (bar_width + 10)
            bar_h = int(val * bar_max_h)
            
            colors = [
                (255, 100, 100), (255, 200, 100), (100, 255, 100),
                (100, 200, 255), (100, 100, 255), (200, 100, 255), (255, 100, 255),
            ]
            
            cv2.rectangle(img, (x, y + bar_max_h - bar_h), (x + bar_width, y + bar_max_h), colors[i], -1)
            cv2.putText(img, label, (x + 10, y + bar_max_h + 15), font, 0.35, colors[i], 1)
        
        y += bar_max_h + 30
        
        # Band powers
        cv2.putText(img, "BAND POWERS (relative):", (10, y), font, 0.45, (200, 200, 200), 1)
        y += 20
        
        band_names = ['d', 't', 'a', 'b', 'g']
        band_keys = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_colors = [(255, 100, 100), (255, 200, 100), (100, 255, 100), (100, 100, 255), (200, 100, 255)]
        
        max_rel = 0.5
        
        for i, (name, key, color) in enumerate(zip(band_names, band_keys, band_colors)):
            x = 20 + i * 60
            rel_power = self.band_powers.get(key, 0)
            bar_h = int(min(rel_power / max_rel * 50, 50))
            
            cv2.rectangle(img, (x, y + 50 - bar_h), (x + 45, y + 50), color, -1)
            cv2.rectangle(img, (x, y), (x + 45, y + 50), (80, 80, 80), 1)
            cv2.putText(img, name, (x + 15, y + 65), font, 0.4, color, 1)
            cv2.putText(img, f"{rel_power:.0%}", (x + 5, y + 78), font, 0.28, (150, 150, 150), 1)
        
        y += 95
        cv2.putText(img, f"Seq: F={len(self.fast_seq)} M={len(self.medium_seq)} S={len(self.slow_seq)}", 
                   (10, y), font, 0.35, (150, 150, 150), 1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, width, height, width*3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("Number of States", "n_states", self.n_states, None),
            ("Settle Steps", "settle_steps", self.settle_steps, None),
            ("Diffusion", "diffusion", self.diffusion, None),
            ("Phase Rate", "phase_rate", self.phase_rate, None),
        ]
    
    def set_config_options(self, options):
        for key, value in options.items():
            if hasattr(self, key):
                if key in ['n_states', 'settle_steps']:
                    setattr(self, key, int(value))
                elif key in ['diffusion', 'phase_rate']:
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)