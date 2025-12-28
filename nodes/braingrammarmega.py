"""
Brain Grammar Mega Node
=======================

ALL-IN-ONE brain grammar analysis.

Combines:
- EDF file loading with region selection
- Band power extraction (delta, theta, alpha, beta, gamma)
- State clustering
- Grammar cracking (attractors, vocabulary, syntax, forbidden)
- Transfer entropy and topology
- Prediction and surprise metrics
- Big comprehensive visualization

No inter-node communication needed. Just load an EDF and see the grammar.

Author: The unified brain grammar tool for Antti
"""

import numpy as np
import cv2
from collections import defaultdict, Counter
from pathlib import Path
import os

# --- CRITICAL IMPORT BLOCK (PyQt6 style) ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -------------------------------------------

# Try to import MNE for EEG loading
try:
    import mne
    from scipy import signal
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available - install with: pip install mne")

# Try sklearn for clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available - using simple clustering")


# Brain regions
EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}


class BrainGrammarMegaNode(BaseNode):
    """
    The all-in-one brain grammar analyzer.
    
    Load EDF → Extract bands → Cluster states → Crack grammar → Predict → Visualize
    
    Everything in one node. No communication issues.
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Brain Grammar MEGA"
    NODE_COLOR = QtGui.QColor(255, 150, 50)  # Orange - mega power
    
    def __init__(self):
        super().__init__()
        
        # No required inputs - originator node
        self.inputs = {
            'external_trigger': 'signal',  # Optional: trigger analysis
        }
        
        self.outputs = {
            # Band powers
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            'latent_out': 'spectrum',
            # State
            'current_state': 'signal',
            'state_sequence': 'spectrum',
            # Grammar metrics
            'forbidden_count': 'signal',
            'transfer_entropy': 'signal',
            'n_cycles': 'signal',
            'syntax_strength': 'signal',
            # Prediction
            'predicted_state': 'signal',
            'surprise': 'signal',
            'conformity': 'signal',
        }
        
        # ===== EDF CONFIGURATION =====
        self.edf_file_path = ""
        self.selected_region = "All"
        self.base_scale = 1.0
        self._last_path = ""
        self._last_region = ""
        
        # ===== PROCESSING CONFIG =====
        self.n_states = 15
        self.window_size = 0.5  # seconds
        self.sfreq = 100.0
        
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
        
        # ===== EEG STATE =====
        self.raw = None
        self.current_time = 0.0
        self.band_powers = {band: 0.0 for band in self.bands}
        self.latent_vector = np.zeros(6, dtype=np.float32)
        
        # ===== CLUSTERING =====
        self.clusterer = None
        self.scaler = None
        self.is_fitted = False
        self.features_buffer = []
        
        # ===== STATE TRACKING =====
        self.state_sequence = []
        self.max_sequence = 50000
        self.current_state = 0
        self.last_state = None
        
        # ===== GRAMMAR ANALYSIS =====
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.states_observed = []
        self.n_states_observed = 0
        
        # Cracked grammar
        self.attractors = {}
        self.escape_routes = {}
        self.words = []
        self.syntax_rules = []
        self.forbidden = set()
        self.transfer_entropy_val = 0.0
        self.n_cycles = 0
        
        # ===== PREDICTION =====
        self.predicted_next = 0
        self.last_surprise = 0.0
        self.last_conformity = 1.0
        self.prediction_correct = 0
        self.prediction_total = 0
        
        # ===== DISPLAY =====
        self.samples_processed = 0
        self.analysis_count = 0
        
        if not MNE_AVAILABLE:
            self.node_title = "Brain Grammar MEGA (MNE Required!)"
    
    # ==================== EDF LOADING ====================
    
    def load_edf(self):
        """Load EDF file with region selection."""
        
        if not MNE_AVAILABLE:
            self.raw = None
            return False
        
        if not os.path.exists(self.edf_file_path):
            self.raw = None
            return False
        
        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            # Region selection
            if self.selected_region != "All":
                region_channels = EEG_REGIONS[self.selected_region]
                available = [ch for ch in region_channels if ch in raw.ch_names]
                if available:
                    raw.pick_channels(available)
            
            raw.resample(self.sfreq, verbose=False)
            self.raw = raw
            self.current_time = 0.0
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            
            # Reset everything
            self._reset_analysis()
            
            fname = os.path.basename(self.edf_file_path)[:20]
            self.node_title = f"Grammar MEGA ({fname})"
            return True
            
        except Exception as e:
            print(f"Error loading EDF: {e}")
            self.raw = None
            return False
    
    def _reset_analysis(self):
        """Reset all analysis state."""
        self.state_sequence = []
        self.features_buffer = []
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.is_fitted = False
        self.attractors = {}
        self.escape_routes = {}
        self.words = []
        self.syntax_rules = []
        self.forbidden = set()
        self.transfer_entropy_val = 0.0
        self.n_cycles = 0
        self.samples_processed = 0
        self.analysis_count = 0
        self.prediction_correct = 0
        self.prediction_total = 0
    
    # ==================== BAND EXTRACTION ====================
    
    def _extract_bands(self, data):
        """Extract band powers from EEG data."""
        
        if data.size == 0:
            return None
        
        nyq = self.sfreq / 2.0
        features = []
        
        for band_name, (low, high) in self.bands.items():
            try:
                # Bandpass filter
                low_n = max(low / nyq, 0.01)
                high_n = min(high / nyq, 0.99)
                
                if low_n >= high_n:
                    power = 0.0
                else:
                    b, a = signal.butter(4, [low_n, high_n], btype='band')
                    filtered = signal.filtfilt(b, a, data)
                    power = float(np.mean(filtered ** 2))
                
                self.band_powers[band_name] = power * self.base_scale
                features.append(power)
                
            except Exception:
                self.band_powers[band_name] = 0.0
                features.append(0.0)
        
        # Add raw power
        raw_power = float(np.mean(data ** 2))
        features.append(raw_power)
        
        self.latent_vector = np.array(features, dtype=np.float32)
        return features
    
    # ==================== STATE CLUSTERING ====================
    
    def _cluster_state(self, features):
        """Assign features to a state cluster."""
        
        if features is None:
            return self.current_state
        
        self.features_buffer.append(features)
        
        # Fit clusterer when we have enough samples
        if not self.is_fitted and len(self.features_buffer) >= 200:
            self._fit_clusterer()
        
        if not self.is_fitted:
            # Simple binning before clustering is ready
            total = sum(features)
            return int(total * 1000) % self.n_states
        
        # Predict cluster
        try:
            feat_scaled = self.scaler.transform([features])
            state = int(self.clusterer.predict(feat_scaled)[0])
            return state
        except Exception:
            return self.current_state
    
    def _fit_clusterer(self):
        """Fit the clusterer on accumulated features."""
        
        if not SKLEARN_AVAILABLE:
            self.is_fitted = True
            return
        
        try:
            X = np.array(self.features_buffer[-1000:])  # Use recent samples
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.clusterer = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            self.clusterer.fit(X_scaled)
            self.is_fitted = True
            
        except Exception as e:
            print(f"Clustering error: {e}")
            self.is_fitted = True  # Proceed anyway
    
    # ==================== GRAMMAR ANALYSIS ====================
    
    def _update_transitions(self, new_state):
        """Update transition counts and state sequence."""
        
        if self.last_state is not None:
            self.transition_counts[self.last_state][new_state] += 1
        
        self.state_sequence.append(new_state)
        if len(self.state_sequence) > self.max_sequence:
            self.state_sequence = self.state_sequence[-self.max_sequence:]
        
        self.last_state = new_state
        self.current_state = new_state
    
    def _analyze_grammar(self):
        """Full grammar analysis."""
        
        if len(self.state_sequence) < 100:
            return
        
        self.states_observed = sorted(set(self.state_sequence))
        self.n_states_observed = len(self.states_observed)
        
        if self.n_states_observed < 2:
            return
        
        # Find attractors
        self._find_attractors()
        
        # Find escape routes
        self._find_escape_routes()
        
        # Find vocabulary (common sequences)
        self._find_words()
        
        # Find syntax rules
        self._find_syntax()
        
        # Find forbidden transitions
        self._find_forbidden()
        
        # Compute transfer entropy
        self._compute_transfer_entropy()
        
        # Compute topology
        self._compute_topology()
        
        self.analysis_count += 1
    
    def _find_attractors(self):
        """Find attractor states (high self-loop)."""
        
        self.attractors = {}
        
        for s in self.states_observed:
            self_loops = self.transition_counts[s][s]
            total_out = sum(self.transition_counts[s].values())
            
            if total_out > 0:
                self_ratio = self_loops / total_out
                if self_ratio > 0.3 or self_loops > 500:
                    self.attractors[s] = {
                        'self_loops': self_loops,
                        'self_ratio': self_ratio,
                        'total_out': total_out,
                        'strength': self_loops * self_ratio
                    }
        
        # Sort by strength
        self.attractors = dict(sorted(
            self.attractors.items(),
            key=lambda x: -x[1]['strength']
        ))
    
    def _find_escape_routes(self):
        """Find escape routes from attractors."""
        
        self.escape_routes = {}
        
        for attractor in self.attractors:
            routes = []
            total = sum(self.transition_counts[attractor].values())
            
            for next_s, count in self.transition_counts[attractor].items():
                if next_s != attractor and count > 0:
                    prob = count / total
                    routes.append({
                        'to': next_s,
                        'count': count,
                        'prob': prob,
                    })
            
            routes.sort(key=lambda x: -x['prob'])
            self.escape_routes[attractor] = routes
    
    def _find_words(self):
        """Find common sequences (vocabulary)."""
        
        seq = self.state_sequence
        
        # Bigrams (excluding self-loops)
        bigrams = Counter()
        for i in range(len(seq) - 1):
            if seq[i] != seq[i+1]:
                bigrams[tuple(seq[i:i+2])] += 1
        
        # Trigrams
        trigrams = Counter()
        for i in range(len(seq) - 2):
            if len(set(seq[i:i+3])) > 1:
                trigrams[tuple(seq[i:i+3])] += 1
        
        # 4-grams
        fourgrams = Counter()
        for i in range(len(seq) - 3):
            if len(set(seq[i:i+4])) > 1:
                fourgrams[tuple(seq[i:i+4])] += 1
        
        self.words = []
        
        for bg, count in bigrams.most_common(10):
            self.words.append({'seq': bg, 'count': count, 'len': 2})
        
        for tg, count in trigrams.most_common(8):
            self.words.append({'seq': tg, 'count': count, 'len': 3})
        
        for fg, count in fourgrams.most_common(5):
            self.words.append({'seq': fg, 'count': count, 'len': 4})
    
    def _find_syntax(self):
        """Find deterministic syntax rules."""
        
        self.syntax_rules = []
        
        for s1 in self.states_observed:
            total = sum(self.transition_counts[s1].values())
            if total < 10:
                continue
            
            for s2, count in self.transition_counts[s1].items():
                prob = count / total
                if prob > 0.7:  # Deterministic threshold
                    self.syntax_rules.append({
                        'from': s1,
                        'to': s2,
                        'prob': prob,
                        'count': count,
                    })
        
        self.syntax_rules.sort(key=lambda x: -x['prob'])
    
    def _find_forbidden(self):
        """Find forbidden transitions."""
        
        self.forbidden = set()
        
        for s1 in self.states_observed:
            total = sum(self.transition_counts[s1].values())
            if total < 20:  # Need enough observations
                continue
            
            for s2 in self.states_observed:
                if self.transition_counts[s1][s2] == 0:
                    self.forbidden.add((s1, s2))
    
    def _compute_transfer_entropy(self):
        """Compute transfer entropy."""
        
        seq = self.state_sequence
        if len(seq) < 100:
            self.transfer_entropy_val = 0.0
            return
        
        # Simple transfer entropy at lag 1
        present_future = defaultdict(lambda: defaultdict(int))
        past_present_future = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(seq) - 1):
            past = seq[i-1]
            present = seq[i]
            future = seq[i+1]
            
            present_future[present][future] += 1
            past_present_future[(past, present)][future] += 1
        
        def cond_entropy(counts):
            H = 0.0
            total = sum(sum(fc.values()) for fc in counts.values())
            if total == 0:
                return 0.0
            for cond, fc in counts.items():
                t = sum(fc.values())
                for c in fc.values():
                    if c > 0:
                        p = c / t
                        H -= (t / total) * p * np.log2(p + 1e-10)
            return H
        
        H1 = cond_entropy(present_future)
        H2 = cond_entropy(past_present_future)
        
        self.transfer_entropy_val = max(0.0, H1 - H2)
    
    def _compute_topology(self):
        """Compute topological properties (Betti-1 cycles)."""
        
        states = self.states_observed
        n = len(states)
        
        if n == 0:
            self.n_cycles = 0
            return
        
        # Build adjacency
        adj = np.zeros((n, n))
        idx = {s: i for i, s in enumerate(states)}
        
        for s1 in states:
            for s2, c in self.transition_counts[s1].items():
                if c > 0 and s2 in idx:
                    adj[idx[s1], idx[s2]] = 1
        
        adj_sym = ((adj + adj.T) > 0).astype(float)
        n_edges = int(np.sum(adj_sym) / 2)
        
        # Connected components
        visited = set()
        n_comp = 0
        for i in range(n):
            if i not in visited:
                n_comp += 1
                queue = [i]
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    for j in np.where(adj_sym[curr] > 0)[0]:
                        if j not in visited:
                            queue.append(int(j))
        
        # Betti-1 estimate
        self.n_cycles = max(0, n_edges - n + n_comp)
    
    # ==================== PREDICTION ====================
    
    def _predict_next(self):
        """Predict next state and compute surprise."""
        
        if self.current_state not in self.transition_counts:
            self.predicted_next = self.current_state
            return
        
        probs = {}
        total = sum(self.transition_counts[self.current_state].values())
        
        if total == 0:
            self.predicted_next = self.current_state
            return
        
        for s, c in self.transition_counts[self.current_state].items():
            probs[s] = c / total
        
        # Most likely next
        self.predicted_next = max(probs.keys(), key=lambda x: probs[x])
    
    def _compute_surprise(self, actual_state):
        """Compute surprise for actual transition."""
        
        if self.last_state is None:
            return
        
        total = sum(self.transition_counts[self.last_state].values())
        if total == 0:
            return
        
        prob = self.transition_counts[self.last_state].get(actual_state, 0) / total
        
        if prob > 0:
            self.last_surprise = -np.log2(prob + 1e-10)
        else:
            self.last_surprise = 10.0  # Forbidden transition!
        
        self.last_conformity = prob
        
        # Track prediction accuracy
        if self.predicted_next == actual_state:
            self.prediction_correct += 1
        self.prediction_total += 1
    
    # ==================== MAIN STEP ====================
    
    def step(self):
        """Main processing step."""
        
        # Check for config changes
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()
        
        if self.raw is None:
            return
        
        # Get current window
        start_sample = int(self.current_time * self.sfreq)
        end_sample = start_sample + int(self.window_size * self.sfreq)
        
        if end_sample >= self.raw.n_times:
            self.current_time = 0.0  # Loop
            start_sample = 0
            end_sample = int(self.window_size * self.sfreq)
        
        data, _ = self.raw[:, start_sample:end_sample]
        
        if data.ndim > 1:
            data = np.mean(data, axis=0)
        
        if data.size == 0:
            return
        
        # Extract bands
        features = self._extract_bands(data)
        
        # Cluster to state
        new_state = self._cluster_state(features)
        
        # Compute surprise BEFORE updating
        self._compute_surprise(new_state)
        
        # Update transitions
        self._update_transitions(new_state)
        
        # Predict next
        self._predict_next()
        
        # Periodic analysis
        self.samples_processed += 1
        if self.samples_processed % 200 == 0:
            self._analyze_grammar()
        
        # Advance time
        self.current_time += self.window_size
    
    # ==================== OUTPUTS ====================
    
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
        elif port_name == 'latent_out':
            return self.latent_vector.copy()
        elif port_name == 'current_state':
            return float(self.current_state)
        elif port_name == 'state_sequence':
            if len(self.state_sequence) > 0:
                return np.array(self.state_sequence[-100:], dtype=np.float32)
            return np.zeros(1, dtype=np.float32)
        elif port_name == 'forbidden_count':
            return float(len(self.forbidden))
        elif port_name == 'transfer_entropy':
            return float(self.transfer_entropy_val)
        elif port_name == 'n_cycles':
            return float(self.n_cycles)
        elif port_name == 'syntax_strength':
            if self.syntax_rules:
                return float(np.mean([r['prob'] for r in self.syntax_rules[:5]]))
            return 0.0
        elif port_name == 'predicted_state':
            return float(self.predicted_next)
        elif port_name == 'surprise':
            return float(self.last_surprise)
        elif port_name == 'conformity':
            return float(self.last_conformity)
        return None
    
    # ==================== DISPLAY ====================
    
    def get_display_image(self):
        """Create the mega display."""
        
        width, height = 700, 800
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # ===== HEADER =====
        cv2.putText(img, "=== BRAIN GRAMMAR MEGA ===", (10, 28), font, 0.65, (255, 150, 50), 2)
        
        if self.edf_file_path:
            fname = os.path.basename(self.edf_file_path)[:30]
            cv2.putText(img, fname, (10, 50), font, 0.35, (150, 150, 150), 1)
        
        cv2.putText(img, f"Region: {self.selected_region} | Samples: {self.samples_processed}", 
                   (10, 68), font, 0.35, (150, 150, 150), 1)
        
        # Divider
        cv2.line(img, (0, 78), (width, 78), (80, 80, 80), 1)
        
        y = 100
        
        # ===== ATTRACTORS =====
        cv2.putText(img, "ATTRACTORS (where mind stays):", (10, y), font, 0.5, (255, 200, 0), 1)
        y += 22
        
        for state, data in list(self.attractors.items())[:5]:
            bar_len = int(min(data['self_ratio'] * 200, 200))
            cv2.rectangle(img, (10, y-12), (10 + bar_len, y+2), (255, 200, 0), -1)
            cv2.putText(img, f"State {state}: {data['self_loops']}x ({data['self_ratio']:.0%})", 
                       (220, y), font, 0.35, (255, 255, 255), 1)
            y += 18
        
        y += 15
        
        # ===== ESCAPE ROUTES =====
        cv2.putText(img, "ESCAPE ROUTES (how to leave attractors):", (10, y), font, 0.5, (0, 255, 200), 1)
        y += 20
        
        for attractor, routes in list(self.escape_routes.items())[:3]:
            cv2.putText(img, f"From {attractor}:", (15, y), font, 0.35, (0, 255, 200), 1)
            y += 15
            for route in routes[:3]:
                cv2.putText(img, f"  -> {route['to']} ({route['prob']:.1%}, {route['count']}x)", 
                           (25, y), font, 0.3, (200, 200, 200), 1)
                y += 13
            y += 5
        
        y += 10
        
        # ===== VOCABULARY =====
        cv2.putText(img, "VOCABULARY (common sequences):", (10, y), font, 0.5, (200, 100, 255), 1)
        y += 20
        
        for word in self.words[:8]:
            seq_str = '->'.join(map(str, word['seq']))
            cv2.putText(img, f"{seq_str}: {word['count']}x", (15, y), font, 0.32, (200, 100, 255), 1)
            y += 14
        
        y += 10
        
        # ===== SYNTAX =====
        cv2.putText(img, "SYNTAX (deterministic rules):", (10, y), font, 0.5, (100, 255, 100), 1)
        y += 20
        
        for rule in self.syntax_rules[:4]:
            cv2.putText(img, f"IF {rule['from']} THEN {rule['to']} ({rule['prob']:.0%})", 
                       (15, y), font, 0.32, (100, 255, 100), 1)
            y += 14
        
        y += 10
        
        # ===== FORBIDDEN =====
        cv2.putText(img, "FORBIDDEN (never happens):", (10, y), font, 0.5, (100, 100, 255), 1)
        y += 18
        
        forbidden_list = list(self.forbidden)[:8]
        forbidden_str = "  ".join([f"{s1}-X->{s2}" for s1, s2 in forbidden_list])
        cv2.putText(img, forbidden_str[:80], (15, y), font, 0.28, (100, 100, 255), 1)
        
        y += 25
        
        # Divider
        cv2.line(img, (0, y), (width, y), (80, 80, 80), 1)
        y += 15
        
        # ===== RIGHT COLUMN - METRICS =====
        col2_x = 380
        col2_y = 100
        
        cv2.putText(img, "GRAMMAR METRICS:", (col2_x, col2_y), font, 0.5, (255, 200, 100), 1)
        col2_y += 25
        
        cv2.putText(img, f"States: {self.n_states_observed}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        col2_y += 18
        cv2.putText(img, f"Forbidden: {len(self.forbidden)}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        col2_y += 18
        cv2.putText(img, f"Transfer Entropy: {self.transfer_entropy_val:.3f} bits", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        col2_y += 18
        cv2.putText(img, f"Cycles (β₁): {self.n_cycles}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        col2_y += 18
        cv2.putText(img, f"Attractors: {len(self.attractors)}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        col2_y += 18
        cv2.putText(img, f"Words: {len(self.words)}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        col2_y += 18
        cv2.putText(img, f"Rules: {len(self.syntax_rules)}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        
        col2_y += 25
        
        # Prediction metrics
        cv2.putText(img, "PREDICTION:", (col2_x, col2_y), font, 0.5, (255, 200, 100), 1)
        col2_y += 25
        
        cv2.putText(img, f"Current: {self.current_state}", (col2_x, col2_y), font, 0.4, (100, 255, 100), 1)
        col2_y += 18
        cv2.putText(img, f"Predicted: {self.predicted_next}", (col2_x, col2_y), font, 0.4, (255, 200, 100), 1)
        col2_y += 18
        cv2.putText(img, f"Surprise: {self.last_surprise:.2f} bits", (col2_x, col2_y), font, 0.4, (255, 100, 100), 1)
        col2_y += 18
        cv2.putText(img, f"Conformity: {self.last_conformity:.1%}", (col2_x, col2_y), font, 0.4, (100, 255, 100), 1)
        col2_y += 18
        
        if self.prediction_total > 0:
            acc = self.prediction_correct / self.prediction_total
            cv2.putText(img, f"Accuracy: {acc:.1%}", (col2_x, col2_y), font, 0.4, (200, 200, 200), 1)
        
        # ===== CURRENT STATE & TRAJECTORY =====
        cv2.putText(img, f"STATE: {self.current_state}", (10, y), font, 0.6, (0, 255, 255), 1)
        y += 25
        
        if self.state_sequence:
            recent = self.state_sequence[-15:]
            traj = '->'.join(map(str, recent))
            cv2.putText(img, f"...{traj}", (10, y), font, 0.3, (180, 180, 180), 1)
        
        y += 30
        
        # ===== MINI TRANSITION MATRIX =====
        if self.n_states_observed > 0:
            cv2.putText(img, "TRANSITION MATRIX:", (10, y), font, 0.4, (200, 200, 200), 1)
            y += 5
            
            # Create mini heatmap
            n = min(self.n_states_observed, 15)
            states = self.states_observed[:n]
            
            matrix = np.zeros((n, n), dtype=np.float32)
            for i, s1 in enumerate(states):
                total = sum(self.transition_counts[s1].values())
                if total > 0:
                    for j, s2 in enumerate(states):
                        matrix[i, j] = self.transition_counts[s1][s2] / total
            
            # Scale to image
            mat_size = 150
            mat_img = (matrix * 255).astype(np.uint8)
            mat_img = cv2.applyColorMap(mat_img, cv2.COLORMAP_HOT)
            mat_img = cv2.resize(mat_img, (mat_size, mat_size), interpolation=cv2.INTER_NEAREST)
            
            # Place in display
            mat_y = y + 5
            mat_x = 10
            img[mat_y:mat_y+mat_size, mat_x:mat_x+mat_size] = mat_img
        
        # ===== BAND POWERS BAR =====
        band_y = height - 60
        cv2.putText(img, "BANDS:", (10, band_y), font, 0.35, (150, 150, 150), 1)
        
        band_x = 70
        band_w = 50
        band_names = ['d', 't', 'a', 'b', 'g']  # delta, theta, alpha, beta, gamma
        band_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
        
        max_power = max(self.band_powers.values()) if self.band_powers else 1
        if max_power < 1e-12:
            max_power = 1
        
        for i, (name, bname) in enumerate(zip(band_names, self.bands.keys())):
            x = band_x + i * (band_w + 10)
            power = self.band_powers.get(bname, 0)
            bar_h = int(min(power / max_power * 30, 30))
            
            cv2.rectangle(img, (x, band_y - bar_h), (x + band_w, band_y), band_colors[i], -1)
            cv2.putText(img, name, (x + 18, band_y + 15), font, 0.4, band_colors[i], 1)
        
        # Analysis count
        cv2.putText(img, f"Analysis #{self.analysis_count}", (width - 120, height - 10), 
                   font, 0.3, (100, 100, 100), 1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, width, height, width*3, QtGui.QImage.Format.Format_RGB888)
    
    # ==================== CONFIG ====================
    
    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("Number of States", "n_states", self.n_states, None),
            ("Base Scale", "base_scale", self.base_scale, None),
            ("Window Size (s)", "window_size", self.window_size, None),
        ]
    
    def set_config_options(self, options):
        for key, value in options.items():
            if hasattr(self, key):
                if key == 'n_states':
                    new_n = int(value)
                    if new_n != self.n_states:
                        self.n_states = new_n
                        self.is_fitted = False
                elif key in ['base_scale', 'window_size']:
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)