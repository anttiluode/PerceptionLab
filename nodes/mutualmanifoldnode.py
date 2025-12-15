"""
Mutual Information Manifold - The Information Content of Observation
======================================================================
This node measures HOW MUCH the EEG actually tells us.

The core insight: If we compare a manifold constrained by EEG observations
to one running on pure priors (no input), the DIFFERENCE is the information
content of the signal.

I(EEG; Hidden State) = H(prior) - H(posterior|EEG)

Where:
- H(prior) = entropy of unconstrained manifold (what we'd believe without data)
- H(posterior|EEG) = entropy of constrained manifold (what we believe with data)
- The difference = bits of information gained from observation

This node:
1. Maintains TWO parallel manifolds - one constrained, one free
2. Computes entropy of each continuously
3. Reports mutual information in BITS
4. Visualizes WHERE information concentrates (which dimensions)
5. Tracks information flow over time

When MI is high: The EEG is telling us something specific
When MI is low: The EEG is ambiguous, many states compatible
When MI oscillates: The brain is switching between determinate/indeterminate states

INPUTS:
- token_stream: EEG tokens to constrain the posterior manifold
- theta_phase: Phase alignment
- temperature: Constraint softness (affects both manifolds equally for fair comparison)

OUTPUTS:
- display: Full visualization
- mutual_information: Bits of information (scalar signal)
- information_map: Per-dimension information content
- prior_entropy: H(unconstrained)
- posterior_entropy: H(constrained|EEG)
- information_rate: dI/dt - how fast we're learning
- surprise_signal: How unexpected was this observation?

The philosophical point: We're not measuring what the brain IS doing.
We're measuring how much LESS uncertain we are because we observed it.
"""

import numpy as np
import cv2
from collections import deque
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy as scipy_entropy

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


class MutualInformationManifold(BaseNode):
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Mutual Information"
    NODE_COLOR = QtGui.QColor(255, 150, 50)  # Orange - information color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Mutual Information Manifold"
        
        self.inputs = {
            'token_stream': 'spectrum',
            'theta_phase': 'signal',
            'temperature': 'signal',
            'structural_prior': 'spectrum',  # NEW: Raj modes from ConnectomePriorNode
        }
        
        self.outputs = {
            'display': 'image',
            'mutual_information': 'signal',
            'information_map': 'spectrum',
            'prior_entropy': 'signal',
            'posterior_entropy': 'signal',
            'information_rate': 'signal',
            'surprise_signal': 'signal',
            'structure_deviation': 'signal',      # NEW: How much EEG deviates from structure
            'riverbed_alignment': 'signal',       # NEW: How aligned is EEG with Raj modes
            'deviation_spectrum': 'spectrum',     # NEW: Per-mode deviation
        }
        
        # === DUAL MANIFOLD SYSTEM ===
        self.embed_dim = 64
        self.n_eigenmodes = 16
        self.manifold_size = 256
        self.latent_dim = self.n_eigenmodes * 2
        
        # PRIOR manifold - no constraints, just dynamics
        self.prior_manifold = np.random.randn(self.manifold_size, self.latent_dim) * 0.5
        self.prior_weights = np.ones(self.manifold_size) / self.manifold_size
        
        # POSTERIOR manifold - constrained by observations
        self.posterior_manifold = np.random.randn(self.manifold_size, self.latent_dim) * 0.5
        self.posterior_weights = np.ones(self.manifold_size) / self.manifold_size
        
        # Shared eigenbasis (learned from data)
        self.eigenvectors = np.eye(self.embed_dim)[:, :self.n_eigenmodes]
        self.eigenvalues = np.ones(self.n_eigenmodes)
        
        # Shared skull filter
        self.skull_filter = np.random.randn(self.latent_dim, self.embed_dim) * 0.1
        
        # Token history for eigenbasis learning
        self.token_history = deque(maxlen=200)
        
        # === INFORMATION TRACKING ===
        self.prior_entropy = np.log(self.manifold_size)  # Maximum entropy initially
        self.posterior_entropy = np.log(self.manifold_size)
        self.mutual_information = 0.0
        
        # Per-dimension information
        self.information_map = np.zeros(self.embed_dim)
        
        # History for rate computation
        self.mi_history = deque(maxlen=100)
        self.information_rate = 0.0
        
        # Surprise tracking
        self.expected_embedding = np.zeros(self.embed_dim)
        self.surprise = 0.0
        
        # === STRUCTURAL PRIOR TRACKING (Raj modes) ===
        self.has_structural_prior = False
        self.structural_embedding = np.zeros(self.embed_dim)
        self.structure_deviation = 0.0
        self.riverbed_alignment = 0.0
        self.deviation_spectrum = np.zeros(self.n_eigenmodes)
        
        # === DISPLAY ===
        self._display = np.zeros((750, 1200, 3), dtype=np.uint8)
        
        # Learning rates
        self.lr = 0.01
        self.epoch = 0
        
    def _tokens_to_embedding(self, tokens):
        """Convert token list to fixed-size embedding"""
        embedding = np.zeros(self.embed_dim)
        
        if tokens is None:
            return embedding
        
        # Handle various input formats
        if isinstance(tokens, (int, float, np.floating)):
            # Single scalar - use as simple amplitude
            embedding[0] = float(tokens)
            return embedding
            
        if isinstance(tokens, np.ndarray):
            if tokens.ndim == 0:
                # 0-d array (scalar)
                embedding[0] = float(tokens)
                return embedding
            elif tokens.ndim == 1:
                # 1D array - could be single token or flat embedding
                if len(tokens) == 3:
                    # Single token [id, amp, phase]
                    tokens = [tokens]
                elif len(tokens) == self.embed_dim:
                    # Already an embedding
                    return tokens.astype(np.float64)
                else:
                    # Unknown format, use as-is
                    embedding[:min(len(tokens), self.embed_dim)] = tokens[:self.embed_dim]
                    return embedding
            elif tokens.ndim == 2:
                # 2D array of tokens
                pass  # Continue to token processing below
            else:
                return embedding
        
        if isinstance(tokens, list):
            if len(tokens) == 0:
                return embedding
            # Check if it's a list of tokens or something else
            first = tokens[0]
            if isinstance(first, (int, float, np.floating)):
                # Flat list of numbers
                if len(tokens) == 3:
                    tokens = [tokens]  # Single token
                else:
                    embedding[:min(len(tokens), self.embed_dim)] = np.array(tokens[:self.embed_dim])
                    return embedding
        
        # Process as list of tokens
        try:
            for tok in tokens:
                if tok is None:
                    continue
                if isinstance(tok, (int, float, np.floating)):
                    continue  # Skip scalars in token list
                if not hasattr(tok, '__len__'):
                    continue
                if len(tok) < 3:
                    continue
                    
                token_id = int(tok[0]) % self.embed_dim
                amplitude = float(tok[1])
                phase = float(tok[2])
                
                embedding[token_id] += amplitude * np.cos(phase)
                embedding[(token_id + self.embed_dim//2) % self.embed_dim] += amplitude * np.sin(phase)
        except (TypeError, IndexError):
            pass
        
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding /= norm
            
        return embedding
    
    def _update_eigenbasis(self, embedding):
        """Update shared eigenspace from observation"""
        self.token_history.append(embedding.copy())
        
        if len(self.token_history) < 20:
            return
        
        X = np.array(list(self.token_history))
        X_centered = X - X.mean(axis=0)
        cov = X_centered.T @ X_centered / len(X)
        
        try:
            eigenvalues, eigenvectors = eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            self.eigenvalues = np.abs(eigenvalues[idx][:self.n_eigenmodes]) + 1e-6
            self.eigenvectors = eigenvectors[:, idx][:, :self.n_eigenmodes]
        except:
            pass
    
    def _project_to_eigenspace(self, embedding):
        """Project embedding to eigenspace"""
        return embedding @ self.eigenvectors
    
    def _compute_entropy(self, weights):
        """Compute entropy of a probability distribution in bits"""
        # Normalize
        w = weights / (weights.sum() + 1e-10)
        # Clip for numerical stability
        w = np.clip(w, 1e-10, 1.0)
        # Entropy in bits (log2)
        return -np.sum(w * np.log2(w))
    
    def _evolve_prior(self, temperature):
        """Evolve the prior manifold - NO observation constraints"""
        # Just diffusion - random walk
        noise_scale = 0.1 * temperature
        self.prior_manifold += np.random.randn(*self.prior_manifold.shape) * noise_scale
        
        # Soft regularization toward origin (prevents unbounded drift)
        self.prior_manifold *= 0.99
        
        # Prior weights stay uniform - no observations to update them
        self.prior_weights = np.ones(self.manifold_size) / self.manifold_size
    
    def _evolve_posterior(self, embedding, temperature):
        """Evolve the posterior manifold - WITH observation constraints"""
        # Project observation to eigenspace
        obs_eigen = self._project_to_eigenspace(embedding)
        
        # Project manifold through skull filter
        projected = self.posterior_manifold @ self.skull_filter
        proj_eigen = projected @ self.eigenvectors
        
        # Compute distances to observation
        weights = self.eigenvalues / (self.eigenvalues.sum() + 1e-10)
        distances = np.sum(weights * (proj_eigen - obs_eigen)**2, axis=1)
        
        # Convert to compatibility (posterior weights)
        temp = max(temperature, 0.01)
        compatibility = np.exp(-distances / (2 * temp**2))
        
        # Ensure proper normalization
        compatibility = np.clip(compatibility, 1e-10, None)
        total = compatibility.sum()
        if total > 1e-10:
            self.posterior_weights = compatibility / total
        else:
            self.posterior_weights = np.ones(self.manifold_size) / self.manifold_size
        
        # Ensure weights sum to 1 and are valid for np.random.choice
        self.posterior_weights = np.clip(self.posterior_weights, 1e-10, 1.0)
        self.posterior_weights = self.posterior_weights / self.posterior_weights.sum()
        
        # Resample posterior manifold based on weights
        n_resample = max(10, int(self.manifold_size * 0.1))
        
        try:
            indices = np.random.choice(
                self.manifold_size,
                size=n_resample,
                p=self.posterior_weights
            )
        except ValueError:
            # Fallback to uniform sampling if weights are invalid
            indices = np.random.randint(0, self.manifold_size, size=n_resample)
        
        # Replace low-weight particles
        low_weight_idx = np.argsort(self.posterior_weights)[:n_resample]
        noise_scale = 0.1 * temperature
        self.posterior_manifold[low_weight_idx] = (
            self.posterior_manifold[indices] + 
            np.random.randn(n_resample, self.latent_dim) * noise_scale
        )
        
        # Small innovation noise
        self.posterior_manifold += np.random.randn(*self.posterior_manifold.shape) * 0.01
        
        # Update skull filter
        weighted_proj = self.posterior_weights @ projected
        error = embedding - weighted_proj
        weighted_manifold = (self.posterior_weights.reshape(-1, 1) * self.posterior_manifold).sum(axis=0)
        gradient = np.outer(weighted_manifold, error)
        self.skull_filter += self.lr * gradient
        
        # Normalize skull filter
        norms = np.linalg.norm(self.skull_filter, axis=1, keepdims=True)
        self.skull_filter /= np.maximum(norms, 0.1)
        
        return distances
    
    def _compute_information_map(self):
        """Compute per-dimension information content"""
        # For each embedding dimension, compute how much the posterior
        # differs from the prior in that dimension
        
        # Project both manifolds to embedding space
        prior_proj = self.prior_manifold @ self.skull_filter  # (manifold_size, embed_dim)
        post_proj = self.posterior_manifold @ self.skull_filter
        
        # Weighted statistics
        prior_mean = self.prior_weights @ prior_proj
        prior_var = self.prior_weights @ (prior_proj - prior_mean)**2
        
        post_mean = self.posterior_weights @ post_proj
        post_var = self.posterior_weights @ (post_proj - post_mean)**2
        
        # Information ≈ reduction in variance (in log scale, gives bits-like quantity)
        # I_dim = 0.5 * log(prior_var / posterior_var)
        var_ratio = (prior_var + 1e-6) / (post_var + 1e-6)
        self.information_map = 0.5 * np.log2(np.maximum(var_ratio, 1.0))
        
        # Also compute expected embedding for surprise calculation
        self.expected_embedding = post_mean
    
    def _compute_surprise(self, embedding):
        """Compute surprise = how unexpected was this observation"""
        # Surprise = -log P(observation | prior)
        # Approximated by distance from expected
        distance = np.linalg.norm(embedding - self.expected_embedding)
        self.surprise = distance  # Simple version
    
    def step(self):
        self.epoch += 1
        
        # === GET INPUTS ===
        raw_tokens = self.get_blended_input('token_stream', 'mean')
        theta = self.get_blended_input('theta_phase', 'sum') or 0.0
        temperature = self.get_blended_input('temperature', 'sum')
        temperature = float(temperature) if temperature else 0.5
        temperature = max(0.1, min(2.0, temperature))
        
        # NEW: Get structural prior (Raj modes from ConnectomePriorNode)
        structural_prior = self.get_blended_input('structural_prior', 'mean')
        
        # === CONVERT TOKENS ===
        embedding = self._tokens_to_embedding(raw_tokens)
        
        # === CONVERT STRUCTURAL PRIOR ===
        self.structural_embedding = self._tokens_to_embedding(structural_prior)
        self.has_structural_prior = np.linalg.norm(self.structural_embedding) > 0.01
        
        has_input = np.linalg.norm(embedding) > 0.01
        
        # === UPDATE EIGENBASIS ===
        if has_input:
            self._update_eigenbasis(embedding)
        
        # === EVOLVE BOTH MANIFOLDS ===
        self._evolve_prior(temperature)
        
        if has_input:
            self._evolve_posterior(embedding, temperature)
        else:
            # Without input, posterior drifts toward prior
            self.posterior_weights = np.ones(self.manifold_size) / self.manifold_size
            self.posterior_manifold += np.random.randn(*self.posterior_manifold.shape) * 0.05
        
        # === COMPUTE ENTROPIES ===
        self.prior_entropy = self._compute_entropy(self.prior_weights)
        self.posterior_entropy = self._compute_entropy(self.posterior_weights)
        
        # === MUTUAL INFORMATION ===
        # I(X;Y) = H(prior) - H(posterior|observation)
        self.mutual_information = max(0, self.prior_entropy - self.posterior_entropy)
        
        # Track history for rate computation
        self.mi_history.append(self.mutual_information)
        
        # Information rate (bits per epoch)
        if len(self.mi_history) > 10:
            recent = list(self.mi_history)[-10:]
            self.information_rate = (recent[-1] - recent[0]) / 10.0
        
        # === PER-DIMENSION INFORMATION ===
        self._compute_information_map()
        
        # === SURPRISE ===
        if has_input:
            self._compute_surprise(embedding)
        
        # === NEW: STRUCTURE-FUNCTION DEVIATION ===
        if has_input and self.has_structural_prior:
            # Compute how much EEG (water) deviates from structure (riverbed)
            
            # Direct deviation in embedding space
            deviation_vector = embedding - self.structural_embedding
            self.structure_deviation = np.linalg.norm(deviation_vector)
            
            # Alignment = cosine similarity (1 = perfectly aligned with riverbed)
            dot_product = np.dot(embedding, self.structural_embedding)
            norm_product = (np.linalg.norm(embedding) * np.linalg.norm(self.structural_embedding) + 1e-10)
            self.riverbed_alignment = dot_product / norm_product
            
            # Per-mode deviation (project both onto eigenbasis)
            eeg_projection = embedding @ self.eigenvectors
            struct_projection = self.structural_embedding @ self.eigenvectors
            self.deviation_spectrum = np.abs(eeg_projection - struct_projection)
            
        elif has_input:
            # No structural prior - just measure EEG variance from learned eigenbasis
            self.structure_deviation = np.linalg.norm(embedding)
            self.riverbed_alignment = 0.0
            self.deviation_spectrum = np.abs(embedding @ self.eigenvectors)
        else:
            self.structure_deviation = 0.0
            self.riverbed_alignment = 0.0
            self.deviation_spectrum = np.zeros(self.n_eigenmodes)
        
        # === SET OUTPUTS ===
        self.outputs['mutual_information'] = float(self.mutual_information)
        self.outputs['prior_entropy'] = float(self.prior_entropy)
        self.outputs['posterior_entropy'] = float(self.posterior_entropy)
        self.outputs['information_rate'] = float(self.information_rate)
        self.outputs['surprise_signal'] = float(self.surprise)
        self.outputs['information_map'] = self.information_map.astype(np.float32)
        
        # NEW outputs
        self.outputs['structure_deviation'] = float(self.structure_deviation)
        self.outputs['riverbed_alignment'] = float(self.riverbed_alignment)
        self.outputs['deviation_spectrum'] = self.deviation_spectrum.astype(np.float32)
        
        # === RENDER ===
        self._render_display(embedding, has_input)
    
    def _render_display(self, embedding, has_input):
        """Render the full visualization"""
        img = self._display
        img[:] = (15, 12, 10)  # Dark warm background
        h, w = img.shape[:2]
        
        # === TITLE ===
        cv2.putText(img, "MUTUAL INFORMATION MANIFOLD", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 100), 2)
        cv2.putText(img, "How much does observation reduce uncertainty?", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 150, 100), 1)
        
        # === LEFT: PRIOR MANIFOLD (Unconstrained) ===
        self._render_manifold(img, self.prior_manifold, self.prior_weights,
                             20, 80, 250, 250, "PRIOR (No Data)", (100, 100, 200))
        
        # === CENTER-LEFT: POSTERIOR MANIFOLD (Constrained) ===
        self._render_manifold(img, self.posterior_manifold, self.posterior_weights,
                             290, 80, 250, 250, "POSTERIOR (With EEG)", (100, 200, 100))
        
        # === CENTER: MAIN INFO DISPLAY ===
        info_x, info_y = 560, 80
        info_w, info_h = 300, 250
        
        cv2.rectangle(img, (info_x, info_y), (info_x + info_w, info_y + info_h), 
                     (30, 25, 20), -1)
        cv2.rectangle(img, (info_x, info_y), (info_x + info_w, info_y + info_h),
                     (255, 180, 100), 2)
        
        cv2.putText(img, "INFORMATION CONTENT", (info_x + 50, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1)
        
        # Big MI number
        mi_str = f"{self.mutual_information:.2f}"
        cv2.putText(img, mi_str, (info_x + 80, info_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 200, 100), 3)
        cv2.putText(img, "BITS", (info_x + 200, info_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 100), 1)
        
        # Entropy breakdown
        cv2.putText(img, f"H(prior):     {self.prior_entropy:.3f} bits", 
                   (info_x + 20, info_y + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        cv2.putText(img, f"H(posterior): {self.posterior_entropy:.3f} bits", 
                   (info_x + 20, info_y + 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        cv2.putText(img, f"I(EEG;State): {self.mutual_information:.3f} bits", 
                   (info_x + 20, info_y + 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # Information rate
        rate_color = (100, 255, 100) if self.information_rate > 0 else (255, 100, 100)
        cv2.putText(img, f"dI/dt: {self.information_rate:+.4f} bits/epoch", 
                   (info_x + 20, info_y + 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, rate_color, 1)
        
        # Surprise
        cv2.putText(img, f"Surprise: {self.surprise:.3f}", 
                   (info_x + 20, info_y + 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 150), 1)
        
        # === RIGHT: INFORMATION MAP ===
        map_x, map_y = 880, 80
        map_w, map_h = 300, 120
        
        cv2.rectangle(img, (map_x, map_y), (map_x + map_w, map_y + map_h),
                     (30, 25, 20), -1)
        cv2.putText(img, "INFORMATION PER DIMENSION", (map_x + 10, map_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw bars for information map
        n_bars = min(32, len(self.information_map))
        bar_w = map_w // n_bars
        max_info = max(self.information_map.max(), 0.1)
        
        for i in range(n_bars):
            val = self.information_map[i]
            bar_h = int((val / max_info) * (map_h - 20))
            bx = map_x + i * bar_w
            by = map_y + map_h - 10
            
            # Color by information amount
            intensity = int(min(255, val / max_info * 255))
            color = (50, intensity, 255 - intensity // 2)
            
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 1, by), color, -1)
        
        # === BOTTOM LEFT: MI HISTORY ===
        hist_x, hist_y = 20, 360
        hist_w, hist_h = 400, 150
        
        cv2.rectangle(img, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h),
                     (25, 22, 18), -1)
        cv2.rectangle(img, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h),
                     (100, 80, 60), 1)
        cv2.putText(img, "MUTUAL INFORMATION HISTORY", (hist_x + 10, hist_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 180, 150), 1)
        
        if len(self.mi_history) > 1:
            mi_array = np.array(list(self.mi_history))
            max_mi = max(mi_array.max(), 0.1)
            
            points = []
            for i, mi in enumerate(mi_array):
                px = hist_x + int(i / len(mi_array) * hist_w)
                py = hist_y + hist_h - 10 - int((mi / max_mi) * (hist_h - 20))
                points.append((px, py))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(img, points[i], points[i+1], (255, 180, 100), 2)
        
        # === BOTTOM CENTER: EIGENSPECTRUM COMPARISON ===
        eigen_x, eigen_y = 440, 360
        eigen_w, eigen_h = 350, 150
        
        cv2.rectangle(img, (eigen_x, eigen_y), (eigen_x + eigen_w, eigen_y + eigen_h),
                     (25, 22, 18), -1)
        cv2.putText(img, "EIGENSPECTRUM (Learned Structure)", (eigen_x + 10, eigen_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        bar_w = eigen_w // self.n_eigenmodes
        max_eigen = self.eigenvalues.max() + 1e-10
        
        for i in range(self.n_eigenmodes):
            bar_h = int((self.eigenvalues[i] / max_eigen) * (eigen_h - 30))
            bx = eigen_x + i * bar_w + 5
            by = eigen_y + eigen_h - 15
            
            hue = int(i / self.n_eigenmodes * 180)
            hsv = np.array([[[hue, 200, 200]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 3, by), rgb, -1)
        
        # === BOTTOM RIGHT: INTERPRETATION ===
        interp_x, interp_y = 810, 360
        interp_w, interp_h = 370, 150
        
        cv2.rectangle(img, (interp_x, interp_y), (interp_x + interp_w, interp_y + interp_h),
                     (25, 22, 18), -1)
        cv2.putText(img, "INTERPRETATION", (interp_x + 10, interp_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Dynamic interpretation based on MI
        if self.mutual_information > 5:
            state = "HIGH INFO: EEG strongly constrains"
            color = (100, 255, 100)
        elif self.mutual_information > 2:
            state = "MODERATE: Some constraints"
            color = (255, 255, 100)
        elif self.mutual_information > 0.5:
            state = "LOW INFO: Weakly informative"
            color = (255, 180, 100)
        else:
            state = "MINIMAL: Almost nothing"
            color = (150, 150, 200)
        
        cv2.putText(img, state, (interp_x + 10, interp_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # NEW: Structure-function interpretation
        if self.has_structural_prior:
            cv2.putText(img, "WATER vs RIVERBED:", (interp_x + 10, interp_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 255), 1)
            
            cv2.putText(img, f"Deviation: {self.structure_deviation:.3f}", (interp_x + 10, interp_y + 78),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            cv2.putText(img, f"Alignment: {self.riverbed_alignment:.3f}", (interp_x + 140, interp_y + 78),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            
            # Riverbed interpretation
            if self.riverbed_alignment > 0.8:
                riverbed_state = "ON RIVERBED (autopilot)"
                riverbed_color = (100, 200, 255)
            elif self.riverbed_alignment > 0.5:
                riverbed_state = "NEAR RIVERBED"
                riverbed_color = (150, 200, 200)
            elif self.riverbed_alignment > 0.2:
                riverbed_state = "DEVIATING from structure"
                riverbed_color = (255, 200, 100)
            else:
                riverbed_state = "OFF RIVERBED - Novel!"
                riverbed_color = (255, 100, 100)
            
            cv2.putText(img, riverbed_state, (interp_x + 10, interp_y + 96),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, riverbed_color, 1)
        else:
            cv2.putText(img, "No structural_prior input", (interp_x + 10, interp_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            cv2.putText(img, "Connect eigenmode_spectrum", (interp_x + 10, interp_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Input status
        input_status = "Receiving EEG" if has_input else "No EEG input"
        input_color = (100, 200, 100) if has_input else (200, 100, 100)
        cv2.putText(img, input_status, (interp_x + 10, interp_y + 118),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, input_color, 1)
        
        cv2.putText(img, f"Epoch: {self.epoch}", (interp_x + 140, interp_y + 118),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        cv2.putText(img, f"Surprise: {self.surprise:.3f}", (interp_x + 240, interp_y + 118),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 150, 150), 1)
        
        # === VERY BOTTOM: Philosophy ===
        cv2.putText(img, "I(X;Y) = H(prior) - H(posterior|Y) : Information = Uncertainty reduction", 
                   (20, 540),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 100, 80), 1)
        cv2.putText(img, "We measure not what the brain IS, but how much LESS uncertain we are from observing it.", 
                   (20, 560),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 80, 60), 1)
        
        # === SKULL FILTER ===
        sf_y = 580
        cv2.putText(img, "LEARNED SKULL FILTER (shared by both manifolds)", 
                   (20, sf_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 120, 100), 1)
        
        sf_img = np.abs(self.skull_filter)
        sf_img = sf_img / (sf_img.max() + 1e-10)
        sf_img_u8 = (sf_img * 255).clip(0, 255).astype(np.uint8)
        sf_colored = cv2.applyColorMap(sf_img_u8, cv2.COLORMAP_INFERNO)
        sf_resized = cv2.resize(sf_colored, (500, 80))
        
        y_end = min(sf_y + 90, h)
        x_end = min(520, w)
        sf_h = y_end - (sf_y + 10)
        sf_w = x_end - 20
        if sf_h > 0 and sf_w > 0:
            sf_final = cv2.resize(sf_colored, (sf_w, sf_h))
            img[sf_y + 10:y_end, 20:x_end] = sf_final
        
        self._display = img
    
    def _render_manifold(self, img, manifold, weights, x0, y0, width, height, title, border_color):
        """Render a manifold as 2D density plot"""
        cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), (25, 22, 18), -1)
        cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), border_color, 2)
        cv2.putText(img, title, (x0 + 10, y0 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
        
        # Project to 2D
        manifold_2d = manifold[:, :2]
        
        # Create density image
        density = np.zeros((height, width), dtype=np.float32)
        
        min_x = manifold_2d[:, 0].min() - 0.5
        max_x = manifold_2d[:, 0].max() + 0.5
        min_y = manifold_2d[:, 1].min() - 0.5
        max_y = manifold_2d[:, 1].max() + 0.5
        
        range_x = max(max_x - min_x, 0.1)
        range_y = max(max_y - min_y, 0.1)
        
        for point, weight in zip(manifold_2d, weights):
            px = int((point[0] - min_x) / range_x * (width - 1))
            py = int((point[1] - min_y) / range_y * (height - 1))
            px = np.clip(px, 0, width - 1)
            py = np.clip(py, 0, height - 1)
            density[py, px] += weight * 1000
        
        density = gaussian_filter(density, sigma=5)
        density = density / (density.max() + 1e-10)
        
        density_u8 = (density * 255).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(density_u8, cv2.COLORMAP_INFERNO)
        
        img[y0:y0+height, x0:x0+width] = colored
        
        # Entropy label
        ent = self._compute_entropy(weights)
        cv2.putText(img, f"H={ent:.2f}", (x0 + 10, y0 + height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display
    
    def get_config_options(self):
        return [
            ("n_eigenmodes", "Eigenmodes", "int", 16, (4, 64)),
            ("manifold_size", "Manifold Size", "int", 256, (64, 1024)),
            ("lr", "Learning Rate", "float", 0.01, (0.001, 0.1)),
        ]