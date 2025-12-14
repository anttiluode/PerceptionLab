"""
Eigenmode Constraint Decoder - The Partial Key Hypothesis
============================================================
Instead of trying to decode EEG -> hidden state directly,
this node treats EEG as CONSTRAINTS on a higher-dimensional manifold.

The EEG is the skull-filtered shadow of brain activity.
Many different internal states could produce the same shadow.
But not ALL states - the shadow constrains the possibilities.

This node:
1. Learns the eigenspectrum of the EEG (what shadows are possible)
2. Maintains a latent manifold of "possible internal states"
3. Projects candidates through a learned skull-filter model
4. Keeps only those whose shadows match the current EEG

The output isn't "the" hidden state - it's the ENVELOPE of possible states
consistent with what we observe. This is epistemically honest.

When the envelope shrinks to a point, we have certainty.
When it's large, many internal configurations are compatible.

INPUTS:
- token_stream: Current EEG tokens from NeuralTransformerNode
- theta_phase: Phase for temporal alignment
- temperature: How tightly to enforce constraints (high = loose)
- prior_strength: How much to trust the learned manifold vs current observation

OUTPUTS:
- display: Visualization of the constraint manifold
- constrained_manifold: Complex field of possible states
- certainty_map: Where constraints are tight vs loose
- eigenmode_spectrum: The basis modes learned from this EEG stream
- compatible_modes: How many modes are compatible with current observation
- constraint_violation: How much current state violates learned constraints

The key insight: We're not trying to invert the prism.
We're asking: what shapes of light COULD have made this rainbow?
"""

import numpy as np
import cv2
from collections import deque
from scipy.linalg import eigh, svd
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

class EigenConstraintDecoder(BaseNode):
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Eigen Constraint Decoder"
    NODE_COLOR = QtGui.QColor(100, 50, 150)  # Deep purple - uncertainty color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Eigen Constraint Decoder"
        
        self.inputs = {
            'token_stream': 'spectrum',
            'theta_phase': 'signal',
            'temperature': 'signal',
            'prior_strength': 'signal',
        }
        
        self.outputs = {
            'display': 'image',
            'constrained_manifold': 'complex_spectrum',
            'certainty_map': 'image',
            'eigenmode_spectrum': 'spectrum',
            'compatible_modes': 'signal',
            'constraint_violation': 'signal',
        }
        
        # === EIGENSPACE LEARNING ===
        self.embed_dim = 64
        self.n_eigenmodes = 16  # Number of eigenmodes to track
        self.history_len = 200
        
        # Token history for covariance estimation
        self.token_history = deque(maxlen=self.history_len)
        
        # Learned eigenbasis of observed EEG
        self.eigenvectors = np.eye(self.embed_dim)[:, :self.n_eigenmodes]
        self.eigenvalues = np.ones(self.n_eigenmodes)
        
        # === LATENT MANIFOLD ===
        self.manifold_size = 256
        self.latent_dim = self.n_eigenmodes * 2  # Real + imaginary parts
        
        # The manifold of "possible internal states"
        # Each point is a candidate configuration
        self.manifold = np.random.randn(self.manifold_size, self.latent_dim) * 0.1
        
        # Learned "skull filter" - how internal states project to EEG
        # This is what we're trying to infer, not invert
        self.skull_filter = np.random.randn(self.latent_dim, self.embed_dim) * 0.1
        
        # === CONSTRAINT STATE ===
        self.current_constraint = np.zeros(self.embed_dim)
        self.constraint_tightness = np.ones(self.embed_dim)  # Per-dimension certainty
        self.compatible_count = self.manifold_size
        
        # === METRICS ===
        self.constraint_violation = 0.0
        self.entropy = 0.0
        self.epoch = 0
        
        # === DISPLAY ===
        self._display = np.zeros((700, 1100, 3), dtype=np.uint8)
        self._certainty_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Learning rates
        self.eigenbasis_lr = 0.01
        self.skull_filter_lr = 0.001
        self.manifold_lr = 0.01
        
    def _tokens_to_embedding(self, tokens):
        """Convert token list to fixed-size embedding"""
        embedding = np.zeros(self.embed_dim)
        
        if tokens is None or len(tokens) == 0:
            return embedding
            
        for tok in tokens:
            if len(tok) < 3:
                continue
            token_id = int(tok[0]) % self.embed_dim
            amplitude = float(tok[1])
            phase = float(tok[2])
            
            # Encode in embedding
            embedding[token_id] += amplitude * np.cos(phase)
            embedding[(token_id + self.embed_dim//2) % self.embed_dim] += amplitude * np.sin(phase)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding /= norm
            
        return embedding
    
    def _update_eigenbasis(self, embedding):
        """Update the learned eigenspace from new observation"""
        self.token_history.append(embedding.copy())
        
        if len(self.token_history) < 20:
            return
        
        # Build data matrix
        X = np.array(list(self.token_history))  # (history_len, embed_dim)
        
        # Center
        X_centered = X - X.mean(axis=0)
        
        # Covariance
        cov = X_centered.T @ X_centered / len(X)
        
        # Eigen decomposition
        try:
            eigenvalues, eigenvectors = eigh(cov)
            
            # Sort by magnitude (largest first)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Keep top n_eigenmodes
            self.eigenvalues = np.abs(eigenvalues[:self.n_eigenmodes]) + 1e-6
            self.eigenvectors = eigenvectors[:, :self.n_eigenmodes]
            
        except Exception as e:
            pass  # Keep previous eigenbasis
    
    def _project_to_eigenspace(self, embedding):
        """Project embedding onto learned eigenbasis"""
        return embedding @ self.eigenvectors  # (n_eigenmodes,)
    
    def _apply_constraints(self, embedding, temperature):
        """
        Filter manifold to keep only states compatible with observation.
        This is the key operation: we're not decoding, we're constraining.
        """
        # Project observation to eigenspace
        obs_eigen = self._project_to_eigenspace(embedding)
        
        # For each manifold point, project through skull filter and compare
        projected = self.manifold @ self.skull_filter  # (manifold_size, embed_dim)
        proj_eigen = projected @ self.eigenvectors  # (manifold_size, n_eigenmodes)
        
        # Distance to observation in eigenspace
        # Weight by eigenvalue (care more about high-variance dimensions)
        weights = self.eigenvalues / self.eigenvalues.sum()
        distances = np.sum(weights * (proj_eigen - obs_eigen)**2, axis=1)
        
        # Convert to compatibility scores (soft constraint)
        # Temperature controls how strict: low temp = hard constraint
        temp = max(temperature, 0.01)
        compatibility = np.exp(-distances / (2 * temp**2))
        
        # Normalize to get probability distribution over manifold
        compatibility /= compatibility.sum() + 1e-10
        
        # Count effective number of compatible states (entropy measure)
        entropy = -np.sum(compatibility * np.log(compatibility + 1e-10))
        self.compatible_count = int(np.exp(entropy))
        self.entropy = entropy
        
        # Update constraint tightness per dimension
        # Dimensions where manifold points agree are tight
        weighted_mean = compatibility @ projected
        weighted_var = compatibility @ (projected - weighted_mean)**2
        self.constraint_tightness = 1.0 / (weighted_var + 0.01)
        
        # Constraint violation = how far is most compatible point from perfect match
        self.constraint_violation = distances.min()
        
        return compatibility, projected
    
    def _update_skull_filter(self, embedding, compatibility, projected):
        """Learn the skull filter from constraint satisfaction"""
        # The skull filter should map internal states to observed EEG
        # We update it to reduce constraint violation
        
        # Weighted reconstruction
        weighted_proj = compatibility @ projected
        
        # Error
        error = embedding - weighted_proj
        
        # Gradient: update skull filter to reduce error
        # This is a weighted outer product
        weighted_manifold = (compatibility.reshape(-1, 1) * self.manifold).sum(axis=0)
        gradient = np.outer(weighted_manifold, error)
        
        self.skull_filter += self.skull_filter_lr * gradient
        
        # Regularization: keep skull filter normalized
        norms = np.linalg.norm(self.skull_filter, axis=1, keepdims=True)
        self.skull_filter /= np.maximum(norms, 0.1)
    
    def _evolve_manifold(self, compatibility, prior_strength):
        """
        Evolve the manifold of possible states.
        
        Key insight: The manifold should represent what we believe
        about the space of possible internal states, updated by observations.
        """
        # Resample: duplicate high-compatibility points, remove low ones
        # This is essentially particle filtering
        
        n_resample = max(10, int(self.manifold_size * 0.1))
        
        # Sample new points proportional to compatibility
        indices = np.random.choice(
            self.manifold_size, 
            size=n_resample,
            p=compatibility
        )
        
        # Add noise to resampled points (diffusion)
        noise_scale = 0.1 * prior_strength
        new_points = self.manifold[indices] + np.random.randn(n_resample, self.latent_dim) * noise_scale
        
        # Replace low-compatibility points
        low_compat_idx = np.argsort(compatibility)[:n_resample]
        self.manifold[low_compat_idx] = new_points
        
        # Also add small innovation noise to maintain diversity
        self.manifold += np.random.randn(*self.manifold.shape) * 0.01
    
    def _build_constrained_manifold_image(self, compatibility, projected):
        """
        Create a 2D visualization of the constrained manifold.
        This shows the "envelope" of possible internal states.
        """
        # Project manifold to 2D for visualization using first 2 eigenmodes
        manifold_2d = self.manifold[:, :2]
        
        # Create density image
        img_size = 256
        img = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Scale manifold to image coordinates
        min_x, max_x = manifold_2d[:, 0].min() - 0.5, manifold_2d[:, 0].max() + 0.5
        min_y, max_y = manifold_2d[:, 1].min() - 0.5, manifold_2d[:, 1].max() + 0.5
        
        range_x = max(max_x - min_x, 0.1)
        range_y = max(max_y - min_y, 0.1)
        
        for i, (point, compat) in enumerate(zip(manifold_2d, compatibility)):
            px = int((point[0] - min_x) / range_x * (img_size - 1))
            py = int((point[1] - min_y) / range_y * (img_size - 1))
            
            px = np.clip(px, 0, img_size - 1)
            py = np.clip(py, 0, img_size - 1)
            
            img[py, px] += compat * 1000  # Scale for visibility
        
        # Smooth
        img = gaussian_filter(img, sigma=3)
        
        # Normalize
        img = img / (img.max() + 1e-10)
        
        # Convert to color (heat map)
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        
        return colored
    
    def _build_certainty_map(self):
        """
        Visualize certainty per dimension.
        Tight constraints = high certainty = we know something.
        Loose constraints = could be anything.
        """
        # Reshape constraint tightness to square for visualization
        side = int(np.ceil(np.sqrt(self.embed_dim)))
        certainty_padded = np.zeros(side * side)
        certainty_padded[:len(self.constraint_tightness)] = self.constraint_tightness
        certainty_2d = certainty_padded.reshape(side, side)
        
        # Normalize
        certainty_2d = certainty_2d / (certainty_2d.max() + 1e-10)
        
        # Resize
        certainty_resized = cv2.resize(certainty_2d.astype(np.float32), (256, 256))
        certainty_u8 = (certainty_resized * 255).clip(0, 255).astype(np.uint8)
        
        colored = cv2.applyColorMap(certainty_u8, cv2.COLORMAP_VIRIDIS)
        return colored
    
    def step(self):
        self.epoch += 1
        
        # === GET INPUTS ===
        raw_tokens = self.get_blended_input('token_stream', 'mean')
        theta = self.get_blended_input('theta_phase', 'sum') or 0.0
        temperature = self.get_blended_input('temperature', 'sum')
        temperature = float(temperature) if temperature else 0.5
        prior_strength = self.get_blended_input('prior_strength', 'sum')
        prior_strength = float(prior_strength) if prior_strength else 0.5
        
        # === CONVERT TOKENS TO EMBEDDING ===
        if isinstance(raw_tokens, list):
            tokens = []
            for t in raw_tokens:
                if hasattr(t, '__iter__') and len(t) >= 3:
                    tokens.append(t)
            embedding = self._tokens_to_embedding(tokens)
        else:
            embedding = self._tokens_to_embedding(raw_tokens)
        
        # === UPDATE EIGENBASIS ===
        self._update_eigenbasis(embedding)
        
        # === APPLY CONSTRAINTS ===
        self.current_constraint = embedding
        compatibility, projected = self._apply_constraints(embedding, temperature)
        
        # === UPDATE SKULL FILTER ===
        self._update_skull_filter(embedding, compatibility, projected)
        
        # === EVOLVE MANIFOLD ===
        self._evolve_manifold(compatibility, prior_strength)
        
        # === BUILD OUTPUTS ===
        manifold_img = self._build_constrained_manifold_image(compatibility, projected)
        certainty_img = self._build_certainty_map()
        self._certainty_img = certainty_img
        
        # Eigenmode spectrum output
        eigenmode_spectrum = np.zeros((self.n_eigenmodes, 3))
        for i in range(self.n_eigenmodes):
            eigenmode_spectrum[i] = [
                i,  # mode index
                self.eigenvalues[i],  # amplitude = eigenvalue
                np.arctan2(self.eigenvectors[1, i], self.eigenvectors[0, i])  # phase from first 2 components
            ]
        
        # Constrained manifold as complex field
        # Take the weighted average position in complex form
        weighted_latent = compatibility @ self.manifold
        complex_manifold = np.zeros((16, 16), dtype=np.complex128)
        for i in range(min(16, self.n_eigenmodes)):
            idx = i % 16
            complex_manifold[idx // 4, idx % 4] = weighted_latent[i] + 1j * weighted_latent[i + self.n_eigenmodes] if i + self.n_eigenmodes < len(weighted_latent) else 0
        
        # Set outputs
        self.outputs['constrained_manifold'] = complex_manifold
        self.outputs['certainty_map'] = certainty_img
        self.outputs['eigenmode_spectrum'] = eigenmode_spectrum.astype(np.float32)
        self.outputs['compatible_modes'] = float(self.compatible_count)
        self.outputs['constraint_violation'] = float(self.constraint_violation)
        
        # === RENDER DISPLAY ===
        self._render_display(embedding, compatibility, manifold_img, certainty_img)
    
    def _render_display(self, embedding, compatibility, manifold_img, certainty_img):
        """Render the full visualization"""
        img = self._display
        img[:] = (15, 15, 20)
        h, w = img.shape[:2]
        
        # === TITLE ===
        cv2.putText(img, "EIGEN-CONSTRAINT DECODER", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 255), 2)
        cv2.putText(img, "The Partial Key Hypothesis", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # === LEFT: Constrained Manifold ===
        man_x, man_y = 20, 70
        man_size = 280
        manifold_resized = cv2.resize(manifold_img, (man_size, man_size))
        img[man_y:man_y+man_size, man_x:man_x+man_size] = manifold_resized
        cv2.rectangle(img, (man_x, man_y), (man_x+man_size, man_y+man_size), (100, 50, 150), 2)
        cv2.putText(img, "CONSTRAINT MANIFOLD", (man_x, man_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 150, 255), 1)
        cv2.putText(img, f"Compatible: {self.compatible_count}/{self.manifold_size}", 
                   (man_x, man_y + man_size + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        
        # === CENTER: Eigenspectrum ===
        eigen_x, eigen_y = 330, 70
        eigen_w, eigen_h = 300, 150
        cv2.rectangle(img, (eigen_x, eigen_y), (eigen_x+eigen_w, eigen_y+eigen_h), (50, 50, 60), -1)
        cv2.rectangle(img, (eigen_x, eigen_y), (eigen_x+eigen_w, eigen_y+eigen_h), (100, 100, 120), 1)
        cv2.putText(img, "EIGENSPECTRUM", (eigen_x, eigen_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Draw eigenvalue bars
        bar_w = eigen_w // self.n_eigenmodes
        max_eigen = self.eigenvalues.max() + 1e-10
        for i in range(self.n_eigenmodes):
            bar_h = int((self.eigenvalues[i] / max_eigen) * (eigen_h - 20))
            bx = eigen_x + i * bar_w + 5
            by = eigen_y + eigen_h - 10
            
            # Color by index
            hue = int(i / self.n_eigenmodes * 180)
            hsv = np.array([[[hue, 200, 200]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 3, by), rgb, -1)
        
        # === CENTER: Certainty Map ===
        cert_x, cert_y = 330, 250
        cert_size = 180
        certainty_resized = cv2.resize(certainty_img, (cert_size, cert_size))
        img[cert_y:cert_y+cert_size, cert_x:cert_x+cert_size] = certainty_resized
        cv2.rectangle(img, (cert_x, cert_y), (cert_x+cert_size, cert_y+cert_size), (50, 100, 50), 2)
        cv2.putText(img, "CERTAINTY MAP", (cert_x, cert_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 150), 1)
        
        # === RIGHT: Current Constraint ===
        const_x, const_y = 650, 70
        const_w, const_h = 200, 280
        cv2.rectangle(img, (const_x, const_y), (const_x+const_w, const_y+const_h), (40, 40, 50), -1)
        cv2.putText(img, "CURRENT CONSTRAINT", (const_x, const_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 150, 100), 1)
        
        # Draw constraint as bar chart (embedding)
        bar_h_max = const_h - 40
        n_bars = min(32, len(embedding))
        bar_w = const_w // n_bars
        for i in range(n_bars):
            val = embedding[i]
            bar_h = int(abs(val) * bar_h_max / 2)
            bx = const_x + i * bar_w
            by = const_y + const_h // 2
            
            if val >= 0:
                cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 1, by), (100, 200, 100), -1)
            else:
                cv2.rectangle(img, (bx, by), (bx + bar_w - 1, by + bar_h), (200, 100, 100), -1)
        
        # === FAR RIGHT: Metrics ===
        met_x = 880
        met_y = 70
        
        cv2.putText(img, "METRICS", (met_x, met_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        metrics = [
            ("Epoch", self.epoch),
            ("Compatible", f"{self.compatible_count}"),
            ("Violation", f"{self.constraint_violation:.4f}"),
            ("Entropy", f"{self.entropy:.2f}"),
            ("Top Eigen", f"{self.eigenvalues[0]:.3f}"),
        ]
        
        for i, (name, val) in enumerate(metrics):
            y = met_y + 30 + i * 25
            cv2.putText(img, f"{name}:", (met_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(img, str(val), (met_x + 80, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
        
        # === BOTTOM: Interpretation ===
        interp_y = 470
        
        # Certainty indicator
        mean_certainty = np.mean(self.constraint_tightness)
        if self.compatible_count < 20:
            state = "HIGH CERTAINTY - Few states compatible"
            color = (100, 255, 100)
        elif self.compatible_count < 100:
            state = "MODERATE - Constraint narrows options"
            color = (255, 255, 100)
        else:
            state = "LOW CERTAINTY - Many states possible"
            color = (100, 150, 255)
        
        cv2.putText(img, state, (20, interp_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Philosophy note
        cv2.putText(img, "The EEG is a shadow. Many objects cast similar shadows.", 
                   (20, interp_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 150), 1)
        cv2.putText(img, "But not ALL objects - the shadow constrains possibilities.", 
                   (20, interp_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 150), 1)
        cv2.putText(img, "We don't decode the hidden. We enumerate what's compatible.", 
                   (20, interp_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 120, 180), 1)
        
        # === SKULL FILTER VISUALIZATION ===
        skull_y = 550
        cv2.putText(img, "LEARNED SKULL FILTER (how internal states project to EEG)", 
                   (20, skull_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 100, 150), 1)
        
        # Show skull filter as small heatmap
        sf_img = np.abs(self.skull_filter)
        sf_img = sf_img / (sf_img.max() + 1e-10)
        sf_img_u8 = (sf_img * 255).clip(0, 255).astype(np.uint8)
        sf_colored = cv2.applyColorMap(sf_img_u8, cv2.COLORMAP_MAGMA)
        sf_resized = cv2.resize(sf_colored, (400, 100))
        img[skull_y + 10:skull_y + 110, 20:420] = sf_resized
        
        self._display = img
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'certainty_map':
            return self._certainty_img
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display
    
    def get_config_options(self):
        return [
            ("n_eigenmodes", "Eigenmodes", "int", 16, (4, 64)),
            ("manifold_size", "Manifold Size", "int", 256, (64, 1024)),
            ("eigenbasis_lr", "Eigenbasis LR", "float", 0.01, (0.001, 0.1)),
            ("skull_filter_lr", "Skull Filter LR", "float", 0.001, (0.0001, 0.01)),
        ]