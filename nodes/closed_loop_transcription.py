"""
Closed-Loop Transcription - Ma's Complete Learning Framework
=============================================================
The INTEGRATION node that closes the loop between encoder and decoder.

FROM THE PAPER:
"Once discriminative and generative models are combined to form a 
complete closed-loop system, learning can become autonomous (without 
exterior supervision), more efficient, stable, and adaptive."

THIS NODE IMPLEMENTS:
1. The full f ∘ g ∘ f cycle (encoder → decoder → re-encode)
2. Minimax game: f tries to distinguish, g tries to fool
3. Autonomous learning without external labels
4. The "self-consistency" loss: min ||f(x) - f(g(f(x)))||

ARCHITECTURE:
         x (tokens)
            ↓
      f(x) = z (encode)
            ↓
      g(z) = x̂ (decode)
            ↓
      f(x̂) = ẑ (re-encode)
            ↓
    Loss = ||z - ẑ|| (consistency)

When Loss → 0, the system has achieved self-consistency.
The learned representation z then captures the "true structure."

OUTPUTS:
- display: Full visualization of the loop
- loop_loss: The self-consistency loss (should decrease)
- is_consistent: Boolean gate when loss is below threshold
- learning_signal: Gradient for both encoder and decoder
- manifold_state: Current state on the learned manifold

CREATED: December 2025
THEORY: Yi Ma et al. "Parsimony and Self-Consistency" (2022)
"""

import numpy as np
import cv2
from collections import deque
from scipy.linalg import svd

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

class ClosedLoopTranscription(BaseNode):
    """
    The complete closed-loop system that achieves autonomous learning
    through self-consistency between encoding and decoding.
    """
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Closed-Loop Transcription"
    NODE_COLOR = QtGui.QColor(200, 50, 200)  # Magenta - integration
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'token_stream': 'spectrum',        # Raw input x
            'theta_phase': 'signal',           # Timing reference
            'learning_rate': 'signal',         # Learning speed
            'temperature': 'signal',           # Softmax sharpness
        }
        
        self.outputs = {
            'display': 'image',
            'compressed_z': 'spectrum',        # First encoding z = f(x)
            'reconstructed': 'spectrum',       # Reconstruction x̂ = g(z)
            're_encoded_z': 'spectrum',        # Re-encoding ẑ = f(x̂)
            'loop_loss': 'signal',             # ||z - ẑ||
            'is_consistent': 'signal',         # 1 if loss < threshold
            'rate_reduction': 'signal',        # ΔR from encoding
            'learning_signal': 'signal',       # Gradient magnitude
            'manifold_state': 'spectrum',      # Current position on manifold
        }
        
        # === DIMENSIONS ===
        self.input_dim = 64
        self.latent_dim = 64
        self.n_subspaces = 5
        self.subspace_dim = 16
        self.n_tokens = 20
        
        # === ENCODER (f: x → z) ===
        np.random.seed(42)
        self.encoder_bases = []
        for j in range(self.n_subspaces):
            U = np.random.randn(self.input_dim, self.subspace_dim)
            U, _, _ = svd(U, full_matrices=False)
            self.encoder_bases.append(U[:, :self.subspace_dim])
        
        # === DECODER (g: z → x̂) ===
        np.random.seed(43)
        self.decoder_weights = []
        for j in range(self.n_subspaces):
            W = np.random.randn(self.n_tokens * 3, self.latent_dim) * 0.1
            self.decoder_weights.append(W)
        
        # === MA'S PARAMETERS ===
        self.epsilon = 0.5
        self.consistency_threshold = 0.1
        
        # === STATE ===
        self.current_z = np.zeros(self.latent_dim)
        self.current_x_hat = np.zeros((self.n_tokens, 3))
        self.current_z_hat = np.zeros(self.latent_dim)
        self.current_loss = 0.0
        self.current_rate_reduction = 0.0
        self.current_assignments = np.zeros(self.n_subspaces)
        
        # === HISTORY ===
        self.loss_history = deque(maxlen=500)
        self.rate_history = deque(maxlen=500)
        self.z_buffer = deque(maxlen=100)
        
        # === LEARNING ===
        self.base_lr = 0.01
        self.epoch = 0
        
        # === DISPLAY ===
        self._display = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def _tokens_to_embedding(self, tokens):
        """Convert token stream to embedding vector"""
        z = np.zeros(self.input_dim, dtype=np.float32)
        
        if tokens is None:
            return z
        
        if isinstance(tokens, np.ndarray):
            if tokens.ndim == 1:
                if len(tokens) == self.input_dim:
                    return tokens.astype(np.float32)
                tokens = tokens.reshape(-1, 3) if len(tokens) % 3 == 0 else np.zeros((0, 3))
            
            for tok in tokens:
                if len(tok) >= 3:
                    token_id = int(tok[0]) % 20
                    amplitude = float(tok[1])
                    phase = float(tok[2])
                    
                    # Distribute across embedding
                    for i in range(3):
                        idx = (token_id * 3 + i) % self.input_dim
                        z[idx] += amplitude * np.cos(phase + i * np.pi / 3)
        
        # Normalize
        norm = np.linalg.norm(z)
        if norm > 1e-9:
            z = z / norm
        
        return z
    
    def _encode(self, x, temperature=1.0):
        """
        f(x) → z: Encode input to latent representation.
        Also computes subspace assignments.
        """
        # Compute projection onto each subspace
        projections = np.zeros(self.n_subspaces)
        z_components = np.zeros((self.n_subspaces, self.latent_dim))
        
        for j, U in enumerate(self.encoder_bases):
            # Project onto subspace
            z_j = U @ (U.T @ x)
            z_components[j] = np.pad(z_j, (0, self.latent_dim - len(z_j)))[:self.latent_dim]
            projections[j] = np.linalg.norm(z_j)
        
        # Soft assignments via softmax
        projections = projections / max(temperature, 0.1)
        exp_proj = np.exp(projections - np.max(projections))
        assignments = exp_proj / (np.sum(exp_proj) + 1e-9)
        
        # Weighted combination
        z = np.zeros(self.latent_dim)
        for j, (z_j, w) in enumerate(zip(z_components, assignments)):
            z += w * z_j
        
        return z, assignments
    
    def _decode(self, z, assignments, phase=0.0):
        """
        g(z) → x̂: Decode latent to reconstruction.
        """
        output = np.zeros(self.n_tokens * 3)
        
        for j, (W, w) in enumerate(zip(self.decoder_weights, assignments)):
            if w < 0.05:
                continue
            output += w * (W @ z)
        
        # Reshape to tokens
        tokens = output.reshape(self.n_tokens, 3)
        
        # Post-process
        for i in range(self.n_tokens):
            tokens[i, 0] = i % 20
            tokens[i, 1] = np.abs(tokens[i, 1])
            tokens[i, 2] = tokens[i, 2] + phase
        
        return tokens
    
    def _compute_coding_rate(self, Z, eps=0.5):
        """Ma's coding rate formula"""
        d, n = Z.shape
        if n == 0:
            return 0.0
        
        alpha = d / (n * eps**2)
        cov_term = alpha * (Z @ Z.T)
        I = np.eye(d)
        
        sign, logdet = np.linalg.slogdet(I + cov_term)
        if sign <= 0:
            return 100.0
        
        return 0.5 * logdet
    
    def _compute_rate_reduction(self, Z, assignments_history):
        """
        ΔR = R(Z) - Σ_j w_j R(Z_j)
        """
        d, n = Z.shape
        if n < 5:
            return 0.0
        
        R_total = self._compute_coding_rate(Z, self.epsilon)
        R_subspaces = 0.0
        
        assignments = np.array(assignments_history)
        
        for j in range(self.n_subspaces):
            w_j = np.mean(assignments[:, j])
            if w_j > 0.01:
                weights = assignments[:, j]
                Z_j = Z * weights
                R_j = self._compute_coding_rate(Z_j, self.epsilon)
                R_subspaces += w_j * R_j
        
        return R_total - R_subspaces
    
    def _update_parameters(self, x, z, x_hat, z_hat, assignments, lr=0.01):
        """
        Gradient step on the self-consistency loss.
        Update both encoder and decoder to minimize ||z - ẑ||.
        """
        # Loss gradient
        dz = z - z_hat
        loss_mag = np.linalg.norm(dz)
        
        if loss_mag < 1e-9:
            return 0.0
        
        # Normalize gradient
        dz_norm = dz / loss_mag
        
        # Update encoder bases (move toward better compression)
        for j, U in enumerate(self.encoder_bases):
            if assignments[j] < 0.05:
                continue
            
            # Gradient: move subspace to better capture the structure
            residual = x - U @ (U.T @ x)
            delta_U = lr * assignments[j] * np.outer(residual, U.T @ x)
            
            if np.linalg.norm(delta_U) < 1.0:
                self.encoder_bases[j] = U + delta_U[:, :self.subspace_dim]
                
                # Re-orthonormalize
                U_new, _, _ = svd(self.encoder_bases[j], full_matrices=False)
                self.encoder_bases[j] = U_new[:, :self.subspace_dim]
        
        # Update decoder weights (move toward better reconstruction)
        x_hat_flat = x_hat.flatten()
        for j, W in enumerate(self.decoder_weights):
            if assignments[j] < 0.05:
                continue
            
            # Gradient: outer product
            grad = np.outer(x_hat_flat - np.zeros_like(x_hat_flat), dz_norm)
            grad = grad[:W.shape[0], :W.shape[1]]
            
            self.decoder_weights[j] -= lr * assignments[j] * grad
        
        return loss_mag
    
    def step(self):
        # Get inputs
        raw_tokens = self.get_blended_input('token_stream', 'mean')
        phase_val = self.get_blended_input('theta_phase', 'sum')
        lr_val = self.get_blended_input('learning_rate', 'sum')
        temp_val = self.get_blended_input('temperature', 'sum')
        
        phase = float(phase_val) if phase_val else 0.0
        lr = float(lr_val) if lr_val and lr_val > 0 else self.base_lr
        temperature = float(temp_val) if temp_val and temp_val > 0 else 1.0
        
        # Convert input to embedding
        x = self._tokens_to_embedding(raw_tokens)
        
        # === FORWARD PASS ===
        # Step 1: Encode x → z
        self.current_z, self.current_assignments = self._encode(x, temperature)
        
        # Step 2: Decode z → x̂
        self.current_x_hat = self._decode(self.current_z, self.current_assignments, phase)
        
        # Step 3: Re-encode x̂ → ẑ
        x_hat_emb = self._tokens_to_embedding(self.current_x_hat)
        self.current_z_hat, _ = self._encode(x_hat_emb, temperature)
        
        # === LOSS COMPUTATION ===
        self.current_loss = np.linalg.norm(self.current_z - self.current_z_hat)
        self.loss_history.append(self.current_loss)
        
        # === RATE REDUCTION ===
        self.z_buffer.append(self.current_z.copy())
        if len(self.z_buffer) >= 10:
            Z = np.array(list(self.z_buffer)).T
            
            # Get assignment history
            assignments_hist = []
            for z_hist in self.z_buffer:
                _, a = self._encode(z_hist, temperature)
                assignments_hist.append(a)
            
            self.current_rate_reduction = self._compute_rate_reduction(Z, assignments_hist)
            self.rate_history.append(self.current_rate_reduction)
        
        # === BACKWARD PASS (Learning) ===
        learning_signal = self._update_parameters(
            x, self.current_z, self.current_x_hat, self.current_z_hat,
            self.current_assignments, lr
        )
        
        self.epoch += 1
        
        # === UPDATE OUTPUTS ===
        self.outputs['compressed_z'] = self.current_z.astype(np.float32)
        self.outputs['reconstructed'] = self.current_x_hat.astype(np.float32)
        self.outputs['re_encoded_z'] = self.current_z_hat.astype(np.float32)
        self.outputs['loop_loss'] = float(self.current_loss)
        self.outputs['is_consistent'] = 1.0 if self.current_loss < self.consistency_threshold else 0.0
        self.outputs['rate_reduction'] = float(self.current_rate_reduction)
        self.outputs['learning_signal'] = float(learning_signal)
        self.outputs['manifold_state'] = self.current_assignments.astype(np.float32)
        
        # Render
        self._render_display(x)
    
    def _render_display(self, x):
        """Full visualization of the closed loop"""
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === TITLE ===
        cv2.putText(img, "CLOSED-LOOP TRANSCRIPTION", (w//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(img, "f ∘ g ∘ f: x → z → x̂ → ẑ", (w//2 - 100, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # === LEFT COLUMN: The Loop ===
        self._render_loop_diagram(img, 30, 80, 350, 400)
        
        # === CENTER: Latent Space ===
        self._render_latent_space(img, 400, 80, 350, 350)
        
        # === RIGHT: Loss History ===
        self._render_loss_history(img, 770, 80, 400, 200)
        
        # === BOTTOM RIGHT: Subspace Assignments ===
        self._render_assignments(img, 770, 300, 400, 150)
        
        # === BOTTOM: Statistics ===
        y_stats = h - 80
        
        # Loss with color coding
        loss_color = (100, 255, 100) if self.current_loss < 0.1 else \
                     (255, 255, 100) if self.current_loss < 0.5 else (255, 100, 100)
        cv2.putText(img, f"Loop Loss ||z - ẑ||: {self.current_loss:.4f}", (30, y_stats),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, loss_color, 1)
        
        cv2.putText(img, f"Rate Reduction ΔR: {self.current_rate_reduction:.4f}", (30, y_stats + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)
        
        cv2.putText(img, f"Epoch: {self.epoch}", (30, y_stats + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Consistency indicator
        is_consistent = self.outputs.get('is_consistent', 0)
        if is_consistent > 0.5:
            cv2.putText(img, "✓ SELF-CONSISTENT", (w - 200, y_stats + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        else:
            cv2.putText(img, "○ Learning...", (w - 180, y_stats + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
        
        self._display = img
    
    def _render_loop_diagram(self, img, x0, y0, width, height):
        """Visual diagram of the f ∘ g ∘ f loop"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        # Nodes
        cx = x0 + width // 2
        
        # x (input)
        x_pos = (cx, y0 + 50)
        cv2.circle(img, x_pos, 25, (100, 200, 100), -1)
        cv2.putText(img, "x", (x_pos[0]-8, x_pos[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # z (encoded)
        z_pos = (cx, y0 + 150)
        cv2.circle(img, z_pos, 25, (100, 100, 255), -1)
        cv2.putText(img, "z", (z_pos[0]-8, z_pos[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # x̂ (decoded)
        xhat_pos = (cx, y0 + 250)
        cv2.circle(img, xhat_pos, 25, (200, 100, 100), -1)
        cv2.putText(img, "x", (xhat_pos[0]-12, xhat_pos[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "^", (xhat_pos[0]-5, xhat_pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # ẑ (re-encoded)
        zhat_pos = (cx, y0 + 350)
        cv2.circle(img, zhat_pos, 25, (200, 100, 200), -1)
        cv2.putText(img, "z", (zhat_pos[0]-12, zhat_pos[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "^", (zhat_pos[0]-5, zhat_pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Arrows
        cv2.arrowedLine(img, (x_pos[0], x_pos[1]+30), (z_pos[0], z_pos[1]-30), (150, 150, 150), 2)
        cv2.putText(img, "f", (cx + 15, y0 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        
        cv2.arrowedLine(img, (z_pos[0], z_pos[1]+30), (xhat_pos[0], xhat_pos[1]-30), (150, 150, 150), 2)
        cv2.putText(img, "g", (cx + 15, y0 + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 150), 1)
        
        cv2.arrowedLine(img, (xhat_pos[0], xhat_pos[1]+30), (zhat_pos[0], zhat_pos[1]-30), (150, 150, 150), 2)
        cv2.putText(img, "f", (cx + 15, y0 + 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        
        # Loss indicator (side arc from z to ẑ)
        cv2.ellipse(img, (cx - 60, (z_pos[1] + zhat_pos[1])//2), (40, 100), 0, -90, 90, (255, 200, 100), 2)
        cv2.putText(img, "||z-z||", (x0 + 20, y0 + 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(img, "^", (x0 + 56, y0 + 243), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 100), 1)
    
    def _render_latent_space(self, img, x0, y0, width, height):
        """Visualize z vs ẑ in latent space"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        cv2.putText(img, "LATENT SPACE", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Plot z as bar chart
        z_height = (height - 60) // 2
        bar_w = max(1, (width - 20) // len(self.current_z))
        
        cv2.putText(img, "z", (x0 + 10, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        for i, val in enumerate(self.current_z[:width//bar_w]):
            bx = x0 + 10 + i * bar_w
            by = y0 + 50 + z_height // 2
            bh = int(val * (z_height // 2 - 5))
            
            if val >= 0:
                cv2.rectangle(img, (bx, by - bh), (bx + bar_w - 1, by), (100, 100, 255), -1)
            else:
                cv2.rectangle(img, (bx, by), (bx + bar_w - 1, by - bh), (100, 100, 200), -1)
        
        # Plot ẑ below
        cv2.putText(img, "ẑ", (x0 + 10, y0 + 55 + z_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 200), 1)
        for i, val in enumerate(self.current_z_hat[:width//bar_w]):
            bx = x0 + 10 + i * bar_w
            by = y0 + 60 + z_height + z_height // 2
            bh = int(val * (z_height // 2 - 5))
            
            if val >= 0:
                cv2.rectangle(img, (bx, by - bh), (bx + bar_w - 1, by), (200, 100, 200), -1)
            else:
                cv2.rectangle(img, (bx, by), (bx + bar_w - 1, by - bh), (180, 100, 180), -1)
        
        # Difference indicator
        diff = np.linalg.norm(self.current_z - self.current_z_hat)
        cv2.putText(img, f"||z - ẑ|| = {diff:.4f}", (x0 + width - 120, y0 + height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    
    def _render_loss_history(self, img, x0, y0, width, height):
        """Plot loss over time"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "CONVERGENCE", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if len(self.loss_history) < 2:
            return
        
        losses = list(self.loss_history)
        max_loss = max(losses) + 0.1
        
        # Loss curve
        for i in range(1, len(losses)):
            x1 = x0 + 10 + int((i-1) * (width-20) / len(losses))
            x2 = x0 + 10 + int(i * (width-20) / len(losses))
            y1 = y0 + height - 20 - int(losses[i-1] / max_loss * (height - 50))
            y2 = y0 + height - 20 - int(losses[i] / max_loss * (height - 50))
            cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 2)
        
        # Rate reduction curve (if available)
        if len(self.rate_history) >= 2:
            rates = list(self.rate_history)
            max_rate = max(abs(r) for r in rates) + 0.1
            
            for i in range(1, len(rates)):
                x1 = x0 + 10 + int((i-1) * (width-20) / len(rates))
                x2 = x0 + 10 + int(i * (width-20) / len(rates))
                y1 = y0 + height//2 - int(rates[i-1] / max_rate * (height//4))
                y2 = y0 + height//2 - int(rates[i] / max_rate * (height//4))
                cv2.line(img, (x1, y1), (x2, y2), (100, 255, 100), 1)
        
        # Legend
        cv2.putText(img, "Loss", (x0 + width - 60, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
        cv2.putText(img, "ΔR", (x0 + width - 60, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        
        # Threshold line
        thresh_y = y0 + height - 20 - int(self.consistency_threshold / max_loss * (height - 50))
        cv2.line(img, (x0 + 10, thresh_y), (x0 + width - 10, thresh_y), (100, 100, 100), 1)
    
    def _render_assignments(self, img, x0, y0, width, height):
        """Subspace assignment bars"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "SUBSPACE ASSIGNMENTS", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        colors = [(255,100,100), (100,255,100), (100,100,255), (255,255,100), (255,100,255)]
        names = ['ATT', 'MEM', 'MOT', 'VIS', 'INT']
        bar_w = (width - 20) // self.n_subspaces - 5
        
        for j, (a, c, n) in enumerate(zip(self.current_assignments, colors, names)):
            bx = x0 + 10 + j * (bar_w + 5)
            by = y0 + height - 20
            bh = int(a * (height - 50))
            
            cv2.rectangle(img, (bx, by - bh), (bx + bar_w, by), c, -1)
            cv2.putText(img, n, (bx + 5, by + 15), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display
