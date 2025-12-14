"""
Rate Reduction Encoder - Ma's Compressive Transcription f(x) → z
================================================================
Implements the ENCODER side of Yi Ma's "Parsimony" principle.

FROM THE PAPER:
"The core idea is to seek the most compact representation z of data x
such that z lies on a low-dimensional manifold."

R(Z) = (1/2) * log det(I + (d/(n*ε²)) * Z @ Z.T)

This is the "coding rate" - the volume of the data cloud.
LOWER coding rate = data has been compressed to a simpler structure.

ARCHITECTURE:
1. Takes token stream from NeuralTransformer
2. Projects tokens into learned subspaces (one per cognitive state)
3. Computes Rate Reduction: ΔR = R(Z) - Σ R(Z_j)
4. High ΔR = tokens have organized into distinct subspaces = UNDERSTANDING

OUTPUTS:
- compressed_z: The low-dimensional representation (64-dim)
- coding_rate: R(Z) - total "messiness" of representation
- rate_reduction: ΔR - the GAIN from organization (the reward signal!)
- subspace_assignments: Which subspace each token belongs to
- manifold_image: Visualization of the learned structure

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

class RateReductionEncoder(BaseNode):
    """
    The f(x) → z mapping from Ma's framework.
    Compresses high-dimensional token streams into structured subspaces.
    """
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Rate Reduction Encoder"
    NODE_COLOR = QtGui.QColor(0, 150, 150)  # Teal - compression
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'token_stream': 'spectrum',      # From NeuralTransformer
            'context_vector': 'spectrum',    # Optional: pre-computed context
            'temperature': 'signal',          # Softmax temperature for assignments
            'learning_rate': 'signal',        # For online subspace learning
        }
        
        self.outputs = {
            'display': 'image',
            'compressed_z': 'spectrum',       # The low-dim representation
            'coding_rate': 'signal',          # R(Z) - total rate
            'rate_reduction': 'signal',       # ΔR - the gain!
            'subspace_assignments': 'spectrum', # Soft assignments to subspaces
            'manifold_image': 'image',        # Visualization
            'optimization_gate': 'signal',    # 1.0 when structure is found
        }
        
        # === DIMENSIONS ===
        self.input_dim = 64      # Token embedding dimension
        self.n_subspaces = 5     # Number of cognitive subspaces (like Ma's K classes)
        self.subspace_dim = 16   # Dimension of each subspace
        
        # === MA'S PARAMETERS ===
        self.epsilon = 0.5       # Error tolerance (ε in the paper)
        self.lambda_reg = 0.1    # Regularization for rate reduction
        
        # === LEARNED SUBSPACE BASES ===
        # U_j for each subspace - these get updated during learning
        np.random.seed(42)
        self.subspace_bases = []
        for j in range(self.n_subspaces):
            # Initialize as random orthonormal basis
            U = np.random.randn(self.input_dim, self.subspace_dim)
            U, _, _ = svd(U, full_matrices=False)
            self.subspace_bases.append(U[:, :self.subspace_dim])
        
        # === BUFFERS ===
        self.history_len = 100
        self.token_buffer = deque(maxlen=self.history_len)
        self.rate_history = deque(maxlen=500)
        self.reduction_history = deque(maxlen=500)
        
        # === CURRENT STATE ===
        self.current_z = np.zeros(self.input_dim)
        self.current_assignments = np.zeros(self.n_subspaces)
        self.current_rate = 0.0
        self.current_reduction = 0.0
        
        # === DISPLAY ===
        self._display = np.zeros((700, 1000, 3), dtype=np.uint8)
        self._manifold_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    def _sanitize_tokens(self, data):
        """Convert input to valid token array (N x 3: id, amplitude, phase)"""
        if data is None:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, str):
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, (list, tuple)):
            try:
                data = np.array(data)
            except:
                return np.zeros((0, 3), dtype=np.float32)
        if not hasattr(data, 'ndim'):
            return np.zeros((0, 3), dtype=np.float32)
        if data.ndim == 1:
            if len(data) == 3:
                return data.reshape(1, 3)
            elif len(data) == self.input_dim:
                # It's a context vector, not tokens
                return data.reshape(1, -1)
            return np.zeros((0, 3), dtype=np.float32)
        if data.ndim != 2:
            return np.zeros((0, 3), dtype=np.float32)
        return data.astype(np.float32)
    
    def _tokens_to_embedding(self, tokens):
        """Convert token array to embedding vector"""
        z = np.zeros(self.input_dim, dtype=np.float32)
        
        if len(tokens) == 0:
            return z
        
        # If already an embedding (from context_vector)
        if tokens.shape[1] == self.input_dim:
            return tokens[0]
        
        # Convert tokens to embedding
        for tok in tokens:
            if tok.shape[0] >= 3:
                token_id = int(tok[0]) % 20
                amplitude = tok[1]
                phase = tok[2]
                
                # Distribute across embedding dimensions
                # Token ID determines which dimensions are activated
                start_idx = (token_id * 3) % self.input_dim
                for i in range(3):
                    idx = (start_idx + i) % self.input_dim
                    z[idx] += amplitude * np.cos(phase + i * np.pi / 3)
        
        # Normalize
        norm = np.linalg.norm(z)
        if norm > 1e-9:
            z = z / norm
        
        return z
    
    def _compute_coding_rate(self, Z, eps):
        """
        Ma's Equation 2: R(Z) = (1/2) * log det(I + (d/(n*ε²)) * Z @ Z.T)
        
        Z: (d, n) matrix where d=features, n=samples
        eps: error tolerance
        
        Returns the "volume" of the data cloud in bits.
        """
        d, n = Z.shape
        if n == 0:
            return 0.0
        
        # Covariance-like term
        alpha = d / (n * eps**2)
        cov_term = alpha * (Z @ Z.T)
        
        # Log determinant (numerically stable)
        I = np.eye(d)
        sign, logdet = np.linalg.slogdet(I + cov_term)
        
        if sign <= 0:
            # Matrix is singular or negative, return high rate
            return 100.0
        
        return 0.5 * logdet
    
    def _compute_subspace_assignments(self, z, temperature=1.0):
        """
        Soft assignment of z to each subspace based on projection magnitude.
        Returns probability distribution over subspaces.
        """
        projections = np.zeros(self.n_subspaces)
        
        for j, U in enumerate(self.subspace_bases):
            # Project z onto subspace j
            z_proj = U @ (U.T @ z)
            projections[j] = np.linalg.norm(z_proj)
        
        # Softmax
        projections = projections / max(temperature, 0.1)
        exp_proj = np.exp(projections - np.max(projections))
        assignments = exp_proj / (np.sum(exp_proj) + 1e-9)
        
        return assignments
    
    def _compute_rate_reduction(self, Z, assignments_history):
        """
        Ma's Rate Reduction: ΔR = R(Z) - Σ_j (n_j/n) * R(Z_j)
        
        The GAIN from organizing data into subspaces.
        High ΔR = data has clear structure = intelligence!
        """
        d, n = Z.shape
        if n < 2:
            return 0.0, 0.0
        
        # Total coding rate
        R_total = self._compute_coding_rate(Z, self.epsilon)
        
        # Rate of each subspace
        R_subspaces = 0.0
        assignments = np.array(assignments_history)  # (n, K)
        
        for j in range(self.n_subspaces):
            # Weight of this subspace
            w_j = np.mean(assignments[:, j]) if len(assignments) > 0 else 1.0 / self.n_subspaces
            
            if w_j > 0.01:
                # Get samples belonging to this subspace (weighted)
                weights = assignments[:, j]
                Z_j = Z * weights  # Weighted samples
                R_j = self._compute_coding_rate(Z_j, self.epsilon)
                R_subspaces += w_j * R_j
        
        # Rate Reduction = how much we save by organizing
        delta_R = R_total - R_subspaces
        
        return R_total, delta_R
    
    def _update_subspaces(self, z, assignments, learning_rate=0.01):
        """
        Online learning: update subspace bases to better capture the data.
        This is gradient descent on the rate reduction objective.
        """
        for j, U in enumerate(self.subspace_bases):
            weight = assignments[j]
            if weight < 0.1:
                continue
            
            # Project z onto subspace
            z_proj = U @ (U.T @ z)
            residual = z - z_proj
            
            # Update basis to reduce residual (simplified gradient step)
            # This moves the subspace toward the data
            delta_U = learning_rate * weight * np.outer(residual, U.T @ z)
            
            # Add to basis (with size limiting)
            if np.linalg.norm(delta_U) < 1.0:
                self.subspace_bases[j] = U + delta_U[:, :self.subspace_dim]
                
                # Re-orthonormalize
                U_new, _, _ = svd(self.subspace_bases[j], full_matrices=False)
                self.subspace_bases[j] = U_new[:, :self.subspace_dim]
    
    def step(self):
        # Get inputs
        raw_tokens = self.get_blended_input('token_stream', 'mean')
        raw_context = self.get_blended_input('context_vector', 'mean')
        temp_val = self.get_blended_input('temperature', 'sum')
        lr_val = self.get_blended_input('learning_rate', 'sum')
        
        temperature = float(temp_val) if temp_val and temp_val > 0 else 1.0
        learning_rate = float(lr_val) if lr_val and lr_val > 0 else 0.01
        
        # Get embedding
        tokens = self._sanitize_tokens(raw_tokens)
        if raw_context is not None and hasattr(raw_context, 'shape'):
            z = raw_context.flatten()[:self.input_dim]
            if len(z) < self.input_dim:
                z = np.pad(z, (0, self.input_dim - len(z)))
        else:
            z = self._tokens_to_embedding(tokens)
        
        # Store in buffer
        self.token_buffer.append(z.copy())
        
        # Compute subspace assignments
        self.current_assignments = self._compute_subspace_assignments(z, temperature)
        
        # Build history matrix for rate calculation
        if len(self.token_buffer) >= 10:
            Z = np.array(list(self.token_buffer)).T  # (d, n)
            
            # Get assignment history
            assignments_hist = []
            for z_hist in self.token_buffer:
                a = self._compute_subspace_assignments(z_hist, temperature)
                assignments_hist.append(a)
            
            # Compute rate reduction
            self.current_rate, self.current_reduction = self._compute_rate_reduction(
                Z, assignments_hist
            )
            
            # Store history
            self.rate_history.append(self.current_rate)
            self.reduction_history.append(self.current_reduction)
            
            # Update subspaces (online learning)
            self._update_subspaces(z, self.current_assignments, learning_rate)
        
        # Current compressed representation
        self.current_z = z
        
        # Update outputs
        self.outputs['compressed_z'] = self.current_z.astype(np.float32)
        self.outputs['coding_rate'] = float(self.current_rate)
        self.outputs['rate_reduction'] = float(self.current_reduction)
        self.outputs['subspace_assignments'] = self.current_assignments.astype(np.float32)
        
        # Optimization gate: high when structure is found
        # Normalized rate reduction as gate signal
        if len(self.reduction_history) > 10:
            mean_reduction = np.mean(list(self.reduction_history)[-50:])
            gate = 1.0 if self.current_reduction > mean_reduction * 1.5 else 0.0
        else:
            gate = 0.0
        self.outputs['optimization_gate'] = gate
        
        # Render
        self._render_manifold()
        self.outputs['manifold_image'] = self._manifold_img
        self._render_display()
    
    def _render_manifold(self):
        """Visualize the learned subspace structure"""
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        center = (size // 2, size // 2)
        
        # Draw subspaces as sectors
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
        ]
        
        for j, (U, assignment, color) in enumerate(zip(
            self.subspace_bases, self.current_assignments, colors
        )):
            # Angle for this subspace
            angle_start = j * 2 * np.pi / self.n_subspaces - np.pi/2
            angle_end = (j + 1) * 2 * np.pi / self.n_subspaces - np.pi/2
            
            # Radius based on assignment strength
            radius = int(30 + assignment * 80)
            
            # Draw arc
            for angle in np.linspace(angle_start, angle_end, 20):
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                cv2.circle(img, (x, y), 3, color, -1)
            
            # Draw current z projection
            z_proj = U @ (U.T @ self.current_z)
            proj_mag = np.linalg.norm(z_proj)
            
            angle_mid = (angle_start + angle_end) / 2
            px = int(center[0] + proj_mag * 50 * np.cos(angle_mid))
            py = int(center[1] + proj_mag * 50 * np.sin(angle_mid))
            cv2.circle(img, (px, py), 5, (255, 255, 255), -1)
        
        # Draw center (the origin)
        cv2.circle(img, center, 8, (200, 200, 200), -1)
        
        # Rate reduction indicator
        cv2.putText(img, f"dR: {self.current_reduction:.2f}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"R: {self.current_rate:.2f}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        self._manifold_img = img
    
    def _render_display(self):
        """Full dashboard"""
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === LEFT: Manifold visualization ===
        manifold_resized = cv2.resize(self._manifold_img, (350, 350))
        img[30:380, 30:380] = manifold_resized
        cv2.putText(img, "SUBSPACE MANIFOLD", (30, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # === CENTER: Rate history ===
        self._render_rate_history(img, 400, 30, 350, 200)
        
        # === CENTER BOTTOM: Assignment bars ===
        self._render_assignment_bars(img, 400, 250, 350, 130)
        
        # === RIGHT: Current z visualization ===
        self._render_z_vector(img, 770, 30, 200, 350)
        
        # === BOTTOM: Statistics ===
        cv2.putText(img, f"Samples: {len(self.token_buffer)}", (30, h-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(img, f"Coding Rate R(Z): {self.current_rate:.4f}", (30, h-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
        cv2.putText(img, f"Rate Reduction ΔR: {self.current_reduction:.4f}", (30, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Gate indicator
        gate = self.outputs.get('optimization_gate', 0.0)
        gate_color = (100, 255, 100) if gate > 0.5 else (100, 100, 100)
        cv2.putText(img, f"STRUCTURE {'FOUND' if gate > 0.5 else 'searching...'}", 
                   (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_color, 1)
        
        self._display = img
    
    def _render_rate_history(self, img, x0, y0, width, height):
        """Plot rate and rate reduction over time"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        if len(self.rate_history) < 2:
            return
        
        # Plot rate (cyan)
        rates = list(self.rate_history)
        max_rate = max(rates) + 0.1
        for i in range(1, len(rates)):
            x1 = x0 + int((i-1) * width / len(rates))
            x2 = x0 + int(i * width / len(rates))
            y1 = y0 + height - 20 - int(rates[i-1] / max_rate * (height - 40))
            y2 = y0 + height - 20 - int(rates[i] / max_rate * (height - 40))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 100), 1)
        
        # Plot reduction (green)
        reductions = list(self.reduction_history)
        max_red = max(abs(r) for r in reductions) + 0.1
        for i in range(1, len(reductions)):
            x1 = x0 + int((i-1) * width / len(reductions))
            x2 = x0 + int(i * width / len(reductions))
            y1 = y0 + height//2 - int(reductions[i-1] / max_red * (height//2 - 20))
            y2 = y0 + height//2 - int(reductions[i] / max_red * (height//2 - 20))
            cv2.line(img, (x1, y1), (x2, y2), (100, 255, 100), 2)
        
        cv2.putText(img, "RATE HISTORY", (x0 + 10, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, "R(Z)", (x0 + width - 40, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 100), 1)
        cv2.putText(img, "ΔR", (x0 + width - 40, y0 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
    
    def _render_assignment_bars(self, img, x0, y0, width, height):
        """Render subspace assignment bars"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        bar_width = width // self.n_subspaces - 10
        colors = [(255,100,100), (100,255,100), (100,100,255), (255,255,100), (255,100,255)]
        names = ['ATT', 'MEM', 'MOT', 'VIS', 'INT']
        
        for j, (assignment, color, name) in enumerate(zip(
            self.current_assignments, colors, names
        )):
            bx = x0 + 5 + j * (bar_width + 10)
            by = y0 + height - 20
            bar_h = int(assignment * (height - 40))
            
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_width, by), color, -1)
            cv2.putText(img, name, (bx, by + 15),
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)
            cv2.putText(img, f"{assignment:.2f}", (bx, by - bar_h - 5),
                       cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)
        
        cv2.putText(img, "SUBSPACE ASSIGNMENTS", (x0 + 10, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _render_z_vector(self, img, x0, y0, width, height):
        """Visualize compressed representation z"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        # Draw z as a vertical bar chart
        bar_height = height // self.input_dim
        for i, val in enumerate(self.current_z):
            y = y0 + i * bar_height
            bar_w = int(abs(val) * (width - 20))
            
            if val >= 0:
                color = (100, 200, 100)
                cv2.rectangle(img, (x0 + width//2, y), (x0 + width//2 + bar_w, y + bar_height - 1), color, -1)
            else:
                color = (200, 100, 100)
                cv2.rectangle(img, (x0 + width//2 - bar_w, y), (x0 + width//2, y + bar_height - 1), color, -1)
        
        # Center line
        cv2.line(img, (x0 + width//2, y0), (x0 + width//2, y0 + height), (100, 100, 100), 1)
        
        cv2.putText(img, "z VECTOR", (x0 + 10, y0 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'manifold_image':
            return self._manifold_img
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display
