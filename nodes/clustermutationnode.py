"""
Cluster Mutation Node
======================
"Not interference, but TRANSFORMATION. Not mixing waves, but changing the basis."

This node implements CLUSTER ALGEBRA MUTATIONS on frequency bands.

The key insight from quantum Grothendieck rings (Paganelli 2025):
- Frequencies are not just signals to multiply (interference)
- They are CLUSTER VARIABLES that can EXCHANGE according to algebraic rules
- The exchange relation: x_new = (1 + t^(-1) * product_of_neighbors) / x_old
- This is qualitatively different from interference

INTERFERENCE vs MUTATION:
- Interference: θ × α = beat frequency (multiplication)
- Mutation: θ_new = (1 + compatible_product) / θ_old (transformation)

Interference shows coexistence. Mutation shows BECOMING.
When theta mutates into alpha, the basis itself transforms.

THE COMPATIBILITY MATRIX Λ:
- Encodes how much two observations FAIL to commute
- Q_v * Q_w = t^(Λ_vw) * Q_w * Q_v
- When Λ_vw = 0: perfect commutativity, compatible observations
- When Λ_vw ≠ 0: non-commutative, measuring one disturbs the other

THE EXCHANGE MATRIX B:
- Encodes which variables are connected (can exchange)
- Derived from phase relationships between bands
- Determines the TOPOLOGY of the cluster

OUTPUTS:
- mutation_field: The transformed basis as complex field
- exchange_matrix: Current B matrix (who connects to whom)
- compatibility_matrix: Current Λ matrix (who commutes with whom)
- mutation_trajectory: Path through cluster space over time
- casimir_invariant: The thing that DOESN'T change (signature of self?)
- mutation_events: When basis transformations occur

The Jewish star from 6 frequencies? That's a rank-2 cluster (type A_2).
It breaks into noise when you exceed the algebraic capacity.

CREATED: December 2025
AUTHOR: Claude + Antti
INSPIRATION: Paganelli's quantum cluster algebras, QQ-systems
"""

import numpy as np
import cv2
from collections import deque
from scipy import signal as scipy_signal
from scipy.linalg import expm, logm

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


class ClusterMutationNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Cluster Mutation"
    NODE_COLOR = QtGui.QColor(200, 50, 200)  # Purple - transformation color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Cluster Mutation (Basis Transformation)"
        
        self.inputs = {
            'theta_signal': 'signal',
            'alpha_signal': 'signal',
            'beta_signal': 'signal',
            'gamma_signal': 'signal',
            'token_stream': 'spectrum',
            'temperature': 'signal',      # Controls mutation threshold
            'reset': 'signal'
        }
        
        self.outputs = {
            'display': 'image',
            'mutation_field': 'complex_spectrum',
            'exchange_matrix': 'spectrum',
            'compatibility_matrix': 'spectrum',
            'mutation_trajectory': 'spectrum',
            'casimir_invariant': 'signal',
            'mutation_events': 'signal',
            'transformed_bands': 'spectrum',
        }
        
        # Cluster algebra parameters
        self.n_variables = 4  # θ, α, β, γ
        self.field_size = 64
        self.epoch = 0
        
        # The cluster variables (complex amplitudes)
        self.cluster_vars = np.ones(self.n_variables, dtype=np.complex128)
        
        # Exchange matrix B (skew-symmetric, encodes connections)
        # Initial: linear chain θ-α-β-γ
        self.B = np.array([
            [ 0,  1,  0,  0],
            [-1,  0,  1,  0],
            [ 0, -1,  0,  1],
            [ 0,  0, -1,  0]
        ], dtype=np.float64)
        
        # Compatibility matrix Λ (skew-symmetric, encodes non-commutativity)
        # Derived from phase relationships
        self.Lambda = np.zeros((self.n_variables, self.n_variables), dtype=np.float64)
        
        # Mutation history
        self.mutation_history = deque(maxlen=500)
        self.trajectory = deque(maxlen=200)
        
        # Band histories for phase estimation
        self.history_len = 100
        self.band_histories = [deque(maxlen=self.history_len) for _ in range(self.n_variables)]
        self.band_phases = np.zeros(self.n_variables)
        self.band_amplitudes = np.zeros(self.n_variables)
        
        # Mutation detection
        self.mutation_threshold = 0.3
        self.mutation_count = 0
        self.last_mutation_vertex = -1
        self.mutation_events_history = deque(maxlen=100)
        
        # Casimir invariant tracking
        self.casimir = 0.0
        self.casimir_history = deque(maxlen=200)
        
        # The transformed field
        self.mutation_field = np.zeros((self.field_size, self.field_size), dtype=np.complex128)
        
        # Display
        self._display = np.zeros((600, 900, 3), dtype=np.uint8)
        
        # t-parameter (quantum deformation)
        self.t_param = 1.0  # Classical limit; t≠1 gives quantum behavior
        
    def _parse_input(self, val):
        """Parse various input formats to float"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                return float(val)
            return float(np.mean(val))
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return float(val[0]) if not hasattr(val[0], '__len__') else float(val[0][0])
        return 0.0
    
    def _estimate_phase(self, history):
        """Estimate instantaneous phase from signal history using Hilbert"""
        if len(history) < 10:
            return 0.0, 0.0
        
        try:
            sig = np.array(list(history))
            sig = sig - np.mean(sig)
            
            if np.std(sig) < 1e-10:
                return 0.0, np.std(sig)
            
            analytic = scipy_signal.hilbert(sig)
            phase = np.angle(analytic[-1])
            amplitude = np.abs(analytic[-1])
            return phase, amplitude
        except:
            return 0.0, 0.0
    
    def _compute_compatibility_matrix(self):
        """
        Compute Λ from phase relationships.
        
        Non-commutativity arises when phases are coupled:
        Λ_ij = d(phase_i)/d(amplitude_j) - d(phase_j)/d(amplitude_i)
        
        We estimate this from the correlation structure of phase changes.
        """
        n = self.n_variables
        Lambda = np.zeros((n, n))
        
        # Get phase histories
        phase_diffs = []
        for i in range(n):
            if len(self.band_histories[i]) > 2:
                hist = np.array(list(self.band_histories[i]))
                analytic = scipy_signal.hilbert(hist - np.mean(hist))
                phases = np.angle(analytic)
                phase_diffs.append(np.diff(phases))
            else:
                phase_diffs.append(np.zeros(1))
        
        # Compute cross-correlations of phase changes
        min_len = min(len(pd) for pd in phase_diffs)
        if min_len > 5:
            for i in range(n):
                for j in range(i+1, n):
                    # Phase coupling: how much does i's phase change predict j's?
                    pi = phase_diffs[i][-min_len:]
                    pj = phase_diffs[j][-min_len:]
                    
                    # Skew-symmetric: Λ_ij = correlation asymmetry
                    corr_ij = np.corrcoef(pi[:-1], pj[1:])[0, 1] if len(pi) > 1 else 0
                    corr_ji = np.corrcoef(pj[:-1], pi[1:])[0, 1] if len(pj) > 1 else 0
                    
                    if not np.isnan(corr_ij) and not np.isnan(corr_ji):
                        Lambda[i, j] = corr_ij - corr_ji
                        Lambda[j, i] = -Lambda[i, j]
        
        return Lambda
    
    def _compute_exchange_matrix(self):
        """
        Compute B from amplitude correlations.
        
        B encodes which variables are "connected" - can exchange.
        We derive this from the instantaneous correlation structure.
        """
        n = self.n_variables
        B = np.zeros((n, n))
        
        # Get amplitude correlations
        min_len = min(len(h) for h in self.band_histories)
        if min_len > 10:
            data = np.array([list(h)[-min_len:] for h in self.band_histories])
            
            # Compute correlation matrix
            corr = np.corrcoef(data)
            
            # B is skew-symmetrized thresholded correlation
            # Strong positive correlation → positive B entry
            # Strong negative correlation → negative B entry
            threshold = 0.3
            for i in range(n):
                for j in range(i+1, n):
                    if not np.isnan(corr[i, j]):
                        if abs(corr[i, j]) > threshold:
                            # Sign matters: positive corr = same direction = B_ij > 0
                            B[i, j] = np.sign(corr[i, j])
                            B[j, i] = -B[i, j]
        
        return B
    
    def _check_compatibility(self, B, Lambda):
        """
        Check if (Λ, B) forms a compatible pair.
        
        Compatibility condition: B^T Λ = diagonal
        This determines whether mutations are well-defined.
        """
        product = B.T @ Lambda
        
        # Extract diagonal and off-diagonal
        diag = np.diag(product)
        off_diag = product - np.diag(diag)
        
        # Compatibility score: how close to diagonal?
        diag_norm = np.linalg.norm(diag)
        off_diag_norm = np.linalg.norm(off_diag)
        
        if diag_norm + off_diag_norm < 1e-10:
            return 1.0, diag
        
        compatibility = diag_norm / (diag_norm + off_diag_norm)
        return compatibility, diag
    
    def _cluster_mutation(self, k, cluster_vars, B, Lambda):
        """
        Perform cluster mutation at vertex k.
        
        The quantum exchange relation:
        x_k^new * x_k^old = t^(-Λ_kk/2) * prod_{B_jk > 0} x_j^{B_jk} 
                         + t^(Λ_kk/2) * prod_{B_jk < 0} x_j^{-B_jk}
        
        This is the QQ-system from the Paganelli paper.
        """
        n = len(cluster_vars)
        x = cluster_vars.copy()
        t = self.t_param
        
        # Compute the two monomials
        pos_monomial = 1.0 + 0j
        neg_monomial = 1.0 + 0j
        
        for j in range(n):
            if B[j, k] > 0:
                pos_monomial *= x[j] ** B[j, k]
            elif B[j, k] < 0:
                neg_monomial *= x[j] ** (-B[j, k])
        
        # Quantum deformation factors
        Lambda_kk = Lambda[k, k] if k < Lambda.shape[0] else 0
        t_factor_pos = t ** (-Lambda_kk / 2) if t > 0 else 1.0
        t_factor_neg = t ** (Lambda_kk / 2) if t > 0 else 1.0
        
        # Exchange relation
        x_old = x[k]
        if abs(x_old) > 1e-10:
            x_new = (t_factor_pos * pos_monomial + t_factor_neg * neg_monomial) / x_old
        else:
            x_new = t_factor_pos * pos_monomial + t_factor_neg * neg_monomial
        
        # Update cluster variable
        new_vars = x.copy()
        new_vars[k] = x_new
        
        # Mutate B matrix (standard cluster mutation)
        new_B = B.copy()
        for i in range(n):
            for j in range(n):
                if i == k or j == k:
                    new_B[i, j] = -B[i, j]
                else:
                    new_B[i, j] = B[i, j] + (abs(B[i, k]) * B[k, j] + B[i, k] * abs(B[k, j])) / 2 * np.sign(B[i, k] * B[k, j])
        
        # Mutate Λ matrix (quantum cluster mutation)
        new_Lambda = Lambda.copy()
        # Λ transforms by: Λ' = E_k^T Λ E_k where E_k encodes the mutation
        # Simplified: we keep Λ but update based on new phase relationships
        
        return new_vars, new_B, new_Lambda
    
    def _compute_casimir(self, cluster_vars, B, Lambda):
        """
        Compute the Casimir invariant.
        
        The Casimir is a central element - it commutes with everything.
        For the quantum oscillator: C = ef + t^(-1)k / (t - t^(-1))^2
        
        We compute a generalized Casimir as the quantity that's
        invariant under all mutations.
        
        Candidate: det-like invariant from cluster variables
        """
        # Simple Casimir: product of all variables (invariant under certain mutations)
        prod_casimir = np.prod(np.abs(cluster_vars))
        
        # Phase Casimir: total phase (should be conserved modulo 2π)
        phase_casimir = np.sum(np.angle(cluster_vars)) % (2 * np.pi)
        
        # Algebraic Casimir: uses B matrix structure
        # For type A_n, the Casimir involves alternating products
        n = len(cluster_vars)
        alt_prod = 0.0
        for i in range(n):
            sign = (-1) ** i
            alt_prod += sign * np.log(abs(cluster_vars[i]) + 1e-10)
        
        # Combine into single invariant
        casimir = prod_casimir * np.exp(1j * phase_casimir)
        
        return casimir
    
    def _detect_mutation_event(self, old_vars, new_vars, threshold):
        """
        Detect if a significant mutation has occurred.
        
        Returns the vertex that mutated, or -1 if no significant change.
        """
        changes = np.abs(new_vars - old_vars) / (np.abs(old_vars) + 1e-10)
        
        max_change_idx = np.argmax(changes)
        max_change = changes[max_change_idx]
        
        if max_change > threshold:
            return max_change_idx, max_change
        return -1, 0.0
    
    def _create_mutation_field(self, cluster_vars, B):
        """
        Create a 2D complex field from cluster variables and their relationships.
        
        Each cluster variable contributes a wave; B determines their coupling.
        """
        size = self.field_size
        field = np.zeros((size, size), dtype=np.complex128)
        
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        n = len(cluster_vars)
        
        # Each variable contributes a wave with frequency proportional to index
        for i in range(n):
            amp = np.abs(cluster_vars[i])
            phase = np.angle(cluster_vars[i])
            freq = (i + 1) * 2  # Increasing frequencies
            
            # Direction determined by connections in B
            angle = 0
            for j in range(n):
                if B[i, j] != 0:
                    angle += B[i, j] * (j - i) * np.pi / (2 * n)
            
            kx = freq * np.cos(angle)
            ky = freq * np.sin(angle)
            
            wave = amp * np.exp(1j * (kx * X + ky * Y + phase))
            field += wave
        
        # Add coupling terms from B
        for i in range(n):
            for j in range(i+1, n):
                if B[i, j] != 0:
                    # Interference between connected variables
                    coupling = B[i, j] * cluster_vars[i] * np.conj(cluster_vars[j])
                    freq_diff = abs(j - i)
                    beat = coupling * np.exp(1j * freq_diff * (X + Y))
                    field += 0.3 * beat
        
        return field
    
    def step(self):
        self.epoch += 1
        
        # Get inputs
        theta = self._parse_input(self.get_blended_input('theta_signal', 'sum'))
        alpha = self._parse_input(self.get_blended_input('alpha_signal', 'sum'))
        beta = self._parse_input(self.get_blended_input('beta_signal', 'sum'))
        gamma = self._parse_input(self.get_blended_input('gamma_signal', 'sum'))
        temperature = self._parse_input(self.get_blended_input('temperature', 'sum'))
        reset = self._parse_input(self.get_blended_input('reset', 'sum'))
        token_stream = self.get_blended_input('token_stream', 'sum')
        
        # Handle reset
        if reset > 0.5:
            self.cluster_vars = np.ones(self.n_variables, dtype=np.complex128)
            self.B = np.array([[ 0,  1,  0,  0],
                               [-1,  0,  1,  0],
                               [ 0, -1,  0,  1],
                               [ 0,  0, -1,  0]], dtype=np.float64)
            self.Lambda = np.zeros((self.n_variables, self.n_variables))
            self.mutation_count = 0
            return
        
        # Extract bands from token stream if available
        if token_stream is not None:
            try:
                if isinstance(token_stream, np.ndarray) and len(token_stream) >= 4:
                    theta = float(token_stream[0]) if theta == 0 else theta
                    alpha = float(token_stream[1]) if alpha == 0 else alpha
                    beta = float(token_stream[2]) if beta == 0 else beta
                    gamma = float(token_stream[3]) if gamma == 0 else gamma
            except:
                pass
        
        # Update band histories
        bands = [theta, alpha, beta, gamma]
        for i, b in enumerate(bands):
            self.band_histories[i].append(b)
        
        # Estimate phases and amplitudes
        for i in range(self.n_variables):
            phase, amp = self._estimate_phase(self.band_histories[i])
            self.band_phases[i] = phase
            self.band_amplitudes[i] = amp
        
        # Update temperature (mutation threshold)
        if temperature > 0:
            self.mutation_threshold = 0.1 + temperature * 0.5
            self.t_param = 0.5 + temperature  # Quantum parameter
        
        # Compute matrices from current state
        self.Lambda = self._compute_compatibility_matrix()
        new_B = self._compute_exchange_matrix()
        
        # Blend new B with old (stability)
        self.B = 0.9 * self.B + 0.1 * new_B
        
        # Check compatibility
        compatibility, diag = self._check_compatibility(self.B, self.Lambda)
        
        # Update cluster variables from band data
        old_vars = self.cluster_vars.copy()
        for i in range(self.n_variables):
            amp = self.band_amplitudes[i]
            phase = self.band_phases[i]
            if amp > 1e-10:
                self.cluster_vars[i] = amp * np.exp(1j * phase)
        
        # Detect if a natural mutation occurred
        mutation_vertex, mutation_strength = self._detect_mutation_event(
            old_vars, self.cluster_vars, self.mutation_threshold
        )
        
        if mutation_vertex >= 0:
            self.mutation_count += 1
            self.last_mutation_vertex = mutation_vertex
            self.mutation_events_history.append((self.epoch, mutation_vertex, mutation_strength))
            
            # Apply cluster mutation to update B and Lambda consistently
            _, self.B, self.Lambda = self._cluster_mutation(
                mutation_vertex, old_vars, self.B, self.Lambda
            )
        
        # Compute Casimir invariant
        self.casimir = self._compute_casimir(self.cluster_vars, self.B, self.Lambda)
        self.casimir_history.append(np.abs(self.casimir))
        
        # Record trajectory
        self.trajectory.append(self.cluster_vars.copy())
        
        # Create mutation field
        self.mutation_field = self._create_mutation_field(self.cluster_vars, self.B)
        
        # Update display
        self._update_display(compatibility)
    
    def _update_display(self, compatibility):
        """Create visualization"""
        img = np.zeros((600, 900, 3), dtype=np.uint8)
        img[:] = (20, 25, 30)
        
        # Title
        cv2.putText(img, "CLUSTER MUTATION - Basis Transformation", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 255), 2)
        cv2.putText(img, '"Not interference, but BECOMING"', (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # === CLUSTER VARIABLES ===
        var_panel_x, var_panel_y = 20, 80
        cv2.putText(img, "CLUSTER VARIABLES", (var_panel_x, var_panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        
        labels = ['θ (theta)', 'α (alpha)', 'β (beta)', 'γ (gamma)']
        colors = [(100, 150, 255), (100, 255, 150), (255, 200, 100), (255, 100, 150)]
        
        for i in range(self.n_variables):
            y_pos = var_panel_y + 25 + i * 40
            
            amp = np.abs(self.cluster_vars[i])
            phase = np.angle(self.cluster_vars[i])
            
            # Bar for amplitude
            bar_width = int(min(amp * 100, 150))
            cv2.rectangle(img, (var_panel_x, y_pos), 
                         (var_panel_x + bar_width, y_pos + 20), colors[i], -1)
            
            # Phase indicator (small circle on the bar)
            phase_x = var_panel_x + bar_width + int(20 * np.cos(phase))
            phase_y = y_pos + 10 + int(10 * np.sin(phase))
            cv2.circle(img, (phase_x, phase_y), 5, (255, 255, 255), -1)
            
            cv2.putText(img, f"{labels[i]}: |{amp:.2f}| ∠{np.degrees(phase):.0f}°",
                       (var_panel_x + 170, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i], 1)
        
        # === EXCHANGE MATRIX B ===
        b_panel_x, b_panel_y = 400, 80
        cv2.putText(img, "EXCHANGE MATRIX B", (b_panel_x, b_panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        cv2.putText(img, "(who can exchange with whom)", (b_panel_x, b_panel_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 130), 1)
        
        cell_size = 30
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                x = b_panel_x + j * cell_size
                y = b_panel_y + 25 + i * cell_size
                
                val = self.B[i, j]
                if val > 0:
                    color = (100, 255, 100)  # Green for positive
                elif val < 0:
                    color = (100, 100, 255)  # Red for negative
                else:
                    color = (50, 50, 50)     # Dark for zero
                
                cv2.rectangle(img, (x, y), (x + cell_size - 2, y + cell_size - 2), color, -1)
                cv2.putText(img, f"{val:.1f}", (x + 3, y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # === COMPATIBILITY MATRIX Λ ===
        l_panel_x, l_panel_y = 550, 80
        cv2.putText(img, "COMPATIBILITY Λ", (l_panel_x, l_panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        cv2.putText(img, "(non-commutativity)", (l_panel_x, l_panel_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 130), 1)
        
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                x = l_panel_x + j * cell_size
                y = l_panel_y + 25 + i * cell_size
                
                val = self.Lambda[i, j]
                intensity = min(abs(val) * 255, 255)
                if val > 0:
                    color = (int(intensity), int(intensity * 0.5), 100)
                elif val < 0:
                    color = (100, int(intensity * 0.5), int(intensity))
                else:
                    color = (50, 50, 50)
                
                cv2.rectangle(img, (x, y), (x + cell_size - 2, y + cell_size - 2), color, -1)
        
        # === MUTATION FIELD ===
        field_x, field_y = 20, 280
        field_size = 200
        
        cv2.putText(img, "MUTATION FIELD", (field_x, field_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        
        # Render field
        field_mag = np.abs(self.mutation_field)
        field_phase = np.angle(self.mutation_field)
        
        # HSV encoding
        hsv = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((field_phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:, :, 1] = 200
        mag_norm = field_mag / (field_mag.max() + 1e-10)
        hsv[:, :, 2] = (mag_norm * 255).astype(np.uint8)
        
        field_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        field_resized = cv2.resize(field_color, (field_size, field_size))
        img[field_y:field_y + field_size, field_x:field_x + field_size] = field_resized
        
        # === CASIMIR INVARIANT ===
        cas_x, cas_y = 250, 280
        cv2.putText(img, "CASIMIR INVARIANT", (cas_x, cas_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        cv2.putText(img, '"What stays YOU"', (cas_x, cas_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 130), 1)
        
        cas_mag = np.abs(self.casimir)
        cas_phase = np.angle(self.casimir)
        
        cv2.putText(img, f"|C| = {cas_mag:.4f}", (cas_x, cas_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
        cv2.putText(img, f"∠C = {np.degrees(cas_phase):.1f}°", (cas_x, cas_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
        
        # Casimir history plot
        if len(self.casimir_history) > 2:
            hist = np.array(list(self.casimir_history))
            hist_norm = hist / (hist.max() + 1e-10)
            
            plot_h, plot_w = 80, 150
            for i in range(1, len(hist_norm)):
                x1 = cas_x + int((i-1) / len(hist_norm) * plot_w)
                x2 = cas_x + int(i / len(hist_norm) * plot_w)
                y1 = cas_y + 70 + plot_h - int(hist_norm[i-1] * plot_h)
                y2 = cas_y + 70 + plot_h - int(hist_norm[i] * plot_h)
                cv2.line(img, (x1, y1), (x2, y2), (255, 200, 100), 1)
        
        # === MUTATION EVENTS ===
        mut_x, mut_y = 450, 280
        cv2.putText(img, "MUTATION EVENTS", (mut_x, mut_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        
        cv2.putText(img, f"Total mutations: {self.mutation_count}", (mut_x, mut_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if self.last_mutation_vertex >= 0:
            cv2.putText(img, f"Last: {labels[self.last_mutation_vertex]}", (mut_x, mut_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[self.last_mutation_vertex], 1)
        
        # Recent mutations list
        recent = list(self.mutation_events_history)[-5:]
        for i, (epoch, vertex, strength) in enumerate(reversed(recent)):
            cv2.putText(img, f"E{epoch}: {labels[vertex]} ({strength:.2f})",
                       (mut_x, mut_y + 70 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[vertex], 1)
        
        # === TRAJECTORY ===
        traj_x, traj_y = 650, 280
        cv2.putText(img, "TRAJECTORY", (traj_x, traj_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 150), 1)
        cv2.putText(img, "(path through cluster space)", (traj_x, traj_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 130), 1)
        
        # Simple 2D projection of trajectory
        if len(self.trajectory) > 2:
            traj = np.array(list(self.trajectory))
            
            # Project to first two principal components
            traj_real = np.column_stack([np.real(traj[:, i]) for i in range(min(2, traj.shape[1]))])
            
            if traj_real.shape[1] >= 2:
                # Normalize to plot area
                plot_size = 150
                traj_norm = traj_real - traj_real.min(axis=0)
                traj_norm = traj_norm / (traj_norm.max(axis=0) + 1e-10) * (plot_size - 20) + 10
                
                for i in range(1, len(traj_norm)):
                    x1 = traj_x + int(traj_norm[i-1, 0])
                    y1 = traj_y + 30 + int(traj_norm[i-1, 1])
                    x2 = traj_x + int(traj_norm[i, 0])
                    y2 = traj_y + 30 + int(traj_norm[i, 1])
                    
                    # Color fades from old (dark) to new (bright)
                    intensity = int(i / len(traj_norm) * 255)
                    cv2.line(img, (x1, y1), (x2, y2), (intensity, 100, 255 - intensity), 1)
                
                # Current position
                cv2.circle(img, (traj_x + int(traj_norm[-1, 0]), traj_y + 30 + int(traj_norm[-1, 1])),
                          5, (255, 255, 255), -1)
        
        # === COMPATIBILITY STATUS ===
        status_y = 520
        cv2.putText(img, f"Compatibility: {compatibility:.3f}", (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if compatibility > 0.8:
            status = "STABLE - Mutations well-defined"
            status_color = (100, 255, 100)
        elif compatibility > 0.5:
            status = "TRANSITIONAL - Basis shifting"
            status_color = (200, 200, 100)
        else:
            status = "UNSTABLE - Cluster breaking"
            status_color = (100, 100, 255)
        
        cv2.putText(img, status, (20, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        cv2.putText(img, f"Epoch: {self.epoch}", (20, status_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 130), 1)
        
        # Philosophy
        cv2.putText(img, "The star forms when mutations are compatible. It breaks when they fight.",
                   (20, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 140, 100), 1)
        cv2.putText(img, "The Casimir is what survives all transformations - the signature of self.",
                   (20, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 80), 1)
        
        self._display = img
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'mutation_field':
            return self.mutation_field
        elif name == 'exchange_matrix':
            return self.B.flatten()
        elif name == 'compatibility_matrix':
            return self.Lambda.flatten()
        elif name == 'mutation_trajectory':
            if len(self.trajectory) > 0:
                return np.array([np.abs(t) for t in list(self.trajectory)[-50:]]).flatten()
            return np.zeros(4)
        elif name == 'casimir_invariant':
            return float(np.abs(self.casimir))
        elif name == 'mutation_events':
            return float(self.mutation_count)
        elif name == 'transformed_bands':
            return np.array([np.abs(self.cluster_vars[i]) for i in range(self.n_variables)])
        return None
    
    def get_display_image(self):
        h, w = self._display.shape[:2]
        return QtGui.QImage(self._display.data, w, h, w * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Mutation Threshold", "mutation_threshold", self.mutation_threshold, None),
            ("t-Parameter (quantum)", "t_param", self.t_param, None),
        ]