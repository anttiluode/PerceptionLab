"""
Cortical Learning System: Bidirectional Geometry ↔ Resonance Loop
==================================================================

Implements the closed loop discovered in the lab:
  FORWARD:  Geometry → Eigenmodes → Signal routing
  BACKWARD: Prediction error → Hebbian plasticity → Geometry modification

Timescales (biologically realistic):
  - Signal propagation: ~10ms (every step)
  - Synaptic plasticity: ~100ms-hours (every 10-100 steps)
  - Geometry evolution: days-weeks (every 1000+ steps)

Based on:
  - Raj et al. (2020): Brain eigenmodes from Laplacian
  - Friston (2010): Predictive coding / free energy
  - Hebb (1949): "Cells that fire together wire together"
  - Van Essen (1997): Axonal tension theory of cortical folding

The key insight: New signals MUST flow through existing geometry.
They get captured by the nearest eigenmode attractor.
Learning slowly reshapes which attractors exist.
"""

import numpy as np
import cv2
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from collections import deque

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None
        def set_output(self, name, val): pass


class CorticalSubstrateNode(BaseNode):
    """
    The Physical Cortical Geometry.
    
    Represents a patch of cortex as a graph:
    - Nodes = cortical columns (~1mm patches, ~100k neurons each)
    - Edges = white matter connections (weighted by fiber count)
    - Positions = 2D embedding (could extend to 3D for real sulci/gyri)
    
    Computes:
    - Graph Laplacian L = D - W
    - Eigenmodes φ_k with eigenvalues λ_k
    - Eigenvalues = resonant frequencies the geometry supports
    
    The geometry can be:
    - Initialized randomly
    - Loaded from template (like a real brain parcellation)
    - Evolved through developmental plasticity
    """
    NODE_CATEGORY = "Cortical Learning"
    NODE_COLOR = QtGui.QColor(100, 150, 200)  # Cortical blue-gray
    
    def __init__(self):
        super().__init__()
        self.node_title = "Cortical Substrate"
        
        self.inputs = {
            'activity_pattern': 'spectrum',   # Current neural activity
            'plasticity_signal': 'signal',    # Triggers geometry update
            'external_geometry': 'spectrum'   # Optional: load geometry
        }
        
        self.outputs = {
            'eigenmodes': 'tensor',           # The φ_k basis functions
            'eigenvalues': 'spectrum',        # The λ_k frequencies
            'laplacian': 'tensor',            # The L matrix
            'connectivity': 'tensor',         # The W matrix
            'positions': 'tensor',            # Node positions (x,y)
            'geometry_view': 'image'
        }
        
        # Geometry parameters
        self.n_nodes = 64  # Cortical columns
        self.n_modes = 16  # Eigenmodes to compute
        
        # Initialize geometry
        self._init_geometry()
        
        # Plasticity parameters
        self.geometry_plasticity_rate = 0.001  # Very slow!
        self.connection_decay = 0.9999  # Slight decay prevents runaway
        self.min_connection = 0.01
        self.max_connection = 2.0
        
        # Activity history for Hebbian geometry learning
        self.activity_history = deque(maxlen=100)
        self.step_count = 0
        
        self.display = np.zeros((256, 256, 3), dtype=np.uint8)
        
    def _init_geometry(self):
        """Initialize cortical geometry - quasi-random with structure."""
        np.random.seed(42)  # Reproducible
        
        # Node positions: arrange in a folded pattern
        # This creates a 2D "cortical surface" with some folding
        t = np.linspace(0, 4*np.pi, self.n_nodes)
        
        # Base circle with perturbation (simulates sulci/gyri)
        radius = 0.8 + 0.2 * np.sin(3*t) + 0.1 * np.sin(7*t)
        self.positions = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t)
        ])
        
        # Add some noise to break perfect symmetry
        self.positions += np.random.randn(self.n_nodes, 2) * 0.05
        
        # Build connectivity based on distance (local connections stronger)
        self.connectivity = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                
                # Connection strength decays with distance
                # But also depends on "cortical distance" (along the surface)
                cortical_dist = min(abs(i-j), self.n_nodes - abs(i-j)) / self.n_nodes
                
                # Combine spatial and cortical distance
                effective_dist = 0.5 * dist + 0.5 * cortical_dist
                
                # Gaussian falloff with some long-range connections
                strength = np.exp(-effective_dist**2 / 0.1)
                
                # Add occasional long-range connections (like corpus callosum)
                if np.random.random() < 0.05:
                    strength += 0.3 * np.random.random()
                
                self.connectivity[i, j] = strength
                self.connectivity[j, i] = strength
        
        # Normalize
        self.connectivity = self.connectivity / (np.max(self.connectivity) + 1e-9)
        
        # Compute initial eigenmodes
        self._compute_eigenmodes()
        
    def _compute_eigenmodes(self):
        """Compute eigenmodes of the graph Laplacian."""
        # Degree matrix
        D = np.diag(np.sum(self.connectivity, axis=1))
        
        # Laplacian
        self.laplacian = D - self.connectivity
        
        # Normalized Laplacian for stability
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-9))
        L_norm = D_inv_sqrt @ self.laplacian @ D_inv_sqrt
        
        # Compute eigenmodes
        eigenvalues, eigenvectors = eigh(L_norm)
        
        self.eigenvalues = eigenvalues[:self.n_modes]
        self.eigenmodes = eigenvectors[:, :self.n_modes]
        
    def _update_geometry_from_activity(self, activity):
        """
        Hebbian geometry plasticity: connections that consistently
        co-activate strengthen, others weaken.
        
        This is the SLOW timescale - geometry evolution.
        In real brains, this happens over days/weeks.
        """
        if activity is None or len(activity) != self.n_nodes:
            return
            
        # Record activity
        self.activity_history.append(activity.copy())
        
        # Only update geometry every N steps (slow timescale)
        if self.step_count % 100 != 0:
            return
            
        if len(self.activity_history) < 50:
            return
            
        # Compute activity correlation matrix
        activity_matrix = np.array(list(self.activity_history))
        
        # Correlation of activity patterns
        activity_centered = activity_matrix - np.mean(activity_matrix, axis=0)
        correlation = activity_centered.T @ activity_centered
        correlation = correlation / (len(self.activity_history) + 1e-9)
        
        # Normalize
        norms = np.sqrt(np.diag(correlation) + 1e-9)
        correlation = correlation / np.outer(norms, norms)
        
        # Hebbian update: strengthen connections between correlated nodes
        delta_W = self.geometry_plasticity_rate * correlation
        
        # Apply update with bounds
        self.connectivity = self.connectivity * self.connection_decay + delta_W
        self.connectivity = np.clip(self.connectivity, 
                                    self.min_connection, 
                                    self.max_connection)
        
        # Keep symmetric
        self.connectivity = (self.connectivity + self.connectivity.T) / 2
        np.fill_diagonal(self.connectivity, 0)
        
        # Recompute eigenmodes (geometry changed!)
        self._compute_eigenmodes()
        
    def step(self):
        self.step_count += 1
        
        # Get activity pattern
        activity = self.get_blended_input('activity_pattern', 'mean')
        if activity is not None:
            if len(activity) < self.n_nodes:
                # Pad or interpolate
                activity = np.interp(
                    np.linspace(0, 1, self.n_nodes),
                    np.linspace(0, 1, len(activity)),
                    activity
                )
            elif len(activity) > self.n_nodes:
                activity = activity[:self.n_nodes]
                
            self._update_geometry_from_activity(activity)
        
        # Check for external geometry update
        ext_geom = self.get_blended_input('external_geometry', 'mean')
        if ext_geom is not None and len(ext_geom) == self.n_nodes:
            # Use external geometry to perturb positions
            perturbation = (ext_geom - 0.5) * 0.1
            angles = np.linspace(0, 2*np.pi, self.n_nodes)
            self.positions[:, 0] += perturbation * np.cos(angles)
            self.positions[:, 1] += perturbation * np.sin(angles)
            self._compute_eigenmodes()
        
        # Visualization
        self._draw_geometry()
        
    def _draw_geometry(self):
        """Visualize the cortical geometry."""
        h, w = 256, 256
        self.display.fill(0)
        
        # Map positions to image coordinates
        pos_normalized = (self.positions - self.positions.min(axis=0)) / \
                        (self.positions.max(axis=0) - self.positions.min(axis=0) + 1e-9)
        pos_img = (pos_normalized * (np.array([w, h]) - 40) + 20).astype(int)
        
        # Draw connections (colored by strength)
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                strength = self.connectivity[i, j]
                if strength > 0.1:
                    color = (int(50 + strength * 150), 
                            int(50 + strength * 100), 
                            int(50 + strength * 50))
                    thickness = max(1, int(strength * 3))
                    cv2.line(self.display, 
                            tuple(pos_img[i]), 
                            tuple(pos_img[j]), 
                            color, thickness)
        
        # Draw nodes (colored by eigenmode 1 - Fiedler vector)
        fiedler = self.eigenmodes[:, 1] if self.eigenmodes.shape[1] > 1 else np.zeros(self.n_nodes)
        fiedler_norm = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-9)
        
        for i in range(self.n_nodes):
            color_val = fiedler_norm[i]
            color = (int(100 + color_val * 155),
                    int(100 * (1 - color_val)),
                    int(100 + (1 - color_val) * 155))
            cv2.circle(self.display, tuple(pos_img[i]), 5, color, -1)
            cv2.circle(self.display, tuple(pos_img[i]), 5, (200, 200, 200), 1)
        
        # Info
        cv2.putText(self.display, f"Nodes: {self.n_nodes}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(self.display, f"Step: {self.step_count}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(self.display, f"λ1: {self.eigenvalues[1]:.3f}" if len(self.eigenvalues) > 1 else "",
                   (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
    def get_output(self, name):
        if name == 'eigenmodes':
            return self.eigenmodes
        elif name == 'eigenvalues':
            return self.eigenvalues
        elif name == 'laplacian':
            return self.laplacian
        elif name == 'connectivity':
            return self.connectivity
        elif name == 'positions':
            return self.positions
        elif name == 'geometry_view':
            return self.display
        return None


class ResonantProcessingNode(BaseNode):
    """
    The Forward Pass: Signal → Eigenmode Activation → Output
    
    Input signals are projected onto the cortical eigenmodes.
    Each eigenmode has a "gain" (synaptic weight) that can be modified.
    
    The output is the weighted sum of eigenmode activations.
    
    This is what happens when a stimulus enters the brain:
    it gets decomposed into the geometric basis functions (eigenmodes)
    that the cortex supports.
    
    Key insight: You can only think thoughts that your geometry allows.
    """
    NODE_CATEGORY = "Cortical Learning"
    NODE_COLOR = QtGui.QColor(200, 150, 100)  # Warm processing
    
    def __init__(self):
        super().__init__()
        self.node_title = "Resonant Processing"
        
        self.inputs = {
            'input_signal': 'spectrum',     # Incoming stimulus
            'eigenmodes': 'tensor',         # From CorticalSubstrate
            'eigenvalues': 'spectrum',      # From CorticalSubstrate
            'gain_modulation': 'spectrum',  # Optional: attention-like modulation
            'learning_signal': 'signal'     # Triggers weight update
        }
        
        self.outputs = {
            'output_signal': 'spectrum',      # Processed output
            'mode_activations': 'spectrum',   # Which eigenmodes are active
            'resonance_energy': 'signal',     # Total resonance strength
            'processing_view': 'image'
        }
        
        self.n_modes = 16
        
        # Synaptic weights for each eigenmode (modifiable!)
        self.mode_gains = np.ones(self.n_modes) * 0.5
        
        # Temporal dynamics (leaky integration)
        self.activation_tau = 0.9  # Decay rate
        self.current_activations = np.zeros(self.n_modes)
        
        # Plasticity
        self.learning_rate = 0.01
        self.activation_history = deque(maxlen=50)
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        self.output_signal = np.zeros(64)
        self.resonance_energy = 0.0
        
    def step(self):
        # Get inputs
        input_sig = self.get_blended_input('input_signal', 'mean')
        eigenmodes = self.get_blended_input('eigenmodes', 'mean')
        eigenvalues = self.get_blended_input('eigenvalues', 'mean')
        gain_mod = self.get_blended_input('gain_modulation', 'mean')
        learn_sig = self.get_blended_input('learning_signal', 'mean')
        
        if eigenmodes is None:
            return
            
        n_nodes = eigenmodes.shape[0]
        n_modes = min(eigenmodes.shape[1], self.n_modes)
        
        # Resize mode gains if needed
        if len(self.mode_gains) != n_modes:
            self.mode_gains = np.ones(n_modes) * 0.5
            self.current_activations = np.zeros(n_modes)
        
        # Prepare input signal
        if input_sig is None:
            input_sig = np.random.randn(n_nodes) * 0.1
        else:
            if len(input_sig) < n_nodes:
                input_sig = np.interp(
                    np.linspace(0, 1, n_nodes),
                    np.linspace(0, 1, len(input_sig)),
                    input_sig
                )
            elif len(input_sig) > n_nodes:
                input_sig = input_sig[:n_nodes]
        
        # === FORWARD PASS ===
        # Project input onto eigenmodes: a_k = <input, φ_k>
        new_activations = np.zeros(n_modes)
        for k in range(n_modes):
            new_activations[k] = np.dot(input_sig, eigenmodes[:, k])
        
        # Apply temporal dynamics (leaky integration)
        self.current_activations = (self.activation_tau * self.current_activations + 
                                   (1 - self.activation_tau) * new_activations)
        
        # Apply mode gains (synaptic weights)
        effective_gains = self.mode_gains.copy()
        if gain_mod is not None and len(gain_mod) >= n_modes:
            effective_gains = effective_gains * (0.5 + gain_mod[:n_modes])
        
        weighted_activations = self.current_activations * effective_gains
        
        # Compute output: weighted sum of eigenmodes
        self.output_signal = np.zeros(n_nodes)
        for k in range(n_modes):
            self.output_signal += weighted_activations[k] * eigenmodes[:, k]
        
        # Resonance energy (how much is the system "ringing")
        self.resonance_energy = np.sum(weighted_activations**2)
        
        # Record for learning
        self.activation_history.append(self.current_activations.copy())
        
        # === LEARNING (if triggered) ===
        if learn_sig is not None and learn_sig > 0.5:
            self._update_gains()
        
        # Visualization
        self._draw_processing(n_modes, eigenvalues)
        
    def _update_gains(self):
        """
        Hebbian learning on mode gains:
        Modes that are consistently active get strengthened.
        This is the MEDIUM timescale - synaptic plasticity.
        """
        if len(self.activation_history) < 10:
            return
            
        # Mean activation per mode
        mean_activation = np.mean(np.abs(list(self.activation_history)), axis=0)
        
        # Normalize to prevent runaway
        mean_activation = mean_activation / (np.max(mean_activation) + 1e-9)
        
        # Hebbian update with homeostatic normalization
        target_mean = 0.5
        delta = self.learning_rate * (mean_activation - target_mean)
        
        self.mode_gains = self.mode_gains + delta
        self.mode_gains = np.clip(self.mode_gains, 0.1, 2.0)
        
        # Normalize total gain (homeostasis)
        self.mode_gains = self.mode_gains / np.mean(self.mode_gains)
        
    def _draw_processing(self, n_modes, eigenvalues):
        h, w = 128, 128
        self.display.fill(0)
        
        # Draw mode activations as bars
        bar_w = max(4, w // n_modes - 2)
        
        for k in range(n_modes):
            x = 5 + k * (bar_w + 2)
            
            # Activation bar
            act = np.clip(np.abs(self.current_activations[k]) * 50, 0, h//2 - 5)
            gain = self.mode_gains[k]
            
            # Color by gain (red = high, blue = low)
            color = (int(50 + gain * 100), 
                    int(100), 
                    int(50 + (2 - gain) * 100))
            
            cv2.rectangle(self.display, 
                         (x, h//2 - int(act)), 
                         (x + bar_w, h//2), 
                         color, -1)
            
            # Gain indicator
            gain_h = int(gain * 20)
            cv2.rectangle(self.display,
                         (x, h - 5 - gain_h),
                         (x + bar_w, h - 5),
                         (200, 200, 100), -1)
        
        # Resonance energy meter
        energy_w = int(np.clip(self.resonance_energy * 10, 0, w - 20))
        cv2.rectangle(self.display, (10, 5), (10 + energy_w, 15),
                     (100, 255, 100), -1)
        cv2.putText(self.display, f"E={self.resonance_energy:.2f}", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Separator line
        cv2.line(self.display, (0, h//2), (w, h//2), (100, 100, 100), 1)
        
    def get_output(self, name):
        if name == 'output_signal':
            return self.output_signal
        elif name == 'mode_activations':
            return self.current_activations
        elif name == 'resonance_energy':
            return self.resonance_energy
        elif name == 'processing_view':
            return self.display
        return None


class PredictiveCodingNode(BaseNode):
    """
    Predictive Processing: The Backward Pass
    
    Implements Friston's predictive coding / free energy principle:
    - The brain constantly predicts its next state
    - Prediction errors drive learning
    - The goal is to minimize surprise (free energy)
    
    This node:
    1. Maintains a generative model (predicts next state)
    2. Computes prediction error
    3. Outputs error signal for plasticity
    
    The prediction error is what drives:
    - Fast: attention shifts to surprising inputs
    - Medium: synaptic weight updates (learning)
    - Slow: geometry remodeling (development)
    """
    NODE_CATEGORY = "Cortical Learning"
    NODE_COLOR = QtGui.QColor(200, 100, 150)  # Prediction pink
    
    def __init__(self):
        super().__init__()
        self.node_title = "Predictive Coding"
        
        self.inputs = {
            'sensory_input': 'spectrum',     # Bottom-up: actual input
            'top_down_prediction': 'spectrum', # Optional: higher-level prediction
            'mode_activations': 'spectrum'   # Current eigenmode state
        }
        
        self.outputs = {
            'prediction_error': 'spectrum',  # The surprise signal
            'prediction': 'spectrum',        # What we predicted
            'free_energy': 'signal',         # Scalar surprise measure
            'learn_trigger': 'signal',       # Triggers learning when error high
            'prediction_view': 'image'
        }
        
        # Internal generative model
        self.model_weights = None  # Will be initialized on first input
        self.prediction = None
        self.prediction_error = None
        self.free_energy = 0.0
        
        # Model parameters
        self.prediction_horizon = 1  # How many steps ahead to predict
        self.model_learning_rate = 0.05
        
        # History for learning
        self.input_history = deque(maxlen=20)
        self.activation_history = deque(maxlen=20)
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
    def step(self):
        sensory = self.get_blended_input('sensory_input', 'mean')
        top_down = self.get_blended_input('top_down_prediction', 'mean')
        activations = self.get_blended_input('mode_activations', 'mean')
        
        if sensory is None:
            return
            
        n_dim = len(sensory)
        
        # Initialize model if needed
        if self.model_weights is None or self.model_weights.shape[0] != n_dim:
            self.model_weights = np.eye(n_dim) * 0.5 + np.random.randn(n_dim, n_dim) * 0.1
            self.prediction = np.zeros(n_dim)
            self.prediction_error = np.zeros(n_dim)
        
        # Record history
        self.input_history.append(sensory.copy())
        if activations is not None:
            self.activation_history.append(activations.copy())
        
        # === PREDICTION ===
        # Use previous input to predict current (simple autoregressive model)
        if len(self.input_history) >= 2:
            prev_input = self.input_history[-2]
            self.prediction = self.model_weights @ prev_input
        else:
            self.prediction = np.zeros(n_dim)
        
        # Incorporate top-down prediction if available
        if top_down is not None and len(top_down) == n_dim:
            self.prediction = 0.7 * self.prediction + 0.3 * top_down
        
        # === PREDICTION ERROR ===
        self.prediction_error = sensory - self.prediction
        
        # Free energy (scalar measure of surprise)
        self.free_energy = np.mean(self.prediction_error**2)
        
        # === MODEL LEARNING ===
        # Update generative model to reduce future errors
        if len(self.input_history) >= 2:
            prev_input = self.input_history[-2]
            
            # Gradient descent on prediction error
            # dW = learning_rate * error * prev_input^T
            delta_W = self.model_learning_rate * np.outer(self.prediction_error, prev_input)
            self.model_weights += delta_W
            
            # Regularization (prevent explosion)
            self.model_weights *= 0.99
        
        # Visualization
        self._draw_prediction()
        
    def _draw_prediction(self):
        h, w = 128, 128
        self.display.fill(0)
        
        if self.prediction is None or len(self.prediction) == 0:
            return
            
        n = len(self.prediction)
        
        # Draw prediction (green) vs actual (blue) vs error (red)
        # Use top third for each
        section_h = h // 3
        
        # Actual input
        if len(self.input_history) > 0:
            actual = self.input_history[-1]
            for i, val in enumerate(actual[:min(n, w)]):
                x = int(i * w / n)
                y = section_h // 2 - int(val * section_h * 0.4)
                y = np.clip(y, 2, section_h - 2)
                cv2.circle(self.display, (x, y), 2, (100, 100, 255), -1)
        cv2.putText(self.display, "Actual", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1)
        
        # Prediction
        for i, val in enumerate(self.prediction[:min(n, w)]):
            x = int(i * w / n)
            y = section_h + section_h // 2 - int(val * section_h * 0.4)
            y = np.clip(y, section_h + 2, 2*section_h - 2)
            cv2.circle(self.display, (x, y), 2, (100, 255, 100), -1)
        cv2.putText(self.display, "Predict", (5, section_h + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        
        # Error
        for i, val in enumerate(self.prediction_error[:min(n, w)]):
            x = int(i * w / n)
            y = 2*section_h + section_h // 2 - int(val * section_h * 0.4)
            y = np.clip(y, 2*section_h + 2, h - 2)
            cv2.circle(self.display, (x, y), 2, (255, 100, 100), -1)
        cv2.putText(self.display, f"Error F={self.free_energy:.3f}", (5, 2*section_h + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
        
        # Free energy bar
        fe_w = int(np.clip(self.free_energy * 500, 0, w - 10))
        cv2.rectangle(self.display, (5, h - 15), (5 + fe_w, h - 5),
                     (255, 100, 100), -1)
        
    def get_output(self, name):
        if name == 'prediction_error':
            return self.prediction_error if self.prediction_error is not None else np.zeros(16)
        elif name == 'prediction':
            return self.prediction if self.prediction is not None else np.zeros(16)
        elif name == 'free_energy':
            return self.free_energy
        elif name == 'learn_trigger':
            # Trigger learning when error is high
            return 1.0 if self.free_energy > 0.1 else 0.0
        elif name == 'prediction_view':
            return self.display
        return None


class GeometryLearnerNode(BaseNode):
    """
    The Slowest Timescale: Geometry Evolution
    
    Over very long timescales (days/weeks in real brains),
    the physical geometry of cortex changes through:
    
    1. Axonal tension: Strongly connected areas pull together
    2. Activity-dependent pruning: Unused connections die
    3. Growth: New connections form along activity gradients
    
    This is the "developmental plasticity" that shapes
    which eigenmodes exist in the first place.
    
    In our simulation, this modifies the connectivity matrix
    based on long-term activity patterns and prediction errors.
    """
    NODE_CATEGORY = "Cortical Learning"
    NODE_COLOR = QtGui.QColor(150, 200, 100)  # Growth green
    
    def __init__(self):
        super().__init__()
        self.node_title = "Geometry Learner"
        
        self.inputs = {
            'connectivity': 'tensor',        # Current connectivity
            'activity_pattern': 'spectrum',  # Current activity
            'prediction_error': 'spectrum',  # Error signal
            'free_energy': 'signal'          # Surprise level
        }
        
        self.outputs = {
            'modified_connectivity': 'tensor',
            'growth_signal': 'spectrum',     # Where to grow connections
            'prune_signal': 'spectrum',      # Where to prune
            'geometry_delta': 'signal',      # How much geometry changed
            'learner_view': 'image'
        }
        
        # Long-term accumulators
        self.activity_accumulator = None
        self.error_accumulator = None
        self.coactivation_accumulator = None
        
        # Learning parameters
        self.growth_rate = 0.0001  # VERY slow
        self.prune_threshold = 0.05
        self.activity_threshold = 0.1
        
        self.step_count = 0
        self.last_geometry_delta = 0.0
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
    def step(self):
        self.step_count += 1
        
        connectivity = self.get_blended_input('connectivity', 'mean')
        activity = self.get_blended_input('activity_pattern', 'mean')
        error = self.get_blended_input('prediction_error', 'mean')
        free_energy = self.get_blended_input('free_energy', 'mean') or 0.0
        
        if connectivity is None:
            return
            
        n = connectivity.shape[0]
        
        # Initialize accumulators
        if self.activity_accumulator is None or len(self.activity_accumulator) != n:
            self.activity_accumulator = np.zeros(n)
            self.error_accumulator = np.zeros(n)
            self.coactivation_accumulator = np.zeros((n, n))
        
        # Accumulate activity
        if activity is not None:
            if len(activity) < n:
                activity = np.interp(np.linspace(0, 1, n),
                                    np.linspace(0, 1, len(activity)),
                                    activity)
            elif len(activity) > n:
                activity = activity[:n]
                
            self.activity_accumulator = 0.99 * self.activity_accumulator + 0.01 * np.abs(activity)
            
            # Coactivation (Hebbian)
            activity_outer = np.outer(activity, activity)
            self.coactivation_accumulator = 0.99 * self.coactivation_accumulator + 0.01 * activity_outer
        
        # Accumulate error
        if error is not None:
            if len(error) < n:
                error = np.interp(np.linspace(0, 1, n),
                                 np.linspace(0, 1, len(error)),
                                 error)
            elif len(error) > n:
                error = error[:n]
            self.error_accumulator = 0.99 * self.error_accumulator + 0.01 * np.abs(error)
        
        # === GEOMETRY UPDATE (every 1000 steps) ===
        if self.step_count % 1000 == 0:
            self._update_geometry(connectivity, n)
        
        # Visualization
        self._draw_learner(n)
        
    def _update_geometry(self, connectivity, n):
        """Apply accumulated learning to modify geometry."""
        
        # Growth: strengthen connections between coactive nodes
        growth = self.growth_rate * self.coactivation_accumulator
        
        # Pruning: weaken connections with low coactivation
        prune_mask = self.coactivation_accumulator < self.prune_threshold
        pruning = -self.growth_rate * prune_mask.astype(float)
        
        # Error-driven growth: strengthen connections that reduce error
        # Areas with high error need more connectivity
        error_gradient = np.outer(self.error_accumulator, self.error_accumulator)
        error_growth = self.growth_rate * 0.5 * error_gradient
        
        # Combined update
        delta = growth + pruning + error_growth
        
        # Apply to connectivity
        modified = connectivity + delta
        
        # Enforce constraints
        modified = np.clip(modified, 0.01, 2.0)
        modified = (modified + modified.T) / 2  # Symmetry
        np.fill_diagonal(modified, 0)
        
        # Track how much geometry changed
        self.last_geometry_delta = np.mean(np.abs(delta))
        
        # Store for output
        self._modified_connectivity = modified
        self._growth_signal = np.mean(growth, axis=1)
        self._prune_signal = np.mean(np.abs(pruning), axis=1)
        
    def _draw_learner(self, n):
        h, w = 128, 128
        self.display.fill(0)
        
        # Activity accumulator as heat map
        if self.activity_accumulator is not None:
            act_norm = self.activity_accumulator / (np.max(self.activity_accumulator) + 1e-9)
            for i, val in enumerate(act_norm):
                x = int(i * w / n)
                bar_h = int(val * 30)
                color = (int(50 + val * 200), int(100 + val * 100), 50)
                cv2.rectangle(self.display, (x, h//2 - bar_h), (x + max(1, w//n), h//2),
                             color, -1)
        
        cv2.putText(self.display, "Activity", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Error accumulator
        if self.error_accumulator is not None:
            err_norm = self.error_accumulator / (np.max(self.error_accumulator) + 1e-9)
            for i, val in enumerate(err_norm):
                x = int(i * w / n)
                bar_h = int(val * 30)
                color = (50, 50, int(50 + val * 200))
                cv2.rectangle(self.display, (x, h - bar_h), (x + max(1, w//n), h),
                             color, -1)
        
        cv2.putText(self.display, "Error", (5, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Geometry delta
        cv2.putText(self.display, f"Geo Δ: {self.last_geometry_delta:.6f}", (5, h//2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        cv2.putText(self.display, f"Step: {self.step_count}", (5, h//2 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
    def get_output(self, name):
        if name == 'modified_connectivity':
            return getattr(self, '_modified_connectivity', None)
        elif name == 'growth_signal':
            return getattr(self, '_growth_signal', np.zeros(16))
        elif name == 'prune_signal':
            return getattr(self, '_prune_signal', np.zeros(16))
        elif name == 'geometry_delta':
            return self.last_geometry_delta
        elif name == 'learner_view':
            return self.display
        return None


class StimulusGeneratorNode(BaseNode):
    """
    Generates structured stimuli to drive the learning system.
    
    Can generate:
    - Random noise (baseline)
    - Structured patterns (learning targets)
    - Sequences (temporal structure)
    - EEG-like band power patterns
    """
    NODE_CATEGORY = "Cortical Learning"
    NODE_COLOR = QtGui.QColor(255, 200, 100)  # Stimulus yellow
    
    def __init__(self):
        super().__init__()
        self.node_title = "Stimulus Generator"
        
        self.inputs = {
            'external_signal': 'spectrum',  # Optional external drive
            'noise_level': 'signal'
        }
        
        self.outputs = {
            'stimulus': 'spectrum',
            'stimulus_type': 'signal',
            'generator_view': 'image'
        }
        
        self.stimulus_length = 64
        self.stimulus_type = 0  # 0=noise, 1=sine, 2=square, 3=sequence
        self.phase = 0.0
        self.sequence_step = 0
        
        # Predefined patterns (like "concepts" to learn)
        self.patterns = self._init_patterns()
        
        self.display = np.zeros((64, 128, 3), dtype=np.uint8)
        self.current_stimulus = np.zeros(self.stimulus_length)
        
    def _init_patterns(self):
        """Create a set of learnable patterns."""
        patterns = []
        n = self.stimulus_length
        
        # Pattern 0: Low frequency (delta-like)
        patterns.append(np.sin(np.linspace(0, 2*np.pi, n)))
        
        # Pattern 1: Medium frequency (alpha-like)
        patterns.append(np.sin(np.linspace(0, 8*np.pi, n)))
        
        # Pattern 2: High frequency (gamma-like)
        patterns.append(np.sin(np.linspace(0, 20*np.pi, n)))
        
        # Pattern 3: Localized bump (like a receptive field)
        x = np.linspace(-3, 3, n)
        patterns.append(np.exp(-x**2))
        
        # Pattern 4: Edge (like a visual edge)
        patterns.append(np.concatenate([np.zeros(n//2), np.ones(n//2)]))
        
        return patterns
        
    def step(self):
        external = self.get_blended_input('external_signal', 'mean')
        noise_level = self.get_blended_input('noise_level', 'mean') or 0.1
        
        if isinstance(noise_level, np.ndarray):
            noise_level = float(np.mean(noise_level))
        
        self.phase += 0.1
        
        if external is not None and len(external) > 0:
            # Use external signal
            if len(external) != self.stimulus_length:
                self.current_stimulus = np.interp(
                    np.linspace(0, 1, self.stimulus_length),
                    np.linspace(0, 1, len(external)),
                    external
                )
            else:
                self.current_stimulus = external.copy()
            self.stimulus_type = -1
        else:
            # Generate internal stimulus
            # Cycle through patterns
            pattern_idx = int(self.phase / 10) % len(self.patterns)
            self.current_stimulus = self.patterns[pattern_idx].copy()
            self.stimulus_type = pattern_idx
        
        # Add noise
        self.current_stimulus += np.random.randn(self.stimulus_length) * noise_level
        
        # Visualization
        self._draw_generator()
        
    def _draw_generator(self):
        h, w = 64, 128
        self.display.fill(0)
        
        # Draw stimulus
        for i, val in enumerate(self.current_stimulus):
            x = int(i * w / self.stimulus_length)
            y = h // 2 - int(val * h * 0.4)
            y = np.clip(y, 2, h - 2)
            cv2.circle(self.display, (x, y), 2, (255, 200, 100), -1)
        
        # Type indicator
        type_names = ['Delta', 'Alpha', 'Gamma', 'Bump', 'Edge', 'Ext']
        idx = self.stimulus_type if self.stimulus_type >= 0 else 5
        cv2.putText(self.display, type_names[min(idx, 5)], (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
    def get_output(self, name):
        if name == 'stimulus':
            return self.current_stimulus
        elif name == 'stimulus_type':
            return float(self.stimulus_type)
        elif name == 'generator_view':
            return self.display
        return None

    def get_config_options(self):
        return [
            ("Stimulus Length", "stimulus_length", self.stimulus_length, None),
        ]


class LearningMonitorNode(BaseNode):
    """
    Monitors the learning process across all timescales.
    
    Tracks:
    - Prediction error over time (should decrease)
    - Geometry changes (should stabilize)
    - Eigenmode evolution (should become more structured)
    - Overall system "intelligence" (pattern recognition accuracy)
    """
    NODE_CATEGORY = "Cortical Learning"
    NODE_COLOR = QtGui.QColor(100, 200, 200)  # Monitor cyan
    
    def __init__(self):
        super().__init__()
        self.node_title = "Learning Monitor"
        
        self.inputs = {
            'free_energy': 'signal',
            'geometry_delta': 'signal',
            'resonance_energy': 'signal',
            'eigenvalues': 'spectrum'
        }
        
        self.outputs = {
            'learning_quality': 'signal',
            'monitor_view': 'image'
        }
        
        # History
        self.free_energy_history = deque(maxlen=500)
        self.geometry_delta_history = deque(maxlen=500)
        self.resonance_history = deque(maxlen=500)
        self.eigenvalue_history = deque(maxlen=100)
        
        self.display = np.zeros((128, 256, 3), dtype=np.uint8)
        
    def step(self):
        fe = self.get_blended_input('free_energy', 'mean') or 0.0
        geo_delta = self.get_blended_input('geometry_delta', 'mean') or 0.0
        resonance = self.get_blended_input('resonance_energy', 'mean') or 0.0
        eigenvalues = self.get_blended_input('eigenvalues', 'mean')
        
        if isinstance(fe, np.ndarray):
            fe = float(np.mean(fe))
        if isinstance(geo_delta, np.ndarray):
            geo_delta = float(np.mean(geo_delta))
        if isinstance(resonance, np.ndarray):
            resonance = float(np.mean(resonance))
        
        self.free_energy_history.append(fe)
        self.geometry_delta_history.append(geo_delta)
        self.resonance_history.append(resonance)
        
        if eigenvalues is not None:
            self.eigenvalue_history.append(eigenvalues.copy())
        
        # Visualization
        self._draw_monitor()
        
    def _draw_monitor(self):
        h, w = 128, 256
        self.display.fill(0)
        
        # Plot free energy (red) - should decrease
        self._plot_history(self.free_energy_history, (255, 100, 100), 0, "Free Energy", scale=100)
        
        # Plot geometry delta (green) - should stabilize
        self._plot_history(self.geometry_delta_history, (100, 255, 100), 1, "Geo Delta", scale=10000)
        
        # Plot resonance (blue) - should become structured
        self._plot_history(self.resonance_history, (100, 100, 255), 2, "Resonance", scale=5)
        
    def _plot_history(self, history, color, row, label, scale=1.0):
        if len(history) < 2:
            return
            
        h, w = 128, 256
        row_h = h // 3
        y_offset = row * row_h
        
        # Draw background
        cv2.rectangle(self.display, (0, y_offset), (w, y_offset + row_h - 1),
                     (30, 30, 30), -1)
        
        # Plot line
        data = np.array(list(history)) * scale
        if len(data) > 0:
            data_min, data_max = np.min(data), np.max(data)
            if data_max > data_min:
                data_norm = (data - data_min) / (data_max - data_min)
            else:
                data_norm = np.zeros_like(data)
            
            for i in range(1, len(data_norm)):
                x1 = int((i-1) * w / len(data_norm))
                x2 = int(i * w / len(data_norm))
                y1 = y_offset + row_h - 5 - int(data_norm[i-1] * (row_h - 10))
                y2 = y_offset + row_h - 5 - int(data_norm[i] * (row_h - 10))
                cv2.line(self.display, (x1, y1), (x2, y2), color, 1)
        
        # Label
        cv2.putText(self.display, label, (5, y_offset + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Current value
        if len(history) > 0:
            cv2.putText(self.display, f"{list(history)[-1]:.4f}", (w - 60, y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
    def get_output(self, name):
        if name == 'learning_quality':
            # Inverse of recent free energy (higher = better learning)
            if len(self.free_energy_history) > 0:
                recent_fe = np.mean(list(self.free_energy_history)[-50:])
                return 1.0 / (recent_fe + 0.01)
            return 0.0
        elif name == 'monitor_view':
            return self.display
        return None