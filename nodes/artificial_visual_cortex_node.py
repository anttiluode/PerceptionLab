"""
Artificial Visual Cortex Node
=============================

A self-contained visual processing system inspired by V1/MT/MST cortical architecture.

Architecture:
- Multiple "V1 Layers" at different symmetry levels (depth/scale processing)
  Each layer does log-polar mandala transformation at increasing symmetry
  Low symmetry = texture-dominant, High symmetry = depth-dominant (radial bands)
  
- "IT Layer" (Inferotemporal) - preserves full image for object identity
  
- All layers connected via Izhikevich neurons with learned/learnable weights
- Layers can "talk" to each other through neural dynamics

The V1 layers simulate the log-polar mapping that converts:
- Expansion (approaching) → horizontal shift on cortex  
- Contraction (receding) → opposite horizontal shift
- Rotation (head turn) → vertical shift on cortex
- Spirals → oblique linear motion

At high symmetry, only the radial depth structure remains - the "what's at what depth"
signal that V1 extracts for downstream MST processing.

Inputs:
- image_in: Visual input (webcam, image file, etc.)
- eeg_in: Optional EEG modulation (delta through gamma bands)
- latent_in: Optional latent vector injection
- learning: Enable/disable STDP learning

Outputs:
- v1_mandala: Combined output of all V1 symmetry layers
- it_layer: The texture/identity layer (original image in neural form)
- depth_signal: Pure radial structure (highest symmetry layer)
- eigenstate: The emergent attractor state across all layers
- activity_view: Full neural activity visualization
- resonance: How much the system is resonating
- energy: Total system energy

Configuration:
- grid_size: Resolution of each layer (32-256)
- n_v1_layers: Number of V1 symmetry layers (4-32)
- base_symmetry: Starting symmetry for V1 layers
- symmetry_multiplier: How much symmetry increases per layer
- coupling_strength: Inter-neuron coupling
- learning_rate: STDP plasticity rate (0 = frozen)

Can optionally boot from a CrystalChip .npz to inherit learned weights.

Author: Built for Antti's consciousness crystallography research
Based on Grossberg et al. 1999 - MST motion processing model
"""

import os
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    # Standalone mode - mock the classes
    QtGui = None
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode):
            return None

PHI = (1 + np.sqrt(5)) / 2


class ArtificialVisualCortexNode(BaseNode):
    """
    Multi-layer visual cortex simulation with V1 log-polar processing
    and Izhikevich neural dynamics.
    """
    
    NODE_NAME = "Artificial Visual Cortex"
    NODE_TITLE = "V1 Cortex"
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(180, 100, 200) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            "image_in": "image",        # Visual input
            "delta_in": "signal",       # EEG delta band modulation
            "theta_in": "signal",       # EEG theta
            "alpha_in": "signal",       # EEG alpha  
            "beta_in": "signal",        # EEG beta
            "gamma_in": "signal",       # EEG gamma
            "latent_in": "spectrum",    # Latent vector injection
            "learning": "signal",       # Enable learning (>0.5 = on)
            "reset": "signal",          # Reset neural state
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Layer outputs
            "v1_mandala": "image",       # Combined V1 layers (depth-processed)
            "it_layer": "image",         # Texture/identity layer
            "depth_signal": "image",     # Pure radial structure (deepest V1)
            
            # Emergent outputs
            "eigenstate": "image",       # The attractor state
            "activity_view": "image",    # Full activity visualization
            
            # Analysis
            "resonance": "signal",       # System resonance
            "energy": "signal",          # Total energy
            "entropy": "signal",         # Weight entropy
            "depth_profile": "spectrum", # Radial depth profile (1D)
            
            # EEG-like outputs from processing
            "lfp_out": "signal",         # Local field potential
        }
        
        # === ARCHITECTURE PARAMETERS ===
        self.grid_size = 64              # Resolution per layer
        self.n_v1_layers = 8             # Number of V1 symmetry layers
        self.base_symmetry = 2           # Starting symmetry
        self.symmetry_multiplier = PHI   # Golden ratio scaling between layers
        
        # === IZHIKEVICH PARAMETERS ===
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 0.5
        
        # === COUPLING & LEARNING ===
        self.coupling_strength = 3.0     # Within-layer coupling
        self.inter_layer_coupling = 1.0  # Between-layer coupling
        self.learning_rate = 0.001       # STDP rate (0 = frozen)
        self.trace_decay = 0.95
        self.weight_max = 2.0
        self.weight_min = 0.01
        
        # === STATE ARRAYS ===
        # Will be initialized in _init_layers()
        self.v = None           # Membrane potentials [n_layers+1, grid, grid]
        self.u = None           # Recovery variables
        self.weights = None     # Coupling weights (4 directions per layer)
        self.inter_weights = None  # Between-layer weights
        self.spike_trace = None
        
        # === PROCESSING BUFFERS ===
        self.symmetry_levels = []  # Computed symmetry for each V1 layer
        self.last_image = None
        self.processed_layers = None
        
        # === STATISTICS ===
        self.step_count = 0
        self.total_spikes = 0
        self.current_resonance = 0.0
        self.current_energy = 0.0
        self.current_entropy = 0.0
        
        # LFP history for frequency analysis
        self.lfp_history = np.zeros(256, dtype=np.float32)
        self.lfp_idx = 0
        
        # Display
        self.display_image = None
        
        # Initialize
        self._init_layers()
        self._update_display()
        
    def _init_layers(self):
        """Initialize all neural layers and weights."""
        n = self.grid_size
        n_layers = self.n_v1_layers + 1  # V1 layers + IT layer
        
        # Compute symmetry levels for V1 layers
        self.symmetry_levels = []
        sym = self.base_symmetry
        for i in range(self.n_v1_layers):
            self.symmetry_levels.append(int(sym))
            sym *= self.symmetry_multiplier
        
        # Neural state: [layer, row, col]
        self.v = np.ones((n_layers, n, n), dtype=np.float32) * self.c
        self.u = self.v * self.b
        
        # Within-layer weights: [layer, direction, row, col]
        # directions: 0=up, 1=down, 2=left, 3=right
        self.weights = np.ones((n_layers, 4, n, n), dtype=np.float32) * 0.5
        
        # Between-layer weights: [from_layer, to_layer]
        # Simple scalar coupling for now, could be spatial
        self.inter_weights = np.ones((n_layers, n_layers), dtype=np.float32) * 0.1
        # Stronger coupling to adjacent layers
        for i in range(n_layers):
            for j in range(n_layers):
                dist = abs(i - j)
                if dist == 1:
                    self.inter_weights[i, j] = 0.3
                elif dist == 0:
                    self.inter_weights[i, j] = 0.0  # No self-coupling
                else:
                    self.inter_weights[i, j] = 0.05 / dist
        
        # STDP trace
        self.spike_trace = np.zeros((n_layers, n, n), dtype=np.float32)
        
        # Processing buffer
        self.processed_layers = np.zeros((n_layers, n, n), dtype=np.float32)
        
        print(f"[V1Cortex] Initialized {n_layers} layers at {n}x{n}")
        print(f"[V1Cortex] V1 symmetries: {self.symmetry_levels}")
        
    def _process_v1_layer(self, image, symmetry):
        """
        Process image through V1-like log-polar transformation.
        Higher symmetry = more radial/depth focused.
        """
        n = self.grid_size
        
        # Ensure image is correct size and grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, (n, n)).astype(np.float32) / 255.0
        
        # FFT for frequency processing
        spectrum = fftshift(fft2(gray))
        
        # Apply radial processing (polar bandpass)
        cy, cx = n // 2, n // 2
        y, x = np.ogrid[:n, :n]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        theta = np.arctan2(y - cy, x - cx)
        
        # High symmetry = average out angular info, keep radial
        # This is the key V1→MST transformation for depth
        if symmetry > 1:
            # Create angular sectors
            sector_angle = 2 * np.pi / symmetry
            sector_idx = ((theta + np.pi) / sector_angle).astype(int) % symmetry
            
            # Average within sectors (kaleidoscope effect in frequency domain)
            processed = np.zeros_like(spectrum)
            for s in range(symmetry):
                mask = (sector_idx == s)
                if np.any(mask):
                    sector_mean = np.mean(spectrum[mask])
                    processed[mask] = sector_mean
            
            spectrum = processed
        
        # Inverse FFT
        result = np.real(ifft2(ifftshift(spectrum)))
        
        # Normalize to 0-1
        result = (result - result.min()) / (result.max() - result.min() + 1e-9)
        
        return result.astype(np.float32)
    
    def _process_it_layer(self, image):
        """
        IT layer - preserves texture/identity information.
        Just normalizes the image for neural input.
        """
        n = self.grid_size
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, (n, n)).astype(np.float32) / 255.0
        
        return gray
    
    def step(self):
        """Main processing step."""
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
        
        # Reset if requested
        if reset_signal is not None and reset_signal > 0.5:
            self._init_layers()
            return
        
        # Determine learning state
        learning = self.learning_rate > 0
        if learning_signal is not None:
            learning = learning_signal > 0.5
        
        # Process image through all layers
        if image_in is not None:
            self.last_image = image_in
            
            # Process V1 layers (each at different symmetry)
            for i, sym in enumerate(self.symmetry_levels):
                self.processed_layers[i] = self._process_v1_layer(image_in, sym)
            
            # Process IT layer (last layer, no symmetry)
            self.processed_layers[-1] = self._process_it_layer(image_in)
        
        # Convert processed layers to input current
        # Scale by coupling and EEG modulation
        input_gain = 30.0
        eeg_mod = 1.0 + 0.1 * (delta + theta + alpha + beta + gamma)
        
        input_current = self.processed_layers * input_gain * eeg_mod
        
        # === NEURAL DYNAMICS ===
        n_layers = self.n_v1_layers + 1
        n = self.grid_size
        
        total_spikes_this_step = 0
        
        for layer in range(n_layers):
            v = self.v[layer]
            u = self.u[layer]
            I = input_current[layer]
            
            # Clamp input
            I = np.clip(I, -100, 100)
            
            # Within-layer neighbor coupling
            v_up = np.roll(v, -1, axis=0)
            v_down = np.roll(v, 1, axis=0)
            v_left = np.roll(v, -1, axis=1)
            v_right = np.roll(v, 1, axis=1)
            
            w = self.weights[layer]
            neighbor_influence = (
                w[0] * v_up +
                w[1] * v_down +
                w[2] * v_left +
                w[3] * v_right
            )
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
            
            # Clamp
            v = np.clip(v, -100, 50)
            u = np.clip(u, -50, 50)
            v = np.nan_to_num(v, nan=self.c, posinf=30.0, neginf=-100.0)
            u = np.nan_to_num(u, nan=self.c * self.b, posinf=20.0, neginf=-20.0)
            
            # Spikes
            spikes = v >= 30.0
            v[spikes] = self.c
            u[spikes] += self.d
            
            total_spikes_this_step += np.sum(spikes)
            
            self.v[layer] = v
            self.u[layer] = u
            
            # STDP learning
            if learning and self.learning_rate > 0:
                self.spike_trace[layer] *= self.trace_decay
                self.spike_trace[layer, spikes] = 1.0
                
                trace_up = np.roll(self.spike_trace[layer], -1, axis=0)
                trace_down = np.roll(self.spike_trace[layer], 1, axis=0)
                trace_left = np.roll(self.spike_trace[layer], -1, axis=1)
                trace_right = np.roll(self.spike_trace[layer], 1, axis=1)
                
                spike_float = spikes.astype(np.float32)
                lr = self.learning_rate
                
                # Potentiation
                dw = np.zeros((4, n, n), dtype=np.float32)
                dw[0] = lr * spike_float * trace_up
                dw[1] = lr * spike_float * trace_down
                dw[2] = lr * spike_float * trace_left
                dw[3] = lr * spike_float * trace_right
                
                # Depression
                spike_up = np.roll(spike_float, -1, axis=0)
                spike_down = np.roll(spike_float, 1, axis=0)
                spike_left = np.roll(spike_float, -1, axis=1)
                spike_right = np.roll(spike_float, 1, axis=1)
                
                dw[0] -= 0.5 * lr * self.spike_trace[layer] * spike_up
                dw[1] -= 0.5 * lr * self.spike_trace[layer] * spike_down
                dw[2] -= 0.5 * lr * self.spike_trace[layer] * spike_left
                dw[3] -= 0.5 * lr * self.spike_trace[layer] * spike_right
                
                self.weights[layer] = np.clip(
                    self.weights[layer] + dw, 
                    self.weight_min, 
                    self.weight_max
                )
        
        self.total_spikes += total_spikes_this_step
        
        # === COMPUTE OUTPUTS ===
        # LFP (mean membrane potential)
        mean_v = np.mean(self.v)
        self.lfp_history[self.lfp_idx % 256] = mean_v
        self.lfp_idx += 1
        
        # Resonance (variance of activity)
        self.current_resonance = float(np.std(self.v))
        
        # Energy
        self.current_energy = float(np.sum(np.abs(self.v - self.c)))
        
        # Entropy
        all_weights = self.weights.flatten()
        w_norm = all_weights / (np.sum(all_weights) + 1e-9)
        self.current_entropy = float(-np.sum(w_norm * np.log(w_norm + 1e-9)))
        
        # Update display periodically
        if self.step_count % 5 == 0:
            self._update_display()
    
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
        return None
    
    def _render_v1_mandala(self):
        """Render combined V1 layers as mandala visualization."""
        n = self.grid_size
        
        # Combine V1 layers (weighted by symmetry - higher = more important for depth)
        combined = np.zeros((n, n), dtype=np.float32)
        total_weight = 0
        
        for i in range(self.n_v1_layers):
            sym = self.symmetry_levels[i]
            weight = np.log(sym + 1)  # Log weighting
            
            # Use neural activity, not just processed input
            layer_v = self.v[i]
            norm_v = (layer_v + 90) / 130  # Normalize to 0-1ish
            combined += norm_v * weight
            total_weight += weight
        
        combined /= (total_weight + 1e-9)
        combined = np.clip(combined, 0, 1)
        
        # Apply colormap
        combined_u8 = (combined * 255).astype(np.uint8)
        colored = cv2.applyColorMap(combined_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        
        return cv2.resize(colored, (256, 256))
    
    def _render_it_layer(self):
        """Render IT (texture/identity) layer."""
        n = self.grid_size
        
        # Last layer is IT
        layer_v = self.v[-1]
        norm_v = np.clip((layer_v + 90) / 130, 0, 1)
        
        norm_u8 = (norm_v * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
        
        return cv2.resize(colored, (256, 256))
    
    def _render_depth_signal(self):
        """Render the highest-symmetry V1 layer (pure depth)."""
        n = self.grid_size
        
        # Highest symmetry layer (last V1 layer, most depth-focused)
        layer_v = self.v[self.n_v1_layers - 1]
        norm_v = np.clip((layer_v + 90) / 130, 0, 1)
        
        # Apply radial colormap to emphasize depth bands
        norm_u8 = (norm_v * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_VIRIDIS)
        
        return cv2.resize(colored, (256, 256))
    
    def _render_eigenstate(self):
        """Render the emergent eigenstate across all layers."""
        n = self.grid_size
        n_layers = self.n_v1_layers + 1
        
        # Create a multi-layer visualization
        # Stack layers horizontally showing flow from low-sym to high-sym to IT
        vis_height = 128
        vis_width = 128 * n_layers
        
        vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        for i in range(n_layers):
            layer_v = self.v[i]
            norm_v = np.clip((layer_v + 90) / 130, 0, 1)
            
            # Resize to thumbnail
            thumb = cv2.resize(norm_v, (128, 128))
            thumb_u8 = (thumb * 255).astype(np.uint8)
            
            # Different colormap for IT layer
            if i == n_layers - 1:
                colored = cv2.applyColorMap(thumb_u8, cv2.COLORMAP_INFERNO)
            else:
                # Hue varies with symmetry level
                colored = cv2.applyColorMap(thumb_u8, cv2.COLORMAP_TWILIGHT)
            
            vis[:, i*128:(i+1)*128] = colored
        
        return vis
    
    def _render_activity(self):
        """Render full neural activity grid."""
        n = self.grid_size
        n_layers = self.n_v1_layers + 1
        
        # Create grid: 3 rows x ceil(n_layers/3) cols
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
            
            # Label
            if i < self.n_v1_layers:
                label = f"V1-{self.symmetry_levels[i]}"
            else:
                label = "IT"
            cv2.putText(vis, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis
    
    def _compute_depth_profile(self):
        """Compute 1D radial depth profile from highest symmetry layer."""
        n = self.grid_size
        
        # Use highest symmetry V1 layer
        layer_v = self.v[self.n_v1_layers - 1]
        norm_v = np.clip((layer_v + 90) / 130, 0, 1)
        
        # Compute radial average
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
    
    def _update_display(self):
        """Update main node display."""
        w, h = 400, 350
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "ARTIFICIAL VISUAL CORTEX", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 100, 200), 2)
        
        # Config info
        cv2.putText(img, f"Grid: {self.grid_size}x{self.grid_size} | V1 Layers: {self.n_v1_layers}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(img, f"Symmetries: {self.symmetry_levels[:4]}...", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
        
        # V1 Mandala preview
        v1_img = self._render_v1_mandala()
        v1_small = cv2.resize(v1_img, (130, 130))
        img[75:205, 10:140] = v1_small
        cv2.putText(img, "V1 Mandala", (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # IT Layer preview
        it_img = self._render_it_layer()
        it_small = cv2.resize(it_img, (130, 130))
        img[75:205, 150:280] = it_small
        cv2.putText(img, "IT Layer", (150, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Depth signal preview
        depth_img = self._render_depth_signal()
        depth_small = cv2.resize(depth_img, (100, 100))
        img[75:175, 290:390] = depth_small
        cv2.putText(img, "Depth", (300, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Stats
        stats_y = 240
        cv2.putText(img, f"Step: {self.step_count}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Spikes: {self.total_spikes:,}", (10, stats_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Resonance: {self.current_resonance:.2f}", (10, stats_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(img, f"Energy: {self.current_energy:.0f}", (10, stats_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        cv2.putText(img, f"Entropy: {self.current_entropy:.2f}", (10, stats_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)
        
        # Learning status
        lr_status = "Learning ON" if self.learning_rate > 0 else "Frozen"
        cv2.putText(img, f"LR: {self.learning_rate:.4f} ({lr_status})", (200, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, 
                               QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image
    
    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", self.grid_size, 
             [("32", 32), ("64", 64), ("128", 128), ("256", 256)]),
            ("V1 Layers", "n_v1_layers", self.n_v1_layers, None),
            ("Base Symmetry", "base_symmetry", self.base_symmetry, None),
            ("Symmetry Multiplier", "symmetry_multiplier", self.symmetry_multiplier, None),
            ("Coupling Strength", "coupling_strength", self.coupling_strength, None),
            ("Inter-layer Coupling", "inter_layer_coupling", self.inter_layer_coupling, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
        ]
    
    def set_config_options(self, options):
        needs_reinit = False
        
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    old_val = getattr(self, key)
                    setattr(self, key, value)
                    
                    # Check if we need to reinitialize
                    if key in ['grid_size', 'n_v1_layers', 'base_symmetry', 'symmetry_multiplier']:
                        if old_val != value:
                            needs_reinit = True
        
        if needs_reinit:
            self._init_layers()
    
    # === STATE PERSISTENCE ===
    def save_custom_state(self, folder_path, node_id):
        """Save current state including learned weights."""
        filename = f"visual_cortex_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        
        np.savez(filepath,
                 # Architecture
                 grid_size=self.grid_size,
                 n_v1_layers=self.n_v1_layers,
                 base_symmetry=self.base_symmetry,
                 symmetry_multiplier=self.symmetry_multiplier,
                 # Neural state
                 v=self.v,
                 u=self.u,
                 # Learned weights
                 weights=self.weights,
                 inter_weights=self.inter_weights,
                 # Stats
                 step_count=self.step_count,
                 total_spikes=self.total_spikes,
                 learning_rate=self.learning_rate)
        
        return filename
    
    def load_custom_state(self, filepath):
        """Load saved state."""
        try:
            data = np.load(filepath, allow_pickle=True)
            
            # Architecture
            self.grid_size = int(data['grid_size'])
            self.n_v1_layers = int(data['n_v1_layers'])
            self.base_symmetry = float(data['base_symmetry'])
            self.symmetry_multiplier = float(data['symmetry_multiplier'])
            
            # Reinitialize with loaded architecture
            self._init_layers()
            
            # Load state
            self.v = data['v']
            self.u = data['u']
            self.weights = data['weights']
            self.inter_weights = data['inter_weights']
            
            # Stats
            self.step_count = int(data['step_count'])
            self.total_spikes = int(data['total_spikes'])
            self.learning_rate = float(data['learning_rate'])
            
            print(f"[V1Cortex] Loaded state: {self.step_count} steps, {self.total_spikes:,} spikes")
            
        except Exception as e:
            print(f"[V1Cortex] Error loading state: {e}")
    
    def load_from_crystal(self, crystal_path):
        """
        Initialize weights from a CrystalChip .npz file.
        This allows the visual cortex to inherit learned patterns from EEG training.
        """
        try:
            data = np.load(crystal_path, allow_pickle=True)
            
            crystal_size = int(data['grid_size'])
            
            # If crystal matches our grid size, use its weights directly
            if crystal_size == self.grid_size:
                # Use crystal weights for the IT layer (last layer)
                self.weights[-1, 0] = data['weights_up']
                self.weights[-1, 1] = data['weights_down']
                self.weights[-1, 2] = data['weights_left']
                self.weights[-1, 3] = data['weights_right']
                
                print(f"[V1Cortex] Loaded crystal weights into IT layer")
            else:
                # Resize crystal weights
                for i in range(4):
                    direction_names = ['weights_up', 'weights_down', 'weights_left', 'weights_right']
                    crystal_w = data[direction_names[i]]
                    resized = cv2.resize(crystal_w, (self.grid_size, self.grid_size))
                    self.weights[-1, i] = resized
                
                print(f"[V1Cortex] Resized crystal {crystal_size}→{self.grid_size} into IT layer")
                
        except Exception as e:
            print(f"[V1Cortex] Error loading crystal: {e}")


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("Artificial Visual Cortex Node - Standalone Test")
    print("=" * 50)
    
    # Create node
    node = ArtificialVisualCortexNode()
    
    # Create test image (gradient with some structure)
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            # Radial gradient
            r = np.sqrt((x-128)**2 + (y-128)**2)
            test_img[y, x] = [int(r), int(255-r), 128]
    
    # Add some rectangles (like cabin boards)
    for i in range(5):
        y = 40 + i * 40
        cv2.rectangle(test_img, (20, y), (236, y+30), (100+i*30, 80, 50), -1)
    
    print(f"Created test image: {test_img.shape}")
    print(f"Architecture: {node.n_v1_layers} V1 layers + IT at {node.grid_size}x{node.grid_size}")
    print(f"Symmetry levels: {node.symmetry_levels}")
    
    # Simulate some steps
    print("\nRunning 100 steps...")
    for i in range(100):
        # Inject image
        node.processed_layers = np.zeros_like(node.processed_layers)
        for j, sym in enumerate(node.symmetry_levels):
            node.processed_layers[j] = node._process_v1_layer(test_img, sym)
        node.processed_layers[-1] = node._process_it_layer(test_img)
        
        node.step()
        
        if (i + 1) % 25 == 0:
            print(f"  Step {i+1}: Resonance={node.current_resonance:.2f}, "
                  f"Energy={node.current_energy:.0f}, Spikes={node.total_spikes}")
    
    print("\nGenerating outputs...")
    
    # Get outputs
    v1_mandala = node._render_v1_mandala()
    it_layer = node._render_it_layer()
    depth_signal = node._render_depth_signal()
    eigenstate = node._render_eigenstate()
    activity = node._render_activity()
    depth_profile = node._compute_depth_profile()
    
    print(f"V1 Mandala: {v1_mandala.shape}")
    print(f"IT Layer: {it_layer.shape}")
    print(f"Depth Signal: {depth_signal.shape}")
    print(f"Eigenstate: {eigenstate.shape}")
    print(f"Activity: {activity.shape}")
    print(f"Depth Profile: {depth_profile.shape} (1D radial)")
    
    # Save test outputs
    cv2.imwrite("test_v1_mandala.png", v1_mandala)
    cv2.imwrite("test_it_layer.png", it_layer)
    cv2.imwrite("test_depth.png", depth_signal)
    cv2.imwrite("test_eigenstate.png", eigenstate)
    cv2.imwrite("test_activity.png", activity)
    
    print("\nTest images saved!")
    print("Done.")
