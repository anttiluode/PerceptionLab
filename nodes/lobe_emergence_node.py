"""
Lobe Emergence Node - Demonstrates how brain lobes emerge from W-matrix optimization
Shows the 'ghost cortex' - spatial localization of frequency filters through learning.

This node bridges:
- IHT Phase Field (quantum substrate)
- W Matrix (holographic decoder)
- Brain Lobes (emergent spatial structure)

Key insight: Lobes aren't designed - they EMERGE from optimization.

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import fft, ifft, fft2, ifft2
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: LobeEmergenceNode requires scipy")

class LobeEmergenceNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(200, 100, 200)  # Purple for emergence
    
    def __init__(self, grid_size=24, learning_rate=0.01, damage_location='None', initialization='Random'):
        super().__init__()
        self.node_title = "Lobe Emergence"
        self.initialization = initialization
        
        self.inputs = {
            'phase_field': 'image',        # Input quantum state
            'train_signal': 'signal',      # Trigger training
            'damage_amount': 'signal',     # How much damage to apply
        }
        
        self.outputs = {
            'ghost_cortex': 'image',           # 2D frequency map (the "lobes")
            'lobe_structure': 'image',         # Segmented lobe regions
            'emergence_metric': 'signal',       # How separated are lobes?
            'theta_lobe': 'image',             # Individual lobe outputs
            'alpha_lobe': 'image',
            'gamma_lobe': 'image',
            'cross_frequency_leakage': 'signal'
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Lobe Emergence (No SciPy!)"
            return
        
        self.grid_size = int(grid_size)
        self.learning_rate = float(learning_rate)
        self.damage_location = damage_location
        
        # The W matrix (complex) - starts random, will develop structure
        self.W = None
        self.training_steps = 0
        
        # State trackers for config changes
        self._last_init_mode = self.initialization
        self._last_grid_size = self.grid_size
        
        # --- FIX: Moved this block *before* _init_W() is called ---
        # Frequency bands (Hz equivalents in normalized units)
        self.freq_bands = {
            'theta': (0.05, 0.15),   # Low frequency
            'alpha': (0.15, 0.30),   # Mid frequency
            'gamma': (0.50, 0.90)    # High frequency
        }
        # --- END FIX ---
        
        self._init_W() # Build the W matrix
        
        # Throttle updates
        self.steps_since_last_visual_update = 0
        self.visual_update_interval = 5  # Only update visualization every N training steps
        
        # Outputs
        self.ghost_cortex_img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        self.lobe_structure_img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        self.emergence_score = 0.0
        self.leakage_score = 0.0
        
        # Lobe-specific outputs
        self.theta_lobe_img = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.alpha_lobe_img = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.gamma_lobe_img = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
    def _init_W(self):
        """Initialize W matrix with small random complex values"""
        
        # Update trackers when W is (re)built
        self._last_init_mode = self.initialization
        self._last_grid_size = self.grid_size
        self.training_steps = 0
        
        n = self.grid_size * self.grid_size
        
        if self.initialization == 'Random':
            # --- FIX: Start with pure noise, not a structured identity matrix ---
            # Pure random (slow to converge)
            noise_scale = 0.05 
            real_noise = np.random.randn(n, n) * noise_scale
            imag_noise = np.random.randn(n, n) * noise_scale
            self.W = (real_noise + 1j * imag_noise).astype(np.complex64)
            # --- END FIX ---
            
        elif self.initialization == 'Frequency-Biased':
            # Pre-bias W to prefer spatial frequency separation
            self.W = np.zeros((n, n), dtype=np.complex64)
            
            for i in range(n):
                y_i = i // self.grid_size
                x_i = i % self.grid_size
                
                if y_i < self.grid_size // 3:
                    freq_preference = 'theta'
                    phase_offset = 0.0
                elif y_i < 2 * self.grid_size // 3:
                    freq_preference = 'alpha'
                    phase_offset = np.pi / 3
                else:
                    freq_preference = 'gamma'
                    phase_offset = 2 * np.pi / 3
                
                for j in range(n):
                    y_j = j // self.grid_size
                    x_j = j % self.grid_size
                    dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    
                    if freq_preference == 'theta':
                        strength = np.exp(-dist / 8.0)
                        freq_mod = np.cos(dist * 0.2 + phase_offset)
                    elif freq_preference == 'alpha':
                        strength = np.exp(-dist / 5.0)
                        freq_mod = np.cos(dist * 0.5 + phase_offset)
                    else:  # gamma
                        strength = np.exp(-dist / 3.0)
                        freq_mod = np.cos(dist * 1.0 + phase_offset)
                    
                    self.W[i, j] = strength * freq_mod * (1.0 + 0.1j)
            
            noise_scale = 0.01
            self.W += (np.random.randn(n, n) + 1j * np.random.randn(n, n)) * noise_scale
            
            # --- FIX: Moved this loop inside the 'Frequency-Biased' block ---
            # It should not run for the 'Random' mode.
            # Encourage spatial locality
            for i in range(n):
                y_i = i // self.grid_size
                x_i = i % self.grid_size
                for j in range(n):
                    y_j = j // self.grid_size
                    x_j = j % self.grid_size
                    dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    if dist < 5.0:
                        self.W[i, j] += 0.1 * np.exp(-dist / 2.0)
            # --- END FIX ---

        # --- FIX: The "Encourage spatial locality" loop was here and has been moved. ---
        
        # Immediately compute the visual state after init
        self.ghost_cortex_img = self._compute_ghost_cortex(self.W)
        self.lobe_structure_img = self._segment_lobes(self.ghost_cortex_img)
        self.emergence_score = self._compute_emergence_metric(self.ghost_cortex_img)
        self.leakage_score = self._compute_cross_frequency_leakage(self.ghost_cortex_img)
        
    def _apply_damage(self, W, damage_amount):
        """Apply damage to specific lobe region"""
        if self.damage_location == 'None' or damage_amount < 0.01:
            return W
        
        h, w = self.grid_size, self.grid_size
        W_damaged = W.copy()
        
        damage_masks = {
            'theta': self._get_region_mask(0, 0, h//2, w//2),
            'alpha': self._get_region_mask(0, w//2, h//2, w),
            'gamma': self._get_region_mask(h//2, 0, h, w//2),
        }
        
        if self.damage_location in damage_masks:
            mask_flat = damage_masks[self.damage_location].flatten()
            for i in range(len(mask_flat)):
                if mask_flat[i]:
                    noise = (np.random.randn(W.shape[1]) + 1j * np.random.randn(W.shape[1])) * damage_amount * 0.3
                    W_damaged[i, :] += noise.astype(np.complex64)
                    W_damaged[i, :] *= (1.0 - damage_amount * 0.5)
        return W_damaged
    
    def _get_region_mask(self, y_start, x_start, y_end, x_end):
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        mask[y_start:y_end, x_start:x_end] = True
        return mask
    

    def _compute_ghost_cortex(self, W):
        h, w = self.grid_size, self.grid_size
        ghost_cortex = np.zeros((h, w, 3), dtype=np.float32)
        test_signals = {}
        
        for freq_name, (low, high) in self.freq_bands.items():
            center_freq = (low + high) / 2.0
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            spatial_wave = np.sin(x_coords * center_freq * np.pi + y_coords * center_freq * np.pi * 0.7)
            test_signals[freq_name] = spatial_wave.flatten().astype(np.complex64)
        
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                if idx >= W.shape[0]: continue
                W_row = W[idx, :]
                
                theta_response = np.abs(np.dot(W_row, test_signals['theta']))
                alpha_response = np.abs(np.dot(W_row, test_signals['alpha']))
                gamma_response = np.abs(np.dot(W_row, test_signals['gamma']))
                
                total = theta_response + alpha_response + gamma_response + 1e-9
                ghost_cortex[i, j, 0] = theta_response / total
                ghost_cortex[i, j, 1] = alpha_response / total
                ghost_cortex[i, j, 2] = gamma_response / total
        
        for c in range(3):
            ghost_cortex[:, :, c] = gaussian_filter(ghost_cortex[:, :, c], sigma=1.0)
        
        return ghost_cortex
    
    def _segment_lobes(self, ghost_cortex):
        dominant = np.argmax(ghost_cortex, axis=2)
        segmented = np.zeros_like(ghost_cortex)
        
        theta_mask = (dominant == 0)
        segmented[theta_mask] = [1.0, 0.0, 0.0]
        
        alpha_mask = (dominant == 1)
        segmented[alpha_mask] = [0.0, 1.0, 0.0]
        
        gamma_mask = (dominant == 2)
        segmented[gamma_mask] = [0.0, 0.0, 1.0]
        
        self.theta_lobe_img = theta_mask.astype(np.float32)
        self.alpha_lobe_img = alpha_mask.astype(np.float32)
        self.gamma_lobe_img = gamma_mask.astype(np.float32)
        
        return segmented
    
    def _compute_emergence_metric(self, ghost_cortex):
        r_var = np.var(ghost_cortex[:, :, 0])
        g_var = np.var(ghost_cortex[:, :, 1])
        b_var = np.var(ghost_cortex[:, :, 2])
        separation = (r_var + g_var + b_var) / 3.0
        separation = np.tanh(separation * 20.0)
        return float(separation)
    
    def _compute_cross_frequency_leakage(self, ghost_cortex):
        dominant = np.argmax(ghost_cortex, axis=2)
        h, w = ghost_cortex.shape[:2]
        leakage_sum = 0.0
        for i in range(h):
            for j in range(w):
                dom_idx = dominant[i, j]
                dom_power = ghost_cortex[i, j, dom_idx]
                other_power = 1.0 - dom_power
                leakage_sum += other_power
        leakage = leakage_sum / (h * w)
        return float(leakage)
    
    def _train_W_step(self, phase_field):
        """
        One gradient descent step to train W. (STABLE VERSION)
        """
        try:
            if phase_field.ndim == 3:
                phase_field = np.mean(phase_field, axis=2)
            
            phase_resized = cv2.resize(phase_field, (self.grid_size, self.grid_size))
            
            if not np.all(np.isfinite(phase_resized)):
                return 
                
            psi_flat = phase_resized.flatten().astype(np.complex64)
            
            psi_norm = np.linalg.norm(psi_flat)
            if psi_norm > 1e-6:
                psi_flat = psi_flat / psi_norm
            else:
                return 

            output = np.dot(self.W, psi_flat)
            
            output_norm = np.linalg.norm(output)
            if output_norm > 1e-6:
                output = output / output_norm
            else:
                output = np.zeros_like(output)

            n_updates = 50
            rows_updated = set() 

            for _ in range(n_updates):
                i_out = np.random.randint(0, self.grid_size)
                j_out = np.random.randint(0, self.grid_size)
                i_in = np.random.randint(0, self.grid_size)
                j_in = np.random.randint(0, self.grid_size)
                
                out_idx = i_out * self.grid_size + j_out
                in_idx = i_in * self.grid_size + j_in
                
                spatial_dist = np.sqrt((i_out - i_in)**2 + (j_out - j_in)**2)
                
                if spatial_dist < 10.0:
                    correlation = output[out_idx] * np.conj(psi_flat[in_idx])
                    
                    MAX_CORR_MAG = 100.0
                    corr_mag = np.abs(correlation)
                    if corr_mag > MAX_CORR_MAG:
                        correlation = correlation * (MAX_CORR_MAG / corr_mag)
                    
                    locality_factor = np.exp(-spatial_dist / 3.0)
                    safe_learning_rate = self.learning_rate * 0.01 
                    update_val = safe_learning_rate * correlation * locality_factor

                    if np.isfinite(update_val):
                        self.W[out_idx, in_idx] += update_val
                        rows_updated.add(out_idx)
            
            for idx in rows_updated:
                row_norm = np.linalg.norm(self.W[idx, :])
                if row_norm > 1.5: 
                    self.W[idx, :] /= row_norm
            
            MAX_W_MAGNITUDE = 5.0 
            np.clip(self.W.real, -MAX_W_MAGNITUDE, MAX_W_MAGNITUDE, out=self.W.real)
            np.clip(self.W.imag, -MAX_W_MAGNITUDE, MAX_W_MAGNITUDE, out=self.W.imag)

            self.training_steps += 1

        except Exception as e:
            print(f"CRITICAL ERROR in _train_W_step, resetting W: {e}")
            self._init_W()
    
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        current_grid_size = int(self.grid_size)
        if (self.initialization != self._last_init_mode or 
            current_grid_size != self._last_grid_size):
            
            print(f"Config changed! Re-initializing W with mode: {self.initialization}")
            self.grid_size = current_grid_size 
            self._init_W() 
            return 
        
        phase_field = self.get_blended_input('phase_field', 'mean')
        train_signal = self.get_blended_input('train_signal', 'sum')
        
        if phase_field is None:
            phase_field = np.random.rand(self.grid_size, self.grid_size).astype(np.float32)
        
        if train_signal is not None and train_signal > 0.5:
            self._train_W_step(phase_field)
    
    def get_output(self, port_name):
        # --- NEW: Re-compute visuals on-demand when output is requested ---
        # This ensures outputs are always fresh, even if the node isn't training
        damage_amount = self.get_blended_input('damage_amount', 'sum')
        damage_amount = np.clip((damage_amount or 0.0) + 1.0, 0, 2.0) / 2.0
        W_current = self._apply_damage(self.W, damage_amount)
        
        # We need to re-compute these here to update the outputs
        ghost_cortex_img = self._compute_ghost_cortex(W_current)
        lobe_structure_img = self._segment_lobes(ghost_cortex_img)
        emergence_score = self._compute_emergence_metric(ghost_cortex_img)
        leakage_score = self._compute_cross_frequency_leakage(ghost_cortex_img)
        # --- END NEW ---

        if port_name == 'ghost_cortex':
            return ghost_cortex_img
        elif port_name == 'lobe_structure':
            return lobe_structure_img
        elif port_name == 'emergence_metric':
            return emergence_score
        elif port_name == 'theta_lobe':
            return self.theta_lobe_img # This is set by _segment_lobes
        elif port_name == 'alpha_lobe':
            return self.alpha_lobe_img
        elif port_name == 'gamma_lobe':
            return self.gamma_lobe_img
        elif port_name == 'cross_frequency_leakage':
            return leakage_score
        return None
    
    def get_display_image(self):
        """
        This function now re-computes the visualization every frame.
        """
        if not SCIPY_AVAILABLE:
            return None
        
        # --- NEW: Re-compute visuals every single frame ---
        damage_amount = self.get_blended_input('damage_amount', 'sum')
        damage_amount = np.clip((damage_amount or 0.0) + 1.0, 0, 2.0) / 2.0
        
        # Apply damage to W *for this frame only*
        W_current = self._apply_damage(self.W, damage_amount)
        
        # Compute ghost cortex (frequency map)
        self.ghost_cortex_img = self._compute_ghost_cortex(W_current)
        
        # Segment into discrete lobes
        self.lobe_structure_img = self._segment_lobes(self.ghost_cortex_img)
        
        # Compute metrics
        self.emergence_score = self._compute_emergence_metric(self.ghost_cortex_img)
        self.leakage_score = self._compute_cross_frequency_leakage(self.ghost_cortex_img)
        # --- END NEW ---
        
        # Create a detailed visualization
        display_h = 256
        display_w = 512
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Left side: Ghost cortex (smooth frequency map)
        ghost_resized = cv2.resize(self.ghost_cortex_img, (display_w//2, display_h))
        ghost_u8 = (np.clip(ghost_resized, 0, 1) * 255).astype(np.uint8)
        display[:, :display_w//2] = ghost_u8
        
        # Right side: Segmented lobes (discrete regions)
        lobe_resized = cv2.resize(self.lobe_structure_img, (display_w//2, display_h))
        lobe_u8 = (np.clip(lobe_resized, 0, 1) * 255).astype(np.uint8)
        display[:, display_w//2:] = lobe_u8
        
        # Add dividing line
        display[:, display_w//2-1:display_w//2+1] = [255, 255, 255]
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Left label
        cv2.putText(display, 'GHOST CORTEX', (10, 20), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, 'GHOST CORTEX', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Right label
        cv2.putText(display, 'LOBES', (display_w//2 + 10, 20), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, 'LOBES', (display_w//2 + 10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add training step counter
        step_text = f"Training: {self.training_steps}"
        cv2.putText(display, step_text, (10, display_h - 10), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display, step_text, (10, display_h - 10), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add emergence metric
        emergence_text = f"Emergence: {self.emergence_score:.2f}"
        cv2.putText(display, emergence_text, (10, display_h - 30), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display, emergence_text, (10, display_h - 30), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Add leakage metric (warning if high)
        leakage_text = f"Leakage: {self.leakage_score:.2f}"
        leakage_color = (0, 0, 255) if self.leakage_score > 0.3 else (200, 200, 200)
        cv2.putText(display, leakage_text, (10, display_h - 50), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display, leakage_text, (10, display_h - 50), font, 0.4, leakage_color, 1, cv2.LINE_AA)
        
        # Add legend (bottom right)
        legend_x = display_w//2 + 10
        legend_y = display_h - 60
        
        cv2.rectangle(display, (legend_x, legend_y), (legend_x + 20, legend_y + 10), (255, 0, 0), -1)
        cv2.putText(display, 'Theta (4-8Hz)', (legend_x + 25, legend_y + 8), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(display, (legend_x, legend_y + 15), (legend_x + 20, legend_y + 25), (0, 255, 0), -1)
        cv2.putText(display, 'Alpha (8-13Hz)', (legend_x + 25, legend_y + 23), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(display, (legend_x, legend_y + 30), (legend_x + 20, legend_y + 40), (0, 0, 255), -1)
        cv2.putText(display, 'Gamma (30-100Hz)', (legend_x + 25, legend_y + 38), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add damage indicator if present
        if self.damage_location != 'None':
            damage_text = f"DAMAGED: {self.damage_location.upper()}"
            cv2.putText(display, damage_text, (display_w//2 + 10, display_h - 10), font, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, damage_text, (display_w//2 + 10, display_h - 10), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Initialization", "initialization", self.initialization, [
                ("Random (Slow)", "Random"),
                ("Frequency-Biased (Fast)", "Frequency-Biased")
            ]),
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Damage Location", "damage_location", self.damage_location, [
                ("None (Healthy)", "None"),
                ("Theta Lobe", "theta"),
                ("Alpha Lobe", "alpha"),
                ("Gamma Lobe", "gamma")
            ]),
        ]
