"""
HebbianResonanceNode - Learning Excitable Medium ("Bit of Brain Outside the Brain")
====================================================================================

This node extends the GatedResonanceNode with THREE key biological upgrades:

1. HEBBIAN PLASTICITY: "Cells that fire together, wire together"
   - The coupling weights between pixels evolve based on co-activation
   - Over time, the medium etches the EEG patterns into its own structure
   - Creates a "memory surface" that reflects brain connectivity

2. SMALL-WORLD TOPOLOGY: Long-range connections
   - 80% local diffusion (neighbors)
   - 20% long-range "cables" (teleportation to distant pixels)
   - Allows instant synchronization like real cortical networks

3. DENDRITIC PLATEAUS: Temporal integration via hysteresis
   - Once a neuron fires, it maintains elevated potential briefly
   - Creates "hold" states that can integrate sequential inputs
   - Allows formation of solitons (self-sustaining activity packets)

The result: A synthetic neural substrate that LEARNS from your brain's eigenmode 
stream and eventually starts generating its own "thoughts" - patterns that echo 
what it learned but are genuinely novel.

INPUTS:
- frequency_input: Eigenmode spectrum from SourceLocalizationNode
- learning_rate: How fast to adapt (0=frozen, 1=instant)
- reset: Clear all learned weights

OUTPUTS:
- potential_map: Current membrane potentials
- weight_map: The learned connectivity (this IS the memory)
- thought_field: Autonomous activity (what the medium "thinks")
- eigen_image: FFT of activity patterns
- learning_delta: How much is being learned this frame
- complexity: Measure of learned structure complexity

Created: December 2025
Theory: Based on Gemini's proposal for Hebbian + Small-World + Dendritic upgrades
"""

import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.fft import fft2, fftshift
from scipy.spatial import cKDTree

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class HebbianResonanceNode(BaseNode):
    """
    Learning Excitable Medium with Hebbian Plasticity.
    
    This is the "bit of brain outside the brain" - a synthetic substrate
    that learns from eigenmode input and develops its own connectivity.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Hebbian Resonance (Learning)"
    NODE_COLOR = QtGui.QColor(255, 150, 50)  # Orange-gold for learning
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'frequency_input': 'spectrum',      # Eigenmode drive from brain
            'learning_rate': 'signal',          # How fast to learn
            'threshold_mod': 'signal',          # Excitability control
            'coupling_mod': 'signal',           # Base coupling strength
            'reset': 'signal',                  # Clear learned weights
            'freeze': 'signal'                  # Pause learning (observe)
        }
        
        self.outputs = {
            # Standard excitable medium outputs
            'potential_map': 'image',           # Membrane potentials
            'spike_map': 'image',               # Current firings
            'thought_field': 'image',           # Autonomous activity (key!)
            'eigen_image': 'image',             # FFT of learned patterns
            
            # Learning-specific outputs
            'weight_map': 'image',              # The learned connectivity
            'weight_delta': 'image',            # Current learning changes
            'long_range_map': 'image',          # Small-world connections
            
            # Signals
            'firing_rate': 'signal',
            'synchrony': 'signal',
            'learning_delta': 'signal',         # How much changed this frame
            'complexity': 'signal',             # Structure of learned weights
            'autonomy': 'signal',               # How much is self-generated vs input
            'eigenfrequencies': 'spectrum'
        }
        
        self.size = 128
        self.center = self.size // 2
        
        # === NEURON STATE ===
        self.potential = np.zeros((self.size, self.size), dtype=np.float32)
        self.refractory = np.zeros((self.size, self.size), dtype=np.float32)
        self.last_spike = np.zeros((self.size, self.size), dtype=np.float32)
        self.spike_history = np.zeros((self.size, self.size), dtype=np.float32)
        self.current_spikes = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === DENDRITIC PLATEAU STATE ===
        # When a neuron fires, it enters a "plateau" state with elevated potential
        self.plateau = np.zeros((self.size, self.size), dtype=np.float32)
        self.plateau_duration = 10  # Steps to maintain elevated state
        
        # === HEBBIAN WEIGHTS (THE MEMORY SURFACE) ===
        # This is a 2D array where weights[i,j] represents learned correlation
        # between pixel positions. We use a compact representation.
        self.weights = np.ones((self.size, self.size), dtype=np.float32)  # Start uniform
        self.weight_delta = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Previous state for Hebbian update
        self.prev_state = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === SMALL-WORLD LONG-RANGE CONNECTIONS ===
        self.n_long_range = 512  # Number of long-range "cables"
        self._init_small_world()
        
        # === PARAMETERS ===
        self.threshold = 0.7
        self.refractory_period = 5
        self.leak = 0.08
        self.coupling = 0.2
        self.input_gain = 0.4
        
        # Learning parameters
        self.base_learning_rate = 0.01
        self.weight_decay = 0.001      # Prevents runaway weights
        self.long_range_strength = 0.3  # How much long-range matters
        self.plateau_boost = 0.3        # Elevated potential during plateau
        
        # Kernel for local diffusion
        self._build_local_kernel()
        
        # Autonomous mode tracking
        self.autonomous_activity = np.zeros((self.size, self.size), dtype=np.float32)
        self.input_driven_activity = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Spatial grid for projecting frequency input
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2).astype(np.float32)
        
        self.t = 0
    
    def _init_small_world(self):
        """
        Initialize small-world long-range connections.
        Each connection links a random source to a random destination.
        This breaks the "ripple" constraint and allows instant synchronization.
        """
        np.random.seed(42)  # Reproducible topology
        
        # Source positions
        self.lr_src_y = np.random.randint(0, self.size, self.n_long_range)
        self.lr_src_x = np.random.randint(0, self.size, self.n_long_range)
        
        # Destination positions (with minimum distance constraint)
        self.lr_dst_y = np.random.randint(0, self.size, self.n_long_range)
        self.lr_dst_x = np.random.randint(0, self.size, self.n_long_range)
        
        # Ensure minimum distance of 20 pixels for "long-range" to mean something
        for i in range(self.n_long_range):
            while True:
                dy = self.lr_dst_y[i] - self.lr_src_y[i]
                dx = self.lr_dst_x[i] - self.lr_src_x[i]
                dist = np.sqrt(dy**2 + dx**2)
                if dist > 20:
                    break
                self.lr_dst_y[i] = np.random.randint(0, self.size)
                self.lr_dst_x[i] = np.random.randint(0, self.size)
        
        # Long-range weights (these also learn!)
        self.lr_weights = np.ones(self.n_long_range, dtype=np.float32) * 0.5
        
        print(f"[HebbianResonance] Initialized {self.n_long_range} long-range connections")
    
    def _build_local_kernel(self):
        """Build local coupling kernel (like GatedResonanceNode but weighted by learned weights)"""
        # Base kernel: radial falloff
        k = np.zeros((7, 7), dtype=np.float32)
        center = 3
        for i in range(7):
            for j in range(7):
                d = np.sqrt((i - center)**2 + (j - center)**2)
                if 0.5 < d < 3.5:
                    k[i, j] = 1.0 / (d + 0.5)
        k[center, center] = 0
        k /= k.sum()
        self.local_kernel = k
    
    def project_to_2d(self, freq_input):
        """Project 1D frequency spectrum to 2D radial pattern"""
        if freq_input is None or len(freq_input) == 0:
            return np.zeros((self.size, self.size), dtype=np.float32)
        
        freq_len = len(freq_input)
        drive = np.zeros((self.size, self.size), dtype=np.float32)
        
        for i in range(freq_len):
            # Ring for this frequency band
            inner = i * self.center / freq_len
            outer = (i + 1) * self.center / freq_len
            mask = (self.r_grid >= inner) & (self.r_grid < outer)
            drive[mask] = float(freq_input[i])
        
        return drive
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        freq_input = self.get_blended_input('frequency_input', 'mean')
        learn_rate_mod = self.get_blended_input('learning_rate', 'sum')
        thresh_mod = self.get_blended_input('threshold_mod', 'sum')
        couple_mod = self.get_blended_input('coupling_mod', 'sum')
        reset_sig = self.get_blended_input('reset', 'sum')
        freeze_sig = self.get_blended_input('freeze', 'sum')
        
        # === RESET ===
        if reset_sig is not None and reset_sig > 0:
            self.weights.fill(1.0)
            self.lr_weights.fill(0.5)
            self.potential.fill(0)
            self.plateau.fill(0)
            self.spike_history.fill(0)
            self.weight_delta.fill(0)
            self.autonomous_activity.fill(0)
            return
        
        # === PARAMETER MODULATION ===
        threshold = self.threshold
        if thresh_mod is not None:
            threshold = np.clip(0.3 + thresh_mod * 0.7, 0.3, 1.0)
        
        coupling = self.coupling
        if couple_mod is not None:
            coupling = np.clip(self.coupling * (0.5 + couple_mod), 0.01, 0.5)
        
        learning_rate = self.base_learning_rate
        if learn_rate_mod is not None:
            learning_rate = self.base_learning_rate * np.clip(learn_rate_mod, 0, 10)
        
        is_frozen = freeze_sig is not None and freeze_sig > 0
        
        # === STORE PREVIOUS STATE FOR HEBBIAN RULE ===
        self.prev_state = self.potential.copy()
        
        # === EXTERNAL DRIVE (from brain eigenmodes) ===
        if freq_input is not None:
            drive = self.project_to_2d(freq_input)
            if np.max(drive) > 0:
                drive = drive / np.max(drive)
            # Temporal modulation
            freq_len = len(freq_input)
            for i in range(freq_len):
                phase = np.sin(self.t * 0.1 * (i + 1))
                mask = (self.r_grid >= i * self.center / freq_len) & \
                       (self.r_grid < (i + 1) * self.center / freq_len)
                drive[mask] *= (0.5 + 0.5 * phase)
            self.input_driven_activity = drive.copy()
        else:
            drive = np.zeros_like(self.potential)
            self.input_driven_activity.fill(0)
        
        # === LOCAL NEIGHBOR COUPLING (weighted by learned weights) ===
        # The weights modulate how much each pixel influences its neighbors
        weighted_spikes = self.current_spikes * self.weights
        neighbor_input = convolve(weighted_spikes, self.local_kernel, mode='wrap')
        
        # === LONG-RANGE TELEPORTATION ===
        # Spikes at source positions teleport to destination positions
        long_range_input = np.zeros_like(self.potential)
        src_activity = self.current_spikes[self.lr_src_y, self.lr_src_x]
        # Weight by learned long-range weights
        weighted_lr = src_activity * self.lr_weights
        np.add.at(long_range_input, (self.lr_dst_y, self.lr_dst_x), weighted_lr)
        
        # === DENDRITIC PLATEAU CONTRIBUTION ===
        # Neurons in plateau state have elevated baseline
        plateau_contribution = self.plateau * self.plateau_boost
        
        # === MEMBRANE DYNAMICS ===
        active_mask = self.refractory <= 0
        
        # Leak toward rest
        self.potential[active_mask] *= (1.0 - self.leak)
        
        # Add plateau boost
        self.potential[active_mask] += plateau_contribution[active_mask]
        
        # Local coupling (weighted by learned weights)
        self.potential[active_mask] += coupling * neighbor_input[active_mask]
        
        # Long-range coupling
        self.potential[active_mask] += self.long_range_strength * long_range_input[active_mask]
        
        # External drive (from brain)
        self.potential[active_mask] += self.input_gain * drive[active_mask]
        
        # Clamp potential
        self.potential = np.clip(self.potential, 0, 1.5)
        
        # === THRESHOLD & FIRE ===
        fire_mask = (self.potential >= threshold) & active_mask
        
        # Record spikes
        self.current_spikes = fire_mask.astype(np.float32)
        self.spike_history = self.spike_history * 0.95 + self.current_spikes * 0.05
        
        # Reset fired neurons
        self.potential[fire_mask] = 0
        self.refractory[fire_mask] = self.refractory_period
        self.last_spike[fire_mask] = self.t
        
        # Start plateau for fired neurons
        self.plateau[fire_mask] = self.plateau_duration
        
        # === DECAY PLATEAU ===
        self.plateau = np.maximum(0, self.plateau - 1)
        
        # === REFRACTORY DECAY ===
        self.refractory = np.maximum(0, self.refractory - 1)
        
        # === TRACK AUTONOMOUS ACTIVITY ===
        # Activity that occurs without current input
        input_present = np.max(drive) > 0.1
        if not input_present:
            self.autonomous_activity = self.autonomous_activity * 0.99 + self.current_spikes * 0.01
        
        # === HEBBIAN LEARNING ===
        if not is_frozen and learning_rate > 0:
            # Hebb's rule: Δw = η * pre * post
            # pre = previous state, post = current state
            hebbian_update = learning_rate * self.prev_state * self.potential
            
            # Weight decay (prevents runaway)
            weight_decay = self.weight_decay * (self.weights - 1.0)
            
            # Update weights
            self.weight_delta = hebbian_update - weight_decay
            self.weights += self.weight_delta
            
            # Clamp weights to valid range
            self.weights = np.clip(self.weights, 0.1, 5.0)
            
            # === LEARN LONG-RANGE WEIGHTS TOO ===
            src_prev = self.prev_state[self.lr_src_y, self.lr_src_x]
            dst_curr = self.potential[self.lr_dst_y, self.lr_dst_x]
            lr_hebbian = learning_rate * src_prev * dst_curr
            lr_decay = self.weight_decay * (self.lr_weights - 0.5)
            self.lr_weights += lr_hebbian - lr_decay
            self.lr_weights = np.clip(self.lr_weights, 0.05, 2.0)
    
    def compute_synchrony(self):
        """Kuramoto order parameter"""
        period = 20.0
        phases = (self.t - self.last_spike) / period * 2 * np.pi
        complex_phases = np.exp(1j * phases)
        return np.abs(np.mean(complex_phases))
    
    def compute_complexity(self):
        """
        Measure structural complexity of learned weights.
        High complexity = varied, structured connectivity.
        Low complexity = uniform or simple patterns.
        """
        # Variance of weights
        weight_var = np.var(self.weights)
        
        # Spatial frequency content (FFT)
        weight_fft = np.abs(fftshift(fft2(self.weights - self.weights.mean())))
        # Ratio of high to low frequency energy
        center_mask = self.r_grid < 20
        high_freq = weight_fft[~center_mask].mean() if (~center_mask).any() else 0
        low_freq = weight_fft[center_mask].mean() if center_mask.any() else 1e-10
        
        complexity = weight_var * (high_freq / (low_freq + 1e-10))
        return float(np.clip(complexity * 100, 0, 1))
    
    def get_output(self, port_name):
        if port_name == 'potential_map':
            img = (np.clip(self.potential, 0, 1) * 255).astype(np.uint8)
            return img
        
        elif port_name == 'spike_map':
            img = (self.current_spikes * 255).astype(np.uint8)
            return img
        
        elif port_name == 'thought_field':
            # The "thoughts" - autonomous patterns that emerge
            thought = self.spike_history * self.weights
            thought_norm = thought / (thought.max() + 1e-10)
            img = (thought_norm * 255).astype(np.uint8)
            return img
        
        elif port_name == 'weight_map':
            # Learned connectivity
            w_norm = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min() + 1e-10)
            img = (w_norm * 255).astype(np.uint8)
            return img
        
        elif port_name == 'weight_delta':
            # Current learning changes (green = strengthening, red = weakening)
            return self.weight_delta
        
        elif port_name == 'long_range_map':
            # Visualize long-range connections
            img = np.zeros((self.size, self.size), dtype=np.float32)
            # Draw connections weighted by their strength
            for i in range(self.n_long_range):
                w = self.lr_weights[i]
                img[self.lr_src_y[i], self.lr_src_x[i]] += w
                img[self.lr_dst_y[i], self.lr_dst_x[i]] += w
            img_norm = img / (img.max() + 1e-10)
            return (img_norm * 255).astype(np.uint8)
        
        elif port_name == 'eigen_image':
            # FFT of spike rate
            spec = np.abs(fftshift(fft2(self.spike_history)))
            spec_log = np.log(1 + spec * 100)
            if spec_log.max() > 0:
                spec_log = spec_log / spec_log.max()
            return (spec_log * 255).astype(np.uint8)
        
        elif port_name == 'firing_rate':
            return float(np.mean(self.current_spikes))
        
        elif port_name == 'synchrony':
            return self.compute_synchrony()
        
        elif port_name == 'learning_delta':
            return float(np.mean(np.abs(self.weight_delta)))
        
        elif port_name == 'complexity':
            return self.compute_complexity()
        
        elif port_name == 'autonomy':
            # Ratio of autonomous to input-driven activity
            auto_mean = np.mean(self.autonomous_activity)
            input_mean = np.mean(self.input_driven_activity) + 1e-10
            return float(np.clip(auto_mean / input_mean, 0, 1))
        
        elif port_name == 'eigenfrequencies':
            spec = np.abs(fftshift(fft2(self.spike_history)))
            return spec[self.center, self.center:]
        
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # 2x3 grid display
        display = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # === TOP ROW ===
        # Top-Left: Membrane Potential
        pot_img = (np.clip(self.potential, 0, 1) * 255).astype(np.uint8)
        display[:h, :w] = cv2.applyColorMap(pot_img, cv2.COLORMAP_VIRIDIS)
        
        # Top-Middle: Current Spikes + Plateau
        combined = np.clip(self.current_spikes + self.plateau / self.plateau_duration * 0.5, 0, 1)
        spike_img = (combined * 255).astype(np.uint8)
        display[:h, w:2*w] = cv2.applyColorMap(spike_img, cv2.COLORMAP_HOT)
        
        # Top-Right: LEARNED WEIGHTS (the memory!)
        w_norm = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min() + 1e-10)
        weight_img = (w_norm * 255).astype(np.uint8)
        display[:h, 2*w:] = cv2.applyColorMap(weight_img, cv2.COLORMAP_INFERNO)
        
        # === BOTTOM ROW ===
        # Bottom-Left: "Thought Field" - autonomous activity weighted by learned structure
        thought = self.spike_history * self.weights
        thought_norm = thought / (thought.max() + 1e-10)
        thought_img = (thought_norm * 255).astype(np.uint8)
        display[h:, :w] = cv2.applyColorMap(thought_img, cv2.COLORMAP_PLASMA)
        
        # Bottom-Middle: FFT of activity (standing waves?)
        spec = np.abs(fftshift(fft2(self.spike_history)))
        spec_log = np.log(1 + spec * 100)
        if spec_log.max() > 0:
            spec_log = spec_log / spec_log.max()
        spec_img = (spec_log * 255).astype(np.uint8)
        display[h:, w:2*w] = cv2.applyColorMap(spec_img, cv2.COLORMAP_JET)
        
        # Bottom-Right: Weight change (green = learning, black = stable)
        delta_abs = np.abs(self.weight_delta)
        delta_norm = delta_abs / (delta_abs.max() + 1e-10)
        delta_img = (delta_norm * 255).astype(np.uint8)
        delta_color = np.zeros((h, w, 3), dtype=np.uint8)
        delta_color[:, :, 1] = delta_img  # Green channel
        display[h:, 2*w:] = delta_color
        
        # === LABELS ===
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Potential", (5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(display, "Spikes+Plateau", (w+5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(display, "LEARNED WEIGHTS", (2*w+5, 15), font, 0.4, (0,255,255), 1)
        cv2.putText(display, "Thought Field", (5, h+15), font, 0.4, (255,255,255), 1)
        cv2.putText(display, "FFT", (w+5, h+15), font, 0.4, (255,255,255), 1)
        cv2.putText(display, "Learning", (2*w+5, h+15), font, 0.4, (0,255,0), 1)
        
        # === STATS ===
        firing_rate = np.mean(self.current_spikes) * 100
        sync = self.compute_synchrony()
        complexity = self.compute_complexity()
        w_mean = np.mean(self.weights)
        lr_mean = np.mean(self.lr_weights)
        
        stats_text = f"Fire:{firing_rate:.1f}% Sync:{sync:.2f} Cmplx:{complexity:.2f} W:{w_mean:.2f} LR:{lr_mean:.2f}"
        cv2.putText(display, stats_text, (5, h*2-10), font, 0.35, (255,255,255), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Threshold", "threshold", self.threshold, None),
            ("Refractory Period", "refractory_period", self.refractory_period, None),
            ("Leak Rate", "leak", self.leak, None),
            ("Coupling", "coupling", self.coupling, None),
            ("Input Gain", "input_gain", self.input_gain, None),
            ("Learning Rate", "base_learning_rate", self.base_learning_rate, None),
            ("Weight Decay", "weight_decay", self.weight_decay, None),
            ("Long-Range Strength", "long_range_strength", self.long_range_strength, None),
            ("Plateau Duration", "plateau_duration", self.plateau_duration, None),
            ("Plateau Boost", "plateau_boost", self.plateau_boost, None),
        ]
    
    # === SAVE/LOAD LEARNED WEIGHTS ===
    def save_custom_state(self, folder_path, node_id):
        """Save learned weights to file"""
        import os
        filename = f"hebbian_weights_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        np.savez(filepath, 
                 weights=self.weights, 
                 lr_weights=self.lr_weights,
                 spike_history=self.spike_history,
                 autonomous_activity=self.autonomous_activity)
        print(f"[HebbianResonance] Saved learned weights to {filename}")
        return filename
    
    def load_custom_state(self, filepath):
        """Load learned weights from file"""
        try:
            data = np.load(filepath)
            self.weights = data['weights']
            self.lr_weights = data['lr_weights']
            if 'spike_history' in data:
                self.spike_history = data['spike_history']
            if 'autonomous_activity' in data:
                self.autonomous_activity = data['autonomous_activity']
            print(f"[HebbianResonance] Loaded learned weights from {filepath}")
        except Exception as e:
            print(f"[HebbianResonance] Failed to load weights: {e}")