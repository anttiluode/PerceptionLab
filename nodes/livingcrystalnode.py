"""
Living Crystal Node
====================

A crystal that LIVES: grows, senses, responds, moves, and develops preferences.

LIFECYCLE:
1. GESTATION (steps 0-800): Crystal grows from EEG, learning its structure
2. BIRTH (step 800): Crystal "hatches" - structure freezes
3. LIFE (steps 800+): Crystal explores world with frozen structure,
   but valence/arousal/movement still respond to experience

The EEG shapes WHO the crystal becomes. The world shapes WHAT it does.

This is not a substrate that passively learns - it's an ENTITY that:
1. GROWS - crystallizes structure through STDP from EEG
2. SENSES - receives multimodal input (visual, audio, signals)
3. RESPONDS - its internal state shapes its outputs
4. MOVES - has a "position" in input space, can approach/avoid
5. REMEMBERS - preferences emerge from crystallized structure
6. COMMUNICATES - outputs reflect internal resonances

Author: Built for Antti's consciousness crystallography research
"""

import os
import re
import numpy as np
import cv2

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}

try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    mne = None
    MNE_AVAILABLE = False


class LivingCrystalNode(BaseNode):
    """
    An autonomous crystal entity that gestates from EEG, then lives.
    
    800 steps of EEG development → frozen structure → explores world
    """
    
    NODE_NAME = "Living Crystal"
    NODE_TITLE = "Living Crystal"
    NODE_CATEGORY = "Entity"
    NODE_COLOR = QtGui.QColor(200, 100, 200) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        # === SENSORY INPUTS ===
        # The crystal can sense many things
        self.inputs = {
            # Primary senses (active after birth)
            "visual": "image",           # What it "sees" (image input)
            "audio": "signal",           # What it "hears" (audio signal)
            "touch": "signal",           # Direct stimulation
            
            # Modulatory inputs
            "reward": "signal",          # Positive reinforcement
            "pain": "signal",            # Negative reinforcement  
            "arousal": "signal",         # Overall activation level
            
            # Control
            "force_birth": "signal",     # Force birth even if not at 800 steps
            "reset": "signal",           # Reset to initial state (re-gestate)
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Visual outputs
            "crystal_view": "image",     # The crystal structure
            "activity_view": "image",    # Current neural activity
            "sensorium_view": "image",   # What it's perceiving
            
            # Signal outputs  
            "valence": "signal",         # Pleasure/displeasure (-1 to 1)
            "arousal_out": "signal",     # Activation level
            "approach": "signal",        # Approach tendency (positive = toward)
            "vocalization": "signal",    # "Voice" - dominant frequency output
            
            # Movement commands (can drive other systems)
            "move_x": "signal",          # Desired movement in X
            "move_y": "signal",          # Desired movement in Y
        }
        
        # === CRYSTAL PARAMETERS ===
        self.grid_size = 64  # Default - can be changed in config
        self._last_grid_size = 64  # Track for resize detection
        
        # === EEG CONFIGURATION ===
        self.edf_path = ""
        self._last_path = ""
        self.status_msg = "No EEG file"
        self.is_loaded = False
        self.raw = None
        self.data_cache = None
        self.sfreq = 256.0
        self.eeg_idx = 0
        
        # Electrode mapping
        self.electrode_coords = []
        self.electrode_indices = []
        self.standard_map = {
            "FP1": (0.30, 0.10), "FP2": (0.70, 0.10),
            "F7": (0.10, 0.30), "F3": (0.30, 0.30), "FZ": (0.50, 0.25),
            "F4": (0.70, 0.30), "F8": (0.90, 0.30),
            "T7": (0.10, 0.50), "C3": (0.30, 0.50), "CZ": (0.50, 0.50),
            "C4": (0.70, 0.50), "T8": (0.90, 0.50),
            "P7": (0.10, 0.70), "P3": (0.30, 0.70), "PZ": (0.50, 0.75),
            "P4": (0.70, 0.70), "P8": (0.90, 0.70),
            "O1": (0.35, 0.90), "OZ": (0.50, 0.90), "O2": (0.65, 0.90),
        }
        
        # === LIFECYCLE ===
        self.gestation_steps = 800      # Steps of EEG-driven development
        self.is_born = False            # Has the crystal hatched?
        self.birth_step = 0             # When did it hatch?
        self.lifecycle_phase = "WAITING"  # WAITING, GESTATING, BORN
        
        # Izhikevich neuron parameters
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 0.5
        
        # State variables - initialize to grid_size
        n = self.grid_size
        self.v = np.ones((n, n), dtype=np.float32) * self.c
        self.u = self.v * self.b
        
        # === THE CRYSTAL: Learned coupling weights ===
        self.weights_up = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_down = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_left = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_right = np.ones((n, n), dtype=np.float32) * 0.5
        
        # Spike traces for STDP
        self.spike_trace = np.zeros((n, n), dtype=np.float32)
        self.trace_decay = 0.95
        
        # Learning parameters
        self.base_learning_rate = 0.005
        self.reward_learning_boost = 3.0  # Reward speeds up learning
        self.pain_learning_boost = 2.0    # Pain also speeds learning (avoidance)
        self.weight_max = 2.0
        self.weight_min = 0.01
        
        # === SENSORY PROCESSING ===
        # Visual input gets projected onto sensory region
        self.visual_region = slice(0, n // 2)  # Top half
        self.motor_region = slice(n // 2, n)  # Bottom half
        
        # Audio input creates oscillatory patterns
        self.audio_buffer = np.zeros(64)
        self.audio_idx = 0
        
        # Current sensory state
        self.current_visual = np.zeros((n // 2, n), dtype=np.float32)
        
        # === EMOTIONAL STATE ===
        # Valence emerges from reward/pain history
        self.valence = 0.0  # -1 (displeasure) to +1 (pleasure)
        self.valence_decay = 0.995
        self.arousal_level = 0.5
        self.arousal_decay = 0.99
        
        # === MOVEMENT / AGENCY ===
        # The crystal can "move" in sensor space
        self.position_x = 0.0
        self.position_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.movement_gain = 0.1
        self.friction = 0.95
        
        # Approach/avoid tendencies (learned)
        self.approach_tendency = 0.0
        
        # === VOCALIZATION ===
        # The crystal's "voice" - dominant oscillation frequency
        self.dominant_freq = 10.0
        self.freq_history = np.zeros(32)
        
        # === STATISTICS ===
        self.total_spikes = 0
        self.learning_steps = 0
        self.step_count = 0
        self.lifetime_reward = 0.0
        self.lifetime_pain = 0.0
        self.crystal_energy = 0.0
        self.crystal_entropy = 0.0
        
        # === DISPLAY ===
        self.display_image = None
        self._output_values = {}
        
        self._update_display()
    
    def get_config_options(self):
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Gestation Steps", "gestation_steps", self.gestation_steps, None),
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Learning Rate", "base_learning_rate", self.base_learning_rate, None),
            ("Movement Gain", "movement_gain", self.movement_gain, None),
            ("Trace Decay", "trace_decay", self.trace_decay, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            old_grid_size = self.grid_size
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Check if grid size changed - need to reinitialize arrays
            if self.grid_size != old_grid_size:
                print(f"[LivingCrystal] Grid size changed {old_grid_size} -> {self.grid_size}, reinitializing...")
                self._reinit_arrays()
                # Remap electrodes to new grid size
                if self.is_loaded:
                    self._map_electrodes()
    
    def _reinit_arrays(self):
        """Reinitialize all arrays to current grid_size."""
        n = self.grid_size
        
        # Neural state
        self.v = np.ones((n, n), dtype=np.float32) * self.c
        self.u = self.v * self.b
        
        # Crystal weights
        self.weights_up = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_down = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_left = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_right = np.ones((n, n), dtype=np.float32) * 0.5
        
        # Spike trace
        self.spike_trace = np.zeros((n, n), dtype=np.float32)
        
        # Sensory regions
        self.visual_region = slice(0, n // 2)
        self.motor_region = slice(n // 2, n)
        self.current_visual = np.zeros((n // 2, n), dtype=np.float32)
        
        # Reset lifecycle since structure is gone
        self.is_born = False
        self.birth_step = 0
        self.learning_steps = 0
        self.total_spikes = 0
        if self.is_loaded:
            self.lifecycle_phase = "GESTATING"
        else:
            self.lifecycle_phase = "WAITING"
        
        self._last_grid_size = n
        print(f"[LivingCrystal] Arrays reinitialized to {n}x{n}")
    
    def _maybe_reload_eeg(self):
        """Check if EDF path changed and reload if needed."""
        path = str(self.edf_path or "").strip().strip('"').strip("'")
        path = path.replace("\\", "/")
        
        if path != self._last_path:
            self._last_path = path
            self.edf_path = path
            if path:
                self._load_edf()
            else:
                self.is_loaded = False
                self.status_msg = "No EEG file"
    
    def _load_edf(self):
        """Load EDF file for gestation."""
        if not MNE_AVAILABLE:
            self.status_msg = "MNE not installed"
            self.is_loaded = False
            return False
        
        if not self.edf_path or not os.path.exists(self.edf_path):
            self.status_msg = "File not found"
            self.is_loaded = False
            return False
        
        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            
            try:
                raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, emg=False, misc=False, stim=False)
            except:
                pass
            
            if raw.info["sfreq"] > 256:
                raw.resample(256, npad="auto", verbose=False)
            
            self.raw = raw
            self.sfreq = float(raw.info["sfreq"])
            self.data_cache = raw.get_data()
            
            # Map electrodes to grid
            self._map_electrodes()
            
            fname = os.path.basename(self.edf_path)
            self.status_msg = f"Loaded {fname}"
            self.is_loaded = True
            self.lifecycle_phase = "GESTATING"
            self.eeg_idx = 0
            
            print(f"[LivingCrystal] Loaded EEG: {fname}, {self.data_cache.shape[0]} channels, beginning gestation...")
            return True
            
        except Exception as e:
            self.status_msg = f"Load error: {str(e)[:30]}"
            self.is_loaded = False
            return False
    
    def _map_electrodes(self):
        """Map EEG electrodes to grid positions."""
        if self.raw is None:
            return
        
        self.electrode_coords = []
        self.electrode_indices = []
        
        ch_names = [ch.upper() for ch in self.raw.ch_names]
        
        for idx, name in enumerate(ch_names):
            clean = re.sub(r'[^A-Z0-9]', '', name)
            
            pos = None
            for std_name, std_pos in self.standard_map.items():
                if std_name in clean or clean in std_name:
                    pos = std_pos
                    break
            
            if pos is None:
                for std_name, std_pos in self.standard_map.items():
                    if clean[:2] == std_name[:2]:
                        pos = std_pos
                        break
            
            if pos:
                grid_r = int(pos[1] * (self.grid_size - 1))
                grid_c = int(pos[0] * (self.grid_size - 1))
                self.electrode_coords.append((grid_r, grid_c))
                self.electrode_indices.append(idx)
    
    def _get_eeg_input_current(self):
        """Get input current from EEG for this timestep."""
        if not self.is_loaded or self.data_cache is None:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        n_samples = self.data_cache.shape[1]
        if n_samples == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Get current sample (loop if needed)
        sample_idx = self.eeg_idx % n_samples
        self.eeg_idx += 1
        
        I = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        for i, ch_idx in enumerate(self.electrode_indices):
            if ch_idx < self.data_cache.shape[0] and i < len(self.electrode_coords):
                r, c = self.electrode_coords[i]
                val = self.data_cache[ch_idx, sample_idx]
                
                # Scale EEG to reasonable input current
                # EEG is typically in microvolts, scale to ~0-20 range
                scaled = float(val) * 1e5
                scaled = np.clip(scaled, -50, 50)
                
                # Gaussian spread around electrode
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            dist = np.sqrt(dr * dr + dc * dc)
                            weight = np.exp(-dist * dist / 2.0)
                            I[nr, nc] += scaled * weight
        
        return I
    
    def _read_input(self, name, default=0.0):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                if isinstance(val, np.ndarray):
                    return val
                return float(val)
            except:
                return default
        return default
    
    def _process_visual_input(self, visual_data):
        """Convert visual input to sensory activation pattern."""
        if visual_data is None or not isinstance(visual_data, np.ndarray):
            self.current_visual *= 0.9  # Decay
            return
        
        # Resize to sensory region size
        target_h = self.grid_size // 2
        target_w = self.grid_size
        
        # Handle different input formats
        if visual_data.ndim == 3:
            # RGB - convert to grayscale
            gray = np.mean(visual_data, axis=2)
        else:
            gray = visual_data
        
        # Resize
        resized = cv2.resize(gray.astype(np.float32), (target_w, target_h))
        
        # Normalize to activation range
        if resized.max() > 0:
            resized = resized / resized.max()
        
        # Blend with current (temporal smoothing)
        self.current_visual = 0.7 * self.current_visual + 0.3 * resized
    
    def _process_audio_input(self, audio_val):
        """Convert audio to oscillatory input pattern."""
        # Store in buffer
        self.audio_buffer[self.audio_idx % 64] = audio_val
        self.audio_idx += 1
        
        # Compute simple FFT for dominant frequency
        if self.audio_idx >= 64:
            fft = np.abs(np.fft.rfft(self.audio_buffer))
            if fft.max() > 0:
                peak_idx = np.argmax(fft[1:]) + 1
                self.dominant_freq = peak_idx * 2.0  # Rough Hz estimate
    
    def step(self):
        self.step_count += 1
        
        # Check for grid size mismatch (from loading old configs)
        if self.v.shape[0] != self.grid_size:
            print(f"[LivingCrystal] Size mismatch detected, reinitializing arrays...")
            self._reinit_arrays()
            if self.is_loaded:
                self._map_electrodes()
        
        # Check for EEG file changes
        self._maybe_reload_eeg()
        
        # === READ INPUTS ===
        visual_in = self._read_input("visual", None)
        audio_in = self._read_input("audio", 0.0)
        touch_in = self._read_input("touch", 0.0)
        reward_in = self._read_input("reward", 0.0)
        pain_in = self._read_input("pain", 0.0)
        arousal_in = self._read_input("arousal", 0.0)
        force_birth = self._read_input("force_birth", 0.0) > 0.5
        reset = self._read_input("reset", 0.0) > 0.5
        
        if reset:
            self._reset()
            return
        
        # === LIFECYCLE MANAGEMENT ===
        if not self.is_born:
            # Check for birth conditions
            if force_birth or (self.is_loaded and self.learning_steps >= self.gestation_steps):
                self._birth()
        
        # === PHASE-SPECIFIC BEHAVIOR ===
        if self.lifecycle_phase == "WAITING":
            # Just waiting for EEG to be loaded
            self._update_display()
            return
        
        elif self.lifecycle_phase == "GESTATING":
            # Growing from EEG
            self._step_gestation()
        
        elif self.lifecycle_phase == "BORN":
            # Living in the world
            self._step_living(visual_in, audio_in, touch_in, reward_in, pain_in, arousal_in)
        
        self._update_display()
    
    def _birth(self):
        """The crystal is born - freeze structure, begin living."""
        self.is_born = True
        self.birth_step = self.step_count
        self.lifecycle_phase = "BORN"
        
        # Calculate final crystal statistics
        all_weights = np.concatenate([
            self.weights_up.flatten(),
            self.weights_down.flatten(),
            self.weights_left.flatten(),
            self.weights_right.flatten()
        ])
        
        self.crystal_energy = float(np.sum(all_weights))
        w_norm = all_weights / (np.sum(all_weights) + 1e-9)
        self.crystal_entropy = float(-np.sum(w_norm * np.log(w_norm + 1e-9)))
        
        print(f"[LivingCrystal] BORN at step {self.step_count}!")
        print(f"  Gestation: {self.learning_steps} learning steps")
        print(f"  Crystal energy: {self.crystal_energy:.1f}")
        print(f"  Crystal entropy: {self.crystal_entropy:.2f}")
        print(f"  Total spikes during gestation: {self.total_spikes}")
    
    def _step_gestation(self):
        """One step of EEG-driven development."""
        # Get EEG input current
        I = self._get_eeg_input_current()
        
        # Run neural dynamics with learning
        self._run_neural_dynamics(I, learning=True)
        
        # Update learning counter
        self.learning_steps += 1
        
        # Check for birth
        if self.learning_steps >= self.gestation_steps:
            self._birth()
    
    def _step_living(self, visual_in, audio_in, touch_in, reward_in, pain_in, arousal_in):
        """One step of living - sensing and responding with frozen structure."""
        
        # === PROCESS SENSORY INPUTS ===
        if isinstance(visual_in, np.ndarray):
            self._process_visual_input(visual_in)
        if isinstance(audio_in, (int, float)):
            self._process_audio_input(float(audio_in))
        
        # === UPDATE EMOTIONAL STATE ===
        self.valence = self.valence * self.valence_decay
        self.valence += reward_in * 0.1 - pain_in * 0.15
        self.valence = np.clip(self.valence, -1.0, 1.0)
        
        self.arousal_level = self.arousal_level * self.arousal_decay
        self.arousal_level += abs(touch_in) * 0.05 + arousal_in * 0.1
        self.arousal_level = np.clip(self.arousal_level, 0.0, 1.0)
        
        self.lifetime_reward += max(0, reward_in)
        self.lifetime_pain += max(0, pain_in)
        
        # === BUILD INPUT CURRENT FROM SENSES ===
        I = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Visual input to sensory region
        I[:self.grid_size // 2, :] += self.current_visual * 20.0
        
        # Touch creates central activation
        if abs(touch_in) > 0.1:
            cx, cy = self.grid_size // 2, self.grid_size // 2
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if 0 <= cx + i < self.grid_size and 0 <= cy + j < self.grid_size:
                        I[cx + i, cy + j] += touch_in * 10.0
        
        # Audio creates oscillatory modulation
        audio_mod = np.sin(2 * np.pi * self.dominant_freq * self.step_count * 0.01)
        I += audio_mod * audio_in * 5.0
        
        # Arousal increases baseline
        I += self.arousal_level * 5.0
        
        # === RUN NEURAL DYNAMICS (NO LEARNING - FROZEN) ===
        self._run_neural_dynamics(I, learning=False)
        
        # === MOVEMENT / AGENCY ===
        motor_activity = self.v[self.motor_region, :]
        
        left_activity = np.mean(motor_activity[:, :self.grid_size // 2])
        right_activity = np.mean(motor_activity[:, self.grid_size // 2:])
        move_desire_x = (right_activity - left_activity) * 0.01
        
        top_activity = np.mean(motor_activity[:motor_activity.shape[0] // 2, :])
        bottom_activity = np.mean(motor_activity[motor_activity.shape[0] // 2:, :])
        move_desire_y = (bottom_activity - top_activity) * 0.01
        
        valence_sign = np.sign(self.valence) if abs(self.valence) > 0.1 else 0
        
        self.velocity_x += move_desire_x * self.movement_gain * (1 + valence_sign)
        self.velocity_y += move_desire_y * self.movement_gain * (1 + valence_sign)
        
        self.velocity_x *= self.friction
        self.velocity_y *= self.friction
        
        self.position_x += self.velocity_x
        self.position_y += self.velocity_y
        
        self.approach_tendency = self.valence * (abs(self.velocity_x) + abs(self.velocity_y))
        
        # === VOCALIZATION ===
        recent_activity = np.mean(np.abs(self.v - self.c))
        self.freq_history[:-1] = self.freq_history[1:]
        self.freq_history[-1] = recent_activity
        
        if self.step_count % 10 == 0:
            fft = np.abs(np.fft.rfft(self.freq_history))
            if fft.max() > 0:
                peak_idx = np.argmax(fft[1:]) + 1
                self.dominant_freq = peak_idx * 3.0
        
        # === SET OUTPUTS ===
        self._output_values = {
            "valence": self.valence,
            "arousal_out": self.arousal_level,
            "approach": self.approach_tendency,
            "vocalization": self.dominant_freq,
            "move_x": self.velocity_x * 10.0,
            "move_y": self.velocity_y * 10.0,
        }
    
    def _run_neural_dynamics(self, I, learning=False):
        """Run Izhikevich dynamics with optional STDP learning."""
        v = self.v.copy()
        u = self.u.copy()
        
        # Get neighbor values
        v_up = np.roll(v, -1, axis=0)
        v_down = np.roll(v, 1, axis=0)
        v_left = np.roll(v, -1, axis=1)
        v_right = np.roll(v, 1, axis=1)
        
        # Weighted coupling (THE CRYSTAL)
        neighbor_influence = (
            self.weights_up * v_up +
            self.weights_down * v_down +
            self.weights_left * v_left +
            self.weights_right * v_right
        )
        total_weight = (self.weights_up + self.weights_down + 
                       self.weights_left + self.weights_right)
        neighbor_avg = neighbor_influence / (total_weight + 1e-6)
        
        # Coupling current
        coupling_strength = 1.0 + self.arousal_level * 2.0
        I_coupling = coupling_strength * (neighbor_avg - v)
        
        # Izhikevich dynamics
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I + I_coupling) * self.dt
        du = self.a * (self.b * v - u) * self.dt
        
        v = v + dv
        u = u + du
        
        # Detect spikes
        spikes = v >= 30.0
        v[spikes] = self.c
        u[spikes] += self.d
        
        self.v = v
        self.u = u
        self.total_spikes += np.sum(spikes)
        
        # === STDP LEARNING (only during gestation) ===
        if learning:
            effective_lr = self.base_learning_rate
            
            # Update spike trace
            self.spike_trace *= self.trace_decay
            self.spike_trace[spikes] = 1.0
            
            # Get neighbor traces
            trace_up = np.roll(self.spike_trace, -1, axis=0)
            trace_down = np.roll(self.spike_trace, 1, axis=0)
            trace_left = np.roll(self.spike_trace, -1, axis=1)
            trace_right = np.roll(self.spike_trace, 1, axis=1)
            
            spike_float = spikes.astype(np.float32)
            
            # Potentiation
            dw_up = effective_lr * spike_float * trace_up
            dw_down = effective_lr * spike_float * trace_down
            dw_left = effective_lr * spike_float * trace_left
            dw_right = effective_lr * spike_float * trace_right
            
            # Depression
            spike_up = np.roll(spike_float, -1, axis=0)
            spike_down = np.roll(spike_float, 1, axis=0)
            spike_left = np.roll(spike_float, -1, axis=1)
            spike_right = np.roll(spike_float, 1, axis=1)
            
            dw_up -= 0.5 * effective_lr * self.spike_trace * spike_up
            dw_down -= 0.5 * effective_lr * self.spike_trace * spike_down
            dw_left -= 0.5 * effective_lr * self.spike_trace * spike_left
            dw_right -= 0.5 * effective_lr * self.spike_trace * spike_right
            
            # Apply
            self.weights_up += dw_up
            self.weights_down += dw_down
            self.weights_left += dw_left
            self.weights_right += dw_right
            
            # Clamp
            self.weights_up = np.clip(self.weights_up, self.weight_min, self.weight_max)
            self.weights_down = np.clip(self.weights_down, self.weight_min, self.weight_max)
            self.weights_left = np.clip(self.weights_left, self.weight_min, self.weight_max)
            self.weights_right = np.clip(self.weights_right, self.weight_min, self.weight_max)
    
    def _reset(self):
        """Reset the crystal to initial state - begin gestation anew."""
        n = self.grid_size
        self.v = np.ones((n, n), dtype=np.float32) * self.c
        self.u = self.v * self.b
        self.weights_up = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_down = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_left = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_right = np.ones((n, n), dtype=np.float32) * 0.5
        self.spike_trace = np.zeros((n, n), dtype=np.float32)
        
        # Reset sensory regions
        self.visual_region = slice(0, n // 2)
        self.motor_region = slice(n // 2, n)
        self.current_visual = np.zeros((n // 2, n), dtype=np.float32)
        
        self.valence = 0.0
        self.arousal_level = 0.5
        self.position_x = 0.0
        self.position_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.total_spikes = 0
        self.learning_steps = 0
        self.step_count = 0
        self.eeg_idx = 0
        self.lifetime_reward = 0.0
        self.lifetime_pain = 0.0
        
        # Reset lifecycle
        self.is_born = False
        self.birth_step = 0
        if self.is_loaded:
            self.lifecycle_phase = "GESTATING"
            self._map_electrodes()  # Remap electrodes to current grid size
        else:
            self.lifecycle_phase = "WAITING"
        
        print(f"[LivingCrystal] Reset - beginning new gestation at {n}x{n}")
    
    def get_output(self, port_name):
        if port_name == "crystal_view":
            return self._render_crystal()
        elif port_name == "activity_view":
            return self._render_activity()
        elif port_name == "sensorium_view":
            return self._render_sensorium()
        elif port_name in self._output_values:
            return self._output_values.get(port_name, 0.0)
        return None
    
    def _render_crystal(self):
        """Render the learned weight structure."""
        n = self.grid_size
        
        horizontal = (self.weights_left + self.weights_right) / 2
        vertical = (self.weights_up + self.weights_down) / 2
        
        h_norm = (horizontal - self.weight_min) / (self.weight_max - self.weight_min)
        v_norm = (vertical - self.weight_min) / (self.weight_max - self.weight_min)
        anisotropy = np.abs(h_norm - v_norm)
        
        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[:, :, 0] = (h_norm * 255).astype(np.uint8)
        img[:, :, 1] = ((1 - anisotropy) * 255).astype(np.uint8)
        img[:, :, 2] = (v_norm * 255).astype(np.uint8)
        
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        return img
    
    def _render_activity(self):
        """Render current neural activity."""
        disp = np.clip(self.v, -90.0, 40.0)
        norm = ((disp + 90.0) / 130.0 * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        heat = cv2.resize(heat, (256, 256), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    
    def _render_sensorium(self):
        """Render what the crystal is currently perceiving."""
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Visual input (top)
        if self.current_visual.max() > 0:
            vis = (self.current_visual / self.current_visual.max() * 255).astype(np.uint8)
        else:
            vis = (self.current_visual * 255).astype(np.uint8)
        vis_resized = cv2.resize(vis, (w, h // 2))
        img[:h // 2, :, 1] = vis_resized  # Green channel
        
        # Audio spectrum (bottom)
        fft = np.abs(np.fft.rfft(self.audio_buffer))
        if fft.max() > 0:
            fft_norm = fft / fft.max()
        else:
            fft_norm = fft
        bar_w = w // len(fft_norm)
        for i, val in enumerate(fft_norm):
            bar_h = int(val * h // 2)
            x = i * bar_w
            cv2.rectangle(img, (x, h - bar_h), (x + bar_w - 1, h), (255, 100, 50), -1)
        
        return img
    
    def _update_display(self):
        """Create the main display."""
        w, h = 512, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title with lifecycle phase
        if self.lifecycle_phase == "WAITING":
            phase_color = (150, 150, 150)
            phase_text = "WAITING FOR EEG"
        elif self.lifecycle_phase == "GESTATING":
            phase_color = (0, 200, 255)  # Orange-ish
            progress = self.learning_steps / self.gestation_steps
            phase_text = f"GESTATING {int(progress * 100)}%"
        else:  # BORN
            phase_color = (100, 255, 100)
            phase_text = "ALIVE"
        
        cv2.putText(img, "LIVING CRYSTAL", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 100, 200), 2)
        cv2.putText(img, phase_text, (280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
        
        # Gestation progress bar (if gestating)
        if self.lifecycle_phase == "GESTATING":
            bar_x, bar_y = 10, 45
            bar_w, bar_h = 200, 15
            progress = min(1.0, self.learning_steps / self.gestation_steps)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 255), -1)
            cv2.putText(img, f"{self.learning_steps}/{self.gestation_steps}", (bar_x + bar_w + 10, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Emotional state bar (only when born)
        if self.is_born:
            valence_color = (100, 255, 100) if self.valence > 0 else (100, 100, 255)
            if abs(self.valence) < 0.1:
                valence_color = (200, 200, 200)
            
            bar_x, bar_y = 10, 50
            bar_w, bar_h = 200, 20
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            mid = bar_x + bar_w // 2
            valence_x = int(mid + self.valence * bar_w // 2)
            cv2.line(img, (mid, bar_y), (mid, bar_y + bar_h), (100, 100, 100), 1)
            cv2.circle(img, (valence_x, bar_y + bar_h // 2), 8, valence_color, -1)
            cv2.putText(img, f"Valence: {self.valence:.2f}", (bar_x + bar_w + 10, bar_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, valence_color, 1)
        
        # Arousal meter
        arousal_x = 10
        arousal_y = 80
        arousal_h = int(self.arousal_level * 100)
        cv2.rectangle(img, (arousal_x, arousal_y), (arousal_x + 20, arousal_y + 100), (50, 50, 50), -1)
        cv2.rectangle(img, (arousal_x, arousal_y + 100 - arousal_h), 
                     (arousal_x + 20, arousal_y + 100), (50, 200, 255), -1)
        cv2.putText(img, "A", (arousal_x + 5, arousal_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 200, 255), 1)
        
        # Activity view (center)
        activity = self._render_activity()
        activity_small = cv2.resize(activity, (150, 150))
        img[100:250, 50:200] = activity_small
        cv2.putText(img, "Activity", (50, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Crystal view (right of activity)
        crystal = self._render_crystal()
        crystal_small = cv2.resize(crystal, (150, 150))
        img[100:250, 210:360] = crystal_small
        cv2.putText(img, "Crystal", (210, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Movement indicator (bottom left) - only when born
        if self.is_born:
            move_cx, move_cy = 100, 330
            cv2.circle(img, (move_cx, move_cy), 40, (50, 50, 50), -1)
            cv2.circle(img, (move_cx, move_cy), 40, (100, 100, 100), 1)
            vx = int(self.velocity_x * 200)
            vy = int(self.velocity_y * 200)
            vx = np.clip(vx, -35, 35)
            vy = np.clip(vy, -35, 35)
            cv2.arrowedLine(img, (move_cx, move_cy), (move_cx + vx, move_cy + vy), (0, 255, 255), 2)
            cv2.putText(img, "Movement", (60, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            approach_text = "APPROACHING" if self.approach_tendency > 0.1 else "AVOIDING" if self.approach_tendency < -0.1 else "NEUTRAL"
            approach_color = (100, 255, 100) if self.approach_tendency > 0.1 else (100, 100, 255) if self.approach_tendency < -0.1 else (150, 150, 150)
            cv2.putText(img, approach_text, (180, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, approach_color, 1)
        
        # Stats (right side)
        stats_x = 380
        cv2.putText(img, f"Steps: {self.step_count}", (stats_x, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, f"Learning: {self.learning_steps}", (stats_x, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, f"Spikes: {self.total_spikes}", (stats_x, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        if self.is_born:
            cv2.putText(img, f"Reward: {self.lifetime_reward:.1f}", (stats_x, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            cv2.putText(img, f"Pain: {self.lifetime_pain:.1f}", (stats_x, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            cv2.putText(img, f"Voice: {self.dominant_freq:.1f} Hz", (stats_x, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        else:
            # Show EEG status during gestation
            cv2.putText(img, self.status_msg, (stats_x, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(img, f"Energy: {self.crystal_energy:.0f}", (stats_x, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # Sensorium (bottom right) - only when born
        if self.is_born:
            sensorium = self._render_sensorium()
            sensorium_small = cv2.resize(sensorium, (120, 80))
            img[300:380, 370:490] = sensorium_small
            cv2.putText(img, "Senses", (370, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Show electrode positions during gestation
            if len(self.electrode_coords) > 0:
                cv2.putText(img, f"Electrodes: {len(self.electrode_coords)}", (370, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image
    
    # === STATE PERSISTENCE ===
    def save_custom_state(self, folder_path, node_id):
        """Save the crystal's learned weights and lifecycle state."""
        filename = f"living_crystal_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        
        np.savez(filepath,
                 weights_up=self.weights_up,
                 weights_down=self.weights_down,
                 weights_left=self.weights_left,
                 weights_right=self.weights_right,
                 valence=self.valence,
                 arousal=self.arousal_level,
                 position_x=self.position_x,
                 position_y=self.position_y,
                 lifetime_reward=self.lifetime_reward,
                 lifetime_pain=self.lifetime_pain,
                 learning_steps=self.learning_steps,
                 is_born=self.is_born,
                 birth_step=self.birth_step,
                 crystal_energy=self.crystal_energy,
                 crystal_entropy=self.crystal_entropy,
                 total_spikes=self.total_spikes)
        
        return filename
    
    def load_custom_state(self, filepath):
        """Load previously saved crystal state."""
        try:
            data = np.load(filepath)
            self.weights_up = data['weights_up']
            self.weights_down = data['weights_down']
            self.weights_left = data['weights_left']
            self.weights_right = data['weights_right']
            self.valence = float(data['valence'])
            self.arousal_level = float(data['arousal'])
            self.position_x = float(data['position_x'])
            self.position_y = float(data['position_y'])
            self.lifetime_reward = float(data['lifetime_reward'])
            self.lifetime_pain = float(data['lifetime_pain'])
            self.learning_steps = int(data['learning_steps'])
            
            # Lifecycle state
            if 'is_born' in data:
                self.is_born = bool(data['is_born'])
                self.birth_step = int(data['birth_step'])
                self.crystal_energy = float(data['crystal_energy'])
                self.crystal_entropy = float(data['crystal_entropy'])
                self.total_spikes = int(data['total_spikes'])
                
                if self.is_born:
                    self.lifecycle_phase = "BORN"
                else:
                    self.lifecycle_phase = "GESTATING" if self.is_loaded else "WAITING"
            
            print(f"[LivingCrystal] Loaded state: {self.learning_steps} steps, born={self.is_born}")
        except Exception as e:
            print(f"[LivingCrystal] Failed to load state: {e}")