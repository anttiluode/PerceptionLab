"""
Crystal Cortex Node
====================

A cortical sheet that CRYSTALLIZES patterns through STDP plasticity.

The key insight: The EEG is a crystallization template.
- Same EEG pattern → same crystal structure grows
- The coupling weights ARE the crystal lattice
- Activity sculpts connectivity sculpts activity

This is how brains develop: activity patterns during development
literally sculpt the connectivity. The signal becomes the structure.

Like UV-burning transformer weights onto silicon:
The computation becomes the geometry becomes the computation.

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


class CrystalCortexNode(BaseNode):
    """
    Cortical sheet with STDP plasticity - crystallizes EEG patterns into structure.
    """
    
    NODE_NAME = "Crystal Cortex"
    NODE_TITLE = "Crystal Cortex"
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(180, 100, 180) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "coupling": "signal",
            "excitability": "signal",
            "learning_rate": "signal",  # STDP strength
            "freeze": "signal",         # Stop learning, just run
            "reset": "signal",
        }
        
        self.outputs = {
            "cortex_view": "image",
            "crystal_view": "image",
            "lfp_signal": "signal",
            "crystal_energy": "signal",
            "crystal_entropy": "signal",
        }
        
        # Actual output values (separate from port type declarations)
        self._output_values = {
            "lfp_signal": 0.0,
            "crystal_energy": 0.0,
            "crystal_entropy": 0.0,
        }
        
        # === EDF Config ===
        self.edf_path = ""
        self._last_path = ""
        self.status_msg = "No file"
        self.is_loaded = False
        self.raw = None
        self.data_cache = None
        self.sfreq = 256.0
        self.current_idx = 0
        
        # === Sheet Parameters ===
        self.grid_size = 64  # Smaller for tractable plasticity
        self.dt = 0.5
        
        # Izhikevich parameters
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        
        # State variables
        self.v = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * self.c
        self.u = self.v * self.b
        
        # === THE CRYSTAL: Learned coupling weights ===
        # 4 directions: up, down, left, right
        # These START uniform and CRYSTALLIZE through STDP
        self.weights_up = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
        self.weights_down = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
        self.weights_left = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
        self.weights_right = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
        
        # Spike timing traces for STDP
        self.spike_trace = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.trace_decay = 0.95  # How fast the trace decays
        
        # STDP parameters
        self.base_learning_rate = 0.001
        self.stdp_window = 20.0  # ms window for STDP
        self.weight_max = 2.0
        self.weight_min = 0.01
        
        # Last spike times (for precise STDP)
        self.last_spike_time = np.full((self.grid_size, self.grid_size), -1000.0, dtype=np.float32)
        self.current_time = 0.0
        
        # === Electrode mapping ===
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
        
        # === Statistics ===
        self.crystal_energy = 0.0
        self.crystal_entropy = 0.0
        self.total_spikes = 0
        self.learning_steps = 0
        
        self.display_image = None
        self._update_display()
    
    def get_config_options(self):
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Base Learning Rate", "base_learning_rate", self.base_learning_rate, None),
            ("Trace Decay", "trace_decay", self.trace_decay, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _maybe_reload(self):
        path = str(self.edf_path or "").strip().strip('"').strip("'")
        path = path.replace("\\", "/")
        
        if path != self._last_path:
            self._last_path = path
            self.edf_path = path
            if path:
                self.load_edf()
            else:
                self.is_loaded = False
                self.status_msg = "No file"
    
    def load_edf(self):
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
            self.current_idx = 0
            
            self._map_electrodes()
            self._init_crystal()  # Reset crystal on new file
            
            self.is_loaded = True
            self.status_msg = f"Loaded {os.path.basename(self.edf_path)} | mapped={len(self.electrode_coords)}"
            return True
            
        except Exception as e:
            self.is_loaded = False
            self.status_msg = f"Error: {str(e)[:40]}"
            return False
    
    def _clean_ch_name(self, name):
        n = name.upper()
        n = n.replace("EEG", "").replace(" ", "").replace("-REF", "").replace("REF", "")
        n = re.sub(r"[-_](A1|A2|M1|M2|LE|RE)$", "", n)
        n = re.sub(r"[^A-Z0-9]", "", n)
        return n
    
    def _map_electrodes(self):
        self.electrode_coords = []
        self.electrode_indices = []
        
        if self.raw is None:
            return
        
        names = [self._clean_ch_name(ch) for ch in self.raw.ch_names]
        margin = 4
        scale = self.grid_size - 2 * margin
        
        for i, cn in enumerate(names):
            pos = None
            if cn in self.standard_map:
                pos = self.standard_map[cn]
            else:
                for key, p in self.standard_map.items():
                    if key in cn:
                        pos = p
                        break
            
            if pos is None:
                continue
            
            c = int(pos[0] * scale + margin)
            r = int(pos[1] * scale + margin)
            r = int(np.clip(r, 0, self.grid_size - 1))
            c = int(np.clip(c, 0, self.grid_size - 1))
            
            self.electrode_coords.append((r, c))
            self.electrode_indices.append(i)
    
    def _init_crystal(self):
        """Initialize/reset the crystal structure."""
        n = self.grid_size
        
        # Reset neurons
        self.v = np.ones((n, n), dtype=np.float32) * self.c
        self.u = self.v * self.b
        
        # Reset crystal to uniform (amorphous state)
        self.weights_up = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_down = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_left = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_right = np.ones((n, n), dtype=np.float32) * 0.5
        
        # Reset traces
        self.spike_trace = np.zeros((n, n), dtype=np.float32)
        self.last_spike_time = np.full((n, n), -1000.0, dtype=np.float32)
        self.current_time = 0.0
        
        self.learning_steps = 0
        self.total_spikes = 0
    
    def _read_input_scalar(self, name, default=0.0):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                return float(val)
            except:
                return default
        return default
    
    def step(self):
        self._maybe_reload()
        
        if not self.is_loaded or self.data_cache is None:
            self._update_display()
            return
        
        # Read inputs
        coupling = self._read_input_scalar("coupling", 0.3)
        gain = self._read_input_scalar("excitability", 1.0)
        lr_mod = self._read_input_scalar("learning_rate", 1.0)
        freeze = self._read_input_scalar("freeze", 0.0)
        reset = self._read_input_scalar("reset", 0.0)
        
        coupling = float(np.clip(coupling, 0.0, 2.0))
        gain = float(np.clip(gain, 0.0, 100.0))
        
        if reset > 0.5:
            self._init_crystal()
            return
        
        learning = (freeze < 0.5)
        effective_lr = self.base_learning_rate * lr_mod if learning else 0.0
        
        n_samples = self.data_cache.shape[1]
        n = self.grid_size
        
        # Advance time
        self.current_time += self.dt
        
        # Get EEG sample
        if self.current_idx >= n_samples:
            self.current_idx = 0
        
        # Build input current
        I = np.zeros((n, n), dtype=np.float32)
        
        if self.electrode_indices:
            vec = self.data_cache[self.electrode_indices, self.current_idx].astype(np.float32)
            
            for (r, c), vch in zip(self.electrode_coords, vec):
                I[r, c] += vch
            
            I = cv2.GaussianBlur(I, (7, 7), 2) * (gain * 0.02)
        
        self.current_idx += 1
        
        # === WEIGHTED COUPLING (the crystal structure) ===
        v = self.v
        u = self.u
        
        # Neighbor values
        v_up = np.roll(v, -1, axis=0)
        v_down = np.roll(v, 1, axis=0)
        v_left = np.roll(v, -1, axis=1)
        v_right = np.roll(v, 1, axis=1)
        
        # Weighted sum (THIS IS THE CRYSTAL - learned weights shape the flow)
        neighbor_influence = (
            self.weights_up * v_up +
            self.weights_down * v_down +
            self.weights_left * v_left +
            self.weights_right * v_right
        )
        
        # Normalize by total weight
        total_weight = (self.weights_up + self.weights_down + 
                       self.weights_left + self.weights_right)
        neighbor_avg = neighbor_influence / (total_weight + 1e-6)
        
        # Coupling current
        I_coupling = coupling * (neighbor_avg - v)
        
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
        
        # Update spike timing
        self.last_spike_time[spikes] = self.current_time
        self.total_spikes += np.sum(spikes)
        
        # === STDP: THE CRYSTALLIZATION ===
        if learning and effective_lr > 0:
            self.learning_steps += 1
            
            # Update spike trace (exponential decay + spike injection)
            self.spike_trace *= self.trace_decay
            self.spike_trace[spikes] = 1.0
            
            # Get neighbor traces
            trace_up = np.roll(self.spike_trace, -1, axis=0)
            trace_down = np.roll(self.spike_trace, 1, axis=0)
            trace_left = np.roll(self.spike_trace, -1, axis=1)
            trace_right = np.roll(self.spike_trace, 1, axis=1)
            
            # STDP rule: if I spike and neighbor had recent activity, strengthen
            # This is simplified Hebbian: "fire together, wire together"
            spike_float = spikes.astype(np.float32)
            
            # Potentiation: I spike after neighbor was active
            dw_up = effective_lr * spike_float * trace_up
            dw_down = effective_lr * spike_float * trace_down
            dw_left = effective_lr * spike_float * trace_left
            dw_right = effective_lr * spike_float * trace_right
            
            # Depression: neighbor spikes after I was active (weaker)
            spike_up = np.roll(spike_float, -1, axis=0)
            spike_down = np.roll(spike_float, 1, axis=0)
            spike_left = np.roll(spike_float, -1, axis=1)
            spike_right = np.roll(spike_float, 1, axis=1)
            
            dw_up -= 0.5 * effective_lr * self.spike_trace * spike_up
            dw_down -= 0.5 * effective_lr * self.spike_trace * spike_down
            dw_left -= 0.5 * effective_lr * self.spike_trace * spike_left
            dw_right -= 0.5 * effective_lr * self.spike_trace * spike_right
            
            # Apply weight changes
            self.weights_up += dw_up
            self.weights_down += dw_down
            self.weights_left += dw_left
            self.weights_right += dw_right
            
            # Clamp weights
            self.weights_up = np.clip(self.weights_up, self.weight_min, self.weight_max)
            self.weights_down = np.clip(self.weights_down, self.weight_min, self.weight_max)
            self.weights_left = np.clip(self.weights_left, self.weight_min, self.weight_max)
            self.weights_right = np.clip(self.weights_right, self.weight_min, self.weight_max)
        
        # === STATISTICS ===
        all_weights = np.concatenate([
            self.weights_up.flatten(),
            self.weights_down.flatten(),
            self.weights_left.flatten(),
            self.weights_right.flatten()
        ])
        
        self.crystal_energy = float(np.sum(all_weights))
        
        # Entropy: how structured is the crystal?
        # Low entropy = crystallized (weights concentrated)
        # High entropy = amorphous (weights uniform)
        w_norm = all_weights / (np.sum(all_weights) + 1e-9)
        self.crystal_entropy = float(-np.sum(w_norm * np.log(w_norm + 1e-9)))
        
        # Outputs
        self._output_values["lfp_signal"] = float(np.mean(self.v))
        self._output_values["crystal_energy"] = self.crystal_energy
        self._output_values["crystal_entropy"] = self.crystal_entropy
        
        self._update_display()
    
    def get_output(self, port_name):
        if port_name == "cortex_view":
            return self.display_image
        elif port_name == "crystal_view":
            return self._render_crystal()
        elif port_name in ["lfp_signal", "crystal_energy", "crystal_entropy"]:
            return self._output_values.get(port_name, 0.0)
        return None
    
    def _render_crystal(self):
        """Render the learned weight structure as an image."""
        n = self.grid_size
        
        # Combine weights into a single visualization
        # Red = horizontal dominance, Blue = vertical dominance, Green = uniform
        horizontal = (self.weights_left + self.weights_right) / 2
        vertical = (self.weights_up + self.weights_down) / 2
        
        # Normalize
        h_norm = (horizontal - self.weight_min) / (self.weight_max - self.weight_min)
        v_norm = (vertical - self.weight_min) / (self.weight_max - self.weight_min)
        
        # Anisotropy: how directional is the crystal at each point?
        anisotropy = np.abs(h_norm - v_norm)
        
        # Create RGB
        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[:, :, 0] = (h_norm * 255).astype(np.uint8)  # Red = horizontal
        img[:, :, 1] = ((1 - anisotropy) * 255).astype(np.uint8)  # Green = isotropic
        img[:, :, 2] = (v_norm * 255).astype(np.uint8)  # Blue = vertical
        
        # Scale up
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return img
    
    def _update_display(self):
        """Create the main display showing activity + crystal overlay."""
        w, h = 512, 512
        n = self.grid_size
        
        if not self.is_loaded:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(img, "CRYSTAL CORTEX", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 100, 180), 2)
            cv2.putText(img, self.status_msg, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(img, "Load EDF to begin crystallization", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            # Activity heatmap
            disp = np.clip(self.v, -90.0, 40.0)
            norm = ((disp + 90.0) / 130.0 * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            
            # Resize to display
            heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Get crystal structure for overlay
            crystal = self._render_crystal()
            crystal = cv2.resize(crystal, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Blend: activity dominant but crystal visible
            img = cv2.addWeighted(heat, 0.7, crystal, 0.3, 0)
            
            # Draw electrodes
            scale = w / n
            for r, c in self.electrode_coords:
                center = (int(c * scale), int(r * scale))
                cv2.circle(img, center, 4, (0, 255, 0), -1)
            
            # HUD
            hud = (255, 255, 255)
            cv2.putText(img, f"Sample: {self.current_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 1)
            cv2.putText(img, f"Learning Steps: {self.learning_steps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 1)
            cv2.putText(img, f"Total Spikes: {self.total_spikes}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 1)
            
            # Crystal stats
            cv2.putText(img, f"Crystal Energy: {self.crystal_energy:.1f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
            cv2.putText(img, f"Crystal Entropy: {self.crystal_entropy:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            
            # Learning indicator
            lr = self.base_learning_rate
            if lr > 0:
                cv2.putText(img, "CRYSTALLIZING", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(img, "FROZEN", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image