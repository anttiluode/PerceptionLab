"""
Crystal Chip Node
==================

Loads a frozen crystal (grown by EEG Crystal Maker) and probes it like a chip.

The crystal's electrode positions become I/O pins:
- FRONTAL pins (FP1, FP2, F3, F4, F7, F8, FZ) → Input region
- POSTERIOR pins (O1, O2, OZ, P3, P4, P7, P8, PZ) → Output region  
- CENTRAL pins (C3, C4, CZ, T7, T8) → Internal processing

Input modes:
- image_in → Projects onto input pins spatially
- latent_in → 16-dim vector distributed across input pins
- signal_in → Direct signal injection at all input pins
- Individual pin signals → Fine control

Output modes:
- image_out → Activity pattern at output pins
- latent_out → 16-dim compressed output state
- signal_out → Mean activity at output pins
- Individual pin signals → Direct readings

The crystal processes inputs through its learned geometry.
What comes out depends on what it learned during gestation.

Author: Built for Antti's consciousness crystallography research
"""

import os
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


class CrystalChipNode(BaseNode):
    """
    A frozen crystal used as a computational chip.
    Input at frontal pins → process through crystal geometry → output at posterior pins.
    """
    
    NODE_NAME = "Crystal Chip"
    NODE_TITLE = "Crystal Chip"
    NODE_CATEGORY = "Processing"
    NODE_COLOR = QtGui.QColor(100, 200, 180) if QtGui else None
    
    # Pin categorization by neuroanatomical region
    INPUT_PINS = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ']  # Frontal = input
    OUTPUT_PINS = ['O1', 'O2', 'OZ', 'P3', 'P4', 'P7', 'P8', 'PZ']  # Posterior = output
    INTERNAL_PINS = ['C3', 'C4', 'CZ', 'T7', 'T8']  # Central = internal processing
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            # Multi-modal inputs
            "image_in": "image",       # Visual input → projected to input pins
            "latent_in": "spectrum",   # 16-dim latent → distributed to input pins
            "signal_in": "signal",     # Raw signal → all input pins equally
            
            # Modulation
            "gain": "signal",          # Input amplification
            "coupling": "signal",      # Neighbor coupling strength
            
            # Control
            "reset": "signal",         # Reset neural state
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Multi-modal outputs
            "image_out": "image",      # Activity at output pins as image
            "latent_out": "spectrum",  # 16-dim compressed output state
            "signal_out": "signal",    # Mean activity at output pins
            
            # Visualization
            "chip_view": "image",      # Main display
            "activity_view": "image",  # Full activity pattern
            
            # Analysis
            "resonance": "signal",     # How much the crystal is resonating
            "energy": "signal",        # Total activity energy
            
            # EEG-like frequency band outputs
            "delta": "signal",         # 0.5-4 Hz - slow oscillations
            "theta": "signal",         # 4-8 Hz - memory, navigation
            "alpha": "signal",         # 8-13 Hz - relaxed awareness
            "beta": "signal",          # 13-30 Hz - active thinking
            "gamma": "signal",         # 30-100 Hz - binding, cognition
            "lfp": "signal",           # Local field potential (raw mean)
        }
        
        # === CRYSTAL STATE ===
        self.crystal_path = ""
        self._last_path = ""
        self.is_loaded = False
        self.status_msg = "No crystal loaded"
        
        # Crystal data
        self.grid_size = 64
        self.weights_up = None
        self.weights_down = None
        self.weights_left = None
        self.weights_right = None
        
        # Pin mapping
        self.pin_coords = []  # [(row, col), ...] from crystal file
        self.pin_names = []   # ['FP1', 'F3', ...] from crystal file
        self.input_pin_indices = []   # Indices into pin_coords for input pins
        self.output_pin_indices = []  # Indices into pin_coords for output pins
        self.internal_pin_indices = []
        
        # Crystal metadata
        self.learning_steps = 0
        self.total_spikes = 0
        self.edf_source = ""
        self.created = ""
        
        # === NEURAL STATE ===
        # Izhikevich parameters
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 0.5
        
        # State arrays (initialized when crystal loads)
        self.v = None
        self.u = None
        
        # Processing parameters
        self.base_coupling = 5.0
        self.input_gain = 50.0
        self.spread_radius = 3  # How far input spreads from pins
        
        # Statistics
        self.step_count = 0
        self.current_resonance = 0.0
        self.current_energy = 0.0
        
        # === EEG-LIKE OUTPUT ===
        # History buffer for frequency analysis
        self.lfp_history_size = 256  # ~2.5 seconds at 100Hz
        self.lfp_history = np.zeros(self.lfp_history_size, dtype=np.float32)
        self.lfp_idx = 0
        
        # Frequency band powers
        self.band_powers = {
            "delta": 0.0,   # 0.5-4 Hz
            "theta": 0.0,   # 4-8 Hz
            "alpha": 0.0,   # 8-13 Hz
            "beta": 0.0,    # 13-30 Hz
            "gamma": 0.0,   # 30-100 Hz
        }
        self.current_lfp = 0.0
        
        # Assume ~100 Hz sample rate for the simulation
        self.sample_rate = 100.0
        
        # Output cache
        self._output_values = {
            "signal_out": 0.0,
            "resonance": 0.0,
            "energy": 0.0,
            "delta": 0.0,
            "theta": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "lfp": 0.0,
        }
        self._latent_out = np.zeros(16, dtype=np.float32)
        
        # Display
        self.display_image = None
        self._update_display()
    
    def get_config_options(self):
        return [
            ("Crystal File (.npz)", "crystal_path", self.crystal_path, None),
            ("Base Coupling", "base_coupling", self.base_coupling, None),
            ("Input Gain", "input_gain", self.input_gain, None),
            ("Spread Radius", "spread_radius", self.spread_radius, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _maybe_reload(self):
        """Check if we need to load a new crystal file."""
        path = str(self.crystal_path or "").strip().strip('"').strip("'")
        path = path.replace("\\", "/")
        
        if path != self._last_path:
            self._last_path = path
            self.crystal_path = path
            if path:
                self._load_crystal()
            else:
                self.is_loaded = False
                self.status_msg = "No crystal loaded"
    
    def _load_crystal(self):
        """Load a frozen crystal from .npz file."""
        if not os.path.exists(self.crystal_path):
            self.status_msg = "File not found"
            self.is_loaded = False
            return
        
        try:
            data = np.load(self.crystal_path, allow_pickle=True)
            
            # Load weights
            self.weights_up = data['weights_up'].astype(np.float32)
            self.weights_down = data['weights_down'].astype(np.float32)
            self.weights_left = data['weights_left'].astype(np.float32)
            self.weights_right = data['weights_right'].astype(np.float32)
            
            self.grid_size = self.weights_up.shape[0]
            
            # Load pin coordinates
            if 'pin_coords' in data:
                self.pin_coords = [tuple(p) for p in data['pin_coords']]
            else:
                self.pin_coords = []
            
            if 'pin_names' in data:
                self.pin_names = list(data['pin_names'])
            else:
                self.pin_names = []
            
            # Load metadata
            self.learning_steps = int(data.get('learning_steps', 0))
            self.total_spikes = int(data.get('total_spikes', 0))
            self.edf_source = str(data.get('edf_source', 'unknown'))
            self.created = str(data.get('created', 'unknown'))
            
            # Initialize neural state
            n = self.grid_size
            self.v = np.ones((n, n), dtype=np.float32) * self.c
            self.u = self.v * self.b
            
            # Categorize pins by region
            self._categorize_pins()
            
            fname = os.path.basename(self.crystal_path)
            self.status_msg = f"Loaded {fname} | {n}x{n} | {len(self.pin_coords)} pins"
            self.is_loaded = True
            
            print(f"[CrystalChip] Loaded crystal: {n}x{n}, {len(self.pin_coords)} pins")
            print(f"  Input pins: {len(self.input_pin_indices)}")
            print(f"  Output pins: {len(self.output_pin_indices)}")
            print(f"  Internal pins: {len(self.internal_pin_indices)}")
            print(f"  Learned from: {self.edf_source}")
            print(f"  Training steps: {self.learning_steps}")
            
        except Exception as e:
            self.status_msg = f"Load error: {str(e)[:30]}"
            self.is_loaded = False
            print(f"[CrystalChip] Error loading crystal: {e}")
    
    def _categorize_pins(self):
        """Sort pins into input/output/internal categories based on position."""
        self.input_pin_indices = []
        self.output_pin_indices = []
        self.internal_pin_indices = []
        
        # First try by name if available
        if self.pin_names and len(self.pin_names) == len(self.pin_coords):
            for i, name in enumerate(self.pin_names):
                name_upper = str(name).upper().strip()
                
                if any(inp in name_upper for inp in self.INPUT_PINS):
                    self.input_pin_indices.append(i)
                elif any(out in name_upper for out in self.OUTPUT_PINS):
                    self.output_pin_indices.append(i)
                elif any(internal in name_upper for internal in self.INTERNAL_PINS):
                    self.internal_pin_indices.append(i)
                else:
                    self.internal_pin_indices.append(i)
        
        # If no categorization worked (no names or names didn't match), use position
        if not self.input_pin_indices and not self.output_pin_indices:
            # Use neuroanatomical position: frontal (top) = input, occipital (bottom) = output
            for i, (r, c) in enumerate(self.pin_coords):
                # Normalize position to 0-1 range
                r_norm = r / self.grid_size
                
                if r_norm < 0.35:  # Top 35% = frontal = input
                    self.input_pin_indices.append(i)
                elif r_norm > 0.65:  # Bottom 35% = occipital = output
                    self.output_pin_indices.append(i)
                else:  # Middle = central = internal
                    self.internal_pin_indices.append(i)
            
            # Ensure we have at least some inputs and outputs
            if not self.input_pin_indices and self.pin_coords:
                # Take first third as inputs
                n = len(self.pin_coords)
                self.input_pin_indices = list(range(n // 3))
            if not self.output_pin_indices and self.pin_coords:
                # Take last third as outputs  
                n = len(self.pin_coords)
                self.output_pin_indices = list(range(2 * n // 3, n))
    
    def _read_input(self, name, default=None):
        """Read an input value."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                return val
            except:
                return default
        return default
    
    def _read_image_input(self, name):
        """Read an image input, converting QImage to numpy if needed."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first")
                if val is None:
                    return None
                
                # If it's already a numpy array, return it
                if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                    return val
                
                # If it's a QImage, convert to numpy
                if hasattr(val, 'width') and hasattr(val, 'height') and hasattr(val, 'bits'):
                    # QImage conversion
                    width = val.width()
                    height = val.height()
                    
                    # Get bytes per line for proper array reshaping
                    bytes_per_line = val.bytesPerLine()
                    
                    # Get pointer to image data
                    ptr = val.bits()
                    if ptr is None:
                        return None
                    
                    # Convert to numpy - handle different formats
                    try:
                        ptr.setsize(height * bytes_per_line)
                        arr = np.array(ptr).reshape(height, bytes_per_line)
                        
                        # Determine channels based on format
                        fmt = val.format()
                        if fmt == 4:  # Format_RGB32 or Format_ARGB32
                            arr = arr[:, :width*4].reshape(height, width, 4)
                            arr = arr[:, :, :3]  # Drop alpha, keep RGB
                        elif fmt == 13:  # Format_RGB888
                            arr = arr[:, :width*3].reshape(height, width, 3)
                        elif fmt == 24:  # Format_Grayscale8
                            arr = arr[:, :width]
                        else:
                            # Try to handle as RGB
                            if bytes_per_line >= width * 3:
                                arr = arr[:, :width*3].reshape(height, width, 3)
                            else:
                                arr = arr[:, :width]
                        
                        return arr.astype(np.float32)
                    except Exception as e:
                        print(f"[CrystalChip] QImage conversion error: {e}")
                        return None
                
            except Exception as e:
                print(f"[CrystalChip] Image read error: {e}")
                pass
        return None
    
    def _read_latent_input(self, name):
        """Read a latent/spectrum input."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first")
                if val is not None and isinstance(val, np.ndarray):
                    return val
            except:
                pass
        return None
    
    def step(self):
        self._maybe_reload()
        
        if not self.is_loaded:
            self._update_display()
            return
        
        self.step_count += 1
        
        # Read modulation inputs
        gain_mod = self._read_input("gain", 1.0)
        coupling_mod = self._read_input("coupling", 1.0)
        reset = self._read_input("reset", 0.0)
        
        if reset and reset > 0.5:
            self._reset_state()
            return
        
        effective_gain = self.input_gain * float(gain_mod)
        effective_coupling = self.base_coupling * float(coupling_mod)
        
        # === BUILD INPUT CURRENT ===
        n = self.grid_size
        I = np.zeros((n, n), dtype=np.float32)
        
        # 1. Signal input → all input pins equally
        signal_in = self._read_input("signal_in", 0.0)
        if signal_in and signal_in != 0.0:
            self._inject_at_pins(I, self.input_pin_indices, float(signal_in) * effective_gain)
        
        # 2. Image input → spatially mapped to input pins
        image_in = self._read_image_input("image_in")
        if image_in is not None:
            self._inject_image(I, image_in, effective_gain)
        
        # 3. Latent input → distributed across input pins
        latent_in = self._read_latent_input("latent_in")
        if latent_in is not None:
            self._inject_latent(I, latent_in, effective_gain)
        
        # === NEURAL DYNAMICS ===
        v = self.v.copy()
        u = self.u.copy()
        
        # Get neighbor voltages
        v_up = np.roll(v, -1, axis=0)
        v_down = np.roll(v, 1, axis=0)
        v_left = np.roll(v, -1, axis=1)
        v_right = np.roll(v, 1, axis=1)
        
        # Weighted coupling through crystal geometry
        neighbor_influence = (
            self.weights_up * v_up +
            self.weights_down * v_down +
            self.weights_left * v_left +
            self.weights_right * v_right
        )
        
        total_weight = (self.weights_up + self.weights_down + 
                       self.weights_left + self.weights_right)
        neighbor_avg = neighbor_influence / (total_weight + 1e-6)
        
        I_coupling = effective_coupling * (neighbor_avg - v)
        
        # Clamp to prevent overflow
        I = np.clip(I, -100, 100)
        I_coupling = np.clip(I_coupling, -50, 50)
        
        # Izhikevich dynamics
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I + I_coupling) * self.dt
        du = self.a * (self.b * v - u) * self.dt
        
        v = v + dv
        u = u + du
        
        # Clamp to prevent overflow
        v = np.clip(v, -100, 50)
        u = np.clip(u, -50, 50)
        
        # Detect spikes
        spikes = v >= 30.0
        v[spikes] = self.c
        u[spikes] += self.d
        
        # Clean up NaN
        v = np.nan_to_num(v, nan=self.c, posinf=50.0, neginf=-100.0)
        u = np.nan_to_num(u, nan=0.0, posinf=50.0, neginf=-50.0)
        
        self.v = v
        self.u = u
        
        # === COMPUTE OUTPUTS ===
        self._compute_outputs()
        
        self._update_display()
    
    def _inject_at_pins(self, I, pin_indices, value):
        """Inject current at specified pins with spatial spread."""
        if not pin_indices:
            return
        
        r = self.spread_radius
        for idx in pin_indices:
            if idx < len(self.pin_coords):
                row, col = self.pin_coords[idx]
                for dr in range(-r, r + 1):
                    for dc in range(-r, r + 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            dist = np.sqrt(dr * dr + dc * dc)
                            weight = np.exp(-dist / max(r, 1))
                            I[nr, nc] += value * weight
    
    def _inject_image(self, I, image, gain):
        """Project image onto input pins based on their spatial arrangement."""
        if len(self.input_pin_indices) == 0:
            return
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float32)
        
        # Normalize
        gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-6)
        
        # For each input pin, sample the image at its relative position
        for idx in self.input_pin_indices:
            if idx < len(self.pin_coords):
                row, col = self.pin_coords[idx]
                
                # Map pin position to image coordinates
                img_row = int((row / self.grid_size) * gray.shape[0])
                img_col = int((col / self.grid_size) * gray.shape[1])
                
                img_row = np.clip(img_row, 0, gray.shape[0] - 1)
                img_col = np.clip(img_col, 0, gray.shape[1] - 1)
                
                value = gray[img_row, img_col] * gain
                self._inject_at_pins(I, [idx], value)
    
    def _inject_latent(self, I, latent, gain):
        """Distribute latent vector across input pins."""
        if len(self.input_pin_indices) == 0:
            return
        
        # Ensure latent is 1D
        if latent.ndim > 1:
            latent = latent.flatten()
        
        # Map latent dimensions to input pins (cycling if needed)
        for i, idx in enumerate(self.input_pin_indices):
            latent_idx = i % len(latent)
            value = float(latent[latent_idx]) * gain
            self._inject_at_pins(I, [idx], value)
    
    def _compute_outputs(self):
        """Compute output signals from output pin activity."""
        
        # Read activity at output pins
        output_activities = []
        for idx in self.output_pin_indices:
            if idx < len(self.pin_coords):
                row, col = self.pin_coords[idx]
                if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                    output_activities.append(self.v[row, col])
        
        if output_activities:
            # Signal out = mean output activity
            self._output_values["signal_out"] = float(np.mean(output_activities))
            
            # Latent out = first 16 output activities (or padded)
            latent = np.zeros(16, dtype=np.float32)
            for i, act in enumerate(output_activities[:16]):
                latent[i] = act
            self._latent_out = latent
        else:
            self._output_values["signal_out"] = float(np.mean(self.v))
            self._latent_out = np.zeros(16, dtype=np.float32)
        
        # Resonance = variance of activity (high variance = resonating)
        self.current_resonance = float(np.var(self.v))
        self._output_values["resonance"] = self.current_resonance
        
        # Energy = sum of squared activity
        self.current_energy = float(np.sum(self.v ** 2))
        self._output_values["energy"] = self.current_energy
        
        # === EEG-LIKE FREQUENCY BAND EXTRACTION ===
        # Compute LFP (local field potential) as mean activity
        self.current_lfp = float(np.mean(self.v))
        
        # Add to history buffer (circular)
        self.lfp_history[self.lfp_idx] = self.current_lfp
        self.lfp_idx = (self.lfp_idx + 1) % self.lfp_history_size
        
        # Extract frequency bands using FFT
        self._extract_frequency_bands()
        
        # Update output values
        self._output_values["lfp"] = self.current_lfp
        self._output_values["delta"] = self.band_powers["delta"]
        self._output_values["theta"] = self.band_powers["theta"]
        self._output_values["alpha"] = self.band_powers["alpha"]
        self._output_values["beta"] = self.band_powers["beta"]
        self._output_values["gamma"] = self.band_powers["gamma"]
    
    def _extract_frequency_bands(self):
        """Extract EEG-like frequency bands from LFP history using FFT."""
        # Reorder history to be chronological
        history = np.roll(self.lfp_history, -self.lfp_idx)
        
        # Remove DC offset
        history = history - np.mean(history)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(len(history))
        windowed = history * window
        
        # Compute FFT
        fft = np.fft.rfft(windowed)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(history), d=1.0/self.sample_rate)
        
        # Extract band powers
        # Delta: 0.5-4 Hz
        delta_mask = (freqs >= 0.5) & (freqs < 4)
        self.band_powers["delta"] = float(np.sum(power[delta_mask])) if np.any(delta_mask) else 0.0
        
        # Theta: 4-8 Hz
        theta_mask = (freqs >= 4) & (freqs < 8)
        self.band_powers["theta"] = float(np.sum(power[theta_mask])) if np.any(theta_mask) else 0.0
        
        # Alpha: 8-13 Hz
        alpha_mask = (freqs >= 8) & (freqs < 13)
        self.band_powers["alpha"] = float(np.sum(power[alpha_mask])) if np.any(alpha_mask) else 0.0
        
        # Beta: 13-30 Hz
        beta_mask = (freqs >= 13) & (freqs < 30)
        self.band_powers["beta"] = float(np.sum(power[beta_mask])) if np.any(beta_mask) else 0.0
        
        # Gamma: 30-50 Hz (limited by Nyquist at 100Hz sample rate)
        gamma_mask = (freqs >= 30) & (freqs < 50)
        self.band_powers["gamma"] = float(np.sum(power[gamma_mask])) if np.any(gamma_mask) else 0.0
        
        # Normalize to reasonable range (log scale for display)
        for band in self.band_powers:
            val = self.band_powers[band]
            if val > 0:
                # Log scale, shifted to be mostly positive
                self.band_powers[band] = np.log10(val + 1) * 10
            else:
                self.band_powers[band] = 0.0
    
    def _reset_state(self):
        """Reset neural state to resting."""
        if self.is_loaded:
            n = self.grid_size
            self.v = np.ones((n, n), dtype=np.float32) * self.c
            self.u = self.v * self.b
    
    def get_output(self, port_name):
        if port_name == "chip_view":
            return self.display_image
        elif port_name == "activity_view":
            return self._render_activity()
        elif port_name == "image_out":
            return self._render_output_image()
        elif port_name == "latent_out":
            return self._latent_out
        elif port_name in self._output_values:
            return self._output_values.get(port_name, 0.0)
        return None
    
    def _render_activity(self):
        """Render full activity pattern."""
        if not self.is_loaded:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        n = self.grid_size
        disp = np.clip(self.v, -90.0, 40.0)
        norm = ((disp + 90.0) / 130.0 * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        heat = cv2.resize(heat, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Draw pins
        scale = 256 / n
        for i, (r, c) in enumerate(self.pin_coords):
            center = (int(c * scale), int(r * scale))
            if i in self.input_pin_indices:
                color = (0, 255, 0)  # Green = input
            elif i in self.output_pin_indices:
                color = (0, 0, 255)  # Red = output
            else:
                color = (255, 255, 0)  # Yellow = internal
            cv2.circle(heat, center, 4, color, -1)
        
        return heat
    
    def _render_output_image(self):
        """Render output pin activity as a small image."""
        # Create image from output pin activities
        n_out = len(self.output_pin_indices)
        if n_out == 0:
            return np.zeros((8, 8, 3), dtype=np.uint8)
        
        # Find grid size that fits output pins
        size = int(np.ceil(np.sqrt(n_out)))
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        for i, idx in enumerate(self.output_pin_indices):
            if idx < len(self.pin_coords):
                row, col = self.pin_coords[idx]
                if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                    activity = self.v[row, col]
                    # Normalize to 0-255
                    val = int(np.clip((activity + 90) / 130 * 255, 0, 255))
                    
                    img_row = i // size
                    img_col = i % size
                    if img_row < size and img_col < size:
                        img[img_row, img_col] = [val, val, val]
        
        # Scale up
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        return img
    
    def _update_display(self):
        """Create main display."""
        w, h = 512, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "CRYSTAL CHIP", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 180), 2)
        
        if not self.is_loaded:
            cv2.putText(img, self.status_msg, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(img, "Load a crystal .npz file", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        else:
            # Status line
            cv2.putText(img, self.status_msg, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Activity view
            activity = self._render_activity()
            activity_small = cv2.resize(activity, (200, 200))
            img[70:270, 10:210] = activity_small
            cv2.putText(img, "Activity", (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Pin legend
            cv2.circle(img, (20, 305), 5, (0, 255, 0), -1)
            cv2.putText(img, f"Input ({len(self.input_pin_indices)})", (30, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            
            cv2.circle(img, (120, 305), 5, (0, 0, 255), -1)
            cv2.putText(img, f"Output ({len(self.output_pin_indices)})", (130, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            # Crystal structure view
            crystal = self._render_crystal()
            crystal_small = cv2.resize(crystal, (200, 200))
            img[70:270, 230:430] = crystal_small
            cv2.putText(img, "Crystal Structure", (230, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Output preview - position it to fit within 512 width
            out_img = self._render_output_image()
            out_img_resized = cv2.resize(out_img, (70, 70))
            img[70:140, 440:510] = out_img_resized
            cv2.putText(img, "Output", (445, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Stats
            stats_y = 200
            cv2.putText(img, f"Step: {self.step_count}", (440, stats_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(img, f"Signal Out: {self._output_values['signal_out']:.1f}", (440, stats_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
            cv2.putText(img, f"Resonance: {self.current_resonance:.1f}", (440, stats_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
            cv2.putText(img, f"Energy: {self.current_energy:.0f}", (440, stats_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
            
            # Crystal metadata
            cv2.putText(img, "Crystal Info:", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(img, f"Source: {os.path.basename(self.edf_source)}", (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(img, f"Training: {self.learning_steps} steps", (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(img, f"Spikes: {self.total_spikes:,}", (10, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def _render_crystal(self):
        """Render the crystal weight structure."""
        if not self.is_loaded:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        n = self.grid_size
        
        # Combine weights into visualization
        horizontal = (self.weights_left + self.weights_right) / 2
        vertical = (self.weights_up + self.weights_down) / 2
        
        # Normalize
        w_min, w_max = 0.01, 2.0
        h_norm = np.clip((horizontal - w_min) / (w_max - w_min), 0, 1)
        v_norm = np.clip((vertical - w_min) / (w_max - w_min), 0, 1)
        
        anisotropy = np.abs(h_norm - v_norm)
        
        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[:, :, 0] = (h_norm * 255).astype(np.uint8)
        img[:, :, 1] = ((1 - anisotropy) * 255).astype(np.uint8)
        img[:, :, 2] = (v_norm * 255).astype(np.uint8)
        
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return img
    
    def get_display_image(self):
        return self.display_image
    
    # === STATE PERSISTENCE ===
    def save_custom_state(self, folder_path, node_id):
        """Save current state."""
        filename = f"crystal_chip_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        
        np.savez(filepath,
                 crystal_path=self.crystal_path,
                 step_count=self.step_count)
        
        return filename
    
    def load_custom_state(self, filepath):
        """Load saved state."""
        try:
            data = np.load(filepath, allow_pickle=True)
            self.crystal_path = str(data.get('crystal_path', ''))
            self.step_count = int(data.get('step_count', 0))
            
            if self.crystal_path:
                self._last_path = ""  # Force reload
                self._maybe_reload()
        except Exception as e:
            print(f"[CrystalChip] Error loading state: {e}")