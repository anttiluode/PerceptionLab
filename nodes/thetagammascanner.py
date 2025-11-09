"""
Theta-Gamma Sweep Scanner Node
-----------------------------------
This node simulates a dynamic cortical map based on concepts from
four key papers:

1.  Fractal Cortex (Wang et al., 2024): The node uses a 2D map
    representing the cortex, which is described as a fractal structure.
    
2.  Theta Sweeps (Vollan et al., 2025): The map is scanned by a
    theta-paced (8Hz) "look around" mechanism that alternates
    left and right, modeling the hippocampal-entorhinal system.
    
3.  Gamma Gating (Drebitz et al., 2025): Information is processed
    (gated) based on its phase-relationship to an internal gamma
    oscillation (40Hz), modeling "communication through coherence".
    [cite: 6244, 6606]
4.  Time-Domain Brain (Baker & Cariani, 2025): The node is
    "signal-centric"  and models the interaction between
    oscillation bands (Theta and Gamma) as a core processing
    mechanism. [cite: 4599]

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
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: ThetaGammaScannerNode requires scipy")

class ThetaGammaScannerNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(0, 150, 200)  # Deep Teal
    
    def __init__(self, map_size=64, learning_rate=0.05, decay_rate=0.99, sweep_angle_deg=30.0, theta_freq_hz=8.0, gamma_freq_hz=40.0):
        super().__init__()
        self.node_title = "Theta-Gamma Scanner"
        
        self.inputs = {
            'phase_field': 'image',       # The sensory input to process
            'internal_direction': 'signal', # Bias for the sweep (e.g., head direction)
            'ext_theta': 'signal',        # Optional external theta to sync with
            'ext_gamma': 'signal'         # The "phase" of the input signal
        }
        
        self.outputs = {
            'gated_output': 'image',      # The input signal, gated by coherence
            'memory_map': 'image',        # The internal holographic/fractal map
            'theta_phase': 'signal',      # Our internal theta clock output
            'gamma_phase': 'signal',      # Our internal gamma clock output
            'coherence_gate': 'signal'    # The resulting gamma gate (0-1)
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Theta-Gamma (No SciPy!)"
            return
            
        # --- Configurable Parameters ---
        self.map_size = int(map_size)
        self.learning_rate = float(learning_rate)
        self.decay_rate = float(decay_rate)
        self.sweep_angle_deg = float(sweep_angle_deg)
        self.theta_freq_hz = float(theta_freq_hz)
        self.gamma_freq_hz = float(gamma_freq_hz)

        # --- Internal State ---
        # 1. The Holographic Map (from Paper 1 & 3)
        self.memory_map = np.random.rand(self.map_size, self.map_size).astype(np.float32) * 0.1
        
        # 2. Oscillators (from Paper 2, 3, 4)
        self.theta_phase_rad = 0.0
        self.gamma_phase_rad = 0.0
        self.last_theta_cos = 1.0
        # Assuming 30 FPS for simulation, pre-calculate increments
        self.theta_increment = (2 * np.pi * self.theta_freq_hz) / 30.0
        self.gamma_increment = (2 * np.pi * self.gamma_freq_hz) / 30.0

        # 3. Theta Sweep State (from Paper 2)
        self.sweep_direction = 1.0  # 1.0 for Right, -1.0 for Left
        
        # --- Output Buffers ---
        self.gated_output_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.float32)
        self.coherence_gate_out = 0.0
        
        # Gaze mask for visualization
        self.gaze_mask = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.sweep_x = self.map_size // 2
        self.sweep_y = self.map_size // 2


    def _create_gaze_mask(self, center_x, center_y, size, max_val=1.0):
        """Creates a soft circular mask at the sweep gaze location."""
        y, x = np.indices((size, size))
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        sigma_sq = (size / 10.0)**2  # Make the gaze area ~10% of the map
        mask = max_val * np.exp(-dist_sq / (2 * sigma_sq))
        return mask

    def step(self):
        if not SCIPY_AVAILABLE:
            return
            
        # --- 1. Get Inputs ---
        phase_field_in = self.get_blended_input('phase_field', 'mean')
        base_direction_in = self.get_blended_input('internal_direction', 'sum') or 0.0
        ext_theta_in = self.get_blended_input('ext_theta', 'sum')
        ext_gamma_in = self.get_blended_input('ext_gamma', 'sum')

        # --- 2. Update Oscillators (Paper 3) ---
        
        # Update Theta
        if ext_theta_in is not None:
            self.theta_phase_rad = np.arccos(np.clip(ext_theta_in, -1, 1))
        else:
            self.theta_phase_rad = (self.theta_phase_rad + self.theta_increment) % (2 * np.pi)
        
        current_theta_cos = np.cos(self.theta_phase_rad)
        
        # Update Gamma
        if ext_gamma_in is not None:
            # If external gamma is provided, we phase-lock to it
            self.gamma_phase_rad = np.arccos(np.clip(ext_gamma_in, -1, 1))
        else:
            self.gamma_phase_rad = (self.gamma_phase_rad + self.gamma_increment) % (2 * np.pi)

        # --- 3. Update Theta Sweep (Paper 2) ---
        
        # Check for theta trough (crossing from negative to positive)
        # This is when the sweep alternates [cite: 1424, 1560]
        if self.last_theta_cos < 0 and current_theta_cos >= 0:
            self.sweep_direction *= -1.0  # Flip direction
            
        self.last_theta_cos = current_theta_cos
        
        # Calculate sweep angle
        sweep_angle_rad = np.deg2rad(base_direction_in + (self.sweep_direction * self.sweep_angle_deg))
        
        # Theta phase drives sweep length (0 at trough, 1 at peak)
        # "sweeps linearly outwards from the animal's location" [cite: 1424]
        sweep_progress = (current_theta_cos + 1.0) / 2.0  # 0 -> 1
        sweep_length = (self.map_size / 2.0) * sweep_progress
        
        # Calculate current "gaze" position of the sweep
        center_x = self.map_size // 2 + sweep_length * np.cos(sweep_angle_rad)
        center_y = self.map_size // 2 + sweep_length * np.sin(sweep_angle_rad)
        self.sweep_x, self.sweep_y = center_x, center_y
        
        # Create a soft mask for the gaze location
        self.gaze_mask = self._create_gaze_mask(center_x, center_y, self.map_size)
        
        # --- 4. Apply Gamma Gating (Paper 4) ---
        
        # "communication through coherence" [cite: 6890]
        # The gate opens if the input gamma phase matches the internal gamma phase.
        if ext_gamma_in is not None:
            ext_gamma_rad = np.arccos(np.clip(ext_gamma_in, -1, 1))
            phase_difference = self.gamma_phase_rad - ext_gamma_rad
            # Gate is max (1) at 0 diff, min (0) at pi diff
            self.coherence_gate_out = (np.cos(phase_difference) + 1.0) / 2.0
        else:
            # No external gamma, so gate is just driven by internal excitability
            # "afferent spikes should be most effective when they arrive during the sensitive phase" [cite: 6256]
            self.coherence_gate_out = (np.cos(self.gamma_phase_rad) + 1.0) / 2.0 # Assumes peak is sensitive
            
        # --- 5. Process Signal (Write to Map) ---
        
        if phase_field_in is None:
            phase_field_in = np.random.rand(self.map_size, self.map_size)
        
        if phase_field_in.shape[0] != self.map_size:
            phase_field_in = cv2.resize(phase_field_in, (self.map_size, self.map_size))
            
        if phase_field_in.ndim == 3:
            phase_field_in = np.mean(phase_field_in, axis=2)
            
        # Apply gating: sensory input * sweep_location * gamma_gate
        gated_signal = phase_field_in * self.gaze_mask * self.coherence_gate_out
        
        # Update the memory map (Holographic/Fractal store)
        self.memory_map += gated_signal * self.learning_rate
        # Apply decay/forgetting
        self.memory_map = (self.memory_map * self.decay_rate).astype(np.float32)
        np.clip(self.memory_map, 0, 1, out=self.memory_map)
        
        # Prepare gated signal for output
        self.gated_output_img = (np.clip(gated_signal, 0, 1) * 255).astype(np.uint8)
        self.gated_output_img = cv2.cvtColor(self.gated_output_img, cv2.COLOR_GRAY2RGB)
        
        
    def get_output(self, port_name):
        if port_name == 'gated_output':
            return self.gated_output_img
        elif port_name == 'memory_map':
            return self.memory_map
        elif port_name == 'theta_phase':
            return np.cos(self.theta_phase_rad)
        elif port_name == 'gamma_phase':
            return np.cos(self.gamma_phase_rad)
        elif port_name == 'coherence_gate':
            return self.coherence_gate_out
        return None

    def get_display_image(self):
        if not SCIPY_AVAILABLE: return None
        
        # Create a detailed visualization
        display_w = 512
        display_h = 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Left side: Memory Map
        map_u8 = (np.clip(self.memory_map, 0, 1) * 255).astype(np.uint8)
        map_resized = cv2.resize(cv2.cvtColor(map_u8, cv2.COLOR_GRAY2RGB), 
                                 (display_h, display_h), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Draw the sweep line on the map
        line_start = (display_h // 2, display_h // 2)
        line_end = (int(self.sweep_x / self.map_size * display_h),
                    int(self.sweep_y / self.map_size * display_h))
        cv2.line(map_resized, line_start, line_end, (255, 0, 255), 2)
        cv2.circle(map_resized, line_end, 8, (255, 0, 255), -1)
        
        display[:, :display_h] = map_resized
        
        # Right side: Gated Input (What's being "seen")
        gated_resized = cv2.resize(self.gated_output_img, 
                                   (display_h, display_h), 
                                   interpolation=cv2.INTER_NEAREST)
        display[:, display_w-display_h:] = gated_resized
        
        # Add dividing line
        display[:, display_h-1:display_h+1] = [255, 255, 255]
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'MEMORY MAP', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'GATED SENSORY INPUT', (display_h + 10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add oscillator info
        theta_val = np.cos(self.theta_phase_rad)
        gamma_val = np.cos(self.gamma_phase_rad)
        sweep_dir_str = "RIGHT" if self.sweep_direction > 0 else "LEFT"
        
        cv2.putText(display, f"THETA: {theta_val:+.2f} ({self.theta_freq_hz}Hz)", (10, display_h - 40), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display, f"GAMMA: {gamma_val:+.2f} ({self.gamma_freq_hz}Hz)", (10, display_h - 25), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display, f"SWEEP: {sweep_dir_str}", (10, display_h - 10), font, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(display, f"COHERENCE: {self.coherence_gate_out:.2f}", (display_h + 10, display_h - 10), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Map Size", "map_size", self.map_size, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Decay Rate", "decay_rate", self.decay_rate, None),
            ("Sweep Angle (deg)", "sweep_angle_deg", self.sweep_angle_deg, None),
            ("Theta Freq (Hz)", "theta_freq_hz", self.theta_freq_hz, None),
            ("Gamma Freq (Hz)", "gamma_freq_hz", self.gamma_freq_hz, None),
        ]