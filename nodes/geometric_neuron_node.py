"""
Geometric Neuron Node - Full Biological ODE
-------------------------------------------
Implements the complete geometric neuron model:

dCable/dt = -α·Cable + D·∇²Cable + I_syn + I_ephap + η(x,t)

Where:
- Cable is a Takens manifold (position = time delay)
- Grating samples every 3 compartments (190nm actin scaffold)
- Templates are Koopman eigenmodes (spectral islands)
- Spike position encodes resonance strength (Nav1.6→Nav1.2 gradient)
- Ephaptic field couples neurons spatially
- Scorching = structural plasticity (long-term memory)
"""

import numpy as np
import cv2
from collections import deque
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class GeometricNeuronNode(BaseNode):
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(180, 50, 80)  # Biological red

    def __init__(self, cable_length=128, n_templates=8, rc_alpha=0.93, 
                 diffusion=0.1, grating_period=3):
        super().__init__()
        self.node_title = "Geometric Neuron"

        # Inputs
        self.inputs = {
            'signal_in': 'signal',      # Synaptic input (spikes from others)
            'ephaptic_in': 'signal'     # Extracellular field from all neurons
        }

        # Outputs
        self.outputs = {
            'spike_out': 'signal',          # Binary spike (0/1)
            'spike_pos_out': 'signal',      # Position along AIS (0-1 normalized)
            'resonance_out': 'signal',      # Current resonance strength
            'ephaptic_out': 'signal',       # Field this neuron contributes
            'manifold_img': 'image'         # Visual of cable state
        }

        # === Cable Parameters (Takens Manifold) ===
        self.cable_length = int(cable_length)
        self.rc_alpha = float(rc_alpha)           # RC decay per step
        self.diffusion = float(diffusion)         # Spread along cable
        self.cable_state = np.zeros(self.cable_length, dtype=np.float32)
        self.manifold_scorch = np.zeros(self.cable_length, dtype=np.float32)
        self.manifold_decay = 0.9998
        self.scorch_rate = 0.001

        # === Grating Sampling (190nm actin scaffold) ===
        self.grating_period = int(grating_period)
        self.grating_idx = np.arange(0, self.cable_length, self.grating_period)
        self.n_samples = len(self.grating_idx)

        # === Template Library (Koopman Spectral Islands) ===
        self.n_templates = int(n_templates)
        # Initialize with different spatial frequencies
        self.templates = np.zeros((self.n_templates, self.n_samples), dtype=np.float32)
        for i in range(self.n_templates):
            # Each template is a different spatial frequency on the grating
            freq = 0.05 + (i / self.n_templates) * 0.5
            phase = np.random.rand() * 2 * np.pi
            for j, pos in enumerate(self.grating_idx):
                self.templates[i, j] = np.sin(2 * np.pi * freq * pos / self.cable_length + phase)
            # Normalize
            norm = np.linalg.norm(self.templates[i])
            if norm > 0:
                self.templates[i] /= norm
        self.template_scorch = np.zeros(self.n_templates, dtype=np.float32)

        # === Nav Threshold Gradient (Nav1.6 distal → Nav1.2 proximal) ===
        self.threshold_min = 0.25    # Nav1.6 (distal, sensitive)
        self.threshold_max = 0.70    # Nav1.2 (proximal, requires strong signal)
        self.thresholds = np.linspace(self.threshold_min, self.threshold_max, self.n_samples)

        # === Membrane Integrator ===
        self.membrane = 0.0
        self.membrane_leak = 0.88
        self.charge_rate = 0.14
        self.spike_threshold = 0.52

        # === Ephaptic Field Parameters ===
        self.ephaptic_strength = 0.03      # How strongly field affects cable
        self.ephaptic_decay = 0.95         # Field decay rate
        self.ephaptic_buffer = 0.0
        self.johnson_nyquist_sigma = 0.035  # Thermal noise

        # === State Tracking ===
        self.current_resonance = 0.0
        self.current_spike = 0.0
        self.current_spike_pos = -1
        self.current_mismatch = 1.0
        self.best_template_idx = 0
        
        # History for visualization
        self.resonance_history = deque(maxlen=200)
        self.spike_history = deque(maxlen=200)
        
        # === Visualization ===
        self.display_image = np.zeros((128, 256, 3), dtype=np.uint8)
        self.step_counter = 0

    def _diffusion_operator(self, state):
        """Compute second spatial derivative (diffusion along cable)"""
        diff = np.zeros_like(state)
        diff[1:-1] = state[2:] - 2*state[1:-1] + state[:-2]
        # Neumann boundaries (no flux at ends)
        diff[0] = state[1] - state[0]
        diff[-1] = state[-2] - state[-1]
        return diff

    def _activate_ephaptic(self, field, cable_pos):
        """
        Ephaptic activating function: second derivative of field along cable
        This is what actually polarizes the membrane
        """
        # Check if the field is a single scalar number
        if np.isscalar(field) or np.ndim(field) == 0:
            distance_norm = cable_pos / self.cable_length  # 0=distal, 1=proximal
            # Field affects distal end more strongly (Nav1.6 is there)
            return field * (1.0 - distance_norm * 0.5)
            
        # If it's an array but doesn't match the cable length
        elif len(field) != len(cable_pos):
            distance_norm = cable_pos / self.cable_length
            return np.mean(field) * (1.0 - distance_norm * 0.5)
            
        # If the field already has full spatial structure
        else:
            diff = np.zeros_like(field)
            diff[1:-1] = field[2:] - 2*field[1:-1] + field[:-2]
            return diff * 0.1

    def step(self):
        self.step_counter += 1
        
        # 1. Gather Inputs
        synaptic_input = self.get_blended_input('signal_in', 'sum') or 0.0
        ephaptic_field = self.get_blended_input('ephaptic_in', 'sum') or 0.0
        
        # Clamp to reasonable range
        synaptic_input = np.clip(synaptic_input, -2.0, 2.0)
        ephaptic_field = np.clip(ephaptic_field, -1.0, 1.0)
        
        # 2. Update Cable State (Takens Manifold ODE)
        # ∂Cable/∂t = -α·Cable + D·∇²Cable + I_syn + I_ephap + η
        
        # Shift right (signal propagation) + RC decay
        new_cable = np.zeros_like(self.cable_state)
        new_cable[0] = synaptic_input
        new_cable[1:] = self.cable_state[:-1] * self.rc_alpha
        
        # Diffusion (spread along cable)
        diffusion_term = self.diffusion * self._diffusion_operator(self.cable_state)
        new_cable += diffusion_term
        
        # Ephaptic perturbation (field effect)
        ephaptic_term = self._activate_ephaptic(ephaptic_field, 
                                                 np.arange(self.cable_length))
        new_cable += ephaptic_term * self.ephaptic_strength
        
        # Johnson-Nyquist noise (thermal, position-dependent)
        # Distal end has more noise (more channels)
        noise_scale = self.johnson_nyquist_sigma * (1.0 + 0.5 * np.random.rand())
        noise = np.random.normal(0, noise_scale, self.cable_length)
        new_cable += noise
        
        # Structural manifold (scorched memory)
        new_cable += self.manifold_scorch * 0.085
        
        self.cable_state = new_cable
        
        # 3. Grating Sampling
        sampled = self.cable_state[self.grating_idx]
        norm_sampled = sampled / (np.linalg.norm(sampled) + 1e-8)
        
        # 4. Template Matching (Attention over Koopman modes)
        alignments = np.dot(self.templates, norm_sampled)
        resonances = alignments ** 2
        # Scorch weighting: learned templates resonate more
        weighted = resonances * (1.0 + self.template_scorch * 0.3)
        
        self.best_template_idx = np.argmax(weighted)
        self.current_resonance = float(resonances[self.best_template_idx])
        self.current_mismatch = 1.0 - self.current_resonance
        
        # 5. Spike Position (Nav gradient threshold crossing)
        # The position along AIS where resonance first exceeds threshold
        self.current_spike_pos = -1
        for i, thresh in enumerate(self.thresholds):
            if self.current_resonance >= thresh:
                self.current_spike_pos = i
                break  # First crossing = most distal
        
        # 6. Membrane Integration
        self.membrane = self.membrane * self.membrane_leak + self.current_resonance * self.charge_rate
        
        # 7. Spike Generation & Structural Plasticity
        if self.membrane > self.spike_threshold:
            self.current_spike = 1.0
            self.membrane = 0.0
            
            # --- SCORCHING (Structural Plasticity) ---
            # Scorch the cable manifold (long-term memory of trajectory)
            scorch_strength = self.current_resonance * 0.001
            self.manifold_scorch = (self.manifold_scorch * self.manifold_decay + 
                                    self.cable_state * scorch_strength)
            
            # Scorch the winning template (Hebbian update)
            strength = 0.004 * self.current_resonance
            t = self.templates[self.best_template_idx]
            new_template = t * (1 - strength) + norm_sampled * strength
            norm = np.linalg.norm(new_template)
            if norm > 0:
                self.templates[self.best_template_idx] = new_template / norm
            self.template_scorch[self.best_template_idx] = min(1.0, 
                self.template_scorch[self.best_template_idx] + 0.002)
        else:
            self.current_spike = 0.0
        
        # 8. Ephaptic Output (What this neuron broadcasts)
        # The field contribution depends on spike and resonance
        if self.current_spike > 0:
            # Stronger spike = larger field, proximal spikes broadcast stronger
            pos_norm = self.current_spike_pos / self.n_samples if self.current_spike_pos >= 0 else 0
            self.ephaptic_buffer = self.current_resonance * (0.5 + pos_norm * 0.5)
        else:
            # Subthreshold resonance still leaks into field
            self.ephaptic_buffer = self.current_resonance * 0.1
        
        # Add Johnson-Nyquist noise to the field itself
        self.ephaptic_buffer += np.random.normal(0, 0.02)
        self.ephaptic_buffer = np.clip(self.ephaptic_buffer, -1.0, 1.0)
        
        # 9. Update History
        self.resonance_history.append(self.current_resonance)
        self.spike_history.append(self.current_spike)
        
        # 10. Update Visualization
        self._update_display()

    def _update_display(self):
        """Render the cable state, grating samples, and spike"""
        img = np.zeros((128, 256, 3), dtype=np.uint8)
        h, w = 128, 256
        mid_y = 64
        
        # Normalize cable for display
        cable_max = max(np.max(np.abs(self.cable_state)), 0.01)
        cable_norm = self.cable_state / cable_max
        
        # Draw cable state (Cyan)
        for i in range(self.cable_length - 1):
            x1 = int((i / self.cable_length) * w)
            x2 = int(((i + 1) / self.cable_length) * w)
            y1 = int(mid_y - cable_norm[i] * 40)
            y2 = int(mid_y - cable_norm[i+1] * 40)
            y1 = np.clip(y1, 0, h-1)
            y2 = np.clip(y2, 0, h-1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # Draw scorched manifold (Dark Red)
        scorch_norm = self.manifold_scorch / (np.max(self.manifold_scorch) + 0.01)
        for i in range(self.cable_length - 1):
            x1 = int((i / self.cable_length) * w)
            x2 = int(((i + 1) / self.cable_length) * w)
            y1 = int(mid_y - scorch_norm[i] * 30)
            y2 = int(mid_y - scorch_norm[i+1] * 30)
            y1 = np.clip(y1, 0, h-1)
            y2 = np.clip(y2, 0, h-1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 150), 2)
        
        # Draw grating sample points (Green dots)
        for i, idx in enumerate(self.grating_idx):
            if idx < self.cable_length:
                x = int((idx / self.cable_length) * w)
                # Sample value at this position
                val = cable_norm[idx]
                y = int(mid_y - val * 40)
                y = np.clip(y, 0, h-1)
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        # Draw threshold gradient bar (bottom)
        for i, thresh in enumerate(self.thresholds):
            x = int((i / self.n_samples) * w)
            color = (0, int(255 * (1 - thresh)), int(255 * thresh))
            cv2.line(img, (x, h-10), (x, h-2), color, 2)
        
        # Draw spike position indicator
        if self.current_spike_pos >= 0:
            x = int((self.current_spike_pos / self.n_samples) * w)
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
            cv2.putText(img, "SPIKE", x-20, 15, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw resonance value
        cv2.putText(img, f"R:{self.current_resonance:.3f}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img, f"M:{self.membrane:.3f}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Flash white if spiking
        if self.current_spike > 0:
            cv2.rectangle(img, (0, 0), (w-1, h-1), (255, 255, 255), 2)
        
        self.display_image = img

    def get_output(self, port_name):
        if port_name == 'spike_out':
            return self.current_spike
        if port_name == 'spike_pos_out':
            if self.current_spike_pos >= 0:
                return self.current_spike_pos / self.n_samples
            return 0.0
        if port_name == 'resonance_out':
            return self.current_resonance
        if port_name == 'ephaptic_out':
            return self.ephaptic_buffer
        if port_name == 'manifold_img':
            return self.display_image
        return None

    def get_display_image(self):
        h, w = self.display_image.shape[:2]
        rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Cable Length", "cable_length", self.cable_length, None),
            ("RC Alpha", "rc_alpha", self.rc_alpha, "float"),
            ("Diffusion", "diffusion", self.diffusion, "float"),
            ("Templates", "n_templates", self.n_templates, None),
            ("Grating Period", "grating_period", self.grating_period, None),
            ("Noise Sigma", "johnson_nyquist_sigma", self.johnson_nyquist_sigma, "float"),
            ("Ephaptic Strength", "ephaptic_strength", self.ephaptic_strength, "float"),
        ]

    def set_config_options(self, options):
        if "cable_length" in options:
            self.cable_length = int(options["cable_length"])
            # Reinitialize arrays
            self.cable_state = np.zeros(self.cable_length, dtype=np.float32)
            self.manifold_scorch = np.zeros(self.cable_length, dtype=np.float32)
            self.grating_idx = np.arange(0, self.cable_length, self.grating_period)
            self.n_samples = len(self.grating_idx)
            self.thresholds = np.linspace(self.threshold_min, self.threshold_max, self.n_samples)
        if "rc_alpha" in options:
            self.rc_alpha = float(options["rc_alpha"])
        if "diffusion" in options:
            self.diffusion = float(options["diffusion"])
        if "n_templates" in options:
            self.n_templates = int(options["n_templates"])
            # Reinitialize templates
            self.templates = np.zeros((self.n_templates, self.n_samples), dtype=np.float32)
            for i in range(self.n_templates):
                freq = 0.05 + (i / self.n_templates) * 0.5
                for j, pos in enumerate(self.grating_idx):
                    self.templates[i, j] = np.sin(2 * np.pi * freq * pos / self.cable_length)
                norm = np.linalg.norm(self.templates[i])
                if norm > 0:
                    self.templates[i] /= norm
            self.template_scorch = np.zeros(self.n_templates, dtype=np.float32)
        if "grating_period" in options:
            self.grating_period = int(options["grating_period"])
            self.grating_idx = np.arange(0, self.cable_length, self.grating_period)
            self.n_samples = len(self.grating_idx)
        if "johnson_nyquist_sigma" in options:
            self.johnson_nyquist_sigma = float(options["johnson_nyquist_sigma"])
        if "ephaptic_strength" in options:
            self.ephaptic_strength = float(options["ephaptic_strength"])