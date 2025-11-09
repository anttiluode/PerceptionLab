"""
CorticalReconstructionNode - Attempts to visualize "brain images" from EEG signals.
---------------------------------------------------------------------------------
This node takes raw EEG or specific frequency band powers and projects them
onto a 2D cortical map, synthesizing a visual representation (reconstructed qualia)
based on brain-inspired principles of spatial organization and dynamic attention.

Inspired by:
- How different frequencies (alpha, theta, gamma) correspond to spatial processing
  (Lobe Emergence node).
- Dynamic scanning and gating mechanisms in perception (Theta-Gamma Scanner node).
- The idea of a holographic/fractal memory map encoding visual information.
- The "signal-centric" view where temporal dynamics are crucial for representation.

This is a speculative node for exploring the *concept* of brain-to-image
reconstruction within the Perception Lab's framework.

Place this file in the 'nodes' folder
"""

import numpy as np
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
    print("Warning: CorticalReconstructionNode requires scipy")

class CorticalReconstructionNode(BaseNode):
    NODE_CATEGORY = "Visualization" # Or "Cognitive"
    NODE_COLOR = QtGui.QColor(100, 50, 200) # Deep Purple

    def __init__(self, output_size=128, decay_rate=0.95, alpha_influence=0.3, theta_influence=0.5, gamma_influence=0.8, noise_level=0.01):
        super().__init__()
        self.node_title = "Cortical Reconstruction"

        self.inputs = {
            'raw_eeg_signal': 'signal',   # Main EEG signal (e.g., raw_signal from EEG node)
            'alpha_power': 'signal',      # Alpha power (e.g., alpha from EEG node)
            'theta_power': 'signal',      # Theta power
            'gamma_power': 'signal',      # Gamma power
            'attention_focus': 'image',   # Optional: an image mask to bias reconstruction focus
        }

        self.outputs = {
            'reconstructed_image': 'image', # The synthesized "brain image"
            'alpha_contribution': 'image',  # Visualizing alpha's part
            'theta_contribution': 'image',  # Visualizing theta's part
            'gamma_contribution': 'image',  # Visualizing gamma's part
            'current_focus': 'image'        # Where the node is 'looking'
        }

        if not SCIPY_AVAILABLE or QtGui is None:
            self.node_title = "Cortical Reconstruction (ERROR)"
            self._error = True
            return
        self._error = False

        self.output_size = int(output_size)
        self.decay_rate = float(decay_rate)
        self.alpha_influence = float(alpha_influence) # Higher influence -> more visual output from this band
        self.theta_influence = float(theta_influence)
        self.gamma_influence = float(gamma_influence)
        self.noise_level = float(noise_level)

        # Internal 2D "mental canvas"
        self.reconstructed_image = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        
        # Initialize some simple spatial filters for each band
        # These are highly speculative and can be made more complex
        self.alpha_filter = self._create_spatial_filter(self.output_size, 'smooth')
        self.theta_filter = self._create_spatial_filter(self.output_size, 'directional')
        self.gamma_filter = self._create_spatial_filter(self.output_size, 'detail')

        self.alpha_map = np.zeros_like(self.reconstructed_image)
        self.theta_map = np.zeros_like(self.reconstructed_image)
        self.gamma_map = np.zeros_like(self.reconstructed_image)
        self.current_focus_map = np.zeros_like(self.reconstructed_image)

    def _create_spatial_filter(self, size, type):
        """Creates a speculative spatial pattern for EEG band influence."""
        filter_map = np.zeros((size, size), dtype=np.float32)
        if type == 'smooth':
            filter_map = gaussian_filter(np.random.rand(size, size), sigma=size/8)
        elif type == 'directional':
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            angle = np.random.uniform(0, 2 * np.pi)
            filter_map = np.cos(X * np.cos(angle) * np.pi * 5 + Y * np.sin(angle) * np.pi * 5)
            filter_map = (filter_map + 1) / 2 # Normalize to 0-1
        elif type == 'detail':
            filter_map = np.random.rand(size, size)
            filter_map = cv2.Canny((filter_map * 255).astype(np.uint8), 50, 150) / 255.0 # Edge detection
        return filter_map / (filter_map.max() + 1e-9) # Normalize

    def step(self):
        if self._error: return

        # 1. Get EEG band powers (normalized roughly)
        raw_eeg = self.get_blended_input('raw_eeg_signal', 'sum') or 0.0
        alpha_power = self.get_blended_input('alpha_power', 'sum') or 0.0
        theta_power = self.get_blended_input('theta_power', 'sum') or 0.0
        gamma_power = self.get_blended_input('gamma_power', 'sum') or 0.0
        
        attention_focus_in = self.get_blended_input('attention_focus', 'mean')
        
        # Basic normalization for input signals (adjust as needed for real EEG ranges)
        alpha_power = np.clip(alpha_power, 0, 1) # Assuming 0-1 range for simplicity
        theta_power = np.clip(theta_power, 0, 1)
        gamma_power = np.clip(gamma_power, 0, 1)
        raw_eeg_norm = np.clip(raw_eeg + 0.5, 0, 1) # Roughly center 0 and scale to 0-1

        # 2. Update internal "mental canvas" based on EEG bands
        
        # Alpha: Influences smooth, global background or overall brightness
        self.alpha_map = self.alpha_filter * alpha_power * self.alpha_influence
        
        # Theta: Influences dynamic, directional elements or larger structures
        # We can make theta shift the filter dynamically based on raw_eeg
        # (This is a simplified way to model theta's role in "scanning" and memory)
        theta_shift_x = int((raw_eeg_norm - 0.5) * 10) # Raw EEG shifts the pattern
        theta_shifted_filter = np.roll(self.theta_filter, theta_shift_x, axis=1)
        self.theta_map = theta_shifted_filter * theta_power * self.theta_influence
        
        # Gamma: Influences fine details, edges, and sharp features
        self.gamma_map = self.gamma_filter * gamma_power * self.gamma_influence
        
        # Combine contributions
        current_reconstruction = (self.alpha_map + self.theta_map + self.gamma_map)
        
        # 3. Apply Attention Focus (if provided)
        if attention_focus_in is not None:
            if attention_focus_in.shape[0] != self.output_size:
                attention_focus_in = cv2.resize(attention_focus_in, (self.output_size, self.output_size))
            if attention_focus_in.ndim == 3:
                attention_focus_in = np.mean(attention_focus_in, axis=2)
            
            # Normalize attention mask
            attention_focus_in = attention_focus_in / (attention_focus_in.max() + 1e-9)
            self.current_focus_map = gaussian_filter(attention_focus_in, sigma=self.output_size / 20)
            
            # Only parts under focus are strongly reconstructed
            current_reconstruction *= (0.5 + 0.5 * self.current_focus_map) # Bias towards focused areas
        else:
            self.current_focus_map.fill(1.0) # Full attention if no input

        # Add some baseline noise for organic feel
        current_reconstruction += np.random.rand(self.output_size, self.output_size) * self.noise_level

        # Update the main reconstructed image with decay and new input
        self.reconstructed_image = self.reconstructed_image * self.decay_rate + current_reconstruction
        np.clip(self.reconstructed_image, 0, 1, out=self.reconstructed_image)
        
        # Apply a light gaussian blur for smoother "qualia"
        self.reconstructed_image = gaussian_filter(self.reconstructed_image, sigma=0.5)

    def get_output(self, port_name):
        if self._error: return None
        if port_name == 'reconstructed_image':
            return self.reconstructed_image
        elif port_name == 'alpha_contribution':
            return self.alpha_map
        elif port_name == 'theta_contribution':
            return self.theta_map
        elif port_name == 'gamma_contribution':
            return self.gamma_map
        elif port_name == 'current_focus':
            return self.current_focus_map
        return None

    def get_display_image(self):
        if self._error: return None
        
        display_w = 512
        display_h = 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Left side: Reconstructed Image
        reco_u8 = (np.clip(self.reconstructed_image, 0, 1) * 255).astype(np.uint8)
        reco_color = cv2.cvtColor(reco_u8, cv2.COLOR_GRAY2RGB)
        reco_resized = cv2.resize(reco_color, (display_h, display_h), interpolation=cv2.INTER_LINEAR)
        display[:, :display_h] = reco_resized
        
        # Right side: Band Contributions and Focus (blended for visualization)
        # Alpha: Green, Theta: Blue, Gamma: Red
        contributions_rgb = np.zeros((self.output_size, self.output_size, 3), dtype=np.float32)
        contributions_rgb[:, :, 0] = self.gamma_map # Red for Gamma (details)
        contributions_rgb[:, :, 1] = self.alpha_map # Green for Alpha (smoothness)
        contributions_rgb[:, :, 2] = self.theta_map # Blue for Theta (motion/structure)
        
        # Overlay focus map as an intensity
        focus_overlay = np.stack([self.current_focus_map]*3, axis=-1)
        contributions_rgb = (contributions_rgb * (0.5 + 0.5 * focus_overlay)) # Dim if not focused

        contr_u8 = (np.clip(contributions_rgb, 0, 1) * 255).astype(np.uint8)
        contr_resized = cv2.resize(contr_u8, (display_h, display_h), interpolation=cv2.INTER_LINEAR)
        display[:, display_w-display_h:] = contr_resized
        
        # Add dividing line
        display[:, display_h-1:display_h+1] = [255, 255, 255]
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'RECONSTRUCTED QUALIA', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'BAND CONTRIBUTIONS & FOCUS', (display_h + 10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add input values for context
        alpha_val = self.get_blended_input('alpha_power', 'sum') or 0.0
        theta_val = self.get_blended_input('theta_power', 'sum') or 0.0
        gamma_val = self.get_blended_input('gamma_power', 'sum') or 0.0

        cv2.putText(display, f"ALPHA: {alpha_val:.2f}", (10, display_h - 40), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(display, f"THETA: {theta_val:.2f}", (10, display_h - 25), font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(display, f"GAMMA: {gamma_val:.2f}", (10, display_h - 10), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA) # Changed to blue for theta, red for gamma

        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Output Size", "output_size", self.output_size, None),
            ("Decay Rate", "decay_rate", self.decay_rate, None),
            ("Alpha Influence", "alpha_influence", self.alpha_influence, None),
            ("Theta Influence", "theta_influence", self.theta_influence, None),
            ("Gamma Influence", "gamma_influence", self.gamma_influence, None),
            ("Noise Level", "noise_level", self.noise_level, None),
        ]