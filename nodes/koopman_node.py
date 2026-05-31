"""
Koopman Spectral Node (Eigenmode Visualizer)
--------------------------------------------
Reads the continuous latent vector (Ephaptic Field Spectrum) of the population.
Uses Spatio-Temporal Fourier analysis to isolate the dominant Koopman Eigenmodes.
Displays the network's thoughts as distinct "Spectral Islands" (Frequencies).
"""

import numpy as np
import cv2
from collections import deque
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class KoopmanSpectralNode(BaseNode):
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(180, 100, 255)  # Deep Koopman Purple

    def __init__(self, history_length=128):
        super().__init__()
        self.node_title = "Koopman Eigenmodes"
        
        self.inputs = {
            'latent_in': 'spectrum'  # Expects the 5D field from EphapticFieldNode
        }
        self.outputs = {
            'eigen_image': 'image',
            'dominant_freq': 'signal'
        }
        
        self.history_length = int(history_length)
        # We don't know the exact number of neurons yet, we adapt dynamically
        self.n_dims = 5 
        
        # Buffer to hold the spatial-temporal state
        self.state_history = deque(maxlen=self.history_length)
        
        self.display_image = np.zeros((128, 256, 3), dtype=np.uint8)
        self.current_dominant_freq = 0.0
        self.step_counter = 0

    def step(self):
        self.step_counter += 1
        
        # 1. Get the continuous latent vector (the Ephaptic Field)
        latent = self.get_blended_input('latent_in', 'first')
        
        if latent is not None:
            latent = np.array(latent).flatten()
            self.n_dims = len(latent)
            self.state_history.append(latent)
        else:
            # Feed zeros if no input
            self.state_history.append(np.zeros(self.n_dims))
            
        # 2. Compute the Koopman Modes (Every 5 steps to save CPU)
        if self.step_counter % 5 == 0 and len(self.state_history) == self.history_length:
            self._compute_koopman_modes()

    def _compute_koopman_modes(self):
        # Convert history to a 2D matrix: [Time, Spatial_Dims] (128 x 5)
        X = np.array(self.state_history)
        
        # Apply Hanning window to reduce FFT edge artifacts
        window = np.hanning(self.history_length)[:, np.newaxis]
        X_windowed = X * window
        
        # Perform 1D FFT over the TIME axis for every spatial dimension
        # This reveals the frequency content of each neuron's field contribution
        fft_result = np.fft.rfft(X_windowed, axis=0)
        
        # Get the power spectrum (Amplitude squared)
        power = np.abs(fft_result) ** 2
        
        # The Koopman Eigenmodes are the dominant frequencies shared across the population
        # We sum the power across all neurons to find the network-wide eigenmodes
        global_power = np.sum(power, axis=1)
        
        # Remove DC offset (0 Hz) to focus on the actual oscillations
        global_power[0] = 0 
        
        # Find the dominant Eigenmode
        if np.max(global_power) > 0:
            dom_idx = np.argmax(global_power)
            self.current_dominant_freq = float(dom_idx)
        
        # 3. Render the Spectral Islands
        self._update_display(power, global_power)

    def _update_display(self, spatial_power, global_power):
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        n_freqs = len(global_power)
        if n_freqs <= 1:
            return
            
        # Normalize for display
        max_power = np.max(global_power) + 1e-8
        norm_global = global_power / max_power
        
        # Width of each frequency bin on the screen
        bin_w = w / n_freqs
        
        # A. Draw the Global Koopman Eigenmodes (The white peaks in the background)
        pts = []
        for f in range(n_freqs):
            x = int(f * bin_w)
            y = int(h - (norm_global[f] * (h - 20)))
            pts.append((x, y))
            
        if len(pts) > 1:
            cv2.fillPoly(img, [np.array([(0, h)] + pts + [(w, h)], dtype=np.int32)], (30, 30, 40))
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), 1)

        # B. Draw the individual Spatial contributions (The colored islands)
        # Colors for up to 8 dimensions
        colors = [
            (0, 255, 255), (255, 100, 255), (255, 255, 0), 
            (100, 255, 100), (100, 150, 255), (255, 100, 100), 
            (100, 255, 200), (200, 100, 255)
        ]
        
        for dim in range(self.n_dims):
            dim_power = spatial_power[:, dim] / max_power
            color = colors[dim % len(colors)]
            
            for f in range(n_freqs):
                if dim_power[f] > 0.05:  # Only draw significant spectral presence
                    x = int(f * bin_w)
                    y = int(h - (dim_power[f] * (h - 20)))
                    # Draw overlapping circles to represent spectral "islands"
                    radius = int(dim_power[f] * 6) + 1
                    cv2.circle(img, (x, y), radius, color, -1)
                    
        # C. UI Text
        cv2.putText(img, "Koopman Eigenmodes", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        if self.current_dominant_freq > 0:
            x_dom = int(self.current_dominant_freq * bin_w)
            cv2.line(img, (x_dom, 20), (x_dom, h), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"Dom Mode: {self.current_dominant_freq:.1f}", (w - 110, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        self.display_image = img

    def get_output(self, port_name):
        if port_name == 'eigen_image':
            return self.display_image
        if port_name == 'dominant_freq':
            return self.current_dominant_freq
        return None

    def get_display_image(self):
        h, w = self.display_image.shape[:2]
        rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("History Length (Time Window)", "history_length", self.history_length, None)
        ]

    def set_config_options(self, options):
        if "history_length" in options:
            self.history_length = int(options["history_length"])
            self.state_history = deque(maxlen=self.history_length)