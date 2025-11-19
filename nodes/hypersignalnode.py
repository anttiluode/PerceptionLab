"""
Hyper-Signal Node (The Soul of Slider2) - FIXED & CONVENTION-COMPLIANT
----------------------------------------------------------------------
Ported from 'slider2.py' and now fully adheres to Perception Lab node conventions:
- No __dict__ hacks
- Outputs stored as proper instance variables (self.xxx_val pattern for signals, direct for arrays/images)
- get_output() returns correct types (float for signals, np.ndarray for spectrum/image)
- get_display_image() returns QImage (uint8 RGB) exactly like other nodes
- Clean, readable, and instantly works when dropped into ./nodes/
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class HyperSignalNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(255, 100, 100)  # Salmon/Pink

    def __init__(self, num_channels=16):
        super().__init__()
        self.node_title = "Hyper-Signal Generator"
        
        self.inputs = {
            'modulation': 'signal',      # 0–1 blend Perlin ↔ Quantum (low = more Perlin/flow, high = more Quantum/structure)
            'phase_shift': 'signal'       # Speed up / slow down / reverse time evolution
        }
        
        self.outputs = {
            'spectrum_out': 'spectrum',   # High-dimensional latent vector (the actual "genetic address")
            'phase_plot': 'image',         # Beautiful Slider2-style phase portrait
            'complexity': 'signal'         # Instantaneous complexity (std of vector)
        }
        
        self.num_channels = int(num_channels)
        self.t = 0.0
        self.history = []
        
        # Quantum oscillator state (this is the real "soul" from slider2)
        self.phases = np.random.rand(self.num_channels) * 2 * np.pi
        self.frequencies = np.random.rand(self.num_channels) * 0.09 + 0.01  # Slightly wider band for more interesting orbits

    # ------------------------------------------------------------------
    # Core noise generators
    # ------------------------------------------------------------------
    def _generate_quantum_noise(self, t):
        """The famous "divine luck" superposition from slider2"""
        signal = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            # Light coupling from next oscillator → emergent coherence
            coupling = np.sin(self.phases[(i + 1) % self.num_channels]) * 0.4
            self.phases[i] += self.frequencies[i] + coupling
            signal[i] = np.sin(self.phases[i] + t * 0.3)  # Extra slow global phase
        return signal

    def _generate_perlin_coherent(self, t):
        """Smooth, flowing, river-like coherent noise"""
        signal = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            oct1 = np.sin(t * (i + 1) * 0.11 + i * 0.5)
            oct2 = 0.5 * np.sin(t * (i + 1) * 0.27 + i * 1.3)
            oct3 = 0.25 * np.sin(t * (i + 1) * 0.61 + i *2.1)
            signal[i] = oct1 + oct2 + oct3
        return signal

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self):
        # --- Inputs ---
        mod = self.get_blended_input('modulation', 'sum')
        if mod is None:
            mod = 1.0
        mod = np.clip(mod, 0.0, 1.0)
        
        shift = self.get_blended_input('phase_shift', 'sum')
        if shift is None:
            shift = 0.0
        
        # --- Time evolution ---
        self.t += 0.08 + shift * 0.6  # Base speed + modulation
        
        # --- Generate & blend the two souls ---
        quantum = self._generate_quantum_noise(self.t)
        perlin  = self._generate_perlin_coherent(self.t)
        
        # mod = 0.0 → pure Perlin (calm, flowing)  
        # mod = 1.0 → pure Quantum (crisp, crystalline, "divine")
        vector = quantum * mod + perlin * (1.0 - mod)
        
        # Optional: normalize to ~[-1, 1] range (keeps VAE happy)
        if np.ptp(vector) > 0:
            vector = 2.0 * (vector - vector.min()) / np.ptp(vector) - 1.0
        
        # --- Phase portrait (the beautiful Slider2 visualization) ---
        if len(vector) >= 2:
            x, y = vector[0], vector[1]
        else:
            x = y = 0.0
            
        self.history.append((x, y))
        if len(self.history) > 300:  # Longer trail = more hypnotic
            self.history.pop(0)
        
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = (10, 10, 20)  # Deep space background
        
        if len(self.history) > 1:
            pts = []
            cx, cy = 128, 128
            scale = 90.0
            for px, py in self.history:
                pts.append([int(cx + px * scale), int(cy + py * scale)])
            pts = np.array(pts, np.int32)
            
            # Fade trail
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (int(0 + alpha*50), int(255 + alpha*100), int(200 + alpha*55))
                cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), color, 1, cv2.LINE_AA)
            
            # Bright head
            cv2.circle(img, tuple(pts[-1]), 6, (180, 255, 240), -1)
            cv2.circle(img, tuple(pts[-1]), 9, (100, 200, 255), 2)

        # --- Store outputs the proper Perception Lab way ---
        self.spectrum_out_val = vector.astype(np.float32)  # This is the latent "address"
        self.complexity_val = float(np.std(vector))
        self.phase_plot_val = (img.astype(np.float32) / 255.0)  # Float 0-1 for other nodes
        self.display_img = img  # uint8 for display

    # ------------------------------------------------------------------
    # Standard node interface
    # ------------------------------------------------------------------
    def get_output(self, port_name):
        if port_name == 'spectrum_out':
            return self.spectrum_out_val if hasattr(self, 'spectrum_out_val') else np.zeros(self.num_channels, np.float32)
        if port_name == 'complexity':
            return self.complexity_val if hasattr(self, 'complexity_val') else 0.0
        if port_name == 'phase_plot':
            return self.phase_plot_val if hasattr(self, 'phase_plot_val') else np.zeros((256,256,3), np.float32)
        return None

    def get_display_image(self):
        if hasattr(self, 'display_img'):
            img = self.display_img
            return QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format.Format_RGB888)
        return QtGui.QImage()

    def get_config_options(self):
        return [
            ("Num Channels", "num_channels", self.num_channels, None)
        ]