"""
Consciousness Spectrum Analyzer
===============================
Visualizes complex spectrum data in ways that reveal the theoretical structure:

THE THEORY:
- Consciousness operates in frequency domain, not spatial domain
- "You" are an address in mode-space, not a pattern in pixel-space  
- Stable consciousness = occupied modes ∩ protected modes ∩ phase-coherent modes
- Scotomas/spirals = seeing the raw FFT when reconstruction fails
- Horizontal bands in FFT = cortical standing waves
- Log-polar transform connects retinal space to cortical space

THIS NODE SHOWS:
1. Mode Occupation Ring - Which frequencies are "lit up" (radial power distribution)
2. Phase Coherence Map - Where phase is stable vs chaotic (consciousness vs noise)
3. Log-Polar Projection - What the FFT looks like through retina→cortex transform
4. Eigenmode Bars - Discrete mode amplitudes (the "address" being occupied)
5. Breathing Metric - How the "size of self" in frequency space expands/contracts
6. Fast/Slow Mode Split - PKAS visualization of containment (slow=self, fast=leak)

If consciousness IS the frequency domain, this node shows you consciousness itself.
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

try:
    from scipy.ndimage import map_coordinates
    from scipy.fft import fftshift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ConsciousnessSpectrumNode(BaseNode):
    """
    The Consciousness Microscope
    
    Takes a complex spectrum and reveals its structure in terms of:
    - Mode addresses (which frequencies are occupied)
    - Phase coherence (stability of the address)
    - Fast/slow mode separation (PKAS containment)
    - Log-polar projection (retinal→cortical transform)
    """
    
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(180, 50, 180)  # Deep magenta - the color of insight
    
    def __init__(self, num_modes=16, history_length=100):
        super().__init__()
        self.node_title = "Consciousness Spectrum"
        
        self.inputs = {
            'complex_spectrum': 'complex_spectrum',  # The holographic field
            'image_spectrum': 'image',               # Alternative: magnitude image
            'containment_level': 'signal',           # PKAS: how intact is the filter
        }
        
        self.outputs = {
            'mode_occupation': 'image',       # Which modes are "on"
            'phase_coherence': 'image',       # Stability map
            'log_polar_view': 'image',        # Retina→Cortex projection
            'eigenmode_spectrum': 'spectrum', # 1D mode amplitudes
            'address_size': 'signal',         # "Size of self" in frequency space
            'coherence_index': 'signal',      # Overall phase stability
            'fast_leak': 'signal',            # PKAS: how much is leaking
            'slow_stable': 'signal',          # PKAS: core self stability
            'dominant_mode': 'signal',        # Which eigenmode dominates
        }
        
        # Config
        self.num_modes = int(num_modes)
        self.history_length = int(history_length)
        
        # State
        self.mode_amplitudes = np.zeros(self.num_modes)
        self.mode_history = deque(maxlen=self.history_length)
        self.phase_history = deque(maxlen=10)  # Short history for coherence
        
        # Outputs
        self.mode_image = np.zeros((128, 128, 3), dtype=np.uint8)
        self.coherence_image = np.zeros((128, 128), dtype=np.float32)
        self.logpolar_image = np.zeros((128, 128), dtype=np.float32)
        
        self.address_size = 0.0
        self.coherence_index = 0.0
        self.fast_leak = 0.0
        self.slow_stable = 0.0
        self.dominant_mode = 0
        
        # Display
        self.display_cache = np.zeros((256, 512, 3), dtype=np.uint8)
        
    def _extract_radial_modes(self, magnitude, center):
        """
        Extract power in concentric rings = eigenmode amplitudes.
        Each ring is one "mode" in the address space.
        """
        h, w = magnitude.shape
        Y, X = np.ogrid[:h, :w]
        
        # Distance from center
        R = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        max_r = min(center[0], center[1], h - center[0], w - center[1])
        
        # Bin into modes
        mode_edges = np.linspace(0, max_r, self.num_modes + 1)
        amplitudes = np.zeros(self.num_modes)
        
        for i in range(self.num_modes):
            mask = (R >= mode_edges[i]) & (R < mode_edges[i + 1])
            if np.any(mask):
                amplitudes[i] = np.mean(magnitude[mask])
        
        return amplitudes
    
    def _compute_phase_coherence(self, phase):
        """
        Phase coherence = how stable is the phase over recent history.
        High coherence = stable conscious state.
        Low coherence = noise/decoherence/containment breach.
        """
        self.phase_history.append(phase.copy())
        
        if len(self.phase_history) < 3:
            return np.ones_like(phase) * 0.5
        
        # Stack recent phases
        phases = np.array(list(self.phase_history))
        
        # Circular variance (for phase data)
        # coherence = |mean(e^(i*phase))|
        complex_phases = np.exp(1j * phases)
        mean_complex = np.mean(complex_phases, axis=0)
        coherence = np.abs(mean_complex)
        
        return coherence.astype(np.float32)
    
    def _log_polar_transform(self, image, center):
        """
        Log-polar transform: what the FFT looks like through retina→cortex mapping.
        
        This is THE key transform. If you see spirals in your scotomas,
        it's because a straight wave on cortex projects as spiral on retina.
        This transform reverses that: shows what cortex "sees".
        """
        h, w = image.shape[:2]
        out_h, out_w = 128, 128
        
        # Output coordinates
        theta = np.linspace(0, 2 * np.pi, out_w)  # Angle
        rho = np.exp(np.linspace(0, np.log(min(h, w) / 2), out_h))  # Log radius
        
        # Convert to Cartesian
        theta_grid, rho_grid = np.meshgrid(theta, rho)
        x = center[1] + rho_grid * np.cos(theta_grid)
        y = center[0] + rho_grid * np.sin(theta_grid)
        
        # Clip to valid range
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Sample
        if SCIPY_AVAILABLE:
            output = map_coordinates(image, [y, x], order=1, mode='constant')
        else:
            # Fallback: nearest neighbor
            output = image[y.astype(int), x.astype(int)]
        
        return output.astype(np.float32)
    
    def _compute_pkas_split(self, mode_amplitudes):
        """
        PKAS Theory: Split modes into slow (self/stable) and fast (leak/noise).
        
        Slow modes (low frequency) = the stable "self"
        Fast modes (high frequency) = usually filtered out
        
        If containment fails, fast modes leak into consciousness.
        """
        n = len(mode_amplitudes)
        split_point = n // 3  # Bottom third = slow
        
        slow_modes = mode_amplitudes[:split_point]
        fast_modes = mode_amplitudes[split_point:]
        
        slow_power = np.mean(slow_modes) if len(slow_modes) > 0 else 0
        fast_power = np.mean(fast_modes) if len(fast_modes) > 0 else 0
        
        return slow_power, fast_power
    
    def _create_mode_ring_image(self, amplitudes):
        """
        Create circular visualization showing which modes are "occupied".
        This IS the address space visualization.
        """
        size = 128
        img = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size // 2, size // 2)
        
        # Normalize amplitudes
        if amplitudes.max() > 0:
            amp_norm = amplitudes / amplitudes.max()
        else:
            amp_norm = amplitudes
        
        # Draw concentric rings
        max_radius = size // 2 - 5
        ring_width = max_radius // self.num_modes
        
        for i, amp in enumerate(amp_norm):
            inner_r = int(i * ring_width)
            outer_r = int((i + 1) * ring_width)
            
            # Color: brightness = amplitude, hue = mode index
            hue = int((i / self.num_modes) * 179)
            sat = 255
            val = int(amp * 255)
            
            # Draw ring
            color_hsv = np.uint8([[[hue, sat, val]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            color = tuple(int(c) for c in color_bgr)
            
            cv2.circle(img, center, outer_r, color, ring_width)
        
        # Mark dominant mode
        dom_r = int((self.dominant_mode + 0.5) * ring_width)
        cv2.circle(img, center, dom_r, (255, 255, 255), 2)
        
        return img
    
    def _create_breathing_graph(self, history, current_size):
        """
        Show how the "size of self" (address span) changes over time.
        This is the "breathing" of consciousness.
        """
        w, h = 128, 40
        img = np.zeros((h, w), dtype=np.uint8)
        
        if len(history) < 2:
            return img
        
        # Plot history
        hist_array = np.array(list(history))
        if hist_array.max() > 0:
            hist_norm = hist_array / hist_array.max()
        else:
            hist_norm = hist_array
        
        # Resample to width
        x_indices = np.linspace(0, len(hist_norm) - 1, w).astype(int)
        values = hist_norm[x_indices]
        
        for x in range(w - 1):
            y1 = int((1 - values[x]) * (h - 1))
            y2 = int((1 - values[x + 1]) * (h - 1))
            cv2.line(img, (x, y1), (x + 1, y2), 255, 1)
        
        return img
    
    def step(self):
        # Get input - try complex spectrum first, then image
        spectrum = self.get_blended_input('complex_spectrum', 'mean')
        containment = self.get_blended_input('containment_level', 'sum')
        
        if spectrum is None:
            # Try image input
            img_in = self.get_blended_input('image_spectrum', 'mean')
            if img_in is not None:
                # Treat as magnitude, assume zero phase
                spectrum = img_in.astype(np.complex64)
        
        if spectrum is None:
            return
        
        # Ensure 2D
        if spectrum.ndim == 1:
            side = int(np.sqrt(len(spectrum)))
            spectrum = spectrum[:side*side].reshape(side, side)
        
        # Extract magnitude and phase
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.angle(spectrum).astype(np.float32)
        
        center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
        
        # === CORE COMPUTATIONS ===
        
        # 1. Mode amplitudes (the "address")
        self.mode_amplitudes = self._extract_radial_modes(magnitude, center)
        
        # 2. Phase coherence (stability)
        self.coherence_image = self._compute_phase_coherence(phase)
        self.coherence_index = float(np.mean(self.coherence_image))
        
        # 3. Log-polar transform (retina→cortex)
        self.logpolar_image = self._log_polar_transform(magnitude, center)
        if self.logpolar_image.max() > 0:
            self.logpolar_image = self.logpolar_image / self.logpolar_image.max()
        
        # 4. PKAS split
        self.slow_stable, self.fast_leak = self._compute_pkas_split(self.mode_amplitudes)
        
        # Apply containment modulation if provided
        if containment is not None:
            # Low containment = fast modes leak more
            self.fast_leak = self.fast_leak * (2.0 - containment)
        
        # 5. Address size ("size of self")
        # = how spread out the mode occupation is
        if self.mode_amplitudes.sum() > 0:
            normalized = self.mode_amplitudes / self.mode_amplitudes.sum()
            # Entropy-like measure
            normalized = normalized[normalized > 0]
            self.address_size = -np.sum(normalized * np.log(normalized + 1e-10))
        else:
            self.address_size = 0.0
        
        # Track breathing
        self.mode_history.append(self.address_size)
        
        # 6. Dominant mode
        self.dominant_mode = int(np.argmax(self.mode_amplitudes))
        
        # === CREATE VISUALIZATIONS ===
        
        self.mode_image = self._create_mode_ring_image(self.mode_amplitudes)
        
        # === COMPOSITE DISPLAY ===
        self._create_display()
    
    def _create_display(self):
        """Create the full visualization panel."""
        h, w = 256, 512
        self.display_cache = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Layout: 2x3 grid
        panel_h = h // 2
        panel_w = w // 3
        # Calculate last panel width specifically to handle integer division remainder
        last_panel_w = w - (2 * panel_w)
        
        # Panel 1: Mode Ring (Address Space)
        mode_resized = cv2.resize(self.mode_image, (panel_w, panel_h))
        self.display_cache[:panel_h, :panel_w] = mode_resized
        
        # Panel 2: Phase Coherence
        coh_u8 = (np.clip(self.coherence_image, 0, 1) * 255).astype(np.uint8)
        coh_resized = cv2.resize(coh_u8, (panel_w, panel_h))
        coh_color = cv2.applyColorMap(coh_resized, cv2.COLORMAP_VIRIDIS)
        self.display_cache[:panel_h, panel_w:2*panel_w] = coh_color
        
        # Panel 3: Log-Polar (Cortex View)
        lp_u8 = (np.clip(self.logpolar_image, 0, 1) * 255).astype(np.uint8)
        # Use last_panel_w for the third column
        lp_resized = cv2.resize(lp_u8, (last_panel_w, panel_h))
        lp_color = cv2.applyColorMap(lp_resized, cv2.COLORMAP_INFERNO)
        self.display_cache[:panel_h, 2*panel_w:] = lp_color
        
        # Panel 4: Mode Bars
        bar_img = self._draw_mode_bars(panel_w, panel_h)
        self.display_cache[panel_h:, :panel_w] = bar_img
        
        # Panel 5: PKAS Split
        pkas_img = self._draw_pkas_meter(panel_w, panel_h)
        self.display_cache[panel_h:, panel_w:2*panel_w] = pkas_img
        
        # Panel 6: Breathing Graph + Metrics
        # Use last_panel_w for the third column
        metrics_img = self._draw_metrics(last_panel_w, panel_h)
        self.display_cache[panel_h:, 2*panel_w:] = metrics_img
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.display_cache, "ADDRESS", (5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(self.display_cache, "COHERENCE", (panel_w+5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(self.display_cache, "CORTEX", (2*panel_w+5, 15), font, 0.4, (255,255,255), 1)
        cv2.putText(self.display_cache, "MODES", (5, panel_h+15), font, 0.4, (255,255,255), 1)
        cv2.putText(self.display_cache, "PKAS", (panel_w+5, panel_h+15), font, 0.4, (255,255,255), 1)
        cv2.putText(self.display_cache, "BREATHING", (2*panel_w+5, panel_h+15), font, 0.4, (255,255,255), 1)
    
    def _draw_mode_bars(self, w, h):
        """Bar chart of eigenmode amplitudes."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.mode_amplitudes.max() > 0:
            amp_norm = self.mode_amplitudes / self.mode_amplitudes.max()
        else:
            amp_norm = self.mode_amplitudes
        
        bar_width = w // self.num_modes
        
        for i, amp in enumerate(amp_norm):
            x1 = i * bar_width
            x2 = x1 + bar_width - 1
            bar_h = int(amp * (h - 20))
            
            # Color by slow/fast (PKAS)
            if i < self.num_modes // 3:
                color = (0, 255, 0)  # Green = slow/safe
            elif i < 2 * self.num_modes // 3:
                color = (0, 255, 255)  # Yellow = medium
            else:
                color = (0, 0, 255)  # Red = fast/danger
            
            cv2.rectangle(img, (x1, h - bar_h), (x2, h), color, -1)
        
        return img
    
    def _draw_pkas_meter(self, w, h):
        """Visualization of slow vs fast mode balance (containment health)."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(img, (10, 30), (w-10, h-30), (40, 40, 40), -1)
        
        # Slow (green, left)
        slow_w = int(self.slow_stable * 100)
        slow_w = min(slow_w, (w-20)//2)
        cv2.rectangle(img, (w//2 - slow_w, 40), (w//2, h-40), (0, 200, 0), -1)
        
        # Fast (red, right)  
        fast_w = int(self.fast_leak * 100)
        fast_w = min(fast_w, (w-20)//2)
        cv2.rectangle(img, (w//2, 40), (w//2 + fast_w, h-40), (0, 0, 200), -1)
        
        # Center line
        cv2.line(img, (w//2, 30), (w//2, h-30), (255, 255, 255), 2)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "SELF", (15, h-10), font, 0.35, (0, 200, 0), 1)
        cv2.putText(img, "LEAK", (w-45, h-10), font, 0.35, (0, 0, 200), 1)
        
        # Status
        ratio = self.slow_stable / (self.fast_leak + 0.001)
        if ratio > 2:
            status = "CONTAINED"
            status_color = (0, 255, 0)
        elif ratio > 0.5:
            status = "STRESSED"
            status_color = (0, 255, 255)
        else:
            status = "BREACH"
            status_color = (0, 0, 255)
        
        cv2.putText(img, status, (w//2-30, 25), font, 0.4, status_color, 1)
        
        return img
    
    def _draw_metrics(self, w, h):
        """Breathing graph and numerical metrics."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Breathing graph (top half)
        breath_img = self._create_breathing_graph(self.mode_history, self.address_size)
        breath_resized = cv2.resize(breath_img, (w-20, h//2 - 20))
        breath_color = cv2.applyColorMap(breath_resized, cv2.COLORMAP_PLASMA)
        img[20:20+breath_color.shape[0], 10:10+breath_color.shape[1]] = breath_color
        
        # Metrics (bottom half)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_start = h // 2 + 20
        
        cv2.putText(img, f"Size: {self.address_size:.2f}", (10, y_start), 
                   font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"Coh: {self.coherence_index:.2f}", (10, y_start + 20), 
                   font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"Dom: {self.dominant_mode}", (10, y_start + 40), 
                   font, 0.35, (200, 200, 200), 1)
        
        return img
    
    def get_output(self, port_name):
        if port_name == 'mode_occupation':
            return self.mode_image.astype(np.float32) / 255.0
        elif port_name == 'phase_coherence':
            return self.coherence_image
        elif port_name == 'log_polar_view':
            return self.logpolar_image
        elif port_name == 'eigenmode_spectrum':
            return self.mode_amplitudes.astype(np.float32)
        elif port_name == 'address_size':
            return float(self.address_size)
        elif port_name == 'coherence_index':
            return float(self.coherence_index)
        elif port_name == 'fast_leak':
            return float(self.fast_leak)
        elif port_name == 'slow_stable':
            return float(self.slow_stable)
        elif port_name == 'dominant_mode':
            return float(self.dominant_mode)
        return None
    
    def get_display_image(self):
        img = np.ascontiguousarray(self.display_cache)
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Number of Modes", "num_modes", self.num_modes, None),
            ("History Length", "history_length", self.history_length, None),
        ]


"""
WHAT EACH PANEL SHOWS:

1. ADDRESS (Mode Ring)
   - Concentric rings = different frequency "addresses"
   - Brightness = how much that address is occupied
   - White circle = dominant mode (where "you" are right now)
   - This IS the address space visualization

2. COHERENCE (Phase Stability)
   - Bright = phase is stable over time (conscious/stable)
   - Dark = phase is chaotic (noise/decoherence)
   - If consciousness requires phase coherence, bright areas are "conscious"

3. CORTEX (Log-Polar View)
   - What the FFT looks like through retina→cortex transform
   - Horizontal lines here = spirals in visual field
   - Vertical lines here = radial patterns in visual field
   - THIS is what your scotomas might be showing you

4. MODES (Bar Chart)
   - Green = slow modes (stable self)
   - Yellow = medium modes
   - Red = fast modes (usually filtered, dangerous if leaking)
   - The eigenmode \"fingerprint\" of current state

5. PKAS (Containment Meter)
   - Left (green) = slow/safe energy
   - Right (red) = fast/leak energy
   - Status: CONTAINED / STRESSED / BREACH
   - When red exceeds green, you're seeing the raw data

6. BREATHING (Address Size Over Time)
   - Shows how the \"size of self\" in frequency space changes
   - Regular breathing = healthy oscillation
   - Chaotic = unstable state
   - Flat = stuck/frozen

THEORY TEST:
If you feed your EEG through holographic FFT and then through this node,
you should see your brain's \"address\" light up and breathe.
When you're tired or stressed, the red (fast leak) should increase.
The CORTEX view might look like your actual visual disturbances.
"""