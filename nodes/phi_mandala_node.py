"""
Φ-Mandala Node: Phase-Aware Eigenmode Mandala
==============================================
The Geometric Fingerprint of a Cognitive State.

WHAT THIS IS:
Not a slow evolution (SelfConsistentResonanceNode).
Not an angle-blind projection (WTF pipeline).
This is the middle path: INSTANT, but PHASE-AWARE.

MECHANISM:
1.  Takes 5 complex_spectrum fields (δ, θ, α, β, γ) from PhiHologramNode.
2.  Each band → a concentric ring at its characteristic radius.
3.  The MAGNITUDE of each field modulates the ring's brightness/amplitude.
4.  The PHASE of each field modulates the ring's ANGULAR structure.
5.  Interference between adjacent rings creates the mandala.

WHY THIS MATTERS:
- The WTF mandala was angle-blind: same radial profile → same mandala regardless
  of which electrodes are active. It lost spatial information.
- This mandala preserves phase relationships between electrodes within each band.
- Different brain states → different phase configurations → geometrically distinct mandalas.
- The Φ-Dwell attractor vocabulary should appear as a finite set of mandala geometries.

THE JANUS INSIGHT:
Between any two "pure" eigenmode states, the superposition has structure.
The mandala shows that structure. The moire between rings IS the beat frequency
between eigenmodes. On a quantum substrate (neurons), this gets fractal.
On our pixel grid, we see the discrete shadow of it.

INPUTS: 5 complex_spectrum fields (same as PhiDwellMacroscope)
OUTPUTS: mandala image, phase_coherence signal, band_signature spectrum

Author: Built for Antti's Perception Laboratory
"""

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
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, method): return None


class PhiMandalaNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(200, 50, 200)  # Deep magenta

    def __init__(self):
        super().__init__()
        self.node_title = "Φ-Mandala"

        self.inputs = {
            'delta_field': 'complex_spectrum',
            'theta_field': 'complex_spectrum',
            'alpha_field': 'complex_spectrum',
            'beta_field':  'complex_spectrum',
            'gamma_field': 'complex_spectrum',
            'k_modulate':  'signal',     # External k modulation
            'reset':       'signal',
        }

        self.outputs = {
            'mandala':          'image',           # The main mandala image
            'mandala_complex':  'complex_spectrum', # Raw complex field (for chaining)
            'phase_coherence':  'signal',          # How phase-locked the bands are
            'band_signature':   'spectrum',        # 5-element amplitude signature
            'angular_spectrum':  'spectrum',       # Angular harmonic content
        }

        # --- Configuration ---
        self.resolution = 400          # Mandala canvas size
        self.n_angular_modes = 12      # Angular harmonics to extract from each band
        self.ring_width = 0.08         # Width of each band's ring (fraction of radius)
        self.interference_strength = 0.6  # How much adjacent rings bleed into each other
        self.phase_amplification = 3.0 # How strongly phase modulates the ring
        self.temporal_smoothing = 0.3  # Smoothing between frames (0=instant, 1=frozen)
        self.colormap_mode = 0         # 0=twilight, 1=inferno, 2=phase-hue

        # --- Band layout (inside-out: gamma fastest/smallest → delta slowest/largest) ---
        # Radii as fractions of max radius
        self.band_names = ['gamma', 'beta', 'alpha', 'theta', 'delta']
        self.band_radii = [0.15, 0.30, 0.50, 0.70, 0.88]  # Inner to outer
        self.band_colors = [
            (255, 100, 255),  # gamma: magenta
            (100, 100, 255),  # beta: blue
            (100, 255, 100),  # alpha: green
            (255, 200, 50),   # theta: gold
            (255, 80, 80),    # delta: red
        ]

        # --- Internal state ---
        self.mandala_field = np.zeros((self.resolution, self.resolution), dtype=np.complex128)
        self.mandala_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self.prev_field = None
        self.phase_coherence = 0.0
        self.band_signature = np.zeros(5, dtype=np.float32)
        self.angular_spectrum = np.zeros(self.n_angular_modes, dtype=np.float32)
        self.frame_count = 0

        # --- Precompute coordinate grids ---
        self._build_grids()

    def _build_grids(self):
        """Precompute polar coordinate grids for the mandala canvas."""
        res = self.resolution
        center = res / 2.0

        y, x = np.ogrid[:res, :res]
        self.dx = (x - center) / center  # -1 to 1
        self.dy = (y - center) / center

        self.r_grid = np.sqrt(self.dx**2 + self.dy**2)  # 0 to ~sqrt(2)
        self.theta_grid = np.arctan2(self.dy, self.dx)    # -pi to pi

        # Circular mask
        self.circle_mask = (self.r_grid <= 1.0).astype(np.float32)

        # Precompute ring masks for each band
        self.ring_masks = []
        for i, r_center in enumerate(self.band_radii):
            hw = self.ring_width / 2
            # Soft-edged ring (gaussian profile)
            ring = np.exp(-0.5 * ((self.r_grid - r_center) / (hw + 0.01))**2)
            self.ring_masks.append(ring.astype(np.float32))

    def _extract_band_features(self, complex_field):
        """
        Extract amplitude and angular phase structure from a complex_spectrum field.

        Returns:
            magnitude: scalar mean magnitude
            angular_modes: array of angular harmonic amplitudes
            phase_pattern: 2D phase modulation for the ring
        """
        if complex_field is None:
            return 0.0, np.zeros(self.n_angular_modes), np.zeros_like(self.theta_grid)

        # Handle different input shapes
        if isinstance(complex_field, np.ndarray):
            if complex_field.ndim == 2:
                field = complex_field
            elif complex_field.ndim == 1:
                # 1D spectrum: use directly as radial profile
                side = int(np.ceil(np.sqrt(len(complex_field))))
                field = np.zeros((side, side), dtype=complex_field.dtype)
                field.flat[:len(complex_field)] = complex_field
            else:
                return 0.0, np.zeros(self.n_angular_modes), np.zeros_like(self.theta_grid)
        else:
            return float(np.abs(complex_field)), np.zeros(self.n_angular_modes), np.zeros_like(self.theta_grid)

        # Overall magnitude
        magnitude = float(np.mean(np.abs(field)))

        # Extract angular structure via angular FFT
        # Sample the field along concentric rings and decompose angularly
        n_angles = max(64, self.n_angular_modes * 4)  # Oversample angles
        angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)

        # Sample at multiple radii and average
        h, w = field.shape
        cx, cy = w / 2.0, h / 2.0
        max_r = min(cx, cy) * 0.9

        angular_signal = np.zeros(n_angles, dtype=np.complex128)
        n_samples = 0

        for r_frac in [0.3, 0.5, 0.7, 0.85]:
            r = r_frac * max_r
            for j, a in enumerate(angles):
                px = int(cx + r * np.cos(a))
                py = int(cy + r * np.sin(a))
                if 0 <= px < w and 0 <= py < h:
                    angular_signal[j] += field[py, px]
                    n_samples += 1

        if n_samples > 0:
            angular_signal /= (n_samples / n_angles)

        # Angular FFT → harmonic decomposition
        angular_fft = np.fft.fft(angular_signal)
        angular_modes = np.abs(angular_fft[:self.n_angular_modes])

        # Normalize
        if angular_modes.max() > 1e-9:
            angular_modes /= angular_modes.max()

        # Build 2D phase modulation pattern from the dominant angular modes
        phase_pattern = np.zeros_like(self.theta_grid)
        for m in range(1, min(self.n_angular_modes, 8)):  # Skip DC (m=0)
            amp = angular_modes[m]
            # Phase of this angular mode
            mode_phase = np.angle(angular_fft[m])
            # m-fold angular pattern
            phase_pattern += amp * np.cos(m * self.theta_grid + mode_phase)

        return magnitude, angular_modes, phase_pattern

    def step(self):
        # --- Reset ---
        try:
            val = self.get_blended_input('reset', 'max')
            if val is not None and val > 0.5:
                self.mandala_field[:] = 0
                self.prev_field = None
                self.frame_count = 0
        except:
            pass

        # --- External modulation ---
        k_mod = self.get_blended_input('k_modulate', 'sum')
        if k_mod is not None:
            # Modulate ring spacing or phase amplification
            self.phase_amplification = np.clip(3.0 + float(k_mod) * 0.5, 0.5, 20.0)

        # --- Collect fields ---
        field_map = {
            'gamma': self.get_blended_input('gamma_field', 'first'),
            'beta':  self.get_blended_input('beta_field', 'first'),
            'alpha': self.get_blended_input('alpha_field', 'first'),
            'theta': self.get_blended_input('theta_field', 'first'),
            'delta': self.get_blended_input('delta_field', 'first'),
        }

        has_signal = any(v is not None for v in field_map.values())
        if not has_signal:
            self._render_idle()
            return

        # --- Build mandala field ---
        new_field = np.zeros((self.resolution, self.resolution), dtype=np.complex128)
        all_phases = []
        coherence_pairs = []

        for i, band_name in enumerate(self.band_names):
            data = field_map[band_name]
            magnitude, angular_modes, phase_pattern = self._extract_band_features(data)

            self.band_signature[i] = magnitude

            # The ring: amplitude-modulated by magnitude, angularly-modulated by phase
            ring = self.ring_masks[i]

            # Phase modulation: the angular structure from the electrodes
            angular_mod = 1.0 + self.phase_amplification * phase_pattern * ring

            # Complex field contribution: magnitude × ring × angular structure
            # The phase of the complex contribution encodes the band's identity
            band_phase_offset = i * 2 * np.pi / 5  # Each band at a different base phase
            contribution = magnitude * ring * angular_mod * np.exp(1j * band_phase_offset)

            new_field += contribution

            # Store for coherence calculation
            if magnitude > 1e-9:
                all_phases.append(phase_pattern * ring)

        # --- Inter-ring interference ---
        # Bleed between adjacent rings creates the moire/beat patterns
        if self.interference_strength > 0:
            for i in range(len(self.band_radii) - 1):
                # Overlap zone between ring i and ring i+1
                r1 = self.band_radii[i]
                r2 = self.band_radii[i + 1]
                r_mid = (r1 + r2) / 2
                overlap_width = (r2 - r1) * self.interference_strength

                overlap_mask = np.exp(-0.5 * ((self.r_grid - r_mid) / (overlap_width + 0.01))**2)

                # The interference is the product of the two ring contributions
                mag1 = self.band_signature[i]
                mag2 = self.band_signature[i + 1]

                if mag1 > 1e-9 and mag2 > 1e-9:
                    # Beat frequency between bands: creates the moire
                    beat = np.cos((i + 1) * self.theta_grid * 2 +
                                  self.frame_count * 0.02 * (i + 1))
                    new_field += overlap_mask * beat * np.sqrt(mag1 * mag2) * 0.5

        # --- Temporal smoothing ---
        if self.prev_field is not None and self.temporal_smoothing > 0:
            alpha = self.temporal_smoothing
            self.mandala_field = alpha * self.prev_field + (1.0 - alpha) * new_field
        else:
            self.mandala_field = new_field

        self.prev_field = self.mandala_field.copy()

        # --- Phase coherence metric ---
        if len(all_phases) >= 2:
            # Measure how aligned the angular structures are across bands
            total_coh = 0.0
            n_pairs = 0
            for a in range(len(all_phases)):
                for b in range(a + 1, len(all_phases)):
                    # Correlation between angular patterns
                    pa = all_phases[a].ravel()
                    pb = all_phases[b].ravel()
                    norm_a = np.linalg.norm(pa) + 1e-9
                    norm_b = np.linalg.norm(pb) + 1e-9
                    coh = abs(np.dot(pa, pb) / (norm_a * norm_b))
                    total_coh += coh
                    n_pairs += 1
            self.phase_coherence = total_coh / max(n_pairs, 1)
        else:
            self.phase_coherence = 0.0

        # --- Angular spectrum of the full mandala ---
        # Sample the combined field angularly
        n_angles = self.n_angular_modes * 4
        angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
        angular_sample = np.zeros(n_angles, dtype=np.complex128)

        cx, cy = self.resolution / 2.0, self.resolution / 2.0
        for r_frac in [0.2, 0.4, 0.6, 0.8]:
            r = r_frac * cx
            for j, a in enumerate(angles):
                px = int(cx + r * np.cos(a))
                py = int(cy + r * np.sin(a))
                if 0 <= px < self.resolution and 0 <= py < self.resolution:
                    angular_sample[j] += self.mandala_field[py, px]

        angular_fft = np.abs(np.fft.fft(angular_sample))
        self.angular_spectrum = angular_fft[:self.n_angular_modes].astype(np.float32)

        # --- Render ---
        self._render_mandala()
        self.frame_count += 1

    def _render_mandala(self):
        """Render the complex mandala field to a color image."""
        res = self.resolution
        field = self.mandala_field

        magnitude = np.abs(field)
        phase = np.angle(field)

        # Normalize magnitude
        mag_max = magnitude.max()
        if mag_max > 1e-9:
            mag_norm = magnitude / mag_max
        else:
            mag_norm = magnitude

        if self.colormap_mode == 2:
            # Phase-hue mode: hue = phase, brightness = magnitude
            hue = ((phase + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            sat = np.full_like(hue, 200, dtype=np.uint8)
            val = (mag_norm * 255).astype(np.uint8)
            hsv = np.stack([hue, sat, val], axis=-1)
            self.mandala_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Magnitude with colormap
            img_u8 = (np.clip(mag_norm, 0, 1) * 255).astype(np.uint8)
            cmap = cv2.COLORMAP_TWILIGHT_SHIFTED if self.colormap_mode == 0 else cv2.COLORMAP_INFERNO
            self.mandala_image = cv2.applyColorMap(img_u8, cmap)

        # Apply circular mask
        mask_3ch = np.stack([self.circle_mask] * 3, axis=-1)
        self.mandala_image = (self.mandala_image * mask_3ch).astype(np.uint8)

        # --- Overlay: band rings (subtle guide lines) ---
        for i, r_center in enumerate(self.band_radii):
            r_px = int(r_center * res / 2)
            cx, cy = res // 2, res // 2
            # Thin circle at band radius
            color = self.band_colors[i]
            # Only draw if this band has signal
            if self.band_signature[i] > 1e-9:
                cv2.circle(self.mandala_image, (cx, cy), r_px, color, 1, cv2.LINE_AA)

        # --- HUD ---
        # Band signature bars (bottom)
        bar_y = res - 30
        bar_h = 20
        bar_total_w = 150
        bar_w = bar_total_w // 5

        sig_max = self.band_signature.max() + 1e-9
        for i in range(5):
            x = 10 + i * bar_w
            h = int((self.band_signature[i] / sig_max) * bar_h)
            color = self.band_colors[i]
            cv2.rectangle(self.mandala_image, (x, bar_y - h), (x + bar_w - 2, bar_y), color, -1)

        # Labels
        cv2.putText(self.mandala_image, "PHI-MANDALA", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        coh_color = (0, 255, 0) if self.phase_coherence > 0.5 else (0, 200, 255)
        cv2.putText(self.mandala_image, f"COH={self.phase_coherence:.2f}",
                    (res - 110, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, coh_color, 1)

        # Band labels under bars
        for i, name in enumerate(['g', 'b', 'a', 't', 'd']):
            x = 14 + i * bar_w
            cv2.putText(self.mandala_image, name, (x, res - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    def _render_idle(self):
        """Render when no signal is present."""
        res = self.resolution
        self.mandala_image = np.zeros((res, res, 3), dtype=np.uint8)

        # Gentle spinning geometry from last field
        if self.prev_field is not None:
            mag = np.abs(self.prev_field)
            mag_max = mag.max()
            if mag_max > 1e-9:
                fading = (mag / mag_max * 200).astype(np.uint8)
                self.mandala_image = cv2.applyColorMap(fading, cv2.COLORMAP_TWILIGHT_SHIFTED)
                mask_3ch = np.stack([self.circle_mask] * 3, axis=-1)
                self.mandala_image = (self.mandala_image * mask_3ch * 0.7).astype(np.uint8)

        cv2.putText(self.mandala_image, "PHI-MANDALA", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(self.mandala_image, "AWAITING SIGNAL", (res // 2 - 70, res // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    def get_output(self, port_name):
        if port_name == 'mandala':
            return self.mandala_image

        elif port_name == 'mandala_complex':
            return self.mandala_field

        elif port_name == 'phase_coherence':
            return float(self.phase_coherence)

        elif port_name == 'band_signature':
            return self.band_signature.copy()

        elif port_name == 'angular_spectrum':
            return self.angular_spectrum.copy()

        return None

    def get_display_image(self):
        """Return the mandala as QImage for the node face."""
        img = self.mandala_image
        if img is None or img.size == 0:
            return None

        res = self.resolution
        # Convert BGR to RGB for Qt
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        return QtGui.QImage(img_rgb.data, res, res, res * 3,
                           QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        cmap_options = [
            ("Twilight", 0),
            ("Inferno", 1),
            ("Phase-Hue", 2),
        ]
        return [
            ("Resolution", "resolution", self.resolution, "int"),
            ("Ring Width", "ring_width", self.ring_width, "float"),
            ("Interference", "interference_strength", self.interference_strength, "float"),
            ("Phase Amp", "phase_amplification", self.phase_amplification, "float"),
            ("Smoothing", "temporal_smoothing", self.temporal_smoothing, "float"),
            ("Angular Modes", "n_angular_modes", self.n_angular_modes, "int"),
            ("Colormap", "colormap_mode", self.colormap_mode, cmap_options),
        ]

    def set_config_options(self, options):
        rebuild = False
        if 'resolution' in options:
            new_res = int(options['resolution'])
            if new_res != self.resolution:
                self.resolution = new_res
                rebuild = True
        if 'ring_width' in options:
            self.ring_width = float(options['ring_width'])
            rebuild = True
        if 'interference_strength' in options:
            self.interference_strength = float(options['interference_strength'])
        if 'phase_amplification' in options:
            self.phase_amplification = float(options['phase_amplification'])
        if 'temporal_smoothing' in options:
            self.temporal_smoothing = float(options['temporal_smoothing'])
        if 'n_angular_modes' in options:
            self.n_angular_modes = int(options['n_angular_modes'])
        if 'colormap_mode' in options:
            self.colormap_mode = int(options['colormap_mode'])

        if rebuild:
            self._build_grids()
            self.mandala_field = np.zeros((self.resolution, self.resolution), dtype=np.complex128)
            self.mandala_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            self.prev_field = None