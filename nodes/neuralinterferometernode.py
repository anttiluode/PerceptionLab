"""
Neural Interferometer: From Spike Trains to Images
===================================================
The "How Many Neurons Does It Take To Build A World" Node.

CONCEPT:
Each EEG channel is treated as a single "neuron" — a 1D signal
with magnitude and phase. This node projects ALL channels simultaneously
as interfering wave patterns onto a shared 2D canvas, then runs
iterative phase recovery to extract the spatial structure that's
consistent with ALL the interference patterns at once.

THE EXPERIMENT:
- Vector Size = 1: Each channel contributes ONE circle. You see
  concentric rings — the minimum structure from pure magnitudes.
- Vector Size = 16: Each channel contributes 16 harmonics. The
  interference between channels creates texture.
- Vector Size = 256: Rich structure emerges. The "image" that was
  never in any single channel appears from their collective interference.

This is literally what the visual cortex does:
  Retinal spike trains → retinotopic projection → cortical interference → perception

BRAIN REGIONS:
  Select different regions to see how each part of the brain
  "builds its world" differently. Occipital (visual) should show
  the most structured patterns. Frontal may be more diffuse.

Author: Built for Antti's Perception Laboratory
"""

import numpy as np
import cv2
import os

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

try:
    import mne
    from scipy import signal as scipy_signal
    from scipy.ndimage import gaussian_filter
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("[NeuralInterferometer] Warning: mne/scipy required. pip install mne scipy")


# Standard 10-20 positions (normalized -1 to 1)
ELECTRODE_POS = {
    'FP1': (-0.3, 0.9), 'FP2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'FZ': (0, 0.6),
    'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'F1': (-0.2, 0.6), 'F2': (0.2, 0.6),
    'FC1': (-0.25, 0.3), 'FC2': (0.25, 0.3),
    'FT7': (-0.8, 0.3), 'FT8': (0.8, 0.3),
    'T3': (-0.9, 0.0), 'T7': (-0.9, 0.0),
    'C3': (-0.4, 0.0), 'C1': (-0.2, 0.0),
    'CZ': (0, 0.0), 'C2': (0.2, 0.0),
    'C4': (0.4, 0.0), 'T4': (0.9, 0.0), 'T8': (0.9, 0.0),
    'CP1': (-0.25, -0.25), 'CP2': (0.25, -0.25),
    'TP7': (-0.8, -0.3), 'TP8': (0.8, -0.3),
    'T5': (-0.7, -0.5), 'P7': (-0.7, -0.5),
    'P3': (-0.35, -0.5), 'P1': (-0.2, -0.5),
    'PZ': (0, -0.5), 'P2': (0.2, -0.5),
    'P4': (0.35, -0.5), 'T6': (0.7, -0.5), 'P8': (0.7, -0.5),
    'PO3': (-0.3, -0.7), 'POZ': (0, -0.7), 'PO4': (0.3, -0.7),
    'PO7': (-0.55, -0.7), 'PO8': (0.55, -0.7),
    'O1': (-0.3, -0.85), 'OZ': (0, -0.85), 'O2': (0.3, -0.85),
}

EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2'],
}

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50),
}


class NeuralInterferometerNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(50, 200, 180)  # Teal - interferometry

    def __init__(self):
        super().__init__()
        self.node_title = "Neural Interferometer"

        self.inputs = {
            'vector_size_mod': 'signal',   # Modulate vector size externally
            'k_mod':           'signal',   # Modulate spatial frequency
            'reset':           'signal',
        }

        self.outputs = {
            'interference':    'image',           # The main interference image
            'reconstruction':  'image',           # Phase-recovered reconstruction
            'channel_mandala': 'image',           # Per-channel ring overlay
            'n_channels':      'signal',          # How many channels active
            'complexity':      'signal',          # Measure of pattern complexity
            'interference_field': 'complex_spectrum',  # Raw complex field
        }

        # --- EEG Config ---
        self.edf_file_path = ""
        self.selected_region = "Occipital"
        self._last_path = ""
        self._last_region = ""

        # --- Interferometer Config ---
        self.vector_size = 32          # How many "harmonics" per channel (the knob!)
        self.canvas_size = 400         # Output resolution
        self.spatial_k = 8.0           # Base spatial frequency
        self.phase_recovery_iters = 8  # Gerchberg-Saxton iterations
        self.smoothing = 0.5           # Output smoothing
        self.use_electrode_positions = True  # Use spatial arrangement or just stack
        self.temporal_smoothing = 0.2  # Smoothing between frames
        self.band_mode = "broadband"   # Which frequency content: broadband, delta, theta, alpha, beta, gamma

        # --- EEG State ---
        self.raw = None
        self.sfreq = 160.0
        self.current_time = 0.0
        self.window_size = 0.5        # 500ms window
        self.channel_names = []
        self.channel_positions = []    # (x, y) normalized positions

        # --- Output State ---
        self.interference_image = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.reconstruction_image = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.channel_mandala_image = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.interference_field = None
        self.n_active = 0
        self.complexity = 0.0
        self.prev_field = None
        self.frame_count = 0

        # --- Precompute ---
        self._build_canvas_grid()

    def _build_canvas_grid(self):
        """Precompute coordinate grids for the canvas."""
        s = self.canvas_size
        center = s / 2.0
        y, x = np.ogrid[:s, :s]
        self.cx_grid = (x - center) / center   # -1 to 1
        self.cy_grid = (y - center) / center
        self.r_grid = np.sqrt(self.cx_grid**2 + self.cy_grid**2)
        self.theta_grid = np.arctan2(self.cy_grid, self.cx_grid)
        self.circle_mask = (self.r_grid <= 1.0).astype(np.float32)

    def _load_edf(self):
        """Load EDF and extract channels for selected region."""
        if not MNE_AVAILABLE:
            self.node_title = "Neural Interf. (MNE needed!)"
            return False

        if not self.edf_file_path or not os.path.exists(self.edf_file_path):
            self.node_title = "Neural Interf. (No file)"
            return False

        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose='error')
            
            # Clean channel names
            rename_map = {}
            for ch in raw.ch_names:
                clean = ch.strip().replace('.', '').upper()
                rename_map[ch] = clean
            raw.rename_channels(rename_map)
            
            self.sfreq = raw.info['sfreq']
            all_ch = raw.ch_names

            # Find channels for region - deduplicated
            if self.selected_region == "All":
                available = []
                seen_raw = set()
                for ch in all_ch:
                    if ch in ELECTRODE_POS and ch not in seen_raw:
                        available.append((ch, ch))
                        seen_raw.add(ch)
            else:
                region_channels = EEG_REGIONS[self.selected_region]
                available = []
                seen_raw = set()  # Track raw channel names to avoid duplicates
                for std_name in region_channels:
                    # Direct match first
                    if std_name in all_ch and std_name not in seen_raw:
                        available.append((std_name, std_name))
                        seen_raw.add(std_name)
                        continue
                    # Fuzzy match
                    for ch in all_ch:
                        if ch in seen_raw:
                            continue
                        if std_name in ch or ch in std_name:
                            available.append((ch, std_name))
                            seen_raw.add(ch)
                            break

            if not available:
                self.node_title = f"Neural Interf. (No {self.selected_region} ch)"
                self.raw = None
                return False

            # Pick only available channels using modern API
            pick_names = [ch for ch, _ in available]
            raw.pick(pick_names)
            self.raw = raw
            self.current_time = 0.0

            # Store channel info
            self.channel_names = [std for _, std in available]
            self.channel_positions = []
            for _, std in available:
                pos = ELECTRODE_POS.get(std, (0, 0))
                self.channel_positions.append(pos)

            self.n_active = len(self.channel_names)
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            self.node_title = f"Neural Interf. ({self.selected_region}: {self.n_active}ch)"
            return True

        except Exception as e:
            self.node_title = f"Neural Interf. (Error)"
            print(f"[NeuralInterferometer] Load error: {e}")
            import traceback
            traceback.print_exc()
            self.raw = None
            return False

    def _get_channel_vectors(self):
        """
        Extract a vector of length `vector_size` for each channel.
        Each vector IS the channel's "spike train" — its contribution
        to the interference pattern.
        """
        if self.raw is None:
            return None

        start_sample = int(self.current_time * self.sfreq)
        n_samples = int(self.window_size * self.sfreq)
        end_sample = start_sample + n_samples

        if end_sample >= self.raw.n_times:
            self.current_time = 0.0
            start_sample = 0
            end_sample = n_samples

        data, _ = self.raw[:, start_sample:end_sample]
        # data shape: (n_channels, n_samples)

        if data.size == 0:
            return None

        # Optional: band-pass filter
        if self.band_mode != "broadband" and self.band_mode in BANDS:
            lo, hi = BANDS[self.band_mode]
            nyq = self.sfreq / 2.0
            lo_n = max(0.5, lo) / nyq
            hi_n = min(hi, nyq - 1) / nyq
            try:
                b, a = scipy_signal.butter(3, [lo_n, hi_n], btype='band')
                for i in range(data.shape[0]):
                    data[i] = scipy_signal.filtfilt(b, a, data[i])
            except:
                pass

        vectors = []
        for i in range(data.shape[0]):
            ch_data = data[i]

            # Compute magnitude spectrum (FFT) of this channel's window
            spectrum = np.abs(np.fft.rfft(ch_data))

            # Resample to vector_size
            if len(spectrum) >= self.vector_size:
                # Downsample: take first vector_size bins (low frequencies)
                vec = spectrum[:self.vector_size]
            else:
                # Pad
                vec = np.zeros(self.vector_size, dtype=np.float64)
                vec[:len(spectrum)] = spectrum

            # Normalize per channel
            vmax = np.max(np.abs(vec))
            if vmax > 1e-15:
                vec = vec / vmax

            vectors.append(vec.astype(np.float32))

        # Advance time
        self.current_time += 1.0 / 30.0  # ~30fps

        return vectors

    def _project_channel(self, vector, position, canvas_shape):
        """
        Project one channel's vector as radial rings onto the canvas.
        Position offsets the center (electrode spatial arrangement).

        Returns a complex field contribution.
        """
        s = canvas_shape[0]
        center = s / 2.0

        if self.use_electrode_positions:
            # Offset center by electrode position (scaled)
            offset_scale = 0.15  # How much spatial arrangement matters
            cx = center + position[0] * center * offset_scale
            cy = center - position[1] * center * offset_scale  # Y flipped
        else:
            cx, cy = center, center

        # Distance from this channel's center
        y, x = np.ogrid[:s, :s]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Normalize radius
        max_r = center
        r_norm = r / max_r

        # Build the interference: each element of the vector becomes a ring
        field = np.zeros((s, s), dtype=np.complex128)

        for k, amplitude in enumerate(vector):
            if amplitude < 1e-6:
                continue

            # Ring frequency: higher index = higher spatial frequency
            freq = (k + 1) * self.spatial_k / self.vector_size

            # The wave: amplitude * cos(freq * r) as real part
            # Phase from the vector index gives the complex structure
            phase_offset = k * np.pi / max(self.vector_size, 1)
            wave = amplitude * np.exp(1j * (freq * r_norm * 2 * np.pi + phase_offset))

            field += wave

        return field

    def step(self):
        # --- Check for reload ---
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self._load_edf()

        # --- Reset ---
        try:
            val = self.get_blended_input('reset', 'max')
            if val is not None and val > 0.5:
                self.current_time = 0.0
                self.prev_field = None
        except:
            pass

        # --- External modulation ---
        vs_mod = self.get_blended_input('vector_size_mod', 'sum')
        if vs_mod is not None:
            # Modulate vector size: base ± modulation
            self.vector_size = max(1, int(32 + float(vs_mod) * 10))

        k_mod = self.get_blended_input('k_mod', 'sum')
        if k_mod is not None:
            self.spatial_k = max(1.0, 8.0 + float(k_mod) * 2.0)

        # --- Get channel data ---
        vectors = self._get_channel_vectors()
        if vectors is None or len(vectors) == 0:
            self._render_idle()
            return

        # --- Build interference field ---
        s = self.canvas_size
        total_field = np.zeros((s, s), dtype=np.complex128)

        for i, vec in enumerate(vectors):
            pos = self.channel_positions[i] if i < len(self.channel_positions) else (0, 0)
            channel_field = self._project_channel(vec, pos, (s, s))
            total_field += channel_field

        # Normalize by channel count
        total_field /= max(len(vectors), 1)

        # --- Temporal smoothing ---
        if self.prev_field is not None and self.temporal_smoothing > 0:
            alpha = self.temporal_smoothing
            total_field = alpha * self.prev_field + (1.0 - alpha) * total_field
        self.prev_field = total_field.copy()
        self.interference_field = total_field

        # --- Render interference pattern ---
        magnitude = np.abs(total_field)
        mag_max = magnitude.max()
        if mag_max > 1e-9:
            mag_norm = magnitude / mag_max
        else:
            mag_norm = magnitude

        # Apply circle mask
        mag_norm *= self.circle_mask

        # Colormap
        img_u8 = (np.clip(mag_norm, 0, 1) * 255).astype(np.uint8)
        self.interference_image = cv2.applyColorMap(img_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        mask3 = np.stack([self.circle_mask] * 3, axis=-1)
        self.interference_image = (self.interference_image * mask3).astype(np.uint8)

        # --- Phase recovery reconstruction ---
        self._phase_recovery_reconstruct(mag_norm)

        # --- Channel mandala (show per-channel rings) ---
        self._render_channel_mandala(vectors)

        # --- Complexity metric ---
        # How much structure vs uniform? Use spatial frequency content
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(mag_norm)))
        # High frequency content relative to total
        cy, cx = s // 2, s // 2
        y, x = np.ogrid[:s, :s]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        high_freq_mask = r > s * 0.2
        total_power = np.sum(fft_mag**2) + 1e-9
        high_freq_power = np.sum(fft_mag[high_freq_mask]**2)
        self.complexity = float(high_freq_power / total_power)

        # --- HUD ---
        self._draw_hud()

        self.frame_count += 1

    def _phase_recovery_reconstruct(self, magnitude):
        """Gerchberg-Saxton phase recovery on the interference magnitude."""
        if self.phase_recovery_iters <= 0:
            self.reconstruction_image = self.interference_image.copy()
            return

        s = self.canvas_size

        # Use magnitude as frequency-domain constraint
        mag_freq = np.fft.fftshift(np.abs(np.fft.fft2(magnitude)))

        # Iterative phase recovery
        phase = np.random.uniform(-np.pi, np.pi, (s, s))

        for _ in range(self.phase_recovery_iters):
            spectrum = mag_freq * np.exp(1j * phase)
            spatial = np.fft.ifft2(np.fft.ifftshift(spectrum))
            spatial = np.abs(spatial)
            spatial = np.clip(spatial, 0, None)
            new_spectrum = np.fft.fftshift(np.fft.fft2(spatial))
            phase = np.angle(new_spectrum)

        # Final
        final = mag_freq * np.exp(1j * phase)
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(final)))

        # Smooth
        if self.smoothing > 0:
            recon = gaussian_filter(recon, self.smoothing)

        # Normalize
        rmax = recon.max()
        if rmax > 1e-9:
            recon = recon / rmax

        recon *= self.circle_mask

        # Colormap
        img_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        self.reconstruction_image = cv2.applyColorMap(img_u8, cv2.COLORMAP_BONE)
        mask3 = np.stack([self.circle_mask] * 3, axis=-1)
        self.reconstruction_image = (self.reconstruction_image * mask3).astype(np.uint8)

    def _render_channel_mandala(self, vectors):
        """Show each channel as a colored ring to see individual contributions."""
        s = self.canvas_size
        canvas = np.zeros((s, s, 3), dtype=np.float32)

        n = len(vectors)
        for i, vec in enumerate(vectors):
            pos = self.channel_positions[i] if i < len(self.channel_positions) else (0, 0)

            # Simple: mean amplitude → ring radius, position → color hue
            amp = np.mean(vec)
            r_frac = 0.2 + i * 0.7 / max(n, 1)

            # Ring
            ring = np.exp(-0.5 * ((self.r_grid - r_frac) / 0.02)**2)

            # Color by channel index
            hue = i / max(n, 1)
            r_c = 0.5 + 0.5 * np.cos(2 * np.pi * hue)
            g_c = 0.5 + 0.5 * np.cos(2 * np.pi * (hue - 0.33))
            b_c = 0.5 + 0.5 * np.cos(2 * np.pi * (hue - 0.66))

            canvas[:, :, 0] += ring * amp * r_c
            canvas[:, :, 1] += ring * amp * g_c
            canvas[:, :, 2] += ring * amp * b_c

        canvas = np.clip(canvas, 0, 1)
        canvas *= self.circle_mask[:, :, np.newaxis]
        self.channel_mandala_image = (canvas * 255).astype(np.uint8)

    def _draw_hud(self):
        """Draw info overlay on the interference image."""
        img = self.interference_image

        cv2.putText(img, "NEURAL INTERFEROMETER", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        info = f"Ch:{self.n_active} Vec:{self.vector_size} K:{self.spatial_k:.1f}"
        cv2.putText(img, info, (10, self.canvas_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Complexity bar
        bar_w = int(self.complexity * 100)
        cv2.rectangle(img, (self.canvas_size - 110, 8),
                      (self.canvas_size - 110 + bar_w, 16), (0, 255, 200), -1)
        cv2.putText(img, f"C={self.complexity:.2f}",
                    (self.canvas_size - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Band mode
        cv2.putText(img, self.band_mode.upper(), (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 255), 1)

    def _render_idle(self):
        """No data state."""
        s = self.canvas_size
        self.interference_image = np.zeros((s, s, 3), dtype=np.uint8)
        cv2.putText(self.interference_image, "NEURAL INTERFEROMETER", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(self.interference_image, "Load EDF to begin", (s // 2 - 60, s // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        self.reconstruction_image = self.interference_image.copy()
        self.channel_mandala_image = self.interference_image.copy()

    def get_output(self, port_name):
        if port_name == 'interference':
            return self.interference_image
        elif port_name == 'reconstruction':
            return self.reconstruction_image
        elif port_name == 'channel_mandala':
            return self.channel_mandala_image
        elif port_name == 'n_channels':
            return float(self.n_active)
        elif port_name == 'complexity':
            return float(self.complexity)
        elif port_name == 'interference_field':
            return self.interference_field
        return None

    def get_display_image(self):
        """Show interference pattern as node face."""
        img = self.interference_image
        if img is None or img.size == 0:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        s = self.canvas_size
        return QtGui.QImage(img_rgb.data, s, s, s * 3,
                           QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        band_options = [
            ("Broadband", "broadband"),
            ("Delta (1-4Hz)", "delta"),
            ("Theta (4-8Hz)", "theta"),
            ("Alpha (8-13Hz)", "alpha"),
            ("Beta (13-30Hz)", "beta"),
            ("Gamma (30-50Hz)", "gamma"),
        ]
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, "file_open"),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("Band Mode", "band_mode", self.band_mode, band_options),
            ("Vector Size (neurons)", "vector_size", self.vector_size, "int"),
            ("Canvas Size", "canvas_size", self.canvas_size, "int"),
            ("Spatial K", "spatial_k", self.spatial_k, "float"),
            ("Phase Recovery Iters", "phase_recovery_iters", self.phase_recovery_iters, "int"),
            ("Smoothing", "smoothing", self.smoothing, "float"),
            ("Temporal Smooth", "temporal_smoothing", self.temporal_smoothing, "float"),
            ("Use Electrode Positions", "use_electrode_positions", self.use_electrode_positions, "bool"),
        ]

    def set_config_options(self, options):
        rebuild = False
        if 'edf_file_path' in options:
            self.edf_file_path = options['edf_file_path']
        if 'selected_region' in options:
            self.selected_region = options['selected_region']
        if 'band_mode' in options:
            self.band_mode = options['band_mode']
        if 'vector_size' in options:
            self.vector_size = max(1, int(options['vector_size']))
        if 'canvas_size' in options:
            new_size = int(options['canvas_size'])
            if new_size != self.canvas_size:
                self.canvas_size = new_size
                rebuild = True
        if 'spatial_k' in options:
            self.spatial_k = float(options['spatial_k'])
        if 'phase_recovery_iters' in options:
            self.phase_recovery_iters = int(options['phase_recovery_iters'])
        if 'smoothing' in options:
            self.smoothing = float(options['smoothing'])
        if 'temporal_smoothing' in options:
            self.temporal_smoothing = float(options['temporal_smoothing'])
        if 'use_electrode_positions' in options:
            self.use_electrode_positions = bool(options['use_electrode_positions'])

        if rebuild:
            self._build_canvas_grid()
            self.prev_field = None