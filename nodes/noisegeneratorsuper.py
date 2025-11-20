#!/usr/bin/env python3
"""
Noise Generator Super Node - Advanced Noise Synthesis
Save as: nodes/noisegeneratorsuper.py

Features:
- Multiple noise types (white, pink, brown, blue, violet, perlin, quantum, fractal)
- 1D 'signal' output and 2D 'array' image output
- Robust host import fallbacks and NumPy 2.0 compatibility
- Class name and NODE_CATEGORY follow host discovery conventions
"""

import os
import sys
import math
import numpy as np
import cv2
import __main__

BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class NoiseGeneratorSuperNode(BaseNode):
    """
    Advanced noise generator node for Perception Lab.

    Inputs:
      - none required (optional GUI/config driven)
    Outputs:
      - 'signal' : scalar (float) - mean or single-sample depending on mode
      - 'array'  : 2D numpy array float32 normalized 0..1 for display

    Config (exposed via get_config_options):
      - noise_type: string
      - dimension: '1D' or '2D'
      - amplitude: float
      - width/height: ints for 2D output size
      - perlin params, quantum coherence, etc.
    """
    NODE_CATEGORY = "Generator"
    NODE_COLOR = QtGui.QColor(100, 150, 100)

    def __init__(self):
        super().__init__()
        self.node_title = "Noise Generator Super"

        # IO
        self.inputs = {}  # no standard inputs required
        self.outputs = {
            'signal': 'signal',
            'array': 'image'
        }

        # Configurable parameters (you can change these from the host UI)
        self.noise_type = 'white'  # white, pink, brown, perlin, quantum, fractal, blue, violet
        self.dimension = '1D'      # '1D' or '2D'
        self.amplitude = 1.0
        self.sample_rate = 44100
        self.buffer_size = 1024

        # 2D output size
        self.width = 256
        self.height = 256

        # Pink noise (Voss-McCartney) state
        self.pink_rows = 16
        self._pink_values = np.zeros(self.pink_rows, dtype=np.float32)
        self._pink_index = 0

        # Brown noise state
        self._brown_value = 0.0

        # Blue/violet filter state
        self._blue_prev = 0.0
        self._violet_prev1 = 0.0
        self._violet_prev2 = 0.0

        # Perlin-like params
        self.perlin_scale = 0.05
        self.perlin_octaves = 4
        self.perlin_persistence = 0.5
        self.perlin_offset_x = 0.0
        self.perlin_offset_y = 0.0

        # Quantum-inspired params
        self.quantum_coherence = 0.5
        self.quantum_phase = 0.0

        # Fractal params
        self.fractal_octaves = 6

        # Outputs / buffers
        self.current_signal = 0.0
        self.current_array = np.zeros((self.height, self.width), dtype=np.float32)

        # Small RNG seed consistency option (optional)
        self._rng = np.random.default_rng()

    # -------------------------
    # Main step
    # -------------------------
    def step(self):
        """
        Called every engine tick. Generates either a 1D sample (signal)
        and a small scroller-array for visualization, or a full 2D field.
        """
        if self.dimension == '1D':
            self._generate_1d()
        else:
            self._generate_2d()

    # -------------------------
    # 1D generators
    # -------------------------
    def _generate_1d(self):
        t = None
        nt = self.noise_type.lower()
        if nt == 'white':
            t = self._white_noise()
        elif nt == 'pink':
            t = self._pink_noise()
        elif nt == 'brown':
            t = self._brown_noise()
        elif nt == 'blue':
            t = self._blue_noise()
        elif nt == 'violet':
            t = self._violet_noise()
        elif nt == 'quantum':
            t = self._quantum_noise_1d()
        else:
            # fallback
            t = self._white_noise()

        self.current_signal = float(np.clip(t * self.amplitude, -self.amplitude, self.amplitude))

        # Create a small scrolling visualization (128x128) if needed
        if self.current_array is None or self.current_array.shape != (128, 128):
            self.current_array = np.zeros((128, 128), dtype=np.float32)

        # Scroll left and insert new column scaled to 0..1
        self.current_array = np.roll(self.current_array, -1, axis=1)
        val = (self.current_signal + self.amplitude) / (2.0 * self.amplitude + 1e-12)
        val = float(np.clip(val, 0.0, 1.0))
        self.current_array[:, -1] = val

    # -------------------------
    # 2D generators
    # -------------------------
    def _generate_2d(self):
        nt = self.noise_type.lower()
        if nt == 'white':
            arr = self._white_noise_2d()
        elif nt == 'pink':
            arr = self._pink_noise_2d()
        elif nt == 'brown':
            arr = self._brown_noise_2d()
        elif nt == 'perlin':
            arr = self._perlin_noise_2d()
        elif nt == 'quantum':
            arr = self._quantum_noise_2d()
        elif nt == 'fractal':
            arr = self._fractal_noise_2d()
        elif nt == 'blue':
            arr = self._blue_noise_2d()
        elif nt == 'violet':
            arr = self._violet_noise_2d()
        else:
            arr = self._white_noise_2d()

        # ensure correct shape
        if arr.shape != (self.height, self.width):
            try:
                arr = cv2.resize(arr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            except Exception:
                arr = np.resize(arr, (self.height, self.width))

        # apply amplitude and normalize for display
        arr = arr.astype(np.float32) * float(self.amplitude)
        self.current_array = self._normalize_array(arr)
        # scalar signal output is mean value (centered to -1..1 then scaled)
        self.current_signal = float(np.mean(arr))

    # ====================
    # Noise implementations
    # ====================
    def _white_noise(self):
        return float(self._rng.uniform(-1.0, 1.0))

    def _white_noise_2d(self):
        return self._rng.uniform(-1.0, 1.0, size=(self.height, self.width)).astype(np.float32)

    def _pink_noise(self):
        # Voss-McCartney simple variant
        i = self._rng.integers(0, self.pink_rows)
        old = self._pink_values[i]
        new = self._rng.uniform(-1.0, 1.0)
        self._pink_values[i] = new
        val = float(np.sum(self._pink_values) / max(1, self.pink_rows))
        return val

    def _pink_noise_2d(self):
        # spectral 1/f approximation
        white = self._rng.normal(size=(self.height, self.width))
        f = np.fft.fft2(white)
        rows, cols = self.height, self.width
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol)**2 + (y - crow)**2) + 1e-12
        pink_filter = 1.0 / dist
        f_filtered = f * pink_filter
        pink = np.fft.ifft2(f_filtered).real
        return pink.astype(np.float32)

    def _brown_noise(self):
        step = self._rng.uniform(-0.1, 0.1)
        self._brown_value += step
        self._brown_value = float(np.clip(self._brown_value, -1.0, 1.0))
        return self._brown_value

    def _brown_noise_2d(self):
        white = self._rng.normal(scale=0.1, size=(self.height, self.width)).astype(np.float32)
        brown = np.cumsum(np.cumsum(white, axis=0), axis=1)
        # normalize dynamic range a bit
        return (brown - np.mean(brown)).astype(np.float32)

    def _blue_noise(self):
        w = self._rng.uniform(-1.0, 1.0)
        blue = w - self._blue_prev
        self._blue_prev = w
        return float(np.clip(blue, -1.0, 1.0))

    def _blue_noise_2d(self):
        white = self._rng.normal(size=(self.height, self.width))
        f = np.fft.fft2(white)
        rows, cols = self.height, self.width
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol)**2 + (y - crow)**2) + 1e-12
        blue_filter = dist
        f_filtered = f * blue_filter
        blue = np.fft.ifft2(f_filtered).real
        return blue.astype(np.float32)

    def _violet_noise(self):
        w = self._rng.uniform(-1.0, 1.0)
        violet = w - 2.0 * self._violet_prev1 + self._violet_prev2
        self._violet_prev2 = self._violet_prev1
        self._violet_prev1 = w
        return float(np.clip(violet, -1.0, 1.0))

    def _violet_noise_2d(self):
        white = self._rng.normal(size=(self.height, self.width))
        f = np.fft.fft2(white)
        rows, cols = self.height, self.width
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol)**2 + (y - crow)**2) + 1e-12
        violet_filter = dist**2
        f_filtered = f * violet_filter
        violet = np.fft.ifft2(f_filtered).real
        return violet.astype(np.float32)

    # Perlin-like implementation (sine-based pseudo-perlin for speed / portability)
    def _perlin_noise_2d(self):
        noise = np.zeros((self.height, self.width), dtype=np.float32)
        amplitude = 1.0
        frequency = self.perlin_scale
        max_value = 0.0
        for octave in range(self.perlin_octaves):
            noise += amplitude * self._perlin_octave(frequency)
            max_value += amplitude
            amplitude *= self.perlin_persistence
            frequency *= 2.0
        if max_value > 0:
            noise /= max_value
        return noise

    def _perlin_octave(self, frequency):
        # fast pseudo-Perlin using sines/cosines (deterministic-ish pattern)
        ys = np.linspace(0.0 + self.perlin_offset_y, (self.height - 1) * frequency + self.perlin_offset_y, self.height, dtype=np.float32)
        xs = np.linspace(0.0 + self.perlin_offset_x, (self.width - 1) * frequency + self.perlin_offset_x, self.width, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        noise = np.sin(xx * 1.5 + np.sin(yy * 2.3)) * np.cos(yy * 1.7 + np.cos(xx * 1.9))
        noise += 0.5 * np.sin(xx * 3.1 - yy * 2.7) * np.cos(yy * 2.9 + xx * 3.3)
        # animate offsets slightly
        self.perlin_offset_x += 0.01
        self.perlin_offset_y += 0.01
        return noise.astype(np.float32)

    def _quantum_noise_1d(self):
        coherent = math.sin(self.quantum_phase) * self.quantum_coherence
        decoherent = self._rng.uniform(-1.0, 1.0) * (1.0 - self.quantum_coherence)
        self.quantum_phase += self._rng.uniform(0.0, 0.2)
        return float(np.clip(coherent + decoherent, -1.0, 1.0))

    def _quantum_noise_2d(self):
        # interference pattern blended with random field
        ys = np.linspace(0, 10, self.height, dtype=np.float32)
        xs = np.linspace(0, 10, self.width, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        wave1 = np.sin(xx * 2.0 + self.quantum_phase)
        wave2 = np.sin(yy * 1.7 + self.quantum_phase * 1.3)
        wave3 = np.sin((xx + yy) * 1.2 + self.quantum_phase * 0.7)
        coherent = (wave1 + wave2 + wave3) / 3.0
        decoherent = self._rng.normal(size=(self.height, self.width))
        self.quantum_phase += 0.05
        q = coherent * self.quantum_coherence + decoherent * (1.0 - self.quantum_coherence)
        return q.astype(np.float32)

    def _fractal_noise_2d(self):
        # Fractional Brownian Motion style with sine-based base noise
        noise = np.zeros((self.height, self.width), dtype=np.float32)
        amplitude = 1.0
        frequency = 0.02
        for octave in range(self.fractal_octaves):
            ys = np.linspace(0, self.height * frequency, self.height, dtype=np.float32)
            xs = np.linspace(0, self.width * frequency, self.width, dtype=np.float32)
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            octave_noise = np.sin(xx * (17.5 + octave)) * np.cos(yy * (11.3 + octave))
            octave_noise += np.sin(yy * (13.7 + octave * 0.5)) * np.cos(xx * (19.1 + octave))
            noise += amplitude * octave_noise
            amplitude *= 0.6
            frequency *= 2.1
        return noise.astype(np.float32)

    # -------------------------
    # Utilities
    # -------------------------
    def _normalize_array(self, arr):
        arr = arr.astype(np.float32)
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if (amax - amin) > 1e-12:
            return (arr - amin) / (amax - amin)
        else:
            return np.zeros_like(arr, dtype=np.float32)

    # -------------------------
    # Host API outputs
    # -------------------------
    def get_output(self, port_name):
        if port_name == 'signal':
            return float(self.current_signal)
        if port_name == 'array':
            # return displayable 0..1 float32 array
            return self.current_array
        return None

    def get_display_image(self):
        # Host expects an array (0..1 float), or a QImage in some hosts.
        return self.current_array

    def get_config_options(self):
        # Provide config tuples: (label, attribute_name, current_value, options_or_type)
        return [
            ("Noise Type", "noise_type", self.noise_type,
             ['white', 'pink', 'brown', 'blue', 'violet', 'perlin', 'quantum', 'fractal']),
            ("Dimension", "dimension", self.dimension, ['1D', '2D']),
            ("Amplitude", "amplitude", self.amplitude, 'float'),
            ("Width", "width", self.width, 'int'),
            ("Height", "height", self.height, 'int'),
            ("Perlin Scale", "perlin_scale", self.perlin_scale, 'float'),
            ("Perlin Octaves", "perlin_octaves", self.perlin_octaves, 'int'),
            ("Perlin Persistence", "perlin_persistence", self.perlin_persistence, 'float'),
            ("Quantum Coherence", "quantum_coherence", self.quantum_coherence, 'float'),
        ]
