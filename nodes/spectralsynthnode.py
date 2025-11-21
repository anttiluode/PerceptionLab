# spectralsynthnode.py
"""
Spectral Synthesizer Node (The True Visual Cochlea)
---------------------------------------------------
A high-performance audio node that takes the 55-dimensional 
Eigenmode vector and synthesizes a continuous, organic soundscape 
using PyAudio.

Updated with internal FFT visualization and fixed time_counter bug.

Requires: pip install pyaudio
"""

import numpy as np
import cv2
import math
from scipy.fft import rfft
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

# Try to import PyAudio, handle failure gracefully
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not found. SpectralSynthesizerNode will be silent.")

class SpectralSynthesizerNode(BaseNode):
    NODE_CATEGORY = "Audio"
    NODE_COLOR = QtGui.QColor(200, 140, 40) # Gold/Brass

    def __init__(self, base_freq=110.0, gain=1.0):
        super().__init__()
        self.node_title = "Spectral Synthesizer"
        
        self.inputs = {
            'dna_55': 'spectrum',   # The vibration modes
            'master_gain': 'signal' # Volume control
        }
        
        self.outputs = {
            'visualizer': 'image',      # Audio visualization
            'audio_signal': 'signal',   # Output for FFTCochlea
            'spectrum': 'spectrum',     # FFT spectrum output
            'fft_image': 'image'        # FFT visualization
        }
        
        self.base_freq = float(base_freq)
        self.master_gain = float(gain)
        self.num_modes = 55
        
        # Audio State
        self.active = PYAUDIO_AVAILABLE
        self.sample_rate = 44100
        self.chunk_size = 1024
        
        # The target amplitudes (from the visual simulation)
        self.target_amps = np.zeros(self.num_modes, dtype=np.float32)
        # The current amplitudes (for smoothing)
        self.current_amps = np.zeros(self.num_modes, dtype=np.float32)
        
        # Phase tracking for 55 oscillators
        self.phases = np.zeros(self.num_modes, dtype=np.float32)
        
        # TIME COUNTER - FIX FOR THE BUG
        self.time_counter = 0.0
        
        # FFT Buffer for analysis
        self.fft_buffer_size = 2048
        self.fft_buffer = np.zeros(self.fft_buffer_size, dtype=np.float32)
        self.spectrum_data = None
        self.fft_display = np.zeros((64, 64), dtype=np.uint8)
        
        # Bessel Ratios (The "Drum" Tuning)
        self.ratios = np.array([
            1.000, 1.593, 2.135, 2.653, 3.155, 
            1.593, 2.295, 2.917, 3.500, 4.058, 
            2.135, 2.917, 3.600, 4.230, 4.831, 
            2.653, 3.500, 4.230, 4.900, 5.550, 
            3.155, 4.058, 4.831, 5.550, 6.200,
            3.650, 4.600, 5.400, 6.150, 6.850
        ], dtype=np.float32)
        
        # Fill the rest with harmonics if 55 modes are used
        if len(self.ratios) < self.num_modes:
            extension = np.linspace(7.0, 15.0, self.num_modes - len(self.ratios))
            self.ratios = np.concatenate([self.ratios, extension])
            
        self.freqs = self.base_freq * self.ratios
        
        # Initialize PyAudio
        if self.active:
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()

    def audio_callback(self, in_data, frame_count, time_info, status):
        # This runs on a separate high-priority thread
        
        # 1. Smoothing: Move current amplitudes towards targets
        lerp_factor = 0.1
        self.current_amps = self.current_amps * (1 - lerp_factor) + self.target_amps * lerp_factor
        
        # 2. Generate Silence
        output = np.zeros(frame_count, dtype=np.float32)
        
        # 3. Synthesize Active Modes (Optimization)
        active_indices = np.where(self.current_amps > 0.001)[0]
        
        buffer_indices = np.arange(frame_count, dtype=np.float32)
        
        for i in active_indices:
            amp = self.current_amps[i]
            freq = self.freqs[i]
            current_phase = self.phases[i]
            
            # Wave = amp * sin(2pi*f*t + phase)
            phase_step = 2 * np.pi * freq / self.sample_rate
            chunk_phases = current_phase + buffer_indices * phase_step
            
            output += amp * np.sin(chunk_phases)
            
            # Update stored phase
            self.phases[i] = (current_phase + frame_count * phase_step) % (2 * np.pi)

        # 4. Master Gain & Clipping
        output *= self.master_gain * 0.1 # Scale down to avoid clipping sum
        output = np.clip(output, -1.0, 1.0)
        
        return (output.astype(np.float32).tobytes(), pyaudio.paContinue)

    def step(self):
        # Get Inputs from the visual graph
        coeffs = self.get_blended_input('dna_55', 'first')
        gain_in = self.get_blended_input('master_gain', 'sum')
        
        if gain_in is not None:
            self.master_gain = np.clip(gain_in, 0.0, 2.0)
            
        # --- Audio Thread Logic (Amplitudes) ---
        if coeffs is not None:
            # Update the targets for the audio thread
            n = min(len(coeffs), self.num_modes)
            new_amps = np.abs(coeffs[:n])
            
            # Apply a slight curve so low modes are louder (Bass)
            new_amps = new_amps * (1.0 / (1.0 + np.arange(n) * 0.1))
            
            self.target_amps[:n] = new_amps
            self.target_amps[n:] = 0.0
        else:
            self.target_amps[:] = 0.0

        # --- Node Logic (Instantaneous Signal) ---
        # Synthesize a single sample for the node graph
        dt = 1.0 / 60.0 # Assuming 60Hz simulation step
        self.time_counter += dt
        
        mix_sample = 0.0
        total_energy = 0.0
        
        # Using current smoothed amplitudes
        for i in range(self.num_modes):
            amplitude = self.current_amps[i]
            if amplitude < 0.001: 
                continue 
            
            freq = self.freqs[i]
            osc_val = amplitude * math.sin(2 * math.pi * freq * self.time_counter)
            
            mix_sample += osc_val
            total_energy += amplitude
            
        if total_energy > 1.0:
            mix_sample /= total_energy
            
        mix_sample *= self.master_gain

        # --- Push to FFT Buffer ---
        self.fft_buffer[:-1] = self.fft_buffer[1:]
        self.fft_buffer[-1] = mix_sample
        
        # --- Compute FFT Spectrum ---
        self.compute_fft_spectrum()

        # --- Set Outputs ---
        self.set_output('audio_signal', float(mix_sample))

        # --- Visualization (Amplitude bars) ---
        spectro_vis = np.zeros((55, 20), dtype=np.float32)
        for i in range(min(self.num_modes, 55)):
            amplitude = self.current_amps[i]
            if amplitude > 0:
                spectro_vis[55-i-1:55, :] += amplitude

        spectro_img = cv2.applyColorMap(
            (np.clip(spectro_vis, 0, 1) * 255).astype(np.uint8), 
            cv2.COLORMAP_MAGMA
        )
        spectro_img = cv2.resize(spectro_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.set_output('visualizer', spectro_img)

    def compute_fft_spectrum(self):
        """Compute FFT spectrum from the audio buffer - EXACT FFT Cochlea style"""
        # Perform FFT (using fftshift like FFT Cochlea)
        f = np.fft.fft(self.fft_buffer)
        fsh = np.fft.fftshift(f)
        mag = np.abs(fsh)
        
        # Extract centered spectrum
        center = len(mag) // 2
        half = 32  # 64 bins total (32 on each side)
        spec = mag[center - half:center + half]
        
        # Store raw spectrum
        self.spectrum_data = spec.copy()
        
        # Create visualization EXACTLY like FFT Cochlea
        arr = np.log1p(spec)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        
        w, h = 64, 64
        self.fft_display = np.zeros((h, w), dtype=np.uint8)
        
        # Draw bars from bottom up, white on black
        for i in range(min(len(arr), w)):
            v = int(255 * arr[i])
            self.fft_display[h - v:, i] = 255
        
        # Flip to match FFT Cochlea orientation
        self.fft_display = np.flipud(self.fft_display)
        
        self.set_output('fft_image', self.fft_display)

    def get_output(self, port_name):
        if port_name == 'spectrum':
            return self.spectrum_data
        elif port_name == 'audio_signal':
            if hasattr(self, 'outputs_data'):
                return self.outputs_data.get('audio_signal', None)
        elif port_name == 'fft_image':
            if hasattr(self, 'outputs_data'):
                return self.outputs_data.get('fft_image', None)
        elif port_name == 'visualizer':
            if hasattr(self, 'outputs_data'):
                return self.outputs_data.get('visualizer', None)
        return None

    def set_output(self, name, val):
        if not hasattr(self, 'outputs_data'): 
            self.outputs_data = {}
        self.outputs_data[name] = val

    def get_display_image(self):
        """Show the FFT spectrum EXACTLY like FFT Cochlea - clean white on black"""
        img = np.ascontiguousarray(self.fft_display)
        h, w = img.shape
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def close(self):
        """Cleanup PyAudio"""
        if self.active:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
            
    def get_config_options(self):
        return [
            ("Base Freq (Hz)", "base_freq", self.base_freq, 'float'),
            ("Master Gain", "master_gain", self.master_gain, 'float')
        ]