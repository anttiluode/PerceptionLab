# spectralsynthnode.py
"""
Spectral Synthesizer Node (The True Visual Cochlea)
---------------------------------------------------
A high-performance audio node that takes the 55-dimensional 
Eigenmode vector and synthesizes a continuous, organic soundscape 
using PyAudio.

Requires: pip install pyaudio
"""

import numpy as np
import cv2
import math
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
            'visualizer': 'image'   # Audio visualization
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
        # Simple linear interpolation for this chunk
        lerp_factor = 0.1
        self.current_amps = self.current_amps * (1 - lerp_factor) + self.target_amps * lerp_factor
        
        # 2. Generate Silence
        output = np.zeros(frame_count, dtype=np.float32)
        
        # 3. Time steps for this chunk
        # t = np.arange(frame_count) / self.sample_rate
        # To maintain continuity, we use phase accumulation
        
        # 4. Synthesize Active Modes (Optimization)
        # Only synthesize modes with audible energy (> 0.01)
        active_indices = np.where(self.current_amps > 0.001)[0]
        
        # Precompute phase steps: 2 * pi * freq / sr
        phase_increments = (2 * np.pi * self.freqs[active_indices]) / self.sample_rate
        
        # Generate samples
        # This vectorization is complex for phase continuity, doing simple loop for stability
        # Note: For true high-performance, we'd use C++/NumPy buffers more cleverly.
        # Here we stick to a simplified additive synthesis loop.
        
        buffer_indices = np.arange(frame_count, dtype=np.float32)
        
        for i in active_indices:
            amp = self.current_amps[i]
            freq = self.freqs[i]
            current_phase = self.phases[i]
            
            # Wave = amp * sin(2pi*f*t + phase)
            # Increment phase for next chunk
            phase_step = 2 * np.pi * freq / self.sample_rate
            chunk_phases = current_phase + buffer_indices * phase_step
            
            output += amp * np.sin(chunk_phases)
            
            # Update stored phase
            self.phases[i] = (current_phase + frame_count * phase_step) % (2 * np.pi)

        # 5. Master Gain & Clipping
        output *= self.master_gain * 0.1 # Scale down to avoid clipping sum
        output = np.clip(output, -1.0, 1.0)
        
        return (output.astype(np.float32).tobytes(), pyaudio.paContinue)

    def step(self):
        # Get Inputs from the visual graph
        coeffs = self.get_blended_input('dna_55', 'first')
        gain_in = self.get_blended_input('master_gain', 'sum')
        
        if gain_in is not None:
            self.master_gain = np.clip(gain_in, 0.0, 2.0)
            
        if coeffs is not None:
            # Update the targets for the audio thread
            # Take magnitude (energy) of coefficients
            # Ensure length match
            n = min(len(coeffs), self.num_modes)
            new_amps = np.abs(coeffs[:n])
            
            # Apply a slight curve so low modes are louder (Bass)
            # and high modes are quieter (Texture)
            new_amps = new_amps * (1.0 / (1.0 + np.arange(n) * 0.1))
            
            # Update safely
            self.target_amps[:n] = new_amps
            self.target_amps[n:] = 0.0
        else:
            self.target_amps[:] = 0.0

    def get_output(self, port_name):
        return None

    def get_display_image(self):
        # Visualizer: Draw the current amplitudes as a bar graph
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.active:
            # Draw bars
            num_bars = min(self.num_modes, 55)
            bar_w = w // num_bars
            
            for i in range(num_bars):
                amp = self.current_amps[i]
                bar_h = int(np.clip(amp * 1000, 0, h))
                
                # Color gradient from Bass (Red) to Treble (Blue)
                color = (255 - i*4, i*4, 100)
                
                cv2.rectangle(img, (i*bar_w, h-bar_h), ((i+1)*bar_w - 1, h), color, -1)
                
            if not PYAUDIO_AVAILABLE:
                cv2.putText(img, "PyAudio Not Found", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)

    def close(self):
        # Cleanup PyAudio
        if self.active:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
            
    def get_config_options(self):
        return [
            ("Base Freq (Hz)", "base_freq", self.base_freq, 'float'),
            ("Master Gain", "master_gain", self.master_gain, 'float')
        ]