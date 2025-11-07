"""
Speaker Output Node - Outputs audio to speakers/headphones
** REBUILT **
This version uses a non-blocking callback and synthesizes a
sine wave, using the input signals for frequency and amplitude.
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import pyaudio
import sys
import os

# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------


class SpeakerOutputNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120) 
    
    def __init__(self, sample_rate=44100, device_index=None):
        super().__init__()
        self.node_title = "Speaker (Synth)"
        # FIX: Inputs are now 'frequency' and 'amplitude'
        self.inputs = {'frequency': 'signal', 'amplitude': 'signal'}
        
        self.pa = PA_INSTANCE
        self.sample_rate = int(sample_rate)
        self.device_index = device_index
        self.stream = None
        
        # Synthesis parameters
        self.current_freq = 440.0 # A4
        self.current_amp = 0.0
        self.phase = 0.0
        
        # Store last values for interpolation
        self._last_amp = 0.0
        self._last_freq = 440.0
        
        if not self.pa:
            self.node_title = "Speaker (NO PA)"
            return
        
        if self.device_index is None:
            try:
                self.device_index = self.pa.get_default_output_device_info()['index']
            except Exception:
                self.device_index = -1 
        
        self.open_stream()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """This is called by a separate audio thread"""
        
        # Get smooth ramps for parameters
        target_freq = self.current_freq
        target_amp = self.current_amp
        
        # Simple linear interpolation for smoothing
        amp_ramp = np.linspace(self._last_amp, target_amp, frame_count, dtype=np.float32)
        freq_ramp = np.linspace(self._last_freq, target_freq, frame_count, dtype=np.float32)
        
        # Calculate phase increments
        phase_inc = (2 * np.pi * freq_ramp) / self.sample_rate
        
        # Generate audio buffer
        phase_buffer = np.cumsum(phase_inc) + self.phase
        audio_buffer = (np.sin(phase_buffer) * amp_ramp).astype(np.float32)
        
        # Store last state for next buffer
        self.phase = phase_buffer[-1] % (2 * np.pi)
        self._last_amp = target_amp
        self._last_freq = target_freq
        
        # Convert to 16-bit int
        audio_int = np.clip(audio_buffer * 32767.0, -32768, 32767).astype(np.int16)
        
        return (audio_int.tobytes(), pyaudio.paContinue)
        
    def open_stream(self):
        """Opens or re-opens the PyAudio stream."""
        if self.stream: 
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
            
        if not self.pa or self.device_index < 0:
            return
            
        # Store last values for interpolation
        self._last_amp = self.current_amp
        self._last_freq = self.current_freq
            
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_index,
                frames_per_buffer=256,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            try:
                device_name = self.pa.get_device_info_by_index(self.device_index)['name']
                self.node_title = f"Speaker ({device_name[:15]}...)"
            except:
                self.node_title = "Speaker (Active)"
            
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.stream = None
            self.node_title = "Speaker (ERROR)"
            
    def step(self):
        # This runs at the SIMULATION frame rate
        # Map input signals to audio parameters
        
        # Freq: map [0, 1] (from analyzer) to audible range [100, 1000 Hz]
        freq_in = self.get_blended_input('frequency', 'sum') or 0.0
        self.current_freq = np.clip(freq_in * 900.0 + 100.0, 100.0, 1000.0)
        
        # Amp: map [0, 1] to [0, 0.5] (so it's not too loud)
        amp_in = self.get_blended_input('amplitude', 'sum')
        if amp_in is None:
            # If amplitude is not connected, set it to 0.5
            self.current_amp = 0.5
        else:
            self.current_amp = np.clip(amp_in * 0.5, 0.0, 0.5)

    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Draw amplitude bar
        amp_h = int(np.clip(self.current_amp * 2.0, 0, 1) * h)
        img[h - amp_h:, :w//2] = 255
        
        # Draw frequency bar
        freq_h = int(np.clip((self.current_freq - 100) / 900, 0, 1) * h)
        img[h - freq_h:, w//2:] = 180 

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        
    def get_config_options(self):
        if not self.pa:
            return [("PyAudio Not Found", "error", "Install PyAudio", [])]
            
        devices = []
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
                if info['max_output_channels'] > 0:
                    devices.append((f"{info['name']} ({i})", i))
            except Exception:
                continue # Skip invalid devices
            
        if not any(v == self.device_index for _, v in devices):
            devices.append((f"Selected Device ({self.device_index})", self.device_index))
            
        return [
            ("Output Device", "device_index", self.device_index, devices),
            ("Sample Rate", "sample_rate", self.sample_rate, None)
        ]
        
    def close(self):
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
        super().close()
