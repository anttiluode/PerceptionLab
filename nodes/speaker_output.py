"""
Speaker Output Node - Outputs audio to speakers/headphones
Place this file in the 'nodes' folder
"""

import numpy as np
from collections import deque
from PyQt6 import QtGui

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

try:
    import pyaudio
except ImportError:
    pyaudio = None

class SpeakerOutputNode(BaseNode):
    """Real signal output device using PyAudio."""
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120) 
    
    def __init__(self, buffer_len=512, sample_rate=44100, device_index=None):
        super().__init__()
        self.node_title = "Speaker Output"
        self.inputs = {'signal': 'signal'}
        self.buffer = deque(maxlen=buffer_len)
        self.rms_level = 0.0
        
        self.pa = PA_INSTANCE
        self.sample_rate = int(sample_rate)
        self.stream = None
        self.device_index = device_index
        self.is_playing = False
        
        if self.pa and self.device_index is None:
            try:
                self.device_index = self.pa.get_default_output_device_info()['index']
            except Exception:
                self.device_index = -1 
#        self.open_stream()
        
    def open_stream(self):
        """Opens or re-opens the PyAudio stream."""
        if self.stream: 
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
            
        self.is_playing = False
        if not self.pa or self.device_index < 0:
            self.node_title = "Speaker (NO PA)"
            return
            
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_index,
                frames_per_buffer=64 
            )
            self.is_playing = True
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
        val = self.get_blended_input('signal', 'sum') or 0.0
        
        clipped_val = np.clip(val, -1.0, 1.0)
        self.buffer.append(clipped_val)
        
        if len(self.buffer) > 0:
            rms_val = np.sqrt(np.mean(np.square(np.array(list(self.buffer)))))
            self.rms_level = self.rms_level * 0.8 + rms_val * 0.2
        
        if self.stream and self.is_playing and len(self.buffer) >= 64:
            audio_data = np.array(list(self.buffer)[:64], dtype=np.float32)
            audio_data = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            
            try:
                self.stream.write(audio_data.tobytes())
                for _ in range(64):
                    self.buffer.popleft() 
            except IOError:
                self.is_playing = False 
                
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        buffer_array = np.array(list(self.buffer))
        if len(buffer_array) > w:
            vis_data = buffer_array[-w:]
            vis_data = (vis_data + 1.0) / 2.0 * (h-1)
            for i in range(w):
                y = int(h - 1 - vis_data[i]) 
                y = max(0, min(h-1, y))
                img[y, i] = 255
                
        bar_height = int(np.clip(self.rms_level * 5.0, 0.0, 1.0) * h)
        img[h - bar_height:, w-5:w-1] = 180 

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        
    def get_config_options(self):
        if not self.pa:
            return [("PyAudio Not Found", "error", "Install PyAudio", [])]
            
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['max_output_channels'] > 0:
                devices.append((f"{info['name']} ({i})", i))
            
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