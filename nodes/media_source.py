"""
Media Source Node - Provides webcam or microphone input
Place this file in the 'nodes' folder
"""

import numpy as np
import cv2
from PyQt6 import QtGui

# Import the base class from parent directory
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

class MediaSourceNode(BaseNode):
    """Source node for video (Webcam) or audio (Microphone) input."""
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80)
    
    def __init__(self, source_type='Webcam', device_id=0, width=160, height=120, sample_rate=44100):
        super().__init__()
        self.device_id = int(device_id) 
        self.source_type = source_type
        self.node_title = f"Source ({source_type})"
        self.w, self.h = width, height
        self.sample_rate = sample_rate
        
        self.outputs = {'signal': 'signal', 'image': 'image'}

        self.frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.signal_output = 0.0 
        
        self.pa = PA_INSTANCE
        self.cap = None 
        self.stream = None
        
        # self.setup_source()
        
    def setup_source(self):
        """Initializes or re-initializes resources based on selected type."""
        # Cleanup existing resources
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
        
        self.cap = None
        self.stream = None

        try:
            if self.source_type == 'Webcam':
                self.cap = cv2.VideoCapture(self.device_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not self.cap.isOpened():
                    print(f"Warning: Cannot open webcam {self.device_id}")
            
            elif self.source_type == 'Microphone':
                if not self.pa:
                    print("Error: PyAudio not available for Microphone input.")
                    return
                
                channels = 1
                
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=channels, 
                    rate=int(self.sample_rate),
                    input=True,
                    input_device_index=self.device_id,
                    frames_per_buffer=1024
                )
        except Exception as e:
            print(f"Error setting up source {self.source_type}: {e}")
            self.node_title = f"Source ({self.source_type} ERROR)"
            return
            
        self.node_title = f"Source ({self.source_type})"

    def step(self):
        self.frame *= 0  # clear frame to black
        
        if self.source_type == 'Webcam' and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.resize(frame, (self.w, self.h))
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.signal_output = np.mean(gray) / 255.0  # Luminance signal
                
        elif self.source_type == 'Microphone' and self.stream and self.stream.is_active():
            try:
                data = self.stream.read(256, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if audio_data.size > 0:
                    self.signal_output = np.sqrt(np.mean(audio_data**2)) * 5.0 
                
                # Visual Feedback
                if audio_data.size > 0:
                    padded_audio = np.pad(audio_data, (0, 1024 - len(audio_data)))
                    spec = np.abs(np.fft.fft(padded_audio))
                    spec = spec[:self.w].copy() 
                    
                    spec = np.log1p(spec)
                    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
                    
                    audio_img = np.zeros((self.h, self.w), dtype=np.uint8)
                    for i in range(self.w):
                        h = int(spec[i] * self.h)
                        audio_img[self.h - h:, i] = 255
                    
                    self.frame = cv2.cvtColor(audio_img, cv2.COLOR_GRAY2BGR)
                    
            except Exception:
                self.signal_output = 0.0
        
    def get_output(self, port_name):
        if port_name == 'image':
            if self.frame.ndim == 3:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            else:
                gray = self.frame.astype(np.float32) / 255.0
            return gray
        elif port_name == 'signal':
            return self.signal_output
        return None
        
    def get_display_image(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
        super().close()
        
    def get_config_options(self):
        webcam_devices = [("Default Webcam (0)", 0), ("Secondary Webcam (1)", 1)]
        mic_devices = []
        if self.pa:
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    mic_devices.append((f"{info['name']} ({i})", i))
        
        device_options = mic_devices if self.source_type == 'Microphone' else webcam_devices
        
        if not any(v == self.device_id for _, v in device_options):
             device_options.append((f"Selected Device ({self.device_id})", self.device_id))
        
        return [
            ("Source Type", "source_type", self.source_type, [("Webcam", "Webcam"), ("Microphone", "Microphone")]),
            ("Device ID", "device_id", self.device_id, device_options),
        ]