import numpy as np
from scipy.signal import welch
import os
import time
import threading

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
    PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui
    PA_INSTANCE = None

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

class AudioPlayerNode(BaseNode):
    """
    Audio Player & Analyzer (Auto-Pause Fix).
    
    Now intelligently pauses audio if the simulation loop stops.
    """
    NODE_CATEGORY = "Input"
    NODE_TITLE = "Audio Player (Realtime)"
    NODE_COLOR = QtGui.QColor(255, 50, 100)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'volume': 'signal',
            'playback_pos': 'signal',
            'pause': 'signal' # Manual pause
        }
        
        self.outputs = {
            'spectrum': 'spectrum',
            'raw_signal': 'signal',
            'band_power': 'image'
        }
        
        self.band_names = ["Sub", "Bass", "Mid", "High", "Air"]
        for i in range(5):
            self.outputs[f'band_{i+1}_{self.band_names[i]}'] = 'signal'
            
        self.file_path = "music.mp3"
        self.gain = 1.0
        
        self.audio_data = None
        self.sample_rate = 44100
        self.play_head = 0
        self.stream = None
        self.is_playing = False
        
        # --- AUTO-PAUSE LOGIC ---
        self.last_step_time = time.time()
        
        self.last_spectrum = np.zeros(16)
        self.last_5bands = np.zeros(5)
        self.spec_edges = np.logspace(np.log10(20), np.log10(20000), 17)
        self.five_edges = [20, 60, 250, 2000, 6000, 20000]
        
        self.load_audio()

    def update(self):
        self.load_audio()

    def load_audio(self):
        self.stop_stream()
        
        if not os.path.exists(self.file_path):
            print(f"Audio file not found: {self.file_path}")
            return

        try:
            print(f"Loading {self.file_path}...")
            if HAS_PYDUB:
                seg = AudioSegment.from_file(self.file_path)
                self.sample_rate = seg.frame_rate
                samples = np.array(seg.get_array_of_samples())
                if seg.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                
                if seg.sample_width == 2:
                    samples = samples.astype(np.float32) / 32768.0
                elif seg.sample_width == 4:
                    samples = samples.astype(np.float32) / 2147483648.0
                    
                self.audio_data = samples
                self.start_stream()
            else:
                print("Pydub not installed.")
        except Exception as e:
            print(f"Error loading audio: {e}")

    def start_stream(self):
        if not HAS_PYAUDIO or PA_INSTANCE is None:
            return
            
        def callback(in_data, frame_count, time_info, status):
            if self.audio_data is None:
                return (None, pyaudio.paComplete)
            
            # --- AUTO-PAUSE CHECK ---
            # If step() hasn't been called in > 200ms, assume Host is Paused
            if time.time() - self.last_step_time > 0.2:
                # Return silence but keep stream alive (Pause)
                return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)

            if self.play_head + frame_count >= len(self.audio_data):
                self.play_head = 0
                
            data = self.audio_data[self.play_head : self.play_head + frame_count]
            self.play_head += frame_count
            
            out_data = (data * self.gain).astype(np.float32)
            return (out_data.tobytes(), pyaudio.paContinue)

        try:
            self.stream = PA_INSTANCE.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                stream_callback=callback,
                frames_per_buffer=1024
            )
            self.stream.start_stream()
            self.is_playing = True
        except Exception as e:
            print(f"Failed to start stream: {e}")

    def stop_stream(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except: pass
            self.stream = None
        self.is_playing = False

    def step(self):
        # Update heartbeat timestamp so audio thread knows we are running
        self.last_step_time = time.time()
        
        vol = self.get_blended_input('volume', 'sum')
        if vol is not None: self.gain = float(vol)
        
        if self.audio_data is None: return
        
        # Analysis
        window_size = int(self.sample_rate * 0.05)
        current_pos = self.play_head
        
        if current_pos + window_size > len(self.audio_data):
            chunk = self.audio_data[current_pos:]
        else:
            chunk = self.audio_data[current_pos : current_pos + window_size]
            
        if len(chunk) < 64: return
        
        freqs, psd = welch(chunk, fs=self.sample_rate, nperseg=len(chunk))
        
        # 16-Band
        spec = np.zeros(16)
        for i in range(16):
            mask = (freqs >= self.spec_edges[i]) & (freqs < self.spec_edges[i+1])
            if np.sum(mask) > 0: spec[i] = np.mean(psd[mask])
        if np.max(spec) > 0: spec /= np.max(spec)
        self.last_spectrum = spec
        
        # 5-Band
        bands = np.zeros(5)
        for i in range(5):
            mask = (freqs >= self.five_edges[i]) & (freqs < self.five_edges[i+1])
            if np.sum(mask) > 0: bands[i] = np.mean(psd[mask])
        bands = np.log1p(bands * 1000)
        if np.max(bands) > 0: bands /= np.max(bands)
        self.last_5bands = bands

    def get_output(self, port_name):
        if port_name == 'spectrum': return self.last_spectrum
        elif port_name == 'raw_signal': return float(np.mean(self.last_spectrum))
        elif port_name.startswith('band_'):
            try:
                idx = int(port_name.split('_')[1]) - 1
                return float(self.last_5bands[idx])
            except: pass
        elif port_name == 'band_power':
            h, w = 64, 128
            img = np.zeros((h, w), dtype=np.float32)
            bar_w = w // 16
            for i, val in enumerate(self.last_spectrum):
                height = int(val * (h-1))
                img[h-height:, i*bar_w:(i+1)*bar_w] = 1.0
            return img
        return None
        
    def get_config_options(self):
        return [
            ("Audio File", "file_path", self.file_path, "file_open"),
            ("Gain", "gain", self.gain, "float")
        ]
        
    def close(self):
        self.stop_stream()