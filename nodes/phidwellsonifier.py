"""
Φ-Dwell Sonifier Node (v5 - Spectral Suite)
==========================================
Convention-compliant Audio Node with Signal & Image outputs.

Outputs:
- port 'spectrum': FFT coefficients of the audio (signal).
- port 'render': Live spectrogram visualization (image).

Wiring:
- Macroscope 'metastability' -> Sonifier 'meta'
- Macroscope 'regime_index'  -> Sonifier 'regime'
- Macroscope 'dominant_band' -> Sonifier 'band'
- Macroscope 'dominant_mode' -> Sonifier 'mode'
"""

import numpy as np
import cv2
import pyaudio
import time
from PyQt6 import QtGui, QtCore, QtWidgets

# --- HOST IMPORT ---
import __main__
try:
    BaseNode = __main__.BaseNode
    PA_INSTANCE = __main__.PA_INSTANCE
except Exception:
    class BaseNode:
        def __init__(self): self.inputs={}; self.outputs={}
    PA_INSTANCE = None

class DwellAudioWorker(QtCore.QObject):
    # Signal to send spectral data back to the main node
    spectrum_ready = QtCore.pyqtSignal(np.ndarray)
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = True
        self.freq = 220.0
        self.amp = 0.0
        self.mod_depth = 0.0
        self.regime = 0 

    @QtCore.pyqtSlot()
    def run(self):
        p = PA_INSTANCE if PA_INSTANCE else pyaudio.PyAudio()
        stream = None
        
        try:
            stream = p.open(
                format=pyaudio.paFloat32, 
                channels=1, 
                rate=44100, 
                output=True,
                frames_per_buffer=1024
            )
            
            t = 0.0
            sr = 44100.0
            
            while self._running:
                if self.amp < 0.001:
                    QtCore.QThread.msleep(30)
                    # Send empty spectrum when silent
                    self.spectrum_ready.emit(np.zeros(256))
                    continue

                # 1. Generate 1024 samples (FM Synthesis)
                samples = np.arange(1024)
                mod = np.sin(2 * np.pi * (self.freq * 0.5) * (samples + t) / sr) * self.mod_depth
                carrier = np.sin(2 * np.pi * self.freq * (samples + t) / sr + mod)
                
                # 2. Envelopes
                envelope = 1.0
                if self.regime == 2: # Bursty
                    envelope = 1.0 if (int(t/2000) % 2 == 0) else 0.0
                elif self.regime == 3: # Clocklike
                    envelope = np.abs(np.sin(t * 0.002))

                audio_chunk = (carrier * self.amp * envelope).astype(np.float32)
                
                # 3. Calculate Spectrum for output
                # Take FFT of the chunk, keep first 256 bins for visualization
                fft_data = np.abs(np.fft.rfft(audio_chunk))[:256]
                self.spectrum_ready.emit(fft_data)

                # 4. Write to speakers
                try:
                    stream.write(audio_chunk.tobytes())
                except:
                    break
                t += 1024

        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except: pass
            if not PA_INSTANCE:
                p.terminate()
            self.finished.emit()

    def stop(self):
        self._running = False

class PhiDwellSonifierNode(BaseNode):
    NODE_CATEGORY = "Audio"
    NODE_COLOR = QtGui.QColor(50, 200, 100) 

    def __init__(self):
        super().__init__()
        self.node_title = "Φ-Dwell Sonifier"
        
        self.inputs = {
            'meta': 'signal',      
            'regime': 'signal',    
            'band': 'signal',      
            'mode': 'signal'       
        }

        self.outputs = {
            'spectrum': 'signal',
            'render': 'image'
        }

        # Threading Setup
        app = QtWidgets.QApplication.instance()
        self.thread = QtCore.QThread(app)
        self.worker = DwellAudioWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect Worker Spectrum to Node logic
        self.worker.spectrum_ready.connect(self.update_spectrum_cache)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()
        
        # Internal Caches
        self.current_spectrum = np.zeros(256)
        self.spectrogram_scroll = np.zeros((150, 256), dtype=np.float32)
        self.output_image = None

    def update_spectrum_cache(self, fft_arr):
        """Called from worker thread at ~40Hz."""
        self.current_spectrum = fft_arr
        
        # Roll the spectrogram buffer
        self.spectrogram_scroll = np.roll(self.spectrogram_scroll, -1, axis=1)
        # Normalize and inject latest column
        norm_fft = np.clip(fft_arr[:150] / 50.0, 0, 1)
        self.spectrogram_scroll[:, -1] = norm_fft[::-1]

    def step(self):
        # Update Worker Params
        meta = self.get_blended_input('meta', 'max') or 0.0
        regime = self.get_blended_input('regime', 'max') or 0
        band = self.get_blended_input('band', 'max') or 2
        mode = self.get_blended_input('mode', 'max') or 0.0

        base_freqs = [55.0, 110.0, 220.0, 440.0, 880.0]
        self.worker.freq = base_freqs[int(np.clip(band, 0, 4))] + (mode * 10.0)
        self.worker.amp = np.clip(meta * 0.4, 0, 0.6)
        self.worker.mod_depth = meta * 12.0  
        self.worker.regime = int(regime)

    def get_output(self, port_name):
        if port_name == 'spectrum':
            return self.current_spectrum
        if port_name == 'render':
            return self.output_image
        return None

    def get_display_image(self):
        """Creates a live Spectrogram for the node face."""
        # Convert the scroll buffer to a heatmap
        heat = (self.spectrogram_scroll * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
        
        # Add HUD
        cv2.putText(color_map, "SONIC MANIFOLD", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mirror current display to output port
        self.output_image = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        
        h, w, ch = self.output_image.shape
        return QtGui.QImage(self.output_image.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888).copy()

    def close(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait(1000)