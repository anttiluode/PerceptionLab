"""
Φ-Dwell Sonifier Node (v4 - Global Parent Fix)
==============================================
The final fix for the QThread Destruction crash.

Key Change:
The QThread is parented to the Global Qt Application instance. 
This prevents the Python Garbage Collector from killing the thread 
object while the audio loop is still closing.
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
    # Use the global PyAudio instance if provided by the host
    PA_INSTANCE = __main__.PA_INSTANCE
except Exception:
    class BaseNode:
        def __init__(self): self.inputs={}; self.outputs={}
    PA_INSTANCE = None

class DwellAudioWorker(QtCore.QObject):
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
        # Access the Host's PyAudio instance or create a local one
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
                    continue

                # Generate 1024 samples
                samples = np.arange(1024)
                mod = np.sin(2 * np.pi * (self.freq * 0.5) * (samples + t) / sr) * self.mod_depth
                carrier = np.sin(2 * np.pi * self.freq * (samples + t) / sr + mod)
                
                # Envelopes
                envelope = 1.0
                if self.regime == 2: # Bursty
                    envelope = 1.0 if (int(t/2000) % 2 == 0) else 0.0
                elif self.regime == 3: # Clocklike
                    envelope = np.abs(np.sin(t * 0.002))

                buf = (carrier * self.amp * envelope).astype(np.float32)
                
                try:
                    stream.write(buf.tobytes())
                except:
                    break
                t += 1024

        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except: pass
            # Only terminate if we created it locally
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

        # --- THE FIX: Parent to the main App instance, NOT self ---
        # This keeps the thread object alive in memory even if this node is deleted
        app = QtWidgets.QApplication.instance()
        self.thread = QtCore.QThread(app)
        self.worker = DwellAudioWorker()
        
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        
        self.thread.start()
        
        # UI Buffer
        self.display_img = np.zeros((150, 250, 3), dtype=np.uint8)
        cv2.rectangle(self.display_img, (0,0), (250, 150), (20, 25, 30), -1)
        cv2.putText(self.display_img, "Φ-DWELL AUDIO", (45, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 1)

    def step(self):
        meta = self.get_blended_input('meta', 'max') or 0.0
        regime = self.get_blended_input('regime', 'max') or 0
        band = self.get_blended_input('band', 'max') or 2
        mode = self.get_blended_input('mode', 'max') or 0.0

        base_freqs = [55.0, 110.0, 220.0, 440.0, 880.0]
        self.worker.freq = base_freqs[int(np.clip(band, 0, 4))] + (mode * 10.0)
        self.worker.amp = np.clip(meta * 0.4, 0, 0.6)
        self.worker.mod_depth = meta * 12.0  
        self.worker.regime = int(regime)

    def get_display_image(self):
        img = self.display_img.copy()
        vol_w = int(self.worker.amp * 300)
        cv2.rectangle(img, (25, 100), (25 + vol_w, 115), (50, 200, 100), -1)
        cv2.rectangle(img, (25, 100), (225, 115), (60, 70, 80), 1)
        
        # Status text for visual confirmation
        status = "CRITICAL" if self.worker.regime == 1 else "BURSTY" if self.worker.regime == 2 else "CLOCKLIKE" if self.worker.regime == 3 else "RANDOM"
        cv2.putText(img, status, (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Convert to QImage
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        return QtGui.QImage(img_rgb.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888).copy()

    def close(self):
        """Cleanly signals the worker and waits for the OS thread to exit."""
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread'):
            self.thread.quit()
            # Hard block for up to 1 second to allow PortAudio to release
            if not self.thread.wait(1000):
                print(f"[{self.node_title}] Thread shutdown timed out. Forcing cleanup.")