import numpy as np
import cv2
from collections import deque
from scipy.signal import butter, lfilter, lfilter_zi

# Strict Boilerplate to find the Host
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    # Standalone fallback
    from PyQt6 import QtGui
    class BaseNode: 
        def __init__(self): self._outs = {}
        def get_blended_input(self, n, m): return 0.0

class BrainSetNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Brain Set (Digital Logic)"
    NODE_COLOR = QtGui.QColor(0, 200, 100)

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal'
        }
        
        self.outputs = {
            'geometry': 'image',
            'box_score': 'signal',
            'state_class': 'signal',
            'x_out': 'signal',
            'y_out': 'signal'
        }
        
        # Internal storage for outputs if BaseNode doesn't handle it
        if not hasattr(self, '_outs'): self._outs = {}
        
        # Physics
        self.pixel_ms = 11.0
        self.buffer_len = 500
        self.raw_buffer = deque(maxlen=self.buffer_len)
        
        # Filter (Theta 4-8Hz)
        self.b, self.a = butter(2, [4, 8], btype='bandpass', fs=100, output='ba')
        self.zi = lfilter_zi(self.b, self.a)
        
        self._output_image = None
        self.last_score = 0.0

    # --- SAFETY FIX: Define set_output locally ---
    def set_output(self, name, value):
        self._outs[name] = value

    def step(self):
        # --- SAFETY FIX: Handle None inputs ---
        sig = self.get_blended_input('signal_in', 'sum')
        if sig is None: sig = 0.0
        
        # Filter
        try:
            filt_sig, self.zi = lfilter(self.b, self.a, [sig], zi=self.zi)
            val = filt_sig[0]
        except:
            val = 0.0
        
        self.raw_buffer.append(val)
        
        if len(self.raw_buffer) < 50:
            return

        # Z-Score
        data = np.array(self.raw_buffer)
        std = np.std(data)
        if std < 1e-6: std = 1.0
        z_data = (data - np.mean(data)) / std
        
        # Delay (11ms)
        delay = 2 
        
        x_traj = z_data[:-delay]
        y_traj = z_data[delay:]
        
        # Current Point for Export
        current_x = x_traj[-1]
        current_y = y_traj[-1]
        
        # Box Score (Entropy)
        angles = np.arctan2(y_traj, x_traj)
        hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi), density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-9))
        
        norm_score = 1.0 - (entropy / 2.77)
        self.last_score = np.clip(norm_score * 3.0, 0.0, 1.0) 
        is_digital = 1.0 if self.last_score > 0.4 else 0.0

        # Visualization
        self.render_geometry(x_traj, y_traj, is_digital)
        
        # Outputs
        self.set_output('box_score', float(self.last_score))
        self.set_output('state_class', float(is_digital))
        self.set_output('geometry', self._output_image)
        self.set_output('x_out', float(current_x))
        self.set_output('y_out', float(current_y))

    def render_geometry(self, x, y, is_digital):
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)
        color = (0, 255, 0) if is_digital else (255, 100, 0) 
        scale = size / 6.0
        center = size / 2.0
        
        # Simple decimation for speed
        step = max(1, len(x)//100)
        pts = []
        for i in range(0, len(x), step):
            px = int(x[i] * scale + center)
            py = int(y[i] * scale + center)
            pts.append([px, py])
            
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_arr], False, color, 1)
            
        self._output_image = img

    def get_output(self, name):
        # This connects the internal calculation to the outside world
        return self._outs.get(name)