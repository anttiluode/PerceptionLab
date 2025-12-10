# SimpleNeuralFieldNode.py
# Works with this file you will see the field immediately + purple port works

import numpy as np
import cv2

# --- HOST ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except:
    class BaseNode: 
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None
    from PyQt6 import QtGui

class SimpleNeuralFieldNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Neural Field → Complex (Fixed)"
    NODE_COLOR = QtGui.QColor(180, 50, 180)

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal',
            'beta':  'signal',
            'gamma': 'signal'
        }
        
        self.outputs = {
            'image':         'image',           # so you can see it immediately
            'complex_field': 'complex_spectrum' # purple port for iFFT
        }

        self.size = 128
        # start with tiny random complex field
        rng = np.random.default_rng(42)
        self.u = rng.normal(0, 0.01, (self.size, self.size)).astype(np.complex64) + \
                 1j * rng.normal(0, 0.01, (self.size, self.size)).astype(np.complex64)

        # Mexican-hat kernel in Fourier domain
        self.kernel_fft = self._make_kernel()

        # Host needs this dictionary
        self.active_outputs = {
            'image':         np.zeros((self.size, self.size, 3), np.uint8),
            'complex_field': self.u.copy()
        }

        self.t = 0

    def _make_kernel(self):
        x = np.linspace(-12, 12, self.size)
        y = np.linspace(-12, 12, self.size)
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2
        excite  = np.exp(-r2 / 4.0)
        inhibit = 0.6 * np.exp(-r2 / 25.0)
        kernel = excite - inhibit
        return np.fft.fft2(kernel).astype(np.complex64)

    def step(self):
        # ---- read the five bands ----
        d = float(self.get_blended_input('delta',  'sum') or 0)
        t = float(self.get_blended_input('theta',  'sum') or 0)
        a = float(self.get_blended_input('alpha',  'sum') or 0)
        b = float(self.get_blended_input('beta',   'sum') or 0)
        g = float(self.get_blended_input('gamma',  'sum') or 0)

        # ---- make a nice radial drive ----
        y, x = np.ogrid[:self.size, :self.size]
        cx = cy = self.size // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / (self.size // 3)

        I = ( np.tanh(d) * np.cos(1 * np.pi * r) +
              np.tanh(t) * np.cos(3 * np.pi * r) +
              np.tanh(a) * np.cos(5 * np.pi * r) +
              np.tanh(b) * np.cos(7 * np.pi * r) +
              np.tanh(g) * np.cos(11 * np.pi * r) )

        # slow global rotation so it breathes
        I = I.astype(np.complex64) * np.exp(1j * self.t * 0.04)

        # ---- classic Amari field step (all complex64) ----
        firing = np.tanh(self.u.real)                     # real part only for firing
        interaction = np.fft.ifft2(np.fft.fft2(firing) * self.kernel_fft)
        du = -self.u + interaction + I * 0.8
        self.u += du * 0.15

        # ---- visualisation (so you see something right now) ----
        mag = np.abs(self.u)
        norm = mag / (mag.max() + 1e-8)
        img = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(img, cv2.COLORMAP_TWILIGHT)

        # ---- push to host ----
        self.active_outputs['image']         = colored
        self.active_outputs['complex_field'] = self.u.copy()

        self.t += 1

    def get_output(self, name):
        return self.active_outputs.get(name)

    def get_display_image(self):
        return QtGui.QImage(
            self.active_outputs['image'].data,
            self.size, self.size,
            self.size * 3,
            QtGui.QImage.Format.Format_RGB888
        )