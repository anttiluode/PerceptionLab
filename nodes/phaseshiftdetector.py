"""
Robust Phase Shift Node - Reality Compass V2 (Fixed)
====================================================
Compares the Spectral Slope (Beta) of Sensory vs. Deep layers.
Uses FFT instead of Box Counting for sub-pixel accuracy.

FIXES:
- Solved 'color' crash on startup.
- Solved 'pywt' warning (we don't use wavelets here).
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.stats import linregress

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class RobustPhaseShiftNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Phase Shift (Spectral)"
    NODE_COLOR = QtGui.QColor(255, 50, 50) # Alert Red
    
    def __init__(self):
        super().__init__()
        self.inputs = {
            'layer_1': 'image', # Sensory
            'layer_3': 'image'  # Deep
        }
        self.outputs = {
            'plot': 'image',
            'phase_delta': 'signal', # Positive = Sane, Negative = Hallucinating
            'l1_beta': 'signal',
            'l3_beta': 'signal'
        }
        
        self.history = []
        self._last_display = None

    def _get_spectral_beta(self, img):
        """Computes Power Spectrum Slope (Beta)."""
        if img is None: return 0.0
        
        # Ensure grayscale
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to safe calculation size (prevents warnings)
        img = cv2.resize(img.astype(np.float32), (64, 64))
        
        # FFT
        f = fft2(img)
        fshift = fftshift(f)
        magnitude_spectrum = np.abs(fshift)**2 + 1e-10
        
        # Radial Average
        h, w = magnitude_spectrum.shape
        cy, cx = h//2, w//2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-10)
        
        # Regression (Log-Log)
        # We skip DC (0) and very low freqs to avoid windowing artifacts
        valid_r = np.arange(3, min(32, len(radial_profile)))
        if len(valid_r) < 3: return 0.0
        
        power = radial_profile[valid_r]
        
        slope, _, _, _, _ = linregress(np.log(valid_r), np.log(power + 1e-10))
        return slope

    def step(self):
        l1 = self.get_blended_input('layer_1', 'mean')
        l3 = self.get_blended_input('layer_3', 'mean')
        
        beta1 = self._get_spectral_beta(l1)
        beta3 = self._get_spectral_beta(l3)
        
        # The Delta: 
        # Normal Brain: Sensory is Noisy (Beta ~ -1), Deep is Smooth (Beta ~ -3)
        # Delta = Beta1 - Beta3 should be POSITIVE (e.g. -1 - (-3) = +2)
        delta = beta1 - beta3
        
        # Update History
        self.history.append(delta)
        if len(self.history) > 100: self.history.pop(0)
        
        self.outputs['phase_delta'] = delta
        self.outputs['l1_beta'] = beta1
        self.outputs['l3_beta'] = beta3
        
        self._render_plot(delta, beta1, beta3)

    def _render_plot(self, delta, b1, b3):
        H, W = 200, 300
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(canvas, (0,0), (W,H), (20,20,20), -1)
        
        # Zero Line
        mid_y = H // 2
        cv2.line(canvas, (0, mid_y), (W, mid_y), (100,100,100), 1)
        
        # Define Color EARLY (Fixes the crash)
        color = (0, 255, 0) if delta > 0.1 else (0, 0, 255)
        
        # Plot History
        if len(self.history) > 1:
            pts = []
            scale_y = 30.0 # Sensitivity
            for i, val in enumerate(self.history):
                px = int((i / 100.0) * W)
                # Invert Y so up is positive
                py = int(mid_y - (val * scale_y))
                py = np.clip(py, 0, H)
                pts.append((px, py))
            
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 2)
            
        # HUD
        state = "PERCEPTION" if delta > 0.1 else "HALLUCINATION"
        if abs(delta) < 0.1: state = "CRITICAL / NOISE"
        
        cv2.putText(canvas, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(canvas, f"Sensory B: {b1:.2f} (Rough)", (10, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200))
        cv2.putText(canvas, f"Deep B:    {b3:.2f} (Smooth)", (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200))

        self._last_display = canvas
        self.outputs['plot'] = canvas

    def get_output(self, name):
        return self.outputs.get(name)

    def get_display_image(self):
        if self._last_display is None: return None
        h, w = self._last_display.shape[:2]
        return QtGui.QImage(self._last_display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)