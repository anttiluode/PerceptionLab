"""
Consciousness Filter Node - Models observer-dependent reality projection
Demonstrates "out-of-band content is invisible to the observer" principle.
Implements a trainable W matrix that learns which frequency bands constitute "experience".

Place this file in the 'nodes' folder as 'consciousnessfilter.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import rfft, irfft, rfftfreq
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: ConsciousnessFilterNode requires scipy")

class ConsciousnessFilterNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(140, 70, 180)  # Deep purple for consciousness
    
    def __init__(self, observer_bandwidth=50.0, field_size=512):
        super().__init__()
        self.node_title = "Consciousness Filter"
        
        self.inputs = {
            'external_field': 'signal',    # The "world out there" 
            'internal_field': 'signal',    # The "thoughts/predictions"
            'attention_shift': 'signal',   # Dynamically shift filter band
            'coherence_demand': 'signal'   # How much to enforce phase lock
        }
        
        self.outputs = {
            'conscious_experience': 'signal',  # What "you" experience
            'invisible_content': 'signal',     # What exists but you can't sense
            'phase_coherence': 'signal',       # How locked internal/external are
            'spectrum_image': 'image',         # Visualization of filter action
            'attractor_strength': 'signal'     # How stable is "you" right now
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Consciousness (No SciPy!)"
            return
        
        self.field_size = int(field_size)
        self.observer_bandwidth = float(observer_bandwidth)  # Hz cutoff
        
        self.fs = 1000.0 # Sample rate for frequency interpretation
        
        # The W matrix: your learned frequency response (what bands you can sense)
        self.W_filter_response = self._initialize_W_filter()
        
        # History for phase coherence tracking
        self.external_history = np.zeros(field_size, dtype=np.float32)
        self.internal_history = np.zeros(field_size, dtype=np.float32)
        
        # Attractor state (are you maintaining coherence?)
        self.attractor_basin_depth = 1.0
        self.coherence_history = []
        
        # --- FIX: Initialize all output variables ---
        self.phase_coherence = 0.0
        self.conscious_experience = 0.0
        self.invisible_content = 0.0
        self.attractor_strength_val = 0.0
        self.last_F_ext = np.zeros(self.field_size // 2 + 1, dtype=np.complex64)
        self.last_F_conscious = np.zeros(self.field_size // 2 + 1, dtype=np.complex64)
        # --- END FIX ---
        
    def _initialize_W_filter(self):
        """
        Initialize the W matrix as a frequency response function.
        This is your "consciousness bandwidth" - what you can sense.
        """
        freqs = rfftfreq(self.field_size, 1.0/self.fs)
        
        low_cutoff = 4.0   # Below theta: unconscious
        high_cutoff = self.observer_bandwidth  # Above this: too fast to integrate
        
        W = np.zeros_like(freqs)
        mask = (freqs >= low_cutoff) & (freqs <= high_cutoff)
        W[mask] = 1.0
        
        transition_width = 5.0
        for i, f in enumerate(freqs):
            if f < low_cutoff:
                W[i] = np.exp(-((low_cutoff - f)**2) / (2 * transition_width**2))
            elif f > high_cutoff:
                W[i] = np.exp(-((f - high_cutoff)**2) / (2 * transition_width**2))
        
        return W
    
    def apply_consciousness_filter(self, signal, attention_shift=0.0):
        """
        Apply the W matrix (consciousness filter) to incoming signal.
        """
        F = rfft(signal)
        freqs = rfftfreq(len(signal), 1.0/self.fs)
        
        shifted_W = np.roll(self.W_filter_response, int(attention_shift * 10))
        shifted_W = shifted_W[:len(F)]  # Match length
        
        F_conscious = F * shifted_W
        F_invisible = F * (1.0 - shifted_W)  # What you CAN'T sense
        
        conscious_signal = irfft(F_conscious, n=len(signal))
        invisible_signal = irfft(F_invisible, n=len(signal))
        
        return conscious_signal, invisible_signal, F, F_conscious
    
    def measure_phase_coherence(self, external, internal):
        """
        Measure how phase-locked external and internal fields are.
        """
        F_ext = rfft(external)
        F_int = rfft(internal)
        
        phase_ext = np.angle(F_ext)
        phase_int = np.angle(F_int)
        phase_diff = np.abs(phase_ext - phase_int)
        
        W_slice = self.W_filter_response[:len(phase_diff)]
        weighted_diff = phase_diff * W_slice
        
        coherence = 1.0 - np.mean(weighted_diff) / np.pi
        coherence = np.clip(coherence, 0, 1)
        
        return coherence
    
    def update_attractor_stability(self, coherence):
        """
        Track attractor stability over time.
        """
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > 100:
            self.coherence_history.pop(0)
        
        if len(self.coherence_history) > 10:
            coherence_variance = np.var(self.coherence_history[-20:])
            self.attractor_basin_depth = 1.0 / (1.0 + coherence_variance * 10)
        
        return self.attractor_basin_depth
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        external = self.get_blended_input('external_field', 'sum') or 0.0
        internal = self.get_blended_input('internal_field', 'sum') or 0.0
        attention_shift = self.get_blended_input('attention_shift', 'sum') or 0.0
        coherence_demand = self.get_blended_input('coherence_demand', 'sum') or 0.5
        
        self.external_history[:-1] = self.external_history[1:]
        self.external_history[-1] = external
        
        self.internal_history[:-1] = self.internal_history[1:]
        self.internal_history[-1] = internal
        
        conscious_ext, invisible_ext, F_ext, F_conscious = self.apply_consciousness_filter(
            self.external_history, attention_shift
        )
        
        conscious_int, invisible_int, F_int, _ = self.apply_consciousness_filter(
            self.internal_history, attention_shift
        )
        
        coherence = self.measure_phase_coherence(
            self.external_history, 
            self.internal_history
        )
        
        attractor_strength = self.update_attractor_stability(coherence)
        
        blend_ratio = 0.5 + coherence_demand * 0.3
        self.conscious_experience = (
            blend_ratio * conscious_ext[-1] + 
            (1 - blend_ratio) * conscious_int[-1]
        )
        
        self.invisible_content = invisible_ext[-1]
        
        self.phase_coherence = coherence
        self.attractor_strength_val = attractor_strength
        
        self.last_F_ext = F_ext
        self.last_F_conscious = F_conscious
    
    def get_output(self, port_name):
        if port_name == 'conscious_experience':
            return self.conscious_experience
        
        elif port_name == 'invisible_content':
            return self.invisible_content
        
        elif port_name == 'phase_coherence':
            return self.phase_coherence
        
        elif port_name == 'attractor_strength':
            return self.attractor_strength_val
        
        elif port_name == 'spectrum_image':
            return self.generate_spectrum_image()
        
        return None
    
    def generate_spectrum_image(self):
        """
        Visualize what you can/cannot sense.
        """
        if not hasattr(self, 'last_F_ext'):
            return np.zeros((64, 128), dtype=np.float32)
        
        h, w = 64, 128
        img = np.zeros((h, w), dtype=np.float32)
        
        mag_original = np.abs(self.last_F_ext)
        mag_conscious = np.abs(self.last_F_conscious)
        
        norm_max = np.max(mag_original) + 1e-9
        mag_original = mag_original / norm_max
        mag_conscious = mag_conscious / norm_max
        
        n_bins = len(mag_original)
        if n_bins > w:
            indices = np.linspace(0, n_bins-1, w).astype(int)
            mag_original = mag_original[indices]
            mag_conscious = mag_conscious[indices]
        
        for i in range(len(mag_original)):
            if i >= w:
                break
            
            height_orig = int(mag_original[i] * (h // 2 - 1))
            img[h//2 - height_orig:h//2, i] = 0.5
            
            height_cons = int(mag_conscious[i] * (h // 2 - 1))
            img[h//2:h//2 + height_cons, i] = 1.0
        
        img[h//2, :] = 0.3
        
        return img
    
    def get_display_image(self):
        if not SCIPY_AVAILABLE:
            return None
        
        spectrum_img = self.generate_spectrum_image()
        img_u8 = (np.clip(spectrum_img, 0, 1) * 255).astype(np.uint8)
        
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_PLASMA)
        
        h, w = img_color.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_color, 'EXISTS', (2, 12), font, 0.3, (255, 255, 255), 1)
        cv2.putText(img_color, 'YOU SENSE', (2, h-4), font, 0.3, (255, 255, 255), 1)
        
        bar_width = 8
        bar_height = int(self.phase_coherence * h)
        img_color[-bar_height:, -bar_width:] = [0, 255, 0]  # Green bar
        
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Observer Bandwidth (Hz)", "observer_bandwidth", self.observer_bandwidth, None),
            ("Field Size (samples)", "field_size", self.field_size, None),
        ]