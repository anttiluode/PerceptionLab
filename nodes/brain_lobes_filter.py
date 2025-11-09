"""
Brain Lobes Node - Phase-lobes hypothesis demonstration
Shows frequency separation across brain regions

Place this in nodes/ folder
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
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class BrainLobesNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(180, 100, 200)
    
    def __init__(self, field_size=512, damage_lobe='None'):
        super().__init__()
        self.node_title = "Brain Lobes"
        
        self.inputs = {
            'external_field': 'signal',
            'damage_amount': 'signal',
        }
        
        self.outputs = {
            'frontal_output': 'signal',
            'parietal_output': 'signal',
            'temporal_output': 'signal',
            'occipital_output': 'signal',
            'integrated_experience': 'signal',
            'cross_frequency_leakage': 'signal',
            'lobe_spectrum_image': 'image',
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Brain Lobes (No SciPy!)"
            return
        
        self.field_size = int(field_size)
        self.damage_lobe = damage_lobe
        self.fs = 1000.0
        
        self.history = np.zeros(field_size, dtype=np.float32)
        self.W_lobes = {}
        self._init_filters()
        
        self.lobe_outputs = {'frontal': 0.0, 'parietal': 0.0, 'temporal': 0.0, 'occipital': 0.0}
        self.integrated_output = 0.0
        self.leakage_metric = 0.0
        self.last_spectra = {lobe: None for lobe in self.lobe_outputs.keys()}
        
    def _init_filters(self):
        freqs = rfftfreq(self.field_size, 1.0/self.fs)
        
        # Frontal: Theta (4-8 Hz)
        W_frontal = np.zeros_like(freqs)
        mask = (freqs >= 4.0) & (freqs <= 8.0)
        W_frontal[mask] = 1.0
        self.W_lobes['frontal'] = self._smooth(W_frontal, freqs, 4.0, 8.0)
        
        # Parietal: Alpha (8-13 Hz)
        W_parietal = np.zeros_like(freqs)
        mask = (freqs >= 8.0) & (freqs <= 13.0)
        W_parietal[mask] = 1.0
        self.W_lobes['parietal'] = self._smooth(W_parietal, freqs, 8.0, 13.0)
        
        # Temporal: Gamma (30-100 Hz)
        W_temporal = np.zeros_like(freqs)
        mask = (freqs >= 30.0) & (freqs <= 100.0)
        W_temporal[mask] = 1.0
        self.W_lobes['temporal'] = self._smooth(W_temporal, freqs, 30.0, 100.0)
        
        # Occipital: Beta-Gamma (13-100 Hz)
        W_occipital = np.zeros_like(freqs)
        mask = (freqs >= 13.0) & (freqs <= 100.0)
        W_occipital[mask] = 1.0
        self.W_lobes['occipital'] = self._smooth(W_occipital, freqs, 13.0, 100.0)
        
    def _smooth(self, W, freqs, low, high, width=3.0):
        for i, f in enumerate(freqs):
            if f < low:
                W[i] = np.exp(-((low - f)**2) / (2 * width**2))
            elif f > high:
                W[i] = np.exp(-((f - high)**2) / (2 * width**2))
        return W
    
    def _filter_lobe(self, signal, lobe_name, damage=0.0):
        F = rfft(signal)
        W = self.W_lobes[lobe_name].copy()
        
        if damage > 0.0:
            noise = np.random.randn(len(W)) * damage * 0.3
            W = W * (1.0 - damage * 0.5) + np.abs(noise)
            W = np.clip(W, 0, 1)
        
        W = W[:len(F)]
        F_filtered = F * W
        signal_filtered = irfft(F_filtered, n=len(signal))
        
        return signal_filtered, F, F_filtered
    
    def _compute_leakage(self):
        if self.last_spectra['frontal'] is None:
            return 0.0
        
        freqs = rfftfreq(self.field_size, 1.0/self.fs)
        frontal_spectrum = np.abs(self.last_spectra['frontal'])
        high_freq_mask = freqs > 20.0
        
        if len(frontal_spectrum) >= len(high_freq_mask):
            high_freq_mask = high_freq_mask[:len(frontal_spectrum)]
            contamination = np.sum(frontal_spectrum * high_freq_mask)
            total = np.sum(frontal_spectrum) + 1e-9
            leakage = contamination / total
        else:
            leakage = 0.0
        
        return float(np.clip(leakage, 0, 1))
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        external = self.get_blended_input('external_field', 'sum') or 0.0
        damage_signal = self.get_blended_input('damage_amount', 'sum') or 0.0
        damage_amount = np.clip((damage_signal + 1.0) / 2.0, 0, 1)
        
        self.history[:-1] = self.history[1:]
        self.history[-1] = external
        
        for lobe_name in ['frontal', 'parietal', 'temporal', 'occipital']:
            lobe_damage = damage_amount if self.damage_lobe == lobe_name else 0.0
            filtered, F_orig, F_filtered = self._filter_lobe(self.history, lobe_name, lobe_damage)
            self.lobe_outputs[lobe_name] = filtered[-1]
            self.last_spectra[lobe_name] = F_filtered
        
        self.integrated_output = (
            self.lobe_outputs['frontal'] * 0.3 +
            self.lobe_outputs['parietal'] * 0.25 +
            self.lobe_outputs['temporal'] * 0.25 +
            self.lobe_outputs['occipital'] * 0.2
        )
        
        self.leakage_metric = self._compute_leakage()
    
    def get_output(self, port_name):
        if port_name == 'frontal_output':
            return self.lobe_outputs['frontal']
        elif port_name == 'parietal_output':
            return self.lobe_outputs['parietal']
        elif port_name == 'temporal_output':
            return self.lobe_outputs['temporal']
        elif port_name == 'occipital_output':
            return self.lobe_outputs['occipital']
        elif port_name == 'integrated_experience':
            return self.integrated_output
        elif port_name == 'cross_frequency_leakage':
            return self.leakage_metric
        elif port_name == 'lobe_spectrum_image':
            return self._gen_spectrum_image()
        return None
    
    def _gen_spectrum_image(self):
        h, w = 128, 256
        img = np.zeros((h, w), dtype=np.float32)
        
        if self.last_spectra['frontal'] is None:
            return img
        
        band_h = h // 4
        lobe_names = ['frontal', 'parietal', 'temporal', 'occipital']
        colors = [0.3, 0.5, 0.7, 0.9]
        
        for i, lobe_name in enumerate(lobe_names):
            spectrum = np.abs(self.last_spectra[lobe_name])
            max_val = np.max(spectrum) + 1e-9
            spectrum_norm = spectrum / max_val
            
            if len(spectrum_norm) > w:
                indices = np.linspace(0, len(spectrum_norm)-1, w).astype(int)
                spectrum_norm = spectrum_norm[indices]
            
            y_start = i * band_h
            y_end = (i + 1) * band_h
            
            for x in range(min(len(spectrum_norm), w)):
                height = int(spectrum_norm[x] * band_h * 0.8)
                if height > 0:
                    y_bottom = y_end - 2
                    y_top = max(y_start, y_bottom - height)
                    img[y_top:y_bottom, x] = colors[i]
            
            if i < 3:
                img[y_end-1:y_end+1, :] = 0.2
        
        return img
    
    def get_display_image(self):
        if not SCIPY_AVAILABLE:
            return None
        
        spectrum_img = self._gen_spectrum_image()
        img_u8 = (np.clip(spectrum_img, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        h, w = img_color.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        band_h = h // 4
        
        labels = ['Frontal (θ)', 'Parietal (α)', 'Temporal (γ)', 'Occipital (β-γ)']
        lobe_keys = ['frontal', 'parietal', 'temporal', 'occipital']
        
        for i, label in enumerate(labels):
            y_pos = i * band_h + band_h // 2
            
            if self.damage_lobe == lobe_keys[i]:
                color = (0, 0, 255)
                label += " [DMG]"
            else:
                color = (255, 255, 255)
            
            cv2.putText(img_color, label, (5, y_pos), font, 0.35, color, 1, cv2.LINE_AA)
        
        if self.leakage_metric > 0.1:
            bar_width = 8
            bar_height = int(self.leakage_metric * h)
            img_color[-bar_height:, -bar_width:] = [0, 0, 255]
        
        img_color = np.ascontiguousarray(img_color)
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Damage Lobe", "damage_lobe", self.damage_lobe, [
                ("None (Healthy)", "None"),
                ("Frontal (Theta)", "frontal"),
                ("Parietal (Alpha)", "parietal"),
                ("Temporal (Gamma)", "temporal"),
                ("Occipital (Beta-Gamma)", "occipital")
            ]),
            ("Field Size", "field_size", self.field_size, None),
        ]