"""
Multi-Region Eigenmode Comparator
---------------------------------
Run multiple resonance systems in parallel with different
EEG regions as input. See how occipital vs frontal vs parietal
produce different stable geometries.
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class MultiRegionResonanceNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Multi-Region Comparator"
    NODE_COLOR = QtGui.QColor(100, 200, 150)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'occipital_spectrum': 'spectrum',
            'parietal_spectrum': 'spectrum',
            'frontal_spectrum': 'spectrum',
            'temporal_spectrum': 'spectrum',
            'reset': 'signal'
        }
        
        self.outputs = {
            'occipital_eigen': 'image',
            'parietal_eigen': 'image',
            'frontal_eigen': 'image',
            'temporal_eigen': 'image',
            'difference_map': 'image',
            'dominant_region': 'signal'
        }
        
        self.size = 64  # Smaller for 4 parallel systems
        self.regions = ['occipital', 'parietal', 'frontal', 'temporal']
        
        # Initialize 4 independent resonance systems
        self.structures = {}
        self.tensions = {}
        self.transfer_functions = {}
        
        for region in self.regions:
            self.structures[region] = np.ones((self.size, self.size), dtype=np.complex128)
            self.structures[region] += (np.random.randn(self.size, self.size) + 
                                        1j * np.random.randn(self.size, self.size)) * 0.1
            self.tensions[region] = np.zeros((self.size, self.size), dtype=np.float32)
            self.transfer_functions[region] = np.ones((self.size, self.size), dtype=np.float32)
        
        # Precompute grid
        self.center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        
        self.eigens = {r: None for r in self.regions}
        self.coherences = {r: 0.0 for r in self.regions}
    
    def project_to_2d(self, spectrum):
        if spectrum is None or len(spectrum) == 0:
            return np.zeros((self.size, self.size))
        freq_len = len(spectrum)
        r_flat = np.clip(self.r_grid.ravel(), 0, freq_len - 1).astype(int)
        return spectrum[r_flat].reshape(self.size, self.size)
    
    def step_region(self, region, spectrum):
        """Run one resonance step for a region"""
        if spectrum is None:
            self.tensions[region] *= 0.9
            return
        
        structure = self.structures[region]
        tension = self.tensions[region]
        transfer = self.transfer_functions[region]
        
        # Eigenfrequencies
        eigen = np.abs(fftshift(fft2(structure)))
        eigen = eigen / (np.max(eigen) + 1e-9)
        self.eigens[region] = eigen
        
        # Input
        input_2d = self.project_to_2d(spectrum)
        input_2d = input_2d / (np.max(input_2d) + 1e-9)
        
        # Mismatch
        resistance = input_2d * (1.0 - eigen)
        tension += resistance * 0.1
        
        # Critical transition
        threshold = 0.6
        critical = tension > threshold
        
        if np.sum(critical) > 0:
            structure[critical] *= -1
            transfer[critical] *= 0.8
            tension[critical] = 0
            structure = gaussian_filter(np.real(structure), sigma=0.5) + \
                       1j * gaussian_filter(np.imag(structure), sigma=0.5)
        
        # Evolution
        structure *= np.exp(1j * 0.05 * transfer)
        
        # Normalize
        mag = np.abs(structure)
        structure[mag > 1.0] /= mag[mag > 1.0]
        
        # Store
        self.structures[region] = structure
        self.tensions[region] = tension
        self.transfer_functions[region] = transfer
        
        # Coherence
        phase = np.angle(structure)
        self.coherences[region] = float(np.abs(np.mean(np.exp(1j * phase))))
    
    def step(self):
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            for region in self.regions:
                self.structures[region] = np.ones((self.size, self.size), dtype=np.complex128)
                self.structures[region] += (np.random.randn(self.size, self.size) + 
                                            1j * np.random.randn(self.size, self.size)) * 0.1
                self.tensions[region][:] = 0
                self.transfer_functions[region][:] = 1.0
            return
        
        # Process each region
        for region in self.regions:
            spectrum = self.get_blended_input(f'{region}_spectrum', 'sum')
            self.step_region(region, spectrum)
    
    def get_output(self, port_name):
        for region in self.regions:
            if port_name == f'{region}_eigen':
                return self.eigens.get(region)
        
        if port_name == 'difference_map':
            # Compute difference between regions
            if all(self.eigens[r] is not None for r in self.regions):
                # Max difference across all pairs
                diff = np.zeros((self.size, self.size))
                for i, r1 in enumerate(self.regions):
                    for r2 in self.regions[i+1:]:
                        diff = np.maximum(diff, np.abs(self.eigens[r1] - self.eigens[r2]))
                return diff
        
        if port_name == 'dominant_region':
            # Return index of highest coherence
            max_coh = max(self.coherences.values())
            for i, region in enumerate(self.regions):
                if self.coherences[region] == max_coh:
                    return float(i)
        
        return None
    
    def get_display_image(self):
        """2x2 grid of all regions"""
        panel_size = 64
        display = np.zeros((panel_size * 2, panel_size * 2, 3), dtype=np.uint8)
        
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        colors = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA]
        
        for (row, col), region, cmap in zip(positions, self.regions, colors):
            eigen = self.eigens.get(region)
            if eigen is not None:
                eigen_vis = (eigen / (eigen.max() + 1e-9) * 255).astype(np.uint8)
                colored = cv2.applyColorMap(eigen_vis, cmap)
                y, x = row * panel_size, col * panel_size
                display[y:y+panel_size, x:x+panel_size] = colored
                
                # Label
                cv2.putText(display, region[:3].upper(), (x + 2, y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(display, f"{self.coherences[region]:.2f}", (x + 2, y + panel_size - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        h, w = display.shape[:2]
        return QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)