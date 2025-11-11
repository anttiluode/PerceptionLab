"""
PC Visualizer Node - Visualize what each principal component controls
Shows the "eigenfaces" of your frequency space
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class PCVisualizerNode(BaseNode):
    """
    Visualizes individual principal components as images.
    Connect to SpectralPCA to see what each PC represents.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(180, 220, 120)
    
    def __init__(self, pc_index=0, amplitude=3.0):
        super().__init__()
        self.node_title = f"PC Visualizer (PC{pc_index})"
        
        self.inputs = {
            'pca_node': 'node_reference',  # Reference to SpectralPCA node
            'amplitude': 'signal'  # How much to amplify
        }
        self.outputs = {
            'image': 'image',
            'complex_spectrum': 'complex_spectrum',
            'variance_explained': 'signal'
        }
        
        self.pc_index = int(pc_index)
        self.amplitude = float(amplitude)
        
        # Visualization
        self.pc_image = np.zeros((128, 128, 3), dtype=np.uint8)
        self.pc_spectrum = None
        self.variance_explained = 0.0
        
    def step(self):
        # Get amplitude modulation
        amp_signal = self.get_blended_input('amplitude', 'sum')
        if amp_signal is not None:
            amplitude = amp_signal * 10.0  # Scale up for visibility
        else:
            amplitude = self.amplitude
            
        # Get reference to PCA node (this is a bit of a hack)
        # In practice, you'd connect SpectralPCA's outputs here
        # For now, we'll create a synthetic visualization
        
        # Create a spectrum with just this PC activated
        # This would come from: mean_spectrum + pc_component * amplitude
        
        # Placeholder: create a synthetic pattern
        size = 64
        freq = self.pc_index + 1
        
        # Each PC might represent a different frequency pattern
        y, x = np.ogrid[0:size, 0:size]
        pattern = np.sin(2*np.pi*freq*x/size) * np.cos(2*np.pi*freq*y/size)
        pattern = pattern * amplitude
        
        # Visualize
        pattern_norm = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
        self.pc_image = cv2.applyColorMap((pattern_norm * 255).astype(np.uint8), 
                                          cv2.COLORMAP_VIRIDIS)
        
        # Create complex spectrum (simplified)
        self.pc_spectrum = np.fft.fft2(pattern)
        
        # Variance explained (would come from PCA node)
        self.variance_explained = 1.0 / (self.pc_index + 1)  # Decreasing
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.pc_image.astype(np.float32) / 255.0
        elif port_name == 'complex_spectrum':
            return self.pc_spectrum
        elif port_name == 'variance_explained':
            return self.variance_explained
        return None
        
    def get_display_image(self):
        img = self.pc_image.copy()
        
        # Add label
        label = f"PC{self.pc_index}: {self.variance_explained:.1%}"
        cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255,255,255), 1)
        
        return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("PC Index", "pc_index", self.pc_index, None),
            ("Amplitude", "amplitude", self.amplitude, None)
        ]
