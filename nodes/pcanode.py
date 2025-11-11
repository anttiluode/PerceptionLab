"""
Spectral PCA Node - Learns principal components of FFT spectra
Discovers which frequency patterns co-occur in your visual environment
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SpectralPCANode(BaseNode):
    """
    Learns PCA basis from complex FFT spectra.
    Compresses spectrum to latent code, reconstructs back.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(120, 180, 220)
    
    def __init__(self, latent_dim=16, buffer_size=100):
        super().__init__()
        self.node_title = "Spectral PCA"
        
        self.inputs = {
            'complex_spectrum': 'complex_spectrum',
            'learn': 'signal',  # 0-1: when to collect samples
            'pc_weights': 'spectrum'  # Optional: manually set latent code
        }
        self.outputs = {
            'latent_code': 'spectrum',  # The compressed representation
            'reconstructed_spectrum': 'complex_spectrum',
            'reconstruction_error': 'signal'
        }
        
        self.latent_dim = int(latent_dim)
        self.buffer_size = int(buffer_size)
        
        # Learning buffers
        self.spectrum_buffer = []
        self.is_learned = False
        
        # PCA parameters (the learned W-matrix!)
        self.mean_spectrum = None
        self.pca_components = None  # The principal components
        self.explained_variance = None
        
        # Current state
        self.latent_code = None
        self.reconstructed_spectrum = None
        self.error = 0.0
        
    def step(self):
        # Get input spectrum
        spec_in = self.get_blended_input('complex_spectrum', 'first')
        learn_signal = self.get_blended_input('learn', 'sum') or 0.0
        
        if spec_in is None:
            return
            
        # Flatten spectrum to vector
        spec_flat = spec_in.flatten()
        
        # LEARNING MODE: Collect samples
        if learn_signal > 0.5 and len(self.spectrum_buffer) < self.buffer_size:
            self.spectrum_buffer.append(spec_flat.copy())
            
            # When buffer full, compute PCA
            if len(self.spectrum_buffer) == self.buffer_size:
                self._compute_pca()
                
        # INFERENCE MODE: Encode/decode
        if self.is_learned:
            # Check if external latent code provided
            external_code = self.get_blended_input('pc_weights', 'first')
            
            if external_code is not None and len(external_code) == self.latent_dim:
                # Use provided latent code
                self.latent_code = external_code
            else:
                # Encode: project onto learned basis
                self.latent_code = self._encode(spec_flat)
            
            # Decode: reconstruct from latent
            self.reconstructed_spectrum = self._decode(self.latent_code)
            
            # Reshape back to 2D
            self.reconstructed_spectrum = self.reconstructed_spectrum.reshape(spec_in.shape)
            
            # Calculate reconstruction error
            self.error = np.mean(np.abs(spec_in - self.reconstructed_spectrum))
    
    def _compute_pca(self):
        """Compute PCA from collected spectra"""
        X = np.array(self.spectrum_buffer, dtype=np.complex64)
        
        # Separate real and imaginary parts
        X_real = X.real
        X_imag = X.imag
        
        # Compute mean
        self.mean_spectrum = X.mean(axis=0)
        
        # Center data
        X_real_centered = X_real - X_real.mean(axis=0)
        X_imag_centered = X_imag - X_imag.mean(axis=0)
        
        # SVD on real part (you could also do on magnitude)
        U, S, Vt = np.linalg.svd(X_real_centered, full_matrices=False)
        
        # Keep top components
        self.pca_components = Vt[:self.latent_dim]
        self.explained_variance = S[:self.latent_dim] ** 2 / len(X)
        
        self.is_learned = True
        print(f"PCA learned! Variance explained: {self.explained_variance.sum() / S.sum():.2%}")
        
    def _encode(self, spectrum):
        """Project spectrum onto learned PCA basis"""
        if not self.is_learned:
            return np.zeros(self.latent_dim)
            
        # Center
        centered = spectrum - self.mean_spectrum
        
        # Project (works for complex, projects real part)
        latent = centered.real @ self.pca_components.T
        
        return latent
    
    def _decode(self, latent_code):
        """Reconstruct spectrum from latent code"""
        if not self.is_learned:
            return np.zeros_like(self.mean_spectrum)
            
        # Reconstruct real part
        reconstructed_real = self.mean_spectrum.real + latent_code @ self.pca_components
        
        # Keep imaginary part from mean (or zero)
        reconstructed = reconstructed_real + 1j * self.mean_spectrum.imag
        
        return reconstructed
        
    def get_output(self, port_name):
        if port_name == 'latent_code':
            return self.latent_code
        elif port_name == 'reconstructed_spectrum':
            return self.reconstructed_spectrum
        elif port_name == 'reconstruction_error':
            return self.error
        return None
        
    def get_display_image(self):
        """Visualize latent code as bar graph"""
        img = np.zeros((128, 256, 3), dtype=np.uint8)
        
        if self.latent_code is None:
            return QtGui.QImage(img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)
            
        # Normalize latent code for display
        code = self.latent_code.copy()
        code_min, code_max = code.min(), code.max()
        if code_max - code_min > 1e-6:
            code_norm = (code - code_min) / (code_max - code_min)
        else:
            code_norm = np.zeros_like(code)
            
        # Draw bars
        bar_width = 256 // self.latent_dim
        for i, val in enumerate(code_norm):
            x = i * bar_width
            h = int(val * 128)
            
            # Color based on explained variance if available
            if self.explained_variance is not None:
                var_ratio = self.explained_variance[i] / self.explained_variance.max()
                color = (int(255 * var_ratio), 100, 255 - int(255 * var_ratio))
            else:
                color = (255, 255, 255)
                
            cv2.rectangle(img, (x, 128-h), (x+bar_width-1, 128), color, -1)
            
        return QtGui.QImage(img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None),
            ("Buffer Size", "buffer_size", self.buffer_size, None)
        ]
