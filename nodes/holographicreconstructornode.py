"""
Curvature-Guided Holographic Reconstructor Node (FIXED)
========================================================
Uses Ricci curvature to optimize holographic reconstruction.

FIXED: Properly handles signal inputs (not images) for EEG and curvature
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode

class HolographicReconstructorNode(BaseNode):
    NODE_CATEGORY = "Deep Math"
    NODE_TITLE = "Holographic Reconstructor"
    NODE_COLOR = QtGui.QColor(150, 50, 150)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'phase_signal': 'signal',    # Simple signal input (like from Webcam phase)
            'curvature': 'signal',       # Ricci curvature scalar
        }
        
        self.outputs = {
            'reconstruction': 'image',   # The reconstructed image
            'confidence': 'signal',      # Reconstruction confidence
            'phase_map': 'image'        # The holographic phase visualization
        }
        
        # Reconstruction parameters
        self.size = 64
        self.reconstruction = np.zeros((self.size, self.size))
        
        # Curvature-adaptive filtering
        self.history_length = 50
        self.phase_history = []
        
        # Current state
        self.current_confidence = 1.0
        self.current_phase = 0.0
        
    def _compute_holographic_phase(self, phase_signal, curvature):
        """
        Accumulate phase from signal with curvature-adaptive filtering.
        
        High curvature = more temporal smoothing
        Low curvature = sharp/responsive
        """
        # Handle None inputs
        if phase_signal is None:
            phase_signal = 0.0
        if curvature is None:
            curvature = 0.0
        
        # Ensure scalar
        if isinstance(phase_signal, (list, np.ndarray)):
            phase_signal = float(np.mean(phase_signal))
        if isinstance(curvature, (list, np.ndarray)):
            curvature = float(np.mean(curvature))
        
        # Confidence inversely proportional to curvature
        # High curvature = uncertain = low confidence = need smoothing
        # FIXED: More responsive scaling
        confidence = 1.0 / (1.0 + abs(curvature) * 0.02)  # Changed from 0.0001 to 0.02
        confidence = np.clip(confidence, 0.1, 1.0)
        
        # Store phase in history
        self.phase_history.append(phase_signal)
        if len(self.phase_history) > self.history_length:
            self.phase_history.pop(0)
        
        # Adaptive temporal window based on curvature
        # High curvature = use more history (temporal smoothing)
        window_size = int(10 + (1.0 - confidence) * 40)
        window_size = min(window_size, len(self.phase_history))
        
        if window_size > 0 and len(self.phase_history) > 0:
            recent = np.array(self.phase_history[-window_size:])
            integrated_phase = np.mean(recent)
        else:
            integrated_phase = phase_signal
        
        # Convert to phase (mod 2π)
        phase = (integrated_phase * 0.1) % (2 * np.pi)
        
        return phase, confidence
    
    def _reconstruct_from_phase(self, phase, confidence):
        """
        Holographic reconstruction using interference patterns.
        Multiple "reference beams" at different frequencies.
        """
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        
        # Radial and angular coordinates
        r = np.sqrt((x - center)**2 + (y - center)**2)
        theta = np.arctan2(y - center, x - center)
        
        # Multiple temporal windows (like EEG models: 50-150ms, 150-250ms, etc)
        n_beams = 5
        reconstruction = np.zeros((self.size, self.size))
        
        for beam_idx in range(n_beams):
            # Each beam = different spatiotemporal frequency
            k_radial = 0.2 + beam_idx * 0.15
            k_angular = beam_idx + 1
            
            # Interference pattern modulated by phase
            # This is the holographic principle: interference creates structure
            interference = np.cos(k_radial * r + k_angular * theta + phase * (beam_idx + 1))
            
            # Weight by confidence
            # Low confidence = smooth (all beams equal)
            # High confidence = sharp (high freq beams weighted more)
            weight = confidence ** (beam_idx + 1)
            reconstruction += interference * weight
        
        # Normalize
        reconstruction = (reconstruction - reconstruction.min())
        if reconstruction.max() > 0:
            reconstruction = reconstruction / reconstruction.max()
        
        return reconstruction
    
    def _create_phase_visualization(self, phase, confidence):
        """Visualize the phase field"""
        # Create radial phase pattern
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        r = np.sqrt((x - center)**2 + (y - center)**2)
        theta = np.arctan2(y - center, x - center)
        
        # Phase determines the pattern
        phase_field = np.cos(r * 0.3 + theta * 2 + phase)
        
        # Confidence modulates brightness
        phase_field = phase_field * confidence
        
        # Normalize to 0-255
        phase_vis = ((phase_field + 1) * 127.5).astype(np.uint8)
        
        # Apply colormap
        phase_color = cv2.applyColorMap(phase_vis, cv2.COLORMAP_TWILIGHT)
        
        return phase_color
    
    def step(self):
        # Get inputs (no default parameter in get_blended_input)
        phase_signal = self.get_blended_input('phase_signal')
        curvature = self.get_blended_input('curvature')
        
        # Handle None values
        if phase_signal is None:
            phase_signal = 0.0
        if curvature is None:
            curvature = 0.0
        
        # Compute holographic phase with curvature-adaptive filtering
        phase, confidence = self._compute_holographic_phase(phase_signal, curvature)
        
        # Reconstruct image from phase
        self.reconstruction = self._reconstruct_from_phase(phase, confidence)
        
        # Create phase visualization
        self.phase_vis = self._create_phase_visualization(phase, confidence)
        
        # Store state
        self.current_confidence = confidence
        self.current_phase = phase
    
    def get_output(self, port_name):
        if port_name == 'reconstruction':
            return self.reconstruction
        
        elif port_name == 'confidence':
            return self.current_confidence
        
        elif port_name == 'phase_map':
            return self.phase_vis
        
        return None
    
    def get_display_image(self):
        # Show the reconstruction
        img = (self.reconstruction * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        
        # Add status text
        text = f"Conf:{self.current_confidence:.2f} Phase:{self.current_phase:.2f}"
        cv2.putText(img, text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        h, w, c = img.shape
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)