"""
Auto-Explorer Node - Automatically animates through PC space
Creates smooth explorations of the learned manifold
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class AutoExplorerNode(BaseNode):
    """
    Automatically explores PCA latent space with smooth animations.
    Multiple modes: sequential, random walk, circular, spiral
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 220, 180)
    
    def __init__(self, mode='sequential'):
        super().__init__()
        self.node_title = "Auto-Explorer"
        
        self.inputs = {
            'latent_in': 'spectrum',
            'speed': 'signal',
            'amplitude': 'signal',
            'chaos': 'signal'  # Randomness amount
        }
        self.outputs = {
            'latent_out': 'spectrum',
            'current_pc': 'signal',
            'phase': 'signal'  # 0-1 oscillation
        }
        
        self.mode = mode  # 'sequential', 'random_walk', 'circular', 'spiral'
        
        # State
        self.base_latent = None
        self.current_latent = None
        self.phase = 0.0
        self.current_pc = 0
        self.random_state = np.random.randn(8)  # For random walk
        
    def step(self):
        latent_in = self.get_blended_input('latent_in', 'first')
        speed = self.get_blended_input('speed', 'sum') or 0.05
        amplitude = self.get_blended_input('amplitude', 'sum') or 2.0
        chaos = self.get_blended_input('chaos', 'sum') or 0.0
        
        if latent_in is not None:
            if self.base_latent is None:
                self.base_latent = latent_in.copy()
            self.current_latent = self.base_latent.copy()
            
            # Advance phase
            self.phase += speed
            
            if self.mode == 'sequential':
                self._sequential_mode(amplitude)
            elif self.mode == 'random_walk':
                self._random_walk_mode(amplitude, chaos)
            elif self.mode == 'circular':
                self._circular_mode(amplitude)
            elif self.mode == 'spiral':
                self._spiral_mode(amplitude)
                
    def _sequential_mode(self, amplitude):
        """Oscillate through PCs one at a time"""
        latent_dim = len(self.base_latent)
        
        # Current PC index (cycles through all)
        self.current_pc = int(self.phase / (2*np.pi)) % latent_dim
        
        # Oscillate that PC
        modulation = np.sin(self.phase) * amplitude
        self.current_latent[self.current_pc] += modulation
        
    def _random_walk_mode(self, amplitude, chaos):
        """Brownian motion in latent space"""
        latent_dim = len(self.base_latent)
        
        # Update random state
        self.random_state += np.random.randn(latent_dim) * chaos * 0.1
        
        # Apply damping
        self.random_state *= 0.98
        
        # Add to latent
        for i in range(min(latent_dim, len(self.random_state))):
            self.current_latent[i] += self.random_state[i] * amplitude
            
    def _circular_mode(self, amplitude):
        """Rotate in PC0-PC1 plane"""
        if len(self.base_latent) >= 2:
            self.current_latent[0] += np.cos(self.phase) * amplitude
            self.current_latent[1] += np.sin(self.phase) * amplitude
            self.current_pc = 0  # Indicate using PC0-PC1
            
    def _spiral_mode(self, amplitude):
        """Spiral outward in PC0-PC1 plane while oscillating PC2"""
        if len(self.base_latent) >= 3:
            # Expanding spiral
            radius = (self.phase / (2*np.pi)) % 5.0  # Expand over 5 cycles
            
            self.current_latent[0] += np.cos(self.phase) * radius * amplitude * 0.3
            self.current_latent[1] += np.sin(self.phase) * radius * amplitude * 0.3
            self.current_latent[2] += np.sin(self.phase * 2) * amplitude * 0.5
            
            self.current_pc = 2  # Indicate complex motion
            
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.current_latent
        elif port_name == 'current_pc':
            return float(self.current_pc)
        elif port_name == 'phase':
            return (self.phase % (2*np.pi)) / (2*np.pi)  # Normalized 0-1
        return None
        
    def get_display_image(self):
        """Show current exploration trajectory"""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        if self.current_latent is None:
            cv2.putText(img, "Waiting for input...", (10, 128),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            return QtGui.QImage(img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)
            
        # Draw mode and state
        mode_text = f"Mode: {self.mode}"
        pc_text = f"PC: {self.current_pc}"
        phase_text = f"Phase: {self.phase:.2f}"
        
        cv2.putText(img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, pc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(img, phase_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # Visualize current latent code as bars
        latent_dim = len(self.current_latent)
        bar_width = max(1, 256 // latent_dim)
        
        delta = self.current_latent - self.base_latent
        delta_max = np.abs(delta).max()
        if delta_max > 1e-6:
            delta_norm = delta / delta_max
        else:
            delta_norm = delta
            
        for i, val in enumerate(delta_norm):
            x = i * bar_width
            h = int(abs(val) * 80)
            y_base = 200
            
            if val >= 0:
                color = (0, 255, 0)
                y_start = y_base - h
                y_end = y_base
            else:
                color = (0, 0, 255)
                y_start = y_base
                y_end = y_base + h
                
            # Highlight current PC
            if i == self.current_pc:
                color = (255, 255, 0)
                
            cv2.rectangle(img, (x, y_start), (x+bar_width-1, y_end), color, -1)
            
        # Draw baseline
        cv2.line(img, (0, 200), (256, 200), (100, 100, 100), 1)
        
        return QtGui.QImage(img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Mode", "mode", self.mode, None)
        ]