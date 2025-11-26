"""
Geometry Analyzer Node
----------------------
Tracks the GEOMETRIC properties of emerging patterns:
- Symmetry order (is it 4-fold, 6-fold, 8-fold?)
- Radial mode (which ring frequencies dominate?)
- Phase coherence (how locked is the system?)
- Rotation (is the pattern precessing?)

This is what makes the star interesting - not its brightness.
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class GeometryAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Geometry Analyzer"
    NODE_COLOR = QtGui.QColor(255, 180, 50)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'structure': 'image',
            'phase_field': 'image',  # Optional - from resonance node
            'reset': 'signal'
        }
        
        self.outputs = {
            'symmetry_order': 'signal',    # 2, 4, 6, 8-fold etc
            'dominant_radius': 'signal',    # Which ring is strongest
            'phase_coherence': 'signal',    # 0-1 how locked
            'rotation_rate': 'signal',      # Angular velocity
            'angular_spectrum': 'spectrum', # Full angular decomposition
            'radial_spectrum': 'spectrum'   # Full radial decomposition
        }
        
        # History for tracking dynamics
        self.history_len = 100
        self.symmetry_history = []
        self.coherence_history = []
        self.angle_history = []  # For tracking rotation
        
        # Current measurements
        self.symmetry_order = 0.0
        self.dominant_radius = 0.0
        self.phase_coherence = 0.0
        self.rotation_rate = 0.0
        self.angular_spectrum = np.zeros(32)
        self.radial_spectrum = np.zeros(64)
        
        # Cached grids
        self.size = 128
        self.center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        self.r_grid = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        self.theta_grid = np.arctan2(y - self.center, x - self.center)
        
        # Previous frame for rotation detection
        self.prev_angular = None
        
    def analyze_symmetry(self, structure):
        """
        Decompose the pattern into angular Fourier modes.
        The dominant mode tells us the symmetry order.
        """
        # Convert to frequency domain
        fft = fftshift(fft2(structure))
        magnitude = np.abs(fft)
        
        # Sample along rings at different radii
        # We care about the MID frequencies (not DC, not noise)
        r_min, r_max = 10, 50
        mask = (self.r_grid >= r_min) & (self.r_grid <= r_max)
        
        # Extract angular profile by averaging along rings
        n_angles = 360
        angular_profile = np.zeros(n_angles)
        
        for i in range(n_angles):
            angle = (i / n_angles) * 2 * np.pi - np.pi
            # Wedge mask
            angle_diff = np.abs(self.theta_grid - angle)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            wedge = angle_diff < (np.pi / n_angles)
            
            combined_mask = mask & wedge
            if np.sum(combined_mask) > 0:
                angular_profile[i] = np.mean(magnitude[combined_mask])
        
        # FFT of angular profile gives us symmetry modes
        angular_fft = np.abs(np.fft.fft(angular_profile))[:n_angles//2]
        
        # Normalize
        if angular_fft[0] > 0:
            angular_fft = angular_fft / angular_fft[0]
        
        # Store first 32 modes
        self.angular_spectrum = angular_fft[:32].astype(np.float32)
        
        # Find dominant symmetry (skip mode 0 = DC, mode 1 = offset)
        if len(angular_fft) > 2:
            peak_mode = np.argmax(angular_fft[2:]) + 2
            self.symmetry_order = float(peak_mode)
        
        return angular_profile
    
    def analyze_radial(self, structure):
        """
        Radial power spectrum - which ring frequencies dominate?
        """
        fft = fftshift(fft2(structure))
        magnitude = np.abs(fft)
        
        # Radial binning
        max_r = min(self.center, 64)
        radial_profile = np.zeros(max_r)
        
        for r in range(max_r):
            ring_mask = (self.r_grid >= r) & (self.r_grid < r + 1)
            if np.sum(ring_mask) > 0:
                radial_profile[r] = np.mean(magnitude[ring_mask])
        
        # Normalize
        if radial_profile.max() > 0:
            radial_profile = radial_profile / radial_profile.max()
        
        self.radial_spectrum = radial_profile.astype(np.float32)
        
        # Dominant radius (skip DC)
        if len(radial_profile) > 3:
            self.dominant_radius = float(np.argmax(radial_profile[3:]) + 3)
    
    def analyze_phase_coherence(self, structure, phase_field=None):
        """
        Phase coherence: how uniform is the phase?
        High coherence = locked state (the stable star)
        Low coherence = chaos/transition
        """
        if phase_field is not None:
            # Use provided phase field
            phase = phase_field
        else:
            # Estimate phase from structure via Hilbert-like transform
            fft = fft2(structure)
            # Zero negative frequencies
            fft_hilbert = fft.copy()
            fft_hilbert[self.size//2:, :] = 0
            analytic = np.fft.ifft2(fft_hilbert * 2)
            phase = np.angle(analytic)
        
        # Phase coherence = magnitude of mean phasor
        # If all phases align, this is 1. If random, this is ~0.
        mean_phasor = np.mean(np.exp(1j * phase))
        self.phase_coherence = float(np.abs(mean_phasor))
    
    def analyze_rotation(self, angular_profile):
        """
        Track if the pattern is rotating by comparing angular profiles.
        """
        if self.prev_angular is None:
            self.prev_angular = angular_profile.copy()
            return
        
        # Cross-correlation to find rotation
        correlation = np.correlate(angular_profile, self.prev_angular, mode='full')
        peak_offset = np.argmax(correlation) - len(angular_profile) + 1
        
        # Convert to degrees per frame
        degrees_per_frame = (peak_offset / len(angular_profile)) * 360
        
        # Smooth
        self.rotation_rate = self.rotation_rate * 0.9 + degrees_per_frame * 0.1
        
        self.prev_angular = angular_profile.copy()
    
    def step(self):
        structure = self.get_blended_input('structure', 'first')
        phase_field = self.get_blended_input('phase_field', 'first')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.symmetry_history.clear()
            self.coherence_history.clear()
            self.prev_angular = None
            return
        
        if structure is None:
            return
        
        # Resize if needed
        if structure.shape[0] != self.size:
            structure = cv2.resize(structure, (self.size, self.size))
        
        # Run all analyses
        angular_profile = self.analyze_symmetry(structure)
        self.analyze_radial(structure)
        self.analyze_phase_coherence(structure, phase_field)
        self.analyze_rotation(angular_profile)
        
        # Update histories
        self.symmetry_history.append(self.symmetry_order)
        self.coherence_history.append(self.phase_coherence)
        
        if len(self.symmetry_history) > self.history_len:
            self.symmetry_history.pop(0)
            self.coherence_history.pop(0)
    
    def get_output(self, port_name):
        if port_name == 'symmetry_order':
            return self.symmetry_order
        elif port_name == 'dominant_radius':
            return self.dominant_radius
        elif port_name == 'phase_coherence':
            return self.phase_coherence
        elif port_name == 'rotation_rate':
            return self.rotation_rate
        elif port_name == 'angular_spectrum':
            return self.angular_spectrum
        elif port_name == 'radial_spectrum':
            return self.radial_spectrum
        return None
    
    def get_display_image(self):
        """
        4-panel diagnostic:
        1. Angular spectrum (what symmetry?)
        2. Radial spectrum (what scale?)
        3. Symmetry history (how did it emerge?)
        4. State space (symmetry vs coherence)
        """
        w, h = 256, 256
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        pw, ph = w // 2, h // 2  # Panel dimensions
        
        # Panel 1: Angular spectrum (top-left)
        # This shows which symmetry modes are present
        if len(self.angular_spectrum) > 0:
            spec = self.angular_spectrum[:16]  # First 16 modes
            max_val = spec.max() + 1e-9
            bar_w = pw // 16
            for i, val in enumerate(spec):
                bar_h = int((val / max_val) * (ph - 20))
                x = i * bar_w
                # Color by mode number
                if i == int(self.symmetry_order):
                    color = (0, 255, 255)  # Yellow for dominant
                else:
                    color = (100, 100, 100)
                cv2.rectangle(panel, (x, ph - bar_h), (x + bar_w - 1, ph), color, -1)
        cv2.putText(panel, f"Sym: {self.symmetry_order:.0f}-fold", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Panel 2: Radial spectrum (top-right)
        if len(self.radial_spectrum) > 0:
            spec = self.radial_spectrum[:32]
            max_val = spec.max() + 1e-9
            bar_w = pw // 32
            for i, val in enumerate(spec):
                bar_h = int((val / max_val) * (ph - 20))
                x = pw + i * bar_w
                color = (0, int(255 * val / max_val), 255)
                cv2.rectangle(panel, (x, ph - bar_h), (x + bar_w - 1, ph), color, -1)
        cv2.putText(panel, f"Radius: {self.dominant_radius:.0f}", (pw + 5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Panel 3: History (bottom-left)
        # Symmetry order over time
        if len(self.symmetry_history) > 1:
            pts = []
            for i, sym in enumerate(self.symmetry_history):
                x = int((i / len(self.symmetry_history)) * (pw - 10)) + 5
                y = ph + ph - 10 - int((sym / 12) * (ph - 20))  # 0-12 fold range
                pts.append((x, y))
            for i in range(len(pts) - 1):
                cv2.line(panel, pts[i], pts[i+1], (255, 100, 100), 1)
        cv2.putText(panel, "Symmetry History", (5, ph + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Panel 4: State space (bottom-right)
        # X = coherence, Y = symmetry order
        # This is the INTERESTING plot - where is the system in phase space?
        # Draw grid
        cv2.rectangle(panel, (pw, ph), (w, h), (30, 30, 30), -1)
        
        # Draw trajectory
        if len(self.symmetry_history) > 1 and len(self.coherence_history) > 1:
            pts = []
            for i in range(min(len(self.symmetry_history), len(self.coherence_history))):
                coh = self.coherence_history[i]
                sym = self.symmetry_history[i]
                x = pw + 10 + int(coh * (pw - 20))
                y = h - 10 - int((sym / 12) * (ph - 20))
                pts.append((x, y))
            
            # Draw with fading trail
            for i in range(len(pts) - 1):
                alpha = i / len(pts)
                color = (int(50 + 200 * alpha), int(100 * alpha), int(255 * (1 - alpha)))
                cv2.line(panel, pts[i], pts[i+1], color, 1)
            
            # Current position
            if pts:
                cv2.circle(panel, pts[-1], 5, (255, 255, 255), -1)
        
        cv2.putText(panel, f"Coh: {self.phase_coherence:.2f}", (pw + 5, ph + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, f"Rot: {self.rotation_rate:.1f}°/f", (pw + 5, ph + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return QtGui.QImage(panel.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)