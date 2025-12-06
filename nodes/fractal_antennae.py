"""
Fractal Antenna Node
--------------------
Concept: DNA as a geometric receiver.
1. Takes a DNA Vector (Information).
2. Folds it into a Fractal Path (Geometry).
3. Overlays it on a Field (Spectrum/Image).
4. Measures 'Resonance' (how much field energy aligns with the path).

Hypothesis: Specific DNA shapes are 'tuned' to receive specific environmental frequencies.
"""

import numpy as np
import cv2

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None

class FractalAntenna2Node(BaseNode):
    NODE_CATEGORY = "Quantum Biology"
    NODE_COLOR = QtGui.QColor(100, 200, 255) # Electric Blue

    def __init__(self):
        super().__init__()
        self.node_title = "Fractal Antenna"
        
        self.inputs = {
            'dna_in': 'spectrum',      # The Tuning Parameter
            'field_in': 'image'        # The Environment (EEG Hologram)
        }
        
        self.outputs = {
            'reception_strength': 'signal', # How well it hears
            'antenna_view': 'image',        # Visual of the wire
            'tuned_signal': 'spectrum'      # The data it picked up
        }
        
        self.display = np.zeros((256, 256, 3), dtype=np.uint8)
        self.points = []

    def step(self):
        # 1. Get Inputs
        dna = self.get_blended_input('dna_in', 'mean')
        field = self.get_blended_input('field_in', 'mean')
        
        if dna is None: return
        
        # 2. Construct Fractal Path from DNA
        # We use the DNA to drive a "Turtle Graphics" style walker
        # or a chaotic attractor to generate a conductive path.
        
        # Reset canvas
        h, w = 256, 256
        if field is not None:
            h, w = field.shape[:2]
            
        # Generate Path
        # Interpret DNA as a series of turns and lengths
        x, y = w//2, h//2
        angle = 0.0
        path_points = []
        
        # Normalize DNA to usable ranges
        turns = (dna - 0.5) * 4.0 * np.pi # -2pi to 2pi
        lengths = np.abs(dna) * 20.0 + 2.0
        
        # Walk
        for i in range(len(dna)):
            angle += turns[i]
            dist = lengths[i]
            
            nx = x + np.cos(angle) * dist
            ny = y + np.sin(angle) * dist
            
            # Wrap around (Toroidal Antenna)
            nx = nx % w
            ny = ny % h
            
            # If we wrapped, break the line segment visually, but logically keep point
            if abs(nx - x) < 50 and abs(ny - y) < 50:
                path_points.append( ((int(x), int(y)), (int(nx), int(ny))) )
            
            x, y = nx, ny

        # 3. Calculate Resonance (Reception)
        total_signal = 0.0
        signal_spectrum = []
        
        if field is not None:
            # Convert field to grayscale float if needed
            if len(field.shape) == 3:
                field_gray = np.mean(field, axis=2)
            else:
                field_gray = field
            
            # Normalize field 0..1
            if field_gray.max() > 0:
                field_gray /= field_gray.max()
                
            # Integrate field intensity along the path
            for p1, p2 in path_points:
                # Sample the field at the midpoint of the segment
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                val = field_gray[int(my), int(mx)]
                total_signal += val
                signal_spectrum.append(val)
        else:
            total_signal = 0.0
            signal_spectrum = np.zeros(len(dna))

        # 4. Visualization
        self.display.fill(0)
        
        # Draw Field (faint background)
        if field is not None:
            # Dim the field so we can see the antenna
            bg = (cv2.resize(field, (w, h)) * 0.3).astype(np.uint8)
            if len(bg.shape) == 2: bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            self.display = bg
        
        # Draw Antenna (Glowing Wire)
        # Color based on signal strength at that segment
        for i, (p1, p2) in enumerate(path_points):
            sig = signal_spectrum[i] if i < len(signal_spectrum) else 0
            
            # Brightness = Signal
            intensity = int(100 + sig * 155)
            color = (intensity, intensity, 50) # Yellow-ish
            
            cv2.line(self.display, p1, p2, color, 2)
            cv2.circle(self.display, p1, 2, (0, 255, 255), -1)

        # 5. Outputs
        self.set_output('reception_strength', total_signal)
        self.set_output('antenna_view', self.display)
        self.set_output('tuned_signal', np.array(signal_spectrum))

    def get_output(self, name):
        if name == 'reception_strength': return self.get_output_helper('reception_strength')
        if name == 'antenna_view': return self.display
        if name == 'tuned_signal': return np.array(self.points) # placeholder
        return None
    
    # Helper for the set/get pattern if your base node needs it
    def set_output(self, name, val):
        if not hasattr(self, '_outs'): self._outs = {}
        self._outs[name] = val
        
    def get_output_helper(self, name):
        if not hasattr(self, '_outs'): return None
        return self._outs.get(name)