"""
Manifold Monitor Node (Deep Past Tracker)
-----------------------------------------
Reads the raw Takens cable (spectrum) from Geometric Neurons.
Isolates the rightmost region (the Deep Past / Scorched Manifold).
Measures the amplitude of this region over time to reveal 
sub-threshold "breathing" and structural resonance.
"""

import numpy as np
import cv2
from collections import deque
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ManifoldMonitorNode(BaseNode):
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(150, 80, 50)  # Rust/Amber color

    def __init__(self, n_neurons=5, right_region_start=96):
        super().__init__()
        self.node_title = "Deep Past Monitor"
        
        self.n_neurons = int(n_neurons)
        self.right_region_start = int(right_region_start)
        
        # Dynamically create spectrum inputs for the cables
        self.inputs = {f'neuron_{i}_cable': 'spectrum' for i in range(self.n_neurons)}
        self.outputs = {'manifold_img': 'image'}
        
        # State tracking: Rolling window of the deep past amplitude (last 250 steps)
        self.history_length = 256
        self.amplitude_history = [deque(np.zeros(self.history_length), maxlen=self.history_length) for _ in range(self.n_neurons)]
        
        self.display_image = np.zeros((128, 256, 3), dtype=np.uint8)
        self.colors = [
            (0, 255, 255),   # Cyan (Neuron 0)
            (255, 100, 255), # Pink (Neuron 1)
            (255, 255, 0),   # Yellow (Neuron 2)
            (100, 255, 100), # Green (Neuron 3)
            (100, 150, 255)  # Blue (Neuron 4)
        ]
        self.step_counter = 0

    def step(self):
        self.step_counter += 1
        
        for i in range(self.n_neurons):
            # Get the raw cable state array
            cable = self.get_blended_input(f'neuron_{i}_cable', 'first')
            
            if cable is not None and len(cable) > self.right_region_start:
                # Isolate the deep past (the rightmost side of the cable)
                deep_past = cable[self.right_region_start:]
                
                # Measure the physical energy/amplitude of this region
                # We use Max Absolute Amplitude to catch sharp resonance waves
                amplitude = np.max(np.abs(deep_past))
                self.amplitude_history[i].append(amplitude)
            else:
                self.amplitude_history[i].append(0.0)
                
        # Render UI every few frames
        if self.step_counter % 3 == 0:
            self._update_display()

    def _update_display(self):
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Auto-scale the Y-axis based on current global maximum
        global_max = 0.01  # Prevent div by zero
        for i in range(self.n_neurons):
            local_max = max(self.amplitude_history[i])
            if local_max > global_max:
                global_max = local_max
                
        display_max = global_max * 1.2  # 20% padding at top

        # Draw scrolling waveforms
        for i in range(self.n_neurons):
            hist_array = np.array(self.amplitude_history[i])
            pts = []
            for t_idx, amp in enumerate(hist_array):
                x = int((t_idx / self.history_length) * w)
                # Scale Y to fit height, invert for OpenCV (0 is top)
                y = int(h - 5 - (amp / display_max) * (h - 25))
                y = np.clip(y, 0, h - 1)
                pts.append((x, y))
                
            if len(pts) > 1:
                pts = np.array(pts, np.int32)
                cv2.polylines(img, [pts], False, self.colors[i % len(self.colors)], 1)
                
                # Draw a bright dot at the leading edge (current time)
                current_x, current_y = pts[-1]
                cv2.circle(img, (current_x, current_y), 3, self.colors[i % len(self.colors)], -1)

        # UI Overlays
        cv2.putText(img, f"Deep Past Amplitude (idx {self.right_region_start}+)", (5, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Peak: {display_max:.3f}", (w - 80, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        # Draw threshold reference lines (subdued gray)
        mid_y = int(h - 5 - (0.5 / display_max) * (h - 25))
        if 0 < mid_y < h:
            cv2.line(img, (0, mid_y), (w, mid_y), (50, 50, 50), 1, lineType=cv2.LINE_AA)

        self.display_image = img

    def get_output(self, port_name):
        if port_name == 'manifold_img':
            return self.display_image
        return None

    def get_display_image(self):
        h, w = self.display_image.shape[:2]
        rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Number of Neurons", "n_neurons", self.n_neurons, None),
            ("Right Region Start Index", "right_region_start", self.right_region_start, None)
        ]

    def set_config_options(self, options):
        if "n_neurons" in options:
            self.n_neurons = int(options["n_neurons"])
            self.inputs = {f'neuron_{i}_cable': 'spectrum' for i in range(self.n_neurons)}
            self.amplitude_history = [deque(np.zeros(self.history_length), maxlen=self.history_length) for _ in range(self.n_neurons)]
        if "right_region_start" in options:
            self.right_region_start = int(options["right_region_start"])