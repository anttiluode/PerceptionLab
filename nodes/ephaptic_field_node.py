"""
Ephaptic Field Node - The Galactic Filament
-------------------------------------------
Summates spike fields from all neurons with spatial decay.
The field then perturbs every neuron's cable.
This is the "control parameter" from Pinotsis & Miller.
"""

import numpy as np
import cv2
from collections import deque
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class EphapticFieldNode(BaseNode):
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(80, 50, 180)  # Purple-blue for field

    def __init__(self, n_neurons=5, spatial_decay=0.3, field_decay=0.95):
        super().__init__()
        self.node_title = "Ephaptic Field"
        
        # Dynamic inputs (created based on n_neurons)
        self.n_neurons = int(n_neurons)
        self.inputs = {}
        self.outputs = {}
        
        for i in range(self.n_neurons):
            self.inputs[f'neuron_{i}_ephaptic'] = 'signal'
            self.inputs[f'neuron_{i}_spike_pos'] = 'signal'
            self.inputs[f'neuron_{i}_resonance'] = 'signal'
        
        # Outputs: field value for each neuron
        for i in range(self.n_neurons):
            self.outputs[f'field_to_{i}'] = 'signal'
        
        # Also output the full field for visualization
        self.outputs['field_spectrum'] = 'spectrum'
        self.outputs['field_image'] = 'image'
        
        # Field parameters
        self.spatial_decay = float(spatial_decay)  # How fast field drops with distance
        self.field_decay = float(field_decay)      # Temporal decay of field
        self.johnson_nyquist_sigma = 0.03          # Field noise
        
        # Neuron positions (assumed in a line for now)
        self.positions = np.linspace(0, 1, self.n_neurons)
        
        # Current field state
        self.field_state = np.zeros(self.n_neurons, dtype=np.float32)
        self.field_history = deque(maxlen=100)
        
        # Visualization
        self.display_image = np.zeros((128, 256, 3), dtype=np.uint8)
        
        # Update node title to show neuron count
        self._update_title()

    def _update_title(self):
        self.node_title = f"Ephaptic Field ({self.n_neurons} neurons)"

    def step(self):
        # 1. Gather inputs from all neurons
        contributions = np.zeros(self.n_neurons, dtype=np.float32)
        
        for i in range(self.n_neurons):
            ephaptic = self.get_blended_input(f'neuron_{i}_ephaptic', 'sum') or 0.0
            spike_pos = self.get_blended_input(f'neuron_{i}_spike_pos', 'sum') or 0.0
            resonance = self.get_blended_input(f'neuron_{i}_resonance', 'sum') or 0.0
            
            # A neuron's field contribution depends on its spike and position
            # Proximal spikes (high pos) create stronger fields
            contributions[i] = ephaptic * (0.5 + spike_pos * 0.5)
        
        # 2. Apply spatial coupling (field from each neuron affects all others)
        new_field = np.zeros(self.n_neurons, dtype=np.float32)
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                distance = abs(self.positions[i] - self.positions[j])
                # Exponential spatial decay
                coupling = np.exp(-distance / self.spatial_decay)
                new_field[i] += contributions[j] * coupling
        
        # 3. Temporal decay (field persists)
        self.field_state = self.field_state * self.field_decay + new_field * (1 - self.field_decay)
        
        # 4. Add Johnson-Nyquist noise to the field itself
        self.field_state += np.random.normal(0, self.johnson_nyquist_sigma, self.n_neurons)
        
        # Clamp to reasonable range
        self.field_state = np.clip(self.field_state, -1.0, 1.0)
        
        # 5. Update history for visualization
        self.field_history.append(self.field_state.copy())
        
        # 6. Update visualization
        self._update_display()

    def _update_display(self):
        img = np.zeros((128, 256, 3), dtype=np.uint8)
        h, w = 128, 256
        
        # Draw field as a waveform
        if len(self.field_history) > 1:
            history_array = np.array(list(self.field_history))
            # Plot each neuron's field over time
            for i in range(self.n_neurons):
                color = (int(100 + 155 * i / self.n_neurons), 
                        50, 
                        int(200 - 150 * i / self.n_neurons))
                
                for t in range(1, len(history_array)):
                    x1 = int((t-1) / len(history_array) * w)
                    x2 = int(t / len(history_array) * w)
                    y1 = int(h/2 - history_array[t-1, i] * 50)
                    y2 = int(h/2 - history_array[t, i] * 50)
                    y1 = np.clip(y1, 0, h-1)
                    y2 = np.clip(y2, 0, h-1)
                    cv2.line(img, (x1, y1), (x2, y2), color, 1)
        
        # Draw current field as bar chart at bottom
        bar_w = w // self.n_neurons
        for i, val in enumerate(self.field_state):
            x = i * bar_w
            bar_h = int(abs(val) * 40)
            if val >= 0:
                y = h - 10 - bar_h
                color = (0, int(255 * val), 0)
            else:
                y = h - 10
                color = (int(255 * -val), 0, 0)
            cv2.rectangle(img, (x, y), (x + bar_w - 2, h - 10), color, -1)
        
        # Draw neuron positions
        for i, pos in enumerate(self.positions):
            x = int(pos * w)
            cv2.circle(img, (x, h//2), 4, (255, 255, 255), -1)
        
        cv2.putText(img, f"FIELD", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        self.display_image = img

    def get_output(self, port_name):
        if port_name.startswith('field_to_'):
            idx = int(port_name.split('_')[-1])
            if idx < len(self.field_state):
                return float(self.field_state[idx])
            return 0.0
        if port_name == 'field_spectrum':
            return self.field_state.copy()
        if port_name == 'field_image':
            return self.display_image
        return None

    def get_display_image(self):
        h, w = self.display_image.shape[:2]
        rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Number of Neurons", "n_neurons", self.n_neurons, None),
            ("Spatial Decay", "spatial_decay", self.spatial_decay, "float"),
            ("Field Decay", "field_decay", self.field_decay, "float"),
            ("Field Noise", "johnson_nyquist_sigma", self.johnson_nyquist_sigma, "float"),
        ]

    def set_config_options(self, options):
        if "n_neurons" in options:
            self.n_neurons = int(options["n_neurons"])
            self.positions = np.linspace(0, 1, self.n_neurons)
            self.field_state = np.zeros(self.n_neurons, dtype=np.float32)
            # Rebuild inputs/outputs
            self.inputs = {}
            self.outputs = {}
            for i in range(self.n_neurons):
                self.inputs[f'neuron_{i}_ephaptic'] = 'signal'
                self.inputs[f'neuron_{i}_spike_pos'] = 'signal'
                self.inputs[f'neuron_{i}_resonance'] = 'signal'
            for i in range(self.n_neurons):
                self.outputs[f'field_to_{i}'] = 'signal'
            self.outputs['field_spectrum'] = 'spectrum'
            self.outputs['field_image'] = 'image'
            self._update_title()
        if "spatial_decay" in options:
            self.spatial_decay = float(options["spatial_decay"])
        if "field_decay" in options:
            self.field_decay = float(options["field_decay"])
        if "johnson_nyquist_sigma" in options:
            self.johnson_nyquist_sigma = float(options["johnson_nyquist_sigma"])