"""
Island Archival Node (The Memory Deep-Freeze)
---------------------------------------------
Implements the Island-Memory-Field dynamics. Takes active working memory
(a vector) and the local Skew Flux (thermodynamic heat). When the flux
crosses a threshold, it 'freezes' the vector into a complex pole (Island)
and pushes older memories deeper into the complex plane, naturally
suppressing interference without destroying weights.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class IslandArchivalNode(BaseNode):
    NODE_CATEGORY = "Memory"
    NODE_COLOR = QtGui.QColor(40, 80, 150) # Deep Ocean Blue

    # FIX 1: Change argument to freeze_threshold to match the variable
    def __init__(self, dim=16, freeze_threshold=2.0):
        super().__init__()
        self.node_title = "Island Archival (Freeze)"
        self.inputs = {
            'vector_in': 'spectrum',
            'skew_flux': 'signal',
            'gamma_thaw': 'signal' 
        }
        self.outputs = {
            'memory_readout': 'spectrum',
            'island_viz': 'image'
        }
        self.dim = int(dim)
        self.freeze_threshold = float(freeze_threshold)

        self.islands = []
        self.memory_readout = np.zeros(self.dim, dtype=np.float32)
        self.display_img = np.zeros((128, 256, 3), dtype=np.uint8)
        self.cooldown = 0 

    def step(self):
        vec = self.get_blended_input('vector_in', 'first')
        flux = self.get_blended_input('skew_flux', 'sum') or 0.0
        gamma = self.get_blended_input('gamma_thaw', 'sum') or 0.0

        if vec is None:
            return

        v = np.array(vec, dtype=np.float32)
        if len(v) > self.dim: v = v[:self.dim]
        elif len(v) < self.dim: v = np.pad(v, (0, self.dim - len(v)))

        if self.cooldown > 0:
            self.cooldown -= 1

        if flux > self.freeze_threshold and self.cooldown == 0:
            for island in self.islands:
                island['depth'] += 1.0

            self.islands.append({'vector': v.copy(), 'depth': 1.0})
            self.cooldown = 20 
            
            if len(self.islands) > 10:
                self.islands.pop(0)

        self.memory_readout = np.zeros(self.dim, dtype=np.float32)
        for island in self.islands:
            effective_depth = max(0.1, island['depth'] - gamma * 2.0)
            suppression = 1.0 / (effective_depth ** 2)
            self.memory_readout += island['vector'] * suppression

        max_val = np.max(np.abs(self.memory_readout)) + 1e-9
        self.memory_readout /= max_val

        self._render_islands(flux, gamma)

    def _render_islands(self, flux, gamma):
        h, w = 128, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.line(img, (0, 20), (w, 20), (255, 255, 255), 1)
        cv2.putText(img, "Rajapinta (Active)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        if len(self.islands) > 0:
            x_spacing = w / max(len(self.islands), 1)
            for i, island in enumerate(self.islands):
                x = int((i + 0.5) * x_spacing)
                effective_depth = max(0.1, island['depth'] - gamma * 2.0)
                y = int(20 + effective_depth * 15)
                y = min(y, h - 10)

                intensity = int(255 / max(1, effective_depth))
                color = (intensity, int(intensity*0.8), 50)
                
                cv2.circle(img, (x, y), 6, color, -1)
                cv2.line(img, (x, 20), (x, y), (50, 50, 50), 1) 

        if self.cooldown > 15:
            cv2.rectangle(img, (0,0), (w-1, h-1), (0, 0, 255), 2)

        self.display_img = img

    def get_output(self, port_name):
        if port_name == 'memory_readout': return self.memory_readout
        if port_name == 'island_viz': return self.display_img
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Vector Dim", "dim", self.dim, None),
            # FIX 2: Match the dictionary key to the variable exactly
            ("Freeze Threshold", "freeze_threshold", self.freeze_threshold, "float")
        ]

    def set_config_options(self, options):
        if "dim" in options:
            self.dim = int(options["dim"])
        # FIX 3: Catch the exact updated key
        if "freeze_threshold" in options:
            self.freeze_threshold = float(options["freeze_threshold"])