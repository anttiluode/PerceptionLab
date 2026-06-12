"""
Island Archival Node V2 (Exhaustion-Triggered Freeze)
-----------------------------------------------------
Implements the Island-Memory-Field dynamics. 
V2 Logic: It monitors the Skew Flux (heat) AND the ATP Metabolism (energy). 
When the system has high heat but runs out of ATP (< 0.2), it forces a phase-lock. 
It crystallizes the active vector into a deep memory pole to save it before the 
active cortex shuts down.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class IslandArchivalNodeV2(BaseNode):
    NODE_CATEGORY = "Memory"
    NODE_COLOR = QtGui.QColor(40, 80, 150) # Deep Ocean Blue

    def __init__(self, dim=16, freeze_threshold=0.08):
        super().__init__()
        self.node_title = "Island Archival V2"
        self.inputs = {
            'vector_in': 'spectrum',
            'skew_flux': 'signal',
            'gamma_thaw': 'signal',
            'atp_level': 'signal' # V2: Metabolism Input
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
        atp = self.get_blended_input('atp_level', 'sum')
        if atp is None: atp = 1.0

        if vec is None: return

        v = np.array(vec, dtype=np.float32)
        if len(v) > self.dim: v = v[:self.dim]
        elif len(v) < self.dim: v = np.pad(v, (0, self.dim - len(v)))

        if self.cooldown > 0:
            self.cooldown -= 1

        # V2 FREEZE LOGIC: High Heat + Exhausted ATP
        if flux > self.freeze_threshold and atp < 0.2 and self.cooldown == 0:
            for island in self.islands:
                island['depth'] += 1.0

            self.islands.append({'vector': v.copy(), 'depth': 1.0})
            self.cooldown = 40 
            
            if len(self.islands) > 10:
                self.islands.pop(0)

        self.memory_readout = np.zeros(self.dim, dtype=np.float32)
        for island in self.islands:
            effective_depth = max(0.1, island['depth'] - gamma * 2.0)
            suppression = 1.0 / (effective_depth ** 2)
            self.memory_readout += island['vector'] * suppression

        max_val = np.max(np.abs(self.memory_readout)) + 1e-9
        self.memory_readout /= max_val

        self._render_islands(flux, gamma, atp)

    def _render_islands(self, flux, gamma, atp):
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

        # V2: Flash red boundary only when actively exhausted and freezing
        if self.cooldown > 30 and atp < 0.2:
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
            ("Freeze Threshold", "freeze_threshold", self.freeze_threshold, "float")
        ]

    def set_config_options(self, options):
        if "dim" in options: self.dim = int(options["dim"])
        if "freeze_threshold" in options: self.freeze_threshold = float(options["freeze_threshold"])