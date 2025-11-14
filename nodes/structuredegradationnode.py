"""
StructureDegradationNode - Simulates "fractal texture degradation"
----------------------------------------------------------------------
This is the "Damage" node. It simulates what happens when the
fractal structure of the information field breaks down.

This is your "floater" simulator.
It takes the "healthy" fractal maps and introduces "holes"
where the texture degrades and information is lost.

Consciousness (the Navigator) will fail to surf these regions.

Place this file in the 'nodes' folder
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class StructureDegradationNode(BaseNode):
    NODE_CATEGORY = "Fractal Substrate"
    NODE_COLOR = QtGui.QColor(150, 50, 50)  # "Damaged" red

    def __init__(self, degradation_rate=0.01, hole_size=10, degradation_threshold=0.3):
        super().__init__()
        self.node_title = "Structure Degradation"

        self.inputs = {
            'alignment_field': 'image',
            'complexity_map': 'image',
            'noise_field': 'image',
            'phase_structure': 'image',  # <-- THE CORRECT INPUT PORT
            'damage_control': 'signal', # Control rate externally
        }

        self.outputs = {
            'degraded_alignment_field': 'image',
            'degraded_complexity_map': 'image',
            'debug_mask': 'image',
        }

        # Configurable
        self.degradation_rate = float(degradation_rate)
        self.hole_size = int(hole_size)
        self.degradation_threshold = float(degradation_threshold)
        
        # Internal state
        self.grid_size = 256
        self.damage_mask = None # This is where the "floaters" are
        
        self.degraded_alignment = None
        self.degraded_complexity = None

    def _initialize_mask(self):
        self.damage_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

    def step(self):
        # 1. Get inputs
        alignment_field = self.get_blended_input('alignment_field', 'first')
        complexity_map = self.get_blended_input('complexity_map', 'first')
        damage_control = self.get_blended_input('damage_control', 'sum')

        # We also need to "get" the other inputs, even if we don't use
        # them in this simple "damage" logic, just so the node knows
        # it depends on them.
        self.get_blended_input('noise_field', 'first')
        self.get_blended_input('phase_structure', 'first')

        if alignment_field is None or complexity_map is None:
            return

        # 2. Initialize mask if needed
        if self.damage_mask is None or self.damage_mask.shape[0] != alignment_field.shape[0]:
            self.grid_size = alignment_field.shape[0]
            self._initialize_mask()
            
        rate = damage_control if damage_control is not None else self.degradation_rate

        # 3. "Degrade" the structure
        # Find a random point
        x, y = np.random.randint(0, self.grid_size, 2)
        
        # Check if this area is "interesting" (worth degrading)
        if alignment_field[y, x] > self.degradation_threshold:
            # Create a "hole" (a "floater")
            s = self.hole_size // 2
            cv2.circle(self.damage_mask, (x, y), s, 0.0, -1) # Set mask to 0

        # 4. Slowly "heal" the damage over time
        self.damage_mask += rate # Grow back slowly
        self.damage_mask = np.clip(self.damage_mask, 0.0, 1.0)
        
        # 5. Apply the damage mask to the fields
        self.degraded_alignment = alignment_field * self.damage_mask
        self.degraded_complexity = complexity_map * self.damage_mask

    def get_output(self, port_name):
        if port_name == 'degraded_alignment_field':
            return self.degraded_alignment
        if port_name == 'degraded_complexity_map':
            return self.degraded_complexity
        if port_name == 'debug_mask':
            return self.damage_mask
        return None

    def get_display_image(self):
        display_w, display_h = 256, 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)

        # Top: Degraded Alignment (What the surfer sees)
        if self.degraded_alignment is not None:
            alignment_u8 = (np.clip(self.degraded_alignment, 0, 1) * 255).astype(np.uint8)
            alignment_color = cv2.applyColorMap(alignment_u8, cv2.COLORMAP_JET)
            alignment_resized = cv2.resize(alignment_color, (display_w, display_h // 2))
            display[:display_h//2, :] = alignment_resized
        
        # Bottom: The Damage Mask (The "Floaters")
        if self.damage_mask is not None:
            mask_u8 = (np.clip(self.damage_mask, 0, 1) * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_u8, (display_w, display_h // 2))
            display[display_h//2:, :] = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'DEGRADED ALIGNMENT', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'DAMAGE MASK (FLOATERS)', (10, display_h//2 + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, display_w * 3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Degradation Rate (Heal)", "degradation_rate", self.degradation_rate, None),
            ("Hole Size (Pixels)", "hole_size", self.hole_size, None),
            ("Degradation Threshold", "degradation_threshold", self.degradation_threshold, None)
        ]