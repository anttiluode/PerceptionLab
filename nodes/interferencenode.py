
import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class InterferenceNode(BaseNode):
    NODE_CATEGORY = "Math"
    NODE_COLOR = QtGui.QColor(0, 150, 200)  # Cyan - Physics/Math color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Wave Interference"
        
        self.inputs = {
            'wave_a': 'image',
            'wave_b': 'image'
        }
        
        self.outputs = {
            'constructive': 'image',  # (A + B) - Amplification
            'destructive': 'image',   # |A - B| - Cancellation/Difference
            'moire_xor': 'image'      # Logical interference (Digital Moiré)
        }
        
        self.output_img = None
        self.display_mode = 'destructive' # What to show on the node itself
    
    def step(self):
        img_a = self.get_blended_input('wave_a', 'first')
        img_b = self.get_blended_input('wave_b', 'first')
        
        if img_a is None or img_b is None:
            return
            
        # Ensure dimensions match (Resize B to A if necessary)
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
            
        # Convert to float 0.0-1.0 for wave math
        wave_a = img_a.astype(np.float32) / 255.0
        wave_b = img_b.astype(np.float32) / 255.0
        
        # 1. Constructive Interference (Amplification)
        # Waves summing up. We normalize back to 0-1.
        constructive = (wave_a + wave_b) / 2.0
        
        # 2. Destructive Interference (Cancellation)
        # This shows the PHASE DIFFERENCE. If A and B are identical, this is black.
        # If they are perfectly out of phase, this is bright.
        destructive = np.abs(wave_a - wave_b)
        
        # 3. Moiré / XOR (Digital Logic Interference)
        # This highlights grid clashes specifically.
        # Logical XOR simulation using absolute difference logic emphasized
        moire = np.abs(wave_a - wave_b) * (wave_a + wave_b)
        
        # Store outputs (Convert back to uint8)
        self.out_constructive = (np.clip(constructive, 0, 1) * 255).astype(np.uint8)
        self.out_destructive = (np.clip(destructive, 0, 1) * 255).astype(np.uint8)
        self.out_moire = (np.clip(moire, 0, 1) * 255).astype(np.uint8)

        # Update display based on config
        if self.display_mode == 'constructive':
            self.output_img = self.out_constructive
        elif self.display_mode == 'moire':
            self.output_img = self.out_moire
        else:
            self.output_img = self.out_destructive

    def get_output(self, port_name):
        if port_name == 'constructive':
            return self.out_constructive
        elif port_name == 'destructive':
            return self.out_destructive
        elif port_name == 'moire_xor':
            return self.out_moire
        return None
        
    def get_display_image(self):
        return self.output_img

    def get_config_options(self):
        return [
            ("Display Mode", "display_mode", self.display_mode, ["destructive", "constructive", "moire"])
        ]
        
    def set_config_options(self, options):
        if "display_mode" in options:
            self.display_mode = options["display_mode"]