import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class InterferenceNodeOld(BaseNode):
    """
    WAVE INTERFERENCE & HETERODYNE REACTOR
    --------------------------------------
    Processes two wave fields.
    Added 'Heterodyne' mode: Multiplication of fields (Mixing).
    This reveals the BEAT FREQUENCY (The Moiré of the Gate).
    """
    NODE_CATEGORY = "Math"
    NODE_COLOR = QtGui.QColor(0, 150, 200)
    
    def __init__(self, display_mode='heterodyne'):
        super().__init__()
        self.node_title = "Wave Interference"
        self.inputs = {'wave_a': 'image', 'wave_b': 'image'}
        self.outputs = {
            'constructive': 'image',
            'destructive': 'image',
            'moire_xor': 'image',
            'heterodyne': 'image'   # <-- The New Magic
        }
        self.display_mode = display_mode
        self.out_heterodyne = None

    def step(self):
        img_a = self.get_blended_input('wave_a', 'first')
        img_b = self.get_blended_input('wave_b', 'first')
        if img_a is None or img_b is None: return
        
        # Normalize to 0.0 - 1.0 for math
        a = img_a.astype(np.float32) / 255.0
        b = img_b.astype(np.float32) / 255.0

        # 1. Standard Interference
        self.out_constructive = (np.clip(a + b, 0, 1) * 255).astype(np.uint8)
        self.out_destructive = (np.abs(a - b) * 255).astype(np.uint8)

        # 2. THE HETERODYNE (Multiplication)
        # This is the 'Beat' between the Large Gong and Small Bits.
        # It creates a new structure that highlights where they resonate.
        mixed = a * b 
        self.out_heterodyne = (np.clip(mixed * 2, 0, 1) * 255).astype(np.uint8) # Gain x2 for visibility

        # 3. Digital Moiré (XOR Logic)
        self.out_moire = (np.abs(a - b) * (a + b) * 255).astype(np.uint8)

        # Select display
        modes = {
            'constructive': self.out_constructive,
            'destructive': self.out_destructive,
            'heterodyne': self.out_heterodyne,
            'moire': self.out_moire
        }
        self.output_img = modes.get(self.display_mode, self.out_heterodyne)

    def get_output(self, port_name):
        outputs = {
            'constructive': self.out_constructive,
            'destructive': self.out_destructive,
            'moire_xor': self.out_moire,
            'heterodyne': self.out_heterodyne
        }
        return outputs.get(port_name)

    def get_config_options(self):
        return [("Display Mode", "display_mode", self.display_mode, ["heterodyne", "constructive", "destructive", "moire"])]