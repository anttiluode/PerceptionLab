# dendriticforestnode.py
# The first node that simulates real dendritic trees + volume neuromodulators
# Electric layer (fast, local) + Chemical sky (slow, global)

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.fft import fft2, fftshift

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class DendriticForestNode(BaseNode):
    NODE_CATEGORY = "Biology"
    NODE_TITLE = "Dendritic Forest"
    NODE_COLOR = QtGui.QColor(120, 0, 180)  # Deep purple – the color of real cortex

    def __init__(self):
        super().__init__()
        
        S = 256  # Forest size – can be 128, 256, 512
        self.S = S
        
        self.inputs = {
            'electric_drive': 'spectrum',     # Fast input (sound, vision, touch)
            'global_modulator': 'signal',     # Cortical input (dopamine, serotonin, attention)
            'reset': 'signal'
        }
        
        self.outputs = {
            'electric_field': 'image',        # Fast dendritic activity
            'chemical_sky': 'image',          # Slow neuromodulator clouds
            'tree_health': 'image',           # Vesicle / energy map
            'combined_view': 'image',         # Electric + Chemical overlay
            'mean_dopamine': 'signal'
        }

        # === ELECTRIC LAYER – 16 independent dendritic trees ===
        self.trees_x = np.random.randint(30, S-30, 16)
        self.trees_y = np.random.randint(30, S-30, 16)
        self.dendritic_mask = np.zeros((S, S), dtype=np.float32)
        
        # Grow realistic dendritic trees
        for x, y in zip(self.trees_x, self.trees_y):
            cv2.circle(self.dendritic_mask, (x, y), 4, 1.0, -1)  # soma
            for angle in range(0, 360, 30):
                length = np.random.randint(20, 60)
                dx = int(length * np.cos(np.radians(angle)))
                dy = int(length * np.sin(np.radians(angle)))
                cv2.line(self.dendritic_mask, (x, y), (x+dx, y+dy), 0.7, 2)

        # Distance from nearest dendrite – used for electric propagation speed
        self.distance_field = distance_transform_edt(1 - self.dendritic_mask)

        # Electric potential (fast)
        self.electric = np.zeros((S, S), dtype=np.float32)

        # === CHEMICAL LAYER – the "sky" ===
        self.dopamine = np.zeros((S, S), dtype=np.float32)
        self.serotonin = np.zeros((S, S), dtype=np.float32)
        self.norepinephrine = np.zeros((S, S), dtype=np.float32)

        # Tree health (internal neurotransmitter stores)
        self.vesicles = np.ones((S, S), dtype=np.float32)

        # Parameters
        self.electric_speed = 0.15
        self.chem_release = 0.08
        self.chem_decay = 0.015
        self.diffusion = 2.0

    def step(self):
        drive = self.get_blended_input('electric_drive') or np.zeros(16)
        mod = self.get_blended_input('global_modulator') or 0.0
        reset = self.get_blended_input('reset')

        if reset and reset > 0.5:
            self.electric[:] = 0
            self.dopamine[:] = 0
            self.serotonin[:] = 0
            self.norepinephrine[:] = 0
            return

        # 1. Electric wave – propagates faster near dendrites
        speed_map = 1.0 / (1.0 + self.distance_field * 0.1)
        self.electric += self.electric_speed * speed_map
        
        # Inject drive at tree roots
        for i, (x, y) in enumerate(zip(self.trees_x, self.trees_y)):
            if i < len(drive):
                self.electric[y-5:y+5, x-5:x+5] += drive[i] * 0.3

        # Electric decay
        self.electric *= 0.97

        # 2. Chemical release – only where electric activity is high
        active = self.electric > 0.5
        self.dopamine[active] += self.chem_release * (1.0 + mod)        # attention
        self.serotonin[active] += self.chem_release * 0.7
        self.norepinephrine[active] += self.chem_release * 1.2

        # 3. Chemical diffusion + decay (the sky moves slowly)
        for chem in [self.dopamine, self.serotonin, self.norepinephrine]:
            chem[:] = gaussian_filter(chem, sigma=self.diffusion)
            chem *= (1.0 - self.chem_decay)

        # 4. Chemical modulation of electric layer
        total_chem = (self.dopamine * 1.5 + self.serotonin * 0.8 + self.norepinephrine * 2.0)
        self.electric += total_chem * 0.05   # chemicals boost electric activity
        self.electric = np.clip(self.electric, 0, 2.0)

        # 5. Vesicle depletion / recovery
        self.vesicles[active] -= 0.08
        self.vesicles += 0.005
        self.vesicles = np.clip(self.vesicles, 0, 1)

    def get_output(self, port_name):
        if port_name == 'electric_field':
            return np.clip(self.electric * 80, 0, 255).astype(np.uint8)
        elif port_name == 'chemical_sky':
            sky = np.stack([self.dopamine*300, self.serotonin*300, self.norepinephrine*300], axis=-1)
            return np.clip(sky, 0, 255).astype(np.uint8)
        elif port_name == 'tree_health':
            return (self.vesicles * 255).astype(np.uint8)
        elif port_name == 'combined_view':
            elec = self.electric / self.electric.max() if self.electric.max() > 0 else self.electric
            chem = np.stack([self.dopamine, self.serotonin*0.5, self.norepinephrine], axis=-1)
            chem = chem / (chem.max() + 1e-8)
            combined = 0.6 * elecmap(elec)[:,:,:3] + 0.4 * chem
            return (np.clip(combined, 0, 1) * 255).astype(np.uint8)
        elif port_name == 'mean_dopamine':
            return float(np.mean(self.dopamine))
        return None

    def get_display_image(self):
        # Beautiful three-layer view
        display = np.zeros((self.S, self.S*3, 3), dtype=np.uint8)
        
        # Left: Electric activity (cyan-white)
        e = np.clip(self.electric * 60, 0, 255).astype(np.uint8)
        display[:, :self.S, 0] = e      # Blue channel
        display[:, :self.S, 1] = e//2
        display[:, :self.S, 2] = e//3
        
        # Middle: Chemical sky (RGB = Dopamine/Serotonin/Norepinephrine)
        c = np.stack([self.dopamine*400, self.serotonin*400, self.norepinephrine*400], axis=-1)
        display[:, self.S:self.S*2] = np.clip(c, 0, 255).astype(np.uint8)
        
        # Right: Tree health (green = alive, red = exhausted)
        h = self.vesicles
        display[:, self.S*2:, 0] = np.clip((1-h)*255, 0, 255).astype(np.uint8)  # red when dead
        display[:, self.S*2:, 1] = np.clip(h*255, 0, 255).astype(np.uint8)      # green when alive
        
        cv2.putText(display, "ELECTRIC", (10, 20), 0, 0.7, (255,255,255), 2)
        cv2.putText(display, "CHEMICAL", (self.S+10, 20), 0, 0.7, (255,255,255), 2)
        cv2.putText(display, "HEALTH", (self.S*2+10, 20), 0, 0.7, (255,255,255), 2)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0], 
                           display.shape[1]*3, QtGui.QImage.Format.Format_RGB888)