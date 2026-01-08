import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

# BaseNode injection
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {} 
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name): return None

class NeuroCrystalNode(BaseNode):
    """
    Neuro-Crystal Node (Plastic Spacetime)
    --------------------------------------
    A grid where the 'Space' itself learns.
    - Waves propagate through the lattice.
    - The lattice connections (weights) get stronger where waves flow.
    - VISUALIZES the hidden 'Riverbeds' of energy.
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(180, 0, 100) # Magenta (Biology + Physics)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Neuro Crystal"
        
        self.inputs = {
            'drive_signal': 'spectrum',
        }
        
        self.outputs = {
            'activity_view': 'image', # The Waves
            'network_view': 'image',  # The Hidden Connections (Veins)
            'combined_view': 'image'  # Both
        }
        
        self.config = {
            'resolution': 512,    # Keep low (32x32) to see individual links
            'plasticity': 0.1,   # How fast the grid rewires
            'decay': 0.02,       # How fast unused paths fade
            'diffusion': 0.2     # Wave speed
        }
        
        self._output_values = {}
        self._init_grid()

    def _init_grid(self):
        res = self.config['resolution']
        
        # The Activity Grid (The Water)
        self.grid = np.zeros((res, res), dtype=np.float32)
        
        # The Connection Grids (The Pipes)
        # horizontal_links[y, x] connects (x,y) to (x+1,y)
        self.h_links = np.ones((res, res), dtype=np.float32) * 0.5
        # vertical_links[y, x] connects (x,y) to (x,y+1)
        self.v_links = np.ones((res, res), dtype=np.float32) * 0.5

    # --- Compatibility ---
    def get_input(self, name):
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value): self._output_values[name] = value
    def get_output(self, name): return self._output_values.get(name, None)
    # ---------------------

    def step(self):
        res = self.config['resolution']
        
        # 1. Input Injection (Inject into center)
        stim = self.get_input('drive_signal')
        if stim is not None:
            stim_vec = np.array(stim, dtype=np.float32).flatten()
            center_start = res // 2 - 2
            # Inject a small patch
            if len(stim_vec) > 0:
                val = np.mean(stim_vec) * 5.0
                self.grid[center_start:center_start+4, center_start:center_start+4] += val

        # 2. Physics (Anisotropic Diffusion)
        # Waves flow easier where links are strong
        
        # Shift grids to get neighbors
        left = np.roll(self.grid, 1, axis=1)
        right = np.roll(self.grid, -1, axis=1)
        up = np.roll(self.grid, 1, axis=0)
        down = np.roll(self.grid, -1, axis=0)
        
        # Calculate Flow based on Link Strength
        # Flow = (Neighbor - Self) * Link_Strength
        # We approximate link strength between cells as average of their local links
        
        # Horizontal flow
        h_flow_in = (left - self.grid) * np.roll(self.h_links, 1, axis=1)
        h_flow_out = (right - self.grid) * self.h_links
        
        # Vertical flow
        v_flow_in = (up - self.grid) * np.roll(self.v_links, 1, axis=0)
        v_flow_out = (down - self.grid) * self.v_links
        
        # Update Grid
        diffusion = self.config['diffusion']
        self.grid += (h_flow_in + h_flow_out + v_flow_in + v_flow_out) * diffusion
        
        # Decay Activity
        self.grid *= 0.95
        
        # 3. Hebbian Learning (Plasticity)
        # If flux is high between nodes, strengthen link
        # "Rivers carve deeper channels"
        
        lr = self.config['plasticity']
        
        # Calculate Flux (Absolute difference)
        h_flux = np.abs(self.grid - right)
        v_flux = np.abs(self.grid - down)
        
        # Strengthen links with high flux
        self.h_links += h_flux * lr
        self.v_links += v_flux * lr
        
        # 4. Decay / Homeostasis (The Sculpting)
        decay = self.config['decay']
        self.h_links *= (1.0 - decay)
        self.v_links *= (1.0 - decay)
        
        # Clip
        np.clip(self.h_links, 0, 1.0, out=self.h_links)
        np.clip(self.v_links, 0, 1.0, out=self.v_links)
        np.clip(self.grid, -10, 10, out=self.grid)

        # 5. Render
        self._render_views()

    def _render_views(self):
        res = self.config['resolution']
        scale = 8 # Zoom factor for visibility
        h, w = res * scale, res * scale
        
        # --- A. Activity View (The Water) ---
        # Normalize -1..1 to 0..255
        act_norm = np.clip((self.grid + 1.0) / 2.0, 0, 1)
        act_img = (act_norm * 255).astype(np.uint8)
        act_img = cv2.applyColorMap(act_img, cv2.COLORMAP_OCEAN)
        act_img = cv2.resize(act_img, (w, h), interpolation=cv2.INTER_NEAREST)
        self.set_output('activity_view', act_img)
        
        # --- B. Network View (The Veins) ---
        net_img = np.zeros((h, w, 3), dtype=np.float32)
        
        # We draw lines for links > threshold
        thresh = 0.2
        
        # Iterate grid (Slow in python, but grid is small 32x32)
        for y in range(res):
            for x in range(res):
                cx, cy = x * scale + scale//2, y * scale + scale//2
                
                # Right Link
                if x < res - 1:
                    weight = float(self.h_links[y, x])
                    if weight > thresh:
                        color = (0.0, weight, weight*0.5) # Greenish
                        cv2.line(net_img, (cx, cy), (cx+scale, cy), color, max(1, int(weight*3)))
                        
                # Down Link
                if y < res - 1:
                    weight = float(self.v_links[y, x])
                    if weight > thresh:
                        color = (0.0, weight, weight*0.5)
                        cv2.line(net_img, (cx, cy), (cx, cy+scale), color, max(1, int(weight*3)))

        self.set_output('network_view', net_img)
        
        # --- C. Combined ---
        combined = cv2.addWeighted(act_img.astype(np.float32)/255.0, 0.4, net_img, 1.0, 0)
        self.set_output('combined_view', combined)