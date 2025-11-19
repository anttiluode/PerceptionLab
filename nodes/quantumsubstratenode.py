"""
Quantum Substrate Node (Stable)
-------------------------------
Simulates a Complex Ginzburg-Landau field.
This creates self-organizing spiral waves and quantum turbulence.
It provides the "Field Energy" signal that the Observer needs to wake up.

Outputs:
- field_energy: The total activity of the vacuum (feeds the Observer).
- field_image: Visual representation of the quantum foam.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class QuantumSubstrateNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(80, 0, 180) # Deep Indigo
    
    def __init__(self, size=64):
        super().__init__()
        self.node_title = "Quantum Substrate"
        
        self.inputs = {
            'perturbation': 'image',   # Optional: Poke the field
            'time_scale': 'signal'     # Speed of simulation
        }
        
        self.outputs = {
            'field_image': 'image',
            'field_energy': 'signal',  # Connect this to Observer!
            'entropy': 'signal'
        }
        
        self.size = int(size)
        self.dt = 0.1
        
        # --- Physics Parameters (Ginzburg-Landau) ---
        self.alpha = 0.5
        self.beta = 2.0
        self.diffusion = 0.5
        
        # --- Initialize Field ---
        # Complex field A = u + iv
        self.u = np.random.randn(self.size, self.size).astype(np.float32) * 0.1
        self.v = np.random.randn(self.size, self.size).astype(np.float32) * 0.1
        
        # Pre-calculate Laplacian kernel
        self.kernel = np.array([[0.05, 0.2, 0.05],
                                [0.2, -1.0, 0.2],
                                [0.05, 0.2, 0.05]], dtype=np.float32)
                                
        # Initialize output variables (Prevents AttributeError)
        self.field_energy_val = 0.0
        self.entropy_val = 0.0
        self.display_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def step(self):
        # 1. Get Inputs
        perturb = self.get_blended_input('perturbation', 'mean')
        speed_sig = self.get_blended_input('time_scale', 'sum')
        
        dt = self.dt * (1.0 + (speed_sig or 0.0))
        
        # 2. Apply Perturbation (if any)
        if perturb is not None:
            if perturb.shape != (self.size, self.size):
                perturb = cv2.resize(perturb, (self.size, self.size))
            if perturb.ndim == 3:
                perturb = np.mean(perturb, axis=2)
            
            # Perturbation adds energy to real component 'u'
            self.u += (perturb - 0.5) * 0.5 * dt

        # 3. Physics: Complex Ginzburg-Landau Equation
        # A_t = A + (1 + i*alpha)*Laplacian(A) - (1 + i*beta)*|A|^2*A
        
        # Laplacian
        lu = cv2.filter2D(self.u, -1, self.kernel)
        lv = cv2.filter2D(self.v, -1, self.kernel)
        
        # Magnitude squared |A|^2
        mag2 = self.u**2 + self.v**2
        
        # Evolution terms
        du = self.u + (lu - self.alpha * lv) - mag2 * (self.u - self.beta * self.v)
        dv = self.v + (lv + self.alpha * lu) - mag2 * (self.v + self.beta * self.u)
        
        # Update
        self.u += du * dt
        self.v += dv * dt
        
        # 4. Calculate Outputs
        # Energy = Total magnitude of the field
        self.field_energy_val = float(np.mean(mag2)) * 10.0
        
        # Entropy = Variance of the field
        self.entropy_val = float(np.var(mag2))
        
        # 5. Visualization
        # Normalize
        vis = np.sqrt(mag2)
        vis = np.clip(vis * 2.0, 0, 1)
        
        # Convert to RGB
        img_u8 = (vis * 255).astype(np.uint8)
        self.display_img = cv2.applyColorMap(img_u8, cv2.COLORMAP_TWILIGHT)

    def get_output(self, port_name):
        if port_name == 'field_energy':
            return self.field_energy_val
        elif port_name == 'entropy':
            return self.entropy_val
        elif port_name == 'field_image':
            # Return normalized magnitude for other nodes
            mag = np.sqrt(self.u**2 + self.v**2)
            return np.clip(mag, 0, 1).astype(np.float32)
        return None

    def get_display_image(self):
        img_resized = cv2.resize(self.display_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        img_resized = np.ascontiguousarray(img_resized)
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size", "size", self.size, None),
            ("Alpha", "alpha", self.alpha, None),
            ("Beta", "beta", self.beta, None)
        ]