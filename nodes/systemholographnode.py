"""
System Holograph Node
---------------------
The "Macroscope" for the Genesis System.
Fuses Matter (Body), Physics (Field), and Mind (Observer) into a single
hyperspectral visualization.

- Red Channel   : Morphological Structure (The Body)
- Green Channel : Quantum Field / Turbulence (The Physics)
- Blue Channel  : Observer Attention / Prediction Error (The Mind)

Also renders a Phase Space Attractor (Entropy vs Free Energy) overlay
to visualize the system's stability regime.
"""

import numpy as np
import cv2
from collections import deque
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SystemHolographNode(BaseNode):
    NODE_CATEGORY = "Visualization"
    NODE_COLOR = QtGui.QColor(200, 200, 200) # Silver

    def __init__(self):
        super().__init__()
        self.node_title = "System Holograph"
        
        self.inputs = {
            'body_structure': 'image',   # From ResonanceMorphogenesis (Red)
            'quantum_field': 'image',    # From QuantumSubstrate (Green)
            'mind_attention': 'image',   # From SelfOrganizingObserver (Blue)
            'system_entropy': 'signal',  # X-axis of Phase Plot
            'free_energy': 'signal'      # Y-axis of Phase Plot
        }
        
        self.outputs = {
            'hologram': 'image',         # The fused RGB image
            'coherence': 'signal'        # How aligned are the 3 layers?
        }
        
        self.resolution = 512
        
        # Phase Space History (for the attractor trail)
        self.phase_history = deque(maxlen=200)
        self.display_img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self.coherence_val = 0.0

    def step(self):
        # 1. Gather Images
        body = self.get_blended_input('body_structure', 'mean')
        field = self.get_blended_input('quantum_field', 'mean')
        mind = self.get_blended_input('mind_attention', 'mean')
        
        # 2. Gather Signals
        entropy = self.get_blended_input('system_entropy', 'sum') or 0.0
        energy = self.get_blended_input('free_energy', 'sum') or 0.0
        
        # Track phase space
        self.phase_history.append((entropy, energy))
        
        # 3. Process Layers (Resize & Normalize)
        def prepare_layer(img):
            if img is None:
                return np.zeros((self.resolution, self.resolution), dtype=np.float32)
            
            # Handle dimensions
            if img.ndim == 3:
                img = np.mean(img, axis=2) # Flatten to grayscale
                
            # Resize
            if img.shape[:2] != (self.resolution, self.resolution):
                img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            
            # Normalize 0..1
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min())
            return img

        L_body = prepare_layer(body)
        L_field = prepare_layer(field)
        L_mind = prepare_layer(mind)
        
        # 4. Compute Coherence (Overlap of all 3)
        # High if all 3 are active in the same spots
        overlap = L_body * L_field * L_mind
        self.coherence_val = float(np.mean(overlap))
        
        # 5. Compose RGB Hologram
        # Body = Red, Field = Green, Mind = Blue
        hologram = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
        hologram[:, :, 0] = L_body   * 1.0  # Red
        hologram[:, :, 1] = L_field  * 0.8  # Green (slightly dim to see structure)
        hologram[:, :, 2] = L_mind   * 1.2  # Blue (bright to show sparse attention)
        
        # Clip
        hologram = np.clip(hologram, 0, 1)
        
        # Convert to uint8 for drawing
        vis = (hologram * 255).astype(np.uint8)
        
        # 6. Draw Phase Space Attractor (Overlay)
        # Map entropy/energy to X/Y coordinates
        if len(self.phase_history) > 1:
            # Auto-scale
            hist = np.array(self.phase_history)
            min_x, max_x = hist[:, 0].min(), hist[:, 0].max() + 1e-6
            min_y, max_y = hist[:, 1].min(), hist[:, 1].max() + 1e-6
            
            # Draw box
            margin = 20
            box_size = 100
            origin_x, origin_y = self.resolution - box_size - margin, self.resolution - margin
            
            # Draw background for plot
            cv2.rectangle(vis, (origin_x, origin_y - box_size), (origin_x + box_size, origin_y), (0, 0, 0), -1)
            cv2.rectangle(vis, (origin_x, origin_y - box_size), (origin_x + box_size, origin_y), (100, 100, 100), 1)
            
            # Draw Trail
            pts = []
            for e, f in self.phase_history:
                # Normalize to 0..1 relative to history window
                nx = (e - min_x) / (max_x - min_x)
                ny = (f - min_y) / (max_y - min_y)
                
                px = int(origin_x + nx * box_size)
                py = int(origin_y - ny * box_size) # Invert Y
                pts.append([px, py])
            
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw polyline (Cyan for the attractor)
            cv2.polylines(vis, [pts], False, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Label
            cv2.putText(vis, "PHASE ATTRACTOR", (origin_x, origin_y - box_size - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        self.display_img = vis

    def get_output(self, port_name):
        if port_name == 'hologram':
            # Return as float 0..1 for other nodes
            return self.display_img.astype(np.float32) / 255.0
        if port_name == 'coherence':
            return self.coherence_val
        return None

    def get_display_image(self):
        # Return RGB image for display
        return self.display_img

    def get_config_options(self):
        return [("Resolution", "resolution", self.resolution, None)]