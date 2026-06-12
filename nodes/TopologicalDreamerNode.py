"""
Topological Dreamer Node (The Memory Decoder)
=============================================
Proves the Island Archival node successfully stores spatial geometry.
Reads the 16-dimensional topological 'memory_readout' vector, decodes it
back into a visual 'Ghost Island', and outputs the magnitude of the memory 
as a 'prior_strength' signal to be used for predictive noise suppression.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class TopologicalDreamerNode(BaseNode):
    NODE_CATEGORY = "Memory"
    NODE_COLOR = QtGui.QColor(100, 40, 150) # Amethyst

    def __init__(self):
        super().__init__()
        self.node_title = "Topological Dreamer"
        self.inputs = {'memory_vector': 'spectrum'}
        self.outputs = {
            'ghost_image': 'image',
            'prior_strength': 'signal'
        }
        self.prior_strength = 0.0
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        vec = self.get_blended_input('memory_vector', 'first')
        
        if vec is None or np.sum(np.abs(vec)) < 1e-5:
            self.prior_strength = 0.0
            self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.putText(self.display_img, "No Memory", (25, 64), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return

        # 1. Calculate the structural weight of the memory field
        self.prior_strength = float(np.mean(vec))

        # 2. Decode the 16D topology back into 2D spatial geometry
        # The PhaseFieldV2 compressed a 128x128 image into a 4x4 grid (16 values)
        grid_4x4 = np.array(vec[:16], dtype=np.float32).reshape((4, 4))
        
        # Upsample back to 128x128 using smooth cubic interpolation
        decoded_gray = cv2.resize(grid_4x4, (128, 128), interpolation=cv2.INTER_CUBIC)
        
        # 3. Apply a "Ghostly" colormap to distinguish it from active reality
        decoded_u8 = (np.clip(decoded_gray, 0, 1) * 255).astype(np.uint8)
        ghost_color = cv2.applyColorMap(decoded_u8, cv2.COLORMAP_DEEPGREEN)
        
        # Overlay text
        cv2.putText(ghost_color, f"Prior: {self.prior_strength:.3f}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        
        self.display_img = ghost_color

    def get_output(self, port_name):
        if port_name == 'ghost_image': 
            return self.display_img.astype(np.float32) / 255.0
        if port_name == 'prior_strength': 
            return self.prior_strength
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)