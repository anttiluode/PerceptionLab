"""
Resonant Columns Node (Eigenmode Processor)
===========================================
Takes the Skew Matrix (A), diagonalizes it, and extracts the purely 
imaginary eigenvalues (omega). Uses these rotational rates to drive 
a bank of phase oscillators. 
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ResonantColumnNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(50, 180, 140)

    def __init__(self, num_columns=4):
        super().__init__()
        self.node_title = "Resonant Columns"
        self.inputs = {'skew_matrix': 'image'} # We extract A from this feed
        self.outputs = {}
        self.num_columns = int(num_columns)
        self.phases = np.zeros(self.num_columns, dtype=np.float32)
        self.display_img = np.zeros((128, 256, 3), dtype=np.uint8)

    def step(self):
        A_img = self.get_blended_input('skew_matrix', 'first')
        if A_img is None: return

        # Reconstruct A from the image feed (normalized 0-1 back to roughly -1 to 1)
        A = (A_img[:, :, 0].astype(np.float32) / 255.0 - 0.5) * 2.0
        
        # Diagonalize Skew Matrix A
        try:
            eigvals = np.linalg.eigvals(A)
            # Skew matrices have purely imaginary eigenvals. Extract the rotation rates (omega)
            omegas = np.abs(np.imag(eigvals))
            omegas = np.sort(omegas)[::-1] # Top rotation rates first
        except:
            omegas = np.zeros(self.num_columns)

        # Drive the columns
        for i in range(min(self.num_columns, len(omegas))):
            self.phases[i] += omegas[i] * 5.0 # Scale for visual speed

        # Visualization: Drawing the resonant phase dials
        img = np.zeros((128, 256, 3), dtype=np.uint8)
        spacing = 256 // self.num_columns
        
        for i in range(self.num_columns):
            cx, cy = int((i + 0.5) * spacing), 64
            radius = 25
            
            # Draw dial ring
            cv2.circle(img, (cx, cy), radius, (50, 50, 50), 2)
            
            # Draw phase arm (Boomerang representation)
            px = int(cx + np.cos(self.phases[i]) * radius)
            py = int(cy + np.sin(self.phases[i]) * radius)
            cv2.line(img, (cx, cy), (px, py), (100, 255, 150), 2)
            
            # Text omega
            rate = omegas[i] if i < len(omegas) else 0.0
            cv2.putText(img, f"w:{rate:.2f}", (cx-20, cy+45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        self.display_img = img

    def get_output(self, port_name):
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)