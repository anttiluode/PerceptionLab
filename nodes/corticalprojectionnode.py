import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class CorticalProjectionNode(BaseNode):
    """
    The Truth Test.
    Transforms the 'Retinal' geometry (The Star) into 'Cortical' coordinates (V1 Strip).
    
    Biology: The eye is circular, but V1 is a rectangular sheet.
    Math: Log-Polar Transform.
    
    Hypothesis: 
    If the Star is a true cortical eigenmode, this node should output
    perfect horizontal lattices (Tunnels) or diagonals (Spirals).
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "V1 Cortical Projection"
    NODE_COLOR = QtGui.QColor(200, 100, 255) # Purple for V1

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'retinal_image': 'image',   # Connect 'eigen_image' (The Star) here
        }
        
        self.outputs = {
            'cortical_view': 'image',   # What the Brain actually "sees"
            'coherence_check': 'signal' # Is it a stable lattice?
        }
        
        self.last_cortical = None

    def step(self):
        inp = self.get_blended_input('retinal_image', 'first')
        if inp is None: return

        # 1. Prepare Input
        # Ensure we have a float image 0-1 or uint8 0-255
        if inp.dtype != np.uint8:
            img = (np.clip(inp, 0, 1) * 255).astype(np.uint8)
        else:
            img = inp
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        max_radius = min(center[0], center[1])
        
        # 2. THE TRANSFORMATION (Retina -> Cortex)
        # We use Log-Polar mapping (Biologically accurate for V1)
        # X-Axis: Angle (Theta)
        # Y-Axis: Log(Radius) (Eccentricity)
        
        # Note: We rotate 90 degrees to align 'Up' with 'Forward' in the tunnel view
        flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
        cortical = cv2.warpPolar(img, (w, h), center, max_radius, flags)
        
        # Rotate output so Left-Right = 0-360 degrees, Up-Down = Depth (Fovea to Periphery)
        cortical = cv2.rotate(cortical, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        self.last_cortical = cortical

    def get_output(self, port):
        if port == 'cortical_view':
            return self.last_cortical
        return None

    def get_display_image(self):
        if self.last_cortical is None: return None
        
        # Visualization
        # We apply a heatmap to make the "Lattice" structure pop
        c_map = cv2.applyColorMap(self.last_cortical, cv2.COLORMAP_INFERNO)
        
        # Add HUD lines
        h, w = c_map.shape[:2]
        
        # Draw "Tunnel" guide lines
        # If the output matches these lines, it is a perfect Tunnel Hallucination
        cv2.line(c_map, (0, h//2), (w, h//2), (100, 100, 100), 1) # Horizon
        
        # Text
        cv2.putText(c_map, "V1 CORTICAL MAP", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(c_map, "Left=Fovea  Right=Periphery", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        return QtGui.QImage(c_map.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)