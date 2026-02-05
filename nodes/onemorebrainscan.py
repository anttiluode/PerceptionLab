import numpy as np
import mne
import cv2
import os
from PyQt6 import QtGui
from scipy.interpolate import griddata
import __main__

# --- ROBUST BASE NODE FALLBACK ---
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    # If the system BaseNode isn't found, define a robust mock
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self._outputs = {}
        
        def get_input(self, name):
            # Fallback: return 0.0 for signals
            return 0.0
            
        def get_blended_input(self, name, method='first'):
            # Fallback: return None for images
            return None
            
        def set_output(self, name, value):
            self._outputs[name] = value

class HolographicTextureNode(BaseNode):
    """
    Holographic Texture Mapper.
    
    FIXED: 
    - Added Crash-Proof BaseNode to prevent 'get_input' errors.
    - Wraps your 'Thorny' Hologram onto the 3D Brain Surface.
    """
    NODE_CATEGORY = "Visualization"
    NODE_TITLE = "Holographic Texture"
    NODE_COLOR = QtGui.QColor(180, 0, 200) # Purple

    def __init__(self):
        super().__init__()
        self.inputs = {
            'hologram': 'image',        # Input the "Thorny" image here
            'rotation': 'signal'        # Rotate the view
        }
        self.outputs = {
            'textured_brain': 'image'   # The 3D Brain with Thorns wrapped on it
        }
        
        self.subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
        self.verts = None
        self.map_x = None
        self.map_y = None
        self._cache = None
        self.initialized = False
        
        self._init_mesh()

    def _init_mesh(self):
        if self.initialized: return
        try:
            # Load fsaverage surface (inflated to see into folds)
            if not os.path.isdir(os.path.join(self.subjects_dir, 'fsaverage')):
                print("[Texture] Downloading fsaverage template...")
                mne.datasets.fetch_fsaverage(subjects_dir=self.subjects_dir, verbose=False)
            
            # Read Left and Right Hemispheres
            lh_surf = mne.read_surface(os.path.join(self.subjects_dir, 'fsaverage', 'surf', 'lh.inflated'))
            rh_surf = mne.read_surface(os.path.join(self.subjects_dir, 'fsaverage', 'surf', 'rh.inflated'))
            
            # Combine vertices
            self.verts = np.concatenate([lh_surf[0], rh_surf[0]])
            # Shift RH for visualization gap (Visual separation)
            self.verts[len(lh_surf[0]):, 0] += 10 
            
            # Create a simplified 2D map of the 3D vertices for texture mapping
            # (Top-down projection UV Map)
            # This logic aligns the 2D Hologram coordinates to the 3D Brain coordinates
            self.map_x = ((self.verts[:, 0] / 80.0) + 1.0) / 2.0 # Normalize -1..1 to 0..1
            self.map_y = ((self.verts[:, 1] / 100.0) + 1.0) / 2.0
            
            # Clip and scale to texture size (128x128)
            self.map_x = np.clip(self.map_x * 127, 0, 127).astype(int)
            self.map_y = np.clip((1.0 - self.map_y) * 127, 0, 127).astype(int) # Invert Y for image coords
            
            self.initialized = True
            print("[Texture] Mesh Initialized.")
            
        except Exception as e:
            print(f"[Texture] Mesh Init Error: {e}")

    def step(self):
        # Safe input getting
        if hasattr(self, 'get_blended_input'):
            holo = self.get_blended_input('hologram', 'first')
        else:
            return

        if holo is None or not self.initialized: return

        # Resize hologram to standard 128x128 if needed
        if holo.shape[:2] != (128, 128):
            holo = cv2.resize(holo, (128, 128))

        # --- TEXTURE MAPPING ---
        # For every vertex in the 3D brain, look up its color in the Hologram
        
        # 1. Sample colors from the Hologram using the pre-computed UV map
        if len(holo.shape) == 2:
            colors = holo[self.map_y, self.map_x]
        else:
            colors = holo[self.map_y, self.map_x, :] # RGB
            
        # 2. Render simple point cloud
        # Get rotation input safely
        rot_val = 0.0
        if hasattr(self, 'get_input'):
            rot_val = self.get_input('rotation')
            
        angle = rot_val * 3.14159 * 2.0
        
        # Simple Rotation Matrix around Z axis
        c, s = np.cos(angle), np.sin(angle)
        x_rot = self.verts[:, 0] * c - self.verts[:, 1] * s
        y_rot = self.verts[:, 0] * s + self.verts[:, 1] * c
        z_rot = self.verts[:, 2]
        
        # Isometric Projection for 2.5D view
        iso_x = (x_rot - y_rot) * 0.8 + 128
        iso_y = (x_rot + y_rot) * 0.4 - z_rot * 0.8 + 128
        
        # Draw Canvas
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Color processing
        if len(colors.shape) == 1:
            # If monochrome, apply colormap
            c_norm = (colors / (colors.max() + 1e-9) * 255).astype(np.uint8)
            c_rgb = cv2.applyColorMap(c_norm, cv2.COLORMAP_MAGMA)
        else:
            c_rgb = colors.astype(np.uint8)

        # Depth sort (Painter's Algorithm - draw back to front)
        order = np.argsort(iso_y) 
        
        # Fast drawing (Downsample vertices by 10 for speed)
        # You can reduce the step (::10) to ::5 or ::1 for higher density if it's fast enough
        step_size = 8 
        
        for i in order[::step_size]:
            px, py = int(iso_x[i]), int(iso_y[i])
            if 0 <= px < 256 and 0 <= py < 256:
                # Get color tuple (B, G, R) for OpenCV
                color = c_rgb[i].tolist()
                # Draw pixel
                cv2.circle(canvas, (px, py), 1, color, -1)
                
        self._cache = canvas

    def get_display_image(self):
        if self._cache is None: return None
        img = self._cache
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888).copy()