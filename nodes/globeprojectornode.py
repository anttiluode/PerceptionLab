import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class GlobeProjectorNode(BaseNode):
    """
    Projects a 2D equirectangular map onto a 3D-like globe.
    Allows for interactive spinning and zooming. (v3 - Fixed lighting bug)
    """
    NODE_CATEGORY = "Visualizer"
    NODE_COLOR = QtGui.QColor(80, 120, 220) # Deep Blue

    def __init__(self, zoom=1.0, spin_x=0.0, spin_y=0.0, lighting=True, output_size=256):
        super().__init__()
        self.node_title = "Globe Projector"
        
        # --- Inputs and Outputs ---
        self.inputs = {'image_in': 'image'}
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable ---
        self.zoom = float(zoom)
        self.spin_x = float(spin_x) # longitude
        self.spin_y = float(spin_y) # latitude
        self.lighting = bool(lighting)
        self.output_size = int(output_size)
        
        # --- Internal State ---
        self.output_image = np.zeros((self.output_size, self.output_size, 3), dtype=np.float32)
        self.map_x = None
        self.map_y = None
        self.light_map = None
        
        self._build_maps() # Initial map calculation

    def get_config_options(self):
        """Returns options for the right-click config dialog."""
        return [
            ("Zoom", "zoom", self.zoom, None),
            ("Spin X (0-360)", "spin_x", self.spin_x, None),
            ("Spin Y (0-360)", "spin_y", self.spin_y, None),
            ("Lighting (0 or 1)", "lighting", 1 if self.lighting else 0, None),
            ("Resolution", "output_size", self.output_size, None),
        ]

    def set_config_options(self, options):
        """Receives a dictionary from the config dialog."""
        size_changed = False
        if "zoom" in options: self.zoom = float(options["zoom"])
        if "spin_x" in options: self.spin_x = float(options["spin_x"])
        if "spin_y" in options: self.spin_y = float(options["spin_y"])
        if "lighting" in options: self.lighting = bool(float(options["lighting"]))
        if "output_size" in options:
            new_size = int(options["output_size"])
            if new_size != self.output_size:
                self.output_size = new_size
                size_changed = True
        
        self._build_maps(force_rebuild=size_changed)

    def _build_maps(self, force_rebuild=False):
        """
        Pre-calculates the cv2.remap matrices. This is the core logic.
        """
        w = h = self.output_size
        
        if self.map_x is not None and not force_rebuild:
             pass 
        else:
            self.map_x = np.zeros((h, w), dtype=np.float32)
            self.map_y = np.zeros((h, w), dtype=np.float32)
            self.light_map = np.zeros((h, w), dtype=np.float32)

        spin_x_rad = (self.spin_x % 360) * (np.pi / 180.0)
        spin_y_rad = (self.spin_y % 360) * (np.pi / 180.0)
        
        xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

        xx /= self.zoom
        yy /= self.zoom
        
        zz_sq = 1.0 - xx*xx - yy*yy
        
        mask = zz_sq >= 0
        zz = np.sqrt(zz_sq[mask]) 
        
        lon = np.arctan2(xx[mask], zz) + spin_x_rad
        lat = np.arcsin(yy[mask]) + spin_y_rad
        
        lat = np.clip(lat, -np.pi/2, np.pi/2)
        
        u = (lon / (2 * np.pi)) + 0.5
        v = 0.5 - (lat / np.pi) 
        
        self.map_x[mask] = u
        self.map_y[mask] = v
        
        self.light_map.fill(0) 
        self.light_map[mask] = np.clip(zz, 0.2, 1.0) 

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is None:
            return

        self._build_maps()

        try:
            in_h, in_w = img_in.shape[:2]
        except Exception as e:
            print(f"GlobeProjector: Bad input image shape. {e}")
            return
            
        map_x_abs = self.map_x * in_w
        map_y_abs = self.map_y * in_h
        
        map_x_abs[~np.isfinite(map_x_abs)] = -1
        map_y_abs[~np.isfinite(map_y_abs)] = -1
        map_x_abs[self.map_x == 0] = -1 
        map_y_abs[self.map_y == 0] = -1
        
        self.output_image = cv2.remap(
            img_in, 
            map_x_abs, 
            map_y_abs, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0) 
        )

        # --- Apply Lighting ---
        if self.lighting:
            
            # --- THIS IS THE FIX ---
            # If the remapped image is grayscale, convert it to 3-channel
            # before applying the 3-channel lighting map.
            if self.output_image.ndim == 2:
                self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_GRAY2BGR)
            # --- END FIX ---

            light_map_3ch = cv2.cvtColor(self.light_map, cv2.COLOR_GRAY2BGR)
            
            # Now both are 3-channel, so this will work
            self.output_image = self.output_image * light_map_3ch
            
        self.output_image = np.clip(self.output_image, 0, 1)

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.output_image
        return None

    def get_display_image(self):
        return self.output_image
