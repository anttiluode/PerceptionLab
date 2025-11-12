import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class GlobeToEquirectangularNode(BaseNode):
    """
    Unwraps a 2D image of a globe (orthographic projection)
    back into a 360-degree equirectangular map.
    """
    NODE_CATEGORY = "Image"
    NODE_COLOR = QtGui.QColor(80, 200, 220) # Light blue

    def __init__(self, spin_x=0.0, spin_y=0.0, output_w=512, output_h=256, center_x=0.5, center_y=0.5, radius_scale=1.0):
        super().__init__()
        self.node_title = "Globe Unwrapper (360)"
        
        # --- Inputs and Outputs ---
        self.inputs = {'image_in': 'image'}
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable ---
        self.spin_x = float(spin_x) # Longitude (0-360) at the center of the globe
        self.spin_y = float(spin_y) # Latitude (0-360) at the center
        self.output_w = int(output_w)
        self.output_h = int(output_h)
        self.center_x = float(center_x) # Normalized center of globe in input (0-1)
        self.center_y = float(center_y) # Normalized center of globe in input (0-1)
        self.radius_scale = float(radius_scale) # Scale radius (1.0 = touch edges)
        
        # --- Internal State ---
        self.output_image = np.zeros((self.output_h, self.output_w, 3), dtype=np.float32)
        
        # Pre-calculated mapping coordinates
        self.map_nx = None
        self.map_ny = None
        self.mask = None
        
        self._build_maps() # Initial map calculation

    def get_config_options(self):
        """Returns options for the right-click config dialog."""
        return [
            ("Center Lon (Spin X)", "spin_x", self.spin_x, None),
            ("Center Lat (Spin Y)", "spin_y", self.spin_y, None),
            ("Output Width (px)", "output_w", self.output_w, None),
            ("Output Height (px)", "output_h", self.output_h, None),
            ("Input Center X (0-1)", "center_x", self.center_x, None),
            ("Input Center Y (0-1)", "center_y", self.center_y, None),
            ("Input Radius Scale (0-1)", "radius_scale", self.radius_scale, None),
        ]

    def set_config_options(self, options):
        """Receives a dictionary from the config dialog."""
        rebuild = False
        if "spin_x" in options:
            self.spin_x = float(options["spin_x"])
            rebuild = True
        if "spin_y" in options:
            self.spin_y = float(options["spin_y"])
            rebuild = True
        if "output_w" in options:
            self.output_w = int(options["output_w"])
            rebuild = True
        if "output_h" in options:
            self.output_h = int(options["output_h"])
            rebuild = True
        
        # These don't require rebuilding the maps, they are applied in step()
        if "center_x" in options: self.center_x = float(options["center_x"])
        if "center_y" in options: self.center_y = float(options["center_y"])
        if "radius_scale" in options: self.radius_scale = float(options["radius_scale"])
            
        if rebuild:
            self._build_maps()

    def _build_maps(self):
        """
        Pre-calculates the normalized [-1, 1] mapping coordinates.
        This defines the shape of the unwrapping.
        """
        w, h = self.output_w, self.output_h
        if w == 0 or h == 0: return

        # Create 2D grid of pixel coordinates for the output map
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert pixel coords (u,v) to spherical coords (lon, lat)
        lon = (u / (w - 1.0)) * 2 * np.pi - np.pi  # -pi to +pi
        lat = (v / (h - 1.0)) * np.pi - (np.pi / 2.0) # -pi/2 to +pi/2
        
        # Apply the "un-rotation" based on the spin settings
        spin_lon_rad = (self.spin_x % 360) * np.pi / 180.0
        spin_lat_rad = (self.spin_y % 360) * np.pi / 180.0
        
        lon_rotated = lon - spin_lon_rad
        lat_rotated = lat # Note: Y-spin (latitude) is more complex, focusing on X-spin
        
        # Convert spherical (lon, lat) to 3D Cartesian (x,y,z)
        # where +z is "out of the screen"
        x_3d = np.cos(lat_rotated) * np.sin(lon_rotated)
        y_3d = np.sin(lat_rotated)
        z_3d = np.cos(lat_rotated) * np.cos(lon_rotated)

        # These are our normalized [-1, 1] coordinates for the orthographic projection
        self.map_nx = x_3d
        self.map_ny = -y_3d  # Invert Y for image coordinates (+y is down)
        
        # The mask tells us which pixels are on the "front"
        self.mask = z_3d >= 0

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is None or self.map_nx is None:
            return

        try:
            h_in, w_in = img_in.shape[:2]
        except Exception as e:
            print(f"GlobeUnwrapper: Bad input image shape. {e}")
            return
            
        # 1. Scale normalized maps to the input image's dimensions
        radius = (min(w_in, h_in) / 2.0) * self.radius_scale
        center_x_abs = w_in * self.center_x
        center_y_abs = h_in * self.center_y
        
        map_x = (self.map_nx * radius) + center_x_abs
        map_y = (self.map_ny * radius) + center_y_abs

        # 2. Apply the mask (set "back" pixels to -1)
        map_x[~self.mask] = -1
        map_y[~self.mask] = -1

        # 3. Create a new output image buffer
        self.output_image = np.zeros((self.output_h, self.output_w, 3), dtype=np.float32)
        if img_in.ndim == 3:
            h, w, c = img_in.shape
            self.output_image = np.zeros((self.output_h, self.output_w, c), dtype=np.float32)
        else:
            self.output_image = np.zeros((self.output_h, self.output_w), dtype=np.float32)

        # 4. Apply the warp
        self.output_image = cv2.remap(
            img_in,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0) # Back of the globe is black
        )
        
        # Ensure output is 0-1 float
        self.output_image = np.clip(self.output_image, 0, 1)

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.output_image
        return None

    def get_display_image(self):
        return self.output_image