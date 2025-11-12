import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2

class EquivalenceNode(BaseNode):
    """
    Converts "Matter" (an image) into "Energy" (a force spectrum)
    based on its complexity (Mass) and structure (Curvature).
    This system's E=mc^2.
    """
    NODE_CATEGORY = "Cosmology"
    NODE_COLOR = QtGui.QColor(255, 253, 230) # Einstein's paper

    def __init__(self, spectrum_size=512):
        super().__init__()
        self.node_title = "Equivalence (E=m*c^2)"
        
        # --- Inputs and Outputs ---
        self.inputs = {'image_in': 'image'}
        self.outputs = {'force_spectrum_out': 'spectrum'}
        
        # --- Configurable ---
        self.spectrum_size = int(spectrum_size)
        
        # --- Internal State ---
        self.force_spectrum = np.zeros(self.spectrum_size, dtype=np.float32)

    def get_config_options(self):
        return [
            ("Spectrum Size", "spectrum_size", self.spectrum_size, None),
        ]

    def set_config_options(self, options):
        if "spectrum_size" in options:
            self.spectrum_size = int(options["spectrum_size"])
            # Resize spectrum buffer
            self.force_spectrum = np.zeros(self.spectrum_size, dtype=np.float32)

    def step(self):
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is None:
            self.force_spectrum.fill(0)
            return

        try:
            # --- 1. Calculate "Mass" (m) ---
            # We define "Mass" as the image's entropy or complexity.
            # A simple measure is the standard deviation of pixel values.
            # A flat gray image has 0 mass. A complex one has high mass.
            img_mass = np.std(img_in)
            
            # --- 2. Calculate "Curvature" (c^2) ---
            # We define "Curvature" as the image's spatial structure.
            # We use the Laplacian (second derivative) to find edges/curves.
            # A smooth image has 0 curvature. A sharp one has high curvature.
            if img_in.ndim == 3:
                gray_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img_in
            
            # Ensure 8-bit for Laplacian
            gray_u8 = (np.clip(gray_img, 0, 1) * 255).astype(np.uint8)
            laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
            img_curvature = np.mean(np.abs(laplacian))

            # --- 3. Calculate "Total Energy" (E) ---
            # E = m * c^2 (A simplified model)
            # This is the "gravity" or "force" of the image.
            total_energy = img_mass * (img_curvature + 1.0) # +1 to avoid zero

            # --- 4. Populate the Force Spectrum ---
            # The spectrum will carry this information.
            
            # Clear old spectrum
            self.force_spectrum.fill(0)
            
            # The first two "slots" are the fundamental laws
            self.force_spectrum[0] = total_energy # The total "Gravity"
            self.force_spectrum[1] = img_mass     # The "Mass" component
            self.force_spectrum[2] = img_curvature # The "Curvature" component

            # The rest of the spectrum is the "Vibrational Energy"
            # (A 1D representation of the image's content)
            
            # Resize image to fit the remaining spectrum
            h, w = gray_img.shape[:2]
            remaining_size = self.spectrum_size - 3
            if remaining_size > 0:
                # Get a 1D "slice" of the image
                flat_slice = cv2.resize(gray_img, (remaining_size, 1), 
                                        interpolation=cv2.INTER_LINEAR).flatten()
                
                self.force_spectrum[3:self.spectrum_size] = flat_slice

            # Normalize (optional, but good practice)
            if total_energy > 0:
                 self.force_spectrum /= np.max(self.force_spectrum)

        except Exception as e:
            print(f"EquivalenceNode Error: {e}")
            self.force_spectrum.fill(0)

    def get_output(self, port_name):
        if port_name == 'force_spectrum_out':
            return self.force_spectrum
        return None

    def get_display_image(self):
        # We can visualize the spectrum itself
        if self.force_spectrum is None: return None
        
        # Create an image from the spectrum
        h = 96
        w = len(self.force_spectrum)
        if w == 0: return None
        
        # Normalize spectrum for display
        spec_norm = self.force_spectrum - self.force_spectrum.min()
        max_val = spec_norm.max()
        if max_val > 0:
            spec_norm /= max_val
            
        spec_img = (spec_norm * 255).astype(np.uint8)
        spec_img = np.tile(spec_img, (h, 1)) # Repeat rows to make an image
        spec_img = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
        
        return spec_img