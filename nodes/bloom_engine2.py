"""
Angular Harmonics Node
======================
The "V1 Orientation Decoder."

Performs Angular Fourier Decomposition on an image.
1. Unwraps the image from Cartesian (x,y) to Polar (r, theta).
2. Computes the 1D FFT along the theta axis.
3. Identifies dominant rotational symmetries (k=2, k=3, k=4, etc.).
4. Can RECONSTRUCT the image using ONLY specific harmonics (filtering).

This allows you to separate the "Cat Face" (k=2) from the "Grid Artifacts" (k=4)
and the "Organic Blobs" (low frequencies).

Inputs:
    image_in: Any square image (mandala, reconstruction, etc.)
Outputs:
    filtered_image: The image reconstructed from ONLY the selected harmonics.
    spectrum_plot: Visual readout of angular strength.
    k2_strength: Strength of bilateral symmetry (V1 simple cells / faces).
    k4_strength: Strength of cross/grid symmetry.
    symmetry_vector: The full angular spectrum (for driving other nodes).
"""

import numpy as np
import cv2
import __main__

try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    # Standalone fallback
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, method): return None

class AngularHarmonicsNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Angular Harmonics"
    NODE_COLOR = QtGui.QColor(0, 150, 200)  # Cyan-ish

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',
            'filter_mode': 'signal',  # 0=Pass all, 1=Pass selected, 2=Reject selected
        }
        
        self.outputs = {
            'filtered_image': 'image',     # The V1-filtered view
            'spectrum_plot': 'image',      # Visual graph
            'k2_strength': 'signal',       # Bilateral/Oval power
            'k4_strength': 'signal',       # Cross/Square power
            'symmetry_vector': 'spectrum'  # Full harmonic series
        }
        
        # Config
        self.max_harmonic = 16
        self.smoothing = 0.2
        self.selected_harmonics = "2, 4" # String input for harmonics to keep/kill
        
        # State
        self.current_spectrum = np.zeros(self.max_harmonic, dtype=np.float32)
        self.display_image = None
        self.polar_img = None

    def step(self):
        img = self.get_blended_input('image_in', 'first')
        if img is None:
            return

        # 1. Preprocess: Grayscale & Resize
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        
        h, w = img.shape
        center = (w // 2, h // 2)
        max_radius = min(center[0], center[1])
        
        # 2. Cartesian -> Polar Transform (The "Unwrap")
        # Output is (radius, angle)
        # We use LinearPolar to map: X-axis = Radius, Y-axis = Angle
        # But cv2.linearPolar maps: X=Radius, Y=Angle. 
        # Wait, usually we want Angle on X for easy 1D FFT. 
        # Let's map: X=Angle, Y=Radius? No, standard is X=Radius.
        # We'll stick to standard and transpose.
        
        self.polar_img = cv2.linearPolar(
            img, center, max_radius, 
            cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR
        )
        
        # Now rows=Angle (0..360 mapped to 0..H), cols=Radius
        # Actually cv2.linearPolar: 
        #   x-axis corresponds to radius
        #   y-axis corresponds to angle
        # So we integrate over the Radius (X-axis) to get the Angular Profile (Y-axis)
        
        # We weigh the outer radius higher? Or just mean?
        # Let's use a weighted mean, favoring the mid-range (where structure is)
        radial_weights = np.linspace(0, 1, w)
        radial_weights = np.exp(-0.5 * ((radial_weights - 0.6) / 0.3)**2) # Gaussian at r=0.6
        
        # angular_profile has shape (h,) -> one value per angle step
        angular_profile = np.average(self.polar_img, axis=1, weights=radial_weights)
        
        # 3. FFT (Angular Decomposition)
        # angular_profile is real, so we use rfft
        fft_coeffs = np.fft.rfft(angular_profile)
        
        # Normalize magnitude
        mags = np.abs(fft_coeffs)
        dc = mags[0]
        if dc > 1e-9:
            mags /= dc # Normalize by DC component
        mags[0] = 1.0 # Restore DC for visual scaling
        
        # Smooth spectrum output
        if len(mags) > self.max_harmonic:
            current_mags = mags[:self.max_harmonic]
        else:
            current_mags = np.pad(mags, (0, self.max_harmonic - len(mags)))
            
        self.current_spectrum = (self.current_spectrum * self.smoothing + 
                                 current_mags * (1.0 - self.smoothing))

        # 4. Filter / Reconstruction (The V1 Filter)
        filter_mode = self.get_blended_input('filter_mode', 'sum')
        if filter_mode is None: filter_mode = 0
        
        target_k = self._parse_harmonics()
        
        # Create a filter mask for the FFT coefficients
        mask = np.ones_like(fft_coeffs, dtype=np.float32)
        
        if filter_mode == 1: # Pass ONLY selected
            mask[:] = 0
            mask[0] = 1 # Always keep DC (brightness)
            for k in target_k:
                if k < len(mask): mask[k] = 1
                
        elif filter_mode == 2: # Reject selected
            for k in target_k:
                if k < len(mask): mask[k] = 0
                
        filtered_coeffs = fft_coeffs * mask
        
        # Inverse FFT -> Filtered Angular Profile
        filtered_profile = np.fft.irfft(filtered_coeffs, n=len(angular_profile))
        
        # Reproject to 2D Polar (expand profile back across radius)
        # We take the filtered profile (angle) and broadcast it across radius
        filtered_polar = np.tile(filtered_profile[:, np.newaxis], (1, w))
        
        # Apply the original radial envelope to keep the "shape" roughly valid
        # (This prevents it from looking like infinite rays)
        radial_envelope = np.mean(self.polar_img, axis=0)
        filtered_polar = filtered_polar * (radial_envelope / (np.max(radial_envelope)+1e-9))
        
        # Inverse Polar -> Cartesian
        filtered_cartesian = cv2.linearPolar(
            filtered_polar, center, max_radius,
            cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR
        )

        # 5. Outputs
        self.display_image = self._draw_spectrum_plot(img, self.current_spectrum)
        
        # Specific signals
        k2 = float(self.current_spectrum[2]) if 2 < len(self.current_spectrum) else 0.0
        k4 = float(self.current_spectrum[4]) if 4 < len(self.current_spectrum) else 0.0
        
        # Store for getters
        self.k2_val = k2
        self.k4_val = k4
        self.filtered_result = filtered_cartesian

    def _parse_harmonics(self):
        try:
            return [int(x.strip()) for x in self.selected_harmonics.split(',') if x.strip()]
        except:
            return []

    def _draw_spectrum_plot(self, original_img, spectrum):
        # Create a side-by-side view: [Original | Filtered | Spectrum]
        h, w = 128, 128
        
        # 1. Spectrum Graph
        plot = np.zeros((h, w, 3), dtype=np.uint8)
        bar_w = w // self.max_harmonic
        for k in range(1, self.max_harmonic): # Skip DC
            val = spectrum[k]
            bh = int(val * h * 2) # Scale up
            x = k * bar_w
            
            # Color coding
            color = (200, 200, 200)
            if k == 2: color = (100, 255, 100) # Green for bilateral
            if k == 4: color = (100, 200, 255) # Blue for cross
            
            cv2.rectangle(plot, (x, h), (x + bar_w - 2, h - bh), color, -1)
            # Label
            if k % 2 == 0:
                cv2.putText(plot, str(k), (x, h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150,150,150), 1)

        cv2.putText(plot, "ANGULAR FFT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # 2. Return composite for display
        # Note: The main node output 'spectrum_plot' will just be this graph.
        # The node FACE will show the graph.
        return plot

    def get_output(self, port_name):
        if port_name == 'filtered_image':
            return (self.filtered_result * 255).astype(np.uint8)
        elif port_name == 'spectrum_plot':
            return self.display_image
        elif port_name == 'k2_strength':
            return self.k2_val
        elif port_name == 'k4_strength':
            return self.k4_val
        elif port_name == 'symmetry_vector':
            return self.current_spectrum
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_image.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Selected Harmonics (csv)", "selected_harmonics", self.selected_harmonics, "str"),
            ("Smoothing", "smoothing", self.smoothing, "float")
        ]
        
    def set_config_options(self, options):
        if "selected_harmonics" in options:
            self.selected_harmonics = options["selected_harmonics"]
        if "smoothing" in options:
            self.smoothing = float(options["smoothing"])