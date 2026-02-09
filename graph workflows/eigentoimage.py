"""
Eigen To Image Node (The Inverse Bridge)
----------------------------------------
Reconstructs an image from the eigenstructure output of SelfConsistentResonanceNode.
Inverts: radial projection → vector → image grid

This is solving the inverse problem: given the mandala, recover the original image.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class EigenToImageNode(BaseNode):
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(180, 100, 180)  # Purple - inverse of gray
    
    def __init__(self):
        super().__init__()
        self.node_title = "Eigen -> Image"
        
        self.inputs = {
            'eigen_image': 'image',           # The mandala/eigenstructure
            'eigenfrequencies': 'spectrum',   # Optional: direct 1D spectrum
            'reference_image': 'image'        # Optional: for phase hints
        }
        
        self.outputs = {
            'reconstructed': 'image',      # Main output - recovered image
            'radial_profile': 'spectrum',  # Extracted 1D spectrum
            'confidence_map': 'image',     # How confident each pixel is
            'residual': 'image'            # Difference from reference if provided
        }
        
        # Configuration
        self.output_size = 128          # Output image resolution
        self.vector_dim = 256           # Must match ImageToVectorNode
        self.use_phase_recovery = True  # Attempt iterative phase recovery
        self.iterations = 10            # Phase recovery iterations
        self.smoothing = 1.0            # Output smoothing
        
        # Internal state
        self.radial_spectrum = np.zeros(128)
        self.reconstructed = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        self.confidence = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        
        # Precompute radial indices for inversion
        self._build_radial_map(128)  # Default eigen size
    
    def _build_radial_map(self, size):
        """Build lookup tables for radial averaging (inverse of project_to_2d)"""
        self.eigen_size = size
        center = size // 2
        
        y, x = np.ogrid[:size, :size]
        self.r_grid = np.sqrt((x - center)**2 + (y - center)**2)
        self.max_radius = center
        
        # Build bins for radial averaging
        self.radial_bins = []
        for r in range(center):
            mask = (self.r_grid >= r) & (self.r_grid < r + 1)
            indices = np.where(mask)
            self.radial_bins.append(indices)
    
    def extract_radial_profile(self, eigen_img):
        """
        Inverse of project_to_2d: average concentric rings to get 1D spectrum.
        This is the key inversion step.
        """
        if eigen_img.shape[0] != self.eigen_size:
            self._build_radial_map(eigen_img.shape[0])
        
        # Handle complex or magnitude input
        if np.iscomplexobj(eigen_img):
            magnitude = np.abs(eigen_img)
        else:
            magnitude = eigen_img.astype(np.float32)
        
        # Normalize
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        
        # Average each radial bin
        profile = np.zeros(self.max_radius)
        for r, indices in enumerate(self.radial_bins):
            if len(indices[0]) > 0:
                profile[r] = np.mean(magnitude[indices])
        
        return profile
    
    def spectrum_to_vector(self, radial_profile):
        """
        Convert radial profile back to the flat vector format.
        Inverse of how ImageToVectorNode's output was used.
        """
        # Resample to vector_dim
        # The original went: image → 16x16 grid → 256 vector
        # The grid was then implicitly converted to radial by project_to_2d
        
        # We need to reconstruct the 16x16 grid from radial info
        side = int(np.ceil(np.sqrt(self.vector_dim)))
        center = side // 2
        
        grid = np.zeros((side, side), dtype=np.float32)
        
        # For each pixel, look up its radial distance and sample spectrum
        for y in range(side):
            for x in range(side):
                r = np.sqrt((x - center)**2 + (y - center)**2)
                # Scale r to match profile length
                r_scaled = r * len(radial_profile) / (side / 2)
                r_idx = int(np.clip(r_scaled, 0, len(radial_profile) - 1))
                
                # Interpolate for smoothness
                if r_idx < len(radial_profile) - 1:
                    frac = r_scaled - r_idx
                    grid[y, x] = (1 - frac) * radial_profile[r_idx] + frac * radial_profile[r_idx + 1]
                else:
                    grid[y, x] = radial_profile[r_idx]
        
        return grid
    
    def grid_to_image(self, grid):
        """
        Upsample the small grid back to full image resolution.
        Inverse of cv2.resize with INTER_AREA.
        """
        # INTER_CUBIC for smooth upsampling (inverse of INTER_AREA averaging)
        upsampled = cv2.resize(grid, (self.output_size, self.output_size), 
                               interpolation=cv2.INTER_CUBIC)
        return upsampled
    
    def phase_recovery(self, magnitude_image, iterations=10):
        """
        Gerchberg-Saxton-like phase recovery.
        Given only magnitude in Fourier domain, estimate phase iteratively.
        
        This helps recover spatial structure lost in the forward transform.
        """
        # Start with random phase
        phase = np.random.uniform(-np.pi, np.pi, magnitude_image.shape)
        
        for i in range(iterations):
            # Construct complex spectrum
            spectrum = magnitude_image * np.exp(1j * phase)
            
            # Inverse FFT to spatial domain
            spatial = np.fft.ifft2(np.fft.ifftshift(spectrum))
            
            # Apply spatial constraints (real, positive)
            spatial = np.abs(spatial)
            spatial = np.clip(spatial, 0, None)
            
            # Forward FFT back to frequency domain
            new_spectrum = np.fft.fftshift(np.fft.fft2(spatial))
            
            # Keep original magnitude, update phase
            phase = np.angle(new_spectrum)
        
        # Final reconstruction
        final_spectrum = magnitude_image * np.exp(1j * phase)
        reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(final_spectrum)))
        
        return reconstructed
    
    def step(self):
        # Get inputs
        eigen_img = self.get_blended_input('eigen_image', 'first')
        direct_spectrum = self.get_blended_input('eigenfrequencies', 'sum')
        reference = self.get_blended_input('reference_image', 'first')
        
        if eigen_img is None and direct_spectrum is None:
            return
        
        # Path 1: From eigen_image (mandala)
        if eigen_img is not None:
            # Handle RGB input
            if eigen_img.ndim == 3:
                eigen_img = np.mean(eigen_img, axis=2)
            
            # Step 1: Extract radial profile (invert 2D projection)
            self.radial_spectrum = self.extract_radial_profile(eigen_img)
            
            # Step 2: Convert spectrum to grid (invert vector flattening)
            grid = self.spectrum_to_vector(self.radial_spectrum)
            
            # Step 3: Upsample to image (invert downsampling)
            self.reconstructed = self.grid_to_image(grid)
            
            # Optional: Phase recovery for better structure
            if self.use_phase_recovery and self.iterations > 0:
                # Use the grid as magnitude constraint in frequency domain
                magnitude_constraint = np.fft.fftshift(np.abs(np.fft.fft2(self.reconstructed)))
                self.reconstructed = self.phase_recovery(magnitude_constraint, self.iterations)
            
            # Smooth output
            if self.smoothing > 0:
                self.reconstructed = gaussian_filter(self.reconstructed, self.smoothing)
            
            # Normalize to 0-1
            if self.reconstructed.max() > self.reconstructed.min():
                self.reconstructed = (self.reconstructed - self.reconstructed.min()) / \
                                    (self.reconstructed.max() - self.reconstructed.min())
        
        # Path 2: From direct 1D spectrum
        elif direct_spectrum is not None:
            self.radial_spectrum = np.array(direct_spectrum)
            grid = self.spectrum_to_vector(self.radial_spectrum)
            self.reconstructed = self.grid_to_image(grid)
            
            if self.smoothing > 0:
                self.reconstructed = gaussian_filter(self.reconstructed, self.smoothing)
            
            if self.reconstructed.max() > self.reconstructed.min():
                self.reconstructed = (self.reconstructed - self.reconstructed.min()) / \
                                    (self.reconstructed.max() - self.reconstructed.min())
        
        # Compute confidence map (higher where we have more radial samples)
        self._compute_confidence()
        
        # If reference provided, compute residual
        if reference is not None:
            if reference.ndim == 3:
                reference = np.mean(reference, axis=2)
            ref_resized = cv2.resize(reference, (self.output_size, self.output_size))
            if ref_resized.max() > 0:
                ref_resized = ref_resized / ref_resized.max()
            self.residual = np.abs(self.reconstructed - ref_resized)
        else:
            self.residual = np.zeros_like(self.reconstructed)
    
    def _compute_confidence(self):
        """
        Confidence is higher at center (more samples averaged there in forward pass)
        and lower at edges (fewer samples, more uncertainty in inversion).
        """
        y, x = np.ogrid[:self.output_size, :self.output_size]
        center = self.output_size // 2
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Confidence decreases with radius (inverse of sampling density)
        max_r = np.sqrt(2) * center
        self.confidence = 1.0 - (r / max_r)
        self.confidence = np.clip(self.confidence, 0.1, 1.0)
    
    def get_output(self, port_name):
        if port_name == 'reconstructed':
            # Return as uint8 image
            return (self.reconstructed * 255).astype(np.uint8)
        
        elif port_name == 'radial_profile':
            return self.radial_spectrum
        
        elif port_name == 'confidence_map':
            return (self.confidence * 255).astype(np.uint8)
        
        elif port_name == 'residual':
            return (self.residual * 255).astype(np.uint8)
        
        return None
    
    def get_display_image(self):
        """Show reconstruction with confidence overlay"""
        # Main reconstruction
        img = (self.reconstructed * 255).astype(np.uint8)
        
        # Apply colormap for visibility
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        
        # Blend with confidence (optional visual feedback)
        # Darker edges show lower confidence
        conf_3ch = np.stack([self.confidence] * 3, axis=-1)
        img_weighted = (img_color * conf_3ch).astype(np.uint8)
        
        # Resize for display
        display_size = 128
        img_display = cv2.resize(img_weighted, (display_size, display_size), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Add label
        cv2.putText(img_display, "RECON", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return QtGui.QImage(img_display.data, display_size, display_size,
                           display_size * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Output Size", "output_size", self.output_size, None),
            ("Vector Dim", "vector_dim", self.vector_dim, None),
            ("Phase Recovery", "use_phase_recovery", self.use_phase_recovery, None),
            ("Iterations", "iterations", self.iterations, None),
            ("Smoothing", "smoothing", self.smoothing, None),
        ]
    
    def set_config_options(self, options):
        if "output_size" in options:
            self.output_size = int(options["output_size"])
            self.reconstructed = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            self.confidence = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        if "vector_dim" in options:
            self.vector_dim = int(options["vector_dim"])
        if "use_phase_recovery" in options:
            self.use_phase_recovery = bool(options["use_phase_recovery"])
        if "iterations" in options:
            self.iterations = int(options["iterations"])
        if "smoothing" in options:
            self.smoothing = float(options["smoothing"])