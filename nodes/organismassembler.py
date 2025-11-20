# organismassemblernode.py
"""
Organism Assembler Node (The Endoskeleton) - FIXED V2
------------------------------------------
Handles the structural closure of the Pac-Man mouth (Gastrulation) 
by generating opposing mechanical forces (Internal Pressure).
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class OrganismAssemblerNode(BaseNode):
    NODE_CATEGORY = "Cabbage Suite"
    NODE_COLOR = QtGui.QColor(255, 100, 50) # Orange for Synthesis

    def __init__(self, pressure_decay=0.98, closure_strength=0.1):
        super().__init__()
        self.node_title = "Organism Assembler"
        
        self.inputs = {
            'tissue_structure': 'image',     # The Pac-Man shape (skin)
            'guide_soliton': 'image',        # The Eyeball/Dipole (Growth Cone)
            'metabolic_signal': 'signal'     # General metabolic demand
        }
        
        self.outputs = {
            'final_structure': 'image',      # Closed, filled organism
            'internal_pressure': 'image',    # The Endoderm/Insides
            'closure_signal': 'signal',      # Negative signal to stop growth
            'topological_genus': 'signal'    # Number of folds/holes (structural complexity)
        }
        
        self.resolution = 256
        self.pressure_decay = float(pressure_decay)
        self.closure_strength = float(closure_strength)
        
        # --- THE FIXES ARE HERE (Initialization for safety) ---
        self.internal_pressure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.final_structure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.closure_signal = 0.0
        self.topological_genus = 0.0
        # -----------------------------------------------------

    def _get_pacman_boundary(self, tissue):
        """Converts the tissue blob into a clean binary mask and finds the boundary."""
        # 1. Binarize
        _, binary = cv2.threshold((tissue * 255).astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
        # 2. Smooth (fills small holes, prevents noise)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5)))
        # 3. Find boundary (Laplacian/Sobel)
        boundary = cv2.Laplacian(binary, cv2.CV_32F)
        boundary = np.abs(boundary)
        boundary = np.clip(boundary, 0, 1)
        return boundary, binary
        
    def _measure_closure_gap(self, binary_tissue):
        """Measures the largest gap in the tissue blob (the Pac-Man mouth)"""
        inverted = 255 - binary_tissue
        
        # Find the connected components in the inverted mask (the holes)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted)
        
        largest_gap_area = 0
        
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            if stats[i, cv2.CC_STAT_AREA] > largest_gap_area:
                 # Check if this component is truly inside the tissue boundary
                 # (Simplification: check if the centroid is far from the image edge)
                 if cx > 10 and cx < self.resolution - 10 and cy > 10 and cy < self.resolution - 10:
                      largest_gap_area = stats[i, cv2.CC_STAT_AREA]
        
        normalized_gap = largest_gap_area / (self.resolution**2)
        
        # Return how much closure is needed (negative growth)
        return -normalized_gap * self.closure_strength * 10.0

    def step(self):
        tissue_in = self.get_blended_input('tissue_structure', 'first')
        soliton_in = self.get_blended_input('guide_soliton', 'first')
        metabolic_sig = self.get_blended_input('metabolic_signal', 'sum') or 0.0
        
        if tissue_in is None:
             self.final_structure = np.zeros((self.resolution, self.resolution))
             return

        # 1. Endoskeleton (Internal Pressure/Metabolism)
        self.internal_pressure = self.internal_pressure * self.pressure_decay + metabolic_sig * 0.05
        
        # 2. Closure Signal (Contraction)
        boundary_vis, binary_tissue = self._get_pacman_boundary(tissue_in)
        
        # Measure how much the tissue needs to contract
        self.closure_signal = self._measure_closure_gap(binary_tissue)
        
        # 3. Final Assembly
        
        # Tissue interior is filled by pressure
        pressure_filled = self.internal_pressure * (binary_tissue / 255.0)
        
        # Final Structure = Boundary + Interior Pressure
        final = np.clip(boundary_vis + pressure_filled, 0, 1)

        # 4. Topological Genus (Folds/Holes)
        self.topological_genus = np.var(boundary_vis) # Simpler proxy: variance in boundary
        
        # Store for outputs
        self.final_structure = final

    def get_output(self, port_name):
        if port_name == 'final_structure':
            return self.final_structure
        elif port_name == 'internal_pressure':
            return self.internal_pressure
        elif port_name == 'closure_signal':
            return self.closure_signal
        elif port_name == 'topological_genus':
            return self.topological_genus
        return None

    def get_display_image(self):
        w, h = 512, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Left: Final Structure (Tissue + Endoskeleton)
        final_u8 = (self.final_structure * 255).astype(np.uint8)
        final_color = cv2.applyColorMap(final_u8, cv2.COLORMAP_JET)
        final_resized = cv2.resize(final_color, (h, h))
        img[:, :h] = final_resized
        
        # Right: Internal Pressure (Metabolism)
        pressure_u8 = (self.internal_pressure * 255).astype(np.uint8)
        pressure_color = cv2.applyColorMap(pressure_u8, cv2.COLORMAP_HOT)
        pressure_resized = cv2.resize(pressure_color, (w-h, h))
        img[:, h:] = pressure_resized
        
        # Overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'ORGANISM (SKIN+INSIDES)', (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(img, 'CLOSURE: {:.4f}'.format(self.closure_signal), (10, h - 30), font, 0.5, (0, 255, 0), 1)
        cv2.putText(img, 'GENUS: {:.3f}'.format(self.topological_genus), (10, h - 10), font, 0.5, (255, 255, 0), 1)
        
        return QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Pressure Decay", "pressure_decay", self.pressure_decay, None),
            ("Closure Strength", "closure_strength", self.closure_strength, None)
        ]