import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name): return None

class HaPPYHolographNode2(BaseNode):
    """
    HaPPY Holograph Node
    --------------------
    Implements a simplified HaPPY (Holographic Pentagon) code visualization.
    
    Maps boundary signal onto a hyperbolic Poincaré disk:
    - Boundary = UV (fine detail, edge of disk)
    - Bulk = IR (coarse structure, center of disk)
    
    Demonstrates:
    - Bulk emergence from boundary data
    - Error correction (center is stable despite boundary noise)
    - AdS/CFT-like correspondence
    
    The center "central charge" measures how much boundary info
    reaches the bulk - this is the holographic content.
    """
    NODE_CATEGORY = "Physics"
    NODE_COLOR = QtGui.QColor(50, 100, 200)  # Blue for holography
    
    def __init__(self):
        super().__init__()
        self.node_title = "HaPPY Holograph"
        
        self.inputs = {
            'boundary_signal': 'spectrum',
            'boundary_image': 'image',
        }
        
        self.outputs = {
            'hyperbolic_view': 'image',
            'central_charge': 'signal',
            'bulk_spectrum': 'spectrum',
            'boundary_entropy': 'signal',
        }
        
        self.config = {
            'disk_size': 256,
            'layers': 6,           # Radial layers from boundary to center
            'error_correction': 0.3,  # How much boundary noise is filtered
        }
        
        self._output_values = {}
        self._init_disk()

    def _init_disk(self):
        size = self.config['disk_size']
        self.disk = np.zeros((size, size, 3), dtype=np.float32)
        
        # Precompute hyperbolic coordinates
        y, x = np.ogrid[:size, :size]
        cx, cy = size // 2, size // 2
        
        # Euclidean radius (0 to 1 within disk)
        self.r_euclid = np.sqrt((x - cx)**2 + (y - cy)**2) / (size // 2)
        
        # Hyperbolic radius: r_hyp = 2 * arctanh(r_euclid)
        # Clamped to avoid infinity at edge
        r_clamped = np.clip(self.r_euclid, 0, 0.99)
        self.r_hyper = 2 * np.arctanh(r_clamped)
        
        # Angle
        self.theta = np.arctan2(y - cy, x - cx)
        
        # Mask for disk interior
        self.disk_mask = self.r_euclid < 1.0
        
        # Layer indices (0 = center, layers-1 = boundary)
        self.layer_idx = np.clip(
            (self.r_euclid * self.config['layers']).astype(int),
            0, self.config['layers'] - 1
        )

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'):
            return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value):
        self._output_values[name] = value
    
    def get_output(self, name):
        return self._output_values.get(name, None)

    def _boundary_to_bulk(self, boundary):
        """
        Propagate boundary signal inward via simplified holographic map.
        Each layer is average of outer layer + error correction.
        """
        layers = self.config['layers']
        ec = self.config['error_correction']
        
        # Partition boundary into angular sectors
        n_sectors = len(boundary)
        sector_values = np.zeros((layers, n_sectors))
        
        # Boundary layer = raw signal
        sector_values[-1, :] = boundary
        
        # Propagate inward (renormalization flow)
        for layer in range(layers - 2, -1, -1):
            outer = sector_values[layer + 1, :]
            
            # Average adjacent sectors (local averaging = error correction)
            smoothed = (
                np.roll(outer, 1) + outer + np.roll(outer, -1)
            ) / 3.0
            
            # Mix raw and smoothed (error correction strength)
            sector_values[layer, :] = outer * (1 - ec) + smoothed * ec
        
        return sector_values

    def step(self):
        # Get boundary signal
        signal = self.get_input('boundary_signal')
        image = self.get_input('boundary_image')
        
        if signal is not None:
            boundary = np.array(signal, dtype=np.float32).flatten()
        elif image is not None:
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            # Extract boundary from image edge
            boundary = np.concatenate([
                image[0, :],      # Top
                image[:, -1],     # Right
                image[-1, ::-1],  # Bottom (reversed)
                image[::-1, 0]    # Left (reversed)
            ])
        else:
            # Default: random boundary
            boundary = np.random.randn(64) * 0.5
        
        # Normalize
        boundary = boundary / (np.max(np.abs(boundary)) + 1e-10)
        
        # Ensure sufficient resolution
        if len(boundary) < 32:
            boundary = np.interp(
                np.linspace(0, 1, 64),
                np.linspace(0, 1, len(boundary)),
                boundary
            )
        
        # === HOLOGRAPHIC MAP ===
        bulk_layers = self._boundary_to_bulk(boundary)
        
        # === RENDER DISK ===
        size = self.config['disk_size']
        layers = self.config['layers']
        n_sectors = len(boundary)
        
        self.disk = np.zeros((size, size, 3), dtype=np.float32)
        
        for y in range(size):
            for x in range(size):
                if not self.disk_mask[y, x]:
                    continue
                
                layer = self.layer_idx[y, x]
                theta = self.theta[y, x]
                
                # Map angle to sector
                sector = int((theta + np.pi) / (2 * np.pi) * n_sectors) % n_sectors
                
                value = bulk_layers[layer, sector]
                
                # Color: depth encodes layer, brightness encodes value
                depth_color = 1.0 - layer / layers  # Center is bright
                
                # Blue channel = depth, Green/Red = value
                self.disk[y, x, 0] = max(0, value) * depth_color  # Red for positive
                self.disk[y, x, 1] = depth_color * 0.3            # Base green
                self.disk[y, x, 2] = max(0, -value) * depth_color + depth_color * 0.5  # Blue for negative + base
        
        # Enhance contrast
        self.disk = np.clip(self.disk * 1.5, 0, 1)
        
        # === OUTPUTS ===
        self.set_output('hyperbolic_view', (self.disk * 255).astype(np.uint8))
        
        # Central charge (bulk center value)
        center_layer = bulk_layers[0, :]
        central_charge = np.mean(np.abs(center_layer))
        self.set_output('central_charge', float(central_charge))
        
        # Bulk spectrum (radial profile)
        bulk_spectrum = np.mean(np.abs(bulk_layers), axis=1)
        self.set_output('bulk_spectrum', bulk_spectrum)
        
        # Boundary entropy
        probs = np.abs(boundary) / (np.sum(np.abs(boundary)) + 1e-10)
        probs = probs[probs > 1e-10]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        self.set_output('boundary_entropy', float(entropy))
