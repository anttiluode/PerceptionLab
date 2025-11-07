"""
Space Screensaver Node - A 3D tensor universe simulation
Ported from the SpaceScreensaver.py script.
Requires: pip install torch scipy
Place this file in the 'nodes' folder
"""

import numpy as np
import cv2
import sys
import os

# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui # <--- THIS IS THE FIX
# ------------------------------------

# --- Dependency Checks ---
try:
    import torch
    from scipy.ndimage import label
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("Warning: SpaceScreensaverNode requires 'torch' and 'scipy'.")
    print("Please run: pip install torch scipy")

# --- Color Map Dictionary ---
# Maps string names to OpenCV colormap constants
CMAP_DICT = {
    "gray": None, # Special case for no colormap
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "hot": cv2.COLORMAP_HOT,
    "jet": cv2.COLORMAP_JET
}

# --- Core Simulation Classes (from SpaceScreensaver.py) ---
# These are helper classes, placed inside the node file for portability

class PhysicalTensorSingularity:
    def __init__(self, dimension=128, position=None, mass=1.0, device='cpu'):
        self.dimension = dimension
        self.device = device
        # Physical properties
        if position is not None:
            if isinstance(position, np.ndarray):
                self.position = torch.from_numpy(position).float().to(self.device)
            else:
                self.position = position.clone().detach().float().to(self.device)
        else:
            self.position = torch.tensor(np.random.rand(3), dtype=torch.float32, device=self.device)
        self.velocity = torch.randn(3, device=self.device) * 0.1
        self.mass = mass
        # Tensor properties
        self.core = torch.randn(dimension, device=self.device)
        self.field = self.generate_gravitational_field()

    def generate_gravitational_field(self):
        field = self.core.clone()
        r = torch.linspace(0, 2 * np.pi, self.dimension, device=self.device)
        field *= torch.exp(-r / self.mass)
        return field

    def update_position(self, dt, force):
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class PhysicalTensorUniverse:
    def __init__(self, size=50, num_singularities=100, dimension=128, device='cpu'):
        self.G = 6.67430e-11  # Gravitational constant
        self.size = size
        self.dimension = dimension
        self.device = device
        self.space = torch.zeros((size, size, size), device=self.device)
        self.singularities = []
        self.initialize_singularities(num_singularities)

    def initialize_singularities(self, num):
        """Initialize singularities with random positions and masses"""
        self.singularities = []  # Reset list
        for _ in range(num):
            position = torch.tensor(np.random.rand(3) * self.size, dtype=torch.float32, device=self.device)
            mass = torch.distributions.Exponential(1.0).sample().item()
            self.singularities.append(
                PhysicalTensorSingularity(
                    dimension=self.dimension,
                    position=position,
                    mass=mass,
                    device=self.device
                )
            )

    def update_tensor_interactions(self):
        """Update tensor field interactions using vectorized operations"""
        if not self.singularities:
            return
            
        positions = torch.stack([s.position for s in self.singularities])
        masses = torch.tensor([s.mass for s in self.singularities], device=self.device)

        delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance = torch.norm(delta, dim=2) + 1e-10
        force_magnitude = self.G * masses.unsqueeze(1) * masses.unsqueeze(0) / (distance ** 2)
        force_direction = delta / (distance.unsqueeze(2) + 1e-10)
        
        # Zero out self-interaction
        force_magnitude.fill_diagonal_(0)
        
        force = torch.sum(force_magnitude.unsqueeze(2) * force_direction, dim=1)

        fields = torch.stack([s.field for s in self.singularities])
        field_interaction = torch.tanh(torch.matmul(fields, fields.T))
        force *= (1 + torch.mean(field_interaction, dim=1)).unsqueeze(1)

        for i, singularity in enumerate(self.singularities):
            singularity.update_position(dt=0.1, force=force[i])

    def update_space(self):
        """Update 3D space based on singularity positions and fields"""
        self.space.fill_(0)
        x = torch.linspace(0, self.size-1, self.size, device=self.device)
        y = torch.linspace(0, self.size-1, self.size, device=self.device)
        z = torch.linspace(0, self.size-1, self.size, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        for s in self.singularities:
            R = torch.sqrt((X - s.position[0]) ** 2 +
                          (Y - s.position[1]) ** 2 +
                          (Z - s.position[2]) ** 2)
            self.space += s.mass / (R + 1) * torch.mean(s.field)

# --- The Main Node Class ---

class SpaceScreensaverNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(50, 80, 160) # Deep space blue
    
    def __init__(self, universe_size=48, num_singularities=100, color_scheme='plasma'):
        super().__init__()
        self.node_title = "Space Screensaver"
        
        self.inputs = {'reset': 'signal'}
        self.outputs = {'image': 'image', 'total_mass': 'signal'}
        
        if not LIBS_AVAILABLE:
            self.node_title = "Space (Libs Missing!)"
            return
            
        self.universe_size = int(universe_size)
        self.num_singularities = int(num_singularities)
        self.color_scheme = str(color_scheme)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize simulation
        self.simulation = PhysicalTensorUniverse(
            size=self.universe_size,
            num_singularities=self.num_singularities,
            device=self.device
        )
        
        self.output_image_data = np.zeros((self.universe_size, self.universe_size), dtype=np.float32)
        self.total_mass = 0.0

    def randomize(self):
        """Called by 'R' button - re-initializes the simulation"""
        if LIBS_AVAILABLE:
            self.simulation.initialize_singularities(self.num_singularities)
            
    def _get_density_slice(self):
        """Internal helper to get a 2D slice from the 3D sim"""
        if not LIBS_AVAILABLE:
            return
            
        # Get the middle slice on the Z axis
        slice_index = self.universe_size // 2
        density_slice = self.simulation.space[:, :, slice_index].cpu().numpy()

        # Normalize the density slice for visualization
        min_v, max_v = density_slice.min(), density_slice.max()
        range_v = max_v - min_v
        if range_v > 1e-9:
            self.output_image_data = (density_slice - min_v) / range_v
        else:
            self.output_image_data.fill(0.0)

    def step(self):
        if not LIBS_AVAILABLE:
            return
            
        # Check for reset signal
        reset_sig = self.get_blended_input('reset', 'sum')
        if reset_sig is not None and reset_sig > 0.5:
            self.randomize()
            
        # Run simulation steps
        self.simulation.update_tensor_interactions()
        self.simulation.update_space()
        
        # Get 2D image data
        self._get_density_slice()
        
        # Get metrics
        self.total_mass = float(torch.sum(self.simulation.space).item())

    def get_output(self, port_name):
        if port_name == 'image':
            return self.output_image_data
        elif port_name == 'total_mass':
            return self.total_mass
        return None
        
    def get_display_image(self):
        if not LIBS_AVAILABLE:
            return None
            
        img_u8 = (np.clip(self.output_image_data, 0, 1) * 255).astype(np.uint8)
        
        # Apply the selected colormap
        cmap_cv2 = CMAP_DICT.get(self.color_scheme)
        
        if cmap_cv2 is not None:
            # Apply CV2 colormap and resize
            img_color = cv2.applyColorMap(img_u8, cmap_cv2)
            img_resized = cv2.resize(img_color, (96, 96), interpolation=cv2.INTER_LINEAR)
            img_resized = np.ascontiguousarray(img_resized)
            h, w = img_resized.shape[:2]
            return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
        else:
            # Just resize (for 'gray')
            img_resized = cv2.resize(img_u8, (96, 96), interpolation=cv2.INTER_LINEAR)
            img_resized = np.ascontiguousarray(img_resized)
            h, w = img_resized.shape
            return QtGui.QImage(img_resized.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)


    def get_config_options(self):
        if not LIBS_AVAILABLE:
            return [("Error", "error", "PyTorch or SciPy not found!", [])]
            
        # Create color scheme options for the dropdown
        color_options = [(name.title(), name) for name in CMAP_DICT.keys()]
        
        return [
            ("Universe Size (3D)", "universe_size", self.universe_size, None),
            ("Num Singularities", "num_singularities", self.num_singularities, None),
            ("Color Scheme", "color_scheme", self.color_scheme, color_options),
        ]