"""
Dual-Timescale Encoder Node
----------------------------
Implements the PKAS architecture: two latent spaces operating at different timescales

FAST PATHWAY (Phase Space / Dendritic):
- Small latent (8-16D)
- Updates every frame
- Captures texture, edges, motion
- Represents ephaptic field dynamics

SLOW PATHWAY (Semantic Space / Somatic):
- Large latent (64-256D)  
- Updates with momentum (temporal smoothing)
- Captures objects, meaning, context
- Represents synaptic integration

CONSCIOUSNESS = Mismatch between fast prediction and slow prediction
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("DualTimescaleEncoder: PyTorch not available")


class SimpleEncoder(nn.Module):
    """Lightweight convolutional encoder"""
    def __init__(self, latent_dim=8, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),   # 64->32
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16->8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)


class DualTimescaleEncoderNode(BaseNode):
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(100, 180, 220)
    
    def __init__(self, fast_dim=8, slow_dim=64, img_size=64, slow_momentum=0.9):
        super().__init__()
        self.node_title = "Dual Timescale Encoder"
        
        self.inputs = {
            'image_in': 'image',
        }
        
        self.outputs = {
            'fast_latent': 'spectrum',      # Phase space (dendritic)
            'slow_latent': 'spectrum',      # Semantic space (somatic)
            'mismatch': 'signal',           # Disagreement between them
            'fast_image': 'image',          # Reconstructed from fast
            'slow_image': 'image',          # Reconstructed from slow
        }
        
        if not TORCH_AVAILABLE:
            self.node_title = "Dual Encoder (NO TORCH!)"
            return
        
        self.fast_dim = int(fast_dim)
        self.slow_dim = int(slow_dim)
        self.img_size = int(img_size)
        self.slow_momentum = float(slow_momentum)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create encoders
        self.fast_encoder = SimpleEncoder(self.fast_dim, self.img_size).to(self.device)
        self.slow_encoder = SimpleEncoder(self.slow_dim, self.img_size).to(self.device)
        
        # State
        self.fast_latent = np.zeros(self.fast_dim, dtype=np.float32)
        self.slow_latent = np.zeros(self.slow_dim, dtype=np.float32)
        self.slow_latent_smoothed = np.zeros(self.slow_dim, dtype=np.float32)
        self.mismatch_value = 0.0
        
        # For visualization
        self.fast_img = np.zeros((img_size, img_size), dtype=np.float32)
        self.slow_img = np.zeros((img_size, img_size), dtype=np.float32)
        
    def step(self):
        if not TORCH_AVAILABLE:
            return
            
        img_in = self.get_blended_input('image_in', 'first')
        if img_in is None:
            return
        
        # Prepare image
        if img_in.dtype != np.float32:
            img_in = img_in.astype(np.float32)
        if img_in.max() > 1.0:
            img_in = img_in / 255.0
            
        img_resized = cv2.resize(img_in, (self.img_size, self.img_size))
        
        if img_resized.ndim == 3:
            img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            img = img_resized
            
        # Convert to torch
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Encode in both pathways
        with torch.no_grad():
            # FAST pathway: updates every frame
            fast = self.fast_encoder(x)
            self.fast_latent = fast.cpu().numpy().flatten().astype(np.float32)
            
            # SLOW pathway: updates with momentum
            slow = self.slow_encoder(x)
            slow_np = slow.cpu().numpy().flatten().astype(np.float32)
            
            # Apply temporal smoothing to slow pathway
            self.slow_latent_smoothed = (self.slow_latent_smoothed * self.slow_momentum + 
                                         slow_np * (1.0 - self.slow_momentum))
            self.slow_latent = self.slow_latent_smoothed
        
        # Calculate mismatch
        # Since dimensions differ, we need to project to common space
        # Use simple approach: normalized correlation in their respective spaces
        
        # Normalize both
        fast_norm = self.fast_latent / (np.linalg.norm(self.fast_latent) + 1e-8)
        slow_norm = self.slow_latent / (np.linalg.norm(self.slow_latent) + 1e-8)
        
        # Measure via reconstruction difference
        # Simple proxy: variance in fast vs variance in slow
        fast_var = np.var(self.fast_latent)
        slow_var = np.var(self.slow_latent)
        
        # Mismatch = how different their "information content" is
        self.mismatch_value = np.abs(fast_var - slow_var) / (fast_var + slow_var + 1e-8)
        
        # Generate simple visualizations
        # Fast: high-frequency patterns
        self.fast_img = np.outer(np.sin(self.fast_latent[:4] * 10), 
                                 np.cos(self.fast_latent[4:8] * 10))
        self.fast_img = cv2.resize(self.fast_img, (self.img_size, self.img_size))
        
        # Slow: low-frequency patterns  
        slow_vis = self.slow_latent[:16].reshape(4, 4)
        self.slow_img = cv2.resize(slow_vis, (self.img_size, self.img_size))
        
        # Normalize for display
        self.fast_img = (self.fast_img - self.fast_img.min()) / (self.fast_img.max() - self.fast_img.min() + 1e-8)
        self.slow_img = (self.slow_img - self.slow_img.min()) / (self.slow_img.max() - self.slow_img.min() + 1e-8)
    
    def get_output(self, port_name):
        if port_name == 'fast_latent':
            return self.fast_latent
        elif port_name == 'slow_latent':
            return self.slow_latent
        elif port_name == 'mismatch':
            return self.mismatch_value
        elif port_name == 'fast_image':
            return self.fast_img
        elif port_name == 'slow_image':
            return self.slow_img
        return None
    
    def get_display_image(self):
        if not TORCH_AVAILABLE:
            img = np.zeros((128, 256, 3), dtype=np.uint8)
            cv2.putText(img, "PyTorch not installed", (10, 64),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            return QtGui.QImage(img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)
        
        # Display: Fast (left) | Slow (right) | Mismatch bar (bottom)
        w, h = 256, 192
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Top: Fast and Slow latent visualizations
        fast_u8 = (np.clip(self.fast_img, 0, 1) * 255).astype(np.uint8)
        fast_color = cv2.applyColorMap(fast_u8, cv2.COLORMAP_TWILIGHT)
        fast_resized = cv2.resize(fast_color, (w//2, h*2//3))
        display[:h*2//3, :w//2] = fast_resized
        
        slow_u8 = (np.clip(self.slow_img, 0, 1) * 255).astype(np.uint8)
        slow_color = cv2.applyColorMap(slow_u8, cv2.COLORMAP_VIRIDIS)
        slow_resized = cv2.resize(slow_color, (w//2, h*2//3))
        display[:h*2//3, w//2:] = slow_resized
        
        # Bottom: Mismatch indicator
        mismatch_bar = int(self.mismatch_value * w)
        cv2.rectangle(display, (0, h*2//3), (mismatch_bar, h), (255, 0, 0), -1)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'FAST', (10, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, f'{self.fast_dim}D', (10, 40), font, 0.3, (200, 200, 200), 1)
        
        cv2.putText(display, 'SLOW', (w//2 + 10, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, f'{self.slow_dim}D', (w//2 + 10, 40), font, 0.3, (200, 200, 200), 1)
        
        cv2.putText(display, f'Mismatch: {self.mismatch_value:.4f}', 
                   (10, h - 10), font, 0.4, (255, 255, 0), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Fast Dim", "fast_dim", self.fast_dim, None),
            ("Slow Dim", "slow_dim", self.slow_dim, None),
            ("Image Size", "img_size", self.img_size, None),
            ("Slow Momentum", "slow_momentum", self.slow_momentum, None),
        ]
    
    def close(self):
        if hasattr(self, 'fast_encoder'):
            del self.fast_encoder
            del self.slow_encoder
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().close()