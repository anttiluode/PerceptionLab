"""
Latent Assembler & Decoder Node
Collects individual signal inputs and assembles them into a latent vector (spectrum).
It then DECODES this vector back into an image using a built-in VAE decoder.

- Assembles [in_0, in_1, ...] into a latent vector.
- Feeds this vector into a VAE decoder.
- Outputs the resulting image.

Requires: pip install torch torchvision
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: LatentAssemblerNode requires PyTorch for image output")
    print("Install with: pip install torch torchvision")


# --- VAE Model Class (Copied from RealVAENode) ---
# We need this class definition to create the decoder
class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder"""
    def __init__(self, latent_dim=16, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder (Not used here, but part of the class)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 8 → 4
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Latent space (Not used here)
        hidden_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: 16D latent → 64x64
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 32 → 64
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
# --- End of VAE Model Class ---


class LatentAssemblerNode(BaseNode):
    """
    Assembles multiple signal inputs into a single latent vector (spectrum).
    Can also passthrough a spectrum and modify specific components.
    **NEW**: Decodes the assembled vector into an image.
    """
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(150, 150, 150)
    
    def __init__(self, latent_dim=16, img_size=64): # Added img_size
        super().__init__()
        self.node_title = "Latent Assembler"
        
        self.latent_dim = int(latent_dim)
        self.img_size = int(img_size) # Store image size
        
        # Create inputs: one for each latent dimension
        self.inputs = {
            'latent_base': 'spectrum',  # Optional base latent vector
        }
        for i in range(self.latent_dim):
            self.inputs[f'in_{i}'] = 'signal'
        
        self.outputs = {
            'latent_out': 'spectrum',
            'image_out': 'image'  # --- NEW IMAGE OUTPUT ---
        }
        
        self.latent_vector = np.zeros(self.latent_dim, dtype=np.float32)
        
        # --- NEW VAE SETUP ---
        if not TORCH_AVAILABLE:
            self.node_title = "Latent Assembler (NO TORCH!)"
            self.model = None
            self.device = None
            self.reconstructed_image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            return

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model (only for decoding)
        try:
            self.model = ConvVAE(self.latent_dim, self.img_size).to(self.device)
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            print(f"Error initializing VAE in LatentAssembler: {e}")
            self.node_title = "Latent Assembler (VAE ERROR)"
            self.model = None
        
        self.reconstructed_image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        # --- END NEW VAE SETUP ---

    
    def step(self):
        # Start with base latent if provided
        base = self.get_blended_input('latent_base', 'first')
        
        if base is not None:
            # Use base as starting point
            if len(base) >= self.latent_dim:
                self.latent_vector = base[:self.latent_dim].astype(np.float32)
            else:
                # Pad if base is too short
                self.latent_vector = np.zeros(self.latent_dim, dtype=np.float32)
                self.latent_vector[:len(base)] = base.astype(np.float32)
        else:
            # Start from zeros
            self.latent_vector = np.zeros(self.latent_dim, dtype=np.float32)
        
        # Override with individual signal inputs (if connected)
        for i in range(self.latent_dim):
            signal_val = self.get_blended_input(f'in_{i}', 'sum')
            if signal_val is not None:
                self.latent_vector[i] = float(signal_val)
                
        # --- NEW: Decode the latent vector into an image ---
        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                try:
                    z = torch.from_numpy(self.latent_vector).float().unsqueeze(0).to(self.device)
                    recon = self.model.decode(z)
                    self.reconstructed_image = recon.squeeze().cpu().numpy()
                except Exception as e:
                    # Handle errors, e.g., if latent_dim doesn't match model
                    print(f"LatentAssembler decode error: {e}")
                    self.reconstructed_image *= 0.9 # Fade out
    
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_vector
        elif port_name == 'image_out':
            return self.reconstructed_image
        return None
    
    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        bar_width = max(1, w // self.latent_dim)
        
        # Normalize for display
        val_max = np.abs(self.latent_vector).max()
        if val_max < 1e-6:
            val_max = 1.0
        
        for i, val in enumerate(self.latent_vector):
            x = i * bar_width
            norm_val = val / val_max
            bar_h = int(abs(norm_val) * (h/2 - 10))
            y_base = h // 2
            
            if val >= 0:
                color = (0, int(255 * abs(norm_val)), 0)
                cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color, -1)
            else:
                color = (0, 0, int(255 * abs(norm_val)))
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color, -1)
            
            # Label every 4th
            if i % 4 == 0:
                cv2.putText(img, str(i), (x+2, h-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Baseline
        cv2.line(img, (0, h//2), (w, h//2), (100,100,100), 1)
        
        # Status
        cv2.putText(img, f"Dim: {self.latent_dim}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        if not TORCH_AVAILABLE:
            cv2.putText(img, "NO TORCH!", (w-80, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None),
            ("Image Size", "img_size", self.img_size, None) # --- NEW ---
        ]

    def close(self):
        # Clean up torch model
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().close()