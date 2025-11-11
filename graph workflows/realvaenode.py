"""
Real VAE Node - Actually learns non-linear visual compression
Trains incrementally on webcam, allows latent space exploration

Requires: pip install torch torchvision
Place this file in the 'nodes' folder as realvaenode.py
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
    print("Warning: RealVAENode requires PyTorch")
    print("Install with: pip install torch torchvision")


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder"""
    def __init__(self, latent_dim=16, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder: 64x64 → 16D latent
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
        
        # Latent space
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


class RealVAENode(BaseNode):
    """
    Real Variational Autoencoder - learns visual compression
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(180, 100, 220)
    
    def __init__(self, latent_dim=16, img_size=64):
        super().__init__()  # NO ARGUMENTS - this is the fix!
        self.node_title = "Real VAE"
        
        self.inputs = {
            'image_in': 'image',
            'latent_in': 'spectrum',
            'train': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'image_out': 'image',
            'latent_out': 'spectrum',
            'loss': 'signal'
        }
        
        if not TORCH_AVAILABLE:
            self.node_title = "Real VAE (NO TORCH!)"
            return
        
        self.latent_dim = int(latent_dim)
        self.img_size = int(img_size)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RealVAENode: Using device: {self.device}")
        
        # Create model
        self.model = ConvVAE(self.latent_dim, self.img_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # State
        self.current_latent = np.zeros(self.latent_dim, dtype=np.float32)
        self.reconstructed = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        self.current_loss = 0.0
        self.training_steps = 0
        
    def vae_loss(self, recon, x, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.1 * kl_loss
    
    def step(self):
        if not TORCH_AVAILABLE:
            return
        
        # Get inputs
        img_in = self.get_blended_input('image_in', 'mean')
        train_signal = self.get_blended_input('train', 'sum') or 0.0
        reset_signal = self.get_blended_input('reset', 'sum') or 0.0
        
        # Reset training
        if reset_signal > 0.5:
            print("RealVAENode: Resetting training...")
            self.model = ConvVAE(self.latent_dim, self.img_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.training_steps = 0
        
        if img_in is None:
            return
        
        # Prepare image
        img = cv2.resize(img_in, (self.img_size, self.img_size))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Convert to torch tensor
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # TRAINING MODE
        if train_signal > 0.5:
            self.model.train()
            self.optimizer.zero_grad()
            
            recon, mu, logvar = self.model(x)
            loss = self.vae_loss(recon, x, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            self.current_loss = loss.item()
            self.training_steps += 1
            
            if self.training_steps % 50 == 0:
                print(f"VAE Step {self.training_steps}, Loss: {self.current_loss:.2f}")
        
        # INFERENCE MODE
        self.model.eval()
        with torch.no_grad():
            external_latent = self.get_blended_input('latent_in', 'first')
            
            if external_latent is not None and len(external_latent) == self.latent_dim:
                z = torch.from_numpy(external_latent).float().unsqueeze(0).to(self.device)
                recon = self.model.decode(z)
                self.current_latent = external_latent.copy()
            else:
                mu, logvar = self.model.encode(x)
                z = mu
                recon = self.model.decode(z)
                self.current_latent = z.cpu().numpy().flatten().astype(np.float32)
            
            self.reconstructed = recon.squeeze().cpu().numpy()
    
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.current_latent
        elif port_name == 'image_out':
            return self.reconstructed
        elif port_name == 'loss':
            return self.current_loss
        return None
    
    def get_display_image(self):
        if not TORCH_AVAILABLE:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.putText(img, "PyTorch not installed", (10, 64),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)
        
        img = (np.clip(self.reconstructed, 0, 1) * 255).astype(np.uint8)
        img = cv2.resize(img, (256, 256))
        
        status = f"Steps: {self.training_steps}"
        loss_text = f"Loss: {self.current_loss:.1f}"
        
        cv2.putText(img, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1)
        cv2.putText(img, loss_text, (5, 35), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1)
        
        device_text = "GPU" if self.device.type == 'cuda' else "CPU"
        cv2.putText(img, device_text, (5, 250), cv2.FONT_HERSHEY_SIMPLEX,
                   0.3, (0, 255, 0) if self.device.type == 'cuda' else (255, 255, 0), 1)
        
        return QtGui.QImage(img.data, 256, 256, 256, QtGui.QImage.Format.Format_Grayscale8)
    
    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None),
            ("Image Size", "img_size", self.img_size, None)
        ]