"""
Real VAE Node - (v5 - Fixed Decoder-Only Training with Proper Loss)
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: RealVAENode requires PyTorch")


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder"""
    def __init__(self, latent_dim=16, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder: 64x64 -> 16D latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        hidden_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: 16D latent -> 64x64
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
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
    Real Variational Autoencoder
    v5: Properly trains decoder from external latent codes
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(180, 100, 220)
    
    def __init__(self, latent_dim=16, img_size=64):
        super().__init__()
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RealVAENode: Using device: {self.device}")
        
        self.model = ConvVAE(self.latent_dim, self.img_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # State
        self.current_latent = np.zeros(self.latent_dim, dtype=np.float32)
        self.reconstructed = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        self.current_loss = 0.0
        self.training_steps = 0
        
        # NEW: Target image buffer for decoder-only training
        self.target_image_buffer = []
        self.max_buffer_size = 50
        
    def vae_loss(self, recon, x, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.1 * kl_loss
    
    def step(self):
        if not TORCH_AVAILABLE:
            return
        
        img_in = self.get_blended_input('image_in', 'mean')
        train_signal = self.get_blended_input('train', 'sum') or 0.0
        reset_signal = self.get_blended_input('reset', 'sum') or 0.0
        external_latent = self.get_blended_input('latent_in', 'first')
        
        has_image = img_in is not None
        has_external_latent = external_latent is not None
        
        if not has_image and not has_external_latent:
            self.reconstructed *= 0.95
            return
        
        # Reset
        if reset_signal > 0.5:
            print("RealVAENode: Resetting training...")
            self.model = ConvVAE(self.latent_dim, self.img_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.training_steps = 0
            self.target_image_buffer = []
        
        # --- MODE 1: Full VAE Training (has image) ---
        if has_image:
            img = cv2.resize(img_in, (self.img_size, self.img_size))
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            
            x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
            if train_signal > 0.5:
                self.model.train()
                self.optimizer.zero_grad()
                
                recon, mu, logvar = self.model(x)
                loss = self.vae_loss(recon, x, mu, logvar)
                
                loss.backward()
                self.optimizer.step()
                
                self.current_loss = loss.item()
                self.training_steps += 1
                
                # Store reconstruction as potential training target
                with torch.no_grad():
                    self.target_image_buffer.append(recon.squeeze().cpu().numpy())
                    if len(self.target_image_buffer) > self.max_buffer_size:
                        self.target_image_buffer.pop(0)
                
                if self.training_steps % 50 == 0:
                    print(f"VAE Step {self.training_steps}, Loss: {self.current_loss:.2f}")

            # Always encode
            self.model.eval()
            with torch.no_grad():
                mu, logvar = self.model.encode(x)
                self.current_latent = mu.cpu().numpy().flatten().astype(np.float32)
        
        # --- MODE 2: Decoder-Only Training (latent only) ---
        elif has_external_latent and train_signal > 0.5:
            if len(external_latent) == self.latent_dim:
                self.model.train()
                
                # Decode the external latent
                z = torch.from_numpy(external_latent).float().unsqueeze(0).to(self.device)
                recon = self.model.decode(z)
                
                # STRATEGY 1: Regularization losses (no ground truth needed)
                # Encourage realistic images through various constraints
                
                # 1. Output should use full range (not collapse to gray)
                range_loss = -torch.mean(torch.abs(recon - 0.5))
                
                # 2. Output should have structure (not uniform)
                grad_x = recon[:, :, :, 1:] - recon[:, :, :, :-1]
                grad_y = recon[:, :, 1:, :] - recon[:, :, :-1, :]
                structure_loss = -torch.mean(torch.abs(grad_x)) - torch.mean(torch.abs(grad_y))
                
                # 3. Temporal consistency (smooth decoder)
                if len(self.target_image_buffer) > 0:
                    # Compare to a recent output
                    target = self.target_image_buffer[-1]
                    target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(self.device)
                    consistency_loss = F.mse_loss(recon, target_tensor) * 0.1
                else:
                    consistency_loss = torch.tensor(0.0).to(self.device)
                
                # 4. Latent magnitude penalty (keep latent codes reasonable)
                latent_loss = torch.mean(z ** 2) * 0.001
                
                # Total loss
                total_loss = (
                    range_loss * 1.0 +
                    structure_loss * 0.5 +
                    consistency_loss +
                    latent_loss
                )
                
                # Make sure loss is positive and meaningful
                total_loss = torch.abs(total_loss) + 0.01
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                self.current_loss = total_loss.item()
                self.training_steps += 1
                
                # Store output
                with torch.no_grad():
                    output_img = recon.squeeze().cpu().numpy()
                    self.target_image_buffer.append(output_img)
                    if len(self.target_image_buffer) > self.max_buffer_size:
                        self.target_image_buffer.pop(0)
                
                if self.training_steps % 50 == 0:
                    print(f"VAE (Decoder-only) Step {self.training_steps}, Loss: {self.current_loss:.4f}")
        
        # --- Decoding (for display) ---
        self.model.eval()
        z_to_decode = None
        
        if has_external_latent:
            if len(external_latent) == self.latent_dim:
                z_to_decode = torch.from_numpy(external_latent).float().unsqueeze(0).to(self.device)
                self.current_latent = external_latent.copy()
        elif has_image:
            z_to_decode = torch.from_numpy(self.current_latent).float().unsqueeze(0).to(self.device)
            
        if z_to_decode is not None:
            with torch.no_grad():
                recon = self.model.decode(z_to_decode)
                self.reconstructed = recon.squeeze().cpu().numpy()
        else:
            self.reconstructed *= 0.95
    
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.current_latent
        elif port_name == 'image_out':
            return self.reconstructed
        elif port_name == 'loss':
            # Better scaling for display
            return float(self.current_loss)
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

    def close(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().close()
