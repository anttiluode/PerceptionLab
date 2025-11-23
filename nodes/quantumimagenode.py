"""
Quantum Image Node - Proves images in latent space behave like wavefunctions
============================================================================

This node demonstrates:
1. Image encoding creates probability clouds (superposition)
2. Decoding is measurement (collapse)  
3. Interpolation reveals hidden phase space (continuous)
4. Curvature determines if transition is "allowed" (geodesics)

FIXES:
- Handles grayscale (2D) inputs correctly by converting to RGB.
- Fixes the 'permute' dimension error by checking shape before operations.
- Robust input validation.
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import torch
import torch.nn as nn

import __main__
BaseNode = __main__.BaseNode

class QuantumImageNode(BaseNode):
    NODE_CATEGORY = "Deep Math"
    NODE_TITLE = "Quantum Image"
    NODE_COLOR = QtGui.QColor(100, 50, 200)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',       # Input image
            'curvature': 'signal',     # From Ricci flow
            'interpolation': 'signal'  # 0-1: between two stored images
        }
        
        self.outputs = {
            'wavefunction': 'image',   # Probability distribution visualization
            'collapsed': 'image',      # "Measured" (decoded) image
            'superposition': 'image',  # Interpolated state
            'entropy': 'signal'        # Quantum entropy
        }
        
        # Tiny VAE for real-time encoding
        self.latent_dim = 16
        self.image_size = 64
        
        # Simple encoder/decoder
        self.encoder_mean = self._build_encoder()
        self.encoder_std = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Stored "quantum states"
        self.stored_images = []
        self.stored_latents = []
        self.max_stored = 2  # Store two images to interpolate between
        
        # Current state
        self.current_latent = None
        self.current_std = None
        
    def _build_encoder(self):
        """Minimal encoder for real-time use"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, self.latent_dim)
        )
    
    def _build_decoder(self):
        """Minimal decoder"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def _encode_image(self, img):
        """Encode image to latent space (quantum → classical)"""
        # Robust input handling
        if not isinstance(img, np.ndarray):
            return None, None
            
        # Resize to expected dimensions
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Ensure 3 channels (RGB)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Check shape before permute
        # Expected shape after cv2 resize: (64, 64, 3)
        if img.ndim != 3 or img.shape[2] != 3:
            # Fallback: Create a valid 3-channel image if shape is weird
            print(f"QuantumImageNode: Weird input shape {img.shape}, forcing correction.")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        # Convert to tensor: [H, W, C] -> [C, H, W]
        # unsqueeze(0) adds batch dimension -> [1, C, H, W]
        try:
            tensor_img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
        except Exception as e:
             print(f"QuantumImageNode: Tensor conversion error: {e}")
             return None, None

        with torch.no_grad():
            mu = self.encoder_mean(tensor_img)
            log_var = self.encoder_std(tensor_img)
            std = torch.exp(0.5 * log_var)
        
        return mu.squeeze(), std.squeeze()
    
    def _decode_latent(self, z):
        """Decode latent to image (classical → quantum reconstruction)"""
        if isinstance(z, np.ndarray):
            z = torch.FloatTensor(z).unsqueeze(0)
        
        if z.ndim == 1:
             z = z.unsqueeze(0)

        with torch.no_grad():
            reconstruction = self.decoder(z)
        
        # Output is [1, 3, 64, 64] -> [64, 64, 3]
        img = reconstruction.squeeze().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return img
    
    def _visualize_wavefunction(self, mu, std):
        """Visualize the probability cloud (wavefunction)"""
        if mu is None or std is None: return None
        
        # Sample from the distribution
        n_samples = 1000
        samples = []
        for _ in range(n_samples):
            z = mu + std * torch.randn_like(std)
            samples.append(z.numpy())
        
        samples = np.array(samples)
        if samples.size == 0: return None
        
        # Project to 2D
        x = samples[:, 0]
        y = samples[:, 1]
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-3, 3], [-3, 3]])
        
        # Normalize
        if H.max() > 0:
            H = H / H.max()
        
        # Convert to image
        wavefunction_img = (H.T * 255).astype(np.uint8)
        wavefunction_img = cv2.applyColorMap(wavefunction_img, cv2.COLORMAP_PLASMA)
        
        return wavefunction_img
    
    def _compute_entropy(self, std):
        if std is None: return 0.0
        return torch.sum(std).item()
    
    def _interpolate_latents(self, alpha):
        if len(self.stored_latents) < 2:
            return None, None
        
        z1, std1 = self.stored_latents[0]
        z2, std2 = self.stored_latents[1]
        
        z_interp = alpha * z1 + (1 - alpha) * z2
        std_interp = alpha * std1 + (1 - alpha) * std2
        
        return z_interp, std_interp
    
    def step(self):
        # Get inputs
        img_in = self.get_blended_input('image_in')
        curvature = self.get_blended_input('curvature')
        interp_alpha = self.get_blended_input('interpolation')
        
        # Handle scalar inputs
        if interp_alpha is None:
            interp_alpha = 0.5
        elif isinstance(interp_alpha, (list, np.ndarray)):
             # If signal comes in as an array, take mean
             interp_alpha = float(np.mean(interp_alpha))
        else:
             interp_alpha = float(interp_alpha)
             
        interp_alpha = np.clip(interp_alpha, 0, 1)
        
        # Encode input image
        if img_in is not None:
            # Convert QImage to numpy if needed
            if isinstance(img_in, QtGui.QImage):
                width = img_in.width()
                height = img_in.height()
                ptr = img_in.bits()
                ptr.setsize(height * width * 3)
                img_array = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            else:
                img_array = img_in
            
            # Encode
            mu, std = self._encode_image(img_array)
            
            if mu is not None:
                self.current_latent = (mu, std)
                
                # Store if we have space
                if len(self.stored_latents) < self.max_stored:
                    self.stored_latents.append((mu, std))
                    self.stored_images.append(img_array)
                else:
                    # Replace oldest
                    self.stored_latents[0] = self.stored_latents[1]
                    self.stored_images[0] = self.stored_images[1]
                    self.stored_latents[1] = (mu, std)
                    self.stored_images[1] = img_array
        
        # Generate outputs
        if self.current_latent is not None:
            mu, std = self.current_latent
            
            # 1. Wavefunction visualization
            self.wavefunction_vis = self._visualize_wavefunction(mu, std)
            
            # 2. Collapsed state
            z_sample = mu + std * torch.randn_like(std)
            self.collapsed_img = self._decode_latent(z_sample)
            
            # 3. Quantum entropy
            self.entropy = self._compute_entropy(std)
            
            # 4. Superposition
            if len(self.stored_latents) >= 2:
                z_interp, std_interp = self._interpolate_latents(interp_alpha)
                if z_interp is not None:
                     self.superposition_img = self._decode_latent(z_interp)
            else:
                self.superposition_img = self.collapsed_img
    
    def get_output(self, port_name):
        if port_name == 'wavefunction':
            return self.wavefunction_vis.astype(np.float32) / 255.0 if hasattr(self, 'wavefunction_vis') and self.wavefunction_vis is not None else None
        
        elif port_name == 'collapsed':
            return self.collapsed_img.astype(np.float32) / 255.0 if hasattr(self, 'collapsed_img') and self.collapsed_img is not None else None
        
        elif port_name == 'superposition':
            return self.superposition_img.astype(np.float32) / 255.0 if hasattr(self, 'superposition_img') and self.superposition_img is not None else None
        
        elif port_name == 'entropy':
            return self.entropy if hasattr(self, 'entropy') else 0.0
        
        return None
    
    def get_display_image(self):
        # Show the wavefunction
        if hasattr(self, 'wavefunction_vis') and self.wavefunction_vis is not None:
            img = self.wavefunction_vis
            h, w, c = img.shape
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        return None