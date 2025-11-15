"""
Latent Decoder Node
-------------------
This node REPLACES the HebbianDecoderNode.

It learns to take an abstract 2D "latent image" from the
LatentEncoderNode and reconstruct the original, full-size photo.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

# --- Dependency Check ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: LatentDecoderNode requires 'torch', 'torchvision', and 'Pillow'.")
    print("Please run: pip install torch torchvision pillow")

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")

# --- Architecture from megalivingmirror3video.py ---
# --- MODIFIED to accept 1-channel latent space ---
class MegaDecoder(nn.Module):
    def __init__(self, out_ch=3):
        super().__init__()
        self.up1 = nn.Sequential(
            # MODIFIED: Input 1 channel (from encoder) instead of 16
            nn.Conv2d(1, 1024, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 768, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(768, 768, 3, 1, 1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(256, out_ch, 3, 1, 1)

    def forward(self, z):
        # z is [batch, 1, 64, 64]
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        x = torch.sigmoid(self.final_conv(z))  # [0,1]
        return x # Output shape [batch, 3, 512, 512]

# --- Perception Lab Node ---
class LatentDecoderNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 80, 120)  # Pink for decoder
    
    def __init__(self, learning_rate=0.0005): # Slower LR for VAEs
        super().__init__()
        self.node_title = "Latent Decoder (Latent-to-Image)"
        
        self.inputs = {
            'latents_in': 'image',       # The 64x64 latent image
            'target_image': 'image',     # Ground truth for training
            'train_signal': 'signal',    # 1.0 = train, 0.0 = inference
        }
        self.outputs = {
            'reconstructed': 'image',    # The decoded image
            'loss': 'signal',            # Reconstruction error
        }
        
        if not TORCH_AVAILABLE:
            self.node_title = "Latent Decoder (MISSING TORCH!)"
            return
        
        self.base_learning_rate = float(learning_rate)
        
        self.model = MegaDecoder().to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.base_learning_rate
        )
        
        # Transform for the target image
        self.target_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor() # Output is 0-1, so target must be 0-1
        ])
        
        self.reconstructed_image = np.zeros((512, 512, 3), dtype=np.float32)
        self.current_loss = 0.0
        self.training_steps = 0
        
    def step(self):
        if not TORCH_AVAILABLE:
            return
        
        latents_in = self.get_blended_input('latents_in', 'first')
        target_image = self.get_blended_input('target_image', 'first')
        train_signal = self.get_blended_input('train_signal', 'sum') or 0.0
        
        if latents_in is None:
            return
        
        # 1. Convert latents (64, 64) numpy to [1, 1, 64, 64] tensor
        latents_tensor = torch.from_numpy(latents_in).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        # 2. Forward pass
        # --- THIS IS THE FIX ---
        # We must cast the numpy.bool_ to a native python bool
        is_training = bool(train_signal > 0.5)
        # --- END FIX ---
        
        with torch.set_grad_enabled(is_training):
            # Output is [1, 3, 512, 512]
            reconstructed_tensor = self.model(latents_tensor)
        
        # 3. Store reconstruction as numpy image
        self.reconstructed_image = reconstructed_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # 4. Training mode
        if is_training and target_image is not None:
            # Prepare target
            img_u8 = (np.clip(target_image, 0, 1) * 255).astype(np.uint8)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
                
            target_tensor = self.target_transform(img_u8).to(DEVICE) # No unsqueeze, T.ToTensor() does it
            
            # Compute loss
            loss = F.mse_loss(reconstructed_tensor.squeeze(0), target_tensor)
            self.current_loss = loss.item()
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.training_steps += 1
            
        elif target_image is not None: # Inference mode, but compute loss
            img_u8 = (np.clip(target_image, 0, 1) * 255).astype(np.uint8)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
            target_np = self.target_transform(img_u8).squeeze(0).permute(1, 2, 0).numpy()
            diff = self.reconstructed_image - target_np
            self.current_loss = np.mean(diff ** 2)
        else:
            self.current_loss = 0.0
    
    def get_output(self, port_name):
        if port_name == 'reconstructed':
            return self.reconstructed_image
        elif port_name == 'loss':
            return self.current_loss
        return None
    
    def get_display_image(self):
        img = self.reconstructed_image
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # --- THIS IS THE FIX ---
        # Force a C-contiguous memory layout *before* passing to OpenCV
        # The original array from .permute() is not compatible.
        img_u8 = np.ascontiguousarray(img_u8)
        # --- END FIX ---
        
        # Add info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        status = "TRAINING" if (self.get_blended_input('train_signal', 'sum') or 0.0) > 0.5 else "INFERENCE"
        cv2.putText(img_u8, status, (10, 25), font, 0.7, (0, 255, 0), 2)
        cv2.putText(img_u8, f"Loss: {self.current_loss:.4f}", (10, 50), 
                   font, 0.7, (0, 255, 0), 2)
        cv2.putText(img_u8, f"Steps: {self.training_steps}", (10, 75),
                   font, 0.7, (0, 255, 0), 2)
        
        img_resized = np.ascontiguousarray(img_u8)
        h, w = img_resized.shape[:2]
        # Display is 512x512, 3-channel
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Learning Rate", "base_learning_rate", self.base_learning_rate, None),
        ]