"""
Latent Encoder Node
-------------------
Takes a full-size image and compresses it down to a 
2D "latent image" using the 'MegaEncoder' architecture.

This node learns the *essence* of the image.
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
    import torchvision.transforms as T
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: LatentEncoderNode requires 'torch', 'torchvision', and 'Pillow'.")
    print("Please run: pip install torch torchvision pillow")

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")

# --- Architecture from megalivingmirror3video.py ---
# --- MODIFIED to output 1-channel latent space ---
class MegaEncoder(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(512, 768, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(768, 768, 3, 1, 1),
            nn.ReLU()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(768, 1024, 3, 1, 1),
            nn.ReLU(),
            # MODIFIED: Output 1 channel (a 2D latent image) instead of 16
            nn.Conv2d(1024, 1, 3, 1, 1) 
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.final_conv(x)
        return x # Output shape [batch, 1, 64, 64]

# --- Perception Lab Node ---
class LatentEncoderNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(80, 120, 255) # Blue for encoder

    def __init__(self):
        super().__init__()
        self.node_title = "Latent Encoder (Image-to-Latent)"
        
        self.inputs = { 'image_in': 'image' }
        self.outputs = { 'latents_out': 'image' }
        
        if not TORCH_AVAILABLE:
            self.node_title = "Latent Encoder (MISSING TORCH!)"
            return
            
        self.model = MegaEncoder().to(DEVICE)
        self.model.eval() # This node doesn't train, it just encodes
        
        # Transform for the input image
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)), # Based on megalivingmirror
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.latents_output = np.zeros((64, 64), dtype=np.float32)

    def step(self):
        if not TORCH_AVAILABLE:
            return
            
        image_in = self.get_blended_input('image_in', 'first')
        if image_in is None:
            return

        # 1. Convert Perception Lab image (float 0-1) to torch tensor
        # We must convert to uint8 for ToPILImage()
        img_u8 = (np.clip(image_in, 0, 1) * 255).astype(np.uint8)
        if img_u8.ndim == 2: # Handle grayscale input
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

        tensor = self.transform(img_u8).unsqueeze(0).to(DEVICE)
        
        # 2. Pass through encoder
        with torch.no_grad():
            # Output is [1, 1, 64, 64]
            latents_tensor = self.model(tensor)
            
        # 3. Convert back to numpy for Perception Lab
        # Squeeze to [64, 64]
        self.latents_output = latents_tensor.detach().cpu().squeeze().numpy()

    def get_output(self, port_name):
        if port_name == 'latents_out':
            return self.latents_output
        return None

    def get_display_image(self):
        # We can visualize the 2D latent space
        img = self.latents_output
        # Normalize for display
        norm_img = img - img.min()
        if norm_img.max() > 0:
            norm_img /= norm_img.max()
            
        img_u8 = (norm_img * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        display_size = 256
        img_resized = cv2.resize(img_color, (display_size, display_size), 
                                 interpolation=cv2.INTER_NEAREST)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)