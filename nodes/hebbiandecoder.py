"""
Hebbian Decoder Node (v2) - "Reading Thoughts"
------------------------------------------------
This node learns to decode/reconstruct the original sensory input
from the Hebbian W-matrix alone.

v2: Adds an "Inference Mode."
- If 'train_signal' is ON and 'target_image' is connected,
  it learns the mapping (updates its "key").
- If 'train_signal' is OFF or 'target_image' is missing,
  it "infers" (applies its frozen "key") to the 'w_matrix_in'.
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

# --- Dependency Check ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: HebbianDecoderNode requires 'torch'.")
    print("Please run: pip install torch")

# Use GPU if available
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cpu")


class SimpleDecoder(nn.Module):
    """Simple MLP decoder: W-matrix -> image"""
    def __init__(self, w_dim, image_size=64):
        super().__init__()
        self.w_dim = w_dim
        self.image_size = image_size
        hidden = 512
        
        self.decoder = nn.Sequential(
            nn.Linear(w_dim * w_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, image_size * image_size),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def forward(self, w_matrix_flat):
        img_flat = self.decoder(w_matrix_flat)
        return img_flat.view(-1, 1, self.image_size, self.image_size)


class HebbianDecoderNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(220, 100, 100) # Decoder Red
    
    def __init__(self, w_dim=16, image_size=64, base_learning_rate=0.001):
        super().__init__()
        self.node_title = "Hebbian Decoder"
        
        self.inputs = {
            'w_matrix_in': 'image',
            'target_image': 'image', # The "Answer Key"
            'train_signal': 'signal' # The "Teacher"
        }
        self.outputs = {
            'reconstructed': 'image',
            'loss': 'signal'
        }

        if not TORCH_AVAILABLE:
            self.node_title = "Decoder (NO TORCH!)"
            return
            
        self.w_dim = int(w_dim)
        self.image_size = int(image_size)
        self.lr = float(base_learning_rate)
        
        # --- The "Student's Brain" (The "Key") ---
        self.decoder_model = SimpleDecoder(self.w_dim, self.image_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.decoder_model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # State
        self.reconstructed_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        self.current_loss = 0.0
        self.training_steps = 0
        self.status = "WAITING"

    def step(self):
        if not TORCH_AVAILABLE:
            return

        # 1. Get Inputs
        w_matrix = self.get_blended_input('w_matrix_in', 'first')
        target_image = self.get_blended_input('target_image', 'first')
        train_signal = self.get_blended_input('train_signal', 'sum') or 0.0
        
        if w_matrix is None:
            return

        # 2. Prepare W-Matrix Input
        # Ensure it's the correct dimensions (w_dim, w_dim)
        if w_matrix.shape[0] != self.w_dim or w_matrix.shape[1] != self.w_dim:
            w_matrix = cv2.resize(w_matrix, (self.w_dim, self.w_dim), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Flatten and send to tensor
        w_flat = w_matrix.flatten().astype(np.float32)
        w_tensor = torch.from_numpy(w_flat).unsqueeze(0).to(DEVICE)

        # 3. --- NEW MODE-SWITCHING LOGIC ---
        
        # Check if we are in "Learning Mode"
        if train_signal > 0.5 and target_image is not None:
            self.status = "LEARNING"
            self.decoder_model.train() # Set model to training mode
            
            # Prepare target image
            if target_image.ndim == 3:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            if target_image.shape[0] != self.image_size:
                target_image = cv2.resize(target_image, (self.image_size, self.image_size))
            if target_image.max() > 1.0:
                target_image = target_image / 255.0
            
            target_tensor = torch.from_numpy(target_image).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            
            # --- Learning Step ---
            # A. Get the "Student's Answer"
            recon_tensor = self.decoder_model(w_tensor)
            
            # B. Compare to "Answer Sheet"
            loss = self.loss_fn(recon_tensor, target_tensor)
            
            # C. Update the "Key"
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.current_loss = loss.item()
            self.training_steps += 1
            
            # Store the reconstruction
            self.reconstructed_image = recon_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        else:
            # --- "Inference Mode" ---
            # (No training, no answer key)
            self.status = "INFERRING"
            self.decoder_model.eval() # Set model to evaluation mode
            
            with torch.no_grad():
                # Just apply the "Key" to the "Lock"
                recon_tensor = self.decoder_model(w_tensor)
                
            self.reconstructed_image = recon_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            # Loss is not calculated, it holds its last value
    
    def get_output(self, port_name):
        if port_name == 'reconstructed':
            return self.reconstructed_image
        elif port_name == 'loss':
            return self.current_loss
        return None
    
    def get_display_image(self):
        # Display the reconstruction
        img = self.reconstructed_image
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        
        # Add info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- NEW: Show current mode ---
        status_color = (0, 255, 0) if self.status == "LEARNING" else (0, 255, 255)
        cv2.putText(img_color, self.status, (5, 15), font, 0.4, status_color, 1)
        
        cv2.putText(img_color, f"Loss: {self.current_loss:.4f}", (5, 30), 
                   font, 0.4, (255, 255, 255), 1)
        cv2.putText(img_color, f"Steps: {self.training_steps}", (5, 45),
                   font, 0.4, (255, 255, 255), 1)
        
        # Resize for display
        display_size = 256
        img_resized = cv2.resize(img_color, (display_size, display_size), 
                                 interpolation=cv2.INTER_NEAREST)
        
        img_resized = np.ascontiguousarray(img_resized)
        return QtGui.QImage(img_resized.data, display_size, display_size, 
                            display_size * 3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("W_Matrix_Dim", "w_dim", self.w_dim, None),
            ("Image_Size", "image_size", self.image_size, None),
            ("Learning_Rate", "base_learning_rate", self.lr, None)
        ]
    
    def close(self):
        if hasattr(self, 'decoder_model'):
            del self.decoder_model
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().close()