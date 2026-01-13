import numpy as np
import cv2
from PyQt6 import QtGui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# --- PORTING THE BRAIN FROM RETRO2.PY ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64  # Downscaled for real-time node performance
LATENT_DIM = 128

class MorphoLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('mask', torch.zeros(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        self.weight.data *= 0.1

    def forward(self, x, override=False):
        m = torch.ones_like(self.mask) if override else self.mask
        return F.linear(x, self.weight * m, self.bias)

    def morphogenesis_step(self, active_grad, current_loss, avg_loss):
        safe_avg = max(avg_loss, 0.0001)
        panic_level = current_loss / safe_avg
        
        grow_rate = 0.001
        prune_rate = 0.05
        
        if panic_level > 1.1: # PANIC
            grow_rate = 0.05 * min(panic_level, 2.0)
            prune_rate = 0.0
        elif panic_level < 0.8: # ZEN
            grow_rate = 0.0
            prune_rate = 0.005

        with torch.no_grad():
            # Demand (Hypothetical)
            demand = torch.abs(active_grad) * (1 - self.mask)
            if demand.max() == 0: demand = torch.rand_like(demand) * (1 - self.mask)
            
            # Grow
            n_dead = (self.mask == 0).sum().item()
            n_grow = int(n_dead * grow_rate)
            if n_grow > 0:
                _, idx = torch.topk(demand.flatten(), n_grow)
                self.mask.view(-1)[idx] = 1.0
                self.weight.view(-1)[idx] += torch.randn(n_grow).to(DEVICE) * 0.05

            # Prune
            n_alive = self.mask.sum().item()
            if n_alive > 100 and prune_rate > 0:
                strength = torch.abs(self.weight * self.mask)
                strength[self.mask == 0] = float('inf')
                n_prune = int(n_alive * prune_rate)
                if n_prune > 0:
                    thresh = torch.topk(strength.flatten(), n_prune, largest=False).values.max()
                    self.mask[strength <= thresh] = 0

class LivingWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Tiny CNN for speed
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), LATENT_DIM),
            nn.Tanh()
        )
        self.brain = MorphoLinear(LATENT_DIM, LATENT_DIM)
        self.dec_fc = nn.Linear(LATENT_DIM, 32 * (IMG_SIZE // 4) * (IMG_SIZE // 4))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x, override_brain=False):
        z = self.enc(x)
        z_next_pred = self.brain(z, override=override_brain)
        x_next_pred = self.dec(
            self.dec_fc(z_next_pred).view(-1, 32, IMG_SIZE // 4, IMG_SIZE // 4)
        )
        return x_next_pred, z, z_next_pred

# --- THE NODE ---
class PredictiveDreamNode(BaseNode):
    """
    RETRO-MORPHO Node
    -----------------
    The Breathing Brain inside your graph.
    Inputs: 
        - Visual Reality (Image)
    Outputs:
        - Dream View (Image): What the AI expects to see.
        - Surprise Map (Image): Where prediction failed.
        - Panic Signal (Signal): 0.0 (Zen) to 1.0+ (Panic). Use this to modulate other nodes!
    """
    NODE_CATEGORY = "AI / Life"
    NODE_COLOR = QtGui.QColor(100, 0, 255) # Deep Purple

    def __init__(self):
        super().__init__()
        self.node_title = "Predictive Dream (Retro-V2)"
        self._output_values = {}
        
        self.inputs = {
            'visual_reality': 'image'
        }
        
        self.outputs = {
            'dream_view': 'image',
            'surprise_map': 'image',
            'panic_signal': 'signal'
        }
        
        self.model = None
        self.opt = None
        self.history_loss = []
        self.frame_curr = None
        self.initialized = False

    def get_input(self, n): 
        if hasattr(self, 'get_blended_input'): return self.get_blended_input(n)
        return self.input_data.get(n, [None])[0]
    def set_output(self, n, v): self._output_values[n] = v
    def get_output(self, n): return self._output_values.get(n)

    def step(self):
        # 1. Init
        if not self.initialized:
            print("[DreamNode] Birthing the organism...")
            self.model = LivingWorldModel().to(DEVICE)
            self.opt = optim.Adam(self.model.parameters(), lr=0.002)
            self.initialized = True
            
        # 2. Input
        img_in = self.get_input('visual_reality')
        if img_in is None: return
        
        # Resize/Norm
        small = cv2.resize(img_in, (IMG_SIZE, IMG_SIZE))
        if small.ndim == 2: small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        
        # Check normalization (0-1 vs 0-255)
        if small.max() > 1.05:
            x_np = small.astype(np.float32) / 127.5 - 1.0
        else:
            x_np = small.astype(np.float32) * 2.0 - 1.0
            
        x = torch.from_numpy(x_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        
        # 3. The Lifecycle (Only if we have a previous frame)
        panic_out = 0.5
        
        if self.frame_curr is not None:
            # A. HYPOTHETICAL PROBE
            self.model.zero_grad()
            _, _, z_next_hypo = self.model(self.frame_curr, override_brain=True)
            with torch.no_grad(): z_target = self.model.enc(x)
            
            probe_loss = F.mse_loss(z_next_hypo, z_target)
            probe_loss.backward()
            
            # Morphogenesis Trigger
            current_loss_val = self.history_loss[-1] if self.history_loss else 0.01
            avg_loss = np.mean(self.history_loss) if self.history_loss else 0.01
            self.model.brain.morphogenesis_step(self.model.brain.weight.grad, current_loss_val, avg_loss)
            
            # B. TRAINING
            self.model.zero_grad()
            x_pred, _, z_next = self.model(self.frame_curr, override_brain=False)
            
            l_vis = F.mse_loss(x_pred, x)
            l_lat = F.mse_loss(z_next, z_target.detach())
            loss = l_vis + l_lat
            loss.backward()
            
            with torch.no_grad():
                self.model.brain.weight.grad *= self.model.brain.mask
            self.opt.step()
            
            # Stats
            loss_val = loss.item()
            self.history_loss.append(loss_val)
            if len(self.history_loss) > 50: self.history_loss.pop(0)
            
            # Calculate Panic Signal for Output
            # Normalize panic around 1.0 (1.0 = normal, >1.0 = panic)
            raw_panic = loss_val / max(avg_loss, 0.0001)
            panic_out = min(max(raw_panic * 0.5, 0.0), 1.0) # Scale roughly 0-1
            
            # C. VISUALIZATION OUTPUTS
            # Prediction
            pred_np = x_pred[0].permute(1,2,0).detach().cpu().numpy()
            pred_np = (pred_np + 1.0) / 2.0 # -1..1 -> 0..1
            pred_img = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
            pred_img = cv2.resize(pred_img, (128, 128)) # Upscale slightly for UI
            self.set_output('dream_view', pred_img)
            
            # Surprise Map
            diff = torch.abs(x_pred - x)
            diff_np = diff[0].permute(1,2,0).detach().cpu().numpy()
            diff_img = (np.clip(diff_np * 3.0, 0, 1) * 255).astype(np.uint8) # Boost contrast
            diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_INFERNO)
            diff_img = cv2.resize(diff_img, (128, 128))
            self.set_output('surprise_map', diff_img)
            
        self.frame_curr = x
        self.set_output('panic_signal', panic_out)