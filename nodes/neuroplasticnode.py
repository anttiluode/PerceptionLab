import numpy as np
import cv2
from PyQt6 import QtGui, QtCore
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
            self._output_values = {}
        def get_input(self, name): return None
        def get_output(self, name): return self._output_values.get(name, None)
        def set_output(self, name, val): pass

# --- The Brain (PyTorch) ---
class MorphogenesisNet(nn.Module):
    def __init__(self, input_size, output_res):
        super().__init__()
        self.input_size = input_size
        self.output_res = output_res
        self.output_dim = output_res * output_res
        self.hidden_size = 128
        
        self.W1 = nn.Parameter(torch.randn(input_size, self.hidden_size) * 0.05)
        self.W2 = nn.Parameter(torch.randn(self.hidden_size, self.output_dim) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.b2 = nn.Parameter(torch.zeros(self.output_dim))
        
        self.register_buffer('mask1', torch.ones(input_size, self.hidden_size))
        self.register_buffer('mask2', torch.ones(self.hidden_size, self.output_dim))

    def forward(self, x):
        w1_masked = self.W1 * self.mask1
        w2_masked = self.W2 * self.mask2
        x = F.relu(x @ w1_masked + self.b1)
        x = x @ w2_masked + self.b2
        return torch.sigmoid(x)

    def morphogenesis_step(self):
        with torch.no_grad():
            grow_threshold = 0.001
            prune_threshold = 0.001
            
            if self.W1.grad is not None:
                self.mask1[self.W1.grad.abs() > grow_threshold] = 1.0
            if self.W2.grad is not None:
                self.mask2[self.W2.grad.abs() > grow_threshold] = 1.0

            self.mask1[self.W1.abs() < prune_threshold] = 0.0
            self.mask2[self.W2.abs() < prune_threshold] = 0.0

# --- The Node ---
class NeuroplasticNode(BaseNode):
    NODE_CATEGORY = "AI / Learning"
    NODE_COLOR = QtGui.QColor(255, 165, 0) # Orange
    
    def __init__(self):
        super().__init__()
        self.node_title = "Neuroplastic Decoder"
        
        self.inputs = {
            'latent_in': 'spectrum',   
            'optic_nerve': 'image',    
            'learning_rate': 'signal'  
        }
        
        self.outputs = {
            'decoded_vision': 'image', 
            'wiring_viz': 'image'
        }
        
        # Ensure output storage exists
        if not hasattr(self, '_output_values'):
            self._output_values = {}
            
        self.res = 32
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_count = 0
        self.last_status_time = 0

    # --- SAFETY OVERRIDES (The Fix for Broken Wires) ---
    def get_input(self, name):
        if hasattr(super(), 'get_input'):
            return super().get_input(name)
        return self.inputs.get(name, None)

    # THIS WAS MISSING -> This allows the wire to actually pull the data
    def get_output(self, name):
        return self._output_values.get(name, None)

    def set_output(self, name, val):
        self._output_values[name] = val

    def print_status(self, msg):
        if time.time() - self.last_status_time > 2.0:
            print(f"[NeuroplasticNode] {msg}")
            self.last_status_time = time.time()

    def step(self):
        # 1. ALWAYS OUTPUT SOMETHING (Static Noise if dead)
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # If output is not set yet, set it to noise so we see "Alive but Waiting"
        if 'decoded_vision' not in self._output_values:
            noise = np.random.randint(0, 50, (128, 128, 3), dtype=np.uint8)
            self.set_output('decoded_vision', noise)
            self.set_output('wiring_viz', blank)

        # 2. FETCH INPUTS
        latent_raw = None
        target_img = None
        lr_raw = None

        if hasattr(self, 'input_data'):
            latent_raw = self.input_data.get('latent_in')
            target_img = self.input_data.get('optic_nerve')
            lr_raw = self.input_data.get('learning_rate')
        
        # 3. VALIDATE INPUT
        if latent_raw is None:
            self.print_status("Waiting for Crystal Chip...")
            return 
        
        if isinstance(latent_raw, str):
            return # Ignore unconnected ports

        try:
            latent_arr = np.array(latent_raw, dtype=np.float32).flatten()
        except:
            return
            
        if latent_arr.size == 0: 
            return
        
        # Normalize
        max_val = np.max(np.abs(latent_arr))
        if max_val > 0: latent_arr /= max_val

        # 4. INITIALIZE BRAIN
        if self.model is None:
            input_dim = latent_arr.shape[0]
            print(f"[NeuroplasticNode] INITIALIZED. Input Dim: {input_dim}")
            self.model = MorphogenesisNet(input_dim, self.res).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

        # 5. FORWARD PASS
        in_tensor = torch.from_numpy(latent_arr).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(in_tensor)
            
        pred_img = prediction.cpu().numpy().reshape(self.res, self.res)
        
        # Visual Processing
        pred_viz = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
        pred_viz = cv2.cvtColor(pred_viz, cv2.COLOR_GRAY2RGB)
        
        try:
            pred_viz = cv2.resize(pred_viz, (128, 128), interpolation=cv2.INTER_NEAREST)
        except: pass
            
        # WRITE OUTPUT
        self.set_output('decoded_vision', pred_viz)

        # 6. LEARNING PASS
        learning_rate = 0.01
        if isinstance(lr_raw, (int, float)):
            learning_rate = float(lr_raw) * 0.05
            
        valid_target = (target_img is not None and 
                        isinstance(target_img, np.ndarray) and 
                        target_img.size > 0)

        if valid_target and learning_rate > 0.0001:
            try:
                self.model.train()
                
                target_small = cv2.resize(target_img, (self.res, self.res))
                if target_small.ndim == 3: target_small = np.mean(target_small, axis=2)
                target_small = target_small.astype(np.float32) / 255.0
                
                target_tensor = torch.from_numpy(target_small.flatten()).unsqueeze(0).to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(in_tensor)
                loss = self.loss_fn(out, target_tensor)
                loss.backward()
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                    
                self.optimizer.step()

                if self.frame_count % 10 == 0:
                    self.model.morphogenesis_step()
            except:
                pass

        self.frame_count += 1

        # 7. WIRING VIZ
        if self.frame_count % 5 == 0:
            w1 = self.model.W1.detach().cpu().numpy()
            mask = self.model.mask1.detach().cpu().numpy()
            viz = np.abs(w1 * mask)
            if viz.max() > 0: viz /= viz.max()
            viz_img = (viz * 255).astype(np.uint8)
            viz_img = cv2.applyColorMap(viz_img, cv2.COLORMAP_JET)
            try:
                viz_img = cv2.resize(viz_img, (128, 128), interpolation=cv2.INTER_NEAREST)
            except: pass
            self.set_output('wiring_viz', viz_img)