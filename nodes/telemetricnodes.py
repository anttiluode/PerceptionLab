"""
Telemetric VAE Node - Reality Physics Probe
============================================
Tests if VAE latent space exhibits quantum behavior by mapping it to a Bloch sphere.

THEORY:
- Stable reconstruction (low KL) = Classical state (qubit still)
- Hallucinating (high KL) = Quantum tunneling (qubit spinning)
- Velocity = Time dilation (rotation speed)

CONNECTIONS:
VAE → BlochQubit:
  kl_loss → ry_angle (vertical spin when uncertain)
  velocity → rx_angle (horizontal spin when changing)
  latent_mean[0] → rz_angle (phase rotation)
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from PyQt6 import QtGui
from collections import deque

import __main__
BaseNode = __main__.BaseNode

class TelemetricVAENode(BaseNode):
    NODE_CATEGORY = "AI / ML"
    NODE_TITLE = "Reality Engine (Telemetric)"
    NODE_COLOR = QtGui.QColor(120, 50, 180)
    
    def __init__(self):
        super().__init__()
        
        # Define I/O FIRST (before slow operations)
        self.inputs = {
            'image_in': 'image',
            'observer_collapse': 'signal'  # Multiplier for variance (1.0 = normal, 0.0 = classical)
        }
        
        self.outputs = {
            'image_out': 'image',
            'latent_mean': 'spectrum',
            'kl_loss': 'signal',
            'velocity': 'signal',
            # Bloch sphere control signals
            'rotation_x': 'signal',  # Velocity-driven horizontal spin
            'rotation_y': 'signal',  # KL-driven vertical spin (main quantum indicator)
            'rotation_z': 'signal'   # Latent phase rotation
        }
        
        # Initialize storage
        self.current_image = None
        self.current_latent = None
        self.current_kl = 0.0
        self.current_velocity = 0.0
        self.current_rot_x = 0.0
        self.current_rot_y = 0.0
        self.current_rot_z = 0.0
        self.prev_latent = None
        
        # Init VAE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = 32
        print(f"TelemetricVAE: Initializing on {self.device}")
        
        try:
            self._build_model()
            print("TelemetricVAE: Model loaded successfully")
        except Exception as e:
            print(f"TelemetricVAE: Error building model: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_model(self):
        """Build a simple convolutional VAE"""
        class ConvVAE(nn.Module):
            def __init__(self, latent_dim):
                super().__init__()
                # Encoder
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 32, 4, 2, 1),   # 64x64 -> 32x32
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, 2, 1),  # 32x32 -> 16x16
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 4, 2, 1), # 16x16 -> 8x8
                    nn.ReLU(),
                    nn.Flatten()
                )
                
                self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
                self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
                
                # Decoder
                self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
                self.dec = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8 -> 16x16
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16 -> 32x32
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32x32 -> 64x64
                    nn.Sigmoid()
                )
            
            def reparameterize(self, mu, logvar, multiplier=1.0):
                """The quantum sampling step"""
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + (eps * std * multiplier)
            
            def decode(self, z):
                x = self.decoder_input(z)
                x = x.view(-1, 128, 8, 8)
                return self.dec(x)
            
            def forward(self, x, obs_mult=1.0):
                # Encode
                features = self.enc(x)
                mu = self.fc_mu(features)
                logvar = self.fc_logvar(features)
                
                # Sample (the measurement!)
                z = self.reparameterize(mu, logvar, obs_mult)
                
                # Decode
                recon = self.decode(z)
                
                return recon, mu, logvar
        
        self.model = ConvVAE(self.latent_dim).to(self.device)
    
    def step(self):
        """Main processing loop"""
        # Get inputs
        img = self.get_blended_input('image_in')
        obs_mult = self.get_blended_input('observer_collapse')
        
        # Handle observer collapse multiplier
        if obs_mult is None:
            obs_mult = 1.0
        elif isinstance(obs_mult, (list, np.ndarray)):
            obs_mult = float(np.mean(obs_mult))
        else:
            obs_mult = float(obs_mult)
        
        if img is None:
            return
        
        try:
            # Preprocess image
            img_small = cv2.resize(img, (64, 64))
            
            # Force RGB (handle grayscale/RGBA)
            if len(img_small.shape) == 2:
                img_small = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)
            elif len(img_small.shape) == 3 and img_small.shape[2] == 4:
                img_small = cv2.cvtColor(img_small, cv2.COLOR_RGBA2RGB)
            
            # To tensor
            tensor_img = torch.from_numpy(img_small).float().permute(2, 0, 1).unsqueeze(0)
            if tensor_img.max() > 1.0:
                tensor_img /= 255.0
            tensor_img = tensor_img.to(self.device)
            
            # Forward pass (no gradients - just inference)
            with torch.no_grad():
                recon, mu, logvar = self.model(tensor_img, obs_mult=obs_mult)
                
                # Calculate KL divergence (quantum uncertainty)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_val = float(kl_loss.item()) * 0.01  # Scale for visualization
                
                # Extract latent
                mu_np = mu.cpu().numpy().flatten()
                
                # Calculate velocity (thought speed)
                velocity = 0.0
                if self.prev_latent is not None:
                    velocity = np.linalg.norm(mu_np - self.prev_latent)
                self.prev_latent = mu_np
                
                # Map to Bloch sphere rotations
                # rotation_x: Velocity drives horizontal spin (time dilation)
                self.current_rot_x = np.tanh(velocity * 5.0) * 0.5  # Scale to [-0.5, 0.5]
                
                # rotation_y: KL loss drives vertical spin (quantum tunneling indicator)
                # High KL = quantum = fast spinning
                self.current_rot_y = np.tanh(kl_val * 0.1) * 0.8  # Main quantum indicator
                
                # rotation_z: First latent dimension drives phase rotation
                self.current_rot_z = np.tanh(mu_np[0]) * 0.3  # Subtle phase shift
                
                # Store outputs
                self.current_kl = kl_val
                self.current_velocity = float(velocity)
                self.current_latent = mu_np
                
                # Reconstruct image
                out_img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_img = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)
                
                # Make contiguous for OpenCV (FIX for putText error)
                out_img = np.ascontiguousarray(out_img)
                
                # Add status text
                cv2.putText(out_img, f"KL:{kl_val:.1f}", (2, 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv2.putText(out_img, f"V:{velocity:.2f}", (2, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                
                # Show quantum state
                if kl_val > 5.0:
                    cv2.putText(out_img, "QUANTUM", (2, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(out_img, "CLASSICAL", (2, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                self.current_image = out_img
                
        except Exception as e:
            print(f"TelemetricVAE step error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_output(self, port_name):
        """Return specific outputs"""
        if port_name == 'image_out':
            return self.current_image
        elif port_name == 'latent_mean':
            return self.current_latent
        elif port_name == 'kl_loss':
            return self.current_kl
        elif port_name == 'velocity':
            return self.current_velocity
        elif port_name == 'rotation_x':
            return self.current_rot_x
        elif port_name == 'rotation_y':
            return self.current_rot_y
        elif port_name == 'rotation_z':
            return self.current_rot_z
        return None
    
    def get_display_image(self):
        """Show reconstruction"""
        if self.current_image is not None:
            h, w, c = self.current_image.shape
            return QtGui.QImage(self.current_image.data, w, h, 3*w,
                              QtGui.QImage.Format.Format_RGB888)
        return None


# Keep your other visualization nodes as-is
class PhaseSpaceNode(BaseNode):
    NODE_CATEGORY = "Visualization"
    NODE_TITLE = "Thought Map"
    NODE_COLOR = QtGui.QColor(50, 150, 200)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'latent_vector': 'spectrum',
            'energy': 'signal'
        }
        self.outputs = {
            'visualization': 'image'
        }
        
        self.history = deque(maxlen=50)
        self.w, self.h = 256, 256
        self.color_map = np.zeros((self.h, self.w, 3), dtype=np.uint8)
    
    def step(self):
        vec = self.get_blended_input('latent_vector')
        energy = self.get_blended_input('energy')
        
        if energy is None:
            energy = 0.0
        elif isinstance(energy, (list, np.ndarray)):
            energy = float(np.mean(energy))
        
        x, y = 0.5, 0.5
        if vec is not None and isinstance(vec, np.ndarray) and vec.size >= 2:
            x = (np.clip(vec.flat[0], -2, 2) + 2) / 4
            y = (np.clip(vec.flat[1], -2, 2) + 2) / 4
        
        self.history.append((x, y, energy))
        
        self.color_map.fill(20)
        for i, (px, py, e) in enumerate(self.history):
            r = min(255, int(abs(e) * 500) + 50)
            g = int(i / 50 * 255)
            cv2.circle(self.color_map, 
                      (int(px * self.w), int(py * self.h)),
                      2, (r, g, 255), -1)
    
    def get_output(self, port_name):
        if port_name == 'visualization':
            return self.color_map
        return None
    
    def get_display_image(self):
        if self.color_map is not None:
            h, w, c = self.color_map.shape
            return QtGui.QImage(self.color_map.data, w, h, 3*w,
                              QtGui.QImage.Format.Format_RGB888)
        return None


class QuantumMonitorNode(BaseNode):
    NODE_CATEGORY = "Visualization"
    NODE_TITLE = "Energy Monitor"
    NODE_COLOR = QtGui.QColor(180, 50, 50)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'tunneling_energy': 'signal',
            'time_dilation': 'signal'
        }
        self.outputs = {}
        
        self.w, self.h = 300, 150
        self.monitor_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.energy_hist = deque(maxlen=self.w)
        self.time_hist = deque(maxlen=self.w)
    
    def step(self):
        e = self.get_blended_input('tunneling_energy')
        t = self.get_blended_input('time_dilation')
        
        if e is None:
            e = 0.0
        elif isinstance(e, (list, np.ndarray)):
            e = float(np.mean(e))
        
        if t is None:
            t = 0.0
        elif isinstance(t, (list, np.ndarray)):
            t = float(np.mean(t))
        
        self.energy_hist.append(e)
        self.time_hist.append(t)
        
        self.monitor_img.fill(0)
        
        # Draw energy (yellow)
        pts_e = [[i, np.clip(int(self.h/2 - v * 500), 0, self.h-1)] 
                for i, v in enumerate(self.energy_hist)]
        if len(pts_e) > 1:
            cv2.polylines(self.monitor_img, [np.array(pts_e)], False, (0, 255, 255), 2)
        
        # Draw velocity (magenta)
        pts_t = [[i, np.clip(int(self.h - v * 50), 0, self.h-1)]
                for i, v in enumerate(self.time_hist)]
        if len(pts_t) > 1:
            cv2.polylines(self.monitor_img, [np.array(pts_t)], False, (255, 0, 255), 2)
        
        # Labels
        cv2.putText(self.monitor_img, "Energy (KL)", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(self.monitor_img, "Velocity", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # State indicator
        if e > 5.0:
            cv2.putText(self.monitor_img, "QUANTUM", (self.w - 80, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(self.monitor_img, "CLASSICAL", (self.w - 80, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def get_display_image(self):
        if self.monitor_img is not None:
            h, w, c = self.monitor_img.shape
            return QtGui.QImage(self.monitor_img.data, w, h, 3*w,
                              QtGui.QImage.Format.Format_RGB888)
        return None