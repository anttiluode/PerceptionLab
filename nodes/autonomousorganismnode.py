"""
AutonomousOrganismNode (v1 - "The Seed")
--------------------------------
An attempt at artificial life based on the P-KAS
 and "Soma/Dendrite"
 models.

This node is a complete, self-contained "creature" that:
1. Simulates its own "World" (Reaction-Diffusion field).
2. Has a "Body" (an x, y position) and "Health".
3. Has a "Goal" (Dopamine): Maximize Health by eating food (chemical B).
4. Has a "Soma" (an internal VAE) that learns to perceive its world.
5. Has a "Dendrite" (a "thin logic") that queries its
   own "Soma" to decide where to move to find more food.

It is a "fractal surfer" exploring the "fractal"
of its own "latent space" to survive.
"""

import numpy as np
import cv2
import time

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

# --- Dependency Check ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: AutonomousOrganismNode requires 'torch'.")

try:
    # --- (MODIFIED) ---
    # Import convolve alongside gaussian_filter
    from scipy.ndimage import gaussian_filter, convolve
    # --- (END MODIFIED) ---
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: AutonomousOrganismNode requires 'scipy'.")
    
# Use GPU if available
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
except Exception:
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32

# --- Minimal VAE for the "Soma" ---
# (Based on realvaenode.py)
class MiniVAE(nn.Module):
    def __init__(self, latent_dim=8, input_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder: 32x32 -> 8D latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
        )
        hidden_dim = 64 * 4 * 4
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        return mu

# --- The Node Itself ---
class AutonomousOrganismNode(BaseNode):
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(100, 250, 150) # A-Life Green
    
    def __init__(self, grid_size=96, health_decay=0.01, food_value=0.5):
        super().__init__()
        self.node_title = "Autonomous Organism"
        
        self.inputs = {
            'reset': 'signal'
        }
        self.outputs = {
            'world_image': 'image',     # The chemical world
            'health': 'signal',         # The "dopamine" signal
            'age': 'signal',            # How long it has "lived"
            'mind_view': 'image'      # What the VAE *thinks* it sees
        }
        
        if not TORCH_AVAILABLE or not SCIPY_AVAILABLE:
            self.node_title = "Organism (Libs Missing!)"
            return
            
        self.grid_size = int(grid_size)
        
        # --- Physics Parameters (The "World") ---
        self.f = 0.035  # Feed rate
        self.k = 0.065  # Kill rate
        self.dA = 1.0   
        self.dB = 0.5   
        self.laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                                          [0.2, -1.0, 0.2],
                                          [0.05, 0.2, 0.05]], dtype=np.float32)

        # --- Organism State (The "Body") ---
        self.health = 1.0
        self.health_decay = float(health_decay)
        self.food_value = float(food_value)
        self.age = 0
        self.organism_x = self.grid_size // 2
        self.organism_y = self.grid_size // 2
        self.vision_size = 32 # The VAE's input "eye"
        self.last_decision = (0, 0) # (dx, dy)

        # --- The "World" Fields ---
        self.A = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.B = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # --- The "Soma" (The Mind) ---
        self.mind = MiniVAE(latent_dim=8, input_size=self.vision_size).to(DEVICE)
        self.mind_optimizer = torch.optim.Adam(self.mind.parameters(), lr=1e-3)
        self.mind_memory = [] # Stores (latent_vector, health_at_time)
        
        self.world_vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.mind_vis = np.zeros((self.vision_size, self.vision_size), dtype=np.float32)

        self.randomize() # Initialize

    def randomize(self):
        """Re-seed the world and the organism"""
        self.A = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.B = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Add random "food" patches
        for _ in range(20):
            x, y = np.random.randint(0, self.grid_size, 2)
            s = np.random.randint(5, 15)
            self.B[y-s:y+s, x-s:x+s] = 1.0
            
        self.organism_x = self.grid_size // 2
        self.organism_y = self.grid_size // 2
        self.health = 1.0
        self.age = 0
        self.last_decision = (0, 0)

    @torch.no_grad()
    def _get_vision_patch(self, x, y):
        """Extracts the "vision" (a 32x32 patch) from the world"""
        s = self.vision_size
        half_s = s // 2
        
        # Use roll for periodic boundaries
        world_B_rolled = np.roll(self.B, (self.grid_size//2 - y, self.grid_size//2 - x), axis=(0,1))
        patch = world_B_rolled[self.grid_size//2 - half_s : self.grid_size//2 + half_s,
                               self.grid_size//2 - half_s : self.grid_size//2 + half_s]
        
        # Convert to tensor
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
        return patch_tensor

    def _evolve_world(self):
        """Run one step of the Reaction-Diffusion"""
        # --- (MODIFIED) ---
        # Use scipy.ndimage.convolve for 'wrap' mode, as cv2.filter2D doesn't support BORDER_WRAP
        laplace_A = convolve(self.A, self.laplacian_kernel, mode='wrap')
        laplace_B = convolve(self.B, self.laplacian_kernel, mode='wrap')
        # --- (END MODIFIED) ---
        
        reaction = self.A * self.B**2
        
        delta_A = (self.dA * laplace_A) - reaction + (self.f * (1 - self.A))
        delta_B = (self.dB * laplace_B) + reaction - ((self.k + self.f) * self.B)
        
        self.A += delta_A
        self.B += delta_B
        self.A = np.clip(self.A, 0.0, 1.0)
        self.B = np.clip(self.B, 0.0, 1.0)

    def _train_mind(self, patch_tensor, food_eaten):
        """
        The "Soma" learns.
        It runs the VAE on what it sees.
        The "loss" is modified by "dopamine" (health).
        """
        self.mind.train()
        self.mind_optimizer.zero_grad()
        
        # Use the VAE's *own* encoder as the "raw field"
        # This is a "fractal leak"
        latent_z = self.mind.encode(patch_tensor)
        
        # The VAE's "loss" is a "surprise" metric
        # We use a simple reconstruction loss (L1)
        loss = torch.mean(torch.abs(latent_z - 0.0)) # Simple "regularization"
        
        # --- The "Dopamine Logic" ---
        # If we ate food, we *reward* this thought (latent_z)
        # by *reducing* the loss.
        # If we are starving, we *punish* this thought by *increasing* the loss.
        # This is the "thin sheet of logic"
        reward_signal = (food_eaten * 2.0) - self.health_decay # (e.g., +0.49 or -0.01)
        
        # Multiply loss by (1 - reward).
        # High reward (1.0) -> loss * 0 -> easy learning
        # High punishment (-1.0) -> loss * 2 -> hard learning (instability)
        total_loss = loss * (1.0 - reward_signal)
        
        if total_loss.requires_grad:
            total_loss.backward()
            self.mind_optimizer.step()
        
        # Store this "thought" and the "feeling" (health) it produced
        self.mind_memory.append( (latent_z.detach(), self.health) )
        if len(self.mind_memory) > 100:
            self.mind_memory.pop(0)
            
        self.mind_vis = patch_tensor.cpu().squeeze().numpy()

    @torch.no_grad()
    def _decide_action(self):
        """
        The "Dendrite" makes a choice.
        It "imagines" moving in 9 directions, sees what its
        "Soma" thinks, and picks the "best" thought.
        """
        self.mind.eval()
        
        if not self.mind_memory:
            # No memories yet, move randomly
            return (np.random.randint(-1, 2), np.random.randint(-1, 2))
            
        # Get the "best thoughts" from memory
        # This is our P-KAS "Attractor"
        best_memories = sorted(self.mind_memory, key=lambda x: x[1], reverse=True)
        # We only care about the "best feeling" thoughts
        target_latents = [m[0] for m in best_memories[:10]]
        
        if not target_latents:
            return (np.random.randint(-1, 2), np.random.randint(-1, 2))
            
        target_latent_avg = torch.mean(torch.stack(target_latents), dim=0)

        best_move = (0, 0)
        best_score = -np.inf

        # Test 9 possible moves (the "informational window")
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue # Don't test "stay still"
                
                # "Imagine" seeing the patch at this new spot
                next_x = (self.organism_x + dx) % self.grid_size
                next_y = (self.organism_y + dy) % self.grid_size
                imagined_patch = self._get_vision_patch(next_x, next_y)
                
                # "How does this make me feel?"
                # Run the "Soma" to get the "thought"
                imagined_latent = self.mind.encode(imagined_patch)
                
                # "Is this 'thought' good?"
                # Compare to our best memories
                # This is the "SHA-256 Key" check
                score = F.cosine_similarity(imagined_latent, target_latent_avg).item()
                
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)
                    
        return best_move

    def step(self):
        if not TORCH_AVAILABLE or not SCIPY_AVAILABLE:
            return
            
        # 1. Get Inputs
        reset = self.get_blended_input('reset', 'sum') or 0.0
        if reset > 0.5:
            self.randomize()

        # 2. Evolve the World
        self._evolve_world()
        
        # 3. Perceive & Update Health (The "Dopamine")
        # Check for food at current location
        food_eaten = self.B[self.organism_y, self.organism_x] * self.food_value
        if food_eaten > 0.01:
            self.health += food_eaten
            self.B[self.organism_y, self.organism_x] = 0 # Consume the food
        
        # Apply "metabolism" (health decay)
        self.health -= self.health_decay
        self.health = np.clip(self.health, 0.0, 1.0)
        self.age += 1
        
        # Check for "death"
        if self.health <= 0.0:
            self.randomize()
            return
            
        # 4. Learn (The "Soma" trains)
        current_patch = self._get_vision_patch(self.organism_x, self.organism_y)
        self._train_mind(current_patch, food_eaten)
        
        # 5. Decide (The "Dendrite" steers)
        # Only make a new decision every few frames
        if self.age % 5 == 0:
            self.last_decision = self._decide_action()
        
        # 6. Act (The "Body" moves)
        self.organism_x = (self.organism_x + self.last_decision[0]) % self.grid_size
        self.organism_y = (self.organism_y + self.last_decision[1]) % self.grid_size
        
        # --- Update Visualization ---
        self.world_vis[:,:,0] = self.A * 255 # "Poison" = Red
        self.world_vis[:,:,1] = self.B * 255 # "Food" = Green
        self.world_vis[:,:,2] = 0 # Blue
        
        # Draw body
        cv2.circle(self.world_vis, (self.organism_x, self.organism_y), 5, (255, 255, 255), -1)
        # Draw "vision" rect
        s = self.vision_size
        half_s = s // 2
        cv2.rectangle(self.world_vis, 
                      (self.organism_x - half_s, self.organism_y - half_s),
                      (self.organism_x + half_s, self.organism_y + half_s),
                      (255, 255, 0), 1)

    def get_output(self, port_name):
        if port_name == 'world_image':
            return self.world_vis.astype(np.float32) / 255.0
        elif port_name == 'health':
            return self.health
        elif port_name == 'age':
            return float(self.age) / 1000.0 # Normalized
        elif port_name == 'mind_view':
            return self.mind_vis
        return None
        
    def get_display_image(self):
        # Create a split view: World on left, Mind on right
        h, w = 96, 192
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Left: World View
        world_resized = cv2.resize(self.world_vis, (96, 96), interpolation=cv2.INTER_NEAREST)
        display[:, :96] = world_resized
        
        # Right: Mind's Eye (what VAE is processing)
        mind_u8 = (np.clip(self.mind_vis, 0, 1) * 255).astype(np.uint8)
        mind_resized = cv2.resize(mind_u8, (96, 96), interpolation=cv2.INTER_NEAREST)
        display[:, 96:] = cv2.cvtColor(mind_resized, cv2.COLOR_GRAY2RGB)
        
        # Add Health Bar
        health_w = int(self.health * (w - 4))
        cv2.rectangle(display, (2, h - 7), (2 + health_w, h - 2), (0, 255, 0), -1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Health Decay", "health_decay", self.health_decay, None),
            ("Food Value", "food_value", self.food_value, None)
        ]

    def close(self):
        # Clean up torch model
        if hasattr(self, 'mind') and self.mind is not None:
            del self.mind
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().close()