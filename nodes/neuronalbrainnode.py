"""
Neuronal Brain Node (Reservoir Computing) - SELF HEALING
--------------------------------------------------------
A node with Short-Term Memory.
NOW FEATURES: Auto-Resize. It adapts to 16, 256, or 1024 inputs automatically.
UPDATED: Learning Rate is now adjustable from GUI!
"""

import numpy as np
import cv2
import os

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class NeuronalBrainNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(100, 255, 200) # Cyan/Mint

    def __init__(self):
        super().__init__()
        self.node_title = "Neuronal Brain (ESN)"
        
        self.inputs = {
            'input_vec': 'spectrum',   # Sensory Input
            'target_vec': 'spectrum',  # Teacher Signal
            'train_gate': 'signal'
        }
        
        self.outputs = {
            'output_vec': 'spectrum',
            'brain_activity': 'image',
            'error': 'signal'
        }
        
        # Architecture Config
        self.input_dim = 16 # Start small, will auto-expand
        self.reservoir_size = 200 
        self.output_dim = 16
        self.leak_rate = 0.3      
        self.spectral_radius = 0.95 
        self.learning_rate = 0.05  # Now adjustable from GUI!
        
        # Initialize Matrices
        self.init_matrices()
        
        # State
        self.frozen = False
        self.error_val = 0.0
        self.prediction = np.zeros(self.output_dim)

    def init_matrices(self):
        # Input Projection (Input -> Reservoir)
        self.W_in = (np.random.rand(self.reservoir_size, self.input_dim) * 2 - 1) * 0.5
        
        # Recurrent Connections (Reservoir -> Reservoir)
        W_res_raw = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        radius = np.max(np.abs(np.linalg.eigvals(W_res_raw)))
        # Safety check for radius 0
        if radius == 0: radius = 1.0
        self.W_res = W_res_raw * (self.spectral_radius / radius)
        
        # Readout (Reservoir -> Output)
        self.W_out = np.zeros((self.output_dim, self.reservoir_size))
        
        # Reset State
        self.state = np.zeros(self.reservoir_size)

    def step(self):
        # 1. Get Inputs
        u = self.get_blended_input('input_vec', 'first')
        if u is None: return

        # --- AUTO-HEAL 1: Check Input Dimension ---
        if len(u) != self.input_dim:
            print(f"Brain: Adapting Input from {self.input_dim} to {len(u)}")
            self.input_dim = len(u)
            # We must re-init W_in to match new shape
            self.W_in = (np.random.rand(self.reservoir_size, self.input_dim) * 2 - 1) * 0.5
            self.frozen = False # Unfreeze to learn new pattern
        
        # Resize input vector container
        u_vec = np.zeros(self.input_dim)
        u_vec[:] = u
        
        # --- SAFETY 1: Clamp Input ---
        raw_input = np.nan_to_num(u_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        u_vec = np.clip(raw_input, -10.0, 10.0)

        # 2. Update Reservoir State (The "Thinking" Step)
        # x(t) = (1-a)*x(t-1) + a * tanh( Win*u + Wres*x(t-1) )
        
        input_injection = np.dot(self.W_in, u_vec)
        internal_echo = np.dot(self.W_res, self.state)
        
        pre_activation = input_injection + internal_echo
        new_state = np.tanh(pre_activation)
        
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
        self.state = np.nan_to_num(self.state, nan=0.0) # Nan Guard

        # 3. Readout (Prediction)
        # y = W_out * state
        self.prediction = np.dot(self.W_out, self.state)

        # 4. Training
        if not self.frozen:
            target = self.get_blended_input('target_vec', 'first')
            gate = self.get_blended_input('train_gate', 'sum')
            
            if target is not None and gate is not None and gate > 0.5:
                
                # --- AUTO-HEAL 2: Check Output Dimension ---
                # If target size changed, we need to reshape W_out
                if len(target) != self.output_dim:
                    print(f"Brain: Adapting Output from {self.output_dim} to {len(target)}")
                    self.output_dim = len(target)
                    self.W_out = np.zeros((self.output_dim, self.reservoir_size))
                    self.prediction = np.zeros(self.output_dim)

                # Align Target
                t_vec = np.zeros(self.output_dim)
                t_vec[:] = target[:self.output_dim]
                
                # --- SAFETY 3: Clamp Target ---
                safe_target = np.nan_to_num(t_vec, nan=0.0, posinf=1.0, neginf=-1.0)
                t_vec = np.clip(safe_target, -10.0, 10.0)
                
                # Error
                error = t_vec - self.prediction
                self.error_val = np.mean(np.abs(error))
                
                # Update W_out
                update_step = np.outer(error, self.state)
                
                # --- SAFETY 4: Gradient Clipping ---
                update_step = np.clip(update_step, -0.1, 0.1)
                
                self.W_out += self.learning_rate * update_step
                
                # --- SAFETY 5: NaN Rescue ---
                if not np.all(np.isfinite(self.W_out)):
                    print("Warning: Brain explosion detected. Resetting W_out.")
                    self.W_out = np.nan_to_num(self.W_out, nan=0.0)

    def get_output(self, port_name):
        if port_name == 'output_vec':
            return self.prediction
        elif port_name == 'error':
            return self.error_val
        elif port_name == 'brain_activity':
            grid_side = int(np.sqrt(self.reservoir_size))
            # Handle cases where reservoir isn't a perfect square
            trunc_len = grid_side * grid_side
            activity = self.state[:trunc_len].reshape(grid_side, grid_side)
            
            img_norm = ((activity + 1) / 2.0 * 255).astype(np.uint8)
            img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_OCEAN)
            return QtGui.QImage(img_color.data, grid_side, grid_side, grid_side*3, QtGui.QImage.Format.Format_RGB888)
        return None

    def get_config_options(self):
        return [
            ("Reservoir Size", "reservoir_size", self.reservoir_size, None),
            ("Leak Rate", "leak_rate", self.leak_rate, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Frozen", "frozen", self.frozen, [(True, True), (False, False)])
        ]
        
    def set_config_options(self, options):
        if "reservoir_size" in options:
            self.reservoir_size = int(options["reservoir_size"])
            self.init_matrices() # Reset everything on size change
            
        if "leak_rate" in options: self.leak_rate = float(options["leak_rate"])
        if "learning_rate" in options: self.learning_rate = float(options["learning_rate"])
        if "frozen" in options: self.frozen = bool(options["frozen"])

    # --- Persistence ---
    def save_custom_state(self, folder_path, node_id):
        filename = f"node_{node_id}_brain.npz"
        filepath = os.path.join(folder_path, filename)
        np.savez(filepath, W_in=self.W_in, W_res=self.W_res, W_out=self.W_out)
        return filename

    def load_custom_state(self, filepath):
        try:
            data = np.load(filepath)
            self.W_in = data['W_in']
            self.W_res = data['W_res']
            self.W_out = data['W_out']
            self.input_dim = self.W_in.shape[1]
            self.output_dim = self.W_out.shape[0]
            self.reservoir_size = self.W_res.shape[0]
            self.state = np.zeros(self.reservoir_size)
            self.frozen = True
            print(f"Loaded Brain State: In:{self.input_dim} Out:{self.output_dim}")
        except Exception as e:
            print(f"Error loading brain: {e}")