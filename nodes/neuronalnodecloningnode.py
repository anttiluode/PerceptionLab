"""
Neuronal Approximator Node
--------------------------
The "Student" node. It approximates the function of another node 
by learning a weight matrix (W) that maps Input -> Target.

Workflow:
1. Connect Teacher Input -> This Input
2. Connect Teacher Output -> This Target
3. Set 'Training' to True.
4. Once Error is low, set 'Frozen' to True.
5. Delete Teacher.

This node saves its W matrix to disk when the graph is saved.
"""

import numpy as np
import cv2
import os

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class NeuronalApproximatorNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 50, 100) # Intense Learning Red

    def __init__(self, input_dim=16, output_dim=16, learning_rate=0.01):
        super().__init__()
        self.node_title = "Neuronal Approximator"
        
        self.inputs = {
            'input_vec': 'spectrum',   # X
            'target_vec': 'spectrum',  # Y (Expected Output)
            'train_gate': 'signal'     # 1.0 = Learn, 0.0 = Inference
        }
        
        self.outputs = {
            'predicted_vec': 'spectrum', # Y_hat (Approximation)
            'error': 'signal'            # |Y - Y_hat|
        }
        
        # Config
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
        
        # State
        self.W = np.zeros((self.output_dim, self.input_dim), dtype=np.float32)
        self.frozen = False
        self.error_val = 0.0
        self.prediction = np.zeros(self.output_dim, dtype=np.float32)
        
        # Initialize W with identity-like structure if dims match, else random
        if self.input_dim == self.output_dim:
            self.W = np.eye(self.input_dim, dtype=np.float32)
        else:
            self.W = np.random.randn(self.output_dim, self.input_dim).astype(np.float32) * 0.1

    def step(self):
        # 1. Get Input X
        x = self.get_blended_input('input_vec', 'first')
        
        # Handling input resizing/padding
        if x is None:
            self.prediction.fill(0)
            return
            
        # Ensure X matches input_dim
        x_vec = np.zeros(self.input_dim, dtype=np.float32)
        n = min(len(x), self.input_dim)
        x_vec[:n] = x[:n]
        
        # 2. Forward Pass (Inference)
        # y_hat = W * x
        self.prediction = np.dot(self.W, x_vec)
        
        # 3. Training Logic
        if not self.frozen:
            train_gate = self.get_blended_input('train_gate', 'sum')
            target = self.get_blended_input('target_vec', 'first')
            
            # Only train if gate is open AND we have a target (Teacher)
            if train_gate is not None and train_gate > 0.5 and target is not None:
                
                # Ensure Target matches output_dim
                t_vec = np.zeros(self.output_dim, dtype=np.float32)
                m = min(len(target), self.output_dim)
                t_vec[:m] = target[:m]
                
                # 4. Calculate Error (Delta)
                # e = t - y_hat
                error_vec = t_vec - self.prediction
                self.error_val = np.mean(np.abs(error_vec))
                
                # 5. Update Weights (Delta Rule / LMS)
                # W_new = W_old + learning_rate * error * input.T
                # Using outer product for vector update
                delta_W = self.learning_rate * np.outer(error_vec, x_vec)
                
                # Optional: Weight Decay (Forgetting) to keep values stable
                self.W *= 0.999
                self.W += delta_W

        # 4. Output
        # (Prediction is set in step 2)

    def get_output(self, port_name):
        if port_name == 'predicted_vec':
            return self.prediction
        elif port_name == 'error':
            return self.error_val
        return None
    
    # --- PERSISTENCE METHODS (Called by Host v7) ---
    def save_custom_state(self, folder_path, node_id):
        """Saves the W matrix to a .npy file"""
        filename = f"node_{node_id}_weights.npy"
        filepath = os.path.join(folder_path, filename)
        np.save(filepath, self.W)
        return filename

    def load_custom_state(self, filepath):
        """Loads the W matrix from a .npy file"""
        try:
            loaded_W = np.load(filepath)
            if loaded_W.shape == self.W.shape:
                self.W = loaded_W.astype(np.float32)
                self.frozen = True # Auto-freeze on load (assumption: training is done)
                print(f"NeuronalApproximator: Loaded weights from {os.path.basename(filepath)}")
            else:
                print(f"NeuronalApproximator: Weight shape mismatch. Expected {self.W.shape}, got {loaded_W.shape}")
        except Exception as e:
            print(f"NeuronalApproximator: Failed to load state: {e}")

    def get_display_image(self):
        # Visualize the W Matrix
        # Normalize for display
        w_min, w_max = self.W.min(), self.W.max()
        if w_max - w_min > 1e-6:
            w_norm = (self.W - w_min) / (w_max - w_min)
        else:
            w_norm = np.zeros_like(self.W)
            
        w_u8 = (w_norm * 255).astype(np.uint8)
        
        # Use heatmap
        img_color = cv2.applyColorMap(w_u8, cv2.COLORMAP_JET)
        
        # Resize for visibility
        img_resized = cv2.resize(img_color, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Overlay status
        status_text = "FROZEN" if self.frozen else "TRAINING"
        status_color = (0, 255, 0) if self.frozen else (0, 0, 255)
        
        cv2.putText(img_resized, status_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        cv2.putText(img_resized, f"Err: {self.error_val:.4f}", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(img_resized.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Input Dim", "input_dim", self.input_dim, None),
            ("Output Dim", "output_dim", self.output_dim, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Frozen", "frozen", self.frozen, [(True, True), (False, False)])
        ]
    
    def set_config_options(self, options):
        # Handle resizing W if dimensions change
        dims_changed = False
        if "input_dim" in options: 
            self.input_dim = int(options["input_dim"])
            dims_changed = True
        if "output_dim" in options: 
            self.output_dim = int(options["output_dim"])
            dims_changed = True
        
        if dims_changed:
            self.W = np.random.randn(self.output_dim, self.input_dim).astype(np.float32) * 0.1
            self.prediction = np.zeros(self.output_dim, dtype=np.float32)
            
        if "learning_rate" in options: self.learning_rate = float(options["learning_rate"])
        if "frozen" in options: self.frozen = bool(options["frozen"])