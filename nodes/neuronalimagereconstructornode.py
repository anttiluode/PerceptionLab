"""
Neuronal Image Reconstructor Node (The Holographic Weaver)
----------------------------------------------------------
Converts a latent vector (thought) into an image (hallucination).
It learns to associate specific input vectors with specific target images
using a Hebbian projection matrix (Holography).

Use this to visualize what your Hebbian Brain is "thinking".
"""
import numpy as np
import cv2
import os

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class NeuronalImageReconstructorNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 100, 150) # Pinkish Red

    def __init__(self):
        super().__init__()
        self.node_title = "Holographic Weaver"
        
        self.inputs = {
            'input_vec': 'spectrum',    # The abstract thought (from Brain/Approximator)
            'target_image': 'image',    # The reality (to learn from)
            'train_gate': 'signal',     # 1.0 = Learn, 0.0 = Dream
            'glitch_mod': 'signal'      # Optional: Quantum interference
        }
        
        self.outputs = {
            'reconstructed_image': 'image',
            'error': 'signal'
        }
        
        # Config
        self.input_dim = 16
        self.output_res = 64
        self.learning_rate = 0.01
        
        # State
        self.W = None # Weights (The Hologram)
        self.current_output = np.zeros((self.output_res, self.output_res, 3), dtype=np.float32)
        self.error_val = 0.0
        self.frozen = False

    def _init_weights(self, in_dim):
        self.input_dim = in_dim
        flat_dim = self.output_res * self.output_res * 3
        # Initialize with small random noise (The "Quantum Foam")
        self.W = np.random.randn(flat_dim, self.input_dim).astype(np.float32) * 0.01
        print(f"Weaver: Initialized W ({flat_dim}x{self.input_dim})")

    def step(self):
        # 1. Get Input
        vec = self.get_blended_input('input_vec', 'first')
        if vec is None: return

        # Auto-init if needed
        if self.W is None or len(vec) != self.input_dim:
            self._init_weights(len(vec))
            
        # 2. Forward Pass (Dreaming)
        # Image = Tanh( W * Vector )
        # This is the holographic projection step
        flat_img = np.dot(self.W, vec)
        
        # Apply Glitch (if connected)
        glitch = self.get_blended_input('glitch_mod', 'sum')
        if glitch is not None and glitch != 0:
            flat_img += np.random.randn(len(flat_img)) * glitch * 5.0
            
        # Activation (squash to -1..1)
        flat_img = np.tanh(flat_img)
        
        # Reshape to Image
        # Map -1..1 to 0..1
        self.current_output = ((flat_img + 1.0) / 2.0).reshape((self.output_res, self.output_res, 3))

        # 3. Learning (if enabled)
        if not self.frozen:
            train = self.get_blended_input('train_gate', 'sum')
            target = self.get_blended_input('target_image', 'first')
            
            if train is not None and train > 0.5 and target is not None:
                # Resize target to match output resolution
                if target.shape[:2] != (self.output_res, self.output_res):
                    target = cv2.resize(target, (self.output_res, self.output_res))
                
                # Flatten target
                if target.ndim == 2: target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
                target_flat = (target.flatten() * 2.0) - 1.0 # Map 0..1 to -1..1
                
                # Error Calculation
                error = target_flat - flat_img
                self.error_val = np.mean(np.abs(error))
                
                # Hebbian/Delta Update: dW = lr * error * input.T
                # This encodes the image structure into the weights
                update = np.outer(error, vec)
                self.W += update * self.learning_rate
                
                # Decay/Stabilize
                self.W *= 0.9995

    def get_output(self, port_name):
        if port_name == 'reconstructed_image': return self.current_output
        if port_name == 'error': return self.error_val
        return None

    def get_display_image(self):
        img = (np.clip(self.current_output, 0, 1) * 255).astype(np.uint8)
        
        if self.frozen:
             cv2.putText(img, "FROZEN", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
        elif self.error_val > 0:
             cv2.putText(img, f"Err: {self.error_val:.2f}", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
             
        return QtGui.QImage(img.data, self.output_res, self.output_res, self.output_res*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Resolution", "output_res", self.output_res, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Frozen", "frozen", self.frozen, [(True, True), (False, False)])
        ]
        
    def set_config_options(self, options):
        if "output_res" in options: 
            new_res = int(options["output_res"])
            if new_res != self.output_res:
                self.output_res = new_res
                self.W = None # Force re-init
        if "learning_rate" in options: self.learning_rate = float(options["learning_rate"])
        if "frozen" in options: self.frozen = bool(options["frozen"])

    # --- State Persistence (Save the learned hologram) ---
    def save_custom_state(self, folder_path, node_id):
        if self.W is not None:
            path = os.path.join(folder_path, f"weaver_{node_id}.npy")
            np.save(path, self.W)
            return f"weaver_{node_id}.npy"
        return None
        
    def load_custom_state(self, path):
        if os.path.exists(path):
            self.W = np.load(path)
            self.input_dim = self.W.shape[1]
            print(f"Weaver loaded weights: {self.W.shape}")