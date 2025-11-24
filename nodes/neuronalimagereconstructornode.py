"""
Neuronal Image Reconstructor Node (Holographic Weaver)
------------------------------------------------------
The "Decoder" sibling to the Approximator.
Features:
1. Auto-Resize (Self-Healing)
2. QUANTUM INTERFERENCE PORT (The Glitch Input)
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
            'input_vec': 'spectrum',
            'target_image': 'image',
            'train_gate': 'signal',
            'quantum_interference': 'signal' # <--- THE MISSING PORT
        }
        
        self.outputs = {
            'reconstructed_image': 'image',
            'error': 'signal'
        }
        
        # Config
        self.input_dim = 16 # Will auto-adapt
        self.output_res = 64
        self.channels = 3
        self.learning_rate = 0.005
        
        # State
        self.frozen = False
        self.flat_dim = self.output_res * self.output_res * self.channels
        self.W = np.random.randn(self.flat_dim, self.input_dim).astype(np.float32) * 0.01
        
        self.current_output = None
        self.error_val = 0.0

    def step(self):
        # 1. Get Input Vector
        x_in = self.get_blended_input('input_vec', 'first')
        if x_in is None: return 
        
        # --- SAFETY: AUTO-HEAL DIMENSIONS ---
        if len(x_in) != self.input_dim:
             print(f"Reconstructor: Adapting input from {self.input_dim} to {len(x_in)}")
             self.input_dim = len(x_in)
             self.W = np.random.randn(self.flat_dim, self.input_dim).astype(np.float32) * 0.01
             self.frozen = False
             
        x_vec = np.zeros(self.input_dim, dtype=np.float32)
        x_vec[:] = x_in
        
        # Clamp Input
        raw_input = np.nan_to_num(x_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        x_vec = np.clip(raw_input, -10.0, 10.0)

        # 2. Forward Pass (Calculate Base Image)
        raw_output = np.dot(self.W, x_vec)
        
        # --- QUANTUM INTERFERENCE INJECTION ---
        # We listen to the Bloch Qubit. If it's spinning, we glitch the matrix.
        interference = self.get_blended_input('quantum_interference', 'sum')
        if interference is not None and interference != 0:
            # Scale up the interference to cause visible tearing
            glitch_strength = 5.0 
            raw_output += interference * glitch_strength

        # 3. Activation (Tanh)
        prediction = np.tanh(raw_output) 
        
        # Reshape to Image for Display
        img_01 = (prediction + 1.0) / 2.0
        self.current_output = img_01.reshape((self.output_res, self.output_res, self.channels))

        # 4. Training Loop
        if not self.frozen:
            train_gate = self.get_blended_input('train_gate', 'sum')
            target_img = self.get_blended_input('target_image', 'first')
            
            if train_gate is not None and train_gate > 0.5 and target_img is not None:
                # Resize and format target
                t_img_s = cv2.resize(target_img, (self.output_res, self.output_res))
                
                # Handle Grayscale/RGBA
                if len(t_img_s.shape) == 2: 
                    t_img_s = cv2.cvtColor(t_img_s, cv2.COLOR_GRAY2RGB)
                elif t_img_s.shape[2] == 4: 
                    t_img_s = cv2.cvtColor(t_img_s, cv2.COLOR_RGBA2RGB)
                
                # Normalize Target
                t_flat = (t_img_s.flatten().astype(np.float32) / 255.0) * 2.0 - 1.0
                
                if t_flat.shape != prediction.shape: return

                # Calculate Error
                diff = t_flat - prediction
                self.error_val = np.mean(np.abs(diff))
                
                # Gradient Calculation
                gradient = diff * (1 - prediction**2)
                
                # Safety Checks
                update_step = np.outer(gradient, x_vec)
                update_step = np.clip(update_step, -0.1, 0.1) 
                
                self.W += self.learning_rate * update_step
                self.W *= 0.999
                
                if not np.all(np.isfinite(self.W)):
                    self.W = np.nan_to_num(self.W, nan=0.0, posinf=0.1, neginf=-0.1)

    def get_output(self, port_name):
        if port_name == 'reconstructed_image': return self.current_output
        elif port_name == 'error': return self.error_val
        return None

    # --- Persistence ---
    def save_custom_state(self, folder_path, node_id):
        filename = f"node_{node_id}_decoder_weights.npy"
        filepath = os.path.join(folder_path, filename)
        np.save(filepath, self.W)
        return filename

    def load_custom_state(self, filepath):
        try:
            loaded_W = np.load(filepath)
            if loaded_W.shape == self.W.shape:
                self.W = loaded_W.astype(np.float32)
                self.frozen = True
                print(f"Loaded decoder weights")
        except Exception as e:
            print(e)
            
    def get_display_image(self):
        if self.current_output is not None:
            disp = np.clip(self.current_output, 0, 1)
            img_u8 = (disp * 255).astype(np.uint8)
            if self.frozen:
                cv2.putText(img_u8, "FROZEN", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            else:
                cv2.putText(img_u8, "TRAINING", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            h, w, c = img_u8.shape
            return QtGui.QImage(img_u8.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        else:
            black_img = np.zeros((64, 64, 3), dtype=np.uint8)
            return QtGui.QImage(black_img.data, 64, 64, 64*3, QtGui.QImage.Format.Format_RGB888)
            
    def get_config_options(self):
        return [
            ("Input Dim", "input_dim", self.input_dim, None),
            ("Output Res", "output_res", self.output_res, None),
            ("Frozen", "frozen", self.frozen, [(True, True), (False, False)])
        ]
        
    def set_config_options(self, options):
        reset = False
        if "input_dim" in options: self.input_dim = int(options["input_dim"])
        if "output_res" in options:
            self.output_res = int(options["output_res"])
            reset = True
        if "frozen" in options: self.frozen = bool(options["frozen"])
            
        if reset:
            self.flat_dim = self.output_res * self.output_res * self.channels
            self.W = np.random.randn(self.flat_dim, self.input_dim).astype(np.float32) * 0.01