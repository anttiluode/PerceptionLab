"""
Neural Flow Encoder Node
------------------------
A monolithic node that:
1. Accepts raw Input Vectors (EEG/Spectrum).
2. Projects them to a variable Latent Size (e.g., 16, 32, 64).
3. Amplifies the signal.
4. Visualizes the history as a "Temporal Flow" (Liquid Blobs).
"""

import numpy as np
import cv2
from collections import deque
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class NeuralFlowEncoderNode(BaseNode):
    NODE_CATEGORY = "Visual"
    NODE_COLOR = QtGui.QColor(0, 180, 200) # Teal

    def __init__(self):
        super().__init__()
        self.node_title = "Neural Flow Encoder"
        
        self.inputs = {
            'input_vector': 'spectrum',  # Raw EEG / Input
            'amplification': 'signal'    # Gain Control
        }
        
        self.outputs = {
            'latent_vector': 'spectrum', # The Projected Output
            'flow_image': 'image'        # The Liquid Visualization
        }
        
        # Configuration defaults
        self.target_latent_dim = 16
        self.buffer_size = 100 # How much "Time" to keep in the image
        self.current_input_dim = 0
        self.projection_matrix = None
        
        # History Buffer (deque is faster for scrolling data)
        self.history = deque(maxlen=self.buffer_size)
        
        # internal state
        self.current_latent = np.zeros(self.target_latent_dim)
        self.generated_image = None

    def step(self):
        # 1. Get Inputs
        raw_in = self.get_blended_input('input_vector', 'first')
        gain = self.get_blended_input('amplification', 'sum')
        if gain is None: gain = 1.0
        
        if raw_in is None:
            return

        # 2. Auto-Projection Logic (The "Self-Healing" Matrix)
        # We need to map Input Dimension -> Target Latent Dimension
        input_dim = len(raw_in)
        
        # If dimensions changed (or first run), create a random projection matrix
        if input_dim != self.current_input_dim or self.projection_matrix is None:
            # Check if target dim changed in config too
            if self.projection_matrix is not None and self.projection_matrix.shape[0] != self.target_latent_dim:
                pass # Trigger rebuild
                
            print(f"FlowEncoder: Creating Projection {input_dim} -> {self.target_latent_dim}")
            self.current_input_dim = input_dim
            # Random orthogonal-ish matrix to mix the signals interestingly
            self.projection_matrix = np.random.randn(self.target_latent_dim, input_dim) * 0.5
            
            # Reset history on structure change
            self.history.clear()

        # 3. Project and Amplify
        # Latent = Matrix * Input * Gain
        projected = np.dot(self.projection_matrix, raw_in) * gain
        
        # Tanh activation to keep it "organic" and prevent infinity
        self.current_latent = np.tanh(projected)
        
        # 4. Update History (The "Time" aspect)
        self.history.append(self.current_latent)
        
        # 5. Generate Flow Image (The "Blob" aspect)
        if len(self.history) > 1:
            # Convert history deque to numpy array (Time x Latent)
            data_block = np.array(self.history)
            
            # Normalize for visualization (0..1)
            # We add 1.0 and divide by 2.0 because tanh is -1..1
            vis_data = (data_block + 1.0) / 2.0
            
            # Resize to look like a liquid flow
            # We stretch the width (Latent Dims) and Height (Time)
            # INTER_CUBIC creates the "Blob/Gradient" look instead of pixels
            self.generated_image = cv2.resize(
                vis_data, 
                (256, 256), 
                interpolation=cv2.INTER_CUBIC
            )

    def get_output(self, port_name):
        if port_name == 'latent_vector':
            return self.current_latent
        elif port_name == 'flow_image':
            return self.generated_image
        return None

    def get_display_image(self):
        if self.generated_image is None:
            return None
            
        # Apply Heatmap to make it look sci-fi
        # COLORMAP_JET or COLORMAP_OCEAN looks best for flows
        img_u8 = (np.clip(self.generated_image, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        
        # Rotate so Time flows Horizontal or Vertical?
        # Let's keep Time = Vertical (Waterfall)
        
        # Resize for the small node display
        display_small = cv2.resize(img_color, (128, 128))
        
        return QtGui.QImage(display_small.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Latent Size", "target_latent_dim", self.target_latent_dim, None),
            ("Flow History", "buffer_size", self.buffer_size, None)
        ]
        
    def set_config_options(self, options):
        if "target_latent_dim" in options:
            new_dim = int(options["target_latent_dim"])
            if new_dim != self.target_latent_dim:
                self.target_latent_dim = new_dim
                self.projection_matrix = None # Force matrix rebuild
                
        if "buffer_size" in options:
            self.buffer_size = int(options["buffer_size"])
            self.history = deque(maxlen=self.buffer_size)