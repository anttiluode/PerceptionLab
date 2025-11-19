"""
Self-Organizing Observer Node (Modulatable)
-------------------------------------------
The "Ghost in the Machine" node.
It implements the Free Energy Principle to drive morphogenesis.

Features:
- Configurable Sensitivity: Tune how "neurotic" or "reactive" the observer is.
- Closed Loop Control: Drives growth, plasticity, and energy based on surprise.
- Meta-Cognition Ready: Accepts 'plasticity_mod' to allow chaining observers.

Inputs:
- Sensation: Real-time input (VAE Latent)
- Prediction: Memory expectation (Hebbian Latent)
- Field Energy: Quantum substrate activity
- Plasticity Mod: (NEW) Modulation from a higher-order observer.

Outputs:
- Growth Drive: Triggers morphogenesis
- Plasticity: Modulates learning rate
- Free Energy: The minimized quantity (Surprise + Entropy)
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SelfOrganizingObserverNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 215, 0) # Gold (The Observer)

    def __init__(self, latent_dim=16, growth_sensitivity=15.0, plasticity_gain=5.0, entropy_weight=0.1):
        super().__init__()
        self.node_title = "Self-Organizing Observer"
        
        self.inputs = {
            'sensation': 'spectrum',      # From RealVAE (What is happening)
            'prediction': 'spectrum',     # From HebbianLearner (What I expect)
            'field_energy': 'signal',     # From Quantum/Phi node (System energy)
            'plasticity_mod': 'signal'    # NEW: From Meta-Observer (Force learning)
        }
        
        self.outputs = {
            'growth_drive': 'signal',     # To CorticalGrowth
            'plasticity': 'signal',       # To HebbianLearner
            'entropy_out': 'signal',      # System disorder
            'free_energy': 'signal',      # The quantity being minimized
            'attention_map': 'image'      # Visualization
        }
        
        # --- Configurable Parameters ---
        self.latent_dim = int(latent_dim)
        self.growth_sensitivity = float(growth_sensitivity) # How hard to drive growth when surprised
        self.plasticity_gain = float(plasticity_gain)       # How fast to learn when surprised
        self.entropy_weight = float(entropy_weight)         # How much to penalize pure chaos
        
        # Internal State
        self.attention_vis = np.zeros((64, 64, 3), dtype=np.float32)
        
        # Output variables
        self.growth_drive_val = 0.0
        self.plasticity_val = 0.0
        self.entropy_val = 0.0
        self.free_energy_val = 0.0

    def step(self):
        # 1. Gather Inputs
        sensation = self.get_blended_input('sensation', 'first')
        prediction = self.get_blended_input('prediction', 'first')
        energy = self.get_blended_input('field_energy', 'sum') or 0.5
        plasticity_mod = self.get_blended_input('plasticity_mod', 'sum')
        
        if sensation is None:
            return

        # Normalize sensation if needed
        if len(sensation) != self.latent_dim:
            new_sens = np.zeros(self.latent_dim, dtype=np.float32)
            min_len = min(len(sensation), self.latent_dim)
            new_sens[:min_len] = sensation[:min_len]
            sensation = new_sens
            
        if prediction is None:
            prediction = np.zeros_like(sensation)
            
        # 2. Calculate Free Energy components
        
        # A. Prediction Error (Surprise)
        error_vector = sensation - prediction
        surprise = np.mean(np.square(error_vector))
        
        # B. Entropy (Uncertainty of the input itself)
        current_entropy = np.var(sensation)
        
        # C. Variational Free Energy
        # F = Surprise + (Entropy * Weight)
        free_energy = surprise + (current_entropy * self.entropy_weight)
        
        # 3. Derive Control Signals (The "Will")
        
        # Growth Drive:
        # Peak growth happens at "moderate" surprise.
        # Too little = boredom (no growth). Too much = chaos (shutdown).
        # The sensitivity knob scales the amplitude of this drive.
        growth_drive = free_energy * np.exp(-free_energy * 2.0) * self.growth_sensitivity
        
        # Plasticity (Learning Rate):
        # Learn fast when wrong.
        base_plasticity = np.tanh(surprise * self.plasticity_gain)
        
        # Apply Modulation from Meta-Observer (if connected)
        if plasticity_mod is not None:
            # If the meta-observer is surprised, it forces this observer to learn HARDER
            plasticity = base_plasticity * (1.0 + plasticity_mod * 5.0)
        else:
            plasticity = base_plasticity
        
        # 4. Visualization (The "Mind's Eye")
        side = int(np.sqrt(self.latent_dim))
        if side * side == self.latent_dim:
            err_grid = error_vector.reshape((side, side))
            err_vis = cv2.resize(err_grid, (64, 64), interpolation=cv2.INTER_NEAREST)
            self.attention_vis = cv2.applyColorMap(
                (np.clip(np.abs(err_vis) * 5.0, 0, 1) * 255).astype(np.uint8), 
                cv2.COLORMAP_HOT
            ).astype(np.float32) / 255.0
            
        # 5. Store Outputs
        self.growth_drive_val = growth_drive
        self.plasticity_val = plasticity
        self.entropy_val = current_entropy
        self.free_energy_val = free_energy

    def get_output(self, port_name):
        if port_name == 'attention_map':
            return self.attention_vis
        elif port_name == 'growth_drive':
            return float(self.growth_drive_val)
        elif port_name == 'plasticity':
            return float(self.plasticity_val)
        elif port_name == 'entropy_out':
            return float(self.entropy_val)
        elif port_name == 'free_energy':
            return float(self.free_energy_val)
        return None

    def get_display_image(self):
        # Overlay text for feedback
        img = (self.attention_vis * 255).astype(np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"FE: {self.free_energy_val:.2f}", (2, 10), font, 0.3, (255, 255, 255), 1)
        cv2.putText(img, f"GR: {self.growth_drive_val:.2f}", (2, 60), font, 0.3, (0, 255, 0), 1)
        
        # Show plasticity if boosted
        if self.plasticity_val > 1.0:
             cv2.putText(img, f"PL++: {self.plasticity_val:.2f}", (2, 35), font, 0.3, (255, 0, 255), 1)
        
        return QtGui.QImage(img.data, 64, 64, 64*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None),
            ("Growth Sensitivity", "growth_sensitivity", self.growth_sensitivity, None),
            ("Plasticity Gain", "plasticity_gain", self.plasticity_gain, None),
            ("Entropy Weight", "entropy_weight", self.entropy_weight, None)
        ]