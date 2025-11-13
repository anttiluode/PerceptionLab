"""
Rhythm Gated Perturbation Node
--------------------------------
Models a "temporal gating" mechanism. It takes a stable latent
vector ("Soma" thought) and a "Rhythm" signal ("Dendritic" clock).

If the rhythm becomes incoherent (unstable), it "breaks the gate"
and "leaks" a high-frequency "Phase Field" (a perturbation)
into the latent vector, simulating a "fractal leak" hallucination
.
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# --------------------------

class RhythmGatedPerturbationNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(140, 70, 180) # Deep Purple
    
    def __init__(self, history_length=50, coherence_threshold=0.8, perturb_strength=1.0):
        super().__init__()
        self.node_title = "Rhythm Gated Perturbation"
        
        self.inputs = {
            'latent_in': 'spectrum',       # The stable "Soma" state
            'rhythm_in': 'signal',       # The "Dendritic" timing signal
            'fractal_field_in': 'spectrum' # Optional: The "raw" field to leak
        }
        self.outputs = {
            'latent_out': 'spectrum',      # The final (potentially corrupted) state
            'leakage_amount': 'signal'   # 0=Stable, 1=Full Leak
        }
        
        # Configurable
        self.history_length = int(history_length)
        self.coherence_threshold = float(coherence_threshold)
        self.perturb_strength = float(perturb_strength)
        
        # Internal state
        self.rhythm_history = deque(maxlen=self.history_length)
        self.current_coherence = 1.0 # Start in a stable state
        self.leakage_amount_out = 0.0
        self.latent_out = None
        
        # Ensure deque is initialized
        for _ in range(self.history_length):
            self.rhythm_history.append(0.0)

    def step(self):
        # 1. Get Inputs
        latent_in = self.get_blended_input('latent_in', 'first')
        rhythm_in = self.get_blended_input('rhythm_in', 'sum')
        fractal_field_in = self.get_blended_input('fractal_field_in', 'first')
        
        # Update rhythm history, even if it's None (to detect drops)
        self.rhythm_history.append(rhythm_in if rhythm_in is not None else 0.0)
        
        # 2. Calculate Rhythm Coherence
        # Coherence = inverse of standard deviation (variance)
        rhythm_std = np.std(self.rhythm_history)
        # This maps std=0 to coherence=1. Higher std -> lower coherence.
        self.current_coherence = 1.0 / (1.0 + rhythm_std * 10.0) 
        self.current_coherence = np.clip(self.current_coherence, 0.0, 1.0)

        # 3. Calculate "Fractal Leakage"
        if self.current_coherence < self.coherence_threshold:
            # The gate is "broken"
            self.leakage_amount_out = (self.coherence_threshold - self.current_coherence) / self.coherence_threshold
        else:
            # The gate is "stable"
            self.leakage_amount_out = 0.0
            
        self.leakage_amount_out = np.clip(self.leakage_amount_out, 0.0, 1.0)
        
        # 4. Apply the Leak
        if latent_in is None:
            self.latent_out = None
            return

        if self.leakage_amount_out > 0.01:
            # --- THE FRACTAL LEAK IS HAPPENING ---
            
            # Get the perturbation vector
            if fractal_field_in is not None and len(fractal_field_in) == len(latent_in):
                perturb_vector = fractal_field_in
            else:
                # If no field is provided, create high-frequency noise
                perturb_vector = np.random.randn(len(latent_in)).astype(np.float32)
            
            # Scale perturbation
            perturb_vector = perturb_vector * self.perturb_strength

            # Blend: (Stable Thought * Coherence) + (Raw Field * Leakage)
            self.latent_out = (latent_in * (1.0 - self.leakage_amount_out)) + \
                              (perturb_vector * self.leakage_amount_out)
        else:
            # --- STABLE OPERATION ---
            self.latent_out = latent_in

    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.latent_out
        elif port_name == 'leakage_amount':
            return self.leakage_amount_out
        return None
        
    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw Coherence / Leakage
        coherence_w = int(self.current_coherence * w)
        leakage_w = int(self.leakage_amount_out * w)
        
        # Coherence Bar (Green)
        cv2.rectangle(img, (0, 0), (coherence_w, h // 3), (0, 150, 0), -1)
        # Leakage Bar (Red)
        cv2.rectangle(img, (0, h // 3), (leakage_w, 2 * h // 3), (150, 0, 0), -1)
        
        cv2.putText(img, f"Coherence: {self.current_coherence:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Leakage: {self.leakage_amount_out:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw a line for the coherence threshold
        thresh_x = int(self.coherence_threshold * w)
        cv2.line(img, (thresh_x, 0), (thresh_x, h // 3), (0, 255, 0), 2)
        
        # Display the output latent vector
        if self.latent_out is not None:
            latent_dim = len(self.latent_out)
            bar_width = max(1, w // latent_dim)
            val_max = np.abs(self.latent_out).max()
            if val_max < 1e-6: val_max = 1.0
            
            for i, val in enumerate(self.latent_out):
                x = i * bar_width
                norm_val = val / val_max
                bar_h = int(np.clip(abs(norm_val) * (h/3 - 5), 0, h/3 - 5))
                y_base = h - (h // 6) # Center of bottom 3rd
                
                color = (200, 200, 200) # Default
                if self.leakage_amount_out > 0.01:
                    color = (255, 255, 0) # Tint yellow during leak

                if val >= 0:
                    cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color, -1)
                else:
                    cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color, -1)

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("History Length", "history_length", self.history_length, None),
            ("Coherence Threshold", "coherence_threshold", self.coherence_threshold, None),
            ("Perturbation Strength", "perturb_strength", self.perturb_strength, None)
        ]