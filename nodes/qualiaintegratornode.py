"""
Qualia Integrator Node - Models qualia as the integration of a
stable latent "Soma" state and a chaotic "Dendrite" phase field.

This node implements the core hypothesis:
Qualia = (Soma_Latent * Coherence) + (Dendrite_Field * (1.0 - Coherence))

- Coherence = 1.0 (Healthy): Output is the stable, learned latent vector.
- Coherence = 0.0 (Damaged): Output is the raw, "leaked" phase field.
- Coherence = 0.5 (Mixed): Output is a blend, a "fractal leak"
  superimposed on reality.
"""

import numpy as np
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# -----------------------------

class QualiaIntegratorNode(BaseNode):
    """
    Blends a stable latent vector (Soma) with a raw field vector (Dendrite)
    based on a 'coherence' (brain health) signal.
    """
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(255, 100, 200) # Bright Magenta (Qualia Pink)

    def __init__(self, latent_dim=16):
        super().__init__()
        self.node_title = "Qualia Integrator"
        self.latent_dim = int(latent_dim)

        self.inputs = {
            'soma_latent_in': 'spectrum',      # Stable latent vector (e.g., from VAE)
            'dendrite_field_in': 'spectrum',   # Raw phase field vector (e.g., from ChaoticField)
            'coherence_in': 'signal'       # 0.0 (Total Leak) to 1.0 (Stable)
        }
        self.outputs = {
            'qualia_out': 'spectrum',          # The final, integrated latent vector
            'leakage_amount': 'signal'       # 1.0 - coherence
        }

        # Internal state
        self.qualia_out = np.zeros(self.latent_dim, dtype=np.float32)
        self.leakage_amount = 0.0
        self.soma_vis = np.zeros(self.latent_dim, dtype=np.float32)
        self.dendrite_vis = np.zeros(self.latent_dim, dtype=np.float32)
        self.coherence = 1.0

    def step(self):
        # 1. Get Inputs
        soma = self.get_blended_input('soma_latent_in', 'first')
        dendrite = self.get_blended_input('dendrite_field_in', 'first')
        coherence_sig = self.get_blended_input('coherence_in', 'sum')

        if coherence_sig is None:
            self.coherence = 1.0 # Default to stable/healthy
        else:
            self.coherence = np.clip(coherence_sig, 0.0, 1.0)
        
        self.leakage_amount = 1.0 - self.coherence

        # 2. Handle missing inputs
        if soma is None:
            soma = np.zeros(self.latent_dim, dtype=np.float32)
        if dendrite is None:
            dendrite = np.zeros(self.latent_dim, dtype=np.float32)

        # 3. Ensure vectors match the target latent dimension
        if len(soma) != self.latent_dim:
            soma = self._resize_vector(soma, self.latent_dim)
        if len(dendrite) != self.latent_dim:
            dendrite = self._resize_vector(dendrite, self.latent_dim)

        # Store for visualization
        self.soma_vis = soma
        self.dendrite_vis = dendrite

        # 4. THE QUALIA EQUATION
        # Qualia = (Soma * Coherence) + (Dendrite * Leakage)
        soma_contribution = soma * self.coherence
        dendrite_contribution = dendrite * self.leakage_amount
        
        self.qualia_out = soma_contribution + dendrite_contribution

    def _resize_vector(self, vec, target_dim):
        """Pads or truncates a vector to the target dimension."""
        current_dim = len(vec)
        if current_dim == target_dim:
            return vec
        
        new_vec = np.zeros(target_dim, dtype=np.float32)
        if current_dim > target_dim:
            new_vec = vec[:target_dim] # Truncate
        else:
            new_vec[:current_dim] = vec # Pad
        return new_vec

    def get_output(self, port_name):
        if port_name == 'qualia_out':
            return self.qualia_out.astype(np.float32)
        elif port_name == 'leakage_amount':
            return float(self.leakage_amount)
        return None

    def get_display_image(self):
        """Visualize the integration: Soma, Dendrite, and final Qualia"""
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # --- Helper to draw a vector bar graph ---
        def draw_vector(vector, y_offset, color_rgb):
            bar_width = max(1, w // len(vector))
            val_max = np.abs(vector).max()
            if val_max < 1e-6: val_max = 1.0
            
            for i, val in enumerate(vector):
                x = i * bar_width
                norm_val = val / val_max
                bar_h = int(np.clip(abs(norm_val) * (h/3 - 5), 0, h/3 - 5))
                y_base = y_offset + (h // 6)
                
                if val >= 0:
                    cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color_rgb, -1)
                else:
                    cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color_rgb, -1)
        
        # Draw all three vectors
        draw_vector(self.soma_vis, 0, (0, 200, 0)) # SOMA = Green (stable)
        draw_vector(self.dendrite_vis, h // 3, (200, 0, 0)) # DENDRITE = Red (raw)
        draw_vector(self.qualia_out, 2 * h // 3, (200, 100, 255)) # QUALIA = Pink (mixed)

        # Draw labels and coherence bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Soma (Latent)", (5, 12), font, 0.3, (0, 255, 0), 1)
        cv2.putText(img, "Dendrite (Field)", (5, h//3 + 12), font, 0.3, (0, 0, 255), 1)
        cv2.putText(img, "Qualia (Final)", (5, 2*h//3 + 12), font, 0.3, (255, 100, 200), 1)
        
        # Coherence Bar
        bar_w = int(self.coherence * (w - 10))
        cv2.rectangle(img, (5, h - 10), (5 + bar_w, h - 5), (0, 255, 255), -1)
        cv2.putText(img, f"Coherence: {self.coherence:.2f}", (w - 80, 12), font, 0.3, (0, 255, 255), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Latent Dim", "latent_dim", self.latent_dim, None)
        ]