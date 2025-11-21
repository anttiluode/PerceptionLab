"""
P-KAS Node (Phase-Keyed Associative Storage)
--------------------------------------------
Simulates a network of coupled oscillators solving a constraint satisfaction problem.
Based on the principle that "intelligence emerges from geometry-driven phase dynamics."

Mechanism:
- Oscillators represent variables (e.g., "Yes/No", "Red/Blue/Green").
- Couplings represent constraints (e.g., "Must be different", "Must be same").
- The system "relaxes" into a low-energy phase configuration that satisfies the constraints.

Visualizes:
- The Phase Landscape (Color).
- The Energy Minimization (Convergence).
"""

import numpy as np
from PyQt6 import QtGui
import cv2

# --- CRITICAL IMPORT BLOCK ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------

class PKASNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(200, 50, 100)  # Energetic Pink
    
    def __init__(self, num_oscillators=16, coupling_strength=0.5):
        super().__init__()
        self.node_title = "P-KAS Solver (Phase Dynamics)"
        
        self.inputs = {
            'input_energy': 'signal',   # Injection of energy (arousal)
            'constraint_mod': 'signal' # Modulate constraint strength
        }
        
        self.outputs = {
            'solution_state': 'image', # Visual phase map
            'system_energy': 'signal'  # How "solved" is it? (Low = Solved)
        }
        
        self.N = int(num_oscillators)
        self.K = float(coupling_strength)
        
        # System State
        self.phases = np.random.rand(self.N) * 2 * np.pi
        self.frequencies = np.random.normal(1.0, 0.1, self.N) # Intrinsic freqs
        
        # Connectivity (The Constraints)
        # We create a random constraint graph (e.g., Graph Coloring)
        # -1 = Anti-phase (Must be different), 1 = In-phase (Must be same)
        self.weights = np.random.choice([-1, 0, 1], size=(self.N, self.N), p=[0.3, 0.4, 0.3])
        np.fill_diagonal(self.weights, 0)
        
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)
        self.energy = 1.0

    def step(self):
        # 1. Get Inputs
        input_e = self.get_blended_input('input_energy', 'sum') or 0.0
        const_mod = self.get_blended_input('constraint_mod', 'sum') or 0.0
        
        eff_K = self.K * (1.0 + const_mod)
        
        # 2. Kuramoto Dynamics (The Solver)
        # dtheta/dt = omega + K * sum( weight * sin(theta_j - theta_i) )
        
        # Calculate phase differences matrix
        diff_matrix = self.phases[None, :] - self.phases[:, None]
        interaction = np.sin(diff_matrix)
        
        # Apply constraints (weights)
        # If weight is -1 (Anti-synchronize), we want sin(diff) to be non-zero (push away)
        # Standard Kuramoto minimizes phase difference for positive K.
        # To maximize difference (anti-sync), we use negative weight.
        
        coupling = np.sum(self.weights * interaction, axis=1)
        
        # Update phases
        dt = 0.1
        noise = np.random.normal(0, 0.01 + input_e * 0.1, self.N) # Injection
        d_theta = self.frequencies + (eff_K / self.N) * coupling + noise
        
        self.phases = (self.phases + d_theta * dt) % (2 * np.pi)
        
        # 3. Calculate System Energy (Frustration)
        # Energy = -0.5 * sum( weight * cos(theta_j - theta_i) )
        # Low energy means constraints are satisfied.
        energy_mat = self.weights * np.cos(diff_matrix)
        self.energy = -0.5 * np.sum(energy_mat) / (self.N**2)
        
        # Normalize energy for output (approx range)
        self.energy = (self.energy + 0.5) # Shift to 0-1 range
        
        # 4. Visualization (The Phase Landscape)
        self._render_state()

    def _render_state(self):
        # Visualize oscillators as a ring
        self.display_img.fill(20)
        
        center = (64, 64)
        radius = 50
        
        # Draw connections (Constraints)
        for i in range(self.N):
            for j in range(i+1, self.N):
                w = self.weights[i, j]
                if w != 0:
                    # Get positions
                    xi = int(center[0] + radius * np.cos(2*np.pi*i/self.N))
                    yi = int(center[1] + radius * np.sin(2*np.pi*i/self.N))
                    xj = int(center[0] + radius * np.cos(2*np.pi*j/self.N))
                    yj = int(center[1] + radius * np.sin(2*np.pi*j/self.N))
                    
                    # Color based on satisfaction
                    # If w=1 (sync) and phases close -> Green
                    # If w=-1 (anti) and phases far -> Green
                    diff = np.abs(self.phases[i] - self.phases[j])
                    diff = min(diff, 2*np.pi - diff)
                    
                    satisfied = False
                    if w > 0: # Want sync (diff ~ 0)
                        satisfied = diff < 0.5
                    else: # Want anti (diff ~ pi)
                        satisfied = diff > 2.5
                        
                    col = (0, 255, 0) if satisfied else (0, 0, 255) # Red if frustrated
                    cv2.line(self.display_img, (xi, yi), (xj, yj), col, 1)

        # Draw oscillators
        for i in range(self.N):
            x = int(center[0] + radius * np.cos(2*np.pi*i/self.N))
            y = int(center[1] + radius * np.sin(2*np.pi*i/self.N))
            
            # Phase color wheel
            hue = int((self.phases[i] / (2*np.pi)) * 179)
            osc_color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0,0]
            osc_color = (int(osc_color[0]), int(osc_color[1]), int(osc_color[2]))
            
            cv2.circle(self.display_img, (x, y), 6, osc_color, -1)
            cv2.circle(self.display_img, (x, y), 7, (255, 255, 255), 1)

    def get_output(self, port_name):
        if port_name == 'solution_state':
            return self.display_img.astype(np.float32) / 255.0
        elif port_name == 'system_energy':
            return float(self.energy)
        return None

    def get_display_image(self):
        # Add Energy Text
        img = self.display_img.copy()
        cv2.putText(img, f"Energy: {self.energy:.3f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Num Oscillators", "N", self.N, None),
            ("Coupling (K)", "K", self.K, None)
        ]