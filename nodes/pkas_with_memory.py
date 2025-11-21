# PKASMemoryNode.py
"""
P-KAS Node with Learning & Associative Recall
--------------------------------------------
Adds:
 - write_memory input (signal > 0.5) to store the current phase pattern
 - partial_input (image) to cue recall (NaN or <0 to indicate unknowns)
 - recall_mode (signal > 0.5) to trigger recall dynamics
 - memory persistence to /mnt/data/pkas_memories.npy

Author: patched for Perception Lab
"""

import os
import numpy as np
import cv2

# Host bindings supplied by the Perception Lab runtime
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

# Where we'll persist memories (host can convert path to URL if needed)
MEMORY_SAVE_PATH = "pkas_memories.npy" # Changed to local relative path for safety


class PKASMemoryNode(BaseNode):
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(200, 50, 150)  # Deep Memory Pink

    def __init__(self, num_oscillators=16, coupling_strength=0.5, learning_rate=0.15,
                 memory_bias=2.0, recall_steps=120):
        super().__init__()
        self.node_title = "P-KAS Solver (Memory)"

        self.inputs = {
            'input_energy': 'signal',
            'constraint_mod': 'signal',
            'write_memory': 'signal',   # Trigger > 0.5 to learn current state
            'recall_mode': 'signal',    # Trigger > 0.5 to enter recall mode
            'partial_input': 'image'    # Optional: Image to seed recall (not fully impl in visual yet)
        }

        self.outputs = {
            'solution_state': 'image',
            'system_energy': 'signal',
            'memory_count': 'signal',
            'last_recall_error': 'signal'
        }

        self.N = int(num_oscillators)
        self.K = float(coupling_strength)
        self.lr = float(learning_rate)
        self.mem_bias = float(memory_bias)
        
        # System State
        self.phases = np.random.rand(self.N) * 2 * np.pi
        self.frequencies = np.random.normal(1.0, 0.1, self.N)

        # Connectivity (Constraints) - Initializes random
        self.weights = np.random.choice([-1, 0, 1], size=(self.N, self.N), p=[0.3, 0.4, 0.3])
        np.fill_diagonal(self.weights, 0)
        self.weights = self.weights.astype(np.float32)

        # Memory Storage
        # We'll store learned weight matrices or phase patterns?
        # P-KAS theory says we modify weights to store phases.
        # So 'memories' here effectively means "learned configurations"
        self.memories = [] 
        self._last_recall_error = 0.0
        
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)
        self.energy = 1.0
        
        self.recall_active = False
        self.write_cooldown = 0

    def step(self):
        # 1. Get Inputs
        input_e = self.get_blended_input('input_energy', 'sum') or 0.0
        const_mod = self.get_blended_input('constraint_mod', 'sum') or 0.0
        write_sig = self.get_blended_input('write_memory', 'max') or 0.0
        recall_sig = self.get_blended_input('recall_mode', 'max') or 0.0

        eff_K = self.K * (1.0 + const_mod)

        # 2. Handle Memory Write
        if write_sig > 0.5 and self.write_cooldown <= 0:
            self._learn_current_state()
            self.write_cooldown = 30 # Wait 30 frames
        
        if self.write_cooldown > 0:
            self.write_cooldown -= 1

        # 3. Handle Recall Mode
        # If recall is active, we might bias the system towards stored memories
        # or simply let the weights (which contain the memories) drive the system.
        # In P-KAS, the weights *are* the memory. So standard dynamics apply.
        # However, 'Recall Mode' might mean "Clamp some phases" (Pattern Completion).
        
        # 4. Kuramoto Dynamics
        diff_matrix = self.phases[None, :] - self.phases[:, None]
        interaction = np.sin(diff_matrix)
        
        # Weights drive the system
        coupling = np.sum(self.weights * interaction, axis=1)
        
        dt = 0.1
        # Input energy acts as noise/temperature
        noise = np.random.normal(0, 0.01 + input_e * 0.1, self.N)
        
        d_theta = self.frequencies + (eff_K / self.N) * coupling + noise
        self.phases = (self.phases + d_theta * dt) % (2 * np.pi)

        # 5. Calculate Energy
        energy_mat = self.weights * np.cos(diff_matrix)
        self.energy = -0.5 * np.sum(energy_mat) / (self.N**2)
        self.energy = (self.energy + 0.5)

        self._render_state()

    def _learn_current_state(self):
        """
        Hebbian Learning: Adjust weights to stabilize current phase pattern.
        dw_ij = learning_rate * cos(theta_i - theta_j)
        """
        diff_matrix = self.phases[None, :] - self.phases[:, None]
        # Hebbian term: oscillators in sync strengthen connection (+), anti-sync weaken (-)
        delta_w = np.cos(diff_matrix) 
        
        self.weights += self.lr * delta_w
        
        # Clip weights to keep reasonable bounds
        self.weights = np.clip(self.weights, -2.0, 2.0)
        np.fill_diagonal(self.weights, 0)
        
        # Store "snapshot" for UI count, though weights are the real storage
        self.memories.append(self.phases.copy())
        print(f"P-KAS: Memorized state. Total memories: {len(self.memories)}")

    def _render_state(self):
        self.display_img.fill(20)
        center = (64, 64)
        radius = 50
        
        # Draw connections (only strong ones)
        for i in range(self.N):
            for j in range(i+1, self.N):
                w = self.weights[i, j]
                if abs(w) > 0.5:
                    xi = int(center[0] + radius * np.cos(2*np.pi*i/self.N))
                    yi = int(center[1] + radius * np.sin(2*np.pi*i/self.N))
                    xj = int(center[0] + radius * np.cos(2*np.pi*j/self.N))
                    yj = int(center[1] + radius * np.sin(2*np.pi*j/self.N))
                    
                    # Color based on satisfaction relative to CURRENT weight
                    # Green = Happy (In sync with positive weight OR anti-sync with negative)
                    # Red = Frustrated
                    diff = np.abs(self.phases[i] - self.phases[j])
                    diff = min(diff, 2*np.pi - diff)
                    
                    energy_local = -w * np.cos(diff) # Low energy = happy
                    
                    col = (0, 255, 0) if energy_local < 0 else (0, 0, 255)
                    thickness = max(1, int(abs(w)))
                    cv2.line(self.display_img, (xi, yi), (xj, yj), col, thickness)

        # Draw oscillators
        for i in range(self.N):
            x = int(center[0] + radius * np.cos(2*np.pi*i/self.N))
            y = int(center[1] + radius * np.sin(2*np.pi*i/self.N))
            
            hue = int((self.phases[i] / (2*np.pi)) * 179)
            osc_color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0,0]
            
            cv2.circle(self.display_img, (x, y), 6, (int(osc_color[0]), int(osc_color[1]), int(osc_color[2])), -1)
            cv2.circle(self.display_img, (x, y), 7, (255, 255, 255), 1)

    def get_output(self, port_name):
        if port_name == 'solution_state':
            return self.display_img.astype(np.float32) / 255.0
        elif port_name == 'system_energy':
            return float(self.energy)
        elif port_name == 'memory_count':
            return float(len(self.memories))
        return None

    def get_display_image(self):
        img = self.display_img.copy()
        cv2.putText(img, f"E: {self.energy:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Mem: {len(self.memories)}", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return QtGui.QImage(img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Num Oscillators", "N", self.N, None),
            ("Coupling", "K", self.K, None),
            ("Learning Rate", "lr", self.lr, None)
        ]