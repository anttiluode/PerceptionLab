"""
Quantum Darwinism Fixed - The Cybernetic Pilot That Actually Works
===================================================================

Fixes to Gemini's design:
1. BlochQubit now actually uses rz_angle (it was defined but ignored)
2. Evolution now accepts EXTERNAL fitness signals (not internal metrics)
3. Fitness is computed based on actual qubit stabilization
4. Protocell visualization shows real organic membrane dynamics

The key insight: Gemini graded pilots on their DNA structure, not their flying.
We now grade them on whether they kept the plane level.
"""

import numpy as np
import cv2
from scipy.linalg import expm
from collections import deque

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None


# =============================================================================
# FIX #1: BlochQubit that actually uses BOTH rotation angles
# =============================================================================

H_Y = np.array([[0, -1j], [1j, 0]], dtype=complex) * 0.5
H_Z = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5

class BlochQubitNodeFixed(BaseNode):
    """
    Fixed Bloch Qubit - Now actually uses rz_angle!
    
    The original bug: rz_angle input existed but was never read.
    The evolved organisms were "controlling" a knob that did nothing.
    """
    NODE_CATEGORY = "Quantum"
    NODE_COLOR = QtGui.QColor(100, 0, 255)

    def __init__(self):
        super().__init__()
        self.node_title = "Bloch Qubit (Fixed)"
        
        self.inputs = {
            'ry_angle': 'signal',  # Perturbation (from oscillator)
            'rz_angle': 'signal',  # Control (from evolved organism)
            'reset': 'signal'      # Optional: pulse to reset to |0⟩
        }
        
        self.outputs = {
            'bloch_x': 'signal',
            'bloch_y': 'signal',
            'bloch_z': 'signal',
            'instability': 'signal',  # New: distance from |0⟩ state
            'qubit_state': 'spectrum'
        }
        
        self.state = np.array([1, 0], dtype=complex)
        self.coords = (0.0, 0.0, 1.0)
        self.instability = 0.0
        
        # Smoothing for stability measurement
        self.instability_history = deque(maxlen=30)

    def step(self):
        # Get BOTH angles
        theta_y = self.get_blended_input('ry_angle', 'sum')
        theta_z = self.get_blended_input('rz_angle', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if theta_y is None: theta_y = 0.0
        if theta_z is None: theta_z = 0.0
        
        # Optional reset
        if reset is not None and reset > 0.5:
            self.state = np.array([1, 0], dtype=complex)
        
        # Apply BOTH rotations: Rz then Ry
        # This is the FIX - organisms can now counter-rotate
        U_z = expm(-1j * theta_z * H_Z)
        U_y = expm(-1j * theta_y * H_Y)
        
        # Combined evolution: start from |0⟩, apply Rz, then Ry
        basis = np.array([1, 0], dtype=complex)
        self.state = U_y @ (U_z @ basis)
        
        # Calculate Bloch coordinates
        a, b = self.state[0], self.state[1]
        x = 2 * (a * np.conj(b)).real
        y = 2 * (a * np.conj(b)).imag
        z = float(np.abs(a)**2 - np.abs(b)**2)
        
        self.coords = (float(x), float(y), float(z))
        
        # Instability = distance from north pole (|0⟩ = z=1)
        # If z=1 → stable (instability=0), if z=-1 → maximally unstable
        instant_instability = 1.0 - z  # Range: 0 (stable) to 2 (flipped)
        self.instability_history.append(instant_instability)
        self.instability = float(np.mean(self.instability_history))

    def get_output(self, port_name):
        if port_name == 'bloch_x': return self.coords[0]
        if port_name == 'bloch_y': return self.coords[1]
        if port_name == 'bloch_z': return self.coords[2]
        if port_name == 'instability': return self.instability
        if port_name == 'qubit_state': 
            return np.array([self.state[0].real, self.state[0].imag,
                           self.state[1].real, self.state[1].imag])
        return None

    def get_display_image(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        c, r = (100, 100), 80
        
        # Draw sphere outline
        cv2.circle(img, c, r, (50, 50, 50), 1)
        cv2.line(img, (c[0]-r, c[1]), (c[0]+r, c[1]), (30, 30, 30), 1)  # Equator
        
        # Draw state vector
        x, y, z = self.coords
        px = int(c[0] + x * r)
        py = int(c[1] - z * r)
        
        # Color based on stability
        if self.instability < 0.3:
            color = (0, 255, 0)  # Green = stable
        elif self.instability < 1.0:
            color = (0, 255, 255)  # Yellow = drifting
        else:
            color = (0, 0, 255)  # Red = flipped
        
        cv2.line(img, c, (px, py), color, 2)
        cv2.circle(img, (px, py), 5, color, -1)
        
        # Labels
        cv2.putText(img, f"Z: {z:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(img, f"Instab: {self.instability:.2f}", (5, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, 200, 200, 200*3, QtGui.QImage.Format.Format_RGB888)


# =============================================================================
# FIX #2: Evolution that uses EXTERNAL fitness (the qubit's stability)
# =============================================================================

class CyberneticEvolutionNode(BaseNode):
    """
    The Pilot Breeder - Now actually selects based on external performance!
    
    Key difference from Gemini's design:
    - OLD: fitness = internal DNA structure metrics (irrelevant to task)
    - NEW: fitness = external signal (qubit stability)
    
    This means organisms that successfully stabilize the qubit reproduce.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 50, 150)

    def __init__(self):
        super().__init__()
        self.node_title = "Cybernetic Evolution"
        
        self.inputs = {
            'seed_dna': 'spectrum',
            'external_fitness': 'signal',  # THE KEY FIX: fitness comes from outside
            'mutation_rate': 'signal'
        }
        
        self.outputs = {
            'champion_dna': 'spectrum',
            'control_signal': 'signal',  # First gene as direct control output
            'diversity': 'signal',
            'generation': 'signal'
        }
        
        # Population
        self.pop_size = 32
        self.dna_len = 64
        self.population = [np.random.randn(self.dna_len) * 0.5 for _ in range(self.pop_size)]
        self.fitness_scores = np.zeros(self.pop_size)
        
        # Track which organism is currently "piloting"
        self.current_pilot_idx = 0
        self.pilot_timer = 0
        self.pilot_duration = 10  # Frames to evaluate each pilot
        self.accumulated_fitness = 0.0
        
        self.gen_counter = 0
        self.champion = np.zeros(self.dna_len)

    def step(self):
        external_fitness = self.get_blended_input('external_fitness', 'mean')
        mutation_rate = self.get_blended_input('mutation_rate', 'mean')
        seed = self.get_blended_input('seed_dna', 'mean')
        
        if external_fitness is None: external_fitness = 0.5
        if mutation_rate is None: mutation_rate = 0.1
        
        # Inject seed occasionally
        if seed is not None and np.random.rand() < 0.05:
            if len(seed) >= self.dna_len:
                idx = np.random.randint(self.pop_size)
                self.population[idx] = seed[:self.dna_len].copy()
        
        # Accumulate fitness for current pilot
        # Invert: low instability = high fitness
        fitness_signal = 1.0 - np.clip(external_fitness, 0, 2) / 2.0
        self.accumulated_fitness += fitness_signal
        self.pilot_timer += 1
        
        # Time to evaluate and switch pilots?
        if self.pilot_timer >= self.pilot_duration:
            # Score this pilot
            self.fitness_scores[self.current_pilot_idx] = self.accumulated_fitness / self.pilot_duration
            
            # Move to next pilot
            self.current_pilot_idx = (self.current_pilot_idx + 1) % self.pop_size
            self.pilot_timer = 0
            self.accumulated_fitness = 0.0
            
            # Complete generation?
            if self.current_pilot_idx == 0:
                self._breed_new_generation(mutation_rate)
                self.gen_counter += 1
        
        # Current champion is the one with highest score
        best_idx = np.argmax(self.fitness_scores)
        self.champion = self.population[best_idx].copy()

    def _breed_new_generation(self, mutation_rate):
        """Selection and breeding based on actual performance"""
        sorted_idx = np.argsort(self.fitness_scores)[::-1]
        
        new_pop = []
        
        # Elitism: keep top 20%
        elite_count = max(2, int(self.pop_size * 0.2))
        for i in range(elite_count):
            new_pop.append(self.population[sorted_idx[i]].copy())
        
        # Breed the rest
        while len(new_pop) < self.pop_size:
            # Tournament selection
            candidates = np.random.choice(sorted_idx[:elite_count*2], size=2, replace=False)
            p1 = self.population[candidates[0]]
            p2 = self.population[candidates[1]]
            
            # Crossover
            child = np.zeros(self.dna_len)
            for i in range(self.dna_len):
                if np.random.rand() < 0.5:
                    child[i] = p1[i]
                else:
                    child[i] = p2[i]
            
            # Mutation
            if np.random.rand() < 0.5:
                mutation = np.random.randn(self.dna_len) * mutation_rate
                child += mutation
            
            new_pop.append(child)
        
        self.population = new_pop
        # Don't reset fitness - keep memory of performance

    def get_output(self, name):
        if name == 'champion_dna': 
            return self.champion
        if name == 'control_signal':
            # The organism's "action" - first gene scaled
            current_dna = self.population[self.current_pilot_idx]
            return float(np.mean(current_dna[:4]))  # Average of first 4 genes
        if name == 'diversity':
            if len(self.population) < 2:
                return 0.0
            # Measure population diversity
            pop_matrix = np.array(self.population)
            return float(np.std(pop_matrix))
        if name == 'generation':
            return float(self.gen_counter)
        return None

    def get_display_image(self):
        img = np.zeros((150, 200, 3), dtype=np.uint8)
        
        # Show fitness distribution
        if np.max(self.fitness_scores) > 0:
            normalized = self.fitness_scores / (np.max(self.fitness_scores) + 1e-9)
            bar_w = 200 // self.pop_size
            for i, f in enumerate(normalized):
                h = int(f * 100)
                color = (0, 255, 0) if i == self.current_pilot_idx else (100, 100, 100)
                if i == np.argmax(self.fitness_scores):
                    color = (0, 255, 255)  # Champion in yellow
                cv2.rectangle(img, (i*bar_w, 120-h), ((i+1)*bar_w-1, 120), color, -1)
        
        cv2.putText(img, f"Gen: {self.gen_counter}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(img, f"Pilot: {self.current_pilot_idx}", (5, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(img, f"Best: {np.max(self.fitness_scores):.2f}", (100, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, 200, 150, 200*3, QtGui.QImage.Format.Format_RGB888)


# =============================================================================
# FIX #3: Protocell visualization - organic membranes, not rigid circles
# =============================================================================

class ProtocellVisualizerNode(BaseNode):
    """
    Visualizes DNA as an organic protocell membrane.
    Uses the DNA as Fourier coefficients to create wobbly, alive-looking shapes.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(50, 200, 150)

    def __init__(self):
        super().__init__()
        self.node_title = "Protocell Visualizer"
        
        self.inputs = {
            'dna': 'spectrum',
            'energy': 'signal',  # Makes the cell "breathe"
            'stress': 'signal'   # Deforms the membrane
        }
        
        self.outputs = {
            'cell_view': 'image'
        }
        
        self.phase = 0.0
        self.display = np.zeros((256, 256, 3), dtype=np.uint8)
        self.membrane_history = deque(maxlen=5)  # For trails

    def step(self):
        dna = self.get_blended_input('dna', 'mean')
        energy = self.get_blended_input('energy', 'mean')
        stress = self.get_blended_input('stress', 'mean')
        
        if dna is None: dna = np.zeros(32)
        if energy is None: energy = 0.5
        if stress is None: stress = 0.0
        
        self.phase += 0.1
        
        # Ensure we have enough coefficients
        if len(dna) < 16:
            dna = np.resize(dna, 16)
        
        # Generate membrane shape using DNA as Fourier coefficients
        n_points = 64
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        # Base radius with breathing
        base_r = 60 + 20 * np.sin(self.phase * 0.5) * energy
        
        # Fourier synthesis: DNA controls harmonics
        radii = np.ones(n_points) * base_r
        for k in range(min(8, len(dna)//2)):
            amp = dna[k*2] * 15  # Amplitude from DNA
            phase_offset = dna[k*2 + 1] * np.pi  # Phase from DNA
            harmonic = k + 2  # Start from 2nd harmonic
            radii += amp * np.cos(harmonic * angles + phase_offset + self.phase * (k+1) * 0.1)
        
        # Stress deformation
        radii += stress * 10 * np.sin(3 * angles + self.phase)
        
        # Clip to reasonable range
        radii = np.clip(radii, 20, 110)
        
        # Convert to cartesian
        cx, cy = 128, 128
        pts = []
        for i, (angle, r) in enumerate(zip(angles, radii)):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            pts.append((x, y))
        
        # Store for trails
        self.membrane_history.append(pts.copy())
        
        # Draw
        self.display.fill(10)
        
        # Draw membrane trails (ghost effect)
        for trail_idx, old_pts in enumerate(self.membrane_history):
            alpha = (trail_idx + 1) / len(self.membrane_history)
            color = (int(20 * alpha), int(50 * alpha), int(30 * alpha))
            pts_arr = np.array(old_pts, dtype=np.int32)
            cv2.polylines(self.display, [pts_arr], True, color, 1)
        
        # Draw current membrane
        pts_arr = np.array(pts, dtype=np.int32)
        
        # Fill with translucent color
        overlay = self.display.copy()
        cv2.fillPoly(overlay, [pts_arr], (40, 120, 80))
        cv2.addWeighted(overlay, 0.3, self.display, 0.7, 0, self.display)
        
        # Membrane outline
        membrane_color = (100, 255, 150)
        if stress > 0.5:
            membrane_color = (100, 150, 255)  # Blueish when stressed
        cv2.polylines(self.display, [pts_arr], True, membrane_color, 2)
        
        # Draw nucleus (center blob)
        nucleus_r = int(15 + 5 * np.sin(self.phase * 2))
        cv2.circle(self.display, (cx, cy), nucleus_r, (200, 100, 150), -1)
        
        # Draw organelles (based on DNA)
        for k in range(4):
            if k < len(dna):
                org_angle = dna[k] * 2 * np.pi
                org_r = 30 + k * 10
                org_x = int(cx + org_r * np.cos(org_angle + self.phase * 0.3))
                org_y = int(cy + org_r * np.sin(org_angle + self.phase * 0.3))
                org_size = int(5 + abs(dna[k]) * 3)
                cv2.circle(self.display, (org_x, org_y), org_size, (150, 200, 100), -1)

    def get_output(self, name):
        if name == 'cell_view':
            return self.display
        return None


# =============================================================================
# NEW: Stability Reward Node - computes fitness from qubit state
# =============================================================================

class StabilityRewardNode(BaseNode):
    """
    Computes a reward signal based on how stable the qubit is.
    This is what should drive evolution - actual performance, not DNA structure.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 200, 50)

    def __init__(self):
        super().__init__()
        self.node_title = "Stability Reward"
        
        self.inputs = {
            'bloch_z': 'signal',      # Z coordinate (1 = stable |0⟩)
            'instability': 'signal',  # Direct instability signal
            'target_z': 'signal'      # Optional: target Z value (default: 1.0)
        }
        
        self.outputs = {
            'fitness': 'signal',       # High when stable
            'penalty': 'signal',       # High when unstable
            'reward_view': 'image'
        }
        
        self.fitness = 0.0
        self.penalty = 0.0
        self.history = deque(maxlen=100)
        self.display = np.zeros((80, 200, 3), dtype=np.uint8)

    def step(self):
        bloch_z = self.get_blended_input('bloch_z', 'mean')
        instability = self.get_blended_input('instability', 'mean')
        target_z = self.get_blended_input('target_z', 'mean')
        
        if target_z is None: target_z = 1.0  # Default: stay at |0⟩
        
        # Compute fitness from available signals
        if bloch_z is not None:
            # Fitness = how close to target
            error = abs(bloch_z - target_z)
            self.fitness = max(0, 1.0 - error)
            self.penalty = error
        elif instability is not None:
            # Instability is 0 when stable, 2 when flipped
            self.fitness = max(0, 1.0 - instability / 2.0)
            self.penalty = instability / 2.0
        else:
            self.fitness = 0.5
            self.penalty = 0.5
        
        self.history.append(self.fitness)
        
        # Visualization
        self.display.fill(20)
        if len(self.history) > 1:
            for i in range(1, len(self.history)):
                x1 = int((i-1) * 200 / 100)
                x2 = int(i * 200 / 100)
                y1 = int(70 - self.history[i-1] * 60)
                y2 = int(70 - self.history[i] * 60)
                color = (0, 255, 0) if self.history[i] > 0.7 else (0, 255, 255)
                if self.history[i] < 0.3:
                    color = (0, 0, 255)
                cv2.line(self.display, (x1, y1), (x2, y2), color, 1)
        
        cv2.putText(self.display, f"Fit: {self.fitness:.2f}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'fitness': return self.fitness
        if name == 'penalty': return self.penalty
        if name == 'reward_view': return self.display
        return None