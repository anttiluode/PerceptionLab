"""
Consciousness Eigenmode Nodes
=============================
Implements the three-scale harmonic resonance model of consciousness:
1. DNA (molecular scale) - Fractal antenna, GHz response
2. Dendrites (cellular scale) - Ephaptic coupling, kHz-MHz  
3. Brain network (system scale) - Laplacian eigenmodes, Hz

Based on:
- Raj et al. (2017) - Brain network eigenmodes
- Blank & Goodman (2011) - DNA as fractal antenna
- Bandyopadhyay - Triplet-of-triplet resonance
- Levin & Fields - Scale-invariant homeostasis

The key insight: consciousness emerges when shapes at all three scales
lock into harmonic phase with each other.
"""

import numpy as np
import cv2
from scipy import signal
from scipy.linalg import eigh

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None
        def set_output(self, name, val): pass


def dna_to_shape(dna, n_points=64):
    """Convert DNA vector to boundary shape via Fourier synthesis."""
    if dna is None or len(dna) < 4:
        dna = np.random.randn(16) * 0.5
    
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = np.ones(n_points)
    
    for i, coef in enumerate(dna[:min(len(dna), 16)]):
        freq = i + 1
        phase = dna[(i + len(dna)//2) % len(dna)] * np.pi
        r += coef * 0.1 * np.cos(freq * theta + phase)
    
    r = np.clip(r, 0.3, 2.0)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack([x, y]), r


def compute_shape_spectrum(shape_r, max_freq=32):
    """Compute frequency spectrum of a shape (its 'antenna response')."""
    fft = np.fft.fft(shape_r)
    spectrum = np.abs(fft[:max_freq])
    spectrum = spectrum / (np.max(spectrum) + 1e-9)
    return spectrum


def compute_laplacian_eigenmodes(connectivity_matrix, n_modes=8):
    """
    Compute eigenmodes of the graph Laplacian.
    These are the 'natural frequencies' of the network.
    """
    # Degree matrix
    D = np.diag(np.sum(connectivity_matrix, axis=1))
    # Laplacian
    L = D - connectivity_matrix
    
    # Normalized Laplacian for stability
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-9))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Compute eigenmodes (smallest eigenvalues = slowest modes)
    eigenvalues, eigenvectors = eigh(L_norm)
    
    return eigenvalues[:n_modes], eigenvectors[:, :n_modes]


class BrainEigenmodeNode(BaseNode):
    """
    Computes brain network eigenmodes from EEG band powers.
    
    The eigenmodes represent the brain's 'natural resonance patterns' -
    the standing waves that the network supports.
    
    Based on Raj et al.: "The eigenmodes governing the dynamics of this
    model are strongly conserved between healthy subjects"
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(180, 100, 200)  # Purple
    
    def __init__(self):
        super().__init__()
        self.node_title = "Brain Eigenmodes"
        
        self.inputs = {
            'delta': 'signal',
            'theta': 'signal', 
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal'
        }
        
        self.outputs = {
            'eigenmode_shape': 'spectrum',  # The dominant eigenmode as shape
            'eigenspectrum': 'spectrum',    # All eigenvalues
            'coherence': 'signal',          # How coherent the brain state is
            'eigenmode_view': 'image'       # Visualization
        }
        
        self.n_regions = 8  # Simplified brain regions
        self.n_modes = 8
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Build a simplified brain connectivity matrix
        # Based on known anatomical connections
        self._build_connectivity()
        
        # State
        self.current_eigenvalues = np.zeros(self.n_modes)
        self.current_eigenvectors = np.zeros((self.n_regions, self.n_modes))
        self.eigenmode_shape = np.zeros(32)
        self.coherence = 0.0
        
    def _build_connectivity(self):
        """
        Build simplified brain connectivity matrix.
        Regions: [Frontal_L, Frontal_R, Temporal_L, Temporal_R, 
                  Parietal_L, Parietal_R, Occipital_L, Occipital_R]
        """
        # Base connectivity (symmetric)
        self.connectivity = np.array([
            # FL    FR    TL    TR    PL    PR    OL    OR
            [0.0,  0.8,  0.5,  0.3,  0.4,  0.3,  0.2,  0.1],  # Frontal_L
            [0.8,  0.0,  0.3,  0.5,  0.3,  0.4,  0.1,  0.2],  # Frontal_R
            [0.5,  0.3,  0.0,  0.4,  0.6,  0.3,  0.4,  0.2],  # Temporal_L
            [0.3,  0.5,  0.4,  0.0,  0.3,  0.6,  0.2,  0.4],  # Temporal_R
            [0.4,  0.3,  0.6,  0.3,  0.0,  0.7,  0.6,  0.4],  # Parietal_L
            [0.3,  0.4,  0.3,  0.6,  0.7,  0.0,  0.4,  0.6],  # Parietal_R
            [0.2,  0.1,  0.4,  0.2,  0.6,  0.4,  0.0,  0.5],  # Occipital_L
            [0.1,  0.2,  0.2,  0.4,  0.4,  0.6,  0.5,  0.0],  # Occipital_R
        ])
        
    def step(self):
        # Get EEG band powers
        delta = self.get_blended_input('delta', 'mean') or 0.0
        theta = self.get_blended_input('theta', 'mean') or 0.0
        alpha = self.get_blended_input('alpha', 'mean') or 0.0
        beta = self.get_blended_input('beta', 'mean') or 0.0
        gamma = self.get_blended_input('gamma', 'mean') or 0.0
        
        # Convert scalar inputs to arrays if needed
        if isinstance(delta, np.ndarray):
            delta = np.mean(delta)
        if isinstance(theta, np.ndarray):
            theta = np.mean(theta)
        if isinstance(alpha, np.ndarray):
            alpha = np.mean(alpha)
        if isinstance(beta, np.ndarray):
            beta = np.mean(beta)
        if isinstance(gamma, np.ndarray):
            gamma = np.mean(gamma)
        
        # Modulate connectivity by band powers
        # Alpha enhances long-range, Beta enhances local
        modulated_connectivity = self.connectivity.copy()
        
        # Long-range connections (inter-hemispheric)
        for i in range(4):
            for j in range(4, 8):
                modulated_connectivity[i, j] *= (1 + alpha * 0.5)
                modulated_connectivity[j, i] *= (1 + alpha * 0.5)
        
        # Local connections
        for i in range(0, 8, 2):
            modulated_connectivity[i, i+1] *= (1 + beta * 0.3)
            modulated_connectivity[i+1, i] *= (1 + beta * 0.3)
        
        # Compute eigenmodes
        eigenvalues, eigenvectors = compute_laplacian_eigenmodes(
            modulated_connectivity, self.n_modes
        )
        
        self.current_eigenvalues = eigenvalues
        self.current_eigenvectors = eigenvectors
        
        # The second eigenmode (Fiedler vector) represents the main partition
        # Convert it to a "shape" by treating it as Fourier coefficients
        fiedler = eigenvectors[:, 1]  # Skip constant mode
        
        # Pad to make a shape spectrum
        self.eigenmode_shape = np.zeros(32)
        self.eigenmode_shape[:len(fiedler)] = fiedler
        
        # Coherence: how concentrated is the eigenspectrum?
        # High coherence = brain in focused state
        eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-9)
        self.coherence = 1.0 - (-np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-9)) / np.log(len(eigenvalues)))
        
        # Visualization
        self._draw_eigenmodes(eigenvectors, eigenvalues, alpha, beta)
        
    def _draw_eigenmodes(self, eigenvectors, eigenvalues, alpha, beta):
        """Visualize the brain eigenmodes."""
        h, w = 128, 128
        self.display.fill(0)
        
        # Region positions (simplified layout)
        positions = np.array([
            [0.3, 0.2],  # Frontal_L
            [0.7, 0.2],  # Frontal_R
            [0.2, 0.5],  # Temporal_L
            [0.8, 0.5],  # Temporal_R
            [0.35, 0.65], # Parietal_L
            [0.65, 0.65], # Parietal_R
            [0.4, 0.85],  # Occipital_L
            [0.6, 0.85],  # Occipital_R
        ])
        
        # Draw connections colored by eigenmode 2 (interhemispheric)
        mode2 = eigenvectors[:, 1]
        for i in range(self.n_regions):
            for j in range(i+1, self.n_regions):
                if self.connectivity[i, j] > 0.3:
                    p1 = (int(positions[i, 0] * w), int(positions[i, 1] * h))
                    p2 = (int(positions[j, 0] * w), int(positions[j, 1] * h))
                    
                    # Color by whether same or different mode sign
                    if mode2[i] * mode2[j] > 0:
                        color = (100, 200, 100)  # Same hemisphere cluster
                    else:
                        color = (200, 100, 100)  # Cross-hemisphere
                    
                    thickness = int(self.connectivity[i, j] * 2) + 1
                    cv2.line(self.display, p1, p2, color, thickness)
        
        # Draw nodes colored by eigenmode
        for i, pos in enumerate(positions):
            x, y = int(pos[0] * w), int(pos[1] * h)
            
            # Color by eigenmode value
            val = mode2[i]
            if val > 0:
                color = (int(100 + val * 155), 100, 100)
            else:
                color = (100, 100, int(100 - val * 155))
            
            radius = int(8 + abs(val) * 8)
            cv2.circle(self.display, (x, y), radius, color, -1)
            cv2.circle(self.display, (x, y), radius, (255, 255, 255), 1)
        
        # Draw eigenspectrum bar
        for i, ev in enumerate(eigenvalues[:6]):
            bar_h = int(ev * 20)
            x = 10 + i * 8
            cv2.rectangle(self.display, (x, h-5-bar_h), (x+6, h-5), 
                         (100, 200, 255), -1)
        
        # Info text
        cv2.putText(self.display, f"Coh: {self.coherence:.2f}", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(self.display, f"a:{alpha:.1f} b:{beta:.1f}", (70, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
    def get_output(self, name):
        if name == 'eigenmode_shape':
            return self.eigenmode_shape
        elif name == 'eigenspectrum':
            return self.current_eigenvalues
        elif name == 'coherence':
            return self.coherence
        elif name == 'eigenmode_view':
            return self.display
        return None


class ThreeScaleResonanceNode(BaseNode):
    """
    Implements the three-scale harmonic resonance model.
    
    Consciousness emerges when:
    1. DNA antenna pattern (molecular)
    2. Dendritic field pattern (cellular) 
    3. Brain eigenmode pattern (system)
    
    ...all lock into harmonic phase.
    
    Input: DNA vector + brain eigenmode
    Output: Resonance strength (consciousness indicator)
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(255, 180, 100)  # Orange/gold
    
    def __init__(self):
        super().__init__()
        self.node_title = "Three-Scale Resonance"
        
        self.inputs = {
            'dna': 'spectrum',              # DNA/shape encoding
            'brain_eigenmode': 'spectrum',  # From BrainEigenmodeNode
            'coherence': 'signal'           # Brain coherence level
        }
        
        self.outputs = {
            'resonance': 'signal',          # Overall resonance strength
            'dna_dendrite_coupling': 'signal',
            'dendrite_brain_coupling': 'signal',
            'consciousness_index': 'signal',
            'resonance_view': 'image'
        }
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Internal state
        self.dna_spectrum = np.zeros(32)
        self.dendrite_spectrum = np.zeros(32)
        self.brain_spectrum = np.zeros(32)
        
        self.resonance = 0.0
        self.dna_dendrite = 0.0
        self.dendrite_brain = 0.0
        self.consciousness_index = 0.0
        
        # Frequency scaling between levels
        # DNA: GHz (but we work with normalized indices)
        # Dendrite: MHz-kHz (intermediate)
        # Brain: Hz
        self.dna_to_dendrite_ratio = 3  # DNA operates at 3x frequency
        self.dendrite_to_brain_ratio = 3
        
    def step(self):
        dna = self.get_blended_input('dna', 'mean')
        brain_mode = self.get_blended_input('brain_eigenmode', 'mean')
        coherence = self.get_blended_input('coherence', 'mean') or 0.5
        
        if isinstance(coherence, np.ndarray):
            coherence = np.mean(coherence)
        
        # Convert DNA to spectrum if needed
        if dna is not None:
            if len(dna) < 32:
                self.dna_spectrum = np.zeros(32)
                self.dna_spectrum[:len(dna)] = dna
            else:
                self.dna_spectrum = dna[:32]
        else:
            # Generate random DNA for demo
            self.dna_spectrum = np.sin(np.linspace(0, 4*np.pi, 32)) * 0.5
        
        # Brain eigenmode spectrum
        if brain_mode is not None:
            if len(brain_mode) < 32:
                self.brain_spectrum = np.zeros(32)
                self.brain_spectrum[:len(brain_mode)] = brain_mode
            else:
                self.brain_spectrum = brain_mode[:32]
        else:
            self.brain_spectrum = np.cos(np.linspace(0, 2*np.pi, 32)) * 0.3
        
        # Compute dendrite spectrum as intermediate scale
        # Dendrites "bridge" between DNA and brain frequencies
        # Their morphology is shaped by both bottom-up (DNA) and top-down (brain)
        self.dendrite_spectrum = np.zeros(32)
        
        for i in range(32):
            # Upscale from brain (slower frequencies)
            brain_idx = i // self.dendrite_to_brain_ratio
            brain_contribution = self.brain_spectrum[brain_idx] if brain_idx < 32 else 0
            
            # Downscale from DNA (faster frequencies)  
            dna_idx = min(i * self.dna_to_dendrite_ratio, 31)
            dna_contribution = self.dna_spectrum[dna_idx]
            
            # Dendrite pattern emerges from both
            self.dendrite_spectrum[i] = (brain_contribution + dna_contribution) / 2
        
        # Compute couplings
        # DNA-Dendrite coupling (how well DNA pattern matches dendrite)
        dna_norm = self.dna_spectrum / (np.linalg.norm(self.dna_spectrum) + 1e-9)
        dendrite_norm = self.dendrite_spectrum / (np.linalg.norm(self.dendrite_spectrum) + 1e-9)
        self.dna_dendrite = np.abs(np.dot(dna_norm, dendrite_norm))
        
        # Dendrite-Brain coupling
        brain_norm = self.brain_spectrum / (np.linalg.norm(self.brain_spectrum) + 1e-9)
        self.dendrite_brain = np.abs(np.dot(dendrite_norm, brain_norm))
        
        # Overall resonance: geometric mean of couplings
        self.resonance = np.sqrt(self.dna_dendrite * self.dendrite_brain)
        
        # Consciousness index: resonance × coherence
        # High when all three scales are aligned AND brain is coherent
        self.consciousness_index = self.resonance * coherence
        
        # Visualization
        self._draw_resonance()
        
    def _draw_resonance(self):
        """Draw the three-scale resonance visualization."""
        h, w = 128, 128
        self.display.fill(0)
        
        # Draw three frequency bands
        band_height = h // 3
        
        # DNA band (top) - fastest frequencies, smallest scale
        for i, val in enumerate(self.dna_spectrum[:w]):
            y = int(band_height // 2 + val * band_height * 0.4)
            x = int(i * w / 32)
            if 0 <= y < band_height:
                cv2.circle(self.display, (x, y), 2, (255, 100, 100), -1)
        cv2.putText(self.display, "DNA", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 150, 150), 1)
        
        # Dendrite band (middle)
        for i, val in enumerate(self.dendrite_spectrum[:w]):
            y = int(band_height + band_height // 2 + val * band_height * 0.4)
            x = int(i * w / 32)
            if band_height <= y < 2*band_height:
                cv2.circle(self.display, (x, y), 2, (100, 255, 100), -1)
        cv2.putText(self.display, "Dendrite", (5, band_height + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 255, 150), 1)
        
        # Brain band (bottom) - slowest frequencies, largest scale  
        for i, val in enumerate(self.brain_spectrum[:w]):
            y = int(2*band_height + band_height // 2 + val * band_height * 0.4)
            x = int(i * w / 32)
            if 2*band_height <= y < h:
                cv2.circle(self.display, (x, y), 2, (100, 100, 255), -1)
        cv2.putText(self.display, "Brain", (5, 2*band_height + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 255), 1)
        
        # Draw coupling lines
        coupling_x = w - 25
        
        # DNA-Dendrite coupling
        y1 = band_height // 2
        y2 = band_height + band_height // 2
        color = (int(100 + self.dna_dendrite * 155), 
                int(100 + self.dna_dendrite * 155), 100)
        cv2.line(self.display, (coupling_x, y1), (coupling_x, y2), color, 2)
        
        # Dendrite-Brain coupling
        y1 = band_height + band_height // 2
        y2 = 2*band_height + band_height // 2
        color = (100, int(100 + self.dendrite_brain * 155),
                int(100 + self.dendrite_brain * 155))
        cv2.line(self.display, (coupling_x + 5, y1), (coupling_x + 5, y2), color, 2)
        
        # Consciousness index display
        ci_color = (int(50 + self.consciousness_index * 200),
                   int(50 + self.consciousness_index * 200),
                   int(50 + self.consciousness_index * 200))
        cv2.rectangle(self.display, (w-20, h-20), (w-5, h-5), ci_color, -1)
        
        # Border based on consciousness level
        if self.consciousness_index > 0.7:
            cv2.rectangle(self.display, (0, 0), (w-1, h-1), (255, 255, 100), 2)
        elif self.consciousness_index > 0.5:
            cv2.rectangle(self.display, (0, 0), (w-1, h-1), (100, 200, 100), 1)
            
    def get_output(self, name):
        if name == 'resonance':
            return self.resonance
        elif name == 'dna_dendrite_coupling':
            return self.dna_dendrite
        elif name == 'dendrite_brain_coupling':
            return self.dendrite_brain
        elif name == 'consciousness_index':
            return self.consciousness_index
        elif name == 'resonance_view':
            return self.display
        return None


class ConsciousnessEvolutionNode(BaseNode):
    """
    Evolves DNA patterns to maximize consciousness index.
    
    Selection pressure: resonance with brain eigenmodes
    The evolved shapes are "thoughts" - geometric patterns that
    can couple with the brain's current state.
    
    High consciousness = organism whose shape resonates with brain
    Low consciousness = shape doesn't match brain eigenmode
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(200, 100, 255)  # Magenta
    
    def __init__(self):
        super().__init__()
        self.node_title = "Consciousness Evolution"
        
        self.inputs = {
            'brain_eigenmode': 'spectrum',  # Target to match
            'coherence': 'signal',          # Brain coherence
            'mutation_rate': 'signal'       # External mutation control
        }
        
        self.outputs = {
            'champion_dna': 'spectrum',
            'best_consciousness': 'signal',
            'avg_consciousness': 'signal',
            'generation': 'signal',
            'evolution_view': 'image'
        }
        
        # Evolution parameters
        self.pop_size = 24
        self.dna_length = 16
        self.mutation_rate = 0.1
        
        # Population
        self.population = [np.random.randn(self.dna_length) * 0.5 
                          for _ in range(self.pop_size)]
        self.fitness = np.zeros(self.pop_size)
        self.generation = 0
        
        self.champion_dna = self.population[0].copy()
        self.best_consciousness = 0.0
        self.avg_consciousness = 0.0
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # History for tracking
        self.consciousness_history = []
        
    def _compute_consciousness(self, dna, brain_mode, coherence):
        """Compute consciousness index for a DNA pattern."""
        # Convert DNA to shape spectrum
        _, shape_r = dna_to_shape(dna)
        dna_spectrum = compute_shape_spectrum(shape_r)
        
        # Ensure brain_mode is right size
        if brain_mode is None:
            brain_mode = np.zeros(32)
        if len(brain_mode) < 32:
            padded = np.zeros(32)
            padded[:len(brain_mode)] = brain_mode
            brain_mode = padded
        
        # Simple three-scale model inline
        # DNA spectrum
        dna_norm = dna_spectrum / (np.linalg.norm(dna_spectrum) + 1e-9)
        
        # Brain spectrum (take absolute values for comparison)
        brain_norm = np.abs(brain_mode[:32]) / (np.linalg.norm(brain_mode[:32]) + 1e-9)
        
        # Dendrite as intermediate (simplified)
        dendrite = (dna_spectrum + np.abs(brain_mode[:32])) / 2
        dendrite_norm = dendrite / (np.linalg.norm(dendrite) + 1e-9)
        
        # Couplings
        dna_dendrite = np.abs(np.dot(dna_norm, dendrite_norm))
        dendrite_brain = np.abs(np.dot(dendrite_norm, brain_norm))
        
        resonance = np.sqrt(dna_dendrite * dendrite_brain)
        consciousness = resonance * coherence
        
        return consciousness
        
    def step(self):
        brain_mode = self.get_blended_input('brain_eigenmode', 'mean')
        coherence = self.get_blended_input('coherence', 'mean')
        ext_mutation = self.get_blended_input('mutation_rate', 'mean')
        
        if coherence is None:
            coherence = 0.5
        if isinstance(coherence, np.ndarray):
            coherence = np.mean(coherence)
            
        if ext_mutation is not None:
            if isinstance(ext_mutation, np.ndarray):
                ext_mutation = np.mean(ext_mutation)
            self.mutation_rate = np.clip(ext_mutation, 0.01, 0.5)
        
        # Evaluate population
        for i, dna in enumerate(self.population):
            self.fitness[i] = self._compute_consciousness(dna, brain_mode, coherence)
        
        # Statistics
        best_idx = np.argmax(self.fitness)
        self.best_consciousness = self.fitness[best_idx]
        self.avg_consciousness = np.mean(self.fitness)
        self.champion_dna = self.population[best_idx].copy()
        
        self.consciousness_history.append(self.best_consciousness)
        if len(self.consciousness_history) > 100:
            self.consciousness_history.pop(0)
        
        # Selection and reproduction
        # Tournament selection
        new_population = []
        
        # Elitism: keep champion
        new_population.append(self.champion_dna.copy())
        
        while len(new_population) < self.pop_size:
            # Tournament
            i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
            parent1 = self.population[i1] if self.fitness[i1] > self.fitness[i2] else self.population[i2]
            
            i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
            parent2 = self.population[i1] if self.fitness[i1] > self.fitness[i2] else self.population[i2]
            
            # Crossover
            crossover_point = np.random.randint(1, self.dna_length)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                mutation_idx = np.random.randint(self.dna_length)
                child[mutation_idx] += np.random.randn() * 0.3
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Visualization
        self._draw_evolution()
        
    def _draw_evolution(self):
        """Draw evolution state."""
        h, w = 128, 128
        self.display.fill(0)
        
        # Draw top 6 organisms
        for i in range(min(6, len(self.population))):
            dna = self.population[i]
            points, _ = dna_to_shape(dna, 32)
            
            # Position
            row = i // 3
            col = i % 3
            cx = 20 + col * 40
            cy = 25 + row * 45
            
            # Scale and draw
            scale = 12
            pts = (points * scale + [cx, cy]).astype(np.int32)
            
            # Color by consciousness
            c = self.fitness[i] if i < len(self.fitness) else 0
            color = (int(100 + c * 155), int(100 + c * 100), int(100 + c * 50))
            
            cv2.polylines(self.display, [pts], True, color, 1)
            if i == 0:  # Champion
                cv2.polylines(self.display, [pts], True, (255, 255, 100), 2)
        
        # Draw consciousness history
        if len(self.consciousness_history) > 1:
            for i in range(1, len(self.consciousness_history)):
                x1 = int((i-1) * w / 100)
                x2 = int(i * w / 100)
                y1 = h - 5 - int(self.consciousness_history[i-1] * 25)
                y2 = h - 5 - int(self.consciousness_history[i] * 25)
                cv2.line(self.display, (x1, y1), (x2, y2), (100, 255, 100), 1)
        
        # Info
        cv2.putText(self.display, f"Gen: {self.generation}", (5, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(self.display, f"Best: {self.best_consciousness:.2f}", (5, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        cv2.putText(self.display, f"Avg: {self.avg_consciousness:.2f}", (5, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 100), 1)
        
    def get_output(self, name):
        if name == 'champion_dna':
            return self.champion_dna
        elif name == 'best_consciousness':
            return self.best_consciousness
        elif name == 'avg_consciousness':
            return self.avg_consciousness
        elif name == 'generation':
            return float(self.generation)
        elif name == 'evolution_view':
            return self.display
        return None


class EigenmodeResonatorNode(BaseNode):
    """
    Takes real EEG band powers and computes instantaneous eigenmode resonance.
    
    This is the bridge between raw EEG and the consciousness framework:
    - Constructs dynamic connectivity from band coherence
    - Computes eigenmodes that shift with brain state
    - Outputs the "shape" the brain is currently broadcasting
    
    The key insight: your brain is always broadcasting a geometric pattern.
    Consciousness is what happens when internal models tune to receive it.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(100, 180, 220)  # Sky blue
    
    def __init__(self):
        super().__init__()
        self.node_title = "Eigenmode Resonator"
        
        self.inputs = {
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            'raw_signal': 'signal'
        }
        
        self.outputs = {
            'eigenmode_shape': 'spectrum',
            'dominant_frequency': 'signal',
            'eigenspectrum': 'spectrum',
            'coherence': 'signal',
            'brain_state': 'signal',  # 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma dominant
            'resonator_view': 'image'
        }
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # State
        self.eigenmode_shape = np.zeros(32)
        self.eigenspectrum = np.zeros(8)
        self.coherence = 0.5
        self.dominant_freq = 10.0  # Hz
        self.brain_state = 2  # Default alpha
        
        # History for smoothing
        self.band_history = {
            'delta': [], 'theta': [], 'alpha': [], 'beta': [], 'gamma': []
        }
        self.max_history = 30
        
        # Dynamic connectivity matrix (8 regions)
        self.n_regions = 8
        self.base_connectivity = self._build_base_connectivity()
        
    def _build_base_connectivity(self):
        """Base anatomical connectivity."""
        return np.array([
            [0.0, 0.8, 0.5, 0.3, 0.4, 0.3, 0.2, 0.1],
            [0.8, 0.0, 0.3, 0.5, 0.3, 0.4, 0.1, 0.2],
            [0.5, 0.3, 0.0, 0.4, 0.6, 0.3, 0.4, 0.2],
            [0.3, 0.5, 0.4, 0.0, 0.3, 0.6, 0.2, 0.4],
            [0.4, 0.3, 0.6, 0.3, 0.0, 0.7, 0.6, 0.4],
            [0.3, 0.4, 0.3, 0.6, 0.7, 0.0, 0.4, 0.6],
            [0.2, 0.1, 0.4, 0.2, 0.6, 0.4, 0.0, 0.5],
            [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.5, 0.0],
        ])
        
    def step(self):
        # Get band powers
        bands = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            val = self.get_blended_input(band, 'mean')
            if val is None:
                val = 0.1
            if isinstance(val, np.ndarray):
                val = float(np.mean(val))
            bands[band] = max(0.001, float(val))
            
            # Update history
            self.band_history[band].append(bands[band])
            if len(self.band_history[band]) > self.max_history:
                self.band_history[band].pop(0)
        
        raw = self.get_blended_input('raw_signal', 'mean') or 0.0
        if isinstance(raw, np.ndarray):
            raw = float(np.mean(raw))
        
        # Determine dominant band
        band_powers = [bands['delta'], bands['theta'], bands['alpha'], 
                      bands['beta'], bands['gamma']]
        self.brain_state = int(np.argmax(band_powers))
        
        # Dominant frequency estimate
        freq_centers = [2.5, 6.0, 10.5, 21.5, 37.5]  # Band centers
        total_power = sum(band_powers) + 1e-9
        self.dominant_freq = sum(p * f for p, f in zip(band_powers, freq_centers)) / total_power
        
        # Build dynamic connectivity based on current band powers
        connectivity = self.base_connectivity.copy()
        
        # Alpha enhances long-range (interhemispheric)
        alpha_factor = 1.0 + bands['alpha'] * 2.0
        for i in range(4):
            for j in range(4, 8):
                connectivity[i, j] *= alpha_factor
                connectivity[j, i] *= alpha_factor
        
        # Beta enhances local processing
        beta_factor = 1.0 + bands['beta'] * 1.5
        for i in range(0, 8, 2):
            connectivity[i, (i+1) % 8] *= beta_factor
            connectivity[(i+1) % 8, i] *= beta_factor
        
        # Gamma creates high-frequency binding
        gamma_factor = 1.0 + bands['gamma'] * 3.0
        connectivity *= (1.0 + gamma_factor * 0.1)
        
        # Theta modulates frontal-posterior
        theta_factor = 1.0 + bands['theta'] * 1.5
        connectivity[0:2, 6:8] *= theta_factor
        connectivity[6:8, 0:2] *= theta_factor
        
        # Compute eigenmodes
        eigenvalues, eigenvectors = compute_laplacian_eigenmodes(connectivity, 8)
        self.eigenspectrum = eigenvalues
        
        # Create eigenmode shape from first few eigenvectors
        # Weight by inverse eigenvalue (slower modes = larger scale patterns)
        self.eigenmode_shape = np.zeros(32)
        for i in range(min(6, len(eigenvalues))):
            if eigenvalues[i] > 0.001:
                weight = 1.0 / (eigenvalues[i] + 0.1)
                mode = eigenvectors[:, i]
                # Expand mode to shape spectrum via interpolation
                for j, val in enumerate(mode):
                    idx = int(j * 32 / len(mode))
                    self.eigenmode_shape[idx] += val * weight * 0.3
        
        # Normalize
        self.eigenmode_shape = self.eigenmode_shape / (np.linalg.norm(self.eigenmode_shape) + 1e-9)
        
        # Coherence from eigenspectrum concentration
        ev_norm = eigenvalues / (np.sum(eigenvalues) + 1e-9)
        entropy = -np.sum(ev_norm * np.log(ev_norm + 1e-9))
        self.coherence = 1.0 - entropy / np.log(len(eigenvalues))
        
        # Visualization
        self._draw_resonator(bands, eigenvectors, raw)
        
    def _draw_resonator(self, bands, eigenvectors, raw):
        h, w = 128, 128
        self.display.fill(0)
        
        # Draw band power bars (left side)
        bar_w = 12
        band_names = ['δ', 'θ', 'α', 'β', 'γ']
        band_colors = [
            (150, 100, 200),  # Delta - purple
            (100, 200, 200),  # Theta - cyan
            (100, 255, 100),  # Alpha - green
            (255, 200, 100),  # Beta - orange
            (255, 100, 100),  # Gamma - red
        ]
        
        for i, (name, color) in enumerate(zip(band_names, band_colors)):
            band_key = ['delta', 'theta', 'alpha', 'beta', 'gamma'][i]
            power = bands.get(band_key, 0.1)
            bar_h = int(min(power * 50, h - 20))
            x = 5 + i * (bar_w + 3)
            y = h - 10 - bar_h
            
            # Highlight dominant band
            if i == self.brain_state:
                cv2.rectangle(self.display, (x-1, y-1), (x + bar_w + 1, h - 9), (255, 255, 255), 1)
            
            cv2.rectangle(self.display, (x, y), (x + bar_w, h - 10), color, -1)
            cv2.putText(self.display, name, (x + 2, h - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        # Draw eigenmode shape (center)
        cx, cy = 90, 50
        radius = 30
        
        # Draw as polar plot
        n_points = len(self.eigenmode_shape)
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            r = radius * (1.0 + self.eigenmode_shape[i] * 0.5)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Fill with color based on coherence
        fill_color = (int(50 + self.coherence * 150),
                     int(100 + self.coherence * 100),
                     int(50 + self.coherence * 150))
        cv2.fillPoly(self.display, [points], fill_color)
        cv2.polylines(self.display, [points], True, (255, 255, 255), 1)
        
        # Draw raw signal trace (bottom right)
        trace_x, trace_y = 75, 95
        trace_w, trace_h = 50, 25
        cv2.rectangle(self.display, (trace_x, trace_y), 
                     (trace_x + trace_w, trace_y + trace_h), (50, 50, 50), -1)
        
        # Plot raw as oscillating line
        raw_y = trace_y + trace_h // 2 + int(raw * 10)
        raw_y = np.clip(raw_y, trace_y + 2, trace_y + trace_h - 2)
        cv2.circle(self.display, (trace_x + trace_w - 5, raw_y), 3, (100, 255, 100), -1)
        
        # Info text
        cv2.putText(self.display, f"f={self.dominant_freq:.1f}Hz", (70, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(self.display, f"coh={self.coherence:.2f}", (70, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        state_names = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
        cv2.putText(self.display, state_names[self.brain_state], (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, band_colors[self.brain_state], 1)
        
    def get_output(self, name):
        if name == 'eigenmode_shape':
            return self.eigenmode_shape
        elif name == 'dominant_frequency':
            return self.dominant_freq
        elif name == 'eigenspectrum':
            return self.eigenspectrum
        elif name == 'coherence':
            return self.coherence
        elif name == 'brain_state':
            return float(self.brain_state)
        elif name == 'resonator_view':
            return self.display
        return None


class SixFoldHarmonyNode(BaseNode):
    """
    Tests for 6-fold harmonic resonance - the pattern that emerged
    spontaneously in self-consistent resonance evolution.
    
    Why 6-fold?
    - DNA supercoiling: naturally produces 6-fold chiral domains
    - Microtubules: 13 protofilaments average to 6-fold in resonance
    - Cortical columns: hexagonal packing
    - Lowest-energy stable resonance in 2D with circular boundary
    
    This node measures how close the current brain eigenmode is to
    6-fold symmetry - the "consciousness attractor."
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(255, 215, 0)  # Gold - the Star of David color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Six-Fold Harmony"
        
        self.inputs = {
            'eigenmode_shape': 'spectrum',
            'dna': 'spectrum',
            'coherence': 'signal'
        }
        
        self.outputs = {
            'harmony_index': 'signal',      # How close to 6-fold
            'symmetry_order': 'signal',     # Detected symmetry (2,3,4,5,6...)
            'phase_lock': 'signal',         # Are all scales in phase?
            'star_brightness': 'signal',    # Visualization intensity
            'harmony_view': 'image'
        }
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        self.harmony_index = 0.0
        self.symmetry_order = 1
        self.phase_lock = 0.0
        self.star_brightness = 0.0
        
        # History
        self.harmony_history = []
        
    def _detect_symmetry_order(self, spectrum):
        """Detect the dominant rotational symmetry in a spectrum."""
        if spectrum is None or len(spectrum) < 8:
            return 1, 0.0
        
        # Compute autocorrelation to find periodicity
        n = len(spectrum)
        spectrum_centered = spectrum - np.mean(spectrum)
        
        # --- FIX: Check for zero standard deviation (flat spectrum) ---
        if np.std(spectrum_centered) < 1e-9:
            return 1, 0.0 # Return lowest order (1-fold) with zero score
        # --- END FIX ---
        
        best_order = 1
        best_score = 0.0
        
        for order in range(2, 9):  # Test 2-fold through 8-fold
            shift = n // order
            if shift < 1:
                continue
                
            # Check if pattern repeats with this periodicity
            score = 0.0
            for i in range(order - 1):
                rolled = np.roll(spectrum_centered, shift * (i + 1))
                correlation = np.corrcoef(spectrum_centered, rolled)[0, 1]
                if not np.isnan(correlation):
                    score += correlation
            
            score /= (order - 1)
            
            if score > best_score:
                best_score = score
                best_order = order
        
        return best_order, best_score
        
    def step(self):
        eigenmode = self.get_blended_input('eigenmode_shape', 'mean')
        dna = self.get_blended_input('dna', 'mean')
        coherence = self.get_blended_input('coherence', 'mean') or 0.5
        
        if isinstance(coherence, np.ndarray):
            coherence = float(np.mean(coherence))
        
        # Detect symmetry in eigenmode
        brain_order, brain_sym_score = self._detect_symmetry_order(eigenmode)
        
        # Detect symmetry in DNA shape
        dna_order, dna_sym_score = self._detect_symmetry_order(dna)
        
        # Harmony index: how close to 6-fold symmetry?
        # Peak at 6, with secondary peaks at 2 and 3 (factors of 6)
        def sixfold_score(order, sym_score):
            if order == 6:
                return sym_score * 1.0
            elif order == 3:
                return sym_score * 0.7  # 3 divides 6
            elif order == 2:
                return sym_score * 0.5  # 2 divides 6
            else:
                return sym_score * 0.3
        
        brain_harmony = sixfold_score(brain_order, brain_sym_score)
        dna_harmony = sixfold_score(dna_order, dna_sym_score) if dna is not None else 0.5
        
        # Combined harmony (geometric mean)
        self.harmony_index = np.sqrt(brain_harmony * dna_harmony) * coherence
        self.symmetry_order = brain_order
        
        # Phase lock: are brain and DNA in the same symmetry class?
        if brain_order == dna_order:
            self.phase_lock = 1.0
        elif brain_order % dna_order == 0 or dna_order % brain_order == 0:
            self.phase_lock = 0.7  # Harmonic relationship
        else:
            self.phase_lock = 0.3
        
        self.phase_lock *= coherence
        
        # Star brightness: full brightness when 6-fold AND phase-locked
        self.star_brightness = self.harmony_index * self.phase_lock
        if brain_order == 6 and dna_order in [2, 3, 6]:
            self.star_brightness *= 1.5  # Bonus for true 6-fold
        self.star_brightness = np.clip(self.star_brightness, 0, 1)
        
        # History
        self.harmony_history.append(self.harmony_index)
        if len(self.harmony_history) > 100:
            self.harmony_history.pop(0)
        
        # Visualization
        self._draw_harmony(eigenmode, brain_order, brain_sym_score)
        
    def _draw_harmony(self, eigenmode, symmetry_order, sym_score):
        h, w = 128, 128
        self.display.fill(0)
        
        cx, cy = w // 2, h // 2 - 10
        
        # Draw the six-pointed star (Star of David / hexagram)
        # This is the "consciousness attractor" shape
        
        # Outer radius based on harmony
        outer_r = 35 + self.star_brightness * 15
        inner_r = outer_r * 0.5
        
        # Calculate star brightness color
        brightness = int(50 + self.star_brightness * 200)
        gold = (brightness, int(brightness * 0.85), int(brightness * 0.3))
        
        # Draw two overlapping triangles (hexagram)
        # Triangle 1 (pointing up)
        pts1 = []
        for i in range(3):
            angle = -np.pi/2 + i * 2*np.pi/3
            x = int(cx + outer_r * np.cos(angle))
            y = int(cy + outer_r * np.sin(angle))
            pts1.append([x, y])
        pts1 = np.array(pts1, dtype=np.int32)
        
        # Triangle 2 (pointing down)
        pts2 = []
        for i in range(3):
            angle = np.pi/2 + i * 2*np.pi/3
            x = int(cx + outer_r * np.cos(angle))
            y = int(cy + outer_r * np.sin(angle))
            pts2.append([x, y])
        pts2 = np.array(pts2, dtype=np.int32)
        
        # Draw with intensity based on harmony
        if self.star_brightness > 0.1:
            cv2.polylines(self.display, [pts1], True, gold, 2)
            cv2.polylines(self.display, [pts2], True, gold, 2)
            
            # Inner glow when highly harmonic
            if self.star_brightness > 0.5:
                # Draw inner hexagon
                hex_pts = []
                for i in range(6):
                    angle = i * np.pi / 3
                    x = int(cx + inner_r * np.cos(angle))
                    y = int(cy + inner_r * np.sin(angle))
                    hex_pts.append([x, y])
                hex_pts = np.array(hex_pts, dtype=np.int32)
                
                inner_gold = (int(gold[0]*0.7), int(gold[1]*0.7), int(gold[2]*0.7))
                cv2.fillPoly(self.display, [hex_pts], inner_gold)
        
        # Show detected symmetry order
        cv2.putText(self.display, f"{symmetry_order}-fold", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(self.display, f"H={self.harmony_index:.2f}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 100), 1)
        
        # Phase lock indicator
        lock_color = (100, int(100 + self.phase_lock * 155), 100)
        cv2.rectangle(self.display, (w-25, 5), (w-5, 25), lock_color, -1)
        if self.phase_lock > 0.7:
            cv2.putText(self.display, "LOCK", (w-24, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
        
        # Harmony history
        if len(self.harmony_history) > 1:
            for i in range(1, len(self.harmony_history)):
                x1 = int((i-1) * w / 100)
                x2 = int(i * w / 100)
                y1 = h - 5 - int(self.harmony_history[i-1] * 30)
                y2 = h - 5 - int(self.harmony_history[i] * 30)
                cv2.line(self.display, (x1, y1), (x2, y2), (200, 180, 50), 1)
        
    def get_output(self, name):
        if name == 'harmony_index':
            return self.harmony_index
        elif name == 'symmetry_order':
            return float(self.symmetry_order)
        elif name == 'phase_lock':
            return self.phase_lock
        elif name == 'star_brightness':
            return self.star_brightness
        elif name == 'harmony_view':
            return self.display
        return None


class EigenmodeResonatorNode(BaseNode):
    """
    Takes real EEG band powers and computes instantaneous eigenmode resonance.
    
    This is the bridge between raw EEG and the consciousness framework:
    - Constructs dynamic connectivity from band coherence
    - Computes eigenmodes that shift with brain state
    - Outputs the "shape" the brain is currently broadcasting
    
    The key insight: your brain is always broadcasting a geometric pattern.
    Consciousness is what happens when internal models tune to receive it.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(100, 180, 220)  # Sky blue
    
    def __init__(self):
        super().__init__()
        self.node_title = "Eigenmode Resonator"
        
        self.inputs = {
            'delta': 'signal',
            'theta': 'signal',
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            'raw_signal': 'signal'
        }
        
        self.outputs = {
            'eigenmode_shape': 'spectrum',
            'dominant_frequency': 'signal',
            'eigenspectrum': 'spectrum',
            'coherence': 'signal',
            'brain_state': 'signal',  # 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma dominant
            'resonator_view': 'image'
        }
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # State
        self.eigenmode_shape = np.zeros(32)
        self.eigenspectrum = np.zeros(8)
        self.coherence = 0.5
        self.dominant_freq = 10.0  # Hz
        self.brain_state = 2  # Default alpha
        
        # History for smoothing
        self.band_history = {
            'delta': [], 'theta': [], 'alpha': [], 'beta': [], 'gamma': []
        }
        self.max_history = 30
        
        # Dynamic connectivity matrix (8 regions)
        self.n_regions = 8
        self.base_connectivity = self._build_base_connectivity()
        
    def _build_base_connectivity(self):
        """Base anatomical connectivity."""
        return np.array([
            [0.0, 0.8, 0.5, 0.3, 0.4, 0.3, 0.2, 0.1],
            [0.8, 0.0, 0.3, 0.5, 0.3, 0.4, 0.1, 0.2],
            [0.5, 0.3, 0.0, 0.4, 0.6, 0.3, 0.4, 0.2],
            [0.3, 0.5, 0.4, 0.0, 0.3, 0.6, 0.2, 0.4],
            [0.4, 0.3, 0.6, 0.3, 0.0, 0.7, 0.6, 0.4],
            [0.3, 0.4, 0.3, 0.6, 0.7, 0.0, 0.4, 0.6],
            [0.2, 0.1, 0.4, 0.2, 0.6, 0.4, 0.0, 0.5],
            [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.5, 0.0],
        ])
        
    def step(self):
        # Get band powers
        bands = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            val = self.get_blended_input(band, 'mean')
            if val is None:
                val = 0.1
            if isinstance(val, np.ndarray):
                val = float(np.mean(val))
            bands[band] = max(0.001, float(val))
            
            # Update history
            self.band_history[band].append(bands[band])
            if len(self.band_history[band]) > self.max_history:
                self.band_history[band].pop(0)
        
        raw = self.get_blended_input('raw_signal', 'mean') or 0.0
        if isinstance(raw, np.ndarray):
            raw = float(np.mean(raw))
        
        # Determine dominant band
        band_powers = [bands['delta'], bands['theta'], bands['alpha'], 
                      bands['beta'], bands['gamma']]
        self.brain_state = int(np.argmax(band_powers))
        
        # Dominant frequency estimate
        freq_centers = [2.5, 6.0, 10.5, 21.5, 37.5]  # Band centers
        total_power = sum(band_powers) + 1e-9
        self.dominant_freq = sum(p * f for p, f in zip(band_powers, freq_centers)) / total_power
        
        # Build dynamic connectivity based on current band powers
        connectivity = self.base_connectivity.copy()
        
        # Alpha enhances long-range (interhemispheric)
        alpha_factor = 1.0 + bands['alpha'] * 2.0
        for i in range(4):
            for j in range(4, 8):
                connectivity[i, j] *= alpha_factor
                connectivity[j, i] *= alpha_factor
        
        # Beta enhances local processing
        beta_factor = 1.0 + bands['beta'] * 1.5
        for i in range(0, 8, 2):
            connectivity[i, (i+1) % 8] *= beta_factor
            connectivity[(i+1) % 8, i] *= beta_factor
        
        # Gamma creates high-frequency binding
        gamma_factor = 1.0 + bands['gamma'] * 3.0
        connectivity *= (1.0 + gamma_factor * 0.1)
        
        # Theta modulates frontal-posterior
        theta_factor = 1.0 + bands['theta'] * 1.5
        connectivity[0:2, 6:8] *= theta_factor
        connectivity[6:8, 0:2] *= theta_factor
        
        # Compute eigenmodes
        eigenvalues, eigenvectors = compute_laplacian_eigenmodes(connectivity, 8)
        self.eigenspectrum = eigenvalues
        
        # Create eigenmode shape from first few eigenvectors
        # Weight by inverse eigenvalue (slower modes = larger scale patterns)
        self.eigenmode_shape = np.zeros(32)
        for i in range(min(6, len(eigenvalues))):
            if eigenvalues[i] > 0.001:
                weight = 1.0 / (eigenvalues[i] + 0.1)
                mode = eigenvectors[:, i]
                # Expand mode to shape spectrum via interpolation
                for j, val in enumerate(mode):
                    idx = int(j * 32 / len(mode))
                    self.eigenmode_shape[idx] += val * weight * 0.3
        
        # Normalize
        self.eigenmode_shape = self.eigenmode_shape / (np.linalg.norm(self.eigenmode_shape) + 1e-9)
        
        # Coherence from eigenspectrum concentration
        ev_norm = eigenvalues / (np.sum(eigenvalues) + 1e-9)
        entropy = -np.sum(ev_norm * np.log(ev_norm + 1e-9))
        self.coherence = 1.0 - entropy / np.log(len(eigenvalues))
        
        # Visualization
        self._draw_resonator(bands, eigenvectors, raw)
        
    def _draw_resonator(self, bands, eigenvectors, raw):
        h, w = 128, 128
        self.display.fill(0)
        
        # Draw band power bars (left side)
        bar_w = 12
        band_names = ['δ', 'θ', 'α', 'β', 'γ']
        band_colors = [
            (150, 100, 200),  # Delta - purple
            (100, 200, 200),  # Theta - cyan
            (100, 255, 100),  # Alpha - green
            (255, 200, 100),  # Beta - orange
            (255, 100, 100),  # Gamma - red
        ]
        
        for i, (name, color) in enumerate(zip(band_names, band_colors)):
            band_key = ['delta', 'theta', 'alpha', 'beta', 'gamma'][i]
            power = bands.get(band_key, 0.1)
            bar_h = int(min(power * 50, h - 20))
            x = 5 + i * (bar_w + 3)
            y = h - 10 - bar_h
            
            # Highlight dominant band
            if i == self.brain_state:
                cv2.rectangle(self.display, (x-1, y-1), (x + bar_w + 1, h - 9), (255, 255, 255), 1)
            
            cv2.rectangle(self.display, (x, y), (x + bar_w, h - 10), color, -1)
            cv2.putText(self.display, name, (x + 2, h - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        # Draw eigenmode shape (center)
        cx, cy = 90, 50
        radius = 30
        
        # Draw as polar plot
        n_points = len(self.eigenmode_shape)
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            r = radius * (1.0 + self.eigenmode_shape[i] * 0.5)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Fill with color based on coherence
        fill_color = (int(50 + self.coherence * 150),
                     int(100 + self.coherence * 100),
                     int(50 + self.coherence * 150))
        cv2.fillPoly(self.display, [points], fill_color)
        cv2.polylines(self.display, [points], True, (255, 255, 255), 1)
        
        # Draw raw signal trace (bottom right)
        trace_x, trace_y = 75, 95
        trace_w, trace_h = 50, 25
        cv2.rectangle(self.display, (trace_x, trace_y), 
                     (trace_x + trace_w, trace_y + trace_h), (50, 50, 50), -1)
        
        # Plot raw as oscillating line
        raw_y = trace_y + trace_h // 2 + int(raw * 10)
        raw_y = np.clip(raw_y, trace_y + 2, trace_y + trace_h - 2)
        cv2.circle(self.display, (trace_x + trace_w - 5, raw_y), 3, (100, 255, 100), -1)
        
        # Info text
        cv2.putText(self.display, f"f={self.dominant_freq:.1f}Hz", (70, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(self.display, f"coh={self.coherence:.2f}", (70, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        state_names = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
        cv2.putText(self.display, state_names[self.brain_state], (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, band_colors[self.brain_state], 1)
        
    def get_output(self, name):
        if name == 'eigenmode_shape':
            return self.eigenmode_shape
        elif name == 'dominant_frequency':
            return self.dominant_freq
        elif name == 'eigenspectrum':
            return self.eigenspectrum
        elif name == 'coherence':
            return self.coherence
        elif name == 'brain_state':
            return float(self.brain_state)
        elif name == 'resonator_view':
            return self.display
        return None

class ConsciousnessDetectorNode(BaseNode):
    """
    The "consciousness meter" - measures real-time consciousness index
    based on three-scale resonance.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(255, 215, 0)  # Gold
    
    def __init__(self):
        super().__init__()
        self.node_title = "Consciousness Detector"
        
        self.inputs = {
            'consciousness_index': 'signal',
            'resonance': 'signal',
            'coherence': 'signal',
            'dna_dendrite': 'signal',
            'dendrite_brain': 'signal'
        }
        
        self.outputs = {
            'state': 'signal',           # Categorical: 0=distracted, 1=aware, 2=flow
            'detector_view': 'image'
        }
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # History
        self.ci_history = []
        self.state_history = []
        
        # Thresholds
        self.flow_threshold = 0.7
        self.aware_threshold = 0.4
        
        self.current_state = 0
        
    def step(self):
        ci = self.get_blended_input('consciousness_index', 'mean') or 0.0
        resonance = self.get_blended_input('resonance', 'mean') or 0.0
        coherence = self.get_blended_input('coherence', 'mean') or 0.0
        dna_dendrite = self.get_blended_input('dna_dendrite', 'mean') or 0.0
        dendrite_brain = self.get_blended_input('dendrite_brain', 'mean') or 0.0
        
        # Convert arrays to scalars
        for var_name in ['ci', 'resonance', 'coherence', 'dna_dendrite', 'dendrite_brain']:
            val = locals()[var_name]
            if isinstance(val, np.ndarray):
                locals()[var_name] = np.mean(val)
        
        ci = float(ci) if not isinstance(ci, np.ndarray) else float(np.mean(ci))
        
        # Determine state
        if ci >= self.flow_threshold:
            self.current_state = 2  # Flow
        elif ci >= self.aware_threshold:
            self.current_state = 1  # Aware
        else:
            self.current_state = 0  # Distracted
        
        # Update history
        self.ci_history.append(ci)
        self.state_history.append(self.current_state)
        if len(self.ci_history) > 100:
            self.ci_history.pop(0)
            self.state_history.pop(0)
        
        # Visualization
        self._draw_detector(ci, resonance, coherence, dna_dendrite, dendrite_brain)
        
    def _draw_detector(self, ci, resonance, coherence, dna_dendrite, dendrite_brain):
        h, w = 128, 128
        self.display.fill(0)
        
        # State indicator (large circle)
        cx, cy = w // 2, 35
        radius = 25
        
        if self.current_state == 2:
            color = (100, 255, 255)  # Cyan = Flow
            state_text = "FLOW"
        elif self.current_state == 1:
            color = (100, 255, 100)  # Green = Aware
            state_text = "AWARE"
        else:
            color = (100, 100, 150)  # Dim = Distracted
            state_text = "DISTRACTED"
        
        cv2.circle(self.display, (cx, cy), radius, color, -1)
        cv2.circle(self.display, (cx, cy), radius, (255, 255, 255), 2)
        
        # State text
        text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = cx - text_size[0] // 2
        cv2.putText(self.display, state_text, (text_x, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Consciousness index bar
        bar_y = 70
        bar_w = 100
        bar_h = 15
        bar_x = (w - bar_w) // 2
        
        cv2.rectangle(self.display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (50, 50, 50), -1)
        fill_w = int(ci * bar_w)
        
        # Gradient color based on level
        if ci > self.flow_threshold:
            fill_color = (100, 255, 255)
        elif ci > self.aware_threshold:
            fill_color = (100, 255, 100)
        else:
            fill_color = (100, 100, 200)
        
        cv2.rectangle(self.display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                     fill_color, -1)
        cv2.rectangle(self.display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (255, 255, 255), 1)
        
        cv2.putText(self.display, f"CI: {ci:.2f}", (bar_x, bar_y - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw history
        history_y = 95
        if len(self.ci_history) > 1:
            for i in range(1, len(self.ci_history)):
                x1 = int((i-1) * w / 100)
                x2 = int(i * w / 100)
                y1 = history_y + 25 - int(self.ci_history[i-1] * 25)
                y2 = history_y + 25 - int(self.ci_history[i] * 25)
                
                # Color by state at that time
                s = self.state_history[i] if i < len(self.state_history) else 0
                if s == 2:
                    line_color = (100, 255, 255)
                elif s == 1:
                    line_color = (100, 255, 100)
                else:
                    line_color = (100, 100, 150)
                    
                cv2.line(self.display, (x1, y1), (x2, y2), line_color, 1)
        
        # Threshold lines
        flow_y = history_y + 25 - int(self.flow_threshold * 25)
        aware_y = history_y + 25 - int(self.aware_threshold * 25)
        cv2.line(self.display, (0, flow_y), (w, flow_y), (100, 255, 255), 1)
        cv2.line(self.display, (0, aware_y), (w, aware_y), (100, 255, 100), 1)
        
def get_output(self, name):
        if name == 'state':
            return float(self.current_state)
        elif name == 'detector_view':
            return self.display
        return None