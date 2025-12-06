"""
Shape Evolution: When Geometry IS the Problem
==============================================

Inspired by: COVID spike protein, antibody-antigen binding, 
receptor-ligand recognition, lock-and-key enzymes.

The insight: Shape isn't a visualization of the solution.
Shape IS the solution. Evolution finds geometries that 
interface with the environment.

Applications:
1. Evolve shapes that "bind" to target patterns (like antibodies)
2. Evolve receptors that resonate with specific signals
3. Co-evolve predator-prey / virus-antibody arms races
4. Create "neural receptors" tuned to EEG frequencies
"""

import numpy as np
import cv2
from collections import deque
from scipy import ndimage
from scipy.spatial.distance import cdist

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None


def dna_to_shape_points(dna, n_points=64, base_radius=50):
    """
    Convert DNA vector to shape boundary points using Fourier synthesis.
    This is the genotype → phenotype mapping.
    """
    if dna is None or len(dna) < 8:
        dna = np.zeros(32)
    
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radii = np.ones(n_points) * base_radius
    
    # DNA encodes Fourier coefficients
    n_harmonics = min(8, len(dna) // 2)
    for k in range(n_harmonics):
        amp = dna[k*2] * 20  # Amplitude
        phase = dna[k*2 + 1] * np.pi  # Phase
        harmonic = k + 2  # Start from 2nd harmonic (1st would just shift)
        radii += amp * np.cos(harmonic * angles + phase)
    
    radii = np.clip(radii, 10, 100)
    
    # Convert to cartesian
    cx, cy = 64, 64  # Center in 128x128 space
    points = []
    for angle, r in zip(angles, radii):
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append((x, y))
    
    return np.array(points)


def shape_to_mask(points, size=128):
    """Convert boundary points to binary mask"""
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_binding_affinity(shape1_pts, shape2_pts, size=128):
    """
    Compute how well two shapes "bind" together.
    Like antibody-antigen or enzyme-substrate.
    
    High affinity = complementary shapes that interlock
    """
    # Create masks
    mask1 = shape_to_mask(shape1_pts, size)
    mask2 = shape_to_mask(shape2_pts, size)
    
    # Compute boundary overlap potential
    # Dilate both shapes slightly
    kernel = np.ones((5, 5), np.uint8)
    dilated1 = cv2.dilate(mask1, kernel, iterations=2)
    dilated2 = cv2.dilate(mask2, kernel, iterations=2)
    
    # Contact zone = where dilated shapes overlap but original shapes don't
    contact = np.logical_and(dilated1 > 0, dilated2 > 0)
    overlap = np.logical_and(mask1 > 0, mask2 > 0)
    
    # Good binding = lots of contact, minimal overlap (they touch but don't collide)
    contact_area = np.sum(contact)
    overlap_area = np.sum(overlap)
    
    # Affinity formula: contact surface area minus collision penalty
    affinity = contact_area - 3 * overlap_area
    
    # Normalize
    max_possible = size * size * 0.1  # rough estimate
    return float(np.clip(affinity / max_possible, 0, 1))


def compute_shape_similarity(shape1_pts, shape2_pts, size=128):
    """
    How similar are two shapes? (for matching, not binding)
    Uses IoU (Intersection over Union)
    """
    mask1 = shape_to_mask(shape1_pts, size)
    mask2 = shape_to_mask(shape2_pts, size)
    
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


# =============================================================================
# Shape Target Node - Defines the "lock" that organisms must fit
# =============================================================================

class ShapeTargetNode(BaseNode):
    """
    Generates or accepts a target shape.
    This is the "ACE2 receptor" - the environmental constraint.
    
    Modes:
    - Internal: generates procedural shapes (star, gear, blob)
    - External: accepts shape from image or other source
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(200, 100, 50)

    def __init__(self):
        super().__init__()
        self.node_title = "Shape Target"
        
        self.inputs = {
            'external_shape': 'spectrum',  # DNA-like vector defining external shape
            'complexity': 'signal',        # How complex the target shape
            'morph_rate': 'signal'         # How fast the target changes
        }
        
        self.outputs = {
            'target_dna': 'spectrum',      # The target as DNA (for comparison)
            'target_view': 'image',        # Visualization
            'n_vertices': 'signal'         # Complexity metric
        }
        
        # Internal shape generation
        self.shape_type = 0  # 0=star, 1=gear, 2=blob, 3=external
        self.internal_dna = np.random.randn(32) * 0.5
        self.phase = 0.0
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        external = self.get_blended_input('external_shape', 'mean')
        complexity = self.get_blended_input('complexity', 'mean')
        morph = self.get_blended_input('morph_rate', 'mean')
        
        if complexity is None: complexity = 0.5
        if morph is None: morph = 0.01
        
        # Use external shape if provided
        if external is not None and len(external) >= 16:
            self.internal_dna = np.array(external[:32]) if len(external) >= 32 else np.resize(external, 32)
        else:
            # Slowly morph the internal shape
            self.phase += morph
            noise = np.sin(np.arange(32) * 0.5 + self.phase) * complexity * 0.1
            self.internal_dna = self.internal_dna * 0.99 + noise
        
        # Generate visualization
        points = dna_to_shape_points(self.internal_dna)
        
        self.display.fill(20)
        pts = points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.display, [pts], (100, 50, 50))
        cv2.polylines(self.display, [pts], True, (255, 100, 100), 2)
        
        # Draw center marker
        cv2.circle(self.display, (64, 64), 3, (255, 255, 255), -1)

    def get_output(self, name):
        if name == 'target_dna': return self.internal_dna.copy()
        if name == 'target_view': return self.display
        if name == 'n_vertices': return float(len(self.internal_dna))
        return None


# =============================================================================
# Shape Fitness Node - Measures binding/matching quality
# =============================================================================

class ShapeFitnessNode(BaseNode):
    """
    Computes fitness based on shape geometry.
    
    Two modes:
    1. MATCHING: How similar is organism to target? (mimicry, camouflage)
    2. BINDING: How well does organism bind to target? (receptor-ligand)
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 150, 50)

    def __init__(self):
        super().__init__()
        self.node_title = "Shape Fitness"
        
        self.inputs = {
            'organism_dna': 'spectrum',
            'target_dna': 'spectrum',
            'mode': 'signal'  # 0 = matching, 1 = binding
        }
        
        self.outputs = {
            'fitness': 'signal',
            'comparison_view': 'image'
        }
        
        self.fitness = 0.0
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        org_dna = self.get_blended_input('organism_dna', 'mean')
        target_dna = self.get_blended_input('target_dna', 'mean')
        mode = self.get_blended_input('mode', 'mean')
        
        if mode is None: mode = 0  # Default to matching
        
        if org_dna is None or target_dna is None:
            self.fitness = 0.0
            return
        
        # Convert to shapes
        org_points = dna_to_shape_points(org_dna)
        target_points = dna_to_shape_points(target_dna)
        
        # Compute fitness based on mode
        if mode < 0.5:
            # MATCHING MODE - maximize similarity
            self.fitness = compute_shape_similarity(org_points, target_points)
        else:
            # BINDING MODE - maximize complementary contact
            self.fitness = compute_binding_affinity(org_points, target_points)
        
        # Visualization
        self.display.fill(10)
        
        # Draw target (red)
        target_pts = target_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.display, [target_pts], True, (100, 100, 255), 1)
        
        # Draw organism (green)
        org_pts = org_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.display, [org_pts], True, (100, 255, 100), 2)
        
        # Fitness indicator
        bar_width = int(self.fitness * 120)
        cv2.rectangle(self.display, (4, 118), (4 + bar_width, 124), (0, 255, 0), -1)
        
        cv2.putText(self.display, f"Fit: {self.fitness:.2f}", (4, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        mode_str = "MATCH" if mode < 0.5 else "BIND"
        cv2.putText(self.display, mode_str, (80, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'fitness': return self.fitness
        if name == 'comparison_view': return self.display
        return None


# =============================================================================
# Shape Evolution Node - Evolves organisms to match/bind targets
# =============================================================================

class ShapeEvolutionNode(BaseNode):
    """
    Evolution engine specifically for shape problems.
    
    Key difference from CyberneticEvolution:
    - Fitness comes from shape comparison, not external control
    - DNA directly encodes morphology
    - Selection pressure is geometric
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 50, 200)

    def __init__(self):
        super().__init__()
        self.node_title = "Shape Evolution"
        
        self.inputs = {
            'target_dna': 'spectrum',
            'mutation_rate': 'signal',
            'mode': 'signal'  # 0=match, 1=bind
        }
        
        self.outputs = {
            'champion_dna': 'spectrum',
            'best_fitness': 'signal',
            'diversity': 'signal',
            'generation': 'signal',
            'population_view': 'image'
        }
        
        # Population
        self.pop_size = 48
        self.dna_len = 32
        self.population = [np.random.randn(self.dna_len) * 0.5 for _ in range(self.pop_size)]
        self.fitness_scores = np.zeros(self.pop_size)
        self.gen = 0
        self.champion = np.zeros(self.dna_len)
        self.best_fitness = 0.0
        
        self.display = np.zeros((128, 256, 3), dtype=np.uint8)

    def step(self):
        target_dna = self.get_blended_input('target_dna', 'mean')
        mutation_rate = self.get_blended_input('mutation_rate', 'mean')
        mode = self.get_blended_input('mode', 'mean')
        
        if target_dna is None:
            return
        
        if mutation_rate is None: mutation_rate = 0.1
        if mode is None: mode = 0
        
        target_pts = dna_to_shape_points(target_dna)
        
        # Evaluate all organisms
        for i, dna in enumerate(self.population):
            org_pts = dna_to_shape_points(dna)
            
            if mode < 0.5:
                self.fitness_scores[i] = compute_shape_similarity(org_pts, target_pts)
            else:
                self.fitness_scores[i] = compute_binding_affinity(org_pts, target_pts)
        
        # Selection
        sorted_idx = np.argsort(self.fitness_scores)[::-1]
        self.champion = self.population[sorted_idx[0]].copy()
        self.best_fitness = float(self.fitness_scores[sorted_idx[0]])
        
        # Breeding
        new_pop = []
        elite = max(2, int(self.pop_size * 0.15))
        
        # Keep elite
        for i in range(elite):
            new_pop.append(self.population[sorted_idx[i]].copy())
        
        # Breed rest
        while len(new_pop) < self.pop_size:
            # Tournament selection
            p1_idx = sorted_idx[np.random.randint(0, elite * 2)]
            p2_idx = sorted_idx[np.random.randint(0, elite * 2)]
            
            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]
            
            # Crossover - blend with random interpolation
            alpha = np.random.rand(self.dna_len)
            child = p1 * alpha + p2 * (1 - alpha)
            
            # Mutation
            if np.random.rand() < 0.6:
                mutation = np.random.randn(self.dna_len) * mutation_rate
                # Occasionally larger mutations for exploration
                if np.random.rand() < 0.1:
                    mutation *= 3
                child += mutation
            
            new_pop.append(child)
        
        self.population = new_pop
        self.gen += 1
        
        # Visualization - show top 6 organisms
        self.display.fill(15)
        cell_w, cell_h = 85, 64
        
        for idx in range(min(6, len(sorted_idx))):
            row = idx // 3
            col = idx % 3
            ox = col * cell_w + 5
            oy = row * cell_h
            
            dna = self.population[sorted_idx[idx]]
            pts = dna_to_shape_points(dna, n_points=32, base_radius=25)
            
            # Shift points to cell
            pts = pts - [64, 64]  # Center at origin
            pts = pts * 0.8  # Scale down
            pts = pts + [ox + cell_w//2, oy + cell_h//2]  # Move to cell
            
            pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
            
            # Color by fitness
            fit = self.fitness_scores[sorted_idx[idx]]
            green = int(fit * 255)
            cv2.polylines(self.display, [pts_int], True, (50, green, 100), 1)
            
            # Fitness label
            cv2.putText(self.display, f"{fit:.2f}", (ox + 2, oy + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        cv2.putText(self.display, f"Gen: {self.gen}", (5, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'champion_dna': return self.champion
        if name == 'best_fitness': return self.best_fitness
        if name == 'generation': return float(self.gen)
        if name == 'diversity':
            pop_arr = np.array(self.population)
            return float(np.std(pop_arr))
        if name == 'population_view': return self.display
        return None


# =============================================================================
# Co-Evolution Node - Two populations competing (virus vs antibody)
# =============================================================================

class CoEvolutionNode(BaseNode):
    """
    Two populations evolving against each other.
    
    Population A: "Pathogens" - try to evade binding
    Population B: "Antibodies" - try to bind pathogens
    
    This creates an evolutionary arms race.
    Shapes become increasingly complex as each side adapts.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(200, 50, 200)

    def __init__(self):
        super().__init__()
        self.node_title = "Co-Evolution"
        
        self.inputs = {
            'mutation_rate': 'signal',
            'selection_strength': 'signal'
        }
        
        self.outputs = {
            'pathogen_dna': 'spectrum',
            'antibody_dna': 'spectrum',
            'arms_race_index': 'signal',  # How complex has it gotten?
            'battle_view': 'image'
        }
        
        # Two populations
        self.pop_size = 24
        self.dna_len = 32
        
        self.pathogens = [np.random.randn(self.dna_len) * 0.3 for _ in range(self.pop_size)]
        self.antibodies = [np.random.randn(self.dna_len) * 0.3 for _ in range(self.pop_size)]
        
        self.pathogen_fitness = np.zeros(self.pop_size)
        self.antibody_fitness = np.zeros(self.pop_size)
        
        self.gen = 0
        self.complexity_history = deque(maxlen=100)
        
        self.champion_pathogen = np.zeros(self.dna_len)
        self.champion_antibody = np.zeros(self.dna_len)
        
        self.display = np.zeros((128, 256, 3), dtype=np.uint8)

    def step(self):
        mutation_rate = self.get_blended_input('mutation_rate', 'mean')
        selection = self.get_blended_input('selection_strength', 'mean')
        
        if mutation_rate is None: mutation_rate = 0.15
        if selection is None: selection = 1.0
        
        # Evaluate fitness through battles
        # Each pathogen fights each antibody
        pathogen_wins = np.zeros(self.pop_size)
        antibody_wins = np.zeros(self.pop_size)
        
        for p_idx, pathogen in enumerate(self.pathogens):
            p_pts = dna_to_shape_points(pathogen)
            
            for a_idx, antibody in enumerate(self.antibodies):
                a_pts = dna_to_shape_points(antibody)
                
                # Binding affinity
                affinity = compute_binding_affinity(p_pts, a_pts)
                
                # High affinity = antibody wins (neutralizes pathogen)
                # Low affinity = pathogen wins (evades)
                antibody_wins[a_idx] += affinity
                pathogen_wins[p_idx] += (1.0 - affinity)
        
        # Normalize
        self.pathogen_fitness = pathogen_wins / self.pop_size
        self.antibody_fitness = antibody_wins / self.pop_size
        
        # Select champions
        best_p = np.argmax(self.pathogen_fitness)
        best_a = np.argmax(self.antibody_fitness)
        self.champion_pathogen = self.pathogens[best_p].copy()
        self.champion_antibody = self.antibodies[best_a].copy()
        
        # Breed both populations
        self.pathogens = self._breed_population(
            self.pathogens, self.pathogen_fitness, mutation_rate, selection)
        self.antibodies = self._breed_population(
            self.antibodies, self.antibody_fitness, mutation_rate, selection)
        
        self.gen += 1
        
        # Track complexity (measure of arms race)
        p_complexity = np.std(self.champion_pathogen)
        a_complexity = np.std(self.champion_antibody)
        self.complexity_history.append(p_complexity + a_complexity)
        
        # Visualization
        self._draw_battle()

    def _breed_population(self, population, fitness, mutation_rate, selection):
        sorted_idx = np.argsort(fitness)[::-1]
        new_pop = []
        
        elite = max(2, int(self.pop_size * 0.2))
        for i in range(elite):
            new_pop.append(population[sorted_idx[i]].copy())
        
        while len(new_pop) < self.pop_size:
            # Selection pressure based on selection strength
            top_k = max(2, int(elite * (2 - selection)))
            p1 = population[sorted_idx[np.random.randint(0, top_k)]]
            p2 = population[sorted_idx[np.random.randint(0, top_k)]]
            
            # Crossover
            alpha = np.random.rand(self.dna_len)
            child = p1 * alpha + p2 * (1 - alpha)
            
            # Mutation
            if np.random.rand() < 0.7:
                child += np.random.randn(self.dna_len) * mutation_rate
            
            new_pop.append(child)
        
        return new_pop

    def _draw_battle(self):
        self.display.fill(10)
        
        # Draw champion pathogen (left, red)
        p_pts = dna_to_shape_points(self.champion_pathogen, base_radius=40)
        p_pts = p_pts - [64, 64] + [64, 64]  # Keep centered in left half
        p_pts_int = p_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.display, [p_pts_int], (40, 40, 100))
        cv2.polylines(self.display, [p_pts_int], True, (100, 100, 255), 2)
        
        # Draw champion antibody (right, green)
        a_pts = dna_to_shape_points(self.champion_antibody, base_radius=40)
        a_pts = a_pts - [64, 64] + [192, 64]  # Shift to right half
        a_pts_int = a_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.display, [a_pts_int], (40, 100, 40))
        cv2.polylines(self.display, [a_pts_int], True, (100, 255, 100), 2)
        
        # Labels
        cv2.putText(self.display, "PATHOGEN", (20, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
        cv2.putText(self.display, "ANTIBODY", (165, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        # Generation and complexity
        cv2.putText(self.display, f"Gen: {self.gen}", (5, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        if len(self.complexity_history) > 0:
            complexity = np.mean(list(self.complexity_history)[-10:])
            cv2.putText(self.display, f"Arms Race: {complexity:.2f}", (140, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'pathogen_dna': return self.champion_pathogen
        if name == 'antibody_dna': return self.champion_antibody
        if name == 'arms_race_index':
            if len(self.complexity_history) == 0:
                return 0.0
            return float(np.mean(list(self.complexity_history)[-10:]))
        if name == 'battle_view': return self.display
        return None


# =============================================================================
# EEG Receptor Node - Evolve shapes that "resonate" with brain signals
# =============================================================================

class EEGReceptorNode(BaseNode):
    """
    Converts EEG spectrum into a target shape.
    Then evolution finds organism shapes that "bind" to your brain state.
    
    The result: organisms that can "hear" specific neural frequencies.
    Different brain states → different optimal receptor shapes.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(100, 200, 255)

    def __init__(self):
        super().__init__()
        self.node_title = "EEG Receptor"
        
        self.inputs = {
            'eeg_spectrum': 'spectrum',  # From FFT of EEG
            'frequency_focus': 'signal'  # Which band to emphasize
        }
        
        self.outputs = {
            'brain_shape_dna': 'spectrum',  # EEG as shape DNA
            'receptor_view': 'image'
        }
        
        self.brain_dna = np.zeros(32)
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        eeg = self.get_blended_input('eeg_spectrum', 'mean')
        focus = self.get_blended_input('frequency_focus', 'mean')
        
        if eeg is None:
            # Demo mode - generate synthetic "brain" pattern
            t = np.linspace(0, 4*np.pi, 32)
            eeg = np.sin(t) * 0.5 + np.sin(3*t) * 0.3
        
        if focus is None: focus = 0.5
        
        # Map EEG spectrum to DNA
        # The spectrum becomes the Fourier coefficients of the shape
        if len(eeg) >= 32:
            self.brain_dna = np.array(eeg[:32])
        else:
            self.brain_dna = np.resize(eeg, 32)
        
        # Apply frequency focus - emphasize certain harmonics
        focus_idx = int(focus * 15)
        weights = np.exp(-0.1 * (np.arange(32) - focus_idx*2)**2)
        self.brain_dna = self.brain_dna * weights
        
        # Normalize
        if np.max(np.abs(self.brain_dna)) > 0:
            self.brain_dna = self.brain_dna / np.max(np.abs(self.brain_dna))
        
        # Visualization - the "brain shape"
        pts = dna_to_shape_points(self.brain_dna, base_radius=45)
        
        self.display.fill(15)
        pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
        
        # Pulsing glow effect
        for r in range(3, 0, -1):
            color = (50 + r*20, 100 + r*30, 150 + r*30)
            cv2.polylines(self.display, [pts_int], True, color, r)
        
        cv2.putText(self.display, "BRAIN STATE", (25, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'brain_shape_dna': return self.brain_dna.copy()
        if name == 'receptor_view': return self.display
        return None