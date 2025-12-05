"""
Synthetic Evolution Ecosystem (v5 - Auto-Scroll Fixed)
------------------------------------------------------
Fixes:
- Breeding Arena now defaults to "Live Feed" (End of history).
- History recording is more sensitive (captures micro-mutations).
- Visualization layout improved.
"""

import numpy as np
import cv2

# --- STRICT COMPATIBILITY IMPORTS ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def get_blended_input(self, name, mode): return None

# --- HELPER: Handle Vector Resizing ---
def match_size(vector, target_len):
    """Ensures vector is exactly target_len via tiling or truncation."""
    if vector is None: return np.zeros(target_len)
    arr = np.array(vector).flatten()
    if len(arr) == 0: return np.zeros(target_len)
    if len(arr) == target_len: return arr
    return np.resize(arr, target_len)

class MiniSolverLite:
    """Stripped down solver for fast population simulation"""
    def __init__(self, n_atoms=16):
        self.N = n_atoms
        self.phases = np.zeros(n_atoms * n_atoms)

    def load_dna(self, dna):
        if dna is None or len(dna) == 0: return
        limit = min(len(dna), len(self.phases))
        self.phases[:limit] = dna[:limit] * 2 * np.pi

    def evaluate_fitness(self):
        n = self.N
        bond_matrix = self.phases[:n*n].reshape(n, n)
        conn = np.cos(bond_matrix)
        
        stability = np.sum(conn > 0.8)
        stress = np.sum((conn > 0.0) & (conn < 0.5))
        
        max_bonds = n * (n-1) / 2
        return (stability / max_bonds, stress / max_bonds)

class SyntheticEvolutionNode(BaseNode):
    """
    The Engine of Life. Manages population and selection.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 100, 200) # Hot Pink

    def __init__(self):
        super().__init__()
        self.node_title = "Evolution Engine"
        
        self.inputs = {
            'seed_dna': 'spectrum',       
            'selection_pressure': 'signal' 
        }
        
        self.outputs = {
            'champion_dna': 'spectrum',   
            'dead_dna': 'spectrum',       
            'avg_fitness': 'signal',
            'generation': 'signal'
        }
        
        # Population State
        self.pop_size = 32
        self.dna_len = 128
        self.population = [np.random.rand(self.dna_len) for _ in range(self.pop_size)]
        self.fitness_scores = np.zeros(self.pop_size)
        self.gen_counter = 0
        
        self.solver = MiniSolverLite(16)
        
        # Output Buffers
        self.out_champion = np.zeros(self.dna_len)
        self.out_dead = np.zeros(self.dna_len)
        self.out_fitness = 0.0
        self.out_gen = 0.0

    def step(self):
        # 1. Inputs
        seed = self.get_blended_input('seed_dna', 'mean')
        pressure = self.get_blended_input('selection_pressure', 'sum')
        if pressure is None: pressure = 0.5
        
        # 2. Inject Seed
        if seed is not None:
            seed_fixed = match_size(seed, self.dna_len)
            indices = np.random.choice(self.pop_size, size=int(self.pop_size*0.1))
            for i in indices:
                self.population[i] = seed_fixed + np.random.randn(self.dna_len) * 0.1

        # 3. Evaluate
        for i in range(self.pop_size):
            self.solver.load_dna(self.population[i])
            stab, stress = self.solver.evaluate_fitness()
            self.fitness_scores[i] = stab - (stress * 0.5)

        # 4. Selection
        sorted_idx = np.argsort(self.fitness_scores)[::-1]
        best_idx = sorted_idx[0]
        worst_idx = sorted_idx[-1]
        
        self.out_champion = self.population[best_idx].copy()
        self.out_dead = self.population[worst_idx].copy()
        self.out_fitness = float(np.mean(self.fitness_scores))
        
        # 5. Breeding
        new_pop = []
        elite_count = int(self.pop_size * 0.2)
        for i in range(elite_count):
            new_pop.append(self.population[sorted_idx[i]])
            
        while len(new_pop) < self.pop_size:
            p1 = self.population[np.random.choice(sorted_idx[:elite_count*2])]
            p2 = self.population[np.random.choice(sorted_idx[:elite_count*2])]
            
            split = np.random.randint(0, self.dna_len)
            child = np.zeros(self.dna_len)
            child[:split] = p1[:split]
            child[split:] = p2[split:]
            
            if np.random.rand() < 0.3:
                child += np.random.randn(self.dna_len) * (0.1 * pressure)
            new_pop.append(child)
            
        self.population = new_pop
        self.gen_counter += 1
        self.out_gen = float(self.gen_counter)

    def get_output(self, name):
        if name == 'champion_dna': return self.out_champion
        if name == 'dead_dna': return self.out_dead
        if name == 'avg_fitness': return self.out_fitness
        if name == 'generation': return self.out_gen
        return None


class FitnessFunctionNode(BaseNode):
    """Analyzes a DNA vector."""
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(200, 200, 50) 

    def __init__(self):
        super().__init__()
        self.node_title = "Fitness Function"
        self.inputs = {'dna_in': 'spectrum'}
        self.outputs = {'stability': 'signal', 'stress': 'signal', 'score': 'signal'}
        self.solver = MiniSolverLite(16)
        
        self.out_stab = 0.0
        self.out_stress = 0.0
        self.out_score = 0.0

    def step(self):
        dna = self.get_blended_input('dna_in', 'mean')
        if dna is None: return
        
        self.solver.load_dna(dna)
        stab, stress = self.solver.evaluate_fitness()
        
        self.out_stab = float(stab)
        self.out_stress = float(stress)
        self.out_score = float(stab - stress)

    def get_output(self, name):
        if name == 'stability': return self.out_stab
        if name == 'stress': return self.out_stress
        if name == 'score': return self.out_score
        return None


class MolecularGraveyardNode(BaseNode):
    """Recycles dead DNA into Ghost DNA."""
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(80, 80, 80)

    def __init__(self):
        super().__init__()
        self.node_title = "Molecular Graveyard"
        self.inputs = {'corpse_dna': 'spectrum'}
        self.outputs = {'ghost_dna': 'spectrum', 'entropy_view': 'image'}
        
        self.memory_size = 50
        self.graveyard = [] 
        self.display = np.zeros((100, 200, 3), dtype=np.uint8)
        self.out_ghost = np.zeros(128)

    def step(self):
        corpse = self.get_blended_input('corpse_dna', 'mean')
        
        if corpse is not None:
            corpse_fixed = match_size(corpse, 128)
            self.graveyard.append(corpse_fixed)
            if len(self.graveyard) > self.memory_size:
                self.graveyard.pop(0)
        
        if len(self.graveyard) > 0:
            ghost = np.mean(self.graveyard, axis=0)
            ghost += np.random.randn(len(ghost)) * 0.05
            self.out_ghost = ghost
            
        # Visualize
        self.display.fill(10)
        if len(self.graveyard) > 0:
            for i, dead in enumerate(self.graveyard):
                y = int(np.mean(dead) * 100)
                y = np.clip(y, 0, 99)
                intensity = int((i / self.memory_size) * 200)
                cv2.line(self.display, (0, y), (200, y), (50, 50, intensity), 1)

    def get_output(self, name):
        if name == 'ghost_dna': return self.out_ghost
        if name == 'entropy_view': return self.display
        return None


class BreedingArenaNode(BaseNode):
    """
    Visualizes the top organisms in a grid.
    Features: Auto-scroll to live generation.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(0, 150, 200)

    def __init__(self):
        super().__init__()
        self.node_title = "Breeding Arena"
        self.inputs = {
            'champion_dna': 'spectrum',
            'scroll': 'signal' # 0.0=Oldest, 1.0=Live
        }
        self.outputs = {'arena_view': 'image'}
        
        self.history_size = 500
        self.hall_of_fame = [] # List of (dna, generation_index)
        self.display = np.zeros((256, 256, 3), dtype=np.uint8)
        self.global_gen_counter = 0

    def step(self):
        dna = self.get_blended_input('champion_dna', 'mean')
        scroll = self.get_blended_input('scroll', 'mean')
        
        # 1. Record History (More sensitive check: 0.001)
        if dna is not None:
            dna_fixed = match_size(dna, 128)
            
            # Check if it's different enough or if it's the first one
            is_new = False
            if len(self.hall_of_fame) == 0:
                is_new = True
            else:
                last_dna = self.hall_of_fame[-1][0]
                # Compare similarity. If distance > threshold, it's a new gene
                dist = np.mean(np.abs(dna_fixed - last_dna))
                if dist > 0.001: # 0.1% change is enough to record
                    is_new = True
            
            if is_new:
                self.hall_of_fame.append( (dna_fixed, self.global_gen_counter) )
                self.global_gen_counter += 1
                if len(self.hall_of_fame) > self.history_size:
                    self.hall_of_fame.pop(0)

        # 2. Determine View Window
        n_items = len(self.hall_of_fame)
        if n_items == 0: 
            self.display.fill(0)
            return

        # LOGIC FIX: If scroll is None, Force Auto-Scroll (Live View)
        if scroll is None:
            start_idx = max(0, n_items - 9)
        else:
            # Map 0..1 to index
            target = int(scroll * (n_items - 9))
            start_idx = np.clip(target, 0, max(0, n_items - 9))
        
        # Get the slice
        view_slice = self.hall_of_fame[start_idx : start_idx+9]

        # 3. Draw Grid
        self.display.fill(20) # Dark bg
        cell_w = 256 // 3
        cell_h = 256 // 3
        
        for idx, (gene, gen_num) in enumerate(view_slice):
            row = idx // 3
            col = idx % 3
            ox = col * cell_w
            oy = row * cell_h
            center = (ox + cell_w//2, oy + cell_h//2)
            
            # Draw Cell BG
            cv2.rectangle(self.display, (ox, oy), (ox+cell_w, oy+cell_h), (40,40,40), 1)
            
            # Draw Glyph (Miniature Organism)
            pts = []
            for k in range(8):
                val = gene[k] if k < len(gene) else 0
                angle = k * (2*np.pi/8)
                r = 10 + val * 25
                px = int(center[0] + r * np.cos(angle))
                py = int(center[1] + r * np.sin(angle))
                pts.append((px, py))
            
            # Draw lines
            for k in range(8):
                cv2.line(self.display, pts[k], pts[(k+1)%8], (0, 255, 150), 1)
            
            # Gen Label
            cv2.putText(self.display, f"#{gen_num}", (ox+5, oy+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        # 4. Scrollbar Indicator
        if n_items > 9:
            # Draw a bar on the right side
            bar_height = max(10, int((9 / n_items) * 256))
            bar_rel_pos = start_idx / max(1, (n_items - 9))
            bar_y = int(bar_rel_pos * (256 - bar_height))
            
            # Color: Blue if scrolling, Red if Locked Live
            color = (100, 100, 255) # Red-ish (BGR)
            if start_idx == max(0, n_items - 9):
                color = (0, 255, 0) # Green (Live)
                
            cv2.rectangle(self.display, (250, bar_y), (256, bar_y+bar_height), color, -1)

    def get_output(self, name):
        if name == 'arena_view': return self.display
        return None