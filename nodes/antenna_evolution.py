"""
Antenna Evolution: Shape as Electromagnetic Interface
======================================================

The insight chain:
1. DNA is a fractal antenna (Blank & Goodman 2011) - structure determines bandwidth
2. Dendrites are frequency-tuned antennas (MIT, ephaptic coupling research)
3. Evolved shapes ARE antenna patterns - not decoration, but functional geometry
4. Multiple antennas with compatible shapes can COUPLE through field effects

This module implements:
- Organisms as antenna patterns with computable radiation characteristics
- Field coupling between organisms based on shape resonance
- An ecosystem where shapes "talk" to each other through their geometry
- Evolution that optimizes for both survival AND communication

The latent vector is the "DNA" that generates the antenna pattern.
The shape IS the antenna. The antenna determines what you can hear.
"""

import numpy as np
import cv2
from collections import deque
from scipy.fft import fft, ifft
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


def dna_to_antenna_pattern(dna, n_points=64):
    """
    Convert DNA to antenna radiation pattern.
    Uses Fourier synthesis - DNA encodes frequency components.
    
    Returns: (boundary_points, frequency_response)
    - boundary_points: the physical shape
    - frequency_response: what frequencies this antenna can receive/transmit
    """
    if dna is None or len(dna) < 8:
        dna = np.zeros(32)
    
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # DNA as Fourier coefficients for the shape
    n_harmonics = min(12, len(dna) // 2)
    
    # Build the shape
    radii = np.ones(n_points) * 50  # base radius
    for k in range(n_harmonics):
        amp = dna[k*2] * 15
        phase = dna[k*2 + 1] * np.pi
        harmonic = k + 1
        radii += amp * np.cos(harmonic * angles + phase)
    
    radii = np.clip(radii, 10, 100)
    
    # The frequency response IS the DNA (Fourier coefficients)
    # Normalized to unit energy
    freq_response = np.abs(dna[:n_harmonics*2:2])  # Just the amplitudes
    if np.sum(freq_response) > 0:
        freq_response = freq_response / np.sum(freq_response)
    
    # Convert to cartesian
    cx, cy = 64, 64
    points = np.array([(cx + r * np.cos(a), cy + r * np.sin(a)) 
                       for a, r in zip(angles, radii)])
    
    return points, freq_response


def compute_antenna_coupling(dna1, dna2):
    """
    Compute coupling strength between two antenna patterns.
    
    Based on:
    1. Frequency overlap (do they resonate at same frequencies?)
    2. Impedance matching (complementary vs similar shapes)
    
    Returns: coupling coefficient (0 to 1)
    """
    _, freq1 = dna_to_antenna_pattern(dna1)
    _, freq2 = dna_to_antenna_pattern(dna2)
    
    # Pad to same length
    max_len = max(len(freq1), len(freq2))
    freq1 = np.resize(freq1, max_len)
    freq2 = np.resize(freq2, max_len)
    
    # Frequency overlap - dot product of normalized spectra
    overlap = np.dot(freq1, freq2)
    
    # Phase coherence - how aligned are their Fourier phases?
    if len(dna1) >= 16 and len(dna2) >= 16:
        phases1 = dna1[1:16:2]  # odd indices = phases
        phases2 = dna2[1:16:2]
        phase_coherence = np.abs(np.mean(np.exp(1j * (phases1 - phases2))))
    else:
        phase_coherence = 0.5
    
    # Combined coupling
    coupling = overlap * 0.6 + phase_coherence * 0.4
    
    return float(np.clip(coupling, 0, 1))


def compute_field_at_point(source_dna, source_pos, target_pos, time=0):
    """
    Compute the electromagnetic field contribution from one antenna at a point.
    
    The field strength depends on:
    1. Distance (inverse square falloff)
    2. Direction (antenna pattern is directional)
    3. Time (oscillating field)
    """
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    distance = np.sqrt(dx*dx + dy*dy) + 1e-6
    angle = np.arctan2(dy, dx)
    
    # Get antenna pattern (angular gain)
    points, freq_response = dna_to_antenna_pattern(source_dna, n_points=32)
    
    # Angular index into pattern
    pattern_idx = int((angle / (2*np.pi) + 0.5) * 32) % 32
    
    # Radial extent at this angle = gain in this direction
    center = np.array([64, 64])
    gain = np.linalg.norm(points[pattern_idx] - center) / 50.0  # normalized
    
    # Field strength with distance falloff
    field_strength = gain / (1 + distance * 0.02)
    
    # Oscillating component (superposition of frequencies)
    oscillation = 0
    for k, amp in enumerate(freq_response):
        freq = (k + 1) * 0.5  # frequency in arbitrary units
        oscillation += amp * np.sin(2 * np.pi * freq * time)
    
    return field_strength * (1 + 0.3 * oscillation)


# =============================================================================
# Antenna Field Node - Visualizes the electromagnetic field of an organism
# =============================================================================

class AntennaFieldNode(BaseNode):
    """
    Visualizes the electromagnetic field pattern of an organism.
    
    The shape determines the radiation pattern - like viewing an antenna
    in a near-field measurement chamber.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(100, 150, 255)

    def __init__(self):
        super().__init__()
        self.node_title = "Antenna Field"
        
        self.inputs = {
            'dna': 'spectrum',
            'frequency': 'signal'  # Which frequency to visualize
        }
        
        self.outputs = {
            'field_view': 'image',
            'bandwidth': 'signal',  # How many frequencies can it receive
            'directivity': 'signal'  # How focused is the pattern
        }
        
        self.time = 0.0
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        dna = self.get_blended_input('dna', 'mean')
        freq_select = self.get_blended_input('frequency', 'mean')
        
        if dna is None:
            dna = np.random.randn(32) * 0.3
        
        if freq_select is None:
            freq_select = 0.5
        
        self.time += 0.1
        
        # Get antenna pattern
        points, freq_response = dna_to_antenna_pattern(dna)
        
        # Compute field on a grid
        self.display.fill(0)
        field = np.zeros((128, 128))
        
        for y in range(0, 128, 2):
            for x in range(0, 128, 2):
                f = compute_field_at_point(dna, (64, 64), (x, y), self.time)
                field[y:y+2, x:x+2] = f
        
        # Normalize and colorize
        field = np.clip(field, 0, 2)
        field_norm = (field / 2 * 255).astype(np.uint8)
        
        # Color map: blue (weak) -> cyan -> green -> yellow (strong)
        self.display[:,:,0] = np.clip(field_norm * 0.3, 0, 255).astype(np.uint8)
        self.display[:,:,1] = np.clip(field_norm * 0.8, 0, 255).astype(np.uint8)
        self.display[:,:,2] = np.clip(255 - field_norm, 0, 255).astype(np.uint8)
        
        # Draw antenna outline
        pts = points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.display, [pts], True, (255, 255, 255), 1)
        
        # Metrics
        self.bandwidth = float(np.sum(freq_response > 0.05))  # Active bands
        self.directivity = float(np.std(freq_response) * 10)  # Pattern variation

    def get_output(self, name):
        if name == 'field_view': return self.display
        if name == 'bandwidth': return self.bandwidth
        if name == 'directivity': return self.directivity
        return None


# =============================================================================
# Field Ecosystem Node - Multiple organisms coupling through field effects
# =============================================================================

class FieldEcosystemNode(BaseNode):
    """
    An ecosystem where organisms influence each other through field coupling.
    
    Each organism:
    - Has a position in 2D space
    - Radiates a field determined by its shape (DNA)
    - Receives energy from other organisms' fields
    - Evolves based on total energy received (fitness)
    
    This creates selection pressure for:
    - Shapes that can receive energy (good antennas)
    - Shapes that couple well with neighbors (resonance)
    - Possibly: shapes that can "communicate" information
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(200, 100, 255)

    def __init__(self):
        super().__init__()
        self.node_title = "Field Ecosystem"
        
        self.inputs = {
            'external_signal': 'spectrum',  # Environmental broadcast (e.g., EEG)
            'mutation_rate': 'signal'
        }
        
        self.outputs = {
            'best_receiver_dna': 'spectrum',
            'best_transmitter_dna': 'spectrum',
            'ecosystem_view': 'image',
            'avg_coupling': 'signal',
            'generation': 'signal'
        }
        
        # Population
        self.pop_size = 16
        self.dna_len = 32
        
        # Each organism: (dna, position, energy_received, energy_transmitted)
        self.organisms = []
        for _ in range(self.pop_size):
            dna = np.random.randn(self.dna_len) * 0.5
            pos = np.random.rand(2) * 100 + 14  # positions in 128x128 space
            self.organisms.append({
                'dna': dna,
                'pos': pos,
                'received': 0.0,
                'transmitted': 0.0
            })
        
        self.gen = 0
        self.time = 0.0
        self.display = np.zeros((256, 256, 3), dtype=np.uint8)
        
        self.coupling_history = deque(maxlen=100)

    def step(self):
        external = self.get_blended_input('external_signal', 'mean')
        mutation_rate = self.get_blended_input('mutation_rate', 'mean')
        
        if mutation_rate is None:
            mutation_rate = 0.1
        
        self.time += 0.1
        
        # Reset energy accumulators
        for org in self.organisms:
            org['received'] = 0.0
            org['transmitted'] = 0.0
        
        # Compute pairwise field interactions
        total_coupling = 0
        n_pairs = 0
        
        for i, org_i in enumerate(self.organisms):
            for j, org_j in enumerate(self.organisms):
                if i >= j:
                    continue
                
                # Distance
                dist = np.linalg.norm(org_i['pos'] - org_j['pos'])
                
                # Antenna coupling (shape-based)
                coupling = compute_antenna_coupling(org_i['dna'], org_j['dna'])
                
                # Field strength falls off with distance
                field_factor = 1.0 / (1 + dist * 0.05)
                
                # Energy exchange
                energy = coupling * field_factor
                
                org_i['received'] += energy
                org_j['received'] += energy
                org_i['transmitted'] += energy
                org_j['transmitted'] += energy
                
                total_coupling += coupling
                n_pairs += 1
        
        # Add external signal reception
        if external is not None and len(external) >= self.dna_len:
            for org in self.organisms:
                ext_coupling = compute_antenna_coupling(org['dna'], external[:self.dna_len])
                org['received'] += ext_coupling * 2  # External signal is strong
        
        # Record average coupling
        if n_pairs > 0:
            self.coupling_history.append(total_coupling / n_pairs)
        
        # Evolution every N steps
        if self.time % 3.0 < 0.15:
            self._evolve_population(mutation_rate)
            self.gen += 1
        
        # Visualization
        self._draw_ecosystem()

    def _evolve_population(self, mutation_rate):
        """Selection and breeding based on energy received"""
        
        # Sort by fitness (received energy)
        sorted_orgs = sorted(self.organisms, 
                            key=lambda o: o['received'], 
                            reverse=True)
        
        new_orgs = []
        elite = max(2, int(self.pop_size * 0.25))
        
        # Keep elite
        for i in range(elite):
            new_orgs.append({
                'dna': sorted_orgs[i]['dna'].copy(),
                'pos': sorted_orgs[i]['pos'].copy(),
                'received': 0.0,
                'transmitted': 0.0
            })
        
        # Breed rest
        while len(new_orgs) < self.pop_size:
            # Select parents from top half
            p1 = sorted_orgs[np.random.randint(0, elite * 2)]
            p2 = sorted_orgs[np.random.randint(0, elite * 2)]
            
            # Crossover
            alpha = np.random.rand(self.dna_len)
            child_dna = p1['dna'] * alpha + p2['dna'] * (1 - alpha)
            
            # Mutation
            if np.random.rand() < 0.5:
                child_dna += np.random.randn(self.dna_len) * mutation_rate
            
            # Position: near parents with some spread
            child_pos = (p1['pos'] + p2['pos']) / 2 + np.random.randn(2) * 10
            child_pos = np.clip(child_pos, 14, 114)
            
            new_orgs.append({
                'dna': child_dna,
                'pos': child_pos,
                'received': 0.0,
                'transmitted': 0.0
            })
        
        self.organisms = new_orgs

    def _draw_ecosystem(self):
        self.display.fill(10)
        
        # Draw field lines between coupled organisms
        for i, org_i in enumerate(self.organisms):
            for j, org_j in enumerate(self.organisms):
                if i >= j:
                    continue
                
                coupling = compute_antenna_coupling(org_i['dna'], org_j['dna'])
                if coupling > 0.3:  # Only show strong couplings
                    p1 = (int(org_i['pos'][0] * 2), int(org_i['pos'][1] * 2))
                    p2 = (int(org_j['pos'][0] * 2), int(org_j['pos'][1] * 2))
                    intensity = int(coupling * 200)
                    cv2.line(self.display, p1, p2, (intensity//2, intensity, intensity//2), 1)
        
        # Draw organisms
        max_received = max(o['received'] for o in self.organisms) + 1e-6
        
        for org in self.organisms:
            # Position (scaled to 256x256)
            cx = int(org['pos'][0] * 2)
            cy = int(org['pos'][1] * 2)
            
            # Get shape points
            points, _ = dna_to_antenna_pattern(org['dna'], n_points=16)
            
            # Scale and translate
            points = (points - 64) * 0.3 + [cx, cy]
            pts = points.astype(np.int32).reshape((-1, 1, 2))
            
            # Color by received energy
            energy_ratio = org['received'] / max_received
            color = (50, int(100 + 155 * energy_ratio), int(100 + 100 * energy_ratio))
            
            cv2.polylines(self.display, [pts], True, color, 1)
            cv2.circle(self.display, (cx, cy), 2, color, -1)
        
        # Labels
        cv2.putText(self.display, f"Gen: {self.gen}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if len(self.coupling_history) > 0:
            avg = np.mean(list(self.coupling_history)[-20:])
            cv2.putText(self.display, f"Coupling: {avg:.2f}", (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'ecosystem_view': return self.display
        if name == 'best_receiver_dna':
            best = max(self.organisms, key=lambda o: o['received'])
            return best['dna'].copy()
        if name == 'best_transmitter_dna':
            best = max(self.organisms, key=lambda o: o['transmitted'])
            return best['dna'].copy()
        if name == 'avg_coupling':
            if len(self.coupling_history) == 0:
                return 0.0
            return float(np.mean(list(self.coupling_history)[-20:]))
        if name == 'generation':
            return float(self.gen)
        return None


# =============================================================================
# Resonance Network Node - Organisms form a wireless neural network
# =============================================================================

class ResonanceNetworkNode(BaseNode):
    """
    Organisms as nodes in a wireless neural network.
    
    Information propagates through field coupling.
    An input signal at one organism propagates to others
    based on their antenna coupling coefficients.
    
    This is ephaptic coupling, scaled up to the ecosystem level.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 150, 100)

    def __init__(self):
        super().__init__()
        self.node_title = "Resonance Network"
        
        self.inputs = {
            'input_signal': 'signal',      # Signal injected into network
            'organism_dnas': 'spectrum',   # DNA patterns of network nodes
            'topology': 'signal'           # 0=ring, 1=random, 2=fully connected
        }
        
        self.outputs = {
            'output_signal': 'signal',     # Signal at output node
            'propagation_view': 'image',
            'network_coherence': 'signal'  # How synchronized is the network
        }
        
        self.n_nodes = 8
        self.node_states = np.zeros(self.n_nodes)
        self.node_dnas = [np.random.randn(32) * 0.5 for _ in range(self.n_nodes)]
        
        # Precompute coupling matrix
        self.coupling_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self._update_coupling_matrix()
        
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)
        self.history = deque(maxlen=50)

    def _update_coupling_matrix(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.coupling_matrix[i, j] = compute_antenna_coupling(
                        self.node_dnas[i], self.node_dnas[j]
                    )

    def step(self):
        input_sig = self.get_blended_input('input_signal', 'mean')
        new_dnas = self.get_blended_input('organism_dnas', 'mean')
        
        # Update DNAs if provided
        if new_dnas is not None and len(new_dnas) >= 32:
            # Use chunks of the input as different node DNAs
            for i in range(min(self.n_nodes, len(new_dnas) // 32)):
                self.node_dnas[i] = new_dnas[i*32:(i+1)*32]
            self._update_coupling_matrix()
        
        if input_sig is None:
            input_sig = np.sin(len(self.history) * 0.2)  # Default oscillation
        
        # Inject signal at first node
        self.node_states[0] = input_sig
        
        # Propagate through network (one step of diffusion)
        new_states = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            # Self-decay
            new_states[i] = self.node_states[i] * 0.8
            
            # Input from coupled neighbors
            for j in range(self.n_nodes):
                if i != j:
                    new_states[i] += self.node_states[j] * self.coupling_matrix[j, i] * 0.3
        
        self.node_states = np.tanh(new_states)  # Nonlinearity
        
        self.history.append(self.node_states.copy())
        
        # Visualization
        self._draw_network()

    def _draw_network(self):
        self.display.fill(10)
        
        # Node positions in a circle
        cx, cy = 64, 64
        radius = 45
        positions = []
        for i in range(self.n_nodes):
            angle = i * 2 * np.pi / self.n_nodes - np.pi/2
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            positions.append((x, y))
        
        # Draw coupling lines
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                coupling = self.coupling_matrix[i, j]
                if coupling > 0.2:
                    intensity = int(coupling * 150)
                    cv2.line(self.display, positions[i], positions[j],
                            (intensity//2, intensity, intensity//2), 1)
        
        # Draw nodes
        for i, (x, y) in enumerate(positions):
            # Size by state amplitude
            size = int(5 + abs(self.node_states[i]) * 10)
            
            # Color by state sign
            if self.node_states[i] > 0:
                color = (50, 200, 50)  # Green = positive
            else:
                color = (50, 50, 200)  # Blue = negative
            
            cv2.circle(self.display, (x, y), size, color, -1)
            cv2.circle(self.display, (x, y), size, (200, 200, 200), 1)
        
        # Input/output markers
        cv2.putText(self.display, "IN", (positions[0][0]-8, positions[0][1]-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(self.display, "OUT", (positions[self.n_nodes//2][0]-10, 
                   positions[self.n_nodes//2][1]-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Coherence indicator
        coherence = self._compute_coherence()
        bar_width = int(coherence * 100)
        cv2.rectangle(self.display, (14, 118), (14 + bar_width, 124),
                     (100, 200, 100), -1)

    def _compute_coherence(self):
        """How synchronized are the node states?"""
        if len(self.history) < 10:
            return 0.0
        
        recent = np.array(list(self.history)[-10:])
        
        # Coherence = how correlated are the oscillations
        correlations = []
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                corr = np.corrcoef(recent[:, i], recent[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if len(correlations) == 0:
            return 0.0
        
        return float(np.mean(correlations))

    def get_output(self, name):
        if name == 'output_signal':
            return float(self.node_states[self.n_nodes // 2])
        if name == 'propagation_view':
            return self.display
        if name == 'network_coherence':
            return self._compute_coherence()
        return None


# =============================================================================
# Fractal Antenna Node - DNA-like self-similar structure
# =============================================================================

class FractalAntennaNode(BaseNode):
    """
    Generates fractal antenna patterns inspired by the DNA structure.
    
    From Blank & Goodman (2011):
    - DNA has multiple scales of coiling (1nm helix → 10nm fiber → 30nm solenoid → 200nm tube)
    - Each scale resonates with different frequencies
    - Self-similarity creates broadband reception
    
    This node generates shapes with explicit fractal structure.
    """
    NODE_CATEGORY = "Artificial Life"
    NODE_COLOR = QtGui.QColor(255, 200, 50)

    def __init__(self):
        super().__init__()
        self.node_title = "Fractal Antenna"
        
        self.inputs = {
            'seed_dna': 'spectrum',
            'fractal_depth': 'signal',  # 1-4 levels of self-similarity
            'base_frequency': 'signal'
        }
        
        self.outputs = {
            'fractal_dna': 'spectrum',
            'antenna_view': 'image',
            'bandwidth': 'signal'  # Number of frequency bands
        }
        
        self.dna = np.zeros(64)
        self.display = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        seed = self.get_blended_input('seed_dna', 'mean')
        depth = self.get_blended_input('fractal_depth', 'mean')
        base_freq = self.get_blended_input('base_frequency', 'mean')
        
        if seed is None:
            seed = np.random.randn(16) * 0.5
        if depth is None:
            depth = 3
        if base_freq is None:
            base_freq = 1.0
        
        depth = int(np.clip(depth, 1, 4))
        
        # Generate fractal DNA
        # Each level adds scaled copies of the base pattern
        self.dna = np.zeros(64)
        
        base = np.resize(seed, 16)
        
        for level in range(depth):
            scale = 2 ** level
            freq_mult = base_freq * scale
            
            # Add base pattern at this scale
            for i, val in enumerate(base):
                idx = int(i * scale) % 64
                self.dna[idx] += val / scale  # Amplitude decreases with scale
        
        # Normalize
        if np.max(np.abs(self.dna)) > 0:
            self.dna = self.dna / np.max(np.abs(self.dna))
        
        # Visualization
        self._draw_fractal_antenna(depth)

    def _draw_fractal_antenna(self, depth):
        self.display.fill(10)
        
        cx, cy = 64, 64
        
        # Draw antenna at multiple scales
        colors = [(100, 200, 255), (100, 255, 200), (255, 200, 100), (255, 100, 200)]
        
        for level in range(depth):
            scale = 2 ** level
            radius = 20 + level * 15
            n_points = 8 * (level + 1)
            
            points = []
            for i in range(n_points):
                angle = i * 2 * np.pi / n_points
                
                # Modulate radius by DNA
                dna_idx = int(i * 64 / n_points) % 64
                r_mod = radius + self.dna[dna_idx] * 10 * (depth - level)
                
                x = int(cx + r_mod * np.cos(angle))
                y = int(cy + r_mod * np.sin(angle))
                points.append((x, y))
            
            # Draw this level
            color = colors[level % len(colors)]
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.display, [pts], True, color, 1)
        
        # Labels
        cv2.putText(self.display, f"Depth: {depth}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(self.display, f"Bands: {depth * 4}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_output(self, name):
        if name == 'fractal_dna':
            return self.dna.copy()
        if name == 'antenna_view':
            return self.display
        if name == 'bandwidth':
            # Count significant frequency components
            fft_result = np.abs(fft(self.dna))
            return float(np.sum(fft_result > 0.1))
        return None
