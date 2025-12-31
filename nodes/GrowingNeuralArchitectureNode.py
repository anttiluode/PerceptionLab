"""
Growing Neural Architecture Node
================================

A paradigm shift from "neurons that process" to "neurons that GROW".

This implements:
1. Axonal growth with growth cones following activity gradients
2. Dendritic trees that extend toward active axons
3. Signal propagation delays based on axon LENGTH
4. Ephaptic coupling between nearby axons (field effects)
5. Synaptic formation when axon meets dendrite + correlation
6. Pruning of weak connections
7. Myelination of strong connections (faster propagation)
8. Emergent layer formation from connectivity patterns
9. 3D structure where paths exist at different Z depths

The geometry IS the algorithm. Structure emerges from growth rules.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque
import json
import os

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}


# =============================================================================
# CORE STRUCTURES
# =============================================================================

class GrowthCone:
    """
    The tip of a growing axon. Senses environment and decides direction.
    Like a little autonomous agent exploring the neural space.
    """
    def __init__(self, position, parent_neuron_id):
        self.position = np.array(position, dtype=np.float32)  # 3D position
        self.parent_id = parent_neuron_id
        self.velocity = np.zeros(3, dtype=np.float32)
        self.age = 0
        self.active = True
        self.sensitivity = 1.0  # How strongly it responds to gradients
        
    def sense_gradient(self, activity_field, chemical_field, target_positions):
        """Compute direction to grow based on multiple cues."""
        gradient = np.zeros(3, dtype=np.float32)
        
        # 1. Activity gradient - grow toward active regions
        if activity_field is not None:
            pos_int = self.position.astype(int)
            pos_int = np.clip(pos_int, 1, np.array(activity_field.shape) - 2)
            
            # Sample activity in neighborhood
            for axis in range(3):
                pos_plus = pos_int.copy()
                pos_minus = pos_int.copy()
                pos_plus[axis] = min(pos_plus[axis] + 1, activity_field.shape[axis] - 1)
                pos_minus[axis] = max(pos_minus[axis] - 1, 0)
                
                grad = activity_field[tuple(pos_plus)] - activity_field[tuple(pos_minus)]
                gradient[axis] += grad * 0.5
        
        # 2. Chemical gradient (layer markers, guidance molecules)
        if chemical_field is not None:
            pos_int = self.position.astype(int)
            pos_int = np.clip(pos_int, 1, np.array(chemical_field.shape[:3]) - 2)
            
            for axis in range(3):
                pos_plus = pos_int.copy()
                pos_minus = pos_int.copy()
                pos_plus[axis] = min(pos_plus[axis] + 1, chemical_field.shape[axis] - 1)
                pos_minus[axis] = max(pos_minus[axis] - 1, 0)
                
                # Chemical field has multiple channels
                chem_grad = np.mean(chemical_field[tuple(pos_plus)] - chemical_field[tuple(pos_minus)])
                gradient[axis] += chem_grad * 0.3
        
        # 3. Target attraction - grow toward dendrites
        if target_positions and len(target_positions) > 0:
            targets = np.array(target_positions)
            distances = np.linalg.norm(targets - self.position, axis=1)
            
            # Attracted to nearby targets (inverse square, with cutoff)
            for i, (target, dist) in enumerate(zip(targets, distances)):
                if 1.0 < dist < 20.0:
                    direction = (target - self.position) / (dist + 0.1)
                    attraction = 1.0 / (dist * dist + 1.0)
                    gradient += direction * attraction * 2.0
        
        # 4. Random exploration (noise)
        gradient += np.random.randn(3) * 0.1
        
        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 0.01:
            gradient = gradient / norm
            
        return gradient * self.sensitivity
    
    def step(self, gradient, growth_rate=0.5):
        """Move the growth cone one step."""
        # Momentum
        self.velocity = 0.7 * self.velocity + 0.3 * gradient
        
        # Move
        self.position = self.position + self.velocity * growth_rate
        self.age += 1
        
        # Decay sensitivity over time (mature axons grow slower)
        self.sensitivity *= 0.999


class Axon:
    """
    A growing axon with a path, length, and propagation delay.
    The PATH is the memory. The LENGTH is the timing.
    """
    def __init__(self, soma_position, neuron_id):
        self.neuron_id = neuron_id
        self.path = [np.array(soma_position, dtype=np.float32)]
        self.growth_cone = GrowthCone(soma_position, neuron_id)
        self.synapses = []  # List of (target_neuron_id, synapse_strength)
        self.myelinated = False
        self.propagation_speed = 1.0  # units per timestep
        self.active_signals = deque()  # (signal_strength, arrival_time)
        
    @property
    def length(self):
        """Total path length."""
        total = 0.0
        for i in range(len(self.path) - 1):
            total += np.linalg.norm(self.path[i+1] - self.path[i])
        return total
    
    @property
    def delay(self):
        """Signal propagation delay based on length and myelination."""
        base_delay = self.length / self.propagation_speed
        if self.myelinated:
            return base_delay * 0.2  # Myelination speeds up 5x
        return base_delay
    
    @property
    def tip(self):
        return self.path[-1] if self.path else None
    
    def grow(self, activity_field, chemical_field, target_positions, growth_rate=0.5):
        """Extend the axon one step."""
        if not self.growth_cone.active:
            return
            
        gradient = self.growth_cone.sense_gradient(
            activity_field, chemical_field, target_positions
        )
        self.growth_cone.step(gradient, growth_rate)
        
        # Add new position to path
        self.path.append(self.growth_cone.position.copy())
        
        # Limit path length (memory constraint)
        if len(self.path) > 500:
            self.path = self.path[-500:]
    
    def send_spike(self, strength, current_time):
        """Send a spike down the axon."""
        arrival_time = current_time + self.delay
        self.active_signals.append((strength, arrival_time))
    
    def get_output(self, current_time):
        """Get signal arriving at synapses now."""
        total = 0.0
        while self.active_signals and self.active_signals[0][1] <= current_time:
            strength, _ = self.active_signals.popleft()
            total += strength
        return total
    
    def form_synapse(self, target_id, initial_strength=0.5):
        """Form a synapse with a target neuron."""
        self.synapses.append([target_id, initial_strength])
        self.growth_cone.active = False  # Stop growing once connected
    
    def myelinate(self):
        """Myelinate this axon for faster conduction."""
        self.myelinated = True
        self.propagation_speed = 5.0


class Dendrite:
    """
    A dendritic branch that receives input.
    Grows toward active axons.
    """
    def __init__(self, soma_position, neuron_id, branch_id=0):
        self.neuron_id = neuron_id
        self.branch_id = branch_id
        self.root = np.array(soma_position, dtype=np.float32)
        self.tip = self.root.copy()
        self.path = [self.root.copy()]
        self.receptive_radius = 2.0
        self.input_buffer = 0.0
        
    def grow_toward(self, axon_tips, growth_rate=0.3):
        """Grow toward nearby active axons."""
        if not axon_tips:
            return
            
        tips = np.array(axon_tips)
        distances = np.linalg.norm(tips - self.tip, axis=1)
        
        # Find nearest tip within range
        mask = distances < 15.0
        if not np.any(mask):
            # Random exploration
            direction = np.random.randn(3) * 0.5
        else:
            nearest_idx = np.argmin(distances[mask])
            nearest = tips[mask][nearest_idx]
            direction = nearest - self.tip
            direction = direction / (np.linalg.norm(direction) + 0.01)
        
        self.tip = self.tip + direction * growth_rate
        self.path.append(self.tip.copy())
        
        if len(self.path) > 100:
            self.path = self.path[-100:]
    
    def receive(self, signal):
        """Receive input signal."""
        self.input_buffer += signal
    
    def drain(self):
        """Get accumulated input and reset."""
        val = self.input_buffer
        self.input_buffer *= 0.9  # Leak
        return val


class GrowingNeuron:
    """
    A neuron with a soma, growing axon, and dendritic tree.
    Implements Izhikevich dynamics but with spatial structure.
    """
    def __init__(self, neuron_id, position, neuron_type='regular'):
        self.id = neuron_id
        self.soma = np.array(position, dtype=np.float32)
        
        # Axon (output)
        self.axon = Axon(position, neuron_id)
        
        # Dendrites (inputs) - multiple branches
        self.dendrites = [Dendrite(position, neuron_id, i) for i in range(3)]
        
        # Izhikevich parameters
        if neuron_type == 'fast':
            self.a, self.b, self.c, self.d = 0.1, 0.2, -65.0, 2.0
        elif neuron_type == 'burst':
            self.a, self.b, self.c, self.d = 0.02, 0.2, -55.0, 4.0
        else:  # regular
            self.a, self.b, self.c, self.d = 0.02, 0.2, -65.0, 8.0
        
        # State
        self.v = -65.0
        self.u = self.b * self.v
        self.spike = False
        self.spike_trace = 0.0
        self.activity = 0.0
        
        # External input
        self.I_ext = 0.0
        
    def collect_dendritic_input(self):
        """Sum input from all dendrites."""
        total = 0.0
        for dendrite in self.dendrites:
            total += dendrite.drain()
        return total
    
    def step(self, dt=0.5, current_time=0):
        """Update neuron state."""
        # Collect synaptic input
        I_syn = self.collect_dendritic_input()
        
        # Get input from axon output (for recurrent connections)
        I_axon = self.axon.get_output(current_time) * 10.0
        
        # Total current
        I = self.I_ext + I_syn + I_axon
        
        # Izhikevich dynamics
        dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I) * dt
        du = self.a * (self.b * self.v - self.u) * dt
        
        self.v += dv
        self.u += du
        
        self.v = np.clip(self.v, -100, 50)
        
        # Spike
        self.spike = self.v >= 30.0
        if self.spike:
            self.v = self.c
            self.u += self.d
            self.axon.send_spike(1.0, current_time)
            self.spike_trace = 1.0
        
        # Decay trace
        self.spike_trace *= 0.95
        self.activity = 0.9 * self.activity + 0.1 * float(self.spike)
        
        # Reset external input
        self.I_ext = 0.0
        
        return self.spike


class AxonBundle:
    """
    Multiple axons passing through the same region of space.
    Implements ephaptic coupling (electrical field effects between axons).
    """
    def __init__(self):
        self.axons = []
        self.z_depths = {}
        self.coupling_strength = 0.1
        
    def add_axon(self, axon, z_depth):
        self.axons.append(axon)
        self.z_depths[axon.neuron_id] = z_depth
    
    def get_ephaptic_coupling(self, position, z_depth, radius=3.0):
        """
        Get coupling signal from nearby axons.
        Axons passing close to each other influence each other's signals.
        """
        coupling = 0.0
        
        for axon in self.axons:
            if axon.neuron_id in self.z_depths:
                axon_z = self.z_depths[axon.neuron_id]
                z_dist = abs(z_depth - axon_z)
                
                if z_dist > radius:
                    continue
                
                # Check distance along axon path
                for point in axon.path[-50:]:  # Only recent path
                    xy_dist = np.linalg.norm(position[:2] - point[:2])
                    
                    if xy_dist < radius:
                        # Coupling falls off with distance
                        dist = np.sqrt(xy_dist**2 + z_dist**2)
                        coupling += self.coupling_strength / (dist + 0.1)
        
        return coupling


# =============================================================================
# MAIN NODE
# =============================================================================

class GrowingNeuralArchitectureNode(BaseNode):
    """
    A self-organizing neural substrate where neurons GROW their connections.
    
    The geometry emerges from:
    - Growth cone guidance (activity + chemical gradients)
    - Competitive synapse formation
    - Activity-dependent pruning
    - Myelination of strong pathways
    
    The resulting structure IS the algorithm.
    """
    
    NODE_NAME = "Growing Neural Network"
    NODE_CATEGORY = "Neural"
    NODE_COLOR = QtGui.QColor(50, 150, 100) if QtGui else None
    
    def __init__(self):
        super().__init__()
        self.node_title = "Growing Neural Network"
        
        self.inputs = {
            'image_in': 'image',
            'signal_in': 'signal',
            'growth_drive': 'signal', 
            'pruning_signal': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'structure_view': 'image',
            'activity_view': 'image',
            'axon_view': 'image',
            'output_signal': 'signal',
            'total_synapses': 'signal',
            'total_length': 'signal',
            'mean_delay': 'signal',
            'layer_count': 'signal'
        }
        
        # Configuration
        self.space_size = 64
        self.n_neurons = 200
        self.growth_rate = 0.5
        self.prune_threshold = 0.1
        self.myelination_threshold = 0.8
        
        # --- FIX: INITIALIZE BUNDLES BEFORE SUBSTRATE ---
        # Axon bundles for ephaptic coupling
        self.bundles = AxonBundle()
        
        # Initialize neural substrate
        self._init_substrate()
        # ------------------------------------------------
        
        # Fields
        self.activity_field = np.zeros((self.space_size,)*3, dtype=np.float32)
        self.chemical_field = np.zeros((self.space_size,)*3 + (3,), dtype=np.float32)
        self._init_chemical_gradients()
        
        # Simulation
        self.step_count = 0
        self.current_time = 0.0
        
        # Display
        self.display_array = None
        self.activity_display = None
        self.axon_display = None
        
        # Statistics
        self.total_synapses = 0
        self.emergent_layers = []
        
    def _init_substrate(self):
        """Initialize neurons in 3D space."""
        self.neurons = []
        
        for i in range(self.n_neurons):
            # Distribute in 3D with some structure
            # Input neurons at z=0, output at z=max
            layer_bias = i / self.n_neurons
            
            x = np.random.uniform(5, self.space_size - 5)
            y = np.random.uniform(5, self.space_size - 5)
            z = layer_bias * (self.space_size - 10) + 5 + np.random.randn() * 3
            z = np.clip(z, 0, self.space_size - 1)
            
            # Neuron type based on position
            if layer_bias < 0.3:
                ntype = 'fast'  # Input layer - fast response
            elif layer_bias > 0.7:
                ntype = 'burst'  # Output layer - bursting
            else:
                ntype = 'regular'  # Middle - regular spiking
            
            neuron = GrowingNeuron(i, [x, y, z], ntype)
            self.neurons.append(neuron)
            
            # Add to bundle
            self.bundles.add_axon(neuron.axon, z)
    
    def _init_chemical_gradients(self):
        """Initialize chemical guidance gradients."""
        # Channel 0: Vertical gradient (guides layer formation)
        for z in range(self.space_size):
            self.chemical_field[:, :, z, 0] = z / self.space_size
        
        # Channel 1: Radial gradient (guides columnar organization)
        center = self.space_size / 2
        for x in range(self.space_size):
            for y in range(self.space_size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                self.chemical_field[x, y, :, 1] = 1.0 - dist / center
        
        # Channel 2: Random patches (diversity)
        self.chemical_field[:, :, :, 2] = np.random.rand(self.space_size, self.space_size, self.space_size) * 0.3
    
    def _read_input(self, name, default=None):
        """Read an input value."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                return val if val is not None else default
            except:
                return default
        return default
    
    def _read_image_input(self, name):
        """Read an image input, converting QImage to numpy if needed."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first")
                if val is None:
                    return None
                if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                    return val
                if hasattr(val, 'width') and hasattr(val, 'height') and hasattr(val, 'bits'):
                    width, height = val.width(), val.height()
                    ptr = val.bits()
                    if ptr is None:
                        return None
                    ptr.setsize(height * val.bytesPerLine())
                    arr = np.array(ptr).reshape(height, val.bytesPerLine())
                    if val.bytesPerLine() >= width * 3:
                        arr = arr[:, :width*3].reshape(height, width, 3)
                    return arr.astype(np.float32)
            except:
                pass
        return None
    
    def step(self):
        self.step_count += 1
        self.current_time += 1.0
        
        # Read inputs
        growth = self._read_input('growth_drive', self.growth_rate)
        prune = self._read_input('pruning_signal', self.prune_threshold)
        image = self._read_image_input('image_in')
        signal = self._read_input('signal_in', 0.0)
        
        # Apply image input to "input layer" neurons (low z)
        if image is not None:
            self._apply_image_input(image)
        
        # Apply signal input
        if signal:
            for neuron in self.neurons[:20]:  # First 20 neurons get signal
                neuron.I_ext += float(signal) * 10.0
        
        # Update activity field
        self._update_activity_field()
        
        # Growth phase
        if self.step_count % 5 == 0:
            self._growth_step(growth)
        
        # Neural dynamics
        self._dynamics_step()
        
        # Synapse formation
        if self.step_count % 10 == 0:
            self._synapse_formation_step()
        
        # Pruning
        if self.step_count % 50 == 0:
            self._pruning_step(prune)
        
        # Myelination
        if self.step_count % 100 == 0:
            self._myelination_step()
        
        # Detect emergent layers
        if self.step_count % 200 == 0:
            self._detect_layers()
        
        # Update displays
        if self.step_count % 4 == 0:
            self._update_display()
    
    def _apply_image_input(self, image):
        """Apply image as input to neurons based on their position."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray.astype(np.float32), (self.space_size, self.space_size))
        gray = gray / 255.0
        
        # Input neurons (z < 15) receive image input at their x,y position
        for neuron in self.neurons:
            if neuron.soma[2] < 15:
                x, y = int(neuron.soma[0]), int(neuron.soma[1])
                x, y = np.clip(x, 0, self.space_size-1), np.clip(y, 0, self.space_size-1)
                neuron.I_ext += gray[y, x] * 20.0
    
    def _update_activity_field(self):
        """Update 3D activity field from neuron spikes."""
        self.activity_field *= 0.95  # Decay
        
        for neuron in self.neurons:
            if neuron.spike:
                pos = neuron.soma.astype(int)
                pos = np.clip(pos, 0, self.space_size - 1)
                
                # Add activity with spatial spread
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        for dz in range(-2, 3):
                            px = np.clip(pos[0] + dx, 0, self.space_size - 1)
                            py = np.clip(pos[1] + dy, 0, self.space_size - 1)
                            pz = np.clip(pos[2] + dz, 0, self.space_size - 1)
                            
                            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                            self.activity_field[px, py, pz] += np.exp(-dist) * 0.5
    
    def _growth_step(self, growth_rate):
        """Grow axons and dendrites."""
        # Collect dendrite tip positions for axon targeting
        dendrite_tips = []
        for neuron in self.neurons:
            for dendrite in neuron.dendrites:
                dendrite_tips.append(dendrite.tip)
        
        # Collect axon tip positions for dendrite targeting
        axon_tips = []
        for neuron in self.neurons:
            if neuron.axon.growth_cone.active:
                axon_tips.append(neuron.axon.tip)
        
        # Grow axons
        for neuron in self.neurons:
            neuron.axon.grow(
                self.activity_field,
                self.chemical_field,
                dendrite_tips,
                growth_rate
            )
        
        # Grow dendrites
        for neuron in self.neurons:
            for dendrite in neuron.dendrites:
                dendrite.grow_toward(axon_tips, growth_rate * 0.5)
    
    def _dynamics_step(self):
        """Update neural dynamics."""
        # First pass: compute all spikes
        for neuron in self.neurons:
            neuron.step(dt=0.5, current_time=self.current_time)
        
        # Second pass: deliver spikes through synapses
        for neuron in self.neurons:
            if neuron.spike:
                # Deliver to synaptic targets
                for target_id, strength in neuron.axon.synapses:
                    if 0 <= target_id < len(self.neurons):
                        target = self.neurons[target_id]
                        # Deliver to random dendrite
                        dendrite = np.random.choice(target.dendrites)
                        dendrite.receive(strength * 5.0)
                
                # Ephaptic coupling
                for other in self.neurons:
                    if other.id != neuron.id:
                        coupling = self.bundles.get_ephaptic_coupling(
                            neuron.soma, 
                            neuron.soma[2],
                            radius=5.0
                        )
                        other.I_ext += coupling * 0.5
    
    def _synapse_formation_step(self):
        """Form new synapses where axons meet dendrites."""
        for neuron in self.neurons:
            if not neuron.axon.growth_cone.active:
                continue
            
            axon_tip = neuron.axon.tip
            
            for target in self.neurons:
                if target.id == neuron.id:
                    continue
                
                for dendrite in target.dendrites:
                    dist = np.linalg.norm(axon_tip - dendrite.tip)
                    
                    if dist < dendrite.receptive_radius:
                        # Check correlation (STDP-like)
                        correlation = neuron.spike_trace * target.spike_trace
                        
                        if correlation > 0.1 or np.random.rand() < 0.01:
                            # Form synapse
                            neuron.axon.form_synapse(target.id, 0.5)
                            self.total_synapses += 1
                            break
    
    def _pruning_step(self, threshold):
        """Prune weak synapses."""
        for neuron in self.neurons:
            surviving = []
            for target_id, strength in neuron.axon.synapses:
                # Weaken unused synapses
                if self.neurons[target_id].activity < 0.01:
                    strength *= 0.95
                
                if strength > threshold:
                    surviving.append([target_id, strength])
                else:
                    self.total_synapses -= 1
            
            neuron.axon.synapses = surviving
    
    def _myelination_step(self):
        """Myelinate strong, active pathways."""
        for neuron in self.neurons:
            if neuron.axon.myelinated:
                continue
            
            # Myelinate if strongly connected and active
            total_strength = sum(s for _, s in neuron.axon.synapses)
            if total_strength > self.myelination_threshold and neuron.activity > 0.1:
                neuron.axon.myelinate()
    
    def _detect_layers(self):
        """Detect emergent layer structure from connectivity."""
        # Group neurons by their primary connection targets
        z_positions = [n.soma[2] for n in self.neurons]
        
        # Simple layer detection: histogram of z positions
        hist, bins = np.histogram(z_positions, bins=10)
        
        # Find peaks (layers)
        self.emergent_layers = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                layer_z = (bins[i] + bins[i+1]) / 2
                self.emergent_layers.append(layer_z)
    
    def _update_display(self):
        """Create visualizations."""
        size = 400
        
        # === Structure View: Neuron positions and axon paths ===
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        scale = size / self.space_size
        
        # Draw axon paths (color by z-depth)
        for neuron in self.neurons:
            path = neuron.axon.path
            if len(path) < 2:
                continue
            
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                
                x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
                x2, y2 = int(p2[0] * scale), int(p2[1] * scale)
                
                # Color by z-depth
                z_norm = p1[2] / self.space_size
                color = (
                    int(50 + 150 * z_norm),
                    int(200 * (1 - z_norm)),
                    int(50 + 200 * z_norm)
                )
                
                # Brighter if myelinated
                if neuron.axon.myelinated:
                    color = tuple(min(255, c + 50) for c in color)
                
                cv2.line(img, (x1, y1), (x2, y2), color, 1)
        
        # Draw somas
        for neuron in self.neurons:
            x, y = int(neuron.soma[0] * scale), int(neuron.soma[1] * scale)
            z_norm = neuron.soma[2] / self.space_size
            
            # Size by activity
            radius = 2 + int(neuron.activity * 5)
            
            # Color: input=cyan, middle=green, output=magenta
            if z_norm < 0.3:
                color = (255, 255, 0)  # Cyan (input)
            elif z_norm > 0.7:
                color = (255, 0, 255)  # Magenta (output)
            else:
                color = (0, 255, 0)  # Green (middle)
            
            cv2.circle(img, (x, y), radius, color, -1)
        
        # Info overlay
        cv2.putText(img, f"GROWING NEURAL NET", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Step: {self.step_count}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Neurons: {self.n_neurons}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Synapses: {self.total_synapses}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(img, f"Layers: {len(self.emergent_layers)}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # Calculate stats
        total_length = sum(n.axon.length for n in self.neurons)
        mean_delay = np.mean([n.axon.delay for n in self.neurons])
        myelinated = sum(1 for n in self.neurons if n.axon.myelinated)
        
        cv2.putText(img, f"Total length: {total_length:.0f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(img, f"Mean delay: {mean_delay:.1f}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(img, f"Myelinated: {myelinated}", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        self.display_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # === Activity View: 2D projection of activity ===
        activity_2d = np.max(self.activity_field, axis=2)  # Max projection
        activity_norm = np.clip(activity_2d / (activity_2d.max() + 0.01), 0, 1)
        activity_img = (activity_norm * 255).astype(np.uint8)
        activity_img = cv2.resize(activity_img, (size, size))
        activity_img = cv2.applyColorMap(activity_img, cv2.COLORMAP_INFERNO)
        self.activity_display = cv2.cvtColor(activity_img, cv2.COLOR_BGR2RGB)
        
        # === Axon View: Side view (X-Z plane) ===
        axon_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        for neuron in self.neurons:
            path = neuron.axon.path
            for i in range(len(path) - 1):
                x1 = int(path[i][0] * scale)
                z1 = int(path[i][2] * scale)
                x2 = int(path[i+1][0] * scale)
                z2 = int(path[i+1][2] * scale)
                
                color = (100, 200, 100) if not neuron.axon.myelinated else (200, 255, 200)
                cv2.line(axon_img, (x1, z1), (x2, z2), color, 1)
        
        # Draw layer lines
        for layer_z in self.emergent_layers:
            y = int(layer_z * scale)
            cv2.line(axon_img, (0, y), (size, y), (100, 100, 255), 1)
        
        cv2.putText(axon_img, "X-Z SIDE VIEW", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.axon_display = cv2.cvtColor(axon_img, cv2.COLOR_BGR2RGB)
    
    def get_output(self, port_name):
        if port_name == 'structure_view':
            return self.display_array
        elif port_name == 'activity_view':
            return self.activity_display
        elif port_name == 'axon_view':
            return self.axon_display
        elif port_name == 'output_signal':
            # Mean activity of output layer neurons
            output_neurons = [n for n in self.neurons if n.soma[2] > self.space_size * 0.7]
            if output_neurons:
                return float(np.mean([n.activity for n in output_neurons]))
            return 0.0
        elif port_name == 'total_synapses':
            return float(self.total_synapses)
        elif port_name == 'total_length':
            return float(sum(n.axon.length for n in self.neurons))
        elif port_name == 'mean_delay':
            return float(np.mean([n.axon.delay for n in self.neurons]))
        elif port_name == 'layer_count':
            return float(len(self.emergent_layers))
        return None
    
    def get_display_image(self):
        if self.display_array is not None and QtGui:
            h, w = self.display_array.shape[:2]
            return QtGui.QImage(self.display_array.data, w, h, w * 3,
                              QtGui.QImage.Format.Format_RGB888).copy()
        return None
    
    def get_config_options(self):
        return [
            ("Number of Neurons", "n_neurons", self.n_neurons, None),
            ("Space Size", "space_size", self.space_size, None),
            ("Growth Rate", "growth_rate", self.growth_rate, None),
            ("Prune Threshold", "prune_threshold", self.prune_threshold, None),
            ("Myelination Threshold", "myelination_threshold", self.myelination_threshold, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            reinit = False
            for key, value in options.items():
                if key in ['n_neurons', 'space_size'] and hasattr(self, key):
                    if getattr(self, key) != value:
                        reinit = True
                if hasattr(self, key):
                    setattr(self, key, value)
            
            if reinit:
                self._init_substrate()
                self.activity_field = np.zeros((self.space_size,)*3, dtype=np.float32)
                self.chemical_field = np.zeros((self.space_size,)*3 + (3,), dtype=np.float32)
                self._init_chemical_gradients()
    
    def save_structure(self, filepath):
        """Save the grown structure to file."""
        data = {
            'n_neurons': self.n_neurons,
            'space_size': self.space_size,
            'neurons': []
        }
        
        for neuron in self.neurons:
            ndata = {
                'id': neuron.id,
                'soma': neuron.soma.tolist(),
                'axon_path': [p.tolist() for p in neuron.axon.path],
                'synapses': neuron.axon.synapses,
                'myelinated': neuron.axon.myelinated,
                'type': 'regular'  # Could store actual type
            }
            data['neurons'].append(ndata)
        
        data['emergent_layers'] = self.emergent_layers
        data['total_synapses'] = self.total_synapses
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_structure(self, filepath):
        """Load a previously grown structure."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.n_neurons = data['n_neurons']
        self.space_size = data['space_size']
        
        self.neurons = []
        for ndata in data['neurons']:
            neuron = GrowingNeuron(ndata['id'], ndata['soma'], ndata.get('type', 'regular'))
            neuron.axon.path = [np.array(p) for p in ndata['axon_path']]
            neuron.axon.synapses = ndata['synapses']
            if ndata['myelinated']:
                neuron.axon.myelinate()
            neuron.axon.growth_cone.active = False
            self.neurons.append(neuron)
        
        self.emergent_layers = data.get('emergent_layers', [])
        self.total_synapses = data.get('total_synapses', 0)