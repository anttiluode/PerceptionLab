"""
Dendritic Web Node - Digital Biology
====================================
Moving beyond "Point Neurons" to "Spatial Trees."

1. Structure: Grows a fractal network of Somas, Axons, and Dendrites.
2. Membrane: Signals travel electrically INSIDE trees, chemically OUTSIDE.
3. Quantization: Neurotransmitters are discrete integers ("Real Bits").

"The thought is the spark. The feeling is the molecule."
"""

import numpy as np
import cv2
from scipy.ndimage import convolve
import random

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class DendriticWebNode(BaseNode):
    NODE_CATEGORY = "Biology"
    NODE_TITLE = "Dendritic Web"
    NODE_COLOR = QtGui.QColor(100, 180, 120)  # Organic Green
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'stimulation': 'image',     # External electrical shock
            'chemical_bath': 'image',   # External drug/chemical bath
            'regrow': 'signal'          # Signal > 0.5 triggers regrowth
        }
        
        self.outputs = {
            'membrane_potential': 'image', # Electrical state (Internal)
            'neurotransmitter_map': 'image',# Chemical state (External)
            'structure_map': 'image',       # Anatomy (Where the trees are)
            'firing_event': 'signal'        # Global spike count
        }
        
        self.size = 128
        
        # === ANATOMY ===
        # 0=Void, 1=Soma, 2=Axon, 3=Dendrite
        self.anatomy = np.zeros((self.size, self.size), dtype=np.uint8)
        self.neuron_id_map = np.zeros((self.size, self.size), dtype=np.int32) # Which neuron owns this pixel?
        
        # === PHYSIOLOGY ===
        # Electrical (Float): Exists mainly inside the anatomy
        self.voltage = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Chemical (Integer): "Real Molecular Bits" floating in the void
        self.transmitters = np.zeros((self.size, self.size), dtype=np.float32) # Using float for smooth diffusion, but treated as packets
        
        # Receptor State (Integer): How many molecules are currently bound
        self.receptors_bound = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === PARAMETERS ===
        self.n_neurons = 12
        self.diffusion_rate = 0.8
        self.decay_rate = 0.05
        self.vesicle_packet_size = 5.0 # How much 'stuff' releases per spike
        self.binding_affinity = 0.2    # How easily dendrites catch molecules
        self.action_threshold = 0.8
        
        self.needs_growth = True
        
    def grow_network(self):
        """Fractal growth algorithm to build the trees."""
        self.anatomy.fill(0)
        self.neuron_id_map.fill(-1)
        self.voltage.fill(0)
        
        somas = []
        
        # 1. Plant Somas (Cell Bodies)
        for i in range(self.n_neurons):
            rx = random.randint(10, self.size-10)
            ry = random.randint(10, self.size-10)
            # Ensure spacing
            if self.anatomy[ry, rx] == 0:
                # Draw Soma (3x3 blob)
                self.anatomy[ry-1:ry+2, rx-1:rx+2] = 1
                self.neuron_id_map[ry-1:ry+2, rx-1:rx+2] = i
                somas.append((rx, ry, i))
        
        # 2. Grow Axons (Outputs - Long, thin wires)
        for sx, sy, nid in somas:
            curr_x, curr_y = sx, sy
            # Random direction
            angle = random.uniform(0, 6.28)
            length = random.randint(15, 40)
            
            for _ in range(length):
                curr_x += np.cos(angle)
                curr_y += np.sin(angle)
                
                ix, iy = int(curr_x), int(curr_y)
                if 0 <= ix < self.size and 0 <= iy < self.size:
                    if self.anatomy[iy, ix] == 0:
                        self.anatomy[iy, ix] = 2 # Axon
                        self.neuron_id_map[iy, ix] = nid
                    
                    # occasional branching
                    if random.random() < 0.1:
                        angle += random.uniform(-0.5, 0.5)
                else:
                    break

        # 3. Grow Dendrites (Inputs - Bushy, surrounding Soma)
        for sx, sy, nid in somas:
            grow_points = [(sx, sy)]
            for _ in range(80): # Mass of dendrites
                if not grow_points: break
                
                # Pick a random point to grow from
                idx = random.randint(0, len(grow_points)-1)
                gx, gy = grow_points[idx]
                
                # Try neighbors
                dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                nx, ny = gx+dx, gy+dy
                
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.anatomy[ny, nx] == 0:
                        self.anatomy[ny, nx] = 3 # Dendrite
                        self.neuron_id_map[ny, nx] = nid
                        grow_points.append((nx, ny))
                    elif self.anatomy[ny, nx] == 3 and self.neuron_id_map[ny, nx] == nid:
                        # Sometimes branch from existing dendrite
                        if random.random() < 0.2:
                            grow_points.append((nx, ny))
                            
        self.needs_growth = False

    def step(self):
        # Handle regrow signal
        regrow = self.get_blended_input('regrow', 'sum')
        if regrow is not None and regrow > 0.5:
            self.needs_growth = True
            
        if self.needs_growth:
            self.grow_network()
            return

        # === 1. EXTERNAL INPUT ===
        stim = self.get_blended_input('stimulation', 'sum')
        if stim is not None:
            # Stimulate Somas directly
            mask = (self.anatomy == 1)
            # Use resize to match shape if needed, simplistic here:
            if isinstance(stim, np.ndarray) and stim.shape == self.voltage.shape:
                self.voltage[mask] += stim[mask] * 0.5

        # === 2. ELECTRICAL PHYSICS (Cable Theory Lite) ===
        # Charge equalizes along the tree instantly (simplified)
        # But we iterate to simulate propagation speed
        
        # Simple diffusion of voltage, but MASKED by anatomy
        # Voltage only flows where anatomy > 0
        v_diffused = convolve(self.voltage, [[0,1,0],[1,0,1],[0,1,0]], mode='constant') / 4.0
        
        # Apply anatomy mask: Charge cannot exist in the void (0)
        # Charge moves from High to Low within the same neuron
        
        # Update Somas and Axons and Dendrites
        # (In reality, dendrites flow TO soma, Axons flow FROM soma. 
        # Here we just let it diffuse for visual coherence)
        structure_mask = (self.anatomy > 0)
        self.voltage[structure_mask] = (self.voltage[structure_mask] * 0.5) + (v_diffused[structure_mask] * 0.5)
        
        # Decay
        self.voltage *= 0.9
        
        # === 3. RELEASE MECHANISM (The Vesicle Pop) ===
        # Axon tips (Anatomy=2) that have High Voltage release Chemicals
        # Detect Axon Tips: Axons with empty neighbors
        # For speed, we just say any Axon pixel with voltage > Threshold releases
        firing_mask = (self.anatomy == 2) & (self.voltage > self.action_threshold)
        
        # Release Packets into the void
        # We add to the transmitter grid at these locations
        self.transmitters[firing_mask] += self.vesicle_packet_size
        
        # Reset voltage of fired axon (Refractory)
        self.voltage[firing_mask] = -0.5 
        
        # === 4. CHEMICAL PHYSICS (The Void) ===
        # Transmitters diffuse into the empty space
        # This is the "Liquid" you liked
        
        # Box blur for diffusion
        t_diffused = convolve(self.transmitters, [[1,1,1],[1,0,1],[1,1,1]], mode='constant') / 8.0
        self.transmitters = (self.transmitters * (1.0 - self.diffusion_rate)) + (t_diffused * self.diffusion_rate)
        
        # Decay (Enzymatic breakdown)
        self.transmitters *= (1.0 - self.decay_rate)
        
        # === 5. RECEPTION (The Counter) ===
        # Dendrites (Anatomy=3) detect local transmitters
        dendrite_mask = (self.anatomy == 3)
        
        # Binding: Amount of chemical * Affinity
        bound = self.transmitters * self.binding_affinity
        
        # Only counting what touches dendrites
        reception_events = np.zeros_like(self.voltage)
        reception_events[dendrite_mask] = bound[dendrite_mask]
        
        # Convert Chemical Binding -> Electrical Charge
        # EPSP (Excitatory Post-Synaptic Potential)
        self.voltage[dendrite_mask] += reception_events[dendrite_mask]
        
        # Consumption: Binding removes chemicals from the void
        self.transmitters[dendrite_mask] *= 0.5 

    def get_output(self, port_name):
        if port_name == 'membrane_potential':
            return (np.clip(self.voltage, 0, 1) * 255).astype(np.uint8)
        elif port_name == 'neurotransmitter_map':
            return (np.clip(self.transmitters * 5, 0, 1) * 255).astype(np.uint8)
        elif port_name == 'structure_map':
            # Visualizing the anatomy
            # Void=Black, Soma=White, Axon=Red, Dendrite=Green (in logic, mapped to gray here)
            return (self.anatomy * 60).astype(np.uint8)
        elif port_name == 'firing_event':
            return float(np.sum(self.voltage > self.action_threshold))
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 1. Draw Chemical Void (Blue Mist)
        chem = np.clip(self.transmitters * 10, 0, 1)
        display[:,:,0] = (chem * 200).astype(np.uint8) # Blue
        
        # 2. Draw Anatomy
        # Somas (White)
        soma_mask = (self.anatomy == 1)
        display[soma_mask] = [255, 255, 255]
        
        # Axons (Red)
        axon_mask = (self.anatomy == 2)
        display[axon_mask] = [50, 50, 200] # BGR Red
        
        # Dendrites (Green)
        dend_mask = (self.anatomy == 3)
        display[dend_mask] = [50, 200, 50] # BGR Green
        
        # 3. Draw Electrical Activity (Yellow Lightning)
        # Overlay bright yellow where voltage is high
        active_mask = (self.voltage > 0.2)
        intensity = np.clip(self.voltage[active_mask], 0, 1)
        
        # Add to existing colors
        display[active_mask, 1] = np.clip(display[active_mask, 1] + (intensity * 255), 0, 255) # G
        display[active_mask, 2] = np.clip(display[active_mask, 2] + (intensity * 255), 0, 255) # R
        # R+G = Yellow
        
        return QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("N Neurons", "n_neurons", self.n_neurons, None),
            ("Diffusion", "diffusion_rate", self.diffusion_rate, None),
            ("Packet Size", "vesicle_packet_size", self.vesicle_packet_size, None),
            ("Threshold", "action_threshold", self.action_threshold, None),
            ("Regrow", "regrow", False, "bool")
        ]