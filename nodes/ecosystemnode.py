"""
Ecosystem Node (The Eigenmode Game of Life)
-------------------------------------------
Simulates a population of "Genesis Loops" interacting in a shared Quantum Field.

Each Agent is a minimal Self-Organizing Observer:
1. Sensation: Samples the Quantum Field at its (x,y) location.
2. Prediction: Uses a Hebbian predictor to guess the next field state.
3. Action (Movement): High Surprise -> Velocity (Flee chaos).
4. Growth (Structure): Low Surprise -> Accumulate Mass (Crystallize).

Visuals:
- Agents are drawn as growing geometric forms (Eigenmodes).
- Shape depends on their internal stability state.
- Color depends on their prediction error (Red=Panic, Blue=Flow).
"""

import numpy as np
import cv2
import random
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class EcosystemNode(BaseNode):
    NODE_CATEGORY = "Simulation"
    NODE_COLOR = QtGui.QColor(46, 204, 113) # Emerald Green

    def __init__(self):
        super().__init__()
        self.node_title = "Ecosystem: Eigenmode Life"
        
        self.inputs = {
            'field_input': 'image',      # The Shared World (Quantum Substrate)
            'global_stress': 'signal'    # Global catastrophe/energy knob
        }
        
        self.outputs = {
            'population_view': 'image',  # The Petri Dish view
            'total_biomass': 'signal',   # Total structure grown
            'avg_surprise': 'signal'     # System-wide free energy
        }
        
        self.width = 512
        self.height = 512
        self.num_agents = 64
        
        # --- Initialize Population ---
        # Agents are dictionaries for performance
        self.agents = []
        for _ in range(self.num_agents):
            self.spawn_agent()
            
        self.display_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.biomass = 0.0
        self.avg_error = 0.0

    def spawn_agent(self, parent=None):
        """Creates a new Genesis Loop Agent"""
        if parent:
            # Evolution: Copy parent with mutation
            x, y = parent['x'] + np.random.randn()*10, parent['y'] + np.random.randn()*10
            params = parent['params'] * (1.0 + np.random.randn()*0.1) # Mutate genes
        else:
            # Abiogenesis: Random spawn
            x, y = np.random.rand() * self.width, np.random.rand() * self.height
            params = np.array([0.05, 0.95, 0.1]) # [Learning Rate, Momentum, Growth Rate]

        agent = {
            'x': np.clip(x, 0, self.width),
            'y': np.clip(y, 0, self.height),
            'vx': 0.0, 'vy': 0.0,
            'prediction': 0.0,     # Internal Model
            'mass': 1.0,           # Physical Structure (Thickness)
            'age': 0,
            'params': params,      # DNA
            'eigenmode': (random.randint(1,4), random.randint(0,3)) # (n, m) Shape Identity
        }
        self.agents.append(agent)

    def step(self):
        # 1. Get Environment
        field = self.get_blended_input('field_input', 'mean')
        stress_mod = self.get_blended_input('global_stress', 'sum') or 0.0
        
        if field is None:
            # Fallback if no input connected
            field = np.zeros((self.height, self.width), dtype=np.float32)
            
        # Resize field to match simulation if needed
        if field.shape[:2] != (self.height, self.width):
            field = cv2.resize(field, (self.width, self.height))
        if field.ndim == 3:
            field = np.mean(field, axis=2)

        # Clear canvas (with trails)
        self.display_img = cv2.addWeighted(self.display_img, 0.9, np.zeros_like(self.display_img), 0.1, 0)
        
        current_biomass = 0.0
        total_error = 0.0
        new_agents = []
        dead_agents = []

        # 2. Update Each Organism
        for i, a in enumerate(self.agents):
            # --- SENSATION ---
            # Sample the field at agent's location
            ix, iy = int(a['x']), int(a['y'])
            # Wrap coords
            ix = ix % self.width
            iy = iy % self.height
            
            sensory_input = float(field[iy, ix])
            
            # --- COGNITION (The Observer Loop) ---
            # 1. Calculate Surprise
            error = abs(sensory_input - a['prediction'])
            total_error += error
            
            # 2. Update Prediction (Hebbian Learning)
            # learning_rate = gene[0]
            lr = a['params'][0] * (1.0 + error) # Plasticity increases with surprise
            a['prediction'] += lr * (sensory_input - a['prediction'])
            
            # --- ACTION (Skin in the Game) ---
            # High Error -> High Mobility (Search/Flee)
            # Low Error -> Low Mobility (Settle)
            drive = error * 50.0 + stress_mod
            
            # Random walk biased by error gradient would be better, 
            # but here we just convert panic into velocity
            angle = np.random.rand() * 2 * np.pi
            a['vx'] = a['vx'] * 0.9 + np.cos(angle) * drive
            a['vy'] = a['vy'] * 0.9 + np.sin(angle) * drive
            
            a['x'] = (a['x'] + a['vx']) % self.width
            a['y'] = (a['y'] + a['vy']) % self.height
            
            # --- MORPHOGENESIS (Growth) ---
            # If error is LOW, we are in a stable niche -> GROW
            # If error is HIGH, we are stressed -> SHRINK/METABOLIZE
            
            metabolic_cost = 0.01 + (drive * 0.001)
            growth_potential = (0.1 - error) * a['params'][2] # Growth Rate gene
            
            if error < 0.1:
                # Stable Resonance! Crystallize!
                a['mass'] += growth_potential
            else:
                # Instability! Atrophy!
                a['mass'] -= metabolic_cost * 2.0
                
            current_biomass += a['mass']
            a['age'] += 1
            
            # --- VISUALIZATION (Render the Eigenmode) ---
            # Draw the agent based on its unique (n, m) symmetry
            radius = int(np.log1p(a['mass']) * 5)
            if radius < 1: radius = 1
            
            color_val = int(np.clip(1.0 - error*5, 0, 1) * 255)
            # Blue = Stable/Happy, Red = Panicked/Surprised
            color = (color_val, 50, 255 - color_val) 
            
            # Simple visual representation of eigenmode n (rings)
            cv2.circle(self.display_img, (int(a['x']), int(a['y'])), radius, color, -1)
            if a['eigenmode'][0] > 1:
                cv2.circle(self.display_img, (int(a['x']), int(a['y'])), radius//2, (0,0,0), 1)
            
            # --- EVOLUTION & DEATH ---
            if a['mass'] <= 0.1:
                dead_agents.append(i)
            elif a['mass'] > 10.0 and len(self.agents) < 200:
                # Mitosis!
                a['mass'] *= 0.5 # Split mass
                new_agents.append(a) # Add child
                
        # Process births and deaths
        for idx in sorted(dead_agents, reverse=True):
            self.agents.pop(idx)
        for parent in new_agents:
            self.spawn_agent(parent)
            
        # Repopulate if extinction event
        if len(self.agents) < 10:
            self.spawn_agent()

        self.biomass = current_biomass
        self.avg_error = total_error / (len(self.agents) + 1e-9)
        
    def get_output(self, port_name):
        if port_name == 'population_view':
            return self.display_img.astype(np.float32) / 255.0
        elif port_name == 'total_biomass':
            return float(self.biomass)
        elif port_name == 'avg_surprise':
            return float(self.avg_error)
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, self.width, self.height, 3*self.width, QtGui.QImage.Format.Format_RGB888)