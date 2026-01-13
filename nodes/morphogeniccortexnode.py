import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class MorphogenicCortexNode(BaseNode):
    """
    Morphogenic Cortex - The Rewiring Brain
    =======================================
    
    PROOF OF CONCEPT:
    Integrates 'killer_app6.py' (Morphogenesis) with 'reflexivenode.py' (Error).
    
    Mechanism:
    1. Receives 'prediction_error' from the Thalamic/Reflexive node.
    2. If Error > Threshold ("Surprise"), it enters PANIC MODE.
    3. PANIC MODE triggers 'Morphogenesis':
       - Kills weak Long-Range connections (Pruning).
       - Sprouts new random connections (Growth).
       - Boosts plasticity (Learning Rate).
       
    This physical rewiring manifests as the "Fractal Glitch" in the visual field.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Morphogenic Cortex"
    NODE_COLOR = QtGui.QColor(255, 100, 100)  # Plastic Red
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'stimulus_input': 'image',      # From Source/Sensors
            'prediction_error': 'signal',   # From Reflexive Node (The "Panic" Signal)
            'reset': 'signal'
        }
        
        self.outputs = {
            'field_state': 'image',         # To Reflexive Node
            'wiring_map': 'image',          # Visualizing the connections
            'plasticity_state': 'signal',   # 0=Crystal, 1=Liquid
            'connection_count': 'signal'
        }
        
        self.size = 64
        
        # --- NEURAL SUBSTRATE ---
        self.potential = np.zeros((self.size, self.size), dtype=np.float32)
        self.spikes = np.zeros((self.size, self.size), dtype=np.float32)
        
        # --- MORPHOGENIC WIRING (The "L" and "R" form dynamic_brain.py) ---
        # Instead of fixed geometry, we have a list of long-range connections
        self.n_connections = 1000
        self.max_connections = 2000
        
        # Connection Lists (Source Y, Source X, Dest Y, Dest X, Weight, Age)
        self.conn_src_y = np.random.randint(0, self.size, self.n_connections)
        self.conn_src_x = np.random.randint(0, self.size, self.n_connections)
        self.conn_dst_y = np.random.randint(0, self.size, self.n_connections)
        self.conn_dst_x = np.random.randint(0, self.size, self.n_connections)
        self.conn_weight = np.ones(self.n_connections, dtype=np.float32) * 0.5
        self.conn_age = np.zeros(self.n_connections, dtype=np.float32)
        
        # Panic State
        self.panic_threshold = 0.05
        self.is_panicking = False
        self.plasticity = 0.01

    def step(self):
        # 1. Get Inputs
        stim_in = self.get_blended_input('stimulus_input', 'first')
        error_sig = self.get_blended_input('prediction_error', 'mean')
        
        # 2. Process Prediction Error (The "Surprise")
        current_error = error_sig if error_sig is not None else 0.0
        
        # HYSTERESIS: Enter panic easily, leave slowly
        if current_error > self.panic_threshold:
            self.is_panicking = True
            self.plasticity = 0.1  # High learning rate during panic
        elif current_error < (self.panic_threshold * 0.5):
            self.is_panicking = False
            self.plasticity = 0.001 # Crystalize when stable
            
        # 3. MORPHOGENESIS (The "Killer App" Logic)
        if self.is_panicking:
            self._run_morphogenesis()
            
        # 4. NEURAL DYNAMICS (Field Physics)
        # Decay
        self.potential *= 0.9
        
        # Input Drive
        if stim_in is not None:
            if stim_in.shape[:2] != (self.size, self.size):
                stim_in = cv2.resize(stim_in, (self.size, self.size))
            if stim_in.ndim == 3: stim_in = np.mean(stim_in, axis=2)
            self.potential += stim_in * 0.1
            
        # Long-Range Transmission (The Wiring)
        # Gather spikes from sources
        src_vals = self.spikes[self.conn_src_y, self.conn_src_x]
        # Transmit to destinations (weighted)
        transmission = src_vals * self.conn_weight
        np.add.at(self.potential, (self.conn_dst_y, self.conn_dst_x), transmission)
        
        # Hebbian Learning (Update weights based on success)
        # If Source fired AND Dest fired -> Strengthen
        dst_vals = self.spikes[self.conn_dst_y, self.conn_dst_x]
        hebbian = src_vals * dst_vals
        
        if self.is_panicking:
            # In panic, we learn fast
            self.conn_weight += hebbian * self.plasticity
        else:
            # In stability, we just drift slowly
            self.conn_weight += hebbian * self.plasticity * 0.1
            
        # Weight Decay (The Reaper)
        self.conn_weight *= 0.99
        self.conn_weight = np.clip(self.conn_weight, 0.0, 2.0)
        
        # Fire
        self.spikes = (self.potential > 0.8).astype(np.float32)
        self.potential[self.potential > 0.8] = 0.0 # Reset
        
        # Age connections
        self.conn_age += 1

    def _run_morphogenesis(self):
        """
        The Rewiring Logic:
        1. Identify weak connections (low weight).
        2. Kill them (Death).
        3. Spawn new random connections (Search).
        """
        # Kill bottom 5% of connections
        kill_count = int(self.n_connections * 0.05)
        if kill_count < 1: return
        
        # Sort by weight
        sorted_indices = np.argsort(self.conn_weight)
        
        # The victims are the ones with lowest weights (useless predictions)
        # We effectively "respawn" them by randomizing their coordinates
        victims = sorted_indices[:kill_count]
        
        self.conn_src_y[victims] = np.random.randint(0, self.size, kill_count)
        self.conn_src_x[victims] = np.random.randint(0, self.size, kill_count)
        self.conn_dst_y[victims] = np.random.randint(0, self.size, kill_count)
        self.conn_dst_x[victims] = np.random.randint(0, self.size, kill_count)
        
        # Reset their stats
        self.conn_weight[victims] = 0.5 # Give them a chance
        self.conn_age[victims] = 0
        
        # Grow: Occasionally add NEW connections if error is VERY high
        if len(self.conn_weight) < self.max_connections:
            # Add 10 new wires
            new_n = 10
            self.conn_src_y = np.append(self.conn_src_y, np.random.randint(0, self.size, new_n))
            self.conn_src_x = np.append(self.conn_src_x, np.random.randint(0, self.size, new_n))
            self.conn_dst_y = np.append(self.conn_dst_y, np.random.randint(0, self.size, new_n))
            self.conn_dst_x = np.append(self.conn_dst_x, np.random.randint(0, self.size, new_n))
            self.conn_weight = np.append(self.conn_weight, np.ones(new_n)*0.5)
            self.conn_age = np.append(self.conn_age, np.zeros(new_n))
            self.n_connections += new_n

    def get_output(self, port_name):
        if port_name == 'field_state':
            return (self.potential * 255).astype(np.uint8)
            
        elif port_name == 'wiring_map':
            # Visualize the long-range connections
            img = np.zeros((self.size, self.size), dtype=np.float32)
            # Draw lines? Too slow. Draw dots at destinations weighted by incoming
            np.add.at(img, (self.conn_dst_y, self.conn_dst_x), self.conn_weight)
            norm = img / (np.max(img) + 1e-10)
            return (norm * 255).astype(np.uint8)
            
        elif port_name == 'plasticity_state':
            return float(self.plasticity * 1000) # Scale up for viz
            
        elif port_name == 'connection_count':
            return float(self.n_connections)
            
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        display = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Left: Activity (The "Thinking")
        act = (self.potential * 255).astype(np.uint8)
        display[:, :w] = cv2.applyColorMap(act, cv2.COLORMAP_VIRIDIS)
        
        # Right: Wiring (The "Structure")
        wiring = np.zeros((h, w), dtype=np.float32)
        np.add.at(wiring, (self.conn_dst_y, self.conn_dst_x), self.conn_weight)
        wiring = wiring / (np.max(wiring) + 1e-10)
        w_img = (wiring * 255).astype(np.uint8)
        
        # Color code based on Panic State
        # Blue = Crystal/Stable, Red = Panic/Rewiring
        if self.is_panicking:
            display[:, w:] = cv2.applyColorMap(w_img, cv2.COLORMAP_HOT)
            status = "PANIC: REWIRING"
        else:
            display[:, w:] = cv2.applyColorMap(w_img, cv2.COLORMAP_OCEAN)
            status = "STABLE: CRYSTAL"
            
        cv2.putText(display, status, (w+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(display.data, w*2, h, (w*2)*3, QtGui.QImage.Format.Format_RGB888)