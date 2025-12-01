"""
Neurotransmitter Resonance Node - The Chemical Underbelly
=======================================================
Extends the Gated Resonance Node with a simulated chemical layer.

Two new dynamics:
1. "The Counter" (Vesicle Depletion): Neurons consume fuel to fire. 
   Over-activity leads to exhaustion (depression).
2. "The Cloud" (Volume Transmission): Firing releases chemical signals 
   that diffuse slowly, creating a "mood" that modulates local thresholds.

"The electricity is the thought. The chemical is the feeling."
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class NeurotransmitterResonanceNode(BaseNode):
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Neurotransmitter Resonance"
    NODE_COLOR = QtGui.QColor(50, 180, 100)  # Chemical Green
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'frequency_input': 'spectrum',      # Electrical Drive
            'reset': 'signal'
        }
        
        self.outputs = {
            'potential_map': 'image',           # Fast (Electrical)
            'chemical_map': 'image',            # Slow (Chemical)
            'vesicle_map': 'image',             # Internal Health (Metabolism)
            'eigen_image': 'image',             # Structure
            'mean_chemical': 'signal'           # Global Mood
        }
        
        self.size = 128
        
        # === LAYER 1: ELECTRICAL (The Wiring) ===
        self.potential = np.zeros((self.size, self.size), dtype=np.float32)
        self.spikes = np.zeros((self.size, self.size), dtype=np.float32)
        self.spike_history = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === LAYER 2: CHEMICAL (The Underbelly) ===
        # Vesicle Count: 1.0 = Full Tank, 0.0 = Empty (Cannot fire)
        self.vesicles = np.ones((self.size, self.size), dtype=np.float32)
        
        # Chemical Field: The "Mood" floating in the extracellular space
        self.chemical_field = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === PARAMETERS ===
        # Electrical
        self.threshold = 0.7
        self.coupling = 0.2
        self.leak = 0.1
        
        # Chemical Costs (The "Counter")
        self.fire_cost = 0.15      # How much fuel a spike costs
        self.recovery_rate = 0.01  # How fast neurons recharge
        
        # Volume Transmission (The "Cloud")
        self.release_amount = 0.05 # How much chemical is dumped per spike
        self.chemical_decay = 0.02 # How fast the cloud dissipates
        self.diffusion_rate = 1.5  # How far the cloud spreads
        self.inhibition_strength = 0.5 # How much the cloud suppresses neighbors
        
        # Wiring Kernel (Standard 8-neighbor)
        self.kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1,  0.0, 0.1],
            [0.05, 0.1, 0.05]
        ], dtype=np.float32)
        
        # Time
        self.t = 0

    def step(self):
        self.t += 1
        
        # 1. Inputs
        freq_in = self.get_blended_input('frequency_input', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.potential[:] = 0
            self.chemical_field[:] = 0
            self.vesicles[:] = 1.0
            return

        # 2. Update Chemical Physics (Slow Layer)
        # Recharge vesicles (Metabolism)
        self.vesicles = np.clip(self.vesicles + self.recovery_rate, 0, 1.0)
        
        # Diffuse the Chemical Cloud (Volume Transmission)
        # This blurs the field, simulating diffusion through tissue
        self.chemical_field = gaussian_filter(self.chemical_field, sigma=self.diffusion_rate)
        
        # Decay the cloud
        self.chemical_field *= (1.0 - self.chemical_decay)
        
        # 3. Electrical Dynamics (Fast Layer)
        # Calculate Input from neighbors (Wiring)
        from scipy.ndimage import convolve
        neighbor_input = convolve(self.spikes, self.kernel, mode='wrap')
        
        # Apply External Drive (if any)
        drive = 0
        if freq_in is not None and len(freq_in) > 0:
            # Simple projection for demo
            drive = np.mean(freq_in) * 0.1
        
        # 4. The Interaction (Where Chemistry meets Electricity)
        # The Chemical Cloud acts as INHIBITION (Turning off branches)
        # High chemical = Higher Threshold = Harder to fire
        effective_threshold = self.threshold + (self.chemical_field * self.inhibition_strength)
        
        # Update Potential
        self.potential *= (1.0 - self.leak)
        self.potential += (neighbor_input * self.coupling) + drive
        
        # Add a tiny bit of noise to prevent deadlock
        self.potential += np.random.uniform(-0.01, 0.01, self.potential.shape)
        
        # 5. Firing Logic (The Counter)
        # A neuron can only fire if it exceeds threshold AND has enough vesicles
        fire_mask = (self.potential > effective_threshold) & (self.vesicles > self.fire_cost)
        
        # Execute Fire
        self.spikes[:] = 0
        self.spikes[fire_mask] = 1.0
        self.potential[fire_mask] = 0 # Reset potential
        
        # 6. Chemical Consequences
        # Pay the cost (Deplete internal counter)
        self.vesicles[fire_mask] -= self.fire_cost
        
        # Release the signal (Add to external cloud)
        self.chemical_field[fire_mask] += self.release_amount
        
        # Update history
        self.spike_history = self.spike_history * 0.9 + self.spikes * 0.1

    def get_output(self, port_name):
        if port_name == 'potential_map':
            return (self.potential * 255).astype(np.uint8)
        elif port_name == 'chemical_map':
            # Normalize for display
            chem = np.clip(self.chemical_field * 5, 0, 1)
            return (chem * 255).astype(np.uint8)
        elif port_name == 'vesicle_map':
            return (self.vesicles * 255).astype(np.uint8)
        elif port_name == 'eigen_image':
             spec = np.abs(fftshift(fft2(self.spike_history)))
             spec = np.log(1 + spec)
             if spec.max() > 0: spec /= spec.max()
             return (spec * 255).astype(np.uint8)
        elif port_name == 'mean_chemical':
            return float(np.mean(self.chemical_field))
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # Create a layered visualization
        # Base: Potential (Grayscale)
        base = np.clip(self.potential, 0, 1) * 255
        img = np.dstack((base, base, base)).astype(np.uint8)
        
        # Overlay: Chemical Cloud (Blue/Purple mist)
        chem_vis = np.clip(self.chemical_field * 4, 0, 1) # Boost contrast
        
        # Apply Blue tint proportional to chemical concentration
        # R stays same, G reduces, B increases
        img[:,:,0] = np.clip(img[:,:,0] * (1 - chem_vis*0.5), 0, 255) # Blue channel (OpenCV is BGR)
        img[:,:,1] = np.clip(img[:,:,1] * (1 - chem_vis), 0, 255)     # Green channel
        img[:,:,2] = np.clip(img[:,:,2] + (chem_vis * 150), 0, 255)   # Red channel (Actually Blue in BGR... wait, CV2 is BGR)
        # Correct logic for CV2 BGR:
        # We want Blue mist. So increase B (0), decrease R (2) and G (1)
        
        # Let's do a robust mix:
        # Electrical = White/Yellow spikes
        # Chemical = Blue fog
        # Vesicle Depletion = Red warning
        
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Blue Channel: Chemical Cloud + Spikes
        display[:,:,0] = np.clip((self.chemical_field * 400) + (self.spikes * 255), 0, 255)
        
        # Green Channel: Spikes + Vesicle Health
        # If vesicles are full (1.0), this is bright. If empty, dark.
        display[:,:,1] = np.clip((self.spikes * 255) + (self.vesicles * 30), 0, 255)
        
        # Red Channel: Spikes + Pain (Low Vesicles)
        # If vesicles are low (<0.2), glow red
        exhaustion = np.clip((0.2 - self.vesicles) * 5, 0, 1)
        display[:,:,2] = np.clip((self.spikes * 255) + (exhaustion * 200), 0, 255)
        
        # Status Text
        chem_level = np.mean(self.chemical_field)
        energy_level = np.mean(self.vesicles)
        
        cv2.putText(display, f"Chem: {chem_level:.3f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,100), 1)
        cv2.putText(display, f"Fuel: {energy_level:.2f}", (5, h-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1)
                   
        return QtGui.QImage(display.data, w, h, w * 3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Threshold", "threshold", self.threshold, None),
            ("Coupling", "coupling", self.coupling, None),
            ("Fire Cost", "fire_cost", self.fire_cost, None),
            ("Recovery Rate", "recovery_rate", self.recovery_rate, None),
            ("Chem Release", "release_amount", self.release_amount, None),
            ("Chem Decay", "chemical_decay", self.chemical_decay, None),
            ("Diffusion", "diffusion_rate", self.diffusion_rate, None),
            ("Inhibition Str", "inhibition_strength", self.inhibition_strength, None),
        ]