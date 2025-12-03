"""
Address Projection & Dynamics Nodes
====================================
These nodes connect AFTER ModeAddressAlgebraNode.

AddressProjectionNode:
- Takes a field and an address
- Projects the field through the address (filters it)
- Shows what an attractor "sees" through its address lens
- Implements: ψ_seen = P_A[ψ]

AttractorDynamicsNode:
- Takes stable_address and metrics from ModeAddressAlgebra
- Implements division-dilution balance from IHT-AI
- Tracks attractor stability over time
- Shows convergence/divergence dynamics

AddressLearnerNode:
- Learns optimal address via gradient descent
- Implements the W-matrix training from IHT-AI
- Finds protected mode combinations
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class AddressProjectionNode(BaseNode):
    """
    Projects a quantum field through an address filter.
    
    Implements: ψ_seen = P_A[ψ] = F^{-1}[A · F[ψ]]
    
    This is what the attractor "sees" - reality filtered through its address.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Address Projection"
    NODE_COLOR = QtGui.QColor(200, 150, 100)  # Orange-brown
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_spectrum': 'complex_spectrum',  # The field ψ(k)
            'address_mask': 'image',                  # The address A (from ModeAddressAlgebra)
            'projection_strength': 'signal'           # How hard to filter (0=pass all, 1=strict)
        }
        
        self.outputs = {
            'projected_field': 'complex_spectrum',   # P_A[ψ]
            'projected_image': 'image',              # |P_A[ψ]| in position space
            'filtered_out': 'image',                 # What was rejected
            'projection_loss': 'signal'              # How much energy was lost
        }
        
        self.size = 128
        
        # State
        self.psi_in = None
        self.psi_projected = None
        self.address = None
        self.projected_spatial = None
        self.filtered_out_spatial = None
        self.projection_loss = 0.0
        
        # Parameters
        self.projection_strength = 1.0
        
    def step(self):
        # Get inputs
        psi = self.get_blended_input('complex_spectrum', 'first')
        address = self.get_blended_input('address_mask', 'first')
        strength = self.get_blended_input('projection_strength', 'sum')
        
        if strength is not None:
            self.projection_strength = np.clip(float(strength), 0.0, 1.0)
        
        if psi is None:
            return
            
        # Ensure correct size
        if psi.shape != (self.size, self.size):
            # Can't easily resize complex, so skip
            return
            
        self.psi_in = psi.astype(np.complex64)
        
        # Process address mask
        if address is not None:
            if address.ndim == 3:
                address = np.mean(address, axis=2)
            if address.shape != (self.size, self.size):
                address = cv2.resize(address.astype(np.float32), (self.size, self.size))
            # Normalize to 0-1
            self.address = address.astype(np.float32) / (np.max(address) + 1e-9)
        else:
            # Default: pass everything
            self.address = np.ones((self.size, self.size), dtype=np.float32)
        
        # Apply projection strength (interpolate between full pass and strict filter)
        effective_address = (1 - self.projection_strength) + self.projection_strength * self.address
        
        # Shift to centered k-space for proper filtering
        psi_k_centered = fftshift(self.psi_in)
        
        # Apply address filter
        psi_projected_k = psi_k_centered * effective_address
        psi_rejected_k = psi_k_centered * (1 - effective_address)
        
        # Shift back and store
        self.psi_projected = ifftshift(psi_projected_k)
        
        # Transform to position space for visualization
        self.projected_spatial = np.abs(ifft2(self.psi_projected))
        self.filtered_out_spatial = np.abs(ifft2(ifftshift(psi_rejected_k)))
        
        # Compute projection loss (fraction of energy filtered out)
        energy_in = np.sum(np.abs(psi_k_centered) ** 2)
        energy_out = np.sum(np.abs(psi_projected_k) ** 2)
        self.projection_loss = 1.0 - (energy_out / (energy_in + 1e-9))
        
    def get_output(self, port_name):
        if port_name == 'projected_field':
            return self.psi_projected
            
        elif port_name == 'projected_image':
            if self.projected_spatial is not None:
                img = self.projected_spatial
                img_norm = img / (np.max(img) + 1e-9)
                return (img_norm * 255).astype(np.uint8)
            return None
            
        elif port_name == 'filtered_out':
            if self.filtered_out_spatial is not None:
                img = self.filtered_out_spatial
                img_norm = img / (np.max(img) + 1e-9)
                return (img_norm * 255).astype(np.uint8)
            return None
            
        elif port_name == 'projection_loss':
            return float(self.projection_loss)
            
        return None
    
    def get_display_image(self):
        if self.projected_spatial is None:
            return None
            
        h, w = self.size, self.size
        
        # Left: What passes through (projected)
        proj_norm = self.projected_spatial / (np.max(self.projected_spatial) + 1e-9)
        proj_vis = (proj_norm * 255).astype(np.uint8)
        proj_color = cv2.applyColorMap(proj_vis, cv2.COLORMAP_VIRIDIS)
        
        # Right: What was filtered out
        filt_norm = self.filtered_out_spatial / (np.max(self.filtered_out_spatial) + 1e-9)
        filt_vis = (filt_norm * 255).astype(np.uint8)
        filt_color = cv2.applyColorMap(filt_vis, cv2.COLORMAP_HOT)
        
        full = np.hstack((proj_color, filt_color))
        
        cv2.putText(full, f"Seen (loss={self.projection_loss:.1%})", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Filtered Out", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Projection Strength", "projection_strength", self.projection_strength, None),
        ]


class AttractorDynamicsNode(BaseNode):
    """
    Implements the division-dilution balance from IHT-AI.
    
    Division: Amplitude spreading (+1+1+1...)
    Dilution: Normalization constraint (→1)
    
    Stable attractors exist only where these balance.
    
    Takes metrics from ModeAddressAlgebra and tracks attractor health.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Attractor Dynamics"
    NODE_COLOR = QtGui.QColor(150, 200, 100)  # Yellow-green
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'stable_address': 'image',       # From ModeAddressAlgebra
            'address_entropy': 'signal',     # S(A)
            'participation_ratio': 'signal', # PR
            'complex_spectrum': 'complex_spectrum',  # Optional: the field itself
            'dilution_rate': 'signal'        # γ parameter
        }
        
        self.outputs = {
            'attractor_health': 'signal',    # 0-1 overall health metric
            'stability_map': 'image',        # Spatial stability
            'division_rate': 'signal',       # How fast it's spreading
            'time_to_collapse': 'signal',    # Estimated steps until collapse
            'evolved_field': 'complex_spectrum'  # Field after dynamics applied
        }
        
        self.size = 128
        
        # History tracking
        self.entropy_history = []
        self.pr_history = []
        self.health_history = []
        self.stable_size_history = []
        
        # Current state
        self.stable_address = None
        self.stability_map = None
        self.attractor_health = 0.5
        self.division_rate = 0.0
        self.time_to_collapse = float('inf')
        
        # Internal field for evolution
        self.psi = None
        
        # Parameters
        self.dilution_rate = 0.02
        self.division_strength = 0.1
        
    def compute_health(self, entropy, pr, stable_size):
        """
        Attractor health based on:
        - Moderate entropy (not too spread, not too concentrated)
        - High participation ratio (uses many modes)
        - Large stable address (many protected modes)
        """
        # Optimal entropy around 0.5 (normalized)
        entropy_score = 1.0 - abs(entropy - 0.5) * 2
        
        # PR should be high but not infinite
        # Normalize assuming max useful PR around 10000
        pr_score = min(pr / 5000.0, 1.0)
        
        # Stable size as fraction of total
        size_score = stable_size / (self.size * self.size)
        
        # Weighted combination
        health = 0.3 * entropy_score + 0.3 * pr_score + 0.4 * size_score
        return np.clip(health, 0, 1)
    
    def estimate_collapse_time(self):
        """Estimate time to collapse based on health trend"""
        if len(self.health_history) < 10:
            return float('inf')
            
        # Linear regression on recent health
        recent = self.health_history[-20:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if slope >= 0:
            return float('inf')  # Improving or stable
            
        # Time to reach 0 from current health
        current = self.health_history[-1]
        return -current / slope
        
    def step(self):
        # Get inputs
        stable_addr = self.get_blended_input('stable_address', 'first')
        entropy = self.get_blended_input('address_entropy', 'sum')
        pr = self.get_blended_input('participation_ratio', 'sum')
        psi = self.get_blended_input('complex_spectrum', 'first')
        dilution = self.get_blended_input('dilution_rate', 'sum')
        
        if dilution is not None:
            self.dilution_rate = np.clip(float(dilution), 0.0, 0.5)
        
        # Process stable address
        if stable_addr is not None:
            if stable_addr.ndim == 3:
                stable_addr = np.mean(stable_addr, axis=2)
            if stable_addr.shape != (self.size, self.size):
                stable_addr = cv2.resize(stable_addr.astype(np.float32), (self.size, self.size))
            self.stable_address = stable_addr.astype(np.float32) / (np.max(stable_addr) + 1e-9)
        else:
            self.stable_address = np.ones((self.size, self.size), dtype=np.float32) * 0.5
        
        # Get metrics with defaults
        entropy_val = float(entropy) if entropy is not None else 0.5
        pr_val = float(pr) if pr is not None else 1000.0
        stable_size = np.sum(self.stable_address > 0.5)
        
        # Store history
        self.entropy_history.append(entropy_val)
        self.pr_history.append(pr_val)
        self.stable_size_history.append(stable_size)
        
        # Trim history
        max_hist = 100
        for hist in [self.entropy_history, self.pr_history, 
                     self.stable_size_history, self.health_history]:
            while len(hist) > max_hist:
                hist.pop(0)
        
        # Compute health
        self.attractor_health = self.compute_health(entropy_val, pr_val, stable_size)
        self.health_history.append(self.attractor_health)
        
        # Estimate collapse time
        self.time_to_collapse = self.estimate_collapse_time()
        
        # Compute division rate (how fast the address is spreading)
        if len(self.stable_size_history) > 1:
            self.division_rate = (self.stable_size_history[-1] - self.stable_size_history[-2]) / self.size**2
        
        # Create stability map
        # High stability = high in stable address AND consistent over time
        self.stability_map = self.stable_address.copy()
        
        # Apply division-dilution to field if provided
        if psi is not None and psi.shape == (self.size, self.size):
            self.psi = psi.astype(np.complex64)
            
            # Division: slight spreading via Laplacian in k-space
            # (equivalent to multiplication by k^2)
            center = self.size // 2
            y, x = np.ogrid[:self.size, :self.size]
            k2 = ((x - center)**2 + (y - center)**2).astype(np.float32)
            k2 = k2 / (center**2)  # Normalize
            
            psi_k = fftshift(fft2(self.psi))
            
            # Division: amplitude wants to spread to higher k
            division = 1.0 + self.division_strength * k2 * 0.01
            
            # Dilution: decay proportional to dilution rate
            dilution_factor = 1.0 - self.dilution_rate
            
            # Apply stable address as protection
            # Modes in stable address are protected from dilution
            protection = fftshift(self.stable_address)
            effective_dilution = dilution_factor + (1 - dilution_factor) * protection
            
            # Apply dynamics
            psi_k = psi_k * division * effective_dilution
            
            # Transform back
            self.psi = ifft2(ifftshift(psi_k)).astype(np.complex64)
    
    def get_output(self, port_name):
        if port_name == 'attractor_health':
            return float(self.attractor_health)
            
        elif port_name == 'stability_map':
            if self.stability_map is not None:
                return (self.stability_map * 255).astype(np.uint8)
            return None
            
        elif port_name == 'division_rate':
            return float(self.division_rate)
            
        elif port_name == 'time_to_collapse':
            if np.isinf(self.time_to_collapse):
                return 9999.0
            return float(self.time_to_collapse)
            
        elif port_name == 'evolved_field':
            return self.psi
            
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # Left: Stability map
        if self.stability_map is not None:
            stab_vis = (self.stability_map * 255).astype(np.uint8)
            stab_color = cv2.applyColorMap(stab_vis, cv2.COLORMAP_VIRIDIS)
        else:
            stab_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Right: Health history plot
        plot = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.health_history) > 1:
            n = len(self.health_history)
            
            # Health line (green when high, red when low)
            for i in range(n - 1):
                x1 = int(i * w / n)
                x2 = int((i + 1) * w / n)
                y1 = int((1 - self.health_history[i]) * (h - 20)) + 10
                y2 = int((1 - self.health_history[i + 1]) * (h - 20)) + 10
                
                # Color based on health value
                health_val = self.health_history[i]
                color = (0, int(255 * health_val), int(255 * (1 - health_val)))
                cv2.line(plot, (x1, y1), (x2, y2), color, 2)
        
        # Health indicator
        cv2.putText(plot, f"Health: {self.attractor_health:.2f}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        ttc_str = f"{self.time_to_collapse:.0f}" if not np.isinf(self.time_to_collapse) else "INF"
        cv2.putText(plot, f"TTC: {ttc_str}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
        
        cv2.putText(plot, f"Div: {self.division_rate:+.4f}", (5, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        full = np.hstack((stab_color, plot))
        
        cv2.putText(full, "Stability Map", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Health Dynamics", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Dilution Rate", "dilution_rate", self.dilution_rate, None),
            ("Division Strength", "division_strength", self.division_strength, None),
        ]


class AddressLearnerNode(BaseNode):
    """
    Learns optimal address via gradient descent.
    
    Implements the W-matrix training from IHT-AI:
    - Objective: maximize coherence under decoherence
    - Method: gradient descent on address weights
    
    Finds the protected mode combinations where attractors survive.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Address Learner (W-Matrix)"
    NODE_COLOR = QtGui.QColor(200, 100, 200)  # Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_spectrum': 'complex_spectrum',  # Field to learn from
            'decoherence_map': 'image',              # γ(k) landscape
            'target_coherence': 'signal',            # Target coherence level
            'learning_rate': 'signal'
        }
        
        self.outputs = {
            'learned_address': 'image',      # The learned W mask
            'coherence': 'signal',           # Current coherence
            'loss': 'signal',                # Training loss
            'projected_field': 'complex_spectrum'  # Field through learned address
        }
        
        self.size = 128
        center = self.size // 2
        
        # The learnable address W (sigmoid of weights)
        # Initialize with low-frequency bias
        y, x = np.ogrid[:self.size, :self.size]
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(np.float32)
        
        # Logits (pre-sigmoid weights)
        self.W_logits = 2.0 - 0.05 * r  # Bias toward center
        
        # Decoherence landscape
        self.gamma = np.clip(r / center, 0, 0.95).astype(np.float32)
        
        # Training state
        self.coherence = 0.0
        self.loss = 1.0
        self.loss_history = []
        self.coherence_history = []
        
        # Parameters
        self.learning_rate = 0.01
        self.target_coherence = 0.9
        
        # Internal state
        self.psi = None
        self.W = None
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    def compute_coherence(self, psi_projected):
        """Coherence = how phase-aligned the projected field is"""
        if psi_projected is None or np.sum(np.abs(psi_projected)) < 1e-9:
            return 0.0
            
        # Coherence = |mean(psi)| / mean(|psi|)
        # = 1 if all phases aligned, 0 if random phases
        mean_psi = np.mean(psi_projected)
        mean_abs = np.mean(np.abs(psi_projected))
        
        if mean_abs < 1e-9:
            return 0.0
            
        return np.abs(mean_psi) / mean_abs
    
    def compute_gradient(self, psi_k, W):
        """
        Compute gradient of coherence w.r.t. W logits
        
        Uses finite differences for simplicity
        """
        eps = 0.01
        grad = np.zeros_like(self.W_logits)
        
        # Sample a subset of points for efficiency
        sample_size = 100
        indices = np.random.choice(self.size * self.size, sample_size, replace=False)
        
        for idx in indices:
            i, j = idx // self.size, idx % self.size
            
            # Perturb up
            self.W_logits[i, j] += eps
            W_up = self.sigmoid(self.W_logits)
            psi_up = psi_k * fftshift(W_up)
            coh_up = self.compute_coherence(ifft2(ifftshift(psi_up)))
            
            # Perturb down
            self.W_logits[i, j] -= 2 * eps
            W_down = self.sigmoid(self.W_logits)
            psi_down = psi_k * fftshift(W_down)
            coh_down = self.compute_coherence(ifft2(ifftshift(psi_down)))
            
            # Restore
            self.W_logits[i, j] += eps
            
            # Gradient
            grad[i, j] = (coh_up - coh_down) / (2 * eps)
        
        return grad
    
    def step(self):
        # Get inputs
        psi = self.get_blended_input('complex_spectrum', 'first')
        gamma = self.get_blended_input('decoherence_map', 'first')
        target = self.get_blended_input('target_coherence', 'sum')
        lr = self.get_blended_input('learning_rate', 'sum')
        
        if target is not None:
            self.target_coherence = np.clip(float(target), 0.1, 1.0)
        if lr is not None:
            self.learning_rate = np.clip(float(lr), 0.001, 0.1)
        
        # Update decoherence map
        if gamma is not None:
            if gamma.ndim == 3:
                gamma = np.mean(gamma, axis=2)
            if gamma.shape != (self.size, self.size):
                gamma = cv2.resize(gamma.astype(np.float32), (self.size, self.size))
            self.gamma = gamma.astype(np.float32) / (np.max(gamma) + 1e-9)
        
        if psi is None or psi.shape != (self.size, self.size):
            return
            
        self.psi = psi.astype(np.complex64)
        
        # Current address (sigmoid of logits)
        self.W = self.sigmoid(self.W_logits)
        
        # Apply decoherence penalty to address
        # Modes with high γ should be suppressed
        protection_penalty = 1.0 - self.gamma
        effective_W = self.W * protection_penalty
        
        # Project field through address
        psi_k = fftshift(fft2(self.psi))
        psi_projected_k = psi_k * fftshift(effective_W)
        psi_projected = ifft2(ifftshift(psi_projected_k))
        
        # Compute coherence
        self.coherence = self.compute_coherence(psi_projected)
        
        # Compute loss (want to maximize coherence toward target)
        self.loss = max(0, self.target_coherence - self.coherence)
        
        # Store history
        self.loss_history.append(self.loss)
        self.coherence_history.append(self.coherence)
        while len(self.loss_history) > 200:
            self.loss_history.pop(0)
            self.coherence_history.pop(0)
        
        # Gradient update (every few steps for efficiency)
        if len(self.loss_history) % 5 == 0 and self.loss > 0.01:
            grad = self.compute_gradient(psi_k, self.W)
            
            # Also add gradient toward protected regions
            protection_grad = protection_penalty - 0.5
            
            # Combined gradient
            total_grad = grad + 0.1 * protection_grad
            
            # Update
            self.W_logits += self.learning_rate * total_grad
            
            # Regularization: slight decay toward zero
            self.W_logits *= 0.999
    
    def get_output(self, port_name):
        if port_name == 'learned_address':
            if self.W is not None:
                return (fftshift(self.W) * 255).astype(np.uint8)
            return None
            
        elif port_name == 'coherence':
            return float(self.coherence)
            
        elif port_name == 'loss':
            return float(self.loss)
            
        elif port_name == 'projected_field':
            if self.psi is not None and self.W is not None:
                psi_k = fftshift(fft2(self.psi))
                effective_W = self.W * (1.0 - self.gamma)
                psi_projected_k = psi_k * fftshift(effective_W)
                return ifftshift(psi_projected_k)
            return None
            
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # Left: Learned address W
        if self.W is not None:
            W_shifted = fftshift(self.W)
            W_vis = (W_shifted * 255).astype(np.uint8)
            W_color = cv2.applyColorMap(W_vis, cv2.COLORMAP_PLASMA)
        else:
            W_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Right: Training plot
        plot = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.coherence_history) > 1:
            n = len(self.coherence_history)
            
            # Coherence (green)
            for i in range(n - 1):
                x1 = int(i * w / n)
                x2 = int((i + 1) * w / n)
                y1 = int((1 - self.coherence_history[i]) * (h - 20)) + 10
                y2 = int((1 - self.coherence_history[i + 1]) * (h - 20)) + 10
                cv2.line(plot, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Loss (red)
            max_loss = max(self.loss_history) + 1e-9
            for i in range(n - 1):
                x1 = int(i * w / n)
                x2 = int((i + 1) * w / n)
                y1 = int((1 - self.loss_history[i] / max_loss) * (h - 20)) + 10
                y2 = int((1 - self.loss_history[i + 1] / max_loss) * (h - 20)) + 10
                cv2.line(plot, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # Target line
        target_y = int((1 - self.target_coherence) * (h - 20)) + 10
        cv2.line(plot, (0, target_y), (w, target_y), (255, 255, 0), 1)
        
        cv2.putText(plot, f"Coh: {self.coherence:.3f}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(plot, f"Loss: {self.loss:.3f}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(plot, f"LR: {self.learning_rate:.4f}", (5, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        full = np.hstack((W_color, plot))
        
        cv2.putText(full, "Learned W", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Training", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Target Coherence", "target_coherence", self.target_coherence, None),
        ]
