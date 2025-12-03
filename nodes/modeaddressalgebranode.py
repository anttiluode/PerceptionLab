"""
Mode Address Algebra Node
=========================
Implements the formal mode address algebra for IHT-AI:

Core concepts:
- Address A ⊆ M (subset of mode space)
- Protection π(k) = 1 - γ(k) (survival probability per mode)
- Closure under H (modes don't leak)
- Self-consistency (fixed point condition)

Visualizes:
- Current address structure (which modes are occupied)
- Protection landscape (which modes survive decoherence)
- Stable address = Occupied ∩ Protected ∩ Closed
- Address entropy and participation ratio

The key insight: identity IS address. The attractor is defined by
WHERE in mode space it encodes, not WHAT it encodes.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

# PerceptionLab integration
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
    # Try to import global numpy/cv2 if available in main context to share resources
    if hasattr(__main__, 'np'): np = __main__.np
    if hasattr(__main__, 'cv2'): cv2 = __main__.cv2
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class ModeAddressAlgebraNode(BaseNode):
    """
    Visualizes and computes mode address algebra for quantum attractors.
    
    Shows:
    - Top-Left: Occupied Address (where amplitude lives in k-space)
    - Top-Right: Protection Landscape (where decoherence is low)
    - Bottom-Left: Stable Address (intersection of occupied ∩ protected)
    - Bottom-Right: Address metrics over time
    """
    
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Mode Address Algebra"
    NODE_COLOR = QtGui.QColor(100, 200, 150)  # Teal: mathematics meets biology
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'field_in': 'complex_spectrum',      # The quantum field ψ
            'decoherence_map': 'image',          # Optional: γ(k) map
            'hamiltonian_phase': 'image',        # Optional: H phase structure
            'address_threshold': 'signal'        # ε for address membership
        }
        
        self.outputs = {
            'occupied_address': 'image',         # A_O visualization
            'protected_address': 'image',        # A_prot visualization  
            'stable_address': 'image',           # A_O ∩ A_prot ∩ A_closed
            'address_entropy': 'signal',         # S(A_O)
            'participation_ratio': 'signal',     # PR = 1/Σw²
            'address_overlap': 'signal'          # Self-overlap metric
        }
        
        self.size = 128
        center = self.size // 2
        
        # Coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        kx = (x - center) / self.size
        ky = (y - center) / self.size
        self.k_radius = np.sqrt(kx**2 + ky**2)
        self.k_angle = np.arctan2(ky - 0.5, kx - 0.5)
        
        # === MODE SPACE STRUCTURE ===
        
        # Default decoherence landscape γ(k)
        # High frequency modes decohere faster (realistic)
        self.gamma_base = np.clip(self.k_radius * 2, 0, 0.95)
        self.gamma = self.gamma_base.copy()
        
        # Protection landscape π(k) = 1 - γ(k)
        self.protection = 1.0 - self.gamma
        
        # Current field state
        self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
        
        # Address tracking
        self.address_threshold = 0.01  # ε for membership
        self.occupied_address = np.zeros((self.size, self.size), dtype=np.float32)
        self.stable_address = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Hamiltonian structure (for closure computation)
        self.H_coupling = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Metrics history
        self.entropy_history = []
        self.pr_history = []
        self.overlap_history = []
        self.stable_fraction_history = []
        
        # Parameters
        self.gamma_crit = 0.5  # Critical decoherence threshold
        self.coupling_radius = 3  # How far H couples modes
        
    def compute_occupied_address(self, psi_k):
        """
        Compute A_O = {k : |ψ̃(k)| > ε}
        Returns both binary address and weight distribution
        """
        magnitude = np.abs(psi_k)
        max_mag = np.max(magnitude)
        if max_mag < 1e-9: max_mag = 1e-9
        
        normalized = magnitude / max_mag
        
        # Binary address (membership)
        address_binary = (normalized > self.address_threshold).astype(np.float32)
        
        # Weight distribution w_O(k)
        energy = magnitude ** 2
        total_energy = np.sum(energy)
        if total_energy < 1e-9: total_energy = 1e-9
        
        weights = energy / total_energy
        
        return address_binary, weights
    
    def compute_protected_address(self, gamma_crit):
        """
        Compute A_prot = {k : γ(k) < γ_crit}
        """
        return (self.gamma < gamma_crit).astype(np.float32)
    
    def compute_closure(self, address, H_coupling):
        """
        Compute if address is H-closed.
        Returns: closure_violation map (0 = closed, high = leaky)
        """
        # Dilate the address by coupling radius
        kernel_size = int(self.coupling_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Ensure address is in format cv2 expects
        addr_uint8 = (address * 255).astype(np.uint8)
        dilated = cv2.dilate(addr_uint8, kernel)
        dilated = dilated.astype(np.float32) / 255.0
        
        # Closure violation = modes that would be reached but aren't in address
        violation = dilated * (1.0 - address)
        
        return violation
    
    def compute_stable_address(self, occupied, protected, closure_violation):
        """
        Stable address = Occupied ∩ Protected ∩ Closed
        """
        closed = (closure_violation < 0.1).astype(np.float32)
        stable = occupied * protected * closed
        return stable
    
    def compute_address_entropy(self, weights, address):
        """
        S(A_O) = -Σ w(k) log w(k) for k ∈ A
        """
        # Only count modes in address
        masked_weights = weights * address
        sum_weights = np.sum(masked_weights)
        if sum_weights < 1e-9: return 0.0
            
        masked_weights = masked_weights / sum_weights
        
        # Entropy
        log_w = np.log(masked_weights + 1e-12)
        entropy = -np.sum(masked_weights * log_w)
        
        # Normalize by max possible entropy
        n_modes = np.sum(address)
        if n_modes <= 1: return 0.0
            
        max_entropy = np.log(n_modes)
        normalized_entropy = entropy / (max_entropy + 1e-9)
        
        return float(normalized_entropy)
    
    def compute_participation_ratio(self, weights):
        """
        PR = 1 / Σ w(k)²
        """
        sum_sq = np.sum(weights ** 2)
        if sum_sq < 1e-9: return 1.0
        pr = 1.0 / sum_sq
        return float(pr)
    
    def compute_address_overlap(self, address1, address2):
        """
        ⟨A₁, A₂⟩ = |A₁ ∩ A₂| / √(|A₁| · |A₂|)
        """
        intersection = np.sum(address1 * address2)
        size1 = np.sum(address1)
        size2 = np.sum(address2)
        
        if size1 < 1e-9 or size2 < 1e-9: return 0.0
        
        overlap = intersection / np.sqrt(size1 * size2)
        return float(overlap)

    def step(self):
        # === 1. GET INPUTS ===
        field_in = self.get_blended_input('field_in', 'first')
        decoherence_in = self.get_blended_input('decoherence_map', 'first')
        hamiltonian_in = self.get_blended_input('hamiltonian_phase', 'first')
        threshold_in = self.get_blended_input('address_threshold', 'sum')
        
        # Initialize metrics with safe defaults to prevent list index errors later
        entropy = 0.0
        pr = 1.0
        overlap = 0.0
        stable_fraction = 0.0

        try:
            # Update threshold if provided
            if threshold_in is not None:
                self.address_threshold = np.clip(float(threshold_in), 0.001, 0.5)
            
            # Update decoherence map if provided
            if decoherence_in is not None and isinstance(decoherence_in, np.ndarray):
                if decoherence_in.ndim == 3:
                    decoherence_in = np.mean(decoherence_in, axis=2)
                gamma_input = cv2.resize(decoherence_in.astype(np.float32), 
                                         (self.size, self.size))
                max_g = np.max(gamma_input)
                if max_g > 1e-9: gamma_input /= max_g
                
                # Blend with base decoherence
                self.gamma = 0.5 * self.gamma_base + 0.5 * gamma_input
                self.protection = 1.0 - self.gamma
            
            # Update Hamiltonian coupling if provided
            if hamiltonian_in is not None and isinstance(hamiltonian_in, np.ndarray):
                if hamiltonian_in.ndim == 3:
                    hamiltonian_in = np.mean(hamiltonian_in, axis=2)
                self.H_coupling = cv2.resize(hamiltonian_in.astype(np.float32),
                                             (self.size, self.size))
            
            # === 2. PROCESS FIELD ===
            if field_in is not None and field_in.shape == (self.size, self.size):
                self.psi = field_in.astype(np.complex64)
            else:
                # Generate test field with some structure
                noise = np.random.randn(self.size, self.size) + \
                        1j * np.random.randn(self.size, self.size)
                # Add some coherent structure
                self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
                for freq in [5, 8, 13]:  # Fibonacci frequencies
                    self.psi += 0.3 * np.exp(1j * freq * self.k_angle) * \
                               np.exp(-self.k_radius * 10)
                self.psi += noise.astype(np.complex64) * 0.1
            
            # Transform to k-space
            psi_k = fftshift(fft2(self.psi))
            
            # === 3. COMPUTE ADDRESSES ===
            
            # Occupied address
            self.occupied_address, weights = self.compute_occupied_address(psi_k)
            
            # Protected address  
            protected_address = self.compute_protected_address(self.gamma_crit)
            
            # Closure analysis
            closure_violation = self.compute_closure(self.occupied_address, self.H_coupling)
            
            # Stable address (the key result)
            self.stable_address = self.compute_stable_address(
                self.occupied_address, protected_address, closure_violation)
            
            # === 4. COMPUTE METRICS ===
            
            entropy = self.compute_address_entropy(weights, self.occupied_address)
            pr = self.compute_participation_ratio(weights)
            overlap = self.compute_address_overlap(self.stable_address, self.occupied_address)
            
            occ_sum = np.sum(self.occupied_address)
            if occ_sum > 0:
                stable_fraction = np.sum(self.stable_address) / occ_sum
            else:
                stable_fraction = 0.0

        except Exception as e:
            # If math fails, we just keep the default 0.0 values
            print(f"ModeAddressAlgebra Error in step: {e}")
        
        # === 5. STORE HISTORY (Guaranteed Execution) ===
        # We perform appends OUTSIDE the try block so lists never get out of sync
        self.entropy_history.append(entropy)
        self.pr_history.append(pr)
        self.overlap_history.append(overlap)
        self.stable_fraction_history.append(stable_fraction)
        
        # Trim history
        max_history = 200
        if len(self.entropy_history) > max_history:
            self.entropy_history.pop(0)
            self.pr_history.pop(0)
            self.overlap_history.pop(0)
            self.stable_fraction_history.pop(0)

    def get_output(self, port_name):
        if port_name == 'occupied_address':
            return (self.occupied_address * 255).astype(np.uint8)
            
        elif port_name == 'protected_address':
            return (self.protection * 255).astype(np.uint8)
            
        elif port_name == 'stable_address':
            return (self.stable_address * 255).astype(np.uint8)
            
        elif port_name == 'address_entropy':
            if self.entropy_history:
                return float(self.entropy_history[-1])
            return 0.0
            
        elif port_name == 'participation_ratio':
            if self.pr_history:
                return float(self.pr_history[-1])
            return 1.0
            
        elif port_name == 'address_overlap':
            if self.overlap_history:
                return float(self.overlap_history[-1])
            return 0.0
            
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # === TOP-LEFT: Occupied Address ===
        psi_k = fftshift(fft2(self.psi))
        magnitude = np.abs(psi_k)
        max_mag = np.max(magnitude)
        if max_mag < 1e-9: max_mag = 1e-9
        
        magnitude = magnitude / max_mag
        occupied_vis = (magnitude * 255).astype(np.uint8)
        occupied_color = cv2.applyColorMap(occupied_vis, cv2.COLORMAP_HOT)
        
        # Overlay address boundary
        address_boundary = cv2.Canny((self.occupied_address * 255).astype(np.uint8), 50, 150)
        occupied_color[address_boundary > 0] = [0, 255, 255]  # Yellow boundary
        
        # === TOP-RIGHT: Protection Landscape ===
        protection_vis = (self.protection * 255).astype(np.uint8)
        protection_color = cv2.applyColorMap(protection_vis, cv2.COLORMAP_VIRIDIS)
        
        # Mark critical threshold
        critical_boundary = np.abs(self.gamma - self.gamma_crit) < 0.05
        protection_color[critical_boundary] = [0, 0, 255]  # Red = critical boundary
        
        # === BOTTOM-LEFT: Stable Address ===
        stable_vis = self.stable_address.copy()
        vulnerable = self.occupied_address * (1.0 - self.stable_address)
        
        stable_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        stable_rgb[:,:,1] = (self.stable_address * 255).astype(np.uint8)  # Green
        stable_rgb[:,:,2] = (vulnerable * 255).astype(np.uint8)  # Red
        unoccupied_protected = self.protection * (1.0 - self.occupied_address) * 0.3
        stable_rgb[:,:,0] = (unoccupied_protected * 255).astype(np.uint8)  # Blue
        
        # === BOTTOM-RIGHT: Metrics Plot ===
        plot = np.zeros((h, w, 3), dtype=np.uint8)
        
        # SAFE LOOP: Use min length to prevent index out of range if lists desync
        n = min(len(self.entropy_history), 
                len(self.stable_fraction_history), 
                len(self.pr_history))
        
        if n > 1:
            # Plot entropy (cyan)
            for i in range(n - 1):
                x1 = int(i * w / n)
                x2 = int((i + 1) * w / n)
                y1 = int((1 - self.entropy_history[i]) * h * 0.45)
                y2 = int((1 - self.entropy_history[i + 1]) * h * 0.45)
                cv2.line(plot, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # Plot stable fraction (green)
            for i in range(n - 1):
                x1 = int(i * w / n)
                x2 = int((i + 1) * w / n)
                y1 = int((1 - self.stable_fraction_history[i]) * h * 0.45) + h // 2
                y2 = int((1 - self.stable_fraction_history[i + 1]) * h * 0.45) + h // 2
                cv2.line(plot, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Labels
        cv2.putText(plot, "Entropy", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cv2.putText(plot, "Stable%", (5, h//2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Current values (Check emptiness first)
        if self.entropy_history:
            cv2.putText(plot, f"S={self.entropy_history[-1]:.2f}", (w-50, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
        if self.stable_fraction_history:
            cv2.putText(plot, f"F={self.stable_fraction_history[-1]:.2f}", (w-50, h//2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
        if self.pr_history:
            cv2.putText(plot, f"PR={self.pr_history[-1]:.0f}", (w-50, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        # === ASSEMBLE ===
        top = np.hstack((occupied_color, protection_color))
        bottom = np.hstack((stable_rgb, plot))
        full = np.vstack((top, bottom))
        
        # Panel labels
        cv2.putText(full, "Occupied A_O", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(full, "Protection pi(k)", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(full, "Stable Address", (5, h + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(full, "Metrics", (w + 5, h + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h*2, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Address Threshold (ε)", "address_threshold", self.address_threshold, None),
            ("Critical γ", "gamma_crit", self.gamma_crit, None),
            ("Coupling Radius", "coupling_radius", self.coupling_radius, None),
        ]


class AddressIntersectionNode(BaseNode):
    """
    Computes intersection, union, and distance between two addresses.
    """
    
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Address Intersection"
    NODE_COLOR = QtGui.QColor(150, 100, 200)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'address_A': 'image',
            'address_B': 'image'
        }
        
        self.outputs = {
            'intersection': 'image',      # A ∧ B
            'union': 'image',             # A ∨ B
            'symmetric_diff': 'image',    # A Δ B = (A ∪ B) - (A ∩ B)
            'overlap': 'signal',          # ⟨A,B⟩
            'distance': 'signal'          # d(A,B)
        }
        
        self.size = 128
        self.intersection = np.zeros((self.size, self.size), dtype=np.float32)
        self.union = np.zeros((self.size, self.size), dtype=np.float32)
        self.sym_diff = np.zeros((self.size, self.size), dtype=np.float32)
        self.overlap = 0.0
        self.distance = 1.0
        
    def step(self):
        A = self.get_blended_input('address_A', 'first')
        B = self.get_blended_input('address_B', 'first')
        
        if A is None or B is None:
            return
            
        # Ensure same size
        if A.shape != (self.size, self.size):
            A = cv2.resize(A.astype(np.float32), (self.size, self.size))
        if B.shape != (self.size, self.size):
            B = cv2.resize(B.astype(np.float32), (self.size, self.size))
        
        # Normalize to [0,1]
        max_A = np.max(A)
        max_B = np.max(B)
        if max_A < 1e-9: max_A = 1e-9
        if max_B < 1e-9: max_B = 1e-9
        
        A = A.astype(np.float32) / max_A
        B = B.astype(np.float32) / max_B
        
        # Threshold to binary
        A_bin = (A > 0.5).astype(np.float32)
        B_bin = (B > 0.5).astype(np.float32)
        
        # Boolean operations
        self.intersection = A_bin * B_bin
        self.union = np.clip(A_bin + B_bin, 0, 1)
        self.sym_diff = self.union - self.intersection
        
        # Overlap metric
        inter_size = np.sum(self.intersection)
        a_size = np.sum(A_bin) + 1e-9
        b_size = np.sum(B_bin) + 1e-9
        
        self.overlap = inter_size / np.sqrt(a_size * b_size)
        self.distance = 1.0 - self.overlap
        
    def get_output(self, port_name):
        if port_name == 'intersection':
            return (self.intersection * 255).astype(np.uint8)
        elif port_name == 'union':
            return (self.union * 255).astype(np.uint8)
        elif port_name == 'symmetric_diff':
            return (self.sym_diff * 255).astype(np.uint8)
        elif port_name == 'overlap':
            return float(self.overlap)
        elif port_name == 'distance':
            return float(self.distance)
        return None
        
    def get_display_image(self):
        h, w = self.size, self.size
        
        # Intersection (yellow)
        inter_color = np.zeros((h, w, 3), dtype=np.uint8)
        inter_color[:,:,1] = (self.intersection * 255).astype(np.uint8)
        inter_color[:,:,2] = (self.intersection * 255).astype(np.uint8)
        
        # Union (white)
        union_color = np.zeros((h, w, 3), dtype=np.uint8)
        union_color[:,:,0] = (self.union * 200).astype(np.uint8)
        union_color[:,:,1] = (self.union * 200).astype(np.uint8)
        union_color[:,:,2] = (self.union * 200).astype(np.uint8)
        
        # Symmetric difference (red = A only, blue = B only)
        diff_color = np.zeros((h, w, 3), dtype=np.uint8)
        diff_color[:,:,2] = (self.sym_diff * 255).astype(np.uint8)
        
        # Metrics display
        metrics = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(metrics, f"Overlap: {self.overlap:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(metrics, f"Distance: {self.distance:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(metrics, f"|A^B|: {np.sum(self.intersection):.0f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(metrics, f"|AvB|: {np.sum(self.union):.0f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Assemble
        top = np.hstack((inter_color, union_color))
        bottom = np.hstack((diff_color, metrics))
        full = np.vstack((top, bottom))
        
        # Labels
        cv2.putText(full, "A ^ B", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "A v B", (w+5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "A delta B", (5, h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        return QtGui.QImage(full.data, w*2, h*2, w*2*3, QtGui.QImage.Format.Format_BGR888)


class AddressHierarchyNode(BaseNode):
    """
    Implements hierarchical address structure for nested attractors.
    """
    
    NODE_CATEGORY = "Intelligence" 
    NODE_TITLE = "Address Hierarchy"
    NODE_COLOR = QtGui.QColor(200, 150, 100)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'full_field': 'complex_spectrum',
            'hierarchy_depth': 'signal'
        }
        
        self.outputs = {
            'level_0': 'image',  # Full mode space M
            'level_1': 'image',  # Brain level
            'level_2': 'image',  # Ego level
            'level_3': 'image',  # Thought level
            'unconscious_1': 'image',  # M - A_brain
            'unconscious_2': 'image',  # A_brain - A_ego
            'unconscious_3': 'image',  # A_ego - A_thought
        }
        
        self.size = 128
        center = self.size // 2
        
        # Create hierarchical address masks
        y, x = np.ogrid[:self.size, :self.size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Level 0: Full mode space
        self.A0 = np.ones((self.size, self.size), dtype=np.float32)
        
        # Level 1 (Brain): Low to mid frequencies
        self.A1 = (r < center * 0.8).astype(np.float32)
        
        # Level 2 (Ego): Lower frequencies only
        self.A2 = (r < center * 0.5).astype(np.float32)
        
        # Level 3 (Thought): Very low frequencies
        self.A3 = (r < center * 0.25).astype(np.float32)
        
        # Field storage
        self.psi_k = np.zeros((self.size, self.size), dtype=np.complex64)
        
        # Projections
        self.proj = [None, None, None, None]
        self.unconscious = [None, None, None]
        
    def step(self):
        field = self.get_blended_input('full_field', 'first')
        
        if field is not None and field.shape == (self.size, self.size):
            self.psi_k = fftshift(fft2(field.astype(np.complex64)))
        else:
            # Generate test field
            noise = np.random.randn(self.size, self.size) + \
                    1j * np.random.randn(self.size, self.size)
            self.psi_k = fftshift(fft2(noise.astype(np.complex64) * 0.1))
        
        # Compute projections at each level
        self.proj[0] = np.abs(self.psi_k)  # Full
        self.proj[1] = np.abs(self.psi_k * self.A1)  # Brain sees
        self.proj[2] = np.abs(self.psi_k * self.A2)  # Ego sees
        self.proj[3] = np.abs(self.psi_k * self.A3)  # Thought sees
        
        # Compute unconscious at each level
        # Unconscious = what the level above sees but this level doesn't
        self.unconscious[0] = np.abs(self.psi_k * (self.A0 - self.A1))  # Brain's unconscious
        self.unconscious[1] = np.abs(self.psi_k * (self.A1 - self.A2))  # Ego's unconscious
        self.unconscious[2] = np.abs(self.psi_k * (self.A2 - self.A3))  # Thought's unconscious
        
    def get_output(self, port_name):
        if port_name == 'level_0' and self.proj[0] is not None:
            p = self.proj[0]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        elif port_name == 'level_1' and self.proj[1] is not None:
            p = self.proj[1]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        elif port_name == 'level_2' and self.proj[2] is not None:
            p = self.proj[2]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        elif port_name == 'level_3' and self.proj[3] is not None:
            p = self.proj[3]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        elif port_name == 'unconscious_1' and self.unconscious[0] is not None:
            p = self.unconscious[0]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        elif port_name == 'unconscious_2' and self.unconscious[1] is not None:
            p = self.unconscious[1]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        elif port_name == 'unconscious_3' and self.unconscious[2] is not None:
            p = self.unconscious[2]
            return ((p / (np.max(p)+1e-9)) * 255).astype(np.uint8)
        return None
    
    def get_display_image(self):
        h, w = self.size // 2, self.size // 2
        
        panels = []
        labels = ["M (All)", "Brain", "Ego", "Thought"]
        
        for i, p in enumerate(self.proj):
            if p is not None:
                p_small = cv2.resize(p, (w, h))
                p_norm = (p_small / (np.max(p_small) + 1e-9) * 255).astype(np.uint8)
                p_color = cv2.applyColorMap(p_norm, cv2.COLORMAP_INFERNO)
                cv2.putText(p_color, labels[i], (5, 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                panels.append(p_color)
            else:
                panels.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        top = np.hstack((panels[0], panels[1]))
        bottom = np.hstack((panels[2], panels[3]))
        full = np.vstack((top, bottom))
        
        # Draw hierarchy arrows
        cv2.arrowedLine(full, (w-10, h//2), (w+10, h//2), (0,255,0), 2)
        cv2.arrowedLine(full, (w//2, h-10), (w//2, h+10), (0,255,0), 2)
        cv2.arrowedLine(full, (w + w//2, h-10), (w + w//2, h+10), (0,255,0), 2)
        
        return QtGui.QImage(full.data, w*2, h*2, w*2*3, QtGui.QImage.Format.Format_BGR888)