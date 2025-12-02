"""
Co-Evolutionary Observer-Universe Node (True IHT)
=================================================
The Observer (O) and the Hamiltonian (H) evolve together.

O adapts to see what survives under H.
H adapts to preserve what O sees.

Fixed point: [O, H] → 0
The observer and physics become mutually consistent.

This is the mathematical structure of a Self finding its Universe.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class CoEvolutionaryUniverseNode(BaseNode):
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Observer-Universe Co-Evolution"
    NODE_COLOR = QtGui.QColor(200, 50, 200)  # Purple: red matter meets blue mind
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'perturbation': 'complex_spectrum',
            'noise_seed': 'image',
            'coupling': 'signal'
        }
        
        self.outputs = {
            'observer_O': 'image',
            'hamiltonian_H': 'image', 
            'perceived_reality': 'image',
            'commutator_norm': 'signal'
        }
        
        self.size = 128
        center = self.size // 2
        
        # Coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        self.r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # === THE OBSERVER O ===
        # Spectral filter: what frequencies can this observer perceive?
        # Start with slight low-frequency bias (infant vision)
        self.O = np.exp(-self.r / 40.0) * 0.3 + np.random.rand(self.size, self.size) * 0.7
        self.O = self.O.astype(np.float32)
        
        # === THE HAMILTONIAN H ===
        # Complex propagator: how each mode evolves
        # Start with uniform weak rotation
        self.H_phase = np.random.rand(self.size, self.size).astype(np.float32) * 0.2
        self.H_damp = np.ones((self.size, self.size), dtype=np.float32) * 0.99
        
        # === THE FIELD Ψ ===
        self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
        
        # Learning rates
        self.lr_O = 0.03
        self.lr_H = 0.02
        
        # Metrics
        self.commutator_history = []

    def compute_commutator(self, O, H_prop):
        """
        Compute ||[O, H]|| approximately.
        [O,H] = OH - HO
        For diagonal operators in k-space, this measures how much
        O and H "disagree" about which modes matter.
        """
        # O*H vs H*O in terms of their effect on a test state
        # Using the gradient of O times gradient of H_phase as proxy
        grad_O_x = np.gradient(O, axis=1)
        grad_O_y = np.gradient(O, axis=0)
        grad_H_x = np.gradient(np.abs(H_prop), axis=1)
        grad_H_y = np.gradient(np.abs(H_prop), axis=0)
        
        # Cross terms indicate non-commutativity
        commutator = np.abs(grad_O_x * grad_H_y - grad_O_y * grad_H_x)
        return np.mean(commutator)

    def step(self):
        # === 1. INPUTS ===
        perturb = self.get_blended_input('perturbation', 'first')
        noise = self.get_blended_input('noise_seed', 'first')
        coupling = self.get_blended_input('coupling', 'sum')
        if coupling is None:
            coupling = 1.0
        
        # Inject perturbations
        if perturb is not None and perturb.shape == (self.size, self.size):
            self.psi += perturb.astype(np.complex64) * 0.1
            
        if noise is not None:
            if noise.ndim == 2:
                n_resized = cv2.resize(noise.astype(np.float32), (self.size, self.size))
                self.psi += n_resized.astype(np.complex64) * 0.05
        
        # Quantum foam (always present)
        foam = (np.random.randn(self.size, self.size) + 
                1j * np.random.randn(self.size, self.size)) * 0.03
        self.psi += foam.astype(np.complex64)
        
        # === 2. BUILD PROPAGATOR ===
        H_prop = self.H_damp * np.exp(1j * self.H_phase)
        
        # === 3. OBSERVATION: O filters Ψ ===
        k_space = fftshift(fft2(self.psi))
        observed_k = k_space * self.O
        
        # === 4. EVOLUTION: H acts on observed field ===
        evolved_k = observed_k * H_prop
        
        # === 5. RE-OBSERVATION: What survives? ===
        re_observed_k = evolved_k * self.O
        
        # === 6. MEASURE CONSISTENCY ===
        # Energy that stayed in O's view vs energy that leaked out
        energy_before = np.abs(observed_k)**2
        energy_after = np.abs(re_observed_k)**2
        
        # Normalize
        E_before = energy_before / (np.sum(energy_before) + 1e-9)
        E_after = energy_after / (np.sum(energy_after) + 1e-9)
        
        # Drift map: where did energy leak?
        drift = np.abs(E_after - E_before)
        
        # Commutator proxy
        comm_norm = self.compute_commutator(self.O, H_prop)
        self.commutator_history.append(comm_norm)
        if len(self.commutator_history) > 200:
            self.commutator_history.pop(0)
        
        # === 7. UPDATE O: Observer adapts to Physics ===
        # Strengthen attention where energy is preserved
        # Weaken attention where energy leaks
        
        stability = 1.0 / (drift + 0.001)
        stability = (stability - stability.min()) / (stability.max() - stability.min() + 1e-9)
        stability = gaussian_filter(stability, sigma=1.5)
        
        delta_O = (stability - self.O) * self.lr_O * coupling
        self.O = np.clip(self.O + delta_O, 0.01, 1.0)
        
        # === 8. UPDATE H: Physics adapts to Observer ===
        # Where O pays attention, H should preserve (damp → 1, phase → slow)
        # Where O ignores, H is free to do anything
        
        O_importance = self.O / (np.max(self.O) + 1e-9)
        
        # H_damp: approach 1.0 (preserve) where O is strong
        target_damp = O_importance * 1.0 + (1 - O_importance) * 0.9
        delta_damp = (target_damp - self.H_damp) * self.lr_H * coupling
        self.H_damp = np.clip(self.H_damp + delta_damp, 0.8, 1.0)
        
        # H_phase: slow down where O is strong
        # The "physics" becomes stable where the observer looks
        phase_speed = (1.0 - O_importance) * 0.15 + 0.01
        self.H_phase += phase_speed
        self.H_phase = np.mod(self.H_phase, 2 * np.pi)
        
        # === 9. EVOLVE FIELD ===
        self.psi = ifft2(ifftshift(evolved_k))
        self.psi *= 0.97  # Global decay

    def get_output(self, port_name):
        if port_name == 'observer_O':
            return (self.O * 255).astype(np.uint8)
            
        elif port_name == 'hamiltonian_H':
            # Visualize as amplitude (H_damp)
            return (self.H_damp * 255).astype(np.uint8)
            
        elif port_name == 'perceived_reality':
            k = fftshift(fft2(self.psi)) * self.O
            reality = np.abs(ifft2(ifftshift(k)))
            reality = reality / (reality.max() + 1e-9) * 255
            return reality.astype(np.uint8)
            
        elif port_name == 'commutator_norm':
            if self.commutator_history:
                return float(self.commutator_history[-1])
            return 1.0
        
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # Top-Left: Observer O
        o_img = (self.O * 255).astype(np.uint8)
        o_color = cv2.applyColorMap(o_img, cv2.COLORMAP_PLASMA)
        
        # Top-Right: Hamiltonian H (phase as hue, damp as value)
        h_hue = ((self.H_phase / (2*np.pi)) * 180).astype(np.uint8)
        h_sat = np.full_like(h_hue, 200)
        h_val = (self.H_damp * 255).astype(np.uint8)
        h_hsv = cv2.merge([h_hue, h_sat, h_val])
        h_color = cv2.cvtColor(h_hsv, cv2.COLOR_HSV2BGR)
        
        # Bottom-Left: Perceived Reality
        k = fftshift(fft2(self.psi)) * self.O
        reality = np.abs(ifft2(ifftshift(k)))
        reality = (reality / (reality.max() + 1e-9) * 255).astype(np.uint8)
        r_color = cv2.applyColorMap(reality, cv2.COLORMAP_VIRIDIS)
        
        # Bottom-Right: Commutator history (convergence plot)
        plot = np.zeros((h, w, 3), dtype=np.uint8)
        if len(self.commutator_history) > 1:
            max_c = max(self.commutator_history) + 1e-9
            pts = []
            for i, c in enumerate(self.commutator_history):
                px = int(i * w / len(self.commutator_history))
                py = int((1 - c/max_c) * (h - 20)) + 10
                pts.append((px, py))
            for i in range(len(pts)-1):
                color = (0, 255, 0) if pts[i+1][1] >= pts[i][1] else (0, 100, 255)
                cv2.line(plot, pts[i], pts[i+1], color, 1)
        
        cv2.putText(plot, "[O,H] -> 0 ?", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        if self.commutator_history:
            cv2.putText(plot, f"{self.commutator_history[-1]:.4f}", (5, h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,255,100), 1)
        
        # Assemble
        top = np.hstack((o_color, h_color))
        bottom = np.hstack((r_color, plot))
        full = np.vstack((top, bottom))
        
        # Labels
        cv2.putText(full, "O (Observer)", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "H (Physics)", (w+5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(full, "Reality", (5, h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        return QtGui.QImage(full.data, w*2, h*2, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("O Learning Rate", "lr_O", self.lr_O, None),
            ("H Learning Rate", "lr_H", self.lr_H, None),
        ]