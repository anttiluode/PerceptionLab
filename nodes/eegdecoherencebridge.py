"""
EEG Decoherence Bridge Node
===========================
Converts EEG band powers directly into a decoherence map for Mode Address Algebra.

The idea: Your brain's frequency bands ARE addresses in mode space.
When alpha is high → alpha frequencies are "protected" (low decoherence)
When beta is high → beta frequencies are "protected"

This creates a decoherence landscape SHAPED BY YOUR ACTUAL BRAIN STATE.

Then ModeAddressAlgebra finds which modes are stable given YOUR neural activity.

Frequency mapping to k-space (radial):
- Delta (1-4 Hz)   → center (k < 0.1)
- Theta (4-8 Hz)   → inner ring (0.1 < k < 0.2)
- Alpha (8-13 Hz)  → middle ring (0.2 < k < 0.35)
- Beta (13-30 Hz)  → outer ring (0.35 < k < 0.6)
- Gamma (30-45 Hz) → edge (k > 0.6)
"""

import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    from PyQt6 import QtGui


class EEGDecoherenceBridgeNode(BaseNode):
    """
    Maps EEG band powers to a 2D decoherence landscape.
    
    High band power = low decoherence = protected modes
    Low band power = high decoherence = vulnerable modes
    
    Wire outputs to ModeAddressAlgebraNode's decoherence_map input.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "EEG → Decoherence"
    NODE_COLOR = QtGui.QColor(60, 180, 140)  # Teal-green: brain meets math
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'delta': 'signal',      # 1-4 Hz band power
            'theta': 'signal',      # 4-8 Hz band power
            'alpha': 'signal',      # 8-13 Hz band power
            'beta': 'signal',       # 13-30 Hz band power
            'gamma': 'signal',      # 30-45 Hz band power
            'baseline_protection': 'signal',  # Base protection level (0-1)
            'sensitivity': 'signal' # How much bands affect protection
        }
        
        self.outputs = {
            'decoherence_map': 'image',   # γ(k) - for ModeAddressAlgebra
            'protection_map': 'image',     # π(k) = 1 - γ(k)
            'dominant_band': 'signal',     # Which band is strongest (0-4)
            'total_power': 'signal'        # Sum of all bands
        }
        
        self.size = 128
        center = self.size // 2
        
        # Create radial coordinate grid (normalized 0-1)
        y, x = np.ogrid[:self.size, :self.size]
        kx = (x - center) / center
        ky = (y - center) / center
        self.k_radius = np.sqrt(kx**2 + ky**2)
        
        # Define radial bands in k-space
        # These map neural frequency bands to spatial frequencies
        self.band_masks = {
            'delta': (self.k_radius < 0.12),
            'theta': (self.k_radius >= 0.12) & (self.k_radius < 0.25),
            'alpha': (self.k_radius >= 0.25) & (self.k_radius < 0.40),
            'beta':  (self.k_radius >= 0.40) & (self.k_radius < 0.65),
            'gamma': (self.k_radius >= 0.65)
        }
        
        # State
        self.decoherence = np.ones((self.size, self.size), dtype=np.float32) * 0.5
        self.protection = np.ones((self.size, self.size), dtype=np.float32) * 0.5
        
        # Band values for display
        self.band_values = {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}
        
        # Parameters
        self.baseline = 0.5      # Default decoherence when no signal
        self.sensitivity = 2.0   # How much band power affects decoherence
        
        # Smoothing for temporal stability
        self.smooth_decoherence = None
        self.smooth_factor = 0.3  # Lower = smoother
        
    def step(self):
        # Get parameters
        base = self.get_blended_input('baseline_protection', 'sum')
        sens = self.get_blended_input('sensitivity', 'sum')
        
        if base is not None:
            self.baseline = np.clip(float(base), 0.1, 0.9)
        if sens is not None:
            self.sensitivity = np.clip(float(sens), 0.5, 5.0)
        
        # Get band powers
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        powers = {}
        
        for band in bands:
            val = self.get_blended_input(band, 'sum')
            if val is not None:
                powers[band] = float(val)
                self.band_values[band] = float(val)
            else:
                powers[band] = 0.0
                self.band_values[band] = 0.0
        
        # Normalize powers (so they're comparable)
        total_power = sum(powers.values()) + 1e-9
        
        # Build decoherence map
        # Start with baseline decoherence everywhere
        gamma_map = np.ones((self.size, self.size), dtype=np.float32) * self.baseline
        
        for band in bands:
            mask = self.band_masks[band]
            # High power → LOW decoherence (protected)
            # Normalized power scaled by sensitivity
            normalized_power = powers[band] / (total_power + 1e-9)
            
            # Protection amount: high power = low gamma (low decoherence)
            protection_boost = normalized_power * self.sensitivity
            
            # Apply: reduce decoherence where this band is active
            gamma_map[mask] = np.clip(
                self.baseline - protection_boost,
                0.05,  # Never fully protected
                0.95   # Never fully decoherent
            )
        
        # Smooth transitions between bands (Gaussian blur)
        gamma_map = cv2.GaussianBlur(gamma_map, (9, 9), 0)
        
        # Temporal smoothing
        if self.smooth_decoherence is None:
            self.smooth_decoherence = gamma_map.copy()
        else:
            self.smooth_decoherence = (
                self.smooth_decoherence * (1 - self.smooth_factor) + 
                gamma_map * self.smooth_factor
            )
        
        self.decoherence = self.smooth_decoherence.astype(np.float32)
        self.protection = 1.0 - self.decoherence
        
    def get_output(self, port_name):
        if port_name == 'decoherence_map':
            # Output as uint8 image for compatibility
            return (self.decoherence * 255).astype(np.uint8)
        elif port_name == 'protection_map':
            return (self.protection * 255).astype(np.uint8)
        elif port_name == 'dominant_band':
            # Return index of dominant band (0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma)
            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            values = [self.band_values[b] for b in bands]
            return float(np.argmax(values))
        elif port_name == 'total_power':
            return float(sum(self.band_values.values()))
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # Create side-by-side view: Protection map + Band bars
        display_w = w * 2
        display = np.zeros((h, display_w, 3), dtype=np.uint8)
        
        # Left: Protection map (colorized)
        prot_vis = (self.protection * 255).astype(np.uint8)
        prot_color = cv2.applyColorMap(prot_vis, cv2.COLORMAP_VIRIDIS)
        display[:, :w] = prot_color
        
        # Right: Band power bars
        bar_panel = np.zeros((h, w, 3), dtype=np.uint8)
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        colors = [
            (255, 100, 100),  # Delta - red
            (255, 200, 100),  # Theta - orange
            (100, 255, 100),  # Alpha - green
            (100, 200, 255),  # Beta - cyan
            (200, 100, 255),  # Gamma - purple
        ]
        
        bar_width = w // 5
        max_val = max(self.band_values.values()) + 1e-9
        
        for i, (band, color) in enumerate(zip(bands, colors)):
            x = i * bar_width
            val = self.band_values[band]
            bar_h = int((val / max_val) * (h - 20))
            
            # Draw bar from bottom
            cv2.rectangle(bar_panel, 
                         (x + 2, h - bar_h - 10), 
                         (x + bar_width - 2, h - 10),
                         color, -1)
            
            # Label
            cv2.putText(bar_panel, band[0].upper(), (x + 5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        display[:, w:] = bar_panel
        
        # Labels
        cv2.putText(display, "Protection", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(display, "EEG Bands", (w + 5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw ring boundaries on protection map (faint)
        center = w // 2
        for r_frac in [0.12, 0.25, 0.40, 0.65]:
            r = int(r_frac * center)
            cv2.circle(display, (center, h // 2), r, (100, 100, 100), 1)
        
        return QtGui.QImage(display.data, display_w, h, display_w * 3, 
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Baseline Decoherence", "baseline", self.baseline, "float"),
            ("Band Sensitivity", "sensitivity", self.sensitivity, "float"),
            ("Temporal Smoothing", "smooth_factor", self.smooth_factor, "float"),
        ]


class EEGAddressAnalyzerNode(BaseNode):
    """
    Combines EEG → Decoherence with Mode Address Algebra in one node.
    
    Takes EEG bands directly, computes stable address, outputs metrics.
    This is the "does your brain state have a signature in address space?" node.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "EEG Address Analyzer"
    NODE_COLOR = QtGui.QColor(80, 200, 160)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'delta': 'signal',
            'theta': 'signal', 
            'alpha': 'signal',
            'beta': 'signal',
            'gamma': 'signal',
            'field_in': 'complex_spectrum',  # Optional external field
        }
        
        self.outputs = {
            'stable_address': 'image',
            'eeg_protection': 'image',
            'address_entropy': 'signal',
            'address_centroid': 'signal',  # Where in k-space is the stable address centered
            'state_signature': 'spectrum',  # Compact descriptor of current brain-address
        }
        
        self.size = 128
        center = self.size // 2
        
        # Coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        kx = (x - center) / center
        ky = (y - center) / center
        self.k_radius = np.sqrt(kx**2 + ky**2)
        
        # Band masks (same as bridge node)
        self.band_masks = {
            'delta': (self.k_radius < 0.12),
            'theta': (self.k_radius >= 0.12) & (self.k_radius < 0.25),
            'alpha': (self.k_radius >= 0.25) & (self.k_radius < 0.40),
            'beta':  (self.k_radius >= 0.40) & (self.k_radius < 0.65),
            'gamma': (self.k_radius >= 0.65)
        }
        
        # State
        self.protection = np.zeros((self.size, self.size), dtype=np.float32)
        self.stable_address = np.zeros((self.size, self.size), dtype=np.float32)
        self.psi = None
        
        # Metrics
        self.entropy = 0.0
        self.centroid = 0.0
        self.signature = np.zeros(8, dtype=np.float32)
        
        # History for signature stability
        self.address_history = []
        self.history_len = 30
        
    def step(self):
        # Get EEG bands
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        powers = {}
        for band in bands:
            val = self.get_blended_input(band, 'sum')
            powers[band] = float(val) if val is not None else 0.0
        
        total_power = sum(powers.values()) + 1e-9
        
        # Build protection map from EEG
        protection = np.ones((self.size, self.size), dtype=np.float32) * 0.3
        
        for band in bands:
            mask = self.band_masks[band]
            norm_power = powers[band] / total_power
            protection[mask] = 0.3 + norm_power * 0.6  # 0.3 to 0.9 range
        
        protection = cv2.GaussianBlur(protection, (7, 7), 0)
        self.protection = protection
        
        # Get or generate field
        field_in = self.get_blended_input('field_in', 'first')
        if field_in is not None and field_in.shape == (self.size, self.size):
            self.psi = field_in.astype(np.complex64)
        else:
            # Generate field based on EEG (band powers seed the frequencies)
            psi = np.zeros((self.size, self.size), dtype=np.complex64)
            for i, band in enumerate(bands):
                mask = self.band_masks[band]
                amplitude = powers[band] / total_power
                phase = np.random.random() * 2 * np.pi
                psi[mask] = amplitude * np.exp(1j * phase)
            self.psi = psi
        
        # Compute occupied address (where amplitude is)
        magnitude = np.abs(self.psi)
        max_mag = np.max(magnitude) + 1e-9
        occupied = (magnitude / max_mag > 0.01).astype(np.float32)
        
        # Compute stable address = occupied AND protected
        protected = (protection > 0.5).astype(np.float32)
        self.stable_address = occupied * protected
        
        # Metrics
        # Entropy
        weights = magnitude ** 2
        weights = weights / (np.sum(weights) + 1e-9)
        log_w = np.log(weights + 1e-12)
        self.entropy = -np.sum(weights * log_w * self.stable_address)
        
        # Centroid (average radius of stable address)
        stable_mask = self.stable_address > 0.5
        if np.any(stable_mask):
            self.centroid = np.mean(self.k_radius[stable_mask])
        else:
            self.centroid = 0.0
        
        # Signature: compact 8D descriptor
        # [delta_stable, theta_stable, alpha_stable, beta_stable, gamma_stable, 
        #  entropy, centroid, total_stable_fraction]
        sig = np.zeros(8, dtype=np.float32)
        for i, band in enumerate(bands):
            mask = self.band_masks[band]
            sig[i] = np.mean(self.stable_address[mask])
        sig[5] = self.entropy / 10.0  # Normalize
        sig[6] = self.centroid
        sig[7] = np.mean(self.stable_address)
        self.signature = sig
        
        # Track history for stability analysis
        self.address_history.append(self.stable_address.copy())
        if len(self.address_history) > self.history_len:
            self.address_history.pop(0)
    
    def get_output(self, port_name):
        if port_name == 'stable_address':
            return (self.stable_address * 255).astype(np.uint8)
        elif port_name == 'eeg_protection':
            return (self.protection * 255).astype(np.uint8)
        elif port_name == 'address_entropy':
            return float(self.entropy)
        elif port_name == 'address_centroid':
            return float(self.centroid)
        elif port_name == 'state_signature':
            return self.signature
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # 2x2 grid: Protection, Stable Address, Signature bars, Centroid history
        display = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-left: Protection map
        prot_vis = (self.protection * 255).astype(np.uint8)
        prot_color = cv2.applyColorMap(prot_vis, cv2.COLORMAP_VIRIDIS)
        display[:h, :w] = prot_color
        cv2.putText(display, "EEG Protection", (5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Top-right: Stable address
        stable_vis = np.zeros((h, w, 3), dtype=np.uint8)
        stable_vis[:, :, 1] = (self.stable_address * 255).astype(np.uint8)  # Green
        display[:h, w:] = stable_vis
        cv2.putText(display, "Stable Address", (w + 5, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Bottom-left: Signature bars
        sig_panel = np.zeros((h, w, 3), dtype=np.uint8)
        labels = ['D', 'T', 'A', 'B', 'G', 'E', 'C', 'F']
        bar_w = w // 8
        for i, (val, label) in enumerate(zip(self.signature, labels)):
            x = i * bar_w
            bar_h = int(np.clip(val, 0, 1) * (h - 20))
            color = (100, 200, 100) if i < 5 else (200, 200, 100)
            cv2.rectangle(sig_panel, (x + 1, h - bar_h - 10), (x + bar_w - 1, h - 10), color, -1)
            cv2.putText(sig_panel, label, (x + 2, h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        display[h:, :w] = sig_panel
        cv2.putText(display, "Signature", (5, h + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Bottom-right: Metrics text
        metrics_panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(metrics_panel, f"Entropy: {self.entropy:.2f}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        cv2.putText(metrics_panel, f"Centroid: {self.centroid:.3f}", (5, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        cv2.putText(metrics_panel, f"Stable%: {self.signature[7]*100:.1f}%", (5, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        
        # Dominant band
        bands = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
        dom_idx = int(np.argmax(self.signature[:5]))
        cv2.putText(metrics_panel, f"Dominant: {bands[dom_idx]}", (5, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
        
        display[h:, w:] = metrics_panel
        cv2.putText(display, "Metrics", (w + 5, h + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(display.data, w * 2, h * 2, w * 2 * 3,
                           QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("History Length", "history_len", self.history_len, "int"),
        ]