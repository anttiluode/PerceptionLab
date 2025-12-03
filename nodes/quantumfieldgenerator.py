"""
Quantum Field Generator Node
============================
Generates complex phase fields for IHT-AI experiments.

Produces:
- complex_spectrum: The quantum field ψ in k-space
- decoherence_map: γ(k) landscape showing where modes decay
- hamiltonian_phase: H structure showing mode coupling

This is the "source" node that feeds into Mode Address Algebra
and other IHT nodes.

Modes:
- Attractor: Coherent structure at specific frequencies
- Noise: Quantum foam / random field
- Harmonic: Multiple frequency components (like EEG bands)
- Soliton: Localized stable structure
- Mixed: Combination of above
"""

import numpy as np
import cv2
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


class QuantumFieldGeneratorNode(BaseNode):
    """
    Generates complex quantum fields for IHT experiments.
    
    Output is in k-space (frequency domain) as complex_spectrum.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Quantum Field Generator"
    NODE_COLOR = QtGui.QColor(150, 50, 200)  # Purple - the color of possibility
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'modulation': 'image',      # Optional external modulation
            'frequency_bias': 'signal', # Shift the center frequency
            'coherence': 'signal',      # 0-1 noise vs structure
            'evolution_rate': 'signal'  # How fast the field evolves
        }
        
        self.outputs = {
            'complex_spectrum': 'complex_spectrum',  # Main output: ψ(k)
            'decoherence_map': 'image',              # γ(k) landscape
            'hamiltonian_phase': 'image',            # H structure
            'magnitude_view': 'image',               # |ψ(k)| for display
            'phase_view': 'image'                    # arg(ψ(k)) for display
        }
        
        self.size = 128
        center = self.size // 2
        
        # Coordinate grids
        y, x = np.ogrid[:self.size, :self.size]
        self.kx = (x - center).astype(np.float32) / self.size
        self.ky = (y - center).astype(np.float32) / self.size
        self.k_radius = np.sqrt(self.kx**2 + self.ky**2)
        self.k_angle = np.arctan2(self.ky, self.kx)
        
        # The quantum field ψ
        self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
        
        # Internal phase accumulator (for time evolution)
        self.phase_accum = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Decoherence landscape γ(k)
        # High frequencies decohere faster (realistic)
        self.gamma = np.clip(self.k_radius * 3.0, 0, 0.95).astype(np.float32)
        
        # Hamiltonian phase structure
        # Determines how modes couple/evolve
        self.H_phase = (self.k_radius * 10.0).astype(np.float32)
        
        # Parameters
        self.mode = 0  # 0=Attractor, 1=Noise, 2=Harmonic, 3=Soliton, 4=Mixed
        self.base_coherence = 0.7
        self.evolution_speed = 0.05
        self.attractor_freqs = [3, 5, 8, 13]  # Fibonacci-ish frequencies
        
        # Time counter
        self.t = 0
        
    def generate_attractor_field(self, coherence):
        """Generate coherent attractor structure at specific frequencies"""
        field = np.zeros((self.size, self.size), dtype=np.complex64)
        
        center = self.size // 2
        
        for i, freq in enumerate(self.attractor_freqs):
            # Create ring at this frequency
            ring_mask = np.abs(self.k_radius * self.size - freq) < 2.0
            
            # Phase varies around the ring (creates spiral structure)
            ring_phase = self.k_angle * (i + 1) + self.phase_accum[ring_mask].mean() if ring_mask.any() else 0
            
            # Add coherent component
            amplitude = coherence * np.exp(-((self.k_radius * self.size - freq) ** 2) / 4.0)
            field += amplitude * np.exp(1j * (ring_phase + self.t * freq * 0.1))
        
        return field
    
    def generate_noise_field(self, strength):
        """Generate quantum foam / random fluctuations"""
        noise_real = np.random.randn(self.size, self.size).astype(np.float32)
        noise_imag = np.random.randn(self.size, self.size).astype(np.float32)
        return (noise_real + 1j * noise_imag).astype(np.complex64) * strength
    
    def generate_harmonic_field(self, coherence):
        """Generate multiple harmonic components (like brain rhythms)"""
        field = np.zeros((self.size, self.size), dtype=np.complex64)
        
        # Delta (1-4 Hz equivalent) - low frequency, high amplitude
        delta_mask = self.k_radius < 0.05
        field[delta_mask] += coherence * 2.0 * np.exp(1j * self.t * 0.02)
        
        # Theta (4-8 Hz) 
        theta_mask = (self.k_radius >= 0.05) & (self.k_radius < 0.1)
        field[theta_mask] += coherence * 1.5 * np.exp(1j * self.t * 0.05)
        
        # Alpha (8-13 Hz)
        alpha_mask = (self.k_radius >= 0.1) & (self.k_radius < 0.15)
        field[alpha_mask] += coherence * 1.0 * np.exp(1j * self.t * 0.1)
        
        # Beta (13-30 Hz)
        beta_mask = (self.k_radius >= 0.15) & (self.k_radius < 0.25)
        field[beta_mask] += coherence * 0.7 * np.exp(1j * self.t * 0.15)
        
        # Gamma (30+ Hz)
        gamma_mask = self.k_radius >= 0.25
        field[gamma_mask] += coherence * 0.3 * np.exp(1j * self.t * 0.3)
        
        return field
    
    def generate_soliton_field(self, coherence):
        """Generate localized stable structure (particle-like)"""
        # Soliton in position space, then FFT
        center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(np.float32)
        
        # Gaussian envelope with internal phase
        width = 10.0
        soliton_spatial = coherence * np.exp(-r**2 / (2 * width**2)) * \
                         np.exp(1j * (r * 0.5 + self.t * 0.1))
        
        # Transform to k-space
        return fft2(soliton_spatial).astype(np.complex64)

    def step(self):
        # Get inputs
        modulation = self.get_blended_input('modulation', 'first')
        freq_bias = self.get_blended_input('frequency_bias', 'sum')
        coherence_in = self.get_blended_input('coherence', 'sum')
        evolution_in = self.get_blended_input('evolution_rate', 'sum')
        
        # Update parameters from inputs
        coherence = self.base_coherence
        if coherence_in is not None:
            coherence = np.clip(float(coherence_in), 0.0, 1.0)
        
        evolution = self.evolution_speed
        if evolution_in is not None:
            evolution = np.clip(float(evolution_in), 0.0, 0.5)
        
        # Update time and phase accumulator
        self.t += 1
        self.phase_accum += self.H_phase * evolution
        self.phase_accum = np.mod(self.phase_accum, 2 * np.pi)
        
        # Generate field based on mode
        if self.mode == 0:  # Attractor
            self.psi = self.generate_attractor_field(coherence)
            self.psi += self.generate_noise_field(0.1 * (1 - coherence))
            
        elif self.mode == 1:  # Noise
            self.psi = self.generate_noise_field(1.0)
            
        elif self.mode == 2:  # Harmonic
            self.psi = self.generate_harmonic_field(coherence)
            self.psi += self.generate_noise_field(0.05)
            
        elif self.mode == 3:  # Soliton
            self.psi = self.generate_soliton_field(coherence)
            self.psi += self.generate_noise_field(0.05)
            
        else:  # Mixed
            self.psi = 0.3 * self.generate_attractor_field(coherence)
            self.psi += 0.3 * self.generate_harmonic_field(coherence)
            self.psi += 0.2 * self.generate_soliton_field(coherence)
            self.psi += self.generate_noise_field(0.1)
        
        # Apply external modulation if provided
        if modulation is not None:
            if modulation.ndim == 3:
                modulation = np.mean(modulation, axis=2)
            mod_resized = cv2.resize(modulation.astype(np.float32), 
                                     (self.size, self.size))
            mod_normalized = mod_resized / (np.max(mod_resized) + 1e-9)
            # Modulation affects amplitude
            self.psi *= (0.5 + 0.5 * mod_normalized)
        
        # Apply frequency bias shift if provided
        if freq_bias is not None:
            shift = int(float(freq_bias) * 10)
            self.psi = np.roll(self.psi, shift, axis=0)
            self.psi = np.roll(self.psi, shift, axis=1)
        
        # Normalize to prevent runaway
        max_amp = np.max(np.abs(self.psi))
        if max_amp > 10.0:
            self.psi /= (max_amp / 10.0)

    def get_output(self, port_name):
        if port_name == 'complex_spectrum':
            # Return the complex field - this is the main output
            return self.psi
            
        elif port_name == 'decoherence_map':
            # Return γ(k) as uint8 image
            return (self.gamma * 255).astype(np.uint8)
            
        elif port_name == 'hamiltonian_phase':
            # Return H phase structure as uint8 image
            h_normalized = (self.H_phase % (2 * np.pi)) / (2 * np.pi)
            return (h_normalized * 255).astype(np.uint8)
            
        elif port_name == 'magnitude_view':
            # Return |ψ(k)| for visualization
            mag = np.abs(fftshift(self.psi))
            mag_log = np.log(mag + 1e-9)
            mag_normalized = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-9)
            return (mag_normalized * 255).astype(np.uint8)
            
        elif port_name == 'phase_view':
            # Return phase for visualization
            phase = np.angle(fftshift(self.psi))
            phase_normalized = (phase + np.pi) / (2 * np.pi)
            return (phase_normalized * 255).astype(np.uint8)
            
        return None

    def get_display_image(self):
        h, w = self.size, self.size
        
        # Shift for display
        psi_shifted = fftshift(self.psi)
        
        # Top-Left: Magnitude (log scale)
        mag = np.abs(psi_shifted)
        mag_log = np.log(mag + 1e-9)
        mag_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-9)
        mag_vis = (mag_norm * 255).astype(np.uint8)
        mag_color = cv2.applyColorMap(mag_vis, cv2.COLORMAP_INFERNO)
        
        # Top-Right: Phase
        phase = np.angle(psi_shifted)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        phase_vis = (phase_norm * 255).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_vis, cv2.COLORMAP_HSV)
        
        # Bottom-Left: Decoherence landscape
        gamma_vis = (self.gamma * 255).astype(np.uint8)
        gamma_color = cv2.applyColorMap(gamma_vis, cv2.COLORMAP_VIRIDIS)
        
        # Bottom-Right: Combined magnitude + phase as HSV
        # Hue = phase, Value = magnitude
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:,:,0] = (phase_norm * 180).astype(np.uint8)  # Hue 0-180
        hsv[:,:,1] = 255  # Full saturation
        hsv[:,:,2] = (mag_norm * 255).astype(np.uint8)  # Value = magnitude
        combined = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Assemble
        top = np.hstack((mag_color, phase_color))
        bottom = np.hstack((gamma_color, combined))
        full = np.vstack((top, bottom))
        
        # Labels
        mode_names = ["Attractor", "Noise", "Harmonic", "Soliton", "Mixed"]
        mode_name = mode_names[self.mode] if self.mode < len(mode_names) else "Unknown"
        
        cv2.putText(full, f"|psi(k)| [{mode_name}]", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Phase", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Decoherence", (5, h + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Combined", (w + 5, h + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h*2, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Mode (0=Attr,1=Noise,2=Harm,3=Sol,4=Mix)", "mode", self.mode, None),
            ("Base Coherence", "base_coherence", self.base_coherence, None),
            ("Evolution Speed", "evolution_speed", self.evolution_speed, None),
        ]


class AttractorFieldNode(BaseNode):
    """
    Specialized generator for stable attractor patterns.
    
    Creates coherent structures at specific "address" frequencies
    that should survive in the Mode Address Algebra.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Attractor Field"
    NODE_COLOR = QtGui.QColor(200, 100, 150)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'seed_pattern': 'image',
            'stability': 'signal',
            'num_modes': 'signal'
        }
        
        self.outputs = {
            'complex_spectrum': 'complex_spectrum',
            'address_mask': 'image'  # Shows which modes are active
        }
        
        self.size = 128
        center = self.size // 2
        
        # Coordinates
        y, x = np.ogrid[:self.size, :self.size]
        self.kx = (x - center).astype(np.float32) / self.size
        self.ky = (y - center).astype(np.float32) / self.size
        self.k_radius = np.sqrt(self.kx**2 + self.ky**2)
        self.k_angle = np.arctan2(self.ky, self.kx)
        
        # Field
        self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
        self.address_mask = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Parameters
        self.stability = 0.8
        self.num_modes = 5
        self.t = 0
        
        # Protected frequency bands (low decoherence regions)
        self.protected_radii = [0.05, 0.1, 0.15, 0.2, 0.3]
        
    def step(self):
        stability_in = self.get_blended_input('stability', 'sum')
        num_modes_in = self.get_blended_input('num_modes', 'sum')
        seed = self.get_blended_input('seed_pattern', 'first')
        
        if stability_in is not None:
            self.stability = np.clip(float(stability_in), 0.1, 1.0)
        if num_modes_in is not None:
            self.num_modes = int(np.clip(float(num_modes_in), 1, 10))
        
        self.t += 1
        
        # Generate attractor at protected frequencies
        self.psi = np.zeros((self.size, self.size), dtype=np.complex64)
        self.address_mask = np.zeros((self.size, self.size), dtype=np.float32)
        
        for i in range(min(self.num_modes, len(self.protected_radii))):
            r = self.protected_radii[i]
            
            # Ring at this radius
            ring_width = 0.02
            ring = np.exp(-((self.k_radius - r) ** 2) / (2 * ring_width**2))
            
            # Phase structure (angular momentum)
            angular_mode = i + 1
            phase = angular_mode * self.k_angle + self.t * 0.05 * (i + 1)
            
            # Add to field
            amplitude = self.stability * (1.0 - 0.1 * i)  # Outer modes weaker
            self.psi += amplitude * ring * np.exp(1j * phase)
            
            # Update address mask
            self.address_mask += ring
        
        # Add small noise for realism
        noise = (np.random.randn(self.size, self.size) + 
                1j * np.random.randn(self.size, self.size)) * 0.05
        self.psi += noise.astype(np.complex64)
        
        # Apply seed modulation if provided
        if seed is not None:
            if seed.ndim == 3:
                seed = np.mean(seed, axis=2)
            seed_resized = cv2.resize(seed.astype(np.float32), (self.size, self.size))
            seed_normalized = seed_resized / (np.max(seed_resized) + 1e-9)
            # Seed modulates phase
            self.psi *= np.exp(1j * seed_normalized * np.pi)
        
        # Normalize address mask
        if self.address_mask.max() > 0:
            self.address_mask /= self.address_mask.max()
    
    def get_output(self, port_name):
        if port_name == 'complex_spectrum':
            return self.psi
        elif port_name == 'address_mask':
            return (self.address_mask * 255).astype(np.uint8)
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # Magnitude
        mag = np.abs(fftshift(self.psi))
        mag_log = np.log(mag + 1e-9)
        mag_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-9)
        mag_vis = (mag_norm * 255).astype(np.uint8)
        mag_color = cv2.applyColorMap(mag_vis, cv2.COLORMAP_PLASMA)
        
        # Address mask
        addr_vis = (fftshift(self.address_mask) * 255).astype(np.uint8)
        addr_color = cv2.applyColorMap(addr_vis, cv2.COLORMAP_VIRIDIS)
        
        # Side by side
        full = np.hstack((mag_color, addr_color))
        
        cv2.putText(full, f"Attractor (n={self.num_modes})", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Address", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Stability", "stability", self.stability, None),
            ("Num Modes", "num_modes", self.num_modes, None),
        ]


class DecoherenceFieldNode(BaseNode):
    """
    Generates configurable decoherence landscapes γ(k).
    
    Different decoherence patterns create different "protected" regions
    where attractors can stably exist.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Decoherence Field"
    NODE_COLOR = QtGui.QColor(100, 150, 100)
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'modulation': 'image',
            'center_protection': 'signal',  # How protected is DC?
            'falloff': 'signal'              # How fast does protection decay?
        }
        
        self.outputs = {
            'decoherence_map': 'image',      # γ(k) 
            'protection_map': 'image'        # π(k) = 1 - γ(k)
        }
        
        self.size = 128
        center = self.size // 2
        
        y, x = np.ogrid[:self.size, :self.size]
        self.k_radius = np.sqrt((x - center)**2 + (y - center)**2).astype(np.float32)
        self.k_radius /= center  # Normalize to 0-1
        
        self.gamma = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Parameters
        self.center_protection = 0.9  # High = DC very protected
        self.falloff = 2.0            # How fast protection drops with frequency
        self.mode = 0                 # 0=radial, 1=angular, 2=spots
        
    def step(self):
        center_in = self.get_blended_input('center_protection', 'sum')
        falloff_in = self.get_blended_input('falloff', 'sum')
        modulation = self.get_blended_input('modulation', 'first')
        
        if center_in is not None:
            self.center_protection = np.clip(float(center_in), 0.0, 1.0)
        if falloff_in is not None:
            self.falloff = np.clip(float(falloff_in), 0.5, 5.0)
        
        # Base decoherence: increases with frequency
        if self.mode == 0:  # Radial
            # Protection = center_protection * exp(-falloff * r)
            protection = self.center_protection * np.exp(-self.falloff * self.k_radius)
            self.gamma = 1.0 - protection
            
        elif self.mode == 1:  # Angular bands
            # Certain angles are protected
            center = self.size // 2
            y, x = np.ogrid[:self.size, :self.size]
            angle = np.arctan2(y - center, x - center)
            # Protect horizontal and vertical bands
            angular_protection = np.cos(4 * angle) ** 2
            radial_decay = np.exp(-self.falloff * self.k_radius * 0.5)
            protection = self.center_protection * angular_protection * radial_decay
            self.gamma = 1.0 - np.clip(protection, 0, 1)
            
        else:  # Spots (specific frequencies protected)
            protection = np.zeros((self.size, self.size), dtype=np.float32)
            # Protected spots at specific radii
            for r in [0.1, 0.2, 0.35, 0.5]:
                spot = np.exp(-((self.k_radius - r) ** 2) / 0.01)
                protection += self.center_protection * spot
            self.gamma = 1.0 - np.clip(protection, 0, 1)
        
        # Apply modulation
        if modulation is not None:
            if modulation.ndim == 3:
                modulation = np.mean(modulation, axis=2)
            mod_resized = cv2.resize(modulation.astype(np.float32), (self.size, self.size))
            mod_normalized = mod_resized / (np.max(mod_resized) + 1e-9)
            # Modulation creates additional protection
            self.gamma *= (1.0 - 0.5 * mod_normalized)
        
        self.gamma = np.clip(self.gamma, 0.0, 0.99).astype(np.float32)
    
    def get_output(self, port_name):
        if port_name == 'decoherence_map':
            return (self.gamma * 255).astype(np.uint8)
        elif port_name == 'protection_map':
            protection = 1.0 - self.gamma
            return (protection * 255).astype(np.uint8)
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        
        # Decoherence (where modes decay)
        gamma_vis = (self.gamma * 255).astype(np.uint8)
        gamma_color = cv2.applyColorMap(gamma_vis, cv2.COLORMAP_HOT)
        
        # Protection (where modes survive)
        protection = 1.0 - self.gamma
        prot_vis = (protection * 255).astype(np.uint8)
        prot_color = cv2.applyColorMap(prot_vis, cv2.COLORMAP_VIRIDIS)
        
        full = np.hstack((gamma_color, prot_color))
        
        mode_names = ["Radial", "Angular", "Spots"]
        mode_name = mode_names[self.mode] if self.mode < len(mode_names) else "?"
        
        cv2.putText(full, f"Decoherence [{mode_name}]", (5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(full, "Protection", (w + 5, 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return QtGui.QImage(full.data, w*2, h, w*2*3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Mode (0=Radial,1=Angular,2=Spots)", "mode", self.mode, None),
            ("Center Protection", "center_protection", self.center_protection, None),
            ("Falloff Rate", "falloff", self.falloff, None),
        ]
