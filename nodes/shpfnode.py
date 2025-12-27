"""
Biological Compute Node - Scale-Integrated Hybrid Point-Field System (SHPF)
============================================================================

Implements the core principles from Milinkovic & Aru (2025) 
"On biological and artificial consciousness: A case for biological computationalism"

Key features:
1. HYBRID DISCRETE-CONTINUOUS: Spikes (discrete) embedded in fields (continuous)
2. SCALE INSEPARABILITY: Bidirectional coupling via inner settle loop
3. DYNAMICO-STRUCTURAL CO-DETERMINATION: Connectivity drifts online
4. METABOLIC EMBEDDING: Energy budget constrains computation

The critical innovation is the INNER SETTLE LOOP (K iterations per timestep)
where micro↔macro settle into consistency before advancing time.
This approximates the "simultaneous mutual constraint" of biological systems.

Architecture:
- Field φ(x,t): Continuous 2D field (like ephaptic/extracellular)
- Units v_i(t): Continuous membrane potentials (128 units)
- Spikes s_i: Discrete events (point process)
- Accumulators c_i(t): Slow protein-like integrators
- Modes z(t): Macro order parameters (eigenmodes)
- Connectivity w: Drifting structural state
- Energy E(t): Metabolic budget

Author: Built for Antti's PerceptionLab
Based on: ChatGPT's SHPF framework translation of Milinkovic & Aru
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, laplace
from scipy.special import jn_zeros, jn

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class SHPFNode(BaseNode):
    """
    Scale-Integrated Hybrid Point-Field System (SHPF)
    
    A single node implementing biological computation with:
    - Bidirectional heterarchy (inner settle loop)
    - Hybrid discrete-continuous dynamics
    - Slow protein-like accumulators
    - Energy-constrained computation
    - Online structural drift
    
    Based on Milinkovic & Aru (2025) "On biological and artificial consciousness"
    """
    
    NODE_CATEGORY = "Biological"
    NODE_TITLE = "SHPF Compute"
    NODE_COLOR = QtGui.QColor(180, 100, 50)  # Organic brown
    
    def __init__(self):
        super().__init__()
        
        # === CONFIGURATION ===
        self.field_size = 64          # φ field resolution
        self.n_units = 128            # Number of "neurons"
        self.n_modes = 10             # Macro order parameters
        self.K_settle = 5             # Inner heterarchy iterations
        self.dt = 0.05                # Timestep
        
        # === INPUTS ===
        self.inputs = {
            'perturbation': 'image',      # External stimulus to field
            'energy_supply': 'signal',    # Metabolic input rate
            'external_drive': 'spectrum', # Direct unit drive (optional)
        }
        
        # === OUTPUTS ===
        self.outputs = {
            'field_image': 'image',           # φ visualization
            'mode_spectrum': 'spectrum',      # z as spectrum (10 modes)
            'spike_activity': 'spectrum',     # Recent spike counts per unit
            'unit_potentials': 'spectrum',    # v_i continuous states
            'accumulator_state': 'spectrum',  # c_i slow states
            'energy_level': 'signal',         # E scalar
            'spike_rate': 'signal',           # Total spikes this step
            'structural_entropy': 'signal',   # Connectivity change metric
            'scale_integration': 'signal',    # Bidirectional info flow
            'regime_indicator': 'signal',     # 0=quiescent, 1=critical, 2=chaotic
            'hologram': 'image',              # Combined visualization
        }
        
        # === STATE INITIALIZATION ===
        self._init_all_state()
        
        # === PRECOMPUTE BASIS FUNCTIONS ===
        self._precompute_mode_basis()
        
        # === METRICS HISTORY ===
        self.spike_history = []
        self.mode_history = []
        self.energy_history = []
        
    def _init_all_state(self):
        """Initialize all state variables"""
        
        # Field state φ(x,t) - continuous 2D
        self.phi = np.random.randn(self.field_size, self.field_size).astype(np.float32) * 0.1
        
        # Unit states v_i(t) - continuous membrane potentials
        self.v = np.random.randn(self.n_units).astype(np.float32) * 0.1
        
        # Spike state - recent spike times/counts
        self.spikes = np.zeros(self.n_units, dtype=np.float32)
        self.spike_counts = np.zeros(self.n_units, dtype=np.float32)
        
        # Slow accumulators c_i(t) - protein/phosphorylation-like
        self.c = np.zeros(self.n_units, dtype=np.float32)
        
        # Macro modes z(t) - order parameters
        self.z = np.zeros(self.n_modes, dtype=np.float32)
        
        # Connectivity w - structural state (sparse for efficiency)
        # Random sparse connectivity
        self.connectivity = np.random.randn(self.n_units, self.n_units).astype(np.float32) * 0.1
        self.connectivity *= (np.random.rand(self.n_units, self.n_units) < 0.2)  # 20% connectivity
        np.fill_diagonal(self.connectivity, 0)  # No self-connections
        self.connectivity_initial = self.connectivity.copy()  # For measuring drift
        
        # Thresholds θ - per-unit, can be modulated
        self.theta = np.ones(self.n_units, dtype=np.float32) * 0.5
        
        # Energy budget E(t)
        self.E = 1.0  # Start with full energy (Python float)
        self.E_supply_rate = 0.1  # Default supply (Python float)
        
        # Unit positions (for field coupling)
        # Arrange units on a grid-ish pattern in field space
        self.unit_positions = np.zeros((self.n_units, 2), dtype=np.int32)
        side = int(np.sqrt(self.n_units))
        for i in range(self.n_units):
            self.unit_positions[i, 0] = int((i % side) * self.field_size / side)
            self.unit_positions[i, 1] = int((i // side) * self.field_size / side)
        
        # Metrics (all Python floats)
        self.structural_entropy_val = 0.0
        self.scale_integration_val = 0.0
        self.regime_val = 1.0  # Start at critical
        self.total_spikes = 0.0
        
    def _precompute_mode_basis(self):
        """Precompute eigenmode basis functions for coarse-graining"""
        self.mode_basis = []
        
        h, w = self.field_size, self.field_size
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        x_norm = (x - cx) / (w / 2)
        y_norm = (y - cy) / (h / 2)
        r = np.sqrt(x_norm**2 + y_norm**2) + 1e-9
        theta = np.arctan2(y_norm, x_norm)
        mask = (r <= 1.0).astype(np.float32)
        
        mode_idx = 0
        for n in range(1, 4):  # Fewer modes for speed
            for m in range(0, 4):
                if mode_idx >= self.n_modes:
                    break
                try:
                    zeros = jn_zeros(m, n)
                    k = zeros[-1]
                    radial = jn(m, k * r)
                    
                    if m == 0:
                        mode = radial * mask
                    else:
                        mode = radial * np.cos(m * theta) * mask
                    
                    mode = mode / (np.linalg.norm(mode) + 1e-9)
                    self.mode_basis.append(mode.astype(np.float32))
                    mode_idx += 1
                except:
                    pass
        
        # Pad if needed
        while len(self.mode_basis) < self.n_modes:
            self.mode_basis.append(np.zeros((h, w), dtype=np.float32))
    
    # =========================================================================
    # CORE DYNAMICS - The biological computation
    # =========================================================================
    
    def _field_dynamics(self, phi, v, z, E):
        """
        Field evolution: Ginzburg-Landau + unit coupling + mode constraint
        
        dφ/dt = φ - |φ|²φ + D∇²φ + unit_influence + mode_constraint + noise
        """
        # Laplacian (diffusion)
        lap = laplace(phi).astype(np.float32) * 0.3
        
        # Nonlinear term (Ginzburg-Landau)
        nonlin = phi - phi * (phi**2) * 0.5
        
        # Unit influence on field (spikes inject energy locally)
        unit_influence = np.zeros_like(phi)
        for i in range(self.n_units):
            if self.spikes[i] > 0:
                px, py = self.unit_positions[i]
                px = np.clip(px, 1, self.field_size-2)
                py = np.clip(py, 1, self.field_size-2)
                unit_influence[py-1:py+2, px-1:px+2] += self.spikes[i] * 0.5
        
        # Mode constraint (macro → micro): modes shape field
        mode_constraint = np.zeros_like(phi)
        for m_idx, mode in enumerate(self.mode_basis):
            if m_idx < len(z):
                mode_constraint += z[m_idx] * mode * 0.1
        
        # Energy gating: low energy → less field activity
        energy_gate = np.clip(E, 0.1, 1.0)
        
        # Noise (stochasticity)
        noise = np.random.randn(*phi.shape).astype(np.float32) * 0.02 * energy_gate
        
        # Combined
        dphi = (nonlin + lap + unit_influence * 0.3 + mode_constraint) * energy_gate + noise
        
        return dphi
    
    def _unit_dynamics(self, v, phi, w, theta, z, E):
        """
        Unit evolution: Leaky integrate-and-fire with field coupling
        
        dv/dt = -v/τ + w·v + field_input + mode_modulation + noise
        """
        tau = 10.0  # Membrane time constant
        
        # Leak
        leak = -v / tau
        
        # Recurrent input (connectivity)
        recurrent = np.dot(w, v) * 0.1
        
        # Field input (sample field at unit positions)
        field_input = np.zeros(self.n_units, dtype=np.float32)
        for i in range(self.n_units):
            px, py = self.unit_positions[i]
            px = np.clip(px, 0, self.field_size-1)
            py = np.clip(py, 0, self.field_size-1)
            field_input[i] = phi[py, px]
        
        # Mode modulation (macro → micro): modes affect excitability
        mode_modulation = np.zeros(self.n_units, dtype=np.float32)
        for m_idx in range(min(len(z), self.n_modes)):
            # Different modes affect different unit subsets
            affected = np.arange(self.n_units) % self.n_modes == m_idx
            mode_modulation[affected] += z[m_idx] * 0.2
        
        # Energy gating
        energy_gate = np.clip(E, 0.1, 1.0)
        
        # Noise
        noise = np.random.randn(self.n_units).astype(np.float32) * 0.05 * energy_gate
        
        # Combined
        dv = (leak + recurrent + field_input * 0.3 + mode_modulation) * energy_gate + noise
        
        return dv
    
    def _sample_spikes(self, v, theta, E):
        """
        Sample discrete spikes from continuous potentials (point process)
        
        Spike probability depends on how far v exceeds threshold
        Energy modulates threshold (low energy → higher threshold)
        """
        # Adjust threshold by energy (low energy = harder to spike)
        effective_theta = theta / np.clip(E, 0.3, 1.0)
        
        # Spike probability (soft threshold)
        excess = v - effective_theta
        prob = 1.0 / (1.0 + np.exp(-excess * 5))  # Sigmoid
        
        # Sample
        spikes = (np.random.rand(self.n_units) < prob * self.dt * 10).astype(np.float32)
        
        # Reset spiked units
        return spikes
    
    def _accumulator_dynamics(self, c, v, phi, spikes, z):
        """
        Slow protein-like accumulators
        
        Continuous evidence accumulation → threshold transitions
        This is the "folding" - activity changes structure slowly
        """
        tau_c = 100.0  # Very slow time constant
        
        # Accumulate based on activity
        accumulation = spikes * 0.1 + np.abs(v) * 0.01
        
        # Mode influence (macro shapes what gets accumulated)
        for m_idx in range(min(len(z), self.n_modes)):
            affected = np.arange(self.n_units) % self.n_modes == m_idx
            accumulation[affected] *= (1.0 + z[m_idx] * 0.5)
        
        # Decay toward baseline
        decay = -c / tau_c
        
        dc = decay + accumulation * 0.1
        
        return dc
    
    def _coarse_grain_modes(self, phi, v, spikes):
        """
        Extract macro order parameters z from micro activity
        
        This is the UPWARD causation: micro → macro
        """
        z = np.zeros(self.n_modes, dtype=np.float32)
        
        # Project field onto mode basis
        for m_idx, mode in enumerate(self.mode_basis):
            if m_idx < self.n_modes:
                z[m_idx] = np.sum(phi * mode)
        
        # Add spike contribution (discretes affect macro)
        spike_contribution = np.zeros(self.n_modes, dtype=np.float32)
        for i in range(self.n_units):
            if spikes[i] > 0:
                m_idx = i % self.n_modes
                spike_contribution[m_idx] += spikes[i]
        
        z += spike_contribution * 0.1
        
        # Normalize
        z_norm = np.linalg.norm(z)
        if z_norm > 1e-6:
            z = z / z_norm
        
        return z
    
    def _macro_constraint_update(self, theta, w, z, E):
        """
        Macro modes constrain micro parameters
        
        This is the DOWNWARD causation: macro → micro
        """
        # Modes modulate thresholds
        for m_idx in range(min(len(z), self.n_modes)):
            affected = np.arange(self.n_units) % self.n_modes == m_idx
            # High mode activity → lower threshold for that subset
            theta[affected] *= (1.0 - z[m_idx] * 0.1 * E)
        
        # Keep thresholds bounded
        theta = np.clip(theta, 0.1, 2.0)
        
        # Modes also subtly shape connectivity
        # (This is a simplification of how fields constrain structure)
        mode_energy = np.sum(z**2)
        w *= (1.0 - 0.001 * mode_energy)  # High mode activity slightly weakens connections
        
        return theta, w
    
    def _structural_drift(self, w, theta, c, spikes, z):
        """
        Online structural modification - "the substrate is the algorithm"
        
        Connectivity and thresholds change based on activity patterns
        This is Hebbian-like but also shaped by accumulators and modes
        """
        # Hebbian: neurons that fire together wire together
        spike_outer = np.outer(spikes, spikes)
        dw_hebb = spike_outer * 0.001
        
        # Anti-Hebbian for balance
        dw_anti = -w * 0.0001
        
        # Accumulator-gated plasticity (c must be high enough)
        c_gate = np.outer(np.tanh(c), np.tanh(c))
        dw = (dw_hebb + dw_anti) * c_gate
        
        # Mode-gated: only drift when modes are active
        mode_gate = np.sum(z**2)
        dw *= mode_gate
        
        w_new = w + dw
        
        # Threshold homeostasis
        # Units that spike too much → raise threshold
        # Units that never spike → lower threshold
        spike_rate = spikes  # Instantaneous, could smooth
        dtheta = (spike_rate - 0.1) * 0.01  # Target ~10% activity
        theta_new = theta + dtheta
        theta_new = np.clip(theta_new, 0.1, 2.0)
        
        return w_new, theta_new
    
    def _energy_dynamics(self, E, spikes, v, w_change):
        """
        Metabolic budget dynamics
        
        Energy is consumed by:
        - Spikes (expensive)
        - Maintaining potentials (ion pumps)
        - Structural changes (protein synthesis)
        
        Energy gates computation when low
        """
        # Costs
        spike_cost = np.sum(spikes) * 0.01
        pump_cost = np.sum(np.abs(v)) * 0.001
        plasticity_cost = np.abs(w_change) * 0.001
        
        total_cost = spike_cost + pump_cost + plasticity_cost
        
        # Supply (from input or default)
        supply = self.E_supply_rate
        
        dE = supply - total_cost
        
        return dE
    
    def _compute_scale_integration(self, z, z_prev, v, v_prev):
        """
        Measure bidirectional information flow between scales
        
        High scale integration = micro and macro mutually predictive
        """
        if z_prev is None or v_prev is None:
            return 0.5
        
        # Macro → micro: do modes predict unit changes?
        dv = v - v_prev
        mode_prediction = np.zeros_like(dv)
        for m_idx in range(min(len(z_prev), self.n_modes)):
            affected = np.arange(self.n_units) % self.n_modes == m_idx
            mode_prediction[affected] = z_prev[m_idx]
        
        # Safe correlation (handle constant arrays)
        if np.std(mode_prediction) > 1e-9 and np.std(dv) > 1e-9:
            macro_micro_corr = np.corrcoef(mode_prediction, dv)[0, 1]
            if np.isnan(macro_micro_corr):
                macro_micro_corr = 0
        else:
            macro_micro_corr = 0
        
        # Micro → macro: do units predict mode changes?
        dz = z - z_prev
        unit_summary = np.array([np.mean(v_prev[np.arange(self.n_units) % self.n_modes == m]) 
                                  for m in range(self.n_modes)])
        
        # Safe correlation
        if np.std(unit_summary[:len(dz)]) > 1e-9 and np.std(dz) > 1e-9:
            micro_macro_corr = np.corrcoef(unit_summary[:len(dz)], dz)[0, 1]
            if np.isnan(micro_macro_corr):
                micro_macro_corr = 0
        else:
            micro_macro_corr = 0
        
        # Scale integration = geometric mean of bidirectional flow
        scale_int = np.sqrt(np.abs(macro_micro_corr) * np.abs(micro_macro_corr))
        
        return float(scale_int)
    
    def _compute_regime(self, spike_history):
        """
        Estimate dynamical regime
        
        0 = quiescent (too little activity)
        1 = critical (balanced, complex)
        2 = chaotic (too much activity)
        """
        if len(spike_history) < 10:
            return 1.0
        
        recent = np.array(spike_history[-50:])
        mean_rate = np.mean(recent)
        var_rate = np.var(recent)
        
        if mean_rate < 0.5:
            return 0.0  # Quiescent
        elif var_rate > mean_rate * 2:
            return 2.0  # Chaotic
        else:
            return 1.0  # Critical
    
    # =========================================================================
    # MAIN STEP FUNCTION
    # =========================================================================
    
    def step(self):
        """
        Main computation step with INNER HETERARCHY SETTLE LOOP
        
        This is the key innovation: K iterations of micro↔macro settling
        before advancing time
        """
        # === GET INPUTS ===
        perturb = self.get_blended_input('perturbation', 'mean')
        energy_input = self.get_blended_input('energy_supply', 'sum')
        external_drive = self.get_blended_input('external_drive', 'first')
        
        if energy_input is not None:
            self.E_supply_rate = float(np.clip(energy_input, 0, 0.5))
        
        # Apply perturbation to field
        if perturb is not None:
            if perturb.ndim == 3:
                perturb = np.mean(perturb, axis=2)
            perturb = cv2.resize(perturb.astype(np.float32), 
                                (self.field_size, self.field_size))
            if perturb.max() > 1:
                perturb = perturb / 255.0
            self.phi += (perturb - 0.5) * 0.3
        
        # Apply external drive to units
        if external_drive is not None:
            n = min(len(external_drive), self.n_units)
            self.v[:n] += external_drive[:n] * 0.1
        
        # === STORE PREVIOUS STATE FOR METRICS ===
        z_prev = self.z.copy()
        v_prev = self.v.copy()
        w_prev = self.connectivity.copy()
        
        # === INNER HETERARCHY SETTLE LOOP ===
        # This is where biological computation happens
        # K iterations of micro↔macro mutual constraint
        
        total_spikes_this_step = 0
        
        for k in range(self.K_settle):
            
            # 1) Continuous field evolution
            dphi = self._field_dynamics(self.phi, self.v, self.z, self.E)
            self.phi += dphi * self.dt
            self.phi = np.clip(self.phi, -2, 2)
            
            # 2) Continuous unit evolution
            dv = self._unit_dynamics(self.v, self.phi, self.connectivity, self.theta, self.z, self.E)
            self.v += dv * self.dt
            self.v = np.clip(self.v, -2, 2)
            
            # 3) Discrete spikes (point process)
            self.spikes = self._sample_spikes(self.v, self.theta, self.E)
            total_spikes_this_step += np.sum(self.spikes)
            
            # Reset spiked units
            self.v[self.spikes > 0] *= 0.2
            
            # 4) Slow accumulator dynamics
            dc = self._accumulator_dynamics(self.c, self.v, self.phi, self.spikes, self.z)
            self.c += dc * self.dt
            self.c = np.clip(self.c, 0, 5)
            
            # Check for accumulator threshold crossings (discrete transitions)
            crossed = self.c > 1.0
            if np.any(crossed):
                # Trigger fast plasticity for crossed units
                self.theta[crossed] *= 0.95  # Lower threshold temporarily
                self.c[crossed] = 0.0  # Reset accumulator
            
            # 5) Coarse-grain upward: micro → macro
            self.z = self._coarse_grain_modes(self.phi, self.v, self.spikes)
            
            # 6) Macro constraint downward: macro → micro
            self.theta, self.connectivity = self._macro_constraint_update(
                self.theta, self.connectivity, self.z, self.E)
        
        # === SLOW UPDATES (after settling) ===
        
        # Structural drift
        self.connectivity, self.theta = self._structural_drift(
            self.connectivity, self.theta, self.c, self.spikes, self.z)
        
        # Energy dynamics
        w_change = np.sum(np.abs(self.connectivity - w_prev))
        dE = self._energy_dynamics(self.E, self.spikes, self.v, w_change)
        self.E = float(self.E + dE * self.dt)
        self.E = float(np.clip(self.E, 0.05, 1.5))
        
        # === COMPUTE METRICS ===
        
        # Structural entropy (how much has connectivity changed from initial)
        w_diff = np.abs(self.connectivity - self.connectivity_initial)
        self.structural_entropy_val = float(np.mean(w_diff))
        
        # Scale integration
        self.scale_integration_val = self._compute_scale_integration(
            self.z, z_prev, self.v, v_prev)
        
        # Regime indicator
        self.spike_history.append(total_spikes_this_step)
        if len(self.spike_history) > 200:
            self.spike_history = self.spike_history[-200:]
        self.regime_val = self._compute_regime(self.spike_history)
        
        # Store for output
        self.total_spikes = float(total_spikes_this_step)
        self.spike_counts = self.spikes.copy()
        
        # Mode history for visualization
        self.mode_history.append(self.z.copy())
        if len(self.mode_history) > 100:
            self.mode_history = self.mode_history[-100:]
        
        self.energy_history.append(self.E)
        if len(self.energy_history) > 200:
            self.energy_history = self.energy_history[-200:]
    
    # =========================================================================
    # OUTPUTS
    # =========================================================================
    
    def get_output(self, port_name):
        if port_name == 'field_image':
            return self.phi
        elif port_name == 'mode_spectrum':
            return self.z
        elif port_name == 'spike_activity':
            return self.spike_counts
        elif port_name == 'unit_potentials':
            return self.v
        elif port_name == 'accumulator_state':
            return self.c
        elif port_name == 'energy_level':
            return float(self.E)
        elif port_name == 'spike_rate':
            return float(self.total_spikes)
        elif port_name == 'structural_entropy':
            return float(self.structural_entropy_val)
        elif port_name == 'scale_integration':
            return float(self.scale_integration_val)
        elif port_name == 'regime_indicator':
            return float(self.regime_val)
        elif port_name == 'hologram':
            return self._create_hologram()
        return None
    
    def _create_hologram(self):
        """Create combined visualization"""
        size = 256
        hologram = np.zeros((size, size, 3), dtype=np.float32)
        
        # Red channel: Field magnitude
        phi_norm = (self.phi - self.phi.min()) / (self.phi.max() - self.phi.min() + 1e-9)
        phi_big = cv2.resize(phi_norm, (size, size))
        hologram[:, :, 2] = phi_big  # Red in BGR
        
        # Green channel: Mode activity projected back to space
        mode_field = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        for m_idx, mode in enumerate(self.mode_basis):
            if m_idx < len(self.z):
                mode_field += np.abs(self.z[m_idx]) * mode
        mode_norm = (mode_field - mode_field.min()) / (mode_field.max() - mode_field.min() + 1e-9)
        mode_big = cv2.resize(mode_norm, (size, size))
        hologram[:, :, 1] = mode_big  # Green
        
        # Blue channel: Spike locations
        spike_field = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        for i in range(self.n_units):
            if self.spike_counts[i] > 0:
                px, py = self.unit_positions[i]
                px = np.clip(px, 0, self.field_size-1)
                py = np.clip(py, 0, self.field_size-1)
                spike_field[py, px] = 1.0
        spike_field = gaussian_filter(spike_field, sigma=2)
        spike_big = cv2.resize(spike_field, (size, size))
        hologram[:, :, 0] = spike_big  # Blue
        
        return np.clip(hologram, 0, 1)
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def get_display_image(self):
        """Create comprehensive display"""
        width = 400
        height = 350
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # === Top left: Field ===
        phi_norm = (self.phi - self.phi.min()) / (self.phi.max() - self.phi.min() + 1e-9)
        phi_u8 = (phi_norm * 255).astype(np.uint8)
        phi_color = cv2.applyColorMap(phi_u8, cv2.COLORMAP_TWILIGHT)
        phi_big = cv2.resize(phi_color, (100, 100))
        display[10:110, 10:110] = phi_big
        
        # === Top middle: Mode bars ===
        mode_panel = np.zeros((100, 100, 3), dtype=np.uint8)
        bar_w = 100 // self.n_modes
        for i, z_val in enumerate(self.z):
            h = int(np.clip(np.abs(z_val) * 80, 0, 80))
            color = (0, 255, 0) if z_val >= 0 else (0, 0, 255)
            cv2.rectangle(mode_panel, (i*bar_w, 90-h), ((i+1)*bar_w-1, 90), color, -1)
        display[10:110, 120:220] = mode_panel
        
        # === Top right: Hologram ===
        holo = self._create_hologram()
        holo_u8 = (holo * 255).astype(np.uint8)
        holo_small = cv2.resize(holo_u8, (100, 100))
        display[10:110, 230:330] = holo_small
        
        # === Middle: Energy bar ===
        energy_w = int(self.E * 300)
        energy_color = (0, 255, 0) if self.E > 0.3 else (0, 165, 255) if self.E > 0.1 else (0, 0, 255)
        cv2.rectangle(display, (10, 120), (10 + energy_w, 135), energy_color, -1)
        cv2.rectangle(display, (10, 120), (310, 135), (100, 100, 100), 1)
        
        # === Middle: Spike history ===
        if len(self.spike_history) > 1:
            hist = np.array(self.spike_history[-200:])
            hist_norm = hist / (hist.max() + 1)
            for i, h in enumerate(hist_norm):
                x = 10 + i
                y = 180 - int(h * 40)
                if x < 310:
                    cv2.line(display, (x, 180), (x, y), (255, 200, 100), 1)
        
        # === Bottom: Metrics text ===
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(display, f"E: {self.E:.2f}", (10, 210), font, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Spikes: {self.total_spikes:.0f}", (100, 210), font, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Scale Int: {self.scale_integration_val:.2f}", (200, 210), font, 0.4, (200, 200, 200), 1)
        
        regime_names = ["QUIESCENT", "CRITICAL", "CHAOTIC"]
        regime_colors = [(100, 100, 100), (0, 255, 0), (0, 0, 255)]
        regime_idx = int(np.clip(self.regime_val, 0, 2))
        cv2.putText(display, f"Regime: {regime_names[regime_idx]}", (10, 235), 
                   font, 0.5, regime_colors[regime_idx], 1)
        
        cv2.putText(display, f"Struct Entropy: {self.structural_entropy_val:.4f}", (10, 260), 
                   font, 0.4, (200, 200, 200), 1)
        
        # Mode values
        mode_str = " ".join([f"{z:.1f}" for z in self.z[:5]])
        cv2.putText(display, f"Modes: {mode_str}...", (10, 285), font, 0.35, (150, 255, 150), 1)
        
        # Accumulator summary
        c_mean = np.mean(self.c)
        c_max = np.max(self.c)
        cv2.putText(display, f"Accum: mean={c_mean:.2f} max={c_max:.2f}", (10, 310), 
                   font, 0.35, (255, 150, 150), 1)
        
        # Title
        cv2.putText(display, "SHPF: Scale-Integrated Hybrid Point-Field", (10, height-10), 
                   font, 0.35, (150, 150, 150), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, width, height, width*3, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("Settle Iterations (K)", "K_settle", self.K_settle, None),
            ("Timestep (dt)", "dt", self.dt, None),
            ("Energy Supply Rate", "E_supply_rate", self.E_supply_rate, None),
            ("Field Size", "field_size", self.field_size, None),
            ("Num Units", "n_units", self.n_units, None),
        ]
    
    def set_config_options(self, options):
        for key, value in options.items():
            if hasattr(self, key):
                old_val = getattr(self, key)
                setattr(self, key, type(old_val)(value))
        
        # Reinitialize if size changed
        if 'field_size' in options or 'n_units' in options:
            self._init_all_state()
            self._precompute_mode_basis()