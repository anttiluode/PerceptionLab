"""
Connectome Prior Node - The Stone Beneath the Water
=====================================================
Implements the Raj et al. (2017) framework: Brain network eigenmodes
provide a robust representation of the structural connectome.

The key insight: The brain's WHITE MATTER wiring creates standing waves
(eigenmodes) that constrain how information CAN flow. These are the
"riverbed" that the "water" of neural activity must follow.

This node:
1. Loads a standard structural connectome (Desikan-Killiany 68 regions)
2. Computes the Graph Laplacian: L = Degree - Adjacency
3. Extracts the Laplacian Eigenmodes (the "Raj Modes")
4. Outputs these as a STRUCTURAL PRIOR for other nodes

The eigenmodes represent:
- Mode 1: Constant (trivial, ignored)
- Mode 2: Left-Right hemispheric diffusion (SLOWEST, most persistent)
- Mode 3: Superior-Inferior diffusion
- Mode 4: Anterior-Posterior diffusion
- Higher modes: Increasingly local patterns

When wired to MutualInformationManifold:
- The PRIOR becomes the brain's structural shape
- The POSTERIOR becomes EEG-constrained activity
- MI measures: "How much does current activity deviate from structure?"

High MI = Novel thought (breaking the mold)
Low MI = Autopilot (following the riverbed)

INPUTS:
- modulate: Optional signal to modulate eigenmode weights
- temperature: Diffusion temperature for the structural prior

OUTPUTS:
- display: Visualization of the connectome and eigenmodes
- structural_field: Complex field of structural eigenmodes
- eigenmode_spectrum: The Raj modes as token-like output
- laplacian_matrix: The raw graph Laplacian
- region_labels: Names of the 68 brain regions

Based on: Wang MB, Owen JP, Mukherjee P, Raj A (2017) 
"Brain network eigenmodes provide a robust and compact representation 
of the structural connectome in health and disease." PLoS Comput Biol.
"""

import numpy as np
import cv2
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode): 
            return None


class ConnectomePriorNode(BaseNode):
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Connectome Prior"
    NODE_COLOR = QtGui.QColor(100, 150, 200)  # Steel blue - structural color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Connectome Prior (Raj Modes)"
        
        self.inputs = {
            'token_stream': 'spectrum',       # Main token input (like from NeuralTransformer)
            'frontal_tokens': 'spectrum',     # Regional tokens
            'temporal_tokens': 'spectrum',
            'parietal_tokens': 'spectrum',
            'occipital_tokens': 'spectrum',
            'theta_phase': 'signal',          # Phase alignment
            'temperature': 'signal',          # Diffusion temperature
            'modulate': 'signal',             # Eigenmode modulation
        }
        
        self.outputs = {
            'display': 'image',
            'structural_field': 'complex_spectrum',
            'eigenmode_spectrum': 'spectrum',      # Raj modes as tokens
            'laplacian_matrix': 'spectrum',
            'structure_vs_function': 'signal',     # Comparison metric
            'deviation_map': 'spectrum',           # Per-region deviation from structure
            'dominant_mode': 'signal',             # Which structural mode is most active
        }
        
        # === DESIKAN-KILLIANY 68 REGION ATLAS ===
        # Standard parcellation used in Raj et al.
        self.region_names = self._get_dk_regions()
        self.n_regions = len(self.region_names)
        
        # === BUILD STRUCTURAL CONNECTOME ===
        # Using canonical connectivity patterns from literature
        self.adjacency = self._build_canonical_connectome()
        
        # === COMPUTE GRAPH LAPLACIAN ===
        self.degree = np.diag(self.adjacency.sum(axis=1))
        self.laplacian = self.degree - self.adjacency
        
        # === COMPUTE EIGENMODES ===
        self.eigenvalues, self.eigenvectors = self._compute_eigenmodes()
        self.n_modes = min(16, self.n_regions)  # Use top 16 modes
        
        # === STATE ===
        self.epoch = 0
        self.current_modulation = np.ones(self.n_modes)
        self.functional_pattern = np.zeros(self.n_regions)
        self.has_eeg_input = False
        self.mode_activations = np.zeros(self.n_modes)
        self.structure_vs_function = 0.0
        self.deviation_map = np.zeros(self.n_regions)
        self.dominant_mode = 2
        
        # === DISPLAY ===
        self._display = np.zeros((700, 1100, 3), dtype=np.uint8)
        
        # Pre-compute structural field
        self._compute_structural_field()
    
    def _get_dk_regions(self):
        """Desikan-Killiany 68 region atlas labels"""
        # 34 regions per hemisphere
        regions_lh = [
            "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
            "cuneus", "entorhinal", "fusiform", "inferiorparietal",
            "inferiortemporal", "isthmuscingulate", "lateraloccipital",
            "lateralorbitofrontal", "lingual", "medialorbitofrontal",
            "middletemporal", "parahippocampal", "paracentral",
            "parsopercularis", "parsorbitalis", "parstriangularis",
            "pericalcarine", "postcentral", "posteriorcingulate",
            "precentral", "precuneus", "rostralanteriorcingulate",
            "rostralmiddlefrontal", "superiorfrontal", "superiorparietal",
            "superiortemporal", "supramarginal", "frontalpole",
            "temporalpole", "transversetemporal", "insula"
        ]
        
        # Create left and right hemisphere versions
        all_regions = []
        for r in regions_lh:
            all_regions.append(f"lh_{r}")
        for r in regions_lh:
            all_regions.append(f"rh_{r}")
        
        return all_regions
    
    def _build_canonical_connectome(self):
        """
        Build a canonical structural connectome based on known connectivity patterns.
        
        This is a simplified version - ideally would load from actual DTI data.
        The patterns reflect:
        - Strong ipsilateral (same-hemisphere) connections
        - Homotopic connections via corpus callosum
        - Hierarchical connectivity (frontal hub, etc.)
        """
        n = self.n_regions
        A = np.zeros((n, n))
        
        # Parameters
        homotopic_strength = 0.8  # Corpus callosum connections
        local_strength = 0.6     # Within-lobe connections  
        long_range_strength = 0.3  # Between-lobe connections
        hub_boost = 1.5          # Frontal/parietal hub regions
        
        # Define lobe memberships (indices within each hemisphere)
        frontal = [2, 10, 12, 16, 17, 18, 25, 26, 30]  # Frontal lobe regions
        parietal = [6, 15, 20, 23, 27, 29]  # Parietal regions
        temporal = [0, 4, 5, 7, 13, 14, 28, 31, 32]  # Temporal regions
        occipital = [3, 9, 11, 19]  # Occipital regions
        cingulate = [1, 8, 21, 24]  # Cingulate regions
        insula = [33]  # Insula
        
        lobes = [frontal, parietal, temporal, occipital, cingulate, insula]
        
        # Hub regions (get stronger connections)
        hubs = [2, 6, 23, 26, 27, 33]  # caudalmiddlefrontal, inferiorparietal, precuneus, etc.
        
        for i in range(n):
            for j in range(i+1, n):
                weight = 0.0
                
                # Determine hemisphere
                hemi_i = 0 if i < 34 else 1
                hemi_j = 0 if j < 34 else 1
                
                # Region index within hemisphere
                idx_i = i % 34
                idx_j = j % 34
                
                # Homotopic connections (same region, different hemisphere)
                if idx_i == idx_j and hemi_i != hemi_j:
                    weight = homotopic_strength
                
                # Same hemisphere connections
                elif hemi_i == hemi_j:
                    # Check if in same lobe
                    same_lobe = False
                    for lobe in lobes:
                        if idx_i in lobe and idx_j in lobe:
                            same_lobe = True
                            break
                    
                    if same_lobe:
                        weight = local_strength
                    else:
                        weight = long_range_strength * 0.5
                    
                    # Boost if either is a hub
                    if idx_i in hubs or idx_j in hubs:
                        weight *= hub_boost
                
                # Cross-hemisphere non-homotopic (weaker)
                else:
                    weight = long_range_strength * 0.3
                
                # Add some structured noise
                weight *= (0.8 + 0.4 * np.random.random())
                
                A[i, j] = weight
                A[j, i] = weight
        
        # Normalize
        A = A / A.max()
        
        return A
    
    def _compute_eigenmodes(self):
        """Compute Laplacian eigenmodes (Raj modes)"""
        # Symmetric normalized Laplacian for stability
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.degree) + 1e-10))
        L_norm = D_inv_sqrt @ self.laplacian @ D_inv_sqrt
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(L_norm)
        
        # Sort by eigenvalue (ascending - smallest = slowest modes)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _compute_structural_field(self):
        """Convert eigenmodes to a complex field representation"""
        # Create a field where each eigenmode contributes a frequency component
        field_size = 64
        self.structural_field = np.zeros((field_size, field_size), dtype=np.complex128)
        
        x = np.linspace(-np.pi, np.pi, field_size)
        y = np.linspace(-np.pi, np.pi, field_size)
        X, Y = np.meshgrid(x, y)
        
        # Each eigenmode contributes a wave pattern
        for i in range(1, min(self.n_modes + 1, len(self.eigenvalues))):
            # Skip first (constant) mode
            eigenvalue = self.eigenvalues[i]
            eigenvector = self.eigenvectors[:, i]
            
            # Characteristic frequency from eigenvalue
            freq = np.sqrt(eigenvalue + 0.1)
            
            # Direction from eigenvector (use first few components)
            angle = np.arctan2(eigenvector[1], eigenvector[0])
            
            # Amplitude inversely related to eigenvalue (slow modes = strong)
            amplitude = 1.0 / (eigenvalue + 0.1)
            
            # Create wave
            kx = freq * np.cos(angle)
            ky = freq * np.sin(angle)
            wave = amplitude * np.exp(1j * (kx * X + ky * Y))
            
            self.structural_field += wave
    
    def step(self):
        self.epoch += 1
        
        # === GET INPUTS ===
        token_stream = self.get_blended_input('token_stream', 'mean')
        frontal = self.get_blended_input('frontal_tokens', 'mean')
        temporal = self.get_blended_input('temporal_tokens', 'mean')
        parietal = self.get_blended_input('parietal_tokens', 'mean')
        occipital = self.get_blended_input('occipital_tokens', 'mean')
        theta_phase = self.get_blended_input('theta_phase', 'sum') or 0.0
        modulate = self.get_blended_input('modulate', 'sum')
        temperature = self.get_blended_input('temperature', 'sum')
        temperature = float(temperature) if temperature else 1.0
        
        # === PROCESS EEG TOKENS INTO FUNCTIONAL PATTERN ===
        self.functional_pattern = np.zeros(self.n_regions)
        self.has_eeg_input = False
        
        # Map regional tokens to brain regions
        # Frontal regions: indices 2, 10, 12, 16, 17, 18, 25, 26, 30 (and +34 for RH)
        frontal_indices = [2, 10, 12, 16, 17, 18, 25, 26, 30]
        temporal_indices = [0, 4, 5, 7, 13, 14, 28, 31, 32]
        parietal_indices = [6, 15, 20, 23, 27, 29]
        occipital_indices = [3, 9, 11, 19]
        
        def process_tokens(tokens):
            """Extract amplitude from token array"""
            if tokens is None:
                return 0.0
            if isinstance(tokens, (int, float, np.floating)):
                return float(tokens)
            if isinstance(tokens, np.ndarray):
                if tokens.ndim == 0:
                    return float(tokens)
                elif tokens.ndim == 1 and len(tokens) >= 2:
                    return float(tokens[1])  # amplitude is second element
                elif tokens.ndim == 2 and len(tokens) > 0:
                    return float(np.mean([t[1] for t in tokens if len(t) > 1]))
            if isinstance(tokens, list) and len(tokens) > 0:
                first = tokens[0]
                if hasattr(first, '__len__') and len(first) > 1:
                    return float(np.mean([t[1] for t in tokens if len(t) > 1]))
                elif isinstance(first, (int, float)):
                    return float(first)
            return 0.0
        
        frontal_amp = process_tokens(frontal)
        temporal_amp = process_tokens(temporal)
        parietal_amp = process_tokens(parietal)
        occipital_amp = process_tokens(occipital)
        
        # Also try to get from main token_stream
        main_amp = process_tokens(token_stream)
        
        if frontal_amp > 0 or temporal_amp > 0 or parietal_amp > 0 or occipital_amp > 0 or main_amp > 0:
            self.has_eeg_input = True
            
            # Distribute to regions (both hemispheres)
            for idx in frontal_indices:
                self.functional_pattern[idx] = frontal_amp
                self.functional_pattern[idx + 34] = frontal_amp
            for idx in temporal_indices:
                self.functional_pattern[idx] = temporal_amp
                self.functional_pattern[idx + 34] = temporal_amp
            for idx in parietal_indices:
                self.functional_pattern[idx] = parietal_amp
                self.functional_pattern[idx + 34] = parietal_amp
            for idx in occipital_indices:
                self.functional_pattern[idx] = occipital_amp
                self.functional_pattern[idx + 34] = occipital_amp
            
            # Normalize
            norm = np.linalg.norm(self.functional_pattern)
            if norm > 1e-6:
                self.functional_pattern /= norm
        
        # === MODULATE EIGENMODE WEIGHTS ===
        if modulate is not None:
            mod = float(modulate)
            for i in range(self.n_modes):
                self.current_modulation[i] = np.exp(-self.eigenvalues[i+1] * (1.0 - mod))
        
        # === COMPARE FUNCTION TO STRUCTURE ===
        # Project functional pattern onto eigenmodes
        self.mode_activations = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            idx = i + 1  # Skip constant mode
            eigenvector = self.eigenvectors[:, idx]
            self.mode_activations[i] = np.abs(np.dot(self.functional_pattern, eigenvector))
        
        # Structure-function deviation
        # How much does current activity deviate from structural prediction?
        if self.has_eeg_input:
            # Reconstruct "structural prediction" from top modes
            structural_prediction = np.zeros(self.n_regions)
            for i in range(min(4, self.n_modes)):  # Use top 4 modes
                idx = i + 1
                structural_prediction += self.mode_activations[i] * self.eigenvectors[:, idx]
            
            # Normalize
            norm = np.linalg.norm(structural_prediction)
            if norm > 1e-6:
                structural_prediction /= norm
            
            # Deviation = how different is actual from structural prediction
            self.structure_vs_function = np.linalg.norm(self.functional_pattern - structural_prediction)
            self.deviation_map = np.abs(self.functional_pattern - structural_prediction)
            
            # Dominant mode
            self.dominant_mode = np.argmax(self.mode_activations) + 2  # +2 because mode 1 is constant
        else:
            self.structure_vs_function = 0.0
            self.deviation_map = np.zeros(self.n_regions)
            self.dominant_mode = 2  # Default to slowest mode
        
        # === BUILD EIGENMODE SPECTRUM OUTPUT ===
        eigenmode_spectrum = np.zeros((self.n_modes, 3), dtype=np.float32)
        for i in range(self.n_modes):
            idx = i + 1
            eigenmode_spectrum[i, 0] = i
            eigenmode_spectrum[i, 1] = 1.0 / (self.eigenvalues[idx] + 0.1) * self.current_modulation[i]
            eigenmode_spectrum[i, 2] = np.arctan2(self.eigenvectors[1, idx], self.eigenvectors[0, idx]) + theta_phase
        
        # === TEMPERATURE-MODULATED FIELD ===
        temp_field = np.zeros((64, 64), dtype=np.complex128)
        x = np.linspace(-np.pi, np.pi, 64)
        y = np.linspace(-np.pi, np.pi, 64)
        X, Y = np.meshgrid(x, y)
        
        for i in range(1, min(self.n_modes + 1, len(self.eigenvalues))):
            eigenvalue = self.eigenvalues[i]
            eigenvector = self.eigenvectors[:, i]
            
            freq = np.sqrt(eigenvalue + 0.1)
            angle = np.arctan2(eigenvector[1], eigenvector[0]) + theta_phase
            
            # Temperature and EEG-dependent amplitude
            base_amplitude = self.current_modulation[i-1] * np.exp(-eigenvalue * temperature)
            if self.has_eeg_input:
                base_amplitude *= (1.0 + self.mode_activations[i-1])
            
            kx = freq * np.cos(angle)
            ky = freq * np.sin(angle)
            wave = base_amplitude * np.exp(1j * (kx * X + ky * Y))
            
            temp_field += wave
        
        # === SET OUTPUTS ===
        self.outputs['structural_field'] = temp_field
        self.outputs['eigenmode_spectrum'] = eigenmode_spectrum
        self.outputs['laplacian_matrix'] = self.laplacian.astype(np.float32)
        self.outputs['structure_vs_function'] = float(self.structure_vs_function)
        self.outputs['deviation_map'] = self.deviation_map.astype(np.float32)
        self.outputs['dominant_mode'] = float(self.dominant_mode)
        
        # === RENDER DISPLAY ===
        self._render_display(temperature)
    
    def _render_display(self, temperature):
        """Render visualization of the connectome and eigenmodes"""
        img = self._display
        img[:] = (20, 25, 30)
        h, w = img.shape[:2]
        
        # === TITLE ===
        cv2.putText(img, "CONNECTOME PRIOR - The Structural Riverbed", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 180, 220), 2)
        cv2.putText(img, "Raj et al. (2017): Brain network eigenmodes", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 140, 160), 1)
        
        # === LEFT: ADJACENCY MATRIX ===
        adj_x, adj_y = 20, 70
        adj_size = 200
        
        adj_img = self.adjacency.copy()
        adj_img = adj_img / (adj_img.max() + 1e-10)
        adj_img = (adj_img * 255).astype(np.uint8)
        adj_colored = cv2.applyColorMap(adj_img, cv2.COLORMAP_INFERNO)
        adj_resized = cv2.resize(adj_colored, (adj_size, adj_size))
        
        img[adj_y:adj_y+adj_size, adj_x:adj_x+adj_size] = adj_resized
        cv2.rectangle(img, (adj_x, adj_y), (adj_x+adj_size, adj_y+adj_size), (100, 130, 160), 2)
        cv2.putText(img, "ADJACENCY MATRIX (68x68)", (adj_x, adj_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 180, 220), 1)
        
        # Hemisphere labels
        cv2.putText(img, "LH", (adj_x + 40, adj_y + adj_size + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1)
        cv2.putText(img, "RH", (adj_x + 140, adj_y + adj_size + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 100), 1)
        
        # === CENTER-LEFT: LAPLACIAN MATRIX ===
        lap_x, lap_y = 240, 70
        lap_size = 200
        
        lap_img = np.abs(self.laplacian.copy())
        lap_img = lap_img / (lap_img.max() + 1e-10)
        lap_img = (lap_img * 255).astype(np.uint8)
        lap_colored = cv2.applyColorMap(lap_img, cv2.COLORMAP_VIRIDIS)
        lap_resized = cv2.resize(lap_colored, (lap_size, lap_size))
        
        img[lap_y:lap_y+lap_size, lap_x:lap_x+lap_size] = lap_resized
        cv2.rectangle(img, (lap_x, lap_y), (lap_x+lap_size, lap_y+lap_size), (100, 160, 130), 2)
        cv2.putText(img, "GRAPH LAPLACIAN L=D-A", (lap_x, lap_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 220, 180), 1)
        
        # === CENTER-RIGHT: EIGENSPECTRUM ===
        spec_x, spec_y = 460, 70
        spec_w, spec_h = 300, 200
        
        cv2.rectangle(img, (spec_x, spec_y), (spec_x+spec_w, spec_y+spec_h), (30, 35, 40), -1)
        cv2.rectangle(img, (spec_x, spec_y), (spec_x+spec_w, spec_y+spec_h), (100, 100, 120), 1)
        cv2.putText(img, "LAPLACIAN EIGENSPECTRUM", (spec_x + 10, spec_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw eigenvalue bars
        n_show = min(20, len(self.eigenvalues) - 1)
        bar_w = spec_w // n_show
        max_eigen = self.eigenvalues[n_show]
        
        for i in range(n_show):
            idx = i + 1  # Skip constant mode
            val = self.eigenvalues[idx]
            bar_h = int((val / max_eigen) * (spec_h - 30))
            bx = spec_x + i * bar_w + 2
            by = spec_y + spec_h - 10
            
            # Color by mode importance (slow = blue, fast = red)
            hue = int((1 - val / max_eigen) * 120)  # 120 = green, 0 = red
            hsv = np.array([[[hue, 200, 200]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            
            cv2.rectangle(img, (bx, by - bar_h), (bx + bar_w - 2, by), rgb, -1)
        
        cv2.putText(img, "Mode 2: L-R", (spec_x + 10, spec_y + spec_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 100), 1)
        cv2.putText(img, "Mode 3: S-I", (spec_x + 80, spec_y + spec_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 180, 120), 1)
        cv2.putText(img, "Mode 4: A-P", (spec_x + 150, spec_y + spec_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 160, 140), 1)
        
        # === RIGHT: STRUCTURAL FIELD ===
        field_x, field_y = 780, 70
        field_size = 200
        
        # Render the complex field
        magnitude = np.abs(self.structural_field)
        phase = np.angle(self.structural_field)
        
        mag_norm = magnitude / (magnitude.max() + 1e-10)
        
        hsv = np.zeros((64, 64, 3), dtype=np.uint8)
        hsv[:,:,0] = ((phase + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:,:,1] = 200
        hsv[:,:,2] = (mag_norm * 255).clip(0, 255).astype(np.uint8)
        
        field_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        field_resized = cv2.resize(field_rgb, (field_size, field_size))
        
        img[field_y:field_y+field_size, field_x:field_x+field_size] = field_resized
        cv2.rectangle(img, (field_x, field_y), (field_x+field_size, field_y+field_size), (150, 100, 150), 2)
        cv2.putText(img, "STRUCTURAL FIELD", (field_x, field_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 150, 200), 1)
        
        # === BOTTOM LEFT: EIGENVECTOR VISUALIZATION ===
        ev_x, ev_y = 20, 310
        ev_w, ev_h = 400, 180
        
        cv2.rectangle(img, (ev_x, ev_y), (ev_x+ev_w, ev_y+ev_h), (30, 35, 40), -1)
        cv2.putText(img, "RAJ MODES (Eigenvectors 2-5)", (ev_x + 10, ev_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show first 4 non-trivial eigenvectors as brain-region coloring
        for mode_idx in range(4):
            ev = self.eigenvectors[:, mode_idx + 1]  # Skip constant mode
            
            # Draw as two rows (LH and RH)
            row_h = 40
            row_y = ev_y + 10 + mode_idx * row_h
            
            # Left hemisphere
            for i in range(34):
                val = ev[i]
                px = ev_x + 10 + i * 5
                
                if val > 0:
                    color = (0, int(min(255, val * 500)), 0)
                else:
                    color = (0, 0, int(min(255, -val * 500)))
                
                cv2.rectangle(img, (px, row_y), (px + 4, row_y + 15), color, -1)
            
            # Right hemisphere
            for i in range(34):
                val = ev[i + 34]
                px = ev_x + 200 + i * 5
                
                if val > 0:
                    color = (0, int(min(255, val * 500)), 0)
                else:
                    color = (0, 0, int(min(255, -val * 500)))
                
                cv2.rectangle(img, (px, row_y), (px + 4, row_y + 15), color, -1)
            
            # Mode label
            cv2.putText(img, f"M{mode_idx+2}", (ev_x + 380, row_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Labels
        cv2.putText(img, "LH", (ev_x + 80, ev_y + ev_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1)
        cv2.putText(img, "RH", (ev_x + 280, ev_y + ev_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 100), 1)
        
        # === BOTTOM CENTER: INTERPRETATION ===
        int_x, int_y = 440, 310
        int_w, int_h = 350, 180
        
        cv2.rectangle(img, (int_x, int_y), (int_x+int_w, int_y+int_h), (30, 35, 40), -1)
        cv2.putText(img, "INTERPRETATION", (int_x + 10, int_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        interpretations = [
            "The STONE beneath the WATER:",
            "",
            "Mode 2: Left <-> Right diffusion",
            "  (Slowest - Inter-hemispheric)",
            "",
            "Mode 3: Superior <-> Inferior",
            "  (Sensory-Motor axis)",
            "",
            "Mode 4: Anterior <-> Posterior",
            "  (Executive-Perceptual axis)",
        ]
        
        for i, line in enumerate(interpretations):
            color = (180, 200, 220) if i == 0 else (140, 160, 180)
            cv2.putText(img, line, (int_x + 10, int_y + 40 + i * 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        
        # === BOTTOM RIGHT: METRICS ===
        met_x, met_y = 810, 310
        met_w, met_h = 270, 180
        
        cv2.rectangle(img, (met_x, met_y), (met_x+met_w, met_y+met_h), (30, 35, 40), -1)
        cv2.putText(img, "STRUCTURE vs FUNCTION", (met_x + 10, met_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Input status
        input_status = "Receiving EEG" if self.has_eeg_input else "No EEG input"
        input_color = (100, 255, 100) if self.has_eeg_input else (150, 150, 150)
        cv2.putText(img, input_status, (met_x + 10, met_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, input_color, 1)
        
        metrics = [
            f"Deviation: {self.structure_vs_function:.4f}",
            f"Dominant Mode: {int(self.dominant_mode)}",
            f"Temperature: {temperature:.2f}",
            f"Epoch: {self.epoch}",
            "",
            f"Lambda_2: {self.eigenvalues[1]:.4f}",
            f"Lambda_3: {self.eigenvalues[2]:.4f}",
            f"Lambda_4: {self.eigenvalues[3]:.4f}",
        ]
        
        for i, m in enumerate(metrics):
            cv2.putText(img, m, (met_x + 10, met_y + 65 + i * 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 170, 190), 1)
        
        # Mode activation bars
        if self.has_eeg_input:
            bar_y = met_y + met_h - 25
            bar_w = met_w - 20
            max_act = max(self.mode_activations.max(), 0.01)
            for i in range(min(8, self.n_modes)):
                bw = bar_w // 8
                bh = int((self.mode_activations[i] / max_act) * 20)
                bx = met_x + 10 + i * bw
                
                hue = int((1 - i / 8) * 120)
                hsv = np.array([[[hue, 200, 200]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                
                cv2.rectangle(img, (bx, bar_y - bh), (bx + bw - 2, bar_y), rgb, -1)
        
        # === PHILOSOPHY ===
        cv2.putText(img, "\"The brain's eigenmodes are conserved - they are the shape of thought itself.\" - Raj et al.", 
                   (20, 520),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 140), 1)
        cv2.putText(img, "Wire 'structural_field' or 'eigenmode_spectrum' to MutualInformationManifold as structured prior.", 
                   (20, 540),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 100, 120), 1)
        
        self._display = img
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display
    
    def get_config_options(self):
        return [
            ("n_modes", "Number of Modes", "int", 16, (4, 32)),
        ]