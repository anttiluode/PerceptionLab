import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {} 
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name): return None

class NeuroCrystalNode2(BaseNode):
    """
    Neuro-Crystal Node (Plastic Spacetime with Surface Optimization)
    ----------------------------------------------------------------
    A grid where Space itself learns, now with Barabási-inspired
    surface tension that produces smooth veins instead of jagged lightning.
    
    Key insight: Biological networks minimize SURFACE AREA, not wire length.
    This creates smooth, river-like channels with proper trifurcations.
    
    Physics:
    - Waves propagate through anisotropic lattice
    - Hebbian learning strengthens high-flux paths
    - Surface tension smooths the learned topology
    - Result: minimal-cost information highways
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(180, 0, 100)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Neuro Crystal"
        
        self.inputs = {
            'drive_signal': 'spectrum',
            'injection_pattern': 'image',  # Optional spatial injection
        }
        
        self.outputs = {
            'activity_view': 'image',
            'network_view': 'image',
            'combined_view': 'image',
            'topology_spectrum': 'spectrum',  # For downstream analysis
        }
        
        self.config = {
            'resolution': 64,
            'plasticity': 0.08,
            'decay': 0.015,
            'diffusion': 0.25,
            'surface_tension': 0.12,  # Barabási smoothing
            'anisotropy': 0.02,       # Directional bias development
        }
        
        self._output_values = {}
        self._init_grid()

    def _init_grid(self):
        res = self.config['resolution']
        
        # Activity grid (the waves)
        self.grid = np.zeros((res, res), dtype=np.float32)
        
        # Connection grids (the pipes) - start uniform
        self.h_links = np.ones((res, res), dtype=np.float32) * 0.5
        self.v_links = np.ones((res, res), dtype=np.float32) * 0.5
        
        # Diagonal links for richer topology
        self.d1_links = np.ones((res, res), dtype=np.float32) * 0.35  # NE-SW
        self.d2_links = np.ones((res, res), dtype=np.float32) * 0.35  # NW-SE
        
        # Curvature accumulator (for surface tension)
        self.curvature = np.zeros((res, res), dtype=np.float32)

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'): 
            return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value): 
        self._output_values[name] = value
    
    def get_output(self, name): 
        return self._output_values.get(name, None)

    def _compute_surface_tension(self):
        """
        Barabási-inspired surface minimization.
        Smooths link weights to minimize local curvature.
        Creates river-like channels instead of jagged paths.
        """
        sigma = self.config['surface_tension']
        if sigma < 0.001:
            return
        
        # Compute local curvature (Laplacian of link field)
        h_lap = (
            np.roll(self.h_links, 1, axis=0) + 
            np.roll(self.h_links, -1, axis=0) +
            np.roll(self.h_links, 1, axis=1) + 
            np.roll(self.h_links, -1, axis=1) - 
            4 * self.h_links
        )
        
        v_lap = (
            np.roll(self.v_links, 1, axis=0) + 
            np.roll(self.v_links, -1, axis=0) +
            np.roll(self.v_links, 1, axis=1) + 
            np.roll(self.v_links, -1, axis=1) - 
            4 * self.v_links
        )
        
        # Mean curvature flow - smooths toward minimal surface
        self.h_links += sigma * h_lap
        self.v_links += sigma * v_lap
        
        # Same for diagonals
        d1_lap = (
            np.roll(self.d1_links, 1, axis=0) + 
            np.roll(self.d1_links, -1, axis=0) +
            np.roll(self.d1_links, 1, axis=1) + 
            np.roll(self.d1_links, -1, axis=1) - 
            4 * self.d1_links
        )
        d2_lap = (
            np.roll(self.d2_links, 1, axis=0) + 
            np.roll(self.d2_links, -1, axis=0) +
            np.roll(self.d2_links, 1, axis=1) + 
            np.roll(self.d2_links, -1, axis=1) - 
            4 * self.d2_links
        )
        
        self.d1_links += sigma * 0.7 * d1_lap
        self.d2_links += sigma * 0.7 * d2_lap
        
        # Store curvature for visualization
        self.curvature = np.abs(h_lap) + np.abs(v_lap)

    def step(self):
        res = self.config['resolution']
        
        # === 1. INPUT INJECTION ===
        stim = self.get_input('drive_signal')
        pattern = self.get_input('injection_pattern')
        
        if stim is not None:
            stim_vec = np.array(stim, dtype=np.float32).flatten()
            if len(stim_vec) > 0:
                energy = np.mean(np.abs(stim_vec)) * 3.0
                
                if pattern is not None:
                    # Spatial injection from pattern
                    if pattern.ndim == 3:
                        pattern = np.mean(pattern, axis=2)
                    pat_small = cv2.resize(pattern.astype(np.float32), (res, res))
                    pat_norm = pat_small / (np.max(pat_small) + 1e-6)
                    self.grid += pat_norm * energy * 0.5
                else:
                    # Default: inject at center with spectral modulation
                    cx, cy = res // 2, res // 2
                    spread = min(len(stim_vec), res // 4)
                    for i in range(spread):
                        angle = 2 * np.pi * i / spread
                        dx = int(np.cos(angle) * (i + 1))
                        dy = int(np.sin(angle) * (i + 1))
                        x, y = cx + dx, cy + dy
                        if 0 <= x < res and 0 <= y < res:
                            self.grid[y, x] += stim_vec[i % len(stim_vec)] * energy

        # === 2. ANISOTROPIC DIFFUSION ===
        left = np.roll(self.grid, 1, axis=1)
        right = np.roll(self.grid, -1, axis=1)
        up = np.roll(self.grid, 1, axis=0)
        down = np.roll(self.grid, -1, axis=0)
        
        # Diagonal neighbors
        ul = np.roll(np.roll(self.grid, 1, axis=0), 1, axis=1)
        ur = np.roll(np.roll(self.grid, 1, axis=0), -1, axis=1)
        dl = np.roll(np.roll(self.grid, -1, axis=0), 1, axis=1)
        dr = np.roll(np.roll(self.grid, -1, axis=0), -1, axis=1)
        
        # Flow weighted by link strength
        h_flow = (
            (left - self.grid) * np.roll(self.h_links, 1, axis=1) +
            (right - self.grid) * self.h_links
        )
        v_flow = (
            (up - self.grid) * np.roll(self.v_links, 1, axis=0) +
            (down - self.grid) * self.v_links
        )
        
        # Diagonal flow (reduced weight)
        diag_weight = 0.707  # 1/sqrt(2)
        d_flow = (
            (ul - self.grid) * np.roll(np.roll(self.d1_links, 1, axis=0), 1, axis=1) * diag_weight +
            (dr - self.grid) * self.d1_links * diag_weight +
            (ur - self.grid) * np.roll(np.roll(self.d2_links, 1, axis=0), -1, axis=1) * diag_weight +
            (dl - self.grid) * self.d2_links * diag_weight
        )
        
        diffusion = self.config['diffusion']
        self.grid += (h_flow + v_flow + d_flow * 0.5) * diffusion
        
        # Activity decay
        self.grid *= 0.94

        # === 3. HEBBIAN LEARNING ===
        lr = self.config['plasticity']
        
        # Flux = coactivation (product, not difference - fire together)
        h_flux = np.abs(self.grid * right)
        v_flux = np.abs(self.grid * down)
        d1_flux = np.abs(self.grid * dr) * diag_weight
        d2_flux = np.abs(self.grid * dl) * diag_weight
        
        # Strengthen where flux is high
        self.h_links += h_flux * lr
        self.v_links += v_flux * lr
        self.d1_links += d1_flux * lr * 0.7
        self.d2_links += d2_flux * lr * 0.7

        # === 4. SURFACE TENSION (Barabási) ===
        self._compute_surface_tension()

        # === 5. DECAY + HOMEOSTASIS ===
        decay = self.config['decay']
        self.h_links *= (1.0 - decay)
        self.v_links *= (1.0 - decay)
        self.d1_links *= (1.0 - decay)
        self.d2_links *= (1.0 - decay)
        
        # Soft bounds (allow some overshoot for dynamics)
        np.clip(self.h_links, 0.01, 1.5, out=self.h_links)
        np.clip(self.v_links, 0.01, 1.5, out=self.v_links)
        np.clip(self.d1_links, 0.01, 1.2, out=self.d1_links)
        np.clip(self.d2_links, 0.01, 1.2, out=self.d2_links)
        np.clip(self.grid, -5, 5, out=self.grid)

        # === 6. RENDER ===
        self._render_views()

    def _render_views(self):
        res = self.config['resolution']
        scale = max(4, 256 // res)
        h, w = res * scale, res * scale
        
        # --- Activity View ---
        act_norm = np.clip((self.grid + 2.0) / 4.0, 0, 1)
        act_img = (act_norm * 255).astype(np.uint8)
        act_img = cv2.applyColorMap(act_img, cv2.COLORMAP_OCEAN)
        act_img = cv2.resize(act_img, (w, h), interpolation=cv2.INTER_NEAREST)
        self.set_output('activity_view', act_img)
        
        # --- Network View (The Veins) ---
        net_img = np.zeros((h, w, 3), dtype=np.float32)
        
        # Combine all links into single strength map
        total_strength = (self.h_links + self.v_links + 
                         self.d1_links + self.d2_links) / 4.0
        
        # Threshold for visibility
        thresh = 0.3
        
        for y in range(res):
            for x in range(res):
                cx, cy = x * scale + scale // 2, y * scale + scale // 2
                
                # Horizontal
                if x < res - 1:
                    weight = float(self.h_links[y, x])
                    if weight > thresh:
                        intensity = (weight - thresh) / (1.5 - thresh)
                        color = (0.1, intensity * 0.9, intensity * 0.4)
                        thickness = max(1, int(intensity * 3))
                        cv2.line(net_img, (cx, cy), (cx + scale, cy), color, thickness)
                
                # Vertical
                if y < res - 1:
                    weight = float(self.v_links[y, x])
                    if weight > thresh:
                        intensity = (weight - thresh) / (1.5 - thresh)
                        color = (0.1, intensity * 0.9, intensity * 0.4)
                        thickness = max(1, int(intensity * 3))
                        cv2.line(net_img, (cx, cy), (cx, cy + scale), color, thickness)
                
                # Diagonals (thinner)
                if x < res - 1 and y < res - 1:
                    w1 = float(self.d1_links[y, x])
                    if w1 > thresh:
                        intensity = (w1 - thresh) / (1.2 - thresh)
                        color = (intensity * 0.3, intensity * 0.6, 0.1)
                        cv2.line(net_img, (cx, cy), (cx + scale, cy + scale), color, 1)
                    
                    w2 = float(self.d2_links[y, x])
                    if w2 > thresh:
                        intensity = (w2 - thresh) / (1.2 - thresh)
                        color = (intensity * 0.3, intensity * 0.6, 0.1)
                        cv2.line(net_img, (cx + scale, cy), (cx, cy + scale), color, 1)

        self.set_output('network_view', (net_img * 255).astype(np.uint8))
        
        # --- Combined View ---
        combined = cv2.addWeighted(
            act_img.astype(np.float32) / 255.0, 0.5,
            net_img, 0.8, 0
        )
        self.set_output('combined_view', (combined * 255).astype(np.uint8))
        
        # --- Topology Spectrum ---
        # Radial average of link strength (like P(λ) from Barabási)
        link_field = (self.h_links + self.v_links) / 2.0
        spectrum = np.zeros(res // 2)
        cy, cx = res // 2, res // 2
        for r in range(res // 2):
            mask = np.abs(np.sqrt((np.arange(res)[:, None] - cy)**2 + 
                                  (np.arange(res)[None, :] - cx)**2) - r) < 1
            if np.any(mask):
                spectrum[r] = np.mean(link_field[mask])
        
        self.set_output('topology_spectrum', spectrum)
