"""
NeuroCrystal Node v3.2 - Balanced Version
------------------------------------------
Fixes from v3.1:
- Removed over-aggressive input saturation
- Homeostasis is gentler (doesn't fight the input)
- Still has overflow protection but allows dynamics

The key insight: We want stability at EXTREME values,
but full responsiveness in the NORMAL operating range.
"""

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

class NeuroCrystalNode3(BaseNode):
    """
    Neuro-Crystal Node v3.2 (Balanced)
    ----------------------------------
    - Responsive to input (no aggressive saturation)
    - Only clips at actual overflow values
    - Gentle homeostasis that allows transients
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(180, 0, 100)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Neuro Crystal v3"
        
        self.inputs = {
            'drive_signal': 'spectrum',
            'injection_pattern': 'image',
            'surface_tension_mod': 'signal',
        }
        
        self.outputs = {
            'activity_view': 'image',
            'network_view': 'image',
            'combined_view': 'image',
            'topology_spectrum': 'spectrum',
            'h_links_flat': 'spectrum',
            'v_links_flat': 'spectrum',
            'link_field': 'image',
            'mean_weight': 'signal',
            'curvature_energy': 'signal',
            'junction_count': 'signal',
            'total_energy': 'signal',
        }
        
        self.config = {
            'resolution': 64,
            'plasticity': 0.08,
            'decay': 0.015,
            'diffusion': 0.25,
            'surface_tension': 0.12,
            'anisotropy': 0.02,
        }
        
        self._output_values = {}
        self._init_grid()

    def _init_grid(self):
        res = self.config.get('resolution', 64)
        self.grid = np.zeros((res, res), dtype=np.float32)
        self.h_links = np.ones((res, res), dtype=np.float32) * 0.5
        self.v_links = np.ones((res, res), dtype=np.float32) * 0.5
        self.d1_links = np.ones((res, res), dtype=np.float32) * 0.35
        self.d2_links = np.ones((res, res), dtype=np.float32) * 0.35
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
        """Mean curvature flow on link weights."""
        sigma = self.config.get('surface_tension', 0.12)
        
        sigma_mod = self.get_input('surface_tension_mod')
        if sigma_mod is not None:
            sigma = sigma * float(np.clip(sigma_mod, 0.1, 3.0))
        
        if sigma < 0.001:
            return 0.0
        
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
        
        self.h_links += sigma * h_lap
        self.v_links += sigma * v_lap
        
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
        
        self.curvature = np.abs(h_lap) + np.abs(v_lap)
        return float(np.sum(self.curvature))

    def _count_junctions(self, threshold=0.4):
        h_strong = (self.h_links > threshold).astype(int)
        v_strong = (self.v_links > threshold).astype(int)
        
        degree = np.zeros_like(self.h_links, dtype=int)
        degree += h_strong
        degree += np.roll(h_strong, 1, axis=1)
        degree += v_strong
        degree += np.roll(v_strong, 1, axis=0)
        
        n_junctions = np.sum(degree >= 3)
        n_trifurc = np.sum(degree >= 4)
        return int(n_junctions), int(n_trifurc)

    def step(self):
        res = self.config.get('resolution', 64)
        
        # === 1. INPUT INJECTION (NO SATURATION - full signal) ===
        stim = self.get_input('drive_signal')
        pattern = self.get_input('injection_pattern')
        
        if stim is not None:
            stim_vec = np.array(stim, dtype=np.float32).flatten()
            if len(stim_vec) > 0:
                # FULL energy - no saturation
                energy = np.mean(np.abs(stim_vec)) * 3.0
                
                if pattern is not None:
                    if pattern.ndim == 3:
                        pattern = np.mean(pattern, axis=2)
                    pat_small = cv2.resize(pattern.astype(np.float32), (res, res))
                    pat_norm = pat_small / (np.max(pat_small) + 1e-6)
                    self.grid += pat_norm * energy * 0.5
                else:
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
        
        ul = np.roll(np.roll(self.grid, 1, axis=0), 1, axis=1)
        ur = np.roll(np.roll(self.grid, 1, axis=0), -1, axis=1)
        dl = np.roll(np.roll(self.grid, -1, axis=0), 1, axis=1)
        dr = np.roll(np.roll(self.grid, -1, axis=0), -1, axis=1)
        
        h_flow = (
            (left - self.grid) * np.roll(self.h_links, 1, axis=1) +
            (right - self.grid) * self.h_links
        )
        v_flow = (
            (up - self.grid) * np.roll(self.v_links, 1, axis=0) +
            (down - self.grid) * self.v_links
        )
        
        diag_weight = 0.707
        d_flow = (
            (ul - self.grid) * np.roll(np.roll(self.d1_links, 1, axis=0), 1, axis=1) * diag_weight +
            (dr - self.grid) * self.d1_links * diag_weight +
            (ur - self.grid) * np.roll(np.roll(self.d2_links, 1, axis=0), -1, axis=1) * diag_weight +
            (dl - self.grid) * self.d2_links * diag_weight
        )
        
        diffusion = self.config.get('diffusion', 0.25)
        self.grid += (h_flow + v_flow + d_flow * 0.5) * diffusion
        self.grid *= 0.94

        # === 3. HEBBIAN LEARNING (full strength) ===
        lr = self.config.get('plasticity', 0.08)
        
        h_flux = np.abs(self.grid * right)
        v_flux = np.abs(self.grid * down)
        d1_flux = np.abs(self.grid * dr) * diag_weight
        d2_flux = np.abs(self.grid * dl) * diag_weight
        
        self.h_links += h_flux * lr
        self.v_links += v_flux * lr
        self.d1_links += d1_flux * lr * 0.7
        self.d2_links += d2_flux * lr * 0.7

        # === 4. SURFACE TENSION ===
        curvature_energy = self._compute_surface_tension()

        # === 5. DECAY ===
        decay = self.config.get('decay', 0.015)
        self.h_links *= (1.0 - decay)
        self.v_links *= (1.0 - decay)
        self.d1_links *= (1.0 - decay)
        self.d2_links *= (1.0 - decay)
        
        # === 6. ONLY clip at actual dangerous values (not aggressive) ===
        # Allow values up to 10 before clipping
        np.clip(self.h_links, 0.01, 10.0, out=self.h_links)
        np.clip(self.v_links, 0.01, 10.0, out=self.v_links)
        np.clip(self.d1_links, 0.01, 8.0, out=self.d1_links)
        np.clip(self.d2_links, 0.01, 8.0, out=self.d2_links)
        np.clip(self.grid, -50, 50, out=self.grid)

        # === 7. NaN/Inf protection only ===
        for arr in [self.h_links, self.v_links, self.d1_links, self.d2_links, self.grid]:
            mask = ~np.isfinite(arr)
            if np.any(mask):
                arr[mask] = 0.5

        # === 8. COMPUTE OUTPUTS ===
        total_energy = (np.mean(self.h_links) + np.mean(self.v_links) + 
                       np.mean(self.d1_links) + np.mean(self.d2_links)) / 4.0
        
        self.set_output('h_links_flat', self.h_links.flatten().astype(np.float32))
        self.set_output('v_links_flat', self.v_links.flatten().astype(np.float32))
        
        link_field = (self.h_links + self.v_links + self.d1_links + self.d2_links) / 4.0
        # Normalize for display but preserve actual values
        lf_max = np.max(link_field)
        if lf_max > 0:
            link_field_norm = (link_field / lf_max * 255).astype(np.uint8)
        else:
            link_field_norm = np.zeros_like(link_field, dtype=np.uint8)
        self.set_output('link_field', link_field_norm)
        
        self.set_output('mean_weight', float(np.mean(link_field)))
        self.set_output('curvature_energy', curvature_energy)
        self.set_output('total_energy', float(total_energy))
        
        n_junctions, n_trifurc = self._count_junctions()
        self.set_output('junction_count', float(n_junctions))
        
        # === 9. RENDER ===
        self._render_views()

    def _render_views(self):
        res = self.config.get('resolution', 64)
        scale = max(4, 256 // res)
        h, w = res * scale, res * scale
        
        # Activity View
        act_norm = np.clip((self.grid + 2.0) / 4.0, 0, 1)
        act_img = (act_norm * 255).astype(np.uint8)
        act_img = cv2.applyColorMap(act_img, cv2.COLORMAP_OCEAN)
        act_img = cv2.resize(act_img, (w, h), interpolation=cv2.INTER_NEAREST)
        self.set_output('activity_view', act_img)
        
        # Network View - fixed threshold for consistency
        net_img = np.zeros((h, w, 3), dtype=np.float32)
        thresh = 0.3
        
        for y in range(res):
            for x in range(res):
                cx, cy = x * scale + scale // 2, y * scale + scale // 2
                
                if x < res - 1:
                    weight = float(self.h_links[y, x])
                    if weight > thresh:
                        intensity = min(1.0, (weight - thresh) / 1.2)
                        color = (0.1, intensity * 0.9, intensity * 0.4)
                        thickness = max(1, int(intensity * 3))
                        cv2.line(net_img, (cx, cy), (cx + scale, cy), color, thickness)
                
                if y < res - 1:
                    weight = float(self.v_links[y, x])
                    if weight > thresh:
                        intensity = min(1.0, (weight - thresh) / 1.2)
                        color = (0.1, intensity * 0.9, intensity * 0.4)
                        thickness = max(1, int(intensity * 3))
                        cv2.line(net_img, (cx, cy), (cx, cy + scale), color, thickness)
                
                if x < res - 1 and y < res - 1:
                    w1 = float(self.d1_links[y, x])
                    if w1 > thresh:
                        intensity = min(1.0, (w1 - thresh) / 0.9)
                        color = (intensity * 0.3, intensity * 0.6, 0.1)
                        cv2.line(net_img, (cx, cy), (cx + scale, cy + scale), color, 1)
                    
                    w2 = float(self.d2_links[y, x])
                    if w2 > thresh:
                        intensity = min(1.0, (w2 - thresh) / 0.9)
                        color = (intensity * 0.3, intensity * 0.6, 0.1)
                        cv2.line(net_img, (cx + scale, cy), (cx, cy + scale), color, 1)

        self.set_output('network_view', (net_img * 255).astype(np.uint8))
        
        # Combined
        combined = cv2.addWeighted(
            act_img.astype(np.float32) / 255.0, 0.5,
            net_img, 0.8, 0
        )
        self.set_output('combined_view', (combined * 255).astype(np.uint8))
        
        # Topology Spectrum
        link_field = (self.h_links + self.v_links) / 2.0
        spectrum = np.zeros(res // 2)
        cy_c, cx_c = res // 2, res // 2
        for r in range(res // 2):
            mask = np.abs(np.sqrt((np.arange(res)[:, None] - cy_c)**2 + 
                                  (np.arange(res)[None, :] - cx_c)**2) - r) < 1
            if np.any(mask):
                spectrum[r] = np.mean(link_field[mask])
        
        self.set_output('topology_spectrum', spectrum)