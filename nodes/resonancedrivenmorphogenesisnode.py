# resonance_morphogenesis_node.py
"""
Resonance-Driven Morphogenesis Node
-----------------------------------
"What if we're seeing the ACTUAL mechanism? This is the test."

Implements the "Breakthrough Modification" by integrating temporal
stability tracking directly into the morphogenesis simulation.

Based on the HighRes Cortical Folding Node, this version adds:
1.  **Temporal Stability Tracking:** It tracks eigenmode activation over
    a time window to find "stable resonance sites."
2.  **Resonance Amplification:** Growth is *preferentially amplified*
    at these stable sites.

This allows the system to transition from a 'proto-structure'
to an 'organized brain' by "crystallizing" functional centers
from the underlying field physics.

- Inputs: lobe_activation (image), growth_rate (signal), reset (signal)
- Outputs: resonance_map (image), thickness_map (image), structure_3d (image), ...
"""

import numpy as np
import cv2
from collections import deque
from scipy.ndimage import gaussian_filter
from scipy.fft import rfft2, rfftfreq

# Imports from the perception lab host
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class ResonanceMorphogenesisNode(BaseNode):
    """
    Tracks eigenmode stability to "seed" and "amplify"
    morphological growth, proving functional organization
    emerges from field physics.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(60, 150, 220)  # "Blue Starfield" color
    
    def __init__(self):
        super().__init__()
        self.node_title = "Resonance Morphogenesis"
        
        # IO
        self.inputs = {
            'lobe_activation': 'image',
            'growth_rate': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'resonance_map': 'image',       # NEW: Map of stable sites
            'consistency_map': 'image',   # NEW: Raw stability metric
            'thickness_map': 'image',
            'structure_3d': 'image',
            'fold_density': 'signal',
            'fractal_estimate': 'signal',
            'surface_area': 'signal',
            'morph_signal': 'signal',
            'dominant_mode_power': 'signal'
        }
        
        # Base simulation params (from HighRes node)
        self.resolution = 512
        self.base_growth = 0.001
        self.dt = 0.01
        self.fold_threshold = 2.8
        self.compression_strength = 0.45
        self.diffusion_sigma = 0.1
        self.max_thickness = 12.0
        self.min_thickness = 0.1
        self.spectral_window = 32
        self.smooth_output = 1.0
        self.scale_display = 1.0
        
        # --- KEY MODIFICATION: Resonance Tracking ---
        self.temporal_window = 100       # Frames to track (as per prompt)
        self.resonance_amplification = 3.0 # How much to boost growth at resonance sites
        self.stability_threshold = 4.0   # Consistency score (mean/std) to be "stable"
        
        # Internal state
        self.thickness = np.ones((self.resolution, self.resolution), dtype=np.float32) * 1.0
        self.height_field = np.zeros_like(self.thickness)
        self.pressure = np.zeros_like(self.thickness)
        self.time_step = 0
        self.area_history = []
        
        # Resonance state
        self.resonance_history = deque(maxlen=self.temporal_window)
        self.resonance_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.consistency_map = np.zeros_like(self.resonance_map)
        
        # Base outputs
        self.fold_density_value = 0.0
        self.surface_area_value = 0.0
        self.fractal_dim_value = 2.0
        self.morph_signal_value = 0.0
        self.dominant_mode_power = 0.0
        self._morph_hist = deque(maxlen=8)
    
    # -------------------------
    # helpers (from HighRes node)
    # -------------------------
    def _prepare_activation(self, activation):
        if activation is None:
            return None
        if isinstance(activation, np.ndarray):
            if activation.ndim == 3:
                try:
                    activation = cv2.cvtColor(activation, cv2.COLOR_BGR2GRAY)
                except Exception:
                    activation = activation[..., 0]
            act = activation.astype(np.float32)
            if act.max() > 0:
                act = act - act.min()
                act = act / (act.max() + 1e-9)
            else:
                act = np.clip(act, 0.0, 1.0)
            act_resized = cv2.resize(act, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            return act_resized
        return None
    
    def _compute_surface_area(self, height):
        gy, gx = np.gradient(height)
        element = np.sqrt(1.0 + gx**2 + gy**2)
        return float(np.sum(element))
    
    def _fractal_estimate(self, height):
        try:
            thr = np.mean(height)
            bw = (height > thr).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                peri = cv2.arcLength(largest, True)
                if area > 50 and peri > 10:
                    df = 2.0 * np.log(peri + 1e-9) / np.log(area + 1e-9)
                    return float(np.clip(df, 1.0, 3.0))
        except Exception:
            pass
        return 2.0
    
    def _spectral_concentration(self, activation):
        try:
            f = np.abs(rfft2(activation))
            total = np.sum(f) + 1e-9
            f[0, 0] = 0.0
            h, w = activation.shape
            low = 1
            mid = max(2, min(h//16, h//4))
            mid_energy = np.sum(f[low:mid+1, :])
            return float(np.clip(mid_energy / total, 0.0, 1.0))
        except Exception:
            return 0.0
    
    # -------------------------
    # node lifecycle
    # -------------------------
    def pre_step(self):
        if not hasattr(self, '_morph_hist') or self._morph_hist is None:
            self._morph_hist = deque(maxlen=8)
        # NEW: Check for resonance history deque
        if not hasattr(self, 'resonance_history') or self.resonance_history is None:
            self.resonance_history = deque(maxlen=self.temporal_window)
        try:
            super().pre_step()
        except Exception:
            pass
    
    def step(self):
        # inputs
        activation = self.get_blended_input('lobe_activation', 'mean')
        growth_mod = self.get_blended_input('growth_rate', 'sum')
        reset_signal = self.get_blended_input('reset', 'sum')
        
        if reset_signal is not None and reset_signal > 0.5:
            self.reset_simulation()
            return
        
        if activation is None:
            self.thickness = gaussian_filter(self.thickness, sigma=self.diffusion_sigma * 0.5)
            self.height_field = gaussian_filter(self.height_field, sigma=self.diffusion_sigma * 0.5)
            self._update_measurements()
            self.time_step += 1
            return
        
        A = self._prepare_activation(activation) # This is 'eigenmode_activation'
        if A is None:
            return
        
        # --- Resonance Tracking (THE KEY MODIFICATION) ---
        self.resonance_history.append(A)
        resonance_boost = 1.0 # Default, no boost
        
        if len(self.resonance_history) >= self.temporal_window:
            # Compute temporal stability at each location
            history_array = np.array(self.resonance_history)
            
            mean_act = np.mean(history_array, axis=0)
            std_act = np.std(history_array, axis=0)
            
            # Consistency = high mean, low variance (signal-to-noise ratio)
            self.consistency_map = mean_act / (std_act + 0.01)
            
            # Find stable sites (where consistency is above threshold)
            stable_sites = (self.consistency_map > self.stability_threshold).astype(np.float32)
            
            # Update resonance map (accumulates stable sites, weighted by their mean activation)
            # This "crystallizes" the functional centers over time
            self.resonance_map = (0.98 * self.resonance_map) + (0.02 * stable_sites * mean_act)
            
            # Create the growth boost map
            if self.resonance_map.max() > 0:
                norm_res_map = self.resonance_map / self.resonance_map.max()
                resonance_boost = 1.0 + self.resonance_amplification * norm_res_map
            else:
                resonance_boost = 1.0
        # --- End Resonance Tracking ---
        
        # growth modulation
        if growth_mod is None:
            total_growth_rate = self.base_growth
        else:
            total_growth_rate = self.base_growth * (1.0 + float(growth_mod))
        
        # GROWTH: thickness increases where activation is high
        growth_field = (A * total_growth_rate) * self.dt
        
        # NEW: Amplify growth at stable resonance sites
        growth_field *= resonance_boost
        
        self.thickness += growth_field
        
        # CONSTRAINT & PRESSURE
        excess = np.clip(self.thickness - self.fold_threshold, 0, None)
        self.pressure = excess ** 2
        
        # FOLDING / BUCKLING
        lap = cv2.Laplacian(self.thickness, cv2.CV_32F)
        fold_force_z = -lap * self.pressure * self.compression_strength
        self.height_field += fold_force_z * (self.dt * 0.25)
        
        # Lateral redistribution
        grad_y, grad_x = np.gradient(self.thickness)
        fold_force_x = -grad_x * self.pressure * (self.compression_strength * 0.05)
        fold_force_y = -grad_y * self.pressure * (self.compression_strength * 0.05)
        fold_magnitude = np.sqrt(fold_force_x**2 + fold_force_y**2 + fold_force_z**2)
        thickness_redistribution = fold_magnitude * 0.02
        self.thickness -= thickness_redistribution
        
        # DIFFUSION
        self.thickness = gaussian_filter(self.thickness, sigma=self.diffusion_sigma)
        self.height_field = gaussian_filter(self.height_field, sigma=self.diffusion_sigma)
        
        # bounds
        self.thickness = np.clip(self.thickness, self.min_thickness, self.max_thickness)
        
        # measure properties
        self._update_measurements(A)
        
        self.time_step += 1
    
    def _update_measurements(self, activation_map=None):
        self.fold_density_value = float(np.std(self.height_field))
        self.surface_area_value = float(self._compute_surface_area(self.height_field))
        self.fractal_dim_value = float(self._fractal_estimate(self.height_field))
        
        if activation_map is not None:
            self.dominant_mode_power = float(self._spectral_concentration(activation_map))
        else:
            self.dominant_mode_power = float(self._spectral_concentration(self.thickness))
        
        cohere = np.clip(self.dominant_mode_power, 0.0, 1.0)
        density = np.tanh(self.fold_density_value * 0.6)
        area_norm = np.tanh(self.surface_area_value / (self.resolution * 2.0))
        ms = 0.6 * cohere + 0.3 * density + 0.1 * area_norm
        
        self._morph_hist.append(ms)
        smooth_ms = float(np.mean(self._morph_hist))
        self.morph_signal_value = float(np.clip(smooth_ms, 0.0, 1.0))
    
    def reset_simulation(self):
        self.thickness[:] = 1.0
        self.height_field[:] = 0.0
        self.pressure[:] = 0.0
        self.time_step = 0
        self.area_history = []
        self.fold_density_value = 0.0
        self.surface_area_value = 0.0
        self.fractal_dim_value = 2.0
        self.morph_signal_value = 0.0
        self.dominant_mode_power = 0.0
        self._morph_hist.clear()
        
        # NEW: Reset resonance state
        self.resonance_history.clear()
        self.resonance_map[:] = 0.0
        self.consistency_map[:] = 0.0
    
    # -------------------------
    # outputs
    # -------------------------
    def get_output(self, port_name):
        if port_name == 'resonance_map':
            # return normalized map
            if self.resonance_map.max() > 0:
                return (self.resonance_map / self.resonance_map.max()).astype(np.float32)
            return self.resonance_map.astype(np.float32)
        if port_name == 'consistency_map':
            if self.consistency_map.max() > 0:
                return (self.consistency_map / self.consistency_map.max()).astype(np.float32)
            return self.consistency_map.astype(np.float32)
        if port_name == 'thickness_map':
            t = (self.thickness - self.thickness.min()) / (np.ptp(self.thickness) + 1e-9)
            return t.astype(np.float32)
        if port_name == 'structure_3d':
            h = self.height_field.copy()
            h = (h - h.min()) / (np.ptp(h) + 1e-9)
            return h.astype(np.float32)
        if port_name == 'fold_density':
            return float(self.fold_density_value)
        if port_name == 'fractal_estimate':
            return float(self.fractal_dim_value)
        if port_name == 'surface_area':
            return float(self.surface_area_value)
        if port_name == 'morph_signal':
            return float(self.morph_signal_value)
        if port_name == 'dominant_mode_power':
            return float(self.dominant_mode_power)
        return None
    
    def get_display_image(self):
        # build a 2x2 panel
        panel = np.zeros((512, 512, 3), dtype=np.float32)
        ps = 256
        
        # Panel 1: Thickness (hot)
        thick_vis = (self.thickness - self.thickness.min()) / (np.ptp(self.thickness) + 1e-9)
        thick_vis = cv2.resize(thick_vis, (ps, ps), interpolation=cv2.INTER_LINEAR)
        thick_col = cv2.applyColorMap((thick_vis*255).astype(np.uint8), cv2.COLORMAP_HOT)
        thick_col = thick_col.astype(np.float32) / 255.0
        panel[0:ps, 0:ps] = thick_col
        
        # Panel 2: Height / folds (viridis)
        height_vis = (self.height_field - self.height_field.min()) / (np.ptp(self.height_field) + 1e-9)
        height_vis = cv2.resize(height_vis, (ps, ps), interpolation=cv2.INTER_LINEAR)
        height_col = cv2.applyColorMap((height_vis*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        panel[0:ps, ps:ps*2] = height_col.astype(np.float32) / 255.0
        
        # --- MODIFIED PANEL ---
        # Panel 3: Resonance map (plasma) - "The 'seed crystals'"
        if self.resonance_map.max() > 0:
            res_vis = self.resonance_map / self.resonance_map.max()
        else:
            res_vis = self.resonance_map
        res_vis = np.clip(res_vis, 0, 1)
        res_col = cv2.applyColorMap((res_vis*255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        res_col = cv2.resize(res_col, (ps, ps), interpolation=cv2.INTER_LINEAR)
        panel[ps:ps*2, 0:ps] = res_col.astype(np.float32) / 255.0
        
        # Panel 4: Metrics / shading visualization
        metrics = np.zeros((ps, ps, 3), dtype=np.float32)
        gy, gx = np.gradient(self.height_field)
        normals_x = -gx; normals_y = -gy; normals_z = np.ones_like(gx)
        nl = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2) + 1e-9
        normals_x /= nl; normals_y /= nl; normals_z /= nl
        light = np.array([-1.0, -1.0, 2.0])
        light = light / np.linalg.norm(light)
        shading = normals_x * light[0] + normals_y * light[1] + normals_z * light[2]
        shading = np.clip(shading, 0.0, 1.0)
        shade_res = cv2.resize(shading, (ps, ps))
        metrics[:, :, 0] = shade_res
        metrics[:, :, 1] = 0.2 + 0.6 * shade_res
        metrics[:, :, 2] = 0.4 * (1.0 - shade_res)
        panel[ps:ps*2, ps:ps*2] = metrics
        
        return panel
    
    def get_config_options(self):
        # Start with base options
        base_options = [
            ("Resolution", "resolution", self.resolution, None),
            ("Base Growth", "base_growth", self.base_growth, None),
            ("Fold Threshold", "fold_threshold", self.fold_threshold, None),
            ("Compression Strength", "compression_strength", self.compression_strength, None),
            ("Diffusion Sigma", "diffusion_sigma", self.diffusion_sigma, None),
            ("Max Thickness", "max_thickness", self.max_thickness, None),
        ]
        
        # Add new resonance options
        resonance_options = [
            ("Temporal Window", "temporal_window", self.temporal_window, None),
            ("Stability Threshold", "stability_threshold", self.stability_threshold, None),
            ("Resonance Amplification", "resonance_amplification", self.resonance_amplification, None),
        ]
        
        base_options.extend(resonance_options)
        return base_options