# highers_cortical_folding_node.py
"""
HighRes Cortical Folding Node (patched)
--------------------------------------
- NumPy 2.0 compatible (uses np.ptp)
- deque import fixed
- 512x512 internal resolution
- Outputs: thickness_map, structure_3d, fold_density, fractal_estimate, surface_area,
           morph_signal, dominant_mode_power
- Advanced folding & spectral analysis

Usage:
 - Feed `lobe_activation` from EigenmodeResonanceNode (image 0..1)
 - Optionally feed `growth_rate` (signal) or `reset` (signal > 0.5)
"""

import numpy as np
import cv2
from collections import deque
from scipy.ndimage import gaussian_filter
from scipy.fft import rfft2, rfftfreq

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class HighResCorticalFoldingNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(150, 60, 160)  # rich purple
    
    def __init__(self):
        super().__init__()
        self.node_title = "HighRes Cortical Folding"
        
        # IO
        self.inputs = {
            'lobe_activation': 'image',
            'growth_rate': 'signal',
            'reset': 'signal'
        }
        self.outputs = {
            'thickness_map': 'image',
            'structure_3d': 'image',
            'fold_density': 'signal',
            'fractal_estimate': 'signal',
            'surface_area': 'signal',
            'morph_signal': 'signal',
            'dominant_mode_power': 'signal'
        }
        
        # config / simulation params (tweakable)
        self.resolution = 512            # core internal resolution (square)
        self.base_growth = 0.001       # base growth rate per step
        self.dt = 0.01                   # time-step scalar
        self.fold_threshold = 2.8       # when to start heavy buckling
        self.compression_strength = 0.45
        self.diffusion_sigma = 0.1      # smoothing to stabilize
        self.max_thickness = 12.0
        self.min_thickness = 0.1
        self.spectral_window = 32       # window size for spectral estimation (pixels)
        self.smooth_output = 1.0        # smoothing on visualization
        self.scale_display = 1.0
        
        # internal state
        self.thickness = np.ones((self.resolution, self.resolution), dtype=np.float32) * 1.0
        self.height_field = np.zeros_like(self.thickness)
        self.pressure = np.zeros_like(self.thickness)
        self.time_step = 0
        self.area_history = []
        
        # outputs
        self.fold_density_value = 0.0
        self.surface_area_value = 0.0
        self.fractal_dim_value = 2.0
        self.morph_signal_value = 0.0
        self.dominant_mode_power = 0.0
        
        # small ring buffer for recent morph_signal smoothing
        self._morph_hist = deque(maxlen=8)
    
    # -------------------------
    # helpers
    # -------------------------
    def _prepare_activation(self, activation):
        if activation is None:
            return None
        # Convert to single-channel float 0..1
        if isinstance(activation, np.ndarray):
            if activation.ndim == 3:
                # assume RGB / BGR
                try:
                    activation = cv2.cvtColor(activation, cv2.COLOR_BGR2GRAY)
                except Exception:
                    activation = activation[..., 0]
            act = activation.astype(np.float32)
            # normalize robustly
            if act.max() > 0:
                act = act - act.min()
                act = act / (act.max() + 1e-9)
            else:
                act = np.clip(act, 0.0, 1.0)
            # resize to internal resolution
            act_resized = cv2.resize(act, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            return act_resized
        return None
    
    def _compute_surface_area(self, height):
        gy, gx = np.gradient(height)
        element = np.sqrt(1.0 + gx**2 + gy**2)
        return float(np.sum(element))
    
    def _fractal_estimate(self, height):
        # Quick perimeter/area-based estimate: threshold and measure contour
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
        # compute radial spectral energy concentration - returns (dominant_power_norm)
        # use rfft2 on activation to get stable spectral magnitude
        try:
            f = np.abs(rfft2(activation))
            total = np.sum(f) + 1e-9
            # zero DC
            f[0, 0] = 0.0
            # choose midband indices
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
        # ensure deque exists if state deserialized
        if not hasattr(self, '_morph_hist') or self._morph_hist is None:
            self._morph_hist = deque(maxlen=8)
        try:
            super().pre_step()
        except Exception:
            # some hosts may not implement pre_step; ignore safely
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
            # decays and gentle smoothing when no input
            self.thickness = gaussian_filter(self.thickness, sigma=self.diffusion_sigma * 0.5)
            self.height_field = gaussian_filter(self.height_field, sigma=self.diffusion_sigma * 0.5)
            self._update_measurements()
            self.time_step += 1
            return
        
        A = self._prepare_activation(activation)
        if A is None:
            return
        
        # growth modulation
        if growth_mod is None:
            total_growth_rate = self.base_growth
        else:
            total_growth_rate = self.base_growth * (1.0 + float(growth_mod))
        
        # GROWTH: thickness increases where activation is high
        growth_field = (A * total_growth_rate) * self.dt
        self.thickness += growth_field
        
        # CONSTRAINT & PRESSURE: where thickness exceeds threshold -> pressure
        excess = np.clip(self.thickness - self.fold_threshold, 0, None)
        self.pressure = excess ** 2
        
        # FOLDING / BUCKLING: curvature-driven deformation
        lap = cv2.Laplacian(self.thickness, cv2.CV_32F)
        fold_force_z = -lap * self.pressure * self.compression_strength
        self.height_field += fold_force_z * (self.dt * 0.25)
        
        # lateral redistribution: thickness moves away from peaks (simple diffusion + compression)
        grad_y, grad_x = np.gradient(self.thickness)
        fold_force_x = -grad_x * self.pressure * (self.compression_strength * 0.05)
        fold_force_y = -grad_y * self.pressure * (self.compression_strength * 0.05)
        fold_magnitude = np.sqrt(fold_force_x**2 + fold_force_y**2 + fold_force_z**2)
        thickness_redistribution = fold_magnitude * 0.02
        self.thickness -= thickness_redistribution
        
        # DIFFUSION: smooth thickness and height for stability
        self.thickness = gaussian_filter(self.thickness, sigma=self.diffusion_sigma)
        self.height_field = gaussian_filter(self.height_field, sigma=self.diffusion_sigma)
        
        # bounds
        self.thickness = np.clip(self.thickness, self.min_thickness, self.max_thickness)
        
        # measure properties
        self._update_measurements(A)
        
        self.time_step += 1
    
    def _update_measurements(self, activation_map=None):
        # fold density
        self.fold_density_value = float(np.std(self.height_field))
        
        # surface area
        self.surface_area_value = float(self._compute_surface_area(self.height_field))
        
        # fractal estimate
        self.fractal_dim_value = float(self._fractal_estimate(self.height_field))
        
        # spectral concentration of current activation (dominant_mode_power)
        if activation_map is not None:
            self.dominant_mode_power = float(self._spectral_concentration(activation_map))
        else:
            # fallback to thickness spectral content
            self.dominant_mode_power = float(self._spectral_concentration(self.thickness))
        
        # morph_signal: combine coherence, fold-density and dominance into 0..1
        cohere = np.clip(self.dominant_mode_power, 0.0, 1.0)
        density = np.tanh(self.fold_density_value * 0.6)  # compress
        area_norm = np.tanh(self.surface_area_value / (self.resolution * 2.0))
        ms = 0.6 * cohere + 0.3 * density + 0.1 * area_norm
        # lowpass smoothing over history
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
    
    # -------------------------
    # outputs
    # -------------------------
    def get_output(self, port_name):
        if port_name == 'thickness_map':
            # return normalized thickness as image 0..1 (float32)
            t = (self.thickness - self.thickness.min()) / (np.ptp(self.thickness) + 1e-9)
            return t.astype(np.float32)
        if port_name == 'structure_3d':
            h = self.height_field.copy()
            # normalize for visualization
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
        # build a 2x2 panel (numpy float 0..1)
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
        
        # Panel 3: Pressure map (jet)
        pres = (self.pressure - self.pressure.min()) / (np.ptp(self.pressure) + 1e-9)
        pres = cv2.resize(pres, (ps, ps), interpolation=cv2.INTER_LINEAR)
        pres_col = cv2.applyColorMap((pres*255).astype(np.uint8), cv2.COLORMAP_JET)
        panel[ps:ps*2, 0:ps] = pres_col.astype(np.float32) / 255.0
        
        # Panel 4: Metrics / shading visualization
        metrics = np.zeros((ps, ps, 3), dtype=np.float32)
        # shading from height normals
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
        return [
            ("Resolution", "resolution", self.resolution, None),
            ("Base Growth", "base_growth", self.base_growth, None),
            ("Fold Threshold", "fold_threshold", self.fold_threshold, None),
            ("Compression Strength", "compression_strength", self.compression_strength, None),
            ("Diffusion Sigma", "diffusion_sigma", self.diffusion_sigma, None),
            ("Max Thickness", "max_thickness", self.max_thickness, None),
        ]
