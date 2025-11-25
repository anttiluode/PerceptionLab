"""
Fractal Attractor Neural Field (Strict Bio-Driven Edition)
----------------------------------------------------------
1. NO AUTO-PILOT. If Delta is 0, Time stops.
2. NO SIGNAL FAKING. If EEG inputs are silent, the screen goes dark (Void Mode).
3. Pure Signal-to-Geometry mapping.
"""

import numpy as np
import cv2
import __main__

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class FractalAttractorNeuralFieldNode(BaseNode):
    NODE_CATEGORY = "Generators"
    NODE_COLOR = QtGui.QColor(255, 100, 50) # Blaze Orange

    def __init__(self):
        super().__init__()
        self.node_title = "Fractal Attractor (Strict)"
        
        self.inputs = {
            # --- EEG Drivers ---
            'delta': 'signal',  # REQUIRED: Time Flow
            'theta': 'signal',  # Scale
            'alpha': 'signal',  # Warp
            'beta': 'signal',   # Turbulence
            'gamma': 'signal',  # Roughness
            
            # --- Base Parameters (Offsets) ---
            'base_scale': 'signal',
            'base_roughness': 'signal',
            'base_warp': 'signal',
            'sensitivity': 'signal'
        }
        
        self.outputs = {
            'field_out': 'image'
        }
        
        # Internal State
        self.time_counter = 0.0
        self.resolution = 256
        self.display_buffer = np.zeros((256, 256, 3), dtype=np.uint8)
        self._output_data = {}
        
        # Defaults (Can be set to 0 for total dependency)
        self.base_scale = 1.0
        self.base_roughness = 0.4
        self.base_warp = 0.5
        self.sensitivity = 2.0 # Higher sensitivity since we removed auto-pilot

    def step(self):
        # --- Helper: Safe Signal Getter ---
        def get_signal(name, default=0.0, scale=1.0):
            val = self.get_blended_input(name)
            if val is None: return default
            if isinstance(val, (list, tuple, np.ndarray)):
                try: val = float(np.mean(np.abs(val)))
                except: return default
            try:
                f_val = float(val)
                if not np.isfinite(f_val): return default
                return f_val * scale
            except:
                return default

        # 1. Get Base Config
        b_scale = get_signal('base_scale', default=self.base_scale)
        b_rough = get_signal('base_roughness', default=self.base_roughness)
        b_warp = get_signal('base_warp', default=self.base_warp)
        sens = get_signal('sensitivity', default=self.sensitivity)

        # 2. Get EEG Modulators
        s_delta = get_signal('delta', default=0.0, scale=sens)
        s_theta = get_signal('theta', default=0.0, scale=sens)
        s_alpha = get_signal('alpha', default=0.0, scale=sens)
        s_beta  = get_signal('beta',  default=0.0, scale=sens)
        s_gamma = get_signal('gamma', default=0.0, scale=sens)
        
        # 3. THE LIFE FORCE CHECK (No Signal = No Show)
        total_activity = s_delta + s_theta + s_alpha + s_beta + s_gamma
        
        if total_activity < 0.01:
            # VOID MODE: If no signal, show a faint static or blackness
            # This proves the system is waiting for input.
            self.display_buffer[:] = 0 # Black out
            
            # Tiny movement just to show the engine is on, but "Idling"
            self.time_counter += 0.001 
            
            # Render a "NO SIGNAL" glyph or just a faint grid
            cv2.putText(self.display_buffer, "AWAITING SIGNAL", (60, 128), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            
            self._output_data['field_out'] = self.display_buffer
            return # Skip the heavy fractal math

        # 4. Map EEG to Physics (Strict Mapping)
        
        # Time Flow (Delta): 
        # If Delta is 0, Time stops (Frozen snapshot of the mind).
        dt = s_delta * 0.1 
        self.time_counter += dt

        # Scale (Theta): 
        scale = np.clip(b_scale + (s_theta * 0.5), 0.1, 10.0)

        # Warp (Alpha):
        warp_strength = b_warp + (s_alpha * 0.5)

        # Turbulence (Beta):
        turbulence = s_beta * 2.0

        # Roughness (Gamma):
        roughness = np.clip(b_rough + (s_gamma * 0.2), 0.1, 0.99)

        # 5. Render The Attractor
        h, w = self.resolution, self.resolution
        
        try:
            x = np.linspace(0, 5, w)
            y = np.linspace(0, 5, h)
            X, Y = np.meshgrid(x, y)
            
            t = self.time_counter
            
            # Pure FBM (No sine-grid fakery)
            n1 = np.sin(X * scale + t) * np.cos(Y * scale - t)
            
            s2 = scale * 2.0
            n2 = (np.sin(X * s2 + t*1.5) * np.cos(Y * s2 - t*1.5)) * roughness
            
            s3 = scale * 4.0
            n3 = (np.sin(X * s3 - t*2.0) * np.cos(Y * s3 + t*2.0)) * (roughness**2)
            
            s4 = scale * 8.0
            n4 = (np.sin(X * s4 + turbulence) * np.cos(Y * s4 + turbulence)) * (roughness**3)
            
            noise_field = n1 + n2 + n3 + n4
            
            # Self-Warp
            dx = noise_field * warp_strength
            dy = noise_field * warp_strength
            
            final_pattern = np.sin((X + dx) * scale + t) * np.cos((Y + dy) * scale - t)
            
            # Output
            final_field = (final_pattern + 1.0) / 2.0
            img_data = (final_field * 255).astype(np.uint8)
            img_color = cv2.applyColorMap(img_data, cv2.COLORMAP_INFERNO)
            
            self.display_buffer = img_color
            self._output_data['field_out'] = img_color

        except Exception:
            pass

    def get_output(self, port_name):
        return self._output_data.get(port_name, None)

    def get_display_image(self):
        return self.display_buffer

    def get_config_options(self):
        return [
            ("Base Scale", "base_scale", self.base_scale, "float"),
            ("Base Roughness", "base_roughness", self.base_roughness, "float"),
            ("Base Warp", "base_warp", self.base_warp, "float"),
            ("Sensitivity", "sensitivity", self.sensitivity, "float"),
            ("Resolution", "resolution", self.resolution, "int")
        ]
        
    def set_config_options(self, options):
        if "base_scale" in options: self.base_scale = float(options["base_scale"])
        if "base_roughness" in options: self.base_roughness = float(options["base_roughness"])
        if "base_warp" in options: self.base_warp = float(options["base_warp"])
        if "sensitivity" in options: self.sensitivity = float(options["sensitivity"])
        if "resolution" in options: self.resolution = int(options["resolution"])