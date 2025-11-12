import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np
import cv2
import random

class SpriteEngineNode(BaseNode):
    """
    Multiplies an input image (sprite) into a particle system,
    arranging copies in a lattice or as randomly moving particles.
    (v2 - Fixed config/init bug)
    """
    NODE_CATEGORY = "Visualizer"
    NODE_COLOR = QtGui.QColor(220, 100, 150) # Pink

    def __init__(self, mode='Random', count=20, scale=1.0, speed=1.0, opacity=0.5, output_size=256):
        super().__init__()
        self.node_title = "Sprite Engine"
        
        # --- Inputs and Outputs ---
        self.inputs = {
            'image_in': 'image',
            'background_in': 'image' # Optional
        }
        self.outputs = {'image_out': 'image'}
        
        # --- Configurable ---
        self.modes = ['Lattice', 'Random']
        self.mode = mode if mode in self.modes else self.modes[0]
        self.count = int(count)
        self.scale = float(scale)
        self.speed = float(speed)
        self.opacity = float(opacity)
        self.output_size = int(output_size)
        
        # --- Internal State ---
        self.output_image = np.zeros((self.output_size, self.output_size, 3), dtype=np.float32)
        self.particles = [] # List of [x, y, vx, vy]
        
        # Store the state that created the particles
        self._last_mode = None
        self._last_count = -1
        self._last_output_size = -1
        
        self._init_particles() # Run once on creation

    def get_config_options(self):
        return [
            ("Mode", "mode", self.mode, [('Lattice', 'Lattice'), ('Random', 'Random')]),
            ("Count", "count", self.count, None),
            ("Scale", "scale", self.scale, None),
            ("Speed", "speed", self.speed, None),
            ("Opacity", "opacity", self.opacity, None),
            ("Resolution", "output_size", self.output_size, None),
        ]

    def set_config_options(self, options):
        # Simply update the values. The `step` function will handle the reset.
        if "mode" in options: self.mode = options["mode"]
        if "count" in options: self.count = int(options["count"])
        if "output_size" in options: self.output_size = int(options["output_size"])
        if "scale" in options: self.scale = float(options["scale"])
        if "speed" in options: self.speed = float(options["speed"])
        if "opacity" in options: self.opacity = float(options["opacity"])

    def _init_particles(self):
        """(Re)Initializes all particle positions and velocities."""
        self.particles = []
        if self.count <= 0: return

        if self.mode == 'Lattice':
            grid_size = int(np.ceil(np.sqrt(self.count)))
            if grid_size == 0: return
            spacing_x = self.output_size / grid_size
            spacing_y = self.output_size / grid_size
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= self.count: break
                    x = (j + 0.5) * spacing_x
                    y = (i + 0.5) * spacing_y
                    self.particles.append([x, y, 0, 0]) # No velocity
                    idx += 1
        
        elif self.mode == 'Random':
            for _ in range(self.count):
                x = random.uniform(0, self.output_size)
                y = random.uniform(0, self.output_size)
                vx = random.uniform(-1.0, 1.0) * self.speed
                vy = random.uniform(-1.0, 1.0) * self.speed
                self.particles.append([x, y, vx, vy])
        
        # Store the settings we just used
        self._last_mode = self.mode
        self._last_count = self.count
        self._last_output_size = self.output_size

    def step(self):
        # --- NEW ROBUSTNESS CHECK ---
        # If the settings have changed, re-init the particles
        if (self.mode != self._last_mode or 
            self.count != self._last_count or 
            self.output_size != self._last_output_size):
            self._init_particles()
        # --- END CHECK ---

        img_in = self.get_blended_input('image_in', 'first')
        bg_in = self.get_blended_input('background_in', 'first')

        if img_in is None:
            return # Need a sprite to draw
        
        # --- 1. Setup Canvas ---
        if bg_in is not None:
            self.output_image = cv2.resize(bg_in, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        else:
            self.output_image = np.zeros((self.output_size, self.output_size, 3), dtype=np.float32)
            
        if self.output_image.ndim == 2:
            self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_GRAY2BGR)

        # --- 2. Prepare Sprite ---
        try:
            if img_in.ndim == 2:
                img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
            
            base_h, base_w = img_in.shape[:2]
            sprite_size = max(base_h, base_w, 1) 
            
            sprite_w = int(sprite_size * self.scale)
            sprite_h = int(sprite_size * self.scale)
            
            if sprite_w <= 0 or sprite_h <= 0:
                return 
                
            sprite = cv2.resize(img_in, (sprite_w, sprite_h), interpolation=cv2.INTER_LINEAR)
            
            sprite_gray = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)
            mask = (sprite_gray > 0.01).astype(np.float32)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
            
            sprite = sprite * self.opacity
            
        except Exception as e:
            print(f"SpriteEngine Error: {e}")
            return 

        # --- 3. Update and Draw Particles ---
        for i in range(len(self.particles)):
            x, y, vx, vy = self.particles[i]
            
            if self.mode == 'Random':
                x += vx
                y += vy
                
                # Update particle velocity based on speed (in case it changed)
                vx = np.sign(vx) * self.speed if self.speed > 0 else 0
                vy = np.sign(vy) * self.speed if self.speed > 0 else 0

                if x <= 0 or x >= self.output_size: vx = -vx
                if y <= 0 or y >= self.output_size: vy = -vy
                
                # Screen wrap (alternative to bounce)
                # x = x % self.output_size
                # y = y % self.output_size
                
                self.particles[i] = [x, y, vx, vy]

            try:
                x1 = int(x - sprite_w / 2)
                y1 = int(y - sprite_h / 2)
                x2 = x1 + sprite_w
                y2 = y1 + sprite_h
                
                s_x1, s_y1, s_x2, s_y2 = 0, 0, sprite_w, sprite_h
                
                if x1 < 0: s_x1 = -x1; x1 = 0
                if y1 < 0: s_y1 = -y1; y1 = 0
                if x2 > self.output_size: s_x2 = sprite_w - (x2 - self.output_size); x2 = self.output_size
                if y2 > self.output_size: s_y2 = sprite_h - (y2 - self.output_size); y2 = self.output_size

                if x1 >= x2 or y1 >= y2 or s_x1 >= s_x2 or s_y1 >= s_y2:
                    continue
                    
                sprite_slice = sprite[s_y1:s_y2, s_x1:s_x2]
                mask_slice = mask[s_y1:s_y2, s_x1:s_x2]
                bg_slice = self.output_image[y1:y2, x1:x2]
                
                blended = bg_slice * (1.0 - mask_slice) + (sprite_slice * mask_slice)
                self.output_image[y1:y2, x1:x2] = blended

            except Exception as e:
                pass 

        self.output_image = np.clip(self.output_image, 0, 1)

    def get_output(self, port_name):
        if port_name == 'image_out':
            return self.output_image
        return None

    def get_display_image(self):
        return self.output_image