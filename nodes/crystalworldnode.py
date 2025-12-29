"""
Crystal World Node
==================

An environment for the Living Crystal to explore.

This creates a 2D world with:
- Reward zones (nutrients) - approaching increases reward signal
- Pain zones (dangers) - approaching increases pain signal  
- Neutral zones - exploration terrain
- Objects with different visual/audio signatures

The crystal's move_x/move_y outputs actually move it through this world,
and the world provides visual/audio/reward/pain signals based on position.

This closes the sensorimotor loop.

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}


class CrystalWorldNode(BaseNode):
    """
    A 2D environment for the Living Crystal to explore and learn in.
    """
    
    NODE_NAME = "Crystal World"
    NODE_TITLE = "Crystal World"
    NODE_CATEGORY = "Environment"
    NODE_COLOR = QtGui.QColor(100, 150, 200) if QtGui else None
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "move_x": "signal",
            "move_y": "signal",
            "reset": "signal",
        }
        
        self.outputs = {
            "visual": "image",
            "audio": "signal",
            "reward": "signal",
            "pain": "signal",
            "world_view": "image",
            "position_x": "signal",
            "position_y": "signal",
        }
        
        self.world_size = 256
        self.view_radius = 32
        
        self.pos_x = self.world_size // 2
        self.pos_y = self.world_size // 2
        self.speed_scale = 2.0
        
        self.reward_zones = []
        self.pain_zones = []
        self.objects = []
        
        self.terrain = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        
        self._generate_world()
        
        self.step_count = 0
        self.total_reward = 0.0
        self.total_pain = 0.0
        self.distance_traveled = 0.0
        
        self.display_image = None
        self._output_values = {}
        
        self._update_display()
    
    def get_config_options(self):
        return [
            ("World Size", "world_size", self.world_size, None),
            ("View Radius", "view_radius", self.view_radius, None),
            ("Speed Scale", "speed_scale", self.speed_scale, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    old_size = self.world_size
                    setattr(self, key, value)
                    if key == "world_size" and value != old_size:
                        self._generate_world()
    
    def _generate_world(self):
        """Generate a random world with features."""
        np.random.seed(42)
        
        size = self.world_size
        
        # Generate terrain
        self.terrain = np.zeros((size, size), dtype=np.float32)
        for scale in [8, 16, 32, 64]:
            noise = np.random.randn(size // scale, size // scale)
            noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_LINEAR)
            self.terrain += noise * (scale / 64.0)
        
        self.terrain = (self.terrain - self.terrain.min()) / (self.terrain.max() - self.terrain.min() + 1e-6)
        
        # Reward zones (green, good)
        self.reward_zones = []
        for _ in range(5):
            x = np.random.randint(30, size - 30)
            y = np.random.randint(30, size - 30)
            radius = np.random.randint(15, 35)
            strength = np.random.uniform(0.5, 1.0)
            self.reward_zones.append((x, y, radius, strength))
        
        # Pain zones (red, bad)
        self.pain_zones = []
        for _ in range(4):
            x = np.random.randint(30, size - 30)
            y = np.random.randint(30, size - 30)
            radius = np.random.randint(10, 25)
            strength = np.random.uniform(0.3, 0.8)
            self.pain_zones.append((x, y, radius, strength))
        
        # Objects with distinct signatures
        self.objects = []
        patterns = ['circle', 'square', 'triangle', 'cross']
        for i, pattern in enumerate(patterns):
            x = np.random.randint(40, size - 40)
            y = np.random.randint(40, size - 40)
            freq = 5.0 + i * 10.0
            self.objects.append((x, y, pattern, freq))
        
        self.pos_x = size // 2
        self.pos_y = size // 2
        
        print(f"[CrystalWorld] Generated world: {len(self.reward_zones)} rewards, {len(self.pain_zones)} dangers")
    
    def _read_input(self, name, default=0.0):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                return float(val)
            except:
                return default
        return default
    
    def step(self):
        self.step_count += 1
        
        move_x = self._read_input("move_x", 0.0) * self.speed_scale
        move_y = self._read_input("move_y", 0.0) * self.speed_scale
        reset = self._read_input("reset", 0.0) > 0.5
        
        if reset:
            self._generate_world()
            self.total_reward = 0.0
            self.total_pain = 0.0
            self.distance_traveled = 0.0
            return
        
        old_x, old_y = self.pos_x, self.pos_y
        self.pos_x += move_x
        self.pos_y += move_y
        
        # Toroidal world
        self.pos_x = self.pos_x % self.world_size
        self.pos_y = self.pos_y % self.world_size
        
        dx = self.pos_x - old_x
        dy = self.pos_y - old_y
        self.distance_traveled += np.sqrt(dx * dx + dy * dy)
        
        # Calculate signals
        reward = 0.0
        pain = 0.0
        
        for (rx, ry, radius, strength) in self.reward_zones:
            dist = np.sqrt((self.pos_x - rx) ** 2 + (self.pos_y - ry) ** 2)
            if dist < radius:
                reward += strength * (1.0 - dist / radius)
        
        for (px, py, radius, strength) in self.pain_zones:
            dist = np.sqrt((self.pos_x - px) ** 2 + (self.pos_y - py) ** 2)
            if dist < radius:
                pain += strength * (1.0 - dist / radius)
        
        self.total_reward += reward
        self.total_pain += pain
        
        audio = 0.0
        for (ox, oy, pattern, freq) in self.objects:
            dist = np.sqrt((self.pos_x - ox) ** 2 + (self.pos_y - oy) ** 2)
            if dist < 50:
                amplitude = (1.0 - dist / 50.0) * 10.0
                audio += amplitude * np.sin(2 * np.pi * freq * self.step_count * 0.01)
        
        self._output_values = {
            "reward": reward,
            "pain": pain,
            "audio": audio,
            "position_x": self.pos_x,
            "position_y": self.pos_y,
        }
        
        self._update_display()
    
    def get_output(self, port_name):
        if port_name == "visual":
            return self._render_view()
        elif port_name == "world_view":
            return self._render_world()
        elif port_name in self._output_values:
            return self._output_values.get(port_name, 0.0)
        return None
    
    def _render_view(self):
        """Render crystal's local view."""
        size = self.view_radius * 2
        view = np.zeros((size, size, 3), dtype=np.uint8)
        
        for dy in range(-self.view_radius, self.view_radius):
            for dx in range(-self.view_radius, self.view_radius):
                wx = int(self.pos_x + dx) % self.world_size
                wy = int(self.pos_y + dy) % self.world_size
                vx = dx + self.view_radius
                vy = dy + self.view_radius
                
                terrain_val = int(self.terrain[wy, wx] * 100)
                view[vy, vx] = [terrain_val, terrain_val, terrain_val]
        
        # Reward zones (green)
        for (rx, ry, radius, strength) in self.reward_zones:
            for dy in range(-self.view_radius, self.view_radius):
                for dx in range(-self.view_radius, self.view_radius):
                    wx = int(self.pos_x + dx) % self.world_size
                    wy = int(self.pos_y + dy) % self.world_size
                    vx = dx + self.view_radius
                    vy = dy + self.view_radius
                    
                    dist = np.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)
                    if dist < radius:
                        intensity = int((1.0 - dist / radius) * strength * 200)
                        view[vy, vx, 1] = min(255, view[vy, vx, 1] + intensity)
        
        # Pain zones (red)
        for (px, py, radius, strength) in self.pain_zones:
            for dy in range(-self.view_radius, self.view_radius):
                for dx in range(-self.view_radius, self.view_radius):
                    wx = int(self.pos_x + dx) % self.world_size
                    wy = int(self.pos_y + dy) % self.world_size
                    vx = dx + self.view_radius
                    vy = dy + self.view_radius
                    
                    dist = np.sqrt((wx - px) ** 2 + (wy - py) ** 2)
                    if dist < radius:
                        intensity = int((1.0 - dist / radius) * strength * 200)
                        view[vy, vx, 2] = min(255, view[vy, vx, 2] + intensity)
        
        # Objects
        for (ox, oy, pattern, freq) in self.objects:
            rel_x = ox - self.pos_x
            rel_y = oy - self.pos_y
            
            if rel_x > self.world_size // 2:
                rel_x -= self.world_size
            if rel_x < -self.world_size // 2:
                rel_x += self.world_size
            if rel_y > self.world_size // 2:
                rel_y -= self.world_size
            if rel_y < -self.world_size // 2:
                rel_y += self.world_size
            
            vx = int(rel_x + self.view_radius)
            vy = int(rel_y + self.view_radius)
            
            if 5 <= vx < size - 5 and 5 <= vy < size - 5:
                color = (255, 200, 100)
                if pattern == 'circle':
                    cv2.circle(view, (vx, vy), 4, color, -1)
                elif pattern == 'square':
                    cv2.rectangle(view, (vx - 4, vy - 4), (vx + 4, vy + 4), color, -1)
                elif pattern == 'triangle':
                    pts = np.array([[vx, vy - 5], [vx - 5, vy + 5], [vx + 5, vy + 5]], np.int32)
                    cv2.fillPoly(view, [pts], color)
                elif pattern == 'cross':
                    cv2.line(view, (vx - 4, vy), (vx + 4, vy), color, 2)
                    cv2.line(view, (vx, vy - 4), (vx, vy + 4), color, 2)
        
        return view
    
    def _render_world(self):
        """Render entire world map."""
        img = np.zeros((self.world_size, self.world_size, 3), dtype=np.uint8)
        
        terrain_vis = (self.terrain * 80).astype(np.uint8)
        img[:, :, 0] = terrain_vis
        img[:, :, 1] = terrain_vis
        img[:, :, 2] = terrain_vis
        
        for (rx, ry, radius, strength) in self.reward_zones:
            cv2.circle(img, (int(rx), int(ry)), radius, (0, int(200 * strength), 0), -1)
        
        for (px, py, radius, strength) in self.pain_zones:
            cv2.circle(img, (int(px), int(py)), radius, (0, 0, int(200 * strength)), -1)
        
        for (ox, oy, pattern, freq) in self.objects:
            cv2.circle(img, (int(ox), int(oy)), 5, (0, 255, 255), -1)
        
        cx, cy = int(self.pos_x), int(self.pos_y)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), self.view_radius, (100, 100, 100), 1)
        
        return img
    
    def _update_display(self):
        """Create main display."""
        w, h = 512, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        cv2.putText(img, "CRYSTAL WORLD", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 200), 2)
        
        world = self._render_world()
        world_small = cv2.resize(world, (200, 200))
        img[50:250, 10:210] = world_small
        cv2.putText(img, "World Map", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        view = self._render_view()
        view_resized = cv2.resize(view, (200, 200), interpolation=cv2.INTER_NEAREST)
        img[50:250, 230:430] = view_resized
        cv2.putText(img, "Crystal View", (230, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        stats_x = 450
        cv2.putText(img, f"Pos: ({self.pos_x:.0f},{self.pos_y:.0f})", (stats_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(img, f"Dist: {self.distance_traveled:.0f}", (stats_x, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(img, f"Reward: {self.total_reward:.1f}", (stats_x, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        cv2.putText(img, f"Pain: {self.total_pain:.1f}", (stats_x, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if QtGui:
            qimg = QtGui.QImage(img_rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
            self.display_image = qimg
    
    def get_display_image(self):
        return self.display_image
