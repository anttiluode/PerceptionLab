"""
H/S/L Fractal Pattern Node - Generates a generative fractal
structure based on H (Hub), S (State), and L (Loop) inputs.

Ported from hslcity.html
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

# --- Core Simulation Classes (from hslcity.html) ---

class HSLPattern:
    def __init__(self, x, y, angle, scale, depth, patternType='h'):
        self.x = x
        self.y = y
        self.angle = angle
        self.scale = scale
        self.depth = depth
        self.patternType = patternType
        self.phase = np.random.rand() * np.pi * 2
        self.children = []
        self.age = 0
        self.time = 0.0
        
        if depth > 0:
            self.generateChildren()
            
    def generateChildren(self):
        branchAngle = 45.0 * np.pi / 180
        childScale = self.scale * 0.6
        childDepth = self.depth - 1
        
        if self.patternType == 'h': # Hubs branch into states
            for i in range(3):
                childAngle = self.angle + (i - 1) * branchAngle
                childType = ['s', 'l', 's'][i]
                self.children.append(HSLPattern(
                    self.x + np.cos(childAngle) * self.scale * 40,
                    self.y + np.sin(childAngle) * self.scale * 40,
                    childAngle, childScale, childDepth, childType
                ))
        elif self.patternType == 'l': # Loops create circular patterns
            for i in range(4):
                childAngle = self.angle + i * np.pi / 2
                childType = 'l'
                self.children.append(HSLPattern(
                    self.x + np.cos(childAngle) * self.scale * 30,
                    self.y + np.sin(childAngle) * self.scale * 30,
                    childAngle, childScale, childDepth, childType
                ))
        else: # 's' states transition
            childType = 'l' if np.random.rand() > 0.5 else 'h'
            self.children.append(HSLPattern(
                self.x + np.cos(self.angle) * self.scale * 50,
                self.y + np.sin(self.angle) * self.scale * 50,
                self.angle + (np.random.rand() - 0.5) * branchAngle,
                childScale, childDepth, childType
            ))
            
    def update(self, dt, global_time):
        self.age += dt
        self.time = global_time
        for child in self.children:
            child.update(dt, global_time)
            
    def draw(self, ctx_img, pulse_intensity):
        # Calculate pulsation
        pulse = 1.0
        if self.patternType == 'h':
            pulse = 1 + np.sin(self.time * 3 + self.phase) * pulse_intensity * 0.5
        elif self.patternType == 'l':
            pulse = 1 + np.sin(self.time + self.phase) * pulse_intensity * 0.2
        else:
            pulse = 1 + np.sin(self.time * 2 + self.phase) * pulse_intensity * 0.3
        
        # Set color (BGR)
        color = (0,0,0)
        if self.patternType == 'h': color = (100, 100, 255) # Red
        elif self.patternType == 'l': color = (100, 255, 100) # Green
        else: color = (255, 100, 100) # Blue
        
        radius = int(self.scale * 15 * pulse)
        if radius < 1: radius = 1
        
        # Draw the node
        pt = (int(self.x), int(self.y))
        cv2.circle(ctx_img, pt, radius, color, -1, cv2.LINE_AA)
        
        # Draw connections
        for child in self.children:
            child_pt = (int(child.x), int(child.y))
            cv2.line(ctx_img, pt, child_pt, (100, 100, 100), 1, cv2.LINE_AA)
            child.draw(ctx_img, pulse_intensity)

# --- The Main Node Class ---

class HSLPatternNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(100, 200, 250) # Crystalline blue
    
    def __init__(self, size=128, speed=1.0, pulse=0.8, depth=4):
        super().__init__()
        self.node_title = "HSL Pattern (MTX)"
        
        self.inputs = {
            'H_in': 'signal', # Hub trigger
            'S_in': 'signal', # State trigger
            'L_in': 'signal'  # Loop trigger
        }
        self.outputs = {'image': 'image'}
        
        self.size = int(size)
        self.speed = float(speed)
        self.pulse = float(pulse)
        self.depth = int(depth)
        
        self.time = 0.0
        self.root_patterns = []
        self.output_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Last trigger values
        self.last_h = 0.0
        self.last_s = 0.0
        self.last_l = 0.0
        
        # Initialize
        self._add_seed(self.size // 2, self.size // 2, 'h')

    def _add_seed(self, x, y, pattern_type):
        """Adds a new root pattern to the simulation."""
        new_pattern = HSLPattern(
            x, y, 
            np.random.rand() * 2 * np.pi, 
            scale=1.0, 
            depth=self.depth, 
            patternType=pattern_type
        )
        self.root_patterns.append(new_pattern)
        # Limit total patterns
        if len(self.root_patterns) > 20:
            self.root_patterns.pop(0)

    def step(self):
        # 1. Handle Inputs (check for rising edge)
        h_in = self.get_blended_input('H_in', 'sum') or 0.0
        s_in = self.get_blended_input('S_in', 'sum') or 0.0
        l_in = self.get_blended_input('L_in', 'sum') or 0.0
        
        rand_x = np.random.randint(self.size * 0.2, self.size * 0.8)
        rand_y = np.random.randint(self.size * 0.2, self.size * 0.8)
        
        if h_in > 0.5 and self.last_h <= 0.5: self._add_seed(rand_x, rand_y, 'h')
        if s_in > 0.5 and self.last_s <= 0.5: self._add_seed(rand_x, rand_y, 's')
        if l_in > 0.5 and self.last_l <= 0.5: self._add_seed(rand_x, rand_y, 'l')
        
        self.last_h, self.last_s, self.last_l = h_in, s_in, l_in
        
        # 2. Update time and simulation
        self.time += self.speed * 0.02
        
        # 3. Draw
        # Fade the background
        self.output_image = (self.output_image * 0.9).astype(np.uint8)
        
        for pattern in self.root_patterns:
            pattern.update(0.016, self.time)
            pattern.draw(self.output_image, self.pulse)

    def get_output(self, port_name):
        if port_name == 'image':
            return self.output_image.astype(np.float32) / 255.0
        return None
        
    def get_display_image(self):
        img_rgb = np.ascontiguousarray(self.output_image)
        h, w = img_rgb.shape[:2]
        return QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution", "size", self.size, None),
            ("Speed", "speed", self.speed, None),
            ("Pulsation", "pulse", self.pulse, None),
            ("Recursion Depth", "depth", self.depth, None),
        ]