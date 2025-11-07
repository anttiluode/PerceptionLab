"""
Living Organism Node - A unified "living system" simulation with:
- A non-linear wave field (the "environment")
- 12 Homeostatic Cognitive Units (HCUs) forming a "soft organism"
- An MTX bus for agent communication

Ported from h_cu_life.py
Requires: pip install numpy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import math
import random
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

# --- Simulation Parameters (from h_cu_life.py) ---
GRID = 96                  # Smaller grid for performance
DT = 0.12                  
C = 0.85                   
DAMP = 0.015               
NONLIN = 0.18              
NOISE_AMP = 0.0007         
NUM_HCU = 12               
RING = True                
SPRING_K = 0.12            
SPRING_REST = 8.0          # Adjusted for smaller grid
SPACING_REPULSION = 150.0  
HCU_SENSE_SIGMA = 3.0      
HCU_STAMP = 0.012          
HCU_MOVE_GAIN = 0.85       
HCU_NOISE = 0.35           
HCU_TARGET_AMP = 0.30      
HCU_BASE_FREQ = 1.6        
BUS_MAX = 60               

# Prebuild a small Gaussian stamp used by HCUs
def gaussian_stamp(radius=7, sigma=HCU_SENSE_SIGMA):
    r = int(radius)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    g /= g.sum()
    return g.astype(np.float32)

STAMP = gaussian_stamp(7, HCU_SENSE_SIGMA)

def splat(field, x, y, amp):
    """Add a Gaussian blob to the field at (x,y) with amplitude amp."""
    h, w = field.shape
    r = STAMP.shape[0]//2
    xi, yi = int(x), int(y)
    x0, x1 = max(0, xi-r), min(w, xi+r+1)
    y0, y1 = max(0, yi-r), min(h, yi+r+1)
    sx0, sx1 = r-(xi-x0), r+(x1-xi)
    sy0, sy1 = r-(yi-y0), r+(y1-yi)
    if x0 < x1 and y0 < y1:
        field[y0:y1, x0:x1] += amp * STAMP[sy0:sy1, sx0:sx1]

# --- Core Simulation Classes (from h_cu_life.py) ---

class HCU:
    """Homeostatic Cognitive Unit with internal Hopf oscillator."""
    def __init__(self, x, y, idx):
        self.x = float(x); self.y = float(y)
        self.vx = 0.0; self.vy = 0.0
        self.idx = idx
        self.z = complex(np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1))
        self.mu = 1.0
        self.omega = np.random.uniform(0.8, 1.2)*HCU_BASE_FREQ
        self.energy = 0.0
        self.energy_smooth = 0.0
        self.last_token = None
        self.token_clock = 0.0

    def hopf_step(self, u, dt):
        z = self.z
        r2 = (z.real*z.real + z.imag*z.imag)
        dz = complex(self.mu - r2, self.omega) * z + u
        z = z + dz*dt
        self.z = z

    def sense(self, field):
        h, w = field.shape
        xi, yi = int(self.x), int(self.y)
        r = STAMP.shape[0]//2
        x0, x1 = max(0, xi-r), min(w, xi+r+1)
        y0, y1 = max(0, yi-r), min(h, yi+r+1)
        sx0, sx1 = r-(xi-x0), r+(x1-xi)
        sy0, sy1 = r-(yi-y0), r+(y1-yi)
        
        patch = field[y0:y1, x0:x1]
        mask = STAMP[sy0:sy1, sx0:sx1]
        val = float((patch * mask).sum())
        
        gx = float((field[yi, (xi+1)%w] - field[yi, (xi-1)%w]) * 0.5)
        gy = float((field[(yi+1)%h, xi] - field[(yi-1)%h, xi]) * 0.5)
        return val, gx, gy

    def act(self, field, dt, bus):
        val, gx, gy = self.sense(field)
        r = abs(self.z)
        amp_err = (HCU_TARGET_AMP - r)
        u = complex(val*0.8, amp_err*0.6)
        self.hopf_step(u, dt)

        energy = abs(amp_err) + 0.3*math.sqrt(gx*gx + gy*gy)
        self.energy = energy
        self.energy_smooth = 0.92*self.energy_smooth + 0.08*energy

        self.vx += (-gx * HCU_MOVE_GAIN + np.random.randn()*HCU_NOISE) * dt
        self.vy += (-gy * HCU_MOVE_GAIN + np.random.randn()*HCU_NOISE) * dt
        self.vx *= 0.96; self.vy *= 0.96

        self.x = (self.x + self.vx) % field.shape[1]
        self.y = (self.y + self.vy) % field.shape[0]

        token = None
        if self.energy_smooth < 0.12:
            splat(field, self.x, self.y, +HCU_STAMP)
            token = 'l3' # focus
        elif self.energy_smooth > 0.28:
            splat(field, self.x, self.y, -HCU_STAMP)
            token = 'h0' # novelty
        else:
            token = 's1' # scan

        if token == self.last_token:
            self.token_clock += dt
        else:
            if self.last_token is not None and self.token_clock > 0.12:
                bus.append((self.idx, self.last_token, self.token_clock))
            self.last_token = token
            self.token_clock = 0.0
        return token

class World:
    """The simulation world, containing the field and agents"""
    def __init__(self, size):
        self.size = size
        self.phi = np.zeros((size, size), dtype=np.float32)
        self.phi_prev = np.zeros((size, size), dtype=np.float32)
        self.field_noise_on = True
        self.bus = []
        self.time = 0.0

        self.agents = []
        cx, cy = size//2, size//2
        for i in range(NUM_HCU):
            angle = (i / NUM_HCU) * 2 * math.pi
            r = size * 0.2
            self.agents.append(HCU(cx + r * math.cos(angle), cy + r * math.sin(angle), i))
        
        self.springs = []
        for i in range(NUM_HCU):
            j = (i + 1) % NUM_HCU if RING else i + 1
            if j < NUM_HCU:
                self.springs.append((self.agents[i], self.agents[j]))

    def step_field(self, dt):
        lap = (np.roll(self.phi, 1, 0) + np.roll(self.phi, -1, 0) +
               np.roll(self.phi, 1, 1) + np.roll(self.phi, -1, 1) - 4*self.phi)
        
        nonlinear_force = NONLIN * (self.phi - self.phi**3)
        phi_dot = (self.phi - self.phi_prev) / dt
        force = C*C * lap - DAMP * phi_dot + nonlinear_force

        phi_new = 2*self.phi - self.phi_prev + force * dt*dt
        self.phi_prev, self.phi = self.phi, phi_new
        
        if self.field_noise_on:
            self.phi += (np.random.randn(self.size, self.size) * NOISE_AMP).astype(np.float32)

    def step_agents(self, dt):
        for a, b in self.springs:
            dx, dy = b.x - a.x, b.y - a.y
            dist = math.hypot(dx, dy) + 1e-6
            force_mag = SPRING_K * (dist - SPRING_REST)
            fx, fy = force_mag * dx / dist, force_mag * dy / dist
            a.vx += fx; a.vy += fy
            b.vx -= fx; b.vy -= fy
        
        for i, a in enumerate(self.agents):
            for j in range(i + 1, len(self.agents)):
                b = self.agents[j]
                dx, dy = b.x - a.x, b.y - a.y
                dist_sq = dx*dx + dy*dy + 1e-6
                if dist_sq < (SPRING_REST * 2.5)**2:
                    force_mag = SPACING_REPULSION / dist_sq
                    fx, fy = force_mag * dx / math.sqrt(dist_sq), force_mag * dy / math.sqrt(dist_sq)
                    a.vx -= fx; a.vy -= fy
                    b.vx += fx; b.vy += fy

        self.bus.clear()
        for agent in self.agents:
            agent.act(self.phi, dt, self.bus)
    
    def step(self, dt):
        self.time += dt
        self.step_field(dt)
        self.step_agents(dt)


# --- The Main Node Class ---

class LivingOrganismNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(20, 150, 150) # Biological Teal
    
    def __init__(self, size=96, steps_per_frame=2):
        super().__init__()
        self.node_title = "Living Organism (HCU)"
        
        self.inputs = {
            'noise_toggle': 'signal', # > 0.5 = noise ON
            'guidance_pulse': 'signal' # > 0.5 = inject guidance
        }
        self.outputs = {
            'field_image': 'image',   # The main wave field (phi)
            'avg_energy': 'signal',   # Average energy of all agents
            'bus_activity': 'signal'  # Number of MTX tokens this frame
        }
        
        self.size = int(size)
        self.steps_per_frame = int(steps_per_frame)
        
        # Initialize simulation
        self.world = World(size=self.size)
        self.last_guidance_trigger = 0.0

    def step(self):
        # 1. Handle Inputs
        noise_sig = self.get_blended_input('noise_toggle', 'sum')
        if noise_sig is not None:
            self.world.field_noise_on = (noise_sig > 0.5)
            
        guidance_sig = self.get_blended_input('guidance_pulse', 'sum')
        if guidance_sig is not None and guidance_sig > 0.5 and self.last_guidance_trigger <= 0.5:
            # Inject a global "thought" (guidance)
            rand_agent = random.choice(self.world.agents)
            self.world.bus.append((-1, 'h0', 0.5)) # -1 for global source
            splat(self.world.phi, rand_agent.x, rand_agent.y, -HCU_STAMP * 5)
        self.last_guidance_trigger = guidance_sig or 0.0

        # 2. Run simulation steps
        for _ in range(self.steps_per_frame):
            self.world.step(DT)

    def get_output(self, port_name):
        if port_name == 'field_image':
            # Normalize phi field [-0.4, 0.4] to [0, 1]
            return np.clip((self.world.phi + 0.4) / 0.8, 0.0, 1.0)
            
        elif port_name == 'avg_energy':
            # Average homeostatic energy of all agents
            if self.world.agents:
                return np.mean([a.energy_smooth for a in self.world.agents])
            return 0.0
            
        elif port_name == 'bus_activity':
            # Number of MTX tokens generated this frame
            return float(len(self.world.bus))
            
        return None
        
    def get_display_image(self):
        # Get the field image
        img_data = self.get_output('field_image')
        if img_data is None: return None
        
        img_u8 = (img_data * 255).astype(np.uint8)
        
        # Apply colormap (Viridis, as in screenshot)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
        
        # Draw the organism (agents and springs)
        for a, b in self.world.springs:
            pt1 = (int(a.x), int(a.y))
            pt2 = (int(b.x), int(b.y))
            cv2.line(img_color, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
            
        for a in self.world.agents:
            pt = (int(a.x), int(a.y))
            # Determine color based on internal state
            if a.last_token == 'l3': color = (0, 255, 0) # Green (focus)
            elif a.last_token == 'h0': color = (0, 0, 255) # Red (novelty)
            else: color = (255, 0, 0) # Blue (scan)
            
            cv2.circle(img_color, pt, 3, color, -1)
            cv2.circle(img_color, pt, 3, (255, 255, 255), 1)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Resolution (NxN)", "size", self.size, None),
            ("Sim Steps / Frame", "steps_per_frame", self.steps_per_frame, None),
        ]