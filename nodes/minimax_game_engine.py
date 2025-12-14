"""
Minimax Game Engine - Ma's Adversarial Learning Dynamic
========================================================
Implements the GAME-THEORETIC aspect of Ma's framework.
(FIXED: Added _safe_radius and robust rendering to prevent OpenCV errors from exploding rewards)

FROM THE PAPER:
"The objective for learning the encoder and decoder can be cast as a 
minimax game: the encoder f tries to distinguish x from x̂, while 
the generator g tries to fool f by making x̂ indistinguishable from x."

min_g max_f [ R(f(X)) - R(f(g(Z))) ]

This is like a GAN but with RATE REDUCTION as the discriminator!

CREATED: December 2025
THEORY: Yi Ma et al. "Parsimony and Self-Consistency" (2022)
"""

import numpy as np
import cv2
from collections import deque

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

class MinimaxGameEngine(BaseNode):
    """
    The adversarial learning engine that balances encoder and decoder.
    """
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Minimax Game"
    NODE_COLOR = QtGui.QColor(255, 100, 50)  # Orange - game
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'real_rate': 'signal',          # R(f(X)) - rate of real data
            'fake_rate': 'signal',          # R(f(g(Z))) - rate of reconstructed
            'loop_loss': 'signal',          # From ClosedLoopTranscription
            'learning_rate': 'signal',      # Base learning rate
        }
        
        self.outputs = {
            'display': 'image',
            'f_reward': 'signal',           # Encoder's payoff
            'g_reward': 'signal',           # Decoder's payoff
            'f_learning_rate': 'signal',    # Adjusted LR for encoder
            'g_learning_rate': 'signal',    # Adjusted LR for decoder
            'game_state': 'signal',         # 0=f_winning, 1=equilibrium, 2=g_winning
            'nash_distance': 'signal',      # Distance from equilibrium
            'total_value': 'signal',        # Game value (should → 0)
        }
        
        # === GAME STATE ===
        self.f_reward = 0.0
        self.g_reward = 0.0
        self.game_value = 0.0
        self.nash_distance = 1.0
        
        # === HISTORY ===
        self.f_history = deque(maxlen=500)
        self.g_history = deque(maxlen=500)
        self.value_history = deque(maxlen=500)
        self.nash_history = deque(maxlen=500)
        
        # === ADAPTIVE LEARNING ===
        self.base_lr = 0.01
        self.f_lr_mult = 1.0
        self.g_lr_mult = 1.0
        
        # === MOMENTUM ===
        self.f_momentum = 0.0
        self.g_momentum = 0.0
        self.momentum_decay = 0.9
        
        # === EQUILIBRIUM DETECTION ===
        self.equilibrium_threshold = 0.05
        self.equilibrium_window = 50
        
        # === DISPLAY ===
        self._display = np.zeros((600, 900, 3), dtype=np.uint8)
    
    # --- FIX 1: Safe radius helper to prevent OpenCV crash ---
    def _safe_radius(self, value, base=20, scale=30, lo=10, hi=100):
        try:
            v = float(value)
            if not np.isfinite(v):
                return base
            r = int(base + abs(v) * scale)
        except Exception:
            r = base
        return max(lo, min(hi, r))

    def step(self):
        # Get inputs
        real_rate = self.get_blended_input('real_rate', 'sum')
        fake_rate = self.get_blended_input('fake_rate', 'sum')
        loop_loss = self.get_blended_input('loop_loss', 'sum')
        lr_val = self.get_blended_input('learning_rate', 'sum')
        
        real_rate = float(real_rate) if real_rate else 0.0
        fake_rate = float(fake_rate) if fake_rate else 0.0
        loop_loss = float(loop_loss) if loop_loss else 0.0
        self.base_lr = float(lr_val) if lr_val and lr_val > 0 else 0.01
        
        # === COMPUTE REWARDS ===
        rate_gap = real_rate - fake_rate
        
        self.f_reward = rate_gap
        self.g_reward = -rate_gap
        self.g_reward -= loop_loss * 0.5
        
        self.game_value = self.f_reward + self.g_reward
        self.nash_distance = abs(self.f_reward) + abs(self.g_reward)
        
        # Store history
        self.f_history.append(self.f_reward)
        self.g_history.append(self.g_reward)
        self.value_history.append(self.game_value)
        self.nash_history.append(self.nash_distance)
        
        # === ADAPTIVE LEARNING RATES ===
        if len(self.f_history) > 10:
            recent_f = np.mean(list(self.f_history)[-20:])
            recent_g = np.mean(list(self.g_history)[-20:])
            
            if recent_f > 0.1:
                self.f_lr_mult = max(0.5, self.f_lr_mult * 0.99)
                self.g_lr_mult = min(2.0, self.g_lr_mult * 1.01)
            elif recent_g > 0.1:
                self.g_lr_mult = max(0.5, self.g_lr_mult * 0.99)
                self.f_lr_mult = min(2.0, self.f_lr_mult * 1.01)
            else:
                self.f_lr_mult = 0.99 * self.f_lr_mult + 0.01 * 1.0
                self.g_lr_mult = 0.99 * self.g_lr_mult + 0.01 * 1.0
        
        # === EQUILIBRIUM DETECTION ===
        game_state = 1
        if len(self.nash_history) >= self.equilibrium_window:
            recent_nash = list(self.nash_history)[-self.equilibrium_window:]
            if np.mean(recent_nash) < self.equilibrium_threshold:
                game_state = 1
            elif np.mean(list(self.f_history)[-self.equilibrium_window:]) > 0.05:
                game_state = 0
            elif np.mean(list(self.g_history)[-self.equilibrium_window:]) > 0.05:
                game_state = 2
        
        # === OUTPUTS ===
        self.outputs['f_reward'] = float(self.f_reward)
        self.outputs['g_reward'] = float(self.g_reward)
        self.outputs['f_learning_rate'] = float(self.base_lr * self.f_lr_mult)
        self.outputs['g_learning_rate'] = float(self.base_lr * self.g_lr_mult)
        self.outputs['game_state'] = float(game_state)
        self.outputs['nash_distance'] = float(self.nash_distance)
        self.outputs['total_value'] = float(self.game_value)
        
        self._render_display(game_state)
    
    def _render_display(self, game_state):
        # Added try/except to prevent rendering from crashing the application
        try:
            img = self._display
            img[:] = (20, 20, 25)
            h, w = img.shape[:2]
            
            cv2.putText(img, "MINIMAX GAME: min_g max_f [R(f(X)) - R(f(g(Z)))]", (w//2 - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            self._render_game_state(img, 30, 60, 280, 280)
            self._render_reward_history(img, 330, 60, 350, 200)
            self._render_nash_distance(img, 700, 60, 180, 200)
            self._render_lr_adaptation(img, 30, 360, 280, 150)
            self._render_game_value(img, 330, 280, 350, 150)
            
            states = ["ENCODER WINNING", "EQUILIBRIUM", "DECODER WINNING"]
            colors = [(100, 100, 255), (100, 255, 100), (255, 100, 100)]
            
            state_text = states[int(game_state)]
            state_color = colors[int(game_state)]
            
            cv2.rectangle(img, (0, h-50), (w, h), (30, 30, 40), -1)
            cv2.putText(img, state_text, (w//2 - 100, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            
            cv2.putText(img, f"Nash Distance: {self.nash_distance:.4f}", (30, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            self._display = img
            self.outputs['display'] = self._display # Ensure output is updated
        except Exception:
            # If rendering fails, do not crash the node/application
            pass
    
    def _render_game_state(self, img, x0, y0, width, height):
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "GAME BALANCE", (x0 + 80, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cx = x0 + width // 2
        cy = y0 + height // 2 + 20
        
        balance_angle = np.clip(self.f_reward - self.g_reward, -1, 1) * 0.3
        beam_len = 100
        
        # --- FIX: Clamped circle radius for the fulcrum ---
        cv2.circle(img, (cx, cy), self._safe_radius(0, base=10, scale=0), (150, 150, 150), -1)
        
        x1 = int(cx - beam_len * np.cos(balance_angle))
        y1 = int(cy - beam_len * np.sin(balance_angle))
        x2 = int(cx + beam_len * np.cos(balance_angle))
        y2 = int(cy + beam_len * np.sin(balance_angle))
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 3)
        
        # --- FIX: Clamped circle radius for f ---
        f_size = self._safe_radius(self.f_reward, base=20, scale=30, hi=60)
        cv2.circle(img, (x1, y1), f_size, (255, 100, 100), -1)
        cv2.putText(img, "f", (x1-8, y1+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # --- FIX: Clamped circle radius for g ---
        g_size = self._safe_radius(self.g_reward, base=20, scale=30, hi=60)
        cv2.circle(img, (x2, y2), g_size, (100, 100, 255), -1)
        cv2.putText(img, "g", (x2-8, y2+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(img, f"f: {self.f_reward:.3f}", (x0 + 20, y0 + height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        cv2.putText(img, f"g: {self.g_reward:.3f}", (x0 + width - 80, y0 + height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
    
    def _render_reward_history(self, img, x0, y0, width, height):
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "REWARD HISTORY", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if len(self.f_history) < 2:
            return
        
        f_vals = list(self.f_history)
        g_vals = list(self.g_history)
        
        max_val = max(max(abs(v) for v in f_vals) if f_vals else 0.1, max(abs(v) for v in g_vals) if g_vals else 0.1) + 0.1
        mid_y = y0 + height // 2
        
        for i in range(1, len(f_vals)):
            x1 = x0 + 10 + int((i-1) * (width-20) / len(f_vals))
            x2 = x0 + 10 + int(i * (width-20) / len(f_vals))
            y1 = mid_y - int(f_vals[i-1] / max_val * (height//2 - 20))
            y2 = mid_y - int(f_vals[i] / max_val * (height//2 - 20))
            cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 1)
        
        for i in range(1, len(g_vals)):
            x1 = x0 + 10 + int((i-1) * (width-20) / len(g_vals))
            x2 = x0 + 10 + int(i * (width-20) / len(g_vals))
            y1 = mid_y - int(g_vals[i-1] / max_val * (height//2 - 20))
            y2 = mid_y - int(g_vals[i] / max_val * (height//2 - 20))
            cv2.line(img, (x1, y1), (x2, y2), (100, 100, 255), 1)
        
        cv2.line(img, (x0 + 10, mid_y), (x0 + width - 10, mid_y), (80, 80, 80), 1)
    
    def _render_nash_distance(self, img, x0, y0, width, height):
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "NASH DISTANCE", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        nash_color = (100, 255, 100) if self.nash_distance < 0.1 else \
                     (255, 255, 100) if self.nash_distance < 0.3 else (255, 100, 100)
        
        cv2.putText(img, f"{self.nash_distance:.3f}", (x0 + 30, y0 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, nash_color, 2)
        
        bar_y = y0 + 110
        bar_w = width - 20
        cv2.rectangle(img, (x0+10, bar_y), (x0+10+bar_w, bar_y+20), (50, 50, 50), -1)
        
        fill_w = int((1.0 - min(self.nash_distance, 1.0)) * bar_w)
        cv2.rectangle(img, (x0+10, bar_y), (x0+10+fill_w, bar_y+20), nash_color, -1)
    
    def _render_lr_adaptation(self, img, x0, y0, width, height):
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "LEARNING RATE ADAPTATION", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        bar_w = width - 60
        # f_lr = self.base_lr * self.f_lr_mult # Not used in rendering, just for context
        f_bar_w = int(min(self.f_lr_mult, 2.0) / 2.0 * bar_w)
        
        cv2.putText(img, "f:", (x0 + 10, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        cv2.rectangle(img, (x0+40, y0+40), (x0+40+bar_w, y0+60), (50, 50, 50), -1)
        cv2.rectangle(img, (x0+40, y0+40), (x0+40+f_bar_w, y0+60), (255, 100, 100), -1)
        
        # g_lr = self.base_lr * self.g_lr_mult # Not used in rendering, just for context
        g_bar_w = int(min(self.g_lr_mult, 2.0) / 2.0 * bar_w)
        
        cv2.putText(img, "g:", (x0 + 10, y0 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        cv2.rectangle(img, (x0+40, y0+80), (x0+40+bar_w, y0+100), (50, 50, 50), -1)
        cv2.rectangle(img, (x0+40, y0+80), (x0+40+g_bar_w, y0+100), (100, 100, 255), -1)
    
    def _render_game_value(self, img, x0, y0, width, height):
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "GAME VALUE (-> 0 at equilibrium)", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if len(self.value_history) < 2:
            return
        
        values = list(self.value_history)
        max_val = max(abs(v) for v in values) + 0.1 if values else 0.1
        mid_y = y0 + height // 2 + 10
        
        for i in range(1, len(values)):
            x1 = x0 + 10 + int((i-1) * (width-20) / len(values))
            x2 = x0 + 10 + int(i * (width-20) / len(values))
            y1 = mid_y - int(values[i-1] / max_val * (height//2 - 20))
            y2 = mid_y - int(values[i] / max_val * (height//2 - 20))
            cv2.line(img, (x1, y1), (x2, y2), (255, 200, 100), 2)
        
        cv2.line(img, (x0 + 10, mid_y), (x0 + width - 10, mid_y), (100, 255, 100), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display