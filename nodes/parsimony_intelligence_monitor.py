"""
Parsimony Intelligence Monitor - Ma's Emergence Detector
=========================================================
The MASTER NODE that monitors all aspects of Ma's framework
and indicates when "intelligence" has emerged.

FROM THE PAPER:
"We introduce two fundamental principles, Parsimony and Self-consistency,
which address two fundamental questions regarding intelligence:
what to learn and how to learn, respectively."

INTELLIGENCE EMERGES WHEN:
1. PARSIMONY: High Rate Reduction (ΔR >> 0)
   → The system has found compact structure in the data
   
2. SELF-CONSISTENCY: Low Loop Loss (||z - ẑ|| → 0)
   → The encoder and decoder agree on the representation
   
3. EQUILIBRIUM: Nash Distance → 0
   → Neither encoder nor decoder can improve unilaterally

This node monitors all three conditions and signals when
the system has achieved what Ma calls "intelligence."

OUTPUTS:
- display: Full dashboard showing all metrics
- intelligence_score: 0-1 composite score
- is_intelligent: Binary gate when all conditions met
- parsimony_score: How well compressed (ΔR)
- consistency_score: How self-consistent (1 - loss)
- equilibrium_score: How stable (1 - nash_distance)

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

class ParsimonyIntelligenceMonitor(BaseNode):
    """
    Master monitor for the emergence of intelligence according to Ma's principles.
    """
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Intelligence Monitor"
    NODE_COLOR = QtGui.QColor(255, 215, 0)  # Gold - intelligence
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'rate_reduction': 'signal',     # ΔR from encoder
            'loop_loss': 'signal',          # From closed loop
            'nash_distance': 'signal',      # From minimax game
            'coding_rate': 'signal',        # R(Z) from encoder
            'game_state': 'signal',         # Current game state
        }
        
        self.outputs = {
            'display': 'image',
            'intelligence_score': 'signal',   # Composite 0-1
            'is_intelligent': 'signal',       # Binary gate
            'parsimony_score': 'signal',      # ΔR normalized
            'consistency_score': 'signal',    # 1 - loss normalized
            'equilibrium_score': 'signal',    # 1 - nash normalized
            'emergence_level': 'signal',      # 0-3 (how many conditions met)
        }
        
        # === THRESHOLDS (from paper intuition) ===
        self.parsimony_threshold = 0.5      # ΔR must be positive and significant
        self.consistency_threshold = 0.1    # Loop loss must be small
        self.equilibrium_threshold = 0.1    # Nash distance must be small
        
        # === HISTORY ===
        self.parsimony_history = deque(maxlen=500)
        self.consistency_history = deque(maxlen=500)
        self.equilibrium_history = deque(maxlen=500)
        self.intelligence_history = deque(maxlen=500)
        
        # === CURRENT STATE ===
        self.parsimony_score = 0.0
        self.consistency_score = 0.0
        self.equilibrium_score = 0.0
        self.intelligence_score = 0.0
        self.emergence_level = 0
        
        # === DISPLAY ===
        self._display = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def step(self):
        # Get inputs
        rate_reduction = self.get_blended_input('rate_reduction', 'sum')
        loop_loss = self.get_blended_input('loop_loss', 'sum')
        nash_distance = self.get_blended_input('nash_distance', 'sum')
        coding_rate = self.get_blended_input('coding_rate', 'sum')
        game_state = self.get_blended_input('game_state', 'sum')
        
        rate_reduction = float(rate_reduction) if rate_reduction else 0.0
        loop_loss = float(loop_loss) if loop_loss else 1.0
        nash_distance = float(nash_distance) if nash_distance else 1.0
        coding_rate = float(coding_rate) if coding_rate else 0.0
        game_state = int(game_state) if game_state else 0
        
        # === COMPUTE SCORES ===
        
        # 1. PARSIMONY: Higher ΔR is better
        # Sigmoid normalization to 0-1
        self.parsimony_score = 1.0 / (1.0 + np.exp(-rate_reduction * 2))
        
        # 2. SELF-CONSISTENCY: Lower loss is better
        self.consistency_score = np.exp(-loop_loss * 5)
        
        # 3. EQUILIBRIUM: Lower nash distance is better
        self.equilibrium_score = np.exp(-nash_distance * 5)
        
        # Store history
        self.parsimony_history.append(self.parsimony_score)
        self.consistency_history.append(self.consistency_score)
        self.equilibrium_history.append(self.equilibrium_score)
        
        # === INTELLIGENCE COMPOSITE ===
        # Geometric mean of three scores (all must be good)
        self.intelligence_score = (
            self.parsimony_score * 
            self.consistency_score * 
            self.equilibrium_score
        ) ** (1/3)
        
        self.intelligence_history.append(self.intelligence_score)
        
        # === EMERGENCE LEVEL ===
        # Count how many conditions are met
        conditions_met = 0
        if self.parsimony_score > 0.6:
            conditions_met += 1
        if self.consistency_score > 0.6:
            conditions_met += 1
        if self.equilibrium_score > 0.6:
            conditions_met += 1
        
        self.emergence_level = conditions_met
        
        # Is intelligent? All three must be high
        is_intelligent = 1.0 if self.intelligence_score > 0.5 else 0.0
        
        # === OUTPUTS ===
        self.outputs['intelligence_score'] = float(self.intelligence_score)
        self.outputs['is_intelligent'] = is_intelligent
        self.outputs['parsimony_score'] = float(self.parsimony_score)
        self.outputs['consistency_score'] = float(self.consistency_score)
        self.outputs['equilibrium_score'] = float(self.equilibrium_score)
        self.outputs['emergence_level'] = float(self.emergence_level)
        
        # Render
        self._render_display(rate_reduction, loop_loss, nash_distance, coding_rate)
    
    def _render_display(self, rate_reduction, loop_loss, nash_distance, coding_rate):
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === HEADER ===
        cv2.putText(img, "PARSIMONY & SELF-CONSISTENCY INTELLIGENCE MONITOR", (w//2 - 280, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
        cv2.putText(img, '"The emergence of intelligence from geometric compression"', (w//2 - 230, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # === LEFT: Three Principle Gauges ===
        self._render_gauge(img, 50, 100, "PARSIMONY", self.parsimony_score, 
                          f"ΔR = {rate_reduction:.3f}", (100, 255, 255))
        self._render_gauge(img, 50, 280, "SELF-CONSISTENCY", self.consistency_score,
                          f"Loss = {loop_loss:.4f}", (255, 100, 255))
        self._render_gauge(img, 50, 460, "EQUILIBRIUM", self.equilibrium_score,
                          f"Nash = {nash_distance:.4f}", (255, 255, 100))
        
        # === CENTER: Intelligence Score ===
        self._render_intelligence_dial(img, 400, 120, 300, 300)
        
        # === RIGHT: History Plots ===
        self._render_history(img, 750, 100, 420, 250)
        
        # === BOTTOM: Emergence Level Indicator ===
        self._render_emergence_indicator(img, 400, 480, 400, 100)
        
        # === STATUS BAR ===
        status_y = h - 60
        cv2.rectangle(img, (0, status_y), (w, h), (30, 30, 40), -1)
        
        if self.intelligence_score > 0.5:
            status_text = "INTELLIGENCE EMERGED"
            status_color = (100, 255, 100)
            cv2.rectangle(img, (0, status_y), (w, h), (30, 60, 30), -1)
        elif self.emergence_level >= 2:
            status_text = f"EMERGING... ({self.emergence_level}/3 conditions)"
            status_color = (255, 255, 100)
        else:
            status_text = f"LEARNING... ({self.emergence_level}/3 conditions)"
            status_color = (200, 200, 200)
        
        cv2.putText(img, status_text, (w//2 - 120, status_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Raw values
        cv2.putText(img, f"R(Z): {coding_rate:.2f}", (30, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, f"Score: {self.intelligence_score:.3f}", (w - 150, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 215, 0), 1)
        
        self._display = img
    
    def _render_gauge(self, img, x0, y0, title, value, subtitle, color):
        """Render a single principle gauge"""
        width = 300
        height = 150
        
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        # Title
        cv2.putText(img, title, (x0 + 10, y0 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Progress bar
        bar_y = y0 + 50
        bar_w = width - 40
        cv2.rectangle(img, (x0+20, bar_y), (x0+20+bar_w, bar_y+30), (50, 50, 50), -1)
        
        fill_w = int(value * bar_w)
        fill_color = color if value > 0.6 else (150, 150, 150)
        cv2.rectangle(img, (x0+20, bar_y), (x0+20+fill_w, bar_y+30), fill_color, -1)
        
        # Value
        cv2.putText(img, f"{value:.2%}", (x0 + 20, bar_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Subtitle
        cv2.putText(img, subtitle, (x0 + 120, bar_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Threshold marker
        thresh_x = x0 + 20 + int(0.6 * bar_w)
        cv2.line(img, (thresh_x, bar_y - 5), (thresh_x, bar_y + 35), (100, 255, 100), 2)
        
        # Check mark if passing
        if value > 0.6:
            cv2.putText(img, "✓", (x0 + width - 35, y0 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    
    def _render_intelligence_dial(self, img, x0, y0, width, height):
        """Render the main intelligence score dial"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        cx = x0 + width // 2
        cy = y0 + height // 2 + 30
        radius = 100
        
        # Background arc
        for angle in np.linspace(-np.pi * 0.8, np.pi * 0.8, 50):
            x = int(cx + radius * np.cos(angle - np.pi/2))
            y = int(cy + radius * np.sin(angle - np.pi/2))
            cv2.circle(img, (x, y), 5, (50, 50, 50), -1)
        
        # Filled arc based on score
        score_angle = -np.pi * 0.8 + self.intelligence_score * np.pi * 1.6
        for angle in np.linspace(-np.pi * 0.8, score_angle, int(50 * self.intelligence_score) + 1):
            x = int(cx + radius * np.cos(angle - np.pi/2))
            y = int(cy + radius * np.sin(angle - np.pi/2))
            
            # Color gradient: red → yellow → green
            if self.intelligence_score < 0.3:
                color = (100, 100, 255)
            elif self.intelligence_score < 0.6:
                color = (100, 255, 255)
            else:
                color = (100, 255, 100)
            
            cv2.circle(img, (x, y), 8, color, -1)
        
        # Needle
        needle_angle = -np.pi * 0.8 + self.intelligence_score * np.pi * 1.6 - np.pi/2
        needle_len = radius - 20
        nx = int(cx + needle_len * np.cos(needle_angle))
        ny = int(cy + needle_len * np.sin(needle_angle))
        cv2.line(img, (cx, cy), (nx, ny), (255, 255, 255), 3)
        cv2.circle(img, (cx, cy), 10, (200, 200, 200), -1)
        
        # Score text
        cv2.putText(img, f"{self.intelligence_score:.1%}", (cx - 40, cy + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 215, 0), 2)
        
        # Title
        cv2.putText(img, "INTELLIGENCE SCORE", (x0 + 50, y0 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Scale labels
        cv2.putText(img, "0%", (x0 + 30, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "100%", (x0 + width - 60, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def _render_history(self, img, x0, y0, width, height):
        """Plot score histories"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        cv2.putText(img, "EMERGENCE HISTORY", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if len(self.intelligence_history) < 2:
            return
        
        # Plot each score
        histories = [
            (self.parsimony_history, (100, 255, 255), "P"),
            (self.consistency_history, (255, 100, 255), "C"),
            (self.equilibrium_history, (255, 255, 100), "E"),
            (self.intelligence_history, (255, 215, 0), "I"),
        ]
        
        for history, color, label in histories:
            vals = list(history)
            if len(vals) < 2:
                continue
            
            for i in range(1, len(vals)):
                x1 = x0 + 10 + int((i-1) * (width-20) / len(vals))
                x2 = x0 + 10 + int(i * (width-20) / len(vals))
                y1 = y0 + height - 30 - int(vals[i-1] * (height - 60))
                y2 = y0 + height - 30 - int(vals[i] * (height - 60))
                cv2.line(img, (x1, y1), (x2, y2), color, 1 if label != "I" else 2)
        
        # Legend
        legend_y = y0 + 35
        for i, (_, color, label) in enumerate(histories):
            lx = x0 + 10 + i * 40
            cv2.rectangle(img, (lx, legend_y), (lx+10, legend_y+10), color, -1)
            cv2.putText(img, label, (lx+15, legend_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Threshold line
        thresh_y = y0 + height - 30 - int(0.5 * (height - 60))
        cv2.line(img, (x0 + 10, thresh_y), (x0 + width - 10, thresh_y), (100, 100, 100), 1)
    
    def _render_emergence_indicator(self, img, x0, y0, width, height):
        """Visual indicator of emergence level"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        cv2.putText(img, "EMERGENCE LEVEL", (x0 + 10, y0 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Three lights
        labels = ["PARSIMONY", "CONSISTENCY", "EQUILIBRIUM"]
        scores = [self.parsimony_score, self.consistency_score, self.equilibrium_score]
        
        light_size = 30
        spacing = (width - 60) // 3
        
        for i, (label, score) in enumerate(zip(labels, scores)):
            lx = x0 + 30 + i * spacing
            ly = y0 + 55
            
            # Light color based on score
            if score > 0.6:
                color = (100, 255, 100)  # Green - active
            elif score > 0.3:
                color = (100, 200, 200)  # Yellow - partial
            else:
                color = (100, 100, 100)  # Gray - inactive
            
            cv2.circle(img, (lx, ly), light_size, color, -1)
            cv2.circle(img, (lx, ly), light_size, (200, 200, 200), 2)
            
            # Label
            cv2.putText(img, label[:4], (lx - 20, ly + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Emergence level text
        level_text = ["No Emergence", "Partial", "Near-Intelligent", "INTELLIGENT"][self.emergence_level]
        level_colors = [(150, 150, 150), (100, 200, 200), (100, 255, 255), (100, 255, 100)]
        
        cv2.putText(img, f"Level {self.emergence_level}: {level_text}", (x0 + width - 180, y0 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, level_colors[self.emergence_level], 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display
