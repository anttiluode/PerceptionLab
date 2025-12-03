"""
Social Topology Node (Evolved Intersection)
===========================================
Analyzes the interaction between two Quantum Agents (A and B).

EVOLVED FEATURES:
- Conflict Metric: Measures overlap weighted by importance (Center = High Value).
- Territory Map: Visualizes the "Social Molecule".
- Dominance: Tracks which agent controls more of the protected subspace.

This node creates the feedback loop for Social Physics experiments.
"""

import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class SocialTopologyNode(BaseNode):
    """
    Advanced intersection analysis for Social Physics.
    Replaces standard AddressIntersectionNode.
    """
    NODE_CATEGORY = "Intelligence"
    NODE_TITLE = "Social Topology"
    NODE_COLOR = QtGui.QColor(160, 80, 180)  # Royal Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'address_a': 'image',        # Agent A (The Diamond?)
            'address_b': 'image',        # Agent B (The Challenger?)
            'protection_map': 'image'    # Optional: The landscape value map
        }
        
        self.outputs = {
            'overlap': 'signal',         # Standard Jaccard Index
            'conflict': 'signal',        # Weighted Overlap (Center fighting)
            'dominance': 'signal',       # A vs B balance (-1=B wins, 1=A wins)
            'social_map': 'image'        # Visualizes the interaction
        }
        
        self.size = 128
        center = self.size // 2
        
        # Pre-compute Center Value Map (The "Prize")
        # Gaussian hill at the center (k=0)
        y, x = np.ogrid[:self.size, :self.size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        self.value_map = np.exp(-0.5 * (r / (self.size * 0.15))**2).astype(np.float32)
        
        # State
        self.overlap_val = 0.0
        self.conflict_val = 0.0
        self.dominance_val = 0.0
        self.map_vis = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def step(self):
        # 1. Get Inputs
        A = self.get_blended_input('address_a', 'first')
        B = self.get_blended_input('address_b', 'first')
        prot = self.get_blended_input('protection_map', 'first')
        
        if A is None or B is None: return
        
        # Normalize inputs (0-1)
        A = self.normalize(cv2.resize(A.astype(np.float32), (self.size, self.size)))
        B = self.normalize(cv2.resize(B.astype(np.float32), (self.size, self.size)))
        
        # Use external protection map if provided, else internal value map
        weights = self.value_map
        if prot is not None:
            prot = cv2.resize(prot.astype(np.float32), (self.size, self.size))
            weights = self.normalize(prot)

        # 2. Compute Social Physics
        
        # Intersection & Union
        intersection = A * B
        union = np.maximum(A, B)
        
        # Overlap (Communication Capacity)
        # Simple Jaccard: Intersection / Union
        sum_inter = np.sum(intersection)
        sum_union = np.sum(union) + 1e-9
        self.overlap_val = float(sum_inter / sum_union)
        
        # Conflict (Resource War)
        # Intersection weighted by Value (Fighting for the Center)
        weighted_inter = np.sum(intersection * weights)
        total_value = np.sum(weights) + 1e-9
        self.conflict_val = float(weighted_inter / total_value)
        
        # Dominance (Power Balance)
        # (Size A - Size B) / Size Union
        sum_A = np.sum(A)
        sum_B = np.sum(B)
        self.dominance_val = float((sum_A - sum_B) / sum_union)
        
        # 3. Visualize "The Molecule"
        # Blue = Agent A
        # Red = Agent B
        # Green = The Value Field (The Prize)
        # White/Purple = The Intersection
        
        vis = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        vis[:,:,0] = A * 0.8  # Blue Channel (A)
        vis[:,:,2] = B * 0.8  # Red Channel (B)
        
        # Green Channel shows the "Prize" (Value Map) 
        # but dimmed where Agents exist
        vis[:,:,1] = weights * 0.5
        
        # Boost intersection (White/Purple flash)
        vis[:,:,0] += intersection * 0.5
        vis[:,:,1] += intersection * 0.5
        vis[:,:,2] += intersection * 0.5
        
        self.map_vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)

    def normalize(self, arr):
        mx = np.max(arr)
        if mx > 1e-9: return arr / mx
        return arr

    def get_output(self, name):
        if name == 'overlap': return self.overlap_val
        if name == 'conflict': return self.conflict_val
        if name == 'dominance': return self.dominance_val
        if name == 'social_map': return self.map_vis
        return 0.0

    def get_display_image(self):
        # Create HUD
        h, w = self.size, self.size
        display = self.map_vis.copy()
        
        # Metrics HUD
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw Conflict Bar (Red, Top)
        bar_w = int(self.conflict_val * w)
        cv2.rectangle(display, (0, 0), (bar_w, 5), (0, 0, 255), -1)
        
        # Draw Overlap Bar (White, Bottom)
        bar_w2 = int(self.overlap_val * w)
        cv2.rectangle(display, (0, h-5), (bar_w2, h), (255, 255, 255), -1)
        
        cv2.putText(display, f"Cnflct: {self.conflict_val:.2f}", (5, 20), font, 0.35, (200,200,255), 1)
        cv2.putText(display, f"Ovrlp: {self.overlap_val:.2f}", (5, h-10), font, 0.35, (255,255,255), 1)
        
        # Dominance Indicator
        # If > 0, A wins (Blue text). If < 0, B wins (Red text).
        dom_color = (255, 100, 100) if self.dominance_val < 0 else (100, 100, 255)
        dom_text = "A > B" if self.dominance_val > 0.1 else "B > A" if self.dominance_val < -0.1 else "A = B"
        cv2.putText(display, dom_text, (w - 40, h//2), font, 0.35, dom_color, 1)
        
        return __main__.numpy_to_qimage(display)