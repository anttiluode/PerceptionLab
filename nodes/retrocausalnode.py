"""
Retrocausal Constraint Node
----------------------------
The present is constrained by BOTH past and future.

In block universe view, "now" is a crystal facet held in place
by what came before AND what comes after.

This node buffers states and creates a "squeezed" present
that's influenced bidirectionally.
"""

import numpy as np
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class RetrocausalConstraintNode(BaseNode):
    NODE_CATEGORY = "Temporal"
    NODE_COLOR = QtGui.QColor(180, 120, 200)
    
    def __init__(self, buffer_size=30, constraint_strength=0.7, 
                 backward_weight=0.5, forward_weight=0.5, noise_scale=0.2):
        super().__init__()
        self.node_title = "Retrocausal Constraint"
        
        self.inputs = {
            'state_in': 'image',
            'constraint_strength': 'signal',
            'noise_field': 'image',  # Optional fractal noise
        }
        
        self.outputs = {
            'present_state': 'image',
            'constraint_violation': 'signal',
            'temporal_flow': 'image',
        }
        
        # Configuration
        self.buffer_size = int(buffer_size)
        self.base_constraint_strength = float(constraint_strength)
        self.backward_weight = float(backward_weight)
        self.forward_weight = float(forward_weight)
        self.noise_scale = float(noise_scale)
        
        # State buffers
        self.state_buffer = deque(maxlen=self.buffer_size)
        
        # Outputs
        self.present_state = None
        self.constraint_violation = 0.0
        self.temporal_flow = None
        
        # For visualization
        self.past_state = None
        self.future_state = None
    
    def step(self):
        # Get inputs
        state_in = self.get_blended_input('state_in', 'first')
        constraint_sig = self.get_blended_input('constraint_strength', 'sum')
        noise_field = self.get_blended_input('noise_field', 'first')
        
        # Use signal or default
        constraint_strength = constraint_sig if constraint_sig is not None else self.base_constraint_strength
        constraint_strength = np.clip(constraint_strength, 0.0, 1.0)
        
        if state_in is None:
            if self.present_state is not None:
                self.present_state *= 0.95  # Fade out
            return
        
        # Ensure consistent format
        if state_in.ndim == 3:
            state_in = cv2.cvtColor(state_in, cv2.COLOR_RGB2GRAY) if state_in.shape[2] == 3 else state_in[:,:,0]
        
        if state_in.dtype != np.float32:
            state_in = state_in.astype(np.float32)
        
        if state_in.max() > 1.0:
            state_in = state_in / 255.0
        
        # Add to buffer
        self.state_buffer.append(state_in.copy())
        
        # Need at least 3 states to do retrocausality
        if len(self.state_buffer) < 3:
            self.present_state = state_in
            self.constraint_violation = 0.0
            return
        
        # Get past, present, and future
        past_idx = 0
        present_idx = len(self.state_buffer) // 2
        future_idx = len(self.state_buffer) - 1
        
        self.past_state = self.state_buffer[past_idx]
        natural_present = self.state_buffer[present_idx]
        self.future_state = self.state_buffer[future_idx]
        
        # Calculate constrained present
        # It's pulled by both past and future
        constrained = (self.past_state * self.backward_weight + 
                      self.future_state * self.forward_weight)
        
        # Normalize weights
        total_weight = self.backward_weight + self.forward_weight
        if total_weight > 0:
            constrained = constrained / total_weight
        
        # Add noise based on constraint strength
        # Low constraint = more freedom = more noise
        freedom = 1.0 - constraint_strength
        
        if noise_field is not None:
            # Use provided noise
            noise = noise_field
            if noise.shape != constrained.shape:
                noise = cv2.resize(noise, (constrained.shape[1], constrained.shape[0]))
            if noise.ndim == 3:
                noise = cv2.cvtColor(noise, cv2.COLOR_RGB2GRAY) if noise.shape[2] == 3 else noise[:,:,0]
        else:
            # Generate noise
            noise = np.random.randn(*constrained.shape).astype(np.float32) * 0.1
        
        self.present_state = constrained + (noise * freedom * self.noise_scale)
        self.present_state = np.clip(self.present_state, 0, 1)
        
        # Calculate violation: how different is constrained from natural?
        self.constraint_violation = np.mean(np.abs(self.present_state - natural_present))
        
        # Calculate temporal flow field (simplified)
        # Flow from past to present
        flow_backward = self.present_state - self.past_state
        # Flow from present to future
        flow_forward = self.future_state - self.present_state
        
        # Combined flow shows the "pressure"
        self.temporal_flow = (flow_backward + flow_forward) / 2.0
    
    def get_output(self, port_name):
        if port_name == 'present_state':
            return self.present_state
        elif port_name == 'constraint_violation':
            return self.constraint_violation
        elif port_name == 'temporal_flow':
            return self.temporal_flow
        return None
    
    def get_display_image(self):
        w, h = 384, 256
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        panel_w = w // 3
        
        # Past | Present | Future
        if self.past_state is not None:
            past_u8 = (np.clip(self.past_state, 0, 1) * 255).astype(np.uint8)
            past_color = cv2.applyColorMap(past_u8, cv2.COLORMAP_TWILIGHT)
            past_resized = cv2.resize(past_color, (panel_w, h//2))
            display[:h//2, :panel_w] = past_resized
        
        if self.present_state is not None:
            present_u8 = (np.clip(self.present_state, 0, 1) * 255).astype(np.uint8)
            present_color = cv2.applyColorMap(present_u8, cv2.COLORMAP_VIRIDIS)
            present_resized = cv2.resize(present_color, (panel_w, h//2))
            display[:h//2, panel_w:2*panel_w] = present_resized
        
        if self.future_state is not None:
            future_u8 = (np.clip(self.future_state, 0, 1) * 255).astype(np.uint8)
            future_color = cv2.applyColorMap(future_u8, cv2.COLORMAP_PLASMA)
            future_resized = cv2.resize(future_color, (panel_w, h//2))
            display[:h//2, 2*panel_w:] = future_resized
        
        # Bottom: Temporal flow
        if self.temporal_flow is not None:
            flow_norm = self.temporal_flow - self.temporal_flow.min()
            flow_max = flow_norm.max()
            if flow_max > 0:
                flow_norm = flow_norm / flow_max
            
            flow_u8 = (np.clip(flow_norm, 0, 1) * 255).astype(np.uint8)
            flow_color = cv2.applyColorMap(flow_u8, cv2.COLORMAP_JET)
            flow_resized = cv2.resize(flow_color, (w, h//2))
            display[h//2:, :] = flow_resized
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'PAST', (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display, 'PRESENT', (panel_w + 10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display, 'FUTURE', (2*panel_w + 10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display, 'TEMPORAL FLOW', (10, h//2 + 20), font, 0.5, (255, 255, 255), 1)
        
        # Stats
        cv2.putText(display, f'Violation: {self.constraint_violation:.4f}', 
                   (10, h - 10), font, 0.4, (255, 255, 0), 1)
        cv2.putText(display, f'Buffer: {len(self.state_buffer)}/{self.buffer_size}', 
                   (w - 150, h - 10), font, 0.4, (255, 255, 255), 1)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Buffer Size", "buffer_size", self.buffer_size, None),
            ("Constraint Strength", "base_constraint_strength", self.base_constraint_strength, None),
            ("Backward Weight", "backward_weight", self.backward_weight, None),
            ("Forward Weight", "forward_weight", self.forward_weight, None),
            ("Noise Scale", "noise_scale", self.noise_scale, None),
        ]