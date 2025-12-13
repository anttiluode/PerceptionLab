"""
Token Decoder - Cognitive State Interpreter
=============================================
Takes context vectors from NeuralTransformer and decodes
them into interpretable cognitive states.

THIS IS THE READOUT.

INPUTS:
- context_vector: 64-dim vector from NeuralTransformer
- token_stream: Active tokens for analysis
- theta_phase: For phase-dependent decoding
- sample_trigger: When we're at a box corner

OUTPUTS:
- display: State visualization
- state_vector: Decoded cognitive state (5-dim: attention, memory, motor, visual, internal)
- dominant_state: Index of dominant state (0-4)
- state_history: Rolling history for analysis
- decoded_pattern: Reconstructed pattern from state

COGNITIVE STATES:
0. ATTENTION: Executive focus, task engagement
1. MEMORY: Retrieval, encoding, working memory
2. MOTOR: Planning, execution preparation
3. VISUAL: Sensory processing, external focus
4. INTERNAL: Default mode, introspection, mind-wandering

The decoder learns from the token patterns:
- Frontal tokens → ATTENTION/INTERNAL
- Temporal tokens → MEMORY
- Parietal tokens → MOTOR
- Occipital tokens → VISUAL

Cross-frequency patterns indicate state transitions:
- Theta-Gamma coupling → Memory encoding
- Theta-Beta coupling → Executive control
- Alpha power → Internal/resting state
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

# Cognitive state names and colors
STATE_NAMES = ['ATTENTION', 'MEMORY', 'MOTOR', 'VISUAL', 'INTERNAL']
STATE_COLORS = [
    (255, 100, 100),  # Red - Attention
    (100, 255, 100),  # Green - Memory
    (255, 255, 100),  # Yellow - Motor
    (100, 100, 255),  # Blue - Visual
    (200, 100, 255),  # Purple - Internal
]

class TokenDecoderNode(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Token Decoder"
    NODE_COLOR = QtGui.QColor(255, 200, 50)  # Gold - decoder color
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'context_vector': 'spectrum',
            'token_stream': 'spectrum',
            'theta_phase': 'signal',
            'sample_trigger': 'signal',
        }
        
        self.outputs = {
            'display': 'image',
            'state_vector': 'spectrum',
            'dominant_state': 'signal',
            'state_history': 'spectrum',
            'decoded_pattern': 'image',
        }
        
        # Decoder dimensions
        self.embed_dim = 64
        self.n_states = 5
        
        # Decoder weights (learned from token patterns)
        # Initialize with bias toward expected mappings
        np.random.seed(123)
        self.decoder_weights = np.random.randn(self.n_states, self.embed_dim) * 0.1
        
        # Set up initial biases based on region->state mapping
        # Frontal (tokens 0-4) -> Attention
        self.decoder_weights[0, 0:5] = 0.5
        # Temporal (tokens 5-9) -> Memory
        self.decoder_weights[1, 5:10] = 0.5
        # Parietal (tokens 10-14) -> Motor
        self.decoder_weights[2, 10:15] = 0.5
        # Occipital (tokens 15-19) -> Visual
        self.decoder_weights[3, 15:20] = 0.5
        # Internal = residual
        self.decoder_weights[4, :] = 0.1
        
        # Normalize
        self.decoder_weights /= np.linalg.norm(self.decoder_weights, axis=1, keepdims=True) + 1e-9
        
        # State tracking
        self.current_state = np.zeros(self.n_states)
        self.state_history = deque(maxlen=500)
        self.dominant_state = 0
        
        # Sample accumulator
        self.sample_count = 0
        self.accumulated_context = np.zeros(self.embed_dim)
        
        # Learning
        self.learning_rate = 0.01
        self.use_online_learning = False  # Can enable for adaptation
        
        # Display
        self._display = np.zeros((700, 1000, 3), dtype=np.uint8)
        self._pattern_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    def _sanitize_context(self, data):
        """Ensure context vector is valid 64-dim array"""
        if data is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        if isinstance(data, str):
            return np.zeros(self.embed_dim, dtype=np.float32)
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if not hasattr(data, 'shape'):
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        data = data.flatten().astype(np.float32)
        if len(data) < self.embed_dim:
            padded = np.zeros(self.embed_dim, dtype=np.float32)
            padded[:len(data)] = data
            return padded
        return data[:self.embed_dim]
    
    def _sanitize_tokens(self, data):
        """Convert input to valid token array"""
        if data is None:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, str):
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(data, (list, tuple)):
            try:
                data = np.array(data)
            except:
                return np.zeros((0, 3), dtype=np.float32)
        if not hasattr(data, 'ndim'):
            return np.zeros((0, 3), dtype=np.float32)
        if data.ndim == 1:
            if len(data) == 3:
                return data.reshape(1, 3)
            return np.zeros((0, 3), dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return data.astype(np.float32)
    
    def _decode_state(self, context_vector):
        """Decode context vector to cognitive state probabilities"""
        # Linear projection
        logits = np.matmul(self.decoder_weights, context_vector)
        
        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-9)
        
        return probs
    
    def _tokens_to_features(self, tokens):
        """Extract features from token stream for decoding"""
        features = np.zeros(self.embed_dim)
        
        if len(tokens) == 0:
            return features
        
        for tok in tokens:
            token_id = int(tok[0]) % 20
            amplitude = tok[1]
            phase = tok[2]
            
            # Simple feature: amplitude at token position
            if token_id < self.embed_dim:
                features[token_id] += amplitude
            
            # Cross-token features
            features[20 + (token_id % 20)] += amplitude * np.cos(phase)
            features[40 + (token_id % 20)] += amplitude * np.sin(phase)
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _generate_pattern(self, state_vector):
        """Generate visual pattern representing the decoded state"""
        size = 256
        pattern = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Each state contributes a different pattern
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        for i, (weight, color) in enumerate(zip(state_vector, STATE_COLORS)):
            if weight < 0.1:
                continue
            
            # Different pattern for each state
            if i == 0:  # Attention - concentric circles
                r = np.sqrt(X**2 + Y**2)
                p = np.sin(r * 5) * weight
            elif i == 1:  # Memory - spirals
                theta = np.arctan2(Y, X)
                r = np.sqrt(X**2 + Y**2)
                p = np.sin(theta * 3 + r * 2) * weight
            elif i == 2:  # Motor - directional
                p = np.sin(X * 4 + Y * 2) * weight
            elif i == 3:  # Visual - checker-like
                p = np.sin(X * 5) * np.sin(Y * 5) * weight
            else:  # Internal - smooth
                p = np.exp(-0.5 * (X**2 + Y**2)) * weight
            
            # Add to pattern with state color
            p_norm = ((p + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pattern[:,:,0] = np.clip(pattern[:,:,0].astype(int) + p_norm * color[0] // 255, 0, 255).astype(np.uint8)
            pattern[:,:,1] = np.clip(pattern[:,:,1].astype(int) + p_norm * color[1] // 255, 0, 255).astype(np.uint8)
            pattern[:,:,2] = np.clip(pattern[:,:,2].astype(int) + p_norm * color[2] // 255, 0, 255).astype(np.uint8)
        
        return pattern
    
    def step(self):
        # Get inputs
        raw_context = self.get_blended_input('context_vector', 'mean')
        raw_tokens = self.get_blended_input('token_stream', 'mean')
        theta_phase = self.get_blended_input('theta_phase', 'sum')
        sample_trigger = self.get_blended_input('sample_trigger', 'sum')
        
        if theta_phase is None:
            theta_phase = 0.0
        if sample_trigger is None:
            sample_trigger = 0.0
        
        # Sanitize
        context = self._sanitize_context(raw_context)
        tokens = self._sanitize_tokens(raw_tokens)
        
        # Combine context vector with token features
        token_features = self._tokens_to_features(tokens)
        combined = 0.7 * context + 0.3 * token_features
        
        # Accumulate at sample moments
        if sample_trigger > 0.5:
            self.accumulated_context = 0.8 * self.accumulated_context + 0.2 * combined
            self.sample_count += 1
        
        # Decode state
        self.current_state = self._decode_state(self.accumulated_context)
        self.dominant_state = int(np.argmax(self.current_state))
        
        # Add to history
        self.state_history.append({
            'state': self.current_state.copy(),
            'dominant': self.dominant_state,
            'phase': theta_phase,
            'sample': sample_trigger > 0.5
        })
        
        # Generate pattern
        self._pattern_img = self._generate_pattern(self.current_state)
        
        # Update outputs
        self.outputs['state_vector'] = self.current_state.astype(np.float32)
        self.outputs['dominant_state'] = float(self.dominant_state)
        self.outputs['decoded_pattern'] = self._pattern_img
        
        # History as 2D array
        if len(self.state_history) > 0:
            hist_arr = np.array([h['state'] for h in list(self.state_history)[-100:]])
            self.outputs['state_history'] = hist_arr.astype(np.float32)
        
        # Render
        self._render_display()
    
    def _render_display(self):
        img = self._display
        img[:] = (20, 20, 25)
        h, w = img.shape[:2]
        
        # === LEFT: State bars ===
        self._render_state_bars(img, 30, 30, 200, 400)
        
        # === CENTER: Decoded pattern ===
        pattern_x = 260
        pattern_y = 30
        pattern_size = 350
        
        pattern_resized = cv2.resize(self._pattern_img, (pattern_size, pattern_size))
        img[pattern_y:pattern_y+pattern_size, pattern_x:pattern_x+pattern_size] = pattern_resized
        cv2.rectangle(img, (pattern_x, pattern_y), 
                     (pattern_x+pattern_size, pattern_y+pattern_size), (100, 100, 100), 2)
        
        # Dominant state label
        dom_name = STATE_NAMES[self.dominant_state]
        dom_color = STATE_COLORS[self.dominant_state]
        cv2.putText(img, f"STATE: {dom_name}", (pattern_x + 10, pattern_y + pattern_size + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, dom_color, 2)
        
        # === RIGHT: History ===
        self._render_history(img, 640, 30, 340, 350)
        
        # === BOTTOM: Sample counter and phase ===
        cv2.putText(img, f"Samples: {self.sample_count}", (30, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # State probabilities text
        y_pos = h - 100
        cv2.putText(img, "PROBABILITIES:", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for i, (name, prob) in enumerate(zip(STATE_NAMES, self.current_state)):
            cv2.putText(img, f"{name}: {prob:.3f}", (30, y_pos + 20 + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, STATE_COLORS[i], 1)
        
        self._display = img
    
    def _render_state_bars(self, img, x0, y0, width, height):
        """Render vertical state probability bars"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        bar_width = width // self.n_states - 10
        bar_max_height = height - 60
        
        for i, (name, prob, color) in enumerate(zip(STATE_NAMES, self.current_state, STATE_COLORS)):
            bx = x0 + 5 + i * (bar_width + 10)
            by = y0 + height - 30
            bar_height = int(prob * bar_max_height)
            
            # Bar
            cv2.rectangle(img, (bx, by - bar_height), (bx + bar_width, by), color, -1)
            
            # Label
            cv2.putText(img, name[:3], (bx, by + 15),
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)
            
            # Value
            cv2.putText(img, f"{prob:.2f}", (bx, by - bar_height - 5),
                       cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)
        
        cv2.putText(img, "COGNITIVE STATES", (x0 + 30, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _render_history(self, img, x0, y0, width, height):
        """Render state history as stacked area chart"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        if len(self.state_history) < 2:
            return
        
        history = list(self.state_history)[-width:]
        n_points = len(history)
        
        if n_points < 2:
            return
        
        # Create stacked areas
        states = np.array([h['state'] for h in history])  # (n_points, 5)
        
        # Normalize to sum to 1
        states = states / (states.sum(axis=1, keepdims=True) + 1e-9)
        
        # Draw from bottom up
        for state_idx in range(self.n_states - 1, -1, -1):
            color = STATE_COLORS[state_idx]
            
            # Cumulative sum for stacking
            cumsum = np.sum(states[:, :state_idx+1], axis=1)
            prev_cumsum = np.sum(states[:, :state_idx], axis=1) if state_idx > 0 else np.zeros(n_points)
            
            # Draw filled area
            pts_top = []
            pts_bottom = []
            
            for i in range(n_points):
                px = x0 + int(i * width / n_points)
                py_top = y0 + height - 20 - int(cumsum[i] * (height - 40))
                py_bottom = y0 + height - 20 - int(prev_cumsum[i] * (height - 40))
                
                pts_top.append((px, py_top))
                pts_bottom.append((px, py_bottom))
            
            # Create polygon
            pts = pts_top + pts_bottom[::-1]
            if len(pts) > 2:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(img, [pts_arr], color)
        
        # Sample markers
        for i, h in enumerate(history):
            if h.get('sample', False):
                px = x0 + int(i * width / n_points)
                cv2.line(img, (px, y0 + 20), (px, y0 + height - 20), (255, 255, 255), 1)
        
        cv2.putText(img, "STATE HISTORY", (x0 + 10, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Legend
        legend_y = y0 + height - 15
        for i, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
            lx = x0 + 10 + i * 60
            cv2.rectangle(img, (lx, legend_y), (lx + 10, legend_y + 10), color, -1)
            cv2.putText(img, name[:3], (lx + 15, legend_y + 10),
                       cv2.FONT_HERSHEY_PLAIN, 0.6, (150, 150, 150), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'decoded_pattern':
            return self._pattern_img
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display