"""
Dendritic Attention Node - Adaptive attention system using dendritic growth principles
Place this file in the 'nodes' folder
Requires: pip install scipy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import time

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: DendriticAttentionNode requires 'scipy'.")


def box_count(data, box_size):
    """Count boxes containing any part of the pattern."""
    S = np.add.reduceat(
        np.add.reduceat(data, np.arange(0, data.shape[0], box_size), axis=0),
        np.arange(0, data.shape[1], box_size), axis=1)
    return np.sum(S > 0)


def fractal_dimension(Z, min_box=2, max_box=None, step=2):
    """Compute fractal dimension using box-counting method."""
    Z = Z > Z.mean()
    
    if max_box is None:
        max_box = min(Z.shape) // 4
    
    max_box = min(max_box, min(Z.shape) // 2)
    min_box = max(2, min_box)
    
    if max_box <= min_box:
        return 1.0
        
    sizes = np.arange(min_box, max_box, step)
    if len(sizes) < 2:
        sizes = np.array([min_box, max_box-1])
        
    counts = []
    for size in sizes:
        count = box_count(Z, size)
        counts.append(max(1, count))

    try:
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
        return -slope
    except:
        return 1.0


class DendriticAttentionNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 100, 200)  # Purple for neural
    
    def __init__(self, n_dendrites=1000, learning_rate=0.05):
        super().__init__()
        self.node_title = "Dendritic Attention"
        
        self.inputs = {
            'image_in': 'image',
            'reset': 'signal'
        }
        
        self.outputs = {
            'attention_field': 'image',
            'visualization': 'image',
            'match_score': 'signal',
            'stability': 'signal',
            'attention_width': 'signal',
            'exploration': 'signal',
            'fractal_dim': 'signal',
            'adj_0': 'signal',  # Frequency adjustments for external control
            'adj_1': 'signal',
            'adj_2': 'signal',
            'adj_3': 'signal',
            'adj_4': 'signal'
        }
        
        if not SCIPY_AVAILABLE:
            self.node_title = "Dendritic (No SciPy!)"
            return
        
        # Parameters
        self.input_size = (64, 64)
        self.n_dendrites = int(n_dendrites)
        self.learning_rate = float(learning_rate)
        
        # Initialize dendrites
        self.positions = np.random.rand(self.n_dendrites, 2) * np.array(self.input_size)
        self.directions = self._normalize(np.random.randn(self.n_dendrites, 2))
        self.strengths = np.ones(self.n_dendrites) * 0.5
        
        # Attention state
        self.attention_field = np.ones(self.input_size)
        self.expected_pattern = None
        self.memory_strength = 0.0
        
        # Metrics
        self.attention_width = 0.5
        self.stability_measure = 0.5
        self.exploration_rate = 0.5
        self.fractal_dim_value = 1.5
        
        # History
        self.activity_history = []
        self.match_history = []
        self.reset_time = time.time()
        
        # Response vectors for frequency adjustments
        self.response_vectors = np.random.randn(4, 5) * 0.1
        self.activity_vector = np.zeros(4)
        
        # Output buffers
        self.vis_output = np.zeros((*self.input_size, 3), dtype=np.uint8)
        
    def _normalize(self, vectors):
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    def _resize_input(self, input_data):
        """Resize input to internal resolution."""
        if input_data.shape != self.input_size:
            return cv2.resize(input_data, (self.input_size[1], self.input_size[0]), 
                            interpolation=cv2.INTER_AREA)
        return input_data
    
    def _compute_match(self, input_data, expected):
        """Calculate pattern match score."""
        if input_data.shape != expected.shape:
            return 0.0
        
        input_flat = input_data.flatten()
        expected_flat = expected.flatten()
        
        input_centered = input_flat - np.mean(input_flat)
        expected_centered = expected_flat - np.mean(expected_flat)
        
        numerator = np.dot(input_centered, expected_centered)
        denominator = np.sqrt(np.sum(input_centered**2) * np.sum(expected_centered**2))
        
        if denominator < 1e-8:
            return 0.0
            
        correlation = numerator / denominator
        return max(0, (correlation + 1) / 2)
    
    def _dilate_attention(self):
        """Update attention field (iris effect)."""
        x, y = np.meshgrid(
            np.linspace(-1, 1, self.input_size[1]),
            np.linspace(-1, 1, self.input_size[0])
        )
        
        distance = np.sqrt(x**2 + y**2)
        sigma = 0.2 + self.attention_width * 1.0
        self.attention_field = np.exp(-(distance**2 / (2.0 * sigma**2)))
    
    def _grow_dendrites(self, input_data):
        """Grow dendrites toward areas of high activity."""
        for i in range(self.n_dendrites):
            x, y = self.positions[i].astype(int) % self.input_size
            x = min(x, self.input_size[0] - 1)
            y = min(y, self.input_size[1] - 1)
            
            activity = input_data[x, y]
            
            # Update strength
            self.strengths[i] = 0.95 * self.strengths[i] + 0.05 * activity
            
            # Grow strong dendrites
            if self.strengths[i] > 0.3:
                # Calculate gradient
                grad_x, grad_y = 0, 0
                if x > 0 and x < self.input_size[0] - 1:
                    grad_x = input_data[x+1, y] - input_data[x-1, y]
                if y > 0 and y < self.input_size[1] - 1:
                    grad_y = input_data[x, y+1] - input_data[x, y-1]
                
                # Update direction
                if abs(grad_x) > 0.01 or abs(grad_y) > 0.01:
                    gradient = np.array([grad_x, grad_y])
                    gradient_norm = np.linalg.norm(gradient)
                    if gradient_norm > 0:
                        gradient = gradient / gradient_norm
                        self.directions[i] = 0.8 * self.directions[i] + 0.2 * gradient
                        self.directions[i] = self.directions[i] / (np.linalg.norm(self.directions[i]) + 1e-8)
                
                # Move dendrite
                growth_rate = self.strengths[i] * 0.1
                self.positions[i] += self.directions[i] * growth_rate
                self.positions[i] = self.positions[i] % np.array(self.input_size)
    
    def _extract_features(self, input_data):
        """Extract features for response calculation."""
        total_activity = np.mean(input_data * self.attention_field)
        
        h, w = self.input_size
        top_left = np.mean(input_data[:h//2, :w//2])
        top_right = np.mean(input_data[:h//2, w//2:])
        bottom_left = np.mean(input_data[h//2:, :w//2])
        bottom_right = np.mean(input_data[h//2:, w//2:])
        
        self.activity_vector = np.array([
            total_activity,
            top_left - bottom_right,
            top_right - bottom_left,
            self.stability_measure
        ])
        
        self.activity_history.append(total_activity)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
    
    def _get_frequency_adjustments(self):
        """Calculate adjustments for external control."""
        raw_adjustments = np.dot(self.activity_vector, self.response_vectors)
        scaled = raw_adjustments * (0.5 + self.exploration_rate)
        
        # Add exploration oscillation
        time_factor = np.sin(time.time() * np.pi * 0.1)
        exploration_wave = np.sin(np.linspace(0, 2*np.pi, 5) + time_factor)
        scaled += exploration_wave * self.exploration_rate * 0.2
        
        # Add instability noise
        if self.stability_measure < 0.5:
            scaled += np.random.randn(5) * (0.5 - self.stability_measure) * 0.3
            
        return scaled
    
    def _generate_visualization(self):
        """Create RGB visualization."""
        vis_img = np.zeros((*self.input_size, 3), dtype=np.float32)
        
        # Blue: attention field
        vis_img[:, :, 2] = self.attention_field
        
        # Green: active dendrites
        for i in range(self.n_dendrites):
            if self.strengths[i] > 0.2:
                x, y = self.positions[i].astype(int) % self.input_size
                try:
                    vis_img[x, y, 1] = min(1.0, vis_img[x, y, 1] + self.strengths[i])
                except IndexError:
                    pass
        
        # Red: expected pattern
        if self.expected_pattern is not None:
            vis_img[:, :, 0] = self.expected_pattern * 0.7
        
        return (vis_img * 255).astype(np.uint8)
    
    def step(self):
        if not SCIPY_AVAILABLE:
            return
        
        # Check for reset
        reset_sig = self.get_blended_input('reset', 'sum')
        if reset_sig is not None and reset_sig > 0.5:
            self._reset()
            return
        
        # Get input
        input_img = self.get_blended_input('image_in', 'mean')
        if input_img is None:
            return
        
        # Resize to internal resolution
        input_data = self._resize_input(input_img)
        
        # Compute match with expected pattern
        if self.expected_pattern is not None:
            match_score = self._compute_match(input_data, self.expected_pattern)
            self.match_history.append(match_score)
            if len(self.match_history) > 50:
                self.match_history.pop(0)
        else:
            self.expected_pattern = input_data.copy()
            self.memory_strength = 0.1
            match_score = 1.0
            self.match_history = [1.0]
        
        # Update stability
        if len(self.match_history) > 5:
            match_variance = np.var(self.match_history[-5:])
            self.stability_measure = 1.0 - min(1.0, match_variance * 10)
        
        # Update attention width (iris effect)
        target_width = 0.3 if match_score > 0.7 else 0.8
        self.attention_width = 0.95 * self.attention_width + 0.05 * target_width
        
        # Update attention field
        self._dilate_attention()
        
        # Grow dendrites
        self._grow_dendrites(input_data)
        
        # Extract features
        self._extract_features(input_data)
        
        # Update expected pattern
        if self.expected_pattern is not None:
            self.expected_pattern = (0.9 * self.expected_pattern + 
                                   0.1 * input_data * self.attention_field)
        
        # Calculate fractal dimension
        vis_img = self._generate_visualization()
        red_channel = vis_img[:, :, 0]
        self.fractal_dim_value = fractal_dimension(red_channel)
        
        # Update exploration rate
        runtime = time.time() - self.reset_time
        base_exploration = max(0.1, 1.0 - min(1.0, runtime / 60.0))
        stability_factor = 1.0 - self.stability_measure
        self.exploration_rate = 0.7 * self.exploration_rate + 0.3 * (base_exploration + 0.5 * stability_factor)
        
        # Store visualization
        self.vis_output = vis_img
    
    def _reset(self):
        """Reset the attention system."""
        self.expected_pattern = None
        self.memory_strength = 0.0
        self.attention_width = 0.5
        self.stability_measure = 0.5
        self.activity_history = []
        self.match_history = []
        self.reset_time = time.time()
        self.strengths = np.ones(self.n_dendrites) * 0.5
        self.directions = self._normalize(np.random.randn(self.n_dendrites, 2))
        self.exploration_rate = 0.5
    
    def get_output(self, port_name):
        if port_name == 'attention_field':
            return self.attention_field
        elif port_name == 'visualization':
            return self.vis_output.astype(np.float32) / 255.0
        elif port_name == 'match_score':
            return np.mean(self.match_history[-5:]) if len(self.match_history) >= 5 else 0.5
        elif port_name == 'stability':
            return self.stability_measure
        elif port_name == 'attention_width':
            return self.attention_width
        elif port_name == 'exploration':
            return self.exploration_rate
        elif port_name == 'fractal_dim':
            return self.fractal_dim_value
        elif port_name.startswith('adj_'):
            idx = int(port_name.split('_')[1])
            adjustments = self._get_frequency_adjustments()
            return adjustments[idx] if idx < len(adjustments) else 0.0
        return None
    
    def get_display_image(self):
        # Show the visualization
        img_resized = cv2.resize(self.vis_output, (96, 96), interpolation=cv2.INTER_LINEAR)
        img_resized = np.ascontiguousarray(img_resized)
        
        h, w = img_resized.shape[:2]
        return QtGui.QImage(img_resized.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Num Dendrites", "n_dendrites", self.n_dendrites, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
        ]