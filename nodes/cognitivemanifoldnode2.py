import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name): return None

class CognitiveManifoldNode2(BaseNode):
    """
    Cognitive Manifold Node
    -----------------------
    Projects high-dimensional thought vectors onto a 2D map.
    
    Uses:
    - PCA for dimensionality reduction
    - K-means for identifying cognitive "Sets" (attractor regions)
    - Trajectory tracking for visualizing thought flow
    
    Outputs:
    - manifold_view: 2D map with trajectory and clusters
    - set_id: Current cognitive set (cluster ID)
    - novelty: How far current state is from known attractors
    """
    NODE_CATEGORY = "Consciousness"
    NODE_COLOR = QtGui.QColor(200, 100, 255)  # Purple for mind
    
    def __init__(self):
        super().__init__()
        self.node_title = "Cognitive Manifold"
        
        self.inputs = {
            'thought_vector': 'spectrum',
            'reset': 'signal',
        }
        
        self.outputs = {
            'manifold_view': 'image',
            'set_id': 'signal',
            'novelty': 'signal',
            'trajectory': 'spectrum',  # Flattened trajectory for downstream
        }
        
        self.config = {
            'n_clusters': 5,
            'trajectory_length': 200,
            'canvas_size': 400,
            'learning_rate': 0.02,
        }
        
        self._output_values = {}
        self._init_state()

    def _init_state(self):
        # History buffer for PCA
        self.history = []
        self.max_history = 500
        
        # Trajectory (projected 2D points)
        self.trajectory = []
        
        # PCA components (learned online)
        self.pca_mean = None
        self.pca_components = None
        
        # Cluster centers
        self.n_clusters = self.config['n_clusters']
        self.cluster_centers = None
        self.cluster_colors = [
            (255, 100, 100),   # Red
            (100, 255, 100),   # Green
            (100, 100, 255),   # Blue
            (255, 255, 100),   # Yellow
            (255, 100, 255),   # Magenta
            (100, 255, 255),   # Cyan
            (255, 180, 100),   # Orange
            (180, 100, 255),   # Purple
        ]

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'):
            return self.get_blended_input(name)
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value):
        self._output_values[name] = value
    
    def get_output(self, name):
        return self._output_values.get(name, None)

    def _update_pca(self, vec):
        """Online PCA update"""
        lr = self.config['learning_rate']
        
        if self.pca_mean is None:
            self.pca_mean = vec.copy()
            self.pca_components = np.random.randn(2, len(vec)) * 0.1
            # Orthonormalize
            self.pca_components[0] /= np.linalg.norm(self.pca_components[0]) + 1e-10
            self.pca_components[1] -= np.dot(self.pca_components[1], self.pca_components[0]) * self.pca_components[0]
            self.pca_components[1] /= np.linalg.norm(self.pca_components[1]) + 1e-10
            return
        
        # Update mean
        self.pca_mean = self.pca_mean * (1 - lr) + vec * lr
        
        # Centered vector
        centered = vec - self.pca_mean
        
        # Oja's rule for online PCA
        for i in range(2):
            proj = np.dot(self.pca_components[i], centered)
            self.pca_components[i] += lr * proj * (centered - proj * self.pca_components[i])
            # Normalize
            norm = np.linalg.norm(self.pca_components[i])
            if norm > 1e-10:
                self.pca_components[i] /= norm
        
        # Orthogonalize second component
        self.pca_components[1] -= np.dot(self.pca_components[1], self.pca_components[0]) * self.pca_components[0]
        norm = np.linalg.norm(self.pca_components[1])
        if norm > 1e-10:
            self.pca_components[1] /= norm

    def _project(self, vec):
        """Project to 2D"""
        if self.pca_mean is None or self.pca_components is None:
            return np.array([0.0, 0.0])
        
        centered = vec - self.pca_mean
        return np.array([
            np.dot(self.pca_components[0], centered),
            np.dot(self.pca_components[1], centered)
        ])

    def _update_clusters(self, point_2d):
        """Online k-means update"""
        if self.cluster_centers is None:
            # Initialize around origin
            self.cluster_centers = np.random.randn(self.n_clusters, 2) * 0.5
        
        # Find nearest cluster
        distances = np.linalg.norm(self.cluster_centers - point_2d, axis=1)
        nearest = np.argmin(distances)
        
        # Move cluster toward point (online update)
        lr = self.config['learning_rate']
        self.cluster_centers[nearest] += lr * (point_2d - self.cluster_centers[nearest])
        
        return nearest, distances[nearest]

    def step(self):
        reset = self.get_input('reset')
        if reset is not None and reset > 0.5:
            self._init_state()
            return
        
        vec = self.get_input('thought_vector')
        if vec is None:
            return
        
        vec = np.array(vec, dtype=np.float32).flatten()
        if len(vec) < 4:
            return
        
        # Add to history
        self.history.append(vec.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update PCA
        self._update_pca(vec)
        
        # Project to 2D
        point_2d = self._project(vec)
        
        # Update trajectory
        self.trajectory.append(point_2d.copy())
        max_traj = self.config['trajectory_length']
        if len(self.trajectory) > max_traj:
            self.trajectory.pop(0)
        
        # Update clusters
        set_id, distance = self._update_clusters(point_2d)
        
        # Compute novelty (distance from nearest cluster, normalized)
        avg_dist = np.mean([np.linalg.norm(c) for c in self.cluster_centers]) + 1e-10
        novelty = float(np.clip(distance / avg_dist, 0, 1))
        
        # === OUTPUTS ===
        self.set_output('set_id', float(set_id))
        self.set_output('novelty', novelty)
        
        # Flatten trajectory for spectrum output
        if len(self.trajectory) > 0:
            traj_flat = np.array(self.trajectory).flatten()
            self.set_output('trajectory', traj_flat)
        
        # Render
        self._render_manifold(point_2d, set_id)

    def _render_manifold(self, current_point, current_set):
        """Render the cognitive manifold visualization"""
        size = self.config['canvas_size']
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        
        if len(self.trajectory) < 2:
            self.set_output('manifold_view', canvas)
            return
        
        # Find bounds for normalization
        traj_array = np.array(self.trajectory)
        all_points = traj_array
        if self.cluster_centers is not None:
            all_points = np.vstack([traj_array, self.cluster_centers])
        
        x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
        y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5
        
        # Ensure non-zero range
        if x_max - x_min < 0.1:
            x_min, x_max = -1, 1
        if y_max - y_min < 0.1:
            y_min, y_max = -1, 1
        
        def to_pixel(p):
            px = int((p[0] - x_min) / (x_max - x_min) * (size - 40) + 20)
            py = int((1 - (p[1] - y_min) / (y_max - y_min)) * (size - 40) + 20)
            return (np.clip(px, 0, size-1), np.clip(py, 0, size-1))
        
        # Draw cluster centers
        if self.cluster_centers is not None:
            for i, center in enumerate(self.cluster_centers):
                px, py = to_pixel(center)
                color = self.cluster_colors[i % len(self.cluster_colors)]
                cv2.circle(canvas, (px, py), 15, color, 2)
                cv2.putText(canvas, f"S{i}", (px - 8, py + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw trajectory (fading)
        n_points = len(self.trajectory)
        for i in range(1, n_points):
            alpha = i / n_points
            p1 = to_pixel(self.trajectory[i - 1])
            p2 = to_pixel(self.trajectory[i])
            
            # Color based on current set
            base_color = self.cluster_colors[current_set % len(self.cluster_colors)]
            color = tuple(int(c * alpha) for c in base_color)
            
            cv2.line(canvas, p1, p2, color, 1)
        
        # Draw current point
        px, py = to_pixel(current_point)
        cv2.circle(canvas, (px, py), 8, (255, 255, 255), -1)
        cv2.circle(canvas, (px, py), 10, self.cluster_colors[current_set % len(self.cluster_colors)], 2)
        
        # Info text
        cv2.putText(canvas, f"Set: {current_set}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Points: {len(self.trajectory)}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        self.set_output('manifold_view', canvas)
