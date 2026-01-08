import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

# Try to import sklearn for the "Math" part
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not found. CognitiveManifoldNode will run in fallback mode.")

# BaseNode Injection
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name): return None

class CognitiveManifoldNode(BaseNode):
    """
    Cognitive Manifold Node ( The Map of Meaning )
    ----------------------------------------------
    FIXED: Uses local output buffer and get_output() hook compatible with 
    Perception Lab v9+.
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(0, 100, 200) # Deep Analytic Blue
    
    def __init__(self):
        super().__init__()
        self.node_title = "Cognitive Manifold"
        
        self.inputs = {
            'thought_vector': 'spectrum',  # From Grammar or Dragon
        }
        
        self.outputs = {
            'manifold_view': 'image',     # The 2D Map
            'set_id': 'signal',           # Which "Set" are we in?
            'novelty': 'signal'           # Distance from known thoughts
        }
        
        self.config = {
            'history_len': 300,           # How many past thoughts to remember
            'n_clusters': 5,              # Number of "Sets" to find
        }
        
        # --- THE FIX: Initialize local output storage ---
        self._output_values = {}
        
        # The Memory Buffer
        self.memory_buffer = [] 
        
        # The Projector (PCA)
        self.pca = PCA(n_components=2) if SKLEARN_AVAILABLE else None
        self.kmeans = KMeans(n_clusters=5) if SKLEARN_AVAILABLE else None
        
        # State
        self.is_fitted = False
        self.current_2d_points = None
        self.labels = None

    # --- COMPATIBILITY LAYER (The Fix) ---
    def get_input(self, name):
        # Safe wrapper for PerceptionLab's input blending
        if hasattr(self, 'get_blended_input'):
            return self.get_blended_input(name)
        # Fallback for older hosts
        if hasattr(self, 'input_data') and name in self.input_data:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) and len(val) > 0 else val
        return None

    def set_output(self, name, value):
        # Store in local buffer
        self._output_values[name] = value

    def get_output(self, name):
        # Host calls this to retrieve data
        return self._output_values.get(name, None)
    # -------------------------------------

    def step(self):
        # 1. Get the current thought
        vec = self.get_input('thought_vector')
        
        # Handle cases where input might be None or empty
        if vec is None: 
            return
        
        # Ensure flat numpy array
        try:
            vec = np.array(vec, dtype=np.float32).flatten()
            if vec.size == 0: return
        except Exception:
            return

        # Store in buffer
        if len(self.memory_buffer) > self.config['history_len']:
            self.memory_buffer.pop(0) # Remove oldest
        self.memory_buffer.append(vec)

        # Need enough data to start
        if len(self.memory_buffer) < 20 or not SKLEARN_AVAILABLE:
            return

        # 2. Update the Manifold (Math Logic)
        try:
            data_matrix = np.array(self.memory_buffer)
            if np.isnan(data_matrix).any():
                return
            
            # Update PCA / KMeans periodically
            if not self.is_fitted or len(self.memory_buffer) % 10 == 0:
                self.current_2d_points = self.pca.fit_transform(data_matrix)
                # KMeans expects at least n_samples >= n_clusters
                if len(data_matrix) > self.config['n_clusters']:
                    self.kmeans.fit(self.current_2d_points)
                    self.labels = self.kmeans.labels_
                    self.is_fitted = True
            else:
                if self.is_fitted:
                    self.current_2d_points = self.pca.transform(data_matrix)
                    self.labels = self.kmeans.predict(self.current_2d_points)
        except Exception as e:
            # print(f"Manifold Math Error: {e}") # Silence error to prevent console spam
            return

        if self.current_2d_points is None or self.labels is None:
            return

        # 3. Calculate Signals
        current_pos = self.current_2d_points[-1]
        current_label = self.labels[-1]
        
        # Novelty = Distance from center of mass
        center = np.mean(self.current_2d_points, axis=0)
        dist = np.linalg.norm(current_pos - center)
        
        self.set_output('set_id', [float(current_label)])
        self.set_output('novelty', [float(dist)])

        # 4. Visualization
        self._render_manifold(current_pos)

    def _render_manifold(self, current_pos):
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.float32)
        
        if self.current_2d_points is None: return

        # Normalize 2D points to fit screen [0, 1]
        pts = self.current_2d_points
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        range_xy = max_xy - min_xy + 1e-6
        
        norm_pts = (pts - min_xy) / range_xy
        
        # Colors for clusters (Sets)
        cluster_colors = [
            (0.2, 0.2, 0.9), # Red
            (0.9, 0.2, 0.2), # Blue
            (0.2, 0.9, 0.2), # Green
            (0.2, 0.9, 0.9), # Yellow
            (0.9, 0.2, 0.9)  # Purple
        ]
        n_colors = len(cluster_colors)

        # Draw Points
        for i, pt in enumerate(norm_pts):
            if i >= len(self.labels): break
            
            screen_x = int(pt[0] * (w - 10) + 5)
            screen_y = int(pt[1] * (h - 10) + 5)
            
            label = self.labels[i]
            color = cluster_colors[label % n_colors]
            
            # Fade out older points
            age_factor = (i / len(norm_pts)) # 0 (old) to 1 (new)
            
            if age_factor > 0.2:
                # Apply alpha manually by scaling color
                col_f = tuple(c * age_factor for c in color)
                cv2.circle(img, (screen_x, screen_y), 2, col_f, -1)

        # Draw Sets (Centers)
        if hasattr(self.kmeans, 'cluster_centers_'):
            centers = self.kmeans.cluster_centers_
            norm_centers = (centers - min_xy) / range_xy
            
            for i, center in enumerate(norm_centers):
                cx = int(center[0] * (w - 10) + 5)
                cy = int(center[1] * (h - 10) + 5)
                col = cluster_colors[i % n_colors]
                
                # Faint circle
                cv2.circle(img, (cx, cy), 15, (col[0]*0.3, col[1]*0.3, col[2]*0.3), 1)
                # Label
                cv2.putText(img, f"S{i}", (cx-5, cy-5), cv2.FONT_HERSHEY_PLAIN, 0.8, col, 1)

        # Draw Current Thought (Bright White Orb)
        cur_screen_x = int(((current_pos[0] - min_xy[0]) / range_xy[0]) * (w - 10) + 5)
        cur_screen_y = int(((current_pos[1] - min_xy[1]) / range_xy[1]) * (h - 10) + 5)
        
        cv2.circle(img, (cur_screen_x, cur_screen_y), 4, (1.0, 1.0, 1.0), -1)
        
        # Trail line
        if len(norm_pts) > 1:
            prev = norm_pts[-2]
            prev_x = int(prev[0] * (w - 10) + 5)
            prev_y = int(prev[1] * (h - 10) + 5)
            cv2.line(img, (prev_x, prev_y), (cur_screen_x, cur_screen_y), (1.0, 1.0, 1.0), 1)

        self.set_output('manifold_view', img)