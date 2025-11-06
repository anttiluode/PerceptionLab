"""
Cognitive Set Analyzer Node - Analyzes signal trajectories as "thought patterns"
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import sys
import os
# --- This is the new, correct block ---
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# ------------------------------------

try:
    from sklearn.cluster import KMeans
    from scipy import stats
    import networkx as nx
    SKLEARN_NX_AVAILABLE = True
except ImportError:
    SKLEARN_NX_AVAILABLE = False
    print("Warning: CognitiveSetAnalyzerNode requires 'scikit-learn' and 'networkx'")
    print("Please run: pip install scikit-learn networkx")

class CognitiveSetAnalyzerNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 180, 40) # A golden/analysis color
    
    def __init__(self, trajectory_length=500, num_states=10, display_mode="Radar Plot"):
        super().__init__()
        self.node_title = "Cognitive Set Analyzer"
        
        self.inputs = {
            'signal_1': 'signal', 
            'signal_2': 'signal', 
            'signal_3': 'signal', 
            'signal_4': 'signal'
        }
        self.outputs = {'image': 'image', 'entropy': 'signal'}
        
        self.trajectory_length = int(trajectory_length)
        self.num_states = int(num_states)
        self.display_mode = display_mode
        
        self.trajectory = []
        self.metrics = {}
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)

        if not SKLEARN_NX_AVAILABLE:
            self.node_title = "Set Analyzer (Libs Missing!)"

    def step(self):
        if not SKLEARN_NX_AVAILABLE:
            return

        # 1. Collect signal vector
        vec = [
            self.get_blended_input('signal_1', 'sum') or 0.0,
            self.get_blended_input('signal_2', 'sum') or 0.0,
            self.get_blended_input('signal_3', 'sum') or 0.0,
            self.get_blended_input('signal_4', 'sum') or 0.0
        ]
        
        self.trajectory.append(vec)
        if len(self.trajectory) > self.trajectory_length:
            self.trajectory.pop(0)

        # 2. Analyze if we have enough data
        if len(self.trajectory) < 50:
            return
            
        traj_np = np.array(self.trajectory)
        
        if self.display_mode == "Radar Plot":
            # 3. Analyze state dynamics (from brain_set_system.py)
            self.metrics = self._analyze_dynamics(traj_np)
            # 4. Draw Radar Plot
            self.display_img = self._draw_radar_plot(self.metrics)
        
        elif self.display_mode == "Similarity Matrix":
            # 3. Analyze correlation
            corr = np.corrcoef(traj_np.T)
            corr = (corr + 1.0) / 2.0 # Normalize -1..1 to 0..1
            corr_u8 = (corr * 255).astype(np.uint8)
            # 4. Draw Matrix
            img = cv2.resize(corr_u8, (128, 128), interpolation=cv2.INTER_NEAREST)
            self.display_img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
            
    def _analyze_dynamics(self, latent_trajectory):
        """Adapted from analyze_state_dynamics in brain_set_system.py"""
        n_states = self.num_states
        if len(latent_trajectory) < n_states:
            return {}
            
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init='auto')
        state_labels = kmeans.fit_predict(latent_trajectory)
        
        transitions = np.zeros((n_states, n_states))
        for i in range(len(state_labels) - 1):
            transitions[state_labels[i], state_labels[i+1]] += 1
        
        row_sums = transitions.sum(axis=1)
        transition_probs = transitions / row_sums[:, np.newaxis]
        transition_probs[np.isnan(transition_probs)] = 0
        
        metrics = {}
        state_probs = np.bincount(state_labels) / len(state_labels)
        metrics['state_entropy'] = stats.entropy(state_probs[state_probs > 0])
        
        flat_transitions = transition_probs.flatten()
        metrics['transition_entropy'] = stats.entropy(flat_transitions[flat_transitions > 0])
        
        loops = 0
        for i in range(n_states):
            if transition_probs[i, i] > 0.3:
                loops += 1
        metrics['loops'] = loops
        
        try:
            G = nx.from_numpy_array(transitions, create_using=nx.DiGraph)
            communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
            metrics['modularity'] = nx.community.modularity(G.to_undirected(), communities)
        except Exception:
            metrics['modularity'] = 0
            
        return metrics

    def _draw_radar_plot(self, metrics):
        """Draw a radar plot using numpy and cv2."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        center = (64, 64)
        radius = 55
        
        categories = ['State Entropy', 'Trans. Entropy', 'Modularity', 'Loops']
        n_cats = len(categories)
        
        # Get values and normalize
        vals = [
            metrics.get('state_entropy', 0) / 2.3, # Normalize (log(10))
            metrics.get('transition_entropy', 0) / 4.6, # Normalize (log(100))
            metrics.get('modularity', 0),
            metrics.get('loops', 0) / self.num_states
        ]
        vals = np.clip(vals, 0, 1)
        
        # Draw grid
        for i in range(n_cats):
            angle = (i / n_cats) * 2 * np.pi - (np.pi / 2)
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            cv2.line(img, center, (x, y), (50, 50, 50), 1)
        
        # Draw data shape
        points = []
        for i in range(n_cats):
            angle = (i / n_cats) * 2 * np.pi - (np.pi / 2)
            r = radius * vals[i]
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
            
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(100, 255, 100), thickness=2)
        cv2.fillPoly(img, [pts], color=(50, 120, 50, 0.5))
        
        return img

    def get_output(self, port_name):
        if port_name == 'image':
            return self.display_img.astype(np.float32) / 255.0
        elif port_name == 'entropy':
            return self.metrics.get('state_entropy', 0.0)
        return None
        
    def get_display_image(self):
        rgb = np.ascontiguousarray(self.display_img)
        h, w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Trajectory Length", "trajectory_length", self.trajectory_length, None),
            ("Number of States", "num_states", self.num_states, None),
            ("Display Mode", "display_mode", self.display_mode, [
                ("Radar Plot", "Radar Plot"), 
                ("Similarity Matrix", "Similarity Matrix")
            ]),
        ]