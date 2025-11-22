"""
System Optimizer Node (Energy-Based Fix)
----------------------------------------
FIX V2: Now measures 'activity' as SIGNAL MAGNITUDE, not just change.
This allows it to learn static logic states (like holding a '1')
without pruning the connection.
"""

import numpy as np
import cv2
import gc
import sys

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SystemOptimizerNode(BaseNode):
    NODE_CATEGORY = "Meta"
    NODE_COLOR = QtGui.QColor(255, 215, 0) # Gold (The Controller)

    def __init__(self, learning_rate=0.05, prune_threshold=0.1):
        super().__init__()
        self.node_title = "System Optimizer (Energy)"
        
        self.inputs = {
            'global_reward': 'signal',   # Teach the system what you like
            'reset': 'signal'
        }
        
        self.outputs = {
            'network_state': 'image',    # Visualization
            'active_connections': 'signal',
            'system_entropy': 'signal'
        }
        
        self.learning_rate = float(learning_rate)
        self.prune_threshold = float(prune_threshold)
        
        self.scene_ref = None
        self.edge_stats = {} 
        self.frame_count = 0
        self.matrix_vis = np.zeros((128, 128, 3), dtype=np.uint8)

    def _find_scene(self):
        if self.scene_ref is not None: return self.scene_ref
        for obj in gc.get_objects():
            if hasattr(obj, 'nodes') and hasattr(obj, 'edges') and hasattr(obj, 'add_node'):
                if isinstance(obj.nodes, list) and isinstance(obj.edges, list):
                    self.scene_ref = obj
                    return obj
        return None

    def step(self):
        scene = self._find_scene()
        if scene is None: return

        reward = self.get_blended_input('global_reward', 'sum') or 0.0
        
        self.frame_count += 1
        total_entropy = 0.0
        active_count = 0
        
        for edge in scene.edges:
            edge_id = id(edge)
            if edge_id not in self.edge_stats:
                self.edge_stats[edge_id] = {'activity': 0.0, 'strength': 0.5}
            
            # Get current flow
            current_flow = getattr(edge, 'effect_val', 0.0)
            
            # --- THE FIX: ENERGY METRIC ---
            # Old way: Change = abs(current - history) -> Fails on square waves
            # New way: Energy = abs(current) -> Works on static '1's
            
            energy = abs(current_flow)
            
            # We smooth it so short pulses count too
            self.edge_stats[edge_id]['activity'] = (self.edge_stats[edge_id]['activity'] * 0.9) + (energy * 0.1)
            
            # --- Hebbian Logic ---
            # Target strength based on activity and reward
            # We boost the reward impact significantly
            activity = self.edge_stats[edge_id]['activity']
            target_strength = activity * (0.5 + reward * 2.0)
            
            current_strength = self.edge_stats[edge_id]['strength']
            
            # Asymmetric learning: Learn fast (0.1), forget slow (0.01)
            if target_strength > current_strength:
                lr = self.learning_rate # Grow fast
            else:
                lr = self.learning_rate * 0.1 # Prune slow
                
            new_strength = current_strength * (1.0 - lr) + target_strength * lr
            new_strength = max(0.0, min(1.0, new_strength))
            
            self.edge_stats[edge_id]['strength'] = new_strength
            
            # Write to physics engine
            edge.learned_weight = new_strength

            # Visual feedback
            if new_strength < self.prune_threshold:
                edge.setOpacity(0.1) 
            else:
                edge.setOpacity(0.5 + new_strength * 0.5) 
                active_count += 1
                
            total_entropy += new_strength

        self.render_matrix(scene)
        self.set_output('active_connections', float(active_count))
        self.set_output('system_entropy', total_entropy)
        self.set_output('network_state', self.matrix_vis)

    def render_matrix(self, scene):
        dim = 128
        img = np.zeros((dim, dim, 3), dtype=np.uint8)
        num_nodes = len(scene.nodes)
        if num_nodes == 0: return
        cell_size = max(2, dim // num_nodes)
        node_map = {id(n): i for i, n in enumerate(scene.nodes)}
        
        for edge in scene.edges:
            try:
                u = node_map.get(id(edge.src.parentItem()), 0)
                v = node_map.get(id(edge.tgt.parentItem()), 0)
                strength = self.edge_stats.get(id(edge), {}).get('strength', 0)
                c = int(strength * 255)
                cv2.rectangle(img, (u*cell_size, v*cell_size), ((u+1)*cell_size, (v+1)*cell_size), (c, c, 50), -1)
            except: continue
        self.matrix_vis = img

    def get_output(self, port_name):
        if hasattr(self, 'outputs_data') and port_name in self.outputs_data: return self.outputs_data[port_name]
        return None
    def set_output(self, name, val):
        if not hasattr(self, 'outputs_data'): self.outputs_data = {}
        self.outputs_data[name] = val
    def get_display_image(self):
        return QtGui.QImage(self.matrix_vis.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)
    def get_config_options(self):
        return [("Learning Rate", "learning_rate", self.learning_rate, None), ("Prune Threshold", "prune_threshold", self.prune_threshold, None)]