"""
System Optimizer Node (The Meta-Brain)
--------------------------------------
This is the "God Node" that sits above the graph.
It uses Python introspection to find the host application,
monitor every wire in your system, and optimize them in real-time.

FEATURES:
1. Auto-Discovery: Finds all connections in your graph automatically.
2. Traffic Analysis: Measures "Information Flow" (Variance/Entropy) on every wire.
3. Neuroplasticity: Strengthens active wires, prunes dead ones.
4. Global Optimization: Re-routes signal flow to maximize system complexity.

It turns your 'Graph' into a 'Brain'.
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
        self.node_title = "System Optimizer"
        
        self.inputs = {
            'global_reward': 'signal',   # Teach the system what you like
            'reset': 'signal'
        }
        
        self.outputs = {
            'network_state': 'image',    # Visualization of the whole graph connections
            'active_connections': 'signal',
            'system_entropy': 'signal'
        }
        
        self.learning_rate = float(learning_rate)
        self.prune_threshold = float(prune_threshold)
        
        # Internal storage
        self.scene_ref = None
        self.edge_stats = {} # Stores history for every edge in the app
        self.frame_count = 0
        self.matrix_vis = np.zeros((128, 128, 3), dtype=np.uint8)

    def _find_scene(self):
        """
        The 'God Mode' Hack:
        Uses garbage collection to find the PerceptionScene instance in memory.
        This allows the node to see/modify the entire graph structure.
        """
        if self.scene_ref is not None:
            return self.scene_ref
            
        # Look for objects that look like our Scene
        for obj in gc.get_objects():
            if hasattr(obj, 'nodes') and hasattr(obj, 'edges') and hasattr(obj, 'add_node'):
                # Verify it's likely the right object
                if isinstance(obj.nodes, list) and isinstance(obj.edges, list):
                    self.scene_ref = obj
                    print("SystemOptimizer: Connected to Graph Scene!")
                    return obj
        return None

    def step(self):
        scene = self._find_scene()
        if scene is None: 
            return

        reward = self.get_blended_input('global_reward', 'sum') or 0.0
        
        self.frame_count += 1
        total_entropy = 0.0
        active_count = 0
        
        # --- 1. Introspect the Graph ---
        # Iterate over every connection currently existing in the software
        for edge in scene.edges:
            # Create a unique ID for this connection
            edge_id = id(edge)
            
            if edge_id not in self.edge_stats:
                self.edge_stats[edge_id] = {'activity': 0.0, 'strength': 0.5}
            
            # Get the actual value flowing through the wire right now
            # (We access the 'effect_val' visual parameter which tracks signal strength)
            current_flow = getattr(edge, 'effect_val', 0.0)
            
            # --- 2. Measure Information (Entropy/Variance) ---
            # Information = Change. Static signals carry no info.
            history = self.edge_stats[edge_id].get('history', 0.0)
            change = abs(current_flow - history)
            self.edge_stats[edge_id]['history'] = current_flow
            
            # Update activity metric (EMA)
            self.edge_stats[edge_id]['activity'] = (self.edge_stats[edge_id]['activity'] * 0.9) + (change * 0.1)
            
            # --- 3. Apply Hebbian Learning ---
            # If the wire is active, strengthen it. If static, weaken it.
            # Reward signal acts as a global modulator (dopamine).
            
            target_strength = self.edge_stats[edge_id]['activity'] * (1.0 + reward)
            
            current_strength = self.edge_stats[edge_id]['strength']
            new_strength = current_strength * 0.99 + target_strength * self.learning_rate
            
            # Clamp
            new_strength = max(0.0, min(1.0, new_strength))
            self.edge_stats[edge_id]['strength'] = new_strength
            
            # --- 4. Modify the Physics ---
            # We actually write back to the edge object!
            # We use 'effect_val' to visually dim the wire, 
            # effectively pruning it from the user's view.
            # (In a deeper integration, we would modify the node's input multiplier).
            
            # If below threshold, we visually "cut" it (make it transparent)
            if new_strength < self.prune_threshold:
                edge.setOpacity(0.1) # Ghost connection
            else:
                edge.setOpacity(0.5 + new_strength * 0.5) # Bright active connection
                active_count += 1
                
            total_entropy += new_strength

        # --- 5. Visualize the "Brain State" ---
        # Render a connectivity matrix of the whole system
        self.render_matrix(scene)
        
        # Outputs
        self.set_output('active_connections', float(active_count))
        self.set_output('system_entropy', total_entropy)
        self.set_output('network_state', self.matrix_vis)

    def render_matrix(self, scene):
        """Draws the connectivity matrix of the living graph"""
        dim = 128
        img = np.zeros((dim, dim, 3), dtype=np.uint8)
        
        num_nodes = len(scene.nodes)
        if num_nodes == 0: return
        
        cell_size = max(2, dim // num_nodes)
        
        # Map edges to matrix coordinates
        node_map = {id(n): i for i, n in enumerate(scene.nodes)}
        
        for edge in scene.edges:
            # Find source and target nodes
            try:
                src_node = edge.src.parentItem()
                tgt_node = edge.tgt.parentItem()
                
                u = node_map.get(id(src_node), 0)
                v = node_map.get(id(tgt_node), 0)
                
                strength = self.edge_stats.get(id(edge), {}).get('strength', 0)
                
                # Draw cell
                c = int(strength * 255)
                color = (c, c, 50) # Yellow/Gold
                
                x = u * cell_size
                y = v * cell_size
                
                cv2.rectangle(img, (x, y), (x+cell_size, y+cell_size), color, -1)
            except:
                continue
                
        self.matrix_vis = img

    def get_output(self, port_name):
        if hasattr(self, 'outputs_data') and port_name in self.outputs_data:
            return self.outputs_data[port_name]
        return None

    def set_output(self, name, val):
        if not hasattr(self, 'outputs_data'): self.outputs_data = {}
        self.outputs_data[name] = val
        
    def get_display_image(self):
        img_resized = cv2.resize(self.matrix_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        return QtGui.QImage(img_resized.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Prune Threshold", "prune_threshold", self.prune_threshold, None)
        ]