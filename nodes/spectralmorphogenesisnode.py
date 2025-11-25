"""
Spectral Morphogenesis Node v4 - Chrono-Topological Monitor
-----------------------------------------------------------
1. Maps EEG/Signal Eigenmodes to 3D Space.
2. Measures "Brain Torque" (Angular Velocity of the Eigenmodes).
3. Visualizes the "Wormhole" (Trajectories of State Space).

Updates:
- Removed artificial camera rotation.
- Added Kabsch Algorithm for precise rotation tracking.
- Added Trail Rendering.
"""

import numpy as np
import cv2
from collections import deque
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class SpectralMorphogenesisNode(BaseNode):
    NODE_CATEGORY = "Experimental"
    NODE_COLOR = QtGui.QColor(180, 100, 255) # Deep Purple

    def __init__(self):
        super().__init__()
        self.node_title = "Spectral Morphogenesis (Topo-Monitor)"
        
        self.inputs = {
            'input_spectrum': 'spectrum',
            'growth_rate': 'signal'
        }
        
        self.outputs = {
            'folded_view': 'image',
            'eigen_coords': 'spectrum',
            'angular_velocity': 'signal', # NEW: Brain RPM
            'fold_coherence': 'signal',   # NEW: Stability
            'rotation_axis_x': 'signal',
            'rotation_axis_y': 'signal',
            'rotation_axis_z': 'signal'
        }
        
        # --- Physics & Geometry State ---
        self.grid_size = 10
        self.n_nodes = self.grid_size * self.grid_size
        self.adj_matrix = self._create_grid_adjacency(self.grid_size)
        self.node_activity = np.zeros(self.n_nodes)
        
        # Position Tracking
        self.node_positions = np.zeros((self.n_nodes, 3))
        self.prev_positions = None # For calculating velocity
        
        # Rotation Metrics
        self.angular_velocity = 0.0
        self.rotation_axis = np.array([0.0, 1.0, 0.0])
        self.coherence_metric = 1.0
        
        # Visualization / Time-Tube
        self.frame_counter = 0
        self.fold_interval = 2 # Update physics every N frames (Faster now)
        self.display_buffer = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Trail History (The Wormhole)
        # Stores list of (x,y) screen coords for previous frames
        self.trail_length = 30
        self.trails = [deque(maxlen=self.trail_length) for _ in range(self.n_nodes)]

        # Output Storage
        self._output_data = {
            'folded_view': self.display_buffer,
            'eigen_coords': np.zeros(self.n_nodes * 3),
            'angular_velocity': 0.0,
            'fold_coherence': 0.0,
            'rotation_axis_x': 0.0,
            'rotation_axis_y': 0.0,
            'rotation_axis_z': 0.0
        }

    def _create_grid_adjacency(self, size):
        n = size * size
        adj = np.zeros((n, n))
        for r in range(size):
            for c in range(size):
                i = r * size + c
                if c < size - 1:
                    j = r * size + (c + 1)
                    adj[i, j] = adj[j, i] = 1.0
                if r < size - 1:
                    j = (r + 1) * size + c
                    adj[i, j] = adj[j, i] = 1.0
        return adj

    def step(self):
        # 1. Inputs
        inp = self.get_blended_input('input_spectrum')
        rate = self.get_blended_input('growth_rate')
        if rate is None: rate = 0.05
        if inp is None: return

        # 2. Input Mapping
        target_len = self.n_nodes
        if isinstance(inp, (int, float)): inp = np.array([inp])
        
        if len(inp) != target_len:
            inp_resampled = np.interp(np.linspace(0, len(inp), target_len), np.arange(len(inp)), inp)
        else:
            inp_resampled = inp

        # 3. Activity Dynamics
        self.node_activity = self.node_activity * 0.9 + inp_resampled * 0.5
        
        # 4. Hebbian Tension
        hot_indices = np.where(self.node_activity > 0.6)[0]
        if len(hot_indices) > 1:
            for i in hot_indices:
                j = np.random.choice(hot_indices)
                if i != j:
                    self.adj_matrix[i, j] += rate * 0.1
                    self.adj_matrix[j, i] += rate * 0.1
        
        self.adj_matrix *= 0.995 
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = np.clip(self.adj_matrix, 0.01, 10.0)

        # 5. Physics: Eigenmode Folding & Rotation Tracking
        self.frame_counter += 1
        if self.frame_counter % self.fold_interval == 0:
            try:
                laplacian = csgraph.laplacian(self.adj_matrix, normed=True)
                # Get k=4 vectors. Index 0 is constant. 1,2,3 are XYZ
                vals, vecs = eigsh(laplacian, k=4, which='SM') 
                
                new_pos = vecs[:, 1:4]
                max_range = np.max(np.abs(new_pos))
                if max_range > 0:
                    new_pos /= max_range
                
                # --- THE GROK LOGIC: Rotation Tracking (Kabsch Algorithm) ---
                if self.prev_positions is not None:
                    # Centering
                    P = self.prev_positions
                    Q = new_pos
                    # Compute covariance matrix
                    H = np.transpose(P) @ Q
                    # SVD
                    U, S, Vt = np.linalg.svd(H)
                    # Rotation matrix
                    R = Vt.T @ U.T
                    
                    # Handle reflection case
                    if np.linalg.det(R) < 0:
                        Vt[2, :] *= -1
                        R = Vt.T @ U.T
                        
                    # Calculate Axis-Angle
                    trace = np.trace(R)
                    # Clip to handle numerical errors > 1.0 or < -1.0
                    theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
                    
                    # Angular Velocity (Degrees per calculation step)
                    deg_per_step = np.degrees(theta)
                    self.angular_velocity = deg_per_step
                    
                    # Coherence: How well did the rigid rotation fit?
                    # Transform prev points by R and compare to new points
                    P_rotated = (P @ R.T)
                    error = np.linalg.norm(Q - P_rotated)
                    self.coherence_metric = 1.0 / (1.0 + error) # Inverse error
                    
                    # Store Axis (Simplified)
                    self.rotation_axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
                    norm = np.linalg.norm(self.rotation_axis)
                    if norm > 0: self.rotation_axis /= norm

                self.prev_positions = new_pos.copy()
                self.node_positions = new_pos

            except Exception as e:
                # print(f"Eigen-error: {e}") 
                pass

        # 6. Render with Time-Tube
        self._render_structure()
        
        # 7. Update Outputs
        self._output_data['eigen_coords'] = self.node_positions.flatten()
        self._output_data['folded_view'] = self.display_buffer
        self._output_data['angular_velocity'] = self.angular_velocity
        self._output_data['fold_coherence'] = self.coherence_metric
        self._output_data['rotation_axis_x'] = self.rotation_axis[0]
        self._output_data['rotation_axis_y'] = self.rotation_axis[1]
        self._output_data['rotation_axis_z'] = self.rotation_axis[2]

    def get_output(self, port_name):
        return self._output_data.get(port_name, 0.0)

    def _render_structure(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        center, scale = 128, 90 # Slightly smaller to fit trails
        
        # No artificial camera rotation! We see the raw data spin.
        x = self.node_positions[:, 0]
        y = self.node_positions[:, 1]
        z = self.node_positions[:, 2] # Use Z for depth cues
        
        screen_x = (x * scale + center).astype(int)
        screen_y = (y * scale + center).astype(int)
        
        # 1. Update Trails
        for i in range(self.n_nodes):
            if 0 <= screen_x[i] < 256 and 0 <= screen_y[i] < 256:
                self.trails[i].append((screen_x[i], screen_y[i]))

        # 2. Draw Trails (The Wormhole)
        # Optimization: Only draw trails for every 3rd node to keep it clean
        for i in range(0, self.n_nodes, 3):
            if len(self.trails[i]) > 2:
                # Convert deque to array for polylines
                pts = np.array(self.trails[i], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Color fades based on Z depth of the head
                depth = z[i]
                c_val = int(120 + depth * 100)
                color = (c_val//2, c_val//3, c_val) # Faded purple trails
                
                cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)

        # 3. Draw Connections (Current State)
        strong_links = np.argwhere(self.adj_matrix > 0.4)
        for (i, j) in strong_links:
            if i < j:
                pt1 = (screen_x[i], screen_y[i])
                pt2 = (screen_x[j], screen_y[j])
                weight = self.adj_matrix[i, j]
                intensity = int(min(255, weight * 80))
                # White/Cyan for the "Head" of the wormhole
                cv2.line(img, pt1, pt2, (intensity, intensity, 255), 1)

        # 4. Draw Heads
        for i in range(self.n_nodes):
            if 0 <= screen_x[i] < 256 and 0 <= screen_y[i] < 256:
                # Hotter color for higher activity
                act = self.node_activity[i]
                color = (int(act*255), 255, 255) # Yellow/White hot
                cv2.circle(img, (screen_x[i], screen_y[i]), 2, color, -1)
                
        # 5. Draw Info Stats
        text = f"RPM: {self.angular_velocity*30:.1f}" # Approx frames/sec * deg
        cv2.putText(img, text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.display_buffer = img

    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", 10, None),
            ("Trail Length", "trail_length", 30, None)
        ]