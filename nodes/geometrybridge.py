"""
Geometric Bridge Node - The "Resonance Detector"
================================================
Connects Raj et al. (Eigenmodes) with Surface Optimization (Energy Minimization).

This node maps input signals (Holographic FFT or Webcam) onto a spherical mesh
and calculates the "Geometric Energy" (Dirichlet/Curvature Energy).

- LOW ENERGY (Blue) = The signal "fits" the brain's geometry (Resonance).
- HIGH ENERGY (Red) = The signal fights the geometry (Noise/Dissonance).

It tests the hypothesis: "Is the brain's shape optimized to minimize the energy of natural signals?"
"""

import numpy as np
import cv2
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    # Fallback for testing
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class GeometricBridgeNode(BaseNode):
    NODE_CATEGORY = "Experimental"
    NODE_TITLE = "Geometric Bridge (Resonance)"
    NODE_COLOR = QtGui.QColor(255, 60, 120)  # Hot Pink for "New Science"
    
    def __init__(self):
        super().__init__()
        
        # Inputs: The "World" (Webcam or FFT)
        self.inputs = {
            'signal_in': 'spectrum',  # Can take image or spectrum
            'smoothing': 'signal'     # optimization strength
        }
        
        self.outputs = {
            'resonance_map': 'image',   # Visual of where the signal fits
            'energy_plot': 'image',     # The "Result" graph
            'geometric_stress': 'signal' # The raw energy value
        }
        
        # Internal: The "Brain" Model (A Geodesic Sphere Proxy)
        self.resolution = 32 # Grid size for the mesh proxy
        self.L = None        # The Laplacian (Geometry Matrix)
        self.eigenvals = None
        self.eigenvecs = None
        
        self.energy_history = []
        self._last_display = None
        
        # Initialize the geometry immediately
        self._build_geometry()

    def _build_geometry(self):
        """
        Builds a simple 2D manifold graph (Grid) that represents a 
        flattened patch of cortex/surface.
        We compute the Laplacian (L) here to measure 'Roughness' or 'Energy'.
        """
        N = self.resolution
        size = N * N
        
        # Build Adjacency Matrix for a grid (nearest neighbors)
        # This represents the physical connections of the surface
        rows, cols, data = [], [], []
        
        for y in range(N):
            for x in range(N):
                idx = y * N + x
                # Neighbors: Up, Down, Left, Right
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < N and 0 <= nx < N:
                        n_idx = ny * N + nx
                        rows.append(idx)
                        cols.append(n_idx)
                        data.append(1.0)
        
        # Create Laplacian L = D - A
        A = coo_matrix((data, (rows, cols)), shape=(size, size))
        degrees = np.array(A.sum(axis=1)).flatten()
        D = diags(degrees)
        self.L = (D - A).tocsr() # The Geometry Operator
        
        # Pre-compute Eigenmodes (The "Raj et al" part)
        # We only need the first few low-frequency modes to test resonance
        try:
            k = 20
            vals, vecs = eigsh(self.L, k=k, which='SM', tol=1e-3)
            self.eigenvals = vals
            self.eigenvecs = vecs
            print(f"[Bridge] Geometry Built. {k} Eigenmodes calculated.")
        except Exception as e:
            print(f"[Bridge] Error building geometry: {e}")

    def step(self):
        # 1. GET INPUT (The Signal)
        inp = self.get_blended_input('signal_in', 'mean')
        if inp is None: return

        # Resize input to match our geometry resolution (32x32)
        # This maps the visual world onto our "Brain Surface"
        if isinstance(inp, np.ndarray):
            # Handle complex spectrum or real image
            if np.iscomplexobj(inp):
                inp = np.abs(inp) # Magnitude for now
            
            # Normalize
            inp = cv2.resize(inp.astype(np.float32), (self.resolution, self.resolution))
            signal_vector = inp.flatten()
            
            # Normalize energy
            if signal_vector.max() > 0:
                signal_vector /= signal_vector.max()
        else:
            return

        # 2. CALCULATE GEOMETRIC ENERGY (The "Optimization" part)
        # Energy E = x.T * L * x
        # This measures how much the signal "fights" the connections.
        # Smooth signals (Resonance) = Low Energy. Noise = High Energy.
        
        # Roughness / Dirichlet Energy
        energy = signal_vector.T @ self.L @ signal_vector
        energy_val = float(energy) / self.resolution**2

        # 3. SPECTRAL FILTERING (Raj et al. Simulation)
        # Project signal onto Eigenmodes and reconstruct
        # This shows what the brain *actually* sees (the filtered version)
        if self.eigenvecs is not None:
            weights = self.eigenvecs.T @ signal_vector
            filtered = self.eigenvecs @ weights
            resonance_map = filtered.reshape(self.resolution, self.resolution)
        else:
            resonance_map = inp

        # 4. TRACK HISTORY
        self.energy_history.append(energy_val)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

        # 5. VISUALIZATION
        self._render_display(resonance_map, energy_val)
        
    def _render_display(self, resonance_map, energy_val):
        """
        Draws the surface resonance and the energy graph.
        """
        H, W = 200, 300
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # A. Draw the Resonance Map (Left)
        map_size = 140
        disp_map = cv2.resize(resonance_map, (map_size, map_size))
        # Color: Blue=Calm, Red=Stress
        disp_map = np.clip(disp_map * 255, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_map, cv2.COLORMAP_OCEAN)
        
        # Place on canvas
        y_off = (H - map_size) // 2
        canvas[y_off:y_off+map_size, 10:10+map_size] = disp_color
        
        # B. Draw the Energy Graph (Right)
        # This is the "Scientific Result" tracker
        graph_x = 160
        graph_w = 130
        graph_h = 100
        graph_y = 50
        
        # Background for graph
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h), (40,40,40), -1)
        
        if len(self.energy_history) > 1:
            pts = []
            max_e = max(self.energy_history) + 1e-6
            min_e = min(self.energy_history)
            
            for i, val in enumerate(self.energy_history):
                px = int(graph_x + (i / 100.0) * graph_w)
                # Flip Y (0 at bottom)
                norm_val = (val - min_e) / (max_e - min_e)
                py = int((graph_y + graph_h) - (norm_val * graph_h))
                pts.append((px, py))
                
            # Draw line
            pts_arr = np.array(pts, np.int32)
            cv2.polylines(canvas, [pts_arr], False, (0, 255, 255), 2)
            
        # C. Text Stats
        cv2.putText(canvas, f"GEOMETRIC ENERGY: {energy_val:.2f}", (graph_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        status = "RESONANCE" if energy_val < 0.5 else "DISSONANCE (NOISE)"
        col = (0, 255, 0) if energy_val < 0.5 else (0, 0, 255)
        cv2.putText(canvas, status, (graph_x, graph_y + graph_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        self._last_display = canvas

    def get_display_image(self):
        if self._last_display is None: return None
        h, w = self._last_display.shape[:2]
        return QtGui.QImage(self._last_display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)