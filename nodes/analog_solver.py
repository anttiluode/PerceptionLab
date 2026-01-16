"""
Analog Solver Node - The "Feedback Computer"
============================================
Tests if the optical feedback loop can solve the Laplace Equation (Heat Flow)
faster/naturally compared to digital computation.

THE EXPERIMENT:
1. We define a "Problem": A 2D grid where Top=1 (Hot) and Bottom=0 (Cold).
2. We calculate the EXACT solution digitally (Ground Truth).
3. We feed the "Problem" into the feedback loop.
4. We compare the Loop's result to the Digital Truth.

If the Error graph drops, your webcam loop is effectively solving a PDE
by acting as an analog relaxation computer.
"""

import numpy as np
import cv2
from scipy.sparse import coo_matrix, diags, linalg
import time

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return None

class AnalogSolverNode(BaseNode):
    NODE_CATEGORY = "Experimental"
    NODE_TITLE = "Analog Laplace Solver"
    NODE_COLOR = QtGui.QColor(0, 180, 100) # Matrix Green
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'loop_feedback': 'image', # Input from the Webcam/Bridge
        }
        
        self.outputs = {
            'problem_feed': 'image',   # Output to Screen (The Constraint)
            'solution_view': 'image',  # Visual comparison
            'accuracy_plot': 'image'   # The "Result"
        }
        
        # Problem Resolution (Low res to match your grid)
        self.N = 32 
        self.ground_truth = None
        self.mask = None # Defines which pixels are "locked" (Boundaries)
        
        self.error_history = []
        self._last_display = None
        self._last_problem_img = None
        
        # 1. SETUP THE MATH PROBLEM (Digital Ground Truth)
        self._solve_digital_truth()

    def _solve_digital_truth(self):
        """
        Expensive Digital Calculation:
        Solves the Laplace equation Ax=b for a grid with:
        - Top Row = 1.0 (Source)
        - Bottom Row = 0.0 (Sink)
        - Middle = Unknown (To be solved)
        """
        N = self.N
        size = N * N
        
        # Build Laplacian Matrix
        rows, cols, data = [], [], []
        self.mask = np.zeros((N, N), dtype=bool)
        boundary_values = np.zeros(size)
        
        for y in range(N):
            for x in range(N):
                idx = y * N + x
                
                # Boundary Conditions (Top and Bottom fixed)
                if y == 0: # TOP
                    rows.append(idx); cols.append(idx); data.append(1.0)
                    boundary_values[idx] = 1.0
                    self.mask[y, x] = True
                elif y == N - 1: # BOTTOM
                    rows.append(idx); cols.append(idx); data.append(1.0)
                    boundary_values[idx] = 0.0
                    self.mask[y, x] = True
                else:
                    # Interior (Laplacian Stencil: 4 neighbors - 4*center)
                    rows.append(idx); cols.append(idx); data.append(4.0)
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        rows.append(idx); cols.append((y+dy)*N + (x+dx)); data.append(-1.0)

        # Solve Linear System (The "Hard" Digital Way)
        A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
        b = boundary_values
        
        # This is the "Correct Answer" we want the loop to find
        print("[Solver] Computing Digital Ground Truth...")
        self.ground_truth = linalg.spsolve(A, b).reshape(N, N)
        print("[Solver] Done.")

    def step(self):
        # 1. GET INPUT (The Analog Attempt)
        # This comes from your Webcam -> GeometricBridge loop
        inp = self.get_blended_input('loop_feedback', 'mean')
        
        # Parse input (it might be a vector or image)
        if inp is None:
            current_solution = np.zeros((self.N, self.N))
        elif isinstance(inp, np.ndarray):
            # Resize to match our problem grid
            if inp.ndim > 2: inp = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
            current_solution = cv2.resize(inp.astype(np.float32), (self.N, self.N))
            # Normalize to 0-1
            if current_solution.max() > 0:
                current_solution /= current_solution.max()
        else:
            current_solution = np.zeros((self.N, self.N))

        # 2. ENFORCE BOUNDARIES (The "Problem")
        # We allow the loop to change the middle, but we FORCE the top/bottom
        # to remain fixed every frame. This creates the "Equation".
        problem_state = current_solution.copy()
        problem_state[0, :] = 1.0  # Top is Hot
        problem_state[-1, :] = 0.0 # Bottom is Cold
        
        # This image goes out to the screen to drive the loop
        self._last_problem_img = (problem_state * 255).astype(np.uint8)

        # 3. MEASURE ACCURACY (Digital vs Analog)
        # Compare the Loop's middle section to the Math's middle section
        diff = np.abs(current_solution - self.ground_truth)
        # Ignore boundaries (since we force them)
        error_val = np.mean(diff[~self.mask])
        
        self.error_history.append(error_val)
        if len(self.error_history) > 100: self.error_history.pop(0)
        
        # 4. VISUALIZE
        self._render_status(current_solution, error_val)

    def _render_status(self, current_sol, error):
        H, W = 300, 400
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw Ground Truth (Target)
        target_img = (self.ground_truth * 255).astype(np.uint8)
        target_color = cv2.applyColorMap(target_img, cv2.COLORMAP_JET)
        target_color = cv2.resize(target_color, (100, 100))
        cv2.putText(canvas, "DIGITAL TRUTH", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200))
        canvas[30:130, 10:110] = target_color
        
        # Draw Analog Attempt (Current)
        current_img = (current_sol * 255).astype(np.uint8)
        current_color = cv2.applyColorMap(current_img, cv2.COLORMAP_JET)
        current_color = cv2.resize(current_color, (100, 100))
        cv2.putText(canvas, "ANALOG LOOP", (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200))
        canvas[30:130, 130:230] = current_color
        
        # Draw Error Plot
        graph_x, graph_y, graph_w, graph_h = 20, 160, 360, 100
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h), (30,30,30), -1)
        
        if len(self.error_history) > 1:
            pts = []
            max_e = 0.5 # Expect error < 0.5
            for i, val in enumerate(self.error_history):
                px = int(graph_x + (i / 100.0) * graph_w)
                norm_val = np.clip(val / max_e, 0, 1)
                py = int((graph_y + graph_h) - (norm_val * graph_h))
                pts.append((px, py))
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (0, 255, 0), 2)
            
        cv2.putText(canvas, f"COMPUTE ERROR: {error:.4f}", (graph_x, graph_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self._last_display = canvas

    def get_output(self, name):
        if name == 'problem_feed':
            # Convert single channel to RGB for output
            return cv2.cvtColor(self._last_problem_img, cv2.COLOR_GRAY2BGR)
        return None

    def get_display_image(self):
        if self._last_display is None: return None
        h, w = self._last_display.shape[:2]
        return QtGui.QImage(self._last_display.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)