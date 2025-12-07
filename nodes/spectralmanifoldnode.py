import numpy as np
import cv2
from collections import deque
from scipy.linalg import eigh, svd

# --- STRICT COMPATIBILITY BOILERPLATE ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    # Fallback for testing
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0
        def step(self): pass
        def get_output(self, name): return None
        def get_display_image(self): return None

class SignalSpaceManifoldNode2(BaseNode):
    """
    Signal Space Manifold (The "lol2" Logic)
    ----------------------------------------
    Uses Singular Spectrum Analysis (SSA) / Time-Delay Embedding to 
    reconstruct the high-dimensional 'Shadow Manifold' of the signal.
    
    It calculates the Eigenvectors (Latent Dimensions) of the signal's history.
    
    Visualizes:
    - The 'Hidden Geometry' of the attractor (Projection on PC1 vs PC2)
    - Entropy (How 'glitchy' or complex the signal is)
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Spectral Manifold"
    NODE_COLOR = QtGui.QColor(70, 70, 90) # Dark Slate

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal',  # Connect EEG or Latent Surfer X here
        }
        
        self.outputs = {
            'manifold_view': 'image', # The Geometry
            'entropy': 'signal',      # 0 = Order, 1 = Chaos
            'eigen_1': 'signal',      # Principal Component 1
            'eigen_2': 'signal'       # Principal Component 2
        }
        
        # PARAMETERS
        self.window_size = 60    # Size of the "sliding window" (The Embedding Dimension)
        self.history_len = 200   # How many windows to analyze (The Trajectory length)
        
        # STATE
        # We keep a raw history buffer of the signal
        self.buffer = deque(maxlen=self.history_len + self.window_size)
        
        self.image_size = 256
        self._output_image = None
        self._outs = {}

    def step(self):
        # 1. GET INPUT
        val = self.get_blended_input('signal_in', 'mean')
        
        # Handle empty/None
        if val is None: val = 0.0
        # Sanitize NaN/Inf
        if not np.isfinite(val): val = 0.0
        
        self.buffer.append(float(val))
        
        # Need enough data to fill the trajectory matrix
        if len(self.buffer) < (self.history_len + self.window_size):
            return

        # 2. CONSTRUCT TRAJECTORY MATRIX (Hankel Matrix)
        # We turn the 1D signal into a 2D matrix of "Time-Delayed Windows"
        # This is the "Embedding" step that recovers hidden dimensions.
        
        data = np.array(self.buffer)
        
        # Create matrix X where each row is a window of the signal
        # Shape: (history_len, window_size)
        # This effectively treats time segments as "vectors" in a high-dim space
        X = np.array([data[i : i + self.window_size] for i in range(self.history_len)])
        
        # Center the data (remove mean)
        X_centered = X - np.mean(X, axis=0)
        
        # 3. SINGULAR VALUE DECOMPOSITION (SVD) / PCA
        # We find the "Principal Axes" of this cloud of history.
        # U contains the projection of the data onto the principal components.
        # s contains the singular values (strengths of each component).
        try:
            # We use randomized SVD or standard SVD. For small matrices, standard is fine.
            # We only need the first 2-3 components.
            U, s, Vt = svd(X_centered, full_matrices=False)
            
            # 4. COMPUTE ENTROPY (Complexity)
            # Normalize singular values to get probabilities
            s_norm = s / (np.sum(s) + 1e-9)
            # Shannon entropy of the spectrum
            entropy = -np.sum(s_norm * np.log(s_norm + 1e-9))
            
            # 5. VISUALIZE PROJECTION
            # Project data onto PC1 and PC2 (The two strongest dimensions)
            # These are columns 0 and 1 of U, scaled by s
            
            pc1 = U[:, 0] * s[0]
            pc2 = U[:, 1] * s[1]
            
            # Normalize for display (Fit to screen)
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # Robust scaling
            max_range = max(np.max(np.abs(pc1)), np.max(np.abs(pc2)))
            if max_range < 1e-6: max_range = 1e-6
            
            scale = (self.image_size * 0.4) / max_range
            center = self.image_size // 2
            
            pts = []
            for i in range(len(pc1)):
                x = int(pc1[i] * scale + center)
                y = int(pc2[i] * scale + center)
                pts.append((x, y))
            
            # Draw the path
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    # Gradient color (Time)
                    alpha = i / len(pts)
                    color = (
                        int(255 * (1 - alpha)),   # B
                        int(255 * alpha),         # G
                        255                       # R
                    )
                    cv2.line(img, pts[i-1], pts[i], color, 1)
            
            # Debug info
            cv2.putText(img, f"H: {entropy:.3f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 6. OUTPUTS
            self._output_image = img
            self._outs['manifold_view'] = img
            self._outs['entropy'] = entropy
            self._outs['eigen_1'] = pc1[-1] # Current state in dim 1
            self._outs['eigen_2'] = pc2[-1] # Current state in dim 2
            
        except Exception as e:
            # Fallback if SVD fails (rare, usually 0 data)
            print(f"Manifold Error: {e}")
            self._output_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def get_output(self, name):
        return self._outs.get(name)

    def get_display_image(self):
        return self._output_image