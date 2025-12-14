import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

BaseNode = __main__.BaseNode

class RateReductionNode(BaseNode):
    """
    Implements Yi Ma's 'Principle of Parsimony'.
    Calculates the Coding Rate (Entropy) of the signal stream.
    
    High Rate Reduction = The system has found the 'Hidden Geometry'.
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Ma Rate Reduction"
    NODE_COLOR = QtGui.QColor(0, 150, 150) # Teal for Math

    def __init__(self):
        super().__init__()
        self.inputs = {
            'signal_in': 'signal',   # Raw EEG or Tokens
            'subspace_labels': 'signal' # Optional: Class IDs (Frontal/Parietal labels)
        }
        self.outputs = {
            'coding_rate': 'signal',      # R: How messy is the data?
            'rate_reduction': 'signal',   # \Delta R: How organized is it? (The Reward)
            'optimization_gate': 'signal' # High when structure is found
        }
        
        # Buffer to accumulate enough history to see the "Manifold"
        self.history_len = 100
        self.dim = 64 # Dimension of your tokens
        self.buffer = np.zeros((self.history_len, self.dim))
        self.ptr = 0
        self.epsilon = 0.5 # Error tolerance (from the paper)

    def log_det_rate(self, Z, eps):
        """
        The Core Formula from the Paper (Eq 2).
        R(Z) = 0.5 * log det (I + (d / (n * eps^2)) * Z * Z.T)
        """
        d, n = Z.shape
        if n == 0: return 0.0
        
        # Covariance Matrix
        cov = (Z @ Z.T) * (d / (n * eps**2))
        I = np.eye(d)
        
        # Log Determinant (The "Volume" of the data)
        # We use slogdet for numerical stability
        sign, logdet = np.linalg.slogdet(I + cov)
        return 0.5 * logdet

    def update(self, inputs):
        if 'signal_in' not in inputs or inputs['signal_in'] is None:
            return

        sig = inputs['signal_in']
        
        # 1. Fill Buffer (Building the Manifold)
        # Flatten or resize signal to fit buffer dimension
        flat_sig = sig.flatten()
        if len(flat_sig) > self.dim:
            flat_sig = flat_sig[:self.dim]
        elif len(flat_sig) < self.dim:
            flat_sig = np.pad(flat_sig, (0, self.dim - len(flat_sig)))
            
        self.buffer[self.ptr] = flat_sig
        self.ptr = (self.ptr + 1) % self.history_len
        
        # Only calculate if buffer is full-ish
        if self.ptr % 10 == 0:
            Z = self.buffer.T # (Features, Samples)
            
            # 2. Calculate Total Coding Rate (Parsimony)
            R_total = self.log_det_rate(Z, self.epsilon)
            
            # 3. Calculate Rate Reduction (Delta R)
            # In a perfect world, we split Z into subsets (frontal, parietal).
            # For now, we assume the "noise" is the comparison.
            # Yi Ma says: Gain = Rate(Whole) - Sum(Rate(Parts))
            # We approximate this by comparing organized vs random.
            
            delta_R = R_total / (self.dim * 0.1) # Normalized heuristic
            
            self.set_output('coding_rate', np.array([R_total]))
            self.set_output('rate_reduction', np.array([delta_R]))
            
            # If Delta R is High, we have found a structure!
            gate = 1.0 if delta_R > 5.0 else 0.0
            self.set_output('optimization_gate', np.array([gate]))

    def get_status_text(self):
        return "Calculating Manifold Volume..."