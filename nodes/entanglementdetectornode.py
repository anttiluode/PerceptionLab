"""
Entanglement Detector Node - Detects correlations between coupled systems
Measures mutual information and correlation to detect entanglement-like behavior
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class EntanglementDetectorNode(BaseNode):
    """
    Detects entanglement-like correlations between two quantum-like states.
    Uses mutual information and correlation metrics.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(200, 100, 200)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Entanglement Detector"
        
        self.inputs = {
            'state_a': 'spectrum',
            'state_b': 'spectrum'
        }
        self.outputs = {
            'entanglement': 'signal',  # 0-1 (0=separable, 1=maximally entangled)
            'correlation': 'signal',  # Pearson correlation
            'mutual_info': 'signal',  # Mutual information (bits)
            'concurrence': 'signal'  # Entanglement measure
        }
        
        self.history_a = []
        self.history_b = []
        self.max_history = 100
        
        # Initialize to valid values
        self.entanglement_value = 0.0
        self.correlation_value = 0.0
        self.mutual_info_value = 0.0
        self.concurrence_value = 0.0
        
    def step(self):
        state_a = self.get_blended_input('state_a', 'first')
        state_b = self.get_blended_input('state_b', 'first')
        
        if state_a is None or state_b is None:
            return
            
        # Ensure same dimensionality
        min_dim = min(len(state_a), len(state_b))
        state_a = state_a[:min_dim]
        state_b = state_b[:min_dim]
        
        # Store history
        self.history_a.append(state_a.copy())
        self.history_b.append(state_b.copy())
        
        if len(self.history_a) > self.max_history:
            self.history_a.pop(0)
            self.history_b.pop(0)
            
        if len(self.history_a) < 10:
            return  # Need more data
            
        # Compute metrics
        history_a_array = np.array(self.history_a)
        history_b_array = np.array(self.history_b)
        
        # 1. Correlation (Pearson) - WITH NaN HANDLING
        # Flatten time series and compute correlation
        flat_a = history_a_array.flatten()
        flat_b = history_b_array.flatten()
        
        if len(flat_a) > 1 and len(flat_b) > 1:
            # Check for constant arrays (which cause NaN in corrcoef)
            if np.std(flat_a) < 1e-9 or np.std(flat_b) < 1e-9:
                self.correlation_value = 0.0
            else:
                corr_matrix = np.corrcoef(flat_a, flat_b)
                self.correlation_value = corr_matrix[0, 1]
                # Handle NaN
                if np.isnan(self.correlation_value):
                    self.correlation_value = 0.0
        else:
            self.correlation_value = 0.0
            
        # 2. Mutual Information (simplified) - WITH SAFETY
        # Discretize states and compute MI
        bins = 10
        hist_a, _ = np.histogram(flat_a, bins=bins)
        hist_b, _ = np.histogram(flat_b, bins=bins)
        hist_joint, _, _ = np.histogram2d(flat_a, flat_b, bins=bins)
        
        # Normalize to probabilities
        p_a = hist_a / (hist_a.sum() + 1e-9)
        p_b = hist_b / (hist_b.sum() + 1e-9)
        p_joint = hist_joint / (hist_joint.sum() + 1e-9)
        
        # MI = sum p(a,b) log(p(a,b) / (p(a)p(b)))
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_joint[i, j] > 1e-9 and p_a[i] > 1e-9 and p_b[j] > 1e-9:
                    mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_a[i] * p_b[j]))
                    
        self.mutual_info_value = max(0.0, mi)
        if np.isnan(self.mutual_info_value):
            self.mutual_info_value = 0.0
        
        # 3. Concurrence (entanglement measure) - WITH NaN HANDLING
        # Simplified: based on covariance matrix
        cov_matrix = np.cov(history_a_array.T, history_b_array.T)
        
        # Extract cross-covariance block
        n = history_a_array.shape[1]
        if cov_matrix.shape[0] >= 2*n:  # Safety check
            cross_cov = cov_matrix[:n, n:]
            self.concurrence_value = np.abs(np.trace(cross_cov)) / (n + 1e-9)
        else:
            self.concurrence_value = 0.0
            
        if np.isnan(self.concurrence_value):
            self.concurrence_value = 0.0
        
        # 4. Overall entanglement metric
        # Combination of correlation, MI, and concurrence
        self.entanglement_value = (
            abs(self.correlation_value) * 0.4 +
            min(self.mutual_info_value, 1.0) * 0.3 +
            min(self.concurrence_value, 1.0) * 0.3
        )
        
        # Final NaN check
        if np.isnan(self.entanglement_value):
            self.entanglement_value = 0.0
        
    def get_output(self, port_name):
        if port_name == 'entanglement':
            return float(self.entanglement_value)
        elif port_name == 'correlation':
            return float(self.correlation_value)
        elif port_name == 'mutual_info':
            return float(self.mutual_info_value)
        elif port_name == 'concurrence':
            return float(self.concurrence_value)
        return None
        
    def get_display_image(self):
        """Visualize entanglement metrics"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Helper for NaN/Inf safety
        def safe_val(v):
            return 0.0 if (np.isnan(v) or np.isinf(v)) else v
            
        # Draw correlation plot (recent history)
        if len(self.history_a) > 1:
            recent = min(50, len(self.history_a))
            
            for i in range(1, recent):
                # Plot state_a vs state_b (first dimension)
                x1 = int((safe_val(self.history_a[-i][0]) + 1) / 2 * w)
                y1 = int((safe_val(self.history_b[-i][0]) + 1) / 2 * h)
                x2 = int((safe_val(self.history_a[-i+1][0]) + 1) / 2 * w)
                y2 = int((safe_val(self.history_b[-i+1][0]) + 1) / 2 * h)
                
                x1 = np.clip(x1, 0, w-1)
                y1 = np.clip(y1, 0, h-1)
                x2 = np.clip(x2, 0, w-1)
                y2 = np.clip(y2, 0, h-1)
                
                alpha = i / recent
                color_val = int(255 * alpha)
                cv2.line(img, (x1, y1), (x2, y2), (color_val, 0, 255 - color_val), 1)
        
        # Entanglement indicator - WITH NaN SAFETY
        ent_val = safe_val(self.entanglement_value)
        
        ent_text = "ENTANGLED" if ent_val > 0.7 else "SEPARABLE" if ent_val < 0.3 else "MIXED"
        ent_color = (255, 0, 255) if ent_val > 0.7 else (0, 255, 0) if ent_val < 0.3 else (255, 255, 0)
        
        cv2.putText(img, ent_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ent_color, 2)
        
        # Metrics - WITH NaN SAFETY
        cv2.putText(img, f"Ent: {safe_val(self.entanglement_value):.3f}", (10, h-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"Cor: {safe_val(self.correlation_value):.3f}", (10, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"MI:  {safe_val(self.mutual_info_value):.3f}", (10, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"Con: {safe_val(self.concurrence_value):.3f}", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Entanglement bar - WITH NaN SAFETY
        ent_width = int(np.clip(safe_val(ent_val), 0.0, 1.0) * w)
        cv2.rectangle(img, (0, h-80), (ent_width, h-75), ent_color, -1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)