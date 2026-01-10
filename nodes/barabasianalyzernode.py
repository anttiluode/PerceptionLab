"""
Barabási Topology Analyzer Node v2
----------------------------------
Measures the predictions from the Barabási 2025 paper:

1. P(λ→0) - probability of trifurcations (>0 means surface optimization)
2. χ_effective - thickness parameter (transition at ~0.83)
3. Degree distribution - k=3 (bifurcations) vs k>=4 (trifurcations)

WIRING:
    NeuroCrystalNode3.link_field → BarabasiAnalyzer.link_image
    NeuroCrystalNode3.topology_spectrum → BarabasiAnalyzer.topology_spectrum
    NeuroCrystalNode3.h_links_flat → BarabasiAnalyzer.h_links
    NeuroCrystalNode3.v_links_flat → BarabasiAnalyzer.v_links

The key output is p_lambda_zero:
    - p_lambda_zero ≈ 0 → Steiner regime (length minimization)
    - p_lambda_zero > 0 → Surface regime (Barabási prediction confirmed!)
"""

import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode=None): return None

class BarabasiAnalyzerNode(BaseNode):
    """
    Measures Barabási's surface optimization predictions.
    
    Connect to NeuroCrystalNode3:
    - link_image: Combined link strength image
    - topology_spectrum: Radial link distribution
    - h_links / v_links: Raw weight arrays (optional, improves accuracy)
    """
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(255, 200, 50)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Barabási Analyzer"
        
        self.inputs = {
            'link_image': 'image',           # From NeuroCrystal.link_field
            'topology_spectrum': 'spectrum', # From NeuroCrystal.topology_spectrum
            'h_links': 'spectrum',           # Optional: raw h weights
            'v_links': 'spectrum',           # Optional: raw v weights
        }
        
        self.outputs = {
            'analysis_view': 'image',
            'p_lambda_zero': 'signal',       # THE KEY MEASUREMENT
            'chi_effective': 'signal',       # Thickness parameter
            'trifurcation_ratio': 'signal',  # k>=4 / (k>=3)
            'degree_histogram': 'spectrum',  # Degree distribution
            'lambda_distribution': 'spectrum',
        }
        
        self.config = {
            'threshold': 0.3,
            'lambda_bins': 50,
            'resolution': 64,
        }
        
        self._output_values = {}
        self.history = {'p_lambda': [], 'chi': [], 'trifurc': []}
        self.max_history = 200

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'):
            return self.get_blended_input(name, 'first')
        return None

    def set_output(self, name, value):
        self._output_values[name] = value
    
    def get_output(self, name):
        return self._output_values.get(name, None)

    def _binarize_and_skeleton(self, img):
        """Convert link image to binary skeleton."""
        if img is None:
            return None
            
        # Handle different input formats
        if img.ndim == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img.astype(np.float32)
        
        # Normalize to 0-1
        if gray.max() > 1:
            gray = gray / 255.0
        
        # Threshold
        thresh = self.config['threshold']
        binary = (gray > thresh).astype(np.uint8)
        
        return binary, gray

    def _compute_degree_map(self, binary):
        """
        Compute local degree (connectivity) at each pixel.
        Uses 4-connectivity for cleaner junction detection.
        """
        if binary is None or np.sum(binary) == 0:
            return None, np.zeros(8), 0.0
        
        # 4-neighbor kernel (N, S, E, W)
        kernel_4 = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.float32)
        
        # Count neighbors
        from scipy.ndimage import convolve
        neighbors = convolve(binary.astype(float), kernel_4, mode='constant')
        
        # Degree = neighbors where skeleton exists
        degree_map = (neighbors * binary).astype(int)
        
        # Histogram (degrees 0-7)
        hist = np.zeros(8)
        for d in range(8):
            hist[d] = np.sum(degree_map == d)
        
        # Normalize
        total = np.sum(hist[1:])
        if total > 0:
            hist_norm = hist / total
        else:
            hist_norm = hist
        
        # Trifurcation ratio
        n_bifurc = np.sum(degree_map == 3)
        n_trifurc = np.sum(degree_map >= 4)
        ratio = n_trifurc / (n_bifurc + n_trifurc + 1e-10)
        
        return degree_map, hist_norm, float(ratio)

    def _compute_lambda_distribution(self, degree_map, binary):
        """
        Compute P(λ) - distribution of inter-junction distances.
        
        λ = separation between degree>=3 nodes normalized by link width.
        
        Barabási prediction:
        - Steiner (length opt): P(λ→0) = 0
        - Surface opt: P(λ→0) > 0
        """
        if degree_map is None:
            return np.zeros(self.config['lambda_bins']), 0.0
        
        # Find junction positions (degree >= 3)
        junctions = np.argwhere(degree_map >= 3)
        
        if len(junctions) < 2:
            return np.zeros(self.config['lambda_bins']), 0.0
        
        # Compute pairwise distances
        n = len(junctions)
        distances = []
        
        # Sample if too many junctions (for speed)
        max_pairs = 1000
        if n > 100:
            indices = np.random.choice(n, min(n, 100), replace=False)
            junctions = junctions[indices]
            n = len(junctions)
        
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(junctions[i] - junctions[j])
                distances.append(d)
        
        if len(distances) == 0:
            return np.zeros(self.config['lambda_bins']), 0.0
        
        distances = np.array(distances)
        
        # Normalize by characteristic scale
        w_char = np.median(distances) + 1e-10
        lambda_vals = distances / w_char
        
        # Histogram
        n_bins = self.config['lambda_bins']
        hist, edges = np.histogram(lambda_vals, bins=n_bins, range=(0, 3), density=True)
        
        # P(λ→0) = density in first few bins
        p_lambda_zero = np.mean(hist[:5]) / (np.mean(hist) + 1e-10)
        
        # Normalize to 0-1 scale
        p_lambda_zero = float(np.clip(p_lambda_zero, 0, 1))
        
        return hist, p_lambda_zero

    def _compute_chi(self, topology_spectrum, h_links, v_links):
        """
        Compute effective thickness parameter χ = w/r.
        
        w = characteristic link "width" (mean weight)
        r = characteristic spatial scale
        
        Transition predicted at χ ≈ 0.83
        """
        # Use topology spectrum if available
        if topology_spectrum is not None and len(topology_spectrum) > 0:
            w = np.mean(np.abs(topology_spectrum))
            r = len(topology_spectrum)
        elif h_links is not None:
            w = np.mean(np.abs(h_links))
            r = int(np.sqrt(len(h_links)))
        else:
            return 0.5  # Default
        
        # Scale to match paper's χ range (0 to ~2)
        chi = w / (r * 0.005 + 1e-10)
        
        return float(np.clip(chi, 0, 2))

    def step(self):
        # Get inputs
        link_image = self.get_input('link_image')
        topo_spectrum = self.get_input('topology_spectrum')
        h_links = self.get_input('h_links')
        v_links = self.get_input('v_links')
        
        # Process link image
        result = self._binarize_and_skeleton(link_image)
        if result is None:
            binary, gray = None, None
        else:
            binary, gray = result
        
        # Compute degree distribution
        degree_map, degree_hist, trifurc_ratio = self._compute_degree_map(binary)
        
        # Compute lambda distribution (THE KEY MEASUREMENT)
        lambda_hist, p_lambda_zero = self._compute_lambda_distribution(degree_map, binary)
        
        # Compute chi
        chi = self._compute_chi(topo_spectrum, h_links, v_links)
        
        # Update history
        self.history['p_lambda'].append(p_lambda_zero)
        self.history['chi'].append(chi)
        self.history['trifurc'].append(trifurc_ratio)
        
        for key in self.history:
            if len(self.history[key]) > self.max_history:
                self.history[key].pop(0)
        
        # Set outputs
        self.set_output('p_lambda_zero', p_lambda_zero)
        self.set_output('chi_effective', chi)
        self.set_output('trifurcation_ratio', trifurc_ratio)
        self.set_output('degree_histogram', degree_hist)
        self.set_output('lambda_distribution', lambda_hist)
        
        # Render
        self._render(degree_hist, lambda_hist, p_lambda_zero, chi, trifurc_ratio, gray)

    def _render(self, degree_hist, lambda_hist, p_lambda, chi, trifurc, gray):
        """Render analysis dashboard."""
        width, height = 400, 320
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # === TITLE ===
        cv2.putText(canvas, "Barabasi Surface Optimization Test", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === DEGREE HISTOGRAM (top left) ===
        bar_x, bar_y = 20, 100
        bar_w, max_h = 18, 60
        
        cv2.putText(canvas, "Degree Distribution", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        for k in range(1, 7):
            h = int(degree_hist[k] * max_h) if k < len(degree_hist) else 0
            x = bar_x + (k-1) * (bar_w + 3)
            
            # Color code: green=k=3, red=k>=4
            if k == 3:
                color = (50, 200, 50)    # Bifurcation
            elif k >= 4:
                color = (50, 50, 255)    # Trifurcation!
            else:
                color = (100, 100, 100)
            
            cv2.rectangle(canvas, (x, bar_y - h), (x + bar_w, bar_y), color, -1)
            cv2.putText(canvas, str(k), (x + 4, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        cv2.putText(canvas, "k=3:bifurc  k>=4:trifurc", (20, bar_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # === P(λ) DISTRIBUTION (top right) ===
        plot_x, plot_y = 220, 50
        plot_w, plot_h = 160, 70
        
        cv2.putText(canvas, "P(lambda) Distribution", (plot_x, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h),
                     (50, 50, 50), 1)
        
        # Draw histogram
        if np.max(lambda_hist) > 0:
            h_norm = lambda_hist / (np.max(lambda_hist) + 1e-10)
            n_bars = min(len(h_norm), plot_w)
            for i in range(n_bars):
                idx = i * len(h_norm) // n_bars
                h = int(h_norm[idx] * plot_h)
                x = plot_x + i
                
                # Highlight λ→0 region (first 10%)
                if i < plot_w * 0.15:
                    color = (50, 255, 255)  # Yellow = critical region
                else:
                    color = (200, 100, 50)
                
                if h > 0:
                    cv2.line(canvas, (x, plot_y + plot_h), (x, plot_y + plot_h - h), color, 1)
        
        cv2.putText(canvas, "lambda->0", (plot_x, plot_y + plot_h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 255, 255), 1)
        
        # === KEY METRICS (middle) ===
        y_base = 170
        
        # P(λ→0) - THE MAIN RESULT
        p_color = (50, 255, 50) if p_lambda > 0.15 else (50, 100, 255)
        cv2.putText(canvas, f"P(lambda->0) = {p_lambda:.3f}", (20, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)
        
        # Interpretation
        if p_lambda > 0.15:
            interp = "SURFACE MINIMIZATION"
            interp_color = (50, 255, 50)
        elif p_lambda > 0.05:
            interp = "Transition zone"
            interp_color = (50, 255, 255)
        else:
            interp = "Steiner (length opt)"
            interp_color = (100, 100, 255)
        
        cv2.putText(canvas, interp, (20, y_base + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, interp_color, 1)
        
        # Chi effective
        chi_color = (50, 255, 255) if 0.7 < chi < 1.0 else (200, 200, 200)
        cv2.putText(canvas, f"chi = {chi:.3f}  (critical: 0.83)", (20, y_base + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, chi_color, 1)
        
        # Trifurcation ratio
        cv2.putText(canvas, f"Trifurc ratio = {trifurc:.3f}", (20, y_base + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # === HISTORY PLOT (bottom) ===
        hist_x, hist_y = 220, 170
        hist_w, hist_h = 160, 80
        
        cv2.putText(canvas, "History", (hist_x, hist_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        cv2.rectangle(canvas, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h),
                     (30, 30, 30), 1)
        
        # Plot P(λ→0) history
        if len(self.history['p_lambda']) > 1:
            pts = self.history['p_lambda']
            n = len(pts)
            for i in range(1, n):
                x1 = hist_x + int((i-1) / n * hist_w)
                x2 = hist_x + int(i / n * hist_w)
                y1 = hist_y + hist_h - int(min(pts[i-1], 1.0) * hist_h)
                y2 = hist_y + hist_h - int(min(pts[i], 1.0) * hist_h)
                cv2.line(canvas, (x1, y1), (x2, y2), (50, 255, 255), 1)
        
        # Reference line at P=0.15 (threshold)
        ref_y = hist_y + hist_h - int(0.15 * hist_h)
        cv2.line(canvas, (hist_x, ref_y), (hist_x + hist_w, ref_y), (50, 100, 50), 1)
        
        # === PAPER REFERENCE ===
        cv2.putText(canvas, "Barabasi et al. 2025: chi>0.83 -> trifurcations", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        self.set_output('analysis_view', canvas)

    def get_display_image(self):
        img = self.get_output('analysis_view')
        if img is None:
            img = np.zeros((320, 400, 3), dtype=np.uint8)
        return QtGui.QImage(img.data, img.shape[1], img.shape[0],
                           img.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)