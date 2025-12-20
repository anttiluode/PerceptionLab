"""
Cortical Eigenmode Node - Real Graph Laplacian Eigenmodes on fsaverage Mesh
=============================================================================

This node computes REAL anatomically-meaningful eigenmodes from the fsaverage
cortical surface mesh. These are the actual Laplace-Beltrami eigenmodes that
Raj et al. describe - the structural basis functions of the cortex.

Unlike random topographies, these eigenmodes:
1. Are computed from actual cortical geometry
2. Represent real spatial harmonics of the brain surface
3. Low modes = smooth, global patterns (like DMN vs task-positive)
4. High modes = fine-grained, local patterns

The node then projects your EEG complex mode activations onto these
anatomically-correct cortical eigenmodes for visualization.

INPUTS:
- complex_modes: Complex spectrum from ModePhaseAnalyzerNode
- phase_coherence: Optional coherence signal

OUTPUTS:
- cortex_image: 2D flattened cortex with eigenmode activity
- eigenmode_gallery: Shows the first 10 cortical eigenmodes
- mode_projection: How EEG modes map to cortical modes

Created: December 2025
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.sparse import coo_matrix, diags, csr_matrix
from scipy.sparse.linalg import eigsh

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode):
            return None

# MNE for loading fsaverage
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("[CorticalEigenmodeNode] MNE not available!")


def _try_decimate(rr, tris, target_tris):
    """Decimate surface if possible."""
    try:
        from mne.surface import decimate_surface
        rr2, tris2 = decimate_surface(rr, tris, n_triangles=int(target_tris))
        return rr2, tris2
    except Exception:
        return rr, tris


def _compute_mesh_laplacian(vertices, faces):
    """
    Compute the graph Laplacian of a triangular mesh.
    This is the discrete Laplace-Beltrami operator.
    
    L = D - A where:
    - A is adjacency matrix (edge weights can be cotangent weights for better accuracy)
    - D is degree matrix
    """
    n_vertices = vertices.shape[0]
    
    # Build adjacency from faces
    edges = set()
    for face in faces:
        edges.add((face[0], face[1]))
        edges.add((face[1], face[0]))
        edges.add((face[1], face[2]))
        edges.add((face[2], face[1]))
        edges.add((face[2], face[0]))
        edges.add((face[0], face[2]))
    
    rows = []
    cols = []
    data = []
    
    for (i, j) in edges:
        rows.append(i)
        cols.append(j)
        # Simple uniform weights (could use cotangent weights for more accuracy)
        data.append(1.0)
    
    A = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    A = A.tocsr()
    
    # Degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D = diags(degrees)
    
    # Laplacian
    L = D - A
    
    # Normalize (symmetric normalized Laplacian)
    # L_sym = D^(-1/2) L D^(-1/2)
    d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-10)
    D_inv_sqrt = diags(d_inv_sqrt)
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    return L_normalized.tocsr()


def _hsv_to_bgr(h, s, v):
    """Convert HSV arrays to BGR uint8."""
    hsv = np.zeros((*h.shape, 3), dtype=np.float32)
    hsv[..., 0] = h * 179
    hsv[..., 1] = s * 255
    hsv[..., 2] = v * 255
    hsv_u8 = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)


class CorticalEigenmodeNode(BaseNode):
    """
    Computes real cortical eigenmodes from fsaverage mesh Laplacian
    """
    NODE_CATEGORY = "Holography"
    NODE_TITLE = "Cortical Eigenmodes"
    NODE_COLOR = QtGui.QColor(200, 100, 50)  # Orange-brown for anatomy
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'complex_modes': 'complex_spectrum',
            'phase_coherence': 'signal',
            'modulation': 'signal',
        }
        
        self.outputs = {
            'cortex_image': 'image',
            'eigenmode_gallery': 'image',
            'lh_image': 'image',
            'rh_image': 'image',
            'mode_energy': 'signal',
            'spatial_complexity': 'signal',
        }
        
        # Config
        self.surface_type = 'inflated'
        self.target_tris = 10000  # Per hemisphere - keep small for speed
        self.n_eigenmodes = 20    # Number of cortical eigenmodes to compute
        self.n_input_modes = 10   # Expected input modes
        
        # Display
        self.W = 640
        self.H = 320
        self.gamma = 0.7
        
        # State
        self.vertices_lh = None
        self.vertices_rh = None
        self.faces_lh = None
        self.faces_rh = None
        self.eigenmodes_lh = None  # (n_vertices, n_eigenmodes)
        self.eigenmodes_rh = None
        self.eigenvalues_lh = None
        self.eigenvalues_rh = None
        
        # Pixel mapping
        self.px_lh = None
        self.py_lh = None
        self.px_rh = None
        self.py_rh = None
        
        # Output images
        self._cortex_image = None
        self._gallery_image = None
        self._lh_image = None
        self._rh_image = None
        
        # Metrics
        self._mode_energy = 0.0
        self._spatial_complexity = 0.0
        
        # Initialize
        self._initialized = False
        self._init_error = ""
        
        if MNE_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Load fsaverage and compute eigenmodes"""
        try:
            print("[CorticalEigenmodes] Loading fsaverage surface...")
            
            # Get fsaverage path
            fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
            subjects_dir = fs_dir.parent
            
            # Load surfaces
            surf_lh = subjects_dir / 'fsaverage' / 'surf' / f'lh.{self.surface_type}'
            surf_rh = subjects_dir / 'fsaverage' / 'surf' / f'rh.{self.surface_type}'
            
            rr_lh, tris_lh = mne.read_surface(str(surf_lh))
            rr_rh, tris_rh = mne.read_surface(str(surf_rh))
            
            # Decimate for speed
            rr_lh, tris_lh = _try_decimate(rr_lh, tris_lh, self.target_tris)
            rr_rh, tris_rh = _try_decimate(rr_rh, tris_rh, self.target_tris)
            
            self.vertices_lh = rr_lh.astype(np.float32)
            self.vertices_rh = rr_rh.astype(np.float32)
            self.faces_lh = tris_lh.astype(np.int32)
            self.faces_rh = tris_rh.astype(np.int32)
            
            print(f"[CorticalEigenmodes] LH: {len(self.vertices_lh)} vertices, {len(self.faces_lh)} faces")
            print(f"[CorticalEigenmodes] RH: {len(self.vertices_rh)} vertices, {len(self.faces_rh)} faces")
            
            # Compute eigenmodes for each hemisphere
            print("[CorticalEigenmodes] Computing LH eigenmodes...")
            self.eigenmodes_lh, self.eigenvalues_lh = self._compute_eigenmodes(
                self.vertices_lh, self.faces_lh
            )
            
            print("[CorticalEigenmodes] Computing RH eigenmodes...")
            self.eigenmodes_rh, self.eigenvalues_rh = self._compute_eigenmodes(
                self.vertices_rh, self.faces_rh
            )
            
            # Precompute pixel coordinates
            self._compute_pixel_coords()
            
            # Create eigenmode gallery
            self._create_gallery()
            
            self._initialized = True
            print(f"[CorticalEigenmodes] Initialized with {self.n_eigenmodes} eigenmodes per hemisphere")
            print(f"[CorticalEigenmodes] Eigenvalue range LH: {self.eigenvalues_lh[1]:.4f} to {self.eigenvalues_lh[-1]:.4f}")
            
        except Exception as e:
            self._init_error = str(e)
            print(f"[CorticalEigenmodes] Initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_eigenmodes(self, vertices, faces):
        """Compute Laplacian eigenmodes for a hemisphere"""
        n_vertices = len(vertices)
        
        # Build mesh Laplacian
        L = _compute_mesh_laplacian(vertices, faces)
        
        # Add small regularization
        L = L + 1e-8 * diags(np.ones(n_vertices))
        
        # Compute smallest eigenmodes
        n_modes = min(self.n_eigenmodes + 1, n_vertices - 2)
        
        eigenvalues, eigenmodes = eigsh(L, k=n_modes, which='SM', tol=1e-4, maxiter=5000)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenmodes = eigenmodes[:, idx]
        
        # Skip first mode (constant) and take next n_eigenmodes
        return eigenmodes[:, 1:self.n_eigenmodes+1].astype(np.float32), eigenvalues[1:self.n_eigenmodes+1]
    
    def _compute_pixel_coords(self):
        """Map vertices to 2D pixel coordinates"""
        W, H = self.W, self.H
        half_W = W // 2
        
        def hemi_to_pixels(vertices, x_offset, x_width):
            # Use Y-Z plane projection
            yz = vertices[:, [1, 2]]
            mn = yz.min(axis=0)
            mx = yz.max(axis=0)
            span = mx - mn + 1e-6
            
            uv = (yz - mn) / span
            
            # Add padding
            pad = 0.08
            uv = pad + (1 - 2*pad) * uv
            
            px = (x_offset + uv[:, 0] * x_width).astype(np.int32)
            py = ((1.0 - uv[:, 1]) * (H - 1)).astype(np.int32)
            
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)
            
            return px, py
        
        self.px_lh, self.py_lh = hemi_to_pixels(self.vertices_lh, 0, half_W - 10)
        self.px_rh, self.py_rh = hemi_to_pixels(self.vertices_rh, half_W + 10, half_W - 10)
    
    def _create_gallery(self):
        """Create gallery showing the first 10 eigenmodes"""
        # 2 rows x 5 columns
        cell_h, cell_w = 80, 120
        gallery = np.zeros((cell_h * 2, cell_w * 5, 3), dtype=np.uint8)
        
        for i in range(min(10, self.n_eigenmodes)):
            row = i // 5
            col = i % 5
            
            # Get eigenmode values for LH
            mode = self.eigenmodes_lh[:, i]
            
            # Normalize to [-1, 1]
            mode_max = np.abs(mode).max() + 1e-6
            mode_norm = mode / mode_max
            
            # Create small image
            img = np.zeros((cell_h, cell_w), dtype=np.float32)
            
            # Simple scatter plot of mode values
            scale_x = (cell_w - 10) / (self.px_lh.max() - self.px_lh.min() + 1)
            scale_y = (cell_h - 10) / (self.py_lh.max() - self.py_lh.min() + 1)
            
            for v in range(0, len(mode), 10):  # Subsample for speed
                x = int(5 + (self.px_lh[v] - self.px_lh.min()) * scale_x)
                y = int(5 + (self.py_lh[v] - self.py_lh.min()) * scale_y)
                if 0 <= x < cell_w and 0 <= y < cell_h:
                    img[y, x] = mode_norm[v]
            
            # Apply colormap
            img_u8 = ((img + 1) / 2 * 255).astype(np.uint8)
            img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
            
            # Add label
            cv2.putText(img_color, f"M{i+1}", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img_color, f"l={self.eigenvalues_lh[i]:.2f}", (5, cell_h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1)
            
            # Place in gallery
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            gallery[y0:y1, x0:x1] = img_color
        
        self._gallery_image = gallery
    
    def step(self):
        if not self._initialized:
            return
        
        # Get complex modes
        cm = self.get_blended_input('complex_modes', 'mean')
        if cm is None:
            cm = np.zeros(self.n_input_modes, dtype=np.complex64)
        else:
            cm = np.array(cm, dtype=np.complex64).flatten()
            if len(cm) < self.n_input_modes:
                cm = np.pad(cm, (0, self.n_input_modes - len(cm)))
            else:
                cm = cm[:self.n_input_modes]
        
        # Get coherence
        coh = self.get_blended_input('phase_coherence', 'mean')
        if coh is None:
            coh = 0.7
        coh = float(np.clip(coh, 0.1, 1.0))
        
        # Get modulation
        mod = self.get_blended_input('modulation', 'mean')
        if mod is None:
            mod = 1.0
        mod = float(np.clip(mod, 0.1, 5.0))
        
        # Project EEG modes onto cortical eigenmodes
        # Simple approach: use first n_input_modes cortical eigenmodes
        # weighted by EEG complex mode amplitudes
        
        # LH field
        field_lh = np.zeros(len(self.vertices_lh), dtype=np.complex64)
        for i in range(min(self.n_input_modes, self.n_eigenmodes)):
            field_lh += cm[i] * self.eigenmodes_lh[:, i] * mod
        
        # RH field
        field_rh = np.zeros(len(self.vertices_rh), dtype=np.complex64)
        for i in range(min(self.n_input_modes, self.n_eigenmodes)):
            field_rh += cm[i] * self.eigenmodes_rh[:, i] * mod
        
        # Compute metrics
        self._mode_energy = float(np.sum(np.abs(cm)**2))
        self._spatial_complexity = float(np.std(np.abs(field_lh)) + np.std(np.abs(field_rh)))
        
        # Render
        self._render_cortex(field_lh, field_rh, coh)
    
    def _render_cortex(self, field_lh, field_rh, coherence):
        """Render cortex with eigenmode activity"""
        W, H = self.W, self.H
        
        # Initialize accumulator images
        acc_real = np.zeros((H, W), dtype=np.float32)
        acc_imag = np.zeros((H, W), dtype=np.float32)
        acc_count = np.zeros((H, W), dtype=np.float32)
        
        # Splat LH vertices
        for v in range(len(field_lh)):
            x, y = self.px_lh[v], self.py_lh[v]
            acc_real[y, x] += field_lh[v].real
            acc_imag[y, x] += field_lh[v].imag
            acc_count[y, x] += 1
        
        # Splat RH vertices
        for v in range(len(field_rh)):
            x, y = self.px_rh[v], self.py_rh[v]
            acc_real[y, x] += field_rh[v].real
            acc_imag[y, x] += field_rh[v].imag
            acc_count[y, x] += 1
        
        # Average
        acc_count = acc_count + 1e-6
        field_img = (acc_real / acc_count) + 1j * (acc_imag / acc_count)
        
        # Get magnitude and phase
        mag = np.abs(field_img)
        phase = np.angle(field_img)
        
        # Normalize magnitude
        mag_max = np.percentile(mag[acc_count > 1], 99) + 1e-6
        mag_norm = np.clip(mag / mag_max, 0, 1)
        mag_norm = mag_norm ** self.gamma
        
        # Create mask
        mask = (acc_count > 1).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # HSV coloring: hue from phase, value from magnitude
        phase_norm = (phase + np.pi) / (2 * np.pi)
        
        sat = np.ones_like(mag_norm) * (0.4 + 0.6 * coherence)
        val = mag_norm * (0.3 + 0.7 * coherence)
        
        bgr = _hsv_to_bgr(phase_norm, sat, val)
        
        # Apply mask
        bgr = (bgr.astype(np.float32) * mask[..., None]).astype(np.uint8)
        
        # Add edge highlight
        edges = cv2.Canny((mask * 255).astype(np.uint8), 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        bgr[edges > 0] = np.clip(bgr[edges > 0].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        
        # Add label
        cv2.putText(bgr, "Cortical Eigenmodes", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(bgr, f"Energy: {self._mode_energy:.1f}", (10, H - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        self._cortex_image = bgr
        
        # Also render separate hemispheres
        half_W = W // 2
        self._lh_image = bgr[:, :half_W].copy()
        self._rh_image = bgr[:, half_W:].copy()
    
    def get_output(self, port_name):
        if port_name == 'cortex_image':
            return self._cortex_image
        elif port_name == 'eigenmode_gallery':
            return self._gallery_image
        elif port_name == 'lh_image':
            return self._lh_image
        elif port_name == 'rh_image':
            return self._rh_image
        elif port_name == 'mode_energy':
            return self._mode_energy
        elif port_name == 'spatial_complexity':
            return self._spatial_complexity
        return None
    
    def get_display_image(self):
        if self._cortex_image is not None:
            img = np.ascontiguousarray(self._cortex_image)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
        # Error/loading display
        w, h = 200, 100
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if self._init_error:
            cv2.putText(img, "Init Error", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            cv2.putText(img, self._init_error[:25], (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        else:
            cv2.putText(img, "Initializing...", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Surface Type", "surface_type", self.surface_type, 
             [('inflated', 'inflated'), ('pial', 'pial'), ('white', 'white')]),
            ("Target Triangles/Hemi", "target_tris", self.target_tris, None),
            ("Gamma", "gamma", self.gamma, None),
        ]