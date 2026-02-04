"""
K-Sweep Analysis Node
=====================

Automatically sweeps through spatial frequency (k) values to find the
optimal separation between CARRIER (geometry) and SIGNAL (brain dynamics).

Key Findings from Antti's experiments:
- CARRIER is stable at ALL k values (pure electrode geometry)
- SIGNAL shows dynamics that change with k
- At k~100, SIGNAL flow direction reverses (nodal frequency)
- Higher k = finer spatial detail but more geometric aliasing

This node:
1. Sweeps k from low to high
2. Measures signal dynamics (variance over time) at each k
3. Measures SNR (signal power / carrier power) at each k
4. Finds optimal k where signal is most dynamic relative to carrier
5. Detects flow reversal points (phase velocity sign changes)

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    try:
        from PyQt6 import QtGui
    except ImportError:
        class MockQtGui:
            @staticmethod
            def QColor(*args): return None
            class QImage:
                Format_RGB888 = 0
                def __init__(self, *args): pass
        QtGui = MockQtGui()
    
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
        def get_blended_input(self, name, mode): 
            data = self.input_data.get(name, [None])
            return data[0] if data else None
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}


# Standard 10-20 electrode positions
ELECTRODE_POS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6), 'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'T7': (-0.9, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0, 0.0), 'C4': (0.4, 0.0), 'T8': (0.9, 0.0),
    'P7': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5), 'P4': (0.35, -0.5), 'P8': (0.7, -0.5),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85)
}


class KSweepAnalysisNode(BaseNode):
    """
    Sweeps spatial frequency k to find optimal carrier/signal separation.
    
    Outputs:
    - Optimal k value for maximum signal dynamics
    - K-response curve showing signal strength vs k
    - Flow reversal detection (phase velocity sign changes)
    - Real-time signal at current/optimal k
    """
    
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "K-Sweep Analyzer"
    NODE_COLOR = QtGui.QColor(200, 100, 50)  # Orange - analysis
    
    def __init__(self):
        super().__init__()
        
        # === INPUTS ===
        self.inputs = {
            'hologram_field': 'complex_spectrum',  # From PhiHologram
            'trigger_sweep': 'signal',              # >0.5 triggers new sweep
            'manual_k': 'signal',                   # Manual k override
        }
        
        # === OUTPUTS ===
        self.outputs = {
            # Optimal values
            'optimal_k': 'signal',                  # Best k for signal/carrier separation
            'optimal_snr': 'signal',                # SNR at optimal k
            
            # K-response curve
            'k_response_curve': 'image',            # Graph of metrics vs k
            
            # Current analysis at optimal k
            'signal_at_optimal': 'complex_spectrum', # Signal field at optimal k
            'carrier_at_optimal': 'complex_spectrum', # Carrier field at optimal k
            'signal_image': 'image',                 # Signal visualization
            'carrier_image': 'image',                # Carrier visualization
            
            # Flow analysis
            'flow_field': 'image',                   # Optical flow of signal
            'flow_reversal_k': 'signal',            # K value where flow reverses
            'flow_direction': 'signal',              # Current flow direction (-1, 0, +1)
            
            # Dynamics metrics
            'signal_variance': 'signal',             # How much signal changes over time
            'carrier_stability': 'signal',           # How stable carrier is (should be ~1.0)
        }
        
        # === PARAMETERS ===
        self.resolution = 128
        self.k_min = 10.0
        self.k_max = 500.0
        self.k_steps = 50
        self.history_length = 20  # Frames to analyze for dynamics
        
        # === STATE ===
        self.is_sweeping = False
        self.current_sweep_idx = 0
        self.sweep_results = []  # List of (k, snr, signal_var, carrier_var, flow_dir)
        
        # History buffers for each k being tested
        self.field_history = deque(maxlen=self.history_length)
        
        # Results
        self.optimal_k = 50.0
        self.optimal_snr = 1.0
        self.flow_reversal_k = 100.0
        
        # Current frame outputs
        self._signal_field = None
        self._carrier_field = None
        self._signal_image = None
        self._carrier_image = None
        self._flow_field = None
        self._k_response_image = None
        
        self._signal_variance = 0.0
        self._carrier_stability = 1.0
        self._flow_direction = 0.0
        
        # Pre-compute geometry
        self._init_geometry()
        
        # K values to test
        self.k_values = np.linspace(self.k_min, self.k_max, self.k_steps)
        
    def _init_geometry(self):
        """Pre-compute electrode distance maps."""
        res = self.resolution
        x = np.linspace(-1.5, 1.5, res).astype(np.float32)
        y = np.linspace(-1.5, 1.5, res).astype(np.float32)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.dist_maps = {}
        for name, (ex, ey) in ELECTRODE_POS.items():
            self.dist_maps[name] = np.sqrt((self.X - ex)**2 + (self.Y - ey)**2)
    
    def _generate_carrier(self, k, rot_deg=0):
        """Generate carrier (flat-phase hologram) at given k."""
        res = self.resolution
        field = np.zeros((res, res), dtype=np.complex64)
        rot_rad = np.deg2rad(rot_deg)
        
        for elec_name, dist in self.dist_maps.items():
            theta = 0 - (k * dist) + rot_rad
            wave = np.cos(theta) + 1j * np.sin(theta)
            field += wave
        
        field = field / len(self.dist_maps)
        return field
    
    def _generate_hologram_at_k(self, phases_dict, k, rot_deg=0):
        """
        Generate hologram with given phases at specific k.
        phases_dict: {electrode_name: phase_value}
        """
        res = self.resolution
        field = np.zeros((res, res), dtype=np.complex64)
        rot_rad = np.deg2rad(rot_deg)
        
        for elec_name, dist in self.dist_maps.items():
            if elec_name in phases_dict:
                phi = phases_dict[elec_name]
            else:
                phi = 0
            theta = phi - (k * dist) + rot_rad
            wave = np.cos(theta) + 1j * np.sin(theta)
            field += wave
        
        field = field / len(self.dist_maps)
        return field
    
    def _compute_optical_flow(self, prev_field, curr_field):
        """
        Compute optical flow between two signal fields.
        Returns flow magnitude and dominant direction.
        """
        # Convert to magnitude images
        prev_mag = np.abs(prev_field)
        curr_mag = np.abs(curr_field)
        
        # Normalize to uint8
        prev_u8 = ((prev_mag / (prev_mag.max() + 1e-9)) * 255).astype(np.uint8)
        curr_u8 = ((curr_mag / (curr_mag.max() + 1e-9)) * 255).astype(np.uint8)
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_u8, curr_u8, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Flow components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # Dominant direction (mean flow angle)
        angles = np.arctan2(flow_y, flow_x)
        
        # Weight angles by magnitude
        weights = magnitude / (magnitude.sum() + 1e-9)
        mean_angle = np.arctan2(
            np.sum(weights * np.sin(angles)),
            np.sum(weights * np.cos(angles))
        )
        
        # Direction sign: positive = outward/expanding, negative = inward/contracting
        # Use radial component
        center = self.resolution // 2
        y_grid, x_grid = np.mgrid[:self.resolution, :self.resolution]
        radial_x = (x_grid - center) / (center + 1e-9)
        radial_y = (y_grid - center) / (center + 1e-9)
        
        # Dot product of flow with radial direction
        radial_flow = flow_x * radial_x + flow_y * radial_y
        direction = np.sign(np.mean(radial_flow))
        
        return magnitude, direction, flow
    
    def _separate_at_k(self, field, k):
        """Separate carrier and signal at given k."""
        carrier = self._generate_carrier(k)
        
        # Scale carrier to match field magnitude
        carrier_scale = np.mean(np.abs(field)) / (np.mean(np.abs(carrier)) + 1e-9)
        carrier = carrier * carrier_scale
        
        signal = field - carrier
        
        return carrier, signal
    
    def _compute_metrics_at_k(self, field_history, k):
        """
        Compute signal dynamics metrics at given k.
        
        Returns:
        - snr: signal power / carrier power
        - signal_var: variance of signal over time (dynamics)
        - carrier_var: variance of carrier over time (should be ~0)
        - flow_dir: dominant flow direction
        """
        if len(field_history) < 2:
            return 1.0, 0.0, 0.0, 0.0
        
        signals = []
        carriers = []
        
        for field in field_history:
            carrier, signal = self._separate_at_k(field, k)
            signals.append(signal)
            carriers.append(carrier)
        
        # Stack for analysis
        signal_stack = np.array([np.abs(s) for s in signals])
        carrier_stack = np.array([np.abs(c) for c in carriers])
        
        # Signal variance over time (how dynamic is it?)
        signal_var = np.mean(np.var(signal_stack, axis=0))
        
        # Carrier variance over time (should be near zero - it's geometric)
        carrier_var = np.mean(np.var(carrier_stack, axis=0))
        
        # SNR
        signal_power = np.mean(signal_stack ** 2)
        carrier_power = np.mean(carrier_stack ** 2)
        snr = signal_power / (carrier_power + 1e-9)
        
        # Flow direction (from last two frames)
        if len(signals) >= 2:
            _, flow_dir, _ = self._compute_optical_flow(signals[-2], signals[-1])
        else:
            flow_dir = 0.0
        
        return snr, signal_var, carrier_var, flow_dir
    
    def _run_sweep(self):
        """Run full k-sweep analysis."""
        if len(self.field_history) < 5:
            return  # Need history
        
        self.sweep_results = []
        
        for k in self.k_values:
            snr, sig_var, car_var, flow_dir = self._compute_metrics_at_k(
                list(self.field_history), k
            )
            self.sweep_results.append({
                'k': k,
                'snr': snr,
                'signal_var': sig_var,
                'carrier_var': car_var,
                'flow_dir': flow_dir
            })
        
        # Find optimal k (max signal variance * snr)
        scores = [r['signal_var'] * r['snr'] for r in self.sweep_results]
        best_idx = np.argmax(scores)
        self.optimal_k = self.sweep_results[best_idx]['k']
        self.optimal_snr = self.sweep_results[best_idx]['snr']
        
        # Find flow reversal point
        flow_dirs = [r['flow_dir'] for r in self.sweep_results]
        for i in range(1, len(flow_dirs)):
            if flow_dirs[i] * flow_dirs[i-1] < 0:  # Sign change
                self.flow_reversal_k = self.sweep_results[i]['k']
                break
        
        # Create k-response visualization
        self._create_k_response_image()
    
    def _create_k_response_image(self):
        """Create visualization of k-sweep results."""
        if not self.sweep_results:
            return
        
        h, w = 200, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Dark background
        
        # Extract data
        ks = [r['k'] for r in self.sweep_results]
        snrs = [r['snr'] for r in self.sweep_results]
        sig_vars = [r['signal_var'] for r in self.sweep_results]
        flow_dirs = [r['flow_dir'] for r in self.sweep_results]
        
        # Normalize for plotting
        k_norm = [(k - self.k_min) / (self.k_max - self.k_min) for k in ks]
        
        snr_max = max(snrs) + 1e-9
        snr_norm = [s / snr_max for s in snrs]
        
        var_max = max(sig_vars) + 1e-9
        var_norm = [v / var_max for v in sig_vars]
        
        # Plot SNR (blue)
        margin = 20
        plot_h = h - 2 * margin
        plot_w = w - 2 * margin
        
        for i in range(len(ks) - 1):
            x1 = int(margin + k_norm[i] * plot_w)
            x2 = int(margin + k_norm[i+1] * plot_w)
            y1 = int(h - margin - snr_norm[i] * plot_h)
            y2 = int(h - margin - snr_norm[i+1] * plot_h)
            cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 2)
        
        # Plot Signal Variance (green)
        for i in range(len(ks) - 1):
            x1 = int(margin + k_norm[i] * plot_w)
            x2 = int(margin + k_norm[i+1] * plot_w)
            y1 = int(h - margin - var_norm[i] * plot_h)
            y2 = int(h - margin - var_norm[i+1] * plot_h)
            cv2.line(img, (x1, y1), (x2, y2), (100, 255, 100), 2)
        
        # Plot Flow Direction (yellow, centered)
        center_y = h // 2
        for i in range(len(ks) - 1):
            x1 = int(margin + k_norm[i] * plot_w)
            x2 = int(margin + k_norm[i+1] * plot_w)
            y1 = int(center_y - flow_dirs[i] * 30)
            y2 = int(center_y - flow_dirs[i+1] * 30)
            cv2.line(img, (x1, y1), (x2, y2), (100, 255, 255), 1)
        
        # Mark optimal k (vertical white line)
        opt_x = int(margin + (self.optimal_k - self.k_min) / (self.k_max - self.k_min) * plot_w)
        cv2.line(img, (opt_x, margin), (opt_x, h - margin), (255, 255, 255), 2)
        
        # Mark flow reversal (vertical magenta line)
        rev_x = int(margin + (self.flow_reversal_k - self.k_min) / (self.k_max - self.k_min) * plot_w)
        cv2.line(img, (rev_x, margin), (rev_x, h - margin), (255, 0, 255), 1)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"K-Sweep Analysis", (10, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Optimal k={self.optimal_k:.0f}", (10, 30), font, 0.35, (255, 255, 255), 1)
        cv2.putText(img, f"Flow reversal k={self.flow_reversal_k:.0f}", (10, 45), font, 0.35, (255, 0, 255), 1)
        
        # Legend
        cv2.putText(img, "SNR", (w - 60, 15), font, 0.3, (255, 100, 100), 1)
        cv2.putText(img, "Var", (w - 60, 30), font, 0.3, (100, 255, 100), 1)
        cv2.putText(img, "Flow", (w - 60, 45), font, 0.3, (100, 255, 255), 1)
        
        # Axis labels
        cv2.putText(img, f"k={self.k_min:.0f}", (margin, h - 5), font, 0.25, (150, 150, 150), 1)
        cv2.putText(img, f"k={self.k_max:.0f}", (w - 50, h - 5), font, 0.25, (150, 150, 150), 1)
        
        self._k_response_image = img
    
    def step(self):
        """Main processing step."""
        # Get inputs
        field = self.get_blended_input('hologram_field', 'first')
        trigger = self.get_blended_input('trigger_sweep', 'max')
        manual_k = self.get_blended_input('manual_k', 'first')
        
        if field is None:
            return
        
        # Resize if needed
        if field.shape[0] != self.resolution:
            mag = cv2.resize(np.abs(field), (self.resolution, self.resolution))
            phase = cv2.resize(np.angle(field), (self.resolution, self.resolution))
            field = (mag * np.exp(1j * phase)).astype(np.complex64)
        
        # Add to history
        self.field_history.append(field.copy())
        
        # Check for sweep trigger
        if trigger is not None and float(trigger) > 0.5:
            self._run_sweep()
        
        # Use manual k if provided, else optimal
        current_k = self.optimal_k
        if manual_k is not None:
            current_k = float(manual_k)
        
        # Separate at current k
        self._carrier_field, self._signal_field = self._separate_at_k(field, current_k)
        
        # Compute current metrics
        if len(self.field_history) >= 2:
            snr, sig_var, car_var, flow_dir = self._compute_metrics_at_k(
                list(self.field_history)[-5:], current_k
            )
            self._signal_variance = sig_var
            self._carrier_stability = 1.0 / (car_var + 0.01)  # Invert: low var = high stability
            self._flow_direction = flow_dir
            
            # Compute flow field visualization
            _, _, flow = self._compute_optical_flow(
                list(self.field_history)[-2], 
                list(self.field_history)[-1]
            )
            
            # Visualize flow as HSV (hue=direction, saturation=1, value=magnitude)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            ang = np.arctan2(flow[..., 1], flow[..., 0])
            
            hsv = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(mag * 50, 0, 255).astype(np.uint8)
            self._flow_field = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Create visualizations
        carrier_mag = np.abs(self._carrier_field)
        carrier_norm = carrier_mag / (carrier_mag.max() + 1e-9)
        carrier_u8 = (carrier_norm * 255).astype(np.uint8)
        self._carrier_image = cv2.applyColorMap(carrier_u8, cv2.COLORMAP_BONE)
        
        signal_mag = np.abs(self._signal_field)
        signal_norm = signal_mag / (signal_mag.max() + 1e-9)
        signal_u8 = (signal_norm * 255).astype(np.uint8)
        self._signal_image = cv2.applyColorMap(signal_u8, cv2.COLORMAP_INFERNO)
    
    def get_output(self, port_name):
        """Return outputs."""
        outputs = {
            'optimal_k': self.optimal_k,
            'optimal_snr': self.optimal_snr,
            'k_response_curve': self._k_response_image,
            'signal_at_optimal': self._signal_field,
            'carrier_at_optimal': self._carrier_field,
            'signal_image': self._signal_image,
            'carrier_image': self._carrier_image,
            'flow_field': self._flow_field,
            'flow_reversal_k': self.flow_reversal_k,
            'flow_direction': self._flow_direction,
            'signal_variance': self._signal_variance,
            'carrier_stability': self._carrier_stability,
        }
        return outputs.get(port_name)
    
    def get_display_image(self):
        """Create node display."""
        # Show k-response curve if available, else signal/carrier split
        if self._k_response_image is not None:
            img = self._k_response_image.copy()
        else:
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.putText(img, "K-Sweep Analyzer", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, "Waiting for data...", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(img, "Send trigger to sweep", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        
        qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        qimg.ndarray = img
        return qimg
    
    def get_config_options(self):
        """Configuration options."""
        return [
            ("K Min", "k_min", self.k_min, 'float'),
            ("K Max", "k_max", self.k_max, 'float'),
            ("K Steps", "k_steps", self.k_steps, 'int'),
            ("History Length", "history_length", self.history_length, 'int'),
            ("Resolution", "resolution", self.resolution, 'int'),
        ]
    
    def set_config_options(self, options):
        """Apply configuration."""
        rebuild = False
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    if key in ['k_min', 'k_max', 'k_steps']:
                        rebuild = True
                    if key == 'resolution':
                        self._init_geometry()
        
        if rebuild:
            self.k_values = np.linspace(self.k_min, self.k_max, self.k_steps)


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("K-Sweep Analysis Node")
    print("=" * 40)
    print()
    print("This node finds the optimal spatial frequency (k) for")
    print("separating electrode geometry (carrier) from brain signal.")
    print()
    print("Key outputs:")
    print("  optimal_k        - Best k for signal/carrier separation")
    print("  flow_reversal_k  - K where signal flow direction inverts")
    print("  k_response_curve - Graph showing metrics vs k")
    print()
    print("Connect PhiHologram → this node")
    print("Send trigger signal to run sweep")
    print()
    
    # Quick test
    node = KSweepAnalysisNode()
    print(f"K range: {node.k_min} to {node.k_max} in {node.k_steps} steps")
    print(f"Initial optimal k: {node.optimal_k}")