"""
ResonantInstantonNode - Simulates self-resonant instanton fields for atomic structures.
Based on instantonassim x.py, modeling atoms as field lumps with intrinsic resonances.
Place this file in the 'nodes' folder as 'resonant_instanton_node.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from scipy.ndimage import gaussian_filter

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class ResonantInstantonNode(BaseNode):
    NODE_CATEGORY = "Simulation"
    NODE_COLOR = QtGui.QColor(100, 50, 200)  # Quantum purple

    def __init__(self, grid_size=96, dt=0.05, c=1.0, a=0.1, b=0.1, gamma=0.02, substrate_noise=0.0005):
        super().__init__()
        self.node_title = "Resonant Instanton"

        self.inputs = {
            'atomic_number': 'signal',  # Input to set atomic number (scaled, e.g., 1-100)
            'stable_isotope': 'signal',  # >0.5 for stable, else unstable
            'perturbation': 'signal',  # External noise or nudge to field
            'reset': 'signal'  # >0.5 to reinitialize
        }

        self.outputs = {
            'field_image': 'image',  # 96x96 float32 of phi field
            'stability': 'signal',  # Stability metric (0-1)
            'instanton_count': 'signal',  # Cumulative instanton events
            'decay_event': 'signal'  # 1 if decay occurred this step, else 0
        }

        self.grid_size = grid_size
        self.dt = float(dt)
        self.c = float(c)
        self.a = float(a)
        self.b = float(b)
        self.gamma = float(gamma)
        self.substrate_noise = float(substrate_noise)

        # Field state
        self.phi = np.zeros((grid_size, grid_size))
        self.phi_prev = np.zeros((grid_size, grid_size))

        # Tracking
        self.mode_energies = []
        self.resonance_peaks = []
        self.instanton_density = np.zeros((grid_size, grid_size))
        self.instanton_count = 0
        self.instanton_events = []
        self.stability_metric = 1.0

        # Time
        self.time = 0.0
        self.frame_count = 0

        # Default atom
        self.current_atomic_number = 2  # Helium
        self.current_stable_isotope = True
        self.initialize_atom(self.current_atomic_number, stable_isotope=self.current_stable_isotope)

    def initialize_atom(self, atomic_number, position=None, stable_isotope=True):
        if position is None:
            position = (self.grid_size // 2, self.grid_size // 2)

        # Clear state
        self.phi.fill(0)
        self.phi_prev.fill(0)
        self.instanton_density.fill(0)
        self.instanton_count = 0
        self.instanton_events = []
        self.stability_metric = 1.0

        # Core radius
        core_radius = 4 + np.log(1 + atomic_number)

        # Core amplitude
        core_amplitude = 1.0 + 0.2 * atomic_number

        # Meshgrid
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        r = np.sqrt((x - position[0])**2 + (y - position[1])**2)

        # Nuclear core
        self.phi = core_amplitude * np.exp(-r**2 / (2 * core_radius**2))

        # Shell config
        shell_config = self._calculate_shell_configuration(atomic_number)

        # Add shells
        for shell, electrons in enumerate(shell_config):
            if electrons > 0:
                shell_radius = self._shell_radius(shell + 1)
                shell_amplitude = 0.3 * (electrons / (2 * (2 * shell + 1)**2))
                shell_wave = shell_amplitude * np.cos(np.pi * r / shell_radius)**2 * (r < 2 * shell_radius)
                self.phi += shell_wave

        # Isotope variation
        if not stable_isotope:
            asymmetry = 0.1 * np.sin(3 * np.arctan2(y - position[1], x - position[0]))
            self.phi += asymmetry * np.exp(-r**2 / (2 * core_radius**2))
            self.stability_metric = 0.7 + 0.3 * np.random.random()

        self.phi_prev = self.phi.copy()
        self.time = 0.0
        self.frame_count = 0
        self.mode_energies = []
        self._analyze_resonant_modes()

    def _calculate_shell_configuration(self, atomic_number):
        shell_capacity = [2, 8, 18, 32, 50]
        shells = []
        electrons_left = atomic_number
        for capacity in shell_capacity:
            if electrons_left >= capacity:
                shells.append(capacity)
                electrons_left -= capacity
            else:
                shells.append(electrons_left)
                electrons_left = 0
                break
        while electrons_left > 0:
            next_capacity = 2 * (len(shells) + 1)**2
            if electrons_left >= next_capacity:
                shells.append(next_capacity)
                electrons_left -= next_capacity
            else:
                shells.append(electrons_left)
                break
        return shells

    def _shell_radius(self, n):
        base_radius = 8
        return base_radius * n**2

    def _laplacian(self, field):
        laplacian = np.zeros_like(field)
        field_padded = np.pad(field, 1, mode='wrap')
        laplacian = (field_padded[:-2, 1:-1] + field_padded[2:, 1:-1] +
                     field_padded[1:-1, :-2] + field_padded[1:-1, 2:] -
                     4 * field_padded[1:-1, 1:-1])
        return laplacian

    def _biharmonic(self, field):
        return self._laplacian(self._laplacian(field))

    def _analyze_resonant_modes(self):
        center = self.grid_size // 2
        x = np.arange(self.grid_size) - center
        y = np.arange(self.grid_size) - center
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        r_values = np.arange(0, self.grid_size // 2)
        radial_avg = np.zeros_like(r_values, dtype=float)
        for i, r in enumerate(r_values):
            mask = (R >= r - 0.5) & (R < r + 0.5)
            if np.sum(mask) > 0:
                radial_avg[i] = np.mean(self.phi[mask])
        peaks = []
        for i in range(1, len(radial_avg) - 1):
            if radial_avg[i] > radial_avg[i-1] and radial_avg[i] > radial_avg[i+1] and radial_avg[i] > 0.05:
                peaks.append((i, radial_avg[i]))
        self.resonance_peaks = peaks

    def _detect_instanton_event(self, phi_old, phi_new):
        delta_phi = phi_new - phi_old
        delta_phi_smoothed = gaussian_filter(delta_phi, sigma=1.0)
        threshold = 0.1 * np.max(np.abs(self.phi))
        significant_changes = np.abs(delta_phi_smoothed) > threshold
        if np.any(significant_changes):
            y_indices, x_indices = np.where(significant_changes)
            if len(x_indices) > 0:
                center_x = np.mean(x_indices)
                center_y = np.mean(y_indices)
                magnitude = np.max(np.abs(delta_phi_smoothed))
                self.instanton_count += 1
                self.instanton_events.append({
                    'time': self.time,
                    'position': (center_x, center_y),
                    'magnitude': magnitude
                })
                x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                self.instanton_density += 0.2 * np.exp(-r**2 / 50)
                return True
        return False

    def _update_stability(self):
        if len(self.instanton_events) > 0:
            recent_count = sum(1 for event in self.instanton_events if event['time'] > self.time - 100 * self.dt)
            if recent_count > 5:
                self.stability_metric -= 0.01
            else:
                self.stability_metric = min(1.0, self.stability_metric + 0.001)
        self.stability_metric = max(0.0, min(1.0, self.stability_metric))

    def step(self):
        # Get inputs
        atomic_number_in = self.get_blended_input('atomic_number', 'sum')
        stable_isotope_in = self.get_blended_input('stable_isotope', 'sum')
        perturbation_in = self.get_blended_input('perturbation', 'sum') or 0.0
        reset_in = self.get_blended_input('reset', 'sum') or 0.0

        # Handle reset or param changes
        if reset_in > 0.5 or atomic_number_in is not None or stable_isotope_in is not None:
            if atomic_number_in is not None:
                self.current_atomic_number = max(1, int(1 + atomic_number_in * 100))  # Scale to 1-101
            if stable_isotope_in is not None:
                self.current_stable_isotope = stable_isotope_in > 0.5
            self.initialize_atom(self.current_atomic_number, stable_isotope=self.current_stable_isotope)

        # Save old phi
        phi_old = self.phi.copy()

        # Compute terms
        laplacian_phi = self._laplacian(self.phi)
        biharmonic_phi = self._biharmonic(self.phi) if self.gamma != 0 else 0
        noise = self.substrate_noise * np.random.normal(size=self.phi.shape) + perturbation_in * 0.1  # Add input perturbation

        accel = (self.c**2 * laplacian_phi +
                 self.a * self.phi -
                 self.b * self.phi**3 -
                 self.gamma * biharmonic_phi +
                 noise)

        # Verlet update
        phi_new = 2 * self.phi - self.phi_prev + self.dt**2 * accel
        self.phi_prev = self.phi
        self.phi = phi_new

        # Detect instanton
        self._detect_instanton_event(phi_old, self.phi)

        # Update stability
        self._update_stability()

        # Analyze modes every 50 frames
        if self.frame_count % 50 == 0:
            self._analyze_resonant_modes()
            energy = np.sum(self.phi**2)
            self.mode_energies.append((self.time, energy))

        # Decay check
        self.decay_event = 0
        decay_probability = (1.0 - self.stability_metric)**2 * 0.001
        if np.random.random() < decay_probability:
            self.decay_event = 1

        # Update time
        self.time += self.dt
        self.frame_count += 1

    def get_output(self, port_name):
        if port_name == 'field_image':
            # Normalize phi to [0,1] for image
            phi_norm = (self.phi - np.min(self.phi)) / (np.max(self.phi) - np.min(self.phi) + 1e-9)
            return phi_norm.astype(np.float32)
        elif port_name == 'stability':
            return self.stability_metric
        elif port_name == 'instanton_count':
            return self.instanton_count / 100.0  # Scaled for signal
        elif port_name == 'decay_event':
            return self.decay_event
        return None

    def get_display_image(self):
        # Render colored field with overlays
        phi_norm = (self.phi - np.min(self.phi)) / (np.max(self.phi) - np.min(self.phi) + 1e-9)
        img_u8 = (phi_norm * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_COOL)

        # Overlay instanton density
        if np.max(self.instanton_density) > 0:
            inst_norm = (self.instanton_density / np.max(self.instanton_density) * 255).astype(np.uint8)
            inst_color = cv2.applyColorMap(inst_norm, cv2.COLORMAP_HOT)
            img_color = cv2.addWeighted(img_color, 0.7, inst_color, 0.3, 0)

        # Add text overlays (simulated, since no plt here)
        cv2.putText(img_color, f"Stab: {self.stability_metric:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_color, f"Inst: {self.instanton_count}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Shell circles
        center = self.grid_size // 2
        for r, _ in self.resonance_peaks:
            cv2.circle(img_color, (center, center), int(r), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3 * w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size", "grid_size", self.grid_size, None),
            ("Time Step (dt)", "dt", self.dt, None),
            ("Wave Speed (c)", "c", self.c, None),
            ("Linear Term (a)", "a", self.a, None),
            ("Nonlinear Term (b)", "b", self.b, None),
            ("Biharmonic (gamma)", "gamma", self.gamma, None),
            ("Substrate Noise", "substrate_noise", self.substrate_noise, None),
        ]