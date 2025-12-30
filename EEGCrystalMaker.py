#!/usr/bin/env python3
"""
EEG Crystal Maker
=================

Standalone GUI for growing neural crystal lattices from EEG data.

Features:
- Load any EDF file
- Set resolution (32x32 to 2048x2048)
- Watch crystallization in real-time
- Save crystal state + pin map
- Load and continue growing

The crystal lattice is a 2D Izhikevich neuron sheet with STDP plasticity.
EEG electrodes inject current at mapped positions (the "pins").
Over time, the coupling weights crystallize into a structure that
reflects the EEG's spatiotemporal patterns.

Output:
- .npz file containing:
  - weights (4 directional coupling matrices)
  - pin_coords (electrode positions on grid)
  - pin_names (electrode labels)
  - metadata (resolution, training steps, etc.)

Author: Built for Antti's consciousness crystallography research
"""

import sys
import os
import re
import json
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QGroupBox, QGridLayout, QComboBox, QCheckBox,
    QSlider, QFrame, QMessageBox, QStatusBar
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont

import cv2

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not installed. EEG loading will not work.")


class CrystalLattice:
    """The neural crystal - Izhikevich sheet with STDP."""
    
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.init_arrays()
        
        # Izhikevich parameters
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 0.5
        
        # STDP parameters
        self.learning_rate = 0.005
        self.trace_decay = 0.95
        self.weight_max = 2.0
        self.weight_min = 0.01
        
        # Coupling strength - how much neighbors influence each other
        self.coupling_strength = 5.0  # Higher = more spread
        
        # Statistics
        self.total_spikes = 0
        self.learning_steps = 0
        
    def init_arrays(self):
        """Initialize all arrays to current grid_size."""
        n = self.grid_size
        
        # Neural state
        self.v = np.ones((n, n), dtype=np.float32) * -65.0
        self.u = self.v * 0.2
        
        # Crystal weights (4 directions)
        self.weights_up = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_down = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_left = np.ones((n, n), dtype=np.float32) * 0.5
        self.weights_right = np.ones((n, n), dtype=np.float32) * 0.5
        
        # Spike trace for STDP
        self.spike_trace = np.zeros((n, n), dtype=np.float32)
        
    def resize(self, new_size):
        """Resize the lattice (resets state)."""
        self.grid_size = new_size
        self.init_arrays()
        self.total_spikes = 0
        self.learning_steps = 0
        
    def step(self, input_current, learning=True):
        """One simulation step with optional STDP learning."""
        v = self.v
        u = self.u
        I = input_current
        
        # Clamp input to prevent explosion
        I = np.clip(I, -100, 100)
        
        # Neighbor coupling
        v_up = np.roll(v, -1, axis=0)
        v_down = np.roll(v, 1, axis=0)
        v_left = np.roll(v, -1, axis=1)
        v_right = np.roll(v, 1, axis=1)
        
        neighbor_influence = (
            self.weights_up * v_up +
            self.weights_down * v_down +
            self.weights_left * v_left +
            self.weights_right * v_right
        )
        total_weight = (self.weights_up + self.weights_down + 
                       self.weights_left + self.weights_right)
        neighbor_avg = neighbor_influence / (total_weight + 1e-6)
        
        I_coupling = self.coupling_strength * (neighbor_avg - v)
        I_coupling = np.clip(I_coupling, -50, 50)  # Prevent coupling explosion
        
        # Izhikevich dynamics
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I + I_coupling) * self.dt
        du = self.a * (self.b * v - u) * self.dt
        
        v = v + dv
        u = u + du
        
        # Clamp voltage to sane range (prevents NaN cascade)
        v = np.clip(v, -100, 50)
        u = np.clip(u, -50, 50)
        
        # Handle any NaN that slipped through
        v = np.nan_to_num(v, nan=self.c, posinf=30.0, neginf=-100.0)
        u = np.nan_to_num(u, nan=self.c * self.b, posinf=20.0, neginf=-20.0)
        
        # Spikes
        spikes = v >= 30.0
        v[spikes] = self.c
        u[spikes] += self.d
        
        self.v = v
        self.u = u
        self.total_spikes += np.sum(spikes)
        
        # STDP
        if learning and self.learning_rate > 0:
            self.learning_steps += 1
            
            self.spike_trace *= self.trace_decay
            self.spike_trace[spikes] = 1.0
            
            trace_up = np.roll(self.spike_trace, -1, axis=0)
            trace_down = np.roll(self.spike_trace, 1, axis=0)
            trace_left = np.roll(self.spike_trace, -1, axis=1)
            trace_right = np.roll(self.spike_trace, 1, axis=1)
            
            spike_float = spikes.astype(np.float32)
            lr = self.learning_rate
            
            # Potentiation
            dw_up = lr * spike_float * trace_up
            dw_down = lr * spike_float * trace_down
            dw_left = lr * spike_float * trace_left
            dw_right = lr * spike_float * trace_right
            
            # Depression
            spike_up = np.roll(spike_float, -1, axis=0)
            spike_down = np.roll(spike_float, 1, axis=0)
            spike_left = np.roll(spike_float, -1, axis=1)
            spike_right = np.roll(spike_float, 1, axis=1)
            
            dw_up -= 0.5 * lr * self.spike_trace * spike_up
            dw_down -= 0.5 * lr * self.spike_trace * spike_down
            dw_left -= 0.5 * lr * self.spike_trace * spike_left
            dw_right -= 0.5 * lr * self.spike_trace * spike_right
            
            self.weights_up = np.clip(self.weights_up + dw_up, self.weight_min, self.weight_max)
            self.weights_down = np.clip(self.weights_down + dw_down, self.weight_min, self.weight_max)
            self.weights_left = np.clip(self.weights_left + dw_left, self.weight_min, self.weight_max)
            self.weights_right = np.clip(self.weights_right + dw_right, self.weight_min, self.weight_max)
        
        return spikes
    
    def get_energy(self):
        """Total weight energy."""
        return float(np.sum(self.weights_up) + np.sum(self.weights_down) + 
                    np.sum(self.weights_left) + np.sum(self.weights_right))
    
    def get_entropy(self):
        """Weight distribution entropy."""
        all_weights = np.concatenate([
            self.weights_up.flatten(),
            self.weights_down.flatten(),
            self.weights_left.flatten(),
            self.weights_right.flatten()
        ])
        w_norm = all_weights / (np.sum(all_weights) + 1e-9)
        return float(-np.sum(w_norm * np.log(w_norm + 1e-9)))
    
    def render_activity(self, size=256):
        """Render activity as image."""
        disp = np.clip(self.v, -90.0, 40.0)
        disp = np.nan_to_num(disp, nan=-65.0, posinf=40.0, neginf=-90.0)
        norm = ((disp + 90.0) / 130.0 * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    
    def render_crystal(self, size=256):
        """Render crystal structure as image."""
        horizontal = (self.weights_left + self.weights_right) / 2
        vertical = (self.weights_up + self.weights_down) / 2
        
        h_norm = (horizontal - self.weight_min) / (self.weight_max - self.weight_min)
        v_norm = (vertical - self.weight_min) / (self.weight_max - self.weight_min)
        anisotropy = np.abs(h_norm - v_norm)
        
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        img[:, :, 0] = (h_norm * 255).astype(np.uint8)
        img[:, :, 1] = ((1 - anisotropy) * 255).astype(np.uint8)
        img[:, :, 2] = (v_norm * 255).astype(np.uint8)
        
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


class EEGSource:
    """Handles EEG loading and electrode mapping."""
    
    STANDARD_MAP = {
        "FP1": (0.30, 0.10), "FP2": (0.70, 0.10),
        "F7": (0.10, 0.30), "F3": (0.30, 0.30), "FZ": (0.50, 0.25),
        "F4": (0.70, 0.30), "F8": (0.90, 0.30),
        "T7": (0.10, 0.50), "T3": (0.10, 0.50),  # T3 alias
        "C3": (0.30, 0.50), "CZ": (0.50, 0.50),
        "C4": (0.70, 0.50), "T8": (0.90, 0.50), "T4": (0.90, 0.50),  # T4 alias
        "P7": (0.10, 0.70), "T5": (0.10, 0.70),  # T5 alias
        "P3": (0.30, 0.70), "PZ": (0.50, 0.75),
        "P4": (0.70, 0.70), "P8": (0.90, 0.70), "T6": (0.90, 0.70),  # T6 alias
        "O1": (0.35, 0.90), "OZ": (0.50, 0.90), "O2": (0.65, 0.90),
        "A1": (0.05, 0.50), "A2": (0.95, 0.50),  # Ear references
    }
    
    def __init__(self):
        self.raw = None
        self.data = None
        self.sfreq = 256.0
        self.ch_names = []
        self.current_idx = 0
        self.amplification = 1e9  # Default amplification (Medium)
        
        # Pin mapping
        self.pin_coords = []  # (row, col) for each channel
        self.pin_names = []   # Channel names
        self.pin_indices = [] # Channel indices in data
        
    def load(self, filepath):
        """Load EDF file."""
        if not MNE_AVAILABLE:
            raise RuntimeError("MNE not installed")
        
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        try:
            raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, 
                          emg=False, misc=False, stim=False)
        except:
            pass
        
        if raw.info["sfreq"] > 256:
            raw.resample(256, npad="auto", verbose=False)
        
        self.raw = raw
        self.data = raw.get_data()
        self.sfreq = float(raw.info["sfreq"])
        self.ch_names = list(raw.ch_names)
        self.current_idx = 0
        
        return len(self.ch_names), self.data.shape[1]
    
    def map_electrodes(self, grid_size):
        """Map electrodes to grid positions."""
        self.pin_coords = []
        self.pin_names = []
        self.pin_indices = []
        
        for idx, name in enumerate(self.ch_names):
            clean = re.sub(r'[^A-Z0-9]', '', name.upper())
            
            pos = None
            # Try exact match first
            for std_name, std_pos in self.STANDARD_MAP.items():
                if std_name in clean or clean in std_name:
                    pos = std_pos
                    break
            
            # Try prefix match
            if pos is None:
                for std_name, std_pos in self.STANDARD_MAP.items():
                    if len(clean) >= 2 and clean[:2] == std_name[:2]:
                        pos = std_pos
                        break
            
            if pos:
                grid_r = int(pos[1] * (grid_size - 1))
                grid_c = int(pos[0] * (grid_size - 1))
                self.pin_coords.append((grid_r, grid_c))
                self.pin_names.append(name)
                self.pin_indices.append(idx)
        
        return len(self.pin_coords)
    
    def get_input_current(self, grid_size):
        """Get input current for one timestep."""
        if self.data is None:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        
        n_samples = self.data.shape[1]
        sample_idx = self.current_idx % n_samples
        self.current_idx += 1
        
        I = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Small spread - electrodes are injection points
        # The coupling between neurons spreads activity, not the electrode radius
        spread_radius = max(2, grid_size // 128)  # ~8 at 1024, ~2 at 256
        spread_sigma = max(1.0, spread_radius / 2.0)
        
        # Pre-compute Gaussian kernel once
        kernel_size = spread_radius * 2 + 1
        y, x = np.ogrid[-spread_radius:spread_radius+1, -spread_radius:spread_radius+1]
        kernel = np.exp(-(x*x + y*y) / (2 * spread_sigma * spread_sigma)).astype(np.float32)
        
        for i, ch_idx in enumerate(self.pin_indices):
            if i < len(self.pin_coords):
                r, c = self.pin_coords[i]
                val = self.data[ch_idx, sample_idx]
                
                # Scale EEG
                scaled = float(val) * self.amplification
                scaled = np.clip(scaled, -500, 500)
                
                # Calculate bounds for kernel placement
                r_start = max(0, r - spread_radius)
                r_end = min(grid_size, r + spread_radius + 1)
                c_start = max(0, c - spread_radius)
                c_end = min(grid_size, c + spread_radius + 1)
                
                # Corresponding kernel bounds
                kr_start = r_start - (r - spread_radius)
                kr_end = kernel_size - ((r + spread_radius + 1) - r_end)
                kc_start = c_start - (c - spread_radius)
                kc_end = kernel_size - ((c + spread_radius + 1) - c_end)
                
                # Add weighted kernel to input
                I[r_start:r_end, c_start:c_end] += scaled * kernel[kr_start:kr_end, kc_start:kc_end]
        
        return I


class CrystalMakerWindow(QMainWindow):
    """Main GUI window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Crystal Maker")
        self.setMinimumSize(1000, 700)
        
        # Core objects
        self.crystal = CrystalLattice(64)
        self.eeg = EEGSource()
        
        # State
        self.is_running = False
        self.eeg_loaded = False
        self.edf_path = ""
        
        # Timer for simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """Build the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left panel - controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)
        
        # EEG Loading
        eeg_group = QGroupBox("EEG Source")
        eeg_layout = QVBoxLayout(eeg_group)
        
        self.edf_label = QLabel("No file loaded")
        self.edf_label.setWordWrap(True)
        eeg_layout.addWidget(self.edf_label)
        
        load_btn = QPushButton("Load EDF File...")
        load_btn.clicked.connect(self.load_edf)
        eeg_layout.addWidget(load_btn)
        
        self.eeg_info = QLabel("Channels: -\nSamples: -\nPins mapped: -")
        eeg_layout.addWidget(self.eeg_info)
        
        left_panel.addWidget(eeg_group)
        
        # Crystal Settings
        crystal_group = QGroupBox("Crystal Settings")
        crystal_layout = QGridLayout(crystal_group)
        
        crystal_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["32", "64", "128", "256", "512", "1024"])
        self.resolution_combo.setCurrentText("64")
        self.resolution_combo.currentTextChanged.connect(self.on_resolution_changed)
        crystal_layout.addWidget(self.resolution_combo, 0, 1)
        
        crystal_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.005)
        self.lr_spin.valueChanged.connect(self.on_lr_changed)
        crystal_layout.addWidget(self.lr_spin, 1, 1)
        
        crystal_layout.addWidget(QLabel("EEG Amplification:"), 2, 0)
        self.amp_combo = QComboBox()
        self.amp_combo.addItems(["1e8 (Low)", "1e9 (Medium)", "1e10 (High)", "1e11 (Very High)"])
        self.amp_combo.setCurrentIndex(1)  # Default to Medium
        self.amp_combo.currentIndexChanged.connect(self.on_amp_changed)
        crystal_layout.addWidget(self.amp_combo, 2, 1)
        
        crystal_layout.addWidget(QLabel("Coupling Strength:"), 3, 0)
        self.coupling_spin = QDoubleSpinBox()
        self.coupling_spin.setRange(0.1, 20.0)
        self.coupling_spin.setSingleStep(0.5)
        self.coupling_spin.setValue(5.0)
        self.coupling_spin.valueChanged.connect(self.on_coupling_changed)
        crystal_layout.addWidget(self.coupling_spin, 3, 1)
        
        crystal_layout.addWidget(QLabel("Target Steps:"), 4, 0)
        self.target_steps_spin = QSpinBox()
        self.target_steps_spin.setRange(100, 100000)
        self.target_steps_spin.setSingleStep(100)
        self.target_steps_spin.setValue(800)
        crystal_layout.addWidget(self.target_steps_spin, 4, 1)
        
        left_panel.addWidget(crystal_group)
        
        # Simulation Control
        control_group = QGroupBox("Simulation")
        control_layout = QVBoxLayout(control_group)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.toggle_simulation)
        btn_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset_crystal)
        btn_layout.addWidget(self.reset_btn)
        control_layout.addLayout(btn_layout)
        
        # Speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        control_layout.addLayout(speed_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 800)
        control_layout.addWidget(self.progress_bar)
        
        left_panel.addWidget(control_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Steps: 0\nSpikes: 0\nEnergy: 0\nEntropy: 0")
        self.stats_label.setFont(QFont("Monospace", 10))
        stats_layout.addWidget(self.stats_label)
        
        left_panel.addWidget(stats_group)
        
        # Save/Load
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        save_btn = QPushButton("💾 Save Crystal...")
        save_btn.clicked.connect(self.save_crystal)
        file_layout.addWidget(save_btn)
        
        load_crystal_btn = QPushButton("📂 Load Crystal...")
        load_crystal_btn.clicked.connect(self.load_crystal)
        file_layout.addWidget(load_crystal_btn)
        
        left_panel.addWidget(file_group)
        
        left_panel.addStretch()
        
        # Right panel - visualization
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)
        
        # Activity view
        activity_group = QGroupBox("Neural Activity")
        activity_layout = QVBoxLayout(activity_group)
        self.activity_label = QLabel()
        self.activity_label.setMinimumSize(400, 400)
        self.activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activity_label.setStyleSheet("background-color: #1a1a1a;")
        activity_layout.addWidget(self.activity_label)
        right_panel.addWidget(activity_group)
        
        # Crystal view
        crystal_view_group = QGroupBox("Crystal Structure")
        crystal_view_layout = QVBoxLayout(crystal_view_group)
        self.crystal_label = QLabel()
        self.crystal_label.setMinimumSize(400, 400)
        self.crystal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crystal_label.setStyleSheet("background-color: #1a1a1a;")
        crystal_view_layout.addWidget(self.crystal_label)
        right_panel.addWidget(crystal_view_group)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load an EDF file to begin")
        
    def load_edf(self):
        """Load EDF file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf);;All Files (*)"
        )
        if filepath:
            try:
                n_channels, n_samples = self.eeg.load(filepath)
                n_pins = self.eeg.map_electrodes(self.crystal.grid_size)
                
                self.edf_path = filepath
                self.eeg_loaded = True
                
                fname = os.path.basename(filepath)
                self.edf_label.setText(f"Loaded: {fname}")
                self.eeg_info.setText(
                    f"Channels: {n_channels}\n"
                    f"Samples: {n_samples}\n"
                    f"Pins mapped: {n_pins}"
                )
                self.status_bar.showMessage(f"Loaded {fname} - {n_pins} electrodes mapped")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load EDF:\n{str(e)}")
    
    def on_resolution_changed(self, text):
        """Handle resolution change."""
        new_size = int(text)
        if new_size != self.crystal.grid_size:
            self.crystal.resize(new_size)
            if self.eeg_loaded:
                n_pins = self.eeg.map_electrodes(new_size)
                self.eeg_info.setText(
                    f"Channels: {len(self.eeg.ch_names)}\n"
                    f"Samples: {self.eeg.data.shape[1]}\n"
                    f"Pins mapped: {n_pins}"
                )
            self.update_display()
            self.status_bar.showMessage(f"Resolution changed to {new_size}x{new_size}")
    
    def on_lr_changed(self, value):
        """Handle learning rate change."""
        self.crystal.learning_rate = value
    
    def on_amp_changed(self, index):
        """Handle amplification change."""
        amp_values = [1e8, 1e9, 1e10, 1e11]
        self.eeg.amplification = amp_values[index]
        self.status_bar.showMessage(f"Amplification set to {amp_values[index]:.0e}")
    
    def on_coupling_changed(self, value):
        """Handle coupling strength change."""
        self.crystal.coupling_strength = value
    
    def on_speed_changed(self, value):
        """Handle speed slider change."""
        if self.is_running:
            # Map 1-100 to 100ms-1ms interval
            interval = max(1, 101 - value)
            self.timer.setInterval(interval)
    
    def toggle_simulation(self):
        """Start/stop simulation."""
        if not self.eeg_loaded:
            QMessageBox.warning(self, "Warning", "Please load an EDF file first.")
            return
        
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.start_btn.setText("▶ Start")
            self.status_bar.showMessage("Simulation paused")
        else:
            interval = max(1, 101 - self.speed_slider.value())
            self.timer.start(interval)
            self.is_running = True
            self.start_btn.setText("⏸ Pause")
            self.status_bar.showMessage("Simulation running...")
    
    def simulation_step(self):
        """One step of simulation."""
        I = self.eeg.get_input_current(self.crystal.grid_size)
        self.crystal.step(I, learning=True)
        
        # Update progress
        target = self.target_steps_spin.value()
        self.progress_bar.setMaximum(target)
        self.progress_bar.setValue(min(self.crystal.learning_steps, target))
        
        # Update display every few steps for performance
        if self.crystal.learning_steps % 5 == 0:
            self.update_display()
        
        # Auto-stop at target
        if self.crystal.learning_steps >= target:
            self.toggle_simulation()
            self.status_bar.showMessage(f"Completed {target} steps - Crystal ready to save!")
    
    def reset_crystal(self):
        """Reset crystal to initial state."""
        self.crystal.init_arrays()
        self.crystal.total_spikes = 0
        self.crystal.learning_steps = 0
        if self.eeg_loaded:
            self.eeg.current_idx = 0
        self.update_display()
        self.status_bar.showMessage("Crystal reset")
    
    def update_display(self):
        """Update visualization."""
        # Activity
        activity_img = self.crystal.render_activity(400)
        
        # Draw electrode pins on activity
        if self.eeg_loaded:
            scale = 400 / self.crystal.grid_size
            for r, c in self.eeg.pin_coords:
                x, y = int(c * scale), int(r * scale)
                cv2.circle(activity_img, (x, y), 3, (0, 255, 0), -1)
        
        h, w, ch = activity_img.shape
        qimg = QImage(activity_img.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self.activity_label.setPixmap(QPixmap.fromImage(qimg))
        
        # Crystal
        crystal_img = self.crystal.render_crystal(400)
        h, w, ch = crystal_img.shape
        qimg = QImage(crystal_img.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self.crystal_label.setPixmap(QPixmap.fromImage(qimg))
        
        # Stats
        self.stats_label.setText(
            f"Steps: {self.crystal.learning_steps}\n"
            f"Spikes: {self.crystal.total_spikes:,}\n"
            f"Energy: {self.crystal.get_energy():.1f}\n"
            f"Entropy: {self.crystal.get_entropy():.2f}"
        )
        
        self.progress_bar.setValue(self.crystal.learning_steps)
    
    def save_crystal(self):
        """Save crystal to file."""
        if self.crystal.learning_steps == 0:
            QMessageBox.warning(self, "Warning", "No crystal to save - run some training first.")
            return
        
        default_name = f"crystal_{self.crystal.grid_size}x{self.crystal.grid_size}_{self.crystal.learning_steps}steps.npz"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Crystal", default_name, "NumPy Archive (*.npz);;All Files (*)"
        )
        
        if filepath:
            try:
                # Prepare pin data
                pin_coords = np.array(self.eeg.pin_coords) if self.eeg.pin_coords else np.array([])
                pin_names = np.array(self.eeg.pin_names) if self.eeg.pin_names else np.array([])
                
                np.savez(filepath,
                    # Weights
                    weights_up=self.crystal.weights_up,
                    weights_down=self.crystal.weights_down,
                    weights_left=self.crystal.weights_left,
                    weights_right=self.crystal.weights_right,
                    # Pin map
                    pin_coords=pin_coords,
                    pin_names=pin_names,
                    # Metadata
                    grid_size=self.crystal.grid_size,
                    learning_steps=self.crystal.learning_steps,
                    total_spikes=self.crystal.total_spikes,
                    learning_rate=self.crystal.learning_rate,
                    edf_source=os.path.basename(self.edf_path) if self.edf_path else "",
                    created=datetime.now().isoformat()
                )
                
                self.status_bar.showMessage(f"Saved crystal to {os.path.basename(filepath)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def load_crystal(self):
        """Load crystal from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Crystal", "", "NumPy Archive (*.npz);;All Files (*)"
        )
        
        if filepath:
            try:
                data = np.load(filepath, allow_pickle=True)
                
                # Get grid size and resize
                grid_size = int(data['grid_size'])
                self.crystal.resize(grid_size)
                self.resolution_combo.setCurrentText(str(grid_size))
                
                # Load weights
                self.crystal.weights_up = data['weights_up']
                self.crystal.weights_down = data['weights_down']
                self.crystal.weights_left = data['weights_left']
                self.crystal.weights_right = data['weights_right']
                
                # Load stats
                self.crystal.learning_steps = int(data['learning_steps'])
                self.crystal.total_spikes = int(data['total_spikes'])
                if 'learning_rate' in data:
                    self.crystal.learning_rate = float(data['learning_rate'])
                    self.lr_spin.setValue(self.crystal.learning_rate)
                
                # Load pin map
                if 'pin_coords' in data and len(data['pin_coords']) > 0:
                    self.eeg.pin_coords = [tuple(c) for c in data['pin_coords']]
                    self.eeg.pin_names = list(data['pin_names'])
                
                self.update_display()
                self.status_bar.showMessage(f"Loaded crystal from {os.path.basename(filepath)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark theme
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = CrystalMakerWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()