"""
Cortical Stack Node
====================

Multiple Izhikevich neuron layers that interact like cortical sheets.
Each layer has different dynamics, and connections BETWEEN layers
learn via STDP - they fire-and-wire together.

Architecture (simplified from 6 layers to 4 functional layers):
- L_input (Layer 4 analog): Receives external input, fast dynamics
- L_process (Layer 2/3 analog): Horizontal processing, regular spiking
- L_output (Layer 5 analog): Output generation, bursting
- L_feedback (Layer 6 analog): Feedback to input, slow dynamics

Key Innovation:
- L_input layer weights loaded from crystal (EEG-trained)
- Grid size automatically read from crystal file
- Inter-layer weights LEARN via STDP as the stack runs
- Different Izhikevich parameters per layer type

Author: Built for Antti's consciousness crystallography research
"""

import os
import numpy as np
import cv2

# --- HOST IMPORT BLOCK ---
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


class CorticalStackNode(BaseNode):
    """
    Multi-layer cortical stack with inter-layer STDP learning.
    Loads crystal and auto-configures grid size from file.
    """
    
    NODE_NAME = "Cortical Stack"
    NODE_CATEGORY = "Neural"
    NODE_COLOR = QtGui.QColor(80, 40, 120) if QtGui else None
    
    # Layer types with different Izhikevich parameters
    LAYER_PARAMS = {
        'L_input': {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0},      # Fast spiking
        'L_process': {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},   # Regular spiking
        'L_output': {'a': 0.02, 'b': 0.2, 'c': -55.0, 'd': 4.0},    # Bursting
        'L_feedback': {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.5}  # Slow
    }
    
    # Inter-layer connectivity pattern
    CONNECTIVITY = {
        'L_input': ['L_process'],
        'L_process': ['L_output', 'L_input'],
        'L_output': ['L_feedback'],
        'L_feedback': ['L_input']
    }
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'image_in': 'image',
            'signal_in': 'signal',
            'learning_rate': 'signal',
            'thalamic_gate': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'stack_view': 'image',
            'layer_views': 'image',
            'output_signal': 'signal',
            'feedback_signal': 'signal',
            'input_signal': 'signal',
            'process_signal': 'signal',
            'coherence': 'signal',
            'total_spikes': 'signal'
        }
        
        # === INTERNAL SETTINGS ===
        self.crystal_path = ""
        self._last_crystal_path = ""
        self.stdp_lr = 0.001
        self.input_gain = 30.0
        self.inter_layer_gain = 1.0
        self.coupling_strength = 5.0
        self.enable_learning = True
        
        # Grid size - will be set when crystal loads
        self.grid_size = 32
        self.layer_names = ['L_input', 'L_process', 'L_output', 'L_feedback']
        
        # Crystal metadata
        self.crystal_loaded = False
        self.crystal_source = "None"
        self.crystal_grid_size = 0
        self.pin_coords = []
        self.pin_names = []
        
        # Layers dict
        self.layers = {}
        self.inter_weights = {}
        self.spike_traces = {}
        
        # STDP parameters
        self.trace_decay = 0.95
        self.weight_max = 2.0
        self.weight_min = 0.01
        
        # Simulation
        self.dt = 0.5
        self.step_count = 0
        self.thalamic_gate_value = 1.0
        
        # Statistics
        self.layer_spike_counts = {name: 0 for name in self.layer_names}
        self.coherence_value = 0.0
        
        # Initialize with default grid
        self._init_layers(self.grid_size)
        self._init_inter_weights()
        
        # Display - store as numpy array, not QImage
        self.display_array = None
        self._update_display()
    
    def get_config_options(self):
        return [
            ("Crystal File (.npz)", "crystal_path", self.crystal_path, None),
            ("STDP Learning Rate", "stdp_lr", self.stdp_lr, None),
            ("Input Gain", "input_gain", self.input_gain, None),
            ("Inter-layer Gain", "inter_layer_gain", self.inter_layer_gain, None),
            ("Coupling Strength", "coupling_strength", self.coupling_strength, None),
            ("Enable Learning", "enable_learning", self.enable_learning, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            new_grid = options.get('grid_size', self.grid_size)
            if new_grid != self.grid_size:
                self._init_layers(new_grid)
                self._init_inter_weights()
            
            for key, value in options.items():
                if key == 'grid_size':
                    continue
                if hasattr(self, key):
                    setattr(self, key, value)
            
            if 'crystal_path' in options and options['crystal_path']:
                self._last_crystal_path = ""
    
    def _init_layers(self, grid_size):
        """Initialize all cortical layers for given grid size."""
        self.grid_size = grid_size
        n = grid_size
        
        for name in self.layer_names:
            params = self.LAYER_PARAMS[name]
            self.layers[name] = {
                'v': np.ones((n, n), dtype=np.float32) * -65.0,
                'u': np.ones((n, n), dtype=np.float32) * -13.0,
                'a': params['a'],
                'b': params['b'],
                'c': params['c'],
                'd': params['d'],
                'weights_up': np.ones((n, n), dtype=np.float32) * 0.5,
                'weights_down': np.ones((n, n), dtype=np.float32) * 0.5,
                'weights_left': np.ones((n, n), dtype=np.float32) * 0.5,
                'weights_right': np.ones((n, n), dtype=np.float32) * 0.5,
                'spikes': np.zeros((n, n), dtype=bool),
                'spike_trace': np.zeros((n, n), dtype=np.float32),
                'activity': np.zeros((n, n), dtype=np.float32),
                'I': np.zeros((n, n), dtype=np.float32)
            }
        
        self.spike_traces = {name: np.zeros((n, n), dtype=np.float32) for name in self.layer_names}
    
    def _init_inter_weights(self):
        """Initialize inter-layer connection weights."""
        n = self.grid_size
        
        for source_layer, target_layers in self.CONNECTIVITY.items():
            for target_layer in target_layers:
                key = f"{source_layer}_to_{target_layer}"
                self.inter_weights[key] = np.ones((n, n), dtype=np.float32) * 0.1
    
    def _check_crystal_path(self):
        """Check if crystal path changed and load if needed."""
        path = str(self.crystal_path or "").strip().strip('"').strip("'")
        path = path.replace("\\", "/")
        
        if path and path != self._last_crystal_path:
            self._last_crystal_path = path
            self.crystal_path = path
            self._load_crystal(path)
    
    def _ensure_consistent_sizes(self):
        """Self-healing: ensure all arrays match current grid_size."""
        n = self.grid_size
        
        needs_layer_reinit = False
        if 'L_input' in self.layers:
            layer_size = self.layers['L_input']['v'].shape[0]
            if layer_size != n:
                needs_layer_reinit = True
        else:
            needs_layer_reinit = True
        
        if needs_layer_reinit:
            print(f"[CorticalStack] Reinitializing layers to {n}x{n}")
            self._init_layers(n)
        
        needs_inter_reinit = False
        if self.inter_weights:
            first_key = list(self.inter_weights.keys())[0]
            if self.inter_weights[first_key].shape[0] != n:
                needs_inter_reinit = True
        else:
            needs_inter_reinit = True
        
        if needs_inter_reinit:
            print(f"[CorticalStack] Reinitializing inter-weights to {n}x{n}")
            self._init_inter_weights()
        
        if self.spike_traces:
            first_name = list(self.spike_traces.keys())[0]
            if self.spike_traces[first_name].shape[0] != n:
                self.spike_traces = {name: np.zeros((n, n), dtype=np.float32) for name in self.layer_names}
    
    def _load_crystal(self, path):
        """Load crystal and auto-configure from its settings."""
        if not os.path.exists(path):
            print(f"[CorticalStack] Crystal file not found: {path}")
            return False
        
        try:
            data = np.load(path, allow_pickle=True)
            
            if 'grid_size' in data:
                crystal_grid_size = int(data['grid_size'])
            elif 'weights_up' in data:
                crystal_grid_size = data['weights_up'].shape[0]
            else:
                print(f"[CorticalStack] Cannot determine grid size from crystal")
                return False
            
            print(f"[CorticalStack] Setting grid to {crystal_grid_size}x{crystal_grid_size}")
            self.grid_size = crystal_grid_size
            self._init_layers(crystal_grid_size)
            self._init_inter_weights()
            
            if 'weights_up' in data:
                self.layers['L_input']['weights_up'] = data['weights_up'].astype(np.float32)
                self.layers['L_input']['weights_down'] = data['weights_down'].astype(np.float32)
                self.layers['L_input']['weights_left'] = data['weights_left'].astype(np.float32)
                self.layers['L_input']['weights_right'] = data['weights_right'].astype(np.float32)
            
            if 'pin_coords' in data and len(data['pin_coords']) > 0:
                self.pin_coords = [tuple(c) for c in data['pin_coords']]
            else:
                self.pin_coords = []
            
            if 'pin_names' in data and len(data['pin_names']) > 0:
                self.pin_names = list(data['pin_names'])
            else:
                self.pin_names = []
            
            self.crystal_source = str(data.get('edf_source', os.path.basename(path)))
            self.crystal_grid_size = crystal_grid_size
            self.crystal_loaded = True
            
            print(f"[CorticalStack] Loaded crystal: {crystal_grid_size}x{crystal_grid_size}")
            print(f"  Source: {self.crystal_source}")
            print(f"  Pins: {len(self.pin_coords)}")
            if 'learning_steps' in data:
                print(f"  Training steps: {int(data['learning_steps'])}")
            
            return True
            
        except Exception as e:
            print(f"[CorticalStack] Error loading crystal: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _read_input(self, name, default=None):
        """Read an input value."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                if val is None:
                    return default
                return val
            except:
                return default
        return default
    
    def _read_image_input(self, name):
        """Read an image input, converting QImage to numpy if needed."""
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first")
                if val is None:
                    return None
                
                # Already numpy array
                if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                    return val
                
                # QImage conversion
                if hasattr(val, 'width') and hasattr(val, 'height') and hasattr(val, 'bits'):
                    width = val.width()
                    height = val.height()
                    bytes_per_line = val.bytesPerLine()
                    ptr = val.bits()
                    if ptr is None:
                        return None
                    
                    try:
                        ptr.setsize(height * bytes_per_line)
                        arr = np.array(ptr).reshape(height, bytes_per_line)
                        fmt = val.format()
                        if fmt == 4:
                            arr = arr[:, :width*4].reshape(height, width, 4)
                            arr = arr[:, :, :3]
                        elif fmt == 13:
                            arr = arr[:, :width*3].reshape(height, width, 3)
                        else:
                            if bytes_per_line >= width * 3:
                                arr = arr[:, :width*3].reshape(height, width, 3)
                            else:
                                arr = arr[:, :width]
                        return arr.astype(np.float32)
                    except Exception as e:
                        return None
            except:
                pass
        return None
    
    def step(self):
        self.step_count += 1
        
        self._check_crystal_path()
        self._ensure_consistent_sizes()
        
        gate = self._read_input('thalamic_gate', 1.0)
        if gate is not None:
            self.thalamic_gate_value = float(np.clip(gate, 0.0, 1.0))
        
        ext_signal = self._read_input('signal_in', 0.0)
        if ext_signal is not None:
            ext_signal = float(ext_signal)
        else:
            ext_signal = 0.0
        
        ext_image = self._read_image_input('image_in')
        
        self._apply_external_input(ext_signal * self.thalamic_gate_value, ext_image)
        self._update_layers()
        
        if self.enable_learning:
            lr = self._read_input('learning_rate', self.stdp_lr)
            if lr is not None:
                self._apply_inter_layer_stdp(float(lr))
        
        self._calculate_coherence()
        
        if self.step_count % 4 == 0:
            self._update_display()
    
    def _apply_external_input(self, signal, image):
        """Apply external input to input layer."""
        layer = self.layers['L_input']
        n = self.grid_size
        
        layer['I'] = np.ones((n, n), dtype=np.float32) * signal * self.input_gain
        
        if image is not None:
            try:
                if len(image.shape) == 3:
                    img_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = image
                img_resized = cv2.resize(img_gray.astype(np.float32), (n, n))
                img_norm = img_resized / 255.0
                layer['I'] += img_norm * self.input_gain
            except:
                pass
    
    def _update_layers(self):
        """Update all layers with Izhikevich dynamics."""
        n = self.grid_size
        
        inter_currents = {name: np.zeros((n, n), dtype=np.float32) for name in self.layer_names}
        
        for source_layer, target_layers in self.CONNECTIVITY.items():
            source_spikes = self.layers[source_layer]['spikes'].astype(np.float32)
            source_v = self.layers[source_layer]['v']
            
            for target_layer in target_layers:
                key = f"{source_layer}_to_{target_layer}"
                weights = self.inter_weights[key]
                
                spike_current = source_spikes * 30.0
                v_normalized = (source_v + 65.0) / 95.0
                v_normalized = np.clip(v_normalized, 0, 1)
                graded_current = v_normalized * 10.0
                spread = cv2.GaussianBlur(spike_current, (5, 5), 1.0)
                
                inter_currents[target_layer] += weights * (spike_current + spread + graded_current) * self.inter_layer_gain
        
        for name in self.layer_names:
            layer = self.layers[name]
            a, b, c, d = layer['a'], layer['b'], layer['c'], layer['d']
            v = layer['v']
            u = layer['u']
            
            I = layer['I'] + inter_currents[name]
            
            v_up = np.roll(v, -1, axis=0)
            v_down = np.roll(v, 1, axis=0)
            v_left = np.roll(v, -1, axis=1)
            v_right = np.roll(v, 1, axis=1)
            
            neighbor_influence = (
                layer['weights_up'] * v_up +
                layer['weights_down'] * v_down +
                layer['weights_left'] * v_left +
                layer['weights_right'] * v_right
            )
            total_weight = (layer['weights_up'] + layer['weights_down'] + 
                           layer['weights_left'] + layer['weights_right'])
            neighbor_avg = neighbor_influence / (total_weight + 1e-6)
            
            I_coupling = self.coupling_strength * (neighbor_avg - v)
            I_coupling = np.clip(I_coupling, -50, 50)
            
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I + I_coupling) * self.dt
            du = a * (b * v - u) * self.dt
            
            v = v + dv
            u = u + du
            
            v = np.clip(v, -100, 50)
            u = np.clip(u, -50, 50)
            
            spikes = v >= 30.0
            v[spikes] = c
            u[spikes] += d
            
            layer['v'] = v
            layer['u'] = u
            layer['spikes'] = spikes
            layer['activity'] = 0.9 * layer['activity'] + 0.1 * spikes.astype(np.float32) * 100
            
            layer['spike_trace'] = layer['spike_trace'] * self.trace_decay
            layer['spike_trace'][spikes] = 1.0
            
            self.spike_traces[name] = layer['spike_trace']
            self.layer_spike_counts[name] = int(np.sum(spikes))
    
    def _apply_inter_layer_stdp(self, lr):
        """Apply STDP to inter-layer connections."""
        if lr <= 0:
            return
        
        for source_layer, target_layers in self.CONNECTIVITY.items():
            source_spikes = self.layers[source_layer]['spikes']
            source_trace = self.spike_traces[source_layer]
            
            for target_layer in target_layers:
                target_spikes = self.layers[target_layer]['spikes']
                target_trace = self.spike_traces[target_layer]
                
                key = f"{source_layer}_to_{target_layer}"
                weights = self.inter_weights[key]
                
                dw_ltp = lr * target_spikes.astype(np.float32) * np.mean(source_trace)
                dw_ltd = 0.5 * lr * source_spikes.astype(np.float32) * np.mean(target_trace)
                
                weights = weights + dw_ltp - dw_ltd
                weights = np.clip(weights, self.weight_min, self.weight_max)
                self.inter_weights[key] = weights
    
    def _calculate_coherence(self):
        """Calculate cross-layer synchronization."""
        activities = [np.mean(self.layers[name]['spikes']) for name in self.layer_names]
        if max(activities) > 0:
            self.coherence_value = float(np.std(activities) / (np.mean(activities) + 0.001))
        else:
            self.coherence_value = 0.0
    
    def _update_display(self):
        """Create visualization - store as numpy array."""
        n = self.grid_size
        cell_size = 128
        info_width = 200
        w = cell_size * 2 + info_width
        h = cell_size * 2 + 60
        
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        cv2.putText(img, "CORTICAL STACK", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 80, 200), 2)
        
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        colors = [(0, 255, 255), (0, 255, 0), (255, 100, 0), (255, 0, 255)]
        
        for idx, name in enumerate(self.layer_names):
            col, row = positions[idx]
            x = col * cell_size
            y = 40 + row * cell_size
            
            activity = self.layers[name]['activity']
            act_norm = np.clip(activity / 50.0, 0, 1)
            act_img = (act_norm * 255).astype(np.uint8)
            act_colored = cv2.applyColorMap(act_img, cv2.COLORMAP_INFERNO)
            act_resized = cv2.resize(act_colored, (cell_size - 4, cell_size - 24))
            
            img[y+20:y+cell_size-4, x+2:x+cell_size-2] = act_resized
            
            color = colors[idx]
            label = name.replace('L_', '')
            spikes = self.layer_spike_counts[name]
            cv2.putText(img, f"{label}: {spikes}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.rectangle(img, (x, y), (x + cell_size - 2, y + cell_size - 2), color, 1)
        
        info_x = cell_size * 2 + 10
        info_y = 50
        
        cv2.putText(img, f"Step: {self.step_count}", (info_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Grid: {self.grid_size}x{self.grid_size}", (info_x, info_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Coherence: {self.coherence_value:.2f}", (info_x, info_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1)
        
        if self.crystal_loaded:
            source_short = self.crystal_source[:12] if len(self.crystal_source) > 12 else self.crystal_source
            cv2.putText(img, f"Crystal: {source_short}", (info_x, info_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        else:
            cv2.putText(img, "Crystal: None", (info_x, info_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        cv2.putText(img, f"Gate: {self.thalamic_gate_value:.2f}", (info_x, info_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 50), 1)
        
        cv2.putText(img, "Inter-layer:", (info_x, info_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y_offset = 130
        for key, weights in self.inter_weights.items():
            mean_w = np.mean(weights)
            short_key = key.replace('L_', '').replace('_to_', '>')
            cv2.putText(img, f"{short_key}: {mean_w:.3f}", (info_x, info_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            y_offset += 15
        
        # Store as RGB numpy array (convert from BGR)
        self.display_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_output(self, port_name):
        if port_name == 'stack_view':
            # Return numpy array, not QImage
            return self.display_array
        elif port_name == 'layer_views':
            return self.display_array
        elif port_name == 'output_signal':
            return float(np.mean(self.layers['L_output']['v']))
        elif port_name == 'feedback_signal':
            return float(np.mean(self.layers['L_feedback']['v']))
        elif port_name == 'input_signal':
            return float(np.mean(self.layers['L_input']['v']))
        elif port_name == 'process_signal':
            return float(np.mean(self.layers['L_process']['v']))
        elif port_name == 'coherence':
            return self.coherence_value
        elif port_name == 'total_spikes':
            return sum(self.layer_spike_counts.values())
        return None
    
    def get_display_image(self):
        """Return QImage for the node's own display panel."""
        if self.display_array is not None and QtGui:
            h, w = self.display_array.shape[:2]
            return QtGui.QImage(self.display_array.data, w, h, w * 3, 
                              QtGui.QImage.Format.Format_RGB888).copy()
        return None