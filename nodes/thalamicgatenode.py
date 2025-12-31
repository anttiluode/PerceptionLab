"""
Thalamic Gate Node
==================

The thalamus is the brain's relay station and gatekeeper. It controls
what information flows between cortical areas, when attention shifts,
and regulates arousal/sleep states.

This node implements thalamic-like gating:
1. RELAY MODE: Pass signals through with gain control
2. BURST MODE: Rhythmic gating (sleep spindles, alpha blocking)
3. ATTENTION MODE: Selective gating based on salience
4. AROUSAL MODE: Global gain modulation based on activity level

Author: Built for Antti's consciousness crystallography research
"""

import numpy as np
import cv2
from collections import deque

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


class ThalamicGateNode(BaseNode):
    """
    Thalamic relay and gating node.
    Controls information flow like the brain's thalamus.
    """
    
    NODE_NAME = "Thalamic Gate"
    NODE_CATEGORY = "Neural"
    NODE_COLOR = QtGui.QColor(200, 100, 50) if QtGui else None
    
    # Operating modes
    MODES = ['relay', 'burst', 'attention', 'arousal', 'oscillator']
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'signal_in': 'signal',
            'image_in': 'image',
            'cortical_feedback': 'signal',
            'arousal_level': 'signal',
            'attention_target': 'signal',
            'reset': 'signal'
        }
        
        self.outputs = {
            'signal_out': 'signal',
            'image_out': 'image',
            'gate_view': 'image',
            'gate_state': 'signal',
            'burst_phase': 'signal',
            'relay_gain': 'signal'
        }
        
        # === INTERNAL SETTINGS ===
        self.mode = 'relay'
        self.base_gain = 1.0
        self.gate_threshold = 0.3
        self.burst_frequency = 10.0
        self.burst_duty_cycle = 0.5
        self.attention_sharpness = 2.0
        self.arousal_sensitivity = 1.0
        
        # Internal state
        self.gate_state = 1.0
        self.phase = 0.0
        self.relay_gain = 1.0
        
        # History
        self.input_history = deque([0.0] * 100, maxlen=100)
        self.feedback_history = deque([0.0] * 100, maxlen=100)
        
        # Burst mode state
        self.trn_state = 0.0
        self.trn_threshold = 0.5
        self.refractory = 0
        self.refractory_period = 10
        
        # Cache for image output - store as numpy array
        self._last_image_in = None
        
        # Statistics
        self.step_count = 0
        self.total_passed = 0.0
        self.total_blocked = 0.0
        
        # Display - store as numpy array, not QImage
        self.display_array = None
        self._update_display()
    
    def get_config_options(self):
        return [
            ("Mode", "mode", self.mode, None),
            ("Base Gain", "base_gain", self.base_gain, None),
            ("Gate Threshold", "gate_threshold", self.gate_threshold, None),
            ("Burst Frequency (Hz)", "burst_frequency", self.burst_frequency, None),
            ("Burst Duty Cycle", "burst_duty_cycle", self.burst_duty_cycle, None),
            ("Attention Sharpness", "attention_sharpness", self.attention_sharpness, None),
            ("Arousal Sensitivity", "arousal_sensitivity", self.arousal_sensitivity, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
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
                        if fmt == 4:  # Format_RGB32 or Format_ARGB32
                            arr = arr[:, :width*4].reshape(height, width, 4)
                            arr = arr[:, :, :3]
                        elif fmt == 13:  # Format_RGB888
                            arr = arr[:, :width*3].reshape(height, width, 3)
                        else:
                            if bytes_per_line >= width * 3:
                                arr = arr[:, :width*3].reshape(height, width, 3)
                            else:
                                arr = arr[:, :width]
                        return arr.astype(np.float32)
                    except Exception as e:
                        print(f"[ThalamicGate] QImage conversion error: {e}")
                        return None
            except Exception as e:
                print(f"[ThalamicGate] Image read error: {e}")
        return None
    
    def step(self):
        self.step_count += 1
        
        # Read inputs
        signal_in = self._read_input('signal_in', 0.0)
        if signal_in is not None:
            signal_in = float(signal_in)
        else:
            signal_in = 0.0
        
        self._last_image_in = self._read_image_input('image_in')
        
        cortical_feedback = self._read_input('cortical_feedback', 0.5)
        if cortical_feedback is not None:
            cortical_feedback = float(cortical_feedback)
        else:
            cortical_feedback = 0.5
        
        arousal = self._read_input('arousal_level', 0.5)
        if arousal is not None:
            arousal = float(arousal)
        else:
            arousal = 0.5
        
        attention = self._read_input('attention_target', 0.5)
        if attention is not None:
            attention = float(attention)
        else:
            attention = 0.5
        
        # Update history
        self.input_history.append(abs(signal_in))
        self.feedback_history.append(cortical_feedback)
        
        # Calculate gate state based on mode
        if self.mode == 'relay':
            self._update_relay_mode(arousal)
        elif self.mode == 'burst':
            self._update_burst_mode()
        elif self.mode == 'attention':
            self._update_attention_mode(attention, cortical_feedback)
        elif self.mode == 'arousal':
            self._update_arousal_mode(arousal, signal_in)
        elif self.mode == 'oscillator':
            self._update_oscillator_mode()
        
        # Apply gating
        self.relay_gain = self.base_gain * self.gate_state
        
        # Statistics
        if self.gate_state > 0.5:
            self.total_passed += 1
        else:
            self.total_blocked += 1
        
        # Display
        if self.step_count % 8 == 0:
            self._update_display()
    
    def _update_relay_mode(self, arousal):
        """Simple relay with arousal-modulated gain."""
        self.gate_state = 1.0
        self.relay_gain = self.base_gain * (0.5 + arousal * self.arousal_sensitivity)
    
    def _update_burst_mode(self):
        """Rhythmic bursting like sleep spindles."""
        dt = 1.0 / 60.0
        self.phase += 2 * np.pi * self.burst_frequency * dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        cycle_position = (np.sin(self.phase) + 1) / 2
        self.gate_state = 1.0 if cycle_position > (1 - self.burst_duty_cycle) else 0.0
    
    def _update_attention_mode(self, attention_target, cortical_feedback):
        """Selective gating based on attention signal."""
        recent_input = np.mean(list(self.input_history)[-20:])
        importance = cortical_feedback * self.attention_sharpness
        
        if recent_input > self.gate_threshold:
            self.gate_state = min(1.0, importance)
        else:
            self.gate_state = max(0.1, importance * 0.5)
    
    def _update_arousal_mode(self, arousal, signal_in):
        """Arousal-dependent gating."""
        if arousal < 0.3:
            self.phase += 2 * np.pi * 2.0 * (1/60.0)
            self.gate_state = 0.5 + 0.5 * np.sin(self.phase)
        elif arousal < 0.6:
            self.gate_state = arousal * 1.5
        else:
            self.gate_state = 1.0
        
        self.gate_state = float(np.clip(self.gate_state, 0, 1))
    
    def _update_oscillator_mode(self):
        """Free-running thalamic oscillator."""
        if self.refractory > 0:
            self.refractory -= 1
            self.trn_state *= 0.9
        else:
            recent = np.mean(list(self.input_history)[-10:])
            self.trn_state += 0.1 * (recent + 0.5)
            
            if self.trn_state > self.trn_threshold:
                self.trn_state = 0.0
                self.refractory = self.refractory_period
                self.gate_state = 0.0
            else:
                self.gate_state = 1.0
        
        self.phase = self.trn_state / self.trn_threshold * np.pi
    
    def get_output(self, port_name):
        if port_name == 'signal_out':
            signal_in = self._read_input('signal_in', 0.0)
            if signal_in is not None:
                return float(signal_in) * self.relay_gain
            return 0.0
        
        elif port_name == 'image_out':
            if self._last_image_in is not None:
                img = self._last_image_in * self.gate_state
                # Ensure proper range
                if img.max() <= 1.0:
                    img = img * 255
                return img.astype(np.uint8)
            return None
        
        elif port_name == 'gate_view':
            # Return numpy array, not QImage
            return self.display_array
        
        elif port_name == 'gate_state':
            return float(self.gate_state)
        
        elif port_name == 'burst_phase':
            return float(self.phase)
        
        elif port_name == 'relay_gain':
            return float(self.relay_gain)
        
        return None
    
    def _update_display(self):
        """Create visualization of gate state - store as numpy array."""
        w, h = 300, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "THALAMIC GATE", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 50), 2)
        
        # Mode
        cv2.putText(img, f"Mode: {self.mode.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Gate state bar
        bar_x, bar_y = 10, 70
        bar_w, bar_h = 180, 30
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        fill_w = int(bar_w * self.gate_state)
        color = (0, int(255 * self.gate_state), int(255 * (1 - self.gate_state)))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        
        cv2.putText(img, f"Gate: {self.gate_state:.2f}", (bar_x + bar_w + 10, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Gain
        cv2.putText(img, f"Gain: {self.relay_gain:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Phase
        if self.mode in ['burst', 'oscillator']:
            phase_deg = np.degrees(self.phase) % 360
            cv2.putText(img, f"Phase: {phase_deg:.0f} deg", (100, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Input history trace
        trace_y = 150
        trace_h = 40
        history = list(self.input_history)[-100:]
        if history and max(history) > 0:
            max_val = max(history) + 0.1
            for i in range(len(history) - 1):
                x1 = 10 + int(i * 2.8)
                x2 = 10 + int((i + 1) * 2.8)
                y1 = trace_y + trace_h - int(history[i] / max_val * trace_h)
                y2 = trace_y + trace_h - int(history[i + 1] / max_val * trace_h)
                cv2.line(img, (x1, y1), (x2, y2), (100, 200, 100), 1)
        
        cv2.putText(img, "Input", (10, trace_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Pass rate
        total = self.total_passed + self.total_blocked
        if total > 0:
            pass_rate = self.total_passed / total
            cv2.putText(img, f"Pass: {pass_rate:.1%}", (200, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        # Store as RGB numpy array (convert from BGR)
        self.display_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_display_image(self):
        """Return QImage for the node's own display panel."""
        if self.display_array is not None and QtGui:
            h, w = self.display_array.shape[:2]
            return QtGui.QImage(self.display_array.data, w, h, w * 3, 
                              QtGui.QImage.Format.Format_RGB888).copy()
        return None