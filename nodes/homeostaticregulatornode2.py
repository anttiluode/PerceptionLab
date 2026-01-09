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
            self.input_data = {}
        def get_blended_input(self, name, mode=None): return None

class HomeostaticRegulatorNode2(BaseNode):
    """
    Homeostatic Regulator Node v2
    -----------------------------
    A virtual battery that manages the energy budget of attention.
    
    Inputs:
    - novelty: surprise signal from manifold (wakes up system)
    - intensity_in: signal from amplifier (higher = more metabolic cost)
    - metabolic_cost: spectrum input (alternative cost signal)
    - override: manual wake/sleep control
    
    Outputs:
    - focus_command: 0 (sleep) to 1 (alert) - controls ephaptic lens
    - energy_level: battery state 0-1
    - state_view: visualization
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(255, 200, 100)
    
    def __init__(self):
        super().__init__()
        self.node_title = "Homeostatic Regulator"
        
        self.inputs = {
            'novelty': 'signal',
            'intensity_in': 'signal',      # From SignalAmplifier
            'metabolic_cost': 'spectrum',   # Alternative: from spectrum
            'override': 'signal',
        }
        
        self.outputs = {
            'focus_command': 'signal',
            'energy_level': 'signal',
            'state_view': 'image',
            'state_name': 'spectrum',
        }
        
        self.config = {
            'max_energy': 100.0,
            'wake_threshold': 30.0,
            'sleep_threshold': 80.0,
            'base_drain': 0.5,
            'focus_drain_mult': 2.0,
            'recharge_rate': 1.5,
            'novelty_sensitivity': 0.4,
            'intensity_scale': 0.1,  # How much intensity affects drain
        }
        
        self._output_values = {}
        self._init_state()

    def _init_state(self):
        self.energy = self.config['max_energy'] * 0.7
        self.focus = 0.5
        self.state = "AWAKE"
        self.sleep_pressure = 0.0
        self.wake_drive = 0.0
        self.history = []
        self.current_intensity = 0.0

    def get_input(self, name):
        if hasattr(self, 'get_blended_input'):
            result = self.get_blended_input(name, 'sum')
            return result
        if name in self.input_data and len(self.input_data[name]) > 0:
            val = self.input_data[name]
            return val[0] if isinstance(val, list) else val
        return None

    def set_output(self, name, value):
        self._output_values[name] = value
    
    def get_output(self, name):
        return self._output_values.get(name, None)

    def _to_scalar(self, val):
        """Safely convert any input to a scalar float"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return 0.0
            return float(np.mean(np.abs(val)))
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return 0.0
            return float(np.mean(np.abs(val)))
        try:
            return float(val)
        except:
            return 0.0

    def step(self):
        # Get inputs - safely convert to scalars
        novelty = np.clip(self._to_scalar(self.get_input('novelty')), 0, 1)
        intensity = self._to_scalar(self.get_input('intensity_in'))
        spectrum_cost = self._to_scalar(self.get_input('metabolic_cost'))
        override_raw = self.get_input('override')
        
        # Combine intensity sources
        self.current_intensity = intensity + spectrum_cost
        ext_cost = np.clip(self.current_intensity * self.config['intensity_scale'], 0, 1)
        
        override = self._to_scalar(override_raw) if override_raw is not None else None
        
        max_e = self.config['max_energy']
        wake_th = self.config['wake_threshold']
        sleep_th = self.config['sleep_threshold']
        base_drain = self.config['base_drain']
        focus_mult = self.config['focus_drain_mult']
        recharge = self.config['recharge_rate']
        nov_sens = self.config['novelty_sensitivity']
        
        # Update drives
        self.sleep_pressure = (max_e - self.energy) / max_e
        self.wake_drive = novelty * nov_sens
        
        # Manual override
        if override is not None and override != 0:
            if override > 0.7:
                self.state = "AWAKE"
                self.focus = 0.8
            elif override < 0.3:
                self.state = "SLEEPING"
                self.focus = 0.1
        
        # State transitions
        if self.state == "AWAKE":
            drain = base_drain + self.focus * focus_mult + ext_cost
            self.energy -= drain
            
            target_focus = 0.4 + novelty * 0.5
            self.focus = self.focus * 0.9 + target_focus * 0.1
            
            if self.energy < wake_th:
                self.state = "DROWSY"
                
        elif self.state == "DROWSY":
            drain = base_drain * 0.5 + ext_cost * 0.5
            self.energy -= drain
            self.focus = self.focus * 0.95
            
            if novelty > 0.6 and self.energy > wake_th * 1.2:
                self.state = "AWAKE"
                self.focus = 0.6
            elif self.energy < wake_th * 0.5:
                self.state = "SLEEPING"
                
        elif self.state == "SLEEPING":
            self.energy += recharge
            self.focus = max(0.05, self.focus * 0.9)
            
            if self.energy > sleep_th and novelty > 0.5:
                self.state = "DROWSY"
            elif self.energy >= max_e * 0.95:
                self.state = "DROWSY"
        
        # Clamp
        self.energy = np.clip(self.energy, 0, max_e)
        self.focus = np.clip(self.focus, 0.05, 1.0)
        
        # Outputs
        self.set_output('focus_command', float(self.focus))
        self.set_output('energy_level', float(self.energy / max_e))
        self.set_output('state_name', np.array([ord(c) for c in self.state], dtype=np.float32))
        
        # History
        self.history.append({
            'energy': self.energy / max_e,
            'focus': self.focus,
            'novelty': novelty,
            'intensity': ext_cost,
            'state': self.state
        })
        if len(self.history) > 200:
            self.history.pop(0)
        
        self._render_state()

    def _render_state(self):
        width = 300
        height = 200
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        max_e = self.config['max_energy']
        wake_th = self.config['wake_threshold']
        sleep_th = self.config['sleep_threshold']
        
        # Battery
        battery_x, battery_y = 20, 30
        battery_w, battery_h = 60, 120
        
        cv2.rectangle(canvas, (battery_x, battery_y), 
                     (battery_x + battery_w, battery_y + battery_h),
                     (100, 100, 100), 2)
        cv2.rectangle(canvas, (battery_x + 15, battery_y - 10),
                     (battery_x + 45, battery_y), (100, 100, 100), -1)
        
        fill_h = int((self.energy / max_e) * (battery_h - 4))
        fill_y = battery_y + battery_h - 2 - fill_h
        
        if self.state == "AWAKE":
            fill_color = (50, 200, 50)
        elif self.state == "DROWSY":
            fill_color = (50, 200, 200)
        else:
            fill_color = (200, 100, 50)
        
        if fill_h > 0:
            cv2.rectangle(canvas, (battery_x + 2, fill_y),
                         (battery_x + battery_w - 2, battery_y + battery_h - 2),
                         fill_color, -1)
        
        # Thresholds
        wake_y = battery_y + battery_h - int((wake_th / max_e) * battery_h)
        sleep_y = battery_y + battery_h - int((sleep_th / max_e) * battery_h)
        cv2.line(canvas, (battery_x - 5, wake_y), (battery_x + battery_w + 5, wake_y),
                (100, 100, 255), 1)
        cv2.line(canvas, (battery_x - 5, sleep_y), (battery_x + battery_w + 5, sleep_y),
                (100, 255, 100), 1)
        
        # Text
        state_colors = {
            "AWAKE": (50, 255, 50),
            "DROWSY": (50, 255, 255),
            "SLEEPING": (255, 150, 50)
        }
        cv2.putText(canvas, self.state, (100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors[self.state], 2)
        
        cv2.putText(canvas, f"Energy: {self.energy:.0f}%", (100, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"Focus: {self.focus:.2f}", (100, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"Intensity: {self.current_intensity:.2f}", (100, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # History graph
        graph_x, graph_y = 100, 140
        graph_w, graph_h = 180, 50
        
        cv2.rectangle(canvas, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h),
                     (50, 50, 50), 1)
        
        if len(self.history) > 1:
            n = len(self.history)
            for i in range(1, n):
                x1 = graph_x + int((i - 1) / n * graph_w)
                x2 = graph_x + int(i / n * graph_w)
                
                y1 = graph_y + graph_h - int(self.history[i-1]['energy'] * graph_h)
                y2 = graph_y + graph_h - int(self.history[i]['energy'] * graph_h)
                cv2.line(canvas, (x1, y1), (x2, y2), (50, 200, 50), 1)
                
                y1 = graph_y + graph_h - int(self.history[i-1]['focus'] * graph_h)
                y2 = graph_y + graph_h - int(self.history[i]['focus'] * graph_h)
                cv2.line(canvas, (x1, y1), (x2, y2), (50, 200, 200), 1)
        
        self.set_output('state_view', canvas)