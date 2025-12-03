"""
IHT Data Logger Node (Robust)
=============================
Logs metrics from the IHT Address pipeline to JSON for analysis.

FEATURES:
- Trigger Input: Wire any signal > 0.5 to 'trigger_export' to save.
- Auto-pathing: Saves to current working directory and PRINTS the path.
- Safety: Handles NaN/Inf values and Numpy types automatically.
- Visual Feedback: Display turns GREEN when saving.
"""

import numpy as np
import json
import time
import os
import cv2
from collections import deque

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class IHTDataLoggerNode(BaseNode):
    """
    Robust Data Logger for IHT Analysis.
    
    HOW TO USE:
    1. Wire signals (Entropy, PR, Health, etc.) to inputs.
    2. Wire a manual trigger or LFO to 'trigger_export'.
    3. When trigger > 0.5, it saves a JSON file.
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "IHT Data Logger"
    NODE_COLOR = QtGui.QColor(180, 60, 60)  # Reddish - recording
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'trigger_export': 'signal',      # RISING EDGE triggers export
            'address_entropy': 'signal',
            'participation_ratio': 'signal',
            'address_overlap': 'signal',
            'attractor_health': 'signal',
            'time_to_collapse': 'signal',
            'projection_loss': 'signal',
            'division_rate': 'signal',
            'stable_address': 'image'        # For computing stable fraction & coherence
        }
        
        self.outputs = {
            'step_count': 'signal',
            'export_done': 'signal',         # Pulses 1.0 when export completes
            'coherence': 'signal'            # Computed coherence output
        }
        
        # Internal State
        self.step_count = 0
        self.data_buffer = {
            'timeseries': {
                'step': [],
                'entropy': [],
                'participation_ratio': [],
                'overlap': [],
                'health': [],
                'ttc': [],
                'loss': [],
                'div_rate': [],
                'coherence': [],
                'stable_fraction': []
            }
        }
        
        # Coherence tracking
        self.history_len = 10
        self.stable_address_history = deque(maxlen=self.history_len)
        self.current_coherence = 0.0
        
        # Trigger State
        self.last_trigger_state = 0.0
        self.just_saved_timer = 0
        self.last_save_path = "None"
        
        # Configuration
        self.log_limit = 2000     # Keep last N points in RAM
        self.force_export_btn = False # Config button
        self.file_prefix = "iht_log"

    def compute_coherence(self, current_addr):
        """Calculates temporal stability of the address"""
        if current_addr is None: return 0.0
        
        # Normalize to 0-1
        curr = current_addr.astype(np.float32)
        if curr.ndim == 3: curr = np.mean(curr, axis=2)
        mx = np.max(curr)
        if mx > 1e-9: curr /= mx
        
        self.stable_address_history.append(curr)
        
        if len(self.stable_address_history) < 2: return 0.0
        
        # Compare current to average of recent history
        # (Simple correlation)
        history_stack = np.array(self.stable_address_history)
        avg = np.mean(history_stack, axis=0)
        
        # Flat correlation
        f1 = curr.flatten()
        f2 = avg.flatten()
        
        if np.std(f1) < 1e-9 or np.std(f2) < 1e-9:
            return 0.0
            
        corr = np.corrcoef(f1, f2)[0, 1]
        return float(max(0, corr))

    def export_json(self):
        """Writes the buffer to disk handling Numpy types safely"""
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.file_prefix}_{timestamp}.json"
            
            # Ensure absolute path
            full_path = os.path.abspath(os.path.join(os.getcwd(), filename))
            
            # Prepare export structure
            export_obj = {
                'meta': {
                    'timestamp': timestamp,
                    'total_steps': self.step_count,
                    'node_version': "2.0_Robust"
                },
                'data': self.data_buffer['timeseries']
            }
            
            # Custom encoder for Numpy/NaN/Inf
            def numpy_encoder(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    if np.isnan(obj): return None # Valid JSON null
                    if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

            with open(full_path, 'w') as f:
                json.dump(export_obj, f, indent=2, default=numpy_encoder)
                
            self.last_save_path = filename
            self.just_saved_timer = 20 # Show visual feedback for 20 frames
            print(f"IHT LOGGER >>> SAVED JSON TO: {full_path}")
            return True
            
        except Exception as e:
            print(f"IHT LOGGER >>> SAVE ERROR: {e}")
            self.last_save_path = f"ERROR: {str(e)[:20]}"
            return False

    def step(self):
        self.step_count += 1
        if self.just_saved_timer > 0: self.just_saved_timer -= 1
        
        # 1. Gather Inputs
        trigger_sig = self.get_blended_input('trigger_export', 'sum') or 0.0
        
        entropy = self.get_blended_input('address_entropy', 'sum')
        pr = self.get_blended_input('participation_ratio', 'sum')
        overlap = self.get_blended_input('address_overlap', 'sum')
        health = self.get_blended_input('attractor_health', 'sum')
        ttc = self.get_blended_input('time_to_collapse', 'sum')
        loss = self.get_blended_input('projection_loss', 'sum')
        div = self.get_blended_input('division_rate', 'sum')
        stable_img = self.get_blended_input('stable_address', 'first')
        
        # 2. Coherence Logic
        self.current_coherence = self.compute_coherence(stable_img)
        
        # 3. Stable Fraction Logic
        stable_frac = 0.0
        if stable_img is not None:
            arr = stable_img.astype(np.float32)
            if arr.max() > 0: 
                stable_frac = np.sum(arr > (arr.max()*0.1)) / arr.size
        
        # 4. Update Buffer
        ts = self.data_buffer['timeseries']
        ts['step'].append(self.step_count)
        ts['entropy'].append(float(entropy) if entropy is not None else 0.0)
        ts['participation_ratio'].append(float(pr) if pr is not None else 0.0)
        ts['overlap'].append(float(overlap) if overlap is not None else 0.0)
        ts['health'].append(float(health) if health is not None else 0.0)
        ts['ttc'].append(float(ttc) if ttc is not None else 0.0)
        ts['loss'].append(float(loss) if loss is not None else 0.0)
        ts['div_rate'].append(float(div) if div is not None else 0.0)
        ts['coherence'].append(float(self.current_coherence))
        ts['stable_fraction'].append(float(stable_frac))
        
        # Limit buffer size (FIFO)
        if len(ts['step']) > self.log_limit:
            for k in ts:
                ts[k].pop(0)
                
        # 5. Trigger Logic (Edge Detection)
        should_export = False
        
        # Trigger on rising edge of signal
        if trigger_sig > 0.5 and self.last_trigger_state <= 0.5:
            should_export = True
            
        # Trigger on Config Button
        if self.force_export_btn:
            should_export = True
            self.force_export_btn = False # Reset button
            
        if should_export:
            self.export_json()
            
        self.last_trigger_state = float(trigger_sig)

    def get_output(self, port_name):
        if port_name == 'export_done':
            return 1.0 if self.just_saved_timer > 0 else 0.0
        elif port_name == 'coherence':
            return float(self.current_coherence)
        elif port_name == 'step_count':
            return float(self.step_count)
        return None

    def get_display_image(self):
        # Create display
        h, w = 64, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background color logic
        if self.just_saved_timer > 0:
            img[:] = (0, 100, 0) # Green flash on save
        else:
            img[:] = (40, 40, 40) # Dark gray default
            
        # Status Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "DATA LOGGER", (5, 15), font, 0.4, (200,200,200), 1)
        
        # Metrics
        cv2.putText(img, f"Steps: {self.step_count}", (5, 30), font, 0.35, (180,180,180), 1)
        cv2.putText(img, f"Buffer: {len(self.data_buffer['timeseries']['step'])}", (5, 42), font, 0.35, (180,180,180), 1)
        
        # Trigger status
        if self.last_trigger_state > 0.5:
             cv2.putText(img, "TRIG: HIGH", (70, 30), font, 0.35, (0,255,0), 1)
        else:
             cv2.putText(img, "TRIG: LOW", (70, 30), font, 0.35, (100,100,100), 1)

        # Last file
        if self.last_save_path != "None":
            cv2.putText(img, f"Saved: {self.last_save_path[-12:]}", (5, 58), font, 0.3, (150,255,150), 1)
            
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Force Export Now", "force_export_btn", self.force_export_btn, "bool"),
            ("Log Buffer Size", "log_limit", self.log_limit, "int"),
            ("File Prefix", "file_prefix", self.file_prefix, "text"),
        ]