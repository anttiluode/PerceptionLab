"""
Token Resonance Node - The "Thought" Detector
=============================================
Calculates the interference/resonance between two token streams.

THEORY:
- A "Thought" occurs when the Executive (Frontal) and Memory (Temporal) 
  regions fire tokens at the same frequency with locked phases.
- This node measures that Phase Locking Value (PLV).

INPUTS:
- input_a: Spectrum (e.g., Frontal Tokens)
- input_b: Spectrum (e.g., Temporal Tokens)
- decay: Visual fade rate

OUTPUTS:
- display: Image (Visual of the lock)
- coherence: Signal (0.0 to 1.0 strength of connection)
- locked_tokens: Spectrum (The resulting "Merged" tokens)
"""

import numpy as np
import cv2

# --- COMPATIBILITY ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0

class TokenResonanceNode(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Token Resonance (Q x K)"
    NODE_COLOR = QtGui.QColor(255, 50, 100) # Hot Pink
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "input_a": "spectrum",  # Frontal (Key)
            "input_b": "spectrum",  # Temporal (Value)
            "decay": "float",
        }
        
        self.outputs = {
            "display": "image",
            "coherence": "signal",
            "locked_tokens": "spectrum"
        }
        
        self.sparks = [] # Visual effects list
        self.last_coherence = 0.0
        self._display = np.zeros((400, 400, 3), dtype=np.uint8)

    def _sanitize_input(self, data):
        """Forces input into (N, 3) numpy array, handling Strings/None safely"""
        # 1. Null check
        if data is None:
            return np.zeros((0, 3), dtype=np.float32)
            
        # 2. String check (The source of your error)
        if isinstance(data, str):
            return np.zeros((0, 3), dtype=np.float32)
            
        # 3. List conversion
        if isinstance(data, (list, tuple)):
            try:
                data = np.array(data)
            except:
                return np.zeros((0, 3), dtype=np.float32)

        # 4. Verify it is now an array
        if not isinstance(data, np.ndarray):
            return np.zeros((0, 3), dtype=np.float32)
            
        # 5. Fix Shape
        if data.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # If 1D array [k, a, p], make it 2D [[k, a, p]]
        if data.ndim == 1:
            if len(data) == 3:
                return data.reshape(1, 3)
            else:
                return np.zeros((0, 3), dtype=np.float32)
                
        # If columns missing, pad or cut
        if data.ndim == 2:
            rows, cols = data.shape
            if cols == 3:
                return data.astype(np.float32)
            elif cols > 3:
                return data[:, :3].astype(np.float32)
            else:
                return np.zeros((0, 3), dtype=np.float32)
                
        return np.zeros((0, 3), dtype=np.float32)

    def step(self):
        # 1. Get Inputs (Safe)
        raw_a = self.inputs.get("input_a", None)
        raw_b = self.inputs.get("input_b", None)
        decay_val = self.inputs.get("decay", 0.9)
        
        # Handle decay input safety
        decay = 0.9
        if isinstance(decay_val, (int, float)):
            decay = decay_val
        elif hasattr(decay_val, 'item'): 
            decay = decay_val.item()

        # 2. Sanitize Data (Robust)
        stream_a = self._sanitize_input(raw_a)
        stream_b = self._sanitize_input(raw_b)
        
        # 3. Compare Tokens (The Logic)
        matches = []
        total_resonance = 0.0
        
        # Only process if we have tokens in both streams
        if len(stream_a) > 0 and len(stream_b) > 0:
            for t_a in stream_a:
                key_a, amp_a, phase_a = t_a
                if amp_a < 0.1: continue
                
                for t_b in stream_b:
                    key_b, amp_b, phase_b = t_b
                    if amp_b < 0.1: continue
                    
                    # Frequency Match (Key structure: Band index is key % 5)
                    band_a = int(key_a) % 5
                    band_b = int(key_b) % 5
                    
                    if band_a == band_b:
                        # Phase Locking Value (PLV)
                        # 1.0 = perfect sync, 0.0 = anti-phase
                        phase_diff = abs(phase_a - phase_b)
                        coherence = (np.cos(phase_diff) + 1) / 2.0
                        
                        # Energy of the connection
                        energy = min(amp_a, amp_b) * coherence
                        
                        if energy > 0.2:
                            matches.append({
                                'band': band_a,
                                'energy': energy,
                                'phase': (phase_a + phase_b)/2
                            })
                            total_resonance += energy
                            
                            # Visual spark
                            if len(self.sparks) < 50: 
                                self.sparks.append({
                                    'x': np.random.randint(100, 300),
                                    'y': np.random.randint(100, 300),
                                    'life': 1.0,
                                    'color': (255, int(255*coherence), 100)
                                })

        # 4. Outputs
        self.outputs['coherence'] = float(total_resonance)
        
        # 5. Render
        self._render_reactor(matches, total_resonance, decay)

    def _render_reactor(self, matches, total_res, decay):
        img = self._display
        # Fade out
        if decay > 0:
            img = cv2.multiply(img, decay)
        else:
            img[:] = 0
        
        cx, cy = 200, 200
        
        # Draw Core (The Resonance Chamber)
        core_size = int(20 + min(total_res * 30, 100))
        # Outer glow
        cv2.circle(img, (cx, cy), core_size, (50, 20, 100), -1)
        # Inner hot core
        cv2.circle(img, (cx, cy), int(core_size*0.7), (150, 100, 255), -1)
        
        # Draw Sparks
        new_sparks = []
        for s in self.sparks:
            s['life'] -= 0.08
            if s['life'] > 0:
                # Physics: vortex
                dx = cx - s['x']
                dy = cy - s['y']
                
                # Orbit + Suck
                s['x'] += dx * 0.1 - dy * 0.1
                s['y'] += dy * 0.1 + dx * 0.1
                
                px, py = int(s['x']), int(s['y'])
                c = s['color']
                draw_c = tuple(int(ch * s['life']) for ch in c)
                
                if 0 <= px < 400 and 0 <= py < 400:
                    cv2.circle(img, (px, py), 2, draw_c, -1)
                
                new_sparks.append(s)
        self.sparks = new_sparks
        
        # Draw Text Info
        bands = ["DELTA", "THETA", "ALPHA", "BETA", "GAMMA"]
        y_txt = 380
        
        if len(matches) > 0:
            # Show strongest match
            matches.sort(key=lambda x: x['energy'], reverse=True)
            top = matches[0]
            b_name = bands[top['band']]
            e = top['energy']
            cv2.putText(img, f"LOCKED: {b_name} ({e:.2f})", (10, y_txt), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
             cv2.putText(img, "WAITING FOR RESONANCE...", (10, y_txt), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        self.outputs['display'] = img
        self._display = img

    def get_display_image(self): return self._display
    def get_output(self, name): return self.outputs.get(name)