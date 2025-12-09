import numpy as np
import cv2
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode: 
        def __init__(self): self._outs = {}
        def get_blended_input(self, n, m): return 0.0

class ThetaSweepNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Theta Sweep (Navigation)"
    NODE_COLOR = QtGui.QColor(255, 165, 0) 

    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'x_in': 'signal', 
            'y_in': 'signal'
        }
        
        self.outputs = {
            'viz': 'image',
            'alternation_score': 'signal',
            'sweep_angle': 'signal'
        }
        
        if not hasattr(self, '_outs'): self._outs = {}
        
        self.cycle_len = 30
        self.trace_x = deque(maxlen=self.cycle_len)
        self.trace_y = deque(maxlen=self.cycle_len)
        self.history_angles = deque(maxlen=10)
        self._output_image = None

    def set_output(self, name, value):
        self._outs[name] = value

    def step(self):
        # --- SAFETY FIX: Handle None ---
        x = self.get_blended_input('x_in', 'sum')
        y = self.get_blended_input('y_in', 'sum')
        
        if x is None: x = 0.0
        if y is None: y = 0.0
        
        self.trace_x.append(x)
        self.trace_y.append(y)
        
        if len(self.trace_x) < self.cycle_len:
            return

        # Fit Line
        arr_x = np.array(self.trace_x)
        arr_y = np.array(self.trace_y)
        
        mx = np.mean(arr_x)
        my = np.mean(arr_y)
        dx = arr_x - mx
        dy = arr_y - my
        
        # Covariance Safety Check
        try:
            cov = np.cov(dx, dy)
            if np.all(np.isfinite(cov)):
                eigvals, eigvecs = np.linalg.eigh(cov)
                major_axis = eigvecs[:, 1]
                linearity = eigvals[1] / (eigvals[0] + 1e-9)
                current_angle = np.arctan2(major_axis[1], major_axis[0])
            else:
                linearity = 0; current_angle = 0; major_axis = [0,0]
        except:
            linearity = 0; current_angle = 0; major_axis = [0,0]
        
        alternation_score = 0.0
        if len(self.history_angles) > 0:
            prev_angle = self.history_angles[-1]
            diff = abs(current_angle - prev_angle)
            if diff > np.pi: diff = 2*np.pi - diff
            
            # Detect 180 flip or significant shift
            if diff > 0.5 and linearity > 2.0:
                alternation_score = 1.0
        
        if linearity > 2.0:
            self.history_angles.append(current_angle)

        self.render(arr_x, arr_y, major_axis, alternation_score)
        
        self.set_output('alternation_score', float(alternation_score))
        self.set_output('sweep_angle', float(current_angle))
        self.set_output('viz', self._output_image)

    def render(self, x_trace, y_trace, vector, score):
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)
        scale = size / 6.0
        center = size / 2.0
        
        pts = []
        for i in range(len(x_trace)):
            px = int(x_trace[i] * scale + center)
            py = int(y_trace[i] * scale + center)
            pts.append([px, py])
            
        if len(pts) > 1:
            c_val = int(score * 255)
            color = (c_val, 255, 255) 
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2)
            
        vx = int(vector[0] * 50)
        vy = int(vector[1] * 50)
        cv2.arrowedLine(img, (int(center), int(center)), (int(center+vx), int(center+vy)), (0, 0, 255), 2)
        
        self._output_image = img

    def get_output(self, name):
        # This connects the internal calculation to the outside world
        return self._outs.get(name)