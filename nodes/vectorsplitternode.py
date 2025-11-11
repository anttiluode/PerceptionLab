"""
Vector Splitter Node - ENHANCED
Splits latent vectors with scaling and visualization improvements
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class VectorSplitterNode(BaseNode):
    """Splits a spectrum into N signal outputs with optional scaling"""
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(150, 150, 150)
    
    def __init__(self, num_outputs=16, scale=1.0):  # Default 16 for VAE
        super().__init__()
        self.node_title = "Vector Splitter"
        
        self.num_outputs = int(num_outputs)
        self.scale = float(scale)
        
        self.inputs = {
            'spectrum_in': 'spectrum',
            'scale': 'signal'  # Optional dynamic scaling
        }
        
        self.outputs = {}
        for i in range(self.num_outputs):
            self.outputs[f'out_{i}'] = 'signal'
        
        self.output_values = np.zeros(self.num_outputs, dtype=np.float32)
    
    def step(self):
        vector = self.get_blended_input('spectrum_in', 'first')
        scale_signal = self.get_blended_input('scale', 'sum')
        
        if scale_signal is not None:
            scale = scale_signal
        else:
            scale = self.scale
        
        if vector is None:
            self.output_values.fill(0.0)
            return
        
        if vector.ndim > 1:
            vector = vector.flatten()
        
        for i in range(self.num_outputs):
            if i < len(vector):
                self.output_values[i] = float(vector[i]) * scale
            else:
                self.output_values[i] = 0.0
    
    def get_output(self, port_name):
        if port_name.startswith('out_'):
            try:
                index = int(port_name.split('_')[1])
                if 0 <= index < self.num_outputs:
                    return self.output_values[index]
            except (ValueError, IndexError):
                pass
        return None
    
    def get_display_image(self):
        w, h = 256, 128  # Bigger display
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.output_values.size == 0:
            return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
        bar_width = max(1, w // self.num_outputs)
        
        val_max = np.abs(self.output_values).max()
        if val_max < 1e-6: 
            val_max = 1.0
        
        for i, val in enumerate(self.output_values):
            x = i * bar_width
            norm_val = val / val_max
            bar_h = int(abs(norm_val) * (h/2 - 10))
            y_base = h // 2
            
            if val >= 0:
                color = (0, int(255 * abs(norm_val)), 0)
                cv2.rectangle(img, (x, y_base-bar_h), (x+bar_width-1, y_base), color, -1)
            else:
                color = (0, 0, int(255 * abs(norm_val)))
                cv2.rectangle(img, (x, y_base), (x+bar_width-1, y_base+bar_h), color, -1)
            
            # Label every 4th output
            if i % 4 == 0:
                cv2.putText(img, str(i), (x+2, h-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Baseline
        cv2.line(img, (0, h//2), (w, h//2), (100,100,100), 1)
        
        # Show scale
        cv2.putText(img, f"x{self.scale:.2f}", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Num Outputs", "num_outputs", self.num_outputs, None),
            ("Scale", "scale", self.scale, None)
        ]
