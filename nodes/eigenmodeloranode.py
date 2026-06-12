"""
Eigenmode LoRA Node (Thermodynamic Bias)
========================================
Captures a specific rotational state (Skew Matrix) and injects it back 
into the live feed. This acts as a physical 'Style Transfer', forcing 
the downstream geometry to process reality through a hallucinated rotation.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class EigenmodeLoRANode(BaseNode):
    NODE_CATEGORY = "Modifier"
    NODE_COLOR = QtGui.QColor(200, 50, 150) # Magenta / LoRA Pink

    def __init__(self):
        super().__init__()
        self.node_title = "Eigenmode LoRA"
        self.inputs = {
            'live_matrix': 'image', 
            'bias_strength': 'signal' # Can be driven by a constant or Coupler
        }
        self.outputs = {
            'biased_matrix': 'image'
        }
        
        self.recorded_matrix = None
        self.record_trigger = 0.0
        self.manual_strength = 0.5
        
        self.display_img = np.zeros((128, 256, 3), dtype=np.uint8)

    def step(self):
        live_img = self.get_blended_input('live_matrix', 'first')
        strength_in = self.get_blended_input('bias_strength', 'sum')
        
        # Use signal if connected, otherwise use manual config slider
        strength = float(strength_in) if strength_in is not None else self.manual_strength
        strength = np.clip(strength, 0.0, 1.0)

        if live_img is None:
            return

        # 1. Capture Logic: If triggered, save a copy of the current physics
        if self.record_trigger > 0.5:
            self.recorded_matrix = live_img.copy()
            self.record_trigger = 0.0 # Reset after capturing

        # 2. Injection Logic: Blend the live reality with the recorded hallucination
        if self.recorded_matrix is not None:
            out_matrix = live_img * (1.0 - strength) + self.recorded_matrix * strength
        else:
            out_matrix = live_img

        # 3. Visualization
        img = np.zeros((128, 256, 3), dtype=np.uint8)
        
        # Draw Live vs Recorded state for the UI
        live_disp = (np.clip(live_img[:,:,0] * 15.0 + 0.5, 0, 1) * 255).astype(np.uint8)
        live_color = cv2.applyColorMap(live_disp, cv2.COLORMAP_JET)
        img[14:114, 10:110] = cv2.resize(live_color, (100, 100), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, "Live Flux", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        if self.recorded_matrix is not None:
            rec_disp = (np.clip(self.recorded_matrix[:,:,0] * 15.0 + 0.5, 0, 1) * 255).astype(np.uint8)
            rec_color = cv2.applyColorMap(rec_disp, cv2.COLORMAP_JET)
            img[14:114, 140:240] = cv2.resize(rec_color, (100, 100), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, f"LoRA (w:{strength:.2f})", (140, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
        else:
            cv2.putText(img, "No LoRA Recorded", (140, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        self.display_img = img

        # Output the biased physics
        self._out_matrix = out_matrix

    def get_output(self, port_name):
        if port_name == 'biased_matrix': 
            return self._out_matrix
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Record Now (Set to 1)", "record_trigger", 0.0, "float"),
            ("Manual Bias Strength", "manual_strength", self.manual_strength, "float")
        ]
    
    def set_config_options(self, options):
        if "record_trigger" in options: 
            self.record_trigger = float(options["record_trigger"])
        if "manual_strength" in options: 
            self.manual_strength = float(options["manual_strength"])