import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import cv2
import numpy as np

class ButtonNode(BaseNode):
    """
    A simple clickable button node.
    """
    NODE_CATEGORY = "Input"
    NODE_COLOR = QtGui.QColor(200, 200, 100)

    def __init__(self, label="Button", mode="Toggle"):
        super().__init__()
        self.node_title = "Button"
        self.label = str(label)
        self.mode = str(mode)  # "Toggle" or "Hold"
        
        self.inputs = {}
        self.outputs = {'signal_out': 'signal'}
        
        self.is_pressed = False
        self.value = 0.0

    def get_output(self, port_name):
        if port_name == 'signal_out':
            return self.value
        return None

    def step(self):
        if self.mode == "Hold":
            self.value = 1.0 if self.is_pressed else 0.0
        # For "Toggle", value is changed in mousePressEvent

    def mousePressEvent(self, event):
        self.is_pressed = True
        if self.mode == "Toggle":
            self.value = 1.0 - self.value # Flip
        self.update_display()

    def mouseReleaseEvent(self, event):
        self.is_pressed = False
        self.update_display()

    def get_display_image(self):
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.value > 0.5:
            # Active state
            cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 255, 0), -1)
            cv2.putText(img, self.label, (w//2 - 4*len(self.label), h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # Inactive state
            cv2.rectangle(img, (5, 5), (w-6, h-6), (100, 100, 100), -1)
            cv2.putText(img, self.label, (w//2 - 4*len(self.label), h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Label", "label", self.label, None),
            ("Mode", "mode", self.mode, ["Toggle", "Hold"])
        ]