"""
Task Switch Node (The Cognitive Load Generator)
===============================================
Forces the Pattern Memory Bank to switch targets at a set interval.
Watch the Skew Flux (Entropy) spike wildly during transitions,
representing the thermodynamic cost of changing the mind's attractor.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class TaskSwitchNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(180, 100, 50) # Alert Orange

    def __init__(self, interval=100):
        super().__init__()
        self.node_title = "Task Switcher"
        self.inputs = {}
        self.outputs = {'select': 'signal'}
        
        self.interval = int(interval)
        self.counter = 0
        self.current_task = 0
        self.display_img = np.zeros((64, 128, 3), dtype=np.uint8)

    def step(self):
        self.counter += 1
        if self.counter >= self.interval:
            self.counter = 0
            self.current_task = (self.current_task + 1) % 5 # Assuming 5 patterns

        img = np.zeros((64, 128, 3), dtype=np.uint8)
        # Draw progress bar to next switch
        progress = self.counter / self.interval
        cv2.rectangle(img, (4, 30), (int(4 + progress * 120), 40), (100, 150, 255), -1)
        cv2.rectangle(img, (4, 30), (124, 40), (200, 200, 200), 1)
        
        cv2.putText(img, f"Task Index: {self.current_task}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.display_img = img

    def get_output(self, port_name):
        if port_name == 'select': return float(self.current_task)
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 128, 64, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [("Switch Interval (frames)", "interval", self.interval, None)]
    
    def set_config_options(self, options):
        if "interval" in options: self.interval = int(options["interval"])