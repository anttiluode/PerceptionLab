"""
TopologyAnalyzerNode

Analyzes the output of an InstantonFieldNode.
It finds the "stable, localized information structures" (instantons)
and calculates metrics like count, total accumulated "action",
and "long-range order."
"""

import numpy as np
import cv2

# --- Magic import block ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
# --------------------------

class TopologyAnalyzerNode(BaseNode):
    """
    Finds instantons and calculates their properties.
    """
    NODE_CATEGORY = "Analyzer"
    NODE_COLOR = QtGui.QColor(220, 200, 100) # Gold

    def __init__(self, size=128):
        super().__init__()
        self.node_title = "Topology Analyzer"
        
        self.inputs = {
            'field_in': 'image',     # The raw 'field_out' from InstantonFieldNode
            'threshold': 'signal'    # 0-1, threshold to define an instanton
        }
        self.outputs = {
            'instanton_count': 'signal', # Number of instantons
            'total_action': 'signal',    # Total accumulated information
            'long_range_order': 'signal' # 0-1, how spread out instantons are
        }
        
        self.size = int(size)
        
        # Internal state
        self.instanton_count = 0.0
        self.total_action = 0.0
        self.long_range_order = 0.0

    def step(self):
        # --- 1. Get and Prepare Image ---
        field = self.get_blended_input('field_in', 'first')
        if field is None:
            return

        # Ensure field is 0-1 float
        if field.dtype != np.float32:
            field = field.astype(np.float32)
        if field.max() > 1.0:
            field /= 255.0
            
        # Resize and ensure grayscale
        field = cv2.resize(field, (self.size, self.size), 
                           interpolation=cv2.INTER_LINEAR)
        if field.ndim == 3:
            field_gray = cv2.cvtColor(field, cv2.COLOR_RGB2GRAY)
        else:
            field_gray = field
        
        # --- 2. Calculate Total "Action" ---
        # "Information weight accumulation"
        self.total_action = np.sum(field_gray) / self.size # Normalize by size

        # --- 3. Find Instantons ---
        threshold = self.get_blended_input('threshold', 'sum') or 0.5
        _ , binary = cv2.threshold(
            (field_gray * 255).astype(np.uint8), 
            int(threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Find contours (the instantons)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        self.instanton_count = len(contours)
        
        # --- 4. Calculate "Long-Range Order" ---
        if self.instanton_count > 1:
            centers = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    centers.append([cx, cy])
            
            if len(centers) > 1:
                # Calculate the std deviation of instanton positions
                # A high std dev means they are spread out (high long-range order)
                centers = np.array(centers)
                std_dev_x = np.std(centers[:, 0])
                std_dev_y = np.std(centers[:, 1])
                
                # Normalize by the max possible std dev (size / 2)
                self.long_range_order = (std_dev_x + std_dev_y) / self.size
                self.long_range_order = np.clip(self.long_range_order, 0, 1)
            else:
                self.long_range_order = 0.0
        else:
            self.long_range_order = 0.0

    def get_output(self, port_name):
        if port_name == 'instanton_count':
            return self.instanton_count
        elif port_name == 'total_action':
            return self.total_action
        elif port_name == 'long_range_order':
            return self.long_range_order
        return None

    def get_display_image(self):
        # Create a simple text display
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(img, f"Instantons: {self.instanton_count}", (10, 20), 
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Total Action: {self.total_action:.2f}", (10, 40), 
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Long-Range Order: {self.long_range_order:.2f}", (10, 60), 
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
        return img.astype(np.float32) / 255.0