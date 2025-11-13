import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np

class SignalMonitorNode(BaseNode):
    """
    Visualizes an incoming signal as a simple bar graph.
    Uniquely named to avoid collisions.
    """
    NODE_CATEGORY = "Display"
    NODE_COLOR = QtGui.QColor(100, 100, 100) # Gray

    def __init__(self):
        super().__init__()
        self.node_title = "Signal Monitor"
        
        # --- Inputs and Outputs ---
        self.inputs = {'signal_in': 'signal'}
        self.outputs = {}
        
        # --- Internal State ---
        self.signal_value = 0.0
        self.display_buffer = np.zeros((96, 96, 3), dtype=np.uint8)

    def step(self):
        # Get the blended (summed) signal
        signal_in = self.get_blended_input('signal_in', 'sum')
        
        if signal_in is None:
            self.signal_value = 0.0
        elif isinstance(signal_in, (int, float)):
            self.signal_value = float(signal_in)
        else:
            self.signal_value = 0.0 # Handle unexpected input

        # Update the display buffer
        self._update_display()

    def _update_display(self):
        """Internal helper to draw the bar graph."""
        
        # Start with a black background
        self.display_buffer.fill(0)
        
        # Normalize the signal value for display
        val = np.clip(self.signal_value, 0.0, 10.0) # Clamp at 10
        
        # Calculate bar width (0-96 pixels)
        bar_width = int(np.clip(val, 0.0, 1.0) * 96)
        
        # Draw the bar
        if bar_width > 0:
            self.display_buffer[20:76, :bar_width] = (255, 255, 255)
            
        # Draw a red "overload" bar if signal > 1.0
        if val > 1.0:
            overload_width = int(np.clip(val - 1.0, 0.0, 9.0) * (96 / 9.0))
            overload_start = 96 - overload_width
            self.display_buffer[20:76, overload_start:] = (255, 0, 0)
            
    def get_output(self, port_name):
        return None

    def get_display_image(self):
        return self.display_buffer