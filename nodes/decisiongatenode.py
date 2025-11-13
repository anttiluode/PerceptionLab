import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

import numpy as np

class DecisionGateNode(BaseNode):
    """
    Acts as a "thin layer of logic" (Hinton).
    It compares input signals based on a user-defined rule
    and outputs a binary signal (0 or 1).
    """
    NODE_CATEGORY = "Logic"
    NODE_COLOR = QtGui.QColor(220, 220, 220) # Pure Logic White

    def __init__(self, rule='A > C', constant=0.5):
        super().__init__()
        self.node_title = "Decision Gate (Logic)"
        
        # --- Inputs and Outputs ---
        self.inputs = {
            'signal_in_a': 'signal',
            'signal_in_b': 'signal'
        }
        self.outputs = {'signal_out': 'signal'}
        
        # --- Configurable ---
        self.rules = ['A > C', 'A < C', 'A > B', 'A < B', 'A == B']
        self.rule = rule if rule in self.rules else self.rules[0]
        self.constant = float(constant) # The 'C' value
        
        # --- Internal State ---
        self.output_signal = 0.0
        self.display_img = np.zeros((96, 96, 3), dtype=np.float32)

    def get_config_options(self):
        """Returns options for the right-click config dialog."""
        options_list = [(rule, rule) for rule in self.rules]
        
        return [
            ("Rule (A, B, C)", "rule", self.rule, options_list),
            ("Constant (C)", "constant", self.constant, None),
        ]

    def set_config_options(self, options):
        """Receives a dictionary from the config dialog."""
        if "rule" in options:
            self.rule = options["rule"]
        if "constant" in options:
            self.constant = float(options["constant"])

    def step(self):
        # Get blended (summed) inputs
        a = self.get_blended_input('signal_in_a', 'sum')
        b = self.get_blended_input('signal_in_b', 'sum')
        c = self.constant
        
        # Default to 0.0 if no signal is connected
        if a is None: a = 0.0
        if b is None: b = 0.0

        # --- The Logic Layer ---
        result = False # Default to False (0.0)
        
        try:
            if self.rule == 'A > C':
                result = (a > c)
            elif self.rule == 'A < C':
                result = (a < c)
            elif self.rule == 'A > B':
                result = (a > b)
            elif self.rule == 'A < B':
                result = (a < b)
            elif self.rule == 'A == B':
                # Use a small epsilon for float comparison
                result = np.isclose(a, b)
                
        except Exception as e:
            print(f"DecisionGateNode Error: {e}")
            result = False

        # Set the final output signal
        self.output_signal = 1.0 if result else 0.0
        
        # Update display
        if self.output_signal > 0:
            self.display_img.fill(1.0) # White for "True"
        else:
            self.display_img.fill(0.0) # Black for "False"

    def get_output(self, port_name):
        if port_name == 'signal_out':
            return self.output_signal
        return None

    def get_display_image(self):
        """Returns a black or white square based on the output."""
        return self.display_img