"""
Math Nodes - An expanded library of nodes for signal math, logic, and boolean operations
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
from PIL import Image, ImageDraw, ImageFont
import math

# --- !! CRITICAL IMPORT BLOCK !! ---
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
# -----------------------------------

class SignalMathNode(BaseNode):
    """Performs a mathematical operation on two input signals."""
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, operation='add'):
        super().__init__()
        self.node_title = "Signal Math"
        self.inputs = {'A': 'signal', 'B': 'signal'}
        self.outputs = {'result': 'signal'}
        
        self.operation = operation
        self.result = 0.0
        self.last_a = 0.0
        self.last_b = 0.0

    def step(self):
        # Use last known value if an input is disconnected
        a = self.get_blended_input('A', 'sum')
        b = self.get_blended_input('B', 'sum')
        
        if a is None: a = self.last_a
        else: self.last_a = a
        
        if b is None: b = self.last_b
        else: self.last_b = b
        
        if self.operation == 'add':
            self.result = a + b
        elif self.operation == 'subtract':
            self.result = a - b
        elif self.operation == 'multiply':
            self.result = a * b
        elif self.operation == 'divide':
            if abs(b) < 1e-6:
                self.result = 0.0
            else:
                self.result = a / b
        elif self.operation == 'pow':
            try:
                # Use numpy for safer power calculation
                self.result = np.nan_to_num(math.pow(a, b))
            except (ValueError, OverflowError):
                self.result = 0.0 # Handle complex results or overflow
        elif self.operation == 'min':
            self.result = min(a, b)
        elif self.operation == 'max':
            self.result = max(a, b)
        elif self.operation == 'avg':
            self.result = (a + b) / 2.0
        
    def get_output(self, port_name):
        if port_name == 'result':
            return self.result
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        op_symbol = {
            'add': '+', 'subtract': '-', 'multiply': '×', 'divide': '÷',
            'pow': '^', 'min': 'min', 'max': 'max', 'avg': 'avg'
        }.get(self.operation, '?')
        
        text = f"A {op_symbol} B\n= {self.result:.2f}"
        
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None 
            
        draw.text((5, 20), text, fill=255, font=font)
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Operation", "operation", self.operation, [
                ("Add (A + B)", "add"),
                ("Subtract (A - B)", "subtract"),
                ("Multiply (A × B)", "multiply"),
                ("Divide (A ÷ B)", "divide"),
                ("Power (A ^ B)", "pow"),
                ("Min(A, B)", "min"),
                ("Max(A, B)", "max"),
                ("Average", "avg")
            ])
        ]

class SignalLogicNode(BaseNode):
    """
    Outputs one of two signals based on a test condition.
    (If Test > Threshold, output if_true, else output if_false)
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, threshold=0.5, condition='>'):
        super().__init__()
        self.node_title = "Signal Logic (If/Else)"
        self.inputs = {'test': 'signal', 'if_true': 'signal', 'if_false': 'signal'}
        self.outputs = {'result': 'signal'}
        
        self.threshold = float(threshold)
        self.condition = condition
        self.result = 0.0
        self.last_true = 0.0
        self.last_false = 0.0
        self.condition_met = False

    def step(self):
        test_val = self.get_blended_input('test', 'sum') or 0.0
        if_true_val = self.get_blended_input('if_true', 'sum')
        if_false_val = self.get_blended_input('if_false', 'sum')
        
        if if_true_val is not None: self.last_true = if_true_val
        if if_false_val is not None: self.last_false = if_false_val
        
        self.condition_met = False
        if self.condition == '>':
            self.condition_met = test_val > self.threshold
        elif self.condition == '<':
            self.condition_met = test_val < self.threshold
        elif self.condition == '==':
            self.condition_met = abs(test_val - self.threshold) < 1e-6
        elif self.condition == '>=':
            self.condition_met = test_val >= self.threshold
        elif self.condition == '<=':
            self.condition_met = test_val <= self.threshold
        elif self.condition == '!=':
            self.condition_met = abs(test_val - self.threshold) > 1e-6
            
        self.result = self.last_true if self.condition_met else self.last_false

    def get_output(self, port_name):
        if port_name == 'result':
            return self.result
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        # Use RGB for color
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.condition_met:
            img[10:h-10, 10:w-10] = (60, 220, 60) # Green
            text = "TRUE"
        else:
            img[10:h-10, 10:w-10] = (220, 60, 60) # Red
            text = "FALSE"
            
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None
            
        condition_text = f"Test {self.condition} {self.threshold}"
        draw.text((5, 2), condition_text, fill=(255,255,255), font=font)
        draw.text((18, 28), text, fill=(255,255,255), font=font)
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Condition", "condition", self.condition, [
                ("Greater Than (>)", ">"),
                ("Less Than (<)", "<"),
                ("Equals (==)", "=="),
                ("Not Equal (!=)", "!="),
                ("Greater/Equal (>=)", ">="),
                ("Less/Equal (<=)", "<="),
            ]),
            ("Threshold", "threshold", self.threshold, None)
        ]

class SignalBooleanNode(BaseNode):
    """
    Performs boolean logic on two signals (A > thresh, B > thresh).
    Outputs 1.0 for True, 0.0 for False.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Transform Orange
    
    def __init__(self, operation='and', threshold=0.0):
        super().__init__()
        self.node_title = "Signal Boolean"
        self.inputs = {'A': 'signal', 'B': 'signal'}
        self.outputs = {'result': 'signal'}
        
        self.operation = operation
        self.threshold = float(threshold)
        self.result = 0.0
        self.last_a = 0.0
        self.last_b = 0.0

    def step(self):
        a = self.get_blended_input('A', 'sum')
        b = self.get_blended_input('B', 'sum')
        
        if a is None: a = self.last_a
        else: self.last_a = a
        
        if b is None: b = self.last_b
        else: self.last_b = b
        
        # Convert signals to boolean based on threshold
        a_true = (a > self.threshold)
        b_true = (b > self.threshold)
        
        res_bool = False
        if self.operation == 'and':
            res_bool = a_true and b_true
        elif self.operation == 'or':
            res_bool = a_true or b_true
        elif self.operation == 'xor':
            res_bool = a_true ^ b_true
        elif self.operation == 'not':
            res_bool = not a_true  # Only uses input A
        elif self.operation == 'nand':
            res_bool = not (a_true and b_true)
        elif self.operation == 'nor':
            res_bool = not (a_true or b_true)
        elif self.operation == 'xnor':
            res_bool = not (a_true ^ b_true)
            
        self.result = 1.0 if res_bool else 0.0

    def get_output(self, port_name):
        if port_name == 'result':
            return self.result
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        op_str = self.operation.upper()
        if op_str == 'NOT':
            text = f"NOT A\n= {self.result:.1f}"
        else:
            text = f"A {op_str} B\n= {self.result:.1f}"
        
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None 
            
        draw.text((5, 20), text, fill=255, font=font)
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Operation", "operation", self.operation, [
                ("AND", "and"),
                ("OR", "or"),
                ("XOR", "xor"),
                ("NOT (A only)", "not"),
                ("NAND", "nand"),
                ("NOR", "nor"),
                ("XNOR", "xnor"),
            ]),
            ("Boolean Threshold", "threshold", self.threshold, None)
        ]