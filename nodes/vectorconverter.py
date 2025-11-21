"""
VectorConverterNode - BULLETPROOF dimension conversion
========================================================
Completely new name to avoid any conflicts.
Handles EVERYTHING: scalars, arrays, None, broken data.
"""

import numpy as np

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class VectorConverterNode(BaseNode):
    """
    Universal vector converter - handles any input type.
    """
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(180, 100, 220)
    
    def __init__(self, target_dim=16):
        super().__init__()
        self.node_title = "Vector Converter"
        
        self.inputs = {
            'input_data': 'spectrum'  # Actually accepts anything
        }
        
        self.outputs = {
            'vector_out': 'spectrum'
        }
        
        self.target_dim = int(target_dim)
        self.output_vector = np.zeros(self.target_dim, dtype=np.float32)
    
    def step(self):
        """Ultra-robust processing"""
        try:
            # Get input
            data = self.get_blended_input('input_data', 'first')
            
            # Handle None
            if data is None:
                self.output_vector = np.zeros(self.target_dim, dtype=np.float32)
                return
            
            # Handle scalar (float or int)
            if isinstance(data, (int, float, np.integer, np.floating)):
                # Broadcast scalar to all dimensions
                self.output_vector = np.full(self.target_dim, float(data), dtype=np.float32)
                return
            
            # Handle numpy array
            if isinstance(data, np.ndarray):
                # Flatten if multidimensional
                if data.ndim > 1:
                    data = data.flatten()
                
                # Convert to 1D array
                data = data.astype(np.float32)
                input_dim = len(data)
                
                # Resize to target dimension
                if input_dim == self.target_dim:
                    self.output_vector = data.copy()
                elif input_dim > self.target_dim:
                    # Truncate
                    self.output_vector = data[:self.target_dim]
                else:
                    # Pad with zeros
                    self.output_vector = np.zeros(self.target_dim, dtype=np.float32)
                    self.output_vector[:input_dim] = data
                return
            
            # Handle list
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
                input_dim = len(data)
                
                if input_dim == self.target_dim:
                    self.output_vector = data
                elif input_dim > self.target_dim:
                    self.output_vector = data[:self.target_dim]
                else:
                    self.output_vector = np.zeros(self.target_dim, dtype=np.float32)
                    self.output_vector[:input_dim] = data
                return
            
            # Unknown type - fill with zeros
            self.output_vector = np.zeros(self.target_dim, dtype=np.float32)
            
        except Exception as e:
            # Ultimate fallback
            print(f"VectorConverter: Unexpected error: {e}, filling with zeros")
            self.output_vector = np.zeros(self.target_dim, dtype=np.float32)
    
    def get_output(self, port_name):
        if port_name == 'vector_out':
            return self.output_vector
        return None