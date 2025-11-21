"""
FIXED: DimensionAdapterNode
Handles scalar floats AND spectrum vectors
"""

import numpy as np

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class DimensionAdapterNode(BaseNode):
    """
    Automatically adapts vector dimensions between nodes.
    NOW HANDLES: scalars, arrays, any dimension
    """
    NODE_CATEGORY = "Utility"
    NODE_COLOR = QtGui.QColor(150, 100, 200)
    
    def __init__(self, target_dim=16, method='truncate_pad'):
        super().__init__()
        self.node_title = "Dimension Adapter"
        
        self.inputs = {
            'spectrum_in': 'spectrum',
            'target_dim_signal': 'signal'
        }
        
        self.outputs = {
            'spectrum_out': 'spectrum',
            'input_dim': 'signal',
            'output_dim': 'signal',
            'compression_ratio': 'signal'
        }
        
        self.target_dim = int(target_dim)
        self.method = method
        
        self.projection_matrix = None
        self.input_history = []
        self.learning_rate = 0.01
        
        self.output_spectrum = np.zeros(self.target_dim, dtype=np.float32)
        self.actual_input_dim = 0
        self.compression_ratio_val = 1.0
    
    def _convert_to_array(self, input_val):
        """Convert ANY input to numpy array"""
        if input_val is None:
            return None
        
        # If it's already an array, ensure it's 1D
        if isinstance(input_val, np.ndarray):
            if input_val.ndim > 1:
                input_val = input_val.flatten()
            return input_val.astype(np.float32)
        
        # If it's a scalar (float/int), convert to 1-element array
        if isinstance(input_val, (int, float)):
            return np.array([float(input_val)], dtype=np.float32)
        
        # If it's a list, convert
        if isinstance(input_val, list):
            return np.array(input_val, dtype=np.float32)
        
        # Unknown type, return None
        return None
    
    def adapt_truncate_pad(self, input_vec):
        """Simple truncation or padding"""
        input_dim = len(input_vec)
        
        if input_dim == self.target_dim:
            return input_vec.copy()
        elif input_dim > self.target_dim:
            return input_vec[:self.target_dim]
        else:
            output = np.zeros(self.target_dim, dtype=np.float32)
            output[:input_dim] = input_vec
            return output
    
    def adapt_interpolate(self, input_vec):
        """Smooth interpolation"""
        input_dim = len(input_vec)
        
        if input_dim == self.target_dim:
            return input_vec.copy()
        
        if input_dim == 1:
            # Special case: broadcast scalar to all dimensions
            return np.full(self.target_dim, input_vec[0], dtype=np.float32)
        
        x_in = np.linspace(0, 1, input_dim)
        x_out = np.linspace(0, 1, self.target_dim)
        
        output = np.interp(x_out, x_in, input_vec)
        return output.astype(np.float32)
    
    def adapt_project(self, input_vec):
        """PCA-like projection"""
        input_dim = len(input_vec)
        
        if input_dim == self.target_dim:
            return input_vec.copy()
        elif input_dim < self.target_dim:
            return self.adapt_truncate_pad(input_vec)
        
        importance = np.abs(input_vec)
        top_indices = np.argsort(importance)[-self.target_dim:]
        top_indices = np.sort(top_indices)
        
        return input_vec[top_indices]
    
    def adapt_learned(self, input_vec):
        """Learned projection matrix"""
        input_dim = len(input_vec)
        
        if self.projection_matrix is None or self.projection_matrix.shape != (self.target_dim, input_dim):
            self.projection_matrix = np.zeros((self.target_dim, input_dim), dtype=np.float32)
            for i in range(min(self.target_dim, input_dim)):
                self.projection_matrix[i, i] = 1.0
        
        output = self.projection_matrix.dot(input_vec)
        
        if len(self.input_history) > 10:
            input_variance = np.var(input_vec)
            output_variance = np.var(output)
            
            if output_variance > 1e-9:
                scale = np.sqrt(input_variance / output_variance)
                self.projection_matrix *= (1.0 - self.learning_rate) + self.learning_rate * scale
        
        self.input_history.append(input_vec.copy())
        if len(self.input_history) > 100:
            self.input_history.pop(0)
        
        return output.astype(np.float32)
    
    def step(self):
        spectrum = self.get_blended_input('spectrum_in', 'first')
        
        if spectrum is None:
            self.output_spectrum = np.zeros(self.target_dim, dtype=np.float32)
            self.actual_input_dim = 0
            self.compression_ratio_val = 1.0
            return
        
        # CRITICAL FIX: Convert any input type to array
        spectrum = self._convert_to_array(spectrum)
        
        if spectrum is None:
            self.output_spectrum = np.zeros(self.target_dim, dtype=np.float32)
            self.actual_input_dim = 0
            self.compression_ratio_val = 1.0
            return
        
        # Get dynamic target dim if provided
        target_dim_sig = self.get_blended_input('target_dim_signal', 'sum')
        if target_dim_sig is not None:
            self.target_dim = max(1, int(target_dim_sig))
        
        self.actual_input_dim = len(spectrum)
        
        # Choose adaptation method
        try:
            if self.method == 'truncate_pad':
                self.output_spectrum = self.adapt_truncate_pad(spectrum)
            elif self.method == 'interpolate':
                self.output_spectrum = self.adapt_interpolate(spectrum)
            elif self.method == 'project':
                self.output_spectrum = self.adapt_project(spectrum)
            elif self.method == 'learned':
                self.output_spectrum = self.adapt_learned(spectrum)
            else:
                self.output_spectrum = self.adapt_truncate_pad(spectrum)
        except Exception as e:
            print(f"DimensionAdapter: Adaptation error: {e}")
            # Fallback to simple broadcast
            if self.actual_input_dim == 1:
                self.output_spectrum = np.full(self.target_dim, spectrum[0], dtype=np.float32)
            else:
                self.output_spectrum = self.adapt_truncate_pad(spectrum)
        
        # Calculate compression ratio
        if self.actual_input_dim > 0:
            self.compression_ratio_val = float(self.target_dim) / float(self.actual_input_dim)
        else:
            self.compression_ratio_val = 1.0
    
    def get_output(self, port_name):
        if port_name == 'spectrum_out':
            return self.output_spectrum
        elif port_name == 'input_dim':
            return float(self.actual_input_dim)
        elif port_name == 'output_dim':
            return float(self.target_dim)
        elif port_name == 'compression_ratio':
            return self.compression_ratio_val
        return None