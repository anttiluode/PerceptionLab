"""
Quantum State Tomography Node - Reconstructs the full state from measurements
Performs multiple measurements in different bases to characterize the state
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class StateTomographyNode(BaseNode):
    """
    Performs quantum state tomography by measuring in multiple bases.
    Builds up a statistical picture of the state.
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(220, 150, 100)
    
    def __init__(self, num_measurements=100):
        super().__init__()
        self.node_title = "State Tomography"
        
        self.inputs = {
            'state_in': 'spectrum',
            'trigger': 'signal',  # Start tomography
            'reset': 'signal'
        }
        self.outputs = {
            'density_matrix': 'spectrum',  # Reconstructed density matrix (flattened)
            'measurement_results': 'spectrum',  # Histogram of outcomes
            'completeness': 'signal',  # How complete the tomography is (0-1)
            'fidelity': 'signal'  # Estimated state fidelity
        }
        
        self.num_measurements = int(num_measurements)
        
        # Measurement bases (Pauli-like)
        self.bases = []
        self.measurements = []
        self.is_measuring = False
        self.measurement_count = 0
        
    def step(self):
        state = self.get_blended_input('state_in', 'first')
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        reset = self.get_blended_input('reset', 'sum') or 0.0
        
        if state is None:
            return
            
        dim = len(state)
        
        # Reset tomography
        if reset > 0.5:
            self.measurements = []
            self.measurement_count = 0
            self.is_measuring = False
            self._initialize_bases(dim)
            
        # Start tomography
        if trigger > 0.5 and not self.is_measuring:
            self.is_measuring = True
            self.measurements = []
            self.measurement_count = 0
            self._initialize_bases(dim)
            
        # Perform measurements
        if self.is_measuring and self.measurement_count < self.num_measurements:
            # Choose random basis
            basis_idx = np.random.randint(len(self.bases))
            basis = self.bases[basis_idx]
            
            # Project state onto basis
            projection = np.abs(np.dot(state, basis)) ** 2
            prob_sum = np.abs(state) ** 2
            prob_sum = prob_sum.sum()
            
            if prob_sum > 1e-9:
                # Measure
                outcome = projection / prob_sum
            else:
                outcome = 0.0
                
            self.measurements.append({
                'basis': basis_idx,
                'outcome': outcome,
                'state_snapshot': state.copy()
            })
            
            self.measurement_count += 1
            
            if self.measurement_count >= self.num_measurements:
                self.is_measuring = False
                self._reconstruct_density_matrix()
                
    def _initialize_bases(self, dim):
        """Create measurement bases (computational, hadamard, etc.)"""
        self.bases = []
        
        # Computational basis (standard basis vectors)
        for i in range(min(dim, 6)):  # Limit to 6 bases
            basis = np.zeros(dim)
            basis[i] = 1.0
            self.bases.append(basis)
            
        # Hadamard-like bases (equal superposition)
        if dim >= 2:
            basis = np.ones(dim) / np.sqrt(dim)
            self.bases.append(basis)
            
        # Phase-rotated bases
        if dim >= 4:
            basis = np.array([np.exp(1j * 2 * np.pi * i / dim) for i in range(dim)])
            self.bases.append(np.real(basis))
            
    def _reconstruct_density_matrix(self):
        """Reconstruct density matrix from measurements (simplified)"""
        if len(self.measurements) == 0:
            return
            
        # Extract dimension from first measurement
        dim = len(self.measurements[0]['state_snapshot'])
        
        # Average all measured states (simplified tomography)
        avg_state = np.mean([m['state_snapshot'] for m in self.measurements], axis=0)
        
        # Density matrix: ρ = |ψ⟩⟨ψ|
        self.density_matrix = np.outer(avg_state, np.conj(avg_state))
        
        # Compute fidelity (purity of density matrix)
        self.fidelity = np.real(np.trace(np.dot(self.density_matrix, self.density_matrix)))
        
    def get_output(self, port_name):
        if port_name == 'density_matrix':
            if hasattr(self, 'density_matrix'):
                return self.density_matrix.flatten().astype(np.complex64)
            return None
        elif port_name == 'measurement_results':
            if len(self.measurements) > 0:
                outcomes = np.array([m['outcome'] for m in self.measurements])
                return outcomes.astype(np.float32)
            return None
        elif port_name == 'completeness':
            return float(self.measurement_count) / float(self.num_measurements)
        elif port_name == 'fidelity':
            return float(self.fidelity) if hasattr(self, 'fidelity') else 0.0
        return None
        
    def get_display_image(self):
        """Visualize tomography progress"""
        w, h = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Progress bar
        progress = self.measurement_count / self.num_measurements
        progress_width = int(progress * w)
        cv2.rectangle(img, (0, 0), (progress_width, 30), (0, 255, 0), -1)
        
        cv2.putText(img, f"Measurements: {self.measurement_count}/{self.num_measurements}",
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0) if progress > 0.5 else (255,255,255), 1)
        
        # Measurement histogram
        if len(self.measurements) > 0:
            outcomes = [m['outcome'] for m in self.measurements[-50:]]  # Last 50
            
            hist, bins = np.histogram(outcomes, bins=20, range=(0, 1))
            hist_max = hist.max() if hist.max() > 0 else 1
            
            bar_width = w // len(hist)
            for i, count in enumerate(hist):
                x = i * bar_width
                bar_h = int((count / hist_max) * 150)
                cv2.rectangle(img, (x, h - bar_h), (x + bar_width - 2, h), (100, 150, 255), -1)
                
        # Status
        if self.is_measuring:
            status = "MEASURING..."
            color = (255, 255, 0)
        elif self.measurement_count >= self.num_measurements:
            status = "COMPLETE"
            color = (0, 255, 0)
        else:
            status = "READY"
            color = (150, 150, 150)
            
        cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Fidelity
        if hasattr(self, 'fidelity'):
            cv2.putText(img, f"Fidelity: {self.fidelity:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        
    def get_config_options(self):
        return [
            ("Num Measurements", "num_measurements", self.num_measurements, None)
        ]