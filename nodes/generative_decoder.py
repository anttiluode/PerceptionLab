"""
Generative Decoder - Ma's Self-Consistency g(z) → x̂
====================================================
Implements the DECODER side of Yi Ma's framework.
(FIXED: Added robust input handling to prevent array-related crashes in live environment)

FROM THE PAPER:
"Self-consistency requires that the learned representation z can
regenerate the original data via a map g(z) → x̂ such that 
the encoder f cannot distinguish x from x̂."

CREATED: December 2025
THEORY: Yi Ma et al. "Parsimony and Self-Consistency" (2022)
"""

import numpy as np
import cv2
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): 
            self.inputs = {}
            self.outputs = {}
        def get_blended_input(self, name, mode): 
            return None

class GenerativeDecoder(BaseNode):
    """
    The g(z) → x̂ mapping from Ma's framework.
    Generates data from compressed representations to ensure self-consistency.
    """
    NODE_CATEGORY = "Ma Framework"
    NODE_TITLE = "Generative Decoder"
    NODE_COLOR = QtGui.QColor(150, 100, 50)  # Brown - expansion
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'compressed_z': 'spectrum',           # From RateReductionEncoder
            'subspace_assignments': 'spectrum',   # Which subspaces to use
            'original_tokens': 'spectrum',        # For error computation
            'theta_phase': 'signal',              # For phase-coherent generation
        }
        
        self.outputs = {
            'display': 'image',
            'reconstructed_tokens': 'spectrum',   # x̂ - the generated tokens
            'reconstruction_error': 'signal',     # ||x - x̂|| 
            'generated_field': 'image',           # Visual of generated pattern
            'consistency_score': 'signal',        # 1 - normalized error
            'surprise': 'signal',                 # High when prediction fails
            'complex_spectrum': 'complex_spectrum', # For consciousness nodes
            'dream_field': 'complex_spectrum',    # 2D complex field
        }
        
        # === DIMENSIONS ===
        self.latent_dim = 64
        self.n_subspaces = 5
        self.n_tokens = 20  # Max tokens to generate
        
        # === LEARNED GENERATORS ===
        # Each subspace has its own generator matrix
        np.random.seed(43)
        self.generators = []
        for j in range(self.n_subspaces):
            # Generator: z → token parameters
            # Maps latent to (token_id, amplitude, phase) for each potential token
            G = np.random.randn(self.n_tokens * 3, self.latent_dim) * 0.1
            self.generators.append(G)
        
        # === STATE ===
        self.current_reconstruction = np.zeros((self.n_tokens, 3))
        self.current_error = 0.0
        self.current_surprise = 0.0
        self.error_history = deque(maxlen=500)
        self.current_field = np.zeros((256, 256), dtype=np.complex64) # Initialize field
        
        # === LEARNING ===
        self.learning_rate = 0.01
        
        # === DISPLAY ===
        self._display = np.zeros((600, 900, 3), dtype=np.uint8)
        self._field_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    def _sanitize_input(self, data, expected_len):
        """Ensure input is valid numpy array of expected length"""
        if data is None:
            return np.zeros(expected_len, dtype=np.float32)
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if not hasattr(data, 'shape') or data.size == 0:
            return np.zeros(expected_len, dtype=np.float32)
        
        data = data.flatten().astype(np.float32)
        if len(data) < expected_len:
            return np.pad(data, (0, expected_len - len(data)))
        return data[:expected_len]
    
    def _generate_tokens(self, z, assignments, phase=0.0):
        """
        Generate tokens from latent representation z.
        Uses weighted combination of subspace generators.
        """
        output = np.zeros(self.n_tokens * 3)
        
        for j, (G, weight) in enumerate(zip(self.generators, assignments)):
            # Robustly check weight and array type before use
            if not isinstance(G, np.ndarray) or not isinstance(z, np.ndarray) or weight < 0.05:
                continue
            
            # Generate token parameters
            params = G @ z
            output += weight * params
        
        # Reshape to (n_tokens, 3)
        tokens = output.reshape(self.n_tokens, 3)
        
        # Post-process
        # Token IDs: softmax to get distribution, then argmax
        for i in range(self.n_tokens):
            tokens[i, 0] = i % 20  # Simple ID assignment
            tokens[i, 1] = np.abs(tokens[i, 1])  # Amplitude must be positive
            tokens[i, 2] = tokens[i, 2] + phase  # Add phase reference
        
        # Keep only tokens with significant amplitude
        mask = tokens[:, 1] > 0.1
        if np.sum(mask) > 0:
            active_tokens = tokens[mask]
        else:
            # Keep at least 3, even if small, to avoid empty array
            active_tokens = tokens[:3]
        
        return active_tokens
    
    def _compute_reconstruction_error(self, original, reconstructed):
        """
        Compute how well the decoder reconstructed the original.
        This is the "surprise" signal for learning.
        """
        if len(original) == 0:
            return 0.0, 0.0
        
        # Convert both to comparable format
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        
        # Pad to same length
        max_len = max(len(orig_flat), len(recon_flat))
        orig_padded = np.zeros(max_len)
        recon_padded = np.zeros(max_len)
        orig_padded[:len(orig_flat)] = orig_flat
        recon_padded[:len(recon_flat)] = recon_flat
        
        # L2 error
        error = np.linalg.norm(orig_padded - recon_padded)
        
        # Normalize by original magnitude
        orig_mag = np.linalg.norm(orig_padded) + 1e-9
        normalized_error = error / orig_mag
        
        # Surprise: high when error is much higher than expected
        return error, normalized_error
    
    def _generate_field(self, tokens, phase_ref=0.0):
        """Generate complex interference field from tokens"""
        size = 256
        field = np.zeros((size, size), dtype=np.complex128)
        
        if len(tokens) == 0:
            return field
        
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        for tok in tokens:
            # Added a check to ensure tok has enough elements
            if len(tok) < 3:
                continue 
            
            token_id = int(tok[0]) % 20
            amplitude = float(tok[1])
            phase = float(tok[2])
            
            # Wave parameters from token
            angle = token_id * (2 * np.pi / 20)
            k = 1 + (token_id % 5)
            
            kx = k * np.cos(angle + phase_ref)
            ky = k * np.sin(angle + phase_ref)
            
            # Complex wave (for proper interference)
            wave = amplitude * np.exp(1j * (kx * X + ky * Y + phase))
            field += wave
        
        # Normalize
        max_mag = np.abs(field).max()
        if max_mag > 1e-9:
            field = field / max_mag
        
        return field
    
    def _field_to_spectrum_1d(self, field):
        """Extract 1D complex spectrum from 2D field"""
        # Radial FFT profile
        fft_2d = np.fft.fftshift(np.fft.fft2(field))
        size = field.shape[0]
        center = size // 2
        n_bins = 64
        
        spectrum = np.zeros(n_bins, dtype=np.complex128)
        
        for r in range(n_bins):
            radius = r * size // (2 * n_bins)
            n_samples = max(8, int(2 * np.pi * radius))
            
            values = []
            for theta in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
                x = int(center + radius * np.cos(theta))
                y = int(center + radius * np.sin(theta))
                if 0 <= x < size and 0 <= y < size:
                    values.append(fft_2d[y, x])
            
            if values:
                spectrum[r] = np.mean(values)
        
        return spectrum
    
    def _update_generators(self, z, assignments, original, reconstructed, lr=0.01):
        """
        Online learning: update generators to reduce reconstruction error.
        This is the learning signal from the self-consistency loop.
        """
        if len(original) == 0:
            return
        
        # Compute gradient (simplified)
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        
        # Pad
        max_len = max(len(orig_flat), len(recon_flat))
        error = np.zeros(max_len)
        error[:len(orig_flat)] = orig_flat
        error[:len(recon_flat)] -= recon_flat[:min(len(recon_flat), max_len)]
        
        # Update each generator proportionally to its assignment weight
        for j, (G, weight) in enumerate(zip(self.generators, assignments)):
            if weight < 0.05:
                continue
            
            # Outer product gradient (simplified)
            error_truncated = error[:self.n_tokens * 3]
            if len(error_truncated) < self.n_tokens * 3:
                error_truncated = np.pad(error_truncated, (0, self.n_tokens * 3 - len(error_truncated)))
            
            # Added a check for array shapes before outer product
            if G.shape[1] != z.shape[0] or error_truncated.shape[0] != self.n_tokens * 3:
                 continue

            grad = np.outer(error_truncated, z)
            
            # Update with weight
            self.generators[j] += lr * weight * grad
    
    def step(self):
        # ===============================
        # 1. GET RAW INPUTS
        # ===============================
        # Use a sensible default for 'mean' blending mode if input is None
        raw_z = self.get_blended_input('compressed_z', 'mean') 
        raw_assignments = self.get_blended_input('subspace_assignments', 'mean')
        raw_original = self.get_blended_input('original_tokens', 'mean')
        phase_val = self.get_blended_input('theta_phase', 'sum')

        phase = float(phase_val) if phase_val is not None else 0.0

        # ===============================
        # 2. SANITIZE LATENT + ASSIGNMENTS
        # ===============================
        z = self._sanitize_input(raw_z, self.latent_dim)
        assignments = self._sanitize_input(raw_assignments, self.n_subspaces)

        if np.sum(assignments) > 0:
            assignments = assignments / np.sum(assignments)
        else:
            assignments = np.ones(self.n_subspaces, dtype=np.float32) / self.n_subspaces

        # ===============================
        # 3. GENERATE TOKENS (x̂ = g(z))
        # ===============================
        self.current_reconstruction = self._generate_tokens(z, assignments, phase)

        # ===============================
        # 4. GET ORIGINAL TOKENS (x)
        # ===============================
        if raw_original is not None and hasattr(raw_original, 'shape') and raw_original.ndim >= 2:
            original = raw_original
        else:
            # Fallback to an empty 2D array if input is not valid
            original = np.zeros((0, 3), dtype=np.float32)

        # ===============================
        # 5. RECONSTRUCTION ERROR / SURPRISE
        # ===============================
        error, normalized_error = self._compute_reconstruction_error(
            original, self.current_reconstruction
        )
        self.current_error = normalized_error

        self.error_history.append(normalized_error)
        if len(self.error_history) > 10:
            mean_error = np.mean(list(self.error_history)[-50:])
            self.current_surprise = max(0.0, normalized_error - mean_error)
        else:
            self.current_surprise = normalized_error

        consistency = 1.0 - min(1.0, normalized_error)

        # ===============================
        # 6. ONLINE LEARNING (OPTIONAL)
        # ===============================
        self._update_generators(
            z,
            assignments,
            original,
            self.current_reconstruction,
            self.learning_rate
        )

        # ===============================
        # 7. GENERATE COMPLEX FIELD
        # ===============================
        field = self._generate_field(self.current_reconstruction, phase)

        if field is None or not isinstance(field, np.ndarray):
            field = np.zeros((256, 256), dtype=np.complex64)
        elif field.dtype != np.complex64:
            field = field.astype(np.complex64)

        self._current_field = field

        # ===============================
        # 8. FIELD → DISPLAY IMAGE
        # ===============================
        magnitude = np.abs(field)
        phase_angle = np.angle(field)

        mag_max = magnitude.max()
        mag_norm = magnitude / (mag_max + 1e-9)

        hsv = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = ((phase_angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = (mag_norm * 255).astype(np.uint8)

        self._field_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # ===============================
        # 9. COMPLEX SPECTRUM (FOR DREAM)
        # ===============================
        spectrum_1d = self._field_to_spectrum_1d(field)

        if spectrum_1d is None or not isinstance(spectrum_1d, np.ndarray):
            # Ensure complex spectrum is always a valid complex array
            spectrum_1d = np.zeros(self.latent_dim, dtype=np.complex64)
        elif spectrum_1d.dtype != np.complex64:
            spectrum_1d = spectrum_1d.astype(np.complex64)

        # ===============================
        # 10. OUTPUTS (STRICT CONTRACT)
        # ===============================
        self.outputs['reconstructed_tokens'] = self.current_reconstruction.astype(np.float32)
        self.outputs['reconstruction_error'] = float(self.current_error)
        self.outputs['generated_field'] = self._field_img
        self.outputs['consistency_score'] = float(consistency)
        self.outputs['surprise'] = float(self.current_surprise)

        self.outputs['complex_spectrum'] = spectrum_1d
        self.outputs['dream_field'] = field

        # ===============================
        # 11. RENDER
        # ===============================
        self._render_display(original)

    
    def _render_display(self, original):
        """Full dashboard"""
        # Added try/except to prevent rendering from crashing the application
        try:
            img = self._display
            img[:] = (20, 20, 25)
            h, w = img.shape[:2]
            
            # === LEFT: Generated field ===
            field_resized = cv2.resize(self._field_img, (300, 300))
            img[30:330, 30:330] = field_resized
            cv2.putText(img, "GENERATED FIELD g(z)", (30, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # === CENTER: Token comparison ===
            self._render_token_comparison(img, 350, 30, 300, 300, original)
            
            # === RIGHT: Error history ===
            self._render_error_history(img, 670, 30, 200, 200)
            
            # === BOTTOM: Statistics ===
            cv2.putText(img, f"Reconstruction Error: {self.current_error:.4f}", (30, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            cv2.putText(img, f"Surprise: {self.current_surprise:.4f}", (30, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            
            consistency = self.outputs.get('consistency_score', 0)
            cons_color = (100, 255, 100) if consistency > 0.7 else (255, 255, 100) if consistency > 0.4 else (255, 100, 100)
            cv2.putText(img, f"Self-Consistency: {consistency:.2%}", (30, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cons_color, 1)
            
            self._display = img
            self.outputs['display'] = self._display # Ensure output is updated
        except Exception:
            pass # Fail silently on render error
    
    def _render_token_comparison(self, img, x0, y0, width, height, original):
        """Side-by-side comparison of original and reconstructed tokens"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        mid_x = x0 + width // 2
        
        # Original tokens (left)
        cv2.putText(img, "ORIGINAL x", (x0 + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        try: # Added try/except for robust rendering of original tokens
            if len(original) > 0 and hasattr(original, 'shape') and original.ndim >= 2:
                bar_h = min(15, (height - 50) // len(original))
                for i, tok in enumerate(original[:20]):
                    if len(tok) >= 2:
                        amp = float(tok[1]) if len(tok) > 1 else 0.5
                        bar_w = int(min(amp * 50, mid_x - x0 - 30))
                        y = y0 + 30 + i * bar_h
                        cv2.rectangle(img, (x0 + 10, y), (x0 + 10 + bar_w, y + bar_h - 2),
                                     (100, 200, 100), -1)
        except Exception:
            cv2.putText(img, "Original Data Error", (x0 + 10, y0 + height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            
        # Reconstructed tokens (right)
        cv2.putText(img, "RECONSTRUCTED x̂", (mid_x + 10, y0 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        recon = self.current_reconstruction
        if len(recon) > 0:
            bar_h = min(15, (height - 50) // len(recon))
            for i, tok in enumerate(recon[:20]):
                amp = float(tok[1])
                bar_w = int(min(amp * 50, width - mid_x + x0 - 30))
                y = y0 + 30 + i * bar_h
                cv2.rectangle(img, (mid_x + 10, y), (mid_x + 10 + bar_w, y + bar_h - 2),
                             (200, 100, 100), -1)
        
        # Divider
        cv2.line(img, (mid_x, y0 + 25), (mid_x, y0 + height - 10), (80, 80, 80), 1)
    
    def _render_error_history(self, img, x0, y0, width, height):
        """Plot error over time"""
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), (30, 30, 40), -1)
        
        if len(self.error_history) < 2:
            return
        
        errors = list(self.error_history)
        max_err = max(errors) + 0.1 if errors else 0.1
        
        for i in range(1, len(errors)):
            x1 = x0 + int((i-1) * width / len(errors))
            x2 = x0 + int(i * width / len(errors))
            y1 = y0 + height - 20 - int(errors[i-1] / max_err * (height - 40))
            y2 = y0 + height - 20 - int(errors[i] / max_err * (height - 40))
            cv2.line(img, (x1, y1), (x2, y2), (255, 100, 100), 1)
        
        cv2.putText(img, "ERROR HISTORY", (x0 + 10, y0 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def get_output(self, name):
        if name == 'display':
            return self._display
        elif name == 'generated_field':
            return self._field_img
        return self.outputs.get(name)
    
    def get_display_image(self):
        return self._display