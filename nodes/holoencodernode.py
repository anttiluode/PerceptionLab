"""
HoloEncoder Node (v4 - Fixed Outputs)
------------------
This node implements holographic/holographic-like compression,
converting a 2D image (spatial domain) into a 1D complex signal
(temporal/frequency domain). It can also decompress this signal
back into an image.

This is inspired by "Time-Domain Brain" concepts, where spatial
information might be encoded as a complex temporal pattern or
wave interference pattern for storage and broadcast.

FIX v4:
- `image_out` (blue port) now correctly outputs the reconstructed
  image when in 'Compress' mode, matching the internal display.
- `signal_out_real` (now orange port) is correctly typed as
  'spectrum' (a 1D float array) instead of 'signal' (a single float).
"""

import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: HoloEncoderNode requires scipy.fft")

if QtGui is None:
    print("CRITICAL: HoloEncoderNode could not import QtGui from host.")


class HoloEncoderNode(BaseNode):
    NODE_CATEGORY = "Cognitive"
    NODE_COLOR = QtGui.QColor(220, 100, 100)  # Holographic Red
    
    def __init__(self, mode='Compress', compression_ratio=0.1, reference_phase_seed=42):
        super().__init__()
        self.node_title = "HoloEncoder"
        
        # Define port types. 'complex_spectrum' is a custom type
        # that the BaseNode will treat as "not 'signal'",
        # which is correct for handling arrays.
        self.inputs = {
            'image_in': 'image',
            'signal_in': 'complex_spectrum', 
        }
        
        self.outputs = {
            'image_out': 'image',
            'signal_out_complex': 'complex_spectrum', # The full complex signal
            # --- FIX: Changed port type from 'signal' to 'spectrum' ---
            # 'signal' is for single floats, 'spectrum' is for 1D float arrays.
            'signal_out_real': 'spectrum'            # The real magnitude for other nodes
        }
        
        if not SCIPY_AVAILABLE or QtGui is None:
            self.node_title = "HoloEncoder (ERROR)"
            self._error = True
            return
        self._error = False
            
        # --- Configurable Parameters ---
        self.mode = str(mode) # 'Compress' or 'Decompress'
        self.compression_ratio = float(compression_ratio)
        self.reference_phase_seed = int(reference_phase_seed)

        # --- Internal State ---
        self._last_seed = self.reference_phase_seed
        self.reference_phase_map = None
        self.input_shape = (64, 64) # Default
        
        self._update_reference_map() # Initialize the map
        
        # Buffers for display
        self.display_in = np.zeros((64, 64, 3), dtype=np.uint8)
        self.display_out = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Output buffers for ports
        self.signal_out_complex_buffer = None
        self.signal_out_real_buffer = None
        self.image_out_buffer = None

    def _update_reference_map(self):
        """
        Creates the complex reference wave based on the seed.
        This is the "holographic plate" or "interference key".
        """
        if self.input_shape is None:
            return
        # Use a fixed seed for a stable reference wave
        rng = np.random.default_rng(self.reference_phase_seed)
        phase_angles = rng.uniform(0, 2 * np.pi, self.input_shape)
        self.reference_phase_map = np.exp(1j * phase_angles).astype(np.complex64)
        self._last_seed = self.reference_phase_seed

    def _check_config_change(self, new_shape=None):
        """Check if we need to regenerate the reference map."""
        shape_changed = False
        if new_shape is not None and new_shape != self.input_shape:
            self.input_shape = new_shape
            shape_changed = True
            
        if self.reference_phase_seed != self._last_seed or shape_changed:
            self._update_reference_map()

    def _normalize_image_in(self, img_in):
        """Converts any input image to a 2D float (0-1) array."""
        if img_in.ndim == 3:
            img_in = np.mean(img_in, axis=2) # Convert to grayscale
        
        if img_in.dtype == np.uint8:
            img_float = img_in.astype(np.float32) / 255.0
        else:
            # Assumes it's a float array (e.g., from CorticalReconstruction)
            img_float = img_in.astype(np.float32)
            max_val = img_float.max()
            if max_val > 1e-6:
                img_float = (img_float - img_float.min()) / (max_val - img_float.min() + 1e-9)
            
        return np.clip(img_float, 0, 1)

    def step(self):
        if self._error: return
        
        if self.mode == 'Compress':
            self._step_compress()
        else:
            self._step_decompress()

    def _step_compress(self):
        # --- Mode: Image -> Signal ---
        self.node_title = "HoloEncoder (Compress)"
        image_in = self.get_blended_input('image_in', 'mean')
        if image_in is None:
            # --- FIX: Clear outputs if no input ---
            self.image_out_buffer = None
            self.signal_out_complex_buffer = None
            self.signal_out_real_buffer = None
            self.display_in = np.zeros_like(self.display_in)
            self.display_out = np.zeros_like(self.display_out)
            return

        # 1. Prepare Input Image
        img_float = self._normalize_image_in(image_in)
        self._check_config_change(img_float.shape)
        
        # Store for display
        self.display_in = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # 2. Holographic Encoding (as per script)
        # Combine image amplitude with reference phase
        object_wave = img_float * self.reference_phase_map
        
        # Transform to frequency domain (the "hologram")
        hologram_freq = fftshift(fft2(object_wave))
        
        # 3. Compress
        # Keep only the central part of the spectrum
        h, w = hologram_freq.shape
        k = int(np.sqrt(h * w * self.compression_ratio))
        k = max(1, k) # Ensure at least 1
        
        start_h, end_h = (h - k) // 2, (h + k) // 2
        start_w, end_w = (w - k) // 2, (w + k) // 2
        
        compressed_spectrum = hologram_freq[start_h:end_h, start_w:end_w]
        
        # 4. Flatten to 1D Signal for output
        self.signal_out_complex_buffer = compressed_spectrum.flatten()
        # --- Create Real (Magnitude) version for other nodes ---
        self.signal_out_real_buffer = np.abs(self.signal_out_complex_buffer).astype(np.float32)
        
        # 5. Decompress for verification display AND output
        # --- FIX: Output the reconstructed image to the blue port ---
        self.image_out_buffer = self._decompress_signal(self.signal_out_complex_buffer, self.input_shape)
        self.display_out = cv2.cvtColor((self.image_out_buffer * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
    def _step_decompress(self):
        # --- Mode: Signal -> Image ---
        self.node_title = "HoloEncoder (Decompress)"
        
        signal_in_list = self.get_blended_input('signal_in', 'raw_list') # Get the list of inputs
        if not signal_in_list:
             # --- FIX: Clear outputs if no input ---
            self.image_out_buffer = None
            self.signal_out_complex_buffer = None
            self.signal_out_real_buffer = None
            self.display_in = np.zeros_like(self.display_in)
            self.display_out = np.zeros_like(self.display_out)
            return
        signal_in = signal_in_list[0] # Get the first (and likely only) signal

        # 1. Check/update reference map
        # We need a target shape, use the last known shape or default
        self._check_config_change() 
        
        # 2. Decompress
        # Convert input signal (which might be float) to complex
        signal_in_complex = np.array(signal_in).astype(np.complex64)
        decomp_img = self._decompress_signal(signal_in_complex, self.input_shape)
        
        self.image_out_buffer = decomp_img # This is the main output
        self.signal_out_complex_buffer = None
        self.signal_out_real_buffer = None
        
        # 3. Prepare for display
        self.display_out = cv2.cvtColor((decomp_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Show the input signal's spectrum as "input"
        self.display_in = self._visualize_spectrum(signal_in_complex, self.input_shape)

    def _decompress_signal(self, signal, target_shape):
        """Internal decompression logic, usable by both modes."""
        h, w = target_shape
        
        # 1. Reconstruct Spectrum
        k_h = k_w = int(np.sqrt(signal.size))
        if k_h * k_w != signal.size: # Handle non-square
             k_h = k_w = int(np.floor(np.sqrt(signal.size)))
             if k_h * k_w == 0: return np.zeros(target_shape, dtype=np.float32) # Not enough data
             signal = signal[:k_h*k_w]
        
        compressed_spectrum = signal.reshape((k_h, k_w))
        
        full_spectrum = np.zeros(target_shape, dtype=np.complex64)
        start_h, end_h = (h - k_h) // 2, (h + k_h) // 2
        start_w, end_w = (w - k_w) // 2, (w + k_w) // 2
        
        # Handle cases where k is odd/even
        h_slice = slice(start_h, start_h + k_h)
        w_slice = slice(start_w, start_w + k_w)

        full_spectrum[h_slice, w_slice] = compressed_spectrum
        
        # 2. Inverse FFT
        reconstructed_wave = ifft2(ifftshift(full_spectrum))
        
        # 3. Decode with reference phase
        # This is the key: multiply by the conjugate of the reference
        reconstructed_image_complex = reconstructed_wave * np.conj(self.reference_phase_map)
        
        # 4. Take absolute value (amplitude)
        reconstructed_image = np.abs(reconstructed_image_complex)
        
        # Normalize for output
        max_val = reconstructed_image.max()
        if max_val > 1e-6:
            reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (max_val - reconstructed_image.min())
            
        return np.clip(reconstructed_image, 0, 1).astype(np.float32)

    def _visualize_spectrum(self, signal, target_shape):
        """Helper for creating a displayable spectrum for Decompress mode."""
        h, w = target_shape
        k_h = k_w = int(np.sqrt(signal.size))
        if k_h * k_w != signal.size:
             k_h = k_w = int(np.floor(np.sqrt(signal.size)))
             if k_h*k_w == 0: return np.zeros((64,64,3), dtype=np.uint8)
             signal = signal[:k_h*k_w]
             
        spectrum = signal.reshape((k_h, k_w))
        
        full_spectrum_vis = np.zeros(target_shape, dtype=np.float32)
        start_h, end_h = (h - k_h) // 2, (h + k_h) // 2
        start_w, end_w = (w - k_w) // 2, (w + k_w) // 2
        
        # Handle cases where k is odd/even
        h_slice = slice(start_h, start_h + k_h)
        w_slice = slice(start_w, start_w + k_w)

        # Log magnitude for visualization
        log_mag = np.log1p(np.abs(spectrum))
        log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-9)
        
        full_spectrum_vis[h_slice, w_slice] = log_mag_norm
        
        img_u8 = (full_spectrum_vis * 255).astype(np.uint8)
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    def get_output(self, port_name):
        if self._error: return None
        if port_name == 'image_out':
            return self.image_out_buffer
        elif port_name == 'signal_out_complex':
            return self.signal_out_complex_buffer
        elif port_name == 'signal_out_real':
            return self.signal_out_real_buffer
        return None

    def get_display_image(self):
        if self._error: return None
        
        display_h = 128
        display_w = 256
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # --- Left side: "Input" ---
        in_resized = cv2.resize(self.display_in, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        display[:, :display_h] = in_resized
        
        # --- Right side: "Output" ---
        out_resized = cv2.resize(self.display_out, (display_h, display_h), interpolation=cv2.INTER_NEAREST)
        display[:, display_w-display_h:] = out_resized
        
        # Add dividing line
        display[:, display_h-1:display_h+1] = [255, 255, 255]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if self.mode == 'Compress':
            in_label = 'IN (Image)'
            out_label = 'OUT (Reconstructed)'
            info_text = f"COMPRESSING (Ratio: {self.compression_ratio:.2f})"
        else:
            in_label = 'IN (Spectrum)'
            out_label = 'OUT (Image)'
            info_text = "DECOMPRESSING"

        cv2.putText(display, in_label, (10, 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, out_label, (display_h + 10, 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, info_text, (10, display_h - 10), font, 0.4, (220, 100, 100), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Mode", "mode", self.mode, [
                ("Compress (Image->Signal)", "Compress"),
                ("Decompress (Signal->Image)", "Decompress")
            ]),
            ("Compression Ratio", "compression_ratio", self.compression_ratio, None),
            ("Reference Phase Seed", "reference_phase_seed", self.reference_phase_seed, None),
        ]

    # This is a special function to tell the host app how to handle array inputs
    def get_blended_input(self, port_name, blend_mode='sum'):
        # --- FIX: This method must be copied from the host BaseNode ---
        # --- so the node can correctly parse its own custom input types ---
        
        values = self.input_data.get(port_name, [])
        if not values:
            return None
            
        if blend_mode == 'raw_list':
            return values # Return the whole list of inputs

        # Check the type of the first item to decide the blend strategy
        first_val = values[0]
        
        if isinstance(first_val, (int, float)):
            # Handle simple signals (sum, mean, or first)
            if blend_mode == 'sum':
                return np.sum(values)
            elif blend_mode == 'mean':
                return np.mean(values)
            return values[0] # Default to 'first'

        elif isinstance(first_val, np.ndarray):
            # Handle array inputs (images, spectrums)
            
            # Check if it's complex
            if np.iscomplexobj(first_val):
                if blend_mode == 'mean':
                    # Safely average complex arrays
                    return np.mean([v for v in values if v is not None and v.size > 0], axis=0)
                return values[0] # Default to 'first'
            else:
                # Safely average real float/int arrays
                if blend_mode == 'mean':
                    return np.mean([v.astype(float) for v in values if v is not None and v.size > 0], axis=0)
                return values[0] # Default to 'first'
                
        # Default fallback for other types
        return values[0]