"""
EDF Loader → Holographic FFT Node
==================================
Loads EEG and outputs it DIRECTLY as a 2D holographic frequency image.

The hypothesis: consciousness operates in frequency domain.
Raw EEG is "cortical space" - we transform it to see "perception space".

This node accumulates EEG samples into a 2D array (time × channels or time × frequency)
then performs 2D FFT to produce a complex holographic field.

Outputs:
- complex_spectrum: The raw complex FFT (for holographic processing)
- magnitude_view: Magnitude image for visualization
- phase_view: Phase image for visualization
- dominant_freq: Signal output of strongest frequency component

Settings:
- window_samples: How many time samples to accumulate (width of 2D array)
- freq_bins: How many frequency bins in spectrogram mode (height)
- output_mode: 'spectrogram' (time-freq) or 'multichannel' (time-channel)
"""

import numpy as np
from PyQt6 import QtGui
import os
import cv2
from collections import deque

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)

try:
    import mne
    from scipy import signal
    from scipy.fft import fft, fft2, fftshift
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: EDFLoaderHolographicNode requires 'mne' and 'scipy'")

# Brain regions
EEG_REGIONS = {
    "All": [],
    "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'],
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']
}


class EDFLoaderholographicNode(BaseNode):
    """
    EEG → Holographic Frequency Domain
    
    Treats EEG as "cortical space" and transforms to "perception space" via 2D FFT.
    If consciousness operates in frequency domain, this should reveal structure
    that raw EEG hides.
    """
    NODE_CATEGORY = "Holography"
    NODE_COLOR = QtGui.QColor(100, 60, 180)  # Deep purple for frequency domain
    
    def __init__(self, edf_file_path="", window_samples=128, freq_bins=64):
        super().__init__()
        self.node_title = "EEG → Holographic"
        
        self.inputs = {
            'external_signal': 'signal',  # Optional: modulate with external signal
        }
        
        self.outputs = {
            'complex_spectrum': 'complex_spectrum',  # The holographic field
            'magnitude_view': 'image',               # |FFT| for visualization
            'phase_view': 'image',                   # arg(FFT) for visualization
            'dominant_freq': 'signal',               # Strongest frequency
            'spectral_entropy': 'signal',            # Complexity measure
        }
        
        # Config
        self.edf_file_path = edf_file_path
        self.selected_region = "Occipital"
        self.window_samples = int(window_samples)  # Time axis of 2D array
        self.freq_bins = int(freq_bins)            # Frequency axis
        self.output_mode = "spectrogram"           # 'spectrogram' or 'multichannel'
        
        self._last_path = ""
        self._last_region = ""
        
        # EEG state
        self.raw = None
        self.fs = 256.0  # Higher sample rate for better frequency resolution
        self.current_sample = 0
        
        # Buffer for building 2D array
        self.time_buffer = deque(maxlen=self.window_samples)
        
        # Output state
        self.complex_field = None
        self.magnitude = None
        self.phase = None
        self.dominant_freq = 0.0
        self.spectral_entropy = 0.0
        
        # Display cache
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        if not MNE_AVAILABLE:
            self.node_title = "EEG Holo (MNE Required!)"
    
    def load_edf(self):
        """Load EDF file and prepare for streaming."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_file_path):
            self.raw = None
            self.node_title = "EEG Holo (No File)"
            return
        
        try:
            raw = mne.io.read_raw_edf(self.edf_file_path, preload=True, verbose=False)
            raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
            
            # Select region
            if self.selected_region != "All":
                region_channels = EEG_REGIONS.get(self.selected_region, [])
                available = [ch for ch in region_channels if ch in raw.ch_names]
                if available:
                    raw.pick_channels(available)
            
            # Resample
            raw.resample(self.fs, verbose=False)
            
            self.raw = raw
            self.current_sample = 0
            self._last_path = self.edf_file_path
            self._last_region = self.selected_region
            self.node_title = f"EEG→Holo ({self.selected_region})"
            
            # Reset buffer
            self.time_buffer.clear()
            
            print(f"Loaded EEG for holographic: {self.edf_file_path}")
            print(f"  Channels: {len(raw.ch_names)}, Samples: {raw.n_times}, Fs: {self.fs}")
            
        except Exception as e:
            self.raw = None
            self.node_title = "EEG Holo (Error)"
            print(f"Error loading EEG: {e}")
    
    def _compute_spectrogram_column(self, data_chunk):
        """
        Compute one column of spectrogram from a chunk of EEG data.
        Returns frequency amplitudes (complex) for this time slice.
        """
        # Average across channels if multi-channel
        if data_chunk.ndim > 1:
            data_chunk = np.mean(data_chunk, axis=0)
        
        # Windowed FFT
        windowed = data_chunk * np.hanning(len(data_chunk))
        spectrum = fft(windowed)
        
        # Take positive frequencies only, up to freq_bins
        n_freqs = min(len(spectrum) // 2, self.freq_bins)
        return spectrum[1:n_freqs+1]  # Skip DC
    
    def step(self):
        # Check for config changes
        if self.edf_file_path != self._last_path or self.selected_region != self._last_region:
            self.load_edf()
        
        if self.raw is None:
            return
        
        # Get chunk of EEG data (enough for one spectrogram column)
        chunk_size = int(self.fs * 0.05)  # 50ms chunks
        
        start = self.current_sample
        end = start + chunk_size
        
        if end >= self.raw.n_times:
            self.current_sample = 0
            return
        
        data, _ = self.raw[:, start:end]
        self.current_sample = end
        
        # Compute frequency content for this time slice
        freq_column = self._compute_spectrogram_column(data)
        
        # Pad/trim to exact freq_bins
        if len(freq_column) < self.freq_bins:
            freq_column = np.pad(freq_column, (0, self.freq_bins - len(freq_column)))
        else:
            freq_column = freq_column[:self.freq_bins]
        
        # Add to rolling buffer
        self.time_buffer.append(freq_column)
        
        # Once buffer is full, compute 2D FFT
        if len(self.time_buffer) >= self.window_samples:
            # Stack into 2D array: (freq_bins, window_samples)
            spectrogram = np.array(list(self.time_buffer)).T  # (freq, time)
            
            # This is already partially in frequency domain (freq axis)
            # Now do FFT on time axis to get full 2D frequency representation
            # This reveals "frequencies of frequency changes" - the meta-structure
            
            self.complex_field = fftshift(fft2(spectrogram))
            
            # Compute outputs
            self.magnitude = np.abs(self.complex_field).astype(np.float32)
            self.phase = np.angle(self.complex_field).astype(np.float32)
            
            # Normalize magnitude for display
            if self.magnitude.max() > 0:
                mag_norm = self.magnitude / self.magnitude.max()
            else:
                mag_norm = self.magnitude
            
            # Dominant frequency (brightest point excluding DC)
            center = np.array(self.magnitude.shape) // 2
            # Mask out center (DC)
            mag_masked = self.magnitude.copy()
            mag_masked[center[0]-2:center[0]+2, center[1]-2:center[1]+2] = 0
            peak_idx = np.unravel_index(np.argmax(mag_masked), mag_masked.shape)
            
            # Convert to frequency value (simplified)
            self.dominant_freq = np.sqrt((peak_idx[0] - center[0])**2 + 
                                         (peak_idx[1] - center[1])**2) / center[0]
            
            # Spectral entropy (complexity measure)
            mag_flat = self.magnitude.flatten()
            mag_flat = mag_flat / (mag_flat.sum() + 1e-10)
            mag_flat = mag_flat[mag_flat > 0]
            self.spectral_entropy = -np.sum(mag_flat * np.log(mag_flat + 1e-10))
            
            # Update display
            self._update_display(mag_norm)
    
    def _update_display(self, mag_norm):
        """Create visualization combining magnitude and phase."""
        h, w = 128, 128
        
        # Resize magnitude to display size
        mag_resized = cv2.resize(mag_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Log scale for better visibility
        mag_log = np.log1p(mag_resized * 10)
        mag_log = mag_log / (mag_log.max() + 1e-10)
        
        # Apply colormap
        mag_u8 = (mag_log * 255).astype(np.uint8)
        self.display_img = cv2.applyColorMap(mag_u8, cv2.COLORMAP_INFERNO)
        
        # Add info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.display_img, f"DomF: {self.dominant_freq:.2f}", 
                   (5, 15), font, 0.4, (255, 255, 255), 1)
        cv2.putText(self.display_img, f"Ent: {self.spectral_entropy:.1f}", 
                   (5, 30), font, 0.4, (255, 255, 255), 1)
    
    def get_output(self, port_name):
        if port_name == 'complex_spectrum':
            return self.complex_field
        elif port_name == 'magnitude_view':
            if self.magnitude is not None:
                # Normalize for image output
                mag = self.magnitude / (self.magnitude.max() + 1e-10)
                return mag.astype(np.float32)
            return None
        elif port_name == 'phase_view':
            if self.phase is not None:
                # Normalize phase from [-pi, pi] to [0, 1]
                phase_norm = (self.phase + np.pi) / (2 * np.pi)
                return phase_norm.astype(np.float32)
            return None
        elif port_name == 'dominant_freq':
            return self.dominant_freq
        elif port_name == 'spectral_entropy':
            return self.spectral_entropy
        return None
    
    def get_display_image(self):
        img = np.ascontiguousarray(self.display_img)
        h, w = img.shape[:2]
        return QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        region_options = [(name, name) for name in EEG_REGIONS.keys()]
        
        return [
            ("EDF File Path", "edf_file_path", self.edf_file_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_options),
            ("Window Samples", "window_samples", self.window_samples, None),
            ("Freq Bins", "freq_bins", self.freq_bins, None),
        ]


"""
WHAT THIS NODE DOES:

1. Loads EEG data
2. Computes running spectrogram (freq × time)
3. Applies 2D FFT to the spectrogram

This gives you "frequencies of frequency changes" - if there are rhythmic 
patterns in how the brain's frequency content oscillates, this will reveal them.

The output is a complex holographic field that can feed into:
- HolographicReconstructionNode (to see what structure emerges)
- ComplexToImageNode (for different visualizations)
- HebbianLearner (to learn stable patterns)
- VAE nodes (to see what a neural net learns from this representation)

WHY THIS MATTERS:

If consciousness operates in frequency domain, then:
- Raw EEG = "cortical pixel space"  
- This node's output = "perception frequency space"

The patterns in this output might correspond more directly to 
conscious experience than raw EEG ever could.
"""