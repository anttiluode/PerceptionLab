"""
NeuralQuantumExplorer - Finding the Smallest Unit
==================================================

"What is the minimum configuration that can predict itself?"

Antti discovered that holographic interference structure disappears below
~65ms at 256 frequency bins, but requires different parameters at 128 bins.
This isn't arbitrary - it's the sampling theorem meeting neural dynamics.

The relationship: bins × time_window must exceed some threshold for
coherent structure to emerge. This threshold might be the NEURAL QUANTUM -
the minimum information unit that brain dynamics can support.

The Free Energy Principle says systems minimize surprise by finding stable
states. The smallest stable state is the minimum configuration that can
self-evidence - predict itself.

THIS NODE:
1. Adaptively searches for the natural resolution of EEG organization
2. Finds the minimum time window where structure survives
3. Identifies natural frequency clusters (eigenmodes) that persist
4. Measures the "quantum" - minimum stable information unit
5. Tracks how the quantum changes with brain state

THEORY:
- Heisenberg: Δt × Δf ≥ 1/(4π)
- But neural systems have additional constraints from:
  - Membrane time constants (~10-50ms)
  - Synaptic integration windows (~1-20ms)  
  - Thalamocortical loop delays (~20-80ms)
  - Refractory periods (~1-5ms)
  
The neural quantum should emerge where these biological constraints
meet the mathematical uncertainty limit.

OUTPUTS:
- quantum_time: Minimum time window with structure (ms)
- quantum_freq: Corresponding frequency resolution (Hz)  
- quantum_bits: Information content at quantum scale
- n_eigenmodes: Number of natural frequency clusters found
- eigenmode_centers: Frequencies of natural modes
- coherence_at_quantum: How stable the structure is
- structure_strength: Measure of pattern emergence

Created: December 2025
For Antti's quest to find the smallest unit
"""

import numpy as np
import cv2
from scipy.fft import fft, rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.ndimage import gaussian_filter, label
from scipy.cluster.hierarchy import fclusterdata
from collections import deque
import os

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


class NeuralQuantumExplorer(BaseNode):
    """
    Adaptively searches for the minimum stable unit of neural organization.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Neural Quantum Explorer"
    NODE_COLOR = QtGui.QColor(0, 255, 255)  # Cyan - the edge of visibility
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'raw_eeg': 'signal',           # External EEG input
            'search_mode': 'signal',        # 0=auto, 1=manual sweep
            'target_structure': 'signal',   # Desired structure strength
            'reset': 'signal'
        }
        
        self.outputs = {
            # Quantum measurements
            'quantum_time': 'signal',       # Minimum time (ms)
            'quantum_freq': 'signal',       # Frequency resolution (Hz)
            'quantum_bits': 'signal',       # Information content
            'quantum_product': 'signal',    # time × freq (uncertainty measure)
            
            # Eigenmode outputs  
            'n_eigenmodes': 'signal',       # Number of natural clusters
            'eigenmode_spectrum': 'spectrum', # Power at eigenmode centers
            'mode_coherence': 'signal',     # Cross-mode phase coherence
            
            # Structure measures
            'structure_strength': 'signal', # Pattern emergence measure
            'coherence_map': 'image',       # Where coherence lives
            'quantum_field': 'image',       # Field at quantum resolution
            
            # Search state
            'current_window': 'signal',     # Current test window (ms)
            'current_bins': 'signal',       # Current test bins
            'search_complete': 'signal',    # 1 when quantum found
            
            # Combined view
            'combined_view': 'image',
            
            # Pass-through for chaining
            'eigenmode_freqs': 'spectrum'   # Centers of found modes
        }
        
        # === EEG SOURCE ===
        self.edf_path = ""
        self.selected_region = "All"
        self.raw_mne = None
        self.fs = 256.0
        self.current_time = 0.0
        self._last_path = ""
        
        # === SEARCH PARAMETERS ===
        # Time window search range (ms)
        self.min_window_ms = 20.0    # Below membrane time constant
        self.max_window_ms = 500.0   # Above typical ERP
        self.window_step_ms = 5.0    # Search resolution
        
        # Frequency bin search range
        self.min_bins = 16
        self.max_bins = 512
        self.bin_steps = [16, 32, 64, 128, 256, 512]
        
        # Structure detection threshold
        self.structure_threshold = 0.1  # Minimum to consider "structured"
        
        # === SEARCH STATE ===
        self.search_mode = 'auto'  # 'auto' or 'manual'
        self.current_window_ms = self.max_window_ms
        self.current_bins = 256
        self.search_phase = 'coarse'  # 'coarse', 'fine', 'complete'
        
        # Results storage
        self.structure_map = {}  # (window_ms, bins) -> structure_strength
        self.quantum_found = False
        self.quantum_time_ms = 0.0
        self.quantum_freq_hz = 0.0
        self.quantum_bits = 0.0
        
        # === EIGENMODE DETECTION ===
        self.eigenmode_centers = []
        self.eigenmode_powers = []
        self.n_eigenmodes = 0
        
        # === BUFFERS ===
        self.eeg_buffer = np.zeros(int(self.fs * self.max_window_ms / 1000))
        self.structure_history = deque(maxlen=100)
        
        # === OUTPUTS ===
        self.coherence_map = np.zeros((64, 64))
        self.quantum_field = np.zeros((128, 128))
        self.structure_strength = 0.0
        self.mode_coherence = 0.0
        
        self.t = 0
    
    def _load_edf(self):
        """Load EEG file."""
        if not MNE_AVAILABLE or not os.path.exists(self.edf_path):
            return False
        
        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            raw.resample(self.fs, verbose=False)
            self.raw_mne = raw
            self._last_path = self.edf_path
            self.current_time = 0.0
            print(f"[QuantumExplorer] Loaded: {self.edf_path}")
            return True
        except Exception as e:
            print(f"[QuantumExplorer] Load error: {e}")
            return False
    
    def _get_eeg_window(self, window_ms):
        """Get EEG window of specified duration."""
        n_samples = int(window_ms * self.fs / 1000)
        
        if self.raw_mne is not None:
            start = int(self.current_time * self.fs)
            end = start + n_samples
            
            if end >= self.raw_mne.n_times:
                self.current_time = 0.0
                start = 0
                end = n_samples
            
            data, _ = self.raw_mne[:, start:end]
            if data.ndim > 1:
                data = np.mean(data, axis=0)
            
            self.current_time += 1.0 / 30.0
            return data
        
        # Use buffer
        return self.eeg_buffer[-n_samples:] if n_samples <= len(self.eeg_buffer) else self.eeg_buffer
    
    def _compute_structure_strength(self, signal, n_bins):
        """
        Compute how much structure exists in the signal at given resolution.
        Returns value 0-1 where higher = more organized structure.
        """
        if len(signal) < 10:
            return 0.0
        
        # Compute spectrum
        spectrum = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), 1/self.fs)
        
        if len(spectrum) < n_bins:
            return 0.0
        
        # Bin the spectrum
        bin_edges = np.linspace(0, len(spectrum), n_bins + 1, dtype=int)
        binned = np.zeros(n_bins)
        for i in range(n_bins):
            start, end = bin_edges[i], bin_edges[i+1]
            if end > start:
                binned[i] = np.mean(spectrum[start:end])
        
        if binned.max() == 0:
            return 0.0
        
        # Normalize
        binned = binned / binned.max()
        
        # Structure measures:
        
        # 1. Peakiness - are there clear peaks vs flat noise?
        peaks, properties = find_peaks(binned, height=0.3, distance=2)
        peakiness = len(peaks) / n_bins * 10  # Scaled
        
        # 2. Entropy - low entropy = more structure
        hist, _ = np.histogram(binned, bins=20, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(20)
        structure_from_entropy = 1 - entropy
        
        # 3. Autocorrelation - structure repeats
        autocorr = np.correlate(binned - binned.mean(), binned - binned.mean(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        # Find secondary peaks
        ac_peaks, _ = find_peaks(autocorr[1:], height=0.2)
        periodicity = len(ac_peaks) / 10
        
        # 4. Variance ratio - signal vs noise floor
        sorted_bins = np.sort(binned)
        noise_floor = np.mean(sorted_bins[:n_bins//4])
        signal_level = np.mean(sorted_bins[-n_bins//4:])
        snr = signal_level / (noise_floor + 1e-10)
        snr_score = min(1.0, snr / 10)
        
        # Combine measures
        structure = (peakiness * 0.3 + 
                    structure_from_entropy * 0.3 + 
                    periodicity * 0.2 + 
                    snr_score * 0.2)
        
        return float(np.clip(structure, 0, 1))
    
    def _find_eigenmodes(self, signal, n_bins):
        """
        Find natural frequency clusters in the signal.
        These are the eigenmodes - stable resonant frequencies.
        """
        if len(signal) < 10:
            return [], []
        
        spectrum = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), 1/self.fs)
        
        if len(spectrum) < 3:
            return [], []
        
        # Smooth spectrum
        if len(spectrum) > 10:
            spectrum_smooth = gaussian_filter(spectrum, sigma=2)
        else:
            spectrum_smooth = spectrum
        
        # Find peaks
        peaks, properties = find_peaks(spectrum_smooth, 
                                       height=spectrum_smooth.max() * 0.1,
                                       distance=max(1, len(spectrum) // 20))
        
        if len(peaks) == 0:
            return [], []
        
        # Get frequencies and powers at peaks
        peak_freqs = freqs[peaks] if len(freqs) > max(peaks) else []
        peak_powers = spectrum[peaks] if len(spectrum) > max(peaks) else []
        
        if len(peak_freqs) == 0:
            return [], []
        
        # Cluster nearby peaks into eigenmodes
        if len(peak_freqs) > 1:
            try:
                # Cluster based on frequency proximity
                freq_array = np.array(peak_freqs).reshape(-1, 1)
                clusters = fclusterdata(freq_array, t=5.0, criterion='distance')
                
                # Average within clusters
                eigenmode_centers = []
                eigenmode_powers = []
                for c in np.unique(clusters):
                    mask = clusters == c
                    center = np.average(peak_freqs[mask], weights=peak_powers[mask])
                    power = np.sum(peak_powers[mask])
                    eigenmode_centers.append(center)
                    eigenmode_powers.append(power)
                
                return eigenmode_centers, eigenmode_powers
            except:
                pass
        
        return list(peak_freqs), list(peak_powers)
    
    def _compute_mode_coherence(self, signal, eigenmode_centers):
        """
        Compute phase coherence between eigenmodes.
        High coherence = modes are phase-locked = stable structure.
        """
        if len(eigenmode_centers) < 2 or len(signal) < 20:
            return 0.0
        
        nyq = self.fs / 2
        phases = []
        
        for freq in eigenmode_centers[:5]:  # Limit to first 5 modes
            if freq <= 0 or freq >= nyq * 0.9:
                continue
            
            try:
                bw = max(1.0, freq * 0.2)
                low = max(0.1, freq - bw) / nyq
                high = min(0.99, freq + bw) / nyq
                
                if low >= high:
                    continue
                
                b, a = butter(2, [low, high], btype='band')
                filtered = filtfilt(b, a, signal)
                analytic = hilbert(filtered)
                phase = np.angle(analytic)
                phases.append(phase)
            except:
                continue
        
        if len(phases) < 2:
            return 0.0
        
        # Compute pairwise phase coherence
        coherences = []
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                # Phase locking value
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                coherences.append(plv)
        
        return float(np.mean(coherences)) if coherences else 0.0
    
    def _search_quantum(self, signal):
        """
        Search for the minimum time window where structure survives.
        """
        if self.search_phase == 'complete':
            return
        
        # Test current parameters
        test_signal = signal[-int(self.current_window_ms * self.fs / 1000):]
        structure = self._compute_structure_strength(test_signal, self.current_bins)
        
        # Store result
        key = (self.current_window_ms, self.current_bins)
        self.structure_map[key] = structure
        self.structure_history.append(structure)
        
        if self.search_phase == 'coarse':
            # Coarse search: find approximate quantum
            if structure < self.structure_threshold:
                # Lost structure - quantum is somewhere above current window
                if self.current_window_ms < self.max_window_ms:
                    # Found lower bound, switch to fine search
                    self.search_phase = 'fine'
                    self.quantum_time_ms = self.current_window_ms + self.window_step_ms * 5
                else:
                    # At max window, try fewer bins
                    current_idx = self.bin_steps.index(self.current_bins) if self.current_bins in self.bin_steps else 0
                    if current_idx > 0:
                        self.current_bins = self.bin_steps[current_idx - 1]
                        self.current_window_ms = self.max_window_ms
            else:
                # Still have structure, try smaller window
                self.current_window_ms -= self.window_step_ms * 5
                if self.current_window_ms < self.min_window_ms:
                    # Reached minimum, found quantum
                    self.quantum_time_ms = self.min_window_ms
                    self.search_phase = 'complete'
                    self.quantum_found = True
        
        elif self.search_phase == 'fine':
            # Fine search around approximate quantum
            if structure >= self.structure_threshold:
                # Still have structure, try smaller
                self.current_window_ms -= self.window_step_ms
                if self.current_window_ms < self.min_window_ms:
                    self.current_window_ms = self.min_window_ms
                    self.search_phase = 'complete'
                    self.quantum_found = True
            else:
                # Lost structure - previous window was the quantum
                self.quantum_time_ms = self.current_window_ms + self.window_step_ms
                self.search_phase = 'complete'
                self.quantum_found = True
        
        if self.quantum_found:
            # Calculate quantum properties
            self.quantum_freq_hz = 1000.0 / self.quantum_time_ms
            self.quantum_bits = np.log2(self.current_bins) + np.log2(self.quantum_time_ms)
    
    def _render_quantum_field(self, signal, n_bins):
        """Render field at current resolution."""
        size = 128
        
        if len(signal) < 10:
            self.quantum_field = np.zeros((size, size))
            return
        
        spectrum = np.abs(rfft(signal))
        
        # Create 2D representation via outer product with phase
        if len(spectrum) > 1:
            # Resize spectrum to size
            spectrum_resized = np.interp(
                np.linspace(0, len(spectrum)-1, size),
                np.arange(len(spectrum)),
                spectrum
            )
            
            # Create 2D field
            phase = np.linspace(0, 2*np.pi, size)
            field = np.outer(spectrum_resized, np.cos(phase))
            field += np.outer(spectrum_resized[::-1], np.sin(phase))
            
            # Normalize
            if field.max() != field.min():
                field = (field - field.min()) / (field.max() - field.min())
            
            self.quantum_field = field
        else:
            self.quantum_field = np.zeros((size, size))
    
    def _render_coherence_map(self, eigenmode_centers, eigenmode_powers):
        """Render coherence between modes as 2D map."""
        n = len(eigenmode_centers)
        size = 64
        
        if n < 2:
            self.coherence_map = np.zeros((size, size))
            return
        
        # Create mode interaction map
        map_size = min(n, size)
        coh_map = np.zeros((map_size, map_size))
        
        for i in range(map_size):
            for j in range(map_size):
                if i < n and j < n:
                    # Interaction strength based on frequency ratio
                    ratio = eigenmode_centers[i] / (eigenmode_centers[j] + 1e-10)
                    # Harmonic relationships show up as integer ratios
                    harmonic = min(abs(ratio - round(ratio)), 0.5) * 2
                    coh_map[i, j] = (1 - harmonic) * np.sqrt(eigenmode_powers[i] * eigenmode_powers[j])
        
        # Resize to standard size
        if map_size != size:
            coh_map = cv2.resize(coh_map, (size, size))
        
        if coh_map.max() > 0:
            coh_map = coh_map / coh_map.max()
        
        self.coherence_map = coh_map
    
    def step(self):
        self.t += 1
        
        # Get inputs
        raw_in = self.get_blended_input('raw_eeg', 'sum')
        mode_in = self.get_blended_input('search_mode', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0.5:
            self.search_phase = 'coarse'
            self.current_window_ms = self.max_window_ms
            self.quantum_found = False
            self.structure_map.clear()
            return
        
        # Load EDF if needed
        if self.edf_path and self.edf_path != self._last_path:
            self._load_edf()
        
        # Update buffer
        if raw_in is not None:
            self.eeg_buffer = np.roll(self.eeg_buffer, -1)
            self.eeg_buffer[-1] = raw_in
        
        # Get signal
        signal = self._get_eeg_window(self.max_window_ms)
        if signal is None or len(signal) < 10:
            return
        
        # Search for quantum
        self._search_quantum(signal)
        
        # Compute at current resolution
        current_signal = signal[-int(self.current_window_ms * self.fs / 1000):]
        self.structure_strength = self._compute_structure_strength(current_signal, self.current_bins)
        
        # Find eigenmodes
        self.eigenmode_centers, self.eigenmode_powers = self._find_eigenmodes(current_signal, self.current_bins)
        self.n_eigenmodes = len(self.eigenmode_centers)
        
        # Compute mode coherence
        self.mode_coherence = self._compute_mode_coherence(current_signal, self.eigenmode_centers)
        
        # Render visualizations
        self._render_quantum_field(current_signal, self.current_bins)
        self._render_coherence_map(self.eigenmode_centers, self.eigenmode_powers)
    
    def get_output(self, port_name):
        if port_name == 'quantum_time':
            return float(self.quantum_time_ms) if self.quantum_found else float(self.current_window_ms)
        
        elif port_name == 'quantum_freq':
            if self.quantum_found and self.quantum_time_ms > 0:
                return float(1000.0 / self.quantum_time_ms)
            return 0.0
        
        elif port_name == 'quantum_bits':
            return float(self.quantum_bits)
        
        elif port_name == 'quantum_product':
            # Uncertainty product: time (s) × freq (Hz)
            if self.quantum_found and self.quantum_time_ms > 0:
                return float((self.quantum_time_ms / 1000) * (1000.0 / self.quantum_time_ms))
            return 0.0
        
        elif port_name == 'n_eigenmodes':
            return float(self.n_eigenmodes)
        
        elif port_name == 'eigenmode_spectrum':
            # Return powers at eigenmode frequencies
            if len(self.eigenmode_powers) > 0:
                return np.array(self.eigenmode_powers, dtype=np.float32)
            return np.zeros(1, dtype=np.float32)
        
        elif port_name == 'mode_coherence':
            return float(self.mode_coherence)
        
        elif port_name == 'structure_strength':
            return float(self.structure_strength)
        
        elif port_name == 'coherence_map':
            return (self.coherence_map * 255).astype(np.uint8)
        
        elif port_name == 'quantum_field':
            return (self.quantum_field * 255).astype(np.uint8)
        
        elif port_name == 'current_window':
            return float(self.current_window_ms)
        
        elif port_name == 'current_bins':
            return float(self.current_bins)
        
        elif port_name == 'search_complete':
            return 1.0 if self.quantum_found else 0.0
        
        elif port_name == 'eigenmode_freqs':
            if len(self.eigenmode_centers) > 0:
                return np.array(self.eigenmode_centers, dtype=np.float32)
            return np.zeros(1, dtype=np.float32)
        
        elif port_name == 'combined_view':
            return self._render_combined()
        
        return None
    
    def _render_combined(self):
        """Render combined visualization."""
        h, w = 256, 384
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Quantum field (left)
        field_vis = (self.quantum_field * 255).astype(np.uint8)
        field_color = cv2.applyColorMap(field_vis, cv2.COLORMAP_PLASMA)
        field_resized = cv2.resize(field_color, (128, 128))
        display[:128, :128] = field_resized
        
        # Coherence map (bottom left)
        coh_vis = (self.coherence_map * 255).astype(np.uint8)
        coh_color = cv2.applyColorMap(coh_vis, cv2.COLORMAP_HOT)
        coh_resized = cv2.resize(coh_color, (128, 128))
        display[128:, :128] = coh_resized
        
        # Structure history (top right)
        hist_h, hist_w = 80, 250
        if len(self.structure_history) > 1:
            hist_arr = np.array(self.structure_history)
            hist_arr = hist_arr / (hist_arr.max() + 1e-10)
            for i, v in enumerate(hist_arr[-hist_w:]):
                x = 130 + i
                bar_h = int(v * hist_h)
                display[hist_h-bar_h:hist_h, x] = [100, 255, 100]
        
        # Threshold line
        thresh_y = int((1 - self.structure_threshold) * hist_h)
        cv2.line(display, (130, thresh_y), (130 + hist_w, thresh_y), (0, 0, 255), 1)
        
        # Info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 100
        
        cv2.putText(display, "NEURAL QUANTUM EXPLORER", (135, y), font, 0.35, (0, 255, 255), 1)
        y += 20
        
        status = "FOUND" if self.quantum_found else self.search_phase.upper()
        cv2.putText(display, f"Status: {status}", (135, y), font, 0.35, (255, 255, 255), 1)
        y += 18
        
        cv2.putText(display, f"Window: {self.current_window_ms:.1f} ms", (135, y), font, 0.35, (255, 255, 255), 1)
        y += 18
        
        cv2.putText(display, f"Bins: {self.current_bins}", (135, y), font, 0.35, (255, 255, 255), 1)
        y += 18
        
        cv2.putText(display, f"Structure: {self.structure_strength:.3f}", (135, y), font, 0.35, (255, 255, 255), 1)
        y += 18
        
        if self.quantum_found:
            cv2.putText(display, f"QUANTUM: {self.quantum_time_ms:.1f} ms", (135, y), font, 0.4, (0, 255, 0), 1)
            y += 18
            cv2.putText(display, f"= {1000/self.quantum_time_ms:.1f} Hz", (135, y), font, 0.4, (0, 255, 0), 1)
        
        y += 25
        cv2.putText(display, f"Eigenmodes: {self.n_eigenmodes}", (135, y), font, 0.35, (255, 200, 100), 1)
        y += 18
        cv2.putText(display, f"Mode Coherence: {self.mode_coherence:.3f}", (135, y), font, 0.35, (255, 200, 100), 1)
        
        # Show eigenmode frequencies
        if len(self.eigenmode_centers) > 0:
            y += 18
            freqs_str = ", ".join([f"{f:.1f}" for f in self.eigenmode_centers[:5]])
            cv2.putText(display, f"Modes: {freqs_str} Hz", (135, y), font, 0.3, (200, 200, 200), 1)
        
        return display
    
    def get_display_image(self):
        display = self._render_combined()
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        region_opts = [
            ('All', 'All'), ('Occipital', 'Occipital'), ('Temporal', 'Temporal'),
            ('Parietal', 'Parietal'), ('Frontal', 'Frontal'), ('Central', 'Central')
        ]
        
        return [
            ("EDF File Path", "edf_path", self.edf_path, None),
            ("Brain Region", "selected_region", self.selected_region, region_opts),
            ("Min Window (ms)", "min_window_ms", self.min_window_ms, None),
            ("Max Window (ms)", "max_window_ms", self.max_window_ms, None),
            ("Window Step (ms)", "window_step_ms", self.window_step_ms, None),
            ("Structure Threshold", "structure_threshold", self.structure_threshold, None),
            ("Starting Bins", "current_bins", self.current_bins, None),
        ]