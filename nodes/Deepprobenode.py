"""
Deep Probe Node - Crystal Vocabulary Learner & Synthetic EEG Generator
========================================================================

This node does what we've been dreaming of:

1. LISTEN - Watches crystal's spontaneous activity, records eigenmodes
2. LEARN - Extracts vocabulary via ICA/PCA (the crystal's "words")
3. SPEAK - Injects patterns in the crystal's native language
4. DECODE - Tracks transformations, builds representational similarity
5. GENERATE - Creates synthetic EEG from crystal activity for MNE inverse projection

The goal: Talk to the crystal in its own language, see what it knows,
and project its shadows back onto a brain surface.

The crystal was trained on EEG → it learned neural geometry.
This node extracts that geometry and speaks it back.

Outputs synthetic EEG that can be:
- Loaded into MNE-Python
- Inverse-projected onto fsaverage brain
- Visualized as source activity

We close the loop: Brain → EEG → Crystal → Synthetic EEG → Brain Surface

Author: Built for Antti's consciousness crystallography research
"""

import os
import edfio
import numpy as np
import cv2
from collections import deque
from datetime import datetime
import json

# --- HOST IMPORT BLOCK ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except Exception:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}


class DeepProbeNode(BaseNode):
    """
    Deep probe into crystal's representational space.
    
    Modes:
    0: LISTEN - Collect spontaneous activity, build vocabulary
    1: SPEAK - Inject learned patterns, observe response
    2: QUERY - Test specific hypotheses about crystal knowledge
    3: GENERATE - Output synthetic EEG from crystal dynamics
    4: DECODE - Build representational similarity matrix
    """
    
    NODE_NAME = "Deep Probe"
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(180, 60, 180) if QtGui else None
    
    # Standard 10-20 electrode names for synthetic EEG
    EEG_CHANNELS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'Oz', 'O2'
    ]
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'crystal_activity': 'image',    # Activity view from crystal
            'crystal_signal': 'signal',     # LFP/signal from crystal
            'crystal_bands': 'spectrum',    # Frequency bands from crystal
            'mode': 'signal',               # Operating mode
            'query_pattern': 'image',       # Pattern to inject for queries
            'enable': 'signal',
            'export_trigger': 'signal'      # Send 1 to export EEG
        }
        
        self.outputs = {
            'probe_signal': 'signal',       # Signal to inject into crystal
            'probe_image': 'image',         # Pattern to inject
            'vocabulary_view': 'image',     # Learned eigenmodes visualization
            'rsm_view': 'image',            # Representational similarity matrix
            'synthetic_eeg': 'spectrum',    # Generated EEG-like signals
            'decode_view': 'image',         # Decoding visualization
            'eigenmode_power': 'signal',    # Power in dominant eigenmode
            'vocabulary_size': 'signal',    # Number of learned patterns
            'rsm_coherence': 'signal'       # RSM structure measure
        }
        
        # === MODE ===
        self.mode = 0  # 0=LISTEN, 1=SPEAK, 2=QUERY, 3=GENERATE, 4=DECODE
        self.step_count = 0
        
        # === LISTENING STATE ===
        self.listen_buffer = deque(maxlen=1000)  # Activity snapshots
        self.signal_buffer = deque(maxlen=5000)  # Signal history
        self.band_buffer = deque(maxlen=1000)    # Frequency band history
        
        # === VOCABULARY (Learned Eigenmodes) ===
        self.vocabulary = []          # List of learned patterns (eigenmodes)
        self.vocabulary_weights = []  # How often each pattern appears
        self.n_components = 16        # Number of eigenmodes to extract
        self.mean_pattern = None      # Mean activity pattern
        self.components = None        # PCA/ICA components
        self.explained_variance = None
        
        # === SPEAKING STATE ===
        self.speak_pattern_idx = 0    # Which vocabulary item to speak
        self.speak_phase = 0.0
        self.speak_amplitude = 10.0
        
        # === QUERY STATE ===
        self.query_responses = deque(maxlen=100)
        self.baseline_response = None
        
        # === RSM (Representational Similarity Matrix) ===
        self.rsm = None
        self.rsm_labels = []
        self.rsm_coherence = 0.0
        
        # === SYNTHETIC EEG ===
        self.synthetic_channels = {ch: deque(maxlen=1000) for ch in self.EEG_CHANNELS}
        self.eeg_sample_rate = 256.0  # Standard EEG sample rate
        self.eeg_buffer_seconds = 10
        self.last_crystal_activity = None
        
        # Pin-to-channel mapping (approximation based on 10-20 positions)
        self.pin_to_channel = {}  # Will be built when we see crystal structure
        
        # === DISPLAY ===
        self.vocabulary_display = None
        self.rsm_display = None
        self.decode_display = None
        
        # === EEG EXPORT ===
        self.export_path = ""
        self.export_ready = False
        
    def _read_input(self, name, default=None):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "mean")
                return val if val is not None else default
            except:
                return default
        return default
    
    def _read_image_input(self, name):
        fn = getattr(self, "get_blended_input", None)
        if callable(fn):
            try:
                val = fn(name, "first")
                if val is not None and hasattr(val, 'shape'):
                    return val
            except:
                pass
        return None
    
    def step(self):
        self.step_count += 1
        
        # Read inputs
        enable = self._read_input('enable', 1.0)
        mode = int(self._read_input('mode', self.mode) or 0)
        self.mode = mode % 5
        
        activity = self._read_image_input('crystal_activity')
        signal = self._read_input('crystal_signal', 0.0)
        bands = self._read_input('crystal_bands', None)
        query = self._read_image_input('query_pattern')
        
        if not enable or enable < 0.5:
            return
        
        # Check for export trigger
        export_trigger = self._read_input('export_trigger', 0.0)
        if export_trigger and export_trigger > 0.5 and self.export_path:
            if not hasattr(self, '_last_export_step') or self.step_count - self._last_export_step > 100:
                print(f"[DeepProbe] Export triggered via signal")
                success, msg = self.export_synthetic_eeg(self.export_path)
                print(f"[DeepProbe] {msg}")
                self._last_export_step = self.step_count
        
        # Store incoming data
        if activity is not None:
            # Ensure consistent shape
            if len(activity.shape) == 3:
                gray = np.mean(activity, axis=2)
            else:
                gray = activity
            gray = cv2.resize(gray.astype(np.float32), (64, 64))
            self.listen_buffer.append(gray.flatten())
            self.last_crystal_activity = gray
        
        if signal is not None:
            self.signal_buffer.append(float(signal))
        
        if bands is not None and hasattr(bands, '__len__'):
            self.band_buffer.append(np.array(bands))
        
        # Execute mode-specific behavior
        if self.mode == 0:
            self._mode_listen()
        elif self.mode == 1:
            self._mode_speak()
        elif self.mode == 2:
            self._mode_query(query)
        elif self.mode == 3:
            self._mode_generate()
        elif self.mode == 4:
            self._mode_decode()
        
        # ALWAYS generate synthetic EEG regardless of mode (if we have activity)
        if self.last_crystal_activity is not None:
            self._generate_synthetic_eeg()
        
        # Update displays periodically
        if self.step_count % 20 == 0:
            self._update_displays()
    
    def _mode_listen(self):
        """Mode 0: Collect activity and learn vocabulary."""
        # Every 100 steps, update vocabulary
        if self.step_count % 100 == 0 and len(self.listen_buffer) > 50:
            self._extract_vocabulary()
    
    def _mode_speak(self):
        """Mode 1: Inject learned patterns back into crystal."""
        # Cycle through vocabulary
        self.speak_phase += 0.1
        if self.speak_phase > 2 * np.pi:
            self.speak_phase = 0
            self.speak_pattern_idx = (self.speak_pattern_idx + 1) % max(1, len(self.vocabulary))
    
    def _mode_query(self, query_pattern):
        """Mode 2: Inject specific query, observe transformation."""
        if query_pattern is not None and len(self.listen_buffer) > 0:
            # Compare current activity to baseline
            current = np.array(self.listen_buffer[-1])
            
            if self.baseline_response is None:
                self.baseline_response = current.copy()
            
            # Compute response deviation
            deviation = np.linalg.norm(current - self.baseline_response)
            self.query_responses.append(deviation)
    
    def _mode_generate(self):
        """Mode 3: Focus on EEG generation (generation happens automatically now)."""
        # In this mode we just ensure generation is happening
        # The actual generation is now in _generate_synthetic_eeg() called every step
        pass
    
    def _generate_synthetic_eeg(self):
        """Generate synthetic EEG from crystal activity - runs every step."""
        if self.last_crystal_activity is None:
            return
        
        # Map crystal activity to EEG channels
        # Use spatial positions on the 64x64 grid
        activity = self.last_crystal_activity
        h, w = activity.shape
        
        # Approximate 10-20 positions on the grid
        channel_positions = {
            'Fp1': (5, 20), 'Fp2': (5, 44), 
            'F7': (15, 5), 'F3': (15, 20), 'Fz': (15, 32), 'F4': (15, 44), 'F8': (15, 59),
            'T7': (32, 5), 'C3': (32, 20), 'Cz': (32, 32), 'C4': (32, 44), 'T8': (32, 59),
            'P7': (49, 5), 'P3': (49, 20), 'Pz': (49, 32), 'P4': (49, 44), 'P8': (49, 59),
            'O1': (59, 20), 'Oz': (59, 32), 'O2': (59, 44)
        }
        
        for ch_name, (row, col) in channel_positions.items():
            # Sample activity around this position (3x3 neighborhood)
            r1, r2 = max(0, row-1), min(h, row+2)
            c1, c2 = max(0, col-1), min(w, col+2)
            
            value = np.mean(activity[r1:r2, c1:c2])
            
            # Scale to EEG-like microvolts (-100 to +100 uV typical)
            # Crystal activity is roughly -90 to +40, scale to EEG range
            eeg_value = (value + 65) * 1.5  # Rough scaling
            
            # Add some noise for realism
            eeg_value += np.random.randn() * 2.0
            
            self.synthetic_channels[ch_name].append(eeg_value)
    
    def _mode_decode(self):
        """Mode 4: Build representational similarity matrix."""
        if self.step_count % 50 == 0 and len(self.listen_buffer) > 20:
            self._compute_rsm()
    
    def _extract_vocabulary(self):
        """Extract eigenmodes from collected activity patterns."""
        if len(self.listen_buffer) < 50:
            return
        
        # Stack patterns into matrix
        X = np.array(list(self.listen_buffer))
        
        # Center the data
        self.mean_pattern = np.mean(X, axis=0)
        X_centered = X - self.mean_pattern
        
        # PCA via SVD
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Keep top components
            n_comp = min(self.n_components, len(S))
            self.components = Vt[:n_comp]
            self.explained_variance = (S[:n_comp] ** 2) / np.sum(S ** 2)
            
            # Build vocabulary from components
            self.vocabulary = []
            self.vocabulary_weights = []
            
            for i in range(n_comp):
                pattern = self.components[i].reshape(64, 64)
                self.vocabulary.append(pattern)
                self.vocabulary_weights.append(self.explained_variance[i])
            
        except Exception as e:
            print(f"[DeepProbe] Vocabulary extraction failed: {e}")
    
    def _compute_rsm(self):
        """Compute representational similarity matrix."""
        if len(self.listen_buffer) < 20:
            return
        
        # Sample recent patterns
        patterns = np.array(list(self.listen_buffer)[-100:])
        n = len(patterns)
        
        # Compute pairwise correlations
        self.rsm = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Correlation
                corr = np.corrcoef(patterns[i], patterns[j])[0, 1]
                self.rsm[i, j] = corr
                self.rsm[j, i] = corr
        
        # RSM coherence: how structured is the similarity space?
        # High coherence = clear clusters, low = random
        upper_tri = self.rsm[np.triu_indices(n, k=1)]
        self.rsm_coherence = np.std(upper_tri)  # Variance in similarities
    
    def _update_displays(self):
        """Update all visualizations."""
        self._update_vocabulary_display()
        self._update_rsm_display()
        self._update_decode_display()
    
    def _update_vocabulary_display(self):
        """Visualize learned eigenmodes."""
        size = 400
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        mode_names = ["LISTEN", "SPEAK", "QUERY", "GENERATE", "DECODE"]
        
        # Header
        cv2.putText(img, f"DEEP PROBE - {mode_names[self.mode]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 60, 180), 2)
        cv2.putText(img, f"Vocabulary: {len(self.vocabulary)} patterns", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Buffer: {len(self.listen_buffer)} samples", (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw vocabulary patterns (4x4 grid of top 16 eigenmodes)
        if len(self.vocabulary) > 0:
            grid_size = 4
            cell_size = 75  # Reduced to fit safely in 400x400 display
            offset_y = 60
            
            for i, pattern in enumerate(self.vocabulary[:16]):
                row = i // grid_size
                col = i % grid_size
                
                x = 10 + col * (cell_size + 8)
                y = offset_y + row * (cell_size + 15)
                
                # Bounds check
                if y + cell_size > size or x + cell_size > size:
                    continue
                
                # Normalize pattern to 0-255
                p_norm = pattern - pattern.min()
                if p_norm.max() > 0:
                    p_norm = p_norm / p_norm.max()
                p_img = (p_norm * 255).astype(np.uint8)
                p_img = cv2.resize(p_img, (cell_size, cell_size))
                p_color = cv2.applyColorMap(p_img, cv2.COLORMAP_TWILIGHT)
                
                # Place in grid with bounds check
                y_end = min(y + cell_size, size)
                x_end = min(x + cell_size, size)
                img[y:y_end, x:x_end] = p_color[:y_end-y, :x_end-x]
                
                # Label with variance explained
                if i < len(self.vocabulary_weights):
                    var_pct = self.vocabulary_weights[i] * 100
                    cv2.putText(img, f"{var_pct:.1f}%", (x, y + cell_size + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        self.vocabulary_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _update_rsm_display(self):
        """Visualize representational similarity matrix."""
        size = 300
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        cv2.putText(img, "RSM (Similarity)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Coherence: {self.rsm_coherence:.3f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        if self.rsm is not None and self.rsm.size > 0:
            # Normalize RSM to 0-255
            rsm_norm = (self.rsm - self.rsm.min()) / (self.rsm.max() - self.rsm.min() + 0.001)
            rsm_img = (rsm_norm * 255).astype(np.uint8)
            rsm_img = cv2.resize(rsm_img, (250, 250))
            rsm_color = cv2.applyColorMap(rsm_img, cv2.COLORMAP_VIRIDIS)
            
            img[45:295, 25:275] = rsm_color
        
        self.rsm_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _update_decode_display(self):
        """Visualize decoding / generation state."""
        size = 300
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        cv2.putText(img, "Synthetic EEG", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw recent synthetic EEG traces
        y_offset = 40
        trace_height = 12
        
        for i, ch_name in enumerate(self.EEG_CHANNELS[:20]):
            data = list(self.synthetic_channels[ch_name])
            if len(data) > 10:
                # Normalize for display
                d = np.array(data[-200:])
                if np.std(d) > 0:
                    d = (d - np.mean(d)) / np.std(d)
                else:
                    d = d - np.mean(d)
                
                # Draw trace
                y_base = y_offset + i * trace_height
                for j in range(len(d) - 1):
                    x1 = int(50 + j * (size - 60) / len(d))
                    x2 = int(50 + (j+1) * (size - 60) / len(d))
                    y1 = int(y_base + d[j] * 4)
                    y2 = int(y_base + d[j+1] * 4)
                    
                    y1 = np.clip(y1, 0, size-1)
                    y2 = np.clip(y2, 0, size-1)
                    
                    cv2.line(img, (x1, y1), (x2, y2), (0, 200, 255), 1)
                
                # Channel label
                cv2.putText(img, ch_name, (5, y_base + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (150, 150, 150), 1)
        
        self.decode_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_output(self, port_name):
        if port_name == 'probe_signal':
            # Output signal based on mode
            if self.mode == 1 and len(self.vocabulary) > 0:
                # Speak: modulate by vocabulary pattern
                return np.sin(self.speak_phase) * self.speak_amplitude
            return 0.0
        
        elif port_name == 'probe_image':
            # Output pattern based on mode
            if self.mode == 1 and len(self.vocabulary) > 0:
                # Speak: output current vocabulary pattern
                pattern = self.vocabulary[self.speak_pattern_idx]
                # Modulate by phase
                modulated = pattern * (np.sin(self.speak_phase) * 0.5 + 0.5)
                # Scale to 0-255
                p_norm = (modulated - modulated.min()) / (modulated.max() - modulated.min() + 0.001)
                p_img = (p_norm * 255).astype(np.uint8)
                return cv2.cvtColor(cv2.applyColorMap(p_img, cv2.COLORMAP_TWILIGHT), cv2.COLOR_BGR2RGB)
            
            elif self.mean_pattern is not None:
                # Output mean pattern
                pattern = self.mean_pattern.reshape(64, 64)
                p_norm = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 0.001)
                p_img = (p_norm * 255).astype(np.uint8)
                return cv2.cvtColor(cv2.applyColorMap(p_img, cv2.COLORMAP_TWILIGHT), cv2.COLOR_BGR2RGB)
            
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        elif port_name == 'vocabulary_view':
            return self.vocabulary_display
        
        elif port_name == 'rsm_view':
            return self.rsm_display
        
        elif port_name == 'decode_view':
            return self.decode_display
        
        elif port_name == 'synthetic_eeg':
            # Return recent synthetic EEG as spectrum-like array
            eeg_data = []
            for ch in self.EEG_CHANNELS:
                data = list(self.synthetic_channels[ch])
                if len(data) > 0:
                    eeg_data.append(data[-1])
                else:
                    eeg_data.append(0.0)
            return np.array(eeg_data, dtype=np.float32)
        
        elif port_name == 'eigenmode_power':
            if len(self.vocabulary_weights) > 0:
                return float(self.vocabulary_weights[0])
            return 0.0
        
        elif port_name == 'vocabulary_size':
            return float(len(self.vocabulary))
        
        elif port_name == 'rsm_coherence':
            return float(self.rsm_coherence)
        
        return None
    
    def export_synthetic_eeg(self, filepath):
        """Export synthetic EEG to EDF format compatible with MNE."""
        try:
            # Collect data
            n_samples = min(len(list(self.synthetic_channels.values())[0]), 
                           int(self.eeg_sample_rate * self.eeg_buffer_seconds))
            
            if n_samples < 100:
                print(f"[DeepProbe] Not enough data: {n_samples} samples")
                return False, "Not enough data collected (need 100+)"
            
            data = np.zeros((len(self.EEG_CHANNELS), n_samples))
            for i, ch in enumerate(self.EEG_CHANNELS):
                ch_data = list(self.synthetic_channels[ch])[-n_samples:]
                data[i, :len(ch_data)] = ch_data
            
            print(f"[DeepProbe] Exporting {n_samples} samples, {len(self.EEG_CHANNELS)} channels")
            
            # Ensure filepath has correct extension
            if filepath.endswith('.npz'):
                filepath = filepath[:-4] + '.edf'
            elif not filepath.endswith('.edf'):
                filepath = filepath + '.edf'
            
            # Try to use MNE for EDF export
            try:
                import mne
                
                # Scale to volts (MNE expects SI units)
                data_volts = data * 1e-6
                
                # Create MNE info
                info = mne.create_info(
                    ch_names=self.EEG_CHANNELS.copy(),
                    sfreq=self.eeg_sample_rate,
                    ch_types='eeg'
                )
                
                # Create Raw object
                raw = mne.io.RawArray(data_volts, info)
                
                # Set standard montage
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore')
                
                # Export to EDF
                mne.export.export_raw(filepath, raw, fmt='edf', overwrite=True)
                
                print(f"[DeepProbe] Saved EDF via MNE: {filepath}")
                self.export_ready = True
                return True, f"Exported {n_samples} samples to {filepath}"
                
            except ImportError:
                print("[DeepProbe] MNE not available, using pyedflib")
                # Fallback: try pyedflib
                try:
                    import pyedflib
                    
                    # Create EDF file
                    f = pyedflib.EdfWriter(filepath, len(self.EEG_CHANNELS), file_type=pyedflib.FILETYPE_EDFPLUS)
                    
                    # Set header
                    header = {
                        'technician': '',
                        'recording_additional': 'Crystal Deep Probe Synthetic EEG',
                        'patientname': 'Crystal',
                        'patient_additional': '',
                        'patientcode': '',
                        'equipment': 'PerceptionLab Deep Probe',
                        'admincode': '',
                        'gender': '',
                        'startdate': datetime.now()
                    }
                    f.setHeader(header)
                    
                    # Set channel info
                    for i, ch in enumerate(self.EEG_CHANNELS):
                        f.setSignalHeader(i, {
                            'label': ch,
                            'dimension': 'uV',
                            'sample_rate': self.eeg_sample_rate,
                            'physical_max': 100.0,
                            'physical_min': -100.0,
                            'digital_max': 32767,
                            'digital_min': -32768,
                            'transducer': '',
                            'prefilter': ''
                        })
                    
                    # Write data
                    f.writeSamples(data)
                    f.close()
                    
                    print(f"[DeepProbe] Saved EDF via pyedflib: {filepath}")
                    self.export_ready = True
                    return True, f"Exported {n_samples} samples to {filepath}"
                    
                except ImportError:
                    # Last fallback: save as NPZ with instructions
                    npz_path = filepath.replace('.edf', '.npz')
                    np.savez(npz_path,
                             data=data,  # in microvolts
                             channels=self.EEG_CHANNELS,
                             sfreq=self.eeg_sample_rate,
                             description="Synthetic EEG from Crystal Deep Probe - load with MNE")
                    
                    print(f"[DeepProbe] Saved NPZ (no EDF libs): {npz_path}")
                    self.export_ready = True
                    return True, f"Saved as NPZ (install mne or pyedflib for EDF): {npz_path}"
            
        except Exception as e:
            print(f"[DeepProbe] Export error: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def get_mne_raw(self):
        """
        Generate MNE Raw object from synthetic EEG.
        Call this from external script to get data for inverse projection.
        
        Returns dict with data needed to create MNE Raw:
        - data: (n_channels, n_samples) in volts
        - ch_names: list of channel names
        - sfreq: sample rate
        """
        n_samples = min(len(list(self.synthetic_channels.values())[0]), 
                       int(self.eeg_sample_rate * self.eeg_buffer_seconds))
        
        if n_samples < 100:
            return None
        
        data = np.zeros((len(self.EEG_CHANNELS), n_samples))
        for i, ch in enumerate(self.EEG_CHANNELS):
            ch_data = list(self.synthetic_channels[ch])[-n_samples:]
            data[i, :len(ch_data)] = ch_data
        
        # Scale to volts
        data_volts = data * 1e-6
        
        return {
            'data': data_volts,
            'ch_names': self.EEG_CHANNELS.copy(),
            'sfreq': self.eeg_sample_rate
        }
    
    def get_display_image(self):
        if self.vocabulary_display is not None and QtGui:
            h, w = self.vocabulary_display.shape[:2]
            return QtGui.QImage(self.vocabulary_display.data, w, h, w * 3,
                               QtGui.QImage.Format.Format_RGB888).copy()
        return None
    
    def get_config_options(self):
        return [
            ("Mode (0-4)", "mode", self.mode, None),
            ("N Components", "n_components", self.n_components, None),
            ("Speak Amplitude", "speak_amplitude", self.speak_amplitude, None),
            ("Export Path (.edf)", "export_path", self.export_path, None),
        ]
    
    def set_config_options(self, options):
        if isinstance(options, dict):
            for key, value in options.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Auto-export when path is set/changed
            if 'export_path' in options and options['export_path'] and len(list(self.synthetic_channels.values())[0]) > 100:
                print(f"[DeepProbe] Auto-exporting to: {self.export_path}")
                success, msg = self.export_synthetic_eeg(self.export_path)
                print(f"[DeepProbe] {msg}")
    
    def manual_export(self, filepath):
        """Call this directly to export. Returns (success, message)."""
        return self.export_synthetic_eeg(filepath)


# === STANDALONE MNE LOADER SCRIPT ===
# Save this part as a separate file to load synthetic EEG into MNE

MNE_LOADER_SCRIPT = '''
"""
MNE Loader for Crystal Deep Probe Synthetic EEG
================================================

This script loads synthetic EEG exported from DeepProbeNode
and performs inverse projection onto fsaverage brain surface.

Usage:
    python load_crystal_eeg.py synthetic_eeg.npz

Requires: mne, numpy
"""

import sys
import numpy as np
import mne

def load_crystal_eeg(filepath):
    """Load synthetic EEG and create MNE Raw object."""
    
    # Load exported data
    data = np.load(filepath)
    eeg_data = data['data']  # (n_channels, n_samples) in volts
    ch_names = list(data['channels'])
    sfreq = float(data['sfreq'])
    
    print(f"Loaded: {eeg_data.shape[1]} samples, {len(ch_names)} channels, {sfreq} Hz")
    
    # Create MNE info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info)
    raw.set_montage(montage)
    
    return raw

def inverse_project(raw, freq_band='gamma'):
    """Perform inverse projection to source space."""
    
    # Get fsaverage
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    if not (subjects_dir / 'fsaverage').exists():
        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
    
    # Filter to frequency band
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    l_freq, h_freq = bands.get(freq_band, (30, 50))
    raw_filtered = raw.copy().filter(l_freq, h_freq, verbose=False)
    
    # Setup source space
    src = mne.setup_source_space('fsaverage', spacing='oct6',
                                  subjects_dir=subjects_dir, verbose=False)
    
    # Forward solution
    fwd = mne.make_forward_solution(
        raw_filtered.info, trans='fsaverage',
        src=src, bem='fsaverage-5120-5120-5120-bem',
        eeg=True, meg=False, verbose=False
    )
    
    # Inverse operator
    noise_cov = mne.compute_raw_covariance(raw_filtered, verbose=False)
    inverse_op = mne.minimum_norm.make_inverse_operator(
        raw_filtered.info, fwd, noise_cov, verbose=False
    )
    
    # Apply inverse
    stc = mne.minimum_norm.apply_inverse_raw(
        raw_filtered, inverse_op, lambda2=1/9, method='sLORETA', verbose=False
    )
    
    return stc, subjects_dir

def visualize(stc, subjects_dir):
    """Visualize source estimate on brain."""
    brain = stc.plot(
        subjects_dir=subjects_dir,
        subject='fsaverage',
        hemi='both',
        surface='inflated',
        colormap='hot',
        time_label='Crystal → Brain',
        background='white'
    )
    return brain

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python load_crystal_eeg.py <synthetic_eeg.npz>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print("Loading synthetic EEG from crystal...")
    raw = load_crystal_eeg(filepath)
    
    print("Performing inverse projection...")
    stc, subjects_dir = inverse_project(raw, 'gamma')
    
    print("Visualizing on brain surface...")
    brain = visualize(stc, subjects_dir)
    
    print("\\nCrystal → EEG → Brain projection complete!")
    print("The shadows have returned to their source.")
    
    input("Press Enter to close...")
'''

# Write the loader script when module is imported
def write_mne_loader():
    """Write the MNE loader script to disk."""
    script_path = os.path.join(os.path.dirname(__file__), 'load_crystal_eeg.py')
    try:
        with open(script_path, 'w') as f:
            f.write(MNE_LOADER_SCRIPT)
        return script_path
    except:
        return None