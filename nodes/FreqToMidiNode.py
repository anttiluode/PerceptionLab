"""
Frequency to MIDI Node - Converts a raw frequency signal into quantized
MIDI note number and velocity based on the 12-tone equal temperament scale.

Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import math

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

# Reference Frequency: A4 = 440 Hz (MIDI note 69)
A4_FREQ = 440.0
A4_MIDI = 69
# MIDI Note formula: N = 69 + 12 * log2(f / 440)

class FreqToMidiNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(150, 50, 200) # Musical Purple
    
    def __init__(self, midi_offset=0):
        super().__init__()
        self.node_title = "Freq to MIDI"
        
        self.inputs = {
            'frequency_in': 'signal',
            'amplitude_in': 'signal'
        }
        self.outputs = {
            'midi_note': 'signal',
            'velocity': 'signal'
        }
        
        self.midi_offset = int(midi_offset) # Shifts the output keyboard range
        self.output_note = 0.0
        self.output_velocity = 0.0

    def _freq_to_midi(self, frequency):
        """Converts frequency (Hz) to the nearest integer MIDI note number."""
        if frequency <= 0:
            return 0 # Off note
        
        try:
            # N = 69 + 12 * log2(f / 440)
            midi_note_float = A4_MIDI + 12 * np.log2(frequency / A4_FREQ)
            
            # Round to the nearest integer note
            midi_note = int(round(midi_note_float))
            
            # Apply offset and clamp to MIDI range [0, 127]
            return np.clip(midi_note + self.midi_offset, 0, 127)
            
        except ValueError:
            return 0

    def step(self):
        # 1. Get raw inputs
        freq_in = self.get_blended_input('frequency_in', 'sum')
        amp_in = self.get_blended_input('amplitude_in', 'sum')
        
        # 2. Process Frequency
        # Map input signal [-1, 1] to an audible range (e.g., 50 Hz to 2000 Hz)
        if freq_in is not None:
            # We assume the input signal is normalized (e.g., from SpectrumAnalyzer)
            # Map [-1, 1] to [50, 2000] Hz
            target_freq = (freq_in + 1.0) / 2.0 * 1950.0 + 50.0
            self.output_note = float(self._freq_to_midi(target_freq))
        
        # 3. Process Amplitude
        if amp_in is not None:
            # Map signal [0, 1] (or [-1, 1]) to normalized velocity [0.0, 1.0]
            # Use abs() to treat negative signals as volume
            velocity_norm = np.clip(np.abs(amp_in), 0.0, 1.0)
            self.output_velocity = float(velocity_norm)
        else:
            self.output_velocity = 0.0

    def get_output(self, port_name):
        if port_name == 'midi_note':
            # Only output the note if the velocity is above a threshold
            return self.output_note if self.output_velocity > 0.05 else 0.0
        elif port_name == 'velocity':
            return self.output_velocity
        return None
        
    def get_display_image(self):
        w, h = 96, 48
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw piano key visualization
        note = int(self.output_note)
        
        # Calculate Octave and Note Name
        note_name_map = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note_name = note_name_map[note % 12]
        octave = note // 12 - 1
        
        # Color based on velocity
        vel_norm = self.output_velocity
        color_val = int(vel_norm * 255)
        
        if vel_norm > 0.05:
            # Draw an active key (white or black key color based on sharp/flat)
            is_sharp = ('#' in note_name)
            fill_color = (255, 0, color_val) if is_sharp else (color_val, color_val, color_val) # Red/Magenta for sharps
            text_color = (0, 0, 0) if not is_sharp else (255, 255, 255)

            cv2.rectangle(img, (0, 0), (w, h), fill_color, -1)
        else:
            text_color = (100, 100, 100)

        # Draw Note Label
        label = f"{note_name}{octave}"
        cv2.putText(img, label, (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # Draw MIDI number
        cv2.putText(img, f"MIDI: {note}", (w//4, h//2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
            
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Keyboard Offset (semitones)", "midi_offset", self.midi_offset, None),
        ]