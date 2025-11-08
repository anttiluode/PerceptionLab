"""
MIDI to Frequency Node - Converts a standard MIDI note number and velocity
into a usable frequency (Hz) and amplitude signal.

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
# MIDI Note formula: f = 440 * 2^((N - 69)/12)

class MidiToFreqNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(150, 50, 200) # Musical Purple
    
    def __init__(self):
        super().__init__()
        self.node_title = "MIDI to Freq (Hz)"
        
        self.inputs = {
            'midi_note_in': 'signal',   # MIDI note number (0-127)
            'velocity_in': 'signal'     # MIDI velocity (0.0 to 1.0)
        }
        self.outputs = {
            'frequency_out': 'signal',
            'amplitude_out': 'signal'
        }
        
        self.output_freq = 0.0
        self.output_amp = 0.0
        self.current_note = 0

        self.midi_offset = 0

    def _midi_to_freq(self, midi_note):
        """Converts integer MIDI note number to frequency (Hz)."""
        if midi_note <= 0:
            return 0.0
        
        # Clamp to reasonable range for calculation
        midi_note = np.clip(midi_note, 0, 127)
        
        # f = 440 * 2^((N - 69)/12)
        exponent = (midi_note - A4_MIDI) / 12.0
        return float(A4_FREQ * math.pow(2, exponent))

    def _get_note_name(self, midi_note):
        """Helper to get note name and octave for display."""
        note_name_map = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note = int(midi_note)
        note_name = note_name_map[note % 12]
        octave = note // 12 - 1
        return f"{note_name}{octave}"

    def step(self):
        # 1. Get raw inputs
        note_in = self.get_blended_input('midi_note_in', 'sum') or 0.0
        amp_in = self.get_blended_input('velocity_in', 'sum') or 0.0
        
        # 2. Quantize Note Input
        # Note numbers are integers; anything less than 0.5 is treated as 'off'
        if amp_in > 0.05 and note_in >= 0:
            self.current_note = int(round(note_in))
        else:
            self.current_note = 0
        
        # 3. Calculate Frequency
        self.output_freq = self._midi_to_freq(self.current_note)
        
        # 4. Calculate Amplitude
        # Amp is just the velocity signal, clamped and smoothed
        self.output_amp = np.clip(amp_in, 0.0, 1.0)

    def get_output(self, port_name):
        if port_name == 'frequency_out':
            # Output frequency only if amplitude is high enough
            return self.output_freq if self.output_amp > 0.05 else 0.0
        elif port_name == 'amplitude_out':
            return self.output_amp
        return None
        
    def get_display_image(self):
        w, h = 96, 48
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        note = self.current_note
        freq = self.output_freq
        amp = self.output_amp
        
        # Color based on activity
        if freq > 0.0:
            fill_color = (0, 150, 255) # Active Cyan
            text_color = (0, 0, 0)
            note_label = self._get_note_name(note)
        else:
            fill_color = (50, 50, 50)
            text_color = (150, 150, 150)
            note_label = "OFF"
        
        cv2.rectangle(img, (0, 0), (w, h), fill_color, -1)

        # Draw Note Label
        cv2.putText(img, note_label, (w//4, h//3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # Draw Frequency
        cv2.putText(img, f"{freq:.1f} Hz", (w//4, h//3 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        
        # Draw Amplitude Bar
        bar_w = int(amp * (w - 10))
        cv2.rectangle(img, (5, h - 10), (5 + bar_w, h - 5), (255, 255, 255), -1)
            
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("MIDI Note Offset", "midi_offset", self.midi_offset, None),
        ]